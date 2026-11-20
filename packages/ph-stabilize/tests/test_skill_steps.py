"""P7-18: a skill that is a procedure, and the loop that finishes it.

This file tests three claims. The middle claim is the reason why the row exists.

**A declared procedure becomes work.** When the model reads a skill with
`steps:`, the steps go into the todo list in order, and each one waits on the
one before it. This is the mechanism that `tool-todo` already had. Here a writer
that is not the model uses it.

**The model cannot delete them.** `write_todos` replaces the whole list, so
without a rule the procedure lasts only until the next plan of the model. The
entries that a skill seeded carry a `source` that the harness issues. A write
that drops, reorders, or rewords one of them is refused. The model may add its
own entries beside them, and it may mark them done. That is the difference
between a procedure and a suggestion.

**The loop objects while work remains.** The row listens on
`agent/turn-stopping`, the same boundary that `/autonomous` uses. It objects by
steering and not by reaching into loop state. A turn that would end while a step
can still be started is nudged instead.

A profile can add one exception. A turn that is ending *to wait for a person* is
allowed to end, where a decision model is there to say so. The tests for that
exception are at the end of this file.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import anyio.lowlevel
import pytest
from stabilize_helpers import PROFILE, events_of, row, run_tool_calls, todo_call

from ph.agent.types import AgentOptions
from ph.keys import AGENTS, DIAGNOSTICS, LLM, SESSIONS
from ph.llm.adapter import (
    DECIDED,
    DECISION_FAILED,
    LlmError,
    Noul,
    ResolvedModel,
)
from ph.llm.replay import tool_call_chunks
from ph.llm.types import CONTEXT_WINDOW_EXCEEDED, GenerateOptions, text_of
from ph.seams.skills import discover_skills, rendered_skill
from ph.session import Session, SurfaceIntent
from ph.testing import (
    EMOJI_FAMILY,
    FAKE_OPTIONS,
    MountProfile,
    StoppingAgent,
    StubDecider,
    assistant_payload,
    llama_refusal,
    log_event,
    result_text,
    run_tool,
    write_skill,
)
from ph_stabilize import skill_steps
from ph_stabilize.skill_steps import (
    HOLD_AT,
    MAX_NAMED,
    MAX_NUDGES,
    MIN_SAID,
    MOST_SAID,
    STEP_BYTES,
    WAITING,
    decision_state,
    nudges_since,
    said_bytes,
    seeded,
    steer_text,
)
from ph_stabilize.todo import (
    MAX_TODO_CONTENT,
    MAX_TODOS,
    SKILL,
    startable,
    steps_of,
    todos_of,
)

pytestmark = pytest.mark.anyio

ROWS: list[dict[str, Any]] = [
    {"id": "tool-todo", "disabled": False},
    {"id": "skill-steps", "disabled": False},
]

STEP_TEXTS = ["survey the callers", "port the row", "gate it"]
STEPS = "steps:\n" + "".join(f"  - {one}\n" for one in STEP_TEXTS)


def _entry(
    content: str,
    status: str = "pending",
    requires: list[str] | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    """One entry, written in one way, including the `source` that the harness issues."""
    entry: dict[str, Any] = {"content": content, "status": status, "requires": requires or []}
    if source is not None:
        entry["source"] = source
    return entry


async def _reading(
    mount: MountProfile,
    tmp_path: Path,
    *,
    skill_budget: int | None = None,
    profile_budget: int | None = None,
    decision: dict[str, Any] | None = None,
) -> Any:  # noqa: ANN401
    """A mounted deployment that has just read a three-step skill.

    `skill_budget` is the `max-nudges` of the skill, and `profile_budget` is the
    `maxNudges` of the row. When either is left out, nothing is set, so the
    default applies. `decision` is the `decision` of the row: the route of a
    decision model to ask.
    """
    extra = STEPS if skill_budget is None else f"{STEPS}max-nudges: {skill_budget}\n"
    write_skill(tmp_path, "port", description="port a row", extra=extra, body="Do it.")
    planning, steering = ROWS
    config: dict[str, Any] = {}
    if profile_budget is not None:
        config["maxNudges"] = profile_budget
    if decision is not None:
        config["decision"] = decision
    if config:
        steering = {**steering, "config": config}
    ctx = await mount(
        planning,
        steering,
        {"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}},
        profile=PROFILE,
    )
    session = ctx.require(SESSIONS).create("procedure")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await run_tool(ctx, "skill", {"name": "port"}, agent=agent, session=session)
    return ctx, session, agent


# ------------------------------------------------------------ the seeding --


async def test_reading_a_skill_turns_its_steps_into_work(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The declaration becomes entries, in order, and each waits on the one before.

    Sequential `requires` is the mechanism that `tool-todo` already had. This
    row adds no ordering of its own. That is why `blocked_by` and the sidebar
    understand a seeded procedure without knowing that skills exist.
    """
    _ctx, session, _agent = await _reading(mount, tmp_path)

    todos = todos_of(session)
    assert steps_of(todos) == ["survey the callers", "port the row", "gate it"]
    # The list and the budget that governs it are written in one batch (S14).
    seeded = session.latest("todo/write")
    budget = session.latest("skill-steps/budget")
    assert seeded is not None and budget is not None
    assert seeded.batch is not None and seeded.batch == budget.batch
    assert [one["requires"] for one in todos] == [[], ["survey the callers"], ["port the row"]]
    assert startable(todos) == ["survey the callers"], "one at a time, in the declared order"


async def test_reading_the_same_skill_twice_does_not_duplicate_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A model re-reads instructions that it half remembers. A second copy of the
    procedure would be a plan that can never be finished."""
    ctx, session, agent = await _reading(mount, tmp_path)
    await run_tool(ctx, "skill", {"name": "port"}, agent=agent, session=session)

    assert len(steps_of(todos_of(session))) == 3


def test_a_procedure_that_grew_seeds_only_what_is_new() -> None:
    """A skill that is edited in the middle of a session adds its new step and
    adds no step a second time.

    This is the one thing that the by-content filter does and the collision
    guard below does not. Without the filter, a procedure that grew seeds
    nothing at all, because its *first* step is already an entry and the whole
    batch is refused.
    """
    current = [
        _entry("survey", "completed", source=SKILL),
        _entry("port", "in_progress", ["survey"], source=SKILL),
    ]

    grown = seeded(current, ["survey", "port", "gate it"])

    assert grown is not None
    assert steps_of(grown) == ["survey", "port", "gate it"]
    # The new step waits on the step that the *skill* put before it (D13). That
    # step is `port`, which is already in the list and still in progress. This
    # assertion used to read `== []`, on the argument that a new step "waits on
    # nothing it did not arrive with". That was the chain built over the steps
    # that are *missing* and not over what the skill declared, so a step that
    # the procedure put last could start immediately. The ordering belongs to
    # the procedure, not to the missing steps.
    assert grown[-1]["requires"] == ["port"]
    assert [one["status"] for one in grown[:2]] == ["completed", "in_progress"], "progress kept"


def test_a_step_whose_text_the_model_already_used_is_not_seeded() -> None:
    """`requires` names entries by content, so two entries with one text make a
    plan that cannot be reproduced. The seeding is refused here. Otherwise the
    list would be written, and `_checked` would refuse the next write of the
    model."""
    assert seeded([_entry("port the row", "completed")], ["port the row"]) is None


# ------------------------------------------------------- the model's hands --


async def test_the_model_may_mark_a_seeded_step_done(mount: MountProfile, tmp_path: Path) -> None:
    """Marking progress is the one change that the model *may* make. That is the purpose."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    steps = steps_of(todos_of(session))

    await run_tool_calls(
        ctx,
        session,
        todo_call(
            "c1",
            [
                _entry(steps[0], "completed"),
                _entry(steps[1], "in_progress", [steps[0]]),
                _entry(steps[2], "pending", [steps[1]]),
            ],
        ),
    )

    todos = todos_of(session)
    assert [one["status"] for one in todos] == ["completed", "in_progress", "pending"]
    assert steps_of(todos) == steps, "and provenance survives a write that never mentions it"


async def test_the_model_may_add_its_own_entries_beside_them(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A procedure does not take over the list. The model keeps its own plan beside it."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    steps = steps_of(todos_of(session))

    await run_tool_calls(
        ctx,
        session,
        todo_call(
            "c1",
            [
                _entry(steps[0]),
                _entry(steps[1], requires=[steps[0]]),
                _entry(steps[2], requires=[steps[1]]),
                _entry("read the upstream diff"),
            ],
        ),
    )

    todos = todos_of(session)
    assert len(todos) == 4
    assert steps_of(todos) == steps, "the model's own entry carries no provenance"


async def test_a_write_that_drops_a_seeded_step_is_refused(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The rule that the row exists for.**

    `write_todos` replaces the whole list. Without this rule, a procedure lasts
    only until the next plan of the model, and a `turn-stopping` listener that
    enforced against that list would enforce against nothing.

    Sabotage: drop `_carried` and the plan below is written, the steps vanish,
    and the loop stops objecting because it can no longer see anything to finish.
    """
    ctx, session, _agent = await _reading(mount, tmp_path)
    steps = steps_of(todos_of(session))

    await run_tool_calls(ctx, session, todo_call("c1", [_entry(steps[0]), _entry("my own plan")]))

    assert steps_of(todos_of(session)) == steps, "nothing was written"
    said = result_text(session, "c1")
    assert "drops 2 steps" in said and repr(steps[1]) in said, "it names what went missing"


async def test_a_write_that_reorders_them_is_refused(mount: MountProfile, tmp_path: Path) -> None:
    """The order is the procedure. `requires` alone would not detect a swap of two
    steps that do not depend on each other."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    steps = steps_of(todos_of(session))

    await run_tool_calls(
        ctx,
        session,
        todo_call("c1", [_entry(steps[2]), _entry(steps[1]), _entry(steps[0])]),
    )

    assert steps_of(todos_of(session)) == steps
    said = result_text(session, "c1")
    assert "reorders" in said and "drops" not in said, (
        "a reorder that named every step as dropped told the model all three were "
        "wrong when only their order was"
    )


# --------------------------------------------------------------- steering --


def test_the_steer_names_a_few_and_counts_the_rest() -> None:
    """The nudge is a pointer, not a second copy of the plan. The model already has
    the list.

    A twenty-line reminder each time a turn tries to end becomes noise that a
    model learns to skim.
    """
    text = steer_text([f"step {n}" for n in range(MAX_NAMED + 2)], blocked=4)

    assert text.count("'step ") == MAX_NAMED, "quoted names only; the blocked count is not one"
    assert "and 2 more" in text
    assert "4 further steps wait" in text, "counted prose agrees, via `count_of`"
    assert "1 further step waits" in steer_text(["a"], blocked=1), "and agrees at one"


EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "skills" / "self-steerings-examples"
"""The OpenMono playbooks, ported as self-steering skills. They are meant to be
run and not only read, so they are gated in the way that a deployment would
install them.

The gate is here and not in ph-core, which owns `discover_skills`. The bound that
matters to a *step* is `MAX_TODO_CONTENT`, and ph-core may not import the package
that declares it. This suite sees both."""

EXAMPLE_ARGUMENTS: dict[str, dict[str, Any]] = {
    "commit": {"scope": "auth"},
    "db-migrate": {"target": "staging"},
    "deploy-ftp": {"host": "ftp.example.com", "user": "deploy"},
    "file-scan": {},
    "graphify": {"action": "query", "args": "how does auth work?"},
    "incident-response": {"service": "checkout", "severity": "P1"},
    "pr-ready": {},
    "release": {"version-type": "minor"},
}
"""Enough arguments to satisfy the required inputs of each example. A generic
placeholder does not work because of the enums, which shows that the declaration
does its job."""


def test_the_ported_examples_all_load_and_render() -> None:
    """Eight worked skills, checked in the way that a deployment would install them.

    They are documentation that is also *input*. A profile points `paths:` at
    that directory and gets all eight. The gate is therefore `discover_skills`,
    the call that a mount makes. A scanner that refuses a skill logs a warning
    and drops it, and a reader would never notice that failure.

    The test renders the skills too, and does not only load them. A body is
    scanned for undeclared placeholders only when something asks for it. A
    mistyped `{{parameters.x}}` in any of these would therefore stay until the
    first model that read it got a tool error and not instructions.

    Every step is also checked against `MAX_TODO_CONTENT`, because a step here
    is an entry in a todo list and not a paragraph of prompt. One step in the
    source playbooks is 510 characters, which `seeded` would refuse outright.
    """
    if not EXAMPLES.is_dir():
        pytest.skip("docs are not part of an installed distribution")
    found = discover_skills([str(EXAMPLES)])

    on_disk = sorted(one.name for one in EXAMPLES.iterdir() if one.is_dir())
    assert [one.name for one in found] == on_disk, "every example directory installs"
    assert set(EXAMPLE_ARGUMENTS) == set(on_disk), "this table covers exactly what is there"

    for one in found:
        assert one.steps, f"{one.name} is a self-steering example and declares no steps"
        body = Path(str(one.path)).read_text(encoding="utf-8")
        filled, steps = rendered_skill(body, one, EXAMPLE_ARGUMENTS[one.name])
        assert seeded([], steps) is not None, f"{one.name} does not seed"
        for step in steps:
            assert len(step) <= MAX_TODO_CONTENT, (
                f"{one.name}: a step too long to be a todo entry seeds nothing at all"
            )
        for orphan in ("{{params.", "{{state.", "{{playbook.", "{{shell:"):
            assert orphan not in filled, (
                f"{one.name} still carries {orphan} — pH has no such substitution, so it "
                "would reach the model as literal text (see the examples' own README)"
            )


async def _nudges(ctx: Any, stopping: StoppingAgent, tries: int) -> int:  # noqa: ANN401
    """How many of `tries` turn-stopping boundaries the row steered at."""
    before = len(stopping.steers)
    for _ in range(tries):
        await ctx.serial("agent/turn-stopping", stopping, 1)
    return len(stopping.steers) - before


async def test_a_turn_trying_to_end_with_a_procedure_unfinished_is_steered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The whole purpose of the row, at the boundary that the agent loop already
    fires.

    Sabotage: drop the `agent.steer(...)` call, or the `ctx.on` that registers
    this listener, and a model that read a three-step procedure stops after
    whatever it felt like doing first.
    """
    ctx, session, _agent = await _reading(mount, tmp_path)
    stopping = StoppingAgent(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(stopping.steers) == 1, "one nudge, not one per outstanding step"
    message = stopping.steers[0]
    assert "survey the callers" in text_of(message.content)
    assert message.source.plugin == "ph_stabilize.skill_steps", (
        "tagged, so the transcript does not read the harness's nudge as the person's"
    )


async def test_a_finished_procedure_lets_the_turn_end(mount: MountProfile, tmp_path: Path) -> None:
    """The other half of the rule. A row that never stopped steering would break it."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    await run_tool_calls(
        ctx, session, todo_call("done", [_entry(step, status="completed") for step in STEP_TEXTS])
    )
    stopping = StoppingAgent(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "nothing is outstanding; the turn is allowed to end"


async def test_a_session_with_no_procedure_is_never_steered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A mounted row must cost nothing for a session that does not use it."""
    ctx = await mount(*ROWS, profile=PROFILE)
    session = ctx.require(SESSIONS).create("plain")
    ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await run_tool_calls(ctx, session, todo_call("own", [_entry("my own work")]))
    stopping = StoppingAgent(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "the model's own list is the model's to finish"


async def test_the_row_stands_down_when_its_nudges_change_nothing(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The ceiling that `/autonomous` has and the first version of this row did not.

    The loop has no step cap (`limits` ships with `turn_limit` unset, on
    purpose). A model that will not mark a step done would therefore be steered
    for the life of the session. The count runs since the list last *changed*,
    which makes it a stall detector and not a quota.

    Sabotage: drop the `nudges_since` check and the loop below never stops
    steering.
    """
    ctx, session, _agent = await _reading(mount, tmp_path)

    assert await _nudges(ctx, StoppingAgent(session), MAX_NUDGES + 2) == MAX_NUDGES, (
        "it stops after the plan has not moved"
    )


async def test_progress_on_the_plan_earns_more_nudges(mount: MountProfile, tmp_path: Path) -> None:
    """A run that makes progress is never cut off. That is what makes the ceiling a
    stall detector and not a quota."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    stopping = StoppingAgent(session)
    await _nudges(ctx, stopping, MAX_NUDGES)

    steps = steps_of(todos_of(session))
    await run_tool_calls(
        ctx,
        session,
        todo_call("moved", [_entry(steps[0], "completed"), *(_entry(one) for one in steps[1:])]),
    )
    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(stopping.steers) == MAX_NUDGES + 1
    assert "port the row" in text_of(stopping.steers[-1].content), "and it points at the next one"


async def test_a_profile_sets_the_nudge_budget(mount: MountProfile, tmp_path: Path) -> None:
    """D16: the deployment chooses the ceiling. It is not a constant.

    Sabotage: compare against `MAX_NUDGES` in `keep_going` again and this
    steers three times.
    """
    ctx, session, _agent = await _reading(mount, tmp_path, profile_budget=1)

    assert await _nudges(ctx, StoppingAgent(session), 5) == 1


async def test_a_skill_budget_overrides_the_profile(mount: MountProfile, tmp_path: Path) -> None:
    """D16: the author knows what a half-finished run of their procedure is worth.

    The budget also outlives the next plan of the model. `write_todos` replaces
    the `todo/write` that seeded the steps, which is why the budget is a record
    of its own.

    Sabotage: drop the `skill-steps/budget` append, or read the budget off the
    latest `todo/write`, and one of the two counts below falls to the profile's 1.
    """
    ctx, session, _agent = await _reading(mount, tmp_path, skill_budget=4, profile_budget=1)
    stopping = StoppingAgent(session)
    assert await _nudges(ctx, stopping, 6) == 4

    steps = steps_of(todos_of(session))
    await run_tool_calls(
        ctx,
        session,
        todo_call("moved", [_entry(steps[0], "completed"), *(_entry(one) for one in steps[1:])]),
    )

    assert await _nudges(ctx, stopping, 6) == 4, "still the skill's budget after a re-plan"


async def test_a_skill_budget_of_zero_seeds_but_never_steers(
    mount: MountProfile, tmp_path: Path
) -> None:
    """D16: an author can say "a partial result is fine"."""
    ctx, session, _agent = await _reading(mount, tmp_path, skill_budget=0)

    assert steps_of(todos_of(session)) == STEP_TEXTS, "the procedure is still the plan"
    assert await _nudges(ctx, StoppingAgent(session), 3) == 0


async def test_a_later_procedure_without_a_budget_uses_the_profile(
    mount: MountProfile, tmp_path: Path
) -> None:
    """D16: the procedure that was seeded last governs. When it sets no budget, the
    budget of the profile applies, and not the budget that the skill before it
    asked for."""
    write_skill(tmp_path, "notes", description="take notes", extra="steps:\n  - write it down\n")
    ctx, session, agent = await _reading(mount, tmp_path, skill_budget=5, profile_budget=1)
    await run_tool(ctx, "skill", {"name": "notes"}, agent=agent, session=session)

    assert await _nudges(ctx, StoppingAgent(session), 5) == 1


@pytest.mark.parametrize("budget", [0, 2])
async def test_a_spent_ceiling_steers_for_exactly_the_nudge_budget(
    mount: MountProfile, tmp_path: Path, budget: int
) -> None:
    """D16: the decided behavior when a `limits` ceiling is spent and a procedure
    is unfinished.

    `_turn` reads the inbox again after `agent/turn-stopping`, so a nudge keeps
    a turn alive past a ceiling that concluded it. The call of the next step is
    denied, the turn concludes again, and the row steers again. This is left as
    it is, on purpose. "The budget is spent" and "the procedure is unfinished"
    are different facts, and only the skill knows whether its outstanding steps
    are necessary. The nudge budget limits the repetition (the budget of the
    skill, or else the budget of the profile). The model calls that are spent
    past the ceiling are therefore exactly that budget.

    The test runs a real turn, because the interaction is in the driver. The
    model never marks a step done and reads the skill again at every step. Such
    a read seeds nothing, so it cannot reset the count. The turn runs under a
    deadline, so a row that never stands down fails and does not hang. A cap on
    the adapter cannot do that, because whatever the adapter answers instead is
    steered too.
    """
    write_skill(tmp_path, "port", description="port a row", extra=f"{STEPS}max-nudges: {budget}\n")
    ctx = await mount(
        *ROWS,
        row("limits", toolCalls={"turnLimit": 1, "exit": "end"}),
        {"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}},
        profile=PROFILE,
    )
    calls = 0

    class Stubborn:
        async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]:
            nonlocal calls
            calls += 1
            # A checkpoint, so the deadline below can cancel a loop that never
            # stops asking.
            await anyio.lowlevel.checkpoint()
            for chunk in tool_call_chunks("", "skill", '{"name": "port"}'):
                yield chunk

    ctx.require(LLM).register_adapter(["stubborn"], Stubborn())
    session = ctx.require(SESSIONS).create("capped")
    agent = ctx.require(AGENTS).create(session, AgentOptions(provider="stubborn", model="m"))

    with anyio.fail_after(10):
        await agent.prompt("port it")

    # One call inside the ceiling, one that crosses it and concludes the turn,
    # then one more per nudge.
    assert calls == 2 + budget
    assert nudges_since(session, -1) == budget, "every call past the ceiling was a nudge's"


def test_a_step_too_long_to_be_a_todo_entry_seeds_nothing() -> None:
    """The seeding is all or nothing, because the alternative is a plan that the
    model cannot write.

    A step longer than `MAX_TODO_CONTENT` would seed without error and then
    fail `WriteTodosArgs` on *every* later call, while `_carried` refuses any
    write that drops it. The model would be locked out of its own list while
    the row steers it to continue. The author does not control the length
    either: `{{parameters.x}}` renders an argument that the model supplies into
    the text of the step.
    """
    assert seeded([], ["fine", "x" * (MAX_TODO_CONTENT + 1)]) is None


def test_seeding_past_the_list_cap_seeds_nothing() -> None:
    """The other half of the same lockout, from the other bound."""
    full = [_entry(f"mine {n}") for n in range(MAX_TODOS - 1)]

    assert seeded(full, ["one more"]) is not None, "there is room for exactly one"
    assert seeded(full, ["one more", "and another"]) is None


def test_a_chain_points_at_one_step_at_a_time() -> None:
    """This is also why the empty-`ready` branch in `keep_going` is a *shape* guard.

    `seeded` writes a sequential chain. While any step in it is unfinished, the
    earliest unfinished step waits on nothing, so there is always exactly one
    step to name. The entries that this row writes cannot reach the branch that
    would decline to steer. The test asserts this here and not through the
    listener, so that no gate claims to cover a state that the feature cannot
    produce.
    """
    todos = [
        _entry("a", "in_progress", source=SKILL),
        _entry("b", "pending", ["a"], source=SKILL),
        _entry("c", "pending", ["b"], source=SKILL),
    ]

    assert startable(todos) == ["a"], "the in-progress head is startable; nothing behind it is"
    todos[0]["status"] = "completed"
    assert startable(todos) == ["b"], "finishing one uncovers exactly the next"
    todos[1]["status"] = "completed"
    todos[2]["status"] = "completed"
    assert startable(todos) == [], "and a finished chain offers nothing, which is how it ends"


def test_a_re_read_links_around_a_step_that_is_already_there() -> None:
    """D13: the chain was built over the missing steps, so it skipped a step that
    was already there.

    It is ordinary to read a skill again in the middle of a session. It is also
    ordinary to find one of its steps already in the list, because a previous
    seed may have been partly carried. With the chain built over `wanted`, a
    procedure `one → two → three` whose middle step is present seeds `three`
    waiting on `one`. The ordering that the skill declared is silently replaced
    by one that lets `three` start while `two` is still pending. `requires`
    exists to prevent exactly that.

    The predecessor can always be named, so this needs no fallback. Every step
    is either already in the list or added in the same batch.
    """
    current = [_entry("two", "pending", source=SKILL)]

    grown = seeded(current, ["one", "two", "three"])

    assert grown is not None
    by_content = {one["content"]: one for one in grown}
    assert by_content["one"]["requires"] == []
    assert by_content["three"]["requires"] == ["two"], "the chain skipped the step that was there"


# ----------------------------------------------------- the decision model --

READY = (["survey the callers"], 2)
"""What can start, and how many wait, after the three-step skill is read."""

ROUTE = {"provider": "decision", "model": "clef"}
"""The `decision` of the row: a route, named in the same way as the route of a session."""

ASKING = "The dry run passed on staging. Shall I apply it to production? Reply yes to go on."
"""How a turn ends when it obeys a step that says "stop for approval"."""

TOO_LONG = f"provider returned 500: {llama_refusal(1100, 512)}"
"""A llama.cpp's refusal of a decision that does not fit its batch, as the
adapter raises it."""


async def _stopping_with(
    mount: MountProfile,
    tmp_path: Path,
    decider: StubDecider | None,
    said: str = ASKING,
    decision: dict[str, Any] = ROUTE,
) -> Any:  # noqa: ANN401
    """A deployment whose `skill-steps` row asks the decision model that `decision`
    names, with a procedure just read in it, and a turn that is ending with
    `said` as its last message."""
    ctx, session, _agent = await _reading(mount, tmp_path, decision=decision)
    if decider is not None:
        ctx.require(LLM).register_adapter(["decision"], decider)
    if said:
        log_event(
            session, "assistant/message", assistant_payload(said, "a9"), SurfaceIntent("append")
        )
    return ctx, session, StoppingAgent(session)


def _decided(session: Session) -> list[dict[str, Any]]:
    return [dict(one.data) for one in events_of(session, skill_steps.DECIDED)]


async def test_a_turn_ending_to_wait_for_a_person_is_let_end(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The row keeps a gate that a skill writes in prose.

    A step says "stop for approval", and the model stops and asks. The row used
    to answer with a nudge to continue, which steered the model past the
    approval that it had been told to wait for. Now the row asks a decision
    model first. The model says that the turn is waiting on a person, and the
    turn is allowed to end.

    Both halves are in the log. The record of the route says what the request
    cost. The record of the row says what the answer was and what threshold
    applied.

    Sabotage: drop the `waits_for_a_person` check in `keep_going` and the turn
    is nudged past the question.
    """
    decider = StubDecider(0.97)
    ctx, session, stopping = await _stopping_with(mount, tmp_path, decider)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "the model asked a person something; the turn is theirs"
    (asked,) = decider.asked
    assert (asked.provider, asked.model) == ("decision", "clef")
    assert list(asked.questions) == [WAITING] and isinstance(asked.questions[WAITING], Noul)
    assert "survey the callers" in asked.state, "it is told what was left undone"
    assert asked.state.endswith(ASKING), "and how the turn ended"
    assert _decided(session) == [{"waiting": 0.97, "holdAt": HOLD_AT}]
    assert [one.data["usage"]["inputTokens"] for one in events_of(session, DECIDED)] == [120]


async def test_a_turn_that_just_stopped_is_still_nudged(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The other answer. The row must then do what it always did. It must also
    record in the log that it asked, so that this nudge can be told from a nudge
    that was sent without a decision."""
    decider = StubDecider(0.02)
    ctx, session, stopping = await _stopping_with(
        mount, tmp_path, decider, "I surveyed the callers; there are four."
    )

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(stopping.steers) == 1
    assert _decided(session) == [{"waiting": 0.02, "holdAt": HOLD_AT}]


@pytest.mark.parametrize(("hold_at", "held"), [(HOLD_AT, False), (0.5, True)])
async def test_the_line_a_nudge_is_held_at_is_the_profile_s(
    mount: MountProfile, tmp_path: Path, hold_at: float, held: bool
) -> None:
    """The answer is a probability, never a verdict. The deployment chooses the
    threshold. The default still nudges the unsure middle. A profile that
    prefers to end a turn over the risk of steering past a question lowers the
    threshold."""
    ctx, _session, stopping = await _stopping_with(
        mount, tmp_path, StubDecider(0.65), "ok?", decision={**ROUTE, "holdAt": hold_at}
    )

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert (not stopping.steers) is held


async def test_a_held_nudge_spends_none_of_the_budget(mount: MountProfile, tmp_path: Path) -> None:
    """A nudge that was not sent is not a nudge. The stall detector counts the
    messages of the row in the log, and a hold writes none. A procedure that
    paused for an answer therefore comes back to its whole budget.

    Sabotage: write the hold as a `user/message` under this row's tag and the
    turn after the person answers has one nudge fewer.
    """
    decider = StubDecider(0.97, 0.01)
    ctx, _session, stopping = await _stopping_with(mount, tmp_path, decider)

    assert await _nudges(ctx, stopping, 1) == 0, "held"
    assert await _nudges(ctx, stopping, MAX_NUDGES + 2) == MAX_NUDGES, "then the whole budget"
    assert len(decider.asked) == 1 + MAX_NUDGES, "and a row that has stood down asks nothing"


@pytest.mark.parametrize(
    ("decider", "code"),
    [
        (None, "NO_ADAPTER"),
        (StubDecider(LlmError("the model is loading", "SERVER_ERROR")), "SERVER_ERROR"),
    ],
)
async def test_a_decision_model_that_cannot_be_asked_leaves_the_nudge(
    mount: MountProfile,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    decider: StubDecider | None,
    code: str,
) -> None:
    """Graceful degradation. When nobody registered the route, or the server is
    down, the deployment loses the exception and not the feature. The turn is
    nudged as it was before there was a decision model.

    The reason is kept where each reader looks. The log of the session records
    it every time, by code and sentence. The log of the operator records it one
    time, and not at the end of every turn.

    Sabotage: let the `LlmError` out of `waits_for_a_person` and a decision
    server that is down fails every turn that ends with a procedure unfinished.
    """
    ctx, session, stopping = await _stopping_with(mount, tmp_path, decider)

    with caplog.at_level("WARNING", logger="ph_stabilize.skill_steps"):
        assert await _nudges(ctx, stopping, 2) == 2

    assert [one.data["code"] for one in events_of(session, DECISION_FAILED)] == [code, code]
    assert _decided(session) == [], "nothing was decided, so nothing says it was"
    warnings = [one.getMessage() for one in caplog.records if "decision model" in one.getMessage()]
    assert len(warnings) == 1 and "decision/clef" in warnings[0] and code in warnings[0]


async def test_a_state_too_long_for_the_server_is_asked_again_with_less(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A decision is read in one batch. The batch of a llama.cpp is 512 tokens
    unless it was started with more, and it must hold the state, the question,
    and the framing of the model together. How many characters that is depends
    on the text. The row therefore sends the end of the message. If the server
    refuses, the row sends less of it, one time, sized by the two numbers that
    the refusal gives.

    The row sends the end, because that is where a question is.

    Sabotage: drop the retry in `_decided` and a reply in a script denser than
    prose is nudged past its question on every server left at the default.
    """
    said = "I checked another caller and it was fine. " * 200 + ASKING
    decider = StubDecider(LlmError(TOO_LONG, CONTEXT_WINDOW_EXCEEDED), 0.97)
    ctx, session, stopping = await _stopping_with(mount, tmp_path, decider, said)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    first, second = (one.state for one in decider.asked)
    assert said_bytes(None, *READY) < len(first) < len(said), "only the end of it, at first"
    assert len(second) < len(first) * 0.5, "the second is cut by what the server said it reads"
    assert first.endswith(ASKING) and second.endswith(ASKING)
    assert stopping.steers == [] and _decided(session)[0]["waiting"] == 0.97
    assert [one.type for one in session.events if one.type.startswith("llm/")] == [
        DECISION_FAILED,
        DECIDED,
    ], "and the refusal is on the record beside the answer"


@pytest.mark.parametrize(
    ("limit", "shown"), [(None, said_bytes(None, *READY)), (8192, MOST_SAID), (300, MIN_SAID)]
)
async def test_how_much_of_the_message_is_shown_is_the_route_s(
    mount: MountProfile, tmp_path: Path, limit: int | None, shown: int
) -> None:
    """How much fits is a fact about the route (`LlmRuntime.decision_limit`). What
    to cut is the choice of the row: it keeps the end of the message. A route
    that declares a bigger batch is shown more, so a question that is followed
    by a long listing stays in view. The most is `MOST_SAID`, because a turn
    waits for the answer. A route that reads less is shown less, down to
    `MIN_SAID`.

    Sabotage: size the tail by a constant again and the 8192-token route is
    shown what a 512-token one is.
    """
    said = ASKING + " This is what I would run:\n" + "\n".join(["rows += 1"] * 800)
    decider = StubDecider(0.97, route=ResolvedModel(max_decision_tokens=limit))
    ctx, _session, stopping = await _stopping_with(mount, tmp_path, decider, said)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    (asked,) = decider.asked
    assert asked.state.endswith("\n…" + said[-(shown - len("…".encode())) :]), (
        "the mark inside the budget"
    )


async def test_a_limit_one_refusal_named_sizes_the_next_ask(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The seam keeps the limit that a refusal reported for the route. The state of
    the next turn is then sized to fit that limit, and is not refused in the
    same way again."""
    small = f"provider returned 500: {llama_refusal(1100, 300)}"
    said = "I checked another caller and it was fine. " * 200 + ASKING
    decider = StubDecider(LlmError(small, CONTEXT_WINDOW_EXCEEDED), 0.01)
    ctx, _session, stopping = await _stopping_with(mount, tmp_path, decider, said)

    await _nudges(ctx, stopping, 2)

    first, second = (one.state for one in decider.asked)
    assert ctx.require(LLM).decision_limit("decision", "clef") == 300
    first_room, second_room = said_bytes(None, *READY), said_bytes(300, *READY)
    assert first.endswith("\n…" + said[-(first_room - len("…".encode())) :])
    assert second.endswith("\n…" + said[-(second_room - len("…".encode())) :]), (
        "sized by what it said"
    )


async def test_a_route_that_timed_out_rests_before_it_is_asked_again(
    mount: MountProfile, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A route that stalls costs every turn that would be nudged the whole
    `timeoutMs`, in every session. After one timeout, the row nudges without
    asking for `REST_S`, and then asks again. A route that is only down is
    asked at every turn, because it refuses in milliseconds (see
    `test_a_decision_model_that_cannot_be_asked_leaves_the_nudge`).

    Sabotage: drop the `resting_until` check in `waits_for_a_person` and the
    second turn waits on the stalled route again.
    """
    clock = [100.0]
    monkeypatch.setattr(skill_steps, "_now", lambda: clock[0])
    stalled = LlmError('provider "decision" did not decide within 3 s', "TIMEOUT")
    decider = StubDecider(stalled, 0.97)
    ctx, session, stopping = await _stopping_with(mount, tmp_path, decider)

    assert await _nudges(ctx, stopping, 2) == 2
    assert len(decider.asked) == 1, "resting: the second turn did not wait on it"
    assert [one.data["code"] for one in events_of(session, DECISION_FAILED)] == ["TIMEOUT"]

    clock[0] += skill_steps.REST_S
    assert await _nudges(ctx, stopping, 1) == 0, "asked again, and it was waiting"
    assert len(decider.asked) == 2


async def test_a_refusal_a_shorter_state_cannot_answer_is_not_asked_twice(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Only one kind of refusal is worth a second request, and only when it says
    by how much the state was too long. In any other case, the row would ask the
    same thing again while a turn waits for the answer."""
    decider = StubDecider(LlmError("it did not fit", CONTEXT_WINDOW_EXCEEDED), 0.97)
    ctx, _session, stopping = await _stopping_with(mount, tmp_path, decider)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(decider.asked) == 1 and len(stopping.steers) == 1


async def test_a_turn_that_said_nothing_is_not_asked_about(
    mount: MountProfile, tmp_path: Path
) -> None:
    """There is nothing to judge. A request about an empty message would be a
    decision about the wording of the question itself."""
    decider = StubDecider(0.97)
    ctx, _session, stopping = await _stopping_with(mount, tmp_path, decider, said="")

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert decider.asked == [] and len(stopping.steers) == 1


async def test_a_finished_procedure_asks_no_decision(mount: MountProfile, tmp_path: Path) -> None:
    """The decision model is asked last, after every cheaper check has said that a
    nudge is due. It is a request to a server, and a turn that would not be
    nudged must not wait for one."""
    decider = StubDecider(0.97)
    ctx, session, stopping = await _stopping_with(mount, tmp_path, decider)
    await run_tool_calls(
        ctx, session, todo_call("done", [_entry(step, status="completed") for step in STEP_TEXTS])
    )

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert decider.asked == [] and stopping.steers == []


async def test_doctor_says_which_decision_model_is_asked_and_whether_it_can_be(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A user runs `phern doctor` when the nudges are not held. The usual reason
    is a route that names no adapter that decides. The line is read on request,
    so an adapter row that is mounted after this row counts."""
    ctx, _session, _stopping = await _stopping_with(mount, tmp_path, None)

    def line() -> str:
        report = dict(ctx.require(DIAGNOSTICS).report())
        said: str = dict(report["Skill steps"])["decision model"]
        return said

    assert "decision/clef" in line() and "no adapter" in line()
    ctx.require(LLM).register_adapter(["decision"], StubDecider(0.5))
    assert "decision/clef" in line() and f"held at {HOLD_AT:g}" in line()


async def test_a_row_with_no_decision_model_has_nothing_for_doctor(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx, _session, _agent = await _reading(mount, tmp_path)
    assert "Skill steps" not in dict(ctx.require(DIAGNOSTICS).report())


def test_the_decision_model_is_shown_the_end_of_a_long_message() -> None:
    """What the decision model is asked about:

    - The step that can start, on one line, however the skill wrapped it.
    - How many steps wait on it.
    - The end of what the assistant said. It is marked as cut, so that a
      clipped sentence is not read as the whole reply.
    """
    step = "Apply the migration\n  to production " + "and verify " * 40
    said = "x" * 5000 + " Shall I go on?"

    state = decision_state([step], 2, said, 100)

    assert "'Apply the migration to production and verify" in state
    assert "…'" in state and "2 further steps wait on these" in state
    assert state.endswith("Shall I go on?")
    assert "\n…" + said[-(100 - len("…".encode())) :] in state
    assert decision_state([step], 0, "short", 100).endswith("\nshort")


def test_the_end_of_the_message_is_cut_by_bytes_between_characters() -> None:
    """The decision model is shown a byte budget of the message, because its
    server counts tokens and bytes are the careful estimate of tokens. The cut
    never falls inside a character: not inside a byte sequence, and not inside
    an emoji sequence or an accent either (`ph.characters.tail_bytes`).

    Sabotage: cut the encoded message at `-room` and decode it, and the tail
    starts with the last members of a family.
    """
    said = "Shall I go on? " + EMOJI_FAMILY * 40

    state = decision_state(["port the row"], 0, said, 50)

    shown = state.rsplit("\n…", 1)[1]
    assert len(shown.encode()) <= 50
    assert not shown.replace(EMOJI_FAMILY, ""), "whole families only"


def test_the_whole_state_is_sized_in_bytes() -> None:
    """Everything that the decision model is shown uses one unit. A step is
    clipped by bytes, as the message is. Three steps written in a script of
    three bytes a character therefore take the room that three steps in English
    take. The message gets what the route reads, less the framing and the steps
    as they are sent. It does not get what is left after a reserve that was
    measured one time on English steps, which three long Chinese steps overran.

    Sabotage: clip steps by characters again and the Chinese steps take three
    times their share, leaving the message the floor.
    """
    english = ["Run the migration against staging and report every table it touched " * 3]
    chinese = ["在预发布环境运行数据库迁移并报告它修改过的每一张表格以及每一个索引和约束" * 3]

    for steps in (english * 3, chinese * 3):
        prefix = decision_state(steps, 4, "", 0)
        named = prefix.split(": ", 1)[1].split(" can be started now")[0]
        assert all(
            len(one.encode()) <= STEP_BYTES + len("''…".encode()) for one in named.split(", ")
        )

    assert said_bytes(None, english * 3, 4) < said_bytes(None, english, 0), (
        "steps shown are room the message loses"
    )
    assert abs(said_bytes(None, chinese * 3, 4) - said_bytes(None, english * 3, 4)) <= 3 * 2, (
        "and three long steps take the same room in either script, to a character's cut"
    )
