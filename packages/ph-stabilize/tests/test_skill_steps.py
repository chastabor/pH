"""P7-18 — a skill that is a procedure, and the loop that finishes it.

Three claims, and the middle one is the row.

**A declared procedure becomes work.** Reading a skill with `steps:` puts them in
the todo list, in order, waiting on each other — the mechanism `tool-todo`
already had, used by something that is not the model.

**The model cannot delete them.** `write_todos` replaces the whole list, so
without a rule the procedure lasts exactly until the model's next plan. Entries a
skill seeded carry a harness-issued `source`, and a write that drops, reorders or
rewords one is refused. It may add its own entries beside them and mark them
done; that is the difference between a procedure and a suggestion.

**The loop objects while work remains.** On `agent/turn-stopping` — the same
boundary `/autonomous` uses, by steering rather than by reaching into loop
state — a turn that would end with a startable step left is nudged instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import anyio
import anyio.lowlevel
import pytest
from stabilize_helpers import PROFILE, result_text, row, run_tool_calls, todo_call

from ph.agent.types import AgentOptions
from ph.keys import AGENTS, LLM, SESSIONS
from ph.llm.replay import tool_call_chunks
from ph.llm.types import GenerateOptions, text_of
from ph.seams.skills import discover_skills, rendered_skill
from ph.session import Session, SurfaceIntent
from ph.testing import FAKE_OPTIONS, MountProfile, log_event, run_tool, write_skill
from ph_stabilize.skill_steps import (
    MAX_NAMED,
    MAX_NUDGES,
    nudges_since,
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
    """One entry, in one spelling — including the harness-issued `source`."""
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
) -> Any:  # noqa: ANN401
    """A mounted deployment that has just read a three-step skill.

    `skill_budget` is the skill's own `max-nudges`; `profile_budget` the row's
    `maxNudges`. Either left out says nothing, so the default applies.
    """
    extra = STEPS if skill_budget is None else f"{STEPS}max-nudges: {skill_budget}\n"
    write_skill(tmp_path, "port", description="port a row", extra=extra, body="Do it.")
    planning, steering = ROWS
    if profile_budget is not None:
        steering = {**steering, "config": {"maxNudges": profile_budget}}
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
    """The declaration becomes entries, in order, waiting on each other.

    Sequential `requires` is the mechanism `tool-todo` already had — this row
    adds no ordering of its own, which is why `blocked_by` and the sidebar
    understand a seeded procedure without knowing skills exist.
    """
    _ctx, session, _agent = await _reading(mount, tmp_path)

    todos = todos_of(session)
    assert steps_of(todos) == ["survey the callers", "port the row", "gate it"]
    assert [one["requires"] for one in todos] == [[], ["survey the callers"], ["port the row"]]
    assert startable(todos) == ["survey the callers"], "one at a time, in the declared order"


async def test_reading_the_same_skill_twice_does_not_duplicate_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A model re-reads instructions it half-remembers; a second copy of the
    procedure would be a plan that can never be finished."""
    ctx, session, agent = await _reading(mount, tmp_path)
    await run_tool(ctx, "skill", {"name": "port"}, agent=agent, session=session)

    assert len(steps_of(todos_of(session))) == 3


def test_a_procedure_that_grew_seeds_only_what_is_new() -> None:
    """A skill edited mid-session adds its new step and re-adds nothing.

    The only thing the by-content filter does that the collision guard below does
    not: without it a grown procedure seeds nothing at all, because its *first*
    step is already an entry and the whole batch is refused.
    """
    current = [
        _entry("survey", "completed", source=SKILL),
        _entry("port", "in_progress", ["survey"], source=SKILL),
    ]

    grown = seeded(current, ["survey", "port", "gate it"])

    assert grown is not None
    assert steps_of(grown) == ["survey", "port", "gate it"]
    # It waits on the step the *skill* put before it (D13), which is `port` —
    # already in the list and still in progress. This assertion used to read
    # `== []`, on the argument that a new step "waits on nothing it did not
    # arrive with"; that was the chain being built over what is *missing* rather
    # than over what the skill declared, so a step the procedure put last became
    # immediately available. The ordering is the procedure's, not the gap's.
    assert grown[-1]["requires"] == ["port"]
    assert [one["status"] for one in grown[:2]] == ["completed", "in_progress"], "progress kept"


def test_a_step_whose_text_the_model_already_used_is_not_seeded() -> None:
    """`requires` names entries by content, so two entries with one text is a
    plan that cannot be reproduced — refused here rather than written and then
    refused by `_checked` on the model's next write."""
    assert seeded([_entry("port the row", "completed")], ["port the row"]) is None


# ------------------------------------------------------- the model's hands --


async def test_the_model_may_mark_a_seeded_step_done(mount: MountProfile, tmp_path: Path) -> None:
    """Marking progress is the one thing it *may* change — the whole point."""
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
    """A procedure is not a cage: its own plan lives alongside."""
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
    """**The rule the row exists for.**

    `write_todos` replaces the whole list, so without this a procedure lasts
    exactly until the model's next plan — and a `turn-stopping` listener
    enforcing against that list would be enforcing against nothing.

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
    """Order is the procedure. `requires` alone would not notice a swap between
    two steps that happen not to depend on each other."""
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
    """A pointer, not a second copy of the plan — the model already has the list.

    A twenty-line reminder every time a turn tries to end is how a steer becomes
    noise a model learns to skim.
    """
    text = steer_text([f"step {n}" for n in range(MAX_NAMED + 2)], blocked=4)

    assert text.count("'step ") == MAX_NAMED, "quoted names only; the blocked count is not one"
    assert "and 2 more" in text
    assert "4 further steps wait" in text, "counted prose agrees, via `count_of`"
    assert "1 further step waits" in steer_text(["a"], blocked=1), "and agrees at one"


EXAMPLES = Path(__file__).resolve().parents[3] / "docs" / "skills" / "self-steerings-examples"
"""The OpenMono playbooks ported as self-steering skills — meant to be run, not
only read, so they are gated the way a deployment would install them.

Here rather than in ph-core, which owns `discover_skills`: the bound that matters
to a *step* is `MAX_TODO_CONTENT`, and ph-core may not import the package that
declares it. This suite sees both."""

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
"""Enough to satisfy each one's required inputs. A generic placeholder is useless
because of the enums, which is the declaration doing its job."""


def test_the_ported_examples_all_load_and_render() -> None:
    """Eight worked skills, checked the way a deployment would install them.

    They are documentation that is also *input*: a profile points `paths:` at that
    directory and gets all eight. So the gate is `discover_skills` — the call a
    mount makes — because a scanner that refuses one logs a warning and drops it,
    which is exactly the failure a reader would never notice.

    Rendering too, not only loading. A body is scanned for undeclared placeholders
    only when something asks for it, so a typo'd `{{parameters.x}}` in any of these
    would sit there until the first model that read it got a tool error instead of
    instructions.

    And every step is checked against `MAX_TODO_CONTENT`, because a step is an
    entry in a todo list here rather than a paragraph of prompt — one playbook in
    the source is 510 characters, which `seeded` would refuse outright.
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


class _Stopping:
    """An agent, as far as `agent/turn-stopping` is concerned.

    The listener is driven through `ctx.serial` on the real waterfall rather than
    through a whole turn, because the loop has no step cap of its own — a fake
    provider that never marks a step done would be steered forever, which is the
    thing `MAX_NUDGES` exists to bound and cannot be used to test itself.

    `steer` **appends what the driver would append**, and that is load-bearing
    rather than decorative: `nudges_since` folds those very messages out of
    the log, so a stub that only collected them in a list would make the
    stand-down untestable and would model the boundary wrongly.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.steers: list[Any] = []

    def steer(self, message: Any) -> None:  # noqa: ANN401
        self.steers.append(message)
        log_event(self.session, "user/message", message.to_wire(), SurfaceIntent("append"))


async def _nudges(ctx: Any, stopping: _Stopping, tries: int) -> int:  # noqa: ANN401
    """How many of `tries` turn-stopping boundaries the row steered at."""
    before = len(stopping.steers)
    for _ in range(tries):
        await ctx.serial("agent/turn-stopping", stopping, 1)
    return len(stopping.steers) - before


async def test_a_turn_trying_to_end_with_a_procedure_unfinished_is_steered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The row's whole purpose, at the boundary the agent loop already fires.

    Sabotage: drop the `agent.steer(...)` call, or the `ctx.on` that registers
    this listener, and a model that read a three-step procedure stops after
    whatever it felt like doing first.
    """
    ctx, session, _agent = await _reading(mount, tmp_path)
    stopping = _Stopping(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(stopping.steers) == 1, "one nudge, not one per outstanding step"
    message = stopping.steers[0]
    assert "survey the callers" in text_of(message.content)
    assert message.source.plugin == "ph_stabilize.skill_steps", (
        "tagged, so the transcript does not read the harness's nudge as the person's"
    )


async def test_a_finished_procedure_lets_the_turn_end(mount: MountProfile, tmp_path: Path) -> None:
    """The other half, and the one a steer that never stands down would break."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    await run_tool_calls(
        ctx, session, todo_call("done", [_entry(step, status="completed") for step in STEP_TEXTS])
    )
    stopping = _Stopping(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "nothing is outstanding; the turn is allowed to end"


async def test_a_session_with_no_procedure_is_never_steered(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A row that is mounted must cost a session that does not use it nothing."""
    ctx = await mount(*ROWS, profile=PROFILE)
    session = ctx.require(SESSIONS).create("plain")
    ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await run_tool_calls(ctx, session, todo_call("own", [_entry("my own work")]))
    stopping = _Stopping(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "the model's own list is the model's to finish"


async def test_the_row_stands_down_when_its_nudges_change_nothing(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The ceiling `/autonomous` has and the first cut of this row did not.

    The loop has no step cap (`limits` ships with `turn_limit` unset, on
    purpose), so a model that will not mark a step done would be steered for the
    life of the session. Counted since the list last *changed*, which makes it a
    stall detector rather than a quota.

    Sabotage: drop the `nudges_since` check and the loop below never stops
    steering.
    """
    ctx, session, _agent = await _reading(mount, tmp_path)

    assert await _nudges(ctx, _Stopping(session), MAX_NUDGES + 2) == MAX_NUDGES, (
        "it stops after the plan has not moved"
    )


async def test_progress_on_the_plan_earns_more_nudges(mount: MountProfile, tmp_path: Path) -> None:
    """A run that is getting somewhere is never cut off — which is what makes the
    ceiling a stall detector rather than a budget."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    stopping = _Stopping(session)
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
    """D16 — the ceiling is the deployment's to choose, not a constant.

    Sabotage: compare against `MAX_NUDGES` in `keep_going` again and this
    steers three times.
    """
    ctx, session, _agent = await _reading(mount, tmp_path, profile_budget=1)

    assert await _nudges(ctx, _Stopping(session), 5) == 1


async def test_a_skill_budget_overrides_the_profile(mount: MountProfile, tmp_path: Path) -> None:
    """D16 — the author knows what a half-finished run of their procedure is worth.

    And it outlives the model's next plan: `write_todos` replaces the seeding
    `todo/write`, which is why the budget is a record of its own. Sabotage: drop
    the `skill-steps/budget` append, or read the budget off the latest
    `todo/write`, and one of the two counts below falls to the profile's 1.
    """
    ctx, session, _agent = await _reading(mount, tmp_path, skill_budget=4, profile_budget=1)
    stopping = _Stopping(session)
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
    """D16 — "a partial result is fine" is a thing an author can say."""
    ctx, session, _agent = await _reading(mount, tmp_path, skill_budget=0)

    assert steps_of(todos_of(session)) == STEP_TEXTS, "the procedure is still the plan"
    assert await _nudges(ctx, _Stopping(session), 3) == 0


async def test_a_later_procedure_without_a_budget_uses_the_profile(
    mount: MountProfile, tmp_path: Path
) -> None:
    """D16 — the procedure seeded last governs, and saying nothing means the
    profile's budget rather than whatever the skill before it asked for."""
    write_skill(tmp_path, "notes", description="take notes", extra="steps:\n  - write it down\n")
    ctx, session, agent = await _reading(mount, tmp_path, skill_budget=5, profile_budget=1)
    await run_tool(ctx, "skill", {"name": "notes"}, agent=agent, session=session)

    assert await _nudges(ctx, _Stopping(session), 5) == 1


@pytest.mark.parametrize("budget", [0, 2])
async def test_a_spent_ceiling_steers_for_exactly_the_nudge_budget(
    mount: MountProfile, tmp_path: Path, budget: int
) -> None:
    """D16 — a spent `limits` ceiling against an unfinished procedure, decided.

    `_turn` re-reads the inbox after `agent/turn-stopping`, so a steer keeps a
    turn alive past a ceiling that concluded it: the next step's call is denied,
    concludes again, and is steered again. That is left as it is, on purpose —
    "the budget is spent" and "the procedure is unfinished" are different facts,
    and the skill is what knows whether its outstanding steps are load-bearing.
    What bounds it is the nudge budget (the skill's own, else the profile's), so
    the model calls spent past the ceiling are exactly that budget.

    Driven through a real turn, because the interaction lives in the driver: a
    model that never marks a step done, re-reading the skill every step (a read
    that seeds nothing, so it cannot reset the count). Under a deadline, so a
    steer that never stands down fails rather than hangs — a cap on the
    adapter cannot do that, because whatever it answers instead is steered too.
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
    """All or nothing, because the alternative is a plan the model cannot write.

    A step past `MAX_TODO_CONTENT` seeds happily and then fails `WriteTodosArgs`
    on *every* later call — while `_carried` refuses any write that drops it. The
    model would be locked out of its own list with the row steering it onward.
    Length is not the author's to control either: `{{parameters.x}}` renders a
    model-supplied argument into the step's text.
    """
    assert seeded([], ["fine", "x" * (MAX_TODO_CONTENT + 1)]) is None


def test_seeding_past_the_list_cap_seeds_nothing() -> None:
    """The other half of the same lockout, from the other bound."""
    full = [_entry(f"mine {n}") for n in range(MAX_TODOS - 1)]

    assert seeded(full, ["one more"]) is not None, "there is room for exactly one"
    assert seeded(full, ["one more", "and another"]) is None


def test_a_chain_points_at_one_step_at_a_time() -> None:
    """Which is also why `keep_going`'s empty-`ready` branch is a *shape* guard.

    `seeded` writes a sequential chain, so while anything in it is unfinished its
    earliest link is waiting on nothing — there is always exactly one thing to
    name, and the branch that would decline to steer cannot be reached by the
    entries this row writes. Asserting it here rather than through the listener
    keeps a gate from claiming to cover a state the feature cannot produce.
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
    """D13 — the chain was built over the missing steps, so it linked *over* one.

    A skill re-read mid-session is ordinary, and so is finding one of its steps
    already in the list — the model may have written the same text itself, or a
    previous seed may have been partly carried. With the chain built over
    `wanted`, a procedure `one → two → three` whose middle step is present seeds
    `three` waiting on `one`: the ordering the skill declared is silently
    replaced by one that lets `three` start while `two` is still pending, which
    is the whole thing `requires` exists to prevent.

    The predecessor is always nameable, which is why this needs no fallback:
    every step is either already in the list or being added in the same batch.
    """
    current = [_entry("two", "pending", source=SKILL)]

    grown = seeded(current, ["one", "two", "three"])

    assert grown is not None
    by_content = {one["content"]: one for one in grown}
    assert by_content["one"]["requires"] == []
    assert by_content["three"]["requires"] == ["two"], "the chain skipped the step that was there"
