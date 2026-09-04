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

from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, result_text, run_tool_calls, todo_call

from ph.llm.types import text_of
from ph.seams.skills import discover_skills, rendered_skill
from ph.session import SurfaceIntent
from ph.testing import FAKE_OPTIONS, run_tool, write_skill
from ph_stabilize.skill_steps import MAX_NAMED, MAX_NUDGES, seeded, steer_text
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


async def _reading(mount: Any, tmp_path: Any) -> Any:
    """A mounted deployment that has just read a three-step skill."""
    write_skill(tmp_path, "port", description="port a row", extra=STEPS, body="Do it.")
    ctx = await mount(
        *ROWS, {"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}}, profile=PROFILE
    )
    session = ctx.sessions.create("procedure")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    await run_tool(ctx, "skill", {"name": "port"}, agent=agent, session=session)
    return ctx, session, agent


# ------------------------------------------------------------ the seeding --


async def test_reading_a_skill_turns_its_steps_into_work(mount: Any, tmp_path: Any) -> None:
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
    mount: Any, tmp_path: Any
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
    assert grown[-1]["requires"] == [], "the new step waits on nothing it did not arrive with"
    assert [one["status"] for one in grown[:2]] == ["completed", "in_progress"], "progress kept"


def test_a_step_whose_text_the_model_already_used_is_not_seeded() -> None:
    """`requires` names entries by content, so two entries with one text is a
    plan that cannot be reproduced — refused here rather than written and then
    refused by `_checked` on the model's next write."""
    assert seeded([_entry("port the row", "completed")], ["port the row"]) is None


# ------------------------------------------------------- the model's hands --


async def test_the_model_may_mark_a_seeded_step_done(mount: Any, tmp_path: Any) -> None:
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


async def test_the_model_may_add_its_own_entries_beside_them(mount: Any, tmp_path: Any) -> None:
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


async def test_a_write_that_drops_a_seeded_step_is_refused(mount: Any, tmp_path: Any) -> None:
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


async def test_a_write_that_reorders_them_is_refused(mount: Any, tmp_path: Any) -> None:
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
    rather than decorative: `nudges_since_plan` folds those very messages out of
    the log, so a stub that only collected them in a list would make the
    stand-down untestable and would model the boundary wrongly.
    """

    def __init__(self, session: Any) -> None:
        self.session = session
        self.steers: list[Any] = []

    def steer(self, message: Any) -> None:
        self.steers.append(message)
        self.session.append("user/message", message.to_wire(), SurfaceIntent("append"))


async def test_a_turn_trying_to_end_with_a_procedure_unfinished_is_steered(
    mount: Any, tmp_path: Any
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


async def test_a_finished_procedure_lets_the_turn_end(mount: Any, tmp_path: Any) -> None:
    """The other half, and the one a steer that never stands down would break."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    await run_tool_calls(
        ctx, session, todo_call("done", [_entry(step, status="completed") for step in STEP_TEXTS])
    )
    stopping = _Stopping(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "nothing is outstanding; the turn is allowed to end"


async def test_a_session_with_no_procedure_is_never_steered(mount: Any, tmp_path: Any) -> None:
    """A row that is mounted must cost a session that does not use it nothing."""
    ctx = await mount(*ROWS, profile=PROFILE)
    session = ctx.sessions.create("plain")
    ctx.agents.create(session, FAKE_OPTIONS)
    await run_tool_calls(ctx, session, todo_call("own", [_entry("my own work")]))
    stopping = _Stopping(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert stopping.steers == [], "the model's own list is the model's to finish"


async def test_the_row_stands_down_when_its_nudges_change_nothing(
    mount: Any, tmp_path: Any
) -> None:
    """The ceiling `/autonomous` has and the first cut of this row did not.

    The loop has no step cap (`limits` ships with `turn_limit` unset, on
    purpose), so a model that will not mark a step done would be steered for the
    life of the session. Counted since the list last *changed*, which makes it a
    stall detector rather than a quota.

    Sabotage: drop the `nudges_since_plan` check and the loop below never stops
    steering.
    """
    ctx, session, _agent = await _reading(mount, tmp_path)
    stopping = _Stopping(session)

    for _ in range(MAX_NUDGES + 2):
        await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(stopping.steers) == MAX_NUDGES, "it stops after the plan has not moved"


async def test_progress_on_the_plan_earns_more_nudges(mount: Any, tmp_path: Any) -> None:
    """A run that is getting somewhere is never cut off — which is what makes the
    ceiling a stall detector rather than a budget."""
    ctx, session, _agent = await _reading(mount, tmp_path)
    stopping = _Stopping(session)
    for _ in range(MAX_NUDGES):
        await ctx.serial("agent/turn-stopping", stopping, 1)

    steps = steps_of(todos_of(session))
    await run_tool_calls(
        ctx,
        session,
        todo_call("moved", [_entry(steps[0], "completed"), *(_entry(one) for one in steps[1:])]),
    )
    await ctx.serial("agent/turn-stopping", stopping, 1)

    assert len(stopping.steers) == MAX_NUDGES + 1
    assert "port the row" in text_of(stopping.steers[-1].content), "and it points at the next one"


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
