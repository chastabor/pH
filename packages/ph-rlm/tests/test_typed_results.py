"""A child that owes its parent a value in a set shape (typed child results, B1-B3).

The gate: a categorizing child returns a value that matches its `output`, through
`submit_result` or as a reply that is the JSON object itself; a value that does not match
is corrected; and a child that returns none, after one reminder, fails with its text
kept. Around it: where the shape comes from (the spawn's own, else its skills'), where
the value goes (the child's own log, its state, its roster row and the notice its parent
is handed), and that the return tool is the child's alone.
"""

from __future__ import annotations

import json
from typing import Any, TypeAlias

import pytest
from rlm_fixtures import BINDINGS_ROW, PROVIDER_ROW

from ph.agent.types import AgentDriver, AgentOptions
from ph.cordis import Context
from ph.json import JsonObject, as_obj, thaw_json
from ph.keys import AGENTS, LLM, SESSIONS, SKILLS, SUBAGENTS, TOOLS
from ph.seams.subagents import (
    RESULT_TOOL,
    RETURNED,
    ChildState,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentSpawnError,
)
from ph.testing import (
    KIND_OUTPUT,
    MountProfile,
    RecordedStep,
    ReplayAdapter,
    not_none,
    run_tool,
    simple_tool,
    skill,
    text_chunks,
    tool_call_chunks,
)
from ph_rlm.bindings import WAIT_TOOL
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio

KIND = KIND_OUTPUT
"""What a categorizing child returns: which kind of report it read."""

Step: TypeAlias = str | tuple[str, dict[str, Any]]
"""One model reply: text, or a call by name with its arguments."""


def _replies(*steps: Step) -> ReplayAdapter:
    """A child's model: one scripted reply per request, and `REPLAY_EXHAUSTED` past the
    last, so a child that asks again where it should have stopped is caught at once."""
    return ReplayAdapter(
        steps=[
            RecordedStep(
                turn=1,
                step=index,
                chunks=text_chunks(step)
                if isinstance(step, str)
                else tool_call_chunks(f"call-{index}", step[0], json.dumps(step[1])),
            )
            for index, step in enumerate(steps)
        ]
    )


def _offered(model: ReplayAdapter, index: int = 0) -> set[str]:
    """The tools the child was offered on one request."""
    return {tool.name for tool in model.requests[index].tools}


def _said_to(model: ReplayAdapter, index: int) -> str:
    """Everything the child was sent on one request, tool results included."""
    return "\n".join(message.model_dump_json() for message in model.requests[index].messages)


async def _parent(mount: MountProfile, *steps: Step) -> tuple[Context, AgentDriver, ReplayAdapter]:
    """A parent on a scripted route its children inherit."""
    ctx = await mount(dict(PROVIDER_ROW), dict(BINDINGS_ROW))
    model = _replies(*steps)
    ctx.require(LLM).register_adapter(("scripted",), model)
    session = ctx.require(SESSIONS).create("parent")
    options = AgentOptions(provider="scripted", model="s1")
    return ctx, ctx.require(AGENTS).create(session, options), model


async def _spawn(ctx: Context, parent: AgentDriver, **kwargs: Any) -> SubagentRun:  # noqa: ANN401
    return await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="sort this report", parent=parent, **kwargs)
    )


async def _settled(ctx: Context, run: SubagentRun) -> tuple[SubagentResult, ChildState]:
    """The child's result, and its state as its own log tells it."""
    result = await not_none(run.result)()
    return result, not_none(ctx.require(SUBAGENTS).state(run.session_id))


def _returned(ctx: Context, run: SubagentRun) -> list[JsonObject]:
    """Every value recorded in the child's own log."""
    child = not_none(ctx.require(SESSIONS).get(run.session_id))
    return [as_obj(thaw_json(dict(event.data))) for event in child.events if event.type == RETURNED]


# ------------------------------------------------------------------ the gate --


async def test_a_child_returns_its_value_through_its_tool(mount: MountProfile) -> None:
    """The gate's first half. A valid call ends the child's turn, so it asks nothing
    more of its model, and the value reaches every reader: the result a waiter gets,
    the child's own log, its state and roster row, and the notice its parent is handed.

    Sabotage: leave `conclude_turn` out of the tool, and the child asks its model again;
    drop `value` from the ending, the fold or the notice, and that reader goes without.
    """
    ctx, parent, script = await _parent(mount, (RESULT_TOOL, {"kind": "bug"}))

    run = await _spawn(ctx, parent, output=KIND)
    result, state = await _settled(ctx, run)

    assert (result.status, result.value) == ("done", {"kind": "bug"})
    assert len(script.requests) == 1, "the call ended the turn"
    assert RESULT_TOOL in _offered(script)
    assert _returned(ctx, run) == [{"value": {"kind": "bug"}, "via": "tool"}]
    assert state.value == {"kind": "bug"}
    assert as_obj(thaw_json(state.to_wire()))["value"] == {"kind": "bug"}
    assert "output" not in state.to_wire(), "the row leaves out the schema its reader holds"
    assert '"kind":"bug"' in not_none(state.notice).text


async def test_a_value_that_does_not_match_is_corrected_in_the_same_turn(
    mount: MountProfile,
) -> None:
    """A bad call is a tool error naming what is wrong, as any tool's arguments are, and
    the model's next step fixes it. Sabotage: record the value unchecked, and `chore`
    is returned."""
    ctx, parent, script = await _parent(
        mount, (RESULT_TOOL, {"kind": "chore"}), (RESULT_TOOL, {"kind": "feature"})
    )

    run = await _spawn(ctx, parent, output=KIND)
    result, _state = await _settled(ctx, run)

    assert result.value == {"kind": "feature"}
    assert "kind: must be one of" in _said_to(script, 1)
    assert [one["value"] for one in _returned(ctx, run)] == [{"kind": "feature"}]


async def test_a_reply_that_is_the_json_object_is_the_value(mount: MountProfile) -> None:
    """A JSON value is text in the end: a child that answers with the object itself,
    fenced or not, has returned it. Sabotage: count only `submit_result`, and this
    child is reminded and then fails."""
    ctx, parent, script = await _parent(mount, '```json\n{"kind": "feature"}\n```')

    run = await _spawn(ctx, parent, output=KIND)
    result, state = await _settled(ctx, run)

    assert (result.status, result.value) == ("done", {"kind": "feature"})
    assert len(script.requests) == 1, "no reminder"
    assert _returned(ctx, run) == [{"value": {"kind": "feature"}, "via": "text"}]
    assert state.value == {"kind": "feature"}


async def test_a_child_that_returns_no_value_fails_with_its_text_kept(
    mount: MountProfile,
) -> None:
    """The gate's second half: the value is a contract the parent may depend on, so a
    child with none is a failure, not prose with a note. It is reminded once first, and
    its text is kept for the parent to weigh.

    Sabotage: end it `done`, or send no reminder, or remind it without end.
    """
    ctx, parent, script = await _parent(mount, "It looks like a bug to me.", "Still a bug.")

    run = await _spawn(ctx, parent, output=KIND)
    result, state = await _settled(ctx, run)

    assert (result.status, result.answer) == ("error", "Still a bug.")
    assert "returned no value in the shape asked for" in not_none(result.error)
    assert len(script.requests) == 2, "one reminder, then the ending"
    assert "You have not returned your result" in _said_to(script, 1)
    assert state.status == "error" and state.value is None
    assert "Still a bug." in not_none(state.notice).text


async def test_a_reminded_child_that_then_returns_its_value_succeeds(
    mount: MountProfile,
) -> None:
    ctx, parent, _script = await _parent(mount, "A bug, I think.", (RESULT_TOOL, {"kind": "bug"}))

    run = await _spawn(ctx, parent, output=KIND)
    result, _state = await _settled(ctx, run)

    assert (result.status, result.value) == ("done", {"kind": "bug"})


# ------------------------------------------------------------ where it comes from --


async def test_a_skill_s_output_reaches_its_child_and_its_admission(
    mount: MountProfile,
) -> None:
    """A categorizing skill says what its child returns, and the admission records the
    shape, so a readmitted child still owes it. Sabotage: drop `resolve_output` from
    `start`, and the child owes nothing."""
    ctx, parent, script = await _parent(mount, (RESULT_TOOL, {"kind": "bug"}))
    ctx.require(SKILLS).register(skill("sort", output=KIND))

    run = await _spawn(ctx, parent, skills=("sort",))
    _result, state = await _settled(ctx, run)

    assert not_none(state.admission).output == KIND
    assert RESULT_TOOL in _offered(script)
    assert "You owe your result in a set shape" in not_none(script.requests[0].system)


async def test_a_spawn_s_own_output_beats_its_skill_s(mount: MountProfile) -> None:
    ctx, parent, _script = await _parent(mount, (RESULT_TOOL, {"label": "x"}))
    ctx.require(SKILLS).register(skill("sort", output=KIND))
    labeled = {"type": "object", "properties": {"label": {"type": "string"}}}

    run = await _spawn(ctx, parent, skills=("sort",), output=labeled)
    result, state = await _settled(ctx, run)

    assert not_none(state.admission).output == labeled
    assert result.value == {"label": "x"}


async def test_skills_that_declare_different_outputs_are_refused(mount: MountProfile) -> None:
    ctx, parent, _script = await _parent(mount)
    ctx.require(SKILLS).register(skill("sort", output=KIND))
    ctx.require(SKILLS).register(skill("tag", output={"type": "object"}))

    with pytest.raises(SubagentSpawnError, match="declare different outputs"):
        await _spawn(ctx, parent, skills=("sort", "tag"))


@pytest.mark.parametrize(
    ("output", "said"),
    [
        ({"type": "array"}, "root is not `type: object`"),
        ({"type": "object", "oneOf": []}, "keywords pH does not check: oneOf"),
    ],
)
async def test_an_output_ph_cannot_check_is_refused(
    mount: MountProfile, output: dict[str, Any], said: str
) -> None:
    """A schema pH would half-check is refused rather than trusted. Sabotage: skip
    `object_schema_problems` for a spawn's own output, and it is admitted."""
    ctx, parent, _script = await _parent(mount)

    with pytest.raises(SubagentSpawnError, match=said):
        await _spawn(ctx, parent, output=output)

    assert ctx.require(SUBAGENTS).children(not_none(parent.session).id) == {}


async def test_a_spawn_s_output_that_may_not_constrain_is_warned_about(
    mount: MountProfile, caplog: pytest.LogCaptureFixture
) -> None:
    """The same warning a skill's gets when it loads, at the spawn that names its own.
    Sabotage: warn on the skill path alone."""
    ctx, parent, _script = await _parent(mount, (RESULT_TOOL, {"kind": "x"}))

    with caplog.at_level("WARNING", logger="ph.seams.subagents"):
        run = await _spawn(ctx, parent, output={"type": "object", "required": ["kind"]})
        await _settled(ctx, run)

    assert "kind have no typed `properties` entry" in caplog.text


# ------------------------------------------------------------- whose tool it is --


async def test_only_a_child_that_owes_a_value_is_offered_the_tool(mount: MountProfile) -> None:
    """Registered on the child's own scope: its parent has no such tool, and nor does
    a sibling that owes nothing. Sabotage: register it on the mount, and every agent
    has it."""
    ctx, parent, script = await _parent(mount, "plain answer")

    run = await _spawn(ctx, parent)
    await _settled(ctx, run)

    assert RESULT_TOOL not in _offered(script)
    assert RESULT_TOOL not in ctx.require(TOOLS).names(scope=parent.ctx)


async def test_a_parent_s_return_tool_is_not_something_it_holds(mount: MountProfile) -> None:
    """A child's own child does not inherit its parent's way of answering, so a value
    it returned would reach nobody. Sabotage: drop the filter in `held_by`, and the
    tool is among what the parent could hand on."""
    ctx, parent, _script = await _parent(mount)
    ctx.require(TOOLS).register(simple_tool(RESULT_TOOL), scope=parent.ctx)

    _skills, tools = ctx.require(SUBAGENTS).held_by(SubagentRequest(prompt="x", parent=parent))

    assert RESULT_TOOL in ctx.require(TOOLS).names(scope=parent.ctx)
    assert RESULT_TOOL not in tools


# -------------------------------------------------------------- waiting for one --


async def _wait(ctx: Context, parent: AgentDriver, child_id: str) -> Any:  # noqa: ANN401
    """`rlm.wait(child_id)` as the parent calls it."""
    return await run_tool(
        ctx, WAIT_TOOL, {"child_id": child_id}, agent=parent, session=parent.session
    )


async def test_a_parent_waits_for_its_child_s_value(mount: MountProfile) -> None:
    """The one wait there is, for a step that needs the value: it hands back what the
    child returned, checked. Sabotage: answer with the child's text, and the value is
    lost."""
    ctx, parent, _script = await _parent(mount, (RESULT_TOOL, {"kind": "bug"}))
    run = await _spawn(ctx, parent, output=KIND)

    waited = await _wait(ctx, parent, run.id)

    assert waited.value["value"] == {"kind": "bug"}
    assert '{"kind":"bug"}' in repr(waited.content)


async def test_a_child_settled_before_the_wait_is_answered_from_its_log(
    mount: MountProfile,
) -> None:
    """A child this process no longer holds, as after a restart, is answered from its
    own log. Sabotage: refuse a child with no live run."""
    ctx, parent, _script = await _parent(mount, (RESULT_TOOL, {"kind": "feature"}))
    run = await _spawn(ctx, parent, output=KIND)
    await _settled(ctx, run)
    ctx.require(SUBAGENTS).forget(run.id)

    waited = await _wait(ctx, parent, run.id)

    assert waited.value["value"] == {"kind": "feature"}


async def test_a_wait_for_a_child_that_failed_fails_with_its_text_kept(
    mount: MountProfile,
) -> None:
    ctx, parent, _script = await _parent(mount, "A bug.", "Still a bug.")
    run = await _spawn(ctx, parent, output=KIND)

    waited = await _wait(ctx, parent, run.id)

    assert waited.is_error
    assert "ended as error" in repr(waited.content)
    assert "Still a bug." in repr(waited.content)


async def test_an_agent_cannot_wait_on_a_child_not_its_own(mount: MountProfile) -> None:
    """Another agent's live child is not yours to wait on, nor is an id no child has.
    Sabotage: look the run up without asking whose child it is."""
    ctx, parent, _script = await _parent(mount, (RESULT_TOOL, {"kind": "bug"}))
    run = await _spawn(ctx, parent, output=KIND)
    stranger = ctx.require(AGENTS).create(
        ctx.require(SESSIONS).create("stranger"), AgentOptions(provider="scripted", model="s1")
    )

    theirs = await _wait(ctx, stranger, run.id)
    nobody = await _wait(ctx, parent, "child-of-nobody")

    assert theirs.is_error and "you have no child" in repr(theirs.content)
    assert nobody.is_error and "you have no child" in repr(nobody.content)
