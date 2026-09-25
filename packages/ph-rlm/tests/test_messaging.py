"""Agent-to-agent messages, and the boundary on them (P3-12, C7/C8).

The load-bearing claim is the asymmetry: the **family boundary** is a monotonic
guard that no later listener can re-permit and whose denial ends the run, while
the **rate limit** is a failure the program handles. Both were one check inside a
comm handler in prime-agent; separating them is what makes the first
unroutable-around and the second survivable.
"""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from typing import Any

import pytest
from rlm_fixtures import MESSAGING_ROW, PROVIDER_ROW, MountedRuntime, logs_after_a_crash

from ph.agent.types import AgentDriver
from ph.cordis import DEPLOYMENT, Context
from ph.keys import AGENTS, JOBS, SESSIONS, SUBAGENTS, TOOLS
from ph.llm.types import text_of
from ph.persistence import resume_session
from ph.seams.subagents import SubagentRequest, family_reach, reachable_family
from ph.session import Session, SessionEvent, outcome_of
from ph.session.json import freeze_json_value
from ph.session.kinds import TOOL_DISPATCH
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    log_interrupted_call,
    not_none,
    result_text,
    run_tool,
    stored_events,
)
from ph.tools import Allow, NotDone
from ph_rlm.keys import RLM_CHILDREN
from ph_rlm.messaging import (
    OBSERVE_GET_TOOL,
    OUT_OF_REACH,
    SEND_TOOL,
    render_received,
)
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio


ROWS: list[dict[str, Any]] = [PROVIDER_ROW, MESSAGING_ROW]


@pytest.fixture
def family_ctx(mount: MountProfile) -> Callable[..., Any]:
    """`await family_ctx()` → `(ctx, parent_session, parent)` with messaging on."""

    async def build(**config: object) -> tuple[Any, Any, Any]:
        rows = [dict(ROWS[0]), dict(ROWS[1])]
        if config:
            rows[1]["config"] = config
        ctx = await mount(*rows)
        session = ctx.require(SESSIONS).create("parent")
        return ctx, session, ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    return build


async def _spawn(ctx: Context, parent: Any, name: str) -> Any:  # noqa: ANN401
    return await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt=f"work on {name}", parent=parent, name=name)
    )


async def _siblings(
    family_ctx: MountedRuntime,
    **config: object,
) -> tuple[Any, Any, Any, Any]:
    """Two root agents, which the reach rule makes siblings of each other.

    Used for the limit tests because they must not race a child's completion:
    a spawned child settles at an unpredictable await point and injects a notice
    into its parent's inbox, which is itself pending input.
    """
    ctx, first_session, first = await family_ctx(**config)
    second_session = ctx.require(SESSIONS).create("other-root")
    return ctx, first_session, first, ctx.require(AGENTS).create(second_session, FAKE_OPTIONS)


def _agent(ctx: Context, run: Any) -> AgentDriver:  # noqa: ANN401
    agent = ctx.require(AGENTS).get(run.session_id)
    assert agent is not None, "the child agent is not running"
    return agent


_CALLS = itertools.count(1)


async def _send(ctx: Context, sender: Any, session: Session, **arguments: Any) -> Any:  # noqa: ANN401
    """One send, as its own call: a relayed message's id is derived from the call
    (`relay_message_id`), and a call is never made twice."""
    call_id = f"send-{next(_CALLS)}"
    return await run_tool(ctx, SEND_TOOL, arguments, agent=sender, session=session, call_id=call_id)


# ------------------------------------------------------------------- reach --


def test_the_reach_rule_and_its_enumeration_are_one_implementation() -> None:
    """`reachable_family` is derived from `family_reach`, so the guard that
    refuses a send and the roster that offers one cannot disagree."""

    class _Header:
        def __init__(self, parent: str | None) -> None:
            self.parent_session = parent

    class _Session:
        def __init__(self, session_id: str, parent: str | None) -> None:
            self.id = session_id
            self.header = _Header(parent)

    sessions = [
        _Session("root", None),
        _Session("other-root", None),
        _Session("a", "root"),
        _Session("b", "root"),
        _Session("grandchild", "a"),
    ]
    assert reachable_family(sessions, "a") == {
        "a": "self",
        "root": "parent",
        "b": "sibling",
        "grandchild": "child",
    }
    # No grandparent, no uncle — and the enumeration agrees with the predicate
    # for every pair, which is the property that matters.
    for one in sessions:
        reach = reachable_family(sessions, one.id)
        for two in sessions:
            predicate = family_reach(
                sender_parent=one.header.parent_session,
                sender_id=one.id,
                target_parent=two.header.parent_session,
                target_id=two.id,
            )
            assert (two.id in reach) is predicate


async def test_a_child_reaches_its_parent(family_ctx: MountedRuntime) -> None:
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)

    child_session = ctx.require(SESSIONS).get(run.session_id)
    assert child_session is not None
    result = await _send(
        ctx,
        child,
        child_session,
        message="found it",
        receiver_role="parent",
    )
    assert result.is_error is False
    assert result.value["receiverId"] == session.id
    assert result.value["receiverRole"] == "parent"
    assert result.value["deliveryStatus"] in {"delivered", "queued"}


async def test_the_message_reaches_the_target_verbatim(family_ctx: MountedRuntime) -> None:
    """Framing is the sender's; a harness that rewrote the body would make the
    log stop saying what the target actually read."""
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)

    body = "the bug is in parser.py line 40"
    child_session = ctx.require(SESSIONS).get(run.session_id)
    assert child_session is not None
    await _send(ctx, child, child_session, message=body, receiver_role="parent")
    delivered = [
        repr(event.data)
        for event in session.events
        if event.type == "agent/inbox/spliced" and "Agent-to-agent" in repr(event.data)
    ]
    assert delivered, "the parent never received it"
    assert body in delivered[0]
    assert "[from child:scout]" in delivered[0]
    assert f"From: {run.session_id}" in delivered[0]


async def test_a_message_is_on_the_receivers_disk_before_the_sender_is_told(
    family_ctx: MountedRuntime,
) -> None:
    """L6. The receiver's record of a message is written before `send` returns.

    `steer` records the message in the receiver's log, in memory, and the sender's
    receipt goes into the sender's log, and each used to reach disk at its own
    agent's next flush. A receiver in the middle of a long tool call flushes last,
    so a crash could leave the sender's log saying `delivered` and the receiver's
    saying nothing: a message lost without anyone being told. With the receiver
    written first, the worst a crash leaves is the sender's call unresolved, which
    repair reports as outcome unknown.

    Asked of the store as `send` returns, through the Protocol: what a resume of the
    receiver would be handed. Siblings, so no child's completion flushes anything.

    Sabotage: drop the write after `steer` and the receiver's stored log has no
    splice when the sender holds its receipt.
    """
    ctx, session, sender, target = await _siblings(family_ctx)
    body = "the fixture moved to tests/data"
    receipt = await _send(ctx, sender, session, message=body, receiver_role="sibling")
    assert receipt.is_error is False

    stored = stored_events(ctx, not_none(target.session).id)
    spliced = [event for event in stored if event.type == "agent/inbox/spliced"]
    assert spliced, "the receiver's log on disk has no record of the message"
    assert body in repr(spliced[-1].data)


async def _reconciled(
    ctx: Context, session: Session, call_id: str, arguments: dict[str, Any]
) -> Any:  # noqa: ANN401
    """What the send tool says, as a resume would ask it, about the call `call_id`."""
    record = SessionEvent(
        type="tool/call",
        seq=0,
        time=0,
        data=freeze_json_value(
            {"callId": call_id, "name": SEND_TOOL, "arguments": json.dumps(arguments)}
        ),
    )
    return await ctx.require(TOOLS).reconciled(record, session, scope=DEPLOYMENT)


async def test_a_send_is_found_in_its_receivers_log_or_ruled_out(
    family_ctx: MountedRuntime,
) -> None:
    """L6b. The send answers for its own call from the receiver's log.

    The message's id is derived from the call, so after a crash the tool can look for
    it where it would be. Found, the model is shown a receipt instead of being told to
    decide whether to send again. Absent from every log the send could have reached,
    the send did not happen.

    Sabotage: mint the message id with `new_message_id` again and the delivered
    message is not found.
    """
    ctx, _session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)
    child_session = not_none(ctx.require(SESSIONS).get(run.session_id))
    arguments = {"message": "found it", "receiver_role": "parent"}
    sent = await run_tool(ctx, SEND_TOOL, arguments, agent=child, session=child_session)
    assert sent.is_error is False

    found = await _reconciled(ctx, child_session, "call-1", arguments)
    assert isinstance(found, tuple), "the delivered message was not found"
    assert text_of(list(found)) == f"recorded to {parent.id} (parent)"

    never = await _reconciled(ctx, child_session, "call-2", arguments)
    assert isinstance(never, NotDone), "every log it could have reached lacks it"


async def test_a_roots_send_to_a_sibling_can_be_found_but_not_ruled_out(
    family_ctx: MountedRuntime,
) -> None:
    """L6b. A root's siblings are the other roots, which only the live ones stand for
    after a restart, so a message missing from them may be in a root not yet resumed.

    Sabotage: treat the live roots as every root and the absent send reads not done.
    """
    ctx, session, sender, _target = await _siblings(family_ctx)
    arguments = {"message": "halves", "receiver_role": "sibling"}
    await run_tool(ctx, SEND_TOOL, arguments, agent=sender, session=session)

    assert isinstance(await _reconciled(ctx, session, "call-1", arguments), tuple)
    unsure = await _reconciled(ctx, session, "call-2", arguments)
    assert unsure is None, "an absent send to a root's sibling is unknown, not ruled out"


async def test_a_send_a_crashed_cell_made_is_reported_delivered_after_a_restart(
    family_ctx: MountedRuntime, mount: MountProfile
) -> None:
    """L6b, end to end: a program that sent a message and was then interrupted.

    Under Code Mode the send is a dispatch inside the cell, so the model reads only
    the cell's result, which used to say its outcome was unknown. The receiver's log
    holds the message (L6), so on resume the cell's result says the send happened,
    and the model need not send it twice.

    Resumed by a fresh deployment over a snapshot of the logs, as a restarted daemon
    would resume it: the first one still holds the leases.
    """
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    arguments = {"message": "look in tests/data", "receiver_role": "child"}
    sent = await run_tool(
        ctx, SEND_TOOL, arguments, agent=parent, session=session, call_id="c1:code:0"
    )
    assert sent.is_error is False
    code = json.dumps({"code": "await agent_message.send('look in tests/data', 'child')"})
    log_interrupted_call(session, "run_code", code, dispatches=[(SEND_TOOL, arguments)])
    await ctx.require(SESSIONS).flush(session)
    snapshot = str(logs_after_a_crash())
    fresh = await mount(*ROWS, {"id": "session-persistence", "config": {"root": snapshot}})

    revived = await resume_session(fresh, session.id)

    settle = not_none(revived.latest("tool/code-dispatch"))
    assert outcome_of(TOOL_DISPATCH, settle) == "done"
    assert f"`{SEND_TOOL}` happened: recorded to {run.session_id} (child)" in result_text(
        revived, "c1"
    )


async def test_siblings_reach_each_other(family_ctx: MountedRuntime) -> None:
    """Two roots are siblings under the rule, which is what lets two top-level
    agents in one deployment talk."""
    ctx, session, sender, target = await _siblings(family_ctx)
    ok = await _send(ctx, sender, session, message="halves", receiver_role="sibling")
    assert ok.is_error is False
    assert ok.value["receiverId"] == target.id
    assert ok.value["receiverRole"] == "sibling"


async def test_addressing_a_settled_child_wakes_it(family_ctx: MountedRuntime) -> None:
    """P3-13: a completed child stays addressable, and a send is the trigger.

    Settlement releases the agent — which is what holds an inbox — but the
    session, the log and the roster row all survive it. So addressing the child
    rehydrates it rather than telling the sender to go read a transcript.
    """
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    await ctx.drain()
    assert ctx.require(AGENTS).get(run.session_id) is None, "the child should have settled"

    result = await _send(
        ctx, parent, session, message="are you there", receiver_role="child", receiver_name="scout"
    )
    assert result.is_error is False
    assert result.value["receiverId"] == run.session_id
    # Checked before draining: the woken child settles again once its job runs.
    assert ctx.require(AGENTS).get(run.session_id) is not None, "the child was not woken"

    await ctx.drain()
    # The roster says it was *woken*, distinctly from a child still on its first
    # task, so a parent reading the roster can tell the two apart.
    statuses = [
        event.data["status"]
        for event in session.events
        if event.type == "subagent/status" and event.data["runId"] == run.id
    ]
    assert statuses == ["running", "done", "running", "done"]
    # `running` with a *cause*, not a `rehydrated` status: the roster folds
    # status last-write-wins, so a woken child that is working must still read as
    # running to anything that branches on it.
    causes = [
        event.data.get("cause")
        for event in session.events
        if event.type == "subagent/status" and event.data["runId"] == run.id
    ]
    assert causes == [None, None, "rehydrated", None]


async def test_repeated_wakes_do_not_accrete_jobs(family_ctx: MountedRuntime) -> None:
    """A drive job is an effect of the delegation, released when the child settles.

    Every message to a settled child wakes it, and each wake starts a job. Without
    an owner that is one permanent table entry per message; with one it is none.
    """
    ctx, session, parent = await family_ctx(rateCapacity=99)
    await _spawn(ctx, parent, "scout")
    await ctx.drain()

    for index in range(4):
        sent = await _send(
            ctx,
            parent,
            session,
            message=f"ping {index}",
            receiver_role="child",
            receiver_name="scout",
        )
        assert sent.is_error is False
        await ctx.drain()

    assert [job for job in ctx.require(JOBS).list() if job.kind == "subagent"] == []


async def test_disposing_the_parent_abandons_a_running_drive(family_ctx: MountedRuntime) -> None:
    """The other half: work still in flight when its owner goes is canceled."""
    ctx, _session, parent = await family_ctx()
    await _spawn(ctx, parent, "scout")
    running = [job for job in ctx.require(JOBS).list() if job.kind == "subagent"]
    assert running, "the drive job was never registered"

    await ctx.require(AGENTS).dispose(parent.id)
    assert [job for job in ctx.require(JOBS).list() if job.kind == "subagent"] == []
    assert running[0].token.canceled


async def test_a_revoked_child_is_not_quietly_revived(family_ctx: MountedRuntime) -> None:
    """The tombstone is the parent's record that it revoked the child; waking one
    behind that record would make the record false."""
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    await ctx.drain()
    assert await ctx.require(RLM_CHILDREN).delete(session, run.id, reason="user") is True

    # Both doors: the service refuses because `forget()` dropped the run, and the
    # provider refuses because `_release` dropped the child.
    assert await ctx.require(SUBAGENTS).rehydrate(run.id) is False
    assert await ctx.require(RLM_CHILDREN).rehydrate(run.id) is False

    result = await _send(
        ctx, parent, session, message="come back", receiver_role="child", receiver_name="scout"
    )
    assert result.is_error is True
    assert result.error.kind == "failed", "not addressable is not a policy refusal"
    assert "could not be woken" in result.error.message
    assert "agent_observe" in result.error.message
    assert ctx.require(AGENTS).get(run.session_id) is None


async def test_a_send_outside_the_family_is_refused_by_a_guard(family_ctx: MountedRuntime) -> None:
    """C7: the guard is deny-only and runs last, so nothing re-permits it."""
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)
    grandchild = await _spawn(ctx, child, "recon")

    # A permissive row that allows everything cannot re-permit it: a guard runs
    # after every waterfall listener and a denial is final (B2).
    ctx.on("tools/pre-execute", lambda _execution, _next: Allow())

    # The parent reaching its *grandchild*: one generation too far.
    result = await _send(
        ctx,
        parent,
        session,
        message="hello down there",
        receiver_role="child",
        receiver_name="recon",
    )
    assert result.is_error is True
    assert result.error.kind == "denied", "the boundary must deny, not merely fail"
    assert OUT_OF_REACH in result.error.message or "no child is named" in result.error.message
    # And the grandchild exists and is reachable from its own parent, so the
    # refusal is about the boundary rather than about a missing agent.
    assert ctx.require(SESSIONS).get(grandchild.session_id) is not None


async def test_the_denial_names_the_reachable_roles(family_ctx: MountedRuntime) -> None:
    """A root agent with no family gets a refusal it can act on."""
    ctx, session, parent = await family_ctx()
    result = await _send(ctx, parent, session, message="anyone?", receiver_role="parent")
    assert result.is_error is True
    assert OUT_OF_REACH in result.error.message


async def test_an_ambiguous_role_asks_for_a_name(family_ctx: MountedRuntime) -> None:
    ctx, session, parent = await family_ctx()
    await _spawn(ctx, parent, "alpha")
    await _spawn(ctx, parent, "beta")

    result = await _send(ctx, parent, session, message="which of you", receiver_role="child")
    assert result.is_error is True
    assert "pass receiver_name" in result.error.message


# ------------------------------------------------------------------ limits --


async def test_an_oversized_message_is_refused_with_the_alternative(
    family_ctx: MountedRuntime,
) -> None:
    ctx, session, sender, _target = await _siblings(family_ctx, maxMessageChars=64)

    result = await _send(ctx, sender, session, message="x" * 100, receiver_role="sibling")
    assert result.is_error is True
    assert "at most 64 characters" in result.error.message
    assert "send the path" in result.error.message


async def test_the_rate_limit_fails_rather_than_denies(family_ctx: MountedRuntime) -> None:
    """C8, and the deliberate deviation: backpressure is the program's to handle.

    A *denial* would end the whole cell under C3, which is not what four messages
    in one second should cost. The distinction is visible in `error.kind`, which
    is exactly what Code Mode branches on.
    """
    ctx, session, sender, _target = await _siblings(
        family_ctx, rateCapacity=2, rateRefillSeconds=1000.0, maxPending=99
    )

    for index in range(2):
        allowed = await _send(
            ctx, sender, session, message=f"note {index}", receiver_role="sibling"
        )
        assert allowed.is_error is False

    limited = await _send(ctx, sender, session, message="one too many", receiver_role="sibling")
    assert limited.is_error is True
    assert limited.error.kind == "failed", "a rate limit must not end the run"
    assert "backpressure, not a refusal" in limited.error.message


async def test_a_target_at_the_pending_cap_refuses_more(family_ctx: MountedRuntime) -> None:
    ctx, session, sender, _target = await _siblings(family_ctx, maxPending=2, rateCapacity=99)

    for index in range(2):
        assert (
            await _send(ctx, sender, session, message=f"m{index}", receiver_role="sibling")
        ).is_error is False
    blocked = await _send(ctx, sender, session, message="m2", receiver_role="sibling")
    assert blocked.is_error is True
    assert blocked.error.kind == "failed"
    assert "unread messages" in blocked.error.message


# ------------------------------------------------------------------ replies --


async def test_a_childs_send_records_that_it_replied(family_ctx: MountedRuntime) -> None:
    """The wiring that suppresses the "finished without replying" notice.

    Asserted on the provider's record rather than on the notice's absence: a
    child driven by the fake adapter settles inside the send's own await, so
    whether the notice has already been written is a race. That the *reply* is
    recorded is not — and `test_subagents.py` covers what the provider does with
    it, in the one place the ordering is controllable.
    """
    ctx, _session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)

    child_session = ctx.require(SESSIONS).get(run.session_id)
    assert child_session is not None
    sent = await _send(
        ctx,
        child,
        child_session,
        message="here it is",
        receiver_role="parent",
    )
    assert sent.is_error is False
    assert ctx.require(RLM_CHILDREN)._children[run.id].replied is True


# ------------------------------------------------------------------ observe --


async def test_observe_is_bounded_by_the_same_reach_rule(family_ctx: MountedRuntime) -> None:
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)
    grandchild = await _spawn(ctx, child, "recon")
    await ctx.drain()

    seen = await run_tool(
        ctx, OBSERVE_GET_TOOL, {"agent_id": run.session_id}, agent=parent, session=session
    )
    assert seen.is_error is False
    assert seen.value["role"] == "child"
    assert any("task from parent" in row["text"] for row in seen.value["messages"])

    # A grandchild is out of reach — two *roots* would be siblings under the
    # rule, so the unreachable case has to be a generation away.
    refused = await run_tool(
        ctx,
        OBSERVE_GET_TOOL,
        {"agent_id": grandchild.session_id},
        agent=parent,
        session=session,
    )
    assert refused.is_error is True
    assert OUT_OF_REACH in not_none(refused.error).message


async def test_an_observe_read_is_capped(family_ctx: MountedRuntime) -> None:
    ctx, session, parent = await family_ctx(observeMaxMessages=1)
    run = await _spawn(ctx, parent, "scout")
    await ctx.drain()

    seen = await run_tool(
        ctx,
        OBSERVE_GET_TOOL,
        {"agent_id": run.session_id, "limit": 50},
        agent=parent,
        session=session,
    )
    assert len(seen.value["messages"]) == 1, "the row cap bounds what one read returns"


# ----------------------------------------------------------------- rendering --


def test_the_received_rendering_is_prime_agents() -> None:
    text = render_received(
        sender_label="child:scout",
        sender_id="s-1",
        target_id="s-0",
        message_id="m-1",
        body="the answer",
    )
    assert text.splitlines()[0] == "[from child:scout]"
    assert "Agent-to-agent message received." in text
    assert "Source: agent_message" in text
    assert "From: s-1" in text
    assert "To: s-0" in text
    assert "Message id: m-1" in text
    # The body is last and verbatim, after a blank line.
    assert text.endswith("\n\nthe answer")
