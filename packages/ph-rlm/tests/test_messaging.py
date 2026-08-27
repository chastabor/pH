"""Agent-to-agent messages, and the boundary on them (P3-12, C7/C8).

The load-bearing claim is the asymmetry: the **family boundary** is a monotonic
guard that no later listener can re-permit and whose denial ends the run, while
the **rate limit** is a failure the program handles. Both were one check inside a
comm handler in prime-agent; separating them is what makes the first
unroutable-around and the second survivable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ph.seams.subagents import SubagentRequest, family_reach, reachable_family
from ph.testing import FAKE_OPTIONS, run_tool
from ph.tools import Allow
from ph_rlm.messaging import (
    OBSERVE_GET_TOOL,
    OUT_OF_REACH,
    SEND_TOOL,
    render_received,
)
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]

ROWS: list[dict[str, Any]] = [
    {"id": "rlm-subagent-provider", "name": "rlm-subagent-provider"},
    {"id": "rlm-messaging", "name": "rlm-messaging"},
]


@pytest.fixture
def family_ctx(mount: Any) -> Callable[..., Any]:
    """`await family_ctx()` → `(ctx, parent_session, parent)` with messaging on."""

    async def build(**config: Any) -> tuple[Any, Any, Any]:
        rows = [dict(ROWS[0]), dict(ROWS[1])]
        if config:
            rows[1]["config"] = config
        ctx = await mount(*rows)
        session = ctx.sessions.create("parent")
        return ctx, session, ctx.agents.create(session, FAKE_OPTIONS)

    return build


async def _spawn(ctx: Any, parent: Any, name: str) -> Any:
    return await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt=f"work on {name}", parent=parent, name=name)
    )


async def _siblings(family_ctx: Mounted, **config: Any) -> tuple[Any, Any, Any, Any]:
    """Two root agents, which the reach rule makes siblings of each other.

    Used for the limit tests because they must not race a child's completion:
    a spawned child settles at an unpredictable await point and injects a notice
    into its parent's inbox, which is itself pending input.
    """
    ctx, first_session, first = await family_ctx(**config)
    second_session = ctx.sessions.create("other-root")
    return ctx, first_session, first, ctx.agents.create(second_session, FAKE_OPTIONS)


def _agent(ctx: Any, run: Any) -> Any:
    agent = ctx.agents.get(run.session_id)
    assert agent is not None, "the child agent is not running"
    return agent


async def _send(ctx: Any, sender: Any, session: Any, **arguments: Any) -> Any:
    return await run_tool(ctx, SEND_TOOL, arguments, agent=sender, session=session)


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


async def test_a_child_reaches_its_parent(family_ctx: Mounted) -> None:
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)

    result = await _send(
        ctx, child, ctx.sessions.get(run.session_id), message="found it", receiver_role="parent"
    )
    assert result.is_error is False
    assert result.value["receiverId"] == session.id
    assert result.value["receiverRole"] == "parent"
    assert result.value["deliveryStatus"] in {"delivered", "queued"}


async def test_the_message_reaches_the_target_verbatim(family_ctx: Mounted) -> None:
    """Framing is the sender's; a harness that rewrote the body would make the
    log stop saying what the target actually read."""
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    child = _agent(ctx, run)

    body = "the bug is in parser.py line 40"
    await _send(ctx, child, ctx.sessions.get(run.session_id), message=body, receiver_role="parent")
    delivered = [
        repr(event.data)
        for event in session.events
        if event.type == "agent/inbox/spliced" and "Agent-to-agent" in repr(event.data)
    ]
    assert delivered, "the parent never received it"
    assert body in delivered[0]
    assert "[from child:scout]" in delivered[0]
    assert f"From: {run.session_id}" in delivered[0]


async def test_siblings_reach_each_other(family_ctx: Mounted) -> None:
    """Two roots are siblings under the rule, which is what lets two top-level
    agents in one deployment talk."""
    ctx, session, sender, target = await _siblings(family_ctx)
    ok = await _send(ctx, sender, session, message="halves", receiver_role="sibling")
    assert ok.is_error is False
    assert ok.value["receiverId"] == target.id
    assert ok.value["receiverRole"] == "sibling"


async def test_a_settled_child_cannot_be_steered_yet(family_ctx: Mounted) -> None:
    """Today's honest answer, and the shape of what P3-13 owes.

    Prime Agent retains completed children as addressable and rehydrates a
    passive one on `send`. Passivation is P3-13; until then a settled child has
    no live inbox, so the refusal names the route that *does* work rather than
    pretending the message was delivered.
    """
    ctx, session, parent = await family_ctx()
    run = await _spawn(ctx, parent, "scout")
    await ctx.drain()
    assert ctx.agents.get(run.session_id) is None, "the child should have settled"

    result = await _send(
        ctx, parent, session, message="are you there", receiver_role="child", receiver_name="scout"
    )
    assert result.is_error is True
    assert result.error.kind == "failed", "not running is not a policy refusal"
    assert "is not running" in result.error.message
    assert "agent_observe" in result.error.message


async def test_a_send_outside_the_family_is_refused_by_a_guard(family_ctx: Mounted) -> None:
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
    assert ctx.sessions.get(grandchild.session_id) is not None


async def test_the_denial_names_the_reachable_roles(family_ctx: Mounted) -> None:
    """A root agent with no family gets a refusal it can act on."""
    ctx, session, parent = await family_ctx()
    result = await _send(ctx, parent, session, message="anyone?", receiver_role="parent")
    assert result.is_error is True
    assert OUT_OF_REACH in result.error.message


async def test_an_ambiguous_role_asks_for_a_name(family_ctx: Mounted) -> None:
    ctx, session, parent = await family_ctx()
    await _spawn(ctx, parent, "alpha")
    await _spawn(ctx, parent, "beta")

    result = await _send(ctx, parent, session, message="which of you", receiver_role="child")
    assert result.is_error is True
    assert "pass receiver_name" in result.error.message


# ------------------------------------------------------------------ limits --


async def test_an_oversized_message_is_refused_with_the_alternative(
    family_ctx: Mounted,
) -> None:
    ctx, session, sender, _target = await _siblings(family_ctx, maxMessageChars=64)

    result = await _send(ctx, sender, session, message="x" * 100, receiver_role="sibling")
    assert result.is_error is True
    assert "at most 64 characters" in result.error.message
    assert "send the path" in result.error.message


async def test_the_rate_limit_fails_rather_than_denies(family_ctx: Mounted) -> None:
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


async def test_a_target_at_the_pending_cap_refuses_more(family_ctx: Mounted) -> None:
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


async def test_a_childs_send_records_that_it_replied(family_ctx: Mounted) -> None:
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

    sent = await _send(
        ctx, child, ctx.sessions.get(run.session_id), message="here it is", receiver_role="parent"
    )
    assert sent.is_error is False
    assert ctx.rlm_children._children[run.id].replied is True


# ------------------------------------------------------------------ observe --


async def test_observe_is_bounded_by_the_same_reach_rule(family_ctx: Mounted) -> None:
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
    assert OUT_OF_REACH in refused.error.message


async def test_an_observe_read_is_capped(family_ctx: Mounted) -> None:
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
