"""Delegation: admission, the depth gate, and what the parent is told (P3-11).

The load-bearing claim is **non-blocking admission**: `start()` returns once the
child is admitted, not once it has answered. Everything else here is about the
parent being able to act on that handle — the roster is a fold, a silent child is
announced, usage is attributed, a revoked child leaves a tombstone.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from ph.seams.subagents import (
    SubagentRequest,
    SubagentSpawnError,
    default_child_name,
    family_reach,
    subagent_roster,
)
from ph.testing import FAKE_OPTIONS
from ph_rlm.subagents import PROVIDER_NAME, TASK_PREFIX, delegation_depth

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]

PROVIDER_ROW: dict[str, Any] = {"id": "rlm-subagent-provider", "name": "rlm-subagent-provider"}


@pytest.fixture
def delegating(mount: Any) -> Callable[..., Any]:
    """`await delegating()` → `(ctx, parent_session, parent)` with the provider on."""

    async def build(**config: Any) -> tuple[Any, Any, Any]:
        rows = [dict(PROVIDER_ROW)]
        if config:
            rows[0]["config"] = config
        ctx = await mount(*rows)
        session = ctx.sessions.create("parent")
        return ctx, session, ctx.agents.create(session, FAKE_OPTIONS)

    return build


async def _spawn(ctx: Any, parent: Any, prompt: str = "research the thing", **kwargs: Any) -> Any:
    return await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt=prompt, parent=parent, **kwargs)
    )


# ------------------------------------------------------------------ admission --


async def test_admission_returns_before_the_child_answers(delegating: Mounted) -> None:
    """The property the whole design exists for: a parent fans out and keeps
    working, instead of blocking on each child in turn."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)

    # Admitted, not finished: the record of it existing is written and nothing
    # has reported on it yet.
    assert [event.type for event in session.events if event.type.startswith("subagent/")] == [
        "subagent/admitted"
    ]
    admitted = [event for event in session.events if event.type == "subagent/admitted"]
    assert len(admitted) == 1
    assert admitted[0].data["runId"] == run.id
    assert admitted[0].data["name"] == run.name
    assert admitted[0].data["prompt"] == "research the thing"

    # The child's own session exists and carries the parent link and depth.
    child_session = ctx.sessions.get(run.session_id)
    assert child_session is not None
    assert child_session.header.parent_session == session.id
    assert delegation_depth(child_session) == 1


async def test_the_admission_is_logged_before_any_status(delegating: Mounted) -> None:
    """A fold that met status for an unadmitted child would show a family that
    does not exist, so the order is not incidental."""
    ctx, session, parent = await delegating()
    await _spawn(ctx, parent)
    await ctx.drain()

    kinds = [event.type for event in session.events if event.type.startswith("subagent/")]
    assert kinds[0] == "subagent/admitted"
    assert "subagent/status" in kinds


async def test_eight_children_are_all_admitted_without_waiting(delegating: Mounted) -> None:
    ctx, session, parent = await delegating()
    runs = [await _spawn(ctx, parent, f"task {index}") for index in range(8)]

    assert len({run.id for run in runs}) == 8
    assert len({run.name for run in runs}) == 8, "names address children, so they are unique"
    assert len([e for e in session.events if e.type == "subagent/admitted"]) == 8
    assert len(ctx.subagents.list(parent_id=parent.id)) == 8


async def test_the_child_gets_the_task_labelled_as_the_parents(delegating: Mounted) -> None:
    """`[task from parent]` is what the child's own prompt recognizes."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "count the files")
    await ctx.drain()

    child_session = ctx.sessions.get(run.session_id)
    assert child_session is not None
    relayed = [
        event
        for event in child_session.events
        if event.type == "user/message" and TASK_PREFIX in repr(event.data)
    ]
    assert relayed, "the child never received the task"
    assert "count the files" in repr(relayed[0].data)
    assert relayed[0].data["source"]["form"] == "relay"


# ----------------------------------------------------------------- the gates --


async def test_the_depth_gate_names_both_numbers(delegating: Mounted) -> None:
    """Prime Agent's wording, so a model that has seen it need not re-learn it."""
    ctx, session, parent = await delegating(maxDepth=0)
    with pytest.raises(SubagentSpawnError, match=r"RLM_DEPTH=0, RLM_MAX_DEPTH=0"):
        await _spawn(ctx, parent)
    # Refused before the child existed, so there is nothing to reconcile: no
    # admission, no session, no artifacts.
    assert [event for event in session.events if event.type.startswith("subagent/")] == []


async def test_a_child_cannot_delegate_past_the_depth_limit(delegating: Mounted) -> None:
    ctx, _session, parent = await delegating(maxDepth=1)
    run = await _spawn(ctx, parent)
    child_session = ctx.sessions.get(run.session_id)
    child = ctx.agents.get(child_session.id) if child_session else None
    assert child is not None

    with pytest.raises(SubagentSpawnError, match=r"RLM_DEPTH=1, RLM_MAX_DEPTH=1"):
        await _spawn(ctx, child, "delegate again")


async def test_a_prompt_is_required(delegating: Mounted) -> None:
    ctx, _session, parent = await delegating()
    with pytest.raises(SubagentSpawnError, match="needs a prompt"):
        await _spawn(ctx, parent, "   ")


async def test_a_sibling_name_collision_is_refused(delegating: Mounted) -> None:
    ctx, _session, parent = await delegating()
    await _spawn(ctx, parent, "first", name="scout")
    with pytest.raises(SubagentSpawnError, match="already named"):
        await _spawn(ctx, parent, "second", name="scout")


async def test_an_unroutable_provider_is_refused_at_admission(delegating: Mounted) -> None:
    """The preflight that exists today: no adapter, no child. Nothing is
    substituted — a child answering on a model the parent did not choose is a
    result the parent cannot interpret."""
    ctx, _session, parent = await delegating()
    with pytest.raises(SubagentSpawnError, match="no registered adapter"):
        await _spawn(ctx, parent, provider="nonexistent", model="m1")


async def test_access_defaults_to_read_and_says_what_it_granted(delegating: Mounted) -> None:
    """E4, plus the honesty D21 requires until `ctx.workspace` exists."""
    ctx, session, parent = await delegating()
    default = await _spawn(ctx, parent, "read some code")
    assert default.requested_access == "read"
    assert default.granted_access == "read"

    asked = await _spawn(ctx, parent, "implement the thing", access="write")
    assert asked.requested_access == "write"
    # Downgraded, and the log says why rather than pretending it was granted.
    assert asked.granted_access == "read"
    rows = {
        event.data["runId"]: event.data
        for event in session.events
        if event.type == "subagent/admitted"
    }
    # A code, not a sentence: a durable log has to stay parseable after the
    # workspace tier lands, and prose in an event goes stale silently.
    assert asked.downgrade_reason == "workspace-not-mounted"
    assert rows[asked.id]["downgradeReason"] == "workspace-not-mounted"
    assert default.downgrade_reason is None
    assert "downgradeReason" not in rows[default.id]


# ------------------------------------------------------- what the parent hears --


def _notices(session: Any) -> list[str]:
    """Notices delivered to the parent's inbox but not yet claimed by a step.

    `inject` is deliberately non-waking, so the notice lands as a splice and
    becomes a `user/message` at the parent's next step — which is what "replies
    arrive on later turns" means. Reading the splice is reading the moment of
    delivery.
    """
    return [
        repr(event.data)
        for event in session.events
        if event.type == "agent/inbox/spliced" and "rlm child" in repr(event.data)
    ]


async def test_a_child_that_never_replies_is_announced(delegating: Mounted) -> None:
    """Silence is indistinguishable from a hang, so it is reported."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "say nothing")
    await ctx.drain()

    delivered = _notices(session)
    assert delivered, "the parent was never told the child finished"
    assert "completed without sending a reply" in delivered[0]
    assert run.name in delivered[0]
    assert "'form': 'notice'" in delivered[0]


async def test_the_notice_reaches_the_parents_context_on_its_next_step(
    delegating: Mounted,
) -> None:
    """Delivered means the model actually sees it, not that it sits in a queue."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "say nothing")
    await ctx.drain()

    await parent.prompt("what did the child say?")
    claimed = [
        repr(event.data)
        for event in session.events
        if event.type == "user/message" and "rlm child" in repr(event.data)
    ]
    assert claimed, "the notice never entered a step"
    assert run.name in claimed[0]


async def test_a_child_that_replied_is_not_announced_as_silent(delegating: Mounted) -> None:
    """The reply is the notice, so the parent is not told the same thing twice.

    `mark_replied` is called here directly, which is the only way to fix the
    ordering: `rlm-messaging` calls it from a send, and a fake-adapter child
    settles inside that send's own await.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "say something")
    ctx.rlm_children.mark_replied(run.session_id)
    await ctx.drain()

    assert _notices(session) == [], "a child that replied was announced as silent"
    # The status record still lands: only the redundant notice is suppressed.
    assert [
        event.data["status"]
        for event in session.events
        if event.type == "subagent/status" and event.data["runId"] == run.id
    ][-1] == "done"


async def test_the_child_status_reaches_the_parents_log(delegating: Mounted) -> None:
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()

    statuses = [
        event.data["status"]
        for event in session.events
        if event.type == "subagent/status" and event.data["runId"] == run.id
    ]
    assert statuses[0] == "running"
    assert statuses[-1] in {"done", "error"}


async def test_a_waiter_can_still_block_on_completion(delegating: Mounted) -> None:
    """The generic `task` contract: the answer is reachable, just never the thing
    admission hands back."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    assert run.result is not None
    outcome = await run.result()
    assert outcome.status == "done"
    # The answer is reachable — just never what admission handed back.
    assert outcome.answer == "ok"


async def test_child_usage_is_attributed_to_the_parent(delegating: Mounted) -> None:
    """Without this a fan-out of eight reads as context pressure on the parent
    and triggers a compaction it does not need."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()

    attributed = [
        event
        for event in session.events
        if event.type == "subagent/usage-attributed" and event.data["runId"] == run.id
    ]
    assert attributed, "the child's tokens were never attributed"
    assert attributed[0].data["origin"] == "spawn_task"
    assert "childUsage" in attributed[0].data


# --------------------------------------------------------- roster and deletion --


async def test_the_roster_is_a_fold_over_the_parents_own_log(delegating: Mounted) -> None:
    """P3-13 by construction: no side table, so restart and compaction are free."""
    ctx, session, parent = await delegating()
    first = await _spawn(ctx, parent, "one", name="alpha")
    second = await _spawn(ctx, parent, "two", name="beta")
    await ctx.drain()

    roster = subagent_roster(session)
    assert set(roster) == {first.id, second.id}
    assert roster[first.id]["name"] == "alpha"
    assert roster[second.id]["status"] in {"done", "error"}, "status folded onto the row"


async def test_deleting_a_child_leaves_a_tombstone(delegating: Mounted) -> None:
    """The transcript stays on disk, so the revocation must be findable."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "doomed")
    provider = ctx.rlm_children

    assert await provider.delete(session, run.id, reason="user") is True
    assert ctx.subagents.get(run.id) is None
    # Deleting twice is not an error, and does not double-tombstone.
    assert await provider.delete(session, run.id) is False

    tombstones = [event for event in session.events if event.type == "subagent/deleted"]
    assert len(tombstones) == 1
    assert tombstones[0].data == {"runId": run.id, "reason": "user"}

    roster = subagent_roster(session)
    assert roster[run.id]["deleted"] is True
    assert roster[run.id]["deletedReason"] == "user"
    # A revoked child has a terminal state, not merely an absence — a panel that
    # knew only `deleted` could not say whether it had ever run.
    assert roster[run.id]["status"] == "cancelled"
    # The child's log is still there — a tombstone is not a deletion.
    assert ctx.sessions.get(run.session_id) is not None


async def test_a_settled_child_releases_its_agent_scope(delegating: Mounted) -> None:
    """A child's scope owns its kernel subprocess, so holding it leaks a CPython
    per delegation. The terminal result survives the release."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "finish and go")
    await ctx.drain()

    assert ctx.agents.get(run.session_id) is None, "the child agent was never disposed"
    # And a caller that awaits after the release still gets the outcome.
    assert run.result is not None
    assert (await run.result()).status == "done"


async def test_disposing_the_parent_unwinds_its_children(delegating: Mounted) -> None:
    """I2: a child is an artifact of the parent's scope, so it is released by the
    same unwinding rather than by someone remembering to."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "outlive me")

    await ctx.agents.dispose(parent.id)
    assert ctx.subagents.get(run.id) is None
    tombstones = [event for event in session.events if event.type == "subagent/deleted"]
    assert [event.data["reason"] for event in tombstones] == ["parent-teardown"]


async def test_the_status_and_usage_records_are_ignorable(delegating: Mounted) -> None:
    """A different build may skip them; it may *not* skip an admission, because
    that would show the parent the wrong family."""
    ctx, session, parent = await delegating()
    await _spawn(ctx, parent)
    await ctx.drain()

    by_type = {event.type: event for event in session.events if event.type.startswith("subagent/")}
    assert by_type["subagent/status"].ignorable is True
    assert by_type["subagent/usage-attributed"].ignorable is True
    assert by_type["subagent/admitted"].ignorable is False


# ------------------------------------------------------------- seam vocabulary --


def test_the_family_reach_rule_is_the_nuclear_family() -> None:
    """C7's rule, in the seam so the guard (P3-12) and the roster cannot disagree."""

    def reach(sender: tuple[str | None, str], target: tuple[str | None, str]) -> bool:
        return family_reach(
            sender_parent=sender[0],
            sender_id=sender[1],
            target_parent=target[0],
            target_id=target[1],
        )

    parent, child, sibling, nephew = (None, "p"), ("p", "c"), ("p", "s"), ("c", "n")
    assert reach(child, parent) is True, "a child reaches its parent"
    assert reach(parent, child) is True, "a parent reaches its child"
    assert reach(child, sibling) is True, "siblings share a parent"
    assert reach(child, nephew) is True, "a direct child of its own"
    assert reach(nephew, parent) is False, "a grandparent is out of reach"
    assert reach(nephew, sibling) is False, "an uncle is out of reach"
    # Roots are siblings of each other, which is what makes two top-level
    # agents in one deployment able to talk.
    assert reach((None, "root-a"), (None, "root-b")) is True


def test_a_default_name_describes_the_task_and_stays_unique() -> None:
    name = default_child_name("Review the authentication middleware for races", "abcdef123456")
    assert name.startswith("subagent-review-the-authentication-")
    assert name.endswith("-abcdef12")
    assert default_child_name("x", "abcdef123456", taken=[name]) != name
    # An unslugifiable prompt still yields an addressable name.
    assert default_child_name("!!!", "abcdef123456") == "subagent-task-abcdef12"
