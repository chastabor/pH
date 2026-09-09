"""Delegation: admission, the depth gate, and what the parent is told (P3-11).

The load-bearing claim is **non-blocking admission**: `start()` returns once the
child is admitted, not once it has answered. Everything else here is about the
parent being able to act on that handle — the roster is a fold, a silent child is
announced, usage is attributed, a revoked child leaves a tombstone.

## `subagent/usage-attributed` is a record, not a correction

This module's own docstring used to say the event exists "so the token meter
**can** subtract a child's tokens from the parent's own context measurement".
That was never true, and "can" was doing the work in the sentence.

`TokenMeter.last_usage` scans only `assistant/message` in the log it is *given*,
and a child's `assistant/message` events are in the **child's** log — so the
parent's context measurement never included them and there is nothing to
subtract. The event is additive, for readers; its only consumer is the TUI panel.

Worth keeping written down because the false version was load-bearing-sounding:
it implied a fan-out of eight would otherwise read as context pressure on the
parent and trigger a compaction it does not need. It would not, and no code path
depends on the event to prevent it.

## Why the parent check sits above `agents.create` in `rehydrate`

Below it, a rehydration with no live parent built an agent and a scope, failed, and
left both behind: **an orphan under the registry root holding the deployment-wide
ceiling that nothing would ever dispose**. The parent owns the drive job *and*,
since P6-27, the scope the child nests in — so a missing parent is a refusal rather
than a degradation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph.llm.types import text_of
from ph.persistence import resume_session
from ph.seams.subagents import (
    STATUS,
    UNRECOVERABLE_DETAIL,
    USAGE,
    SubagentRequest,
    SubagentSpawnError,
    child_is_live,
    default_child_name,
    exhausted_detail,
    family_reach,
    subagent_roster,
)
from ph.seams.workspace import workspace_survivors
from ph.session import derive_event_message
from ph.testing import FAKE_OPTIONS, StubWorkspaceProvider, skill
from ph.testing.git import WORKTREE_ROWS, git_repo
from ph_rlm.subagents import PROVIDER_NAME, TASK_PREFIX, delegation_depth

pytestmark = pytest.mark.anyio

Mounted = Callable[..., Any]

from conftest import PROVIDER_ROW  # noqa: E402 - a fixture row, not a symbol


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


async def test_the_default_shapes_the_request_and_the_tier_answers_it(
    delegating: Mounted,
) -> None:
    """E4 and E3 together, at the `advisory` tier — which is the shipped default.

    The default is about the *request*: an omitted `access` asks for `read`, and
    that is what makes a child research-shaped no matter what the tier does with
    it. What comes back is the tier's answer, and at `advisory` the answer to
    both requests is the same shared checkout, because **nothing here can make a
    repository read-only** (§4.8's table says exactly this, and calls the row
    "not read-only"). So `granted` says `write` for a `read` request, and the
    pair — requested beside granted — is the honest record. Claiming `read`
    would be a promise nothing keeps, which is the one thing this field must
    never do.
    """
    ctx, session, parent = await delegating()
    default = await _spawn(ctx, parent, "read some code")
    assert default.requested_access == "read"
    assert default.granted_access == "write"

    asked = await _spawn(ctx, parent, "implement the thing", access="write")
    assert asked.requested_access == "write"
    assert asked.granted_access == "write"

    rows = {
        event.data["runId"]: event.data
        for event in session.events
        if event.type == "subagent/admitted"
    }
    # Nothing was *narrowed*, so there is no downgrade to report: the widening a
    # `read` request meets at this tier is visible in the pair itself, and the
    # child is told plainly by its own workspace prompt line.
    assert asked.downgrade_reason is None
    assert default.downgrade_reason is None
    assert "downgradeReason" not in rows[default.id]


async def test_a_read_child_gets_an_isolated_checkout_where_a_tier_can_give_one(
    delegating: Mounted, tmp_path: Path
) -> None:
    """E3, through the spawn path: the same request, a tier that can answer it.

    `worktree-ephemeral` is what `access="read"` buys where a tier exists — the
    child writes freely and **merges nothing**, so what it was granted *of the
    project* is `read`. That is the seam's own reading of `repo_writable` applied
    one level up, and the reason `granted` is not copied from the request.

    Branching from the *parent's* root rather than the process's is the other
    half: it is what puts a fan-out on sibling branches instead of one shared
    checkout (E2).
    """
    ctx, _session, parent = await delegating()
    parent_root = tmp_path / "parent-tree"
    parent_root.mkdir()
    await ctx.workspace.acquire(session_id="parent", agent_id=parent.id, base=parent_root)
    tier = StubWorkspaceProvider()
    ctx.workspace.register_provider(tier)

    child = await _spawn(ctx, parent, "read some code")

    assert child.requested_access == "read"
    assert child.granted_access == "read"
    assert tier.bases == [parent_root]


async def test_a_profile_with_no_workspace_row_refuses_to_promise_one(mount: Any) -> None:
    """The conservative claim, and the only case that still downgrades.

    With no seam at all nothing can enforce a writable repo, so nothing promises
    one — and the reason travels as a *code*, not a sentence, because a durable
    log has to stay parseable and prose in an event goes stale silently.
    """
    ctx = await mount(
        dict(PROVIDER_ROW),
        {"id": "workspace-lifecycle", "remove": True},
        {"id": "workspace", "remove": True},
    )
    session = ctx.sessions.create("parent")
    parent = ctx.agents.create(session, FAKE_OPTIONS)

    child = await _spawn(ctx, parent, "implement the thing", access="write")

    assert child.granted_access == "read"
    assert child.downgrade_reason == "workspace-not-mounted"
    rows = {
        event.data["runId"]: event.data
        for event in session.events
        if event.type == "subagent/admitted"
    }
    assert rows[child.id]["downgradeReason"] == "workspace-not-mounted"


# ------------------------------------------------------- what the parent hears --


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Gate]:
    """A held model, released whatever the test does — see `_Gate`."""
    held = _Gate()
    held.patch(monkeypatch)
    yield held
    held.release_all()


def _statuses(session: Any, run_id: str) -> list[str]:
    """Every status this child reached, in order. One spelling, three readers."""
    return [
        str(event.data["status"])
        for event in session.events
        if event.type == STATUS and event.data.get("runId") == run_id
    ]


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
    assert _statuses(session, run.id)[-1] == "done"


async def test_the_child_status_reaches_the_parents_log(delegating: Mounted) -> None:
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()

    statuses = _statuses(session, run.id)
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


# ------------------------------------------------------------------- the grant --


async def test_a_real_child_is_narrowed_by_its_spawn(delegating: Mounted) -> None:
    """P4-13b through the provider that actually ships it.

    The seam refuses what the parent does not hold and the provider applies the
    narrowing, so this is the half a unit test of `apply_grant` cannot reach: a
    child agent created by `agents.create` — whose scope is the parent's
    *sibling* — really does end up with the subset.
    """
    ctx, _session, parent = await delegating()
    for name in ("review", "deploy"):
        ctx.skills.register(skill(name))

    run = await _spawn(ctx, parent, skills=("review",), tools=("read",))
    child_scope = next(one.ctx for one in ctx.agents.list() if one.session.id == run.session_id)

    assert [one.name for one in ctx.skills.list(child_scope)] == ["review"]
    assert "read" in ctx.tools.view(child_scope).visible
    assert "write" not in ctx.tools.view(child_scope).visible
    # The parent kept everything, which is what makes this narrowing.
    assert "write" in ctx.tools.view(parent.ctx).visible


async def test_a_real_spawn_cannot_widen(delegating: Mounted) -> None:
    ctx, _session, parent = await delegating()

    with pytest.raises(SubagentSpawnError) as refused:
        await _spawn(ctx, parent, skills=("nonesuch",))

    assert "Grant it to the parent first" in str(refused.value)


async def test_a_rehydrated_child_is_narrowed_again(delegating: Mounted) -> None:
    """The hole this row nearly shipped, and the reason the seam owns the ceiling.

    Settlement disposes the child's scope, and the filters bounding it are
    effects of that scope — so they go too. `rehydrate` then builds a **fresh**
    scope from the retained options, and narrowed nothing: a child that outlived
    its own restriction came back holding the whole deployment, which is exactly
    what the ruling forbids and is reachable from a public method on the
    shipping provider.
    """
    ctx, _session, parent = await delegating()
    for name in ("review", "deploy"):
        ctx.skills.register(skill(name))
    run = await _spawn(ctx, parent, skills=("review",))
    assert [one.name for one in ctx.skills.list(run.scope)] == ["review"]

    await ctx.drain()
    assert ctx.agents.get(run.session_id) is None, "the child should have settled"

    assert await ctx.subagents.rehydrate(run.id)

    assert [one.name for one in ctx.skills.list(run.scope)] == ["review"], (
        "a rehydrated child came back holding more than its parent granted"
    )


# --------------------------------------------------- P6-28: the evidence policy --


async def _tiered_child(
    ctx: Any, parent: Any, tmp_path: Path, prompt: str, *, access: str = "read"
) -> Any:
    """A child under a tier that hands out worktrees, `read` by default.

    `access` is a parameter rather than a second copy of the setup: the write
    case differs from the read case in exactly that word, and the two had already
    drifted apart on `mkdir(exist_ok=)` and on whether the provider was given a
    root.
    """
    parent_root = tmp_path / "parent-tree"
    parent_root.mkdir(exist_ok=True)
    await ctx.workspace.acquire(session_id="parent", agent_id=parent.id, base=parent_root)
    ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    return await _spawn(ctx, parent, prompt, access=access)


def _marks(ctx: Any, run: Any) -> list[str]:
    session = ctx.sessions.get(run.session_id)
    return [
        str(event.data.get("retained", ""))
        for event in session.events
        if event.type == "workspace/retained"
    ]


async def test_a_child_is_retained_from_the_moment_its_tree_exists(
    delegating: Mounted, tmp_path: Path
) -> None:
    """Retain-by-default, and why it cannot be retain-on-failure (P6-28).

    The window to mark a workspace closes with the child's scope, and the
    outcomes that most need the evidence are the ones that close it first: on the
    `parent-teardown` path the worktree is released *before* `_release` runs,
    which that method's own docstring states in its own terms. So there is
    nowhere to put "retain when it fails" — the mark has to exist from the moment
    the tree does, and a clean finish withdraws it.
    """
    ctx, _session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "get cancelled")

    assert _marks(ctx, run) == ["the child has not settled cleanly"]
    (record,) = workspace_survivors(ctx.sessions.get(run.session_id))
    assert record.outcome == "retained"


async def test_a_clean_child_leaves_nothing_behind(delegating: Mounted, tmp_path: Path) -> None:
    """The other half, and the one that keeps the kind's promise.

    Without the withdrawal, retain-by-default would invert `worktree-ephemeral`
    for *every* outcome rather than for the ones that produce evidence — which
    is the trade the row forbids, since it buys a checkout per delegation and
    saves nothing anyone will read.
    """
    ctx, _session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "finish and go")
    await ctx.drain()

    assert _marks(ctx, run) == ["the child has not settled cleanly", ""]
    # The *closing* half is what the disposal policy reads, and the reason is
    # gone from it. Asserted rather than `outcome`, because this tier is a stub
    # with no `release` and so keeps every tree — under the real `worktree`
    # provider this checkout is discarded and there is no survivor at all, which
    # is precisely the promise the withdrawal restores.
    (record,) = workspace_survivors(ctx.sessions.get(run.session_id))
    assert record.outcome != "retained", "a successful child's checkout is not evidence"
    assert record.reason == ""


async def test_a_cancelled_child_keeps_its_evidence(delegating: Mounted, tmp_path: Path) -> None:
    """The case the row was written for, through the path that cannot mark.

    `parent-teardown` disposes the child's scope before the settle handler runs,
    so the retention this asserts is one nothing on that path could have set. It
    is there because it was taken at acquire and never withdrawn.
    """
    ctx, _session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "outlive me")
    child_session = ctx.sessions.get(run.session_id)

    await ctx.agents.dispose(parent.id)

    (record,) = workspace_survivors(child_session)
    assert record.outcome == "retained"
    assert record.reason == "the child has not settled cleanly"
    assert record.closed is True, "the pair still closed; only the discard was skipped"


async def test_a_write_child_is_not_retained(delegating: Mounted, tmp_path: Path) -> None:
    """Only the kind that discards, because only that kind can lose evidence.

    An ordinary `worktree` already keeps a dirty tree for review and a committed
    branch survives release regardless, so retaining those would grow the pile
    without saving anything from it.
    """
    ctx, _session, parent = await delegating()

    run = await _tiered_child(ctx, parent, tmp_path, "write some code", access="write")

    assert _marks(ctx, run) == []


async def test_a_failed_child_tells_its_parent_where_the_tree_is(
    delegating: Mounted, tmp_path: Path
) -> None:
    """Retention keeps the checkout; this is what stops it being evidence nobody
    can find.

    A child's workspace events are in the *child's* log, so a parent told only
    "it failed" is left diagnosing from a transcript — the sentence the row opens
    with. Read from the log rather than from the seam, so the same notice is
    true on a path where the child's scope has already gone.
    """
    ctx, session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "fail please")
    child = ctx.agents.get(run.session_id)
    root = ctx.workspace.of(child.id).root
    _breaks(child)
    await ctx.drain()

    (told,) = _notices(session)
    assert "failed" in told
    assert str(root) in told, "the parent was left to find the evidence itself"


def _breaks(agent: Any) -> None:
    """Make this agent's next run raise, which is what `_drive` catches."""

    async def broken() -> None:
        raise RuntimeError("the child broke")

    agent.run = broken


async def test_a_child_with_no_tree_is_announced_without_naming_one(
    delegating: Mounted,
) -> None:
    """Silence rather than a fabricated path.

    A profile with no tier, a `shared` workspace, a child whose tree really was
    discarded — a notice that named a directory for every failure would be
    naming ones that are not there.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "fail with no workspace")
    _breaks(ctx.agents.get(run.session_id))
    await ctx.drain()

    (told,) = _notices(session)
    assert "failed" in told
    assert "workspace is kept" not in told


# ---------------------------------------------------------------- the queue --


class _Gate:
    """Holds every child's model call until the test lets one through.

    Patched over the fake adapter's `stream`, so a child is "running" for as long
    as the test says and no timing is guessed. `arrived` counts calls, since the
    held list shrinks as they are released.

    The `gate` fixture below is what opens it again, and that is not tidiness: a
    child left parked here makes the mount's `drain()` wait forever, so one failed
    assertion becomes a hung suite. Written as a `finally` in each test, that is a
    rule the fourth test has to remember.
    """

    def __init__(self) -> None:
        self.held: list[anyio.Event] = []
        self.arrived = 0
        self.open = False

    def release_one(self) -> None:
        self.held.pop(0).set()

    def twice(self) -> bool:
        """Whether a second child has reached the model — the readmit's proof."""
        return self.arrived >= 2

    def release_all(self) -> None:
        self.open = True
        for event in self.held:
            event.set()
        self.held.clear()

    def patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from ph.llm.fake import FakeAdapter

        original = FakeAdapter.stream
        gate = self

        async def gated(self: Any, options: Any) -> Any:
            gate.arrived += 1
            if not gate.open:
                event = anyio.Event()
                gate.held.append(event)
                await event.wait()
            async for chunk in original(self, options):
                yield chunk

        monkeypatch.setattr(FakeAdapter, "stream", gated)


async def _until(predicate: Callable[[], bool], what: str) -> None:
    """Poll until `predicate()`, or fail saying what was being waited for.

    The `except` is the point, and `daemon_helpers.until` makes the same argument
    for its own copy: `fail_after` raises a bare `TimeoutError`, so a call site's
    `what` reaches nobody without this — and a wedged queue is exactly the failure
    that arrives as a timeout with no other evidence.
    """
    try:
        with anyio.fail_after(5):
            while not predicate():
                await anyio.sleep(0.005)
    except TimeoutError:
        pytest.fail(f"timed out waiting for {what}")


async def test_a_full_parent_queues_the_next_child_until_a_slot_frees(
    delegating: Mounted, gate: _Gate
) -> None:
    """`maxConcurrent` is a queue: the parent gets every child it asked for, one
    slot at a time, in admission order — and never a refusal."""
    ctx, session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")

    roster = ctx.subagents.roster(session)
    assert roster[first.id]["status"] == "running"
    assert roster[second.id]["status"] == "queued", "admitted, not refused — and waiting"
    assert _statuses(session, second.id) == ["queued"], "the wait is in the log"

    gate.release_one()
    await _until(lambda: gate.arrived == 2, "the second child to take the freed slot")
    assert ctx.subagents.roster(session)[first.id]["status"] == "done"
    assert _statuses(session, second.id) == ["queued", "running"]

    gate.release_one()
    assert (await second.result()).status == "done"
    assert _statuses(session, first.id) == ["running", "done"], "no wait, no queued record"


async def test_a_child_that_failed_frees_its_slot(
    delegating: Mounted, gate: _Gate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue cannot wedge on a failure: the provider's `error` path releases
    the slot exactly as `done` does.

    The failure is the child's *run* raising — the provider's own error path —
    rather than a model call failing, which the agent loop contains as a turn that
    ended in error and the provider records as `done`.
    """
    from ph.agent_loop.driver import ReactLoopAgent

    original_run = ReactLoopAgent.run
    failed: list[str] = []

    async def run(self: Any) -> None:
        # The parent is never run in this test, so the first `run()` is the first
        # child's; every later one is genuine.
        if not failed:
            failed.append(self.id)
            raise RuntimeError("the child fell over")
        await original_run(self)

    monkeypatch.setattr(ReactLoopAgent, "run", run)
    ctx, session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")

    assert (await first.result()).status == "error"
    await _until(lambda: gate.arrived == 1, "the second child to run after the failure")
    gate.release_one()

    assert (await second.result()).status == "done"
    assert _statuses(session, first.id) == ["running", "error"]
    assert _statuses(session, second.id)[-2:] == ["running", "done"]


async def test_deleting_a_queued_child_stops_its_wait_and_takes_no_slot(
    delegating: Mounted, gate: _Gate
) -> None:
    """A child revoked before it ran is cancelled where it waits, and the slot it
    never held is not leaked — the next child still gets it."""
    ctx, session, parent = await delegating(maxConcurrent=1)
    provider = ctx.subagents.require(PROVIDER_NAME).provider
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")

    assert await provider.delete(session, second.id, reason="user") is True
    assert _statuses(session, second.id) == ["queued", "cancelled"]

    third = await _spawn(ctx, parent, "third")
    gate.release_one()
    await _until(lambda: gate.arrived == 2, "the third child to take the slot the first freed")
    gate.release_one()
    assert (await first.result()).status == "done"
    assert (await third.result()).status == "done"
    assert _statuses(session, third.id) == ["queued", "running", "done"]


# ------------------------------------------------------------ across a restart --


async def _persisted(ctx: Any, session: Any) -> None:
    """Put on disk what a restart will read, with the harness holding still.

    A flush and nothing else. The first harness is parked at the model for the
    whole of these tests, so it appends nothing more to this log until the gate
    opens at teardown — which is what lets a second one open the same file
    without the two writers P5-03 refuses. Draining here instead would wait on
    the very child that is meant to be caught mid-flight.
    """
    await ctx.sessions.flush(session)


RETRIES = 3
"""This suite's own ladder bound.

Stated here rather than imported: `resume_children` takes the limit because it is
the *host's* policy, and a test that reached into `ph-app` for the daemon's
number would be asserting against a value it does not control — and coupling
`ph-rlm`'s tests to a package they do not depend on.
"""


async def _restart(
    mount: Any, session_id: str, *, skills: tuple[str, ...] = (), concurrent: int = 1
) -> Any:
    """A second harness over the same `$PH_HOME`, resuming one root from its log.

    What a daemon restart *is* from the seam's side: a fresh mount, nothing in
    memory, and a session that has to come off disk. `resume_children` is the
    call `Supervisor` makes at the same point, against the same agent.

    `skills` is what the *deployment* still provides. It is a parameter because
    a readmit re-derives the child's ceiling against what the parent holds now,
    not against what it held then — so a skill this deployment no longer mounts
    is a child refused rather than one quietly readmitted without it.
    """
    ctx = await mount(dict(PROVIDER_ROW, config={"maxConcurrent": concurrent}))
    for name in skills:
        ctx.skills.register(skill(name))
    session = await resume_session(ctx, session_id)
    parent = ctx.agents.create(session, FAKE_OPTIONS)
    await ctx.subagents.resume_children(parent, retry_limit=RETRIES)
    return ctx, session, parent


async def test_a_queued_child_is_re_driven_after_a_restart(
    delegating: Mounted, gate: _Gate, mount: Any
) -> None:
    """The work was described in the parent's log and running nowhere (P5-04).

    A child that never reached its first turn has claimed nothing and spent
    nothing, so the next harness runs it for the first time — under its original
    id, so the parent's roster gains no second child it never asked for.

    Reaching the model is the proof of "re-driven": the gate counts arrivals, and
    the readmitted child is the only one the second harness can run.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")
    assert _statuses(session, second.id) == ["queued"]
    await _persisted(ctx, session)

    # Room for both, so which one this asserts about is not a race: the
    # interrupted sibling is on the ladder and comes back too.
    revived_ctx, revived, _parent = await _restart(mount, session.id, concurrent=2)

    assert second.id in {run.id for run in revived_ctx.subagents.list()}, (
        "the queued child came back under its own id"
    )
    await _until(
        lambda: _statuses(revived, second.id)[-1] == "running",
        "the readmitted child to reach the model",
    )
    assert len(subagent_roster(revived)) == 2, "no child was invented or lost"
    assert "attempts" not in subagent_roster(revived)[second.id], (
        "a child that never ran is a first attempt, not a retry"
    )


async def test_a_child_caught_mid_turn_climbs_the_ladder_with_its_task_re_presented(
    delegating: Mounted, gate: _Gate, mount: Any
) -> None:
    """The ladder, and the thing that makes it a real attempt rather than a lie.

    Starting a turn *claims* the task from the inbox and the claim is a logged
    splice, so a resumed child that was simply driven again would find an empty
    inbox, end at step zero and report `completed` for work it never did —
    P5-04's finding at the root. Re-presenting the task is the fix, and saying
    the turn was cut short is what stops the transcript reading as one
    instruction given twice.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    interrupted = await _spawn(ctx, parent, "only")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    assert _statuses(session, interrupted.id) == ["running"]
    await _persisted(ctx, session)

    revived_ctx, revived, _parent = await _restart(mount, session.id)

    assert interrupted.id in {run.id for run in revived_ctx.subagents.list()}
    assert child_is_live(subagent_roster(revived)[interrupted.id]), "a child owed a turn is live"
    # The count follows the restart's own `running` record, which a detached
    # drive job writes a moment later — so this waits for the fact rather than
    # reading the roster before it exists.
    await _until(
        lambda: subagent_roster(revived)[interrupted.id].get("attempts") == 1,
        "the restart to be counted",
    )
    assert subagent_roster(revived)[interrupted.id]["starts"] == 2, "one first run, one restart"
    await _until(gate.twice, "the resumed child to reach the model again")

    child = revived_ctx.sessions.get(interrupted.session_id)
    assert child is not None
    tasks = [
        text_of(derive_event_message(event).content)
        for event in child.events
        if event.type == "user/message" and TASK_PREFIX in repr(event.data)
    ]
    assert len(tasks) == 2, "the task was not presented again, so the retry answers nothing"
    assert "the harness stopped while you were working on this" in tasks[-1]
    assert "this is attempt 2" in tasks[-1]


def _resumed(session: Any, run_id: str, times: int) -> None:
    """Record `times` restarts, the way a restart actually records one.

    The real facts rather than a seeded count: the ladder folds `running` records
    carrying `cause: resumed`, so a test that wrote an `attempts` number would be
    asserting against a field production no longer has.
    """
    for _ in range(times):
        session.append(STATUS, {"runId": run_id, "status": "running", "cause": "resumed"})


async def _stalled(ctx: Any, session: Any, parent: Any, gate: _Gate, *, restarts: int) -> Any:
    """A child at the model that has already been restarted `restarts` times."""
    child = await _spawn(ctx, parent, "only")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    _resumed(session, child.id, restarts)
    return child


async def test_the_ladder_gives_up_and_says_so(
    delegating: Mounted, gate: _Gate, mount: Any
) -> None:
    """Three restarts with nothing achieved between them is not bad luck.

    Re-driving forever would spend a parent's budget on a transcript nobody
    reads, which is the failure the root's own ladder is bounded to avoid.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    spent = await _stalled(ctx, session, parent, gate, restarts=RETRIES)
    assert subagent_roster(session)[spent.id]["attempts"] == RETRIES
    await _persisted(ctx, session)

    revived_ctx, revived, _parent = await _restart(mount, session.id)

    assert revived_ctx.subagents.list() == [], "a spent ladder put a child back to work"
    row = subagent_roster(revived)[spent.id]
    assert row["status"] == "error"
    assert row["detail"] == exhausted_detail(RETRIES)
    assert str(RETRIES) in row["detail"], "the sentence names the bound it hit"
    assert not child_is_live(row), "a root cannot be passivated while this reads live"


async def test_progress_since_the_last_restart_clears_the_ladder(
    delegating: Mounted, gate: _Gate, mount: Any
) -> None:
    """A child stopped, working an hour, then stopped again met two incidents.

    Without a reset the ladder counts a lifetime's interruptions rather than
    consecutive ones, and fails work that was going fine. The same setup as the
    test above plus one fact: the child was attributed a model answer, which is
    something a turn that did nothing cannot produce.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    moved = await _stalled(ctx, session, parent, gate, restarts=RETRIES)
    session.append(USAGE, {"runId": moved.id, "targetSeq": 0, "childUsage": {}, "origin": "probe"})
    assert subagent_roster(session)[moved.id]["attempts"] == 0, "progress clears the count"
    await _persisted(ctx, session)

    revived_ctx, revived, _parent = await _restart(mount, session.id)

    row = subagent_roster(revived)[moved.id]
    assert child_is_live(row), "a child that got somewhere is owed another attempt"
    assert moved.id in {run.id for run in revived_ctx.subagents.list()}
    # **The restart is still recorded as one**, counting up from the cleared
    # ladder. Derived from `attempts` instead, this readmit would look like a
    # first run, write no `resumed`, and the ladder would never count it again.
    await _until(
        lambda: subagent_roster(revived)[moved.id].get("attempts") == 1,
        "the restart to be counted from a cleared ladder",
    )


async def test_a_readmitted_child_does_not_come_back_wider_than_it_was_admitted(
    delegating: Mounted, gate: _Gate, mount: Any
) -> None:
    """§6.5 across a power cut, which is why the narrowing is in the record.

    The ceiling is re-derived from the admission, so a child admitted with one
    skill does not return holding every skill its parent has. Rebuilt from the
    run alone it would, and nothing would have said so.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    ctx.skills.register(skill("review"))
    ctx.skills.register(skill("audit"))
    await _spawn(ctx, parent, "first")
    narrowed = await _spawn(ctx, parent, "second", skills=("review",), tools=("read",))
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")
    await _persisted(ctx, session)

    revived_ctx, revived, _parent = await _restart(mount, session.id, skills=("review", "audit"))

    # Read back off the *resumed* log, which is the only copy a restart has.
    admitted = next(
        event.data
        for event in revived.events
        if event.type == "subagent/admitted" and event.data["runId"] == narrowed.id
    )
    assert list(admitted["skills"]) == ["review"], "the narrowing has to survive the round trip"
    assert list(admitted["tools"]) == ["read"]

    back = next(run for run in revived_ctx.subagents.list() if run.id == narrowed.id)
    assert back.grant is not None
    assert back.grant.skills == ("review",)
    assert back.grant.tools == ("read",)


async def test_a_child_no_provider_can_resume_is_settled_not_left_queued(
    delegating: Mounted, gate: _Gate, mount: Any
) -> None:
    """A `queued` row nothing will pick up is worse than an honest failure.

    It reads as live, so the root can never be passivated, and the parent waits
    on a slot no one will ever give it. Deciding where the capability probe
    answers is what stops the sweep writing a status it cannot honour — here the
    next harness mounts no subagent provider at all.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    orphan = await _spawn(ctx, parent, "only")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)

    bare = await mount()
    revived = await resume_session(bare, session.id)
    await bare.subagents.resume_children(
        bare.agents.create(revived, FAKE_OPTIONS), retry_limit=RETRIES
    )

    row = subagent_roster(revived)[orphan.id]
    assert row["status"] == "error"
    assert row["detail"] == UNRECOVERABLE_DETAIL
    assert not child_is_live(row), "a root cannot be passivated while this reads live"


# ------------------------------------------------- a child asked a second thing --


@pytest.mark.needs_git
async def test_a_re_addressed_child_comes_back_to_its_own_work(mount: Any, tmp_path: Path) -> None:
    """A second question reaches the child that answered the first, tree and all.

    Disposal commits the checkout to the child's branch before removing it, and
    `_add` attaches an existing branch rather than resetting it — so the same run
    id resolves to the same branch, and re-acquiring restores what the child did.
    Nothing arranges that here; what was missing is that `rehydrate` never asked.
    """
    ctx = await mount(*WORKTREE_ROWS, PROVIDER_ROW)
    base = await git_repo(ctx, tmp_path / "repo")
    session = ctx.sessions.create("parent")
    parent = ctx.agents.create(session, FAKE_OPTIONS)
    await ctx.workspace.acquire(
        session_id=session.id, agent_id=parent.id, base=base, access="write", session=session
    )

    run = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="first question", parent=parent, access="write")
    )
    first = ctx.workspace.of(run.session_id)
    assert first is not None and first.kind == "worktree"
    (first.root / "child-work.txt").write_text("what the first answer produced\n", encoding="utf-8")
    await ctx.drain()
    assert not first.root.exists(), "the checkout goes; the branch is what survives"

    assert await ctx.subagents.ensure_addressable(run.session_id) is True

    again = ctx.workspace.of(run.session_id)
    assert again is not None, "a child given a runtime back and no tree writes into its parent's"
    assert again.root == first.root, "the same child, so the same checkout"
    assert again.ref == first.ref
    assert (again.root / "child-work.txt").read_text(encoding="utf-8").startswith("what the first")


@pytest.mark.needs_git
async def test_a_re_addressed_child_is_no_wider_than_it_was_admitted(
    mount: Any, tmp_path: Path
) -> None:
    """The containment half. A child admitted `read` comes back ephemeral.

    Its access is read from the admission, not from the caller re-addressing it,
    so nothing about being asked a second question can widen what the first was
    allowed to keep.
    """
    ctx = await mount(*WORKTREE_ROWS, PROVIDER_ROW)
    base = await git_repo(ctx, tmp_path / "repo")
    session = ctx.sessions.create("parent")
    parent = ctx.agents.create(session, FAKE_OPTIONS)
    await ctx.workspace.acquire(
        session_id=session.id, agent_id=parent.id, base=base, access="write", session=session
    )

    run = await ctx.subagents.start(
        PROVIDER_NAME, SubagentRequest(prompt="look only", parent=parent, access="read")
    )
    assert ctx.workspace.of(run.session_id).kind == "worktree-ephemeral"
    await ctx.drain()

    assert await ctx.subagents.ensure_addressable(run.session_id) is True

    again = ctx.workspace.of(run.session_id)
    assert again is not None
    assert again.kind == "worktree-ephemeral", "a second question must not widen the first's grant"
