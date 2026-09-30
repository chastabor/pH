"""Delegation: admission, the depth gate, and what the parent is told (P3-11).

The load-bearing claim is **non-blocking admission**: `start()` returns once the
child is admitted, not once it has answered. Everything else here is about the
parent being able to act on that handle — a child's state is read from its own
log, a silent child is announced, what a child spent is counted where it spent it,
a revoked child leaves a tombstone.

## Each session owns its log (Phase 11)

Every record about a child — its admission, each start, each wait, its ending and
its tombstone — is in the **child's own** log, and nothing about it is written to
its parent's. So these tests read a child's story from the child (`_log_of`,
`_state`), and the ones about durability ask the child's store, not the parent's.

The parent used to keep a roster of its children in its own log, and every
durability rule in this file existed to keep that second account in step with the
first: the parent flushed before a child's gate opened (S2), before a restart
(S10), after the child's own log (F1), and a resume copied the answers a crash had
left only in the child's log back into the parent's (L5). With one account there
is nothing to keep in step, and each rule is about one log.
`test_a_parents_log_holds_no_record_of_its_child` is what keeps it that way.

## A child's spend never was a correction to its parent's context

`TokenMeter.last_usage` folds only `assistant/message` in the log it is *given*,
and a child's `assistant/message` events are in the **child's** log — so the
parent's context measurement never included them and there is nothing to
subtract. The mirror this package once wrote into the parent's log
(`subagent/usage-attributed`) was additive, for readers, and those readers — a
goal's budget, the retry ladder, the TUI panel — read the child's own answers now
(`ChildState.tokens`).

Worth keeping written down because the false version was load-bearing-sounding:
it implied a fan-out of eight would otherwise read as context pressure on the
parent and trigger a compaction it does not need. It would not, and no code path
depends on a parent-side record to prevent it.

## Why the parent check sits above `agents.create` in `rehydrate`

Below it, a rehydration with no live parent built an agent and a scope, failed, and
left both behind: **an orphan under the registry root holding the deployment-wide
ceiling that nothing would ever dispose**. The parent owns the drive job *and*,
since P6-27, the scope the child nests in — so a missing parent is a refusal rather
than a degradation.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from rlm_fixtures import (
    BINDINGS_ROW,
    PROVIDER_ROW,
    ModelGate,
    MountedRuntime,
    logs_after_a_crash,
)

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.json import as_obj, as_seq
from ph.keys import (
    AGENTS,
    CREDENTIALS,
    JOBS,
    LLM,
    SESSION_PERSISTENCE,
    SESSIONS,
    SKILLS,
    SUBAGENTS,
    TOOLS,
    WORKSPACE,
)
from ph.llm.adapter import ResolvedModel
from ph.llm.fake import FakeAdapter, text_script
from ph.llm.types import text_of, user_text
from ph.persistence import SessionBusy, open_session, resume_session
from ph.seams.credentials import waiting_for
from ph.seams.subagents import (
    ADMITTED,
    DELETED,
    PARENT_TEARDOWN,
    STATUS,
    SUSPENDED_DETAIL,
    UNRECOVERABLE_DETAIL,
    ChildState,
    StatusCause,
    SubagentRequest,
    SubagentRun,
    SubagentSpawnError,
    child_is_live,
    child_state,
    child_state_of,
    default_child_name,
    exhausted_detail,
    family_reach,
    record_started,
    restarts_since_progress,
)
from ph.seams.token_meter import reported_usage
from ph.seams.workspace import workspace_survivors
from ph.session import (
    Session,
    SessionEvent,
    SurfaceIntent,
    derive_event_message,
)
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    StubWorkspaceProvider,
    assistant_payload,
    log_event,
    not_none,
    reconciled_call,
    run_tool,
    skill,
    stored_events,
    stored_types,
)
from ph.testing.git import WORKTREE_ROWS, git_repo
from ph.tools.definition import NotDone
from ph_rlm.bindings import RUN_TOOL
from ph_rlm.keys import RLM_CHILDREN
from ph_rlm.subagents import (
    PROVIDER_NAME,
    TASK_PREFIX,
    RlmChildProvider,
    delegation_depth,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def delegating(mount: MountProfile) -> Callable[..., Any]:
    """`await delegating(*rows)` → `(ctx, parent_session, parent)` with the provider on,
    and any further rows beside it."""

    async def build(*extra: dict[str, Any], **config: object) -> tuple[Any, Any, Any]:
        rows = [dict(PROVIDER_ROW), *extra]
        if config:
            rows[0]["config"] = config
        ctx = await mount(*rows)
        session = ctx.require(SESSIONS).create("parent")
        return ctx, session, ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    return build


async def _spawn(
    ctx: Context,
    parent: Any,  # noqa: ANN401
    prompt: str = "research the thing",
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    return await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt=prompt, parent=parent, **kwargs)
    )


def _log_of(ctx: Context, run: SubagentRun) -> list[SessionEvent]:
    """A child's own log: its live session's events, else what its store holds.

    The live one while this process has the child open, since that is what its doors
    write. The stored one for a child this process let go — one a resume sweep ended
    without readmitting it, whose log it opened, wrote, flushed and closed again — for
    which the disk is the only copy left.
    """
    live = ctx.require(SESSIONS).get(run.session_id)
    return list(live.events) if live is not None else stored_events(ctx, run.session_id)


def _state(ctx: Context, run: SubagentRun) -> ChildState:
    """One child as its own log tells it, through the service every reader asks — the
    prompt, the roster tool, the resume sweep and the budgets alike."""
    return not_none(ctx.require(SUBAGENTS).state(run.session_id), f"the child {run.id}")


def _stored_state(ctx: Context, run: SubagentRun) -> ChildState:
    """One child as its **store** holds it: what a resume would be handed, and so what a
    durability rule is about."""
    header, events = ctx.require(SESSION_PERSISTENCE).read(run.session_id)
    return child_state_of(run.session_id, header, events)


def _statuses(ctx: Context, run: SubagentRun, field: str = "status") -> list[str]:
    """Every status this child reached, in order — or another `field` of the same
    records. Its own log's, which is the only place a status is written."""
    return [str(event.data.get(field)) for event in _log_of(ctx, run) if event.type == STATUS]


def _about_children(session: Session) -> list[str]:
    """The `subagent/*` records in a parent's own log — none, by design."""
    return [event.type for event in session.events if event.type.startswith("subagent/")]


# ------------------------------------------------------------------ admission --


@pytest.mark.parametrize("step", ["building-its-workspace", "starting-its-drive"])
async def test_a_parent_that_goes_away_mid_admission_refuses_the_child(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """A spawn whose parent's scope is disposed under one of its awaits is refused.

    It escaped as a raw `InactiveScopeError` from whichever registration met the
    dead scope first — a workspace effect on the child's scope, the release effect or
    the job on the parent's — carrying no spawn code, and a child it had got as far
    as starting was left open with nothing that would ever end it. Now it is a
    `SubagentSpawnError`, and what was built is released: its session let go while
    it is being built, tombstoned in its own log once its drive exists. Either way
    the seam never writes its admission, so its log names no child, and nothing
    reads as a live child of the parent.

    Sabotage: drop either `except InactiveScopeError` in `_admit`.
    """
    ctx, session, parent = await delegating()
    # Two awaits in building the child: its workspace, and the job that drives it,
    # which registers on the parent's scope.
    owner, name = (
        (RlmChildProvider, "_workspace")
        if step == "building-its-workspace"
        else (type(ctx.require(JOBS)), "start")
    )
    original = getattr(owner, name)

    async def parent_goes(self: Any, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        await ctx.require(AGENTS).dispose(parent.id)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(owner, name, parent_goes)

    with pytest.raises(SubagentSpawnError, match="went away"):
        await _spawn(ctx, parent)

    assert not any(
        child_is_live(state) for state in ctx.require(SUBAGENTS).children(session.id).values()
    )
    assert ctx.require(SUBAGENTS).list() == []
    assert _about_children(session) == []
    logs = [
        one for one in ctx.require(SESSIONS).list() if one.header.delegating_parent == session.id
    ]
    if step == "building-its-workspace":
        assert logs == [], "the child's session outlived its refusal"
    else:
        (child,) = logs
        kinds = [event.type for event in child.events if event.type.startswith("subagent/")]
        assert ADMITTED not in kinds and DELETED in kinds, kinds


async def test_admission_returns_before_the_child_answers(delegating: MountedRuntime) -> None:
    """The property the whole design exists for: a parent fans out and keeps
    working, instead of blocking on each child in turn."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)

    # Admitted, not finished: the record of it existing is written — in its own log,
    # and not its parent's — and nothing has reported on it yet.
    own = _log_of(ctx, run)
    assert [event.type for event in own if event.type.startswith("subagent/")] == [ADMITTED]
    (admitted,) = [event for event in own if event.type == ADMITTED]
    assert admitted.data["runId"] == run.id
    assert admitted.data["name"] == run.name
    assert admitted.data["prompt"] == "research the thing"
    assert admitted.data["owner"] == PROVIDER_NAME, "readmission finds its provider by this"
    assert _about_children(session) == []

    # The child's own session exists and carries the parent link and depth.
    child_session = ctx.require(SESSIONS).get(run.session_id)
    assert child_session is not None
    assert child_session.header.parent_session == session.id
    assert delegation_depth(child_session) == 1


async def test_the_admission_is_logged_before_any_status(delegating: MountedRuntime) -> None:
    """The child's drive is gated on its admission (`SubagentRun.ready`), so every
    status the drive writes follows the admission in the child's own log.

    The fold no longer needs the order — a log with statuses and no admission is not a
    child, and every reader skips it — but a drive that ran ahead of its admission
    would be a child working before anything recorded it existed, which is S2's
    failure seen from the other side."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()

    kinds = [event.type for event in _log_of(ctx, run) if event.type.startswith("subagent/")]
    assert kinds[0] == ADMITTED
    assert STATUS in kinds


async def test_the_admission_is_on_disk_before_the_child_takes_a_step(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2 — the admission is the one record a resume finds a child by.

    It is in the child's own log, and the seam flushes that log before it opens the
    child's gate: a crash any time after the child's first step leaves a log on disk
    that says what the child was asked, by whom and under which ceiling — which is
    what the store lists a parent's children by (`descendants_of`) and the sweep
    readmits them from. It used to ride the parent's log in memory, while the child's
    own flushes never wrote its parent, so a crash mid-run left a child with a log and
    a tree that nothing on disk named.

    Asked at the gate, not at the model: the child's first model request flushes its
    log anyway, so by then the question is moot. The drive's first act after the gate
    is its `running` record, so the store is read there.

    Sabotage: drop the child's flush from `SubagentService._admit`, and the stored
    log is empty when the child starts.
    """
    ctx, _session, parent = await delegating()
    seen: list[list[str]] = []
    original = record_started

    async def watched(
        scope: Context, child: Session, *, cause: StatusCause | None = None
    ) -> SessionEvent:
        seen.append(stored_types(ctx, child.id))
        return await original(scope, child, cause=cause)

    monkeypatch.setattr("ph_rlm.subagents.record_started", watched)
    await _spawn(ctx, parent, "work")
    await ctx.drain()

    assert len(seen) == 1, "the child never started"
    assert ADMITTED in seen[0], "a child started and its own log on disk does not say so"


async def test_a_child_whose_admission_cannot_be_written_is_not_started(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """Fail-closed, as a durable intent is: no admission on disk, no child.

    The refusal reaches the caller, the child never reaches the model, and it is
    ended in its own log — so nothing holds the parent out of passivation for a child
    that never ran.

    Sabotage: open the gate whether or not the write worked, and the child runs.
    """
    ctx, session, parent = await delegating()

    def refuse(target: Session) -> None:
        # The child's log, which is the one the admission is written to: its id is
        # minted inside the spawn, and its header names the parent.
        if target.header.delegating_parent == session.id:
            raise OSError("the disk is full")

    ctx.on("session/flush", refuse)
    with pytest.raises(SubagentSpawnError, match="could not be written"):
        await _spawn(ctx, parent, "never")
    await ctx.drain()

    assert gate.arrived == 0, "the child ran without its admission on disk"
    (state,) = ctx.require(SUBAGENTS).children(session.id).values()
    assert state.status == "error"
    assert not child_is_live(state)


async def test_an_interrupted_spawn_is_answered_with_the_handle_it_admitted(
    delegating: MountedRuntime,
) -> None:
    """S2 — an `rlm.run` a crash cut short is found among its parent's children by its
    call.

    Its whole value is the admission handle, and the admission is on the child's disk
    before the child runs — carrying the call's id — so a resume can show the program
    the handle it would have had, rather than "outcome unknown", on which a model
    spawns the same child beside the one the resume has already put back to work. No
    admission under the call is a spawn that never happened.

    Sabotage: drop `call_id` from the request `run_child` builds, and the admitted
    child is not found.
    """
    ctx, session, parent = await delegating(dict(BINDINGS_ROW))
    arguments = {"prompt": "scout the repo"}
    spawned = await run_tool(ctx, RUN_TOOL, arguments, agent=parent, session=session)
    assert spawned.is_error is False

    found = await reconciled_call(ctx, session, RUN_TOOL, arguments)
    assert isinstance(found, tuple), "the admitted child was not found by its call"
    assert "admitted" in text_of(list(found))
    never = await reconciled_call(ctx, session, RUN_TOOL, arguments, call_id="call-2")
    assert isinstance(never, NotDone), "a call that admitted nothing started nothing"


async def test_eight_children_are_all_admitted_without_waiting(delegating: MountedRuntime) -> None:
    ctx, session, parent = await delegating()
    runs = [await _spawn(ctx, parent, f"task {index}") for index in range(8)]

    assert len({run.id for run in runs}) == 8
    assert len({run.name for run in runs}) == 8, "names address children, so they are unique"
    assert [
        event.type for run in runs for event in _log_of(ctx, run) if event.type == ADMITTED
    ] == [ADMITTED] * 8, "one admission each, in each child's own log"
    assert list(ctx.require(SUBAGENTS).children(session.id)) == [run.id for run in runs]
    assert len(ctx.require(SUBAGENTS).list(parent_id=parent.id)) == 8


async def test_the_child_gets_the_task_labeled_as_the_parents(delegating: MountedRuntime) -> None:
    """`[task from parent]` is what the child's own prompt recognizes."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "count the files")
    await ctx.drain()

    child_session = ctx.require(SESSIONS).get(run.session_id)
    assert child_session is not None
    relayed = [
        event
        for event in child_session.events
        if event.type == "user/message" and TASK_PREFIX in repr(event.data)
    ]
    assert relayed, "the child never received the task"
    assert "count the files" in repr(relayed[0].data)
    assert as_obj(relayed[0].data["source"])["form"] == "relay"


# ----------------------------------------------------------------- the gates --


async def test_the_depth_gate_names_both_numbers(delegating: MountedRuntime) -> None:
    """Prime Agent's wording, so a model that has seen it need not re-learn it."""
    ctx, session, parent = await delegating(maxDepth=0)
    with pytest.raises(SubagentSpawnError, match=r"RLM_DEPTH=0, RLM_MAX_DEPTH=0"):
        await _spawn(ctx, parent)
    # Refused before the child existed, so there is nothing to reconcile: no
    # admission, no session, no artifacts.
    assert ctx.require(SUBAGENTS).children(session.id) == {}
    assert [
        one.id for one in ctx.require(SESSIONS).list() if one.header.delegating_parent == session.id
    ] == [], "a refused spawn opened a log for its child"


async def test_a_child_cannot_delegate_past_the_depth_limit(delegating: MountedRuntime) -> None:
    ctx, _session, parent = await delegating(maxDepth=1)
    run = await _spawn(ctx, parent)
    child_session = ctx.require(SESSIONS).get(run.session_id)
    child = ctx.require(AGENTS).get(child_session.id) if child_session else None
    assert child is not None

    with pytest.raises(SubagentSpawnError, match=r"RLM_DEPTH=1, RLM_MAX_DEPTH=1"):
        await _spawn(ctx, child, "delegate again")


async def test_a_prompt_is_required(delegating: MountedRuntime) -> None:
    ctx, _session, parent = await delegating()
    with pytest.raises(SubagentSpawnError, match="needs a prompt"):
        await _spawn(ctx, parent, "   ")


async def test_a_sibling_name_collision_is_refused(delegating: MountedRuntime) -> None:
    ctx, _session, parent = await delegating()
    await _spawn(ctx, parent, "first", name="scout")
    with pytest.raises(SubagentSpawnError, match="already named"):
        await _spawn(ctx, parent, "second", name="scout")


async def test_an_unroutable_provider_is_refused_at_admission(delegating: MountedRuntime) -> None:
    """The preflight that exists today: no adapter, no child. Nothing is
    substituted — a child answering on a model the parent did not choose is a
    result the parent cannot interpret."""
    ctx, _session, parent = await delegating()
    with pytest.raises(SubagentSpawnError, match="no registered adapter"):
        await _spawn(ctx, parent, provider="nonexistent", model="m1")


async def test_the_default_shapes_the_request_and_the_tier_answers_it(
    delegating: MountedRuntime,
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
        run_id: not_none(state.admission)
        for run_id, state in ctx.require(SUBAGENTS).children(session.id).items()
    }
    # Nothing was *narrowed*, so there is no downgrade to report: the widening a
    # `read` request meets at this tier is visible in the pair itself, and the
    # child is told plainly by its own workspace prompt line.
    assert asked.downgrade_reason is None
    assert default.downgrade_reason is None
    assert rows[default.id].downgrade_reason is None
    assert "downgradeReason" not in rows[default.id].to_wire()


async def test_a_read_child_gets_an_isolated_checkout_where_a_tier_can_give_one(
    delegating: MountedRuntime, tmp_path: Path
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
    await ctx.require(WORKSPACE).acquire(session_id="parent", agent_id=parent.id, base=parent_root)
    tier = StubWorkspaceProvider()
    ctx.require(WORKSPACE).register_provider(tier)

    child = await _spawn(ctx, parent, "read some code")

    assert child.requested_access == "read"
    assert child.granted_access == "read"
    assert tier.bases == [parent_root]


async def test_a_profile_with_no_workspace_row_refuses_to_promise_one(mount: MountProfile) -> None:
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
    session = ctx.require(SESSIONS).create("parent")
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    child = await _spawn(ctx, parent, "implement the thing", access="write")

    assert child.granted_access == "read"
    assert child.downgrade_reason == "workspace-not-mounted"
    admitted = not_none(ctx.require(SUBAGENTS).children(session.id)[child.id].admission)
    assert admitted.downgrade_reason == "workspace-not-mounted"


# ------------------------------------------------------- what the parent hears --


def _notices(session: Session) -> list[str]:
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


async def test_a_child_that_never_replies_is_announced(delegating: MountedRuntime) -> None:
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
    delegating: MountedRuntime,
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


async def test_a_child_that_replied_is_not_announced_as_silent(delegating: MountedRuntime) -> None:
    """The reply is the notice, so the parent is not told the same thing twice.

    `mark_replied` is called here directly, which is the only way to fix the
    ordering: `rlm-messaging` calls it from a send, and a fake-adapter child
    settles inside that send's own await.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "say something")
    ctx.require(RLM_CHILDREN).mark_replied(run.session_id)
    await ctx.drain()

    assert _notices(session) == [], "a child that replied was announced as silent"
    # The status record still lands: only the redundant notice is suppressed.
    assert _statuses(ctx, run)[-1] == "done"


async def test_the_child_status_reaches_its_own_log(delegating: MountedRuntime) -> None:
    """Each start and the ending are the child's records, in its log; the parent's
    log hears of the child only as a notice in its inbox."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()

    statuses = _statuses(ctx, run)
    assert statuses[0] == "running"
    assert statuses[-1] in {"done", "error"}
    assert _about_children(session) == []


async def test_a_childs_outcome_is_on_disk_before_its_parent_is_told(
    delegating: MountedRuntime,
) -> None:
    """F1. The parent hears a child finished — the notice in its inbox, the result
    its waiters are handed — only once the child's own log on disk holds the ending,
    and the answer before it.

    Nothing used to flush a child after its last model request — its last barrier
    comes *before* that request, and a parent's flush walks ancestors, never
    children — so every child log on disk ended before its answer while the parent
    was told "done" with a preview of it. Repair then closed the child's turn as
    interrupted on the next open, and readmitted it to do the work again. The rule is
    one log's now: `record_ended` writes the ending and flushes the child before the
    drive tells anyone.

    Asked of the store at the moment the notice lands in the parent's log, through
    the Protocol: what it would hand a resume, not what is in memory.

    Sabotage: drop the flush from `ph.seams.subagents.record_ended`, and the child's
    stored log is missing its ending when the parent is told.
    """
    ctx, session, parent = await delegating()
    children: list[SubagentRun] = []
    seen: list[tuple[list[SessionEvent], list[SessionEvent]]] = []

    def watch(_session: Session, event: SessionEvent) -> None:
        if event.type == "agent/inbox/spliced" and "rlm child" in repr(event.data):
            child = not_none(ctx.require(SESSIONS).get(children[0].session_id))
            seen.append((stored_events(ctx, child.id), list(child.events)))

    session.observe(watch)
    children.append(await _spawn(ctx, parent))
    await ctx.drain()

    assert len(seen) == 1, "the parent was never told the child finished"
    stored, held = seen[0]
    ending = next(
        i for i, event in enumerate(held) if event.type == STATUS and event.data["status"] == "done"
    )
    # Everything up to and including the ending — the answer among it. What follows
    # it (the withdrawn workspace mark) is the child's own business, not the news.
    assert [event.seq for event in stored[: ending + 1]] == [
        event.seq for event in held[: ending + 1]
    ], "the parent was told before the child's disk held its answer and its ending"
    assert _stored_state(ctx, children[0]).status == "done"


async def test_a_waiter_can_still_block_on_completion(delegating: MountedRuntime) -> None:
    """The generic `task` contract: the answer is reachable, just never the thing
    admission hands back."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    assert run.result is not None
    outcome = await run.result()
    assert outcome.status == "done"
    # The answer is reachable — just never what admission handed back.
    assert outcome.answer == "ok"


async def test_a_childs_spend_is_counted_from_its_own_answers(
    delegating: MountedRuntime,
) -> None:
    """What a child spent is read where it spent it: the `usage` on its own
    `assistant/message`s, which a goal's budget (`delegated_tokens`) and the panel
    add up. Nothing is copied into the parent's log for them to read — a copy that
    could lag the child's own after a crash, and had to be caught up on every resume.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()

    answers = [event for event in _log_of(ctx, run) if event.type == "assistant/message"]
    spent = sum(not_none(reported_usage(event)).total for event in answers)
    assert spent > 0, "the child's answer carries no usage to count"
    assert _state(ctx, run).tokens == spent
    assert ctx.require(SUBAGENTS).delegated_tokens(session.id) == spent
    assert _about_children(session) == []


# ------------------------------------------------------- children and deletion --


async def test_the_children_are_read_from_their_own_logs(delegating: MountedRuntime) -> None:
    """P3-13 by construction: no side table, so restart and compaction are free.
    A parent's children are whatever logs name it, each folded on its own — the
    service's cached fold equals a fresh fold of the child's log (I6)."""
    ctx, session, parent = await delegating()
    first = await _spawn(ctx, parent, "one", name="alpha")
    second = await _spawn(ctx, parent, "two", name="beta")
    await ctx.drain()

    children = ctx.require(SUBAGENTS).children(session.id)
    assert list(children) == [first.id, second.id], "in admission order"
    assert children[first.id].name == "alpha"
    assert children[second.id].status in {"done", "error"}, "status folded onto the state"
    for run in (first, second):
        assert children[run.id] == child_state(not_none(ctx.require(SESSIONS).get(run.session_id)))


async def test_deleting_a_child_leaves_a_tombstone(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """The transcript stays on disk, so the revocation must be findable — in the
    child's own log, which alone then tells its whole story.

    Held at the model, so it is revoked while it works: the delete reads the parent's
    stored children off the event loop first, and an unheld child answers meanwhile.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "doomed")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    subagents = ctx.require(SUBAGENTS)

    assert await subagents.delete(session, run.id, reason="user") is True
    assert subagents.get(run.id) is None
    # Deleting twice is not an error, and does not double-tombstone.
    assert await subagents.delete(session, run.id, reason="again") is False

    own = _log_of(ctx, run)
    tombstones = [event for event in own if event.type == DELETED]
    assert len(tombstones) == 1
    assert tombstones[0].data == {"reason": "user"}

    state = _state(ctx, run)
    assert state.deleted is True
    assert state.deleted_reason == "user"
    # A revoked child has a terminal state, not merely an absence — a panel that
    # knew only `deleted` could not say whether it had ever run.
    assert state.status == "canceled"
    # And it lands with the tombstone (S14): apart, a flush between them left a
    # child canceled and not deleted.
    canceled = next(
        event for event in own if event.type == STATUS and event.data.get("status") == "canceled"
    )
    assert canceled.batch is not None and canceled.batch == tombstones[0].batch
    # On its disk before anything was let go.
    assert _stored_state(ctx, run).deleted is True
    # The child's log is still there — a tombstone is not a deletion.
    assert ctx.require(SESSIONS).get(run.session_id) is not None
    assert _about_children(session) == []


async def test_a_parents_log_holds_no_record_of_its_child(delegating: MountedRuntime) -> None:
    """The one-writer gate (Phase 11). A child's whole life — spawned, answered,
    woken and answered again, deleted — is written in its own log and not its
    parent's: the parent's log gains no `subagent/*` record, nothing reaches it but
    the notices delivered to its own inbox, and it is never flushed on the child's
    account.

    One writer per log is one writer per lock. Every door a child's drive called used
    to take the parent's `Session`, so each child appended to its parent's log and
    sometimes flushed it — a writer the parent's lease never saw, and a second account
    of the child that every durability rule then had to keep in step with the first.

    Sabotage: have a door write the parent's log as well as the child's — the roster
    this replaced — and the parent's log holds a `subagent/*` record.
    """
    ctx, session, parent = await delegating()
    before = session.seq
    flushed: list[str] = []

    def note(target: Session) -> None:
        flushed.append(target.id)

    ctx.on("session/flush", note)

    run = await _spawn(ctx, parent, "answer twice")
    await ctx.drain()
    assert await ctx.require(SUBAGENTS).rehydrate(run.id)
    not_none(ctx.require(AGENTS).get(run.session_id)).steer(user_text("and once more"))
    await ctx.drain()
    assert await ctx.require(SUBAGENTS).delete(session, run.id, reason="user")

    # The child's own log tells the whole story…
    assert [event.type for event in _log_of(ctx, run) if event.type.startswith("subagent/")] == [
        ADMITTED,
        STATUS,
        STATUS,
        STATUS,
        STATUS,
        DELETED,
    ]
    assert _statuses(ctx, run) == ["running", "done", "running", "done"]
    assert _stored_state(ctx, run).deleted, "on the child's disk"
    # …and the parent's tells none of it.
    appended = {event.type for event in session.events if event.seq >= before}
    assert appended <= {"agent/inbox/spliced"}, f"the child wrote its parent's log: {appended}"
    assert len(_notices(session)) == 2, "each answer was announced to the parent's inbox"
    assert session.id not in flushed, "the child flushed its parent's log"
    await ctx.require(SESSIONS).flush(session)
    assert [kind for kind in stored_types(ctx, session.id) if kind.startswith("subagent/")] == []


async def test_a_settled_child_releases_its_agent_scope(delegating: MountedRuntime) -> None:
    """A child's scope owns its kernel subprocess, so holding it leaks a CPython
    per delegation. The terminal result survives the release."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "finish and go")
    await ctx.drain()

    assert ctx.require(AGENTS).get(run.session_id) is None, "the child agent was never disposed"
    # And a caller that awaits after the release still gets the outcome.
    assert run.result is not None
    assert (await run.result()).status == "done"


async def test_disposing_the_parent_unwinds_its_children(delegating: MountedRuntime) -> None:
    """I2: a child is an artifact of the parent's scope, so it is released by the
    same unwinding rather than by someone remembering to."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "outlive me")

    await ctx.require(AGENTS).dispose(parent.id)
    assert ctx.require(SUBAGENTS).get(run.id) is None
    tombstones = [event for event in _log_of(ctx, run) if event.type == DELETED]
    assert [event.data["reason"] for event in tombstones] == [PARENT_TEARDOWN]
    assert _about_children(session) == []


async def test_every_record_of_a_child_is_required_reading(delegating: MountedRuntime) -> None:
    """No build may skip one. An admission never could — skipping it shows the
    parent the wrong family — and a status no longer may either: the retry ladder
    counts starts from it, so a reader that skipped one would miscount the ladder or
    bring back a child that ended."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await ctx.drain()
    await ctx.require(SUBAGENTS).delete(not_none(parent.session), run.id, reason="user")

    by_type = {
        event.type: event for event in _log_of(ctx, run) if event.type.startswith("subagent/")
    }
    assert set(by_type) == {ADMITTED, STATUS, DELETED}
    assert [event.ignorable for event in by_type.values()] == [False] * 3


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
    name = default_child_name("Review the authentication middleware for races", "abcdef12")
    assert name.startswith("subagent-review-the-authentication-")
    assert name.endswith("-abcdef12")
    assert default_child_name("x", "abcdef12", taken=[name]) != name
    # An unslugifiable prompt still yields an addressable name.
    assert default_child_name("!!!", "abcdef12") == "subagent-task-abcdef12"


# ------------------------------------------------------------------- the grant --


async def test_a_real_child_is_narrowed_by_its_spawn(delegating: MountedRuntime) -> None:
    """P4-13b through the provider that actually ships it.

    The seam refuses what the parent does not hold and the provider applies the
    narrowing, so this is the half a unit test of `apply_grant` cannot reach: a
    child agent created by `agents.create` — whose scope is the parent's
    *sibling* — really does end up with the subset.
    """
    ctx, _session, parent = await delegating()
    for name in ("review", "deploy"):
        ctx.require(SKILLS).register(skill(name))

    run = await _spawn(ctx, parent, skills=("review",), tools=("read",))
    child_scope = next(
        one.ctx
        for one in ctx.require(AGENTS).list()
        if one.session is not None and one.session.id == run.session_id
    )

    assert [one.name for one in ctx.require(SKILLS).list(child_scope)] == ["review"]
    assert "read" in ctx.require(TOOLS).view(child_scope).visible
    assert "write" not in ctx.require(TOOLS).view(child_scope).visible
    # The parent kept everything, which is what makes this narrowing.
    assert "write" in ctx.require(TOOLS).view(parent.ctx).visible


async def test_a_real_spawn_cannot_widen(delegating: MountedRuntime) -> None:
    ctx, _session, parent = await delegating()

    with pytest.raises(SubagentSpawnError) as refused:
        await _spawn(ctx, parent, skills=("nonesuch",))

    assert "Grant it to the parent first" in str(refused.value)


async def test_a_rehydrated_child_is_narrowed_again(delegating: MountedRuntime) -> None:
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
        ctx.require(SKILLS).register(skill(name))
    run = await _spawn(ctx, parent, skills=("review",))
    assert [one.name for one in ctx.require(SKILLS).list(run.scope)] == ["review"]

    await ctx.drain()
    assert ctx.require(AGENTS).get(run.session_id) is None, "the child should have settled"

    assert await ctx.require(SUBAGENTS).rehydrate(run.id)

    assert [one.name for one in ctx.require(SKILLS).list(run.scope)] == ["review"], (
        "a rehydrated child came back holding more than its parent granted"
    )


# --------------------------------------------------- P6-28: the evidence policy --


async def _tiered_child(
    ctx: Context,
    parent: Any,  # noqa: ANN401
    tmp_path: Path,
    prompt: str,
    *,
    access: str = "read",
) -> Any:  # noqa: ANN401
    """A child under a tier that hands out worktrees, `read` by default.

    `access` is a parameter rather than a second copy of the setup: the write
    case differs from the read case in exactly that word, and the two had already
    drifted apart on `mkdir(exist_ok=)` and on whether the provider was given a
    root.
    """
    parent_root = tmp_path / "parent-tree"
    parent_root.mkdir(exist_ok=True)
    await ctx.require(WORKSPACE).acquire(session_id="parent", agent_id=parent.id, base=parent_root)
    ctx.require(WORKSPACE).register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    return await _spawn(ctx, parent, prompt, access=access)


def _marks(ctx: Any, run: Any) -> list[str]:  # noqa: ANN401
    session = ctx.require(SESSIONS).get(run.session_id)
    return [
        str(event.data.get("retained", ""))
        for event in session.events
        if event.type == "workspace/retained"
    ]


async def test_a_child_is_retained_from_the_moment_its_tree_exists(
    delegating: MountedRuntime, tmp_path: Path
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
    run = await _tiered_child(ctx, parent, tmp_path, "get canceled")

    assert _marks(ctx, run) == ["the child has not settled cleanly"]
    (record,) = workspace_survivors(
        not_none(ctx.require(SESSIONS).get(run.session_id), "the child session")
    )
    assert record.outcome == "retained"


async def test_a_clean_child_leaves_nothing_behind(
    delegating: MountedRuntime, tmp_path: Path
) -> None:
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
    (record,) = workspace_survivors(
        not_none(ctx.require(SESSIONS).get(run.session_id), "the child session")
    )
    assert record.outcome != "retained", "a successful child's checkout is not evidence"
    assert record.reason == ""


async def test_a_canceled_child_keeps_its_evidence(
    delegating: MountedRuntime, tmp_path: Path
) -> None:
    """The case the row was written for, through the path that cannot mark.

    `parent-teardown` disposes the child's scope before the settle handler runs,
    so the retention this asserts is one nothing on that path could have set. It
    is there because it was taken at acquire and never withdrawn.
    """
    ctx, _session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "outlive me")
    child_session = ctx.require(SESSIONS).get(run.session_id)

    await ctx.require(AGENTS).dispose(parent.id)

    (record,) = workspace_survivors(not_none(child_session, "the child session"))
    assert record.outcome == "retained"
    assert record.reason == "the child has not settled cleanly"
    assert record.closed is True, "the pair still closed; only the discard was skipped"


async def test_a_write_child_is_not_retained(delegating: MountedRuntime, tmp_path: Path) -> None:
    """Only the kind that discards, because only that kind can lose evidence.

    An ordinary `worktree` already keeps a dirty tree for review and a committed
    branch survives release regardless, so retaining those would grow the pile
    without saving anything from it.
    """
    ctx, _session, parent = await delegating()

    run = await _tiered_child(ctx, parent, tmp_path, "write some code", access="write")

    assert _marks(ctx, run) == []


async def test_a_failed_child_tells_its_parent_where_the_tree_is(
    delegating: MountedRuntime, tmp_path: Path
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
    child = not_none(ctx.require(AGENTS).get(run.session_id), "the child agent")
    root = not_none(ctx.require(WORKSPACE).of(child.id), "the child workspace").root
    _breaks(child)
    await ctx.drain()

    (told,) = _notices(session)
    assert "failed" in told
    assert str(root) in told, "the parent was left to find the evidence itself"


def _breaks(agent: Any) -> None:  # noqa: ANN401
    """Make this agent's next run raise, which is what `_drive` catches."""

    async def broken() -> None:
        raise RuntimeError("the child broke")

    agent.run = broken


async def test_a_child_with_no_tree_is_announced_without_naming_one(
    delegating: MountedRuntime,
) -> None:
    """Silence rather than a fabricated path.

    A profile with no tier, a `shared` workspace, a child whose tree really was
    discarded — a notice that named a directory for every failure would be
    naming ones that are not there.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "fail with no workspace")
    _breaks(ctx.require(AGENTS).get(run.session_id))
    await ctx.drain()

    (told,) = _notices(session)
    assert "failed" in told
    assert "workspace is kept" not in told


# ---------------------------------------------------------------- the queue --


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
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """`maxConcurrent` is a queue: the parent gets every child it asked for, one
    slot at a time, in admission order — and never a refusal."""
    ctx, session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")

    children = ctx.require(SUBAGENTS).children(session.id)
    assert children[first.id].status == "running"
    assert children[second.id].status == "queued", "admitted, not refused — and waiting"
    assert _statuses(ctx, second) == ["queued"], "the wait is in its log"

    gate.release_one()
    await _until(lambda: gate.arrived == 2, "the second child to take the freed slot")
    assert ctx.require(SUBAGENTS).children(session.id)[first.id].status == "done"
    assert _statuses(ctx, second) == ["queued", "running"]

    gate.release_one()
    assert (await second.result()).status == "done"
    assert _statuses(ctx, first) == ["running", "done"], "no wait, no queued record"


async def test_a_child_that_failed_frees_its_slot(
    delegating: MountedRuntime, gate: ModelGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The queue cannot wedge on a failure: the provider's `error` path releases
    the slot exactly as `done` does.

    The failure is the child's *run* raising — the provider's own error path —
    rather than a model call failing, which the agent loop contains as a turn that
    ended in error and the provider records as `done`.
    """
    original_run = ReactLoopAgent.run
    failed: list[str] = []

    async def run(self: Any) -> None:  # noqa: ANN401
        # The parent is never run in this test, so the first `run()` is the first
        # child's; every later one is genuine.
        if not failed:
            failed.append(self.id)
            raise RuntimeError("the child fell over")
        await original_run(self)

    monkeypatch.setattr(ReactLoopAgent, "run", run)
    ctx, _session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")

    assert (await first.result()).status == "error"
    await _until(lambda: gate.arrived == 1, "the second child to run after the failure")
    gate.release_one()

    assert (await second.result()).status == "done"
    assert _statuses(ctx, first) == ["running", "error"]
    assert _statuses(ctx, second)[-2:] == ["running", "done"]


async def test_deleting_a_queued_child_stops_its_wait_and_takes_no_slot(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """A child revoked before it ran is canceled where it waits, and the slot it
    never held is not leaked — the next child still gets it."""
    ctx, session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")

    assert await ctx.require(SUBAGENTS).delete(session, second.id, reason="user") is True
    assert _statuses(ctx, second) == ["queued", "canceled"]

    third = await _spawn(ctx, parent, "third")
    gate.release_one()
    await _until(lambda: gate.arrived == 2, "the third child to take the slot the first freed")
    gate.release_one()
    assert (await first.result()).status == "done"
    assert (await third.result()).status == "done"
    assert _statuses(ctx, third) == ["queued", "running", "done"]


# ------------------------------------------------------------ across a restart --


async def _persisted(ctx: Context, session: Session) -> None:
    """Put the parent's own log on disk, with the harness holding still, so a restart
    has a root to resume.

    Only the parent's: each child's log is on its disk by its own barriers — the
    admission's flush before its gate opened, the checkpoint before each model
    request — and flushing a child here would put on disk what a crash might not
    have. A flush and nothing else. The first harness is parked at the model for the
    whole of these tests; draining here instead would wait on the very child that is
    meant to be caught mid-flight. `_restart` reads a snapshot of what is on disk, so
    nothing the first harness does afterwards reaches the second.
    """
    await ctx.require(SESSIONS).flush(session)


RETRIES = 3
"""This suite's own ladder bound.

Stated here rather than imported: `resume_children` takes the limit because it is
the *host's* policy, and a test that reached into `ph-app` for the daemon's
number would be asserting against a value it does not control — and coupling
`ph-rlm`'s tests to a package they do not depend on.
"""


async def _restart(
    mount: MountProfile,
    session_id: str,
    *,
    skills: tuple[str, ...] = (),
    concurrent: int = 1,
    prepare: Callable[[Context], None] | None = None,
) -> Any:  # noqa: ANN401
    """A second harness resuming one root from its log, as a restarted process would.

    What a daemon restart *is* from the seam's side: a fresh mount, nothing in
    memory, and a session that has to come off disk. `resume_children` is the
    call `Supervisor` makes at the same point, against the same agent.

    **Over a snapshot of the logs, not the live files** (`logs_after_a_crash`): the
    sessions directory as `_persisted` left it, since the first harness is still
    alive, parked, and holding its children's leases.

    `skills` is what the *deployment* still provides. It is a parameter because
    a readmit re-derives the child's ceiling against what the parent holds now,
    not against what it held then — so a skill this deployment no longer mounts
    is a child refused rather than one quietly readmitted without it.

    `prepare` sets up the restarted deployment before anything is resumed, as a
    daemon's profile would be: a route of its own, say.
    """
    ctx = await mount(
        dict(PROVIDER_ROW, config={"maxConcurrent": concurrent}),
        {"id": "session-persistence", "config": {"root": str(logs_after_a_crash())}},
    )
    for name in skills:
        ctx.require(SKILLS).register(skill(name))
    if prepare is not None:
        prepare(ctx)
    session = await resume_session(ctx, session_id)
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=RETRIES)
    return ctx, session, parent


async def test_a_live_childs_log_refuses_a_second_opener(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """L2. A child's log is a session of its own, openable by its id — `phern -p
    --session <child>`, a daemon's `session/new` naming it — so while this harness
    drives the child, a second opener is refused as it would be for a root (I-5),
    rather than made a second writer on one log.

    Sabotage: open the child with `resume_session` or `sessions.create` directly in
    `_child_session`, and the second open is granted.
    """
    ctx, _session, parent = await delegating()
    child = await _spawn(ctx, parent, "first")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    elsewhere = await mount()

    with pytest.raises(SessionBusy):
        await open_session(elsewhere, child.session_id)


async def test_a_queued_child_is_re_driven_after_a_restart(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """The work was described in the child's log and running nowhere (P5-04).

    A child that never reached its first turn has claimed nothing and spent
    nothing, so the next harness runs it for the first time — under its original
    id, so the parent gains no second child it never asked for.

    Reaching the model is the proof of "re-driven": the gate counts arrivals, and
    the readmitted child is the only one the second harness can run.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")
    assert _statuses(ctx, second) == ["queued"]
    await _persisted(ctx, session)

    # Room for both, so which one this asserts about is not a race: the
    # interrupted sibling is on the ladder and comes back too.
    revived_ctx, revived, _parent = await _restart(mount, session.id, concurrent=2)

    assert second.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}, (
        "the queued child came back under its own id"
    )
    await _until(
        lambda: _statuses(revived_ctx, second)[-1] == "running",
        "the readmitted child to reach the model",
    )
    assert len(revived_ctx.require(SUBAGENTS).children(revived.id)) == 2, (
        "no child was invented or lost"
    )
    assert _state(revived_ctx, second).resumes == 0, (
        "a child that never ran is a first attempt, not a retry"
    )


async def test_a_child_caught_mid_turn_climbs_the_ladder_with_its_task_re_presented(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
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
    assert _statuses(ctx, interrupted) == ["running"]
    await _persisted(ctx, session)

    revived_ctx, _revived, _parent = await _restart(mount, session.id)

    assert interrupted.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}
    assert child_is_live(_state(revived_ctx, interrupted)), "a child owed a turn is live"
    # The count follows the restart's own `running` record, which a detached
    # drive job writes a moment later — so this waits for the fact rather than
    # reading the child's state before it exists.
    await _until(
        lambda: restarts_since_progress(_state(revived_ctx, interrupted)) == 1,
        "the restart to be counted",
    )
    assert _state(revived_ctx, interrupted).starts == 2, "one first run, one restart"
    await _until(gate.twice, "the resumed child to reach the model again")

    child = revived_ctx.require(SESSIONS).get(interrupted.session_id)
    assert child is not None
    tasks = [
        text_of(not_none(derive_event_message(event)).content)
        for event in child.events
        if event.type == "user/message" and TASK_PREFIX in repr(event.data)
    ]
    assert len(tasks) == 2, "the task was not presented again, so the retry answers nothing"
    assert "the harness stopped while you were working on this" in tasks[-1]
    assert "this is attempt 2" in tasks[-1]


async def test_a_mount_that_unwinds_suspends_its_children_rather_than_revoking_them(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """S1 — a clean stop tombstoned every child, so a restart brought none back.

    A daemon stopping, or a root remounted on a new profile, unwinds the parent's
    scope, and each child's release wrote `canceled` and `subagent/deleted` as it
    went: `resume_children` skips a deleted child, so an ordinary restart abandoned
    every delegation in flight. The mount going away now suspends its children —
    `queued`, saying why, in each child's own log, and no tombstone — and the next
    mount readmits them.

    Sabotage: drop the `suspend` disposer from `apply`, and the release raises on
    the way down and writes nothing, so the restart reads the child as crashed.
    """
    ctx, session, parent = await delegating()
    working = await _spawn(ctx, parent, "keep going")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")

    await ctx.root.dispose()
    revived_ctx, _revived, _parent = await _restart(mount, session.id)

    assert not _state(revived_ctx, working).deleted, "the stop revoked the child"
    assert SUSPENDED_DETAIL in _statuses(revived_ctx, working, "detail"), (
        "the child's log does not say why it stopped"
    )
    assert working.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}
    await _until(gate.twice, "the resumed child to reach the model again")


async def test_a_suspended_child_keeps_what_was_queued_for_it(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A child coming back comes back to its inbox, not an empty one.

    `suspend` stops the child's agent keeping its inbox, where a revocation clears
    it — so a message steered to a working child and not yet taken is still pending
    when the next mount readmits it.

    Sabotage: cancel the agent in `suspend` without `keep_inbox`, and the message is
    canceled in the child's log.
    """
    ctx, _session, parent = await delegating()
    working = await _spawn(ctx, parent, "keep going")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    child = not_none(ctx.require(AGENTS).get(working.session_id))
    child.steer(user_text("also this"))

    await ctx.root.dispose()
    revived_ctx, _revived, _parent = await _restart(mount, parent.id)
    await _until(gate.twice, "the resumed child to reach the model again")

    log = not_none(revived_ctx.require(SESSIONS).get(working.session_id)).events
    assert not [
        event
        for event in log
        if event.type == "agent/inbox/spliced" and event.data.get("outcome") == "canceled"
    ], "what was queued for the child was canceled as it stopped"
    assert "also this" in repr([event.data for event in log if event.type == "agent/inbox/spliced"])


async def test_a_child_suspended_before_its_first_step_is_handed_its_task_once(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A child that never took a step still has its task, in its own log.

    The inbox gives up a batch only in the step that records it (S3), so a child
    queued behind a sibling when its mount unwound comes back with the task still
    pending — and presenting it again on readmit handed it the same work twice.

    Sabotage: present the task in `_admit` whatever the inbox holds, and the
    child's log inserts it twice.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    await _spawn(ctx, parent, "first")
    waiting = await _spawn(ctx, parent, "second")
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")

    await ctx.root.dispose()
    revived_ctx, _revived, _parent = await _restart(mount, session.id, concurrent=2)
    await _until(lambda: gate.arrived >= 3, "both children to reach the model again")

    child = revived_ctx.require(SESSIONS).get(waiting.session_id)
    assert child is not None
    presented = [
        event
        for event in child.events
        if event.type == "agent/inbox/spliced" and TASK_PREFIX in repr(event.data.get("inserted"))
    ]
    assert len(presented) == 1, "the readmit handed the child its task a second time"


async def test_a_settled_child_is_not_canceled_when_its_parent_goes(
    delegating: MountedRuntime,
) -> None:
    """A child that finished keeps its ending.

    Its parent's teardown released it with `canceled`, written over the `done` its
    drive had recorded, so every finished child read as a revoked one once its
    parent was gone.

    Sabotage: write `canceled` in `_release` whatever the child's result, and the
    child's log says canceled.
    """
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "finish first")
    await ctx.drain()

    await ctx.require(AGENTS).dispose(parent.id)

    state = _state(ctx, run)
    assert state.status == "done"
    assert state.deleted is True, "released with its parent all the same"


async def test_a_restart_is_on_disk_before_the_attempt_it_counts(
    delegating: MountedRuntime,
    gate: ModelGate,
    mount: MountProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """S10 — the ladder counts `running{cause: resumed}`, so that record is on the
    child's own disk before the attempt it counts.

    A restart that reached only memory before the child took the daemon down again
    was never counted: a crash loop never advanced the count on disk, and the ladder
    never gave up. It used to ride the parent's log, which nothing flushed between
    readmitting a child and the child working.

    Asked as the attempt starts — the drive's call into the child's run — rather
    than at the model: the child's first model request flushes its log anyway, so by
    then the question is moot.

    Sabotage: drop the flush from `ph.seams.subagents.record_started`, and the stored
    child has no restart when its attempt begins.
    """
    ctx, session, parent = await delegating()
    interrupted = await _spawn(ctx, parent, "only")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)
    revived: list[Context] = []
    seen: list[int] = []
    original = ReactLoopAgent.run

    async def watched(self: ReactLoopAgent) -> None:
        # The first harness's child is already inside its own `run`, and the revived
        # parent is never run, so the one call this sees is the resumed child's.
        if self.id == interrupted.session_id:
            seen.append(restarts_since_progress(_stored_state(revived[0], interrupted)))
        await original(self)

    monkeypatch.setattr(ReactLoopAgent, "run", watched)
    await _restart(mount, session.id, prepare=revived.append)
    await _until(gate.twice, "the resumed child to reach the model again")

    assert seen == [1], "the attempt began before its restart was on the child's disk"


async def _resumed(ctx: Context, run: SubagentRun, times: int) -> None:
    """Record `times` restarts in the child's own log, the way a restart actually
    records one, and put them on its disk as the door does.

    The real facts rather than a seeded count: the ladder folds `running` records
    carrying `cause: resumed`, so a test that wrote an `attempts` number would be
    asserting against a field production no longer has.
    """
    child = not_none(ctx.require(SESSIONS).get(run.session_id))
    for _ in range(times):
        log_event(child, STATUS, {"status": "running", "cause": "resumed"})
    await ctx.require(SESSIONS).flush(child)


async def _stalled(
    ctx: Context,
    parent: object,
    gate: ModelGate,
    *,
    restarts: int,
) -> Any:  # noqa: ANN401
    """A child at the model that has already been restarted `restarts` times."""
    child = await _spawn(ctx, parent, "only")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    await _resumed(ctx, child, restarts)
    return child


async def test_the_ladder_gives_up_and_says_so(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """Three restarts with nothing achieved between them is not bad luck.

    Re-driving forever would spend a parent's budget on a transcript nobody
    reads, which is the failure the root's own ladder is bounded to avoid.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    spent = await _stalled(ctx, parent, gate, restarts=RETRIES)
    assert restarts_since_progress(_state(ctx, spent)) == RETRIES
    await _persisted(ctx, session)

    revived_ctx, revived, _parent = await _restart(mount, session.id)

    assert revived_ctx.require(SUBAGENTS).list() == [], "a spent ladder put a child back to work"
    state = _state(revived_ctx, spent)
    assert state.status == "error"
    assert state.detail == exhausted_detail(RETRIES)
    assert str(RETRIES) in state.detail, "the sentence names the bound it hit"
    assert not child_is_live(state), "a root cannot be passivated while this reads live"
    # Ended in its own log, on its own disk — the one place the next start will look.
    assert _stored_state(revived_ctx, spent).status == "error"
    assert _about_children(revived) == []


async def test_progress_since_the_last_restart_clears_the_ladder(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A child stopped, working an hour, then stopped again met two incidents.

    Without a reset the ladder counts a lifetime's interruptions rather than
    consecutive ones, and fails work that was going fine. The same setup as the
    test above plus one fact: the child gave a model answer, in its own log, which is
    something a turn that did nothing cannot produce.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    moved = await _stalled(ctx, parent, gate, restarts=RETRIES)
    await _answered_on_its_own_disk(ctx, moved)
    state = _state(ctx, moved)
    assert restarts_since_progress(state) == 0, "progress forgives the restarts before it"
    assert state.resumes == RETRIES, "and the child's state still says how many there were"
    await _persisted(ctx, session)

    revived_ctx, _revived, _parent = await _restart(mount, session.id)

    assert child_is_live(_state(revived_ctx, moved)), "a child that got somewhere is owed more"
    assert moved.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}
    # **The restart is still recorded as one**, counting up from the answer.
    # Derived from the ladder instead, this readmit would look like a first run,
    # write no `resumed`, and the ladder would never count it again.
    await _until(
        lambda: restarts_since_progress(_state(revived_ctx, moved)) == 1,
        "the restart to be counted from a cleared ladder",
    )


CHILD_ANSWER_USAGE = {"inputTokens": 120, "outputTokens": 30}


async def _answered_on_its_own_disk(ctx: Context, run: SubagentRun) -> SessionEvent:
    """The child answers, and its own log reaches its disk.

    What its checkpoint does before its next request, while a parent waiting on it
    makes no request and writes nothing. The answer's `usage` is the whole record of
    what it spent: nothing is copied anywhere else.
    """
    child = not_none(ctx.require(SESSIONS).get(run.session_id))
    payload = {**assistant_payload("found it", "a1"), "usage": CHILD_ANSWER_USAGE}
    answer = log_event(child, "assistant/message", payload, SurfaceIntent("append", ()))
    await ctx.require(SESSIONS).flush(child)
    return answer


@pytest.mark.parametrize("parent_written", ["before-the-answer", "after-the-answer"])
async def test_a_crash_after_an_answer_is_counted_exactly(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile, parent_written: str
) -> None:
    """L5, fixed by construction. A child answers, only the child's log is written,
    and the mount restarts: the ladder forgives the restarts before the answer, and
    the budget counts what the answer spent — once — with no catch-up step.

    Two decisions read a child's answers. The ladder takes an answer as progress, so
    a child whose answer a restart could not see read as stuck, and after `RETRIES`
    restarts was failed as exhausted while it was getting somewhere. A goal's budget
    counts what the answer spent. Both used to read a copy of each answer mirrored
    into the parent's log, which a crash could leave behind the child's own — so a
    resume had to copy the missing answers back, and not copy one twice when the
    parent's log had reached disk after it. Both read the child's own log now, which
    is where the answer is, whenever the parent's log was last written.

    Sabotage: have the fold skip `assistant/message`, as a reader of anything but the
    child's own answers would, and the child is failed as exhausted with nothing
    counted.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    moved = await _stalled(ctx, parent, gate, restarts=RETRIES)
    if parent_written == "before-the-answer":
        await _persisted(ctx, session)
    await _answered_on_its_own_disk(ctx, moved)
    if parent_written == "after-the-answer":
        await _persisted(ctx, session)
    assert _about_children(session) == [], "nothing about the answer reached the parent's log"

    revived_ctx, revived, _parent = await _restart(mount, session.id)

    state = _state(revived_ctx, moved)
    assert state.status != "error", f"failed as {state.detail!r} though it had answered"
    assert moved.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}
    assert state.tokens == 150, "the budget reads what the child's answer spent, once"
    assert revived_ctx.require(SUBAGENTS).delegated_tokens(revived.id) == 150
    assert _about_children(revived) == [], "and the resume wrote no catch-up into the parent"


async def _grandchild(
    ctx: Context,
    parent: Any,  # noqa: ANN401
    gate: ModelGate,
    **leaf: Any,  # noqa: ANN401
) -> tuple[Any, Any]:
    """A child and the child it delegated to, both at the model, both logs written."""
    middle = await _spawn(ctx, parent, "delegate further")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    child = not_none(ctx.require(AGENTS).get(middle.session_id))
    grandchild = await _spawn(ctx, child, "do the work", **leaf)
    await _until(lambda: gate.arrived == 2, "the grandchild to reach the model")
    await _persisted(ctx, not_none(parent.session))
    await _persisted(ctx, not_none(child.session))
    return middle, grandchild


async def test_a_grandchild_the_restart_interrupted_is_put_back_to_work_too(
    delegating: MountedRuntime,
    gate: ModelGate,
    mount: MountProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L5b. A readmitted child's own children are swept the way the root's are.

    `RLM_MAX_DEPTH` lets a child delegate, and its children name *it* as their
    parent. The resume swept the root's children alone, so a grandchild caught
    mid-turn stayed `running` with nothing driving it, readmitted or failed, and its
    parent's model was told it was still working.

    **Before the child's first step** (`SubagentRun.ready`): the grandchild is
    readmitted ahead of the child's new attempt, so the children the child's first
    request is built from are the swept ones.

    Sabotage: drop `_sweep_readmitted` from `_readmit_children` and the grandchild
    is never readmitted; open a readmitted child's gate at admission and its new
    attempt starts ahead of the sweep.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    middle, leaf = await _grandchild(ctx, parent, gate)
    revived: list[Context] = []
    swept_first: list[bool] = []
    original = record_started

    async def watched(
        scope: Context, child: Session, *, cause: StatusCause | None = None
    ) -> SessionEvent:
        if child.id == middle.session_id:
            readmitted = {run.id for run in revived[0].require(SUBAGENTS).list()}
            swept_first.append(leaf.id in readmitted)
        return await original(scope, child, cause=cause)

    monkeypatch.setattr("ph_rlm.subagents.record_started", watched)
    revived_ctx, _revived, _parent = await _restart(mount, session.id, prepare=revived.append)

    assert leaf.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}, (
        "the grandchild came back under its own id"
    )
    await _until(
        lambda: restarts_since_progress(_state(revived_ctx, leaf)) == 1,
        "the grandchild's restart to be counted in its own log",
    )
    await _until(lambda: gate.arrived == 4, "both to reach the model again")
    assert swept_first == [True], "the child's new attempt started before its children were swept"


GRANDCHILD_KEY = "PH_L5B_GRANDCHILD_KEY"
"""What the `keyed` route's adapter resolves at its edge, and the environment lacks."""


def _keyed(ctx: Context) -> None:
    """A route of its own, whose adapter names `GRANDCHILD_KEY`."""
    ctx.require(LLM).register_adapter(
        ("keyed",),
        FakeAdapter(respond=text_script("done"), route=ResolvedModel(credential=GRANDCHILD_KEY)),
    )


async def test_a_grandchild_held_for_its_key_is_released_when_the_key_arrives(
    delegating: MountedRuntime,
    gate: ModelGate,
    mount: MountProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L5b, the credential half. A grandchild waiting for a name is released by it.

    The nested sweep holds a grandchild whose route names a credential the restarted
    deployment lacks, in the grandchild's own log (T5) — the hold a session waiting on
    its own route writes. `readmit_waiting` asked only the root's children when the
    name arrived, so that hold was never released.

    Sabotage: stop `readmit_waiting` descending into live children, and the grandchild
    stays held with its key supplied.
    """
    monkeypatch.delenv(GRANDCHILD_KEY, raising=False)
    ctx, session, parent = await delegating(maxConcurrent=1)
    _keyed(ctx)
    middle, leaf = await _grandchild(ctx, parent, gate, provider="keyed", model="k1")

    revived_ctx, _revived, revived_parent = await _restart(mount, session.id, prepare=_keyed)
    middle_log = not_none(revived_ctx.require(SESSIONS).get(middle.session_id))
    assert _state(revived_ctx, leaf).awaiting == GRANDCHILD_KEY, "held by name"
    assert _stored_state(revived_ctx, leaf).awaiting == GRANDCHILD_KEY, "in its own log, on disk"
    assert waiting_for(revived_ctx, middle_log) == {}, "and not in its parent's"

    revived_ctx.require(CREDENTIALS).provide_value(GRANDCHILD_KEY, "supplied")
    revived = await revived_ctx.require(SUBAGENTS).readmit_waiting(
        revived_parent, retry_limit=RETRIES
    )

    assert leaf.id in revived, "the key arrived and the grandchild stayed held"
    assert _state(revived_ctx, leaf).awaiting is None


async def test_a_readmitted_child_does_not_come_back_wider_than_it_was_admitted(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """§6.5 across a power cut, which is why the narrowing is in the record.

    The ceiling is re-derived from the admission, so a child admitted with one
    skill does not return holding every skill its parent has. Rebuilt from the
    run alone it would, and nothing would have said so. Its writable directories
    too (S7b, item 3). Sabotage: drop `paths` from `_request_of`, and the child
    comes back binding its parent's.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    ctx.require(SKILLS).register(skill("review"))
    ctx.require(SKILLS).register(skill("audit"))
    await _spawn(ctx, parent, "first")
    narrowed = await _spawn(
        ctx, parent, "second", skills=("review",), tools=("read",), paths=("/srv/cache",)
    )
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")
    await _persisted(ctx, session)

    revived_ctx, _revived, _parent = await _restart(mount, session.id, skills=("review", "audit"))

    # Read back off the child's own log as the restart found it, which is the only
    # copy a restart has.
    admitted = next(
        event.data for event in _log_of(revived_ctx, narrowed) if event.type == ADMITTED
    )
    assert list(as_seq(admitted["skills"])) == ["review"], "the narrowing survives the round trip"
    assert list(as_seq(admitted["tools"])) == ["read"]

    back = next(run for run in revived_ctx.require(SUBAGENTS).list() if run.id == narrowed.id)
    assert back.grant is not None
    assert back.grant.skills == ("review",)
    assert back.grant.tools == ("read",)
    assert back.grant.paths == ("/srv/cache",)


async def test_a_child_no_provider_can_resume_is_settled_not_left_queued(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A `queued` row nothing will pick up is worse than an honest failure.

    It reads as live, so the root can never be passivated, and the parent waits
    on a slot no one will ever give it. Deciding where the capability probe
    answers is what stops the sweep writing a status it cannot honor — here the
    next harness mounts no subagent provider at all.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    orphan = await _spawn(ctx, parent, "only")
    await _until(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)

    # Over a snapshot, as `_restart` is: ending the child means claiming its log,
    # which the first harness — alive, and parked at the model — still holds.
    bare = await mount({"id": "session-persistence", "config": {"root": str(logs_after_a_crash())}})
    revived = await resume_session(bare, session.id)
    await bare.require(SUBAGENTS).resume_children(
        bare.require(AGENTS).create(revived, FAKE_OPTIONS), retry_limit=RETRIES
    )

    state = _state(bare, orphan)
    assert state.status == "error"
    assert state.detail == UNRECOVERABLE_DETAIL
    assert not child_is_live(state), "a root cannot be passivated while this reads live"
    assert _stored_state(bare, orphan).status == "error", "ended in its own log, on disk"


async def test_a_child_the_ceiling_now_refuses_is_settled_rather_than_skipped(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """K4 — a readmit that *raises* used to be logged and then dropped.

    A readmit re-derives the ceiling from the admission record against what the
    parent holds **now**, so a deployment that stopped mounting a skill refuses
    the child that was narrowed to it. That refusal is correct — the sibling test
    above is why the narrowing is logged at all. What was wrong is what followed:
    the exception was logged and the loop moved on, leaving the row `queued`,
    which `child_is_live` reads as waiting for a slot. The parent then waits for
    a child nothing will ever drive, and the root can never be passivated.

    Distinct from the no-provider case below it: there the sweep *decides* it
    cannot recover the child, here a readmit it fully intended threw on the way.
    The two reached the log differently and only one of them reached it at all.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    ctx.require(SKILLS).register(skill("review"))
    await _spawn(ctx, parent, "first")
    narrowed = await _spawn(ctx, parent, "second", skills=("review",))
    await _until(lambda: gate.arrived == 1, "the first child to reach the model")
    await _persisted(ctx, session)

    # The deployment no longer mounts `review`, so the parent does not hold it
    # and `check_grant` refuses the child that was admitted with it.
    revived_ctx, _revived, _parent = await _restart(mount, session.id, concurrent=2)

    state = _state(revived_ctx, narrowed)
    assert state.status == "error"
    assert "could not be resumed" in not_none(state.detail)
    assert not child_is_live(state), "a root cannot be passivated while this reads live"


# ------------------------------------------------- a child asked a second thing --


@pytest.mark.needs_git
async def test_a_re_addressed_child_comes_back_to_its_own_work(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A second question reaches the child that answered the first, tree and all.

    Disposal commits the checkout to the child's branch before removing it, and
    `_add` attaches an existing branch rather than resetting it — so the same run
    id resolves to the same branch, and re-acquiring restores what the child did.
    Nothing arranges that here; what was missing is that `rehydrate` never asked.
    """
    ctx = await mount(*WORKTREE_ROWS, PROVIDER_ROW)
    base = await git_repo(ctx, tmp_path / "repo")
    session = ctx.require(SESSIONS).create("parent")
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await ctx.require(WORKSPACE).acquire(
        session_id=session.id, agent_id=parent.id, base=base, access="write", session=session
    )

    run = await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="first question", parent=parent, access="write")
    )
    first = ctx.require(WORKSPACE).of(run.session_id)
    assert first is not None and first.kind == "worktree"
    (first.root / "child-work.txt").write_text("what the first answer produced\n", encoding="utf-8")
    await ctx.drain()
    assert not first.root.exists(), "the checkout goes; the branch is what survives"

    assert await ctx.require(SUBAGENTS).ensure_addressable(run.session_id) is True

    again = ctx.require(WORKSPACE).of(run.session_id)
    assert again is not None, "a child given a runtime back and no tree writes into its parent's"
    assert again.root == first.root, "the same child, so the same checkout"
    assert again.ref == first.ref
    assert (again.root / "child-work.txt").read_text(encoding="utf-8").startswith("what the first")


@pytest.mark.needs_git
async def test_a_re_addressed_child_is_no_wider_than_it_was_admitted(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The containment half. A child admitted `read` comes back ephemeral.

    Its access is read from the admission, not from the caller re-addressing it,
    so nothing about being asked a second question can widen what the first was
    allowed to keep.
    """
    ctx = await mount(*WORKTREE_ROWS, PROVIDER_ROW)
    base = await git_repo(ctx, tmp_path / "repo")
    session = ctx.require(SESSIONS).create("parent")
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await ctx.require(WORKSPACE).acquire(
        session_id=session.id, agent_id=parent.id, base=base, access="write", session=session
    )

    run = await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt="look only", parent=parent, access="read")
    )
    assert not_none(ctx.require(WORKSPACE).of(run.session_id)).kind == "worktree-ephemeral"
    await ctx.drain()

    assert await ctx.require(SUBAGENTS).ensure_addressable(run.session_id) is True

    again = ctx.require(WORKSPACE).of(run.session_id)
    assert again is not None
    assert again.kind == "worktree-ephemeral", "a second question must not widen the first's grant"
