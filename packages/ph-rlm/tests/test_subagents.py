"""Delegation: admission, the depth gate, and what the parent is told (P3-11).

The main claim is **non-blocking admission**: `start()` returns when the child
is admitted, not when it has answered. Everything else here is about how the
parent can act on that handle:

- The state of a child is read from its own log.
- A silent child is announced.
- What a child spent is counted where the child spent it.
- A revoked child leaves a tombstone.

## Each session owns its log (Phase 11)

Every record about a child is in the log of **the child itself**: its admission,
each start, each wait, its ending and its tombstone. Nothing about the child is
written to the log of its parent. These tests therefore read the records of a
child from the child (`_log_of`, `_state`). The tests about durability ask the
store for the log of the child, not for the log of the parent.

The parent used to keep a roster of its children in its own log. Every
durability rule in this file existed to keep that second account in step with
the first. The parent flushed before the gate of a child opened (S2), before a
restart (S10), and after the log of the child (F1). A resume copied the answers
that a crash had left only in the log of the child back into the log of the
parent (L5). With one account there is nothing to keep in step, and each rule is
about one log. `test_a_parents_log_holds_no_record_of_its_child` keeps it that
way.

## The spend of a child was never a correction to the context of its parent

`TokenMeter.last_usage` folds only the `assistant/message` events of the log
that it is *given*, and the `assistant/message` events of a child are in the log
of **the child**. The context measurement of the parent therefore never included
them, and there is nothing to subtract. This package used to write a mirror into
the log of the parent (`subagent/usage-attributed`). The mirror was additive and
was for readers: the budget of a goal, the retry ladder and the TUI panel. Those
readers now read the answers of the child itself (`ChildState.tokens`).

This is written down because the false version sounded important. It implied
that a fan-out of eight would otherwise read as context pressure on the parent
and trigger a compaction that the parent does not need. It would not, and no
code path depends on a record in the log of the parent to prevent it.

## Why the parent check is above `agents.create` in `rehydrate`

With the check below `agents.create`, a rehydration with no live parent built an
agent and a scope, failed, and left both behind: **an orphan under the registry
root that held the deployment-wide ceiling and that nothing would ever
dispose**. The parent owns the drive job *and*, since P6-27, the scope that the
child nests in. A missing parent is therefore a refusal and not a degradation.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from rlm_fixtures import (
    BINDINGS_ROW,
    MODELS_ROW,
    PROVIDER_ROW,
    ModelGate,
    MountedRuntime,
    logs_after_a_crash,
    reply_to_parent,
)

from ph.agent.types import AgentHandle
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
from ph.persistence import SessionBusy, open_session
from ph.seams.credentials import waiting_for
from ph.seams.models import ModelChoice
from ph.seams.subagents import (
    ADMITTED,
    DELETED,
    PARENT_TEARDOWN,
    STATUS,
    SUSPENDED_DETAIL,
    UNRECOVERABLE_DETAIL,
    ChildState,
    ModelFallback,
    StatusCause,
    SubagentRequest,
    SubagentRun,
    SubagentService,
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
    delivered_ids,
    log_event,
    not_none,
    raising,
    reconciled_call,
    resume_stored,
    run_tool,
    settle_background,
    settled,
    skill,
    stored_events,
    stored_types,
)
from ph.testing.git import WORKTREE_ROWS, git_repo
from ph.tools.definition import NotDone
from ph_rlm.bindings import RUN_TOOL
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
    """The log of one child: the events of its live session, or else what the
    store holds.

    The live session is used while this process has the child open, because the
    record functions of the seam write to it. The stored log is used for a child
    that this process let go. An example is a child that a resume sweep ended
    and did not readmit: the sweep opened its log, wrote it, flushed it and
    closed it again. For such a child, the disk holds the only copy.
    """
    live = ctx.require(SESSIONS).get(run.session_id)
    return list(live.events) if live is not None else stored_events(ctx, run.session_id)


def _state(ctx: Context, run: SubagentRun) -> ChildState:
    """The state of one child as its own log gives it. The state is read through
    the service that every reader asks: the prompt, the roster tool, the resume
    sweep and the budgets."""
    return not_none(ctx.require(SUBAGENTS).state(run.session_id), f"the child {run.id}")


def _stored_state(ctx: Context, run: SubagentRun) -> ChildState:
    """The state of one child as the **store** holds it. A resume would receive
    this state, so a durability rule is about this state."""
    header, events = ctx.require(SESSION_PERSISTENCE).read(run.session_id)
    return child_state_of(run.session_id, header, events)


def _statuses(ctx: Context, run: SubagentRun, field: str = "status") -> list[str]:
    """Every status that this child reached, in order, or another `field` of the
    same records. The records come from its own log, which is the only place
    where a status is written."""
    return [str(event.data.get(field)) for event in _log_of(ctx, run) if event.type == STATUS]


def _teardowns(parent: AgentHandle) -> list[str]:
    """The children whose revocation is still registered on `parent`'s scope."""
    prefix = "subagent:"
    return [
        effect.label.removeprefix(prefix)
        for effect in parent.ctx._effects
        if effect.label.startswith(prefix)
    ]


def _about_children(session: Session) -> list[str]:
    """The `subagent/*` records in the log of a parent. By design there are none."""
    return [event.type for event in session.events if event.type.startswith("subagent/")]


# ------------------------------------------------------------------ admission --


@pytest.mark.parametrize("step", ["building-its-workspace", "starting-its-drive"])
async def test_a_parent_that_goes_away_mid_admission_refuses_the_child(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch, step: str
) -> None:
    """A spawn is refused when the scope of its parent is disposed during one of
    its awaits.

    The failure used to escape as a raw `InactiveScopeError` from the first
    registration that met the dead scope: a workspace effect on the scope of the
    child, or the release effect or the job on the scope of the parent. The
    error carried no spawn code. A child that the spawn had already started was
    left open, and nothing would ever end it.

    Now the failure is a `SubagentSpawnError`, and what was built is released.
    While the child is being built, its session is let go. After its drive
    exists, the child is tombstoned in its own log. In both cases the seam never
    writes the admission, so the log names no child, and nothing reads as a live
    child of the parent.

    Sabotage: in `RlmChildProvider._admit`, drop the `except BaseException`
    around `_workspace` or the `except InactiveScopeError`.
    """
    ctx, session, parent = await delegating()
    # The build of the child has two awaits: its workspace, and the job that
    # drives it. The job registers on the scope of the parent.
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
    """The property that the design exists for: a parent fans out and continues
    to work. It does not block on each child in turn."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)

    # Admitted, not finished. The record that the child exists is written in its
    # own log and not in the log of its parent. Nothing has reported on the child
    # yet.
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
    """The drive of the child waits for its admission (`SubagentRun.ready`).
    Every status that the drive writes therefore follows the admission in the
    log of the child.

    The fold no longer needs the order: a log with statuses and no admission is
    not a child, and every reader skips it. But a drive that ran ahead of its
    admission would be a child that works before any record says that it exists.
    That is the failure of S2, seen from the other side."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await settle_background(ctx)

    kinds = [event.type for event in _log_of(ctx, run) if event.type.startswith("subagent/")]
    assert kinds[0] == ADMITTED
    assert STATUS in kinds


async def test_the_admission_is_on_disk_before_the_child_takes_a_step(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S2: the admission is the one record by which a resume finds a child.

    The admission is in the log of the child, and the seam flushes that log
    before it opens the gate of the child. A crash at any time after the first
    step of the child therefore leaves a log on disk that says what the child
    was asked, by whom, and under which ceiling. The store lists the children of
    a parent by that record (`descendants_of`), and the sweep readmits them from
    it. The admission used to be held in memory in the log of the parent, and
    the flushes of the child never wrote the log of its parent. A crash in the
    middle of a run therefore left a child with a log and a tree that nothing on
    disk named.

    The test asks at the gate, not at the model. The first model request of the
    child flushes its log anyway, so at that point the question no longer
    matters. The first act of the drive after the gate is its `running` record,
    so the test reads the store there.

    Sabotage: drop the flush of the child from `SubagentService._admit`, and the
    stored log is empty when the child starts.
    """
    ctx, _session, parent = await delegating()
    seen: list[list[str]] = []
    original = record_started

    async def watched(
        scope: Context, child: Session, *, cause: StatusCause | None = None
    ) -> SessionEvent | None:
        seen.append(stored_types(ctx, child.id))
        return await original(scope, child, cause=cause)

    monkeypatch.setattr("ph_rlm.subagents.record_started", watched)
    await _spawn(ctx, parent, "work")
    await settle_background(ctx)

    assert len(seen) == 1, "the child never started"
    assert ADMITTED in seen[0], "a child started and its own log on disk does not say so"


async def test_a_child_whose_admission_cannot_be_written_is_not_started(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """Fail-closed, like a durable intent: no admission on disk means no child.

    The refusal reaches the caller, the child never reaches the model, and the
    child is ended in its own log. Nothing therefore keeps the parent out of
    passivation for a child that never ran. The child is not revoked, because a
    refusal is not a deletion. The release that used to tombstone it said
    `parent-teardown` about a parent that was still there.

    Sabotage: open the gate whether or not the write worked, and the child runs.
    Point `run.dispose` back at `_release(run_id, PARENT_TEARDOWN)`, and the
    child is deleted.
    """
    ctx, session, parent = await delegating()

    def refuse(target: Session) -> None:
        # The log of the child is the log that the admission is written to. Its
        # id is created inside the spawn, and its header names the parent.
        if target.header.delegating_parent == session.id:
            raise OSError("the disk is full")

    ctx.on("session/flush", refuse)
    with pytest.raises(SubagentSpawnError, match="could not be written"):
        await _spawn(ctx, parent, "never")
    await settle_background(ctx)

    assert gate.arrived == 0, "the child ran without its admission on disk"
    (state,) = ctx.require(SUBAGENTS).children(session.id).values()
    assert state.status == "error"
    assert not child_is_live(state)
    assert not state.deleted, "a refusal is not a revocation"


async def test_an_interrupted_spawn_is_answered_with_the_handle_it_admitted(
    delegating: MountedRuntime,
) -> None:
    """S2: an `rlm.run` that a crash cut short is found among the children of its
    parent by its call.

    The whole value of an `rlm.run` is the admission handle. The admission is on
    disk in the log of the child before the child runs, and it carries the id of
    the call. A resume can therefore show the program the handle that it would
    have had. The alternative is "outcome unknown", and on that answer a model
    spawns the same child again beside the one that the resume has already put
    back to work. If no admission exists under the call, the spawn never
    happened.

    Sabotage: drop `call_id` from the request that `run_child` builds, and the
    admitted child is not found.
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
    await settle_background(ctx)

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
    """Prime-agent's wording, so a model that has seen it need not re-learn it."""
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
    """No adapter means no child. The spawn is refused where the route is chosen
    (`choose`). Nothing is substituted, because a child that answers on a model
    that the parent did not choose gives a result that the parent cannot
    interpret."""
    ctx, _session, parent = await delegating()
    with pytest.raises(SubagentSpawnError, match='no adapter here serves "nonexistent"'):
        await _spawn(ctx, parent, choice=ModelChoice(provider="nonexistent", model="m1"))


async def test_the_default_shapes_the_request_and_the_tier_answers_it(
    delegating: MountedRuntime,
) -> None:
    """E4 and E3 together, at the `advisory` tier, which is the shipped default.

    The default is about the *request*. An omitted `access` asks for `read`, and
    that makes a child suited to research whatever the tier does with the
    request. What comes back is the answer of the tier. At `advisory` the answer
    to both requests is the same shared checkout, because **nothing here can
    make a repository read-only**. (The table in §4.8 says exactly this, and
    calls the row "not read-only".) So `granted` says `write` for a `read`
    request, and the pair of requested and granted is the correct record. A
    claim of `read` would be a promise that nothing keeps, and this field must
    never make such a promise.
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
    # Nothing was *narrowed*, so there is no downgrade to report. The widening
    # that a `read` request gets at this tier is visible in the pair itself, and
    # the workspace line in the prompt of the child states it plainly.
    assert asked.downgrade_reason is None
    assert default.downgrade_reason is None
    assert rows[default.id].downgrade_reason is None
    assert "downgradeReason" not in rows[default.id].to_wire()


async def test_a_read_child_gets_an_isolated_checkout_where_a_tier_can_give_one(
    delegating: MountedRuntime, tmp_path: Path
) -> None:
    """E3, through the spawn path: the same request, with a tier that can answer it.

    Where a tier exists, `access="read"` gets `worktree-ephemeral`. The child
    writes freely and **merges nothing**, so what it was granted *of the
    project* is `read`. That is how the seam itself reads `repo_writable`,
    applied one level up, and it is the reason that `granted` is not copied from
    the request.

    The other half is that the branch starts from the root of the *parent* and
    not from the root of the process. That puts a fan-out on sibling branches
    and not in one shared checkout (E2).
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

    With no seam at all, nothing can enforce a writable repo, so nothing promises
    one. The reason is recorded as a *code* and not as a sentence, because a
    durable log must stay parseable and prose in an event goes stale silently.
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
    """The notices that were delivered to the inbox of the parent but that no
    step has claimed yet.

    The seam delivers a notice with `steer` (`_deliver_notice`). The notice
    arrives as a splice and becomes a `user/message` at the next step of the
    parent. That is the meaning of "replies arrive on later turns". To read the
    splice is to read the moment of delivery.
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
    await settle_background(ctx)

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
    await settle_background(ctx)

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

    The test writes the reply into the log of the child before the child runs.
    That is the only way to fix the order, because a child on the fake adapter
    settles inside the await of a real send.

    Sabotage: decide the notice from anything but the log of the child itself,
    and the notice is sent.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "say something")
    reply_to_parent(not_none(ctx.require(SESSIONS).get(run.session_id)))
    await settle_background(ctx)

    assert _notices(session) == [], "a child that replied was announced as silent"
    # The status record is still written. Only the redundant notice is suppressed.
    assert _statuses(ctx, run)[-1] == "done"


async def test_the_child_status_reaches_its_own_log(delegating: MountedRuntime) -> None:
    """Each start and the ending are records of the child, in its own log. The
    log of the parent learns of the child only from a notice in the inbox of the
    parent."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await settle_background(ctx)

    statuses = _statuses(ctx, run)
    assert statuses[0] == "running"
    assert statuses[-1] in {"done", "error"}
    assert _about_children(session) == []


async def test_a_childs_outcome_is_on_disk_before_its_parent_is_told(
    delegating: MountedRuntime,
) -> None:
    """F1. The parent is told that a child finished only after the log of the
    child on disk holds the ending, and the answer before the ending. The parent
    is told in two ways: the notice in its inbox, and the result that its
    waiters receive.

    Nothing used to flush a child after its last model request. The last barrier
    of a child comes *before* that request, and the flush of a parent walks
    ancestors and never children. Every child log on disk therefore ended before
    the answer, while the parent was told "done" with a preview of that answer.
    On the next open, repair closed the turn of the child as interrupted and
    readmitted the child to do the work again. The rule now concerns one log:
    `record_ended` writes the ending and flushes the child before the drive
    tells anyone.

    The test asks the store at the moment that the notice arrives in the log of
    the parent. It asks through the Protocol, so the answer is what the store
    would give a resume, not what is in memory.

    Sabotage: drop the flush from `ph.seams.subagents.record_ended`, and the
    stored log of the child lacks its ending when the parent is told.
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
    await settle_background(ctx)

    assert len(seen) == 1, "the parent was never told the child finished"
    stored, held = seen[0]
    ending = next(
        i for i, event in enumerate(held) if event.type == STATUS and event.data["status"] == "done"
    )
    # Everything up to and including the ending, which includes the answer. What
    # follows the ending (the withdrawn workspace mark) concerns only the child
    # and is not part of what the parent is told.
    assert [event.seq for event in stored[: ending + 1]] == [
        event.seq for event in held[: ending + 1]
    ], "the parent was told before the child's disk held its answer and its ending"
    assert _stored_state(ctx, children[0]).status == "done"


async def test_a_waiter_can_still_block_on_completion(delegating: MountedRuntime) -> None:
    """The generic `task` contract: the answer can be reached, but admission
    never returns it."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    assert run.result is not None
    outcome = await run.result()
    assert outcome.status == "done"
    # The answer can be reached, but admission did not return it.
    assert outcome.answer == "ok"


async def test_a_childs_spend_is_counted_from_its_own_answers(
    delegating: MountedRuntime,
) -> None:
    """What a child spent is read where the child spent it: the `usage` on its
    own `assistant/message` events. The budget of a goal (`delegated_tokens`)
    and the panel add those up. Nothing is copied into the log of the parent for
    them to read. Such a copy could lag behind the log of the child after a
    crash, and every resume had to bring it up to date.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await settle_background(ctx)

    answers = [event for event in _log_of(ctx, run) if event.type == "assistant/message"]
    spent = sum(not_none(reported_usage(event)).total for event in answers)
    assert spent > 0, "the child's answer carries no usage to count"
    assert _state(ctx, run).tokens == spent
    assert ctx.require(SUBAGENTS).delegated_tokens(session.id) == spent
    assert _about_children(session) == []


# ------------------------------------------------------- children and deletion --


async def test_the_children_are_read_from_their_own_logs(delegating: MountedRuntime) -> None:
    """P3-13 by construction: there is no side table, so restart and compaction
    cost nothing. The children of a parent are the logs that name it, and each
    log is folded on its own. The cached fold of the service equals a fresh fold
    of the log of the child (I6)."""
    ctx, session, parent = await delegating()
    first = await _spawn(ctx, parent, "one", name="alpha")
    second = await _spawn(ctx, parent, "two", name="beta")
    await settle_background(ctx)

    children = ctx.require(SUBAGENTS).children(session.id)
    assert list(children) == [first.id, second.id], "in admission order"
    assert children[first.id].name == "alpha"
    assert children[second.id].status in {"done", "error"}, "status folded onto the state"
    for run in (first, second):
        assert children[run.id] == child_state(not_none(ctx.require(SESSIONS).get(run.session_id)))


async def test_deleting_a_child_leaves_a_tombstone(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """The transcript stays on disk, so a reader must be able to find the
    revocation. It is in the log of the child, and that log alone then holds the
    whole record of the child.

    The child is held at the model, so it is revoked while it works. The delete
    first reads the stored children of the parent off the event loop, and a
    child that is not held answers in that time.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "doomed")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    subagents = ctx.require(SUBAGENTS)
    assert _teardowns(parent) == [run.id]

    assert await subagents.delete(session, run.id, reason="user") is True
    assert subagents.get(run.id) is None
    # The revocation of the child went with it. It did not stay on the scope of
    # the parent until the parent ends. Otherwise one revocation would stay for
    # each deleted child, for as long as a root lives.
    assert _teardowns(parent) == []
    # Deleting twice is not an error, and does not double-tombstone.
    assert await subagents.delete(session, run.id, reason="again") is False

    own = _log_of(ctx, run)
    tombstones = [event for event in own if event.type == DELETED]
    assert len(tombstones) == 1
    assert tombstones[0].data == {"reason": "user"}

    state = _state(ctx, run)
    assert state.deleted is True
    assert state.deleted_reason == "user"
    # A revoked child has a terminal state and is not merely absent. A panel
    # that knew only `deleted` could not say whether the child had ever run.
    assert state.status == "canceled"
    # The status is written in the same batch as the tombstone (S14). When they
    # were apart, a flush between them left a child canceled and not deleted.
    canceled = next(
        event for event in own if event.type == STATUS and event.data.get("status") == "canceled"
    )
    assert canceled.batch is not None and canceled.batch == tombstones[0].batch
    # On disk before anything was let go.
    assert _stored_state(ctx, run).deleted is True
    # The log of the child still exists: a tombstone is not a deletion.
    assert ctx.require(SESSIONS).get(run.session_id) is not None
    assert _about_children(session) == []


async def test_a_child_deleted_as_it_is_admitted_stays_revoked(
    delegating: MountedRuntime,
) -> None:
    """The child is deleted before its drive took a step, and nothing holds it.

    The release writes the tombstone and waits for the disk. The drive comes
    through its gate during that wait, before the release has marked the child
    finished. The drive used to write `running`, run an agent whose task the
    cancel had taken, write `done`, and tell the parent that the child it had
    just deleted "completed without sending a reply".

    Sabotage: drop the `_tombstoned` check from `record_started` and the
    `child_is_live` check from `record_ended`, and the statuses are `canceled`,
    `running`, `done`, with the notice delivered. Make the drive continue past a
    refused start, and whoever awaits the child is told `done`.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "doomed")

    assert await ctx.require(SUBAGENTS).delete(session, run.id, reason="user") is True
    await settle_background(ctx)

    assert _statuses(ctx, run) == ["canceled"]
    state = _state(ctx, run)
    assert (state.deleted, state.status, state.starts) == (True, "canceled", 0)
    assert (await run.result()).status == "error", "revoked, not finished"
    assert _notices(session) == []


async def test_a_failure_after_a_child_finished_does_not_end_it_again(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child is ended one time.

    Something that raised after the `done` of the child was written used to
    reach the failure path of the drive. Here that something is the withdrawal
    of the mark on its tree. The failure path wrote `error` over the `done`,
    with a notice of its own. The parent was told that the child finished and
    then that it failed, and whoever awaited the child received the failure.
    Now `record_ended` refuses the second ending, and the waiter receives what
    the log holds.

    Sabotage: drop the `child_is_live` check from `record_ended`, and the
    statuses end `done`, `error`, with two notices.
    """
    monkeypatch.setattr(
        RlmChildProvider, "_withdraw", raising(RuntimeError("the mark could not be withdrawn"))
    )
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "finish quietly")
    await settle_background(ctx)

    assert _statuses(ctx, run) == ["running", "done"]
    assert (await run.result()).status == "done"
    assert len(_notices(session)) == 1


async def test_a_parents_log_holds_no_record_of_its_child(delegating: MountedRuntime) -> None:
    """The one-writer gate (Phase 11).

    The whole life of a child is written in its own log and not in the log of
    its parent: spawned, answered, woken and answered again, deleted. The log of
    the parent gains no `subagent/*` record. Nothing reaches it except the
    notices that are delivered to the inbox of the parent, and it is never
    flushed because of the child.

    One writer per log is one writer per lock. Every record function that the
    drive of a child called used to take the `Session` of the parent. Each child
    therefore appended to the log of its parent and sometimes flushed it. That
    was a writer that the lease of the parent never saw. It was also a second
    account of the child, which every durability rule then had to keep in step
    with the first.

    Sabotage: make a record function write the log of the parent as well as the
    log of the child, as the roster that this design replaced did, and the log
    of the parent holds a `subagent/*` record.
    """
    ctx, session, parent = await delegating()
    before = session.seq
    flushed: list[str] = []

    def note(target: Session) -> None:
        flushed.append(target.id)

    ctx.on("session/flush", note)

    run = await _spawn(ctx, parent, "answer twice")
    await settle_background(ctx)
    assert await ctx.require(SUBAGENTS).rehydrate(run.id)
    not_none(ctx.require(AGENTS).get(run.session_id)).steer(user_text("and once more"))
    await settle_background(ctx)
    assert await ctx.require(SUBAGENTS).delete(session, run.id, reason="user")

    # The log of the child holds the whole record…
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
    # …and the log of the parent holds none of it.
    appended = {event.type for event in session.events if event.seq >= before}
    assert appended <= {"agent/inbox/spliced"}, f"the child wrote its parent's log: {appended}"
    assert len(_notices(session)) == 2, "each answer was announced to the parent's inbox"
    assert session.id not in flushed, "the child flushed its parent's log"
    await ctx.require(SESSIONS).flush(session)
    assert [kind for kind in stored_types(ctx, session.id) if kind.startswith("subagent/")] == []


async def test_a_settled_child_releases_its_agent_scope(delegating: MountedRuntime) -> None:
    """The scope of a child owns its kernel subprocess, so a scope that is kept
    leaks one CPython subprocess for each delegation. The terminal result
    survives the release."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "finish and go")
    await settle_background(ctx)

    assert ctx.require(AGENTS).get(run.session_id) is None, "the child agent was never disposed"
    # And a caller that awaits after the release still gets the outcome.
    assert run.result is not None
    assert (await run.result()).status == "done"


async def test_disposing_the_parent_unwinds_its_children(delegating: MountedRuntime) -> None:
    """I2: a child is an artifact of the scope of its parent. The same unwinding
    therefore releases it, and nobody has to remember to release it."""
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "outlive me")

    await ctx.require(AGENTS).dispose(parent.id)
    assert ctx.require(SUBAGENTS).get(run.id) is None
    tombstones = [event for event in _log_of(ctx, run) if event.type == DELETED]
    assert [event.data["reason"] for event in tombstones] == [PARENT_TEARDOWN]
    assert _about_children(session) == []


async def test_every_record_of_a_child_is_required_reading(delegating: MountedRuntime) -> None:
    """No build may skip a record of a child. A build could never skip an
    admission, because that shows the parent the wrong family. A build may no
    longer skip a status either. The retry ladder counts starts from the status
    records, so a reader that skipped one would miscount the ladder or bring
    back a child that had ended."""
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent)
    await settle_background(ctx)
    await ctx.require(SUBAGENTS).delete(not_none(parent.session), run.id, reason="user")

    by_type = {
        event.type: event for event in _log_of(ctx, run) if event.type.startswith("subagent/")
    }
    assert set(by_type) == {ADMITTED, STATUS, DELETED}
    assert [event.ignorable for event in by_type.values()] == [False] * 3


# ------------------------------------------------------------- seam vocabulary --


def test_the_family_reach_rule_is_the_nuclear_family() -> None:
    """The rule of C7. It is in the seam so that the guard (P3-12) and the roster
    cannot disagree."""

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
    # Roots are siblings of each other. That lets two top-level agents in one
    # deployment talk to each other.
    assert reach((None, "root-a"), (None, "root-b")) is True


def test_a_default_name_describes_the_task_and_stays_unique() -> None:
    name = default_child_name("Review the authentication middleware for races", "abcdef12")
    assert name.startswith("subagent-review-the-authentication-")
    assert name.endswith("-abcdef12")
    assert default_child_name("x", "abcdef12", taken=[name]) != name
    # A prompt that cannot be made into a slug still gives an addressable name.
    assert default_child_name("!!!", "abcdef12") == "subagent-task-abcdef12"


# ------------------------------------------------------------------- the grant --


async def test_a_real_child_is_narrowed_by_its_spawn(delegating: MountedRuntime) -> None:
    """P4-13b through the provider that actually ships it.

    The seam refuses what the parent does not hold, and it applies the narrowing
    to the scope that the provider returns. This test therefore covers the half
    that a unit test of `Grant.apply` cannot reach: a child agent whose scope
    `agents.create` nested inside the scope of its parent really does end with
    the subset.
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
    # The parent kept everything, so this is a narrowing.
    assert "write" in ctx.require(TOOLS).view(parent.ctx).visible


async def test_a_real_spawn_cannot_widen(delegating: MountedRuntime) -> None:
    ctx, _session, parent = await delegating()

    with pytest.raises(SubagentSpawnError) as refused:
        await _spawn(ctx, parent, skills=("nonesuch",))

    assert "Grant it to the parent first" in str(refused.value)


async def test_a_rehydrated_child_is_narrowed_again(delegating: MountedRuntime) -> None:
    """The defect that this row nearly shipped, and the reason that the seam owns
    the ceiling.

    Settlement disposes the scope of the child. The filters that bound the child
    are effects of that scope, so they are disposed too. `rehydrate` then builds
    a **fresh** scope from the retained options. It used to narrow nothing, so a
    child that outlived its own restriction came back with the whole deployment.
    That is exactly what the ruling forbids, and a public method on the shipped
    provider could reach it.
    """
    ctx, _session, parent = await delegating()
    for name in ("review", "deploy"):
        ctx.require(SKILLS).register(skill(name))
    run = await _spawn(ctx, parent, skills=("review",))
    assert [one.name for one in ctx.require(SKILLS).list(run.scope)] == ["review"]

    await settle_background(ctx)
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
    """A child under a tier that gives out worktrees, with `read` access by
    default.

    `access` is a parameter so that the setup is not copied. The write case
    differs from the read case in exactly that word. The two copies had already
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

    The time in which a workspace can be marked ends with the scope of the
    child. The outcomes that most need the evidence are the ones that end it
    first. On the `parent-teardown` path the worktree is released *before*
    `_release` runs, as the docstring of that method states. No place therefore
    exists to put "retain when it fails". The mark must exist from the moment
    that the tree exists, and a clean finish withdraws it.
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
    """The other half, which keeps the promise of the kind.

    Without the withdrawal, retain-by-default would invert `worktree-ephemeral`
    for *every* outcome and not only for the outcomes that produce evidence. The
    row forbids that trade, because it costs one checkout for each delegation
    and keeps nothing that anyone will read.
    """
    ctx, _session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "finish and go")
    await settle_background(ctx)

    assert _marks(ctx, run) == ["the child has not settled cleanly", ""]
    # The disposal policy reads the *closing* half, and the reason is gone from
    # it. The test asserts the reason and not a particular `outcome`, because
    # this tier is a stub with no `release` and therefore keeps every tree.
    # Under the real `worktree` provider this checkout is discarded and no
    # survivor exists. That is exactly the promise that the withdrawal restores.
    (record,) = workspace_survivors(
        not_none(ctx.require(SESSIONS).get(run.session_id), "the child session")
    )
    assert record.outcome != "retained", "a successful child's checkout is not evidence"
    assert record.reason == ""


async def test_a_canceled_child_keeps_its_evidence(
    delegating: MountedRuntime, tmp_path: Path
) -> None:
    """The case that the row was written for, through the path that cannot mark.

    `parent-teardown` disposes the scope of the child before the settle handler
    runs. Nothing on that path could therefore have set the retention that this
    test asserts. The retention is present because it was taken at acquire and
    never withdrawn.
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
    """Only the kind that discards is retained, because only that kind can lose
    evidence.

    An ordinary `worktree` already keeps a dirty tree for review, and a
    committed branch survives release in any case. To retain those would add to
    the kept trees and save nothing.
    """
    ctx, _session, parent = await delegating()

    run = await _tiered_child(ctx, parent, tmp_path, "write some code", access="write")

    assert _marks(ctx, run) == []


async def test_a_failed_child_tells_its_parent_where_the_tree_is(
    delegating: MountedRuntime, tmp_path: Path
) -> None:
    """Retention keeps the checkout. This notice makes sure that someone can find
    that evidence.

    The workspace events of a child are in the log of the *child*. A parent that
    is told only "it failed" must diagnose from a transcript, which is the
    sentence that the row opens with. The location is read from the log and not
    from the seam, so the same notice is true on a path where the scope of the
    child is already gone.
    """
    ctx, session, parent = await delegating()
    run = await _tiered_child(ctx, parent, tmp_path, "fail please")
    child = not_none(ctx.require(AGENTS).get(run.session_id), "the child agent")
    root = not_none(ctx.require(WORKSPACE).of(child.id), "the child workspace").root
    _breaks(child)
    await settle_background(ctx)

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
    """The notice names no path when there is none. It does not invent one.

    The cases are a profile with no tier, a `shared` workspace, and a child
    whose tree really was discarded. A notice that named a directory for every
    failure would name directories that do not exist.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "fail with no workspace")
    _breaks(ctx.require(AGENTS).get(run.session_id))
    await settle_background(ctx)

    (told,) = _notices(session)
    assert "failed" in told
    assert "workspace is kept" not in told


# ---------------------------------------------------------------- the queue --


async def test_a_full_parent_queues_the_next_child_until_a_slot_frees(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """`maxConcurrent` is a queue. The parent gets every child that it asked for,
    one slot at a time, in admission order. It never gets a refusal."""
    ctx, session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await settled(lambda: gate.arrived == 1, "the first child to reach the model")

    children = ctx.require(SUBAGENTS).children(session.id)
    assert children[first.id].status == "running"
    assert children[second.id].status == "queued", "admitted, not refused — and waiting"
    assert _statuses(ctx, second) == ["queued"], "the wait is in its log"

    gate.release_one()
    await settled(lambda: gate.arrived == 2, "the second child to take the freed slot")
    assert ctx.require(SUBAGENTS).children(session.id)[first.id].status == "done"
    assert _statuses(ctx, second) == ["queued", "running"]

    gate.release_one()
    assert (await second.result()).status == "done"
    assert _statuses(ctx, first) == ["running", "done"], "no wait, no queued record"


async def test_a_child_that_failed_frees_its_slot(
    delegating: MountedRuntime, gate: ModelGate, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failure cannot block the queue: the `error` path of the provider
    releases the slot exactly as `done` does.

    The failure here is that the *run* of the child raises, which is the error
    path of the provider itself. It is not a failed model call. The agent loop
    contains a failed model call as a turn that ended in error, and the provider
    records that as `done`.
    """
    original_run = ReactLoopAgent.run
    failed: list[str] = []

    async def run(self: Any) -> None:  # noqa: ANN401
        # The parent is never run in this test, so the first `run()` is that of
        # the first child. Every later call runs the original.
        if not failed:
            failed.append(self.id)
            raise RuntimeError("the child fell over")
        await original_run(self)

    monkeypatch.setattr(ReactLoopAgent, "run", run)
    ctx, _session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")

    assert (await first.result()).status == "error"
    await settled(lambda: gate.arrived == 1, "the second child to run after the failure")
    gate.release_one()

    assert (await second.result()).status == "done"
    assert _statuses(ctx, first) == ["running", "error"]
    assert _statuses(ctx, second)[-2:] == ["running", "done"]


async def test_deleting_a_queued_child_stops_its_wait_and_takes_no_slot(
    delegating: MountedRuntime, gate: ModelGate
) -> None:
    """A child that is revoked before it ran is canceled where it waits. The slot
    that it never held is not leaked: the next child still gets it."""
    ctx, session, parent = await delegating(maxConcurrent=1)
    first = await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await settled(lambda: gate.arrived == 1, "the first child to reach the model")

    assert await ctx.require(SUBAGENTS).delete(session, second.id, reason="user") is True
    assert _statuses(ctx, second) == ["queued", "canceled"]

    third = await _spawn(ctx, parent, "third")
    gate.release_one()
    await settled(lambda: gate.arrived == 2, "the third child to take the slot the first freed")
    gate.release_one()
    assert (await first.result()).status == "done"
    assert (await third.result()).status == "done"
    assert _statuses(ctx, third) == ["queued", "running", "done"]


# ------------------------------------------------------------ across a restart --


async def _persisted(ctx: Context, session: Session) -> None:
    """Put the log of the parent on disk while the harness does nothing else, so
    that a restart has a root to resume.

    Only the log of the parent is flushed. The log of each child reaches disk
    through its own barriers: the flush of the admission before its gate opened,
    and the checkpoint before each model request. To flush a child here would
    put on disk what a crash might not have put there.

    This is a flush and nothing else. The first harness waits at the model for
    the whole of these tests. To drain here would wait for the very child that
    the test means to catch in the middle of its turn. `_restart` reads a
    snapshot of what is on disk, so nothing that the first harness does
    afterwards reaches the second.
    """
    await ctx.require(SESSIONS).flush(session)


RETRIES = 3
"""The ladder bound of this suite.

The bound is stated here and not imported. `resume_children` takes the limit as
an argument because the limit is the policy of the *host*. A test that read the
number of the daemon from `phern` would assert against a value that it does not
control. It would also couple the tests of `ph-rlm` to a package that they do
not depend on.
"""


async def _restart(
    mount: MountProfile,
    session_id: str,
    *,
    skills: tuple[str, ...] = (),
    concurrent: int = 1,
    rows: tuple[dict[str, Any], ...] = (),
    prepare: Callable[[Context], None] | None = None,
) -> Any:  # noqa: ANN401
    """A second harness that resumes one root from its log, as a restarted
    process would.

    This is what a daemon restart *is* from the side of the seam: a fresh mount,
    nothing in memory, and a session that must be read from disk.
    `resume_children` is the call that `Supervisor` makes at the same point,
    against the same agent.

    **The harness runs over a snapshot of the logs, not the live files**
    (`logs_after_a_crash`). The snapshot is the sessions directory as
    `_persisted` left it. The first harness is still alive, waits at the model,
    and holds the leases of its children.

    `skills` is what the *deployment* still provides. It is a parameter because
    a readmit derives the ceiling of the child again from what the parent holds
    now, not from what it held then. A child whose skill this deployment no
    longer mounts is therefore refused. It is not quietly readmitted without
    that skill.

    `rows` are further rows of the restarted deployment, for example a `models`
    row. `prepare` sets up the deployment before anything is resumed, as the
    profile of a daemon would, for example with a route of its own.
    """
    ctx = await mount(
        dict(PROVIDER_ROW, config={"maxConcurrent": concurrent}),
        {"id": "session-persistence", "config": {"root": str(logs_after_a_crash())}},
        *rows,
    )
    for name in skills:
        ctx.require(SKILLS).register(skill(name))
    if prepare is not None:
        prepare(ctx)
    session = await resume_stored(ctx, session_id)
    parent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await ctx.require(SUBAGENTS).resume_children(parent, retry_limit=RETRIES)
    return ctx, session, parent


async def test_a_live_childs_log_refuses_a_second_opener(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """L2. The log of a child is a session of its own, and a caller can open it
    by its id: `phern -p --session <child>`, or a `session/new` to a daemon that
    names it. While this harness drives the child, a second opener is therefore
    refused, as it would be for a root (I-5). It does not become a second writer
    on one log.

    Sabotage: open the child with `resume_session` or `sessions.create` directly
    in `_child_session`, and the second open is granted.
    """
    ctx, _session, parent = await delegating()
    child = await _spawn(ctx, parent, "first")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    elsewhere = await mount()

    with pytest.raises(SessionBusy):
        await open_session(elsewhere, child.session_id)


async def test_a_queued_child_is_re_driven_after_a_restart(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """The work was described in the log of the child and was running nowhere
    (P5-04).

    A child that never reached its first turn has claimed nothing and spent
    nothing, so the next harness runs it for the first time. It runs under its
    original id, so the parent does not gain a second child that it never asked
    for.

    The proof of "re-driven" is that the readmitted child starts to run under
    the second harness: its log gains a `running` record. Its log keeps the one
    admission that it was rebuilt from. Sabotage: make `_readmit_one` pass
    `record=True`, and the readmission writes a second admission.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    await _spawn(ctx, parent, "first")
    second = await _spawn(ctx, parent, "second")
    await settled(lambda: gate.arrived == 1, "the first child to reach the model")
    assert _statuses(ctx, second) == ["queued"]
    await _persisted(ctx, session)

    # There is room for both children, so a race does not decide which child
    # this test asserts about. The interrupted sibling is on the ladder and
    # comes back too.
    revived_ctx, revived, _parent = await _restart(mount, session.id, concurrent=2)

    assert second.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}, (
        "the queued child came back under its own id"
    )
    await settled(
        lambda: _statuses(revived_ctx, second)[-1] == "running",
        "the readmitted child to reach the model",
    )
    assert len(revived_ctx.require(SUBAGENTS).children(revived.id)) == 2, (
        "no child was invented or lost"
    )
    assert _state(revived_ctx, second).resumes == 0, (
        "a child that never ran is a first attempt, not a retry"
    )
    admissions = [event for event in _log_of(revived_ctx, second) if event.type == ADMITTED]
    assert len(admissions) == 1, "a readmission is not a second admission"


async def test_a_child_caught_mid_turn_climbs_the_ladder_with_its_task_re_presented(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """The ladder, and what makes a restart a real attempt and not a false
    report of one.

    The start of a turn *claims* the task from the inbox, and the claim is a
    logged splice. A resumed child that was only driven again would therefore
    find an empty inbox, end at step zero, and report `completed` for work that
    it never did. That is what P5-04 found at the root. The fix is to present
    the task again. The note that the turn was cut short stops the transcript
    from reading as one instruction given twice.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    interrupted = await _spawn(ctx, parent, "only")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    assert _statuses(ctx, interrupted) == ["running"]
    await _persisted(ctx, session)

    revived_ctx, _revived, _parent = await _restart(mount, session.id)

    assert interrupted.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}
    assert child_is_live(_state(revived_ctx, interrupted)), "a child owed a turn is live"
    # The count follows the `running` record of the restart itself. A detached
    # drive job writes that record a moment later, so this waits for the fact
    # and does not read the state of the child before the record exists.
    await settled(
        lambda: restarts_since_progress(_state(revived_ctx, interrupted)) == 1,
        "the restart to be counted",
    )
    assert _state(revived_ctx, interrupted).starts == 2, "one first run, one restart"
    await settled(gate.twice, "the resumed child to reach the model again")

    child = revived_ctx.require(SESSIONS).get(interrupted.session_id)
    assert child is not None
    tasks = [
        text_of(not_none(derive_event_message(event)).content)
        for event in child.events
        if event.type == "user/message" and TASK_PREFIX in repr(event.data)
    ]
    assert len(tasks) == 2, "the task was not presented again, so the retry answers nothing"
    assert "the harness stopped while you were working" in tasks[-1]
    assert "this is attempt 2" in tasks[-1]


async def test_a_child_that_replied_before_a_restart_is_not_announced_as_silent(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """Whether a child replied used to be a flag on its in-memory `_Child`, and a
    readmitted child gets a new `_Child`. A child that answered its parent, was
    caught by a restart and then finished was therefore announced as finished
    without a reply. The reply is in the log of the child itself, which comes
    back with the child.

    Sabotage: decide the notice from anything that the readmitted `_Child` holds
    in memory, and the child is announced as silent.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    run = await _spawn(ctx, parent, "only")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    child = not_none(ctx.require(SESSIONS).get(run.session_id))
    reply_to_parent(child)
    await ctx.require(SESSIONS).flush(child)
    await _persisted(ctx, session)

    revived_ctx, revived, _parent = await _restart(mount, session.id)
    await settled(gate.twice, "the readmitted child to reach the model")
    gate.release_all()
    await settle_background(revived_ctx)

    assert _statuses(revived_ctx, run)[-1] == "done"
    assert _notices(revived) == [], "a child that replied before the restart was announced"


async def test_a_mount_that_unwinds_suspends_its_children_rather_than_revoking_them(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """S1: a clean stop used to tombstone every child, so a restart brought none
    back.

    A daemon that stops, or a root that is remounted on a new profile, unwinds
    the scope of the parent. The release of each child used to write `canceled`
    and `subagent/deleted` during that unwind. `resume_children` skips a deleted
    child, so an ordinary restart abandoned every delegation that was in
    progress. A mount that goes away now suspends its children. Each child gets
    a `queued` record with the reason in its own log, and no tombstone, and the
    next mount readmits it.

    Sabotage: drop the `suspend` disposer from `apply`, and the release raises
    during the unwind and writes nothing, so the restart reads the child as
    crashed.
    """
    ctx, session, parent = await delegating()
    working = await _spawn(ctx, parent, "keep going")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")

    await ctx.root.dispose()
    revived_ctx, _revived, _parent = await _restart(mount, session.id)

    assert not _state(revived_ctx, working).deleted, "the stop revoked the child"
    assert SUSPENDED_DETAIL in _statuses(revived_ctx, working, "detail"), (
        "the child's log does not say why it stopped"
    )
    assert working.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}
    await settled(gate.twice, "the resumed child to reach the model again")


async def test_a_suspended_child_keeps_what_was_queued_for_it(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A child that comes back finds its inbox as it was, not an empty one.

    `suspend` stops the agent of the child and keeps its inbox, where a
    revocation clears the inbox. A message that was steered to a working child
    and not yet taken is therefore still pending when the next mount readmits
    the child.

    Sabotage: cancel the agent in `suspend` without `keep_inbox`, and the
    message is canceled in the log of the child.
    """
    ctx, _session, parent = await delegating()
    working = await _spawn(ctx, parent, "keep going")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    child = not_none(ctx.require(AGENTS).get(working.session_id))
    child.steer(user_text("also this"))

    await ctx.root.dispose()
    revived_ctx, _revived, _parent = await _restart(mount, parent.id)
    await settled(gate.twice, "the resumed child to reach the model again")

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

    The inbox releases a batch only in the step that records it (S3). A child
    that was queued behind a sibling when its mount unwound therefore comes back
    with the task still pending. To present the task again on readmit gave the
    child the same work twice.

    Sabotage: present the task in `_admit` whatever the inbox holds, and the log
    of the child inserts it twice.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    await _spawn(ctx, parent, "first")
    waiting = await _spawn(ctx, parent, "second")
    await settled(lambda: gate.arrived == 1, "the first child to reach the model")

    await ctx.root.dispose()
    revived_ctx, _revived, _parent = await _restart(mount, session.id, concurrent=2)
    await settled(lambda: gate.arrived >= 3, "both children to reach the model again")

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

    The teardown of its parent used to release it with `canceled`, written over
    the `done` that its drive had recorded. Every finished child therefore read
    as a revoked child after its parent was gone.

    Sabotage: write `canceled` in `_tombstone` (`ph.seams.subagents`) whatever
    the status of the child, and the log of the child says canceled.
    """
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "finish first")
    await settle_background(ctx)

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
    """S10: the ladder counts `running{cause: resumed}`, so that record is on
    disk in the log of the child before the attempt that it counts.

    A restart that reached only memory before the child took the daemon down
    again was never counted. A crash loop never advanced the count on disk, and
    the ladder never gave up. The record used to be in the log of the parent,
    which nothing flushed between the readmission of a child and the work of
    that child.

    The test asks as the attempt starts, which is the call of the drive into the
    run of the child, and not at the model. The first model request of the child
    flushes its log anyway, so at that point the question no longer matters.

    Sabotage: drop the flush from `ph.seams.subagents.record_started`, and the
    stored child has no restart when its attempt begins.
    """
    ctx, session, parent = await delegating()
    interrupted = await _spawn(ctx, parent, "only")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)
    revived: list[Context] = []
    seen: list[int] = []
    original = ReactLoopAgent.run

    async def watched(self: ReactLoopAgent) -> None:
        # The child of the first harness is already inside its own `run`, and
        # the revived parent is never run. The one call that this sees is
        # therefore the call of the resumed child.
        if self.id == interrupted.session_id:
            seen.append(restarts_since_progress(_stored_state(revived[0], interrupted)))
        await original(self)

    monkeypatch.setattr(ReactLoopAgent, "run", watched)
    await _restart(mount, session.id, prepare=revived.append)
    await settled(gate.twice, "the resumed child to reach the model again")

    assert seen == [1], "the attempt began before its restart was on the child's disk"


async def _resumed(ctx: Context, run: SubagentRun, times: int) -> None:
    """Record `times` restarts in the log of the child, in the way that a restart
    really records one, and put them on disk as `record_started` does.

    These are the real facts and not a seeded count. The ladder folds `running`
    records that carry `cause: resumed`, so a test that wrote an `attempts`
    number would assert against a field that production no longer has.
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
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    await _resumed(ctx, child, restarts)
    return child


async def test_the_ladder_gives_up_and_says_so(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """Three restarts with nothing achieved between them is not bad luck.

    To drive the child again forever would spend the budget of a parent on a
    transcript that nobody reads. The ladder of the root itself has a bound to
    avoid the same failure.
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
    # Ended in its own log, on disk, which is the one place where the next start
    # will look.
    assert _stored_state(revived_ctx, spent).status == "error"
    assert _about_children(revived) == []


async def test_progress_since_the_last_restart_clears_the_ladder(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A child that was stopped, worked for an hour, and was stopped again met
    two separate incidents.

    Without a reset, the ladder counts every interruption in the life of the
    child and not only consecutive ones, and it fails work that was going well.
    The setup is that of the test above plus one fact: the child gave a model
    answer, in its own log. A turn that did nothing cannot produce that.
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
    # **The restart is still recorded as a restart**, and the count starts again
    # from the answer. If the readmit took its count from the ladder, it would
    # look like a first run and write no `resumed`, and the ladder would never
    # count it again.
    await settled(
        lambda: restarts_since_progress(_state(revived_ctx, moved)) == 1,
        "the restart to be counted from a cleared ladder",
    )


CHILD_ANSWER_USAGE = {"inputTokens": 120, "outputTokens": 30}


async def _answered_on_its_own_disk(ctx: Context, run: SubagentRun) -> SessionEvent:
    """The child answers, and its own log reaches disk.

    This is what the checkpoint of the child does before its next request. In
    that time a parent that waits for the child makes no request and writes
    nothing. The `usage` of the answer is the whole record of what the child
    spent: nothing is copied anywhere else.
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
    """L5, fixed by construction. A child answers, only the log of the child is
    written, and the mount restarts. The ladder forgives the restarts before the
    answer, and the budget counts what the answer spent, one time, with no
    catch-up step.

    Two decisions read the answers of a child. The ladder takes an answer as
    progress. A child whose answer a restart could not see therefore read as
    stuck, and after `RETRIES` restarts it was failed as exhausted while it was
    making progress. The budget of a goal counts what the answer spent.

    Both decisions used to read a copy of each answer that was mirrored into the
    log of the parent. A crash could leave that copy behind the log of the
    child. A resume therefore had to copy the missing answers back, and it had
    to avoid a second copy of an answer when the log of the parent had reached
    disk after it. Both decisions now read the log of the child itself, which is
    where the answer is, whenever the log of the parent was last written.

    Sabotage: make the fold skip `assistant/message`, as a reader of anything
    but the answers of the child itself would, and the child is failed as
    exhausted with nothing counted.
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
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    child = not_none(ctx.require(AGENTS).get(middle.session_id))
    grandchild = await _spawn(ctx, child, "do the work", **leaf)
    await settled(lambda: gate.arrived == 2, "the grandchild to reach the model")
    await _persisted(ctx, not_none(parent.session))
    await _persisted(ctx, not_none(child.session))
    return middle, grandchild


async def test_a_grandchild_the_restart_interrupted_is_put_back_to_work_too(
    delegating: MountedRuntime,
    gate: ModelGate,
    mount: MountProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """L5b. The children of a readmitted child are swept in the way that the
    children of the root are.

    `RLM_MAX_DEPTH` lets a child delegate, and its children name *it* as their
    parent. The resume used to sweep only the children of the root. A grandchild
    that was caught in the middle of a turn therefore stayed `running`. Nothing
    drove it, readmitted it or failed it, and the model of its parent was told
    that it was still working.

    **The sweep runs before the first step of the child**
    (`SubagentRun.ready`). The grandchild is readmitted ahead of the new attempt
    of the child, so the first request of the child is built from the swept
    children.

    Sabotage: drop `_sweep_readmitted` from `_resume` and the grandchild is
    never readmitted. Open the gate of a readmitted child at admission and its
    new attempt starts ahead of the sweep.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    middle, leaf = await _grandchild(ctx, parent, gate)
    revived: list[Context] = []
    swept_first: list[bool] = []
    original = record_started

    async def watched(
        scope: Context, child: Session, *, cause: StatusCause | None = None
    ) -> SessionEvent | None:
        if child.id == middle.session_id:
            readmitted = {run.id for run in revived[0].require(SUBAGENTS).list()}
            swept_first.append(leaf.id in readmitted)
        return await original(scope, child, cause=cause)

    monkeypatch.setattr("ph_rlm.subagents.record_started", watched)
    revived_ctx, _revived, _parent = await _restart(mount, session.id, prepare=revived.append)

    assert leaf.id in {run.id for run in revived_ctx.require(SUBAGENTS).list()}, (
        "the grandchild came back under its own id"
    )
    await settled(
        lambda: restarts_since_progress(_state(revived_ctx, leaf)) == 1,
        "the grandchild's restart to be counted in its own log",
    )
    await settled(lambda: gate.arrived == 4, "both to reach the model again")
    assert swept_first == [True], "the child's new attempt started before its children were swept"


GRANDCHILD_KEY = "PH_L5B_GRANDCHILD_KEY"
"""The name of the credential that the adapter of the `keyed` route resolves at
its edge, and that the environment lacks."""


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
    """L5b, the credential half. A grandchild that waits for a credential name is
    released when that name arrives.

    The nested sweep holds a grandchild whose route names a credential that the
    restarted deployment lacks. The hold is in the log of the grandchild itself
    (T5), and it is the hold that a session writes when it waits on its own
    route. `readmit_waiting` used to ask only the children of the root when the
    name arrived, so that hold was never released.

    Sabotage: stop `readmit_waiting` from descending into live children, and the
    grandchild stays held although its key is supplied.
    """
    monkeypatch.delenv(GRANDCHILD_KEY, raising=False)
    ctx, session, parent = await delegating(maxConcurrent=1)
    _keyed(ctx)
    keyed = ModelChoice(provider="keyed", model="k1")
    middle, leaf = await _grandchild(ctx, parent, gate, choice=keyed)

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


async def test_two_sweeps_of_one_parent_readmit_a_child_once(
    delegating: MountedRuntime,
    gate: ModelGate,
    mount: MountProfile,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """S1. Sweeps of one parent take turns, so only one of them readmits a child.

    `_resume` used to read `state.run_id in self._runs` and then await before
    `_admit` wrote the entry. Two sweeps that overlapped therefore both
    readmitted the child. The `open_child_log` of the losing sweep failed on the
    live session, and its `except Exception` ended the child of the winning
    sweep. That put a false failure in the log of the child and revoked the
    children of that child permanently. Two `credentials/store` requests in a
    row are two such sweeps.

    The second sweep waits for the first and reads the state that the first
    left, so a credential that arrived in that time is not missed.

    In this test the hold write of the losing sweep meets the log that the
    winning sweep holds (`SessionForkError`), and its attempt to end the child
    fails too. The child therefore survives only because of the order in which
    two writes happened to arrive. The test asserts on the attempt.

    Sabotage: drop the per-parent lock from `readmit_waiting`, and the second
    sweep tries to end the child.
    """
    monkeypatch.delenv(GRANDCHILD_KEY, raising=False)
    ctx, session, parent = await delegating(maxConcurrent=1)
    _keyed(ctx)
    keyed = ModelChoice(provider="keyed", model="k1")
    _middle, leaf = await _grandchild(ctx, parent, gate, choice=keyed)

    revived_ctx, _revived, revived_parent = await _restart(mount, session.id, prepare=_keyed)
    assert _state(revived_ctx, leaf).awaiting == GRANDCHILD_KEY, "held by name"

    revived_ctx.require(CREDENTIALS).provide_value(GRANDCHILD_KEY, "supplied")
    seam = revived_ctx.require(SUBAGENTS)
    revived: list[str] = []
    # The first sweep is held inside the provider, between its check of `_runs`
    # and its admission, until the second sweep has readmitted the child. That
    # is the window that two overlapping sweeps share. The hold has a time
    # limit, because sweeps that run one at a time never make the second call
    # that this waits for.
    original = RlmChildProvider.readmit
    calls: list[str] = []
    second_done = anyio.Event()

    async def readmit(self: RlmChildProvider, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        calls.append(kwargs["run_id"])
        if len(calls) == 1:
            with anyio.move_on_after(0.3):
                await second_done.wait()
            return await original(self, *args, **kwargs)
        try:
            return await original(self, *args, **kwargs)
        finally:
            second_done.set()

    monkeypatch.setattr(RlmChildProvider, "readmit", readmit)

    async def sweep() -> None:
        revived.extend(await seam.readmit_waiting(revived_parent, retry_limit=RETRIES))

    async with anyio.create_task_group() as sweeps:
        sweeps.start_soon(sweep)
        sweeps.start_soon(sweep)

    assert revived.count(leaf.id) == 1, revived
    assert child_is_live(_state(revived_ctx, leaf)), "the second sweep ended the first's child"
    gave_up = [record.message for record in caplog.records if "could not be" in record.message]
    assert gave_up == [], "a sweep gave up on a child another sweep was readmitting"


async def test_a_readmitted_child_does_not_come_back_wider_than_it_was_admitted(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """§6.5 across a power cut, which is why the narrowing is in the record.

    The ceiling is derived again from the admission, so a child that was
    admitted with one skill does not return with every skill that its parent
    has. A child that was rebuilt from the run alone would, and nothing would
    have reported it. The same applies to its writable directories (S7b, item
    3). Sabotage: drop `paths` from `_request_of`, and the child comes back
    bound to the writable directories of its parent.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    ctx.require(SKILLS).register(skill("review"))
    ctx.require(SKILLS).register(skill("audit"))
    await _spawn(ctx, parent, "first")
    narrowed = await _spawn(
        ctx, parent, "second", skills=("review",), tools=("read",), paths=("/srv/cache",)
    )
    await settled(lambda: gate.arrived == 1, "the first child to reach the model")
    await _persisted(ctx, session)

    revived_ctx, _revived, _parent = await _restart(mount, session.id, skills=("review", "audit"))

    # Read from the log of the child as the restart found it, which is the only
    # copy that a restart has.
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


def _sorting(ctx: Context) -> None:
    """The `sort` skill, which asks for `classify` (`MODELS_ROW`)."""
    ctx.require(SKILLS).register(skill("sort", model="classify"))


async def test_a_model_set_up_since_reaches_a_readmitted_child(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A child runs on the model of its parent only until the user sets up the
    model that its skill names. A restart is when that change reaches a child
    that is already at work: its route is worked out again, so it comes back on
    `classify`.

    Sabotage: read the route back from the admission in `_request_of`, and the
    child comes back on `fake-1`, the model of its parent, and still falls back.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    _sorting(ctx)
    child = await _spawn(ctx, parent, "sort these", skills=("sort",))
    assert child.model_fallback == ModelFallback(asked=("classify",), reason="unlisted")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)

    revived_ctx, _revived, _parent = await _restart(
        mount, session.id, rows=(MODELS_ROW,), prepare=_sorting
    )

    back = not_none(revived_ctx.require(SUBAGENTS).get(child.id))
    assert (back.model, back.model_fallback) == ("fake-9", None)
    assert not_none(revived_ctx.require(AGENTS).get(child.session_id)).options.model_key == (
        "classify"
    )


async def test_a_readmitted_child_whose_model_is_gone_runs_on_its_parent_s_and_says_so(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """The opposite case: the model was listed at the spawn and is gone at the
    restart. To end the child would discard its work because of a setting. So
    the child runs on the route of its parent, and the run that its readmission
    returns says so, as the run of a spawn does.

    Sabotage: drop the `model_fallback` stamp in `SubagentService._admit`, and
    the readmitted run says nothing.
    """
    ctx, session, parent = await delegating(dict(MODELS_ROW), maxConcurrent=1)
    _sorting(ctx)
    child = await _spawn(ctx, parent, "sort these", skills=("sort",))
    assert (child.model, child.model_fallback) == ("fake-9", None)
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)

    revived_ctx, _revived, revived_parent = await _restart(mount, session.id, prepare=_sorting)

    back = not_none(revived_ctx.require(SUBAGENTS).get(child.id))
    assert back.model_fallback == ModelFallback(asked=("classify",), reason="unlisted")
    assert back.model == revived_parent.options.model
    assert child_is_live(_state(revived_ctx, child)), "readmitted, not ended"


async def test_a_child_no_provider_can_resume_is_settled_not_left_queued(
    delegating: MountedRuntime, gate: ModelGate, mount: MountProfile
) -> None:
    """A `queued` child that nothing will pick up is worse than a reported
    failure.

    The child reads as live, so the root can never be passivated, and the parent
    waits for a slot that nobody will ever give it. The sweep decides at the
    point where the capability probe answers, and that stops it from writing a
    status that it cannot honor. In this test the next harness mounts no
    subagent provider at all.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    orphan = await _spawn(ctx, parent, "only")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)

    # This runs over a snapshot, as `_restart` does. To end the child, the
    # harness must claim its log, and the first harness still holds that log: it
    # is alive and waits at the model.
    bare = await mount({"id": "session-persistence", "config": {"root": str(logs_after_a_crash())}})
    revived = await resume_stored(bare, session.id)
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
    """K4: a readmit that *raises* used to be logged and then dropped.

    A readmit derives the ceiling again from the admission record, against what
    the parent holds **now**. A deployment that stopped mounting a skill
    therefore refuses the child that was narrowed to that skill. That refusal is
    correct, and the sibling test above is why the narrowing is logged at all.
    What followed was wrong. The exception was logged and the loop moved on. The
    child stayed `queued`, which `child_is_live` reads as waiting for a slot.
    The parent then waited for a child that nothing would ever drive, and the
    root could never be passivated.

    This case is distinct from the no-provider case above it. There the sweep
    *decides* that it cannot recover the child. Here a readmit that the sweep
    fully intended raised partway. The two cases reached the log differently,
    and only one of them reached it at all.
    """
    ctx, session, parent = await delegating(maxConcurrent=1)
    ctx.require(SKILLS).register(skill("review"))
    await _spawn(ctx, parent, "first")
    narrowed = await _spawn(ctx, parent, "second", skills=("review",))
    await settled(lambda: gate.arrived == 1, "the first child to reach the model")
    await _persisted(ctx, session)

    # The deployment no longer mounts `review`, so the parent does not hold it
    # and `check_grant` refuses the child that was admitted with it.
    revived_ctx, _revived, _parent = await _restart(mount, session.id, concurrent=2)

    state = _state(revived_ctx, narrowed)
    assert state.status == "error"
    assert "could not be resumed" in not_none(state.detail)
    assert not child_is_live(state), "a root cannot be passivated while this reads live"


async def test_a_readmission_the_seam_refuses_is_ended_and_not_revoked(
    delegating: MountedRuntime,
    gate: ModelGate,
    mount: MountProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A child that the seam refuses at readmission is ended with the reason. Its
    parent is told one time, and the child is not revoked.

    The refusal used to release the child through the revocation of the
    provider, which tombstoned it `parent-teardown` for a parent that was still
    there. Worse, the notice was then attached to a deleted child. The sweep
    that brings a parent back delivers no notice of a deleted child, so a crash
    before the next write of the parent lost the notice permanently. A child
    that is not deleted is one whose notice that sweep delivers again
    (`test_child_logs`).

    The revocation of the child is dropped from the scope of the parent without
    being run (`Releaser.drop`), so nothing is left there to revoke the child
    later.

    Sabotage: point `run.dispose` back at `_release(run_id, PARENT_TEARDOWN)`,
    and the child is deleted, `parent-teardown`. Skip the drop in `_stop`, which
    `_take` calls, and the scope of the parent still holds the revocation.
    """
    ctx, session, parent = await delegating()
    run = await _spawn(ctx, parent, "refused next time")
    await settled(lambda: gate.arrived == 1, "the child to reach the model")
    await _persisted(ctx, session)

    monkeypatch.setattr(
        SubagentService,
        "_enforce",
        raising(SubagentSpawnError("its ceiling cannot be applied now")),
    )
    revived_ctx, revived, revived_parent = await _restart(mount, session.id)

    state = _state(revived_ctx, run)
    assert (state.status, state.deleted) == ("error", False)
    assert "its ceiling cannot be applied now" in not_none(state.detail)
    assert delivered_ids(revived) == [not_none(state.notice).id]
    assert _teardowns(revived_parent) == []


# ------------------------------------------------- a child asked a second thing --


@pytest.mark.needs_git
async def test_a_re_addressed_child_comes_back_to_its_own_work(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A second question reaches the child that answered the first, with its
    tree.

    Disposal commits the checkout to the branch of the child before it removes
    the checkout, and `_add` attaches an existing branch and does not reset it.
    The same run id therefore resolves to the same branch, and a second acquire
    restores what the child did. Nothing here arranges that. What was missing
    was that `rehydrate` never asked for the workspace.
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
    await settle_background(ctx)
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
    """The containment half. A child that was admitted with `read` comes back
    ephemeral.

    Its access is read from the admission, not from the caller that addresses it
    again. A second question therefore cannot widen what the first was allowed
    to keep.
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
    await settle_background(ctx)

    assert await ctx.require(SUBAGENTS).ensure_addressable(run.session_id) is True

    again = ctx.require(WORKSPACE).of(run.session_id)
    assert again is not None
    assert again.kind == "worktree-ephemeral", "a second question must not widen the first's grant"


async def test_a_rehydration_that_is_stopped_leaves_the_child_wakeable(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S4. A rehydration that fails or is stopped after its agent is made ends
    that agent.

    `child.agent` was set before the runtime and the drive were set up. A
    rehydration that was stopped between those steps left an agent that nothing
    drove. A kernel cancels a binding call when its cell settles, so a cell that
    is stopped in the middle of a send is enough to cause this. Every later wake
    then answered "already running", and every later send went into an inbox
    that nobody read.

    Sabotage: drop the `except BaseException` from `rehydrate`, and the second
    wake is refused.
    """
    ctx, _session, parent = await delegating()
    run = await _spawn(ctx, parent, "answer twice")
    await settle_background(ctx)
    answered = await run.result()

    original = RlmChildProvider._runtime
    calls: list[int] = []

    async def stopped_once(self: RlmChildProvider, *args: Any) -> None:  # noqa: ANN401
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("the cell that asked was stopped")
        await original(self, *args)

    monkeypatch.setattr(RlmChildProvider, "_runtime", stopped_once)
    with pytest.raises(RuntimeError, match="was stopped"):
        await ctx.require(SUBAGENTS).rehydrate(run.id)
    assert ctx.require(AGENTS).get(run.session_id) is None, "an agent nothing drives"
    assert await run.result() == answered, "a waiter lost the answer it had"

    assert await ctx.require(SUBAGENTS).rehydrate(run.id), "the child can no longer be woken"
    await settle_background(ctx)


async def test_a_spawn_stopped_while_its_workspace_is_taken_leaves_no_agent(
    delegating: MountedRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S5. The rlm build ends what it made on every exit path, not only when the
    parent is gone.

    `_admit` used to let go of the session of the child when the parent went
    away during the acquire of the workspace. When the spawn was stopped there
    instead, it let go of nothing: the agent and its open session stayed until
    the scope of the parent went.

    Sabotage: catch only `InactiveScopeError` around `_workspace` again, and the
    agent of the child is still registered.
    """
    ctx, _session, parent = await delegating()
    taking = anyio.Event()

    async def stuck(self: RlmChildProvider, *args: Any) -> Any:  # noqa: ANN401
        taking.set()
        await anyio.sleep_forever()

    monkeypatch.setattr(RlmChildProvider, "_workspace", stuck)
    agents = {agent.id for agent in ctx.require(AGENTS).list()}
    sessions = {one.id for one in ctx.require(SESSIONS).list()}
    async with anyio.create_task_group() as spawning:
        spawning.start_soon(partial(_spawn, ctx, parent, "do the work"))
        await taking.wait()
        spawning.cancel_scope.cancel()

    assert {agent.id for agent in ctx.require(AGENTS).list()} == agents
    assert {one.id for one in ctx.require(SESSIONS).list()} == sessions
