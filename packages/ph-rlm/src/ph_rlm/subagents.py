"""`rlm-subagent-provider`: `rlm()` as a `ctx.subagents` provider (P3-11).

This module is a port of `AgentSession._startRlmChildRun` from prime-agent. The
semantics come from prime-agent, and the mechanism is the seams of pH. The design
has five properties. The obvious implementation gets each of them wrong.

**The handle returns before the child answers.** `ph.seams.subagents` states this
contract one time, and this file keeps it. `start()` creates the session and the
agent, starts the job, and returns. The seam writes the admission of the child
into the log of the child. The reply of the child arrives on a later turn as an
ordinary inbox message.

**Every record of a child is in the log of that child** (Phase 11). This provider
writes nothing to the log of a parent. The starts, the waits, the ending and the
tombstone of a child go into the log of the child through the record functions of
the seam. A parent reads them from there: the roster tool, the prompt and the
resume sweep all do. The agent is *created* before the admission, so a `create`
failure leaves no admitted child behind. The agent is not started until the seam
opens its gate, and the seam opens the gate after the admission is on disk.

**A child is an artifact of the scope of its parent.** The provider acquires the
child through `parent.ctx.effect()`. A disposed parent therefore unwinds its
children (I2), and `delete()` only releases the child early. Without that, a
settled child kept its agent scope alive for the lifetime of the host. That scope
owns the kernel subprocess of the child, so every delegation leaked a CPython
process.

**Usage stays where it was spent.** The answers of a child carry their usage in
the log of the child. The budget of a goal, the retry ladder and the TUI panel
read it there (`ChildState.tokens`, `restarts_since_progress`). This provider
used to mirror a copy into the log of the parent. After a crash that copy could
lag behind the log of the child, and every resume had to bring it up to date.

**This provider takes the workspace of the child. The lifecycle row does not**
(P4-08). The base of the workspace is the root of the parent, and the access is
the decision of the parent. The lifecycle row knows neither. `granted` is what
the child got *of the project*. A `worktree-ephemeral` child writes freely and
merges nothing, so it was granted `read`. The *reason* for any narrowing travels
as a code and not as prose, so that the log stays parseable.

@module ph_rlm.subagents
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field
from functools import partial

import anyio
from pydantic import Field

from ph.agent.types import AgentCancelCause, AgentDriver, AgentHandle, AgentOptions
from ph.characters import head
from ph.cordis import Context, InactiveScopeError, Releaser, plugin, releasing
from ph.json import JsonObject, JsonValue, dumps
from ph.keys import AGENTS, FS, JOBS, SESSIONS, SUBAGENTS, WORKSPACE
from ph.llm.types import PluginSource, create_user_message
from ph.persistence import cut_short_note
from ph.seams.subagents import (
    PARENT_TEARDOWN,
    SUSPENDED_DETAIL,
    Access,
    ChildNotice,
    DowngradeReason,
    SettledStatus,
    StatusCause,
    SubagentAwaiter,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentSpawnError,
    child_options,
    last_answer,
    offer_shipped_skills,
    open_child_log,
    record_deleted,
    record_ended,
    record_started,
    record_waiting,
    said_text,
    settle_value,
)
from ph.seams.workspace import discards_writes, project_access, workspace_survivors
from ph.session import (
    Session,
    child_session_id,
)
from ph.wire import WireModel

from .keys import RLM_CHILDREN
from .messaging import replied_to_parent

__all__ = [
    "PROVIDER_NAME",
    "RLM_MAX_DEPTH",
    "TASK_PREFIX",
    "Config",
    "RlmChildProvider",
    "apply",
    "delegation_depth",
]

log = logging.getLogger("ph_rlm.subagents")

PROVIDER_NAME = "rlm-child"
RLM_MAX_DEPTH = 2
"""The `RLM_MAX_DEPTH` of prime-agent. Depth 0 delegates, depth 1 delegates, and
depth 2 does the work. Three levels is already a lot of indirection between the
question of a human and the tokens that answer it."""
TASK_PREFIX = "[task from parent]"
"""Ported verbatim: the child's prompt recognizes this label."""


class Config(WireModel):
    """Row config for `rlm-subagent-provider`."""

    max_depth: int = RLM_MAX_DEPTH
    max_concurrent: int | None = Field(default=None, ge=1)
    """How many children of one parent run at the same time. `None` means no cap.

    **The cap is a queue, not a refusal.** A parent that delegates nine tasks
    asked for nine. To refuse the ninth because eight are running would answer a
    question about resources with a refusal about intent. Only the *drive* of a
    child waits. Admission is not affected.

    **The cap is per parent, so it limits *fairness* and not host load.** Ten
    roots with four children each make forty children. The cap exists so that the
    fan-out of one agent cannot starve the fan-out of another root. A deployment
    that wants to limit the *host* sets the `concurrency` of `jobs` instead. Both
    limits apply.

    The queue belongs to `ctx.jobs` (`slot=`). That seam states the mechanics:
    the wait, the release on every ending, and the lifetime of the queue. They
    are not repeated here, because that would be a second account of code that
    belongs to another module."""
    answer_preview_chars: int = 240
    """How many characters of the answer of the child the status record carries,
    for the trajectory viewer and the P3-19 panel. There is deliberately no
    `default_access` setting here. The request default is on `SubagentRequest`. A second copy would
    be a documented setting that silently did nothing while the tier is
    unmounted."""


def delegation_depth(session: Session) -> int:
    """How many delegations deep this session is, read from its own header.

    The function reads the typed field and not `to_wire()["delegationDepth"]`. A
    wire alias that is rebuilt by hand returns 0 after a rename and does not
    fail, and 0 *opens* the depth gate. The depth is read from the header and not
    counted at spawn time, because a resumed child has no live parent to ask and
    the gate must still hold.
    """
    return session.header.delegation_depth or 0


@dataclass(slots=True)
class _Child:
    """The live half of one delegation: the part that a fold cannot reconstruct.

    Settlement releases what a *finished* child does not need: its agent scope
    (and with it the kernel subprocess), its job entry, and the session handle.
    It keeps what a late `result()` or a rehydration needs: the run, the outcome,
    and the options to rebuild the agent with.
    """

    run: SubagentRun
    finished: anyio.Event
    options: AgentOptions
    """The options that the agent of the child was built with (`child_options`).
    For a listed route these are the options of that route. For a child on the
    route of its parent they are the options of the parent at spawn. They are kept
    for a rehydration, which rebuilds the agent after the parent that lent them
    may have gone. The admission records the route, the settings of a listed
    route, and the reasoning effort that the spawn *asked* for
    (`Admission.reasoning_effort`). For a child on the route of its parent, the
    settings that it inherited survive only here."""
    agent: AgentDriver | None = None
    session: Session | None = None
    job_id: str | None = None
    """The drive job. The scope of the *parent* owns it, not the scope of the
    child. The last act of the drive job disposes the scope of the child, and a
    job that abandoned itself would report `canceled` for work that finished."""
    result: SubagentResult | None = None
    teardown: Releaser | None = None
    """The revocation that `_admit` registers on the scope of the parent. A parent
    that goes away runs it. It is registered exactly while this provider holds
    the child. When the child is taken off the provider (`_take`, `suspend`), the
    revocation is dropped without being run, because it would find nothing here."""


@dataclass(slots=True)
class RlmChildProvider:
    """Runs a child agent in this process, owned by the parent's scope."""

    ctx: Context
    config: Config
    _children: dict[str, _Child] = field(default_factory=dict)

    @property
    def depth_limit(self) -> int:
        """How deep delegation may go. `rlm-prompt` reads it, and this class
        enforces it.

        The prompt reads this property and does not reach into `self.config`, so
        the limit that the model is told and the limit that `start()` applies are
        one value.
        """
        return self.config.max_depth

    # ------------------------------------------------------------ admission --

    @staticmethod
    def _parent_session(parent: AgentHandle) -> Session:
        """The log of the delegating agent, which every admission path needs.

        `AgentHandle.session` is optional, because a handle need not have bound a
        session. `ph.testing.StubAgent` is one that has not. Everything below
        needs the log of the parent: the depth is folded from it, the header of
        the child cites it, and the concurrency slot is keyed by it. A missing
        session is refused with an error and not asserted, because it is a fact
        about the *caller*, and admission is where this seam refuses. The two
        neighboring checks raise the same error for a depth limit and for an
        empty prompt.
        """
        session = parent.session
        if session is None:
            raise SubagentSpawnError("a subagent needs a parent with a session to delegate from")
        return session

    async def start(self, request: SubagentRequest) -> SubagentRun:
        """Admit a child. The call returns before the child answers: that is the
        purpose."""
        parent = request.parent
        parent_session = self._parent_session(parent)
        depth = delegation_depth(parent_session)
        if depth >= self.depth_limit:
            # This is the wording of prime-agent. A model that has seen this
            # text before should not have to learn again what it means.
            raise SubagentSpawnError(
                f"RLM recursion depth limit reached (RLM_DEPTH={depth}, "
                f"RLM_MAX_DEPTH={self.depth_limit})"
            )
        prompt = request.prompt.strip()
        if not prompt:
            raise SubagentSpawnError("a subagent needs a prompt describing its task")

        run_id = f"child-{secrets.token_hex(6)}"
        # The seam chose the name before it called this provider. The name is
        # unique among the siblings of the child.
        return await self._admit(request, run_id=run_id, name=request.name or run_id)

    async def readmit(
        self, request: SubagentRequest, *, run_id: str, session_id: str, restarts: int = 0
    ) -> SubagentRun | None:
        """Take an admitted child back from the log alone, after a restart (P5-04).

        The record is the whole input. The seam rebuilt this request from the
        record and derived the ceiling again. Only the build is left to do here.
        It is the same build that `start` does, but with the ids that the log
        names and not new ones. The child therefore keeps its identity, its log
        and its place among the children of its parent.

        **No second `subagent/admitted` is written.** The admission already
        happened and is in the log of the child. The seam writes an admission
        only for a new child, so this delegation stays one delegation.

        The method declines and does not raise when this provider cannot build a
        child under the parent. The sweep of the seam is starting a *root*, and
        one unrecoverable child must not stop the rest from coming back.
        """
        parent_session = request.parent.session
        if parent_session is None:
            return None
        if session_id != child_session_id(parent_session.id, run_id):
            # A child that this provider opens is named after its parent
            # (`open_child_log`). A child with another id was never a child of
            # this provider, and it is not opened as if it were.
            log.warning(
                "ph_rlm.subagents: %s is not %s's child; not readmitted", session_id, run_id
            )
            return None
        if not request.prompt.strip():
            # Nothing can run a record that has no task again. The skip is logged,
            # because a silent skip here is a child that stays queued forever.
            log.warning("ph_rlm.subagents: %s has no prompt in its record; not readmitted", run_id)
            return None
        return await self._admit(
            request, run_id=run_id, name=request.name or run_id, restarts=restarts
        )

    async def _admit(
        self,
        request: SubagentRequest,
        *,
        run_id: str,
        name: str,
        restarts: int = 0,
    ) -> SubagentRun:
        """Build one child and start it. Admission and readmission share this.

        `restarts` is how many times this child has already been started. When it
        is above zero, the task text says so. Without that, the transcript shows
        the same instruction twice, and a model cannot follow it.
        """
        parent = request.parent
        parent_session = self._parent_session(parent)
        prompt = request.prompt.strip()
        # The seam makes sure that the request names a route that an adapter here
        # serves. It does that when it resolves the model (`resolve_model`), for
        # a spawn and for a readmission.
        options = child_options(request)

        child_session = await self._child_session(parent_session, run_id)
        # The agent is created before the seam writes the admission. The ways
        # that `agents.create` can fail (no driver, no route, options that the
        # driver rejects) therefore cannot leave an admitted child behind. The
        # agent does not *run* yet, so the order that the log needs still holds.
        try:
            # `parent=` nests the scope of the child inside the scope of its
            # parent (P6-27), so the ceiling is inherited and not applied.
            child_agent = self.ctx.require(AGENTS).create(child_session, options, parent=parent)
        except Exception as error:
            self.ctx.require(SESSIONS).dispose(child_session.id)
            raise SubagentSpawnError(f"the child agent could not be created: {error}") from error

        try:
            granted, downgrade = await self._workspace(
                parent, child_agent, child_session, request.access
            )
        except BaseException:
            # The parent went away while this child was being built, or the build
            # was stopped there (S5). In the first case the scope of the parent,
            # and the agent of the child inside it, were disposed under the await.
            # In both cases the agent is disposed. The session of the child, which
            # nothing names yet, is disposed and not tombstoned. The seam turns
            # the scope error of a parent that is gone into the refusal. The
            # cleanup is shielded, because a cancel is one of the ways to get here.
            with releasing():
                await self.ctx.require(AGENTS).dispose(child_agent.id)
            self.ctx.require(SESSIONS).dispose(child_session.id)
            raise

        run = SubagentRun(
            id=run_id,
            name=name,
            session_id=child_session.id,
            parent_id=parent.id,
            model_provider=options.provider,
            model=options.model,
            requested_access=request.access,
            granted_access=granted,
            downgrade_reason=downgrade,
            # The scope is handed back so that the *seam* can bound this child
            # (P4-13b). The provider creates the scope, but the provider must not
            # be the one that remembers the ceiling: `rehydrate` below makes a
            # second scope, and it forgot the ceiling.
            scope=child_agent.ctx,
            dispose=partial(self._dispose, run_id),
        )
        child = _Child(
            run=run,
            finished=anyio.Event(),
            options=options,
            agent=child_agent,
            session=child_session,
        )
        self._children[run_id] = child
        run.result = self._awaiter(child)

        # **The task is presented again, and that makes a retry real.** A step
        # claims the task from the inbox when it starts, and the claim is a logged
        # splice. A resumed child whose task was not presented again therefore
        # finds an empty inbox, ends at step zero, and reports `completed` for
        # work that it never did.
        # **The exception is a task that is still there** (S3). The inbox gives up
        # a batch only in the step that records it. A child that was stopped
        # before its first step therefore has its task pending in its own log,
        # and a second copy would hand over the same work twice.
        if not _task_pending(child_agent):
            child_agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": _task_text(prompt, restarts)}],
                    source=_TASK_SOURCE,
                )
            )
        try:
            # The child is an artifact of the scope of the *parent* (I2). A
            # disposed parent therefore unwinds its children and does not leave
            # them running with nobody to answer. `delete()` releases the child
            # early, and is not a second cleanup path that must remember
            # everything. That release is a revocation. The release of the seam
            # (`run.dispose`) is not.
            child.teardown = await parent.ctx.effect(
                lambda: partial(self._release, run_id, PARENT_TEARDOWN),
                label=f"subagent:{run_id}",
            )
            # `resumed` makes a restart countable. Without it, the log holds one
            # `running` record per start and no way to tell a first start from a
            # fourth, and the ladder needs that fold. A parent that was torn down
            # under the await of the effect is refused here too, because the job
            # is registered on the scope of the parent.
            await self._attach(child, parent, cause="resumed" if restarts else None)
        except InactiveScopeError:
            # The child is ended and not left live. This is the release that a
            # disposed parent runs, run now. If the teardown of the parent got
            # here first, this call finds nothing.
            await self._release(run_id, PARENT_TEARDOWN)
            raise
        except BaseException:
            # The build was stopped while the child was wired up, or the job would
            # not start (S5). The child is let go without a tombstone, in the same
            # way as a child that the seam would not admit (`_dispose`). The seam
            # writes the admission of a spawn after this method returns. The
            # admission of a readmission is left for the next sweep to try again.
            with releasing():
                await self._dispose(run_id)
            raise
        return run

    async def _child_session(self, parent_session: Session, run_id: str) -> Session:
        """The session of this child, opened through `open_child_log`.

        `open_child_log` claims the session, then resumes it when a log for it
        survived, and otherwise creates it. Every child is opened through that
        function of the seam, which names and files the log in the way that the
        parent finds it.

        A readmitted child usually has no log at all. It never ran a turn, and
        the little that its session held was still in a buffer when the daemon
        stopped. But a child that *did* reach disk must be resumed and not
        recreated, for the reason that `open_session` gives. Creating a session
        over a stored log puts `seq` backwards in the middle of the file, and
        the log can then no longer be read at all (P5-03).

        **The session is claimed, as a root is (I-5, L2).** The log of a child is
        a session of its own, and since Phase 11 it is the only record of the
        child. It therefore has one writer, which is the mount of the root that
        spawned it, and no host opens it as a root (P11-08). The lease keeps a
        *second process* out, for example another daemon that mounted the same
        root. The scope of the mount holds the lease, as it holds the lease of
        the root (`open_session` claims on `ctx.root`). The session of the child
        lives in the store of the deployment for as long as the root is mounted.
        The last act of the mount writes the session before the lease is given
        back. A child that another process holds is refused with `SessionBusy`,
        and a readmission settles that child as one that it could not resume.
        """
        return await open_child_log(self.ctx, parent_session, run_id, agentPreset="rlm")

    # ------------------------------------------------------------- lifecycle --

    async def _attach(
        self, child: _Child, parent: AgentHandle, *, cause: StatusCause | None = None
    ) -> None:
        """Wire a live child to its parent and start to drive it.

        A new admission and a rehydration use this one path, so the two cannot
        attach different things. They attach the job, and the `cause` that the
        log of the child records beside `running`.
        """
        assert child.session is not None
        limit = self.config.max_concurrent
        # `ctx.jobs` detaches the work and does not run it inline. The job gives
        # the run an id, a cancel and `job/*` events at no cost. A subagent is
        # the example that the jobs seam itself gives of work that outlives the
        # step that started it.
        job = await self.ctx.require(JOBS).start(
            kind="subagent",
            label=f"{child.run.name} ({child.run.id})",
            run=lambda _job: self._drive(child, cause=cause),
            # The lifetime of the delegation, which is the lifetime of the parent.
            # A disposed parent abandons the drive. A settled child releases its
            # own entry, so that a chatty exchange does not leave one job per
            # message behind.
            scope=parent.ctx,
            # The queue belongs to the jobs seam, and `Config.max_concurrent` sets
            # its size. The key is the session of the *parent*, so a cap limits
            # the fan-out of one agent, and the children of one parent cannot
            # starve the children of another root. The jobs seam is responsible
            # for the wait, the release on every ending and the lifetime of the
            # queue. This provider does not derive them again.
            slot=None if limit is None else (self._parent_session(parent).id, limit),
            # Written only when the child actually waited. An admitted child with
            # no status already reads as `queued`. This record is for the case
            # where that is true *for a reason*.
            on_queued=lambda: self._waiting(child, slots=limit),
        )
        child.job_id = job.id

    async def rehydrate(self, run_id: str) -> bool:
        """Give a settled child a runtime again, so that it can be addressed (P3-13).

        The session and the log of the child survived settlement. `_quiesce`
        released the agent, and the agent is what holds an inbox. Rehydration
        therefore creates the agent again on the same session and drives it
        again. The `Inbox` rebuilds itself from `agent/inbox/spliced` in that
        log, so anything that was queued before the child settled is still there.

        A *deleted* child is not rehydrated. The tombstone in its log records
        that its parent revoked it, and to revive it quietly would make that
        record false. Passivation across a restart, where the session itself is
        gone and must be loaded from disk, belongs to the daemon (Phase 5).
        """
        child = self._children.get(run_id)
        if child is None or child.agent is not None:
            # The child is unknown or already running. A *revoked* child never
            # reaches here. `_release` takes it out of `_children` (`_take`) and
            # calls `forget()` (`_drop`), so the lookup of the service refuses it
            # first.
            return False
        session = self.ctx.require(SESSIONS).get(child.run.session_id)
        if session is None:
            log.debug("ph_rlm.subagents: %s has no live session to rehydrate", run_id)
            return False
        child.session = session
        # **This check comes before anything is built.** The parent owns the drive
        # job *and*, since P6-27, the scope that the child nests in. A missing
        # parent is therefore a refusal and not a degradation. If this guard came
        # after `agents.create`, it would leave an agent and a scope behind: an
        # orphan under the registry root that holds the deployment-wide ceiling
        # and that nothing would ever dispose.
        parent = self.ctx.require(AGENTS).get(child.run.parent_id)
        if parent is None:
            log.debug("ph_rlm.subagents: %s has no live parent to own its drive", run_id)
            return False
        agents = self.ctx.require(AGENTS)
        child.agent = agents.create(session, child.options, parent=parent)
        settled = child.finished, child.result
        try:
            await self._runtime(child, parent, session)
            # A new `finished` event. The awaiter that is already on the run reads
            # it at await time, because its closure holds the `_Child` and not the
            # old Event. The result is cleared until the child gives a new one.
            # The result that it holds answers the previous question, and a waiter
            # on this question must not get it. `suspend` would also read it as an
            # ending. A child that `suspend` stopped in the middle of a question
            # would then stay `running`, and a restart would count it.
            child.finished = anyio.Event()
            child.result = None
            await self._attach(child, parent, cause="rehydrated")
        except BaseException:
            # **The child is settled again, as it was** (S4). An agent that was set
            # and never driven made every later wake answer "already running", and
            # every later send landed in an inbox that nobody read. A kernel
            # cancels a binding call when its cell settles, so a cell that is
            # stopped in the middle of a send is enough to get here. The answer
            # that the child held is put back for any caller that still waits on
            # it.
            child.finished, child.result = settled
            agent, child.agent = child.agent, None
            if agent is not None:
                with releasing():
                    await agents.dispose(agent.id)
            raise
        return True

    async def _runtime(self, child: _Child, parent: AgentHandle, session: Session) -> None:
        """Set up again everything that a child needs around a new agent.

        **This is one list, because it has been discovered twice.** The agent of
        a child is not finished when `agents.create` returns. Its scope must be
        handed back, its ceiling applied again, and its workspace taken again,
        each from what the admission recorded. `rehydrate` got those steps late
        and one at a time. The comment on the spawn path says so about the second
        (*"`rehydrate` below makes a second scope, and it forgot the ceiling"*),
        and the third arrived the same way. Neither omission failed loudly. A
        child came back holding the whole deployment, and later a child came back
        writing into a tree that was not its own.

        The work of the child comes back with it. The tier gives that result, and
        nothing here arranges it. Disposal commits the checkout to the branch of
        the child before it removes the checkout, and `_add` attaches an existing
        branch and does not reset it. The same run id resolves to the same
        branch, so a second question starts where the first stopped.
        """
        # A new scope means a new ceiling. The filters that were applied at
        # admission were disposed with the scope that settled (P4-13b).
        agent = child.agent
        assert agent is not None, "a child is re-asked only after it has an agent"
        child.run.scope = agent.ctx
        if child.run.grant is not None:
            child.run.grant.apply(self.ctx, agent.ctx)
        # This is the access that the **admission** recorded, never the access of
        # a caller. Being asked a second question must not widen what the first
        # question was allowed.
        await self._workspace(parent, agent, session, child.run.requested_access)

    def _awaiter(self, child: _Child) -> SubagentAwaiter:
        async def wait() -> SubagentResult:
            await child.finished.wait()
            return child.result or SubagentResult(status="error", error="the child never settled")

        return wait

    def _child_log(self, child: _Child) -> Session | None:
        """The log of the child, whether or not the child still has an agent.

        `child.session` goes with the agent when a child settles (`_quiesce`).
        The log itself stays live in the store for the life of the mount. A
        rehydration, a revocation or a suspension records there what happened to
        the child. This uses `get` for the same reason as `_drop`: when a mount
        unwinds, the store may be gone.
        """
        if child.session is not None:
            return child.session
        sessions = self.ctx.get(SESSIONS)
        return sessions.get(child.run.session_id) if sessions is not None else None

    def _waiting(self, child: _Child, **extra: JsonValue) -> None:
        """Record `queued` in the log of the child, when the child has a log to
        write to."""
        own = self._child_log(child)
        if own is not None:
            record_waiting(own, **extra)

    async def _drive(self, child: _Child, *, cause: StatusCause | None) -> None:
        """Run the child to quiescence, tell the parent, then let it go."""
        run = child.run
        own = child.session
        assert own is not None, "a child is driven only while it has its own log"
        try:
            # The drive waits until the seam opens the gate (`SubagentRun.ready`).
            # By then the child is bounded, and for a readmitted child its own
            # children are swept. Neither its ceiling nor the children that its
            # first prompt shows are still being worked out.
            await run.ready.wait()
            # **After a release, the release owns the ending.** The drive checks
            # for that here, at the gate, and after the run below. A child that
            # was revoked, refused or suspended already has its ending written,
            # and an ending from the drive would overwrite it. `Job.cancel` no
            # longer reaches a job that is past its slot, so the drive must make
            # this check itself.
            if child.finished.is_set():
                return
            # The status is `running` in both cases, and `cause` says *why*. The
            # state of a child folds the status with the last write winning, and a
            # woken child that is working must not read as not running. A restart
            # is on the disk of the child before the attempt that it counts (S10).
            # `record_started` keeps that rule. `record_started` also refuses a
            # child that is already tombstoned. A release that still waits on the
            # disk has not set `finished` yet, but it owns the ending all the same.
            if await record_started(self.ctx, own, cause=cause) is None:
                return
            agent = child.agent
            assert agent is not None, "a child runs only after it has an agent"
            await agent.run()
            if child.finished.is_set():
                return
            answer = last_answer(own)
            # The seam has the rule for what a child with an `output` owes. This
            # provider writes the ending.
            returned = settle_value(own)
            if returned.problem:
                # The text of the child is kept. The parent gets the prose and
                # learns that the shape it asked for is missing.
                await self._fail(child, returned.problem, answer=answer)
                return
            # From the side of the parent, silence looks the same as a hang, so a
            # child that never sent a message is announced. A child that *did*
            # reply needs no notice, because the reply is the notice. The log of
            # the child says whether it replied, also across a restart. A value
            # is announced whatever else the child said, because the value is
            # what the parent asked for.
            notice: ChildNotice | None = None
            if returned.value is not None:
                notice = ChildNotice(
                    text=f"[rlm child {run.name} ({run.id}) returned: {dumps(returned.value)}]",
                    summary=f"{run.name} returned its result",
                )
            elif not replied_to_parent(own):
                notice = ChildNotice(
                    text=f"[rlm child {run.name} ({run.id}) completed without sending a "
                    f"reply.{said_text(answer)}]",
                    summary=f"{run.name} finished without replying",
                )
            # The ending of the child is on its own disk before its parent gets the
            # answer (F1). `record_ended` flushes, and the waiters wake after it.
            # The notice is recorded with the ending, so a restart delivers a
            # notice that a crash held back.
            if not await self._end(child, answer=answer, value=returned.value, notice=notice):
                # The child was revoked while its run was returning from the
                # cancel of the release. It did not finish, and its tree keeps the
                # mark that it has.
                return
            # This is the other half of retain-by-default (P6-28). The checkout
            # of a child that finished is not evidence of anything. The mark that
            # was set at acquire is therefore withdrawn, and `worktree-ephemeral`
            # keeps its promise for the case that it was written for. This runs
            # before `_quiesce` in the `finally` below. `_quiesce` disposes the
            # scope that holds the workspace, and after that there is nothing
            # left to unmark.
            self._withdraw(child)
        except Exception as error:
            if child.finished.is_set():
                return
            await self._fail(child, f"{type(error).__name__}: {error}")
            log.debug("ph_rlm.subagents: child %s failed", run.id, exc_info=True)
        finally:
            child.finished.set()
            # A settled child holds an agent scope, and that scope owns the kernel
            # subprocess of the child (`code-runtime:<namespace>` is an effect of
            # it). Keeping the scope for the lifetime of the host leaked one
            # CPython process, and the whole namespace of the child, for each
            # delegation.
            await self._quiesce(child)

    async def _fail(self, child: _Child, message: str, *, answer: str = "") -> None:
        """End a child that failed, in the same order as any ending.

        The account in the log of the child comes first: why it failed, where its
        checkout is, and what it last said. The notice to its parent comes after.
        """
        run = child.run
        await self._end(
            child,
            answer=answer,
            error=message,
            notice=ChildNotice(
                text=f"[rlm child {run.name} ({run.id}) failed: {message}"
                f"{self._evidence(child)}{said_text(answer)}]",
                summary=f"{run.name} failed",
            ),
        )

    async def _end(
        self,
        child: _Child,
        /,
        *,
        answer: str,
        notice: ChildNotice | None,
        value: JsonObject | None = None,
        error: str | None = None,
    ) -> bool:
        """End `child` in its own log (`record_ended`), and hold the result that
        its waiters are told.

        The result is `done`, or `error` with the text of `error` when there is
        one. The method returns `False` when `record_ended` refused the ending,
        because the child was revoked or had ended already. The record carries
        the head of `answer` and, for a failure, `error` as its `detail`.

        **The result is held before the ending is written**, not after. `suspend`
        reads a child that holds no result as one that is still working, and
        writes `queued`. The flush of the ending is an await, and `suspend` can
        run during it. A refused ending puts back the result that was held
        before. A waiter then hears what the log of the child says, and not an
        ending that the log does not hold. That is nothing for a revoked child,
        or the `done` that this drive wrote before a later step raised.
        """
        own = child.session
        assert own is not None, "a child ends only while it has its own log"
        status: SettledStatus = "done" if error is None else "error"
        fields: dict[str, JsonValue] = {}
        if preview := head(answer, self.config.answer_preview_chars):
            fields["answerPreview"] = preview
        if error is not None:
            fields["detail"] = error
        held = child.result
        child.result = SubagentResult(status=status, answer=answer, value=value, error=error)
        if await record_ended(self.ctx, own, status, notice=notice, **fields) is not None:
            return True
        child.result = held
        return False

    def _evidence(self, child: _Child) -> str:
        """Where the tree of the failed child is, for the message that reports
        the failure.

        **This answers the motivating case of the row at the place where the
        parent looks** (P6-28). Retention keeps the checkout, and this text stops
        the checkout from being evidence that nobody can find. The workspace
        events of a child are in the log of the *child*. A parent that is told
        only "it failed" must diagnose from a transcript, and the row opens with
        that sentence.

        The text is read from the log and not from the seam, deliberately. This
        runs on the way to `_quiesce`, and the same sentence must be true on a
        path where the scope is already gone. The fold answers from events that
        are already written.

        The method returns an empty string when there is nothing to report: no
        tier, a `shared` workspace, or a child whose tree was discarded. A notice
        that named a directory for every failure would name directories that are
        not there.
        """
        if child.session is None:
            return ""
        kept = [one for one in workspace_survivors(child.session) if one.outcome == "retained"]
        return f" Its workspace is kept at {kept[0].root}." if kept else ""

    def _withdraw(self, child: _Child) -> None:
        """Take back the mark that the tree of this child got at acquire (P6-28).

        This method only withdraws, so it takes no reason. The marking half runs
        in `_workspace`, where the seam and the kind are already available. A
        shared helper with a `reason` parameter, to which one caller always
        passed `""`, was two spellings of one operation.

        The method uses `get` and not attribute access, and it tolerates an agent
        that never took a workspace. It runs on a settle path, and both a profile
        with no workspace row and a child whose tier declined are ordinary. The
        seam answers `False` and does not raise, for exactly this caller.
        """
        seam = self.ctx.get(WORKSPACE)
        if seam is not None and child.agent is not None:
            seam.retain(child.agent.id, "")

    async def _quiesce(self, child: _Child) -> None:
        """Drop everything that a settled child no longer needs.

        The terminal `result` stays, so a caller that awaits `result()` after the
        child is gone still gets its answer. Failures are contained. This runs in
        a `finally`, and a teardown that raised would replace the outcome of a
        settled child with a disposal error.
        """
        # This uses `get` for the same reason as `_drop`. The unwind of a mount
        # reaches here after the rows that provide these services have gone, and
        # there is nothing left to release.
        jobs = self.ctx.get(JOBS)
        if child.job_id is not None and jobs is not None:
            # The entry is released, not abandoned. The work finished, so the
            # entry goes and the job is not reported as canceled.
            jobs.forget(child.job_id)
        child.job_id = None
        agent, child.agent, child.session = child.agent, None, None
        agents = self.ctx.get(AGENTS)
        if agent is None or agents is None:
            return
        try:
            await agents.dispose(agent.id)
        except Exception:  # pragma: no cover - teardown must not mask an outcome
            log.debug("ph_rlm.subagents: disposing child %s failed", child.run.id, exc_info=True)

    # ---------------------------------------------------------------- delete --

    async def _workspace(
        self,
        parent: AgentHandle,
        child_agent: AgentHandle,
        child_session: Session,
        access: Access,
    ) -> tuple[Access, DowngradeReason | None]:
        """Take the workspace of the child, and report what it actually got (D21, E3).

        The method takes the `access` and not the request, because the other
        caller has no request. A child that gets its runtime back has only what
        its admission recorded, and it must come back with exactly that.

        **The workspace is acquired here and not left to the lifecycle row.** The
        base and the access of a child are the decision of its *parent*, and the
        row knows neither. The row would give the child a `write` workspace over
        the directory of the process itself, which is both the wrong tree and the
        wrong guarantee. Because the child branches from the root of the parent,
        a fan-out lands on sibling branches and not on one shared checkout (E2).

        `granted` is access **to the project**, not to a directory. A
        `worktree-ephemeral` child may write its checkout freely, and none of it
        is ever merged. What it was granted of the project is therefore `read`.
        This is the seam's own reading of `repo_writable`, applied one level up.
        """
        seam = self.ctx.get(WORKSPACE)
        if seam is None:
            # The profile has no workspace row at all. The conservative claim is
            # the only correct one: nothing here can enforce a writable repo, so
            # nothing promises one.
            return "read", "workspace-not-mounted"
        workspace = await seam.acquire(
            session_id=child_session.id,
            agent_id=child_agent.id,
            # The base comes from `ctx.fs` and is not derived again from the
            # workspace of the parent. "Where does a relative path of this agent
            # land" already has one implementation. A second one in this package
            # would be the one that must not disagree with it.
            base=self.ctx.require(FS).root_for(parent),
            access=access,
            session=child_session,
            # The scope of the child itself. A revoked or finished child releases
            # its checkout with everything else that it took (I2).
            scope=child_agent.ctx,
        )
        if discards_writes(workspace.kind):
            # **The tree is retained from the moment that it exists** (P6-28). The
            # window to mark it closes with the scope of the child, and the
            # outcomes that most need the evidence close that scope first. On the
            # `parent-teardown` path the worktree is released *before* `_release`
            # runs, as the docstring of that method states. The policy therefore
            # cannot be "retain when it fails", because there is nowhere to put
            # that. It has to be "retain unless it succeeds", and `_drive` clears
            # the mark.
            #
            # This applies only to the kind that discards. Every other kind
            # already keeps a dirty tree for review, and a committed branch
            # survives release in any case. Retaining those would add to the pile
            # and save nothing from it.
            #
            # `phern workspaces gc` limits the pile that this creates. That is
            # not a note about future work. This policy was held back until the
            # command existed, because inverting the promise of
            # `worktree-ephemeral` with no collector is a worse trade than the
            # lost evidence that it fixes.
            seam.retain(child_agent.id, "the child has not settled cleanly")
        return project_access(workspace.kind), None

    async def revoke(self, run_id: str, reason: str) -> bool:
        """`RevokingProvider`: stop one child that this provider holds, and write a
        tombstone in the log of the child. Returns `False` for a child that this
        provider does not hold.

        `SubagentService.delete` calls this, and every revocation goes through
        that method. The seam tombstones a child that this provider is not
        running, such as one that an earlier process settled.
        """
        return await self._release(run_id, reason)

    async def _dispose(self, run_id: str) -> None:
        """`SubagentRun.dispose`: let go of a child that the seam would not admit,
        without revoking it.

        The seam ended the child already and said why (`_abandon`). A refusal is
        not a deletion, so this is `_release` without the tombstone.
        """
        child = self._take(run_id)
        if child is not None:
            await self._drop(child)

    async def _release(self, run_id: str, reason: str) -> bool:
        """The one revocation path, whether the model asked or the parent unwound.

        **On the parent-teardown path the scope of the child is already gone**
        (P6-27). `Context.dispose` unwinds `_children` before its own effects, and
        this method runs as one of those effects. The scope, the worktree and the
        kernel of the child are therefore released *before* this is called.

        Nothing here breaks because of that. `cancel` touches only the phase and
        the inbox, and disposing an inactive scope returns at once. But it
        **limits what this path may do**. The log of the child and its tombstone
        belong to the store and not to the scope of the child, so they are live.
        Anything that needs the scope of the *child* (its own services, its
        workspace) is not live. Such a step would work when the model calls
        `delete()` and fail here.

        **This is only ever a revocation.** A mount that goes away reaches its
        children first, through `suspend`, and leaves nothing here to release. A
        teardown of a parent that does reach a child is therefore one that the
        mount outlives: the children of a settled child, or a parent that
        `ctx.agents` disposed. Those children really are revoked.

        **A settled child is not settled again.** Its drive wrote `done` or
        `error`, and a `canceled` written over that turned a finished child into
        a revoked one.
        """
        child = self._take(run_id)
        if child is None:
            return False
        # The ending and the tombstone are written together (S14), in the log of
        # the child and on its disk, before anything is let go. The ending is
        # written at all because a revoked child is not merely absent: a panel
        # that knew only `deleted` could not say whether the child had run.
        own = self._child_log(child)
        if own is not None:
            await record_deleted(self.ctx, own, reason)
        await self._drop(child)
        return True

    def _take(self, run_id: str) -> _Child | None:
        """Take the child of `run_id` off this provider and stop it (`_stop`).
        Returns `None` for a child that this provider does not hold, or no longer
        holds."""
        child = self._children.pop(run_id, None)
        if child is not None:
            self._stop(child)
        return child

    def _stop(self, child: _Child, *, keep_inbox: bool = False) -> None:
        """Stop a child that this provider is letting go of, before anything about
        it is written. This drops its revocation on the scope of the parent,
        cancels its queued job and cancels its agent. `keep_inbox` is for a child
        that is suspended and not released (`suspend`)."""
        if child.teardown is not None:
            child.teardown.drop()
        # This uses `get` for the same reason as `_drop`: when a mount unwinds, the
        # row may be gone.
        jobs = self.ctx.get(JOBS)
        if child.job_id is not None and jobs is not None:
            # The job was queued and never ran. `Job.cancel` stops the wait for a
            # slot that the job will now never take. A body that is parked on a
            # limiter reads no token. Without this cancel, a revoked child stays
            # queued behind children that may never settle, and it holds its place
            # and `JobService.settled()` until a slot is free. The cancel is
            # harmless for a child that is already running: the agent stops that
            # child, one line down.
            jobs.cancel(child.job_id)
        if child.agent is not None:
            child.agent.cancel(AgentCancelCause(kind="parent"), keep_inbox=keep_inbox)

    async def _drop(self, child: _Child) -> None:
        """Let go of a child that `_take` stopped: wake its waiters, release its
        scope, and tell the seam that it is gone."""
        self._let_go(child)
        await self._quiesce(child)
        # This uses `get` and not attribute access. On the parent-teardown path
        # this runs while scopes are unwinding, and the service that the seam
        # provides may already be gone. A teardown that raised would abort the
        # rest of the unwind.
        registry = self.ctx.get(SUBAGENTS)
        if registry is not None:
            registry.forget(child.run.id)

    def suspend(self) -> None:
        """Stop every child that this provider holds, and revoke none of them (S1).

        **This is the disposer of this row.** The row injects every row that it
        uses, so it activates after them and unwinds before them: before `jobs`,
        `subagents`, and the `agent` row whose scopes hold every parent. A mount
        that goes away (a daemon that stops, or a root that is remounted on a new
        profile) reaches here while all of those are still up. This method also
        drops the revocation of each child from the scope of its parent. Before
        this disposer existed, the unwind of a mount reached its children only
        through those effects, after `jobs` had gone. The release of a working
        child raised before it wrote anything, so the next start read the child
        as a crash and spent a rung of its ladder. A settled child was
        tombstoned.

        A suspended child stays live. Its status is `queued` with the reason, it
        has no tombstone, and it keeps its inbox. Its agent is stopped with the
        inbox kept. The agent is then let go of here and not left for the
        teardown of the drive to dispose. Disposing an agent clears what is
        queued for it, and a drive whose model call honors the cancel wakes while
        this row can still reach the registry. The unwind of the mount takes the
        scope. The resume sweep readmits the child, and any caller that waited on
        it is answered `queued` with the same reason. A settled child is left as
        it ended.

        The method uses `get` and not `require` for the one case that this does
        not cover: the row deactivates while the mount lives on, because a row
        that it injects went away.
        """
        registry = self.ctx.get(SUBAGENTS)
        for child in self._children.values():
            self._stop(child, keep_inbox=True)
            child.agent = None
            if child.result is None:
                self._waiting(child, detail=SUSPENDED_DETAIL)
                child.result = SubagentResult(status="queued", error=SUSPENDED_DETAIL)
            self._let_go(child)
            if registry is not None:
                registry.forget(child.run.id)
        self._children.clear()

    @staticmethod
    def _let_go(child: _Child) -> None:
        """Mark a child as ended for everyone that waits on it.

        The first waiter is the caller that awaits its `result()`. The second is
        a drive that is still parked at the gate of the seam, for a child that
        was refused or that was released before its admission was written. That
        drive is let through and finds itself released (`_drive`). Otherwise it
        would wait on an event that nobody sets, because `Job.cancel` trips a
        token that `ready.wait()` never reads.
        """
        child.finished.set()
        child.run.ready.set()


_TASK_SOURCE = PluginSource(plugin="ph_rlm.subagents", form="relay")
"""Where a child's task comes from, as its inbox and its transcript record it."""


def _task_pending(agent: AgentDriver) -> bool:
    """Whether the child's task is still in its inbox, not yet taken by a step."""
    return any(
        message.source == _TASK_SOURCE
        for message in (*agent.inbox.next_turn, *agent.inbox.next_step)
    )


def _task_text(prompt: str, restarts: int) -> str:
    """The task as the child reads it. After an interruption, the text also says
    why the task appears twice.

    The transcript of a resumed child already holds the turn that its harness
    cut short, closed by the repair of the resume. If the task is presented again
    with no mention of that, the model sees the same instruction twice and can
    conclude that it already answered. Naming the interruption lets the model
    read the second attempt as a second attempt.

    The number is how many times this child has been started. It is not the
    position of the child on the ladder of the harness. The two differ as soon as
    progress clears the ladder, and the history of the model itself is the
    correct one to tell it about.
    """
    if restarts < 1:
        return f"{TASK_PREFIX}\n\n{prompt}"
    return f"{TASK_PREFIX}\n\n{prompt}\n\n{cut_short_note(restarts + 1)}"


@plugin(
    "rlm-subagent-provider",
    affects="environment",
    config=Config,
    inject=[SUBAGENTS, AGENTS, SESSIONS, JOBS],
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the `rlm-child` provider and expose it for the bindings row. Also
    offer the skills that ph-core ships for handing to a child (`categorize`)."""
    provider = RlmChildProvider(ctx=ctx, config=config)
    ctx.require(SUBAGENTS).register_provider(PROVIDER_NAME, provider)
    ctx.provide(RLM_CHILDREN, provider)
    offer_shipped_skills(ctx)
    # Registered after the provider, so it runs before the provider is withdrawn.
    ctx.add_disposer(provider.suspend, label="rlm-children.suspend")
