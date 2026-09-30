"""`rlm-subagent-provider` — `rlm()` as a `ctx.subagents` provider (P3-11).

Ported from prime-agent's `AgentSession._startRlmChildRun`. The semantics are
its; the mechanism is pH's seams. Five properties are the whole design, and each
is a thing the obvious implementation gets wrong:

**The handle returns before the child answers.** Stated once, in
`ph.seams.subagents` — this file's job is to keep the promise. `start()` creates
the session and the agent, starts the job, and returns; the seam writes the child's
admission into the child's own log, and the child's reply arrives on a later turn
as an ordinary inbox message.

**Every record of a child is in the child's own log** (Phase 11). This provider
writes nothing to a parent's log: a child's starts, waits, ending and tombstone go
through the seam's doors into the child's log, and a parent — the roster tool, the
prompt, the resume sweep — reads them from there. The agent is *created* before
the admission, so a `create` failure leaves no admitted child behind — and not
started until the seam opens its gate, after the admission is on disk.

**A child is an artifact of its parent's scope.** Acquired through
`parent.ctx.effect()`, so a disposed parent unwinds its children (I2) and
`delete()` is just "release it early". Without that, a settled child kept its
agent scope alive for the host's lifetime — and that scope owns the child's kernel
subprocess, so every delegation leaked a CPython.

**Usage stays where it was spent.** A child's answers carry their usage in its own
log, and a goal's budget, the retry ladder and the TUI panel read it there
(`ChildState.tokens`, `restarts_since_progress`). The copy this provider used to
mirror into the parent's log could lag the child's own after a crash, and had to be
caught up on every resume.

**The child's workspace is taken here, not by the lifecycle row** (P4-08): its
base is the parent's root and its access is the parent's decision, and the row
knows neither. `granted` is what the child got *of the project* — a
`worktree-ephemeral` child writes freely and merges nothing, so it was granted
`read` — and the *reason* for any narrowing travels as a code, not prose, so the
log stays parseable.

@module ph_rlm.subagents
"""

from __future__ import annotations

import logging
import secrets
from dataclasses import dataclass, field, replace
from functools import partial

import anyio
from pydantic import Field

from ph.agent.types import AgentCancelCause, AgentDriver, AgentHandle, AgentOptions
from ph.cordis import Context, InactiveScopeError, plugin
from ph.json import JsonValue
from ph.keys import AGENTS, FS, JOBS, LLM, SESSIONS, SUBAGENTS, WORKSPACE
from ph.llm.adapter import LlmError
from ph.llm.types import PluginSource, create_user_message, text_of
from ph.seams.subagents import (
    PARENT_TEARDOWN,
    SUSPENDED_DETAIL,
    Access,
    ChildNotice,
    DowngradeReason,
    StatusCause,
    SubagentAwaiter,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentSpawnError,
    child_model_key,
    child_route,
    open_child_log,
    record_deleted,
    record_ended,
    record_started,
    record_waiting,
)
from ph.seams.workspace import discards_writes, project_access, workspace_survivors
from ph.session import (
    Session,
    SessionEvent,
    child_session_id,
    derive_event_message,
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
"""Prime Agent's `RLM_MAX_DEPTH`. Depth 0 delegates, depth 1 delegates, depth 2
does the work — three levels is already a lot of indirection between a human's
question and the tokens that answer it."""
TASK_PREFIX = "[task from parent]"
"""Ported verbatim: the child's prompt recognizes this label."""


class Config(WireModel):
    """Row config for `rlm-subagent-provider`."""

    max_depth: int = RLM_MAX_DEPTH
    max_concurrent: int | None = Field(default=None, ge=1)
    """How many of one parent's children run at once. `None` is no cap.

    **A queue, not a refusal.** A parent that delegates nine tasks asked for nine,
    and refusing the ninth because eight are running answers a question about
    resources with a refusal about intent. The *drive* waits; admission is
    untouched.

    **Per parent, so it bounds *fairness*, not host load**: ten roots at four
    apiece is forty children. It exists so one agent's fan-out cannot starve
    another root's, and a deployment bounding the *host* wants `jobs`'
    `concurrency` instead. Both apply.

    The queue itself is `ctx.jobs`' (`slot=`), and the mechanics — waiting,
    releasing on every ending, the queue's own lifetime — are stated there rather
    than here, where they would be a second account of somebody else's code."""
    answer_preview_chars: int = 240
    """How much of the child's answer the status record carries, for `ph trace`
    and the P3-19 panel. There is deliberately no `default_access` knob here: the
    request default lives on `SubagentRequest`, and a second copy would be a
    documented lever that silently did nothing while the tier is unmounted."""


def delegation_depth(session: Session) -> int:
    """How many delegations deep this session is, from its own header.

    The typed field, not `to_wire()["delegationDepth"]`: reconstructing a wire
    alias by hand means a rename returns 0 rather than failing — and 0 *opens*
    the depth gate. Read from the header rather than counted at spawn time,
    because a resumed child has no live parent to ask and the gate must hold.
    """
    return session.header.delegation_depth or 0


@dataclass(slots=True)
class _Child:
    """The live half of one delegation: what a fold cannot reconstruct.

    Settlement releases what a *finished* child does not need — its agent scope
    (and with it the kernel subprocess), the session observer, the session
    handle. What it keeps is what a late `result()` or a rehydration needs: the
    run, the outcome and the options to rebuild the agent with.
    """

    run: SubagentRun
    finished: anyio.Event
    options: AgentOptions
    """The options the child's agent was built with: its parent's at spawn, with the
    child's route resolved over them. Kept for a rehydration, which rebuilds the agent
    after the parent that lent them may have gone. The admission records the route and
    the reasoning effort the spawn *asked* for (`Admission.reasoning_effort`); an
    effort it left unnamed was its parent's, and survives only here."""
    agent: AgentDriver | None = None
    session: Session | None = None
    job_id: str | None = None
    """The drive job, owned by the *parent's* scope. Not the child's: the child's
    scope is disposed *by* the drive job's own last act, and a job that abandoned
    itself would report `canceled` for work that finished."""
    result: SubagentResult | None = None


@dataclass(slots=True)
class RlmChildProvider:
    """Runs a child agent in this process, owned by the parent's scope."""

    ctx: Context
    config: Config
    _children: dict[str, _Child] = field(default_factory=dict)

    @property
    def depth_limit(self) -> int:
        """How deep delegation may go. Read by `rlm-prompt`, enforced here.

        Exposed rather than letting the prompt reach into `self.config`, so the
        limit the model is told and the limit `start()` applies are one value.
        """
        return self.config.max_depth

    # ------------------------------------------------------------ admission --

    @staticmethod
    def _parent_session(parent: AgentHandle) -> Session:
        """The delegating agent's log, which every admission path needs.

        `AgentHandle.session` is optional because a handle need not have bound one
        — `ph.testing.StubAgent` is one that has not — while everything below
        needs the parent's log: the depth is folded from it, the child's header
        cites it, and the concurrency slot is keyed by it. Refused at the door
        rather than asserted, because it is a fact about the *caller*, and
        admission is where this seam says no (its two neighbors raise the same
        error for a depth limit and an empty prompt).
        """
        session = parent.session
        if session is None:
            raise SubagentSpawnError("a subagent needs a parent with a session to delegate from")
        return session

    async def start(self, request: SubagentRequest) -> SubagentRun:
        """Admit a child. Returns before it answers (the whole point)."""
        parent = request.parent
        parent_session = self._parent_session(parent)
        depth = delegation_depth(parent_session)
        if depth >= self.depth_limit:
            # Prime Agent's wording; a model that has seen this text before
            # should not have to re-learn what it means.
            raise SubagentSpawnError(
                f"RLM recursion depth limit reached (RLM_DEPTH={depth}, "
                f"RLM_MAX_DEPTH={self.depth_limit})"
            )
        prompt = request.prompt.strip()
        if not prompt:
            raise SubagentSpawnError("a subagent needs a prompt describing its task")

        run_id = f"child-{secrets.token_hex(6)}"
        # Named by the seam, unique among its siblings, before this was asked.
        return await self._admit(request, run_id=run_id, name=request.name or run_id)

    async def readmit(
        self, request: SubagentRequest, *, run_id: str, session_id: str, restarts: int = 0
    ) -> SubagentRun | None:
        """Take an admitted child back from the log alone, after a restart (P5-04).

        The record is the whole input: the seam rebuilt this request from it and
        re-derived the ceiling, so what is left here is the build — the same one
        `start` does, with the ids the log names rather than fresh ones, so the
        child keeps its identity, its log and its place among its parent's children.

        **No second `subagent/admitted`.** The admission already happened and is in
        the child's log; the seam writes one only for a new child, so this delegation
        stays one.

        Declines rather than raises when the parent is not one this provider can
        build a child under — the seam's sweep is starting a *root*, and one
        unrecoverable child must not stop the rest from coming back.
        """
        parent_session = request.parent.session
        if parent_session is None:
            return None
        if session_id != child_session_id(parent_session.id, run_id):
            # A child this provider opens is named for its parent (`open_child_log`);
            # one that is not was never this provider's, and is not opened as if it were.
            log.warning(
                "ph_rlm.subagents: %s is not %s's child; not readmitted", session_id, run_id
            )
            return None
        if not request.prompt.strip():
            # A record with no task is one nothing can re-run. Said out loud: a
            # silent skip here is a child that stays queued forever.
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
        """Build one child and set it running. Shared by admission and readmit.

        `restarts` is how many times this child has already been started. Above
        zero the task is presented with that said, which is the difference
        between a transcript a model can follow and one where the same
        instruction simply appears twice.
        """
        parent = request.parent
        parent_session = self._parent_session(parent)
        prompt = request.prompt.strip()
        provider_name, model, effort = self._resolve_model(request, parent)

        child_session = await self._child_session(parent_session, run_id)
        # Created before the seam writes the admission, so the ways `agents.create`
        # can fail — no driver, no route, options the driver rejects — cannot leave
        # an admitted child behind. It does not *run* yet, so the ordering the log
        # cares about still holds.
        options = replace(
            parent.options,
            provider=provider_name,
            model=model,
            reasoning_effort=effort,
            model_key=child_model_key(request),
        )
        try:
            # `parent=` nests the child's scope inside its parent's (P6-27), so
            # the ceiling is inherited rather than applied.
            child_agent = self.ctx.require(AGENTS).create(child_session, options, parent=parent)
        except Exception as error:
            self.ctx.require(SESSIONS).dispose(child_session.id)
            raise SubagentSpawnError(f"the child agent could not be created: {error}") from error

        try:
            granted, downgrade = await self._workspace(
                parent, child_agent, child_session, request.access
            )
        except InactiveScopeError:
            # The parent went away while this child was being built — its scope,
            # and the child's agent inside it, were disposed under the await.
            # Nothing names the child yet, so its session is let go rather than
            # tombstoned; the seam turns the scope error into the refusal.
            self.ctx.require(SESSIONS).dispose(child_session.id)
            raise

        run = SubagentRun(
            id=run_id,
            name=name,
            session_id=child_session.id,
            parent_id=parent.id,
            model_provider=provider_name,
            model=model,
            requested_access=request.access,
            granted_access=granted,
            downgrade_reason=downgrade,
            # Handed back so the *seam* can bound this child (P4-13b). The scope
            # is the provider's to create and the ceiling is not the provider's
            # to remember — `rehydrate` below makes a second one, and forgot.
            scope=child_agent.ctx,
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

        # **Presented again, and that is what makes a retry real.** Starting a
        # step claims the task from the inbox, and the claim is a logged splice —
        # so a resumed child whose task was not re-presented finds an empty inbox,
        # ends at step zero, and reports `completed` for work it never did.
        # **Unless it is still there** (S3): the inbox gives up a batch only in the
        # step that records it, so a child stopped before its first step has its
        # task pending in its own log, and a second copy would be the same work
        # handed over twice.
        if not _task_pending(child_agent):
            child_agent.followup(
                create_user_message(
                    content=[{"type": "text", "text": _task_text(prompt, restarts)}],
                    source=_TASK_SOURCE,
                )
            )
        try:
            # The child is an artifact of the *parent's* scope (I2), so a disposed
            # parent unwinds its children instead of leaving them running with nobody
            # to answer — and `delete()` becomes "release it early" rather than a
            # second cleanup path that has to remember everything.
            run.dispose = await parent.ctx.effect(
                lambda: partial(self._release, run_id, PARENT_TEARDOWN),
                label=f"subagent:{run_id}",
            )
            # `resumed` is what makes a restart countable: without it the log holds
            # a `running` record per start and no way to tell a first one from a
            # fourth, which is the fold the ladder needs. A parent torn down under the
            # effect's await refuses here too: the job is registered on its scope.
            await self._attach(child, parent, cause="resumed" if restarts else None)
        except InactiveScopeError:
            # Ended rather than left live: the release a disposed parent runs, run now.
            # Idempotent, so the parent's own teardown reaching it as well does
            # nothing twice.
            await self._release(run_id, PARENT_TEARDOWN)
            raise
        return run

    async def _child_session(self, parent_session: Session, run_id: str) -> Session:
        """This child's session — claimed, then resumed when a log for it survived,
        else fresh — through `open_child_log`, the seam's door every child is opened
        by, which names and files it the way its parent finds it.

        A readmitted child usually has no log at all: it never ran a turn, and what
        little its session held was still in a buffer when the daemon stopped. But a
        child that *did* reach disk must be resumed rather than recreated, for the
        reason `open_session` gives — creating over a stored log puts `seq`
        backwards mid-file and the log stops being readable at all (P5-03).

        **Claimed, as a root is (I-5, L2).** A child's log is a session of its own,
        and since Phase 11 its only record — so it has one writer, the mount of the
        root that spawned it, and no host opens it as a root (P11-08). The lease is
        what keeps a *second process* out: another daemon that mounted the same
        root, say. It is held by the mount's scope, like the root's (`open_session`
        claims on `ctx.root`): the child's session lives in the deployment's store
        for as long as the root is mounted, and the mount's last act writes it
        before the lease is given back. A child another process holds refuses with
        `SessionBusy`, which a readmission settles as one it could not resume.
        """
        return await open_child_log(self.ctx, parent_session, run_id, agentPreset="rlm")

    def _resolve_model(
        self, request: SubagentRequest, parent: AgentHandle
    ) -> tuple[str, str, str | None]:
        """The child's model, with **no fallback** on an explicit selector.

        Falling back would answer the parent's question on a model it did not
        choose, and the parent has no way to tell from the reply.
        """
        options = parent.options
        provider_name, model = child_route(request)
        if not provider_name or not model:
            raise SubagentSpawnError(
                "a subagent needs a provider and a model; the parent has none to inherit"
            )
        # The preflight that exists: the route must resolve to an adapter. There
        # is no model *catalog* on `ctx.llm` yet — `rlm.find_models` needs one
        # and will bring it — so an unknown model name is caught at its first
        # request rather than at admission. What matters either way is that
        # nothing substitutes a different model.
        try:
            self.ctx.require(LLM).adapter_for(provider_name)
        except LlmError as error:
            raise SubagentSpawnError(
                f'provider "{provider_name}" has no registered adapter, so a child cannot '
                f"run on it: {error}"
            ) from error
        return provider_name, model, request.reasoning_effort or options.reasoning_effort

    # ------------------------------------------------------------- lifecycle --

    async def _attach(
        self, child: _Child, parent: AgentHandle, *, cause: StatusCause | None = None
    ) -> None:
        """Wire a live child to its parent and start driving it.

        One path for a fresh admission and for a rehydration, so the two cannot
        attach different things: the job, and the `cause` its own log records beside
        `running`.
        """
        assert child.session is not None
        limit = self.config.max_concurrent
        # `ctx.jobs`, which detaches rather than running inline: the job gives the
        # run an id, a cancel and `job/*` events for free, and a subagent is the
        # seam's own example of work that outlives the step that started it.
        job = await self.ctx.require(JOBS).start(
            kind="subagent",
            label=f"{child.run.name} ({child.run.id})",
            run=lambda _job: self._drive(child, cause=cause),
            # The delegation's lifetime, which is the parent's: a disposed parent
            # abandons the drive, and a settled child releases its own entry so a
            # chatty exchange does not leave one job per message behind.
            scope=parent.ctx,
            # The queue is the seam's (`Config.max_concurrent`). Keyed by the
            # *parent's* session, so a cap bounds one agent's fan-out and one
            # parent's children cannot starve another root's — and the wait, the
            # release on every ending and the queue's own lifetime are the seam's
            # to get right rather than this provider's to re-derive.
            slot=None if limit is None else (self._parent_session(parent).id, limit),
            # Written only when there was actually a wait. An admitted child with
            # no status already reads as `queued`; this is the record for the case
            # where that is true *for a reason*.
            on_queued=lambda: self._waiting(child, slots=limit),
        )
        child.job_id = job.id

    async def rehydrate(self, run_id: str) -> bool:
        """Give a settled child a runtime again so it can be addressed (P3-13).

        The child's session and its log survived settlement — what `_quiesce`
        released was the agent, which is what holds an inbox. So rehydration
        re-creates the agent against the same session and drives it again; the
        `Inbox` rebuilds itself from `agent/inbox/spliced` in that log, so anything
        queued before it settled is still there.

        A *deleted* child is not rehydrated: the tombstone in its log records that its
        parent revoked it, and quietly reviving it would make that record false.
        Passivation across a restart — where the session itself is gone and has to
        come off disk — is the daemon's (Phase 5).
        """
        child = self._children.get(run_id)
        if child is None or child.agent is not None:
            # Unknown or already running. A *revoked* child never reaches here:
            # `_release` both pops `_children` and calls `forget()`, so the
            # service's own lookup refuses it first.
            return False
        session = self.ctx.require(SESSIONS).get(child.run.session_id)
        if session is None:
            log.debug("ph_rlm.subagents: %s has no live session to rehydrate", run_id)
            return False
        child.session = session
        # **Before anything is built.** The parent owns the drive job *and*, since
        # P6-27, the scope the child nests in — so a missing parent is a refusal,
        # not a degradation. Below `agents.create` this guard would leave an agent
        # and a scope behind: an orphan under the registry root, holding the
        # deployment-wide ceiling, that nothing would ever dispose.
        parent = self.ctx.require(AGENTS).get(child.run.parent_id)
        if parent is None:
            log.debug("ph_rlm.subagents: %s has no live parent to own its drive", run_id)
            return False
        child.agent = self.ctx.require(AGENTS).create(session, child.options, parent=parent)
        await self._runtime(child, parent, session)
        # A fresh gate, which the awaiter already on the run reads at await time —
        # its closure holds the `_Child`, not the old Event.
        child.finished = anyio.Event()
        await self._attach(child, parent, cause="rehydrated")
        return True

    async def _runtime(self, child: _Child, parent: AgentHandle, session: Session) -> None:
        """Everything a child needs re-established around a fresh agent.

        **One list, because it has been discovered twice.** A child's agent is not
        finished when `agents.create` returns: its scope must be handed back, its
        ceiling re-applied, and its workspace re-taken — each from what the
        admission recorded. `rehydrate` grew those late and one at a time; the
        spawn path's own comment says so about the second (*"the ceiling is not
        the provider's to remember — `rehydrate` below makes a second one, and
        forgot"*), and the third arrived the same way. Neither omission failed
        loudly: a child came back holding the whole deployment, and later one came
        back writing into a tree that was not its own.

        Its work comes back with it, and that falls out of the tier rather than
        being arranged here: disposal commits the checkout to the child's branch
        before removing it, and `_add` attaches an existing branch rather than
        resetting one. The same run id resolves to the same branch, so a second
        question starts where the first stopped.
        """
        # A fresh scope means a fresh ceiling: the filters applied at admission
        # were disposed with the scope that settled (P4-13b).
        agent = child.agent
        assert agent is not None, "a child is re-asked only after it has an agent"
        child.run.scope = agent.ctx
        if child.run.grant is not None:
            child.run.grant.apply(self.ctx, agent.ctx)
        # The access the **admission** recorded, never a caller's: nothing about
        # being asked a second question may widen what the first was allowed.
        await self._workspace(parent, agent, session, child.run.requested_access)

    def _awaiter(self, child: _Child) -> SubagentAwaiter:
        async def wait() -> SubagentResult:
            await child.finished.wait()
            return child.result or SubagentResult(status="error", error="the child never settled")

        return wait

    def _child_log(self, child: _Child) -> Session | None:
        """The child's own log, whether or not it still has an agent.

        `child.session` goes with the agent when a child settles (`_quiesce`), while
        the log itself stays live in the store for the mount's life — and is where a
        rehydration, a revocation or a suspension records what happened to it. `get`,
        for `_release`'s reason: on a mount's unwind the store may be gone.
        """
        if child.session is not None:
            return child.session
        sessions = self.ctx.get(SESSIONS)
        return sessions.get(child.run.session_id) if sessions is not None else None

    def _waiting(self, child: _Child, **extra: JsonValue) -> None:
        """`queued`, in the child's own log, when it has one to write to."""
        own = self._child_log(child)
        if own is not None:
            record_waiting(own, **extra)

    async def _drive(self, child: _Child, *, cause: StatusCause | None) -> None:
        """Run the child to quiescence, tell the parent, then let it go."""
        run = child.run
        own = child.session
        assert own is not None, "a child is driven only while it has its own log"
        try:
            # Not before the seam says so (`SubagentRun.ready`): bounded, and for a
            # readmitted child its own children swept, so neither its ceiling nor
            # the children its first prompt shows are still being worked out.
            await run.ready.wait()
            # **Once released, the release owns the ending** — here, at the gate, and
            # after the run below: revoked, refused or suspended, the child's ending
            # is already written, and the drive's would overwrite it. A job past its
            # slot is one `Job.cancel` no longer reaches, so the check is the drive's.
            if child.finished.is_set():
                return
            # `running` either way; `cause` says *why* it is running, because a
            # child's state folds status last-write-wins and a woken child that is
            # working must not read as not-running. A restart is on its own disk
            # before the attempt it counts (S10) — the seam's door keeps that rule.
            await record_started(self.ctx, own, cause=cause)
            agent = child.agent
            assert agent is not None, "a child runs only after it has an agent"
            await agent.run()
            if child.finished.is_set():
                return
            answer = _last_assistant_text(own)
            child.result = SubagentResult(status="done", answer=answer)
            # Silence is indistinguishable from a hang from the parent's side,
            # so a child that never sent a message is announced. A child that
            # *did* reply needs no notice — the reply is the notice — and its own
            # log says whether it did, across a restart too.
            tail = f" Last assistant text: {answer}" if answer else ""
            silent = ChildNotice(
                text=f"[rlm child {run.name} ({run.id}) completed without sending a reply.{tail}]",
                summary=f"{run.name} finished without replying",
            )
            # The child's ending is on its own disk before its parent is handed the
            # answer (F1): `record_ended` flushes, and the waiters wake after it. The
            # notice rides on the ending, so a restart delivers one a crash held back.
            await record_ended(
                self.ctx,
                own,
                "done",
                notice=None if replied_to_parent(own) else silent,
                answerPreview=answer[: self.config.answer_preview_chars] or None,
            )
            # The other half of retain-by-default (P6-28): a child that finished
            # is a child whose checkout is not evidence of anything, so the mark
            # taken at acquire is withdrawn and `worktree-ephemeral` keeps its
            # promise for the case it was written for. Before `_quiesce` in the
            # `finally` below, which disposes the scope that holds the workspace
            # — after it there is nothing left to unmark.
            self._withdraw(child)
        except Exception as error:
            if child.finished.is_set():
                return
            message = f"{type(error).__name__}: {error}"
            child.result = SubagentResult(status="error", error=message)
            # The same order for a failure: the child's account of it first.
            await record_ended(
                self.ctx,
                own,
                "error",
                notice=ChildNotice(
                    text=f"[rlm child {run.name} ({run.id}) failed: {message}"
                    f"{self._evidence(child)}]",
                    summary=f"{run.name} failed",
                ),
                detail=message,
            )
            log.debug("ph_rlm.subagents: child %s failed", run.id, exc_info=True)
        finally:
            child.finished.set()
            # A settled child holds an agent scope, and that scope owns the
            # child's kernel subprocess (`code-runtime:<namespace>` is an effect
            # of it). Keeping it for the host's lifetime leaked one CPython — and
            # the child's whole namespace — per delegation.
            await self._quiesce(child)

    def _evidence(self, child: _Child) -> str:
        """Where the failed child's tree is, for the message that says it failed.

        **The row's motivating case, answered where the parent is looking**
        (P6-28). Retention keeps the checkout; this is what stops it being
        evidence nobody can find. A child's workspace events are in the *child's*
        log, so a parent told only "it failed" is left diagnosing from a
        transcript — which is the sentence the row opens with.

        Read from the log rather than from the seam, and that is not incidental:
        this runs on the way to `_quiesce`, and the same sentence has to be true
        on a path where the scope is already gone. The fold answers from events
        that are already written.

        Silent when there is nothing to say — no tier, a `shared` workspace, a
        child whose tree was discarded. A notice that named a directory for every
        failure would be naming ones that are not there.
        """
        if child.session is None:
            return ""
        kept = [one for one in workspace_survivors(child.session) if one.outcome == "retained"]
        return f" Its workspace is kept at {kept[0].root}." if kept else ""

    def _withdraw(self, child: _Child) -> None:
        """Take back the mark this child's tree got at acquire (P6-28).

        Only ever the withdrawal, which is why it takes no reason: the marking
        half runs in `_workspace`, where the seam and the kind are already in
        hand. A shared helper with a `reason` parameter that one caller always
        passed `""` was two spellings of one operation.

        `get`, not attribute access, and tolerant of an agent that never took a
        workspace: this runs on a settle path, and a profile with no workspace
        row or a child whose tier declined are both ordinary. The seam answers
        `False` rather than raising for exactly this caller.
        """
        seam = self.ctx.get(WORKSPACE)
        if seam is not None and child.agent is not None:
            seam.retain(child.agent.id, "")

    async def _quiesce(self, child: _Child) -> None:
        """Drop everything a settled child no longer needs.

        The terminal `result` stays, so a caller awaiting `result()` after the
        child is gone still gets its answer. Failures are contained: this runs in
        a `finally`, and a teardown that raised would replace a settled child's
        outcome with a disposal error.
        """
        # `get`, for `_release`'s reason: a mount's unwind reaches here after the
        # rows that provide these have gone, and there is nothing left to release.
        jobs = self.ctx.get(JOBS)
        if child.job_id is not None and jobs is not None:
            # Released, not abandoned: the work finished, so the entry goes
            # without the job being reported as canceled.
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
        """Take the child's workspace, and report what it actually got (D21, E3).

        Takes the `access` rather than the request, because the other caller has
        no request: a child being given its runtime back has only what its
        admission recorded, and that is exactly what it should come back with.

        **Acquired here rather than left to the lifecycle row**, because a
        child's base and access are its *parent's* decision and the row knows
        neither: it would hand the child a `write` workspace over the process's
        own directory, which is both the wrong tree and the wrong guarantee.
        Branching from the parent's root is what makes a fan-out land on
        sibling branches instead of one shared checkout (E2).

        `granted` is access **to the project**, not to a directory. A
        `worktree-ephemeral` child may write its checkout freely and none of it
        is ever merged, so what it was granted of the project is `read` — the
        seam's own reading of `repo_writable`, applied one level up.
        """
        seam = self.ctx.get(WORKSPACE)
        if seam is None:
            # A profile with no workspace row at all. The conservative claim is
            # the only honest one: nothing here can enforce a writable repo, so
            # nothing promises one.
            return "read", "workspace-not-mounted"
        workspace = await seam.acquire(
            session_id=child_session.id,
            agent_id=child_agent.id,
            # Asked of `ctx.fs` rather than re-derived from the parent's
            # workspace: "where does this agent's relative path land" already has
            # one implementation, and a second one in this package is the one
            # that must not disagree with it.
            base=self.ctx.require(FS).root_for(parent),
            access=access,
            session=child_session,
            # The child's own scope: a revoked or finished child releases its
            # checkout with everything else it took (I2).
            scope=child_agent.ctx,
        )
        if discards_writes(workspace.kind):
            # **Retained from the moment the tree exists** (P6-28), because the
            # window to mark it closes with the child's scope and the outcomes
            # that most need the evidence are the ones that close it first: on
            # the `parent-teardown` path the worktree is released *before*
            # `_release` runs, which that method's own docstring states. So this
            # cannot be "retain when it fails" — there is nowhere to put that —
            # and it has to be "retain unless it succeeds", cleared in `_drive`.
            #
            # Only for the kind that discards. Every other kind already keeps a
            # dirty tree for review, and a committed branch survives release
            # regardless, so retaining those would grow the pile without saving
            # anything from it.
            #
            # The pile this creates is bounded by `phern workspaces gc`, which is
            # not a note about future work: this policy was held back until that
            # command existed, because inverting `worktree-ephemeral`'s promise
            # with no collector is a worse trade than the lost evidence it fixes.
            seam.retain(child_agent.id, "the child has not settled cleanly")
        return project_access(workspace.kind), None

    async def revoke(self, run_id: str, reason: str) -> bool:
        """`RevokingProvider`: stop one child this provider holds, with a tombstone in
        its own log. `False` for a child it does not hold.

        Asked by `SubagentService.delete`, the door every revocation goes through: a
        child this provider is not running — settled by an earlier process — is the
        seam's to tombstone.
        """
        return await self._release(run_id, reason)

    async def _release(self, run_id: str, reason: str) -> bool:
        """The one revocation path, whether the model asked or the parent unwound.

        **On the parent-teardown path the child's scope is already gone** (P6-27).
        `Context.dispose` unwinds `_children` before its own effects, and this runs as one
        of those effects — so the child's scope, its worktree and its kernel are released
        *before* this is called.

        Nothing here breaks on that: `cancel` touches only the phase and the inbox, and
        disposing an inactive scope returns at once. But it **bounds what this path may
        do**: the child's log and its tombstone are the store's, not the child scope's,
        and live; anything needing the *child's* scope — its own services, its
        workspace — is not, and would work when the model calls `delete()` and fail
        here.

        **Only ever a revocation.** A mount going away reaches its children first, through
        `suspend`, and leaves nothing here to release; so a parent's teardown that does
        reach a child is one the mount lives on past — a settled child's own children, a
        parent `ctx.agents` disposed — and those children really are revoked.

        **A settled child is not settled again.** Its drive wrote `done` or `error`, and a
        `canceled` written over that turned a finished child into a revoked one.
        """
        child = self._children.pop(run_id, None)
        if child is None:
            return False
        if child.job_id is not None:
            # Queued, never ran: `Job.cancel` stops the wait for a slot it will
            # now never take. A body parked on a limiter reads no token, so
            # without this a revoked child stays queued behind children that may
            # never settle and `ctx.drain()` waits for it — a revocation that
            # hangs the teardown. Harmless for a child already running: its agent
            # is what stops that one, one line down.
            self.ctx.require(JOBS).cancel(child.job_id)
        if child.agent is not None:
            child.agent.cancel(AgentCancelCause(kind="parent"))
        # The ending and the tombstone together (S14), in the child's own log and on
        # its disk before anything is let go — and the ending at all, because a
        # revoked child is not merely absent: a panel that knew only `deleted` could
        # not say whether it had run.
        own = self._child_log(child)
        if own is not None:
            await record_deleted(self.ctx, own, reason)
        self._let_go(child)
        await self._quiesce(child)
        # `get`, not attribute access: on the parent-teardown path this runs while
        # scopes are unwinding, and the seam's own provision may already be gone —
        # a teardown that raised would abort the rest of the unwind.
        registry = self.ctx.get(SUBAGENTS)
        if registry is not None:
            registry.forget(run_id)
        return True

    def suspend(self) -> None:
        """Stop every child this provider holds, and revoke none of them (S1).

        **This row's own disposer.** The row injects every row it uses, so it activates
        after them and unwinds before them — before `jobs`, `subagents`, and the `agent`
        row whose scopes hold every parent. A mount going away (a daemon stopping, a root
        remounted on a new profile) reaches here with all of that still up, and each
        parent's own effect then finds nothing left for `_release`. Before this, a mount's
        unwind reached its children only through those effects, after `jobs` had gone: a
        working child's release raised before it wrote anything, so the next start read
        it as a crash and spent a rung of its ladder, and a settled child was tombstoned.

        A suspended child keeps a live row — `queued`, saying why, and no tombstone — and
        its inbox. Its agent is stopped keeping the inbox, and let go of rather than
        left for the drive's teardown to dispose — disposing an agent clears what is
        queued for it, and a drive whose model call honors the cancel wakes while this
        row can still reach the registry. The mount's unwind takes the scope. The resume
        sweep readmits it, and whoever waited on it is answered `queued` with the same
        reason. A settled child is left as it ended.

        `get` rather than `require` for the one case this does not cover: the row
        deactivating while the mount lives on, because a row it injects went away.
        """
        jobs, registry = self.ctx.get(JOBS), self.ctx.get(SUBAGENTS)
        for child in self._children.values():
            if child.result is None:
                self._waiting(child, detail=SUSPENDED_DETAIL)
                child.result = SubagentResult(status="queued", error=SUSPENDED_DETAIL)
            if child.job_id is not None and jobs is not None:
                jobs.cancel(child.job_id)
            if child.agent is not None:
                child.agent.cancel(AgentCancelCause(kind="parent"), keep_inbox=True)
                child.agent = None
            self._let_go(child)
            if registry is not None:
                registry.forget(child.run.id)
        self._children.clear()

    @staticmethod
    def _let_go(child: _Child) -> None:
        """Mark a child ended for everyone waiting on it: the caller awaiting its
        `result()`, and a drive still parked at the seam's gate — a child refused, or
        released before its admission was written — which is let through to find itself
        released (`_drive`) rather than left on an event nobody would set: `Job.cancel`
        trips a token that `ready.wait()` never reads."""
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
    """The task as the child reads it — and, after an interruption, why twice.

    A resumed child's transcript already holds the turn its harness cut short,
    closed by the resume's own repair. Re-presenting the task without a word
    about that shows a model the same instruction twice and lets it conclude it
    already answered; naming the interruption is what makes the second attempt
    legible as one.

    The number is how many times this child has been started, not where it sits
    on the harness's ladder — the two part company the moment progress clears the
    ladder, and the model's own history is the honest one to tell it about.
    """
    if restarts < 1:
        return f"{TASK_PREFIX}\n\n{prompt}"
    return (
        f"{TASK_PREFIX}\n\n{prompt}\n\n[the harness stopped while you were working on "
        f"this, so the turn above was cut short; this is attempt {restarts + 1}. "
        "Anything you finished is in your transcript — continue from there rather "
        "than starting over.]"
    )


def _assistant_text(event: SessionEvent) -> str | None:
    """One event's assistant text, or `None` when it carries none.

    Through `derive_event_message` + `text_of` rather than reaching into the event
    payload: those two own the rules for what an event projects to and which
    blocks carry text, and a hand-rolled copy here would go quietly wrong the day
    a new text-bearing block type lands.

    Empty text answers `None` — the same as a tool-only turn — so the fold keeps
    scanning back to the last turn that actually said something.
    """
    message = derive_event_message(event)
    if message is None or message.role != "assistant":
        return None
    return text_of(message.content).strip() or None


def _last_assistant_text(session: Session | None) -> str:
    """The child's last non-empty assistant text — its answer, by convention.

    An incremental fold rather than a reverse walk of `session.events`, which
    materialized a snapshot of the child's whole log to read one turn — and this
    is asked once per child, at the moment the log is longest.

    Scoped to `assistant/message`, which is the only event type
    `derive_event_message` projects into the assistant role; the role check stays
    in the parser because that is the rule being relied on, not an assumption
    about the type.
    """
    if session is None:
        return ""
    return session.projection("assistant/message", _assistant_text) or ""


@plugin(
    "rlm-subagent-provider",
    affects="environment",
    config=Config,
    inject=[SUBAGENTS, AGENTS, SESSIONS, JOBS, LLM],
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the `rlm-child` provider and expose it for the bindings row."""
    provider = RlmChildProvider(ctx=ctx, config=config)
    ctx.require(SUBAGENTS).register_provider(PROVIDER_NAME, provider)
    ctx.provide(RLM_CHILDREN, provider)
    # Registered after the provider, so it runs before the provider is withdrawn.
    ctx.add_disposer(provider.suspend, label="rlm-children.suspend")
