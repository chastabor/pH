"""`rlm-subagent-provider` — `rlm()` as a `ctx.subagents` provider (P3-11).

Ported from prime-agent's `AgentSession._startRlmChildRun`. The semantics are
its; the mechanism is pH's seams. Four properties are the whole design, and each
one is a thing the obvious implementation gets wrong:

**The handle returns before the child answers.** Stated once, in
`ph.seams.subagents` — this file's job is to keep the promise. `start()` creates
the session and the agent, appends `subagent/admitted`, starts the job, and
returns; the child's reply arrives on a later turn as an ordinary inbox message.

**Admission is logged before anything runs.** The agent is *created* first, so a
`create` failure cannot leave a phantom child in an append-only roster, but it is
not started until after the record exists — so no status event can precede the
record of the child it describes.

**A child is an artifact of its parent's scope.** Acquired through
`parent.ctx.effect()`, so a disposed parent unwinds its children (I2) and
`delete()` is just "release it early". Without that, a settled child kept its
agent scope alive for the host's lifetime — and that scope owns the child's
kernel subprocess, so every delegation leaked a CPython.

**Usage is attributed, not double-counted.** Each child `assistant/message`
appends `subagent/usage-attributed` to the *parent's* log, so the token meter can
subtract a child's tokens from the parent's own context measurement while billing
totals still include them. Without it a fan-out of eight reads as context
pressure on the parent and triggers a compaction it does not need.

**No workspace yet.** `granted` is `read` until `ctx.workspace` lands (Phase 4,
D21) — a child cannot be handed a guarantee nothing enforces — and the *reason*
travels as a code, not prose, so the log stays parseable once the tier exists.

@module ph_rlm.subagents
"""

from __future__ import annotations

import logging
import secrets
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import anyio

from ph.agent.types import AgentCancelCause, AgentOptions
from ph.cordis import Context, Disposer, plugin
from ph.llm.adapter import LlmError
from ph.llm.types import CONTEXT_SUMMARY_MAX_CHARS, PluginSource, create_user_message, text_of
from ph.seams.subagents import (
    ADMITTED,
    DELETED,
    STATUS,
    USAGE,
    Access,
    DowngradeReason,
    StatusCause,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentSpawnError,
    SubagentStatus,
    default_child_name,
)
from ph.session import Session, derive_event_message
from ph.session.json import thaw_json
from ph.wire import WireModel

__all__ = [
    "MAX_NAME_CHARS",
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
MAX_NAME_CHARS = 64
TASK_PREFIX = "[task from parent]"
"""Ported verbatim: the child's prompt recognizes this label."""


class Config(WireModel):
    """Row config for `rlm-subagent-provider`."""

    max_depth: int = RLM_MAX_DEPTH
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
    run, the outcome, the parent's log and the options to rebuild the agent with.
    """

    run: SubagentRun
    finished: anyio.Event
    parent_session: Session
    """The log the child reports to — the single source for every append about
    this child, so a rehydration and a tombstone cannot land in different logs."""
    options: AgentOptions
    """The child's resolved options, because at rehydration time the parent agent
    may be gone and `reasoning_effort` survives nowhere else."""
    agent: Any = None
    session: Session | None = None
    unobserve: Disposer | None = None
    job_id: str | None = None
    """The drive job, owned by the *parent's* scope. Not the child's: the child's
    scope is disposed *by* the drive job's own last act, and a job that abandoned
    itself would report `cancelled` for work that finished."""
    result: SubagentResult | None = None
    replied: bool = False
    """Whether the child sent its parent a message (`rlm-messaging` sets it).

    Held here rather than folded, because it decides whether the terminal notice
    fires — a decision made at completion, in this process, about a child this
    process ran."""


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

    async def start(self, request: SubagentRequest) -> SubagentRun:
        """Admit a child. Returns before it answers (the whole point)."""
        parent = request.parent
        parent_session: Session = parent.session
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
        taken = [str(row.get("name")) for row in self.ctx.subagents.roster(parent_session).values()]
        name = self._resolve_name(request.name, prompt, run_id, taken)
        provider_name, model, effort = self._resolve_model(request, parent)

        # `read` until `ctx.workspace` exists (D21): a granted guarantee nothing
        # enforces is worse than an honest refusal to grant it.
        granted: Access = "read"
        downgrade: DowngradeReason | None = (
            "workspace-not-mounted" if request.access != granted else None
        )

        child_session = self.ctx.sessions.create(
            f"{parent_session.id}-{run_id}",
            meta={
                "parentSession": parent_session.id,
                "origin": "subagent",
                "delegationDepth": depth + 1,
                "agentPreset": "rlm",
            },
        )
        # Created before the admission is appended, so the ways `agents.create`
        # can fail — no driver, no route, options the driver rejects — cannot
        # leave a phantom child in an append-only roster. It does not *run* yet,
        # so the ordering the log cares about still holds.
        options = replace(
            parent.options, provider=provider_name, model=model, reasoning_effort=effort
        )
        try:
            child_agent = self.ctx.agents.create(child_session, options)
        except Exception as error:
            self.ctx.sessions.dispose(child_session.id)
            raise SubagentSpawnError(f"the child agent could not be created: {error}") from error

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
        )
        child = _Child(
            run=run,
            finished=anyio.Event(),
            parent_session=parent_session,
            options=options,
            agent=child_agent,
            session=child_session,
        )
        self._children[run_id] = child
        run.result = self._awaiter(child)

        parent_session.append(ADMITTED, {**run.to_wire(), "prompt": prompt})

        # The child is an artifact of the *parent's* scope (I2), so a disposed
        # parent unwinds its children instead of leaving them running with nobody
        # to answer — and `delete()` becomes "release it early" rather than a
        # second cleanup path that has to remember everything.
        run.dispose = await parent.ctx.effect(
            lambda: partial(self._release, parent_session, run_id, "parent-teardown"),
            label=f"subagent:{run_id}",
        )
        child_agent.followup(
            create_user_message(
                content=[{"type": "text", "text": f"{TASK_PREFIX}\n\n{prompt}"}],
                source=PluginSource(plugin="ph_rlm.subagents", form="relay"),
            )
        )
        await self._attach(child, parent)
        return run

    def _resolve_name(
        self, requested: str | None, prompt: str, run_id: str, taken: list[str]
    ) -> str:
        if requested is None:
            return default_child_name(prompt, run_id, taken=taken)
        name = requested.strip()
        if not name or len(name) > MAX_NAME_CHARS:
            raise SubagentSpawnError(f"a subagent name must be 1..{MAX_NAME_CHARS} characters")
        if name in taken:
            raise SubagentSpawnError(
                f'a sibling is already named "{name}"; names address children, so they '
                "must be unique among siblings"
            )
        return name

    def _resolve_model(self, request: SubagentRequest, parent: Any) -> tuple[str, str, str | None]:
        """The child's model, with **no fallback** on an explicit selector.

        Falling back would answer the parent's question on a model it did not
        choose, and the parent has no way to tell from the reply.
        """
        options = parent.options
        provider_name = request.provider or options.provider
        model = request.model or options.model
        if not provider_name or not model:
            raise SubagentSpawnError(
                "a subagent needs a provider and a model; the parent has none to inherit"
            )
        # The preflight that exists: the route must resolve to an adapter. There
        # is no model *catalogue* on `ctx.llm` yet — `rlm.find_models` needs one
        # and will bring it — so an unknown model name is caught at its first
        # request rather than at admission. What matters either way is that
        # nothing substitutes a different model.
        try:
            self.ctx.llm.adapter_for(provider_name)
        except LlmError as error:
            raise SubagentSpawnError(
                f'provider "{provider_name}" has no registered adapter, so a child cannot '
                f"run on it: {error}"
            ) from error
        return provider_name, model, request.reasoning_effort or options.reasoning_effort

    # ------------------------------------------------------------- lifecycle --

    async def _attach(
        self, child: _Child, parent: Any, *, cause: StatusCause | None = None
    ) -> None:
        """Wire a live child to its parent and start driving it.

        One path for a fresh admission and for a rehydration, so the two cannot
        attach different things: the usage mirror, the job, and the `cause` the
        roster shows beside `running`.
        """
        assert child.session is not None
        child.unobserve = child.session.observe(self._mirror(child))
        # `ctx.jobs`, which detaches rather than running inline: the job gives the
        # run an id, a cancel and `job/*` events for free, and a subagent is the
        # seam's own example of work that outlives the step that started it.
        job = await self.ctx.jobs.start(
            kind="subagent",
            label=f"{child.run.name} ({child.run.id})",
            run=lambda _job: self._drive(child, cause=cause),
            # The delegation's lifetime, which is the parent's: a disposed parent
            # abandons the drive, and a settled child releases its own entry so a
            # chatty exchange does not leave one job per message behind.
            scope=parent.ctx,
        )
        child.job_id = job.id

    async def rehydrate(self, run_id: str) -> bool:
        """Give a settled child a runtime again so it can be addressed (P3-13).

        The child's session, log and roster row all survived settlement — what
        `_quiesce` released was the agent, which is what holds an inbox. So
        rehydration re-creates the agent against the same session and drives it
        again; the `Inbox` rebuilds itself from `agent/inbox/spliced` in that log,
        so anything queued before it settled is still there.

        A *deleted* child is not rehydrated: the tombstone is the parent's record
        that it revoked the child, and quietly reviving it would make that record
        false. Passivation across a restart — where the session itself is gone
        and has to come off disk — is the daemon's (Phase 5).
        """
        child = self._children.get(run_id)
        if child is None or child.agent is not None:
            # Unknown or already running. A *revoked* child never reaches here:
            # `_release` both pops `_children` and calls `forget()`, so the
            # service's own lookup refuses it first.
            return False
        session = self.ctx.sessions.get(child.run.session_id)
        if session is None:
            log.debug("ph_rlm.subagents: %s has no live session to rehydrate", run_id)
            return False
        child.session = session
        child.agent = self.ctx.agents.create(session, child.options)
        # A fresh gate, which the awaiter already on the run reads at await time —
        # its closure holds the `_Child`, not the old Event.
        child.finished = anyio.Event()
        parent = self.ctx.agents.get(child.run.parent_id)
        if parent is None:
            log.debug("ph_rlm.subagents: %s has no live parent to own its drive", run_id)
            return False
        await self._attach(child, parent, cause="rehydrated")
        return True

    def _awaiter(self, child: _Child) -> Any:
        async def wait() -> SubagentResult:
            await child.finished.wait()
            return child.result or SubagentResult(status="error", error="the child never settled")

        return wait

    def _mirror(self, child: _Child) -> Any:
        """Attribute the child's usage to the parent as it is produced."""
        parent_session, run_id = child.parent_session, child.run.id

        def observer(_source: Session, event: Any) -> None:
            if event.type != "assistant/message":
                return
            usage = event.data.get("usage")
            # `Mapping`, not `dict`: a committed event's data is frozen into
            # `MappingProxyType`, which is a Mapping and is *not* a dict
            # instance — so an `isinstance(..., dict)` guard here silently
            # attributed nothing at all.
            if not isinstance(usage, Mapping):
                return
            parent_session.append(
                USAGE,
                {
                    "runId": run_id,
                    "targetSeq": event.seq,
                    "childUsage": thaw_json(usage),
                    "origin": "spawn_task",
                },
            )

        return observer

    def _status(self, child: _Child, status: SubagentStatus, **extra: Any) -> None:
        # Only the event, and only from the child's own parent log. A copy on the
        # handle would be a second source of truth for a fact the roster folds,
        # frozen at the last in-process update.
        child.parent_session.append(STATUS, {"runId": child.run.id, "status": status, **extra})

    async def _drive(self, child: _Child, *, cause: StatusCause | None) -> None:
        """Run the child to quiescence, tell the parent, then let it go."""
        run = child.run
        parent_session = child.parent_session
        try:
            # `running` either way; `cause` says *why* it is running, because the
            # roster folds status last-write-wins and a woken child that is
            # working must not read as not-running.
            self._status(child, "running", **({"cause": cause} if cause else {}))
            await child.agent.run()
            answer = _last_assistant_text(child.session)
            child.result = SubagentResult(status="done", answer=answer)
            self._status(
                child, "done", answerPreview=answer[: self.config.answer_preview_chars] or None
            )
            # Silence is indistinguishable from a hang from the parent's side,
            # so a child that never sent a message is announced. A child that
            # *did* reply needs no notice — the reply is the notice.
            if not child.replied:
                tail = f" Last assistant text: {answer}" if answer else ""
                self._inject(
                    parent_session,
                    f"[rlm child {run.name} ({run.id}) completed without sending a reply.{tail}]",
                    f"{run.name} finished without replying",
                )
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            child.result = SubagentResult(status="error", error=message)
            self._status(child, "error", detail=message)
            self._inject(
                parent_session,
                f"[rlm child {run.name} ({run.id}) failed: {message}]",
                f"{run.name} failed",
            )
            log.debug("ph_rlm.subagents: child %s failed", run.id, exc_info=True)
        finally:
            child.finished.set()
            # A settled child holds an agent scope, and that scope owns the
            # child's kernel subprocess (`code-runtime:<namespace>` is an effect
            # of it). Keeping it for the host's lifetime leaked one CPython — and
            # the child's whole namespace — per delegation.
            await self._quiesce(child)

    async def _quiesce(self, child: _Child) -> None:
        """Drop everything a settled child no longer needs.

        The terminal `result` stays, so a caller awaiting `result()` after the
        child is gone still gets its answer. Failures are contained: this runs in
        a `finally`, and a teardown that raised would replace a settled child's
        outcome with a disposal error.
        """
        if child.unobserve is not None:
            child.unobserve()
            child.unobserve = None
        if child.job_id is not None:
            # Released, not abandoned: the work finished, so the entry goes
            # without the job being reported as cancelled.
            self.ctx.jobs.forget(child.job_id)
            child.job_id = None
        agent, child.agent, child.session = child.agent, None, None
        if agent is None:
            return
        try:
            await self.ctx.agents.dispose(agent.id)
        except Exception:  # pragma: no cover - teardown must not mask an outcome
            log.debug("ph_rlm.subagents: disposing child %s failed", child.run.id, exc_info=True)

    def _inject(self, parent_session: Session, text: str, summary: str) -> None:
        """Put one notice in the parent's inbox — and only if it is still there.

        A parent disposed while a child was running has no inbox to deliver to;
        the child's own log already records what it did, so the notice is dropped
        rather than raising inside a detached task.
        """
        parent = self.ctx.agents.get(parent_session.id)
        if parent is None:
            return
        parent.inject(
            create_user_message(
                content=[{"type": "text", "text": text}],
                source=PluginSource(
                    plugin="ph_rlm.subagents",
                    form="notice",
                    summary=summary[:CONTEXT_SUMMARY_MAX_CHARS],
                ),
            )
        )

    def mark_replied(self, agent_id: str) -> None:
        """Record that a child sent its parent a message (`rlm-messaging`).

        Keyed by the child's *agent* id, which is what a sender knows about
        itself — `_children` is keyed by run id, and deriving one from the other
        would depend on how a session id happens to be composed.
        """
        for child in self._children.values():
            if child.run.session_id == agent_id:
                child.replied = True
                return

    # ---------------------------------------------------------------- delete --

    async def delete(self, parent_session: Session, run_id: str, *, reason: str = "user") -> bool:
        """Revoke one child early. Its transcript stays on disk.

        A tombstone rather than a removal, because the child's log and artifacts
        outlive it: a parent looking for what a revoked child did should find the
        revocation, not a gap.
        """
        return await self._release(parent_session, run_id, reason)

    async def _release(self, parent_session: Session, run_id: str, reason: str) -> bool:
        """The one teardown path, whether the model asked or the parent unwound."""
        child = self._children.pop(run_id, None)
        if child is None:
            return False
        if child.agent is not None:
            child.agent.cancel(AgentCancelCause(kind="parent"))
        # A terminal state for the roster: a revoked child is not merely absent,
        # and a panel that knew only `deleted` could not say whether it had run.
        self._status(child, "cancelled", reason=reason)
        child.finished.set()
        await self._quiesce(child)
        # `get`, not attribute access: on the parent-teardown path this runs while
        # scopes are unwinding, and the seam's own provision may already be gone —
        # a teardown that raised would abort the rest of the unwind.
        registry = self.ctx.get("subagents")
        if registry is not None:
            registry.forget(run_id)
        parent_session.append(DELETED, {"runId": run_id, "reason": reason})
        return True


def _last_assistant_text(session: Session | None) -> str:
    """The child's last non-empty assistant text — its answer, by convention.

    Through `derive_event_message` + `text_of` rather than reaching into the event
    payload: those two own the rules for what an event projects to and which
    blocks carry text, and a hand-rolled copy here would go quietly wrong the day
    a new text-bearing block type lands.
    """
    if session is None:
        return ""
    for event in reversed(session.events):
        message = derive_event_message(event)
        if message is None or message.role != "assistant":
            continue
        if text := text_of(message.content).strip():
            return text
    return ""


@plugin(
    "rlm-subagent-provider",
    config=Config,
    inject=["subagents", "agents", "sessions", "jobs", "llm"],
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the `rlm-child` provider and expose it for the bindings row."""
    provider = RlmChildProvider(ctx=ctx, config=config)
    ctx.subagents.register_provider(PROVIDER_NAME, provider)
    ctx.provide("rlm_children", provider)
