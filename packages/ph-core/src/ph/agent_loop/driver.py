"""`ReactLoopAgent` — one session driven through turn and step boundaries.

The Phase 0 driver implements dsh's lifecycle verbatim **except tool
execution**, which lands with the pipeline in Phase 1 (P1-02). The order below
is the contract, and every stabilization feature in Phase 4 attaches to one of
these seams rather than to the loop itself (D12):

```
turn/start
  ├ inbox.claim               → agent/inbox/claimed
  ├ system_prompt.assemble    → system-prompt/assemble
  ├ agent/pre-step            → reject | enter(messages)
  │    reject         → turn/end{blocked}
  │    enter, empty, first step → turn/end{completed}
  ├ step/start
  │    user/message*          (the claimed batch, surface: append)
  │    agent/request          → LlmCallConfig
  │    request/header         (appended only when it changed — A12)
  │    request/context        (appended only when the route changed)
  │    llm/stream             → assistant/chunk* → assistant/message
  │    agent/request-error    → retry | None
  ├ step/end
  ├ agent/turn-stopping       (a listener objects by steering)
turn/end
```

Two rules are easy to lose in a port and expensive to lose in production:

* every request's `messages` is `session.derive_messages()` — never a
  separately-maintained array (invariant I3, asserted by P0-14);
* `max-tokens` is **sticky** for the turn: a later completed step must not
  downgrade the outcome, or a truncated answer is reported as a clean one.

The driver runs inside the scope `ctx.agents` created for it. Services resolve
through that scope most-specific-first, so `self.ctx.llm` is the global adapter
seam unless something shadowed it for this agent alone.

@module ph.agent_loop.driver
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import anyio

from ..agent.inbox import Inbox, InboxNotifications, InboxTarget
from ..agent.types import (
    AgentCancelCause,
    AgentOptions,
    AgentStatus,
    PreStepDecision,
    PreStepRequest,
    RequestErrorAction,
    RequestFailure,
    RequestProposal,
    TurnEndReason,
)
from ..cordis import Context
from ..llm.adapter import LlmError
from ..llm.assembler import BlockAssembler
from ..llm.types import (
    FinishReason,
    GenerateOptions,
    LlmCallConfig,
    LlmFailure,
    Message,
    PluginSource,
    TokenUsage,
    create_assistant_message,
    create_user_message,
)
from ..session import Session, SurfaceIntent
from ..session.request_header import EpochHeader, RequestContext, canonical_header, header_equals
from ..system_prompt.assembly import (
    AssembleContext,
    PromptAssembly,
    join_context_sections,
    render_context_sections,
    render_prompt,
)

__all__ = ["AgentCancelled", "ReactLoopAgent"]

log = logging.getLogger("ph.agent_loop")


class AgentCancelled(Exception):
    """The active driver was cancelled; carries the cause for `turn/end`."""

    def __init__(self, cause: AgentCancelCause) -> None:
        super().__init__(f"agent cancelled: {cause.kind}")
        self.cause = cause


@dataclass(slots=True)
class _Phase:
    kind: str = "idle"
    turn: int = 0
    step: int = 0
    cancelled: AgentCancelCause | None = None


@dataclass(frozen=True, slots=True)
class _PreparedStep:
    kind: str
    messages: tuple[Message, ...] = ()
    assembly: PromptAssembly | None = None


class ReactLoopAgent:
    """The default agent driver."""

    def __init__(self, ctx: Context, session: Session, options: AgentOptions) -> None:
        self.ctx = ctx
        self.session = session
        self.options = options
        self.id = session.id
        self._phase = _Phase(turn=_last_turn_of(session))
        self._request_header_logged = False
        self._context_snapshot: str | None = None
        self._idle = anyio.Event()
        self._idle.set()
        self.inbox = Inbox(
            session,
            InboxNotifications(
                inserted=lambda message: ctx.emit("agent/inbox/inserted", self, message),
                discarded=lambda message: ctx.emit("agent/inbox/discarded", self, message),
                claimed=lambda message, turn: ctx.emit("agent/inbox/claimed", self, message, turn),
            ),
        )

    # -------------------------------------------------------------- identity --

    def __repr__(self) -> str:
        return f"<ReactLoopAgent {self.id} {self.status}>"

    @property
    def status(self) -> AgentStatus:
        return "idle" if self._phase.kind == "idle" else "running"

    def _set_phase(self, kind: str) -> None:
        previous = self.status
        self._phase.kind = kind
        if self.status != previous:
            self.ctx.emit("agent/status", self, self.status)

    # ----------------------------------------------------------------- inbox --

    def send(self, message: Message, target: InboxTarget, wakeup: bool) -> None:
        # Waking input cannot join an aborted activity, so it starts the next
        # turn. Classified BEFORE the insertion, so a reentrant cancel from a
        # splice observer cannot reclassify it.
        waking_after_abort = (
            wakeup and self._phase.kind != "idle" and self._phase.cancelled is not None
        )
        self.inbox.append("next-turn" if waking_after_abort else target, message)

    def followup(self, message: Message) -> None:
        """Deliver at the next turn boundary and wake the agent."""
        self.send(message, "next-turn", True)

    def steer(self, message: Message) -> None:
        """Deliver at the next step boundary and wake the agent."""
        self.send(message, "next-step", True)

    def inject(self, message: Message) -> None:
        """Deliver at the next step boundary without waking — it waits."""
        self.send(message, "next-step", False)

    def cancel(self, cause: AgentCancelCause, *, keep_inbox: bool = False) -> None:
        if not keep_inbox:
            self.inbox.clear()
        self._phase.cancelled = cause

    def _throw_if_cancelled(self) -> None:
        if self._phase.cancelled is not None:
            raise AgentCancelled(self._phase.cancelled)

    # ------------------------------------------------------------------ run --

    async def when_idle(self) -> None:
        await self._idle.wait()

    async def run(self) -> None:
        """Drive turns until the inbox is empty.

        The public entry point; a daemon or the CLI calls it after queueing
        input. Failures are reported at their live boundary (`agent/error`) and
        contained here, because a driver that propagated would take the process
        down with one bad turn.
        """
        if self._phase.kind != "idle":
            raise RuntimeError(f'agent "{self.id}" already has active work')
        self._idle = anyio.Event()
        self._set_phase("running")
        self._phase.step = 0
        self._phase.cancelled = None
        try:
            while await self._turn():
                pass
        except AgentCancelled:
            pass
        except Exception:
            log.debug("ph.agent_loop: driver contained a failure", exc_info=True)
        finally:
            self._set_phase("idle")
            self._idle.set()

    async def prompt(self, text: str) -> None:
        """Queue one human prompt and drive the loop to idle."""
        self.followup(
            create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})
        )
        await self.run()

    async def dispose(self) -> None:
        self.cancel(AgentCancelCause(kind="disposed"))

    # ---------------------------------------------------------------- phases --

    def _report(self, error: BaseException) -> None:
        self.ctx.emit("agent/error", self, self._phase.turn, self._phase.step, error)

    async def _pre_step(self, target: InboxTarget, turn: int, step: int) -> _PreparedStep:
        self._throw_if_cancelled()
        claimed = self.inbox.claim(target, turn)
        assembly = await self.ctx.system_prompt.assemble(
            AssembleContext(scope=self.ctx, agent=self)
        )
        self._throw_if_cancelled()
        context_message = self._project_context(assembly)
        messages = (*claimed, context_message) if context_message is not None else tuple(claimed)

        async def inner(request: PreStepRequest) -> PreStepDecision:
            return PreStepDecision(kind="enter", messages=request.messages)

        request = PreStepRequest(agent=self, messages=messages, turn=turn, step=step)
        decision = await self.ctx.waterfall("agent/pre-step", request, inner=inner)
        self._throw_if_cancelled()
        if not isinstance(decision, PreStepDecision):
            raise TypeError("agent/pre-step must resolve to a PreStepDecision")
        if decision.kind == "reject":
            return _PreparedStep(kind="reject")
        return _PreparedStep(kind="enter", messages=decision.messages, assembly=assembly)

    def _project_context(self, assembly: PromptAssembly) -> Message | None:
        """Materialize `context()` providers, but only when the text changed.

        This is the whole reason `context()` exists separately from `section`:
        re-sending unchanged context on every step would invalidate the cached
        prefix each turn (A12).
        """
        sections = render_context_sections(assembly)
        if not sections:
            return None
        text = join_context_sections(sections)
        if text == self._context_snapshot:
            return None
        self._context_snapshot = text
        return create_user_message(
            content=[{"type": "text", "text": text}],
            source=PluginSource(
                plugin="ph.system-prompt", form="snapshot", sections=list(sections)
            ),
        )

    async def _turn(self) -> bool:
        phase = self._phase
        self._throw_if_cancelled()
        turn = phase.turn + 1
        self.session.append("turn/start", {"turn": turn})
        phase.turn = turn
        turn_ends: TurnEndReason | None = None
        target: InboxTarget = "next-turn"
        try:
            while True:
                self._throw_if_cancelled()
                step = phase.step + 1
                decision = await self._pre_step(target, turn, step)
                if decision.kind == "reject":
                    turn_ends = TurnEndReason(kind="blocked")
                    return False
                if turn_ends is not None and not decision.messages:
                    break
                # A removed waking message, or an `enter` rewritten to empty,
                # still owns the turn boundary it opened — it just spends no
                # model call.
                if phase.step == 0 and not decision.messages:
                    turn_ends = TurnEndReason(kind="completed")
                    return False
                self._throw_if_cancelled()
                self.session.append("step/start", {"turn": turn, "step": step})
                phase.step = step
                try:
                    for message in decision.messages:
                        self.session.append(
                            "user/message", message.to_wire(), SurfaceIntent("append")
                        )
                    assert decision.assembly is not None
                    step_end = await self._step(decision.assembly)
                    # max-tokens stays sticky: a later completed step must not
                    # downgrade the turn outcome.
                    if turn_ends is None or turn_ends.kind != "max-tokens":
                        turn_ends = step_end
                finally:
                    self.session.append("step/end", {"turn": turn, "step": step})
                self._throw_if_cancelled()
                if turn_ends is not None and not self.inbox.next_step:
                    await self.ctx.serial("agent/turn-stopping", self, turn)
                    self._throw_if_cancelled()
                if turn_ends is not None and not self.inbox.next_step:
                    break
                target = "next-step"
        except AgentCancelled as cancelled:
            turn_ends = TurnEndReason(kind="aborted", reason=cancelled.cause)
            raise
        except Exception as error:
            failure = (
                error.failure
                if isinstance(error, LlmError)
                else LlmFailure(message=_error_chain(error), code="UNKNOWN")
            )
            turn_ends = TurnEndReason(kind="error", error=failure)
            self._report(error)
            raise
        finally:
            self.session.append(
                "turn/end",
                {"turn": turn, "reason": (turn_ends or TurnEndReason(kind="completed")).to_wire()},
            )
        if not self.inbox.has_pending:
            return False
        phase.cancelled = None
        phase.step = 0
        return True

    async def _step(self, assembly: PromptAssembly) -> TurnEndReason | None:
        phase = self._phase
        turn, step = phase.turn, phase.step
        self._throw_if_cancelled()
        system = render_prompt(assembly)

        while True:
            request = await self._build_request(turn, step, assembly, system)
            assembler = BlockAssembler()
            chunk_seqs: list[int] = []
            try:
                stream = await self.ctx.llm.stream(request)
                self._throw_if_cancelled()
                async for chunk in stream:
                    self._throw_if_cancelled()
                    # Raw chunks are logged before assembly, so the log carries
                    # token-level replay fidelity even for a stream that later
                    # fails.
                    chunk_seqs.append(
                        self.session.append(
                            "assistant/chunk",
                            {"turn": turn, "step": step, "chunk": chunk.to_wire()},
                        ).seq
                    )
                    assembler.push(chunk)
                self._throw_if_cancelled()
            except AgentCancelled:
                content = assembler.interrupted_blocks()
                if content:
                    # An interrupted turn still finalizes what the user saw:
                    # dropping it would leave the transcript claiming the model
                    # said nothing.
                    self._append_assistant_message(
                        turn, step, request, content, chunk_seqs, assembler.usage, interrupted=True
                    )
                raise

            finish = assembler.finish
            if finish.kind in ("error", "aborted"):
                action = await self._request_error(turn, step, request, finish)
                if action is None or action.kind != "retry":
                    failure = finish.failure or LlmFailure(
                        message="model request failed", code="UNKNOWN"
                    )
                    raise LlmError(failure.message, failure.code, failure)
                continue

            self._append_assistant_message(
                turn, step, request, assembler.blocks(), chunk_seqs, assembler.usage
            )
            if finish.kind == "max-tokens":
                return TurnEndReason(kind="max-tokens")
            # Phase 1 dispatches tool calls here; until then every step with a
            # message completes the turn.
            return TurnEndReason(kind="completed")

    async def _request_error(
        self, turn: int, step: int, request: GenerateOptions, finish: FinishReason
    ) -> RequestErrorAction | None:
        async def inner(request_failure: RequestFailure) -> RequestErrorAction | None:
            return None

        failure = RequestFailure(
            agent=self,
            turn=turn,
            step=step,
            provider=request.provider,
            failure=finish.failure or LlmFailure(message="model request failed", code="UNKNOWN"),
        )
        action = await self.ctx.waterfall("agent/request-error", failure, inner=inner)
        self._throw_if_cancelled()
        return action if isinstance(action, RequestErrorAction) else None

    def _append_assistant_message(
        self,
        turn: int,
        step: int,
        request: GenerateOptions,
        content: list[Any],
        chunk_seqs: list[int],
        usage: TokenUsage | None,
        *,
        interrupted: bool = False,
    ) -> None:
        message = create_assistant_message(
            content=content, provider=request.provider, model=request.model
        )
        data: dict[str, Any] = {"turn": turn, "step": step, "message": message.to_wire()}
        if usage is not None:
            data["usage"] = usage.to_wire()
        if interrupted:
            data["interrupted"] = True
        self.session.append("assistant/message", data, SurfaceIntent("append", tuple(chunk_seqs)))

    async def _build_request(
        self, turn: int, step: int, assembly: PromptAssembly, system: str
    ) -> GenerateOptions:
        """Compose one frozen request and log its header when it changed.

        The message list is `session.derive_messages()` and nothing else — the
        invariant plugin will refuse the request otherwise.
        """
        session = self.session
        persisted = session.request_header()
        seed = (
            _request_proposal(persisted)
            if self._request_header_logged and persisted is not None
            else self.options.seed_config()
        )

        async def inner(proposal: RequestProposal) -> LlmCallConfig:
            return proposal.config

        proposal = RequestProposal(agent=self, turn=turn, step=step, config=seed)
        proposed = await self.ctx.waterfall("agent/request", proposal, inner=inner)
        self._throw_if_cancelled()
        if not isinstance(proposed, LlmCallConfig):
            raise TypeError("agent/request must resolve to an LlmCallConfig")
        if not proposed.provider or not proposed.model:
            raise ValueError(
                f'agent "{self.id}" has no provider/model: set AgentOptions.provider '
                "and AgentOptions.model, or supply both via the agent/request waterfall"
            )

        header = canonical_header(
            EpochHeader(config=proposed, system=system or None, tools=list(assembly.tools) or None)
        )
        baseline = session.request_header()
        if not self._request_header_logged:
            session.append(
                "request/header",
                {"header": header.to_wire(), "reason": "initial" if baseline is None else "resume"},
            )
            self._request_header_logged = True
        elif baseline is None or not header_equals(baseline, header):
            session.append("request/header", {"header": header.to_wire(), "reason": "change"})

        resolved = self.ctx.llm.resolve_model(proposed.provider, proposed.model)
        request_context = RequestContext(
            provider=proposed.provider, model=proposed.model, context_window=resolved.context_window
        )
        if session.request_context() != request_context:
            session.append("request/context", request_context.to_wire())
        self._throw_if_cancelled()

        return GenerateOptions(
            provider=header.config.provider,
            model=header.config.model,
            messages=session.derive_messages(),
            system=header.system,
            tools=tuple(header.tools or ()),
            reasoning_effort=header.config.reasoning_effort,
            temperature=header.config.temperature,
            max_tokens=header.config.max_tokens,
            stop=tuple(header.config.stop or ()),
            session_id=session.id,
        )


def _request_proposal(header: EpochHeader) -> LlmCallConfig:
    """Strip adapter-materialized values before plugins propose the next config.

    A default the adapter chose is re-resolved per step against the exact model;
    freezing one into the conversation would outlive the route that produced it.
    """
    defaults = header.adapter_defaults
    if defaults is None:
        return header.config
    data = header.config.model_dump(by_alias=True, exclude_none=True)
    if defaults.reasoning_effort is True:
        data.pop("reasoningEffort", None)
    if defaults.max_tokens is True:
        data.pop("maxTokens", None)
    return LlmCallConfig.model_validate(data)


def _last_turn_of(session: Session) -> int:
    event = session.last_event_of("turn/start")
    return int(event.data.get("turn", 0)) if event is not None else 0


def _error_chain(error: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current) or type(current).__name__)
        current = current.__cause__ or current.__context__
    return ": ".join(parts)
