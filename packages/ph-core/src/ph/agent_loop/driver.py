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
from collections.abc import Sequence
from dataclasses import dataclass, field
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
from ..cancel import Canceled, CancelToken
from ..cordis import Context, settled, settled_or_none
from ..json import as_int
from ..keys import LLM, SYSTEM_PROMPT
from ..llm.adapter import LlmError
from ..llm.assembler import BlockAssembler, named_call
from ..llm.types import (
    ContentBlock,
    FinishReason,
    GenerateOptions,
    LlmCallConfig,
    LlmFailure,
    Message,
    PluginSource,
    TokenUsage,
    ToolCallBlock,
    create_assistant_message,
    create_user_message,
)
from ..session import Session, SurfaceIntent
from ..session.request_header import EpochHeader, RequestContext, canonical_header, header_equals
from ..system_prompt.assembly import (
    PromptAssembly,
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from ..tools.batch import execute_tool_calls
from ..tools.errors import error_info

__all__ = ["AgentCanceled", "ReactLoopAgent"]

log = logging.getLogger("ph.agent_loop")


class AgentCanceled(Exception):
    """The active driver was canceled; carries the cause for `turn/end`."""

    def __init__(self, cause: AgentCancelCause) -> None:
        super().__init__(f"agent canceled: {cause.kind}")
        self.cause = cause


@dataclass(slots=True)
class _Phase:
    kind: str = "idle"
    turn: int = 0
    step: int = 0
    canceled: AgentCancelCause | None = None
    token: CancelToken = field(default_factory=CancelToken)
    """The live cancellation view handed to tool calls.

    A flag the loop checks between awaits is not enough once a tool body can run
    for minutes: the body needs to observe cancellation itself, and the pipeline
    needs to tell "aborted before dispatch" from "aborted" to know whether the
    call had an effect."""

    def begin_turn(self) -> None:
        """Fresh per-turn state: a new token, so an old cancellation cannot leak
        into the next turn's tool calls."""
        self.step = 0
        self.canceled = None
        self.token = CancelToken()


@dataclass(frozen=True, slots=True)
class _PreparedStep:
    kind: str
    messages: tuple[Message, ...] = ()
    assembly: PromptAssembly | None = None


class ReactLoopAgent:
    """The default agent driver."""

    def __init__(
        self,
        ctx: Context,
        session: Session,
        options: AgentOptions,
        *,
        max_parallel_tool_calls: int = 10,
    ) -> None:
        self.ctx = ctx
        self.session = session
        self.options = options
        self.id = session.id
        self.max_parallel_tool_calls = max_parallel_tool_calls
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
            wakeup and self._phase.kind != "idle" and self._phase.canceled is not None
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

    def interject(self, message: Message) -> None:
        """Deliver as soon as this loop will take it: mid-turn if one is running.

        The verb for *"also this"* from outside — a person typing while the agent
        works. A running turn takes it at the next step, so the answer does not
        wait for whatever the agent is in the middle of; an idle one has no turn
        to join, and a prompt to an idle agent means a new one.

        **Decided here rather than by the caller**, because the phase is this
        object's and a caller reading `status` and then choosing a verb is
        branching on a value that can change between the two. It also gives every
        front end the same reach without each re-deriving it — and the reach a
        child already had, since a child's message is delivered by `steer`.
        """
        self.send(message, "next-step" if self._phase.kind != "idle" else "next-turn", True)

    @property
    def signal(self) -> CancelToken:
        """The live cancellation view — the same token `cancel` below trips."""
        return self._phase.token

    def cancel(self, cause: AgentCancelCause, *, keep_inbox: bool = False) -> None:
        if not keep_inbox:
            self.inbox.clear()
        self._phase.canceled = cause
        self._phase.token.cancel(cause.kind)

    def _throw_if_canceled(self) -> None:
        if self._phase.canceled is not None:
            raise AgentCanceled(self._phase.canceled)

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
        self._phase.begin_turn()
        try:
            while await self._turn():
                pass
        except (AgentCanceled, Canceled):
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
        self._throw_if_canceled()
        # **Assembled before the batch is claimed** (C4). `claim` is durable —
        # it appends `agent/inbox/spliced`, which is what takes the messages out
        # of the inbox for good — and `assemble` is an await that a person's
        # interrupt can land in. Claimed first, a cancel there consumed the
        # prompt and ran nothing: the typed line was gone from the inbox and
        # never reached a model call. Nothing here needs the batch, so the
        # ordering costs nothing and the claim now happens on the far side of
        # the last cancel check before the step is proposed.
        assembly = await self.ctx.require(SYSTEM_PROMPT).assemble(self.ctx, agent=self)
        self._throw_if_canceled()
        claimed = self.inbox.claim(target, turn)
        context_message, context_text = self._project_context(assembly)
        messages = (*claimed, context_message) if context_message is not None else tuple(claimed)

        async def inner(request: PreStepRequest) -> PreStepDecision:
            return PreStepDecision(kind="enter", messages=request.messages)

        request = PreStepRequest(
            agent=self, session=self.session, messages=messages, turn=turn, step=step
        )
        answered = await self.ctx.waterfall("agent/pre-step", request, inner=inner)
        self._throw_if_canceled()
        decision = settled("agent/pre-step", answered, PreStepDecision)
        if decision.kind == "reject":
            return _PreparedStep(kind="reject")
        if context_message is not None and any(one is context_message for one in decision.messages):
            # **Advanced where the message survives, not where it was built**
            # (C3). The snapshot is what stops unchanged context re-invalidating
            # the cached prefix every step, so moving it forward is a promise
            # that the model has been told — and a pre-step that rejected, was
            # canceled, or dropped the message from `messages` made that promise
            # falsely. An `AGENTS.md` edit then went unseen until the file
            # changed again, which for a file somebody edits once is never.
            #
            # By identity, because a listener may substitute the batch: the
            # snapshot belongs to *this* text reaching the step, not to some
            # equal-looking message a row put in its place.
            self._context_snapshot = context_text
        return _PreparedStep(kind="enter", messages=decision.messages, assembly=assembly)

    def _project_context(self, assembly: PromptAssembly) -> tuple[Message | None, str]:
        """Materialize `context()` providers, but only when the text changed.

        This is the whole reason `context()` exists separately from `section`:
        re-sending unchanged context on every step would invalidate the cached
        prefix each turn (A12).

        **Returns the text and advances nothing** (C3). The snapshot is the
        record of what the model has been *told*, and this only knows what was
        built; a step that never happened would otherwise mark its context as
        delivered. `_pre_step` commits it once the message is in the batch the
        step will run.

        The text is empty whenever the message is `None`, because the caller
        reads it only alongside a message: the two are one answer, not two. An
        earlier shape returned the *current* snapshot on the no-op path, which
        made a stale commit look like a plausible reading of this function.
        """
        sections = render_context_sections(assembly)
        if not sections:
            return None, ""
        text = join_context_sections(sections)
        if text == self._context_snapshot:
            return None, ""
        return create_user_message(
            content=[{"type": "text", "text": text}],
            source=PluginSource(
                plugin="ph.system-prompt", form="snapshot", sections=list(sections)
            ),
        ), text

    async def _turn(self) -> bool:
        phase = self._phase
        self._throw_if_canceled()
        turn = phase.turn + 1
        self.session.append("turn/start", {"turn": turn})
        phase.turn = turn
        turn_ends: TurnEndReason | None = None
        capped = False
        """Whether any step in this turn stopped at `max_tokens`.

        Kept beside `turn_ends` rather than inside it, because the two answer
        different questions: what ends the turn, and what the person should be
        told about it. Folding the second into the first made a `max-tokens` step
        *end the turn* even where the loop had more to do — see where it is read
        below.
        """
        target: InboxTarget = "next-turn"
        try:
            while True:
                self._throw_if_canceled()
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
                self._throw_if_canceled()
                self.session.append("step/start", {"turn": turn, "step": step})
                phase.step = step
                try:
                    for message in decision.messages:
                        self.session.append(
                            "user/message", message.to_wire(), SurfaceIntent("append")
                        )
                    assert decision.assembly is not None
                    step_end = await self._step(decision.assembly)
                    # **The outcome is sticky; the stopping is not.** A step that
                    # hit `max_tokens` must still be reported as such at the end
                    # of the turn, however the turn actually finishes — but a
                    # `None` here means "tools ran, there is more to do", and
                    # holding the earlier reason for it ended the turn with a
                    # `tool/result` the model never saw. The two were one
                    # variable, so the sticky answer was also the deciding one.
                    capped = capped or (step_end is not None and step_end.kind == "max-tokens")
                    turn_ends = step_end
                finally:
                    self.session.append("step/end", {"turn": turn, "step": step})
                self._throw_if_canceled()
                if turn_ends is not None and not self.inbox.next_step:
                    await self.ctx.serial("agent/turn-stopping", self, turn)
                    self._throw_if_canceled()
                if turn_ends is not None and not self.inbox.next_step:
                    break
                target = "next-step"
        except (AgentCanceled, Canceled) as canceled:
            cause = (
                canceled.cause
                if isinstance(canceled, AgentCanceled)
                else self._phase.canceled or AgentCancelCause(kind="user")
            )
            turn_ends = TurnEndReason(kind="aborted", reason=cause)
            raise
        except Exception as error:
            # A harness error's own code rather than `UNKNOWN`, because that is
            # what `HarnessError` is for: *"a failure's routing matters as much as
            # its message — retry policy, the sandbox layer and replay all branch
            # on the code"*. Flattening every one of them here made `turn/end`
            # unable to say which failure it was, in the record a client reads to
            # decide whether retrying could possibly help.
            failure = error.failure if isinstance(error, LlmError) else _as_failure(error)
            turn_ends = TurnEndReason(kind="error", error=failure)
            self._report(error)
            raise
        finally:
            self.session.append(
                "turn/end",
                {"turn": turn, "reason": _turn_reason(turn_ends, capped).to_wire()},
            )
        if not self.inbox.has_pending:
            return False
        phase.begin_turn()
        return True

    async def _step(self, assembly: PromptAssembly) -> TurnEndReason | None:
        phase = self._phase
        turn, step = phase.turn, phase.step
        self._throw_if_canceled()
        system = render_prompt(assembly)

        # **Which attempt this is, because a retried step keeps its coordinates**
        # (G12). That is deliberate — `llm/retry` is legible in the log because
        # of it — but it leaves `(turn, step)` naming a *step* where a model
        # *call* is what needs the name. Two attempts minted the same tool-call
        # ids and `recorded_steps` had to infer the boundary between them.
        #
        # **Counted here because this is the only place every retry passes.**
        # Not `retry.attempts_so_far`, which is a different number: it folds
        # `llm/retry` records, which only the `llm-retry` row writes, so a retry
        # `compaction` asks for after an overflow never reaches it. That is right
        # for a retry *budget* and wrong for a call's identity — the two used to
        # be described as one fact, and they diverge on the first non-`llm-retry`
        # retry.
        attempt = 0
        while True:
            request = await self._build_request(turn, step, assembly, system)
            assembler = BlockAssembler()
            chunk_seqs: list[int] = []
            try:
                stream = await self.ctx.require(LLM).stream(request)
                self._throw_if_canceled()
                async for chunk in stream:
                    self._throw_if_canceled()
                    # **Named before it is logged** (G11): a tool call the
                    # provider did not name gets its id here, so the raw record
                    # and the assembled message agree about it. This is the only
                    # consumer that pairs a call to a result, and the only one
                    # holding the coordinates the id is built from.
                    chunk = named_call(chunk, turn, step, attempt)
                    # Raw chunks are logged before assembly, so the log carries
                    # token-level replay fidelity even for a stream that later
                    # fails.
                    chunk_seqs.append(
                        self.session.append(
                            "assistant/chunk",
                            {
                                "turn": turn,
                                "step": step,
                                # The call's own coordinate, so `recorded_steps`
                                # reads the boundary between two attempts rather
                                # than inferring it (G12).
                                "attempt": attempt,
                                "chunk": chunk.to_wire(),
                            },
                        ).seq
                    )
                    assembler.push(chunk)
                self._throw_if_canceled()
            except (AgentCanceled, Canceled):
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
                attempt += 1
                continue

            blocks = assembler.blocks()
            self._append_assistant_message(turn, step, request, blocks, chunk_seqs, assembler.usage)
            if finish.kind == "max-tokens":
                # Sticky, and checked before dispatch: max-tokens already
                # dropped any truncated tool call, so there is nothing to run.
                return TurnEndReason(kind="max-tokens")
            tool_calls = [block for block in blocks if isinstance(block, ToolCallBlock)]
            if not tool_calls:
                return TurnEndReason(kind="completed")
            return await self._dispatch_tools(turn, step, tool_calls)

    async def _dispatch_tools(
        self, turn: int, step: int, tool_calls: list[ToolCallBlock]
    ) -> TurnEndReason | None:
        """Run the step's tool batch; `None` continues the turn with another step.

        Result context is spliced into the next-step inbox rather than appended
        here, so it lands *after* every `tool/result` and call/result adjacency
        survives.
        """
        outcome = await execute_tool_calls(
            self.ctx,
            self,
            turn,
            step,
            tool_calls,
            self._phase.token,
            lambda context: self.inbox.append("next-step", context),
            max_parallel=self.max_parallel_tool_calls,
        )
        if outcome.aborted:
            self._throw_if_canceled()
            raise Canceled("tool batch aborted")
        return TurnEndReason(kind="completed") if outcome.concluded else None

    async def _request_error(
        self, turn: int, step: int, request: GenerateOptions, finish: FinishReason
    ) -> RequestErrorAction | None:
        async def inner(request_failure: RequestFailure) -> RequestErrorAction | None:
            return None

        failure = RequestFailure(
            agent=self,
            session=self.session,
            turn=turn,
            step=step,
            provider=request.provider,
            failure=finish.failure or LlmFailure(message="model request failed", code="UNKNOWN"),
        )
        action = await self.ctx.waterfall("agent/request-error", failure, inner=inner)
        self._throw_if_canceled()
        return settled_or_none("agent/request-error", action, RequestErrorAction)

    def _append_assistant_message(
        self,
        turn: int,
        step: int,
        request: GenerateOptions,
        content: Sequence[ContentBlock],
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

        proposal = RequestProposal(
            agent=self, session=self.session, turn=turn, step=step, config=seed
        )
        answered = await self.ctx.waterfall("agent/request", proposal, inner=inner)
        self._throw_if_canceled()
        proposed = settled("agent/request", answered, LlmCallConfig)
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

        resolved = self.ctx.require(LLM).resolve_model(proposed.provider, proposed.model)
        request_context = RequestContext(
            provider=proposed.provider, model=proposed.model, context_window=resolved.context_window
        )
        if session.request_context() != request_context:
            session.append("request/context", request_context.to_wire())
        self._throw_if_canceled()

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
    return as_int(event.data.get("turn")) if event is not None else 0


def _error_chain(error: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(str(current) or type(current).__name__)
        current = current.__cause__ or current.__context__
    return ": ".join(parts)


def _turn_reason(ends: TurnEndReason | None, capped: bool) -> TurnEndReason:
    """What `turn/end` records, given how the loop left and what it passed through.

    **A cap is not erased by finishing.** A step that hit `max_tokens` cut the
    model off, and the turn may then have carried on — a steer, a tool call,
    another step — and ended tidily. Reporting that as `completed` tells the
    person their answer is whole when part of it was dropped, so `max-tokens`
    outranks the two endings that mean "nothing went wrong".

    It does not outrank the three that say more: `aborted`, `blocked` and
    `error` each name something the reader has to act on, and a cap somewhere in
    the turn is the smaller fact beside them.

    **`None` means an exception left the body, so it is `aborted`** (C9). Every
    normal exit from `_turn` assigns `ends` first — both `break`s are guarded by
    `turn_ends is not None` and both `return` paths set it — so a `None` arriving
    here cannot be a turn that finished. It used to read as `completed`, and each
    `except` branch in `_turn` existed to write over that default before the
    `finally` could believe it.

    Deriving it here instead of catching it there is what makes the coverage
    total. A branch for `anyio.get_cancelled_exc_class()` still misses the
    `BaseExceptionGroup` an anyio task group raises around one, and
    `KeyboardInterrupt`, and `SystemExit` — each of which would have recorded a
    turn that finished nothing as *completed*, so the resume path saw no open
    turn to repair and a reader was told a story the run did not have.

    `kind="user"` is the honest cause for that case: something outside the turn
    decided, and the loop cannot see what. The harness's own cancellations arrive
    as `AgentCanceled`/`Canceled`, are caught in `_turn`, and carry their real
    cause.
    """
    if ends is None:
        return TurnEndReason(kind="aborted", reason=AgentCancelCause(kind="user"))
    if ends.kind == "completed":
        return TurnEndReason(kind="max-tokens" if capped else "completed")
    return ends


def _as_failure(error: Exception) -> LlmFailure:
    """A raised error as a turn's `reason.error`, keeping its code if it has one.

    A harness error's own code rather than `UNKNOWN`, because that is what
    `HarnessError` is for: *"a failure's routing matters as much as its message —
    retry policy, the sandbox layer and replay all branch on the code"*. Flattening
    every one of them left `turn/end` unable to say which failure it was, in the
    record a client reads to decide whether retrying could possibly help.
    """
    coded = error_info(error)
    return LlmFailure(
        message=_error_chain(error), code="UNKNOWN" if coded is None else coded["code"]
    )
