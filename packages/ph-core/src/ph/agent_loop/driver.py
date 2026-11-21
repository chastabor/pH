"""`ReactLoopAgent`: one session driven through turn and step boundaries.

The driver implements the lifecycle of dsh verbatim. Tool execution is not in
the driver: a step hands its tool calls to the tool pipeline
(`execute_tool_calls`, P1-02). The order below is the contract. Every
stabilization feature (Phase 4) attaches to one of these seams and not to the
loop itself (D12):

```
turn/start
  ├ system_prompt.assemble    → system-prompt/assemble
  ├ inbox.propose             (nothing taken yet)
  ├ agent/pre-step            → reject | enter(messages)
  │    reject         → agent/inbox/spliced, turn/end{blocked}
  │    enter, empty, first step → agent/inbox/spliced, turn/end{completed}
  ├ step/start                (one batch with the splice → agent/inbox/claimed)
  │    user/message*          (the batch, surface: append)
  │    agent/request          → LlmCallConfig
  │    request/header         (on this driver's first request, then on a change — A12)
  │    request/context        (appended only when the route or its window changed)
  │    llm/stream             → assistant/chunk* → assistant/message
  │    agent/request-error    → retry | None
  │    step/retry             (the record of a model call that is made again)
  │    execute_tool_calls     (the tool calls of the message, when it has any)
  ├ step/end
  ├ agent/turn-stopping       (a listener objects by steering)
turn/end
```

Two rules are easy to lose in a port and expensive to lose in production:

* The `messages` of every request is `session.derive_messages()`. It is never
  an array that is maintained separately (invariant I3, asserted by P0-14).
* `max-tokens` is **sticky** for the turn. A later completed step must not
  downgrade the outcome, or a truncated answer is reported as a clean one.

The driver runs inside the scope that `ctx.agents` created for it. Services
resolve through that scope, the most specific first. `self.ctx.require(LLM)` is
therefore the global adapter seam, unless something shadowed it for this agent
alone.

@module ph.agent_loop.driver
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio

from ..agent.inbox import Inbox, InboxBatch, InboxNotifications, InboxTarget
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
from ..cancel import Canceled, CancelToken, reason_of
from ..cordis import Context, settled, settled_or_none
from ..json import as_int
from ..keys import LLM, MODELS, SYSTEM_PROMPT
from ..llm.adapter import LlmError
from ..llm.assembler import BlockAssembler, named_call
from ..llm.types import (
    ContentBlock,
    FinishReason,
    GenerateOptions,
    LlmCallConfig,
    LlmFailure,
    Message,
    TokenUsage,
    ToolCallBlock,
    create_assistant_message,
    user_text,
)
from ..session import Session, SurfaceIntent
from ..session.request_header import EpochHeader, RequestContext, canonical_header, header_equals
from ..session.writers import log_writer
from ..system_prompt.assembly import PromptAssembly, context_message, render_prompt
from ..tools.batch import execute_tool_calls
from ..tools.errors import error_info

_LOG = log_writer(__name__)

__all__ = ["ReactLoopAgent"]

log = logging.getLogger("ph.agent_loop")


@dataclass(slots=True)
class _Phase:
    kind: str = "idle"
    turn: int = 0
    step: int = 0
    cause: AgentCancelCause | None = None
    """Who stopped this turn, as the `cancel` of the loop was told, for its `turn/end`.
    The token says whether the turn is canceled. A stop that is sent straight to
    the token has no cause, and it stops the loop all the same."""
    token: CancelToken = field(default_factory=CancelToken)
    """The live cancellation view that is handed to tool calls. The loop checks the
    same one.

    A flag that the loop checks between awaits is not enough when a tool body
    can run for minutes. The body needs to observe cancellation itself. The
    pipeline also needs to tell "aborted before dispatch" from "aborted", to
    know whether the call had an effect."""

    def begin_turn(self) -> None:
        """Fresh per-turn state. The token is new, so an old cancellation cannot
        leak into the tool calls of the next turn."""
        self.step = 0
        self.cause = None
        self.token = CancelToken()


@dataclass(frozen=True, slots=True)
class _PreparedStep:
    kind: str
    held: InboxBatch
    """The inbox batch that the step was proposed with. It stays pending until it is taken."""
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
        self._rerouted = False
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

    def reroute(self, options: AgentOptions) -> None:
        """Run on `options` from the next request on.

        Setting `options` alone would change nothing after the first request of
        this process. From then on a request is seeded from the `request/header`
        of the conversation, so what a waterfall proposed stays proposed. This
        method marks the next seed as coming from `options` instead. The header
        that this request logs, with reason `change`, is the record of the new
        route in the log. A turn in flight finishes its current request on the
        old route.
        """
        self.options = options
        self._rerouted = True

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
        # turn. It is classified BEFORE the insertion, so that a reentrant
        # cancel from a splice observer cannot reclassify it.
        waking_after_abort = wakeup and self._phase.kind != "idle" and self._phase.token.canceled
        self.inbox.append("next-turn" if waking_after_abort else target, message)
        if wakeup:
            # This event is the wake that the table in `ph.agent.inbox`
            # promises. This loop runs only when something calls `run()`. The
            # host that holds it (the root task of a daemon, an rpc server
            # between prompts) hears this event and runs the loop. The event is
            # emitted after the append, so a listener that runs the agent at
            # once finds the message pending.
            self.ctx.emit("agent/woken", self)

    def followup(self, message: Message) -> None:
        """Deliver at the next turn boundary and wake the agent."""
        self.send(message, "next-turn", True)

    def steer(self, message: Message) -> None:
        """Deliver at the next step boundary and wake the agent."""
        self.send(message, "next-step", True)

    def inject(self, message: Message) -> None:
        """Deliver at the next step boundary without waking the agent. The message waits."""
        self.send(message, "next-step", False)

    def interject(self, message: Message) -> None:
        """Deliver as soon as this loop will take it: mid-turn if a turn is running.

        This is the verb for *"also this"* from outside: a user who types while
        the agent works. A running turn takes the message at the next step, so
        the answer does not wait for whatever the agent is in the middle of. An
        idle agent has no turn to join, and a prompt to an idle agent means a
        new turn.

        **The choice is made here and not by the caller**, because the phase
        belongs to this object. A caller that reads `status` and then chooses a
        verb branches on a value that can change between the two. This also
        gives every front end the same reach, without each front end deriving
        it again. A child already had that reach, because the message of a
        child is delivered by `steer`.
        """
        self.send(message, "next-step" if self._phase.kind != "idle" else "next-turn", True)

    @property
    def signal(self) -> CancelToken:
        """The live cancellation view. It is the same token that `cancel` below trips."""
        return self._phase.token

    def cancel(self, cause: AgentCancelCause, *, keep_inbox: bool = False) -> None:
        if not keep_inbox:
            self.inbox.clear()
        self._phase.cause = cause
        self._phase.token.cancel(cause.kind)

    def _throw_if_canceled(self) -> None:
        self._phase.token.raise_if_canceled()

    # ------------------------------------------------------------------ run --

    async def when_idle(self) -> None:
        await self._idle.wait()

    async def run(self) -> None:
        """Drive turns until the inbox is empty.

        This is the public entry point. A daemon or the CLI calls it after
        queueing input. Failures are reported at their live boundary
        (`agent/error`) and contained here, because a driver that propagated
        them would take the process down with one bad turn.
        """
        if self._phase.kind != "idle":
            raise RuntimeError(f'agent "{self.id}" already has active work')
        self._idle = anyio.Event()
        self._set_phase("running")
        self._phase.begin_turn()
        try:
            while await self._turn():
                pass
        except Canceled:
            pass
        except Exception:
            log.debug("ph.agent_loop: driver contained a failure", exc_info=True)
        finally:
            self._set_phase("idle")
            self._idle.set()

    async def prompt(self, text: str) -> None:
        """Queue one human prompt and drive the loop to idle."""
        self.followup(user_text(text))
        await self.run()

    async def dispose(self) -> None:
        self.cancel(AgentCancelCause(kind="disposed"))

    # ---------------------------------------------------------------- phases --

    def _report(self, error: BaseException) -> None:
        self.ctx.emit("agent/error", self, self._phase.turn, self._phase.step, error)

    async def _pre_step(self, target: InboxTarget, turn: int, step: int) -> _PreparedStep:
        self._throw_if_canceled()
        # **The batch is proposed, not taken** (C4, S3). Taking the batch
        # appends `agent/inbox/spliced`, which removes the messages from the
        # inbox for good. Both `assemble` and `agent/pre-step` are awaits. The
        # interrupt of a user can land in either. A pre-step listener may also
        # make a model call, as the summary of compaction does, and the
        # checkpoint barrier of that call flushes the log. When the batch was
        # taken first, an interrupt consumed the prompt and ran nothing, and a
        # crash during that call left the prompt out of the inbox on disk and
        # nowhere in the conversation. So the batch is only proposed here, and
        # `_turn` takes it in the batch that records the step.
        assembly = await self.ctx.require(SYSTEM_PROMPT).assemble(self.ctx, agent=self)
        self._throw_if_canceled()
        held = self.inbox.propose(target)
        # Only when the conversation that the model is shown lacks it (C12).
        # See `context_message`.
        context = context_message(assembly, self.session.derive_messages())
        messages = (*held.messages, context) if context is not None else held.messages

        async def inner(request: PreStepRequest) -> PreStepDecision:
            return PreStepDecision(kind="enter", messages=request.messages)

        request = PreStepRequest(
            agent=self, session=self.session, messages=messages, turn=turn, step=step
        )
        try:
            answered = await self.ctx.waterfall("agent/pre-step", request, inner=inner)
        except Exception:
            # **A pre-step that fails still consumes its batch**, as it did
            # when the batch was taken first. `run` contains a turn failure and
            # nothing runs the turn again (the `_drive` of the daemon says
            # why). A batch that was left pending would therefore be taken by
            # the turn of the *next* prompt, ahead of the prompt that woke it.
            # A stop is not a failure, and it leaves the batch pending.
            if not self._phase.token.canceled:
                self.inbox.take(held, turn)
            raise
        self._throw_if_canceled()
        decision = settled("agent/pre-step", answered, PreStepDecision)
        if decision.kind == "reject":
            # A refusal is handled as a failure is: the batch is consumed, and
            # it reaches no model call.
            self.inbox.take(held, turn)
            await self.ctx.serial("agent/step-rejected", request)
            return _PreparedStep(kind="reject", held=held)
        return _PreparedStep(kind="enter", held=held, messages=decision.messages, assembly=assembly)

    async def _turn(self) -> bool:
        phase = self._phase
        self._throw_if_canceled()
        turn = phase.turn + 1
        _LOG.append(self.session, "turn/start", {"turn": turn})
        phase.turn = turn
        turn_ends: TurnEndReason | None = None
        capped = False
        """Whether any step in this turn stopped at `max_tokens`.

        This is kept beside `turn_ends` and not inside it, because the two
        answer different questions: what ends the turn, and what the user
        should be told about it. When the second was folded into the first, a
        `max-tokens` step *ended the turn* even where the loop had more to do.
        See where it is read below.
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
                if not decision.messages and (turn_ends is not None or phase.step == 0):
                    # The pre-step emptied the batch, so it opens no step and
                    # consumes what it was proposed with. On the first step (a
                    # waking message that was removed, an `enter` that was
                    # rewritten to empty) it still owns the turn boundary that
                    # it opened. It only spends no model call.
                    self.inbox.take(decision.held, turn)
                    if turn_ends is not None:
                        break
                    turn_ends = TurnEndReason(kind="completed")
                    return False
                self._throw_if_canceled()
                with self.inbox.claim(decision.held, turn) as batch:
                    _LOG.append(batch, "step/start", {"turn": turn, "step": step})
                    for message in decision.messages:
                        _LOG.append(
                            batch, "user/message", message.to_wire(), SurfaceIntent("append")
                        )
                phase.step = step
                try:
                    assert decision.assembly is not None
                    step_end = await self._step(decision.assembly)
                    # **The outcome is sticky. The stopping is not.** A step
                    # that hit `max_tokens` must still be reported as such at
                    # the end of the turn, however the turn finishes. But a
                    # `None` here means "tools ran, there is more to do".
                    # Holding the earlier reason over it ended the turn with a
                    # `tool/result` that the model never saw. The two were one
                    # variable, so the sticky answer was also the deciding one.
                    capped = capped or (step_end is not None and step_end.kind == "max-tokens")
                    turn_ends = step_end
                finally:
                    _LOG.append(self.session, "step/end", {"turn": turn, "step": step})
                self._throw_if_canceled()
                if turn_ends is not None and not self.inbox.next_step:
                    await self.ctx.serial("agent/turn-stopping", self, turn)
                    self._throw_if_canceled()
                if turn_ends is not None and not self.inbox.next_step:
                    break
                target = "next-step"
        except Canceled as error:
            # A stop, recorded over whatever ending a step had already decided.
            turn_ends = _aborted(error, phase.cause)
            raise
        except Exception as error:
            # The failure keeps the code of a harness error and does not become
            # `UNKNOWN`. That is the purpose of `HarnessError`: the routing of
            # a failure matters as much as its message, because retry policy,
            # the sandbox layer and replay all branch on the code. When every
            # error was flattened here, `turn/end` could not say which failure
            # it was, in the record that a client reads to decide whether a
            # retry could possibly help.
            failure = error.failure if isinstance(error, LlmError) else _as_failure(error)
            turn_ends = TurnEndReason(kind="error", error=failure)
            self._report(error)
            raise
        except BaseException as error:
            # A stop from outside, a canceled scope, an interrupt or an exit.
            # Each is recorded as the stops of the loop itself are.
            turn_ends = _aborted(error, phase.cause)
            raise
        finally:
            _LOG.append(
                self.session,
                "turn/end",
                {
                    "turn": turn,
                    "reason": _turn_reason(turn_ends, capped).to_wire(),
                },
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

        # **Which attempt this is, because a retried step keeps its
        # coordinates** (G12). That is deliberate, because it makes `llm/retry`
        # readable in the log. But it leaves `(turn, step)` naming a *step*
        # where a model *call* is what needs the name. Two attempts created the
        # same tool-call ids, and `recorded_steps` had to infer the boundary
        # between them.
        #
        # **The count is here because this is the only place that every retry
        # passes**, and it is kept by the row that asked (G13). The attempt is
        # the total. The budget of a policy is its own share or the total, as
        # the policy chooses.
        retries: Counter[str] = Counter()
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
                    # **Named before it is logged** (G11). A tool call that the
                    # provider did not name gets its id here, so the raw record
                    # and the assembled message agree about it. This is the
                    # only consumer that pairs a call to a result, and the only
                    # one that holds the coordinates that the id is built from.
                    chunk = named_call(chunk, turn, step, attempt)
                    # Raw chunks are logged before assembly. The log therefore
                    # has token-level replay fidelity even for a stream that
                    # fails later.
                    chunk_seqs.append(
                        _LOG.append(
                            self.session,
                            "assistant/chunk",
                            {
                                "turn": turn,
                                "step": step,
                                # The coordinate of the call itself, so that
                                # `recorded_steps` reads the boundary between
                                # two attempts and does not infer it (G12).
                                "attempt": attempt,
                                "chunk": chunk.to_wire(),
                            },
                        ).seq
                    )
                    assembler.push(chunk)
                self._throw_if_canceled()
            except Canceled:
                content = assembler.interrupted_blocks()
                if content:
                    # An interrupted turn still finalizes what the user saw.
                    # Dropping it would leave a transcript that claims the
                    # model said nothing.
                    self._append_assistant_message(
                        turn, step, request, content, chunk_seqs, assembler.usage, interrupted=True
                    )
                raise

            finish = assembler.finish
            if finish.kind in ("error", "aborted"):
                action, by = await self._request_error(turn, step, retries, request, finish)
                if action is None or action.kind != "retry":
                    failure = finish.failure or LlmFailure(
                        message="model request failed", code="UNKNOWN"
                    )
                    raise LlmError(failure.message, failure.code, failure)
                retries[by] += 1
                attempt = retries.total()
                if action.granted is not None:
                    action.granted()
                # The record that the loop itself keeps of the call that it is
                # about to make again, beside `step/start` for the first call
                # (P1). It is the answer of the log to "how many model calls",
                # whichever row asked for the retry. `llm/retry` is the account
                # that `llm-retry` gives of itself, and a compaction retry
                # writes none, so a ceiling that counts those records
                # undercounts.
                _LOG.append(
                    self.session,
                    "step/retry",
                    {"turn": turn, "step": step, "attempt": attempt, "by": by},
                )
                # **The loop does the waiting**, after the waterfall has
                # settled. A row that slept inside the waterfall answered only
                # after a later veto could no longer save the time, and a gate
                # had to run *ahead* of that row to refuse at all (P1 review).
                if action.delay_ms > 0:
                    await anyio.sleep(action.delay_ms / 1000)
                    self._throw_if_canceled()
                continue

            blocks = assembler.blocks()
            self._append_assistant_message(turn, step, request, blocks, chunk_seqs, assembler.usage)
            if finish.kind == "max-tokens":
                # Sticky, and checked before dispatch. `max-tokens` already
                # dropped any truncated tool call, so there is nothing to run.
                return TurnEndReason(kind="max-tokens")
            tool_calls = [block for block in blocks if isinstance(block, ToolCallBlock)]
            if not tool_calls:
                return TurnEndReason(kind="completed")
            return await self._dispatch_tools(turn, step, tool_calls)

    async def _dispatch_tools(
        self, turn: int, step: int, tool_calls: list[ToolCallBlock]
    ) -> TurnEndReason | None:
        """Run the tool batch of the step. `None` continues the turn with another step.

        Result context is spliced into the next-step inbox and not appended
        here. It therefore lands *after* every `tool/result`, and each call
        stays adjacent to its result.
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
        self,
        turn: int,
        step: int,
        retries: Mapping[str, int],
        request: GenerateOptions,
        finish: FinishReason,
    ) -> tuple[RequestErrorAction | None, str]:
        """The answer of the chain, and the row whose listener gave it.

        Cordis attributes the answer, and the action does not claim it (G13).
        The retry is counted under the plugin that granted it. A row could
        misname that plugin, and cordis already knows it.
        """

        async def inner(request_failure: RequestFailure) -> RequestErrorAction | None:
            return None

        failure = RequestFailure(
            agent=self,
            session=self.session,
            turn=turn,
            step=step,
            provider=request.provider,
            failure=finish.failure or LlmFailure(message="model request failed", code="UNKNOWN"),
            # A copy: the payload is a snapshot, and the loop goes on counting.
            retries_by=dict(retries),
        )
        action, by = await self.ctx.waterfall_attributed(
            "agent/request-error", failure, inner=inner
        )
        self._throw_if_canceled()
        return settled_or_none("agent/request-error", action, RequestErrorAction), by

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
        _LOG.append(
            self.session, "assistant/message", data, SurfaceIntent("append", tuple(chunk_seqs))
        )

    async def _build_request(
        self, turn: int, step: int, assembly: PromptAssembly, system: str
    ) -> GenerateOptions:
        """Compose one frozen request, and log its header when the header changed.

        The message list is `session.derive_messages()` and nothing else. The
        invariant plugin refuses the request otherwise.
        """
        session = self.session
        persisted = session.request_header()
        seed = (
            _request_proposal(persisted)
            if self._request_header_logged and not self._rerouted and persisted is not None
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

        # What the route cannot carry is withheld before the header is logged.
        # The header therefore says what was sent and, under `withheld`, what
        # was asked for and not sent.
        llm = self.ctx.require(LLM)
        resolved = llm.resolve_model(proposed.provider, proposed.model)
        # What each effort sends on the route that is sent, wherever the
        # waterfall moved it, according to the list of the session itself. The
        # entry that these options came from is looked up first.
        listed = self.ctx.get(MODELS)
        efforts = (
            listed.efforts_for(proposed.provider, proposed.model, key=self.options.model_key)
            if listed is not None
            else {}
        )
        sent, withheld = llm.withhold(proposed, resolved, efforts)
        header = canonical_header(
            EpochHeader(
                config=sent,
                withheld=withheld,
                system=system or None,
                tools=list(assembly.tools) or None,
            )
        )
        baseline = session.request_header()
        if not self._request_header_logged:
            _LOG.append(
                session,
                "request/header",
                {"header": header.to_wire(), "reason": "initial" if baseline is None else "resume"},
            )
            self._request_header_logged = True
        elif baseline is None or not header_equals(baseline, header):
            _LOG.append(session, "request/header", {"header": header.to_wire(), "reason": "change"})
        # The flag is cleared when the route is in the log, and not when it was
        # read. Otherwise a request that was canceled in the waterfall would
        # take the reroute with it.
        self._rerouted = False

        request_context = RequestContext(
            provider=proposed.provider, model=proposed.model, context_window=resolved.context_window
        )
        if session.request_context() != request_context:
            _LOG.append(session, "request/context", request_context.to_wire())
        self._throw_if_canceled()

        # An effort that was not withheld is unset, `default`, or an effort that
        # the route defines. `default` is not a key of `efforts` (`ModelRoute`),
        # so it sends no fields.
        effort = header.config.reasoning_effort
        return GenerateOptions(
            provider=header.config.provider,
            model=header.config.model,
            messages=session.derive_messages(),
            system=header.system,
            tools=tuple(header.tools or ()),
            effort_fields=None if effort is None else efforts.get(effort),
            sampling=header.config.sampling,
            max_tokens=header.config.max_tokens,
            stop=tuple(header.config.stop or ()),
            session_id=session.id,
        )


def _request_proposal(header: EpochHeader) -> LlmCallConfig:
    """The config that the last request was asked with, for plugins to propose the
    next one from. It is what was sent, with what was withheld asked for again.
    Whether a route carries a setting is the answer of the route at that time. A
    profile that learns to carry a setting therefore sends it, and the user does
    not have to set it twice.
    """
    return header.config.asking(header.withheld) if header.withheld else header.config


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


def _aborted(stop: BaseException, cause: AgentCancelCause | None) -> TurnEndReason:
    """What a stop ends a turn as: `aborted`, whatever a step had decided (C9).

    A step may already have decided how the turn ends when the stop lands:
    while `agent/turn-stopping` runs, or in a step that a steer added. `_turn`
    records every stop by catching `BaseException`. It does not catch
    `anyio.get_cancelled_exc_class()`, because that misses the
    `BaseExceptionGroup` that an anyio task group raises around a cancel, and
    it misses `KeyboardInterrupt` and `SystemExit`.

    **The cause is the one that the `cancel` of the loop was told, or none.** A
    stop from outside has an author that the loop cannot see: the Ctrl-C of a
    user and the shutdown of a host reach the loop as the same cancel. The loop
    used to record `user`, which a shutdown is not. It now names nobody, as the
    `interrupted` closer of repair does. What the loop has in its place is what
    the stop said of itself (`reason_of`), and that is kept as the `detail`. A
    stop that is sent straight to the token of the loop is one of these, and
    its reason is the detail. A host that wants its reason on record gives it
    to `cancel`, or to the scope that it cancels, as `until_signaled` does with
    the name of the signal.
    """
    if cause is not None:
        return TurnEndReason(kind="aborted", reason=cause)
    return TurnEndReason(kind="aborted", detail=reason_of(stop))


def _turn_reason(ends: TurnEndReason | None, capped: bool) -> TurnEndReason:
    """What `turn/end` records, given how the loop left and what it passed through.

    **Finishing does not erase a cap.** A step that hit `max_tokens` cut the
    model off. The turn may then have continued (a steer, a tool call, another
    step) and ended cleanly. Reporting that as `completed` tells the user that
    their answer is whole when part of it was dropped. So `max-tokens` outranks
    `completed`, the ending that means "nothing went wrong".

    It does not outrank the endings that say more. `aborted`, `blocked` and
    `error` each name something that the reader has to act on, and a cap
    somewhere in the turn is the smaller fact beside them.

    **No ending reads as `aborted`, never as `completed`** (C9). Every exit from
    `_turn` assigns an ending first: both `break`s are guarded by
    `turn_ends is not None`, both `return` paths set it, and every raise is
    caught. `None` is therefore a path that nobody has written yet. It used to
    read as `completed`. A turn that finished nothing then told the resume path
    that there was no open turn to repair, and told a reader a story that the
    run did not have.
    """
    if ends is None:
        return TurnEndReason(kind="aborted")
    if ends.kind == "completed":
        return TurnEndReason(kind="max-tokens" if capped else "completed")
    return ends


def _as_failure(error: Exception) -> LlmFailure:
    """A raised error as the `reason.error` of a turn. It keeps its code if it has one.

    The failure keeps the code of a harness error and does not become
    `UNKNOWN`. That is the purpose of `HarnessError`: the routing of a failure
    matters as much as its message, because retry policy, the sandbox layer and
    replay all branch on the code. When every error was flattened, `turn/end`
    could not say which failure it was, in the record that a client reads to
    decide whether a retry could possibly help.
    """
    coded = error_info(error)
    return LlmFailure(
        message=_error_chain(error), code="UNKNOWN" if coded is None else coded["code"]
    )
