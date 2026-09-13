"""`limits` — hard boundaries on a loop that has stopped making progress (P4-04, G5).

Upstream's `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware`, plus the
consecutive-failure breaker the companion plan asks for, expressed as listeners
on seams that already exist: `agent/pre-step` rejects a step, `tools/pre-execute`
denies a call. D12 again — none of it is a parameter on the driver.

**The counts are a fold over the log, not a counter in memory.** A limit that
lives in a field is a limit a resume forgets, and "how many model calls has this
session made" is exactly the question the log already answers: `step/start` per
model call, `tool/call` per tool call, `turn/start` as the reset. Folded through
`SessionFoldCache`, which folds only the slice appended since the last read —
this runs on every step and every call, and a session's log is mostly chunks.

**Two vocabularies, mapped once.** Upstream counts per *thread* (durable across
runs) and per *run* (one invocation). pH's equivalents are the **session** and
the **turn**, and the rename happens here rather than in each message so a diff
against `model_call_limit.py` stays readable.

**`>=`, like upstream.** Both folds hold every call *before* the one being
judged — `step/start` and `tool/call` are appended after their gates decide
(P7-15) — so the comparison is upstream's. (It was `>` while `tool/call` preceded
the gate; the plan row has the story.)

**Ending a turn is a step-boundary decision, not a reach into the batch.**
Upstream's `end` jumps to the graph's end and synthesizes results for the calls
it skipped. pH's batch is already in flight by the time a limit trips, so `end`
denies the offending call, denies its siblings with upstream's own wording, and
lets the *next* `agent/pre-step` reject — the turn closes at the boundary the
loop already has. Nothing has to be un-dispatched.

@module ph_stabilize.limits
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import Field

from ph.agent.types import PreStepDecision, PreStepRequest
from ph.cordis import Context, plugin
from ph.json import as_bool, as_str
from ph.keys import SESSIONS, SUBAGENTS, TUI_STATUS
from ph.llm.types import ToolResultBlock
from ph.seams._registry import contribute_via
from ph.seams.invariants import contribute_fold_cache
from ph.seams.subagents import ADMITTED, SubagentRequest
from ph.seams.tui_status import StatusField, StatusReading
from ph.session import Session, SessionEvent, SessionFoldCache, derive_event_message
from ph.tools.definition import Deny, ToolExecution
from ph.wire import WireModel

from .compaction import TRIGGER_FRACTION

__all__ = [
    "BREAKER_DENIAL",
    "MODEL_LIMIT_MESSAGE",
    "SIBLING_STOPPED",
    "TOOL_DENIAL",
    "WARN_FRACTION",
    "CallBudget",
    "Config",
    "Counts",
    "ModelCallLimitExceeded",
    "ToolCallLimitExceeded",
    "apply",
    "counts_of",
]

log = logging.getLogger("ph_stabilize.limits")

# --------------------------------------------------------------------------
# Verbatim from `langchain.agents.middleware.model_call_limit` and
# `.tool_call_limit` at 1.3.18, with `thread`/`run` renamed to pH's `session`
# and `turn`. Copied rather than imported, as every other ported prompt in this
# bundle is: pH does not depend on langchain, and this is text a model reads.
# --------------------------------------------------------------------------

MODEL_LIMIT_MESSAGE = "Model call limits exceeded: {limits}"
"""`_build_limit_exceeded_message`, whose body joins `turn limit (3/3)`-shaped
clauses with `", "`."""

TOOL_DENIAL = "Tool call limit exceeded. Do not call '{tool}' again."
"""`_build_tool_message_content`, the named-tool branch. Upstream's comment is
worth keeping in view: this text is *for the model*, so it names no limit the
model has no notion of.

Its `tool_name=None` sibling is not ported: nothing here denies "all tools"
without naming one, and an unreachable constant is a claim nobody can check."""

CHILD_DENIAL = "Subagent limit reached: {limits}. No child was created."
"""The child caps' refusal. No upstream wording to copy — the middleware pair
this module ports has no notion of a child — so it says the two things a model
can act on: which ceiling, and that nothing was made."""

SIBLING_STOPPED = (
    "Execution stopped before this tool call could run because another tool "
    "call in the same batch exceeded the limit."
)

BREAKER_DENIAL = (
    "'{tool}' has failed {count} times in a row and is not being called again. "
    "The last error is above. Change the approach — read the file, check the "
    "arguments, or use a different tool — rather than retrying this one."
)
"""pH's own, because upstream has no breaker. Says what happened, where to look,
and what to do instead: a denial that only refuses teaches a model to retry."""


WARN_FRACTION = TRIGGER_FRACTION
"""Where the footer's reading turns amber.

*The* compaction threshold, imported rather than restated: two readings on one
line that turned colour at different fractions would be two things to learn
instead of one, and three copies of `0.85` with three comments asserting they
must stay equal is how they stop being equal."""


class ModelCallLimitExceeded(RuntimeError):
    """`exit: error` on the model-call limit."""


class ToolCallLimitExceeded(RuntimeError):
    """`exit: error` on the tool-call limit."""


# ------------------------------------------------------------------- config --


class CallBudget(WireModel):
    """A per-turn and per-session ceiling. `None` on both means no limit."""

    turn_limit: int | None = None
    """The ceiling for one turn — and a turn is longer than one prompt.

    **A person interjecting joins the running turn rather than starting one.**
    The daemon steers a prompt at a *busy* root so it lands at the next step
    instead of after the whole turn (P5-01), and the loop keeps turning while
    anything is waiting there — so no `turn/start` is appended, and every
    per-turn counter here keeps running. The same is true of a child's message
    (`rlm-messaging`) and of `/autonomous`'s own steer.

    The consequence to plan around: an interjection **shares this ceiling with
    the work already done**, where before it got a fresh one. With
    `modelCalls.turnLimit: 10` against a turn that has spent nine steps, the
    typed line gets one step and then the turn ends as `blocked` — which reads
    as the harness ignoring what was just asked. A deployment that expects
    people to interject mid-fan-out wants this set for the *conversation* a turn
    can become, or set on `sessionLimit` instead, which counts the same work
    either way and cannot be extended by talking.

    It moves the other way for `/autonomous`'s `max_turns`, for the same reason:
    fewer turns carry the same work, so that budget is spent more slowly."""
    session_limit: int | None = None
    """The ceiling for the whole session. Unaffected by where turns are drawn."""

    @property
    def unlimited(self) -> bool:
        return self.turn_limit is None and self.session_limit is None


class ModelCallLimits(CallBudget):
    """How many model calls a turn and a session may make."""

    exit: Literal["end", "error"] = "end"
    """`end` closes the turn as `blocked` and records why; `error` raises.

    No `continue`, and upstream has none either: a model call that is not made
    cannot be continued past — the step *is* the call."""


class ToolCallLimits(CallBudget):
    """How many tool calls a turn and a session may make."""

    per_tool: dict[str, CallBudget] = Field(default_factory=dict)
    """Budgets for named tools, checked beside the aggregate above.

    Upstream mounts one middleware instance per tool with its own limits; one
    table says the same thing without a row per tool."""
    exit: Literal["continue", "end", "error"] = "continue"


class ChildLimits(CallBudget):
    """How many children an agent may spawn, per turn and per session.

    Folded from the parent's own `subagent/admitted` records, so a resumed
    session keeps its count, and enforced as a `ctx.subagents.guard` before the
    provider is asked, so a refused spawn creates nothing. Every ceiling is unset
    by default, for the reason the others are.

    **No live ceiling, deliberately.** How many children *run at once* is a
    question about resources, not about what the parent may ask for, and a
    refusal is the wrong instrument for it: the parent asked for nine. The
    subagent provider answers it with a queue (`rlm-subagent-provider`'s
    `maxConcurrent`), and this row keeps to the totals."""


class BreakerConfig(WireModel):
    """The consecutive-failure breaker."""

    consecutive_failures: int | None = 5
    """`None` disables it. Counted per tool and reset by any success, so a tool
    that works intermittently never trips — the failure this catches is a model
    retrying one broken call until the budget is gone."""


class Config(WireModel):
    """Row config.

    **Every ceiling here is unset by default and the breaker ships on**, which is
    the row's whole posture: a limit nobody chose fires on somebody's longest
    legitimate turn, while five identical failures in a row is not a long task
    but a stuck one. A deployment that wants a ceiling says so.

    The four options nest, so `ph config` shows each as one line with its
    sub-fields in the default — the per-field reasoning lives on those fields
    (`CallBudget.turn_limit` is the one with a consequence worth reading before
    setting it).
    """

    # No docstrings here, deliberately: each of these *is* another config model,
    # and `ph config` takes an option's summary from the model that defines it
    # when the field says nothing (`catalog._nested_doc`). A paragraph here would
    # paraphrase the four classes below and go stale against them first, in the
    # one command whose claim is that it cannot drift from the code.
    model_calls: ModelCallLimits = ModelCallLimits()
    tool_calls: ToolCallLimits = ToolCallLimits()
    children: ChildLimits = ChildLimits()
    breaker: BreakerConfig = BreakerConfig()


# -------------------------------------------------------------- the counting --


@dataclass(frozen=True, slots=True)
class Counts:
    """Everything the limits ask, folded from one pass over the log."""

    session_steps: int = 0
    turn_steps: int = 0
    session_tools: int = 0
    turn_tools: int = 0
    session_children: int = 0
    turn_children: int = 0
    per_tool_session: Mapping[str, int] = field(default_factory=dict)
    per_tool_turn: Mapping[str, int] = field(default_factory=dict)
    consecutive_failures: Mapping[str, int] = field(default_factory=dict)
    names_by_call: Mapping[str, str] = field(default_factory=dict)
    """`callId` → tool name, so a `tool/result` can be attributed.

    Carried in the fold rather than looked up per result: the result payload
    names the call it answers but not the tool that ran, and a second scan to
    find out would be the fold done twice."""


_COUNTED = frozenset(
    {
        "turn/start",
        "step/start",
        "tool/call",
        "tool/result",
        "tool/code-dispatch-start",
        "tool/code-dispatch",
        ADMITTED,
    }
)
"""The only event types this fold reads. Named so the slice can be filtered
before anything is copied.

**A Code Mode dispatch is a tool call** (C1, C2). A `tools.x(...)` from inside a
cell logs `tool/code-dispatch-start` rather than `tool/call`, and the fold read
only the latter — so every dispatch was *judged* at `tools/pre-execute` against a
count that never moved, and a per-tool budget or the breaker could not bind on
the door P3-21 (b) said Phase 4 would count. Both records land after the gate
decides, as `tool/call` does since P7-15, so the two doors share one arithmetic."""


def _extend(previous: Counts, session: Session, from_seq: int) -> Counts:
    """Fold the slice appended since `from_seq` onto an earlier count.

    A left fold with a reset, which is what makes it extendable: `turn/start`
    zeroes the turn counters wherever it appears, so folding a prefix and then
    the rest gives the same answer as folding the whole log.

    Takes the *session* and slices its events, which is the shape
    `SessionFoldCache` hands an `extend` — `from_seq` is the log length when the
    cached value was computed, so `events[from_seq:]` is exactly what is new.
    """
    slice_ = [event for event in session.events_from(from_seq) if event.type in _COUNTED]
    if not slice_:
        # The same object, so a caller can tell nothing changed — and, far more
        # often, so the four dict copies below are not paid to fold a slice that
        # is entirely `assistant/chunk`. `session.seq` bumps on every event, so
        # the cache misses on almost every read while a model is streaming.
        return previous
    session_steps, turn_steps = previous.session_steps, previous.turn_steps
    session_tools, turn_tools = previous.session_tools, previous.turn_tools
    session_children, turn_children = previous.session_children, previous.turn_children
    per_session = dict(previous.per_tool_session)
    per_turn = dict(previous.per_tool_turn)
    failures = dict(previous.consecutive_failures)
    names = dict(previous.names_by_call)

    for event in slice_:
        if event.type == "turn/start":
            turn_steps = 0
            turn_tools = 0
            turn_children = 0
            per_turn = {}
            # A call id from a finished turn can never be answered, so the map
            # that carries a name from a call to its settle is a turn's. Left to
            # the settle alone it is not bounded: `_log_settle` is not a
            # `finally`, so a dispatch that raises writes its start and never its
            # end, and that entry is then copied on every fold for the session's
            # life (~4.7 µs at zero, ~8.8 µs at two thousand).
            names = {}
        elif event.type == "step/start":
            session_steps += 1
            turn_steps += 1
        elif event.type in ("tool/call", "tool/code-dispatch-start"):
            name = as_str(event.data.get("name"))
            # `callId` for a model's call, `subCallId` for a dispatch: the id the
            # settling record will cite, so the breaker can pair them.
            call_id = as_str(event.data.get("callId") or event.data.get("subCallId"))
            if call_id:
                names[call_id] = name
            session_tools += 1
            turn_tools += 1
            per_session[name] = per_session.get(name, 0) + 1
            per_turn[name] = per_turn.get(name, 0) + 1
        elif event.type in ("tool/result", "tool/code-dispatch"):
            call_id, is_error = _result_facts(event)
            # Popped, not read: the map exists to carry a name from a call to
            # the settle that answers it, and both are in the same turn — which
            # is also the reset above, for the dispatch that never settles.
            name = names.pop(call_id, "")
            if name:
                failures[name] = failures.get(name, 0) + 1 if is_error else 0
        elif event.type == ADMITTED:
            session_children += 1
            turn_children += 1
    return Counts(
        session_steps=session_steps,
        turn_steps=turn_steps,
        session_tools=session_tools,
        turn_tools=turn_tools,
        session_children=session_children,
        turn_children=turn_children,
        per_tool_session=per_session,
        per_tool_turn=per_turn,
        consecutive_failures=failures,
        names_by_call=names,
    )


def _result_facts(event: SessionEvent) -> tuple[str, bool]:
    """The call a settled call answers, and whether it failed — both doors.

    Through `derive_event_message` — THE projection — rather than by indexing
    the payload. This module was the *fourth* reader of that shape and reached
    the call id by a different route than the other three, so a change to
    `_append_result` would have left the breaker silently counting nothing.
    Paid once per settle, because the fold visits each event once.

    A Code Mode dispatch settles as `tool/code-dispatch`, which is its own shape
    and not a `ToolResultBlock` — so the branch is here, where this module's one
    reader of "what did a settle say" already lives, rather than in the fold. The
    start branch unified the same way, on `callId or subCallId`.
    """
    if event.type == "tool/code-dispatch":
        return as_str(event.data.get("subCallId")), as_bool(event.data.get("isError"))
    message = derive_event_message(event)
    block = next(
        (one for one in (message.content if message else ()) if isinstance(one, ToolResultBlock)),
        None,
    )
    return ("", False) if block is None else (block.tool_call_id, bool(block.is_error))


def counts_of(session: Session) -> Counts:
    """The counts folded from scratch — the cache's cold path, and a test's."""
    return _extend(Counts(), session, 0)


# ------------------------------------------------------------- the breaches --


def _exceeded(budget: CallBudget | None, turn: int, session: int) -> list[tuple[str, int, int]]:
    """`(scope, used, limit)` for every ceiling the call being judged would pass."""
    if budget is None:
        return []
    pairs = (("turn", turn, budget.turn_limit), ("session", session, budget.session_limit))
    return [
        (scope, used, limit) for scope, used, limit in pairs if limit is not None and used >= limit
    ]


def _breaches(budget: CallBudget, turn: int, session: int) -> list[str]:
    """`ModelCallLimitMiddleware`'s phrasing: `turn limit (3/3)`."""
    return [
        f"{scope} limit ({used}/{limit})" for scope, used, limit in _exceeded(budget, turn, session)
    ]


def _over(budget: CallBudget | None, turn: int, session: int, *, noun: str = "calls") -> list[str]:
    """`ToolCallLimitMiddleware`'s: `turn limit exceeded (4/3 calls)`.

    `used + 1`: the sentence counts the call being refused, as upstream's does.
    `noun` because the same arithmetic bounds children (P4-04)."""
    return [
        f"{scope} limit exceeded ({used + 1}/{limit} {noun})"
        for scope, used, limit in _exceeded(budget, turn, session)
    ]


def _record(session: Session, kind: str, detail: dict[str, Any]) -> None:
    session.append("limits/exceeded", {"limit": kind, **detail})


@plugin("limits", inject=[SESSIONS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Arm the model-call limit, the tool-call limit and the breaker."""
    counts = SessionFoldCache(counts_of, extend=_extend)
    # The cache is a local rather than a service, so the poll is declared where it
    # is built. It has the strongest claim of the six to being polled: this fold
    # is the only thing standing between a looping agent and an unbounded run, and
    # every way it could drift drifts *low* — the direction that stops refusing.
    contribute_fold_cache(ctx, id="limits-fold-cache", subject="limit count", stale=counts.stale)

    # ------------------------------------------------------- model calls --

    async def on_pre_step(
        request: PreStepRequest,
        next_: Callable[..., Awaitable[PreStepDecision]],
    ) -> PreStepDecision:
        settings = config.model_calls
        session = request.session
        if settings.unlimited:
            return await next_(request)
        current = counts.read(session)
        exceeded = _breaches(settings, current.turn_steps, current.session_steps)
        if not exceeded:
            return await next_(request)
        message = MODEL_LIMIT_MESSAGE.format(limits=", ".join(exceeded))
        if settings.exit == "error":
            raise ModelCallLimitExceeded(message)
        _record(
            session,
            "model-calls",
            {"turn": current.turn_steps, "session": current.session_steps, "message": message},
        )
        # Vetoed: returning without `next_` is how a policy row owns this
        # decision, and it stops every later listener doing work for a step that
        # will not happen.
        return PreStepDecision(kind="reject", reason=message)

    # -------------------------------------------------------- tool calls --

    async def on_pre_execute(
        execution: ToolExecution,
        next_: Callable[..., Awaitable[Any]],
    ) -> Any:  # noqa: ANN401
        session = execution.session
        settings = config.tool_calls
        breaker = config.breaker.consecutive_failures
        if session is None or (settings.unlimited and not settings.per_tool and breaker is None):
            return await next_(execution)
        current = counts.read(session)
        return _deny(session, execution, current) or await next_(execution)

    def _deny(session: Session, execution: ToolExecution, current: Counts) -> Deny | None:
        settings = config.tool_calls
        name = execution.name
        turn, in_session = current.per_tool_turn.get(name, 0), current.per_tool_session.get(name, 0)
        exceeded = [
            *_over(settings, current.turn_tools, current.session_tools),
            *_over(settings.per_tool.get(name), turn, in_session),
        ]
        if not exceeded:
            return _breaker(session, execution, current)
        if settings.exit == "error":
            raise ToolCallLimitExceeded(f"'{name}' call limit reached: {', '.join(exceeded)}.")
        if settings.exit == "continue":
            return Deny(reason=TOOL_DENIAL.format(tool=name))
        if not _is_first_over(settings, current, name):
            # A sibling of the call that breached, told apart by arithmetic
            # rather than by a latch: the breaching call is the one that found
            # the count *at* a ceiling. Upstream's wording, because this call
            # did nothing wrong and the model should not read the denial as
            # being about it.
            return Deny(reason=SIBLING_STOPPED, concludes_turn=True)
        _record(
            session,
            "tool-calls",
            {
                "tool": name,
                "message": f"'{name}' tool call limit reached: {' and '.join(exceeded)}.",
            },
        )
        # The loop's own way to end a turn, rather than a second one: the batch
        # in flight still settles, and nothing after it runs.
        return Deny(reason=TOOL_DENIAL.format(tool=name), concludes_turn=True)

    def _is_first_over(settings: ToolCallLimits, current: Counts, name: str) -> bool:
        """Whether this call is the one that crossed a ceiling, not a follower."""
        counted = (
            (settings.turn_limit, current.turn_tools),
            (settings.session_limit, current.session_tools),
            *(
                (limit, used)
                for budget in (settings.per_tool.get(name),)
                if budget is not None
                for limit, used in (
                    (budget.turn_limit, current.per_tool_turn.get(name, 0)),
                    (budget.session_limit, current.per_tool_session.get(name, 0)),
                )
            ),
        )
        return any(limit is not None and used == limit for limit, used in counted)

    # ----------------------------------------------------------- breaker --

    def _breaker(session: Session, execution: ToolExecution, current: Counts) -> Deny | None:
        limit = config.breaker.consecutive_failures
        failures = current.consecutive_failures.get(execution.name, 0)
        if limit is None or failures < limit:
            return None
        session.append(
            "limits/breaker-tripped",
            {"tool": execution.name, "failures": failures, "limit": limit},
        )
        return Deny(reason=BREAKER_DENIAL.format(tool=execution.name, count=failures))

    # ------------------------------------------------------ the footer --

    def _reading(session: Session) -> StatusReading | None:
        """The tightest active budget, as one short reading (P4-04).

        A *live* number rather than only the notice that lands when a budget is
        spent: upstream announces the limit on the step it stops you, which is
        the one moment the information can no longer change anything. This is
        the same argument the context gauge already makes one field over.

        Read on every redraw, so it is guarded by the cheapest possible question
        first and then rides the same `SessionFoldCache` the limits themselves
        read — which returns its previous value outright when the events since
        the last read were all chunks.
        """
        model, tools = config.model_calls, config.tool_calls
        if model.unlimited and tools.unlimited:
            return None
        current = counts.read(session)
        gauges = (
            ("steps", current.turn_steps, model.turn_limit),
            ("steps", current.session_steps, model.session_limit),
            ("tools", current.turn_tools, tools.turn_limit),
            ("tools", current.session_tools, tools.session_limit),
        )
        tightest = max(
            ((used / limit, label, used, limit) for label, used, limit in gauges if limit),
            default=None,
        )
        if tightest is None:
            return None
        fraction, label, used, limit = tightest
        # Amber from the same fraction the context gauge warns at, so the two
        # readings mean the same thing by the time a person has learned one.
        return StatusReading(
            text=f"{label} {used}/{limit}",
            level="warning" if fraction >= WARN_FRACTION else "normal",
        )

    status = ctx.get(TUI_STATUS)
    if status is not None:
        status.register(StatusField(id="limits", read=_reading, order=10), scope=ctx)

    ctx.on("agent/pre-step", on_pre_step)
    ctx.on("tools/pre-execute", on_pre_execute)

    # ----------------------------------------------------------- children --

    def refuse_child(request: SubagentRequest) -> str | None:
        """The child caps, asked before every admission (P4-04)."""
        settings = config.children
        session = request.parent.session
        if session is None:
            return None
        current = counts.read(session)
        exceeded = _over(settings, current.turn_children, current.session_children, noun="children")
        if not exceeded:
            return None
        message = CHILD_DENIAL.format(limits=" and ".join(exceeded))
        _record(
            session,
            "children",
            {
                "turn": current.turn_children,
                "session": current.session_children,
                "message": message,
            },
        )
        return message

    # `contribute_via`, which carries the wait-for-the-key rule and its reason:
    # a profile that mounts no subagent seam has nothing to cap and must not lose
    # its call limits for want of one, and either mount order has to work. Not
    # registered at all when nothing is capped, so a spawn pays no guard for a
    # ceiling nobody chose.
    if not config.children.unlimited:
        contribute_via(
            ctx,
            SUBAGENTS,
            lambda subagents, scope: subagents.guard(refuse_child, scope=scope),
            label="limits-children",
        )
