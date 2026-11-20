"""`limits`: hard boundaries on a loop that has stopped making progress (P4-04, G5).

This module ports `ModelCallLimitMiddleware` and `ToolCallLimitMiddleware` from
upstream, and adds the consecutive-failure breaker that the companion plan asks
for. All three are listeners on seams that already exist:

* `agent/pre-step` rejects a step.
* `agent/request-error` refuses a retry.
* `tools/pre-execute` denies a call.

This is D12 again: none of it is a parameter on the driver.

**The counts are a fold over the log, not a counter in memory.** A resume
forgets a limit that is held in a field. The log already answers the question
"how many model calls has this session made":

* `step/start` records the first model call of a step.
* `step/retry` records each call that is made again.
* `tool/call` records each tool call.
* `turn/start` is the reset.

The fold goes through `SessionFoldCache`, which folds only the slice that was
appended since the last read. The fold runs on every step and every call, and
most of the log of a session is chunks.

**Children are counted in their own logs, not in the log of the parent**
(Phase 11). The `subagent/admitted` event of a child is in the log of the child.
It is stamped with the seq of the latest `turn/start` of the parent. The child
caps therefore ask the seam for the count when a spawn is judged
(`SubagentService.child_counts`). The fold above stays a fold over one log.

**Two vocabularies, mapped one time.** Upstream counts per *thread* (durable
across runs) and per *run* (one invocation). The pH equivalents are the
**session** and the **turn**. The rename happens here and not in each message,
so that a diff against `model_call_limit.py` stays readable.

**`>=`, like upstream.** Both folds hold every call *before* the call that is
being judged, because `step/start` and `tool/call` are appended after their
gates decide (P7-15). The comparison is therefore the same as upstream's. (It
was `>` while `tool/call` came before the gate. The plan row has the history.)

**Ending a turn is a decision at a boundary that the loop already has. It does
not reach into the batch.** The `end` of upstream jumps to the end of the graph
and synthesizes results for the calls that it skipped. In pH the batch is
already in flight when a limit trips. `end` therefore denies the offending call
with `Deny(concludes_turn=True)`, and denies the siblings of that call with the
wording of upstream. The batch in flight still settles, and the loop then closes
the turn. No call has to be un-dispatched.

@module ph_stabilize.limits
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal

from pydantic import Field

from ph.agent.types import (
    PreStepDecision,
    PreStepRequest,
    RequestErrorAction,
    RequestFailure,
)
from ph.cordis import Context, Next, plugin
from ph.json import as_str
from ph.keys import SESSIONS, SUBAGENTS, TUI_STATUS
from ph.seams._registry import contribute_via
from ph.seams.invariants import contribute_fold_cache
from ph.seams.subagents import SubagentRequest, SubagentService
from ph.seams.tui_status import StatusField, StatusReading
from ph.session import Session, SessionFoldCache, settle_of
from ph.session.writers import log_writer
from ph.tools import TOOL_DISPATCH_EVENT_TYPES, call_id_of
from ph.tools.definition import Deny, PreToolDecision, ToolExecution
from ph.wire import WireModel

from .compaction import TRIGGER_FRACTION

_LOG = log_writer(__name__)

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
    "apply",
    "counts_of",
]

log = logging.getLogger("ph_stabilize.limits")

# --------------------------------------------------------------------------
# Copied verbatim from `langchain.agents.middleware.model_call_limit` and
# `.tool_call_limit` at 1.3.18, with `thread`/`run` renamed to the pH terms
# `session` and `turn`. The text is copied and not imported, like every other
# ported prompt in this bundle. pH does not depend on langchain, and a model
# reads this text.
# --------------------------------------------------------------------------

MODEL_LIMIT_MESSAGE = "Model call limits exceeded: {limits}"
"""From `_build_limit_exceeded_message`. The body of that function joins clauses
of the form `turn limit (3/3)` with `", "`."""

TOOL_DENIAL = "Tool call limit exceeded. Do not call '{tool}' again."
"""From `_build_tool_message_content`, the branch for a named tool. The upstream
comment is worth remembering: this text is *for the model*, so it names no limit
that the model does not know about.

The `tool_name=None` branch is not ported. Nothing here denies "all tools"
without naming one, and an unreachable constant would be a claim that nobody can
check."""

CHILD_DENIAL = "Subagent limit reached: {limits}. No child was created."
"""The refusal of the child caps. There is no upstream wording to copy, because
the two middlewares that this module ports have no notion of a child. The text
therefore says the two things that a model can act on: which ceiling was
reached, and that nothing was created."""

SIBLING_STOPPED = (
    "Execution stopped before this tool call could run because another tool "
    "call in the same batch exceeded the limit."
)

BREAKER_DENIAL = (
    "'{tool}' has failed {count} times in a row and is not being called again. "
    "The last error is above. Change the approach — read the file, check the "
    "arguments, or use a different tool — rather than retrying this one."
)
"""Written for pH, because upstream has no breaker. It says what happened, where
to look, and what to do instead. A denial that only refuses teaches a model to
retry."""


WARN_FRACTION = TRIGGER_FRACTION
"""The fraction at which the footer reading turns amber.

This is the compaction threshold itself, imported and not restated. Two readings
on one line that changed color at different fractions would be two things to
learn and not one. Three copies of `0.85`, with three comments that say the
copies must stay equal, are how the copies stop being equal."""


class ModelCallLimitExceeded(RuntimeError):
    """`exit: error` on the model-call limit."""


# ------------------------------------------------------------------- config --


class CallBudget(WireModel):
    """A per-turn and per-session ceiling. `None` on both means no limit."""

    turn_limit: int | None = None
    """The ceiling for one turn. A turn can be longer than one prompt.

    **A user who interjects joins the running turn and does not start a new
    one.** The daemon steers a prompt at a *busy* root so that the prompt lands
    at the next step, and not after the whole turn (P5-01). The loop keeps
    running while anything waits there. No `turn/start` is appended, so every
    per-turn counter here keeps running. The same applies to a message from a
    child (`rlm-messaging`) and to the steer that `/autonomous` itself sends.

    The consequence to plan for: an interjection **shares this ceiling with the
    work already done**. Before, an interjection got a new ceiling. With
    `modelCalls.turnLimit: 10` and a turn that has used nine steps, the typed
    line gets one step and then the turn ends as `blocked`. To the user, the
    harness appears to ignore the request that was just made. A deployment that
    expects users to interject during a fan-out has two options. It can set this
    limit for the *conversation* that a turn can become. It can set
    `sessionLimit` instead, which counts the same work in both cases and cannot
    be extended by talking.

    The effect is the opposite for the `max_turns` of `/autonomous`, for the
    same reason: fewer turns carry the same work, so that budget is used more
    slowly."""
    session_limit: int | None = None
    """The ceiling for the whole session. Turn boundaries do not affect it."""

    @property
    def unlimited(self) -> bool:
        return self.turn_limit is None and self.session_limit is None


class ModelCallLimits(CallBudget):
    """How many model calls a turn and a session may make."""

    exit: Literal["end", "error"] = "end"
    """`end` closes the turn as `blocked` and records the reason. `error` raises.

    A retry is also a model call (P1). When the ceiling refuses a retry, the
    failure stands. The turn then ends with the error of the provider itself,
    and the record says why the call was not retried.

    There is no `continue`, and upstream has none either: nothing can continue
    past a model call that is not made."""


class ToolCallLimits(CallBudget):
    """How many tool calls a turn and a session may make."""

    per_tool: dict[str, CallBudget] = Field(default_factory=dict)
    """Budgets for named tools, checked together with the aggregate budget above.

    Upstream mounts one middleware instance for each tool, each with its own
    limits. One table says the same thing without a row for each tool."""
    exit: Literal["continue", "end", "error"] = "continue"
    """What a call over the ceiling does. There are three readings. They differ
    in *who is told*, not in how hard they stop.

    `continue` denies this call and lets the turn go on. The model reads a
    denial, which is a policy decision, and may do something else. `end` denies
    the call and concludes the turn. `error` also concludes the turn, and the
    model reads a **spent budget** and not a policy decision
    (`TOOL_BUDGET_SPENT`, `kind="failed"`), because nothing judged the call.

    **Each of the three either stops the turn or does not, and says which fact
    it was** (D7). The three used to differ on both axes at the same time.
    `error` raised out of `tools/pre-execute`. The pipeline there turns any
    exception from a policy row into a failed result. That is deliberate, so
    that a broken row cannot take down the call that it was asked about. The
    ceiling inherited that behavior by accident. The turn continued, and every
    later call got the same failure. Under Code Mode, a generated program could
    `except` the failure and keep calling. The posture that was meant to report
    a breach most loudly was therefore the one that a program could suppress.

    The fix was one field, not a new exception contract (N2). The loop already
    ends a turn when a policy row says so: `end` does it through
    `Deny(concludes_turn=True)`. The only thing that `error` adds over `end` is
    the `FailureKind` that the model reads, and `Deny.failure_kind` now carries
    it. The two stay separate because the *readings* are meant to differ: `end`
    is a policy refusal, and `error` is a spent budget.
    `code_mode.CodeRunFailure` and `fs.FsDenied`/`fs.FileTooLarge` already make
    the same distinction."""


class ChildLimits(CallBudget):
    """How many children an agent may spawn, per turn and per session.

    The count comes from the admissions of the children themselves, plus the
    spawns that are in progress (`ctx.subagents.child_counts`). A resumed
    session therefore keeps its count after its children are loaded, and the
    resume sweep loads them first. The limit is enforced as a
    `ctx.subagents.guard` before the provider is asked, so a refused spawn
    creates nothing. Every ceiling is unset by default, for the same reason as
    the other ceilings.

    **A deleted child still counts.** A delete frees nothing that this budget
    limits: the spawn was requested and made. A delete also never reduced the
    count when the log of the parent held the admissions.

    **A fork starts at zero** (decision 4). A fork is a new root, and no child
    names it as its parent. The count used to be folded from the log of the
    parent from seq 0, including the seeded prefix. A fork therefore inherited
    the children of its source and spent them again. Only this budget changed:
    the model calls and tool calls of a fork are still folded over its seeded
    prefix.

    **There is deliberately no ceiling on live children.** How many children
    *run at the same time* is a question about resources, not about what the
    parent may request. A refusal is the wrong tool for that question, because
    the parent asked for nine. The subagent provider answers it with a queue
    (the `maxConcurrent` of `rlm-subagent-provider`), and this row limits only
    the totals."""

    exit: Literal["continue", "end", "error"] = "continue"
    """What a spawn over the ceiling does, in the terms of `ToolCallLimits.exit`
    (D15).

    This budget used to have one posture, with this argument for it: "a child
    that is not created cannot be continued past or raised about". The first
    half is also true of a tool call that is not executed, and D7 still gave
    that ceiling three readings. The argument therefore did not separate the
    two ceilings. This was the last ceiling on the posture that D7 was filed to
    replace.

    The three values mean here exactly what they mean there. `continue` refuses
    the spawn and lets the turn go on. That is what this budget always did, and
    it stays the default. `end` refuses the spawn and concludes the turn.
    `error` also concludes the turn, and the model reads a spent budget and not
    a policy decision (`TOOL_BUDGET_SPENT`, `kind="failed"`), because nothing
    judged the request.

    A refusal still creates nothing, whichever value is chosen: the guard runs
    before the provider is asked. That half of the old argument was always
    about *this* budget and not about postures."""


class BreakerConfig(WireModel):
    """The consecutive-failure breaker."""

    consecutive_failures: int | None = 5
    """`None` disables the breaker. Failures are counted per tool, and any
    success resets the count, so a tool that works intermittently never trips
    the breaker. The failure that the breaker catches is a model that retries
    one broken call until the budget is gone."""


class Config(WireModel):
    """Row config.

    **Every ceiling here is unset by default, and the breaker is on by
    default.** That is the whole posture of the row. A limit that nobody chose
    fires on the longest legitimate turn of some user. Five identical failures
    in a row are not a long task but a stuck one. A deployment that wants a
    ceiling sets one.

    The four options nest, so `phern config` shows each option as one line with
    its sub-fields in the default. The reasoning for each field is on that
    field. `CallBudget.turn_limit` has a consequence that is worth reading
    before you set it.
    """

    # These fields deliberately have no docstrings. Each one *is* another config
    # model. When a field says nothing, `phern config` takes the summary of the
    # option from the model that defines it (`catalog._nested_doc`). A paragraph
    # here would paraphrase the four classes that these fields name, and it
    # would go stale against them first. `phern config` is the one command that
    # claims it cannot drift from the code.
    model_calls: ModelCallLimits = ModelCallLimits()
    tool_calls: ToolCallLimits = ToolCallLimits()
    children: ChildLimits = ChildLimits()
    breaker: BreakerConfig = BreakerConfig()


# -------------------------------------------------------------- the counting --


@dataclass(frozen=True, slots=True)
class Counts:
    """Everything that the limits need from the log of the session itself, folded
    in one pass.

    The children are not included. They are counted in their own logs
    (`SubagentService.child_counts`)."""

    session_calls: int = 0
    turn_calls: int = 0
    """Model calls: the first call of each step (`step/start`) and every retry of
    one (`step/retry`, P1). Both events record the same fact, a model call, at
    the two places where one is made, so there is one count."""
    session_tools: int = 0
    turn_tools: int = 0
    per_tool_session: Mapping[str, int] = field(default_factory=dict)
    per_tool_turn: Mapping[str, int] = field(default_factory=dict)
    consecutive_failures: Mapping[str, int] = field(default_factory=dict)
    names_by_call: Mapping[str, str] = field(default_factory=dict)
    """Maps a `callId` to a tool name, so that a `tool/result` can be attributed.

    The fold carries the map, so no lookup is needed for each result. The result
    payload names the call that it answers but not the tool that ran. A second
    scan to find the tool would repeat the fold."""


_COUNTED = frozenset(
    {
        "turn/start",
        "step/start",
        "step/retry",
        *TOOL_DISPATCH_EVENT_TYPES,
        "tool/result",
        "tool/code-dispatch",
    }
)
"""The only event types that this fold reads. They are named so that the slice
can be filtered before anything is copied.

**A Code Mode dispatch is a tool call** (C1, C2). A `tools.x(...)` call from
inside a cell logs `tool/code-dispatch-start` and not `tool/call`, and the fold
used to read only `tool/call`. Every dispatch was therefore *judged* at
`tools/pre-execute` against a count that never moved. A per-tool budget or the
breaker could not apply to the transport that P3-21 (b) said Phase 4 would
count. Both records are appended after the gate decides, as `tool/call` has been
since P7-15, so the two transports share one arithmetic."""


def _extend(previous: Counts, session: Session, from_seq: int) -> Counts:
    """Fold the slice that was appended since `from_seq` onto an earlier count.

    This is a left fold with a reset, and the reset makes it extendable.
    `turn/start` sets the turn counters to zero wherever it appears. Folding a
    prefix and then the rest therefore gives the same answer as folding the
    whole log.

    The function takes the *session* and slices its events, because that is the
    shape that `SessionFoldCache` passes to an `extend`. `from_seq` is the
    length of the log when the cached value was computed, so `events[from_seq:]`
    is exactly the new part.
    """
    slice_ = [event for event in session.events_from(from_seq) if event.type in _COUNTED]
    if not slice_:
        # Return the same object, so that a caller can tell that nothing
        # changed. Far more often, this avoids the four dict copies below for a
        # slice that is entirely `assistant/chunk`. `session.seq` increases on
        # every event, so the cache misses on almost every read while a model is
        # streaming.
        return previous
    session_calls, turn_calls = previous.session_calls, previous.turn_calls
    session_tools, turn_tools = previous.session_tools, previous.turn_tools
    per_session = dict(previous.per_tool_session)
    per_turn = dict(previous.per_tool_turn)
    failures = dict(previous.consecutive_failures)
    names = dict(previous.names_by_call)

    for event in slice_:
        if event.type == "turn/start":
            turn_calls = 0
            turn_tools = 0
            per_turn = {}
            # A call id from a finished turn can never be answered. The map
            # carries a name from a call to its settle, so the map belongs to
            # one turn. The settle alone did not bound the map. `_log_settle`
            # was not in a `finally`, so a dispatch that raised wrote its start
            # and never its end. That entry was then copied on every fold for
            # the life of the session (~4.7 µs at zero, ~8.8 µs at two
            # thousand). `code_mode._abandon` now settles such a dispatch. This
            # reset still bounds the map for any call that stays open.
            names = {}
        elif event.type in ("step/start", "step/retry"):
            session_calls += 1
            turn_calls += 1
        elif event.type in TOOL_DISPATCH_EVENT_TYPES:
            name = as_str(event.data.get("name"))
            # `callId` for a call from the model, `subCallId` for a dispatch.
            # This is the id that the settling record cites, so the breaker can
            # pair the two records.
            call_id = call_id_of(event)
            if call_id:
                names[call_id] = name
            session_tools += 1
            turn_tools += 1
            per_session[name] = per_session.get(name, 0) + 1
            per_turn[name] = per_turn.get(name, 0) + 1
        elif (settled := settle_of(event)) is not None:
            call_id, is_error = settled
            # The name is popped, not read. The map exists to carry a name from
            # a call to the settle that answers it, and both are in the same
            # turn. For a dispatch that never settles, the reset above removes
            # the name.
            name = names.pop(call_id, "")
            if name:
                failures[name] = failures.get(name, 0) + 1 if is_error else 0
    return Counts(
        session_calls=session_calls,
        turn_calls=turn_calls,
        session_tools=session_tools,
        turn_tools=turn_tools,
        per_tool_session=per_session,
        per_tool_turn=per_turn,
        consecutive_failures=failures,
        names_by_call=names,
    )


def counts_of(session: Session) -> Counts:
    """The counts, folded from the start of the log. The cold path of the cache
    uses this function, and so do the tests."""
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
    """The phrasing of `ModelCallLimitMiddleware`: `turn limit (3/3)`."""
    return [
        f"{scope} limit ({used}/{limit})" for scope, used, limit in _exceeded(budget, turn, session)
    ]


def _over(budget: CallBudget | None, turn: int, session: int, *, noun: str = "calls") -> list[str]:
    """The phrasing of `ToolCallLimitMiddleware`: `turn limit exceeded (4/3 calls)`.

    `used + 1`: the sentence counts the call that is refused, as the upstream
    sentence does. `noun` exists because the same arithmetic limits children
    (P4-04)."""
    return [
        f"{scope} limit exceeded ({used + 1}/{limit} {noun})"
        for scope, used, limit in _exceeded(budget, turn, session)
    ]


def _record(session: Session, kind: str, posture: str, detail: dict[str, Any]) -> None:
    """Record one breach in the log (D7/N2). The record reaches disk at the next
    barrier: the flush of the pre-step reject, or the flush of the next request.

    **`posture` is not decoration.** `limits/exceeded` had come to mean two
    different things: a breach that ended the turn, and a breach that did not.
    Nothing on the event told them apart, and the third posture wrote nothing
    at all. A reviewer, `phern doctor` and a resumed session all read this
    record and not the tool result. "The ceiling was hit" is a different fact
    from "and here is what happened next".
    """
    _LOG.append(session, "limits/exceeded", {"limit": kind, "posture": posture, **detail})


@plugin("limits", affects="environment", inject=[SESSIONS], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Arm the model-call limit, the tool-call limit, the breaker and the child caps."""
    counts = SessionFoldCache(counts_of, extend=_extend)
    # The cache is a local and not a service, so the poll is declared where the
    # cache is built. Of the fold caches that declare a poll, this one has the
    # strongest claim to it. This fold is the only thing between a looping agent
    # and an unbounded run, and every way that it could drift makes it drift
    # *low*, which is the direction that stops refusing. The declaration also
    # lets the counts of a session go away with the session. This row never
    # subscribed to `session/disposed`, and it kept the counts for every child
    # that a long-lived root ever ran.
    contribute_fold_cache(ctx, id="limits-fold-cache", subject="limit count", stale=counts.stale)

    # ------------------------------------------------------- model calls --

    def _model_breach(session: Session, **detail: object) -> str | None:
        """The sentence of the model-call ceiling if the next call would pass it.
        The breach is recorded before the sentence is returned.

        Returns `None` while there is room. A model call is made in two places:
        a new step and a retry. Both use this one check, so the way that a
        breach is recorded cannot come to differ between them, as it did for the
        tool limit before D7.

        **The breach is recorded before it is raised** (N2). D7 made this change
        on the tool limit and left the model-call limit as it was. The one
        posture that reports the breach most loudly to the model was therefore
        still the one that left no trace of it in the log. A raise unwinds the
        turn, and an unwound turn appends nothing.
        """
        settings = config.model_calls
        if settings.unlimited:
            return None
        current = counts.read(session)
        exceeded = _breaches(settings, current.turn_calls, current.session_calls)
        if not exceeded:
            return None
        message = MODEL_LIMIT_MESSAGE.format(limits=", ".join(exceeded))
        _record(
            session,
            "model-calls",
            settings.exit,
            {
                "turn": current.turn_calls,
                "session": current.session_calls,
                "message": message,
                **detail,
            },
        )
        if settings.exit == "error":
            raise ModelCallLimitExceeded(message)
        return message

    async def on_pre_step(
        request: PreStepRequest,
        next_: Next[PreStepDecision],
    ) -> PreStepDecision:
        message = _model_breach(request.session)
        if message is None:
            return await next_(request)
        # This is a veto. A policy row owns this decision by returning without
        # calling `next_`. That also stops every later listener from doing work
        # for a step that will not happen.
        return PreStepDecision(kind="reject", reason=message)

    async def on_request_error(
        failure: RequestFailure, next_: Next[RequestErrorAction | None]
    ) -> RequestErrorAction | None:
        """A retry is a model call, so a granted retry is held to the ceiling (P1).

        `agent/pre-step` fires one time per step, and a retried step makes its
        calls without passing it again. A ceiling that was checked only there
        limited steps, and a step that was retried three times cost one. This
        seam is where a retry is decided, whichever row grants it.

        **The rest of the chain decides first, and this listener refuses what
        the chain granted.** If the ceiling were checked before `next_`, it
        would record a refusal for a failure that nobody meant to retry. An auth
        error at the limit would read as "not retried because of the ceiling".
        The listener is prepended so that it *wraps* the chain. `llm-retry`
        answers without calling `next_` when it grants, so a listener behind it
        never sees the retry. A veto after the grant costs nothing, because the
        loop sleeps the backoff and the row does not.

        To refuse, the listener returns `None`. The failure stands, and the turn
        ends with the error of the provider itself. The breach is recorded
        beside that error as the reason that the call was not retried.
        `exit: error` raises instead, as `on_pre_step` does.
        """
        action = await next_(failure)
        if action is None or action.kind != "retry":
            return action
        return action if _model_breach(failure.session, retry=True) is None else None

    # -------------------------------------------------------- tool calls --

    async def on_pre_execute(
        execution: ToolExecution,
        next_: Next[PreToolDecision],
    ) -> PreToolDecision:
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
            # **A decision, not a raise** (D7). This branch used to raise out of
            # `tools/pre-execute`. The pipeline there turns any exception from a
            # policy row into a failed result. That is deliberate, so that a
            # broken row cannot take down the call that it was asked about. The
            # ceiling inherited that behavior by accident. The turn continued,
            # every later call got the same failure, and under Code Mode the
            # program could `except` the failure and keep going. A generated
            # program could therefore defeat a policy gate.
            #
            # `Deny` now says both halves. `concludes_turn` ends the turn, as
            # `end` already did. `failure_kind="failed"` is everything that
            # `error` adds over `end`: the model reads a spent budget and not a
            # policy refusal. The two postures exist to differ on that reading.
            #
            # The breach is recorded before the decision is returned, for the
            # same reason that it was recorded before it was raised.
            # `phern doctor`, a reviewer and a resumed session all read this
            # event and not the tool result.
            message = f"'{name}' call limit reached: {', '.join(exceeded)}."
            if not _is_first_over(settings, current, name):
                return Deny(reason=SIBLING_STOPPED, concludes_turn=True, failure_kind="failed")
            _record(session, "tool-calls", settings.exit, {"tool": name, "message": message})
            return Deny(reason=message, concludes_turn=True, failure_kind="failed")
        if settings.exit == "continue":
            # **The quietest posture is still a breach** (N2). This posture
            # denies the call and lets the turn go on. It is therefore the
            # posture that a deployment is most likely to run when a budget
            # starts to apply. It was also the only posture that wrote nothing.
            # "The ceiling is doing something" was then invisible in exactly the
            # case where nothing else reports it.
            _record(
                session,
                "tool-calls",
                settings.exit,
                {
                    "tool": name,
                    "message": f"'{name}' tool call limit reached: {' and '.join(exceeded)}.",
                },
            )
            return Deny(reason=TOOL_DENIAL.format(tool=name))
        if not _is_first_over(settings, current, name):
            # This call is a sibling of the call that breached. Arithmetic tells
            # the two apart, not a latch: the breaching call is the one that
            # found the count *at* a ceiling. The wording is from upstream,
            # because this call did nothing wrong and the model should not read
            # the denial as being about this call.
            #
            # **Both postures that end the turn do this.** `error` did not while
            # it raised. An exception has no siblings to suppress, so a parallel
            # fan-out over a spent ceiling wrote one `limits/exceeded` for each
            # call and told every call that it was the breach. The two postures
            # are documented to differ in what the model reads. A difference in
            # the number of breaches that they record was drift.
            return Deny(reason=SIBLING_STOPPED, concludes_turn=True)
        _record(
            session,
            "tool-calls",
            settings.exit,
            {
                "tool": name,
                "message": f"'{name}' tool call limit reached: {' and '.join(exceeded)}.",
            },
        )
        # This is the way that the loop itself ends a turn, so there is no
        # second way. The batch in flight still settles, and nothing after it
        # runs.
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
        _LOG.append(
            session,
            "limits/breaker-tripped",
            {"tool": execution.name, "failures": failures, "limit": limit},
        )
        return Deny(reason=BREAKER_DENIAL.format(tool=execution.name, count=failures))

    # ------------------------------------------------------ the footer --

    def _reading(session: Session) -> StatusReading | None:
        """The tightest active budget, as one short reading (P4-04).

        This is a *live* number, not only the notice that arrives when a budget
        is spent. Upstream announces the limit on the step where it stops you,
        which is the one moment when the information can no longer change
        anything. The context gauge, one field over, already makes the same
        argument.

        The reading is read on every redraw. It is therefore guarded first by
        the cheapest possible check. It then uses the same `SessionFoldCache`
        that the limits themselves read. That cache returns its previous value
        directly when the events since the last read were all chunks.
        """
        model, tools = config.model_calls, config.tool_calls
        if model.unlimited and tools.unlimited:
            return None
        current = counts.read(session)
        gauges = (
            ("calls", current.turn_calls, model.turn_limit),
            ("calls", current.session_calls, model.session_limit),
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
        # Amber starts at the same fraction at which the context gauge warns. A
        # user who has learned one reading then knows what the other one means.
        return StatusReading(
            text=f"{label} {used}/{limit}",
            level="warning" if fraction >= WARN_FRACTION else "normal",
        )

    status = ctx.get(TUI_STATUS)
    if status is not None:
        status.register(StatusField(id="limits", read=_reading, order=10), scope=ctx)

    ctx.on("agent/pre-step", on_pre_step)
    ctx.on("agent/request-error", on_request_error, prepend=True)
    ctx.on("tools/pre-execute", on_pre_execute)

    # ----------------------------------------------------------- children --

    def refuse_child(subagents: SubagentService, request: SubagentRequest) -> Deny | None:
        """The child caps, checked before every admission (P4-04).

        The count comes from the service that the guard is on. That service
        knows the children and the spawns that are in progress (`child_counts`).
        The count does not come from the fold that the other ceilings read.
        """
        settings = config.children
        session = request.parent.session
        if session is None:
            return None
        current = subagents.child_counts(session)
        exceeded = _over(settings, current.turn, current.session, noun="children")
        if not exceeded:
            return None
        message = CHILD_DENIAL.format(limits=" and ".join(exceeded))
        # Record the posture that the deployment chose, as the other two
        # ceilings record theirs (D15). This used to be the literal `"refuse"`,
        # because there was only one posture.
        _record(
            session,
            "children",
            settings.exit,
            {
                "turn": current.turn,
                "session": current.session,
                "message": message,
            },
        )
        # This is the same decision that the tool ceiling returns, in the same
        # type. `error` is the only reading that is not a policy refusal: the
        # ceiling judged nothing about *this* request, because it was already
        # spent. The seam derives the code from the reading, as `registry._gate`
        # does.
        return Deny(
            reason=message,
            concludes_turn=settings.exit in ("end", "error"),
            failure_kind="failed" if settings.exit == "error" else "denied",
        )

    # `contribute_via` carries the wait-for-the-key rule and the reason for it.
    # A profile that mounts no subagent seam has nothing to cap, and it must not
    # lose its call limits because that seam is missing. Either mount order must
    # work. The guard is not registered at all when nothing is capped, so a
    # spawn pays for no guard when nobody chose a ceiling.
    if not config.children.unlimited:
        contribute_via(
            ctx,
            SUBAGENTS,
            lambda subagents, scope: subagents.guard(partial(refuse_child, subagents), scope=scope),
            label="limits-children",
        )
