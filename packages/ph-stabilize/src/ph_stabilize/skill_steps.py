"""`skill-steps`: a skill that is a procedure, and a loop that finishes it (P7-18).

A `SKILL.md` may declare `steps:`. When the model reads such a skill, the row
seeds the steps into the todo list. The model **may mark these entries done and
may not delete them**. A listener on `agent/turn-stopping` objects while any of
them can still be started. That is the whole feature, and none of it is new
machinery.

**The loop already had the walker.** `commands/autonomous.py` describes its own
row this way: *"the driver is not a daemon, a scheduler or a second loop, it is a
listener on the boundary the agent loop already fires… a listener objects **by
steering** rather than by reaching into loop state, which is why continuing costs
no new machinery."* `/autonomous` asks whether the gates of a goal pass. This row
asks whether the procedure is finished. Both use the same boundary and the same
`agent.steer`. Both tag the message with a `PluginSource`, so that the transcript
does not attribute the nudge of the harness to the user who reads it.

**The addition was provenance, not a walker.** `write_todos` replaces the *whole*
list. Without a rule, the model could delete the seeded steps on its next write,
and a listener that enforced against that list would enforce against nothing.
The `SKILL` source and `_carried` in `tool-todo` are the rule that prevents this.
This row puts the entries there for that rule to protect.

**The row has two limits, and both are deliberate.** First, a *loop* step is not
a *plan* step. `agent/turn-stopping` fires when a turn is about to end, and a
plan step spans many. The row therefore asserts an invariant at a boundary: work
remains that can begin. It never claims to know which step is running. Second,
the row holds the model to *its own accepted plan*, not to reality (P5-16). When
the model marks a step done, that is still only the word of the model. The gates
that check the world are `ctx.goals` and `ctx.approval`. A procedure that needs
one says so in the text of the step.

**The row can ask one question before it objects** (`Decision`). A step that says
"stop for approval" is prose. When the model obeyed such a step, this row used to
answer with a nudge to continue, which steered the model past the gate that the
skill wrote. Where a profile names a decision model, the row asks that model
whether the turn is ending to wait for a person. If it is, the row holds the
nudge. The row asks a decision model and not the model of the session. A decision
model answers in a third of a second with a probability. A chat model would spend
the tokens of a whole turn to write a sentence that this row would then have to
parse.

@module ph_stabilize.skill_steps
"""

from __future__ import annotations

import logging
from itertools import islice
from typing import Any

import anyio
from pydantic import Field, ValidationError

from ph.agent.types import AgentDriver
from ph.cordis import Context, plugin
from ph.json import as_int, as_obj, as_str
from ph.keys import LLM
from ph.llm.adapter import (
    DecisionRequest,
    Decisions,
    LlmError,
    LlmRuntime,
    Noul,
    NoulAnswer,
    refused_share,
)
from ph.llm.tokens import bytes_of
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    TIMEOUT,
    PluginSource,
    create_user_message,
    text_of,
)
from ph.seams.diagnostics import Diagnostic, contribute
from ph.session import Session, derive_event_message
from ph.session.writers import log_writer
from ph.text import count_of, elide
from ph.wire import WireModel

from .todo import (
    MAX_TODOS,
    SKILL,
    PlanError,
    TodoItem,
    _checked,
    outstanding_steps,
    startable,
    steps_of,
    todos_of,
)

_LOG = log_writer(__name__)

__all__ = [
    "DECIDED",
    "Config",
    "Decision",
    "apply",
    "decision_state",
    "latest_skill_budget",
    "nudges_since",
    "said_bytes",
    "seeded",
    "steer_text",
]

log = logging.getLogger("ph_stabilize.skill_steps")

PLUGIN = "ph_stabilize.skill_steps"
"""The name of this row on the messages that it steers with. `nudges_since` finds
those messages by the same name, so the tag and the fold cannot drift apart."""

MAX_NAMED = 3
"""How many outstanding steps the nudge names before it counts the rest.

The model that reads a nudge already has the list in its context. A nudge is a
pointer, not a second copy of the plan. A twenty-line reminder each time a turn
tries to end becomes noise that the model learns to skim."""

MAX_NUDGES = 3
"""The default number of times that this row steers while the plan does not change.

**`/autonomous` has a ceiling, and this row at first did not.** `/autonomous` is
the same listener on the same boundary, and it carries four ceilings:
continuations, turns, tokens, and wall clock. It has them because a driver whose
only exit is that the model complies is not bounded at all. The loop has no step
cap of its own (`limits` ships with `turn_limit` unset, deliberately). Without
this ceiling, a model that will not mark a step done is steered for as long as
the session lives.

The count runs **since the list last changed**, not since the turn began. That
makes it a stall detector and not a quota. Any `write_todos` resets the count: a
step marked done, an entry added, or a new plan. A run that makes progress is
therefore never cut off. When a run makes no progress, the row stands down and
the user sees the list.

This is the default, not the rule (D16). A profile sets its own value with
`maxNudges`, and a skill may set one for its procedure with `max-nudges`. See
`keep_going`."""

BUDGET = "skill-steps/budget"
"""The event that records the budget of the procedure that was seeded last (D16).

The budget is a record of its own and not a field on the `todo/write` that seeds
the steps. The next `write_todos` of the model replaces that event, so a budget
stored on it would last for exactly one plan. The budget is also not a field on
the seeded entries. `tool-todo` echoes those entries back to the model on every
write, so a number that only the harness uses would cost tokens on every write."""


DECIDED = "skill-steps/decided"
"""The event that records what a decision model said about a turn that stopped
with a procedure unfinished. `waiting` is how likely it is that the turn is
waiting on a person. `holdAt` is the threshold that applied. The nudge was held
when `waiting` reached `holdAt`. This event is in addition to the `llm/decided`
event of the route, which records what the request cost and not what was asked."""

WAITING = "waiting"
"""The name of the one question that this row asks. The answer comes back under
the same name."""

ASKS = (
    "Did the assistant stop to wait for the person: a question it needs answered, or an "
    "approval or confirmation it asked for, before it goes on with the procedure?"
)
"""The question, as the decision model reads it. It is deliberately a `noul`, which
is a yes-or-no question. The answer is one number, and the rule that applies to
it is one line (`HOLD_AT`)."""

HOLD_AT = 0.8
"""The default likelihood of "waiting on a person" at which the nudge is held.

The value is deliberately above an even chance, because the two mistakes cost
differently. A nudge that is held wrongly ends a turn with the list visible, and
a person says "go on". A nudge that is sent wrongly steers a model past an
approval that it had stopped for. The answers leave room for a high threshold.
These values were measured against Clef on a llama.cpp, with this question and
`decision_state`:

- A request to approve, a choice put to the user, and a missing password read
  0.98 to 0.99.
- A turn that reported and stopped read 0.01.
- A sign-off (*"let me know if you want anything else"*) read 0.03 to 0.07.
- A reply that was unsure of its own result, with no question in it, read 0.3
  to 0.4.

The threshold is therefore above the unsure middle, and an unsure reply still
gets its nudge."""

DECIDE_WITHIN_MS = 3_000
"""The default time that a turn that is ending waits for the decision, for both
tries together. A decision model answers in well under a second (0.4 s against
Clef), and every turn that would be nudged waits for it. A model that takes this
long is not going to answer. The turn then gets the nudge that it would have had
without a decision model."""

REST_S = 60.0
"""How long the row stops asking a decision model that did not answer in time.

A route that is down refuses in milliseconds. The row asks it again at the end of
the next turn, and that is how the row finds a route that has come back. A route
that stalls is different. It costs every turn that would be nudged the whole
`timeoutMs`, in every session, for as long as it stalls. After one timeout, the
row therefore nudges without asking for this long, and then tries again."""

_now = anyio.current_time
"""The clock that times a rest. It has a name of its own so that a test can step
it and does not have to wait for `REST_S`."""

DEFAULT_BATCH = 512
"""The number of tokens that a decision route is assumed to read in one request
when `LlmRuntime.decision_limit` gives no limit. It is the default physical batch
of a llama.cpp. A llama.cpp reads a decision in one batch and refuses a request
that does not fit the batch, whatever its context window is. The state, the
question, and the framing of the model must therefore fit together in the batch,
and not in the window."""

FRAMING_TOKENS = 175
"""The tokens of a decision that are neither the steps nor the message: the framing
of the model and the question. The measured value was 170, against Clef, for a
state of one character."""

MOST_SAID = 4_000
"""The most of the message that the decision model is shown, in bytes, however
much its route reads. A question or a request to confirm is at the end of a
message. A turn that is ending waits for the answer, and the answer takes longer
when the model reads more."""

MIN_SAID = 200
"""The shortest tail that is worth a request, in bytes. Less than this is a
fragment of a sentence. If a server cannot take this much, the row nudges without
asking it."""

STEP_BYTES = 120
"""How much of the text of a step the decision model is shown. The unit is bytes,
as for the rest of the state. This is enough to tell which step it is. A step may
run to 500 characters, and the state must fit a batch."""


def said_bytes(limit: int | None, ready: list[str], blocked: int) -> int:
    """How much of the end of the last assistant message the decision model is
    shown, in UTF-8 bytes.

    A route reads `limit` tokens in one decision, or `DEFAULT_BATCH` when `limit`
    is `None`. The result is what is left of that after the framing and the
    steps that the model is shown are counted. This is about 800 bytes at the
    default of a llama.cpp with one step named, which is 800 characters of
    English. Where the route reads more, the result goes up to `MOST_SAID`.

    The whole state uses one unit. `bytes_of` converts the tokens of the route.
    It is the careful estimate that a caller with no tokenizer shares with
    text-index (against Clef, prose ran 3.2 characters a token and code 4). The
    steps count as the bytes that they are sent as. The unit is bytes and not
    characters, so that a step or a message in a script of three bytes a
    character is not sent at three times what fits. The server refuses text
    that is denser still, and the row then asks again with a shorter tail
    (`_fitting`)."""
    room = bytes_of((DEFAULT_BATCH if limit is None else limit) - FRAMING_TOKENS)
    steps = len(decision_state(ready, blocked, "", 0).encode("utf-8"))
    return max(MIN_SAID, min(MOST_SAID, room - steps))


class Decision(WireModel):
    """The decision model that a profile gives this row to ask before it nudges.

    It is a route, named in the same way as the route of a session: the
    `provider` that an adapter row registered and the `model` that it serves
    there. The adapter must be one that decides
    (`ph.llm.adapter.DecisionAdapter`). The `llm-openai-compatible` adapter of
    phern is one, against a llama.cpp or an Ollama that serves a decision
    model."""

    provider: str
    model: str = ""
    """Left empty for a server that serves one model and needs no name."""
    hold_at: float = Field(default=HOLD_AT, gt=0.0, le=1.0)
    """The nudge is held when "waiting on a person" is at least this likely."""
    timeout_ms: int = Field(default=DECIDE_WITHIN_MS, gt=0)
    """How long a turn that is ending waits for the answer before it is nudged
    without one."""

    @property
    def route(self) -> str:
        """The route as a warning and `phern doctor` name it."""
        return f"{self.provider}/{self.model}" if self.model else self.provider


class Config(WireModel):
    """The row config: the nudge budget that a profile gives every procedure
    (D16), and the decision model that the row asks first, where the profile
    names one."""

    max_nudges: int = Field(default=MAX_NUDGES, ge=0)
    """The number of nudges with the plan unchanged before the row stands down.
    The `max-nudges` of a skill overrides it for the procedure of that skill.
    `0` seeds the steps but never steers."""
    decision: Decision | None = None
    """The decision model that is asked whether a turn is ending to wait for a
    person. If it is, the row lets the turn end. Without a decision model, every
    unfinished procedure is nudged. The same happens when the model cannot be
    asked."""


def seeded(current: list[dict[str, Any]], steps: list[str]) -> list[dict[str, Any]] | None:
    """`current` with `steps` appended as the steps of a skill, or `None` when
    nothing is seeded. A refusal to seed logs the reason.

    **The function is idempotent by content.** Reading a skill twice is
    ordinary, because a model re-reads instructions that it half remembers. A
    second copy of the procedure would be a plan that can never be finished.

    The steps are appended and not merged into position. The entries of the
    model stay where the model put them. A procedure that arrives in the middle
    of a session adds work to the end. It does not rewrite a plan that somebody
    is following.

    Each step requires the step before it, and only within the steps of the
    same skill. Two procedures read in one session are two orderings, not one
    queue. The chain follows the order that the *skill* declared, and not the
    order of the steps that are missing. When a second read finds one step
    already present, the step after it still requires that step. The chain does
    not skip it (D13).

    **The entries are built through `TodoItem` and checked by `_checked`,
    because this function is the second writer of the list.** `write_todos` is
    bounded by its own schema and its own coherence rule. An entry that is
    appended straight to the log is bounded by neither, and the failure is not
    cosmetic. A step longer than `MAX_TODO_CONTENT`, or one that pushes the
    list past `MAX_TODOS`, would seed without error and then make *every* later
    `write_todos` fail validation, while `_carried` refuses any write that
    drops the step. The model would be locked out of its own plan while the row
    steers it to continue. A refusal to seed is the recoverable answer, and
    the function logs a warning that gives the reason.

    The seeding is all or nothing for the same reason: half a procedure is a
    plan that nobody wrote.
    """
    already = set(steps_of(current))
    wanted = [step for step in steps if step not in already]
    if not wanted:
        return None
    if len(current) + len(wanted) > MAX_TODOS:
        log.warning(
            "ph_stabilize.skill_steps: seeding %s would put the list past %s entries; not seeding",
            count_of(len(wanted), "step"),
            MAX_TODOS,
        )
        return None
    grown = list(current)
    # **The chain follows the order of the skill, not the missing steps** (D13).
    # When the chain was built over `wanted`, a procedure whose middle step was
    # already in the list linked the step after it to the step *before* it:
    # `[one, three]`, with `three` waiting on `one`. The ordering that the skill
    # declared was silently replaced by one that let `three` start while `two`
    # was still pending. The predecessor can always be named, because every step
    # is either already present or added now.
    predecessor = {step: steps[index - 1] for index, step in enumerate(steps) if index}
    for step in wanted:
        try:
            entry = TodoItem(
                content=step,
                status="pending",
                requires=[predecessor[step]] if step in predecessor else [],
            ).model_dump(mode="json")
        except ValidationError as error:
            log.warning("ph_stabilize.skill_steps: a step cannot be a todo entry: %s", error)
            return None
        # `source` is set on the dict and is not a `TodoItem` field, in the same
        # way as `worked`. If the model could write the field, it could remove
        # the label.
        grown.append({**entry, "source": SKILL})
    try:
        # `_checked` is the one authority for "is this list coherent", so there
        # is no second collision check here. `_checked` already refuses two
        # entries that share content. A step whose text the model has already
        # used would make such a pair, and `requires` could not then name
        # either entry unambiguously.
        _checked(grown)
    except PlanError as refusal:
        log.warning("ph_stabilize.skill_steps: not seeding — %s", refusal)
        return None
    return grown


def _startable(ready: list[str], blocked: int, *, room: int | None = None) -> str:
    """The one sentence that the nudge and the state of the decision model share.

    It names a few of the steps that can start, counts the others, and says how
    many steps wait on them. `room` shortens each named step to one line of that
    many bytes."""
    named = ", ".join(
        repr(one if room is None else elide(" ".join(one.split()), room, in_bytes=True))
        for one in ready[:MAX_NAMED]
    )
    rest = len(ready) - MAX_NAMED
    more = f", and {rest} more" if rest > 0 else ""
    waiting = ""
    if blocked:
        verb = "waits" if blocked == 1 else "wait"
        waiting = f" {count_of(blocked, 'further step')} {verb} on these."
    return f"{named}{more} can be started now.{waiting}"


def steer_text(outstanding: list[str], blocked: int) -> str:
    """What the model is told when it tries to stop with a procedure unfinished."""
    return (
        f"A skill you read set out a procedure and it is not finished: "
        f"{_startable(outstanding, blocked)} Continue with it, or if a step genuinely does "
        "not apply, mark it completed and say why in your next message."
    )


def decision_state(ready: list[str], blocked: int, said: str, room: int) -> str:
    """What the decision model is shown: the steps that can start, and how the
    last message of the assistant ended. The end is the last `room` bytes of the
    message, and it starts between characters (`ph.characters.tail_bytes`).

    The steps are included so that the question has a subject. "Is it waiting"
    is asked about a turn that left *these steps* undone. A model that is shown
    only the message cannot tell a question about the procedure from small
    talk."""
    tail = elide(said, room, keep="tail", in_bytes=True)
    return (
        f"An assistant is following a procedure and has not finished it: "
        f"{_startable(ready, blocked, room=STEP_BYTES)}\n\nThe assistant ended its turn "
        f"with this message:\n{tail}"
    )


def _last_said(session: Session) -> str:
    """The text of the latest assistant message: what the assistant said as it
    stopped. The text is empty when the assistant said nothing, and then there
    is nothing to ask about.

    This reads the latest message and does not use
    `ph.seams.subagents.last_answer`, which scans back to the last message that
    has text. With that function, a turn that ended on tool calls alone would be
    judged by what an earlier turn said."""
    event = session.latest("assistant/message")
    message = derive_event_message(event) if event is not None else None
    return text_of(message.content).strip() if message is not None else ""


def _fitting(error: LlmError, shown: int) -> int | None:
    """How many bytes of the message to show a server that refused `shown` bytes
    of it as too long, or `None` when a shorter tail does not answer the
    refusal.

    The numbers in the refusal set the size (`refused_share`). The server said
    how many tokens it was sent and how many it reads. The tail is cut in that
    proportion and by a quarter more, because the question and the steps are
    part of the count and do not get shorter. The number of tokens that the
    server reads is a fact about the route, and `LlmRuntime.decision_limit`
    keeps it. How dense this message was applies to this message only."""
    if error.code != CONTEXT_WINDOW_EXCEEDED:
        return None
    share = refused_share(error.failure.sent_tokens, error.failure.limit_tokens)
    if share is None:
        return None
    fits = int(shown * share * 0.75)
    return fits if fits >= MIN_SAID else None


async def _decided(
    llm: LlmRuntime,
    session: Session,
    decision: Decision,
    ready: list[str],
    blocked: int,
    said: str,
) -> Decisions:
    """The answer of the decision model about this stop.

    When the first try was too long for the server, the function asks one more
    time with less of the message. Both tries share the one `timeout_ms` that a
    turn waits."""
    deadline = anyio.current_time() + decision.timeout_ms / 1000

    async def ask(room: int) -> Decisions:
        request = DecisionRequest(
            provider=decision.provider,
            model=decision.model,
            state=decision_state(ready, blocked, said, room),
            questions={WAITING: Noul(ASKS)},
        )
        left = max(deadline - anyio.current_time(), 0.0)
        return await llm.decide(request, session=session, timeout_s=left)

    first = said_bytes(llm.decision_limit(decision.provider, decision.model), ready, blocked)
    try:
        return await ask(first)
    except LlmError as error:
        fits = _fitting(error, min(len(said.encode("utf-8")), first))
        if fits is None:
            raise
        return await ask(fits)


def latest_skill_budget(session: Session) -> int | None:
    """The budget that the latest seeded skill recorded, or `None`. This is a
    fact, not a rule.

    The result is `None` both when no skill has seeded and when the latest skill
    set no budget (it records `null`). `keep_going` decides which record governs
    and what `None` falls back to (P4). The value is folded from the log, for
    the same reason as in `nudges_since`: a budget held on the listener would be
    gone after a resume.
    """
    event = session.latest(BUDGET)
    if event is None:
        return None
    recorded = event.data.get("maxNudges")
    return None if recorded is None else as_int(recorded)


def nudges_since(session: Session, seq: int, *, up_to: int | None = None) -> int:
    """How many times this row has steered after `seq`. This is a count, not a
    window.

    The count is folded from the log and is not kept on the row, for the reason
    in P5-04. A counter on a listener starts at zero after a resume or a
    passivation, and the stall that this count limits is exactly the kind of run
    that outlives a process. The nudge is in the log as a `user/message` that
    carries the `PluginSource` of this row. The tag *is* the record, so there is
    nothing else to keep in step with it.

    The caller decides where the count starts (P4). The rule "since the todo
    list last changed" used to be written in this function, where it read as a
    fact about the log and not as the policy of the stall detector. `up_to`
    stops the walk when the count reaches it. The only question that
    `keep_going` asks is "is the budget spent". Without `up_to`, a run that
    stood down (or a skill whose budget is `0`) would scan a tail that only
    grows, again on every turn.
    """
    # The walk goes by index through `Session.at`, not through `events_from`.
    # `events_from` copies the whole tail before the first event is looked at,
    # so `up_to` saved the filtering and not the walk. The tail of a run that
    # stood down only grows.
    events = (session.at(index) for index in range(seq + 1, session.seq))
    nudges = (
        event
        for event in events
        if event is not None
        and event.type == "user/message"
        and as_str(as_obj(event.data.get("source")).get("plugin")) == PLUGIN
    )
    return sum(1 for _ in islice(nudges, up_to))


@plugin("skill-steps", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Seed the steps of a skill that was read, and object while they are
    unfinished.

    The row declares no `inject`. The body registers two listeners and
    contributes one diagnostic, and needs no service for either. `inject` is an
    activation gate: a row is reported inactive when a key is unmet, and its
    scope unwinds when the service goes. To name `tools` here would tie the life
    of this row to a service that it never calls. To name `llm` would stop the
    seeding and the nudge because a decision model is missing, and both work
    without one. The row looks up `ctx.llm` when it asks for a decision. If
    `ctx.llm` is absent, that is one more reason why no answer was available.
    """

    # `warned` holds each code for which the decision model could not be asked.
    # Each code is warned about one time, so a route that is down is reported
    # one time and not at the end of every turn. `resting_until` is the time
    # when a route that timed out is worth asking again (`REST_S`).
    warned: set[str] = set()
    resting_until: float | None = None

    async def waits_for_a_person(session: Session, ready: list[str], blocked: int) -> bool:
        """Whether this turn is ending to wait for a person, as the decision
        model of the profile reads it.

        The result is `False` wherever the model cannot say: no model is named,
        the assistant said nothing, or the route did not answer. `False` gives
        the nudge that this row always sent. A decision that is not available
        degrades to the behavior before there was a decision model, with a
        warning. It never holds up a turn because a server is down."""
        decision = config.decision
        if decision is None or (resting_until is not None and _now() < resting_until):
            return False
        said = _last_said(session)
        if not said:
            return False
        llm = ctx.get(LLM)
        if llm is None:
            unasked(decision, "ctx.llm", "not mounted")
            return False
        try:
            decided = await _decided(llm, session, decision, ready, blocked, said)
        except LlmError as error:
            # The route records the failure in the log of the session
            # (`llm/decision-failed`). The warning here is the line that an
            # operator reads.
            unasked(decision, error.code, str(error))
            return False
        waiting = decided.answer(WAITING, NoulAnswer).noul
        _LOG.append(session, DECIDED, {"waiting": waiting, "holdAt": decision.hold_at})
        return waiting >= decision.hold_at

    def unasked(decision: Decision, code: str, reason: str) -> None:
        """Rest a route that timed out, and warn one time for each code that the
        decision model could not be asked."""
        nonlocal resting_until
        rests = code == TIMEOUT
        if rests:
            resting_until = _now() + REST_S
        if code in warned:
            return
        warned.add(code)
        log.warning(
            "ph_stabilize.skill_steps: the decision model %s could not be asked (%s: %s); "
            "nudging without it%s",
            decision.route,
            code,
            reason,
            f", and not asking it again for {REST_S:g} s" if rests else "",
        )

    def describe() -> list[tuple[str, str]]:
        """The line for `phern doctor`: which decision model is asked, and whether
        it can be asked. The line is read on request, so an adapter row that is
        mounted after this row counts."""
        decision = config.decision
        if decision is None:
            return []
        llm = ctx.get(LLM)
        if llm is not None and llm.decides(decision.provider):
            said = f"a nudge is held at {decision.hold_at:g} likely to be waiting on a person"
        else:
            said = "no adapter registered for that provider decides, so every procedure is nudged"
        return [("decision model", f"{decision.route}; {said}")]

    def on_skill_read(payload: Any) -> None:  # noqa: ANN401
        session = payload.get("session")
        # The listener takes the *rendered* steps from the payload, not
        # `skill.steps`. An author may write `Run {{parameters.gate}}` in a
        # step. The arguments that fill it in belong to the call that read the
        # skill. This listener fires after that call and does not have them.
        steps = [str(one) for one in payload.get("steps") or ()]
        if not steps or session is None:
            return
        grown = seeded(todos_of(session), steps)
        if grown is None:
            return
        # This is the same event that the tool writes, because it means one
        # thing: "the list is now this". A second type would give `todos_of` two
        # things to fold and the sidebar two things to draw.
        # The budget is recorded even when it is `null`. That record returns a
        # later procedure to the budget of the profile, and not to the budget of
        # the previous skill. The budget is in one batch with the list that it
        # governs (S14), so a torn tail cannot seed the steps under the budget
        # of the last skill.
        with session.batch() as batch:
            _LOG.append(batch, "todo/write", {"todos": grown})
            _LOG.append(
                batch,
                BUDGET,
                {"skill": payload["skill"].name, "maxNudges": payload.get("max_nudges")},
            )

    async def keep_going(agent: AgentDriver, turn: int) -> None:
        session = agent.session
        if session is None:
            return
        # This check reads the frozen payload, and it runs before anything
        # else. The listener fires one time per turn for every session in the
        # deployment. A session that never read a skill must not pay to thaw a
        # hundred-entry list to find that out.
        outstanding = outstanding_steps(session)
        if not outstanding:
            return
        # The two rules of the stall detector are here, where they are applied
        # (P4). **The procedure that was seeded last governs.** A nudge names
        # whatever can be started, so it cannot be charged to one skill, and the
        # latest read is the procedure that the model was most recently asked
        # to follow. A procedure that set no budget returns the row to the
        # budget of the profile and does not inherit the budget before it.
        recorded = latest_skill_budget(session)
        budget = config.max_nudges if recorded is None else recorded
        # **The count runs since the plan last changed.** Any `write_todos` (a
        # step marked done, an entry added, a new plan) therefore resets the
        # count, and only a run that makes no progress reaches the budget.
        plan = session.latest("todo/write")
        if nudges_since(session, plan.seq if plan else -1, up_to=budget) >= budget:
            # The row stands down. More nudges would change nothing.
            log.info(
                "ph_stabilize.skill_steps: %s with the plan unchanged; standing down",
                count_of(budget, "nudge"),
            )
            return
        todos = todos_of(session)
        ready = [step for step in startable(todos) if step in outstanding]
        if not ready:
            # This is a shape guard and not a policy. `seeded` writes a
            # *sequential* chain, so while any step is unfinished, the earliest
            # unfinished step waits on nothing. The entries that this row writes
            # therefore cannot reach this branch. The guard is here because a
            # nudge that names no step is worse than no nudge. The list is also
            # a fold of a log, and a hand edit, a future seeding shape, or a
            # second row could put another arrangement into that log.
            return
        # The decision model is asked last, after every cheaper check has said
        # that a nudge is due. It is a request to a server, and a turn that
        # would not be nudged anyway must not wait for one. A held nudge is not
        # a nudge, so it spends none of the budget. The row asks again about
        # the next turn that stops.
        blocked = len(outstanding) - len(ready)
        if await waits_for_a_person(session, ready, blocked):
            return
        agent.steer(
            create_user_message(
                content=[{"type": "text", "text": steer_text(ready, blocked)}],
                source=PluginSource(
                    plugin=PLUGIN,
                    form="notice",
                    summary="a skill's procedure is unfinished",
                ).to_wire(),
            )
        )

    ctx.on("skills/read", on_skill_read)
    ctx.on("agent/turn-stopping", keep_going)
    contribute(ctx, Diagnostic(id="skill-steps", title="Skill steps", read=describe, order=45))
