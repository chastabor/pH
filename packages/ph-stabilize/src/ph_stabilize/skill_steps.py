"""`skill-steps` — a skill that is a procedure, and a loop that finishes it (P7-18).

A `SKILL.md` may declare `steps:`. Reading such a skill seeds them into the todo
list as entries the model **may mark done and may not delete**, and a listener on
`agent/turn-stopping` objects while any of them is still startable. That is the
whole of the feature, and none of it is new machinery.

**The loop already had the walker.** `commands/autonomous.py` says it best about
its own row: *"the driver is not a daemon, a scheduler or a second loop, it is a
listener on the boundary the agent loop already fires… a listener objects **by
steering** rather than by reaching into loop state, which is why continuing costs
no new machinery."* `/autonomous` asks whether a goal's gates pass; this asks
whether the procedure is finished. Same boundary, same `agent.steer`, same
`PluginSource` tag so the transcript does not attribute the harness's nudge to
the person reading it.

**What had to be added was provenance, not a walker.** `write_todos` replaces the
*whole* list, so steps seeded into it were the model's to delete on its next
write — and a listener enforcing against that list would have been enforcing
against nothing. `tool-todo`'s `SKILL` source and `_carried` are the rule that
closes it; this row is what puts entries there for it to protect.

**Two limits, and they are the point rather than an apology.** A *loop* step is
not a *plan* step: `agent/turn-stopping` fires when a turn is about to end, and a
plan step spans many, so this asserts a boundary invariant — work remains that
can begin — and never claims to know which step is running. And it holds the
model to *its own accepted plan*, not to reality (P5-16): marking a step done is
still the model's word. Gates that check the world are `ctx.goals` and
`ctx.approval`, and a procedure that needs one says so in the step's own text.

**And one question it can now have answered before it objects** (`Decision`).
A step that says "stop for approval" is prose, and this row used to answer the
model obeying it with a nudge to carry on — steering it past the very gate the
skill wrote. Where a profile names a decision model, the row asks it whether the
turn is ending to wait for a person, and holds the nudge when it is. A decision
model and not the session's own: it answers in a third of a second with a
probability, where a chat model would spend a turn's worth of tokens to write a
sentence this row would then have to parse.

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
"""This row's name on the messages it steers with — the key `nudges_since`
reads them back by, so the tag and the fold cannot drift apart."""

MAX_NAMED = 3
"""How many outstanding steps the steer names before it counts the rest.

A nudge is read by a model that already has the list in its context — it is a
pointer, not a second copy of the plan, and a twenty-line reminder every time a
turn tries to end is how a steer becomes noise the model learns to skim."""

MAX_NUDGES = 3
"""How many times this row will steer without the plan moving, by default.

**The ceiling the shape this row copies has and it did not.** `/autonomous` is
the same listener on the same boundary and carries four — continuations, turns,
tokens, wall clock — because a driver whose only exit is the model complying is
not bounded at all. The loop has no step cap of its own (`limits` ships with
`turn_limit` unset, deliberately), so without this a model that will not mark a
step done is steered for as long as the session lives.

Counted **since the list last changed**, not since the turn began, which is what
makes it a stall detector rather than a quota: any `write_todos` — marking a step
done, adding an entry, re-planning — resets it, so a run making progress is never
cut off, and one going in circles stands down and lets the person see the list.

The default, not the rule (D16): a profile sets its own with `maxNudges`, and a
skill may set one for its procedure with `max-nudges` — see `keep_going`."""

BUDGET = "skill-steps/budget"
"""The event recording the budget of the procedure seeded last (D16).

A record of its own rather than a field on the seeding `todo/write`: the model's
next `write_todos` replaces that event, so a budget stored on it would last
exactly one plan. Nor a field on the seeded entries, which `tool-todo` echoes
back to the model on every write — a harness number paid for in tokens forever."""


DECIDED = "skill-steps/decided"
"""What a decision model said of a turn that stopped with a procedure
unfinished: how likely it is waiting on a person (`waiting`), and the line that
was held to (`holdAt`) — the nudge was held where the first reaches the second.
Beside the route's own `llm/decided`, which says what asking cost and not what
was asked."""

WAITING = "waiting"
"""The one question this row asks, by the name its answer comes back under."""

ASKS = (
    "Did the assistant stop to wait for the person: a question it needs answered, or an "
    "approval or confirmation it asked for, before it goes on with the procedure?"
)
"""The question, as the decision model reads it. A `noul` — a yes-or-no — on
purpose: the answer is one number, and the rule over it is one line
(`HOLD_AT`)."""

HOLD_AT = 0.8
"""How likely "waiting on a person" has to be before the nudge is held, by
default.

Above an even chance on purpose. The two mistakes cost differently: a nudge
held wrongly ends a turn with the list on show, and a person says "go on"; a
nudge sent wrongly steers a model past an approval it had stopped for. The
answers leave room for it. Measured against Clef on a llama.cpp, with this
question and `decision_state`: a request to approve, a choice put to the person
and a missing password read 0.98 to 0.99; a turn that reported and stopped read
0.01; a sign-off — *"let me know if you want anything else"* — 0.03 to 0.07;
and a reply unsure of its own result, with no question in it, 0.3 to 0.4. So
the line sits above the unsure middle, which still gets its nudge."""

DECIDE_WITHIN_MS = 3_000
"""How long a turn that is ending waits for the decision, by default, both tries
together. A decision model answers in well under a second — 0.4 s against Clef —
and every turn that would be nudged waits on it; one that takes this long is not
going to answer, and the turn gets the nudge it would have had without it."""

REST_S = 60.0
"""How long the row stops asking a decision model that did not answer in time.

A route that is down refuses in milliseconds and is asked again at the next
turn's end, which is how one that comes back is found. A route that stalls
costs every turn that would be nudged its whole `timeoutMs`, in every session,
for as long as it stalls — so after one timeout the row nudges without asking
for this long, and then tries again."""

_now = anyio.current_time
"""The clock a rest is kept by: a name of its own so a test steps it rather than
waiting `REST_S` out."""

DEFAULT_BATCH = 512
"""What a decision route is taken to read in one request when nothing says
otherwise (`LlmRuntime.decision_limit`): a llama.cpp's default physical batch.
A llama.cpp reads a decision in one batch and refuses what does not fit it,
whatever its context window is, so this and not the window is what the state,
the question and the model's own framing have to fit together."""

FRAMING_TOKENS = 175
"""What of a decision's tokens is neither the steps nor the message: the model's
own framing and the question — 170, measured against Clef for a state of one
character."""

MOST_SAID = 4_000
"""The most of the message the decision model is shown, in bytes, however much
its route reads. The end of a message is where a question or a request to
confirm is, and a turn that is ending waits on an answer that takes longer the
more is read."""

MIN_SAID = 200
"""The shortest tail worth asking about, in bytes. Less than this is a sentence
fragment, and a server that cannot take it is one this row nudges without."""

STEP_BYTES = 120
"""How much of a step's text the decision model is shown, in bytes as the rest
of the state is sized: enough to tell which step it is. A step may run to 500
characters, and the state has a batch to fit."""


def said_bytes(limit: int | None, ready: list[str], blocked: int) -> int:
    """How much of the assistant's last message — its end, in UTF-8 bytes — the
    decision model is shown: what a route that reads `limit` tokens in one
    decision (`DEFAULT_BATCH` where nothing says) has left once the framing and
    the steps it is shown are counted. About 800 bytes at a llama.cpp's default
    with one step named, which is 800 characters of English; up to `MOST_SAID`
    where the route reads more.

    One unit for the whole state: the route's tokens by `bytes_of`, the careful
    estimate a caller with no tokenizer shares with text-index (prose ran 3.2
    characters a token and code 4 against Clef), and the steps by the bytes they
    are sent as. Bytes rather than characters, so neither a step nor a message
    in a script three bytes a character is sent three times what fits. Text
    denser still is refused, and answered with a shorter tail (`_fitting`)."""
    room = bytes_of((DEFAULT_BATCH if limit is None else limit) - FRAMING_TOKENS)
    steps = len(decision_state(ready, blocked, "", 0).encode("utf-8"))
    return max(MIN_SAID, min(MOST_SAID, room - steps))


class Decision(WireModel):
    """The decision model a profile gives this row to ask before it nudges.

    A route, named the way a session's is: the `provider` an adapter row
    registered and the `model` it serves there. The adapter has to be one that
    decides (`ph.llm.adapter.DecisionAdapter`); phern's `llm-openai-compatible`
    is, against a llama.cpp or an Ollama serving a decision model."""

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
    """Row config: the nudge budget a profile gives every procedure (D16), and
    the decision model it asks first, where it has one."""

    max_nudges: int = Field(default=MAX_NUDGES, ge=0)
    """Nudges without the plan moving before the row stands down. A skill's own
    `max-nudges` overrides it for that skill's procedure; `0` seeds steps but
    never steers."""
    decision: Decision | None = None
    """The decision model asked whether a turn is ending to wait for a person,
    in which case it is let end. Without one every unfinished procedure is
    nudged, as it is when the model cannot be asked."""


def seeded(current: list[dict[str, Any]], steps: list[str]) -> list[dict[str, Any]] | None:
    """`current` with `steps` appended as a skill's, or `None` with a reason logged.

    **Idempotent by content**, because reading a skill twice is ordinary — a
    model re-reads instructions it half-remembers — and a second copy of the
    procedure would be a plan that can never be finished.

    Appended rather than merged into position: the model's own entries stay where
    it put them, and a procedure that arrives mid-session is work added to the
    end rather than a plan rewritten underneath somebody.

    Sequential `requires` within the skill's own steps, and only within them —
    two procedures read in one session are two orderings, not one queue. The
    chain follows the order the *skill* declared rather than the order of what is
    missing, so a re-read that finds one step already present links the rest
    around the gap instead of over it (D13).

    **Built through `TodoItem` and checked by `_checked`, because this is the
    list's second writer.** `write_todos` is bounded by its own schema and its own
    coherence rule; an entry appended straight to the log is bounded by neither,
    and the failure is not a cosmetic one: a step longer than `MAX_TODO_CONTENT`,
    or one that pushes the list past `MAX_TODOS`, seeds fine and then makes
    *every* later `write_todos` fail validation — while `_carried` refuses any
    write that drops it. The model would be locked out of its own plan with the
    row steering it to keep going. Refusing to seed is the recoverable answer,
    and the author gets a warning naming the skill's file.

    All or nothing for the same reason: half a procedure is a plan nobody wrote.
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
    # **The chain follows the skill's own order, not the gap** (D13). Built over
    # `wanted`, a procedure whose middle step was already in the list linked the
    # step after it to the step *before* — `[one, three]` with `three` waiting on
    # `one` — so the ordering the skill declared was quietly replaced by one that
    # let `three` start while `two` was still pending. The predecessor is always
    # nameable: every step is either already present or being added now.
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
        # `source` is out of band on the dict rather than a `TodoItem` field, the
        # same shape `worked` uses: a field the model could write is a field it
        # could label away.
        grown.append({**entry, "source": SKILL})
    try:
        # One authority for "is this list coherent", rather than a second
        # collision check here. `_checked` already refuses two entries sharing
        # content — which is what a step whose text the model has used would be,
        # and which `requires` could not then name unambiguously.
        _checked(grown)
    except PlanError as refusal:
        log.warning("ph_stabilize.skill_steps: not seeding — %s", refusal)
        return None
    return grown


def _startable(ready: list[str], blocked: int, *, room: int | None = None) -> str:
    """The steps that can start — a few named, the rest counted — and how many
    wait on them: the one sentence the steer and the decision model's state
    share. `room` shortens each named step to a line of that many bytes."""
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
    assistant's last message ended — its last `room` bytes, starting between
    characters (`ph.characters.tail_bytes`).

    The steps are here so the question has something to be about: "is it
    waiting" is asked of a turn that left *this* undone, and a model shown only
    the message cannot tell a question about the procedure from small talk."""
    tail = elide(said, room, keep="tail", in_bytes=True)
    return (
        f"An assistant is following a procedure and has not finished it: "
        f"{_startable(ready, blocked, room=STEP_BYTES)}\n\nThe assistant ended its turn "
        f"with this message:\n{tail}"
    )


def _last_said(session: Session) -> str:
    """The text of the assistant's latest message: what it said as it stopped.
    Empty when it said nothing, and then there is nothing to ask about.

    The latest message and not `ph.seams.subagents.last_answer`, which scans
    back to the last one with text: a turn that ended on tool calls alone would
    then be judged by what an earlier turn said."""
    event = session.latest("assistant/message")
    message = derive_event_message(event) if event is not None else None
    return text_of(message.content).strip() if message is not None else ""


def _fitting(error: LlmError, shown: int) -> int | None:
    """How many bytes of the message to show a server that refused `shown` bytes
    of it as too long, or `None` when the refusal is not one a shorter tail
    answers.

    Aimed by the refusal's own numbers (`refused_share`): the server said how
    many tokens it was sent and how many it reads, and the tail is cut in that
    proportion and a quarter more, since the question and the steps are part of
    what was counted and are not what gets shorter. What it reads is the
    route's to keep (`LlmRuntime.decision_limit`); how dense this message ran
    is this message's alone."""
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
    """The decision model's answer about this stop, asked again once with less
    of the message when the first try was too long for the server — both tries
    within the one `timeout_ms` a turn waits."""
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
    """The budget the latest seeded skill recorded, or `None` — a fact, not a rule.

    `None` both when no skill has seeded and when the latest one set no budget
    (it records `null`). Which record governs, and what `None` falls back to, is
    `keep_going`'s to decide (P4). Folded from the log, for `nudges_since`'s
    reason: a budget held on the listener would be gone after a resume.
    """
    event = session.latest(BUDGET)
    if event is None:
        return None
    recorded = event.data.get("maxNudges")
    return None if recorded is None else as_int(recorded)


def nudges_since(session: Session, seq: int, *, up_to: int | None = None) -> int:
    """How many times this row has steered after `seq` — a count, not a window.

    Folded from the log rather than counted on the row, for P5-04's reason: a
    counter on a listener starts at zero after a resume or a passivation, and the
    stall it bounds is exactly the kind of run that outlives a process. The steer
    lands in the log as a `user/message` carrying this row's `PluginSource`, so
    the tag *is* the record — there is nothing to keep in step with it.

    Where the count starts is the caller's rule (P4): it was "since the todo list
    last changed", written in here, where it read as a fact about the log rather
    than the stall detector's policy. `up_to` stops the walk once the count
    reaches it — the only question `keep_going` asks is "is the budget spent",
    and a run that stood down (or a skill whose budget is `0`) would otherwise
    rescan an ever-growing tail on every turn.
    """
    # By index through `Session.at`, not `events_from`: that copies the whole
    # tail before the first event is looked at, so `up_to` saved the filtering
    # and not the walk — and the tail of a run that stood down only grows.
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
    """Seed a read skill's steps, and object while they are unfinished.

    No `inject`: the body registers two listeners and needs no service to do
    it. `inject` is an activation gate — a row is reported inactive on an unmet
    key and its scope unwinds when the service goes — so naming `tools` here
    would tie this row's life to something it never calls, and naming `llm`
    would stop the seeding and the nudge for want of a decision model they work
    without. `ctx.llm` is looked up when a decision is asked for, and its
    absence is one more reason the answer could not be had.
    """

    # Each code the decision model could not be asked for, warned about once, so
    # a route that is down says so once rather than at every turn's end; and
    # when a route that timed out is worth asking again (`REST_S`).
    warned: set[str] = set()
    resting_until: float | None = None

    async def waits_for_a_person(session: Session, ready: list[str], blocked: int) -> bool:
        """Whether this turn is ending to wait for a person, as the profile's
        decision model reads it. `False` wherever it cannot say — no model
        named, nothing said, a route that would not answer — which is the nudge
        this row always sent: a decision that cannot be had degrades to the
        behavior before there was one, with a warning, and never holds a turn
        up on a server that is down."""
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
            # The route's own record of the failure is in the session's log
            # (`llm/decision-failed`); this is the line an operator reads.
            unasked(decision, error.code, str(error))
            return False
        waiting = decided.answer(WAITING, NoulAnswer).noul
        _LOG.append(session, DECIDED, {"waiting": waiting, "holdAt": decision.hold_at})
        return waiting >= decision.hold_at

    def unasked(decision: Decision, code: str, reason: str) -> None:
        """Rest a route that timed out, and say once for each code that the
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
        """`phern doctor`'s line: which decision model is asked, and whether it
        can be. Read when asked, so an adapter row mounted after this one counts."""
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
        # The *rendered* steps off the payload, not `skill.steps`: an author may
        # write `Run {{parameters.gate}}` in a step, and the arguments that fill
        # it in belong to the call that read the skill — which this listener,
        # firing after the fact, does not have.
        steps = [str(one) for one in payload.get("steps") or ()]
        if not steps or session is None:
            return
        grown = seeded(todos_of(session), steps)
        if grown is None:
            return
        # The same event the tool writes, because it means one thing — "the list
        # is now this" — and a second type would give `todos_of` two things to
        # fold and the sidebar two things to draw.
        # The budget is recorded even when `null`: that is what hands a later
        # procedure back to the profile's budget rather than the previous
        # skill's. One batch with the list it governs (S14), so a torn tail
        # cannot seed the steps under the last skill's budget.
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
        # Frozen, and before anything else: this fires once per turn for every
        # session in the deployment, and a session that never read a skill must
        # not pay a thaw of somebody's hundred-entry list to find that out.
        outstanding = outstanding_steps(session)
        if not outstanding:
            return
        # The stall detector's two rules, here where they are applied (P4).
        # **The procedure seeded last governs**, because a nudge names whatever
        # is startable and cannot be charged to one skill — and the latest read
        # is the procedure the model was most recently asked to follow. One that
        # set no budget hands the row back to the profile's rather than
        # inheriting the one before it.
        recorded = latest_skill_budget(session)
        budget = config.max_nudges if recorded is None else recorded
        # **Counted since the plan last changed**, so any `write_todos` — a step
        # marked done, an entry added, a re-plan — earns fresh nudges, and only
        # a run going in circles reaches the budget.
        plan = session.latest("todo/write")
        if nudges_since(session, plan.seq if plan else -1, up_to=budget) >= budget:
            # Stood down rather than steering into a wall.
            log.info(
                "ph_stabilize.skill_steps: %s with the plan unchanged; standing down",
                count_of(budget, "nudge"),
            )
            return
        todos = todos_of(session)
        ready = [step for step in startable(todos) if step in outstanding]
        if not ready:
            # A shape guard, and said as one rather than dressed up as a policy:
            # `seeded` writes a *sequential* chain, so while anything is
            # unfinished its earliest link is waiting on nothing and this branch
            # cannot be reached by the entries this row writes. It is here
            # because a steer naming no step is worse than no steer at all, and
            # the list is a fold of a log that a hand-edit, a future seeding
            # shape or a second row could put another arrangement into.
            return
        # Asked last, once everything cheaper has said a nudge is due: it is a
        # request to a server, and a turn that would not be nudged anyway owes
        # nobody the wait. A held nudge is not a nudge, so it spends none of the
        # budget, and the next turn that stops is asked about afresh.
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
