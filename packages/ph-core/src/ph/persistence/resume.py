"""Resuming a stored log: the repair of a crashed tail, what tools say about the
calls that the crash cut off, the `session/resumed` record, and the readers of
that record.

This module is backend-neutral, so it does not belong to a backend. It lived in
the module of the JSONL store, though nothing in it touches a file.
`open_session` reads the log through the Protocol and hands it here. So a
database backend resumes through the same code, and this module writes
`session/resumed`, whichever backend is mounted.

@module ph.persistence.resume
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..cordis import DEPLOYMENT, Context
from ..json import as_int, as_obj, as_str
from ..keys import SESSIONS, TOOLS
from ..session import Session, SessionEvent, SessionHeader, declared_intents, open_intents
from ..session.writers import log_writer
from .repair import CallOutcome, interrupted_turn_closers, unresolved_calls

_LOG = log_writer(__name__)

__all__ = [
    "CutTurnOwner",
    "Resumption",
    "cut_short_note",
    "cut_turn_owner",
    "resume_session",
    "resumption_of",
    "resumptions",
]

CutTurnOwner: TypeAlias = Literal["person", "unattended"]
"""Who a turn that a crash cut short was owed to (P13-04).

`person`: a prompt in the turn was the words of a person, and that person is at
a keyboard somewhere to send it again. `unattended`: every prompt in the turn
came from something that will not send it again: a schedule, a notice from a
child, or a continuation by a supervisor."""


@dataclass(frozen=True, slots=True)
class Resumption:
    """One `session/resumed` record, as `resume_session` wrote it."""

    time: int
    """When the resume began, and so when this incarnation of the log did."""
    events: int
    """How many events the store held when it began: all of the log that survived."""
    interrupted: bool
    """Whether the repair closed anything: a turn, or an intent left open between turns."""
    closed: int
    turn: int | None = None
    """The turn that the repair closed, or `None` when the crash cut none. An intent
    left open between turns is `interrupted` without a turn. A resumer asks for
    this before it continues a turn (P13-04)."""

    @classmethod
    def of(cls, event: SessionEvent) -> Resumption:
        data = event.data
        return cls(
            time=event.time,
            events=as_int(data.get("events")),
            interrupted=data.get("interrupted") is True,
            closed=as_int(data.get("closed")),
            turn=as_int(data.get("turn")) if "turn" in data else None,
        )


def resumption_of(session: Session) -> Resumption | None:
    """What the last resume of this session recorded, or `None` if it was never
    resumed.

    The answer is read from the log and not returned from `resume_session`. So a
    front end that did not perform the resume learns it in the same way as the
    process that did: a TUI that attaches to a daemon root, or a trajectory
    reader that opens a file. Only the own resumes of the log count
    (`resumptions`).
    """
    event = session.latest("session/resumed")
    if event is None or event.seq < session.header.first_own_seq:
        return None
    return Resumption.of(event)


def cut_turn_owner(session: Session) -> CutTurnOwner | None:
    """Who the turn that the last resume of this log closed was owed to, or `None`
    when it is owed to nobody.

    It is owed to nobody when the resume closed no turn (`Resumption.turn`), or
    closed a turn that was cut before it took a prompt. That turn did nothing.
    Whatever it was started for is still in the inbox, where the next turn takes
    it. A continuation as well would be a second turn for one prompt.

    The answer is read from the prompts of the cut turn itself. The walk goes
    back from the last event that survived to the `turn/start` of that turn, so
    its length is that of one turn and not of the log. A `user` source anywhere
    in the turn makes it `person`. Any other prompt makes it `unattended`.
    """
    resumed = resumption_of(session)
    if resumed is None or resumed.turn is None:
        return None
    owner: CutTurnOwner | None = None
    for seq in range(resumed.events - 1, session.header.first_own_seq - 1, -1):
        event = session.at(seq)
        if event is None or event.type == "turn/start":
            break
        if event.type == "user/message":
            if as_obj(event.data.get("source")).get("kind") == "user":
                return "person"
            owner = "unattended"
    return owner


def cut_short_note(attempt: int, *, limit: int | None = None) -> str:
    """What an agent is told when a turn that a restart cut short is presented to it
    again: a readmitted child, or a root whose turn is continued. There is one
    wording, so the two cannot come to describe one interruption differently."""
    of = f" of {limit}" if limit is not None else ""
    return (
        "[the harness stopped while you were working, so the turn above was cut short; "
        f"this is attempt {attempt}{of}. Anything you finished is in your transcript — "
        "continue from there rather than starting over, and check a tool call whose "
        "outcome reads as unknown before repeating it.]"
    )


def resumptions(session: Session) -> list[Resumption]:
    """Every resume of this log, oldest first.

    **Only its own.** The seed of a fork carries the `session/resumed` records
    of its source. Taken as records of the fork, they would date the fork by the
    restarts of its source.
    """
    return [
        Resumption.of(event) for event in session.own_events() if event.type == "session/resumed"
    ]


async def resume_session(
    ctx: Context, session_id: str, header: SessionHeader, events: list[SessionEvent]
) -> Session:
    """Repair the crashed tail of a stored log and publish it. `events` is the log
    as `read` gave it. The caller reads it: `open_session`, which reads the log
    to learn whether there is one.

    The repair runs on the seed and not after publication. So a resumed session
    is provider-valid the first time that anything reads it. An open turn that
    reached `derive_messages()` would be rejected by the provider before anyone
    noticed that it was unclosed (A5).

    `interrupted` says whether the tail had to be closed. That is the correct
    signal for "this crashed" as against "this was reopened", because a clean
    stop synthesizes no closers.
    """
    calls, intents = await _reconciled(ctx, session_id, header, events)
    closers = interrupted_turn_closers(events, calls, intents=intents)
    cut = next((closer for closer in closers if closer.type == "turn/end"), None)
    revived = Session(session_id, seed=[*events, *closers], header=header, durable=len(events))
    # `durable=len(events)`: **the store already holds `events`, and nothing
    # else.** The closers are synthesized here. They are in the log and have not
    # been written. A backend that inferred durability from "the file exists"
    # dropped them and left a gap in the seq space, which `_readmit` refuses. So
    # the session resumed one time and never again. The rule is stated here
    # because this is the only place that knows the difference.
    sessions = ctx.require(SESSIONS)
    async with sessions.opening(sessions.adopt(revived)) as session:
        # The resume is recorded, not only returned. A resume is a fact about
        # *provenance*: this process picked up work that somebody else started.
        # Nothing else in the log gives that fact, because a session that was
        # reopened and one that ran straight through look identical afterwards.
        # It matters most where nobody is watching: in the daemon and in an
        # agent that cron started. The record lets `phern doctor`, a trajectory
        # reader or a person who scrolls back find where the resume happened.
        # There is one event for each reopen, not for each turn.
        _LOG.append(
            session,
            "session/resumed",
            {
                "events": len(events),
                "interrupted": bool(closers),
                "closed": len(closers),
                # Which turn, when one was cut. A resumer continues a turn, and
                # not an intent that repair settled between turns.
                **({"turn": as_int(cut.data.get("turn"))} if cut is not None else {}),
            },
        )
        # After the resume is recorded and before anyone holds the session:
        # settle what a log read off disk owes, which includes the trees that a
        # crash leaked.
        await sessions.loaded(session)
    return session


async def _reconciled(
    ctx: Context, session_id: str, header: SessionHeader, events: list[SessionEvent]
) -> tuple[dict[str, CallOutcome], dict[str, dict[str, CallOutcome]]]:
    """Ask each tool about its own started, unresolved call (P10-13), and about each
    open intent of a kind that declares `reconciled` (L6b).

    The intents are asked about because under Code Mode every call that the
    model makes is a dispatch (`TOOL_DISPATCH` declares `reconciled`). A `send`
    that a program made before it was interrupted is a dispatch. When only
    top-level calls were asked about, no tool that could check was ever asked.
    Answers come back by call id. For intents they come back by the opening type
    of the kind and the key of the intent, which is how repair looks them up.

    The question is asked here, where a deployment is mounted, and not in
    repair. Repair stays a pure fold that a stored log can be put through with
    nothing mounted, and only the answers are handed to it. The question is
    asked at `DEPLOYMENT` scope, because no agent exists yet. An agent-scoped
    tool is not seen and keeps `TOOL_OUTCOME_UNKNOWN`, as
    `ToolDefinition.reconcile` says. It goes through `ToolRuntime.reconciled`,
    which is the question that the pipeline asks. So a raise or an `Unknown` is
    no answer here either, and a `Done` is rendered as the row that registered
    the tool.

    The session that the tool is shown is a read-only copy of the stored log. It
    is built only when some call has a tool that can answer. That path is a
    crash with a call in flight. The copy is the price of letting a tool read
    its own context (a workspace root, for example) in the way that it would
    read a live one.
    """
    tools = ctx.get(TOOLS)
    if tools is None:
        return {}, {}
    view: Session | None = None

    async def ask(record: SessionEvent) -> CallOutcome | None:
        nonlocal view
        definition = tools.get(as_str(record.data.get("name")), scope=DEPLOYMENT)
        if definition is None or definition.reconcile is None:
            return None
        view = view or Session(session_id, seed=events, header=header)
        said = await tools.reconciled(record, view, scope=DEPLOYMENT)
        if said is None:
            return None
        if isinstance(said, tuple):
            return CallOutcome(done=True, content=tuple(block.to_wire() for block in said))
        return CallOutcome(done=False)

    calls: dict[str, CallOutcome] = {}
    for call in unresolved_calls(events):
        if (outcome := await ask(call)) is not None:
            calls[as_str(call.data.get("callId"))] = outcome
    intents: dict[str, dict[str, CallOutcome]] = {}
    for kind in declared_intents():
        if kind.reconciled is None:
            continue
        for intent in open_intents(events, kind):
            if (outcome := await ask(intent.opened)) is not None:
                intents.setdefault(kind.opened, {})[intent.key] = outcome
    return calls, intents
