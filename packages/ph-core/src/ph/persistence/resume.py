"""Resuming a stored log: the repair of a crashed tail, what tools say of the calls
it cut off, the `session/resumed` record, and the readers of that record.

Backend-neutral, and so not a backend's. It lived in the JSONL store's module,
though nothing in it touches a file: `open_session` reads the log through the
Protocol and hands it here, so a database backend resumes through the same code —
and the `session/resumed` writer is this module, not whichever backend is mounted.

@module ph.persistence.resume
"""

from __future__ import annotations

from dataclasses import dataclass

from ..cordis import DEPLOYMENT, Context
from ..json import as_int, as_str
from ..keys import SESSIONS, TOOLS
from ..session import Session, SessionEvent, SessionHeader, declared_intents, open_intents
from ..session.writers import log_writer
from .repair import CallOutcome, interrupted_turn_closers, unresolved_calls

_LOG = log_writer(__name__)

__all__ = ["Resumption", "resume_session", "resumption_of", "resumptions"]


@dataclass(frozen=True, slots=True)
class Resumption:
    """One `session/resumed` record, as `resume_session` wrote it."""

    time: int
    """When the resume began, and so when this incarnation of the log did."""
    events: int
    """How many events the store held when it began: all of the log that survived."""
    interrupted: bool
    closed: int

    @classmethod
    def of(cls, event: SessionEvent) -> Resumption:
        data = event.data
        return cls(
            time=event.time,
            events=as_int(data.get("events")),
            interrupted=data.get("interrupted") is True,
            closed=as_int(data.get("closed")),
        )


def resumption_of(session: Session) -> Resumption | None:
    """What this session's last resume recorded, or `None` if it never was.

    Read from the log rather than returned from `resume_session`, so a front end
    that did not perform the resume — a TUI attaching to a daemon root, a
    trajectory reader opening a file — learns it the same way as the process
    that did. Only the log's own resumes count (`resumptions`).
    """
    event = session.latest("session/resumed")
    if event is None or event.seq < session.header.first_own_seq:
        return None
    return Resumption.of(event)


def resumptions(session: Session) -> list[Resumption]:
    """Every resume of this log, oldest first.

    **Its own only.** A fork's seed carries its source's `session/resumed`, and
    taken as the fork's they would date the fork by its source's restarts.
    """
    return [
        Resumption.of(event) for event in session.own_events() if event.type == "session/resumed"
    ]


async def resume_session(
    ctx: Context, session_id: str, header: SessionHeader, events: list[SessionEvent]
) -> Session:
    """Repair a stored log's crashed tail and publish it — the log as `read` gave it,
    read by the caller: `open_session`, which reads it to learn whether there is one.

    The repair runs on the seed rather than after publication, so a resumed
    session is provider-valid the first time anything reads it — an open turn
    that reached `derive_messages()` would be rejected by the provider before
    anyone noticed it was unclosed (A5).

    `interrupted` says whether the tail had to be closed, which is the honest
    signal for "this crashed" as against "this was reopened": a clean stop
    synthesizes no closers.
    """
    calls, intents = await _reconciled(ctx, session_id, header, events)
    closers = interrupted_turn_closers(events, calls, intents=intents)
    revived = Session(session_id, seed=[*events, *closers], header=header, durable=len(events))
    # `durable=len(events)`: **what the store already holds is `events`, and
    # nothing else.** The closers are synthesized here; they are in the log and
    # have not been written. A backend that inferred durability from "the file
    # exists" dropped them and left a gap in the seq space, which `_readmit`
    # refuses — so the session resumed once and never again. Said here because
    # this is the only place that knows the difference.
    sessions = ctx.require(SESSIONS)
    async with sessions.opening(sessions.adopt(revived)) as session:
        # Recorded, not just returned. A resume is a fact about *provenance* — this
        # process picked up work somebody else started — and it is not derivable
        # from anything else in the log: a session that was reopened and one that
        # ran straight through look identical afterwards. It matters most where
        # nobody is watching, which is the daemon and a cron-started agent, and it
        # is what lets `phern doctor`, a trajectory reader or a person scrolling
        # back find the seam. One event per reopen, not per turn.
        _LOG.append(
            session,
            "session/resumed",
            {
                "events": len(events),
                "interrupted": bool(closers),
                "closed": len(closers),
            },
        )
        # After the resume is recorded and before anyone holds the session: what a
        # log read off disk owes, a crash's leaked trees among it.
        await sessions.loaded(session)
    return session


async def _reconciled(
    ctx: Context, session_id: str, header: SessionHeader, events: list[SessionEvent]
) -> tuple[dict[str, CallOutcome], dict[str, dict[str, CallOutcome]]]:
    """Ask each tool about its own started, unresolved call (P10-13), and about each
    open intent of a kind that declares `reconciled` (L6b).

    The intents because under Code Mode every call the model makes is a dispatch
    (`TOOL_DISPATCH` declares `reconciled`): a `send` a program made before it was
    interrupted is one, and asked only about top-level calls, no tool that could
    check ever was. Answers come back by call id, and for intents by the kind's
    opening type and the intent's key, which is how repair looks them up.

    Here, mounted, and not in repair — which stays a pure fold a stored log can
    be put through with nothing mounted; only the answers are handed to it. At
    `DEPLOYMENT` scope, since no agent exists yet: an agent-scoped tool is not
    seen, and keeps `TOOL_OUTCOME_UNKNOWN`, as `ToolDefinition.reconcile` says.
    Through `ToolRuntime.reconciled`, the pipeline's own question, so a raise or
    an `Unknown` is no answer here either and a `Done` is rendered as the row that
    registered the tool.

    The session the tool is shown is a read-only copy of the stored log, built
    only when some call has a tool that can answer — the path is a crash with a
    call in flight, and the copy is the price of letting a tool read its own
    context (a workspace root, say) the way it would read a live one.
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
