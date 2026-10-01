"""`ctx.session_telemetry` — records, redaction, and no span tracer.

**The session log is the trace** (§8). There is deliberately no span
hierarchy: spans would be a second, lossier account of what already exists
event-by-event in the log, and two accounts of one run diverge.

What this seam adds is *export*: a record stream a sink can ship somewhere,
with one hard ordering rule — **every record passes the
`session-telemetry/record` redaction waterfall before any sink sees it**. A sink
registered as a listener alongside redaction could observe an unredacted record
by winning a race; a sink registered through `add_sink` cannot, because it runs
after the waterfall settles.

Ledger records mirror session events one-to-one with one exception: only the
*first* `assistant/chunk` per step ships, because a token-by-token export is
thousands of records saying the same thing, and the first is the one that
carries the latency signal.

**A ledger record ships once the log holds its event, and not before.** The
record names its event by `(session, seq)`, and `session/event` fires on the
in-memory append. A crash loses whatever no flush had written, and the resumed
log appends at its own length, so a record exported off the firehose could name
one event while the log, after a resume, put another at that seq. So the row
keeps a cursor per session and, when `session/durable` says how much the log
holds (`SessionStore.flush`), reads the events up to there off the log, the way a
store reads what it owes. Nothing else changes: the record still carries the
event's own `time`, so the latency the first chunk reports is the one it had, and
the export lags only by the distance to the next flush, which every step, tool
call and turn end makes.

@module ph.seams.telemetry
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, TypeAlias

import anyio

from ..cordis import (
    Context,
    Disposer,
    MaybeAwaitable,
    Running,
    events,
    maybe_await,
    plugin,
    running,
    settled_or_none,
)
from ..json import as_int, dumps
from ..keys import SESSION_TELEMETRY, SESSIONS
from ..paths import default_home_path, write_text_under
from ..session import Session, SessionEvent, now_ms
from ..wire import WireModel
from ._registry import claim_entry

__all__ = [
    "SessionTelemetry",
    "SessionTelemetryRecord",
    "apply",
    "ops_record",
]

log = logging.getLogger("ph.seams.telemetry")

Channel: TypeAlias = Literal["ledger", "ops"]
Severity: TypeAlias = Literal["debug", "info", "warn", "error"]

events.declare(
    "session-telemetry/record",
    "waterfall",
    owner="ph.seams.telemetry",
    doc="Redaction. Runs before any sink; a listener may rewrite or drop a record.",
)


class SessionTelemetryRecord(WireModel):
    """One exportable record."""

    channel: Channel
    time: int
    severity: Severity
    attributes: dict[str, Any]
    body: str


@dataclass(frozen=True, slots=True)
class _Sink:
    """An exporter and who registered it (P6-29).

    A sink is a row's body this seam invokes, once per record — the same category
    as a tool's `execute`, and it ran unbound, so an exporter that registered
    anything landed it on the seam and outlived its row."""

    export: Callable[[SessionTelemetryRecord], MaybeAwaitable[None]]
    by: Running


_EXPORTING: ContextVar[bool] = ContextVar("ph.seams.telemetry.exporting", default=False)
"""Whether *this task* is already inside a fan-out, so a sink cannot feed itself.

A sink that records what it just failed to ship — through `ops_record`, or any
indirection reaching it — would re-enter `record` and fan out to itself, a loop
with no floor. The rule was a docstring asking each future sink author to
remember; this is the same rule where it cannot be forgotten. Dropping the
nested record is the intended answer, not a cost: it is a record *about* the
export path, made while that path is the thing failing.

**Per task, and that is the whole correction.** It was an instance field, so the
flag was raised for the *deployment* while any one record was in flight — and
records are shipped from many tasks (a batch from each flush, `ops_record` from
wherever the operation is). Two records a millisecond apart therefore raced, and
the loser was dropped silently: exactly the bursts worth exporting — a turn's
`step/start`, its first chunk, its `turn/end` — arrived as one record in three.
Re-entrancy is a property of the call stack, so it is tracked where the call
stack is. A task a sink spawns inherits the flag, which is right: it is still the
export path.
"""


@dataclass(slots=True)
class SessionTelemetry:
    """The service published as `ctx.session_telemetry`."""

    ctx: Context
    _sinks: list[_Sink] = field(default_factory=list)
    _last_chunked_step: dict[Session, tuple[int, int]] = field(default_factory=dict)
    """Per session, the step whose first chunk already shipped. One entry per
    session rather than one per step, so it does not grow with the conversation."""
    _shipped: dict[Session, int] = field(default_factory=dict)
    """Per live session, how far the ledger has read its log: every event below
    this seq, and none past it.

    Keyed by the session and not its id, because a resume in the same process
    makes a second copy of a log under the same id, and a flush of the first
    copy finishing late says nothing about the second's events."""

    def add_sink(
        self,
        sink: Callable[[SessionTelemetryRecord], MaybeAwaitable[None]],
        *,
        scope: Context | None = None,
    ) -> Disposer:
        """Register an exporter. It sees only post-redaction records."""
        by = self.ctx.running_for(scope)
        return claim_entry(by.owner, self._sinks, _Sink(sink, by), label="telemetry.sink")

    async def record(self, record: SessionTelemetryRecord) -> None:
        """Redact, then fan out. A dropped record reaches no sink."""

        async def inner(candidate: SessionTelemetryRecord) -> SessionTelemetryRecord | None:
            return candidate

        answered = await self.ctx.waterfall("session-telemetry/record", record, inner=inner)
        redacted = settled_or_none("session-telemetry/record", answered, SessionTelemetryRecord)
        if redacted is None or _EXPORTING.get():
            return
        token = _EXPORTING.set(True)
        try:
            for sink in list(self._sinks):
                try:
                    # No target: telemetry is deployment-wide, so both halves are
                    # what registration recorded.
                    with running(sink.by):
                        await maybe_await(sink.export(redacted))
                except Exception:
                    log.exception("ph.seams.telemetry: a sink failed")
        finally:
            _EXPORTING.reset(token)

    def _wants(self, session: Session, event: SessionEvent) -> bool:
        """Whether this event ships. Only the first `assistant/chunk` per step does."""
        if event.type != "assistant/chunk":
            return True
        step = (as_int(event.data.get("turn")), as_int(event.data.get("step")))
        if self._last_chunked_step.get(session) == step:
            return False
        self._last_chunked_step[session] = step
        return True

    def _track(self, session: Session) -> None:
        """Start a session's cursor at its end: its seed was never an event here."""
        self._shipped.setdefault(session, session.seq)

    def _durable(self, session: Session, through: int) -> list[SessionEvent]:
        """What the ledger ships now that the log holds every event below `through`.

        Nothing is read when no sink would receive it: a record is fanned out to
        the sinks there are when it is made, so there is nobody to read it for."""
        start = self._shipped.get(session)
        if start is None or through <= start:
            return []
        self._shipped[session] = through
        if not self._sinks:
            return []
        owed = session.events_from(start, through - start)
        return [event for event in owed if self._wants(session, event)]

    def _forget(self, session: Session) -> None:
        """Drop one session's cursor. What no flush confirmed is not exported:
        nothing says the log holds it."""
        self._shipped.pop(session, None)
        self._last_chunked_step.pop(session, None)

    async def observe(self, session: Session, events: Sequence[SessionEvent]) -> None:
        """Mirror session events onto the ledger channel, in the order given."""
        for event in events:
            await self.record(
                SessionTelemetryRecord(
                    channel="ledger",
                    time=event.time,
                    severity="info",
                    attributes={
                        "session.id": session.id,
                        "event.type": event.type,
                        "event.seq": event.seq,
                    },
                    body=event.type,
                )
            )

    async def ops(
        self,
        body: str,
        *,
        severity: Severity = "info",
        **attributes: object,
    ) -> None:
        """Record something about the harness rather than the conversation."""
        await self.record(
            SessionTelemetryRecord(
                channel="ops",
                time=now_ms(),
                severity=severity,
                attributes=attributes,
                body=body,
            )
        )


async def ops_record(
    ctx: Context,
    body: str,
    *,
    severity: Severity = "info",
    **attributes: object,
) -> None:
    """Record something about the harness, if this deployment has the seam (P5-09).

    The producer's door. A row or a host holding a fact that is *not* a session
    event — an open refused because another process holds the log, a store that
    cannot take the I-5 lease, a workspace provider that broke and left an agent
    uncontained — says it here and never looks for the seam itself. No seam, no
    record and no error: telemetry is optional, and a producer must not fail its
    own work for want of somewhere to put a note about it.

    A sink that failed and reported its own failure here would fan out to the
    same sink, which is a loop with no floor; `record` refuses a nested fan-out
    for that reason, so reaching this from inside a sink drops the record rather
    than recursing.
    """
    telemetry = ctx.get(SESSION_TELEMETRY)
    if telemetry is None:
        return
    try:
        await telemetry.ops(body, severity=severity, **attributes)
    except Exception:
        log.exception("ph.seams.telemetry: an ops record could not be made")


class Config(WireModel):
    """Row config for the JSONL telemetry sink."""

    path: str | None = None
    enabled: bool = True


@plugin("session-telemetry", affects="deployment", config=Config, inject=[SESSIONS])
async def apply(ctx: Context, config: Config) -> None:
    """Mount the telemetry seam and, when enabled, the JSONL sink."""
    telemetry = SessionTelemetry(ctx=ctx)
    ctx.provide(SESSION_TELEMETRY, telemetry)

    if config.enabled:
        path = default_home_path(config.path, "telemetry.jsonl")

        async def write(record: SessionTelemetryRecord) -> None:
            await anyio.to_thread.run_sync(
                partial(write_text_under, path, f"{dumps(record.to_wire())}\n", append=True)
            )

        telemetry.add_sink(write)

    def on_durable(session: Session, through: int) -> MaybeAwaitable[None]:
        # The returned coroutine is scheduled by `emit`, never awaited on the
        # flush path; a flush that released nothing costs no task at all.
        ready = telemetry._durable(session, through)
        return telemetry.observe(session, ready) if ready else None

    # Sessions already live when the row (re)activates are read from here on,
    # as the firehose listener it replaced would have heard them.
    for session in ctx.require(SESSIONS).list():
        telemetry._track(session)
    ctx.on("session/created", telemetry._track)
    ctx.on("session/durable", on_durable)
    ctx.on("session/disposed", telemetry._forget)
