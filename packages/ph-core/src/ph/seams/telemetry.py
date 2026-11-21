"""`ctx.session_telemetry`: records, redaction, and no span tracer.

**The session log is the trace** (§8). The seam deliberately has no span
hierarchy. Spans would be a second, lossier account of what the log already
holds event by event, and two accounts of one run diverge.

This seam adds *export*: a stream of records that a sink can ship somewhere. It
has one hard ordering rule: **every record passes the
`session-telemetry/record` redaction waterfall before any sink sees it**. A
sink that was registered as a listener beside redaction could win a race and
see an unredacted record. A sink that is registered through `add_sink` cannot,
because it runs after the waterfall settles.

Ledger records mirror session events one-to-one, with one exception: only the
*first* `assistant/chunk` of each step ships. A token-by-token export is
thousands of records that say the same thing, and the first record carries the
latency signal. A record names its event and carries only the fields that the
type of the event exports (`ph.session.exported_fields`): why the work failed,
was refused or stopped.

**A ledger record ships after the log holds its event, and not before.** The
record names its event by `(session, seq)`, and `session/event` fires on the
in-memory append. A crash loses whatever no flush had written, and the resumed
log appends at its own length. So a record exported from `session/event` could
name one event while the log, after a resume, put another event at that seq.
The row therefore keeps a cursor for each session. When `session/durable`
reports how much the log holds (`SessionStore.flush`), the row reads the events
up to that point from the log, as a store reads what it owes. Nothing else
changes. The record still carries the `time` of the event itself, so the
latency that the first chunk reports is the latency that it had. The export
lags only by the distance to the next flush, and every step, tool call and turn
end makes a flush.

**The shutdown tail ships before the row goes.** This row sits in the log
layer, beneath the agents, and rows unwind in reverse. So the last barrier of
the checkpoint policy flushes what the agents append as they let go, while this
row and the sinks beside it are still mounted. That tail ships like any other
record. The barrier waits for that fan-out before the rows beneath it go.

**A write that failed ships too, on the ops channel.** No event names it,
because the log is what could not take it. So the row ships the
`session/unwritten` of the store as an `error` record that carries what the
write reported.
`docs/seams/session_telemetry.md` has the rest.

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

from ..cancel import reason_of
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
from ..json import PlainJsonValue, as_int, dumps, thaw_json
from ..keys import SESSION_TELEMETRY, SESSIONS
from ..paths import default_home_path, write_text_under
from ..session import Session, SessionEvent, SessionTable, exported_fields, now_ms
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

    A sink is the body of a row, and this seam invokes it one time for each
    record. It is in the same category as the `execute` of a tool. It used to
    run unbound, so anything that an exporter registered landed on the seam and
    outlived its row."""

    export: Callable[[SessionTelemetryRecord], MaybeAwaitable[None]]
    by: Running


_EXPORTING: ContextVar[bool] = ContextVar("ph.seams.telemetry.exporting", default=False)
"""Whether *this task* is already inside a fan-out, so that a sink cannot feed itself.

A sink that records what it just failed to ship, through `ops_record` or
through any indirection that reaches it, would enter `record` again and fan out
to itself. That loop has no end. The rule used to be a docstring that asked
each future sink author to remember it. This flag is the same rule, in a place
where nobody can forget it. To drop the nested record is the intended answer
and not a cost: it is a record *about* the export path, made while that path is
the thing that fails.

**The flag is per task, and that is the complete correction.** It was an
instance field, so the flag was raised for the *deployment* while any one
record was in flight. Records are shipped from many tasks: a batch from each
flush, and `ops_record` from wherever the operation is. So two records a
millisecond apart raced, and the loser was dropped silently. Those are exactly
the bursts worth exporting: the `step/start` of a turn, its first chunk and its
`turn/end` arrived as one record in three. Re-entrancy is a property of the
call stack, so the flag is tracked where the call stack is. A task that a sink
spawns inherits the flag. That is correct, because the task is still on the
export path.
"""


@dataclass(slots=True)
class _Cursor:
    """How far the ledger has come through the log of one session."""

    read: int
    """Every event below this seq has been read for shipping, and none past it."""
    chunked: tuple[int, int] | None = None
    """The step whose first chunk already shipped. There is one per session and
    not one per step, so it does not grow with the conversation."""

    def wants(self, event: SessionEvent) -> bool:
        """Whether this event ships. Only the first `assistant/chunk` per step does."""
        if event.type != "assistant/chunk":
            return True
        step = (as_int(event.data.get("turn")), as_int(event.data.get("step")))
        if self.chunked == step:
            return False
        self.chunked = step
        return True


@dataclass(slots=True)
class SessionTelemetry:
    """The service published as `ctx.session_telemetry`."""

    ctx: Context
    _sinks: list[_Sink] = field(default_factory=list)
    _cursors: SessionTable[_Cursor] = field(default_factory=SessionTable)
    """For each live session, how far the ledger has come through its log. The
    entry goes with the session.

    The key is the session and not its id. A resume in the same process makes a
    second copy of a log under the same id, and a late flush of the first copy
    says nothing about the events of the second copy."""

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

    def _track(self, session: Session) -> None:
        """Give a session that is already live when the row activates a cursor at the
        current end of its log. What the session held before then was for the
        earlier life of this row to ship."""
        self._cursors.keep(session, _Cursor(read=session.seq))

    def _created(self, session: Session) -> None:
        """Start the cursor of a new session where the writing of this process begins.

        That point is `durable_length`: what a store already held when the
        session was made. The prefix of a fork was for its parent to ship, and a
        stored log was for the run that wrote it to ship. So the repair closers
        of a resume ship, because the store does not hold them yet. A cursor at
        the end of the seed skipped them."""
        self._cursors.keep(session, _Cursor(read=session.durable_length))

    def _durable(self, session: Session, through: int) -> MaybeAwaitable[None]:
        """Ship what the ledger owes, now that the log holds every event below `through`.

        The fan-out is returned for `emit` to schedule, and is never awaited on
        the flush path. A flush that released nothing costs no task. Nothing is
        read when no sink would receive it: a record is fanned out to the sinks
        that exist when it is made, so there is nobody to read it for."""
        cursor = self._cursors.get(session)
        if cursor is None or through <= cursor.read:
            return None
        start, cursor.read = cursor.read, through
        if not self._sinks:
            return None
        owed = session.events_from(start, through - start)
        ready = [event for event in owed if cursor.wants(event)]
        return self.observe(session, ready) if ready else None

    def _unwritten(self, session: Session, error: Exception) -> MaybeAwaitable[None]:
        """Ship a write of `session` that raised, with its reason. Ship nothing when
        no sink would receive it, as `_durable` does."""
        if not self._sinks:
            return None
        reason = reason_of(error)
        return self.ops(
            f"session {session.id} could not be written: {reason}",
            severity="error",
            session_id=session.id,
            reason=reason,
        )

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
                        **_exported(event),
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


def _exported(event: SessionEvent) -> dict[str, PlainJsonValue]:
    """The fields that the type of `event` exports, each as `event.<field>`. They
    are thawed, so a sink receives plain JSON. A field that the payload leaves
    out ships nothing."""
    return {
        f"event.{name}": thaw_json(event.data[name])
        for name in exported_fields(event.type)
        if event.data.get(name) is not None
    }


async def ops_record(
    ctx: Context,
    body: str,
    *,
    severity: Severity = "info",
    **attributes: object,
) -> None:
    """Record something about the harness, if this deployment has the seam (P5-09).

    This is the function that a producer calls. A row or a host can hold a fact
    that is *not* a session event: an open that was refused because another
    process holds the log, a store that cannot take the I-5 lease, or a
    workspace provider that broke and left an agent uncontained. The producer
    reports the fact here and never looks for the seam itself. With no seam
    there is no record and no error. Telemetry is optional, and a producer must
    not fail its own work because it has nowhere to put a note about that work.

    A sink that failed and reported its own failure here would fan out to the
    same sink, which is a loop with no end. `record` refuses a nested fan-out
    for that reason, so a call from inside a sink drops the record and does not
    recurse.
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


@plugin(
    "session-telemetry",
    affects="deployment",
    config=Config,
    inject=[SESSIONS],
    log_layer="telemetry",
)
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

    for session in ctx.require(SESSIONS).list():
        telemetry._track(session)
    ctx.on("session/created", telemetry._created)
    ctx.on("session/durable", telemetry._durable)
    ctx.on("session/unwritten", telemetry._unwritten)
