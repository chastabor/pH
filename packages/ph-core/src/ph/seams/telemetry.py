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

@module ph.seams.telemetry
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Literal, TypeAlias

import anyio

from ..cordis import Context, Disposer, Running, events, maybe_await, plugin, running
from ..paths import default_home_path, write_text_under
from ..session import Session, SessionEvent, dumps, now_ms
from ..wire import WireModel
from ._registry import claim_entry

__all__ = ["SessionTelemetry", "SessionTelemetryRecord", "apply"]

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

    export: Callable[[SessionTelemetryRecord], Any]
    by: Running


@dataclass(slots=True)
class SessionTelemetry:
    """The service published as `ctx.session_telemetry`."""

    ctx: Context
    _sinks: list[_Sink] = field(default_factory=list)
    _last_chunked_step: dict[str, tuple[int, int]] = field(default_factory=dict)
    """Per session, the step whose first chunk already shipped. One entry per
    session rather than one per step, so it does not grow with the conversation."""

    def add_sink(
        self, sink: Callable[[SessionTelemetryRecord], Any], *, scope: Context | None = None
    ) -> Disposer:
        """Register an exporter. It sees only post-redaction records."""
        by = self.ctx.running_for(scope)
        return claim_entry(by.owner, self._sinks, _Sink(sink, by), label="telemetry.sink")

    async def record(self, record: SessionTelemetryRecord) -> None:
        """Redact, then fan out. A dropped record reaches no sink."""

        async def inner(candidate: SessionTelemetryRecord) -> SessionTelemetryRecord | None:
            return candidate

        redacted = await self.ctx.waterfall("session-telemetry/record", record, inner=inner)
        if redacted is None:
            return
        for sink in list(self._sinks):
            try:
                # No target: telemetry is deployment-wide, so both halves are
                # what registration recorded.
                with running(sink.by):
                    await maybe_await(sink.export(redacted))
            except Exception:
                log.exception("ph.seams.telemetry: a sink failed")

    def wants(self, session: Session, event: SessionEvent) -> bool:
        """Whether this event ships — decided synchronously, so a dropped chunk
        costs no task. Only the first `assistant/chunk` per step does."""
        if event.type != "assistant/chunk":
            return True
        step = (int(event.data.get("turn", 0)), int(event.data.get("step", 0)))
        if self._last_chunked_step.get(session.id) == step:
            return False
        self._last_chunked_step[session.id] = step
        return True

    async def observe(self, session: Session, event: SessionEvent) -> None:
        """Mirror one session event onto the ledger channel."""
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

    async def ops(self, body: str, *, severity: Severity = "info", **attributes: Any) -> None:
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


class Config(WireModel):
    """Row config for the JSONL telemetry sink."""

    path: str | None = None
    enabled: bool = True


@plugin("session-telemetry", config=Config, inject=["sessions"])
async def apply(ctx: Context, config: Config) -> None:
    """Mount the telemetry seam and, when enabled, the JSONL sink."""
    telemetry = SessionTelemetry(ctx=ctx)
    ctx.provide("session_telemetry", telemetry)

    if config.enabled:
        path = default_home_path(config.path, "telemetry.jsonl")

        async def write(record: SessionTelemetryRecord) -> None:
            await anyio.to_thread.run_sync(
                partial(write_text_under, path, f"{dumps(record.to_wire())}\n", append=True)
            )

        telemetry.add_sink(write)

    def on_event(session: Session, event: SessionEvent) -> Any:
        # The returned coroutine is scheduled by `emit`, never awaited on the
        # append path; a chunk that will not ship returns nothing and costs no
        # task at all.
        return telemetry.observe(session, event) if telemetry.wants(session, event) else None

    ctx.on("session/event", on_event)
