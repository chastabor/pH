"""P5-09 — the OTel sink: what a collector actually receives.

The seam's own ordering guarantee is asserted in `test_telemetry.py`, where it
belongs. What is left here is this row: that a mounted `session-telemetry-otel`
registers through `add_sink` and therefore inherits that ordering end to end,
and that the severity table it writes out by hand still matches OTel's scale.

## The half-mount this row refuses

An earlier draft treated the two optional distributions differently: it refused
without the SDK, but built a provider **with no exporter** when only the OTLP half
was missing. That is the silent no-op reached through the other door — `phern doctor`
would have reported a healthy telemetry sink shipping to nowhere. Either the row
can export or it refuses.

## Two per-record costs resolved at mount instead

**`SCALARS` is a module-level tuple.** A `str | int | float | bool` expression
inside `_flat` is evaluated per attribute per record: **136 ns against 40 ns** for
the tuple.

**The severity table is built once.** `SeverityNumber(...)` per record goes
through `Enum.__call__` and costs **~180 ns a record** for a table of four that
cannot change while the row is mounted.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, get_args

import pytest

from ph.cordis import MountRefusal
from ph.keys import AGENTS, SESSION_TELEMETRY, SESSIONS
from ph.persistence import open_session
from ph.seams.telemetry import SessionTelemetryRecord
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import MountProfile, log_event, not_none

if TYPE_CHECKING:
    from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter

pytestmark = pytest.mark.anyio

# Declared at the end, as a deployment adds it; its plugin places it in the log layer.
OTEL_ROW = {"id": "telemetry-otel", "name": "session-telemetry-otel"}


@pytest.fixture
def exported(monkeypatch: pytest.MonkeyPatch) -> InMemoryLogRecordExporter:
    """The row's transport, swapped for an in-memory exporter behind a simple processor.

    Only the transport: a test has to observe the export before a batch would have
    flushed. The sink registration, the ordering, the severity and the attribute
    mapping stay the shipped ones, so what a test asserts is what a collector gets.
    """
    from opentelemetry.sdk._logs.export import (
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    import ph.seams.telemetry_otel as row

    # opentelemetry's in-memory exporter ships untyped.
    exporter = InMemoryLogRecordExporter()  # type: ignore[no-untyped-call]
    monkeypatch.setattr(row, "_pipeline", lambda config: SimpleLogRecordProcessor(exporter))
    return exporter


def _record(body: str, **attributes: object) -> SessionTelemetryRecord:
    return SessionTelemetryRecord(
        channel="ledger", time=1_000, severity="info", attributes=attributes, body=body
    )


async def test_the_shipped_sink_exports_post_redaction_records(
    mount: MountProfile, exported: InMemoryLogRecordExporter
) -> None:
    """End to end through the row itself: what a collector would receive.

    Only the *transport* is swapped — `_pipeline` returns an in-memory exporter
    behind a simple processor instead of an OTLP client behind a batching one,
    because a test has to observe the export before a batch would have flushed.
    The sink registration, the ordering, the severity and the attribute mapping
    are all the shipped ones, so what is asserted here is what a collector gets.
    A test that registered its own sink would have proved the SDK works and
    nothing about this row.
    """
    ctx = await mount(OTEL_ROW)

    async def redact(record: SessionTelemetryRecord, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        return await next_(
            record.model_copy(update={"body": record.body.replace("hunter2", "«x»")})
        )

    ctx.on("session-telemetry/record", redact)
    await ctx.require(SESSION_TELEMETRY).record(_record("token hunter2", tool="bash"))

    emitted = exported.get_finished_logs()
    assert emitted, "the row's sink exported nothing"
    body = emitted[-1].log_record
    assert body.body == "token «x»", "an unredacted record reached the exporter"
    assert not_none(body.attributes)["ph.channel"] == "ledger"
    assert not_none(body.attributes)["ph.tool"] == "bash"
    assert body.severity_text == "info"


async def test_both_shipped_sinks_receive_the_shutdown_tail(
    mount: MountProfile, exported: InMemoryLogRecordExporter, tmp_path: Path
) -> None:
    """The row is written last, after `agent`, and its plugin places it in the log
    layer beside the JSONL sink, so what the agents append as they let go reaches
    both before either goes. The JSONL sink takes a thread per record, so the
    fan-out is still running once the last barrier has flushed, and this row shuts
    its provider down as it unwinds: the barrier has to wait for the fan-out, or the
    tail reaches neither sink, or only the JSONL one.

    Sabotage: drop the barrier's `drain`, or the row's `log_layer`, and the tail's
    `turn/end` misses the collector.
    """
    path = tmp_path / "telemetry.jsonl"
    ctx = await mount(OTEL_ROW, {"id": "telemetry", "config": {"enabled": True, "path": str(path)}})
    session = await open_session(ctx, "s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    agent.ctx.add_disposer(
        lambda: log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "aborted"}}),
        label="cancel",
    )
    log_event(session, "turn/start", {"turn": 1})
    await ctx.require(SESSIONS).flush(session)
    await ctx.dispose()

    collected = [one.log_record.body for one in exported.get_finished_logs()]
    written = [json.loads(line)["body"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert collected[-1] == "turn/end", "the shutdown tail never reached the collector"
    assert written[-1] == "turn/end", "the shutdown tail never reached the JSONL file"


async def test_the_row_refuses_rather_than_shipping_nowhere(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half the extra is not a working exporter, and must not mount as one.

    The SDK brings the provider; the OTLP *distribution* brings a transport. An
    earlier draft refused for the first and quietly no-opped for the second, so
    a deployment with `opentelemetry-sdk` alone got a `phern doctor` reporting a
    healthy telemetry sink that shipped nothing. Blocking the import is how CI
    reaches a branch the dev group otherwise makes unreachable — the extra is
    installed for every other test in this file.
    """
    from ph.seams.telemetry_otel import MISSING

    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http._log_exporter", None)
    with pytest.raises(MountRefusal, match="ph-core\\[otel\\]") as refusal:
        await mount(OTEL_ROW)
    assert str(refusal.value) == MISSING


def test_every_severity_maps_to_an_otel_number() -> None:
    """The four pH severities, against OTel's own scale — and against pH's.

    Written out rather than imported so the mapping reads without the SDK in
    front of you, which means a test has to hold the two together. Held against
    `get_args(Severity)` too, so a fifth severity added to the seam fails here
    rather than reaching a collector as INFO through a fallback.
    """
    from opentelemetry._logs import SeverityNumber

    from ph.seams.telemetry import Severity
    from ph.seams.telemetry_otel import SEVERITY

    assert set(SEVERITY) == set(get_args(Severity))
    assert {
        "debug": SeverityNumber.DEBUG.value,
        "info": SeverityNumber.INFO.value,
        "warn": SeverityNumber.WARN.value,
        "error": SeverityNumber.ERROR.value,
    } == SEVERITY
