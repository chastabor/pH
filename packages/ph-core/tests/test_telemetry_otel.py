"""P5-09 — the OTel sink: what a collector actually receives.

The seam's own ordering guarantee is asserted in `test_telemetry.py`, where it
belongs. What is left here is this row: that a mounted `session-telemetry-otel`
registers through `add_sink` and therefore inherits that ordering end to end,
and that the severity table it writes out by hand still matches OTel's scale.

## The half-mount this row refuses

An earlier draft treated the two optional distributions differently: it refused
without the SDK, but built a provider **with no exporter** when only the OTLP half
was missing. That is the silent no-op reached through the other door — `ph doctor`
would have reported a healthy telemetry sink shipping to nowhere. Either the row
can export or it refuses.
"""

from __future__ import annotations

import sys
from typing import Any, get_args

import pytest

from ph.seams.telemetry import SessionTelemetryRecord

pytestmark = pytest.mark.anyio

OTEL_ROW = {"insert": [{"id": "session-telemetry-otel", "name": "session-telemetry-otel"}]}


def _record(body: str, **attributes: Any) -> SessionTelemetryRecord:
    return SessionTelemetryRecord(
        channel="ledger", time=1_000, severity="info", attributes=attributes, body=body
    )


async def test_the_shipped_sink_exports_post_redaction_records(
    mount: Any, monkeypatch: pytest.MonkeyPatch
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
    from opentelemetry.sdk._logs.export import (
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )

    import ph.seams.telemetry_otel as row

    exported = InMemoryLogRecordExporter()
    monkeypatch.setattr(row, "_pipeline", lambda config: SimpleLogRecordProcessor(exported))

    ctx = await mount(OTEL_ROW)

    async def redact(record: SessionTelemetryRecord, next_: Any) -> Any:
        return await next_(
            record.model_copy(update={"body": record.body.replace("hunter2", "«x»")})
        )

    ctx.on("session-telemetry/record", redact)
    await ctx.session_telemetry.record(_record("token hunter2", tool="bash"))

    emitted = exported.get_finished_logs()
    assert emitted, "the row's sink exported nothing"
    body = emitted[-1].log_record
    assert body.body == "token «x»", "an unredacted record reached the exporter"
    assert body.attributes["ph.channel"] == "ledger"
    assert body.attributes["ph.tool"] == "bash"
    assert body.severity_text == "info"


async def test_the_row_refuses_rather_than_shipping_nowhere(
    mount: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Half the extra is not a working exporter, and must not mount as one.

    The SDK brings the provider; the OTLP *distribution* brings a transport. An
    earlier draft refused for the first and quietly no-opped for the second, so
    a deployment with `opentelemetry-sdk` alone got a `ph doctor` reporting a
    healthy telemetry sink that shipped nothing. Blocking the import is how CI
    reaches a branch the dev group otherwise makes unreachable — the extra is
    installed for every other test in this file.
    """
    from ph.seams.telemetry_otel import MISSING

    monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.http._log_exporter", None)
    with pytest.raises(RuntimeError, match="ph-core\\[otel\\]") as refusal:
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
