"""`session-telemetry-otel` — ship the ledger to an OTel collector (P5-09).

A **sink**, registered through `SessionTelemetry.add_sink`, which is the whole
security argument: `add_sink` runs after the `session-telemetry/record`
redaction waterfall settles, so an exporter physically cannot observe a record
the redactors dropped or rewrote. A row that listened on the waterfall event
*alongside* redaction could see the unredacted form by winning a race; this one
has no such race to win, and the row's gate — "redaction still precedes export"
— is that property asserted rather than a rule anyone has to remember.

**Logs, not spans, and that is §8's decision rather than a shortcut.** The seam
says it out loud: "the session log is the trace", and a span hierarchy would be
a second, lossier account of what the log already holds event by event. So each
record becomes one OTel *log record*, carrying the channel and severity pH
already assigned. Anything wanting a waterfall view builds it from the log.

**Optional, and honest when absent.** `opentelemetry-sdk` and the OTLP exporter
distribution are an extra (`ph-core[otel]`), imported inside `apply` so a
deployment that never exports does not pay them, and a profile that mounts this
row without them gets a refusal naming what to install rather than an
`ImportError` from the loader.

**Nothing here half-mounts.** An earlier draft treated the two distributions
differently — refusing without the SDK, but building a provider with no
exporter when only the OTLP half was missing. That is the silent no-op the
refusal below exists to avoid, reached through the other door: `ph doctor` would
have reported a healthy telemetry sink shipping to nowhere. Either the row can
export or it refuses.

@module ph.seams.telemetry_otel
"""

from __future__ import annotations

from typing import Any

from ..cordis import Context, plugin
from ..session import dumps
from ..wire import WireModel
from .diagnostics import Diagnostic, contribute
from .telemetry import SessionTelemetryRecord

__all__ = ["Config", "apply"]

MISSING = (
    "session-telemetry-otel needs the OpenTelemetry SDK and its OTLP exporter, "
    "which are an extra: install `ph-core[otel]`, or remove this row."
)
"""Why the row refused, and the two ways out.

A refusal rather than a silent no-op: a deployment that mounted an exporter and
got nothing exported would discover it from the absence of data somewhere else,
which is the worst place to learn it.
"""

SEVERITY = {"debug": 5, "info": 9, "warn": 13, "error": 17}
"""pH's four severities as OTel severity numbers.

The numbers are OTel's own scale (DEBUG=5, INFO=9, WARN=13, ERROR=17) written
out rather than imported, so this mapping is readable without the SDK in front
of you — and so the module's only hard dependency stays inside `apply`. The
cost of writing them out is that a fifth pH severity could map to nothing; the
test holds these keys against `get_args(Severity)` so that fails loudly rather
than exporting the new severity as INFO.
"""

SCALARS = (str, int, float, bool)
"""What OTel accepts as an attribute value. A tuple at module scope rather than
a `str | int | float | bool` expression inside `_flat`, because that expression
is evaluated per attribute per record — 136 ns against 40 ns for the tuple."""


class Config(WireModel):
    """Row config. `endpoint` empty means "whatever the OTEL_* environment says".

    Deliberately thin: the OTel SDK already reads `OTEL_EXPORTER_OTLP_ENDPOINT`
    and friends, and a second place to configure one exporter is a second place
    for them to disagree. Empty is passed straight through — the exporter's own
    `endpoint or environ.get(...)` gives exactly the unconfigured behaviour, so
    there is nothing for this row to branch on.
    """

    service_name: str = "ph"
    endpoint: str = ""


@plugin("session-telemetry-otel", inject=["session_telemetry"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register the exporter as a sink, or refuse with a reason."""
    try:
        from opentelemetry._logs import SeverityNumber
        from opentelemetry.sdk._logs import LoggerProvider
        from opentelemetry.sdk.resources import Resource

        # Inside the guard, because `_pipeline` imports the *other* half of the
        # extra: a deployment with the SDK but no OTLP exporter refuses here
        # rather than mounting a sink that ships nowhere.
        processor = _pipeline(config)
    except ImportError as error:
        raise RuntimeError(MISSING) from error

    provider = LoggerProvider(resource=Resource.create({"service.name": config.service_name}))
    provider.add_log_record_processor(processor)
    logger = provider.get_logger("ph.session")
    # Resolved once at mount rather than `SeverityNumber(...)` per record: enum
    # lookup by value goes through `Enum.__call__` and costs ~180 ns a record
    # for a table of four that cannot change while the row is mounted.
    severities = {name: SeverityNumber(value) for name, value in SEVERITY.items()}

    def ship(record: SessionTelemetryRecord) -> None:
        """One post-redaction record, as one OTel log record.

        No `try` here: `SessionTelemetry.record` already wraps every sink call,
        so a guard at this level could not change any outcome — it would only
        choose which of two log lines prints, while making the seam's own
        containment unreachable for the one sink that talks to a network. Sink
        health is the seam's to own, and it is already where it can see it.
        """
        logger.emit(
            timestamp=record.time * 1_000_000,
            body=record.body,
            severity_number=severities[record.severity],
            severity_text=record.severity,
            attributes=_flat({"channel": record.channel, **record.attributes}),
        )

    # Through `add_sink`, never `ctx.on("session-telemetry/record", …)`: the
    # sink list is fanned out *after* the waterfall settles, so this cannot see
    # a record the redactors have not finished with. A listener could.
    ctx.session_telemetry.add_sink(ship, scope=ctx)
    ctx.add_disposer(provider.shutdown, label="telemetry.otel")

    contribute(
        ctx,
        Diagnostic(
            id="telemetry-otel",
            title="Telemetry export",
            order=70,
            read=lambda: [
                ("sink", "opentelemetry"),
                ("service.name", config.service_name),
                ("endpoint", config.endpoint or "from OTEL_* environment"),
            ],
        ),
    )


def _pipeline(config: Config) -> Any:
    """The OTLP exporter behind a batching processor — the row's one test seam.

    Batching is right in production, where a record per turn must not be a
    request per turn, and wrong in a test, which has to observe the export
    before the batch would have flushed. One substitutable hook rather than two
    (an exporter factory *and* a processor factory) because both existed for
    this single substitution, and the pair had `apply` importing a class purely
    to hand it to a one-line helper.
    """
    from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    return BatchLogRecordProcessor(OTLPLogExporter(endpoint=config.endpoint))


def _flat(attributes: dict[str, Any]) -> dict[str, Any]:
    """OTel attributes are scalars; anything else goes as its JSON text.

    Flattened here rather than dropped, because a record's attributes are the
    part an operator actually queries on — and a nested value silently omitted
    is a field that looks absent rather than unsupported. The `ph.` prefix is
    applied here and only here, so the caller hands in `channel` rather than
    spelling the convention a second time.
    """
    return {
        f"ph.{key}": value if isinstance(value, SCALARS) else dumps(value)
        for key, value in attributes.items()
    }
