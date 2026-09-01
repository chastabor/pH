"""The `session-telemetry` seam — redaction, then fan-out, and nothing else.

The seam's gate is one sentence: **a redaction listener runs before any sink**.
It has been structurally true since P1-14 — `add_sink` fans out after the
`session-telemetry/record` waterfall settles — but nothing asserted it, because
no shipped profile mounts a sink (`base.yaml` carries the row with
`enabled: false`) and until P5-09 no row registered one. These tests are that
proof, and they live here rather than with the OTel row because the property is
the *seam's*: renaming or dropping an exporter must not take the seam's only
ordering test with it.

## What the media estimates replaced

A media block matched none of `measure`'s branches and contributed **zero**, so a
conversation of forty images reported no pressure at all — and G2/G3's character
thresholds, counted over text, never fired on it either.

That is why `MEDIA_TOKENS_UNKNOWN` is deliberately order-of-magnitude rather than
precise: being wrong by a factor is a rounding error against being wrong by
everything, and this estimate only ever answers "should we compact *before*
asking".
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.seams.telemetry import SessionTelemetryRecord

pytestmark = pytest.mark.anyio


def _record(body: str, **attributes: Any) -> SessionTelemetryRecord:
    return SessionTelemetryRecord(
        channel="ledger", time=1_000, severity="info", attributes=attributes, body=body
    )


async def test_a_sink_sees_only_what_redaction_left(mount: Any) -> None:
    """The gate.

    A redactor registered on the waterfall rewrites the record; the sink is
    registered through `add_sink` and therefore runs after the waterfall has
    settled. The secret must not reach it — and the point is that this holds
    *by construction*, not because the two were registered in a lucky order.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []

    # A waterfall listener takes the value and `next_`; rewriting means passing
    # a changed value onward rather than returning one.
    async def redact(record: SessionTelemetryRecord, next_: Any) -> Any:
        return await next_(
            record.model_copy(update={"body": record.body.replace("hunter2", "«redacted»")})
        )

    ctx.on("session-telemetry/record", redact)
    ctx.session_telemetry.add_sink(seen.append)

    await ctx.session_telemetry.record(_record("the password is hunter2"))

    assert [record.body for record in seen] == ["the password is «redacted»"]


async def test_a_record_a_redactor_drops_reaches_no_sink(mount: Any) -> None:
    """Dropping is stronger than rewriting, and must be just as absolute.

    A redactor returning `None` removes the record; a sink that still saw it
    would make every "this never leaves the machine" claim false.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []

    async def drop(record: SessionTelemetryRecord, next_: Any) -> Any:
        return None

    ctx.on("session-telemetry/record", drop)
    ctx.session_telemetry.add_sink(seen.append)

    await ctx.session_telemetry.record(_record("secret"))
    assert seen == []


async def test_a_failing_sink_does_not_take_the_others_with_it(mount: Any) -> None:
    """Sink containment belongs to the seam, so this is where it is asserted.

    Every sink talks to something that can be down — a collector, a disk — and
    telemetry that can break the thing it observes is worse than no telemetry.
    The exporters therefore carry no guard of their own; this is the one that
    holds, and it must fail here rather than in each of them.
    """
    ctx = await mount()
    seen: list[str] = []

    def explode(record: SessionTelemetryRecord) -> None:
        raise RuntimeError("the collector is down")

    ctx.session_telemetry.add_sink(explode)
    ctx.session_telemetry.add_sink(lambda record: seen.append(record.body))

    await ctx.session_telemetry.record(_record("still recorded"))
    assert seen == ["still recorded"]
