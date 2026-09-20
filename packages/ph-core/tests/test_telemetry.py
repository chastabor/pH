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

from collections.abc import Awaitable, Callable
from typing import Any

import anyio
import pytest

from ph.keys import SESSION_TELEMETRY
from ph.seams.telemetry import SessionTelemetryRecord
from ph.testing import MountProfile

pytestmark = pytest.mark.anyio


def _record(body: str, **attributes: object) -> SessionTelemetryRecord:
    return SessionTelemetryRecord(
        channel="ledger", time=1_000, severity="info", attributes=attributes, body=body
    )


async def test_a_sink_sees_only_what_redaction_left(mount: MountProfile) -> None:
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
    async def redact(record: SessionTelemetryRecord, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        return await next_(
            record.model_copy(update={"body": record.body.replace("hunter2", "«redacted»")})
        )

    ctx.on("session-telemetry/record", redact)
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)

    await ctx.require(SESSION_TELEMETRY).record(_record("the password is hunter2"))

    assert [record.body for record in seen] == ["the password is «redacted»"]


async def test_a_record_a_redactor_drops_reaches_no_sink(mount: MountProfile) -> None:
    """Dropping is stronger than rewriting, and must be just as absolute.

    A redactor returning `None` removes the record; a sink that still saw it
    would make every "this never leaves the machine" claim false.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []

    async def drop(record: SessionTelemetryRecord, next_: object) -> None:
        return None

    ctx.on("session-telemetry/record", drop)
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)

    await ctx.require(SESSION_TELEMETRY).record(_record("secret"))
    assert seen == []


async def test_a_failing_sink_does_not_take_the_others_with_it(mount: MountProfile) -> None:
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

    ctx.require(SESSION_TELEMETRY).add_sink(explode)
    ctx.require(SESSION_TELEMETRY).add_sink(lambda record: seen.append(record.body))

    await ctx.require(SESSION_TELEMETRY).record(_record("still recorded"))
    assert seen == ["still recorded"]


async def test_records_that_land_together_all_reach_the_sink(mount: MountProfile) -> None:
    """Two records in flight at once are two exports, not one (G4).

    The re-entrancy guard was a flag on the *service*, so a second record
    arriving while the first was still in a sink was dropped — and every record
    comes from its own task, so this is the ordinary case rather than a corner
    one: a turn's events are emitted a millisecond apart and the export was
    keeping whichever won. A telemetry seam that silently sheds load is worse
    than one that is switched off, because the gaps look like the thing being
    measured.

    The sink parks on an event so both fan-outs are provably overlapping: with
    the old flag the second record never arrives, and `started` never reaches
    two, so this deadlocks rather than fails — which is why it runs under a
    timeout.
    """
    ctx = await mount()
    started = anyio.Event()
    inside = 0
    seen: list[str] = []

    async def parked(record: SessionTelemetryRecord) -> None:
        nonlocal inside
        inside += 1
        if inside == 2:
            started.set()
        await started.wait()
        seen.append(record.body)

    telemetry = ctx.require(SESSION_TELEMETRY)
    telemetry.add_sink(parked)

    with anyio.fail_after(5):
        async with anyio.create_task_group() as group:
            group.start_soon(telemetry.record, _record("first"))
            group.start_soon(telemetry.record, _record("second"))

    assert sorted(seen) == ["first", "second"]


async def test_a_sink_that_records_is_still_refused(mount: MountProfile) -> None:
    """And the loop the guard exists for stays closed.

    The per-task flag is the same rule, not a weaker one: a sink recording from
    inside its own export is on the same call stack, so it is still dropped —
    and without that this test does not terminate.
    """
    ctx = await mount()
    telemetry = ctx.require(SESSION_TELEMETRY)
    seen: list[str] = []

    async def feeds_itself(record: SessionTelemetryRecord) -> None:
        seen.append(record.body)
        await telemetry.record(_record("about the failure"))

    telemetry.add_sink(feeds_itself)

    with anyio.fail_after(5):
        await telemetry.record(_record("the record"))

    assert seen == ["the record"]
