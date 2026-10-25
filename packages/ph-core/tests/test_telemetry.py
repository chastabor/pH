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

from ph.keys import AGENTS, SESSION_TELEMETRY, SESSIONS
from ph.persistence import open_session
from ph.seams.telemetry import SessionTelemetryRecord
from ph.session import Session, SessionEvent, SessionHeader, declare_log_type
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import (
    MountProfile,
    isolated_log_types,
    log_event,
    raising,
    stored_events,
)

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


# ------------------------------------------------------- behind the log --


def _records(seen: list[SessionTelemetryRecord], session_id: str) -> list[SessionTelemetryRecord]:
    """What the ledger said about one session."""
    return [
        record
        for record in seen
        if record.channel == "ledger" and record.attributes["session.id"] == session_id
    ]


def _ledger(seen: list[SessionTelemetryRecord], session_id: str) -> list[tuple[str, int]]:
    """What the ledger said about one session, as the `(type, seq)` it names."""
    return [(record.body, record.attributes["event.seq"]) for record in _records(seen, session_id)]


async def test_a_ledger_record_ships_once_the_log_holds_its_event(mount: MountProfile) -> None:
    """A record names its event by `(session, seq)`, so it must not name one the
    log can lose.

    `session/event` fires on the in-memory append. A crash takes whatever no flush
    wrote, and the resumed log appends at its own length, so a record shipped off
    the firehose named one event while the log put another at that seq. Here the
    unwritten tail is let go the way a crash lets it go, and the ledger has to
    agree with what a resume would be handed.

    Sabotage: ship on the append again, and the first assertion fails.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)
    sessions = ctx.require(SESSIONS)
    session = sessions.create("s")

    log_event(session, "turn/start", {"turn": 1})
    await anyio.wait_all_tasks_blocked()
    assert _ledger(seen, "s") == [], "exported before the log held it"

    await sessions.flush(session)
    lost = log_event(session, "turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    sessions.dispose("s")
    await anyio.wait_all_tasks_blocked()

    stored = stored_events(ctx, "s")
    assert "turn/start" in [event.type for event in stored]
    assert _ledger(seen, "s") == [(event.type, event.seq) for event in stored]
    assert lost.seq == len(stored), "the tail the store does not hold was not shipped"


async def test_a_record_carries_the_fields_its_type_exports(mount: MountProfile) -> None:
    """A record names its event and carries only what its type exports: why the work
    failed, was refused or stopped, which is what an operator looks for without the
    log at hand. ph-core's types and a package's declared ones ship the same way.

    Sabotage: drop `_exported` from `observe`, or hand a sink the payload's frozen
    view.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)
    sessions = ctx.require(SESSIONS)
    session = sessions.create("s")
    reason = {"kind": "aborted", "detail": "SIGTERM"}

    with isolated_log_types():
        declare_log_type(
            "sample/declined",
            owner="sample.plugin",
            ignorable=True,
            audit_only=True,
            exported=("reason", "code"),
        )
        log_event(session, "turn/start", {"turn": 1})
        log_event(session, "turn/end", {"turn": 1, "reason": reason})
        log_event(session, "sample/declined", {"reason": "busy", "via": "write"})
        await sessions.flush(session)
        await anyio.wait_all_tasks_blocked()

    shipped = {record.body: record.attributes for record in _records(seen, "s")}
    assert shipped["turn/end"]["event.reason"] == reason
    assert type(shipped["turn/end"]["event.reason"]) is dict, "a sink was handed a frozen view"
    assert "event.turn" not in shipped["turn/end"], "a field the type does not export shipped"
    assert not [key for key in shipped["turn/start"] if key.startswith("event.reason")]
    declined = shipped["sample/declined"]
    assert declined["event.reason"] == "busy"
    assert "event.code" not in declined, "a field the payload left out shipped as null"
    assert "event.via" not in declined


async def test_a_flush_that_failed_ships_nothing(mount: MountProfile) -> None:
    """Only a flush every backend finished says the log holds anything, so one that
    raised releases nothing, and the next that succeeds releases it all.

    Sabotage: emit `session/durable` whether or not the dispatch raised, and the
    first assertion fails.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)
    sessions = ctx.require(SESSIONS)
    session = sessions.create("s")

    refusing = ctx.on("session/flush", raising(OSError("no space left on device")))
    started = log_event(session, "turn/start", {"turn": 1})
    with pytest.raises(ExceptionGroup):
        await sessions.flush(session)
    await anyio.wait_all_tasks_blocked()
    assert _ledger(seen, "s") == []

    refusing()
    await sessions.flush(session)
    await anyio.wait_all_tasks_blocked()
    assert (started.type, started.seq) in _ledger(seen, "s")


async def test_one_copy_of_a_log_cannot_release_another(mount: MountProfile) -> None:
    """A resume in the same process makes a second copy of a log under the same id,
    and a flush of the first can still be finishing. What it wrote says nothing
    about the new copy's events, which begin wherever that copy's stored log ended.

    Sabotage: key the ledger's cursors by session id, and the new copy's event
    ships on the old copy's word.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)
    sessions = ctx.require(SESSIONS)
    old = sessions.create("s")
    sessions.dispose("s")
    new = sessions.adopt(Session("s", header=SessionHeader(id="s", created_at=1)))
    started = log_event(new, "turn/start", {"turn": 1})

    ctx.emit("session/durable", old, new.seq)
    await anyio.wait_all_tasks_blocked()
    assert _ledger(seen, "s") == []

    await sessions.flush(new)
    await anyio.wait_all_tasks_blocked()
    assert _ledger(seen, "s") == [(started.type, started.seq)]


async def test_the_shutdown_tail_ships_before_the_mount_lets_go(mount: MountProfile) -> None:
    """What the agents append as they let go ships in the same process. The log layer
    sits beneath `agent`, so the checkpoint policy's last barrier writes that tail
    through the store while this row is still mounted, and waits for the ledger to
    ship it before the rows beneath go. The sink takes its time, as the JSONL one
    does with a thread per record, so the fan-out is still running when the barrier
    returns from its flush.

    Sabotage: drop the barrier's `drain`, drop the barrier, or move the log layer
    back above `agent` in `ph-base`, and the tail never ships.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []

    async def slow(record: SessionTelemetryRecord) -> None:
        await anyio.sleep(0.01)
        seen.append(record)

    ctx.require(SESSION_TELEMETRY).add_sink(slow)
    session = await open_session(ctx, "s")
    agent = ctx.require(AGENTS).create(session, FAKE)
    # A canceled turn closing itself as its agent lets go.
    tail: list[SessionEvent] = []
    reason = {"kind": "aborted", "detail": "SIGTERM"}
    agent.ctx.add_disposer(
        lambda: tail.append(log_event(session, "turn/end", {"turn": 1, "reason": reason})),
        label="cancel",
    )
    log_event(session, "turn/start", {"turn": 1})
    await ctx.require(SESSIONS).flush(session)
    await ctx.dispose()

    (closing,) = tail
    (ended,) = [record for record in _records(seen, "s") if record.body == closing.type]
    assert ended.attributes["event.seq"] == closing.seq
    assert ended.attributes["event.reason"] == reason


async def test_a_resumed_logs_repair_ships(mount: MountProfile) -> None:
    """A resume closes a crashed turn with records its store does not hold yet. They
    are this process's to write, so they are the ledger's to ship, with the reason
    the repair gives. What the store held was the last run's to ship, so nothing
    before them ships again.

    Sabotage: start `_created` at the seed's end, and the closing `turn/end` never
    ships.
    """
    crashed = await mount()
    session = crashed.require(SESSIONS).create("s")
    started = log_event(session, "turn/start", {"turn": 1})
    await crashed.require(SESSIONS).flush(session)
    await crashed.dispose()  # the turn is never closed, as after a crash

    reopened = await mount()
    seen: list[SessionTelemetryRecord] = []
    reopened.require(SESSION_TELEMETRY).add_sink(seen.append)
    revived = await open_session(reopened, "s")
    await reopened.require(SESSIONS).flush(revived)
    await anyio.wait_all_tasks_blocked()

    shipped = _records(seen, "s")
    assert [record.body for record in shipped][:2] == ["turn/end", "session/resumed"]
    assert shipped[0].attributes["event.reason"]["kind"] == "interrupted"
    assert all(record.attributes["event.seq"] > started.seq for record in shipped)


async def test_only_the_first_chunk_of_a_step_ships(mount: MountProfile) -> None:
    """A token-by-token export is thousands of records saying the same thing, and the
    first chunk of a step is the one that carries the latency signal.

    Sabotage: let `_Cursor.wants` pass every chunk, and the second of each step ships.
    """
    ctx = await mount()
    seen: list[SessionTelemetryRecord] = []
    ctx.require(SESSION_TELEMETRY).add_sink(seen.append)
    sessions = ctx.require(SESSIONS)
    session = sessions.create("s")
    chunks = [
        log_event(session, "assistant/chunk", {"turn": 1, "step": step, "delta": "x"})
        for step in (1, 1, 2, 2)
    ]
    await sessions.flush(session)
    await anyio.wait_all_tasks_blocked()

    assert _ledger(seen, "s") == [(chunk.type, chunk.seq) for chunk in chunks[::2]]
