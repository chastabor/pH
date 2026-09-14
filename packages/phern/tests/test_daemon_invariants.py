"""I6 in a live process — the poll that makes the pollable invariants mean something.

`ph.seams.invariants` draws its central distinction between invariants enforced
*inline*, which refuse on the path they govern, and *pollable* ones, which carry
a `check` because a projection either equals its fold right now or it does not.
Every pollable check in the tree existed and **nothing called it**: the one caller
of `ctx.diagnostics.report()` is `phern doctor`, which mounts a fresh profile with no
session created, no view cached and no scope disposed. It reported that the checks
run, never that a deployment's live state passed them.

That is the wrong way round. The caches those checks exist to catch drifting — the
surface, the derivation, `ToolRuntime`'s views, six `SessionFoldCache`s — cannot
drift in a process that has just started. They drift in one that has been up for
hours, which is exactly the process `phern doctor` cannot be pointed at.

**The subject here is the wiring, not the checks.** Whether a stale derivation is
found is `ph-core`'s question and `test_invariants.py` answers it. What these ask
is whether a running daemon notices, and whether the noticing survives the process
that did it.
"""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from daemon_helpers import running, until

from ph.json import as_obj, as_seq, as_str
from ph.keys import SESSION_PERSISTENCE
from ph.session import SurfaceIntent
from ph.testing import user_payload
from ph_app.daemon.recovery import VIOLATED

pytestmark = pytest.mark.anyio


def _drift(root: object) -> None:
    """Empty the derivation memo while leaving its node count — I6's own sabotage.

    The same injection `test_a_stale_derivation_trips_the_session_invariant` uses,
    and it is what a forgotten invalidation looks like from the inside: the memo
    still believes it covers N nodes and holds none of them. Written through the
    private attribute deliberately — there is no public way to produce this state,
    which is the point of the invariant.
    """
    session = root.session  # type: ignore[attr-defined]
    session.append("user/message", user_payload("hello", "m1"), SurfaceIntent("append"))
    assert session.derive_messages(), "nothing derived, so there is nothing to go stale"
    session._derived = ()


async def test_a_deployment_that_holds_records_nothing(tmp_path: Path) -> None:
    """The baseline that stops every other test here from passing vacuously.

    A poll that appended a record unconditionally would satisfy the assertions
    below without ever having run a check, and the failure would look like a pass.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("quiet")
        root.session.append("user/message", user_payload("hello", "m1"), SurfaceIntent("append"))

        assert await daemon.running.supervisor.verify_invariants() == {}
        assert VIOLATED not in [one.type for one in root.session.events]


async def test_a_stale_projection_is_recorded_in_the_root_it_concerns(tmp_path: Path) -> None:
    """The poll finds real drift, and the finding lands in the log.

    **In the log, not only in stderr**, which is the whole reason `VIOLATED`
    exists. This is a finding about bookkeeping the reporting process itself owns,
    so that process's own buffer is the one place it must not sit — and a person
    reading the transcript afterwards is the reader who can act on it.

    Recorded against the root it is about rather than fanned out the way
    `supervisor/unreachable` is: an unreachable socket is the daemon's condition
    and belongs to every root, while a drifted projection belongs to one session's
    caches and naming the others would be a false accusation.
    """
    async with running(tmp_path) as daemon:
        first = await daemon.root("drifted")
        second = await daemon.root("healthy")
        _drift(first)

        found = await daemon.running.supervisor.verify_invariants()

        assert list(found) == ["drifted"], "only the root that drifted"
        assert any("derive_messages holds 0" in one.detail for one in found["drifted"]), found

        recorded = [one for one in first.session.events if one.type == VIOLATED]
        assert len(recorded) == 1, "one record for one poll"
        violation = as_obj(as_seq(recorded[0].data["violations"])[0])
        assert as_str(violation.get("invariant")), "the record names which promise broke"
        assert as_str(violation.get("detail")), "and how it looked when it was caught"

        assert VIOLATED not in [one.type for one in second.session.events], (
            "the healthy root is not accused of its neighbour's drift"
        )


async def test_the_record_is_flushed_rather_than_left_in_the_buffer(tmp_path: Path) -> None:
    """Durable before the process that wrote it goes away.

    `announce_unreachable`'s argument, one turn harder. Every other record here
    survives a crash because the log is written on the way out; this one is
    written precisely when the process's own bookkeeping is what is in doubt, and
    a finding that needs a clean shutdown to reach disk is a finding that is
    absent whenever it matters most.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("drifted")
        _drift(root)

        await daemon.running.supervisor.verify_invariants()

        stored = root.ctx.require(SESSION_PERSISTENCE).locate(root.id)
        assert stored is not None and stored.is_file(), "the log is on disk"
        assert VIOLATED in stored.read_text(encoding="utf-8"), "and the record is in it"


async def test_a_persistent_violation_is_recorded_once_rather_than_every_poll(
    tmp_path: Path,
) -> None:
    """A record, not a sample — the difference between an alarm and a leak.

    A drifted cache does not repair itself, so a poll that appended what it saw
    would write the same finding every five minutes for the life of the daemon:
    288 identical records a day, in the log a person opens *because* something is
    wrong. The condition is reported when it starts, and again if it changes.

    Deliberately not a latch, which is what `unreachable_since` is: that
    transition is one-way by construction and this one is not — see the test
    below.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("drifted")
        _drift(root)

        for _ in range(3):
            assert await daemon.running.supervisor.verify_invariants(), "still violating"

        recorded = [one for one in root.session.events if one.type == VIOLATED]
        assert len(recorded) == 1, f"three polls, one record; got {len(recorded)}"


async def test_a_busy_drifting_root_is_still_recorded_only_once(tmp_path: Path) -> None:
    """The case the first version of this dedup got wrong.

    A violation's *detail* is the sentence a check builds, and `Session.stale()`
    builds it out of counts — "derive_messages holds 3 message(s) where a fresh
    derivation gives 4". Those numbers move every time the session grows. Keying
    the comparison on the detail therefore made a root that is **both drifting
    and still busy** differ from its own last record on every poll, and re-append
    every five minutes: exactly the flood the dedup exists to prevent, surviving
    in the one case anybody would care about.

    So the identity is the invariant id and the detail rides the record. This
    test is the difference: it grows the log between polls, which the
    persistent-violation test above deliberately does not.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("busy")
        _drift(root)

        details: list[str] = []
        for turn in range(3):
            found = await daemon.running.supervisor.verify_invariants()
            details.extend(one.detail for one in found["busy"])
            # The session grows, so the *next* poll's detail says something new.
            payload = user_payload(f"more {turn}", f"m{turn + 2}")
            root.session.append("user/message", payload, SurfaceIntent("append"))

        assert len(set(details)) > 1, "the detail really does change as the log grows"
        recorded = [one for one in root.session.events if one.type == VIOLATED]
        assert len(recorded) == 1, f"one condition, one record; got {len(recorded)}"


async def test_a_cleared_violation_is_recorded_too(tmp_path: Path) -> None:
    """The clearing is news, and it is why this is not a latch.

    A transcript that says "violated" and then goes quiet leaves a reader unable
    to tell a repaired cache from a daemon that stopped looking — which is
    `supervisor/passivated`'s own argument about unexplained gaps. And a latch
    would miss the *next* drift, which is the one this cadence exists to catch.
    """
    async with running(tmp_path) as daemon:
        root = await daemon.root("drifted")
        _drift(root)
        await daemon.running.supervisor.verify_invariants()

        # What a generation bump does: the memo rebuilds from node zero. This is
        # the repair path the invariant is written to allow for.
        root.session._derived_nodes = 0
        assert root.session.derive_messages(), "repaired"

        assert await daemon.running.supervisor.verify_invariants() == {}, "it holds again"

        recorded = [one for one in root.session.events if one.type == VIOLATED]
        assert len(recorded) == 2, "one for the drift, one for the repair"
        cleared = as_seq(recorded[-1].data["violations"])
        assert cleared == (), "and the second says nothing is broken"

        # A third poll adds nothing: holding is already what the log says.
        await daemon.running.supervisor.verify_invariants()
        assert len([one for one in root.session.events if one.type == VIOLATED]) == 2


async def test_the_poll_runs_on_its_own_cadence(tmp_path: Path) -> None:
    """End to end: `serve` actually starts the timer, and it actually fires.

    The two tests above call `verify_invariants` directly, which proves the method
    and proves nothing about the wiring — a cadence that was never started would
    pass both. This is the one that fails if the `start_soon` is dropped.

    Its own `if invariants_every > 0` in `serve`, so a fast cadence here does not
    also speed up the scheduler or the sweep and pull their timers into the
    assertion.
    """
    async with running(tmp_path, invariants_every=0.05) as daemon:
        root = await daemon.root("drifted")
        _drift(root)

        await until(
            lambda: VIOLATED in [one.type for one in root.session.events],
            what="the invariant poll to notice a stale derivation",
        )


async def test_the_cadence_can_be_turned_off(tmp_path: Path) -> None:
    """`0` means off, matching the other four.

    A soak or a benchmark has to be able to silence an O(events) refold without
    also silencing the scheduler — which is why this cadence got its own `if`
    rather than riding on the sweep.
    """
    async with running(tmp_path, invariants_every=0) as daemon:
        root = await daemon.root("drifted")
        _drift(root)
        # Four times the cadence the test above needed to fire, so "nothing
        # happened" is a result rather than a race the assertion won.
        await anyio.sleep(0.2)

        assert VIOLATED not in [one.type for one in root.session.events]
