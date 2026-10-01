"""I6 in a live process — the check that makes the pollable invariants mean something.

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

**When it notices** (P12-04): once a root has settled with nothing written in its
mount for `verify_after`, and not again until something is. It used to poll every
live root every five minutes.
Socket-free through `supervised`, except the one test whose subject is a command
arriving over the wire.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import anyio
import pytest
from daemon_helpers import running, spawned, supervised, until

from ph.json import as_obj, as_seq, as_str
from ph.keys import SESSION_PERSISTENCE
from ph.seams.invariants import Violation
from ph.seams.subagents import STATUS
from ph.session import SurfaceIntent, session_written
from ph.testing import log_event, user_payload
from ph_app.daemon.recovery import PASS_FLOOR, PASSIVATED, VIOLATED
from ph_app.daemon.supervisor import Root, Supervisor

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
    log_event(session, "user/message", user_payload("hello", "m1"), SurfaceIntent("append"))
    assert session.derive_messages(), "nothing derived, so there is nothing to go stale"
    session._derived = ()


async def test_a_deployment_that_holds_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The baseline that stops every other test here from passing vacuously.

    A check that appended a record unconditionally would satisfy the assertions
    below without ever having run a check, and the failure would look like a pass.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("quiet")
        log_event(
            root.session, "user/message", user_payload("hello", "m1"), SurfaceIntent("append")
        )

        assert await supervisor.verify_root(root) == []
        assert VIOLATED not in [one.type for one in root.session.events]


async def test_a_stale_projection_is_recorded_in_the_root_it_concerns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check finds real drift, and the finding lands in the log.

    **In the log, not only in stderr**, which is the whole reason `VIOLATED`
    exists. This is a finding about bookkeeping the reporting process itself owns,
    so that process's own buffer is the one place it must not sit — and a person
    reading the transcript afterwards is the reader who can act on it.

    Recorded against the root it is about rather than fanned out the way
    `supervisor/unreachable` is: an unreachable socket is the daemon's condition
    and belongs to every root, while a drifted projection belongs to one session's
    caches and naming the others would be a false accusation.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        first = await supervisor.start("drifted")
        second = await supervisor.start("healthy")
        _drift(first)

        found = await supervisor.verify_root(first)

        assert await supervisor.verify_root(second) == [], "only the root that drifted"
        assert any("derive_messages holds 0" in one.detail for one in found), found

        recorded = [one for one in first.session.events if one.type == VIOLATED]
        assert len(recorded) == 1, "one record for one check"
        violation = as_obj(as_seq(recorded[0].data["violations"])[0])
        assert as_str(violation.get("invariant")), "the record names which promise broke"
        assert as_str(violation.get("detail")), "and how it looked when it was caught"

        assert VIOLATED not in [one.type for one in second.session.events], (
            "the healthy root is not accused of its neighbor's drift"
        )


async def test_the_record_is_flushed_rather_than_left_in_the_buffer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Durable before the process that wrote it goes away.

    `announce_unreachable`'s argument, one turn harder. Every other record here
    survives a crash because the log is written on the way out; this one is
    written precisely when the process's own bookkeeping is what is in doubt, and
    a finding that needs a clean shutdown to reach disk is a finding that is
    absent whenever it matters most.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("drifted")
        _drift(root)

        await supervisor.verify_root(root)

        stored = root.ctx.require(SESSION_PERSISTENCE).locate(root.id)
        assert stored is not None and stored.is_file(), "the log is on disk"
        assert VIOLATED in stored.read_text(encoding="utf-8"), "and the record is in it"


async def test_a_persistent_violation_is_recorded_once_rather_than_every_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record, not a sample — the difference between an alarm and a leak.

    A drifted cache does not repair itself, so a check that appended what it saw
    would write the same finding at every settle for the life of the root, in the
    log a person opens *because* something is wrong. The condition is reported
    when it starts, and again if it changes.

    Deliberately not a latch, which is what `unreachable_since` is: that
    transition is one-way by construction and this one is not — see the test
    below.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("drifted")
        _drift(root)

        for _ in range(3):
            assert await supervisor.verify_root(root), "still violating"

        recorded = [one for one in root.session.events if one.type == VIOLATED]
        assert len(recorded) == 1, f"three checks, one record; got {len(recorded)}"


async def test_a_busy_drifting_root_is_still_recorded_only_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The case the first version of this dedup got wrong.

    A violation's *detail* is the sentence a check builds, and `Session.stale()`
    builds it out of counts — "derive_messages holds 3 message(s) where a fresh
    derivation gives 4". Those numbers move every time the session grows. Keying
    the comparison on the detail therefore made a root that is **both drifting
    and still busy** differ from its own last record on every check, and re-append
    each time: exactly the flood the dedup exists to prevent, surviving in the one
    case anybody would care about.

    So the identity is the invariant id and the detail rides the record. This
    test is the difference: it grows the log between checks, which the
    persistent-violation test above deliberately does not.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("busy")
        _drift(root)

        details: list[str] = []
        for turn in range(3):
            details.extend(one.detail for one in await supervisor.verify_root(root))
            # The session grows, so the *next* check's detail says something new.
            payload = user_payload(f"more {turn}", f"m{turn + 2}")
            log_event(root.session, "user/message", payload, SurfaceIntent("append"))

        assert len(set(details)) > 1, "the detail really does change as the log grows"
        recorded = [one for one in root.session.events if one.type == VIOLATED]
        assert len(recorded) == 1, f"one condition, one record; got {len(recorded)}"


async def test_a_cleared_violation_is_recorded_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The clearing is news, and it is why this is not a latch.

    A transcript that says "violated" and then goes quiet leaves a reader unable
    to tell a repaired cache from a daemon that stopped looking — which is
    `supervisor/passivated`'s own argument about unexplained gaps. And a latch
    would miss the *next* drift, which is the one this check exists to catch.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("drifted")
        _drift(root)
        await supervisor.verify_root(root)

        # What a generation bump does: the memo rebuilds from node zero. This is
        # the repair path the invariant is written to allow for.
        root.session._derived_nodes = 0
        assert root.session.derive_messages(), "repaired"

        assert await supervisor.verify_root(root) == [], "it holds again"

        recorded = [one for one in root.session.events if one.type == VIOLATED]
        assert len(recorded) == 2, "one for the drift, one for the repair"
        cleared = as_seq(recorded[-1].data["violations"])
        assert cleared == (), "and the second says nothing is broken"

        # A third check adds nothing: holding is already what the log says.
        await supervisor.verify_root(root)
        assert len([one for one in root.session.events if one.type == VIOLATED]) == 2


@asynccontextmanager
async def _verifying(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[Supervisor]:
    """A supervisor whose verifier is running, due a root a fifth of a second after
    the root settles with nothing more written."""
    async with supervised(tmp_path, monkeypatch) as supervisor:
        supervisor.verify_after = 0.2
        stop = anyio.Event()
        supervisor.tasks.start_soon(supervisor.verifier.keep, stop)
        try:
            yield supervisor
        finally:
            stop.set()


def _count_checks(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """The roots `verify_root` is asked about from here on, still checked for real."""
    checked: list[str] = []
    verify = Supervisor.verify_root

    async def counting(self: Supervisor, target: Root) -> list[Violation]:
        checked.append(target.id)
        return await verify(self, target)

    monkeypatch.setattr(Supervisor, "verify_root", counting)
    return checked


async def test_a_drift_is_recorded_once_the_turn_that_wrote_settles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the turn's flushes are heard, and the root is checked once it is
    quiet (P12-04).

    The tests above call `verify_root` directly, which proves the method and proves
    nothing about the wiring — a check nothing ever called would pass all of them.

    Sabotage: drop the `session/durable` listener from `_start`, and nothing tells
    the verifier a root was written to.
    """
    async with _verifying(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("drifted")
        _drift(root)

        await supervisor.prompt("drifted", "hello")

        await until(
            lambda: VIOLATED in [one.type for one in root.session.events],
            what="the settled root to be checked",
        )


async def test_with_the_check_off_a_settled_turn_records_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`check_invariants=False` is off. A soak or a benchmark has to be able to
    silence an O(events) refold."""
    async with _verifying(tmp_path, monkeypatch) as supervisor:
        supervisor.check_invariants = False
        root = await supervisor.start("drifted")
        _drift(root)

        await supervisor.prompt("drifted", "hello")
        await until(
            lambda: root.status == "idle" and root.last_turn is not None,
            what="the turn to settle",
        )
        # Past the moment a check would have come, and the floor after a pass.
        await anyio.sleep(supervisor.verify_after + PASS_FLOOR + 0.3)

        assert VIOLATED not in [one.type for one in root.session.events]


async def test_a_root_is_not_checked_again_until_something_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only a writer can drift a cache, so a root nothing was written to since its
    last check is not checked: `session/durable` is what marks it.

    Sabotage: drop the `unverified_at` gate from `verify_settled`, and every call
    refolds.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("idle")
        checked = _count_checks(monkeypatch)

        log_event(root.session, "user/message", user_payload("hi", "m1"), SurfaceIntent("append"))
        await session_written(root.ctx, root.session)
        await supervisor.verify_settled(root)
        await supervisor.verify_settled(root)
        assert checked == ["idle"], "a second look with nothing written checked again"

        log_event(root.session, "user/message", user_payload("hi", "m2"), SurfaceIntent("append"))
        await session_written(root.ctx, root.session)
        await supervisor.verify_settled(root)
        assert checked == ["idle", "idle"], "a write since the last check went unchecked"


async def test_a_childs_write_makes_its_parent_due_a_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child owns its log, so its writes never move the parent's; the fold caches
    a check walks are the whole mount's. `session/durable` fires for the child's
    flush too, and the parent is checked once the child is done.

    Sabotage: mark only the root's own flushes in the `session/durable` listener,
    and the parent of a child that finished is never checked.
    """
    async with _verifying(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("parent")
        child = spawned(root, "c")
        await supervisor.verify_root(root)
        checked = _count_checks(monkeypatch)

        log_event(child, STATUS, {"status": "done"})
        await session_written(root.ctx, child)

        await until(lambda: "parent" in checked, what="the parent to be checked")


async def test_a_root_is_checked_before_it_is_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The last look at a root's caches, recorded in the log a resume will read.

    Sabotage: drop `verify_settled` from `_passivate`, and the drift leaves with
    the caches that held it.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("drifted")
        _drift(root)
        await session_written(root.ctx, root.session)

        assert await supervisor.sweep(after=0) == ["drifted"]

        types = [one.type for one in root.session.events]
        assert VIOLATED in types, "the drift was not recorded before the release"
        assert types.index(VIOLATED) < types.index(PASSIVATED), "and it came first"


async def test_a_command_on_an_idle_root_is_checked(tmp_path: Path) -> None:
    """A writer that ran outside any turn, heard the way every writer is.

    Over the wire, because the subject is the daemon as a whole: a permission preset
    is recorded on an idle root, no turn follows it, and the verifier `serve`
    started checks it once it has been quiet.

    Sabotage: stop starting `supervisor.verifier` in `serve`, and the drift goes
    unrecorded.
    """
    async with running(tmp_path, check_invariants=True) as daemon:
        daemon.running.supervisor.verify_after = 0.1
        root = await daemon.root("drifted")
        _drift(root)
        client = await daemon.client()

        await client.call("session/preset", sessionId=root.id, preset="workspace-write")

        await until(
            lambda: VIOLATED in [one.type for one in root.session.events],
            what="the command to be checked",
        )
