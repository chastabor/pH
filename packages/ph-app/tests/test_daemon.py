"""P5-01 — the supervisor, and the property every earlier mode lacked.

Every mode before this ties an agent's life to a connection: `ph -p` exits with
the turn, the TUI's root dies with the terminal, `--mode rpc` lives as long as
stdin. The gate here is the negation of that — *TUI close leaves the root
running* — so the tests that matter are the ones where a client goes away and
the work does not.

The socket is a real unix socket under `tmp_path` and the frames are real JSONL;
there is no in-process shortcut, because what this row delivers is precisely the
transport and a fake one would agree with whatever the code did.

## Measurements the supervisor's shapes are chosen for

Kept here rather than in `supervisor.py`, because each one is the reason a shape
that looks over-thought is not.

**One `anyio.Lock` in `start`, not one per root id.** `prompt` calls `start` on
*every turn*, so a per-id table minted a lock per call to throw it away — 832 B
and 0.65 µs each — and retained an entry per id ever started, cleared nowhere, in
the one process built to run for weeks. A single lock costs a serialized mount
(~2 ms) between two *different* new roots, which happens at most once per root,
while the returning-client path skips the lock's checkpoint entirely.

**The lease is acquired inline, not on a worker thread.** Wrapping it in
`to_thread.run_sync` measured **+340 µs on a 1.9 ms root start — twice the 166 µs
the acquire itself costs** — because a real start is seconds after the last one
and pays a cold thread plus a cold selector wakeup every time. At `timeout=0` the
acquire is one `os.open` and a non-blocking `flock`: it cannot wait, so there is
no blocking to move off the loop. Its neighbours settle it — `path.is_file()` two
lines down and `resume_session`'s whole-log read are both on the loop thread.

**`filelock(thread_local=False)` cost three tests to find**, two of them
P5-01's, all failing as "this session is already active" against a daemon that
had cleanly shut down. filelock keeps its re-entrancy counter in a thread-local,
so a lease taken on a worker thread and released from the event loop finds a
counter of zero and returns having released nothing — no error, no warning, and a
lock file held until the process dies.

**`Root.accepted` folds once and keeps the set.** The first draft re-scanned the
whole log per command, which measured **4.9 ms at 200 000 events** on the
daemon's own event loop, stalling every other connection for that long.

**The recovery fold must not be called from `Root.status`.** `describe()` reads
`status` for every root on every `sessions/list`, and folding there measured
**2.3 ms per root at 100 000 events, 12.5 ms at 500 000, and 128 ms for fifty
roots at once, with no await point between them.** It is folded once at root
start and maintained through `Root.retry`/`give_up`/`recovered`.

## Why the ladder does not retry a failed *turn*

The first draft of `recovery` retried `turn/end{error}`, and this row's own tests
are what showed it completing with **no request made at all**. The failed turn had
already *claimed* its message from the inbox, so the second `run()` found nothing
pending and ended at `phase.step == 0` with `kind="completed"` — a trivially
successful empty turn that clears the ladder and reports a healthy root which
answered nothing. Strictly worse than not retrying.

Re-splicing the claimed message instead was the other candidate: it appends a
second `user/message` and shows the model the same prompt twice.

## Why framing uses anyio's `receive_until`

The hand-rolled buffer was quadratic twice over — it re-scanned the whole buffer
per chunk and recopied the tail per frame — measuring **57 ms for 4 096 frames
arriving in one read**. It also bounded the *accumulated buffer* rather than one
frame, so many small frames in one chunk tripped a limit documented as per-frame.

## Why only success may clear the retry ladder

The first version of `recovered` reset the count on any `turn/end` — which the
retry itself *manufactures*: a re-entered `run()` finds an empty inbox and appends
`turn/start` + `turn/end{completed}` before the same crash happens again, so the
ladder cleared the counter that bounds it.

Measured against a persistently failing flush: **165 retries in two seconds, no
give-up, the fold pinned at one attempt and the root reporting "idle"** — the
unbounded retry the row exists to prevent, growing the log by three events an
iteration. A marker that only *success* writes cannot be forged by the failure.

## Why attach does not replay the gap

The first draft streamed the whole gap inside `attach`, one `session.event` frame
per event, straight into a 1024-slot outbox with no await point — so a client
reattaching to a root that had moved on by more than a thousand events got a
`WouldBlock` out of its own attach, *after* the subscription had already been made.
It failed at exactly **1 025**. The gate test passed only because its log was three
events long.

Catch-up now has one mechanism and it is the paged one, which also brings replay
under the 512 KiB-class bound it never had before.

## Three places the supervisor must not fold the log

**`Root.recovery` is held, not re-folded.** A whole-log scan per read is **4.9 ms
at 200 000 events**, and `status` is read for every root on every `sessions/list`.

**`relay` builds nothing before there is a subscriber.** It runs once per streamed
chunk, and rendering a payload for zero watchers measured **6.6 µs an event —
13 ms of a 2 000-chunk turn, all discarded**.

**The schedule tick flushes only when something was appended.** The condition also
read `or schedule.live(...)`, which folded the whole log a second time to decide to
flush a buffer the first fold had just left empty: **24 ms a root at 500 000
events, every five seconds**.

## Why `passivatable` checks quiet before it folds

The root reaching that line is idle and unwatched — the steady state the sweeper
exists for — so a fold above it runs on every sweep of the ninety-minute window and
is discarded eighty-nine times out of ninety. Measured over one idle window at
500 000 events across 50 roots: **60.9 s of event loop as written, 1.3 ms with the
quiet check first**, where `idle_for` itself costs **43 ns**.

The subagent fold goes through `SessionFoldCache` for the same reason: **0.09 µs
against 13.5 ms at 500 000 events**. Without it the root that returns `False` —
idle, unwatched, one unsettled child — re-folds every sixty seconds for the life of
the daemon: **16 minutes of event loop a day at 50 such roots**.

## Why each cadence is its own task

The heartbeat rode the ticker on a counter that only advanced when a tick
*succeeded*, so **a run of failing ticks starved the liveness record** as a side
effect of an unrelated failure. The socket watch has its own `if` for the mirror
reason: a test that turns the scheduler off to keep a timer out of its assertions
must not thereby turn off the thing that notices the daemon has no door.

**And why `_await` names `DaemonGone`.** It used to return an empty `{}` when woken
by the pump ending rather than by an answer, which every caller then read as a
successful reply with no fields in it.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import PROFILE, running

from ph.seams.schedule import Schedule
from ph.testing import stored_log
from ph_app.daemon import DaemonClient, recovery, serve, server
from ph_app.daemon import supervisor as supervisor_module
from ph_app.daemon.server import DaemonUnavailable
from ph_app.daemon.supervisor import Supervisor
from ph_app.protocol import DaemonError

pytestmark = pytest.mark.anyio

ReapedHost = Callable[..., Path]
"""The repo-root `reaped_host` fixture, spelled where it is read.

Structurally rather than by `from conftest import …`: that name resolves to the
*nearest* conftest on `sys.path`, which for this package is
`packages/ph-app/tests/conftest.py` and not the root one the fixture lives in.
"""


async def _history(
    client: DaemonClient, session_id: str, cursor: Any = None
) -> list[dict[str, Any]]:
    """Everything from `cursor` to now, paged the way a client must page it.

    Catch-up has one mechanism — `session/snapshot` — because `session/attach`
    deliberately does not replay: streaming a gap of unknown size into a bounded
    outbox is how a reattach fails at exactly the moment it matters.
    """
    collected: list[dict[str, Any]] = []
    page = await client.call("session/snapshot", sessionId=session_id, cursor=cursor)
    collected.extend(page["events"])
    while page["more"]:
        page = await client.call("session/snapshot", sessionId=session_id, cursor=page["cursor"])
        collected.extend(page["events"])
    return collected


async def _settled(client: DaemonClient, root_id: str, *, events: int) -> dict[str, Any]:
    """Poll the root until its log has grown and it is idle again."""
    with anyio.fail_after(10):
        while True:
            listed = await client.call("sessions/list")
            # A default rather than a bare `next()`: a root can legitimately
            # leave the listing mid-poll now that P5-05 releases idle ones, and
            # a `StopIteration` inside a coroutine surfaces as
            # `RuntimeError: coroutine raised StopIteration` — naming neither
            # the session nor the reason.
            row = next((one for one in listed["sessions"] if one["sessionId"] == root_id), None)
            assert row is not None, f'session "{root_id}" left the listing while settling'
            if row["status"] == "idle" and row["cursor"]["sequence"] >= events:
                return row
            await anyio.sleep(0.01)


# ------------------------------------------------------------------ the gate --


async def test_a_root_keeps_running_after_its_client_disconnects(tmp_path: Path) -> None:
    """P5-01's gate. The client is not the thing doing the work.

    A prompt is queued, the client *disconnects entirely* — the socket closes,
    which is what a closed terminal looks like from here — and a second client
    connects to find the same root, still there, with the turn it was given.
    """
    async with running(tmp_path) as daemon:
        first = await daemon.client()
        await first.call("session/new", sessionId="alpha")
        await first.call("session/prompt", sessionId="alpha", prompt="keep going")
        await first.aclose()

        second = await daemon.client()
        row = await _settled(second, "alpha", events=1)

        assert row["sessionId"] == "alpha"
        assert row["watchers"] == 0, "a disconnected client is still counted as watching"
        assert row["cursor"]["sequence"] > 0, "the root lost the turn its client queued"
        await second.notify("shutdown")


async def test_attaching_streams_events_and_detaching_stops_them(tmp_path: Path) -> None:
    """Attach and detach are symmetric, and neither touches the work.

    The root is prompted *after* the detach, so the second silence is evidence
    the subscription ended rather than that nothing happened.
    """
    async with running(tmp_path) as daemon:
        seen: list[str] = []
        client = await daemon.client(lambda method, params: seen.append(method))
        await client.call("session/new", sessionId="beta")
        await client.call("session/attach", sessionId="beta")
        await client.call("session/prompt", sessionId="beta", prompt="first")
        await _settled(client, "beta", events=1)
        assert "session.event" in seen, "attaching did not stream anything"

        await client.call("session/detach", sessionId="beta")
        seen.clear()
        await client.call("session/prompt", sessionId="beta", prompt="second")
        await _settled(client, "beta", events=2)

        assert seen == [], "a detached client was still being sent events"
        await client.notify("shutdown")


async def test_a_reattaching_client_reads_what_it_missed(tmp_path: Path) -> None:
    """The work happened while nobody watched, and the log is what proves it.

    Through `session/snapshot`, which is the only catch-up path: the root's
    events are the root's, not the connection's, and a client that was away
    reads them back at its own pace.
    """
    async with running(tmp_path) as daemon:
        starter = await daemon.client()
        await starter.call("session/new", sessionId="gamma")
        await starter.call("session/prompt", sessionId="gamma", prompt="unwatched")
        await _settled(starter, "gamma", events=1)
        await starter.aclose()

        watcher = await daemon.client()
        history = await _history(watcher, "gamma")

        assert history, "the session's history was not readable"
        assert [one["seq"] for one in history] == sorted(one["seq"] for one in history)
        await watcher.notify("shutdown")


# ------------------------------------------------------------ many and one --


async def test_two_roots_are_two_deployments(tmp_path: Path) -> None:
    """One task each, one mounted profile each. Two roots are not two agents in
    one deployment: separate sessions, separate seams, separate everything a row
    provides — sharing a `Context` would make one root's registration visible to
    a root that never asked for it."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="one", prompt="a")
        await client.call("session/prompt", sessionId="two", prompt="b")
        await _settled(client, "one", events=1)
        await _settled(client, "two", events=1)

        listed = await client.call("sessions/list")

        assert {row["sessionId"] for row in listed["sessions"]} == {"one", "two"}
        await client.notify("shutdown")


async def test_a_socket_the_kernel_will_not_bind_is_the_same_refusal(tmp_path: Path) -> None:
    """The other way a daemon cannot start, and it used to be a traceback.

    `AF_UNIX` paths are capped at 107 bytes, so a deep `$PH_RUNTIME` fails at
    `bind` — which happened *inside* `serve`'s task group, arrived wrapped in an
    `ExceptionGroup` no `except` clause could see, and reached the person as a
    full traceback. Binding is a precondition and now sits with the stale-socket
    check, ahead of the group, under the same named refusal.
    """
    deep = tmp_path.joinpath(*["directory"] * 16)
    deep.mkdir(parents=True)
    with pytest.raises(DaemonUnavailable, match="cannot listen on"):
        await serve(PROFILE, path=deep / "daemon.sock")


async def test_a_second_daemon_refuses_a_live_socket(tmp_path: Path) -> None:
    """Two supervisors both believing they own this user's roots is I-5's
    question, and taking the socket would answer it wrongly and silently. The
    refusal is this row's; the lease that arbitrates properly is P5-03."""
    async with running(tmp_path) as daemon:
        with pytest.raises(DaemonUnavailable, match="already listening") as refusal:
            await serve(PROFILE, path=daemon.path)
        # Named, because the CLI catches a type: `(RuntimeError, OSError)` is
        # two builtins wide enough to swallow a `typer.Exit`.
        assert refusal.value.code == "daemon_unavailable"

        client = await daemon.client()
        await client.notify("shutdown")


async def test_a_stale_socket_is_cleared_rather_than_inherited(tmp_path: Path) -> None:
    """The ordinary aftermath of a crash. A path nobody answers makes every
    client hang on a connect that is never completed, so it is removed — the
    opposite of the live case, and distinguishable only by trying it."""
    stale = tmp_path / "daemon.sock"
    stale.write_bytes(b"")
    assert stale.exists()

    async with running(tmp_path) as daemon:
        client = await daemon.client()

        assert (await client.call("initialize"))["capabilities"]["attach"] is True

        await client.notify("shutdown")


# ------------------------------------------------------------------- resume --


async def test_a_restarted_daemon_continues_the_log_rather_than_appending_to_it(
    tmp_path: Path,
) -> None:
    """The defect this row shipped for one commit.

    `sessions.create` mints a *fresh* session and the JSONL store appends, so a
    daemon restarted with the same root id concatenated a second session onto
    the first: one file, one header, and `seq` restarting at zero halfway
    through — which breaks A1 and makes every fold over that file double-count.

    Resuming is also what lets P4-14's reconciliation fire for a daemon root:
    `session/created` is published for an adopted session too, so a root that
    died holding a worktree gets it reclaimed on the way back up.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="delta", prompt="first")
        first = await _settled(client, "delta", events=1)
        await client.notify("shutdown")

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="delta")
        second = await _settled(client, "delta", events=first["cursor"]["sequence"])

        assert second["cursor"]["sequence"] > first["cursor"]["sequence"], (
            "the resumed root lost its history"
        )
        await client.notify("shutdown")


async def test_a_resumed_root_says_so_in_its_own_log(tmp_path: Path) -> None:
    """The durable half of the notice. stderr is for whoever is watching; a
    cron-started agent has nobody watching, and "this picked up somebody else's
    work" is a fact about provenance that belongs in the trace."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="epsilon", prompt="first")
        await _settled(client, "epsilon", events=1)
        await client.notify("shutdown")

    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="epsilon")

        history = await _history(client, "epsilon")

        assert any(one["type"] == "session/resumed" for one in history)
        await client.notify("shutdown")


# --------------------------------------------------------------- P5-02 gate --


async def test_a_cursor_resumes_reading_exactly_where_it_stopped(tmp_path: Path) -> None:
    """Half the gate: *reattach preserves streaming position*.

    The client stores the cursor from what it has read, work happens while it is
    away, and it comes back to be handed exactly the gap — not one event more,
    and not the whole log again. A count would have made the client guess; the
    cursor is what it can actually prove about what it has seen.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="zeta", prompt="first")
        first = await _settled(client, "zeta", events=1)
        cursor = first["cursor"]

        await client.call("session/prompt", sessionId="zeta", prompt="second")
        second = await _settled(client, "zeta", events=first["cursor"]["sequence"] + 1)

        missed = await _history(client, "zeta", cursor)

        assert missed, "the cursor read back nothing"
        assert missed[0]["seq"] == cursor["sequence"], "the gap did not start at the cursor"
        assert len(missed) == second["cursor"]["sequence"] - cursor["sequence"]
        await client.notify("shutdown")


async def test_a_cursor_from_another_log_reads_from_the_start(tmp_path: Path) -> None:
    """A sequence only means something against the log that counted it.

    Refusing would strand a client that did nothing wrong; honouring it would
    skip events it never saw. So a stale generation reads as "you have seen
    nothing of *this* log" — and `session/attach` says where it actually
    resumed, so the client is not left inferring it from sequence numbers.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="eta", prompt="first")
        settled = await _settled(client, "eta", events=1)
        stale = {"generation": "not-this-log", "sequence": settled["cursor"]["sequence"]}

        history = await _history(client, "eta", stale)
        attached = await client.call("session/attach", sessionId="eta", cursor=stale)

        assert len(history) == settled["cursor"]["sequence"], "a stale cursor skipped events"
        assert attached["from"] == 0, "attach did not say it was resuming from the start"
        await client.notify("shutdown")


async def test_the_same_command_twice_runs_once(tmp_path: Path) -> None:
    """The other half: *duplicate command is idempotent*.

    A reconnecting client cannot know whether its last `session/prompt` landed,
    so it sends it again. Answering "yes, that one" is what makes asking twice
    safe — and the record is in the log, so it survives the restart that caused
    the retry.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call(
            "session/prompt", sessionId="theta", prompt="once", clientId="c1", commandId="k1"
        )
        settled = await _settled(client, "theta", events=1)

        await client.call(
            "session/prompt", sessionId="theta", prompt="once", clientId="c1", commandId="k1"
        )
        await anyio.sleep(0.05)
        again = await _settled(client, "theta", events=settled["cursor"]["sequence"])

        assert again["cursor"]["sequence"] == settled["cursor"]["sequence"], (
            "the duplicate ran a second turn"
        )
        await client.notify("shutdown")


async def test_a_snapshot_is_paged_rather_than_one_huge_frame(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A resumed root can hold hundreds of thousands of events, and one reply
    carrying all of them would trip the transport's own frame bound.

    The page size is patched down because the first version of this test looped
    on `page["more"]` over a sixteen-event session — `more` was always `False`,
    the loop body never ran, and the bound it claimed to exercise was never
    reached by any test in the suite.
    """
    monkeypatch.setattr(server, "SNAPSHOT_EVENTS", 4)
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("iota", "first")
        settled = await _settled(client, "iota", events=5)

        pages = 0
        collected: list[dict[str, Any]] = []
        page = await client.call("session/snapshot", sessionId="iota")
        while True:
            pages += 1
            collected.extend(page["events"])
            if not page["more"]:
                break
            page = await client.call("session/snapshot", sessionId="iota", cursor=page["cursor"])

        assert pages > 1, "the page bound was never reached"
        assert all(len(one) <= 4 for one in [page["events"]]), "a page exceeded the bound"
        assert len(collected) == settled["cursor"]["sequence"], "paging lost or duplicated events"
        assert [one["seq"] for one in collected] == sorted(one["seq"] for one in collected)
        await client.notify("shutdown")


async def test_the_client_makes_its_own_retries_safe(tmp_path: Path) -> None:
    """Idempotence as a property of the protocol rather than of a caller's
    discipline: `DaemonClient.prompt` mints the ids, so the same call twice is
    two commands and a *replayed* call is one."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        first = await client.prompt("kappa", "once")
        settled = await _settled(client, "kappa", events=1)

        # The retry a reconnect forces: the same command id, sent again.
        await client.call(
            "session/prompt",
            sessionId="kappa",
            prompt="once",
            clientId=client.id,
            commandId="1",
        )
        await anyio.sleep(0.05)
        again = await _settled(client, "kappa", events=settled["cursor"]["sequence"])

        assert first["sessionId"] == "kappa"
        assert again["cursor"]["sequence"] == settled["cursor"]["sequence"], (
            "the replayed command ran a second turn"
        )
        await client.notify("shutdown")


# --- P5-03: leases ----------------------------------------------------------
#
# I-5 names the hazard as "two writers on one JSONL" and the remedy in two
# halves: an in-process lock per root, and a file lock on the canonical path
# against a *second daemon*. They answer different questions, and the tests
# below are paired to that split — inside one process, two clients naming one
# root should get that root; across processes, the second should be refused.


async def test_a_second_daemon_is_refused_the_same_session(tmp_path: Path) -> None:
    """The gate: concurrent open → `session_already_active`.

    Two supervisors over one `$PH_HOME`, which is what a person actually
    produces — a daemon they forgot was running, plus a fresh one — and what
    P5-01's `_clear_stale` explicitly deferred to this row. Without the lease
    both append to the same file.
    """
    async with (
        running(tmp_path, name="a") as first,
        running(tmp_path, name="b") as second,
    ):
        held = await first.client()
        await held.call("session/new", sessionId="shared")

        intruder = await second.client()
        with pytest.raises(DaemonError) as refusal:
            await intruder.call("session/new", sessionId="shared")

        # Named, not narrated: a client branches on this, and matching the
        # message text would be a contract that every rewording breaks.
        assert refusal.value.reason == "session_already_active"
        # And the refusal is per session, not per daemon — the second
        # supervisor is still a working supervisor.
        assert await intruder.call("session/new", sessionId="its-own")


async def test_the_lease_ends_with_the_daemon_that_held_it(tmp_path: Path) -> None:
    """A lease is held for a root's life, not a session file's.

    The failure this pins is the one that makes leases unusable in practice: a
    lock left behind by a daemon that exited cleanly, so the session can never
    be opened again and the fix is "delete a file you were never told about".
    """
    async with running(tmp_path, name="a") as first:
        client = await first.client()
        await client.call("session/new", sessionId="handover")
        await client.notify("shutdown")

    async with running(tmp_path, name="b") as second:
        client = await second.client()
        # Same id, same `$PH_HOME`, no refusal — and it resumes rather than
        # starting over, which is the P5-01 behaviour the lease must not break.
        assert (await client.call("session/new", sessionId="handover"))["sessionId"] == "handover"


async def test_two_clients_naming_one_new_root_share_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside one process the answer is "here it is", not "it is taken".

    `_start` checks `self.roots` and then awaits twice before it assigns, so two
    clients arriving together both pass the membership test and both build a
    root. The file lease would notice that pair — and would answer the wrong
    question, refusing a client whose only mistake was arriving at the same
    moment as another one asking for the same thing.

    **The interleaving is forced, not hoped for.** The first version of this
    test started two `session/new` calls concurrently and trusted the scheduler
    to overlap them. It did, for one commit — until the lease stopped hopping to
    a worker thread, which removed the suspension point that had been doing it,
    and the test went on passing with the lock deleted. So the first caller is
    now parked *inside* `_start`, past the membership check, and the second is
    released only once it is there: without the lock the second must build a
    second root, and there is no ordering left for luck to supply.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        original = Supervisor._session_for
        parked, release = anyio.Event(), anyio.Event()

        # On the class: `Supervisor` is a `slots=True` dataclass, so an instance
        # attribute cannot be shadowed.
        async def hold(self: Supervisor, *args: Any, **kwargs: Any) -> Any:
            if not parked.is_set():
                parked.set()
                await release.wait()
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(Supervisor, "_session_for", hold)
        roots: list[Any] = []

        async with anyio.create_task_group() as both:

            async def first() -> None:
                roots.append(await supervisor.start("contested"))

            async def second() -> None:
                await parked.wait()
                # The first caller is now past `roots` membership and inside
                # the awaits. A second `start` here is the exact race — and
                # releasing it *before* awaiting is safe, since `set()` cannot
                # yield: the first task does not resume until this one blocks,
                # which it does either on the lock or on its own mount.
                release.set()
                roots.append(await supervisor.start("contested"))

            both.start_soon(first)
            both.start_soon(second)

        assert len(roots) == 2
        assert roots[0] is roots[1], "the second caller built a second root on one log"
        assert list(supervisor.roots) == ["contested"]


async def test_a_refused_start_leaves_nothing_behind(tmp_path: Path) -> None:
    """A start that fails registers no root and strands no mount.

    `start` holds its `AsyncExitStack` by hand so a root can outlive the `async
    with` that made it — which means a failure partway through is the one path
    where `mounted`'s own `finally` does not run. The observable half is that
    the id is not listed and can be opened later; the mount is checked by
    disposing the supervisor, which would raise on a context it never took.
    """
    async with running(tmp_path, name="a") as first:
        await (await first.client()).call("session/new", sessionId="taken")

        async with running(tmp_path, name="b") as second:
            intruder = await second.client()
            with pytest.raises(DaemonError):
                await intruder.call("session/new", sessionId="taken")
            assert (await intruder.call("sessions/list"))["sessions"] == []


# --- P5-04: the retry ladder -------------------------------------------------
#
# The ladder answers the root's *task* crashing — a flush that cannot write, a
# disposed context, a bug — where the work is still in the inbox and running
# again is meaningful. A failed *turn* is deliberately not its business:
# `llm-retry` has already retried what a model failure makes sense to retry, and
# the failed turn claimed its message, so a second `run()` would produce an empty
# turn that reports false success.


@pytest.fixture
def short_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ladder spends 6.25 s, which is not a thing to put in a suite.

    Patched on the module rather than passed in, because `Recovery.total` and
    `Recovery.delay` read the global when asked — which is why they are
    properties and not values copied at import.
    """
    monkeypatch.setattr(recovery, "RETRY_DELAYS", (0.01, 0.01, 0.01))


def _crash(patch: pytest.MonkeyPatch, root: Any, times: int) -> None:
    """Make this root's task raise for its first `times` wakes.

    At `run()`'s own boundary, which is where a *task* crash actually appears:
    the driver contains everything inside a turn, so a failure injected further
    in (a flush during the turn, a model error) is caught there and arrives as
    `turn/end{error}` — not as a crash, and not this ladder's business.

    Raising before `run()` claims anything also preserves the property the
    ladder depends on: the work is still in the inbox, so running again is
    meaningful rather than an empty turn reporting false success.

    Patching and counting together, on the class — `Supervisor` and the driver
    are both `slots=True`, so an instance attribute cannot be shadowed, and
    every call site was writing `type(root.agent)` twice to say one thing.
    """
    driver, original, remaining = type(root.agent), type(root.agent).run, times

    async def run(self: Any) -> None:
        nonlocal remaining
        if remaining > 0:
            remaining -= 1
            raise RuntimeError("injected crash")
        await original(self)

    patch.setattr(driver, "run", run)


def _on_disk(path: Path, marker: str) -> bool:
    """Whether `marker` has reached this log yet, before the log exists.

    A session's file appears on its first flush, so a wait that starts earlier
    reads a path that is not there — `read_text` raises `FileNotFoundError`
    rather than answering "not yet", which turns a poll into a crash.
    """
    return path.exists() and marker in path.read_text()


async def _until(done: Callable[[], bool], *, what: str) -> None:
    """Poll until `done()`, or fail saying what was being waited for.

    `prompt` returns as soon as the message is *logged* — the turn has not
    started, let alone crashed — so a wait written as "while it still looks
    fine" exits on its first check and asserts against an empty log.
    """
    try:
        with anyio.fail_after(10):
            while not done():
                await anyio.sleep(0.01)
    except TimeoutError:
        # `fail_after` raises a bare `TimeoutError`, so without this the `what=`
        # every call site passes reached no message at all.
        pytest.fail(f"timed out waiting for {what}")


async def test_an_injected_crash_is_retried_and_the_root_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The gate's first half: an injected crash recovers.

    One crash, then the world works again. The root must come back and finish
    the turn — the message is still in its inbox, which is precisely why this
    failure is worth retrying and a failed turn is not.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="recovers")
        root = daemon.server.supervisor.roots["recovers"]
        _crash(monkeypatch, root, 1)

        await client.prompt("recovers", "hello")
        # Waiting on the *answer*, not on "idle": a root is idle throughout the
        # ladder's own backoff, so a wait for idle exits during the sleep and
        # asserts against a log the retry has not written to yet.
        await _until(
            lambda: any(e.type == "assistant/message" for e in root.session.events_from(0)),
            what="the retried task to finish its turn",
        )

        await _until(
            lambda: root.recovery.attempts == 0, what="the ladder to clear after recovering"
        )
        types = [event.type for event in root.session.events_from(0)]
        assert types.count(recovery.RETRY) == 1
        assert recovery.FAILED not in types, "a root that recovered was reported failed"
        # The marker that clears the count, and the only thing that may: a
        # ladder resettable by its own retry does not terminate.
        assert types.count(recovery.RECOVERED) == 1
        assert root.status == "idle"
        assert "assistant/message" in types, "the retried task never finished its turn"
        assert root.status == "idle"


async def test_the_ladder_gives_up_after_its_last_attempt_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The gate's second half: the third failure reports.

    A ladder that never gave up would be worse than none — a permanently broken
    root would retry forever while reporting itself busy. So the count is
    bounded, the give-up is *recorded*, and the status a client reads changes.
    """
    seen: list[dict[str, Any]] = []
    async with running(tmp_path) as daemon:
        client = await daemon.client(lambda method, params: seen.append(params))
        await client.call("session/new", sessionId="doomed")
        await client.call("session/attach", sessionId="doomed")
        root = daemon.server.supervisor.roots["doomed"]
        _crash(monkeypatch, root, 99)

        await client.prompt("doomed", "hello")
        await _until(lambda: root.status == "failed", what="the ladder to give up")

        types = [event.type for event in root.session.events_from(0)]
        assert types.count(recovery.RETRY) == len(recovery.RETRY_DELAYS)
        assert types.count(recovery.FAILED) == 1
        # Recorded, not merely announced: an unattended run leaves the fact in
        # its own trace whether or not a client was ever attached.
        given_up = next(e for e in root.session.events_from(0) if e.type == recovery.FAILED)
        assert given_up.data["attempts"] == len(recovery.RETRY_DELAYS)
        assert "injected crash" in given_up.data["reason"]
        # Waited for, not sampled: `status` flips when `give_up` appends, while
        # the notification is still crossing the outbox and the pump. Reading
        # `seen` at that instant caught the client mid-delivery — `retrying` had
        # landed and `failed` had not — about one run in four.
        await _until(
            lambda: "failed" in {str(params.get("status", "")) for params in seen},
            what="the failed status to reach the client",
        )

        # **On disk, with the daemon still running and no shutdown sent.** The
        # give-up is the record that matters most and it was the one
        # write-through missed; a clean shutdown flushes it either way, which is
        # why asserting it here rather than after teardown is what pins the
        # behaviour. Waited for rather than read once: `status` flips when the
        # event is *appended*, and the flush that carries it to disk is the next
        # await — so reading immediately raced the very ordering under test,
        # about one run in six. With the flush deleted this waits out its
        # timeout instead, which is still a failure.
        written = stored_log(tmp_path / "sessions", "doomed")
        await _until(lambda: _on_disk(written, recovery.FAILED), what="the give-up to reach disk")


async def test_a_root_resumed_mid_ladder_does_not_start_the_count_over(
    tmp_path: Path, short_ladder: None
) -> None:
    """Why the ladder's state is folded and not remembered.

    A supervisor keeping the attempt count in memory would come back from every
    crash with a fresh ladder, so a root failing for a permanent reason would
    retry forever — three attempts per daemon lifetime, with nothing in the log
    to show it had ever been tried.
    """
    # Its own patch context, deliberately. `monkeypatch.undo()` would revert
    # every patch on the shared fixture — including the autouse `_isolated_home`
    # — so the second daemon would resume from the developer's real `~/.ph`.
    # This test did precisely that and wrote a session there.
    with pytest.MonkeyPatch.context() as crash:
        async with running(tmp_path, name="first") as daemon:
            client = await daemon.client()
            await client.call("session/new", sessionId="stubborn")
            root = daemon.server.supervisor.roots["stubborn"]
            _crash(crash, root, 99)
            await client.prompt("stubborn", "hello")
            # Waited on *disk*, not on status: `status` flips when `give_up`
            # appends, and this test is about what the next daemon can read
            # back — so shutting down at the in-memory flip raced the flush that
            # makes the claim true, and the resumed root came back `retrying`.
            written = stored_log(tmp_path / "sessions", "stubborn")
            await _until(
                lambda: _on_disk(written, recovery.FAILED),
                what="the spent ladder to reach disk",
            )
            await client.notify("shutdown")

    async with running(tmp_path, name="second") as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="stubborn")
        root = daemon.server.supervisor.roots["stubborn"]
        # Read straight off the resumed log, nothing carried in memory between
        # the two daemons.
        assert root.status == "failed"
        state = recovery.recovery_of(root.session)
        assert state.attempts == len(recovery.RETRY_DELAYS)
        assert state.spent


async def test_one_root_crashing_does_not_take_the_daemon_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The failure a supervisor exists to prevent.

    A root's task runs in the supervisor's task group, so anything raising out
    of it cancels the group — every *other* root with it, plus the listener.
    This kills one root outright and asserts the blast radius is exactly that.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("bystander", "hello")
        await _settled(client, "bystander", events=1)

        await client.call("session/new", sessionId="casualty")
        casualty = daemon.server.supervisor.roots["casualty"]
        _crash(monkeypatch, casualty, 99)
        await client.prompt("casualty", "hello")
        await _until(lambda: casualty.status == "failed", what="the doomed root to give up")
        monkeypatch.undo()

        # The daemon still answers, the bystander is still there, and it works.
        listed = await client.call("sessions/list")
        assert "bystander" in {row["sessionId"] for row in listed["sessions"]}
        await client.prompt("bystander", "still here?")
        await _settled(client, "bystander", events=2)


async def test_the_tree_is_restored_from_the_latest_checkpoint_before_a_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry starts from a restore point, not from a half-mutated tree.

    A retry run against whatever the crashed attempt left behind would be a
    different attempt from the one that failed — the model would see edits from
    a run nobody kept — and a ladder that compounds its own damage is worse than
    no ladder. `workspace/checkpoint` is P4-09's record and already a fold, so
    the *latest* one is the tree to go back to.

    Driven directly rather than through a worktree-tier daemon: what this pins
    is the selection and the best-effort contract, and standing up a real git
    worktree would test P4-09's capture again instead.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        root = await supervisor.start("restores")
        root.session.append("workspace/checkpoint", {"agentId": root.agent.id, "tree": "older"})
        root.session.append("workspace/checkpoint", {"agentId": root.agent.id, "tree": "newest"})

        asked: list[str] = []

        async def fake_restore(ctx: Any, workspace: Any, tree: str) -> tuple[str, ...]:
            asked.append(tree)
            return ()

        monkeypatch.setattr(supervisor_module, "restore", fake_restore)
        monkeypatch.setattr(type(root.ctx.workspace), "of", lambda self, agent_id: object())

        assert await supervisor._restore(root) is True
        assert asked == ["newest"], "the retry went back to a stale restore point"

        # Best-effort: a restore that fails must not cost the retry, and must
        # not claim a rollback that did not happen.
        async def angry_restore(ctx: Any, workspace: Any, tree: str) -> tuple[str, ...]:
            raise RuntimeError("git said no")

        monkeypatch.setattr(supervisor_module, "restore", angry_restore)
        assert await supervisor._restore(root) is False


async def test_a_root_with_no_workspace_restores_nothing_and_says_so(
    tmp_path: Path,
) -> None:
    """The ordinary case: an advisory-tier root has no worktree to put back.

    `restored: false` in the record, rather than silence — a transcript that
    implied a rollback nobody performed would misread the attempt that follows.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        root = await supervisor.start("advisory")
        assert await supervisor._restore(root) is False


async def test_a_failing_flush_climbs_the_ladder_instead_of_retrying_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The shape that made the ladder unbounded, and the one nothing tested.

    Every other test here injects at `run()`'s entry, so no turn is ever
    completed. A *persistently failing flush* is different in the one way that
    matters: `run()` succeeds first and writes a `turn/end`. The first version
    of the fold reset the count on any `turn/end`, and the retry manufactures
    one — a re-entered `run()` finds an empty inbox and appends
    `turn/start` + `turn/end{completed}` before the same flush fails again — so
    the ladder cleared the bound that was supposed to stop it. Measured at **165
    retries in two seconds, no give-up, the fold pinned at one attempt, and the
    root reporting "idle"**, growing the log by three events an iteration.

    Only `supervisor/recovered` resets the count now, and nothing but success
    writes it.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="unflushable")
        root = daemon.server.supervisor.roots["unflushable"]

        async def broken(self: Any, session: Any) -> None:
            raise RuntimeError("flush is broken")

        monkeypatch.setattr(type(root.ctx.sessions), "flush", broken)
        await client.prompt("unflushable", "hello")
        await _until(lambda: root.status == "failed", what="the ladder to give up")

        types = [event.type for event in root.session.events_from(0)]
        assert types.count(recovery.RETRY) == len(recovery.RETRY_DELAYS), (
            "the ladder did not terminate — its own retry cleared the count"
        )
        assert types.count(recovery.FAILED) == 1
        assert types.count(recovery.RECOVERED) == 0


# --- P5-05: passivation ------------------------------------------------------
#
# A daemon built to run for weeks accumulates roots, and each one holds a
# mounted profile, a session, an agent and a workspace. Passivation releases the
# process-side half and keeps the session on disk. Rehydration is not a second
# mechanism: `start()` already resumes any root whose log exists (P5-01), which
# is why the round-trip is a property here rather than a feature.


async def test_an_idle_root_is_released_and_comes_back_with_its_history(
    tmp_path: Path,
) -> None:
    """The gate: round-trip.

    Released while nobody wants it, and the next message brings it back — with
    what it already said still in the log, because the log is where it lived the
    whole time.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        client = await daemon.client()
        await client.prompt("napper", "first")
        settled = await _settled(client, "napper", events=1)
        before = settled["cursor"]["sequence"]

        assert await supervisor.sweep(after=0) == ["napper"]
        assert "napper" not in supervisor.roots, "a passivated root is still mounted"

        # The ordinary path wakes it: no rehydrate call, no second mechanism.
        await client.prompt("napper", "second")
        after = await _settled(client, "napper", events=before + 1)
        assert after["cursor"]["sequence"] > before, "the rehydrated root lost its history"

        history = [event["type"] for event in await _history(client, "napper")]
        assert history.count("user/message") == 2, "the first turn did not survive the round-trip"
        assert recovery.PASSIVATED in history, "the pause left no record"


async def test_the_release_is_recorded_before_it_happens(tmp_path: Path) -> None:
    """A gap nobody explained reads as a crash.

    On disk, checked while the daemon is still running: the record is appended
    and then flushed as part of releasing, so a reader opening this log finds
    out why it stops rather than inferring a failure. The `session/resumed` on
    the way back says only that something resumed, not that nothing was wrong.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("recorded", "hello")
        await _settled(client, "recorded", events=1)
        await daemon.server.supervisor.sweep(after=0)

        assert _on_disk(stored_log(tmp_path / "sessions", "recorded"), recovery.PASSIVATED), (
            "the record did not reach disk with the release"
        )


async def test_a_root_somebody_is_watching_is_not_released(tmp_path: Path) -> None:
    """An attached client is a reason the root is still wanted.

    From the root's own subscriber set — the same fact that makes it receive
    events — so a client cannot be receiving a session that was released out
    from under it.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="watched")
        await client.call("session/attach", sessionId="watched")
        assert await daemon.server.supervisor.sweep(after=0) == []

        await client.call("session/detach", sessionId="watched")
        assert await daemon.server.supervisor.sweep(after=0) == ["watched"]


async def test_a_root_that_has_not_been_quiet_long_enough_is_not_released(
    tmp_path: Path,
) -> None:
    """The timeout is read from the log, not from a timer.

    `now` is the log's own last event here, so nothing has been quiet for any
    time at all — which is also what makes a root rehydrated from a three-day-old
    log immediately eligible, correctly.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        client = await daemon.client()
        await client.prompt("busy", "hello")
        await _settled(client, "busy", events=1)
        root = supervisor.roots["busy"]

        last = root.session.last_event
        assert last is not None
        assert await supervisor.sweep(after=60, now=int(last.time)) == []
        assert await supervisor.sweep(after=60, now=int(last.time) + 61_000) == ["busy"]


async def test_a_root_mid_ladder_is_not_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """`retrying` is not `idle`, and this is why `status` derives it.

    A root in P5-04's backoff is doing nothing between attempts. Releasing it
    there would passivate a root part-way up a ladder it had already recorded —
    and the condition that catches it is the one the cleanup pass added when it
    found `retrying` announced as a notification while `sessions/list` still
    said `idle`.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="climbing")
        root = supervisor.roots["climbing"]
        _crash(monkeypatch, root, 99)

        await client.prompt("climbing", "hello")
        await _until(lambda: root.recovery.attempts > 0, what="the ladder to start")
        assert root.status == "retrying"
        assert await supervisor.sweep(after=0) == [], "released a root mid-ladder"


async def test_a_passivated_session_may_be_opened_by_another_process(
    tmp_path: Path,
) -> None:
    """Releasing gives back the I-5 lease, which is the point rather than a leak.

    A root holds its session's lease for as long as it is mounted (P5-03). If
    passivation kept it, a released session would be one *no* process could
    open — unopenable by the daemon that let it go and refused to everyone else.
    """
    async with (
        running(tmp_path, name="a") as first,
        running(tmp_path, name="b") as second,
    ):
        held = await first.client()
        await held.call("session/new", sessionId="handed-over")

        other = await second.client()
        with pytest.raises(DaemonError) as refusal:
            await other.call("session/new", sessionId="handed-over")
        assert refusal.value.reason == "session_already_active"

        await first.server.supervisor.sweep(after=0)
        # Now it is nobody's, so the second daemon may have it.
        assert (await other.call("session/new", sessionId="handed-over"))["sessionId"] == (
            "handed-over"
        )


async def test_attaching_wakes_a_passivated_root(tmp_path: Path) -> None:
    """A client attaching to a released session gets it back, not an error.

    Through `start`, the same path `session/prompt` takes — so waking has one
    mechanism. Without this, a session released while its watcher was away came
    back as `no_such_session`, which reads as "gone" for something still on
    disk.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("awaited", "hello")
        await _settled(client, "awaited", events=1)
        await daemon.server.supervisor.sweep(after=0)
        assert "awaited" not in daemon.server.supervisor.roots

        attached = await client.call("session/attach", sessionId="awaited")
        assert attached["sessionId"] == "awaited"
        assert "awaited" in daemon.server.supervisor.roots


async def test_the_sweeper_actually_runs(tmp_path: Path) -> None:
    """The timer, not just the predicate.

    Everything above drives `sweep()` directly, which would pass just as well if
    nothing ever called it — the shape of dead code that looks tested.
    """
    async with running(tmp_path, passivate_after=0.0, sweep_every=0.02) as daemon:
        client = await daemon.client()
        # Attached first, and that is the test's own setup rather than an
        # accident: with `passivate_after=0` the sweeper is eligible to release
        # this root the instant it is idle and unwatched, which is *during* the
        # turn we are waiting on. Attaching is the same condition a real client
        # relies on to keep a session it is using.
        await client.call("session/new", sessionId="swept")
        await client.call("session/attach", sessionId="swept")
        await client.prompt("swept", "hello")
        await _settled(client, "swept", events=1)
        await client.call("session/detach", sessionId="swept")
        await _until(
            lambda: "swept" not in daemon.server.supervisor.roots,
            what="the sweeper to release an idle root on its own",
        )


async def test_a_root_with_a_live_child_is_not_released(tmp_path: Path) -> None:
    """A parent whose subagent is still working stays mounted.

    Releasing it would be worse than wasteful: the child's `subagent/status`
    events are appended to the *parent's* log, so a released parent gets
    rehydrated by its own child's bookkeeping — a root that puts itself back
    every time the sweeper lets it go.

    Folded from `subagent/*` (P3-13) rather than tracked beside it, and asked of
    the seam that owns the vocabulary — the first version of this test spelled
    the settled statuses itself as `{"completed", "failed", "cancelled",
    "deleted"}`, of which only one is a string any producer emits. The writer
    says `done` and `error`; deletion is a tombstone that leaves `status` alone.
    The effect was that a root which had ever run a child to completion could
    never be released, and the test passed only because it appended the same
    invented status the predicate was checking for.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor

        for label, settle in (
            ("finished", ("subagent/status", {"runId": "c", "status": "done"})),
            ("errored", ("subagent/status", {"runId": "c", "status": "error"})),
            ("cancelled", ("subagent/status", {"runId": "c", "status": "cancelled"})),
            ("revoked", ("subagent/deleted", {"runId": "c", "reason": "revoked"})),
        ):
            root = await supervisor.start(label)
            root.session.append("subagent/admitted", {"runId": "c"})
            assert await supervisor.sweep(after=0) == [], f"{label}: released with a live child"

            root.session.append(*settle)
            assert await supervisor.sweep(after=0) == [label], f"{label}: child never settled"

        # An unrecognised status keeps the parent alive rather than releasing one
        # whose child may still be running.
        root = await supervisor.start("unknown")
        root.session.append("subagent/admitted", {"runId": "c"})
        root.session.append("subagent/status", {"runId": "c", "status": "who-knows"})
        assert await supervisor.sweep(after=0) == [], "an unknown status released the parent"


def test_every_type_this_package_writes_is_in_the_vocabulary() -> None:
    """The writer's half of the deal `known_event_types` records.

    That module states it: *"a producer in another package … owes the same proof
    through its own bundle's tests"*. ph-core pays it and ph-rlm pays it; ph-app
    now appends six types and paid nothing, so a record written here and not
    declared in ph-core would produce a log this build writes and then refuses to
    seed — found by whoever resumes the session rather than by whoever added it.

    Against the constants rather than a source scan, because that is how this
    package appends: ph-rlm's regex over `append("…")` literals would match none
    of these.
    """
    from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES

    written = {
        supervisor_module.COMMAND_ACCEPTED,
        recovery.RETRY,
        recovery.FAILED,
        recovery.RECOVERED,
        recovery.PASSIVATED,
        recovery.UNREACHABLE,
    }
    undeclared = written - KNOWN_SESSION_EVENT_TYPES
    assert not undeclared, f"ph-app writes types ph-core would refuse at seed: {undeclared}"


# --- P5-06: the scheduler ----------------------------------------------------
#
# The seam is tested in ph-core against a log; these are the two things only the
# daemon can answer — that a due tick becomes a real turn, and that a root with
# work scheduled is not released by P5-05 while it waits for it.


async def test_a_due_schedule_starts_a_turn(tmp_path: Path) -> None:
    """The tick delivers through `prompt`, so a scheduled turn is an ordinary one.

    Same path a person's message takes, with a `schedule/tick` beside it saying
    why it started — rather than a second way into the loop that would have its
    own bugs and its own transcript shape.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="cron")
        root = supervisor.roots["cron"]

        root.ctx.schedule.create(
            root.session,
            Schedule(id="nightly", kind="interval", spec="60000", prompt="do the thing"),
        )
        # Relative to creation: a schedule is anchored where it was made, so a
        # clock starting at zero is decades before its own schedule exists.
        made = root.ctx.schedule.states(root.session)["nightly"].created_at
        assert await supervisor.tick(now=made + 1_000) == [], "fired before it was due"

        assert await supervisor.tick(now=made + 90_000) == ["nightly"]
        await _settled(client, "cron", events=1)

        types = [event.type for event in root.session.events_from(0)]
        assert types.count("schedule/tick") == 1
        assert "assistant/message" in types, "the scheduled turn never ran"


async def test_a_root_with_work_scheduled_is_not_released(tmp_path: Path) -> None:
    """P5-05's fourth condition, which that row left open for this one.

    A root that has said when it comes back is a root that is still wanted.
    Releasing it would drop the only thing that knows the appointment — and
    since passivation unwinds the `Context`, the schedule would stop being
    watched while still sitting in the log claiming it fires at nine.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        root = await supervisor.start("appointed")
        assert await supervisor.sweep(after=0) == ["appointed"], "an empty root should release"

        root = await supervisor.start("appointed")
        root.ctx.schedule.create(
            root.session, Schedule(id="s", kind="cron", spec="0 9 * * *", prompt="morning")
        )
        assert await supervisor.sweep(after=0) == [], "released a root with work scheduled"

        root.ctx.schedule.cancel(root.session, "s")
        assert await supervisor.sweep(after=0) == ["appointed"]


async def test_a_heartbeat_records_that_something_is_still_watching(tmp_path: Path) -> None:
    """Liveness for a schedule that fires monthly.

    Without it such a root's log ends with a line a month old, which reads
    exactly like a log nobody is writing any more. A record, not a keep-alive:
    what keeps the root mounted is the schedule itself.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        root = await supervisor.start("monthly")

        await supervisor.heartbeat(now=1_000)
        assert not [e for e in root.session.events_from(0) if e.type == "schedule/heartbeat"]

        root.ctx.schedule.create(
            root.session, Schedule(id="m", kind="cron", spec="0 0 1 * *", prompt="monthly")
        )
        await supervisor.heartbeat(now=2_000)
        beats = [e for e in root.session.events_from(0) if e.type == "schedule/heartbeat"]
        assert len(beats) == 1 and beats[0].data["live"] == 1


async def test_one_root_with_a_broken_schedule_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every root's schedules fire from one loop, so one of them must not end it."""
    async with running(tmp_path) as daemon:
        supervisor = daemon.server.supervisor
        broken = await supervisor.start("broken")
        working = await supervisor.start("working")
        working.ctx.schedule.create(
            working.session, Schedule(id="ok", kind="interval", spec="1000", prompt="hi")
        )

        original = type(broken.ctx.schedule).claim

        def claim(self: Any, session: Any, *, now: int) -> Any:
            if session.id == "broken":
                raise RuntimeError("this schedule is unreadable")
            return original(self, session, now=now)

        monkeypatch.setattr(type(broken.ctx.schedule), "claim", claim)
        made = working.ctx.schedule.states(working.session)["ok"].created_at
        assert await supervisor.tick(now=made + 10_000) == ["ok"]


# --- P5-11: lingering detection (I-6) ----------------------------------------
#
# Gate: *simulated session end → clear diagnostic, not a silent failure.*
#
# The simulation is exact rather than metaphorical. The `reaped_host` fixture in
# the repo-root conftest pins `$XDG_RUNTIME_DIR` at a directory these tests own
# and puts `$PH_RUNTIME` inside it; the socket is bound there, and "the login
# session ended" is that directory being removed — which is what logind does to
# `/run/user/$UID` for a user who is not lingering. What must not happen after
# that is *nothing*: the daemon keeps running with no door, and the only reader
# left is whoever opens a transcript afterwards.


def _notices(root: Any) -> list[Any]:
    return [event for event in root.session.events_from(0) if event.type == recovery.UNREACHABLE]


async def test_a_reaped_runtime_dir_reaches_every_root_as_a_record(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """The gate. A session that ends takes the socket; the log says so, and why.

    Every root, not the busy ones: what became unreachable is the daemon, and a
    transcript that stops without a word is the same puzzle either way. The
    record carries the advice because the reader who finds it is by then some
    distance from the terminal that could have warned them.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket, watch_every=0.05) as daemon:
        supervisor = daemon.server.supervisor
        first = await supervisor.start("alpha")
        second = await supervisor.start("beta")

        # Logout, as logind performs it: the whole directory, not just the file.
        shutil.rmtree(tmp_path / "xdg")
        with anyio.fail_after(10):
            while not (_notices(first) and _notices(second)):
                await anyio.sleep(0.01)

        said = _notices(first)[0].data
        assert said["reason"] == "removed"
        assert said["socket"] == str(socket)
        assert said["linger"] == "off"
        assert said["advice"] == "loginctl enable-linger someone"
        # Which daemon, and which incident. Without these, "were these two
        # sessions in the same outage" can only be answered by correlating clock
        # times across every log and hoping the payloads happen to be equal.
        assert said["pid"] == os.getpid()
        assert _notices(second)[0].data["since"] == said["since"]
        # Once, not once per watch pass: the transition is one-way, and a record
        # appended every thirty seconds forever would bury the log it explains.
        await anyio.sleep(0.2)
        assert len(_notices(first)) == 1


async def test_the_roots_keep_working_when_the_socket_goes(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """Detection, not shutdown — P5-01's inversion holds through this too.

    A root's task holds no reference to a connection, so losing the front door
    is not losing the work. Ending an hour of in-flight turns over a socket
    problem would be this row's own failure mode arriving from the other side.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        supervisor = daemon.server.supervisor
        root = await supervisor.start("working")
        shutil.rmtree(tmp_path / "xdg")
        assert await daemon.server.check_reachable() == "removed"

        await supervisor.prompt("working", "still there?")

        def answered() -> bool:
            return any(event.type == "assistant/message" for event in root.session.events_from(0))

        # Polled on the answer rather than on `status`, which reads `idle` in the
        # window between the prompt being queued and the root's task waking: a
        # wait on it would pass before the turn it is waiting for had started.
        with anyio.fail_after(10):
            while not answered():
                await anyio.sleep(0.01)


async def test_a_second_daemons_socket_is_not_mistaken_for_a_recovery(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """The half an existence check gets wrong, and the worse half.

    Log back in, run the `ph daemon` a client just recommended, and the path
    exists and answers again — while the first daemon still holds every lease
    the second one is about to be refused. That is I-5's hazard reached through
    a door P5-03 does not watch, so the identity is a `(dev, inode)` pair.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        await daemon.server.supervisor.start("held")
        assert await daemon.server.check_reachable() == "", "its own socket, unchanged"

        socket.unlink()
        socket.touch()  # logind remade the directory; somebody remade the socket
        assert await daemon.server.check_reachable() == "replaced"
        assert _notices(daemon.server.supervisor.roots["held"])[0].data["reason"] == "replaced"


async def test_the_record_is_on_disk_before_anyone_could_read_it(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """Flushed, which is not the usual bar here and is the point.

    Every other record survives a crash because the log is written on the way
    out. This one is written exactly when the way out has stopped being
    reliable: `ph agents shutdown` has no door to knock on, so the person's next
    move is often `kill`, and an unflushed record explains nothing to anyone.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        await daemon.server.supervisor.start("durable")
        shutil.rmtree(tmp_path / "xdg")
        await daemon.server.check_reachable()

        stored = stored_log(tmp_path / "sessions", "durable").read_text(encoding="utf-8")
        assert recovery.UNREACHABLE in stored


async def test_daemon_status_says_it_cannot_be_reached_and_what_would_fix_it(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """Reported from the running process, for the client that is already attached.

    A connection accepted before the path went away outlives it, so there is a
    reader for this — and after somebody restores the path by hand there are
    more. The lifetime rows are the daemon's own, asked of the socket it bound
    rather than of whatever `$PH_RUNTIME` derives now.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        healthy = daemon.server.status()
        assert healthy["unreachableSince"] is None
        # One encoding, not three. The reply used to carry `survivesLogout` and
        # `linger` beside the rendered rows; nothing but this assertion read
        # them, and a second spelling of one fact is one that can disagree.
        #
        # Selected by title rather than asserted as the whole list: the envelope
        # is the *daemon's* sections, and P5-12 added a second one the moment
        # after this row landed. A test that enumerates a list it does not own
        # fails for other rows' correct changes.
        lifetime_section = next(
            one for one in healthy["sections"] if one["title"] == "socket lifetime"
        )
        rows = {row["label"]: row["value"] for row in lifetime_section["rows"]}
        assert rows["survives logout"].startswith("no —"), "reaped host, no lingering"
        assert "off for someone" in rows["linger"]
        assert rows["enable it"] == "loginctl enable-linger someone"

        shutil.rmtree(tmp_path / "xdg")
        await daemon.server.check_reachable()
        assert isinstance(daemon.server.status()["unreachableSince"], int)


async def test_a_hand_built_server_with_no_bound_socket_watches_nothing(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """No identity means no moment to compare against, not "everything is gone".

    `serve` captures the pair immediately after the bind, which is the one
    instant at which "the socket at this path" and "the socket this daemon is
    listening on" are the same file by construction. A `DaemonServer` built
    without one — a test's, or a future caller's — has no transition to find,
    and reporting one would be inventing evidence.
    """
    reaped_host()
    async with anyio.create_task_group() as tasks:
        built = server.DaemonServer(
            supervisor=Supervisor(profile=PROFILE, tasks=tasks),
            stop=anyio.Event(),
            path=tmp_path / "nowhere.sock",
        )
        assert built.identity is None
        assert await built.check_reachable() == ""
        assert built.status()["unreachableSince"] is None
        tasks.cancel_scope.cancel()


# ------------------------------------------------------- P6-23: rehydration --


async def test_a_daemon_wakes_a_root_whose_schedule_came_due_while_it_was_down(
    tmp_path: Path,
) -> None:
    """**P6-23's gate, and the inversion of a non-guarantee.**

    P5-06 argues a schedule outlives the process holding it, and of the *log* it
    was always right — `schedule/created` is still there. What did not outlive the
    process was the thing that reads it: `tick` iterates `self.roots` and a boot
    has none, so the appointment survived and was never kept. The failure shape
    was silence — no error, no log, found by somebody noticing a run that did not
    happen — which `test_non_guarantees.py` asserted verbatim until this landed.

    Driven at a simulated `now` on both halves, because the point is the *window*:
    a real boot a second later has nothing due yet, which is the reason the old
    assertion went on passing after rehydration was wired in.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        root = await first.server.supervisor.start("appointed")
        root.ctx.schedule.create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.schedule.states(root.session)["s"].created_at
        await first.server.supervisor._flush(root)

    async with running(tmp_path, name="second") as second:
        supervisor = second.server.supervisor
        assert list(supervisor.roots) == [], "nothing is mounted until something is due"

        fired = await supervisor.wake_and_tick(now=made + 600_000)

        assert fired == ["s"], "the appointment was kept without a client asking"
        assert "appointed" in supervisor.roots, "and its root is mounted to keep it"


async def test_a_session_with_no_appointment_is_left_alone(tmp_path: Path) -> None:
    """**The half that makes the fix safe**, and the reason this is an index.

    `Supervisor.start` takes P5-03's lease, so "mount every stored session at
    boot" would claim every session on the machine and refuse the next `ph -p`
    over any of them with `session_already_active` — loud, immediate, and hitting
    sessions that have no schedule at all. Strictly worse than the silence it
    would be fixing. So only what the index names is woken.

    Both sessions are in one daemon on purpose: a test that only showed the plain
    one staying asleep would pass just as well against a daemon that wakes
    nothing, which is the state this row replaces.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        plain = await first.server.supervisor.start("no-appointment")
        await first.server.supervisor._flush(plain)
        root = await first.server.supervisor.start("appointed")
        root.ctx.schedule.create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.schedule.states(root.session)["s"].created_at
        await first.server.supervisor._flush(root)

    async with running(tmp_path, name="second") as second:
        supervisor = second.server.supervisor

        await supervisor.wake_and_tick(now=made + 600_000)

        assert "appointed" in supervisor.roots, "the mechanism is live"
        assert "no-appointment" not in supervisor.roots, "and it woke only what is due"

        # And the untouched session is still openable, which is what unleased means.
        reopened = await supervisor.start("no-appointment")
        assert reopened.id == "no-appointment"


async def test_catch_up_is_unbounded_by_default(tmp_path: Path) -> None:
    """**The scheduler's whole promise, and its whole scope.**

    pH keeps an appointment while `ph daemon` runs and picks up where it left off
    when it starts — however long it was down, coalesced by `claim` to one run per
    missed window. Bounding that by default would be a second policy on top of the
    one P5-06 already settled, and the OS already ships cron, anacron and systemd
    timers for anyone who wants a run to happen without a daemon at all.

    A year is well past any bound a default could reasonably have carried, which
    is what makes this an assertion about the *absence* of one.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        root = await first.server.supervisor.start("long-gone")
        root.ctx.schedule.create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.schedule.states(root.session)["s"].created_at
        await first.server.supervisor._flush(root)

    async with running(tmp_path, name="second") as second:
        supervisor = second.server.supervisor
        assert supervisor.wake_within is None, "the shipped default is to catch up"

        a_year_later = made + 365 * 24 * 60 * 60 * 1000

        assert await supervisor.wake_and_tick(now=a_year_later) == ["s"]


async def test_a_deployment_can_bound_how_stale_an_appointment_may_be(
    tmp_path: Path,
) -> None:
    """The knob, for a deployment that would rather not resurrect a session
    somebody abandoned — a shape the OS tools do not have, because a schedule here
    is attached to a *conversation* and not to a crontab entry.

    Off by default; this is what turning it on does. Lifting it again on the same
    daemon is what makes the test say something: it proves the appointment was due
    and wakeable all along, and that the *bound* declined it.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        root = await first.server.supervisor.start("abandoned")
        root.ctx.schedule.create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.schedule.states(root.session)["s"].created_at
        await first.server.supervisor._flush(root)

    async with running(tmp_path, name="second", wake_within=60.0) as second:
        supervisor = second.server.supervisor
        long_after = made + 600_000 + 60_000 + 1

        assert await supervisor.wake_and_tick(now=long_after) == []
        assert list(supervisor.roots) == [], "a stale appointment is not resurrected"

        supervisor.wake_within = None

        assert await supervisor.wake_and_tick(now=long_after) == ["s"]
        assert "abandoned" in supervisor.roots, "the bound declined it, not the mechanism"
