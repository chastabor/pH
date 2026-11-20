"""P5-01: the supervisor, and the property that every earlier mode lacked.

Every mode before this one ties the life of an agent to a connection. `phern -p`
exits when the turn ends. The root of the TUI dies with the terminal.
`--mode rpc` lives as long as stdin. The gate here is the opposite: *TUI close
leaves the root running*. The tests that matter are therefore the tests where a
client goes away and the work does not.

The socket is a real unix socket under `tmp_path`, and the frames are real
JSONL. There is no in-process shortcut. This row delivers the transport, and a
fake transport would agree with whatever the code did.

## The measurements behind the design of the supervisor

These are kept here and not in `supervisor.py`. Each one shows that a design
that looks too elaborate is needed.

**One `anyio.Lock` for every mount (`_starting`), not one lock for each root
id.** `prompt` calls `start` on *every turn*. A table of locks by id created a
lock for each call and then discarded it, at 832 B and 0.65 µs each. The table
also kept an entry for every id that was ever started and never removed one, in
the one process that is built to run for weeks. A single lock costs a serialized
mount (~2 ms) between two *different* new roots, which happens at most one time
for each root. The path of a returning client skips the checkpoint of the lock
completely.

**The lease is acquired inline, not on a worker thread.** With the acquire
wrapped in `to_thread.run_sync`, the measurement was **+340 µs on a 1.9 ms root
start, which is twice the 166 µs that the acquire itself costs**. A real start
comes seconds after the last one, so it pays for a cold thread and a cold
selector wakeup every time. At `timeout=0` the acquire is one `os.open` and a
non-blocking `flock`. It cannot wait, so there is no blocking to move off the
loop. When this was measured, the code beside the acquire settled the question:
`path.is_file()` two lines down and the whole-log read of `resume_session` both
ran on the loop thread. That read has since moved to a worker thread
(`ph.persistence.opening`).

**`filelock(thread_local=False)` took three failing tests to find.** Two of the
tests belong to P5-01. All three failed with "this session is already active"
against a daemon that had shut down cleanly. filelock keeps its re-entrancy
counter in a thread-local. A lease that is taken on a worker thread and released
from the event loop therefore finds a counter of zero, and returns without
releasing anything. There is no error and no warning, and the lock file is held
until the process dies.

**The commands that a root accepted are folded one time, and the set is kept.**
The first draft scanned the whole log again for each command. That measured
**4.9 ms at 200 000 events** on the event loop of the daemon itself, and it
stalled every other connection for that long. The set was `Root.accepted`. It is
now the `CLIENT_COMMAND` index of the journal, which is cached and extended, not
scanned.

**The recovery fold must not be called from `Root.status`.** `describe()` reads
`status` for every root on every `sessions/list`. A fold there measured **2.3 ms
per root at 100 000 events, 12.5 ms at 500 000, and 128 ms for fifty roots at
once, with no await point between them.** The state is folded one time when the
root starts, and `Root.retry`, `give_up` and `recovered` maintain it after that.

## Why the ladder does not retry a failed *turn*

The first draft of `recovery` retried `turn/end{error}`. The tests of this row
showed that the retry completed with **no request made at all**. The failed turn
had already *claimed* its message from the inbox. The second `run()` therefore
found nothing pending, and ended at `phase.step == 0` with `kind="completed"`.
That is an empty turn that succeeds trivially. It clears the ladder and reports
a healthy root that answered nothing, which is strictly worse than no retry.

The other candidate was to splice the claimed message again. That appends a
second `user/message` and shows the model the same prompt twice.

## Why framing uses `receive_until` from anyio

The hand-written buffer was quadratic in two ways. It scanned the whole buffer
again for each chunk, and it copied the tail again for each frame. It measured
**57 ms for 4 096 frames that arrive in one read**. It also limited the
*accumulated buffer* and not one frame. Many small frames in one chunk therefore
reached a limit that was documented as a limit for one frame.

## Why only success may clear the retry ladder

The first version of `recovered` reset the count on any `turn/end`, and the
retry itself *produces* a `turn/end`. A `run()` that is entered again finds an
empty inbox and appends `turn/start` + `turn/end{completed}` before the same
crash happens again. The ladder therefore cleared the counter that limits it.

Measured against a flush that fails persistently: **165 retries in two seconds,
no give-up, the fold stuck at one attempt and the root reporting "idle"**. That
is the unbounded retry that the row exists to prevent, and it grew the log by
three events for each iteration. Only *success* writes the marker, so the
failure cannot forge it.

## Why attach does not replay the gap

The first draft streamed the whole gap inside `attach`: one `session.event`
frame for each event, directly into a 1024-slot outbox with no await point. A
client that attached again to a root that had moved on by more than a thousand
events got a `WouldBlock` from its own attach, *after* the subscription had
already been made. It failed at exactly **1 025**. The gate test passed only
because its log was three events long.

Catch-up now has one mechanism, and it is the paged one. Replay therefore also
has the 512 KiB-class bound, which it never had before.

## Three places where the supervisor must not fold the log

**`Root.recovery` is held and not folded again.** A scan of the whole log for
each read is **4.9 ms at 200 000 events**, and `status` is read for every root
on every `sessions/list`.

**`relay` builds nothing before there is a subscriber.** It runs one time for
each streamed chunk. To render a payload for zero watchers measured **6.6 µs for
each event, which is 13 ms of a 2 000-chunk turn, all discarded**.

**The schedule tick flushes only when something was appended.** The condition
used to read `or schedule.live(...)` as well. That folded the whole log a second
time, to decide to flush a buffer that the first fold had just left empty:
**24 ms a root at 500 000 events, on every pass**.

## Why `passivatable` checks quiet before it folds

The root that reaches that line is idle and unwatched, which is the steady state
that the sweep exists for. When this was measured, a sweep ran every sixty
seconds. A fold above the quiet check then ran on every sweep of the
ninety-minute window, and its result was discarded eighty-nine times out of
ninety. Measured over one idle window at 500 000 events across 50 roots:
**60.9 s of event loop as written, 1.3 ms with the quiet check first**.
`idle_for` itself costs **43 ns**.

The subagent fold goes through `SessionFoldCache` for the same reason: **0.09 µs
against 13.5 ms at 500 000 events**. Without the cache, a root that returns
`False` (idle, unwatched, one unsettled child) folded again every sixty seconds
for the life of the daemon: **16 minutes of event loop a day at 50 such roots**.
Since P12-01 there is no sixty-second sweep. The question is asked when a
release is planned or a pass runs, and a root whose only hold is an unsettled
child is still asked on every plan.

## Why each cadence is its own task

A cadence that depends on the counter of another cadence advances only when the
other one *succeeds*. **A run of failing passes then starves an unrelated
task**, as a side effect of a failure that has nothing to do with it. The socket
watch has its own `if` for the reverse reason. A test that turns the scheduler
off, to keep a timer out of its assertions, must not also turn off the thing
that notices that the daemon can no longer be reached.

**And why `Peer.ask` names `DaemonGone`.** The wait for an answer used to return
an empty `{}` when the end of the pump woke it and no answer had arrived. Every
caller then read that as a successful reply with no fields in it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import anyio
import pytest
from daemon_helpers import (
    PROFILE,
    ask_a_person,
    break_the_provider,
    private_runtime,
    running,
    spawned,
    supervised,
    unplugged,
    until,
)
from rlm_fixtures import ModelGate

from ph import wall_clock
from ph.agent.inbox import InboxTarget
from ph.agent_loop.driver import ReactLoopAgent
from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Profile, ProfileDocument, load_profile_documents
from ph.json import JsonObject, JsonValue, as_str
from ph.keys import SCHEDULE, SESSIONS, WORKSPACE
from ph.lingering import socket_identity
from ph.orphans import JOURNAL_NAME
from ph.path_watch import EntryWatch
from ph.paths import resolve_roots
from ph.persistence import session_path
from ph.seams.models import ModelChoice
from ph.seams.schedule import Schedule
from ph.seams.schedule_index import INDEX_NAME, ScheduleIndex
from ph.seams.subagents import ADMITTED, DELETED, PARENT_TEARDOWN, STATUS, SubagentService
from ph.session import Session, SessionEvent, SessionHeader, now_ms, session_written
from ph.session.kinds import SESSION_HOLDER, WORKSPACE_RESTORE, credential_hold
from ph.testing import (
    ReapedHost,
    log_event,
    not_none,
    noted,
    noting,
    raising,
    searches_into,
    stored_log,
    stored_types,
)
from ph_app import runtime as runtime_module
from ph_app import verbs
from ph_app.attach import prompt_message
from ph_app.daemon import recovery, server
from ph_app.daemon import supervisor as supervisor_module
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.projections import family_of, family_rows
from ph_app.daemon.recovery import CHILD_RETRY_LIMIT, PASS_FLOOR
from ph_app.daemon.server import DaemonUnavailable, serve
from ph_app.daemon.supervisor import NotARoot, Root, RootStartAbandoned, Supervisor
from ph_app.payloads import SessionChildrenNotice, SessionEventNotice
from ph_app.protocol import DaemonError, NoParams
from ph_app.runtime import mounted

RESTORING, RESTORED = WORKSPACE_RESTORE.opened, WORKSPACE_RESTORE.settled


@dataclass(slots=True)
class _Checkpoints:
    """A tier that can put a tree back (`CheckpointingProvider`), and that does
    what the test says. It sits beneath the `restore` of the seam, where the
    record is now kept. A test that replaced that method would skip the thing
    that it tests."""

    put_back: Callable[[str], Awaitable[tuple[str, ...]]]
    tier: str = "worktree"

    async def capture(self, workspace: object) -> str | None:
        return None

    async def restore(self, workspace: object, token: str) -> tuple[str, ...]:
        return await self.put_back(token)


pytestmark = pytest.mark.anyio


async def _history(
    client: DaemonClient,
    session_id: str,
    cursor: object = None,
) -> list[dict[str, Any]]:
    """Everything from `cursor` to now, paged as a client must page it.

    Catch-up has one mechanism, `session/snapshot`, because `session/attach`
    deliberately does not replay. To stream a gap of unknown size into a bounded
    outbox makes a reattach fail at exactly the moment when it matters.
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
            # A default, and not a bare `next()`. P5-05 releases idle roots, so
            # a root can legitimately leave the listing during the poll. A
            # `StopIteration` inside a coroutine surfaces as
            # `RuntimeError: coroutine raised StopIteration`, which names
            # neither the session nor the reason.
            row = next((one for one in listed["sessions"] if one["sessionId"] == root_id), None)
            assert row is not None, f'session "{root_id}" left the listing while settling'
            if row["status"] == "idle" and row["cursor"]["sequence"] >= events:
                return dict(row)
            await anyio.sleep(0.01)


async def test_a_resumed_root_hands_the_child_ladder_its_own_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume_children` takes the limit, because the policy belongs to the host
    (P6-32).

    This is therefore the only place where the answer of the daemon is stated,
    and the only place where the wiring can be checked. The seam states no bound
    of its own, so nothing in `ph-core` would notice if the daemon passed a
    different number. The tests of the ladder itself supply their own number.
    """
    seen: list[int] = []
    original = SubagentService.resume_children

    async def spy(self: Any, parent: Any, *, retry_limit: int) -> Any:  # noqa: ANN401
        seen.append(retry_limit)
        return await original(self, parent, retry_limit=retry_limit)

    monkeypatch.setattr(SubagentService, "resume_children", spy)
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="wired")

    assert seen == [CHILD_RETRY_LIMIT], "the daemon's own ladder bound, not the seam's"


async def test_a_person_reaches_a_busy_root_at_its_next_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A child could already interrupt its parent in the middle of a turn. A
    person could not.

    `rlm-messaging` delivers by steer, so that "a running agent picks it up
    without finishing first". A typed line waited for the whole turn. During a
    long fan-out, the user therefore had strictly less reach than the children
    that the user had spawned. The test asserts on the inbox target and not on
    timing. The target is the fact, and timing would be a race.
    """

    targets: list[str] = []
    original = ReactLoopAgent.send

    def spy(self: Any, message: Any, target: InboxTarget, wakeup: bool) -> None:  # noqa: ANN401
        targets.append(target)
        original(self, message, target, wakeup)

    monkeypatch.setattr(ReactLoopAgent, "send", spy)
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        root = await supervisor.start("busy")

        await supervisor.prompt("busy", "the first thing")
        assert targets[-1] == "next-turn", "an idle root has no turn to join"

        driver = root.agent
        assert isinstance(driver, ReactLoopAgent)
        driver._phase.kind = "running"
        await supervisor.prompt("busy", "also this")
        assert targets[-1] == "next-step", "a person waited for the whole turn to end"

        # A schedule means something else. The contract of `tick` is that a
        # scheduled turn is an *ordinary* turn. It must therefore not join a
        # turn that it has nothing to do with, and it must not merge with
        # another schedule that is due in the same pass.
        await supervisor.prompt("busy", "the appointment", reach="next-turn")

    assert targets[-1] == "next-turn", "a scheduled turn joined a running one"


async def test_a_failed_turn_is_named_beside_an_idle_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What P5-04 deferred is closed here, and the supervisor adds no opinion of
    its own.

    `status` stays the status of the agent. It is `idle` after a turn that ended
    in error, because the ladder does not count turn failures. But the
    `turn/end` of the agent is reported beside it. A client that polls the list
    can then tell "answered" from "the last answer was an error", which
    `--until-idle` could not.
    """
    break_the_provider(monkeypatch)
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="sour")
        await client.call("session/prompt", sessionId="sour", prompt="hello")

        row = await _settled(client, "sour", events=3)

        assert row["status"] == "idle"
        assert row["lastTurn"] == "error"
        assert daemon.running.supervisor.roots["sour"].recovery.attempts == 0, (
            "a failed turn is not a crashed task; the ladder must not have moved"
        )


# ------------------------------------------------------------------ the gate --


async def test_a_root_keeps_running_after_its_client_disconnects(tmp_path: Path) -> None:
    """The gate of P5-01. The client is not what does the work.

    A prompt is queued, and the client then *disconnects completely*. The socket
    closes, which is how a closed terminal looks from here. A second client
    connects and finds the same root, still there, with the turn that it was
    given.
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

    The root is prompted *after* the detach. The second silence is therefore
    evidence that the subscription ended, and not that nothing happened.

    The check is filtered to the frames of the session itself, because
    `daemon.lifetime` is not one of them. It goes to every *connection* by
    design (P9-07), and a turn that starts and ends moves it twice. A detach
    ends a subscription to a root, not the connection. The filter asserts that
    distinction.
    """
    async with running(tmp_path) as daemon:
        seen: list[str] = []
        client = await daemon.client(on_notify=lambda method, params: seen.append(method))
        await client.call("session/new", sessionId="beta")
        await client.call("session/attach", sessionId="beta")
        await client.call("session/prompt", sessionId="beta", prompt="first")
        await _settled(client, "beta", events=1)
        assert "session.event" in seen, "attaching did not stream anything"

        await client.call("session/detach", sessionId="beta")
        seen.clear()
        await client.call("session/prompt", sessionId="beta", prompt="second")
        await _settled(client, "beta", events=2)

        assert [one for one in seen if one.startswith("session.")] == [], (
            "a detached client was still being sent events"
        )
        await client.notify("shutdown")


async def test_a_reattaching_client_reads_what_it_missed(tmp_path: Path) -> None:
    """The work happened while nobody watched, and the log proves it.

    The client reads through `session/snapshot`, which is the only catch-up
    path. The events of the root belong to the root and not to the connection,
    and a client that was away reads them back at its own pace.
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
    """Each root has one task and one mounted profile. Two roots are not two
    agents in one deployment. They have separate sessions, separate seams, and a
    separate copy of everything that a row provides. A shared `Context` would
    make the registration of one root visible to a root that never asked for
    it."""
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
    """The other way that a daemon cannot start. It used to be a traceback.

    `AF_UNIX` paths are limited to 107 bytes, so a deep `$PH_RUNTIME` fails at
    `bind`. The bind happened *inside* the task group of `serve`. The failure
    arrived wrapped in an `ExceptionGroup` that no `except` clause could see,
    and reached the user as a full traceback. Binding is a precondition. It is
    now beside the stale-socket check, before the group, under the same named
    refusal.
    """
    deep = tmp_path.joinpath(*["directory"] * 16)
    deep.mkdir(parents=True)
    with pytest.raises(DaemonUnavailable, match="cannot listen on"):
        await serve(PROFILE, path=deep / "daemon.sock")


async def test_a_second_daemon_refuses_a_live_socket(tmp_path: Path) -> None:
    """Two supervisors that both believe they own the roots of this user are the
    question of I-5. To take the socket would answer that question wrongly and
    silently. The refusal belongs to this row. The lease that decides the
    question properly is P5-03."""
    async with running(tmp_path) as daemon:
        with pytest.raises(DaemonUnavailable, match="already listening") as refusal:
            await serve(PROFILE, path=daemon.path)
        # The refusal is named, because the CLI catches a type.
        # `(RuntimeError, OSError)` is two builtins that are wide enough to
        # swallow a `typer.Exit`.
        assert refusal.value.code == "daemon_unavailable"

        client = await daemon.client()
        await client.notify("shutdown")


async def test_a_stale_socket_is_cleared_rather_than_inherited(tmp_path: Path) -> None:
    """The ordinary result of a crash. A path that nobody answers makes every
    client hang on a connect that never completes, so the path is removed. This
    is the opposite of the live case, and the only way to tell the two apart is
    to try the connect."""
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
    """The defect that this row shipped for one commit.

    `sessions.create` mints a *fresh* session, and the JSONL store appends. A
    daemon that was restarted with the same root id therefore concatenated a
    second session onto the first: one file, one header, and `seq` restarting at
    zero halfway through. That breaks A1 and makes every fold over that file
    count twice.

    The resume is also what lets the reconciliation of P4-14 run for a daemon
    root. A resumed log awaits `session/loaded` before it is handed out, so a
    root that died while it held a worktree gets the worktree reclaimed on the
    way back up.
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


async def test_a_resumed_root_says_so_in_its_own_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The durable half of the notice. stderr is for whoever watches. An agent
    that cron started has nobody watching, and "this picked up somebody else's
    work" is a fact about provenance that belongs in the trace.

    The test also checks how the root was found. The start of a root is read
    from disk before its mount. That read is a search of the store, and for a
    fork a read of the whole lineage. It used to run on the event loop that
    every other root shares, and the resume then searched for the same log
    again. Now the read runs on a worker thread, and the open reads the log
    where the read found it.

    Sabotage: read the start on the loop in `read_start`, or drop the family
    from `_session_for`'s open, and this fails.
    """
    from ph.persistence import jsonl
    from ph_app import sessions

    async with running(tmp_path, name="first") as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="epsilon", prompt="first")
        await _settled(client, "epsilon", events=1)
        await client.notify("shutdown")

    loop = threading.get_ident()
    read_on: list[int] = []
    searched: list[str] = []
    real_start = sessions.recorded_start
    monkeypatch.setattr(
        runtime_module,
        "recorded_start",
        lambda directory, root: noted(read_on, threading.get_ident(), real_start(directory, root)),
    )
    monkeypatch.setattr(jsonl, "locate_under", searches_into(searched))

    async with running(tmp_path, name="second") as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="epsilon")

        history = await _history(client, "epsilon")

        assert any(one["type"] == "session/resumed" for one in history)
        await client.notify("shutdown")

    assert read_on and loop not in read_on, "the start was read on the event loop"
    assert searched.count("epsilon") == 1, searched


# --------------------------------------------------------------- P5-02 gate --


async def test_a_cursor_resumes_reading_exactly_where_it_stopped(tmp_path: Path) -> None:
    """Half of the gate: *reattach preserves streaming position*.

    The client stores the cursor from what it has read, and work happens while
    it is away. When it comes back, it is given exactly the gap: not one event
    more, and not the whole log again. A count would have made the client guess.
    The cursor is what the client can prove about what it has seen.
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
    """A sequence has a meaning only against the log that counted it.

    A refusal would strand a client that did nothing wrong. To honor the cursor
    would skip events that the client never saw. A stale generation is therefore
    read as "you have seen nothing of *this* log". The reply names the
    generation that it *does* count against, and a client pages from that and
    not from the cursor that it sent.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="eta", prompt="first")
        settled = await _settled(client, "eta", events=1)
        stale = {"generation": "not-this-log", "sequence": settled["cursor"]["sequence"]}

        history = await _history(client, "eta", stale)
        # Attach takes no cursor, because it does not replay. Catch-up is the
        # paged read above, from the point that this reply names.
        attached = await client.call("session/attach", sessionId="eta")

        assert len(history) == settled["cursor"]["sequence"], "a stale cursor skipped events"
        assert attached["cursor"]["generation"] != stale["generation"], "the reply names this log"
        await client.notify("shutdown")


async def test_the_same_command_twice_runs_once(tmp_path: Path) -> None:
    """The other half: *duplicate command is idempotent*.

    A client that reconnects cannot know whether its last `session/prompt`
    arrived, so it sends the prompt again. The answer "yes, that one" makes it
    safe to ask twice. The record is in the log, so it survives the restart
    that caused the retry.
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
    """A resumed root can hold hundreds of thousands of events. One reply that
    carried all of them would exceed the frame bound of the transport itself.

    The page size is patched down for a reason. The first version of this test
    looped on `page["more"]` over a session of sixteen events. `more` was always
    `False`, the loop body never ran, and no test in the suite reached the bound
    that this test claimed to exercise.
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
            assert len(page["events"]) <= 4, "a page exceeded the bound"
            collected.extend(page["events"])
            if not page["more"]:
                break
            page = await client.call("session/snapshot", sessionId="iota", cursor=page["cursor"])

        assert pages > 1, "the page bound was never reached"
        assert len(collected) == settled["cursor"]["sequence"], "paging lost or duplicated events"
        assert [one["seq"] for one in collected] == sorted(one["seq"] for one in collected)
        await client.notify("shutdown")


async def test_the_client_makes_its_own_retries_safe(tmp_path: Path) -> None:
    """Idempotence is a property of the protocol, and does not depend on the
    discipline of a caller. `DaemonClient.prompt` mints the ids, so the same
    call made twice is two commands, and a *replayed* call is one."""
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        first = await client.prompt("kappa", "once")
        settled = await _settled(client, "kappa", events=1)

        # The retry that a reconnect forces: the same command id, sent again.
        await client.call(
            "session/prompt",
            sessionId="kappa",
            prompt="once",
            clientId=client.id,
            commandId="1",
        )
        await anyio.sleep(0.05)
        again = await _settled(client, "kappa", events=settled["cursor"]["sequence"])

        assert first.session_id == "kappa"
        assert again["cursor"]["sequence"] == settled["cursor"]["sequence"], (
            "the replayed command ran a second turn"
        )
        await client.notify("shutdown")


# --- P5-03: leases ----------------------------------------------------------
#
# I-5 names the hazard as "two writers on one JSONL", and gives the remedy in
# two halves: an in-process lock for each root, and a file lock on the canonical
# path against a *second daemon*. The halves answer different questions, and the
# tests below are paired to match. Inside one process, two clients that name one
# root should get that root. Across processes, the second should be refused.


async def test_a_second_daemon_is_refused_the_same_session(tmp_path: Path) -> None:
    """The gate: concurrent open → `session_already_active`.

    The test runs two supervisors over one `$PH_HOME`. That is what a user
    really produces: a daemon that the user forgot was running, plus a fresh
    one. It is also what `_clear_stale` of P5-01 explicitly deferred to this
    row. Without the lease, both append to the same file.
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

        # The assertion is on the name and not on the message. A client
        # branches on this name, and a match on the message text would be a
        # contract that every rewording breaks.
        assert refusal.value.reason == "session_already_active"
        # The refusal is for one session, not for the daemon. The second
        # supervisor is still a working supervisor.
        assert await intruder.call("session/new", sessionId="its-own")


async def test_the_lease_ends_with_the_daemon_that_held_it(tmp_path: Path) -> None:
    """A lease is held for the life of a root, not for the life of a session
    file.

    This test guards against the failure that makes leases unusable in
    practice: a daemon that exited cleanly leaves a lock behind. The session
    can then never be opened again, and the fix is "delete a file you were
    never told about".
    """
    async with running(tmp_path, name="a") as first:
        client = await first.client()
        await client.call("session/new", sessionId="handover")
        await client.notify("shutdown")

    async with running(tmp_path, name="b") as second:
        client = await second.client()
        # The same id and the same `$PH_HOME`, and no refusal. The session
        # resumes and does not start over, which is the P5-01 behavior that
        # the lease must not break.
        assert (await client.call("session/new", sessionId="handover"))["sessionId"] == "handover"


async def test_two_clients_naming_one_new_root_share_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Inside one process the answer is "here it is", not "it is taken".

    `_start` checks `self.roots` and then awaits before a root is assigned. Two
    clients that arrive together would therefore both pass the membership test,
    and both build a root. The file lease would notice that pair, and would
    answer the wrong question. It would refuse a client whose only mistake was
    to arrive at the same moment as another client that asked for the same
    thing.

    **The interleaving is forced, not hoped for.** The first version of this
    test started two `session/new` calls concurrently and trusted the scheduler
    to overlap them. The scheduler did, for one commit. Then the lease stopped
    hopping to a worker thread, which removed the suspension point that had
    caused the overlap, and the test continued to pass with the ordering
    deleted. The first caller is therefore now held *inside* `_start`, past the
    membership check, and the second is released only when the first is there.
    Without the ordering, the second caller must build a second root, and no
    timing is left for luck to supply.

    `_mounting` does the ordering, and not a lock that a caller holds. `start`
    gives the build to the task group of the supervisor and waits on the
    record, so the second caller finds the entry of the first. `builds` is the
    part of the test that shows this. The other assertions below would also
    pass if the second caller built its own root over the same log.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        original = Supervisor._session_for
        build = Supervisor._start
        builds: list[str] = []
        parked, release = anyio.Event(), anyio.Event()

        async def counted(
            self: Supervisor,
            root_id: str,
            *,
            cwd: str | None,
            choice: ModelChoice,
            profile: str,
            asks: bool,
        ) -> Root:
            builds.append(root_id)
            return await build(self, root_id, cwd=cwd, choice=choice, profile=profile, asks=asks)

        # The patch is on the class. `Supervisor` is a `slots=True` dataclass,
        # so an instance attribute cannot be shadowed.
        async def hold(self: Supervisor, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            if not parked.is_set():
                parked.set()
                await release.wait()
            return await original(self, *args, **kwargs)

        monkeypatch.setattr(Supervisor, "_session_for", hold)
        monkeypatch.setattr(Supervisor, "_start", counted)
        roots: list[Any] = []

        async with anyio.create_task_group() as both:

            async def first() -> None:
                roots.append(await supervisor.start("contested"))

            async def second() -> None:
                await parked.wait()
                # The first caller is now past the `roots` membership check and
                # inside the awaits. A second `start` here is the exact race.
                # To release the first caller *before* this await is safe,
                # because `set()` cannot yield. The first task does not resume
                # until this one blocks, and this one blocks either on the lock
                # or on its own mount.
                release.set()
                roots.append(await supervisor.start("contested"))

            both.start_soon(first)
            both.start_soon(second)

        assert len(roots) == 2
        assert roots[0] is roots[1], "the second caller built a second root on one log"
        assert list(supervisor.roots) == ["contested"]
        assert builds == ["contested"], "two callers, one mount"


async def test_a_refused_start_leaves_nothing_behind(tmp_path: Path) -> None:
    """A start that fails registers no root and leaves no mount behind.

    `_start` holds the mount on an `AsyncExitStack`, so that a root can outlive
    the `async with` that made it. A failure partway through is therefore the
    one path where the `finally` of `mounted` does not run unless that stack is
    unwound. The observable half is that the id is not listed and can be opened
    later. The mount is checked when the supervisor is disposed, which would
    raise on a context that it never took.
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
# The ladder answers a crash of the *task* of a root: a flush that cannot write,
# a disposed context, a bug. In those cases the work is still in the inbox, and
# to run again is meaningful. A failed *turn* is deliberately not the concern of
# the ladder. `llm-retry` has already retried what is worth a retry after a
# model failure. The failed turn also claimed its message, so a second `run()`
# would produce an empty turn that reports false success.


@pytest.fixture
def short_ladder(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real ladder takes 6.25 s, which is too long for a test suite.

    The delays are patched on the module and not passed in, because
    `Recovery.total` and `Recovery.delay` read the global when they are asked.
    That is why they are properties, and not values that are copied at import.
    """
    monkeypatch.setattr(recovery, "RETRY_DELAYS", (0.01, 0.01, 0.01))


def _crash(patch: pytest.MonkeyPatch, root: Any, times: int) -> None:  # noqa: ANN401
    """Make the task of this root raise for its first `times` wakes.

    The crash is injected at the boundary of `run()` itself, which is where a
    *task* crash really appears. The driver contains everything inside a turn.
    A failure that is injected further in (a flush during the turn, a model
    error) is caught there and arrives as `turn/end{error}`. That is not a
    crash, and it is not the concern of this ladder.

    To raise before `run()` claims anything also keeps the property that the
    ladder depends on. The work is still in the inbox, so to run again is
    meaningful, and is not an empty turn that reports false success.

    This helper patches and counts together, on the class. `Supervisor` and the
    driver are both `slots=True`, so an instance attribute cannot be shadowed,
    and every call site wrote `type(root.agent)` twice to say one thing.
    """
    driver, original, remaining = type(root.agent), type(root.agent).run, times

    async def run(self: object) -> None:
        nonlocal remaining
        if remaining > 0:
            remaining -= 1
            raise RuntimeError("injected crash")
        await original(self)

    patch.setattr(driver, "run", run)


def _on_disk(path: Path, marker: str) -> bool:
    """Whether `marker` has reached this log yet. The answer is also correct
    before the log exists.

    The file of a session appears on its first flush. A wait that starts earlier
    reads a path that is not there. `read_text` then raises `FileNotFoundError`
    and does not answer "not yet", which turns a poll into a crash.
    """
    return path.exists() and marker in path.read_text()


async def test_an_injected_crash_is_retried_and_the_root_recovers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The first half of the gate: an injected crash recovers.

    There is one crash, and then everything works again. The root must come
    back and finish the turn. The message is still in its inbox, which is
    exactly why this failure is worth a retry and a failed turn is not.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="recovers")
        root = daemon.running.supervisor.roots["recovers"]
        _crash(monkeypatch, root, 1)

        await client.prompt("recovers", "hello")
        # The wait is for the *answer*, not for "idle". A root used to read as
        # idle for the whole backoff of the ladder. A wait for idle then exited
        # during the sleep, and asserted against a log that the retry had not
        # written to yet. (`Root.status` now reports `retrying` there.)
        await until(
            lambda: any(e.type == "assistant/message" for e in root.session.events_from(0)),
            what="the retried task to finish its turn",
        )

        await until(
            lambda: root.recovery.attempts == 0, what="the ladder to clear after recovering"
        )
        types = [event.type for event in root.session.events_from(0)]
        assert types.count(recovery.RETRY) == 1
        assert recovery.FAILED not in types, "a root that recovered was reported failed"
        # This is the marker that clears the count, and nothing else may clear
        # it. A ladder that its own retry can reset does not terminate.
        assert types.count(recovery.RECOVERED) == 1
        assert "assistant/message" in types, "the retried task never finished its turn"
        assert root.status == "idle"


async def test_the_ladder_gives_up_after_its_last_attempt_and_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The second half of the gate: the third failure reports.

    A ladder that never gave up would be worse than no ladder. A permanently
    broken root would retry forever and report itself busy. The count is
    therefore bounded, the give-up is *recorded*, and the status that a client
    reads changes.
    """
    seen: list[dict[str, Any]] = []
    async with running(tmp_path) as daemon:
        client = await daemon.client(on_notify=lambda method, params: seen.append(params))
        await client.call("session/new", sessionId="doomed")
        await client.call("session/attach", sessionId="doomed")
        root = daemon.running.supervisor.roots["doomed"]
        _crash(monkeypatch, root, 99)

        await client.prompt("doomed", "hello")
        await until(lambda: root.status == "failed", what="the ladder to give up")

        types = [event.type for event in root.session.events_from(0)]
        assert types.count(recovery.RETRY) == len(recovery.RETRY_DELAYS)
        assert types.count(recovery.FAILED) == 1
        # The give-up is recorded, and not only announced. An unattended run
        # leaves the fact in its own trace, whether or not a client was ever
        # attached.
        given_up = next(e for e in root.session.events_from(0) if e.type == recovery.FAILED)
        assert given_up.data["attempts"] == len(recovery.RETRY_DELAYS)
        assert "injected crash" in str(given_up.data["reason"])
        # The test waits for the status and does not sample it. `status` changes
        # when `give_up` appends, while the notification is still on its way
        # through the outbox and the pump. A read of `seen` at that instant
        # caught the client in the middle of delivery, about one run in four:
        # `retrying` had arrived and `failed` had not.
        await until(
            lambda: "failed" in {str(params.get("status", "")) for params in seen},
            what="the failed status to reach the client",
        )

        # **On disk, with the daemon still running and no shutdown sent.** The
        # give-up is the record that matters most, and it was the one record
        # that write-through missed. A clean shutdown flushes it in either
        # case, so only an assertion here, and not after teardown, tests the
        # behavior. The test waits for the record and does not read one time.
        # `status` changes when the event is *appended*, and the flush that
        # carries it to disk is the next await. An immediate read therefore
        # raced the ordering under test, about one run in six. With the flush
        # deleted, this wait runs to its timeout, which is still a failure.
        written = stored_log(tmp_path / "sessions", "doomed")
        await until(lambda: _on_disk(written, recovery.FAILED), what="the give-up to reach disk")


async def test_a_root_resumed_mid_ladder_does_not_start_the_count_over(
    tmp_path: Path, short_ladder: None
) -> None:
    """Why the state of the ladder is folded and not remembered.

    A supervisor that kept the attempt count in memory would come back from
    every crash with a fresh ladder. A root that fails for a permanent reason
    would then retry forever: three attempts for each daemon lifetime, with
    nothing in the log to show that it had ever been tried.
    """
    # The crash has its own patch context, deliberately. `monkeypatch.undo()`
    # would revert every patch on the shared fixture, including the autouse
    # `_isolated_home`. The second daemon would then resume from the real
    # `~/.ph` of the developer. This test did exactly that, and wrote a session
    # there.
    with pytest.MonkeyPatch.context() as crash:
        async with running(tmp_path, name="first") as daemon:
            client = await daemon.client()
            await client.call("session/new", sessionId="stubborn")
            root = daemon.running.supervisor.roots["stubborn"]
            _crash(crash, root, 99)
            await client.prompt("stubborn", "hello")
            # The wait is for the *disk*, not for the status. `status` changes
            # when `give_up` appends, and this test is about what the next
            # daemon can read back. A shutdown at the change in memory raced
            # the flush that makes the claim true, and the resumed root came
            # back as `retrying`.
            written = stored_log(tmp_path / "sessions", "stubborn")
            await until(
                lambda: _on_disk(written, recovery.FAILED),
                what="the spent ladder to reach disk",
            )
            await client.notify("shutdown")

    async with running(tmp_path, name="second") as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="stubborn")
        root = daemon.running.supervisor.roots["stubborn"]
        # This is read directly from the resumed log. Nothing is carried in
        # memory between the two daemons.
        assert root.status == "failed"
        state = recovery.recovery_of(root.session)
        assert state.attempts == len(recovery.RETRY_DELAYS)
        assert state.spent


async def test_one_root_crashing_does_not_take_the_daemon_down(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The failure that a supervisor exists to prevent.

    The task of a root runs in the task group of the supervisor. Anything that
    raises out of the task therefore cancels the group, and with it every
    *other* root and the listener. This test kills one root completely and
    asserts that nothing else is affected.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("bystander", "hello")
        await _settled(client, "bystander", events=1)

        await client.call("session/new", sessionId="casualty")
        casualty = daemon.running.supervisor.roots["casualty"]
        _crash(monkeypatch, casualty, 99)
        await client.prompt("casualty", "hello")
        await until(lambda: casualty.status == "failed", what="the doomed root to give up")
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

    A retry that ran against whatever the crashed attempt left behind would be
    a different attempt from the one that failed. The model would see edits
    from a run that nobody kept. A ladder that adds to its own damage is worse
    than no ladder. `workspace/checkpoint` is the record of P4-09 and already a
    fold, so the *latest* one is the tree to go back to.

    The test drives the supervisor directly, and not through a daemon on the
    worktree tier. It tests the selection and the best-effort contract. A real
    git worktree would test the capture of P4-09 again.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("restores")
        log_event(root.session, "workspace/checkpoint", {"agentId": root.agent.id, "tree": "older"})
        log_event(
            root.session, "workspace/checkpoint", {"agentId": root.agent.id, "tree": "newest"}
        )

        asked: list[str] = []

        async def put_back(token: str) -> tuple[str, ...]:
            asked.append(token)
            return ()

        # The patch is on the mounted **tier**, not on a module import. The
        # ladder asks the seam, so a patch of a name in this module would have
        # continued to pass while the call that it stands for went somewhere
        # else.
        seam = root.ctx.require(WORKSPACE)
        monkeypatch.setattr(type(seam), "of", lambda self, agent_id: object())
        monkeypatch.setattr(seam, "provider", _Checkpoints(put_back))

        workspace, tree = supervisor._restore_point(root)
        assert tree == "newest", "the retry went back to a stale restore point"
        assert workspace is not None
        await supervisor._restore(root, workspace, tree)
        assert asked == ["newest"]
        assert not_none(root.session.latest(RESTORED)).data["ok"] is True

        # Best-effort: a restore that fails must not cost the retry, and must
        # not claim a rollback that did not happen.
        async def refuse(token: str) -> tuple[str, ...]:
            raise RuntimeError("the tier said no")

        monkeypatch.setattr(seam, "provider", _Checkpoints(refuse))
        await supervisor._restore(root, workspace, tree)
        assert not_none(root.session.latest(RESTORED)).data["ok"] is False


async def test_a_root_with_no_workspace_restores_nothing_and_says_so(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The ordinary case: a root on the advisory tier has no worktree to put
    back.

    No restore is opened at all. The alternative is a restore that names
    nothing. A transcript that implied a rollback that nobody performed would
    make a reader misread the attempt that follows.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("advisory")
        assert supervisor._restore_point(root) == (None, None)


async def test_a_root_that_took_no_restore_point_has_no_tree_to_go_back_to(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root can have a workspace and no `workspace/checkpoint` yet.

    The ladder then has no tree to put back, and it must not open a restore.
    The seam answers `""` for such an agent. `_restore_point` turns that into
    `None`, which is what the ladder tests before it restores.

    Sabotage: drop the `stated` around `latest_checkpoint` in `_restore_point`,
    and the ladder is handed a tree named `""` to restore.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("unmarked")
        seam = root.ctx.require(WORKSPACE)
        monkeypatch.setattr(type(seam), "of", lambda self, agent_id: object())

        workspace, tree = supervisor._restore_point(root)

        assert workspace is not None
        assert tree is None


async def test_a_retry_is_on_disk_before_its_restore_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """S12: the restore is part of the attempt, and it used to run unrecorded.

    The restore ran first, and the retry was written after it. A daemon that
    died during the restore had then rewritten the tree with no record of the
    attempt, and a crash loop there never advanced the count. Now the retry
    reaches disk first, and then the opening record of the restore itself
    (`WORKSPACE_RESTORE`). A restore that fails is settled as `ok: false`, so
    the transcript claims no rollback.

    Sabotage: restore before recording again, and the restore finds neither
    record on disk.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("restores")
        log_event(root.session, "workspace/checkpoint", {"agentId": root.agent.id, "tree": "t1"})
        seen: list[tuple[bool, bool]] = []

        def written() -> list[str]:
            return stored_types(root.ctx, root.id)

        async def refused(token: str) -> tuple[str, ...]:
            seen.append((recovery.RETRY in written(), RESTORING in written()))
            raise RuntimeError("the tier said no")

        seam = root.ctx.require(WORKSPACE)
        monkeypatch.setattr(type(seam), "of", lambda self, agent_id: object())
        monkeypatch.setattr(seam, "provider", _Checkpoints(refused))
        _crash(monkeypatch, root, 1)
        await supervisor.prompt("restores", "hello")
        await until(lambda: RESTORED in written(), what="the restore's settle")

        assert seen == [(True, True)], "the tree was touched before the records were on disk"
        ladder = [
            one for one in root.session.events if one.type in {recovery.RETRY, RESTORING, RESTORED}
        ]
        assert [one.type for one in ladder] == [recovery.RETRY, RESTORING, RESTORED]
        assert dict(ladder[1].data) == {"agentId": root.agent.id, "tree": "t1"}
        assert ladder[2].data["restoringSeq"] == ladder[1].seq
        assert ladder[2].data["ok"] is False
        assert "the tier said no" in as_str(ladder[2].data["detail"])


async def test_a_failing_flush_climbs_the_ladder_instead_of_retrying_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """The case that made the ladder unbounded, and that nothing tested.

    Every other test here injects at the entry of `run()`, so no turn ever
    completes. A *flush that fails persistently* is different in the one way
    that matters: `run()` succeeds first and writes a `turn/end`. The first
    version of the fold reset the count on any `turn/end`, and the retry
    produces one. A `run()` that is entered again finds an empty inbox and
    appends `turn/start` + `turn/end{completed}` before the same flush fails
    again. The ladder therefore cleared the bound that was supposed to stop it.
    This measured **165 retries in two seconds, no give-up, the fold stuck at
    one attempt, and the root reporting "idle"**, and the log grew by three
    events for each iteration.

    Only `supervisor/recovered` resets the count now, and only success writes
    it.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="unflushable")
        root = daemon.running.supervisor.roots["unflushable"]

        async def broken(self: object, session: Session) -> None:
            raise RuntimeError("flush is broken")

        monkeypatch.setattr(type(root.ctx.require(SESSIONS)), "flush", broken)
        await client.prompt("unflushable", "hello")
        await until(lambda: root.status == "failed", what="the ladder to give up")

        types = [event.type for event in root.session.events_from(0)]
        assert types.count(recovery.RETRY) == len(recovery.RETRY_DELAYS), (
            "the ladder did not terminate — its own retry cleared the count"
        )
        assert types.count(recovery.FAILED) == 1
        assert types.count(recovery.RECOVERED) == 0


# --- P5-05: passivation ------------------------------------------------------
#
# A daemon that is built to run for weeks accumulates roots. Each root holds a
# mounted profile, a session, an agent and a workspace. Passivation releases the
# half that is in the process and keeps the session on disk. Rehydration is not
# a second mechanism: `start()` already resumes any root whose log exists
# (P5-01). The round-trip is therefore a property here, and not a feature.


async def test_an_idle_root_is_released_and_comes_back_with_its_history(
    tmp_path: Path,
) -> None:
    """The gate: round-trip.

    The root is released while nobody wants it, and the next message brings it
    back. What it already said is still in the log, because the log held it the
    whole time.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
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
    """A gap that nobody explained looks like a crash.

    The record is checked on disk while the daemon is still running. It is
    appended and then flushed as part of the release. A reader who opens this
    log therefore finds out why the log stops, and does not infer a failure.
    The `session/resumed` on the way back says only that something resumed, not
    that nothing was wrong.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("recorded", "hello")
        await _settled(client, "recorded", events=1)
        await daemon.running.supervisor.sweep(after=0)

        assert _on_disk(stored_log(tmp_path / "sessions", "recorded"), recovery.PASSIVATED), (
            "the record did not reach disk with the release"
        )


async def test_a_root_somebody_is_watching_is_not_released(tmp_path: Path) -> None:
    """An attached client is a reason why the root is still wanted.

    The answer comes from the subscriber set of the root itself, which is the
    same fact that makes the client receive events. A client therefore cannot
    be receiving a session that was released under it.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/new", sessionId="watched")
        await client.call("session/attach", sessionId="watched")
        assert await daemon.running.supervisor.sweep(after=0) == []

        await client.call("session/detach", sessionId="watched")
        assert await daemon.running.supervisor.sweep(after=0) == ["watched"]


async def test_a_root_that_has_not_been_quiet_long_enough_is_not_released(
    tmp_path: Path,
) -> None:
    """The timeout is read from the log, not from a timer.

    Here `now` is the time of the last event of the log itself, so nothing has
    been quiet for any time at all. The same rule makes a root that is
    rehydrated from a three-day-old log eligible immediately, which is correct.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
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

    A root in the backoff of P5-04 does nothing between attempts. A release
    there would passivate a root partway up a ladder that it had already
    recorded. The condition that prevents this is the one that the cleanup pass
    added, when it found `retrying` announced as a notification while
    `sessions/list` still said `idle`.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="climbing")
        root = supervisor.roots["climbing"]
        _crash(monkeypatch, root, 99)

        await client.prompt("climbing", "hello")
        await until(lambda: root.recovery.attempts > 0, what="the ladder to start")
        assert root.status == "retrying"
        assert await supervisor.sweep(after=0) == [], "released a root mid-ladder"


async def test_a_passivated_session_may_be_opened_by_another_process(
    tmp_path: Path,
) -> None:
    """A release gives back the I-5 lease. That is the purpose, and not a leak.

    A root holds the lease of its session for as long as it is mounted (P5-03).
    If passivation kept the lease, *no* process could open a released session.
    The daemon that let it go could not open it, and everyone else would be
    refused.
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

        await first.running.supervisor.sweep(after=0)
        # Now no daemon holds it, so the second daemon may have it.
        assert (await other.call("session/new", sessionId="handed-over"))["sessionId"] == (
            "handed-over"
        )


async def test_attaching_wakes_a_passivated_root(tmp_path: Path) -> None:
    """A client that attaches to a released session gets it back, not an error.

    The attach goes through `start`, which is the path that `session/prompt`
    takes, so a wake has one mechanism. Without this, a session that was
    released while its watcher was away came back as `no_such_session`. That
    reads as "gone" for something that is still on disk.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.prompt("awaited", "hello")
        await _settled(client, "awaited", events=1)
        await daemon.running.supervisor.sweep(after=0)
        assert "awaited" not in daemon.running.supervisor.roots

        attached = await client.call("session/attach", sessionId="awaited")
        assert attached["sessionId"] == "awaited"
        assert "awaited" in daemon.running.supervisor.roots


async def test_release_actually_runs(tmp_path: Path) -> None:
    """This tests the planner that `serve` starts, not only the predicate.

    Everything above drives `sweep()` directly. Those tests would pass equally
    well if nothing ever called `sweep()`, which is how dead code comes to look
    tested.
    """
    async with running(tmp_path, passivate_after=0.0) as daemon:
        client = await daemon.client()
        # The client attaches first, and that is deliberate setup and not an
        # accident. With `passivate_after=0` the root is releasable at the
        # instant that it is idle and unwatched, which is *during* the turn
        # that this test waits for. A real client relies on the same condition,
        # an attach, to keep a session that it uses.
        await client.call("session/new", sessionId="swept")
        await client.call("session/attach", sessionId="swept")
        await client.prompt("swept", "hello")
        await _settled(client, "swept", events=1)
        await client.call("session/detach", sessionId="swept")
        await until(
            lambda: "swept" not in daemon.running.supervisor.roots,
            what="an idle root to be released on its own",
        )


@asynccontextmanager
async def _releasing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, window: float = 0.2
) -> AsyncIterator[Supervisor]:
    """A supervisor whose releaser is running, with a short quiet window (P12-01).

    The window is above zero and not zero. A release then comes after the turn
    or the pass that made the root quiet has finished. A test that asserts
    "still mounted" can wait past the window, and the assertion then means
    something.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        supervisor.passivate_after = window
        stop = anyio.Event()
        supervisor.tasks.start_soon(supervisor.releaser.keep, stop)
        try:
            yield supervisor
        finally:
            stop.set()


async def _still_mounted(supervisor: Supervisor, root_id: str) -> None:
    """Wait past the quiet window and past the floor after the last pass, and
    assert that the root is still there. Whatever holds the root then really
    holds it."""
    await anyio.sleep(not_none(supervisor.passivate_after) + PASS_FLOOR + 0.2)
    assert root_id in supervisor.roots, f"{root_id} was released while held"


async def _released(supervisor: Supervisor, root_id: str) -> None:
    await until(lambda: root_id not in supervisor.roots, what=f"{root_id} to be released")


async def test_an_idle_daemon_releases_at_the_deadline_without_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The releaser sleeps until the quiet window ends, with no pass in between
    (P12-01).

    There is one pass at boot and one when the root mounts. The next pass is
    the one that releases the root. A sweep would have run every sixty seconds,
    whether or not anything could have come due.

    Sabotage: put back a cadence, and the passes counted while it waits grow.
    """
    async with _releasing(tmp_path, monkeypatch, window=2.0) as supervisor:
        passes: list[int] = []
        sweep = supervisor.sweep

        async def counted() -> list[str]:
            passes.append(now_ms())
            return await sweep()

        supervisor.releaser.run = counted
        supervisor.releaser.notice()  # the boot pass again, through the counter
        await until(lambda: len(passes) == 1, what="the counted pass")
        await supervisor.start("quiet")
        await until(lambda: len(passes) == 2, what="the mount's pass")

        await anyio.sleep(1.0)
        assert len(passes) == 2, "nothing could have come due, and nothing ran"
        await _released(supervisor, "quiet")
        assert len(passes) == 3


async def test_a_root_mounting_is_planned_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A root that does nothing at all after it mounts is still released.

    Sabotage: drop the releaser's notice from `_start`, and a root that never
    runs a turn is never planned for.
    """
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        await supervisor.start("untouched")
        await _released(supervisor, "untouched")


async def test_a_turn_ending_lets_the_root_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: ModelGate
) -> None:
    """The most common way that a quiet window starts: a turn ends.

    The turn is held open at the model until after the window. The pass that
    the mount planned therefore finds the root `running` and plans nothing. A
    turn that is shorter than the window needs no notice: the pass would find
    the turn finished and plan from its last record.

    Sabotage: drop `recheck()` from the `agent/status` listener, and nothing
    wakes the releaser when the turn ends.
    """
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("worked")
        await supervisor.prompt("worked", "hello")
        await until(lambda: gate.arrived == 1, what="the turn to reach the model")
        assert root.status == "running"
        await _still_mounted(supervisor, "worked")

        gate.release_all()
        await _released(supervisor, "worked")


async def test_the_last_watcher_leaving_lets_the_root_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sabotage: drop `emptied()` from `Audience.drop`, and the root stays."""
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("watched")

        def watcher(_method: str, _params: dict[str, Any]) -> None:
            return None

        root.subscribe(watcher)
        await _still_mounted(supervisor, "watched")

        root.audience.drop(watcher)
        await _released(supervisor, "watched")


async def test_a_root_parking_on_a_person_lets_it_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`waiting` is releasable, and the desk reports that the root now reads
    `waiting`.

    The root is first held by the ladder (`retrying` is work in hand). The only
    thing that can then make it releasable is the ask, because the desk takes
    priority over the ladder in `Root.status`.

    Sabotage: drop the desk's `root.recheck()` when it opens an ask, and the
    root stays.
    """
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("parked")
        root.retry(reason="the provider is down")
        await _still_mounted(supervisor, "parked")

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(ask_a_person, root)
            await until(lambda: root.status == "waiting", what="the root to park")
            await _released(supervisor, "parked")
            tasks.cancel_scope.cancel()


async def test_a_child_settling_lets_the_root_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The record of the child is heard on the bus, before the watcher guard of
    `beneath`.

    Sabotage: move the releaser's notice back under `beneath`'s
    `not root.subscribers` return, and an unwatched parent stays mounted after
    its child is done.
    """
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("parent")
        child = spawned(root, "c")
        await _still_mounted(supervisor, "parent")

        log_event(child, STATUS, {"status": "done"})
        await _released(supervisor, "parent")


async def test_a_schedule_ending_lets_the_root_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sabotage: drop `_recheck()`, which carries the releaser's notice, from
    `_watch_schedules`, and the root stays after its last appointment is
    withdrawn."""
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("booked")
        seam = root.ctx.require(SCHEDULE)
        seam.create(root.session, Schedule(id="s", kind="cron", spec="0 9 * * *", prompt="go"))
        await _still_mounted(supervisor, "booked")

        seam.cancel(root.session, "s")
        await _released(supervisor, "booked")


@pytest.mark.parametrize("end", ["give_up", "recovered"])
async def test_the_ladder_ending_lets_the_root_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, end: str
) -> None:
    """`retrying` holds a root. `failed` and a recovered `idle` do not. Neither
    change moves the status of the agent itself, so the root reports the change
    itself.

    Sabotage: drop `recheck()` from `Root.give_up` or `Root.recovered`, and that
    case stays mounted.
    """
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("laddered")
        root.retry(reason="the provider is down")
        await _still_mounted(supervisor, "laddered")

        if end == "give_up":
            root.give_up("the provider is down", attempts=3)
        else:
            root.recovered()
        await _released(supervisor, "laddered")


async def test_a_root_with_a_live_child_is_not_released(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A parent whose subagent is still working stays mounted. The answer is
    **read from the log of the child itself** (P11-07).

    A release would take the mount of the child with it. The child is suspended
    in the middle of its task, and comes back only when something wakes the
    root. The log of a root no longer holds any record of its children. The
    question is therefore asked of the log of each child, through
    `child_is_live` of the seam, whose settled set is the set that every
    producer writes. The first version of this test wrote that set itself, as
    `{"completed", "failed", "canceled", "deleted"}`. Only one of those is a
    string that any producer emits, and a root that had ever run a child to
    completion could never be released.

    **A grandchild counts too**, at every level. The resume sweep has revoked
    what an ended child left unfinished, so a live grandchild is real work
    (`test_what_an_ended_child_left_unfinished_is_revoked_when_its_root_comes_back`).

    Sabotage: read the root's own log for `subagent/*` again, and every child
    reads as absent. Each root is then released under a working child.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        for label, settle in (
            ("finished", (STATUS, {"status": "done"})),
            ("errored", (STATUS, {"status": "error", "detail": "boom"})),
            ("canceled", (STATUS, {"status": "canceled"})),
            ("revoked", (DELETED, {"reason": "revoked"})),
        ):
            root = await supervisor.start(label)
            child = spawned(root, "c")
            assert await supervisor.sweep(after=0) == [], f"{label}: released with a live child"

            log_event(child, *settle)
            assert await supervisor.sweep(after=0) == [label], f"{label}: child never settled"

        # An unrecognized status keeps the parent mounted, because its child
        # may still be running.
        root = await supervisor.start("unknown")
        log_event(spawned(root, "c"), STATUS, {"status": "who-knows"})
        assert await supervisor.sweep(after=0) == [], "an unknown status released the parent"

        # A grandchild that is working holds the root, even under a child that has
        # already ended.
        root = await supervisor.start("deep")
        child = spawned(root, "c")
        log_event(child, STATUS, {"status": "done"})
        grandchild = spawned(root, "g", under=child)
        assert await supervisor.sweep(after=0) == [], "released under a running grandchild"

        log_event(grandchild, STATUS, {"status": "done"})
        assert await supervisor.sweep(after=0) == ["deep"], "the grandchild never settled"


async def test_what_an_ended_child_left_unfinished_is_revoked_when_its_root_comes_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The whole tree is read when its root comes back, and the unfinished
    children of an ended child are ended in their own logs.**

    The children of a child are artifacts of its scope, and are revoked as the
    scope unwinds. But a crash between the ending of a child and the ending of
    its children leaves a grandchild `running` on disk, beneath a child that
    ended. When the tree was read one level at a time, the next process never
    saw that grandchild. Its spend reached no goal, the panel never listed it,
    and nothing ended it. Now the whole tree of the root is read as its log
    opens, and the sweep revokes what the ended child left. The grandchild
    therefore holds nothing, and is reported as what it is.

    Sabotage: skip `_revoke_beneath` for a child that had ended, and the
    grandchild is still working after the restart.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("deep")
        child = spawned(root, "c")
        grandchild = spawned(root, "g", under=child)
        log_event(grandchild, STATUS, {"status": "running"})
        log_event(child, STATUS, {"status": "done"})
        for log in (root.session, child, grandchild):
            assert await session_written(root.ctx, log)

    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("deep")
        family = {state.run_id: state for state in family_of(root)}
        assert set(family) == {"c", "g"}, "the grandchild was not read with its root"
        revoked = family["g"]
        assert (revoked.deleted, revoked.deleted_reason) == (True, PARENT_TEARDOWN)
        assert revoked.status == "canceled"
        assert await supervisor.sweep(after=0) == ["deep"], "held by what the child left"


# --- P11-07: the family, from the children's own logs ------------------------


async def test_a_childs_status_reaches_the_client_as_a_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate of P11-07: the state of a child reaches a watcher from the log
    of the child itself.**

    The stream of a root no longer carries anything about its children, because
    each child writes only its own log (Phase 11). A client that watches the
    root would therefore see an empty family. The supervisor listens to every
    log in the mount of the root, and pushes `session.children`, the whole
    family, whenever a row changes:

    - **One frame for a batch**, because the `canceled` record of a revocation
      and its tombstone arrive together.
    - **No frame for a chunk**, which changes no row.

    The events of the child itself never reach the watchers of the root.

    Sabotage: stop listening to `session/event` in `_start`, and the notice
    never comes.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("lead")
        heard: list[tuple[str, dict[str, Any]]] = []

        def watch(method: str, params: dict[str, Any]) -> None:
            heard.append((method, params))

        def families() -> list[SessionChildrenNotice]:
            return [
                SessionChildrenNotice.model_validate(params)
                for method, params in heard
                if method == SessionChildrenNotice.METHOD
            ]

        root.subscribe(watch)
        child = spawned(root, "c1", name="scout")
        await until(lambda: len(families()) == 1, what="the admission to be pushed")
        log_event(child, STATUS, {"status": "running"})
        await until(lambda: len(families()) == 2, what="the child's start to be pushed")
        assert [(row.name, row.status) for row in families()[-1].children] == [("scout", "running")]

        log_event(child, "assistant/chunk", {"turn": 1, "step": 1, "chunk": {"type": "x"}})
        await anyio.wait_all_tasks_blocked()
        assert len(families()) == 2, "a chunk moved no row and was pushed anyway"

        with child.batch() as batch:
            log_event(batch, STATUS, {"status": "canceled"})
            log_event(batch, DELETED, {"reason": "revoked"})
        await until(
            lambda: any(row.deleted for row in families()[-1].children),
            what="the revocation to be pushed",
        )
        await anyio.wait_all_tasks_blocked()
        assert len(families()) == 3, "one batch was pushed as two frames"
        (row,) = families()[-1].children
        assert (row.status, row.deleted, row.deleted_reason) == ("canceled", True, "revoked")

        streamed = {
            as_str(params["event"].get("type"))
            for method, params in heard
            if method == SessionEventNotice.METHOD
        }
        assert not streamed & {ADMITTED, STATUS, DELETED, "assistant/chunk"}, (
            "a child's own record reached the root's stream"
        )


async def test_the_children_projection_lists_grandchildren(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`session/children` is the whole family, with a parent before its own
    children.

    The log of a grandchild names the child that spawned it, not the root. A
    list of only the children of the root would therefore leave out every level
    beneath them, which is the work that a fan-out delegated again. Each row
    carries its `parentId`, which lets a reader draw the tree in one pass.

    Sabotage: list only `children(root.id)` in `family_of`, and the grandchild
    is missing.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("lead")
        first = spawned(root, "c1")
        spawned(root, "c2")
        spawned(root, "g1", under=first)

        rows = family_rows(root)

        assert [(row.run_id, row.parent_id) for row in rows] == [
            ("c1", "lead"),
            ("g1", "lead-c1"),
            ("c2", "lead"),
        ]
        # This is what `session/children` answers, through the reply model of
        # its verb.
        wire = SessionChildrenNotice(session_id="lead", children=rows).to_wire()
        assert verbs.SESSION_CHILDREN.read(wire).children == rows


async def test_a_held_child_is_listed_as_awaiting_a_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sub-agent that is held for a key is reported as held, at any depth. It
    is reported in the family and in the credential rows of the doctor (T5).

    The hold is now in the log of the child itself, as a session that waits on
    its own route (`SESSION_HOLDER`). It used to be a row in the log of the
    parent. The log of a root therefore answers for the root alone, and
    `awaited` asks every level beneath it. It names each holder by the run ids
    from the root down.

    Sabotage: read holds from the root's log alone in `_waiting_on`, and the
    grandchild waits unlisted.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("lead")
        child = spawned(root, "c1")
        grandchild = spawned(root, "g1", under=child)

        log_event(grandchild, "credential/needed", credential_hold(SESSION_HOLDER, "EXAMPLE_KEY"))

        assert supervisor.awaited() == [("EXAMPLE_KEY", "lead/c1/g1")]
        assert [row.awaiting for row in family_rows(root)] == [None, "EXAMPLE_KEY"]

        log_event(grandchild, "credential/supplied", credential_hold(SESSION_HOLDER, "EXAMPLE_KEY"))
        assert supervisor.awaited() == []


# --- P5-06: the scheduler ----------------------------------------------------
#
# The seam is tested in ph-core against a log. These tests cover the two things
# that only the daemon can answer: that a due tick becomes a real turn, and that
# P5-05 does not release a root with work scheduled while the root waits for
# that work.


async def test_a_due_schedule_starts_a_turn(tmp_path: Path) -> None:
    """The tick delivers through `prompt`, so a scheduled turn is an ordinary
    turn.

    It takes the same path as the message of a person, with a `schedule/tick`
    beside it that says why the turn started. The alternative is a second way
    into the loop, with its own bugs and its own transcript format.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="cron")
        root = supervisor.roots["cron"]

        root.ctx.require(SCHEDULE).create(
            root.session,
            Schedule(id="nightly", kind="interval", spec="60000", prompt="do the thing"),
        )
        # The times are relative to creation. A schedule is anchored at the
        # time when it was made, so a clock that starts at zero is decades
        # before its own schedule exists.
        made = root.ctx.require(SCHEDULE).states(root.session)["nightly"].created_at
        assert await supervisor.tick(now=made + 1_000) == [], "fired before it was due"

        assert await supervisor.tick(now=made + 90_000) == ["nightly"]
        await _settled(client, "cron", events=1)

        types = [event.type for event in root.session.events_from(0)]
        assert types.count("schedule/tick") == 1
        assert "assistant/message" in types, "the scheduled turn never ran"


async def test_a_due_schedule_starts_its_own_turn_even_mid_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An *ordinary* turn is the contract of this path, and a busy root must not
    break it.

    An interjection from a person is delivered at the next step, deliberately.
    But a schedule is not "also this". If it joined a turn that it has nothing
    to do with, it would share the per-turn ceilings of that turn. Two
    schedules that are due in one pass would also become one turn and not two.
    """

    targets: list[str] = []
    original = ReactLoopAgent.send

    def spy(self: Any, message: Any, target: InboxTarget, wakeup: bool) -> None:  # noqa: ANN401
        targets.append(target)
        original(self, message, target, wakeup)

    monkeypatch.setattr(ReactLoopAgent, "send", spy)
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        root = await supervisor.start("cron-busy")
        root.ctx.require(SCHEDULE).create(
            root.session,
            Schedule(id="nightly", kind="interval", spec="60000", prompt="do the thing"),
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["nightly"].created_at

        driver = root.agent
        assert isinstance(driver, ReactLoopAgent)
        driver._phase.kind = "running"
        assert await supervisor.tick(now=made + 90_000) == ["nightly"]

    assert targets[-1] == "next-turn", "a scheduled turn joined a turn already running"


async def test_a_root_with_work_scheduled_is_not_released(tmp_path: Path) -> None:
    """The fourth condition of P5-05, which that row left open for this one.

    A root that has said when it comes back is a root that is still wanted. A
    release would drop the only thing that knows the appointment. Passivation
    also unwinds the `Context`, so the schedule would no longer be watched,
    while the log still claimed that it fires at nine.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        root = await supervisor.start("appointed")
        assert await supervisor.sweep(after=0) == ["appointed"], "an empty root should release"

        root = await supervisor.start("appointed")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="cron", spec="0 9 * * *", prompt="morning")
        )
        assert await supervisor.sweep(after=0) == [], "released a root with work scheduled"

        root.ctx.require(SCHEDULE).cancel(root.session, "s")
        assert await supervisor.sweep(after=0) == ["appointed"]


async def test_one_root_with_a_broken_schedule_does_not_stop_the_others(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The schedules of every root fire from one loop, so one of them must not
    end the loop."""
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
        broken = await supervisor.start("broken")
        working = await supervisor.start("working")
        working.ctx.require(SCHEDULE).create(
            working.session, Schedule(id="ok", kind="interval", spec="1000", prompt="hi")
        )

        original = type(broken.ctx.require(SCHEDULE)).claim

        def claim(self: Any, session: Session, *, now: int) -> Any:  # noqa: ANN401
            if session.id == "broken":
                raise RuntimeError("this schedule is unreadable")
            return original(self, session, now=now)

        monkeypatch.setattr(type(broken.ctx.require(SCHEDULE)), "claim", claim)
        made = working.ctx.require(SCHEDULE).states(working.session)["ok"].created_at
        assert await supervisor.tick(now=made + 10_000) == ["ok"]


# --- P5-11: lingering detection (I-6) ----------------------------------------
#
# Gate: *simulated session end → clear diagnostic, not a silent failure.*
#
# The simulation is exact and not a metaphor. The `reaped_host` fixture in the
# repo-root conftest sets `$XDG_RUNTIME_DIR` to a directory that these tests
# own, and puts `$PH_RUNTIME` inside it. The socket is bound there. "The login
# session ended" is the removal of that directory, which is what logind does to
# `/run/user/$UID` for a user who is not lingering. What must not happen after
# that is *nothing*. The daemon keeps running with no socket that a client can
# reach, and the only reader left is whoever opens a transcript afterwards.


def _notices(root: Root) -> list[SessionEvent]:
    return [event for event in root.session.events_from(0) if event.type == recovery.UNREACHABLE]


async def test_a_reaped_runtime_dir_reaches_every_root_as_a_record(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """The gate. A login session that ends takes the socket with it. The log
    records that, and why.

    The record goes to every root, not only to the busy ones. The daemon is
    what became unreachable, and a transcript that stops with no explanation is
    the same puzzle for a busy root and for an idle one. The record carries the
    advice, because the reader who finds it is by then far from the terminal
    that could have warned them.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        supervisor = daemon.running.supervisor
        first = await supervisor.start("alpha")
        second = await supervisor.start("beta")

        # Logout, as logind performs it: the whole directory, not just the file.
        shutil.rmtree(tmp_path / "xdg")
        await until(
            lambda: _notices(first) and _notices(second), what="both roots to hear of the logout"
        )

        said = _notices(first)[0].data
        assert said["reason"] == "removed"
        assert said["socket"] == str(socket)
        assert said["linger"] == "off"
        assert said["advice"] == "loginctl enable-linger someone"
        # Which daemon, and which incident. Without these fields, the question
        # "were these two sessions in the same outage" can be answered only by
        # correlating clock times across every log, and only if the payloads
        # happen to be equal.
        assert said["pid"] == os.getpid()
        assert _notices(second)[0].data["since"] == said["since"]
        # The record is written one time, and not one time for each change that
        # is heard. The transition is one-way, and a record that was appended
        # at every later change would bury the log that it explains.
        await anyio.sleep(0.2)
        assert len(_notices(first)) == 1


async def test_the_roots_keep_working_when_the_socket_goes(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """This is detection, not shutdown. The inversion that P5-01 makes holds here
    too.

    The task of a root holds no reference to a connection, so to lose the socket
    is not to lose the work. If a socket problem ended an hour of work in
    flight, that would be the failure that this row guards against, arriving
    from the other side.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        supervisor = daemon.running.supervisor
        root = await supervisor.start("working")
        shutil.rmtree(tmp_path / "xdg")
        assert await daemon.running.check_reachable() == "removed"

        await supervisor.prompt("working", "still there?")

        def answered() -> bool:
            return any(event.type == "assistant/message" for event in root.session.events_from(0))

        # The poll is on the answer and not on `status`. `status` used to read
        # `idle` in the window between the queueing of the prompt and the wake
        # of the task of the root. A wait on `status` then passed before the
        # turn that it waited for had started. (`Root.status` now reports
        # `running` while the inbox holds work.)
        await until(answered, what="the root's answer")


async def test_a_second_daemons_socket_is_not_mistaken_for_a_recovery(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """The case that an existence check gets wrong, which is also the worse
    case.

    The user logs back in and runs the `phern daemon` that a client just
    recommended. The path then exists and answers again, while the first daemon
    still holds every lease that the second one is about to be refused. That is
    the hazard of I-5, reached by a route that P5-03 does not watch. The
    identity is therefore a `(dev, inode)` pair.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        await daemon.running.supervisor.start("held")
        assert await daemon.running.check_reachable() is None, "its own socket, unchanged"

        socket.unlink()
        socket.touch()  # logind remade the directory; somebody remade the socket
        assert await daemon.running.check_reachable() == "replaced"
        assert _notices(daemon.running.supervisor.roots["held"])[0].data["reason"] == "replaced"


async def test_the_record_is_on_disk_before_anyone_could_read_it(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """The record is flushed. That is not the usual requirement here, and it is
    the purpose of this test.

    Every other record survives a crash because the log is written on the way
    out. This record is written exactly when the way out is no longer reliable.
    `phern agents shutdown` has no socket to connect to, so the next move of
    the user is often `kill`, and an unflushed record explains nothing to
    anyone.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        await daemon.running.supervisor.start("durable")
        shutil.rmtree(tmp_path / "xdg")
        await daemon.running.check_reachable()

        stored = stored_log(tmp_path / "sessions", "durable").read_text(encoding="utf-8")
        assert recovery.UNREACHABLE in stored


async def test_daemon_status_says_it_cannot_be_reached_and_what_would_fix_it(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """The report comes from the running process, for the client that is already
    attached.

    A connection that was accepted before the path went away outlives the path,
    so there is a reader for this report. After somebody restores the path by
    hand, there are more readers. The lifetime rows belong to the daemon itself.
    They are asked of the socket that the daemon bound, and not of whatever
    `$PH_RUNTIME` derives now.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        healthy = daemon.running.status()
        assert healthy.unreachable_since is None
        # One encoding, not three. The reply used to carry `survivesLogout` and
        # `linger` beside the rendered rows. Nothing except this assertion read
        # them, and a second spelling of one fact can disagree with the first.
        #
        # The section is selected by title, and the whole list is not asserted.
        # The envelope holds the sections of the *daemon*, and P5-12 added a
        # second section just after this row landed. A test that enumerates a
        # list that it does not own fails when other rows make correct changes.
        lifetime_section = next(one for one in healthy.sections if one.title == "socket lifetime")
        rows = {row.label: row.value for row in lifetime_section.rows}
        assert rows["survives logout"].startswith("no —"), "reaped host, no lingering"
        assert "off for someone" in rows["linger"]
        assert rows["enable it"] == "loginctl enable-linger someone"

        shutil.rmtree(tmp_path / "xdg")
        await daemon.running.check_reachable()
        assert isinstance(daemon.running.status().unreachable_since, int)


async def test_a_hand_built_server_with_no_bound_socket_watches_nothing(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """No identity means that there is no moment to compare against. It does not
    mean "everything is gone".

    `serve` takes the pair immediately after the bind. That is the one instant
    at which "the socket at this path" and "the socket this daemon is listening
    on" are the same file by construction. A `DaemonServer` that is built
    without an identity, by a test or by a future caller, has no transition to
    find. To report one would invent evidence.
    """
    reaped_host()
    async with anyio.create_task_group() as tasks:
        built = server.DaemonServer(
            supervisor=Supervisor(profile=PROFILE, tasks=tasks),
            stop=anyio.Event(),
            path=tmp_path / "nowhere.sock",
        )
        assert built.identity is None
        assert await built.check_reachable() is None
        assert built.status().unreachable_since is None
        tasks.cancel_scope.cancel()


def _socket_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A plain file where the daemon socket of this test would be, in its
    `$PH_RUNTIME`.

    The watch is about a name in a directory, so a plain file stands in for the
    socket. What `serve` takes as `identity` is an `lstat` of whatever is
    there.
    """
    path = private_runtime(tmp_path, monkeypatch) / "daemon.sock"
    path.touch()
    return path


@asynccontextmanager
async def _watched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[tuple[server.DaemonServer, Path]]:
    """A daemon watching its socket by event, with no socket (P12-05)."""
    path = _socket_file(tmp_path, monkeypatch)
    async with unplugged(tmp_path, monkeypatch, path=path) as daemon:
        daemon.identity = socket_identity(path)
        watch = EntryWatch(path)
        daemon.supervisor.tasks.start_soon(daemon.keep_watching, watch)
        try:
            yield daemon, path
        finally:
            watch.close()


def _replaced(path: Path) -> None:
    """Move a new file over the path in one step, as a daemon that binds again
    leaves it. There is one event, and the entry that it names is already the
    new one."""
    fresh = path.with_name("fresh")
    fresh.touch()
    os.replace(fresh, path)


@pytest.mark.parametrize(
    ("change", "reason"),
    [(Path.unlink, "removed"), (_replaced, "replaced")],
    ids=["removed", "replaced"],
)
async def test_the_watch_hears_its_socket_taken(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: Callable[[Path], object],
    reason: str,
) -> None:
    """Without a cadence: the record follows the change, not the next tick.

    Sabotage: stop `keep_watching` asking after a change, and nothing is recorded.
    """
    async with _watched(tmp_path, monkeypatch) as (daemon, path):
        root = await daemon.supervisor.start("watched")
        change(path)

        await until(lambda: bool(_notices(root)), what="the change to be recorded")
        assert _notices(root)[0].data["reason"] == reason


async def test_the_watch_asks_nothing_while_nothing_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is one `lstat` at the start, and none after it while the socket does
    not move, however busy the directory around the socket is.

    Sabotage: put a cadence back in `keep_watching`, and the count grows.
    """
    asked: list[Path] = []
    monkeypatch.setattr(
        server, "socket_identity", lambda path: noted(asked, path, socket_identity(path))
    )
    async with _watched(tmp_path, monkeypatch) as (daemon, path):
        await until(lambda: len(asked) == 1, what="the opening check")
        other = path.with_name(JOURNAL_NAME)
        for _ in range(3):
            other.touch()
            other.unlink()
        await anyio.sleep(0.3)
        if sys.platform == "linux":
            # `NOTE_WRITE` of kqueue on a directory does not name the entry. On
            # macOS each of those changes therefore costs one `lstat`, which is
            # still not a cadence.
            assert len(asked) == 1, f"{len(asked)} checks with the socket untouched"
        assert daemon.unreachable_since is None


async def test_a_failing_check_does_not_end_the_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that raises is logged and the watch continues, as with a planner
    pass that fails. If the failure were raised, it would end the watch and the
    task group of the daemon with it.

    Sabotage: drop the `except` from `DaemonServer._asked`, and the task group
    fails.
    """
    calls: list[Path] = []

    def flaky(path: Path) -> tuple[int, int] | None:
        calls.append(path)
        if len(calls) == 1:
            raise RuntimeError("a bad lstat")
        return socket_identity(path)

    monkeypatch.setattr(server, "socket_identity", flaky)
    async with _watched(tmp_path, monkeypatch) as (daemon, path):
        root = await daemon.supervisor.start("watched")
        # At least one call, not exactly one. kqueue reports any write in the
        # directory of the socket, and the daemon writes there
        # (`processes.jsonl`) as the root mounts. On macOS the watch has
        # therefore usually asked again by now.
        await until(lambda: len(calls) >= 1, what="the opening check to fail")
        path.unlink()

        await until(lambda: bool(_notices(root)), what="the removal to be recorded anyway")


async def test_an_orderly_shutdown_records_no_unreachable(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """Teardown unlinks the socket of the daemon itself. That is the daemon
    closing its own socket, not losing it. Without this, every root would carry
    a false `supervisor/unreachable` from every ordinary shutdown.

    The daemon is shut down in the way that a person does it, so that teardown
    runs in the order of `serve` itself.

    Sabotage: move `watching.cancel()` after the unlink in `serve`, and this
    records.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        root = await daemon.running.supervisor.start("calm")
        client = await daemon.client()
        await client.notify(verbs.SHUTDOWN, NoParams())
        await until(lambda: not socket.exists(), what="teardown to unlink the socket")
        await anyio.sleep(0.2)
    assert _notices(root) == []


async def test_a_watch_that_could_not_be_armed_is_checked_at_each_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no cadence to fall back on. The request of a connected client is
    the check, because new clients cannot reach a daemon that lost its path.

    Sabotage: drop the `check_reachable` from `check_unwatched`, and a socket
    that is gone stays unreported.
    """
    path = _socket_file(tmp_path, monkeypatch)
    async with unplugged(tmp_path, monkeypatch, path=path, watch_refused="no inotify") as daemon:
        daemon.identity = socket_identity(path)
        path.unlink()

        await daemon.check_unwatched()

        assert daemon.status().socket_watch == "unavailable: no inotify"
        assert daemon.unreachable_since is not None


# --- P7-19: a root's lifetime is the daemon's --------------------------------


async def test_a_client_that_leaves_mid_mount_still_gets_a_whole_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A disconnect must not cancel a mount, because it cancels the teardown too.

    The docstring of `Supervisor.start` gives the argument. This test checks the
    outcome. The client leaves in the middle of the build, and the daemon still
    ends up with a complete root, with its lease. It does not end up with a
    half-taken root that nothing released.

    The mount is held open at the one instant that matters, and is not raced
    against a clock. The test is about where the cancellation lands, and a
    sleep would be the same test with a timer in the way.
    """
    entered, release = anyio.Event(), anyio.Event()

    @asynccontextmanager
    async def slow_mount(
        profile: Profile, *, project: Path | None = None
    ) -> AsyncIterator[Context]:
        """`runtime.mounted`, held open at the instant the root is half-built."""
        async with mounted(profile, project=project) as ctx:
            entered.set()
            await release.wait()
            yield ctx

    async with supervised(tmp_path, monkeypatch) as supervisor:
        monkeypatch.setattr(runtime_module, "mounted", slow_mount)
        async with anyio.create_task_group() as client:
            client.start_soon(supervisor.start, "left-early")
            await entered.wait()
            # The socket closed. That is not the concern of this root.
            client.cancel_scope.cancel()
            release.set()

        # The caller is gone and the mount is not. The mount belongs to the task
        # group of the supervisor, so it finishes by itself.
        await until(lambda: "left-early" in supervisor.roots, what="the mount to finish")
        root = supervisor.roots["left-early"]
        assert root.ctx.active, "the mount finished rather than being abandoned halfway"
        assert root.session.id == "left-early"


async def test_a_mount_that_fails_leaves_no_root_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-built root must not be in the table, and its id must stay free.

    The failure is placed at `resume_children`, which is late in the build: the
    profile is mounted, the lease is taken and the agent exists. The root used
    to go into `self.roots` before `resume_children`. An unwind that did not
    take the entry back out then left the fast path of `start` handing out a
    root with no task behind it, forever. The table now holds only complete
    roots (D5): `_start` puts a root in the table as its last step, after its
    children are readmitted.

    `reached` stops this test passing for the wrong reason. It shows that the
    build got as far as `resume_children`, and did not fail earlier.
    """
    reached: list[str] = []

    async def boom(self: object, parent: object, *, retry_limit: int) -> Sequence[str]:
        reached.append("resume_children")
        raise RuntimeError("the children could not be readmitted")

    async with supervised(tmp_path, monkeypatch) as supervisor:
        monkeypatch.setattr(SubagentService, "resume_children", boom)
        with pytest.raises(RuntimeError, match="could not be readmitted"):
            await supervisor.start("broken")

        assert reached == ["resume_children"], "the mount did not get as far as its children"
        assert supervisor.roots == {}, "the half-built root did not stay in the table"
        # The id is also free. The lease went back with the context that it
        # was taken on.
        monkeypatch.undo()
        root = await supervisor.start("broken")
        assert root.ctx.active


@pytest.mark.parametrize("asked", ["in-flight", None], ids=["named", "new"])
async def test_shutdown_waits_for_a_mount_in_flight_and_admits_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, asked: str | None
) -> None:
    """`aclose` releases `self.roots`, and a mount in flight is by definition not
    in that table yet.

    There are two windows. Both exist because the mount is a task of the
    supervisor and not of its caller:

    - A root that finishes *after* the release loop holds a session that nothing
      flushes and a worktree that nothing reclaims (F6). Teardown exists to
      prevent exactly that.
    - A handler outlives the start of teardown. A client can therefore ask for
      a root while the roots that are already built are being unwound, and that
      root is work that nothing would ever release.

    The mount of a new root is waited for too, although it has no id to be
    found by until it opens.
    """
    entered, release = anyio.Event(), anyio.Event()

    @asynccontextmanager
    async def slow_mount(
        profile: Profile, *, project: Path | None = None
    ) -> AsyncIterator[Context]:
        async with mounted(profile, project=project) as ctx:
            entered.set()
            await release.wait()
            yield ctx

    private_runtime(tmp_path, monkeypatch)
    async with anyio.create_task_group() as tasks:
        supervisor = Supervisor(profile=PROFILE, tasks=tasks)
        monkeypatch.setattr(runtime_module, "mounted", slow_mount)
        async with anyio.create_task_group() as client:
            client.start_soon(supervisor.start, asked)
            await entered.wait()
            client.cancel_scope.cancel()

        async with anyio.create_task_group() as closing:
            closing.start_soon(supervisor.aclose)
            await until(lambda: supervisor._closing, what="aclose to take charge")
            # A root that is asked for while the daemon stops is refused, not
            # built.
            with pytest.raises(RootStartAbandoned, match="shutting down"):
                await supervisor.start("too-late")
            release.set()

        assert supervisor.roots == {}, "the mount that finished under aclose was released"
        assert not supervisor._mounting and not supervisor._in_flight
        tasks.cancel_scope.cancel()


async def test_a_resumed_root_runs_the_work_its_log_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt that was acked and flushed before the last process died, and
    never claimed. `Inbox` replays it, and `status` reports it as `running`.
    Before this fix, nothing rang for it. The root then stayed "running" with
    nothing running until the next prompt or tick, which ran both as one turn.

    Sabotage: drop the `has_pending` ring from `_start`, and the turn never
    comes.
    """
    async with supervised(tmp_path, monkeypatch) as first:
        root = await first.start("owed")
        # Logged, not rung: what `prompt` does up to the line the old process
        # never reached.
        root.agent.followup(prompt_message("still owed"))
        await first._flush(root)
        assert root.status == "running", "the inbox is work in hand"

    async with supervised(tmp_path, monkeypatch) as second:
        root = await second.start("owed")
        await until(
            lambda: any(e.type == "assistant/message" for e in root.session.events_from(0)),
            what="the owed turn to run on resume",
        )
        assert root.status == "idle"


# ------------------------------------------------------- P6-23: rehydration --


async def test_a_daemon_wakes_a_root_whose_schedule_came_due_while_it_was_down(
    tmp_path: Path,
) -> None:
    """**The gate of P6-23, and the reversal of a non-guarantee.**

    P5-06 argues that a schedule outlives the process that holds it. For the
    *log* that was always correct: `schedule/created` is still there. What did
    not outlive the process was the reader of the log. `tick` iterates
    `self.roots`, and a boot has none, so the appointment survived and was
    never kept. The failure was silent: no error and no log, found only when
    somebody noticed a run that did not happen. `test_non_guarantees.py`
    asserted that silence verbatim until this landed.

    Both halves are driven at a simulated `now`, because the subject is the
    *window*. A real boot one second later has nothing due yet. For that reason
    the old assertion continued to pass after rehydration was wired in.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        root = await first.running.supervisor.start("appointed")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first.running.supervisor._flush(root)

    async with running(tmp_path, name="second") as second:
        supervisor = second.running.supervisor
        assert list(supervisor.roots) == [], "nothing is mounted until something is due"

        fired = await supervisor.wake_and_tick(now=made + 600_000)

        assert fired == ["s"], "the appointment was kept without a client asking"
        assert "appointed" in supervisor.roots, "and its root is mounted to keep it"


async def test_a_session_with_no_appointment_is_left_alone(tmp_path: Path) -> None:
    """**The half that makes the fix safe**, and the reason why this is an index.

    `Supervisor.start` takes the lease of P5-03. To "mount every stored session
    at boot" would therefore claim every session on the machine, and refuse the
    next `phern -p` over any of them with `session_already_active`. That
    refusal is loud and immediate, and it hits sessions that have no schedule
    at all. It is strictly worse than the silence that it would fix. So only
    what the index names is woken.

    Both sessions are in one daemon on purpose. A test that showed only the
    plain session staying asleep would pass equally well against a daemon that
    wakes nothing, which is the state that this row replaces.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        plain = await first.running.supervisor.start("no-appointment")
        await first.running.supervisor._flush(plain)
        root = await first.running.supervisor.start("appointed")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first.running.supervisor._flush(root)

    async with running(tmp_path, name="second") as second:
        supervisor = second.running.supervisor

        await supervisor.wake_and_tick(now=made + 600_000)

        assert "appointed" in supervisor.roots, "the mechanism is live"
        assert "no-appointment" not in supervisor.roots, "and it woke only what is due"

        # The untouched session can still be opened, which is what unleased
        # means.
        reopened = await supervisor.start("no-appointment")
        assert reopened.id == "no-appointment"


async def _lost_appointment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Profile | None = None
) -> int:
    """Leave the interval schedule of a root written to its log, and an index
    that lost it. Returns when the schedule was made, so that a pass can be run
    after it is due."""
    async with supervised(tmp_path, monkeypatch, profile=profile) as first:
        root = await first.start("appointed")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first._flush(root)
    (tmp_path / INDEX_NAME).write_text("{ not json", encoding="utf-8")
    return made


async def test_a_daemon_rebuilds_an_index_it_cannot_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S18: an index that lost its contents is rebuilt from the logs before the
    pass reads it. It is not left to correct itself one session at a time, as
    each session is opened. For a session that nobody opens, that would be the
    silence that P6-23 closed.

    With nothing mounted, the read is a guess. What it finds is therefore woken
    and not vouched for. The root that wakes brings the store that can vouch.

    Sabotage: drop the rebuild from `rehydrate`, and the appointment never
    fires.
    """
    made = await _lost_appointment(tmp_path, monkeypatch)

    async with supervised(tmp_path, monkeypatch) as second:
        fired = await second.wake_and_tick(now=made + 600_000)

        assert fired == ["s"], "the logs said what the file had lost"
        assert not ScheduleIndex(tmp_path).survey().trusted, "a guess vouches for nothing"
        await second.wake_and_tick(now=made + 600_000)
        assert ScheduleIndex(tmp_path).survey().trusted, "the woken root's store did"


async def test_a_daemon_rebuilds_through_its_roots_store_whatever_its_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rebuild of S18 read JSONL files under one directory. A deployment
    that keeps its logs in Turso therefore had none of them read. The index was
    also marked complete, so an appointment that it had lost stayed lost.

    With nothing mounted, the daemon can only guess, and the guess finds
    nothing. When a root mounts, the rebuild reads through the store of that
    root, whatever its kind.

    Sabotage: have `_rebuild` read the guessed directory with a root mounted
    too, and the lost appointment never fires.
    """
    # The JSONL row of Base is disabled and the Turso row is inserted, as a
    # profile swaps them.
    turso: list[JsonValue] = [
        {"id": "session-persistence", "disabled": True},
        {"insert": [{"id": "session-persistence-turso", "name": "session-persistence-turso"}]},
    ]
    profile = Profile.from_documents(
        [*load_profile_documents([BASE, HEADLESS]), ProfileDocument("turso", turso)]
    )
    made = await _lost_appointment(tmp_path, monkeypatch, profile)

    async with supervised(tmp_path, monkeypatch, profile=profile) as second:
        stamp = made + 600_000
        assert await second.wake_and_tick(now=stamp) == []
        assert not ScheduleIndex(tmp_path).survey().trusted, "a guess vouches for nothing"

        await second.start("bystander")
        assert await second.wake_and_tick(now=stamp) == ["s"], "the store said what was lost"
        assert ScheduleIndex(tmp_path).survey().trusted


async def test_a_rebuild_that_failed_waits_for_a_change_not_a_timer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild that failed, for example on a read-only `$PH_HOME`, fails in the
    same way on the next pass. The daemon used to try every minute for as long
    as the failure lasted. Now it tries again only when something that the
    rebuild depends on has moved: a schedule is made or canceled, a schedule
    comes due, or a store becomes available to read through.

    Sabotage: have `_rebuild` try whenever the index is in doubt, and the second
    pass tries again with nothing changed.
    """
    attempts: list[int] = []
    monkeypatch.setattr(
        supervisor_module,
        "rebuild_index",
        lambda *_args, **_kwargs: noting(attempts, 1, raising(OSError("read-only file system"))),
    )
    async with supervised(tmp_path, monkeypatch) as supervisor:
        (tmp_path / INDEX_NAME).write_text("{ not json", encoding="utf-8")
        await supervisor.wake_and_tick(now=1)
        await supervisor.wake_and_tick(now=2)
        assert len(attempts) == 1, "nothing changed, so nothing was tried again"

        supervisor.notice_schedules()
        await supervisor.wake_and_tick(now=3)
        assert len(attempts) == 2, "a change is a reason to try again"


async def test_an_appointment_the_pass_left_does_not_hold_the_sleep_at_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An appointment that was due by the last pass was attempted then. An
    appointment that the pass declined (`wake_within`) or could not mount is
    still overdue in the index. If the plan used the index as it stands, the
    overdue entry would wake the scheduler immediately, every time, which is a
    busy loop. The entry waits for the next wake or change.

    Sabotage: drop the last-pass filter from `next_wake`, and the overdue entry
    is the next wake.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        supervisor.wake_within = 1.0
        ScheduleIndex(tmp_path).record("abandoned", next_at=5, now=0)

        assert await supervisor.wake_and_tick(now=100_000) == []

        assert "abandoned" not in supervisor.roots, "declined: confirmed too long ago"
        assert supervisor.next_wake(now=100_000) is None


async def _kept_appointment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """Leave the interval schedule of a root written to its log and to the index,
    with the root released. This is what a second daemon finds at boot. Returns
    when the schedule was made."""
    async with supervised(tmp_path, monkeypatch) as first:
        root = await first.start("appointed")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first._flush(root)
    return made


def _mount_refusing(monkeypatch: pytest.MonkeyPatch, *, times: int | None) -> list[int]:
    """Replace `mounted` with one that refuses the first `times` mounts (every
    mount, for `None`), in the way that a lease held by another process refuses
    them. After that it mounts as usual. Returns the list that each attempt
    appends to."""
    attempts: list[int] = []
    real = runtime_module.mounted

    @asynccontextmanager
    async def refusing(profile: Profile, *, project: Path | None = None) -> AsyncIterator[Context]:
        attempts.append(1)
        if times is None or len(attempts) <= times:
            raise OSError("the session's log is held by another process")
        async with real(profile, project=project) as ctx:
            yield ctx

    monkeypatch.setattr(runtime_module, "mounted", refusing)
    return attempts


async def test_a_session_that_would_not_mount_for_its_appointment_is_woken_later(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wake that the pass could not make is planned again. The retry mounts
    the root with nothing else happening: no client, no schedule change, no
    other wake.

    The pass drops the moments that it already attempted (`next_wake`). That is
    correct for an appointment that it fired or declined, and wrong for one
    that it could not reach. A `phern -p` that held the lease of the session at
    the appointed minute cost the appointment for as long as the daemon stayed
    quiet, while `booked()` kept the daemon up for it.

    Sabotage: drop `_wake_retry` from `next_wake`, and the retry never comes.
    Drop the attempt count assertion's floor (`PASS_FLOOR`) and a cadence would
    pass it.
    """
    await _kept_appointment(tmp_path, monkeypatch)
    attempts = _mount_refusing(monkeypatch, times=1)
    async with supervised(tmp_path, monkeypatch) as second:
        second.wake_retry_delays = (0.2,)
        second.tasks.start_soon(second.scheduler.keep, anyio.Event())
        await until(lambda: len(attempts) == 1, what="the first wake to be refused")
        assert "appointed" not in second.roots
        await until(
            lambda: second.scheduler.planned is not None,
            what="the refused wake to be planned again",
        )

        await until(lambda: "appointed" in second.roots, what="the retry to mount the root")
        assert attempts == [1, 1], "one retry, not a cadence"
        assert second.next_wake(now=now_ms()) is not None, "its schedule is planned from its log"
        assert second._wake_retry is None, "and the retry is dropped once the wake is made"


async def test_a_refused_wake_backs_off_and_stops_growing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each pass that leaves an appointment behind plans the next pass one rung
    further up `wake_retry_delays`, and the last rung repeats. A declined
    appointment is not a failure, so it plans no retry.

    Sabotage: plan every retry from the first rung, and the second pass plans
    `+2 s` where `+5 s` is asserted.
    """
    made = await _kept_appointment(tmp_path, monkeypatch)
    _mount_refusing(monkeypatch, times=None)
    async with supervised(tmp_path, monkeypatch) as second:
        second.wake_retry_delays = (2.0, 5.0)
        # The stamp is well past due. The pass therefore reaches the mount,
        # and does not find that nothing is due yet.
        stamp = made + 600_000

        await second.wake_and_tick(now=stamp)
        assert second.next_wake(now=stamp) == stamp + 2_000
        await second.wake_and_tick(now=stamp + 2_000)
        assert second.next_wake(now=stamp + 2_000) == stamp + 7_000
        await second.wake_and_tick(now=stamp + 7_000)
        assert second.next_wake(now=stamp + 7_000) == stamp + 12_000, "the last rung repeats"

        # Too stale to wake is a decision, not a failure: nothing to try again.
        second.wake_within = 0.001
        await second.wake_and_tick(now=stamp + 12_000)
        assert second.next_wake(now=stamp + 12_000) is None


async def test_no_pass_comes_sooner_than_the_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no tick, nothing else limits how often the scheduler runs. An
    interval of a millisecond would wake it a thousand times a second.

    Sabotage: drop `PASS_FLOOR` from `next_wake`, and the next wake is a
    millisecond away.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("eager")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1", prompt="again")
        )
        stamp = now_ms()
        await supervisor.wake_and_tick(now=stamp)

        planned = supervisor.next_wake(now=stamp)
        assert planned is not None and planned >= stamp + PASS_FLOOR * 1000


async def test_the_scheduler_sleeps_until_something_is_due(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """There is no tick. With nothing scheduled, the scheduler has no deadline.
    A schedule that is made on a root wakes the scheduler, through the log of
    the root. The scheduler then sleeps until that schedule is due, and fires
    it.

    **The wait is for that moment on the wall clock** (P12-00), and not a delay
    on the monotonic clock. The monotonic clock does not count a suspend, so a
    scheduler that slept through a closed lid woke late by the time that the
    machine had slept. The recorder checks the hand-off to `ph.wall_clock`, and
    `test_wall_clock` holds `ph.wall_clock` to `CLOCK_REALTIME`.

    Sabotage: drop the schedule watch from `_start`, and the sleeper never
    learns of the appointment. Put back
    `move_on_after((planned - now_ms()) / 1000)`, and nothing is armed on the
    wall clock.
    """
    armed: list[int] = []
    wall_sleep = wall_clock.sleep_until

    async def record(at: int) -> None:
        armed.append(at)
        await wall_sleep(at)

    monkeypatch.setattr(wall_clock, "sleep_until", record)
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("timed")
        assert supervisor.next_wake(now=now_ms()) is None, "nothing scheduled, nothing to wake for"
        supervisor.tasks.start_soon(supervisor.scheduler.keep, anyio.Event())
        await until(lambda: supervisor._last_pass > 0, what="the first pass")
        assert supervisor.scheduler.planned is None, "asleep until told"

        at = now_ms() + 300
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="soon", kind="once", spec=str(at), prompt="go")
        )
        await until(lambda: supervisor.scheduler.planned is not None, what="the plan to move")
        assert not_none(supervisor.scheduler.planned) >= at
        await until(
            lambda: armed[-1:] == [supervisor.scheduler.planned],
            what="the wake to be armed at the planned instant",
        )

        await until(
            lambda: any(e.type == "schedule/tick" for e in root.session.events_from(0)),
            what="the appointment to fire",
        )


async def test_catch_up_is_unbounded_by_default(tmp_path: Path) -> None:
    """**The complete promise of the scheduler, and its complete scope.**

    pH keeps an appointment while `phern daemon` runs. When the daemon starts,
    it continues from where it stopped, however long it was down, and `claim`
    coalesces the gap to one run for each missed window. A default bound on
    that would be a second policy on top of the one that P5-06 already settled.
    The OS already ships cron, anacron and systemd timers for anyone who wants
    a run to happen with no daemon at all.

    A year is well past any bound that a default could reasonably have had.
    This test is therefore an assertion that there is *no* bound.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        root = await first.running.supervisor.start("long-gone")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first.running.supervisor._flush(root)

    async with running(tmp_path, name="second") as second:
        supervisor = second.running.supervisor
        assert supervisor.wake_within is None, "the shipped default is to catch up"

        a_year_later = made + 365 * 24 * 60 * 60 * 1000

        assert await supervisor.wake_and_tick(now=a_year_later) == ["s"]


async def test_a_deployment_can_bound_how_stale_an_appointment_may_be(
    tmp_path: Path,
) -> None:
    """The setting for a deployment that does not want to bring back a session
    that somebody abandoned. The OS tools do not have this case, because a
    schedule here is attached to a *conversation* and not to a crontab entry.

    The bound is off by default, and this test shows what it does when it is
    on. The test then lifts the bound again on the same daemon, which gives the
    test its meaning. It proves that the appointment was due and wakeable the
    whole time, and that the *bound* declined it.
    """
    socket = tmp_path / "first.sock"
    async with running(tmp_path, path=socket) as first:
        root = await first.running.supervisor.start("abandoned")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first.running.supervisor._flush(root)

    async with running(tmp_path, name="second", wake_within=60.0) as second:
        supervisor = second.running.supervisor
        long_after = made + 600_000 + 60_000 + 1

        assert await supervisor.wake_and_tick(now=long_after) == []
        assert list(supervisor.roots) == [], "a stale appointment is not resurrected"

        supervisor.wake_within = None

        assert await supervisor.wake_and_tick(now=long_after) == ["s"]
        assert "abandoned" in supervisor.roots, "the bound declined it, not the mechanism"


# --- P11-08: a sub-agent's log is never mounted as a root --------------------


def _stored_line(*chain: str) -> dict[str, bytes]:
    """Write a root and a line of sub-agents under it, as an earlier process
    would leave them on the disk of this daemon. `chain[0]` is the root, and
    each id after it was spawned by the id before it.

    The logs have the form in which ph-rlm opens the log of a child: its
    spawner in `parentSession`, `origin: "subagent"`, and the family of the
    root, so the whole line is in one directory. Returns the bytes of each
    file, so that a test can assert that none of them changed.
    """
    sessions = resolve_roots().sessions_dir()
    family = SessionHeader(id=chain[0], created_at=1).family
    written: dict[str, bytes] = {}
    for depth, session_id in enumerate(chain):
        session = Session(
            session_id,
            header=SessionHeader(
                id=session_id,
                created_at=1,
                family=family,
                parent_session=chain[depth - 1] if depth else None,
                origin="subagent" if depth else None,
                delegation_depth=depth or None,
            ),
        )
        # One event after the header, so that a second writer has a log to
        # resume.
        log_event(session, "workspace/checkpoint", {"agentId": session_id, "tree": "t1"})
        lines: list[JsonObject] = [{"type": "session/header", "header": session.header.to_wire()}]
        lines += [event.to_wire() for event in session.events]
        path = session_path(sessions, session_id, family)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(f"{json.dumps(line)}\n" for line in lines), encoding="utf-8")
        written[session_id] = path.read_bytes()
    return written


async def test_a_subagents_log_cannot_be_attached_as_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate of P11-08: the log of a child has one writer, the mount of its
    root.**

    `session/attach` reaches `Supervisor.start`, which mounted whatever id it
    was given, including the id of a sub-agent (defect 6). The log of a
    sub-agent is written by the mount of the root that spawned it, and that
    mount readmits the sub-agent whenever the root comes up. A root of its own
    on the same file is therefore a second writer. A second writer is the one
    thing that can corrupt a log that is now the only record of the child.

    The start is refused from the header, before anything is mounted or
    claimed. Every file is therefore byte for byte what its root left. The
    refusal names the **root**, which is the one id that is worth attaching to
    instead. That holds even for a grandchild whose spawner is itself a
    sub-agent.

    Sabotage: drop the header check in `_start`, and the child mounts as a root.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        written = _stored_line("lead", "helper", "nested")

        for child in ("helper", "nested"):
            with pytest.raises(NotARoot, match=rf"{child} is a sub-agent's log.*root lead;"):
                await supervisor.start(child)

        assert supervisor.roots == {}, "a child was mounted as a root"
        sessions = resolve_roots().sessions_dir()
        family = SessionHeader(id="lead", created_at=1).family
        assert {
            session_id: session_path(sessions, session_id, family).read_bytes()
            for session_id in written
        } == written, "a refused attach wrote to the family's logs"
        # The root that the refusal names is the one attach that is still
        # served.
        assert (await supervisor.start("lead")).session.id == "lead"
