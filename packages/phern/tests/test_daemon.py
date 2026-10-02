"""P5-01 — the supervisor, and the property every earlier mode lacked.

Every mode before this ties an agent's life to a connection: `phern -p` exits with
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
no blocking to move off the loop. Its neighbors settle it — `path.is_file()` two
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
events, on every pass**.

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

A cadence that rides another's counter advances only when that one *succeeds*, so
**a run of failing passes starves an unrelated task** as a side effect of a
failure that has nothing to do with it. The socket watch has its own `if` for the mirror
reason: a test that turns the scheduler off to keep a timer out of its assertions
must not thereby turn off the thing that notices the daemon has no door.

**And why `_await` names `DaemonGone`.** It used to return an empty `{}` when woken
by the pump ending rather than by an answer, which every caller then read as a
successful reply with no fields in it.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
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
    """A tier that can put a tree back (`CheckpointingProvider`), doing what the test
    says. Beneath the seam's `restore`, which is where the record is now kept — so a
    test that replaced that method would skip the very thing it is about."""

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
                return dict(row)
            await anyio.sleep(0.01)


async def test_a_resumed_root_hands_the_child_ladder_its_own_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`resume_children` takes the limit because the policy is the host's (P6-32).

    Which makes *this* the only place the daemon's answer is stated, so it is the
    only place the wiring can be checked: the seam states no bound of its own, so
    nothing in `ph-core` would notice the daemon passing a different number, and
    the ladder's own tests supply their own.
    """
    from ph.seams.subagents import SubagentService

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
    """A child could already interrupt its parent mid-turn; a person could not.

    `rlm-messaging` delivers by steer so "a running agent picks it up without
    finishing first", while a typed line waited for the whole turn — so during a
    long fan-out the person had strictly less reach than the children they had
    spawned. Asserted through the inbox target rather than through timing, which
    is the fact and not a race.
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

        # A schedule means something else: `tick`'s contract is that a scheduled
        # turn is an *ordinary* turn, so it must not join one it has nothing to
        # do with — nor merge with another schedule due in the same pass.
        await supervisor.prompt("busy", "the appointment", reach="next-turn")

    assert targets[-1] == "next-turn", "a scheduled turn joined a running one"


async def test_a_failed_turn_is_named_beside_an_idle_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P5-04's deferral, closed without a second opinion from the supervisor.

    `status` stays the agent's — `idle` after a turn that ended in error, since
    the ladder does not count turn failures — but the agent's own `turn/end`
    rides beside it, so a client polling the list can tell "answered" from "the
    last answer was an error", which `--until-idle` could not.
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

    Filtered to the session's own frames, because `daemon.lifetime` is not one:
    it goes to every *connection* by design (P9-07) and a turn starting and
    ending moves it twice. A detach ends a subscription to a root, not the
    connection — which is the distinction the filter is asserting.
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

    Refusing would strand a client that did nothing wrong; honoring it would
    skip events it never saw. So a stale generation reads as "you have seen
    nothing of *this* log" — and the reply names the generation it *does* count
    against, which is what a client pages from rather than from the cursor it
    sent.
    """
    async with running(tmp_path) as daemon:
        client = await daemon.client()
        await client.call("session/prompt", sessionId="eta", prompt="first")
        settled = await _settled(client, "eta", events=1)
        stale = {"generation": "not-this-log", "sequence": settled["cursor"]["sequence"]}

        history = await _history(client, "eta", stale)
        # Attach takes no cursor — it does not replay, and catch-up is the
        # paged read above from the point this reply names.
        attached = await client.call("session/attach", sessionId="eta")

        assert len(history) == settled["cursor"]["sequence"], "a stale cursor skipped events"
        assert attached["cursor"]["generation"] != stale["generation"], "the reply names this log"
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

        assert first.session_id == "kappa"
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
        # starting over, which is the P5-01 behavior the lease must not break.
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
    and the test went on passing with the ordering deleted. So the first caller
    is now parked *inside* `_start`, past the membership check, and the second is
    released only once it is there: unordered, the second must build a second
    root, and there is no timing left for luck to supply.

    What does the ordering is `_mounting` rather than a lock a caller holds —
    `start` hands the build to the supervisor's task group and waits on the
    record, so the second caller finds the first one's entry. `builds` is the
    half that says so: the assertions below would also pass if the second caller
    built its own root over the same log.
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

        # On the class: `Supervisor` is a `slots=True` dataclass, so an instance
        # attribute cannot be shadowed.
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
        assert builds == ["contested"], "two callers, one mount"


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


def _crash(patch: pytest.MonkeyPatch, root: Any, times: int) -> None:  # noqa: ANN401
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

    async def run(self: object) -> None:
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
        root = daemon.running.supervisor.roots["recovers"]
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
        client = await daemon.client(on_notify=lambda method, params: seen.append(params))
        await client.call("session/new", sessionId="doomed")
        await client.call("session/attach", sessionId="doomed")
        root = daemon.running.supervisor.roots["doomed"]
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
        assert "injected crash" in str(given_up.data["reason"])
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
        # behavior. Waited for rather than read once: `status` flips when the
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
            root = daemon.running.supervisor.roots["stubborn"]
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
        root = daemon.running.supervisor.roots["stubborn"]
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
        casualty = daemon.running.supervisor.roots["casualty"]
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

        # The mounted **tier**, not a module import: the ladder asks the seam, so
        # patching a name in this module would have kept passing while the call it
        # stands for went somewhere else.
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
    """The ordinary case: an advisory-tier root has no worktree to put back.

    No restore is opened at all, rather than one that names nothing — a
    transcript that implied a rollback nobody performed would misread the
    attempt that follows.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("advisory")
        assert supervisor._restore_point(root) == (None, "")


async def test_a_retry_is_on_disk_before_its_restore_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, short_ladder: None
) -> None:
    """S12: the restore is part of the attempt, and it used to run unrecorded.

    It ran first and the retry was written after, so a daemon that died while
    restoring had rewritten the tree with no record of the attempt — and a crash
    loop there never advanced the count. The retry reaches disk first, then the
    restore's own opening record (`WORKSPACE_RESTORE`); a restore that fails is
    settled `ok: false`, so the transcript claims no rollback.

    Sabotage: restore before recording again, and the restore finds neither record
    on disk.
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
        root = daemon.running.supervisor.roots["unflushable"]

        async def broken(self: object, session: Session) -> None:
            raise RuntimeError("flush is broken")

        monkeypatch.setattr(type(root.ctx.require(SESSIONS)), "flush", broken)
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
        await daemon.running.supervisor.sweep(after=0)

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
        assert await daemon.running.supervisor.sweep(after=0) == []

        await client.call("session/detach", sessionId="watched")
        assert await daemon.running.supervisor.sweep(after=0) == ["watched"]


async def test_a_root_that_has_not_been_quiet_long_enough_is_not_released(
    tmp_path: Path,
) -> None:
    """The timeout is read from the log, not from a timer.

    `now` is the log's own last event here, so nothing has been quiet for any
    time at all — which is also what makes a root rehydrated from a three-day-old
    log immediately eligible, correctly.
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

    A root in P5-04's backoff is doing nothing between attempts. Releasing it
    there would passivate a root part-way up a ladder it had already recorded —
    and the condition that catches it is the one the cleanup pass added when it
    found `retrying` announced as a notification while `sessions/list` still
    said `idle`.
    """
    async with running(tmp_path) as daemon:
        supervisor = daemon.running.supervisor
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

        await first.running.supervisor.sweep(after=0)
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
        await daemon.running.supervisor.sweep(after=0)
        assert "awaited" not in daemon.running.supervisor.roots

        attached = await client.call("session/attach", sessionId="awaited")
        assert attached["sessionId"] == "awaited"
        assert "awaited" in daemon.running.supervisor.roots


async def test_release_actually_runs(tmp_path: Path) -> None:
    """The planner `serve` starts, not just the predicate.

    Everything above drives `sweep()` directly, which would pass just as well if
    nothing ever called it — the shape of dead code that looks tested.
    """
    async with running(tmp_path, passivate_after=0.0) as daemon:
        client = await daemon.client()
        # Attached first, and that is the test's own setup rather than an
        # accident: with `passivate_after=0` the root is releasable the instant
        # it is idle and unwatched, which is *during* the turn we are waiting on.
        # Attaching is the same condition a real client relies on to keep a
        # session it is using.
        await client.call("session/new", sessionId="swept")
        await client.call("session/attach", sessionId="swept")
        await client.prompt("swept", "hello")
        await _settled(client, "swept", events=1)
        await client.call("session/detach", sessionId="swept")
        await _until(
            lambda: "swept" not in daemon.running.supervisor.roots,
            what="an idle root to be released on its own",
        )


@asynccontextmanager
async def _releasing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, window: float = 0.2
) -> AsyncIterator[Supervisor]:
    """A supervisor whose releaser is running, with a short quiet window (P12-01).

    A window above zero rather than at it, so a release lands after the turn or
    pass that made the root quiet has finished, and a test asserting "still
    mounted" can wait past the window and mean it.
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
    """Wait past the quiet window, and the floor after the last pass, and say the
    root is still there: what holds it is holding it."""
    await anyio.sleep(not_none(supervisor.passivate_after) + PASS_FLOOR + 0.2)
    assert root_id in supervisor.roots, f"{root_id} was released while held"


async def _released(supervisor: Supervisor, root_id: str) -> None:
    await until(lambda: root_id not in supervisor.roots, what=f"{root_id} to be released")


async def test_an_idle_daemon_releases_at_the_deadline_without_polling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Release sleeps until the quiet window ends: no pass in between (P12-01).

    One pass at boot, one when the root mounts, and the next is the one that
    releases it. A sweep would have run every sixty seconds whether or not anything
    could have come due.

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

    Sabotage: drop the releaser's notice from `start`, and a root that never runs a
    turn is never planned for.
    """
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        await supervisor.start("untouched")
        await _released(supervisor, "untouched")


async def test_a_turn_ending_lets_the_root_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gate: ModelGate
) -> None:
    """The commonest way a quiet window starts: a turn ends.

    The turn is held open at the model past the window, so the pass the mount
    planned finds the root `running` and plans nothing. A turn shorter than that
    needs no notice: the pass would find it over and plan from its last record.

    Sabotage: drop `recheck()` from the `agent/status` listener, and nothing wakes
    the releaser when the turn ends.
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
    """Sabotage: drop `recheck()` from `Root.unsubscribe`, and the root stays."""
    async with _releasing(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("watched")

        def watcher(_method: str, _params: dict[str, Any]) -> None:
            return None

        root.subscribe(watcher)
        await _still_mounted(supervisor, "watched")

        root.unsubscribe(watcher)
        await _released(supervisor, "watched")


async def test_a_root_parking_on_a_person_lets_it_go(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`waiting` is releasable, and the desk is what says the root reads it now.

    Held first by the ladder (`retrying` is work in hand), so the only thing that
    can make it releasable is the ask: the desk outranks the ladder in
    `Root.status`.

    Sabotage: drop the desk's `root.recheck()` when it opens an ask, and the root
    stays.
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
    """The child's record is heard on the bus, before `beneath`'s watcher guard.

    Sabotage: move the releaser's notice back under `beneath`'s `not root.subscribers`
    return, and an unwatched parent stays mounted after its child is done.
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
    """Sabotage: drop the releaser's notice from `_watch_schedules`, and the root
    stays after its last appointment is withdrawn."""
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
    """`retrying` holds a root; `failed` and a recovered `idle` do not, and neither
    moves the agent's own status, so the root says so itself.

    Sabotage: drop `recheck()` from `Root.give_up` or `Root.recovered`, and that case
    stays mounted.
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
    """A parent whose subagent is still working stays mounted — **read from the
    child's own log** (P11-07).

    Releasing it would take the child's mount with it: the child is suspended
    mid-task and comes back only when something wakes the root. A root's log holds
    no record of its children any more, so the question is asked of each child's
    log, through the seam's `child_is_live` — whose settled set is the one every
    producer writes. The first version of this test spelled that set itself as
    `{"completed", "failed", "canceled", "deleted"}`, of which only one is a
    string any producer emits, and a root that had ever run a child to completion
    could never be released.

    **A grandchild counts too**, at every level: the resume sweep has revoked what
    an ended child left unfinished, so a live one is real work
    (`test_what_an_ended_child_left_unfinished_is_revoked_when_its_root_comes_back`).

    Sabotage: read the root's own log for `subagent/*` again, and every child
    reads as absent — each root is released under a working child.
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

        # An unrecognized status keeps the parent alive rather than releasing one
        # whose child may still be running.
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
    """**The whole tree is read when its root comes back, and an ended child's
    unfinished children are ended in their own logs.**

    A child's own children are artifacts of its scope, revoked as it unwinds — but a
    crash between a child's ending and theirs leaves a grandchild `running` on disk
    beneath a child that ended. Read one level at a time, the next process never saw
    it: its spend reached no goal, the panel never listed it, and nothing ended it.
    Now the root's whole tree is read as its log opens, and the sweep revokes what
    the ended child left — so it holds nothing, and reads as what it is.

    Sabotage: skip `_revoke_beneath` for a child that had ended, and the grandchild
    is still working after the restart.
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
    """**P11-07's gate: a child's state reaches a watcher from the child's own log.**

    A root's stream carries nothing about its children any more — each child writes
    only its own log (Phase 11) — so a client watching the root would see an empty
    family. The supervisor listens to every log in the root's mount and pushes
    `session.children`, the whole family, whenever a row moves: **one frame for a
    batch**, since a revocation's `canceled` and its tombstone land together, and
    **none for a chunk**, which moves no row. The child's own events never reach the
    root's watchers.

    Sabotage: stop listening to `session/event` in `_start`, and the notice never
    comes.
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
    """`session/children` is the whole family, a parent before its own children.

    A grandchild's log names the child that spawned it, not the root, so a list
    of the root's own children would leave out every level beneath them — the
    work a fan-out delegated again. Each row carries its `parentId`, which is what
    lets a reader draw the tree in one pass.

    Sabotage: list only `children(root.id)` in `family_of`, and the grandchild is
    missing.
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
        # What `session/children` answers, through its verb's own reply model.
        wire = SessionChildrenNotice(session_id="lead", children=rows).to_wire()
        assert verbs.SESSION_CHILDREN.read(wire).children == rows


async def test_a_held_child_is_listed_as_awaiting_a_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sub-agent held for a key says so, at any depth — in the family and in the
    doctor's credential rows (T5).

    The hold is in the child's own log now, as a session waiting on its own route
    (`SESSION_HOLDER`), where it used to be a row in its parent's log. So a root's
    log answers for the root alone, and `awaited` asks every level beneath it —
    naming each by the run ids from the root down.

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
        supervisor = daemon.running.supervisor
        client = await daemon.client()
        await client.call("session/new", sessionId="cron")
        root = supervisor.roots["cron"]

        root.ctx.require(SCHEDULE).create(
            root.session,
            Schedule(id="nightly", kind="interval", spec="60000", prompt="do the thing"),
        )
        # Relative to creation: a schedule is anchored where it was made, so a
        # clock starting at zero is decades before its own schedule exists.
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
    """An *ordinary* turn is this path's contract, and a busy root must not break it.

    A person interjecting is delivered at the next step, deliberately — but a
    schedule is not "also this". Joining a turn it has nothing to do with would
    share that turn's per-turn ceilings with it, and two schedules due in one
    pass would become one turn rather than two.
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
    """P5-05's fourth condition, which that row left open for this one.

    A root that has said when it comes back is a root that is still wanted.
    Releasing it would drop the only thing that knows the appointment — and
    since passivation unwinds the `Context`, the schedule would stop being
    watched while still sitting in the log claiming it fires at nine.
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
    """Every root's schedules fire from one loop, so one of them must not end it."""
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
# The simulation is exact rather than metaphorical. The `reaped_host` fixture in
# the repo-root conftest pins `$XDG_RUNTIME_DIR` at a directory these tests own
# and puts `$PH_RUNTIME` inside it; the socket is bound there, and "the login
# session ended" is that directory being removed — which is what logind does to
# `/run/user/$UID` for a user who is not lingering. What must not happen after
# that is *nothing*: the daemon keeps running with no door, and the only reader
# left is whoever opens a transcript afterwards.


def _notices(root: Root) -> list[SessionEvent]:
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
    async with running(tmp_path, path=socket) as daemon:
        supervisor = daemon.running.supervisor
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
        # Once, not once per change heard: the transition is one-way, and a
        # record appended at every later change would bury the log it explains.
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
        supervisor = daemon.running.supervisor
        root = await supervisor.start("working")
        shutil.rmtree(tmp_path / "xdg")
        assert await daemon.running.check_reachable() == "removed"

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

    Log back in, run the `phern daemon` a client just recommended, and the path
    exists and answers again — while the first daemon still holds every lease
    the second one is about to be refused. That is I-5's hazard reached through
    a door P5-03 does not watch, so the identity is a `(dev, inode)` pair.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        await daemon.running.supervisor.start("held")
        assert await daemon.running.check_reachable() == "", "its own socket, unchanged"

        socket.unlink()
        socket.touch()  # logind remade the directory; somebody remade the socket
        assert await daemon.running.check_reachable() == "replaced"
        assert _notices(daemon.running.supervisor.roots["held"])[0].data["reason"] == "replaced"


async def test_the_record_is_on_disk_before_anyone_could_read_it(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """Flushed, which is not the usual bar here and is the point.

    Every other record survives a crash because the log is written on the way
    out. This one is written exactly when the way out has stopped being
    reliable: `phern agents shutdown` has no door to knock on, so the person's next
    move is often `kill`, and an unflushed record explains nothing to anyone.
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
    """Reported from the running process, for the client that is already attached.

    A connection accepted before the path went away outlives it, so there is a
    reader for this — and after somebody restores the path by hand there are
    more. The lifetime rows are the daemon's own, asked of the socket it bound
    rather than of whatever `$PH_RUNTIME` derives now.
    """
    socket = reaped_host() / "daemon.sock"
    async with running(tmp_path, path=socket) as daemon:
        healthy = daemon.running.status()
        assert healthy.unreachable_since is None
        # One encoding, not three. The reply used to carry `survivesLogout` and
        # `linger` beside the rendered rows; nothing but this assertion read
        # them, and a second spelling of one fact is one that can disagree.
        #
        # Selected by title rather than asserted as the whole list: the envelope
        # is the *daemon's* sections, and P5-12 added a second one the moment
        # after this row landed. A test that enumerates a list it does not own
        # fails for other rows' correct changes.
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
        assert built.status().unreachable_since is None
        tasks.cancel_scope.cancel()


def _socket_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A plain file where this test's daemon socket would be, in its `$PH_RUNTIME`.

    The watch is about a name in a directory, so a plain file stands in for the
    socket, and what `serve` captures as `identity` is an `lstat` of whatever is
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
    """A new file moved over the path in one step, as a rebinding daemon leaves it:
    one event, and the entry it names is already the new one."""
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
    """One `lstat` at the start, and none after while the socket stays put, however
    busy the directory around it.

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
            # kqueue's `NOTE_WRITE` on a directory does not name the entry, so on
            # macOS each of those changes costs one `lstat` — still no cadence.
            assert len(asked) == 1, f"{len(asked)} checks with the socket untouched"
        assert daemon.unreachable_since is None


async def test_a_failing_check_does_not_end_the_watch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A check that raises is logged and the watch goes on, as a failing planner pass
    does: raised, it would end the watch and take the daemon's task group with it.

    Sabotage: drop the `except` from `DaemonServer._asked`, and the task group fails.
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
        # At least one, not exactly one: kqueue reports any write in the socket's
        # directory, and the daemon writes there (`processes.jsonl`) as the root
        # mounts, so on macOS the watch has usually asked again by now.
        await until(lambda: len(calls) >= 1, what="the opening check to fail")
        path.unlink()

        await until(lambda: bool(_notices(root)), what="the removal to be recorded anyway")


async def test_an_orderly_shutdown_records_no_unreachable(
    tmp_path: Path, reaped_host: ReapedHost
) -> None:
    """Teardown unlinks the daemon's own socket. That is the daemon closing its door,
    not losing it, and every root would otherwise carry a false
    `supervisor/unreachable` from every ordinary shutdown.

    Shut down the way a person does it, so teardown runs in `serve`'s own order.

    Sabotage: move `watching.cancel()` after the unlink in `serve`, and this records.
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
    """No cadence to fall back on: a connected client's request is the check, since
    a daemon that lost its path is unreachable to new ones.

    Sabotage: drop the `check_reachable` from `check_unwatched`, and a socket that is
    gone stays unreported.
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

    `Supervisor.start` carries the argument; what this holds is the outcome — the
    client leaves mid-build and the daemon still ends up with a whole root, lease
    and all, rather than a half-taken one nothing released.

    The mount is held open at the one instant that matters rather than raced
    against a clock: what is under test is where the cancellation lands, and a
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
            # The socket closed. Nothing about that is this root's business.
            client.cancel_scope.cancel()
            release.set()

        # The caller is gone and the mount is not: it belongs to the supervisor's
        # task group, so it finishes on its own.
        await until(lambda: "left-early" in supervisor.roots, what="the mount to finish")
        root = supervisor.roots["left-early"]
        assert root.ctx.active, "the mount finished rather than being abandoned halfway"
        assert root.session.id == "left-early"


async def test_a_mount_that_fails_leaves_no_root_behind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A half-built root must not stay in the table, and its id must stay free.

    The failure is placed *after* the root reaches `self.roots`, which is the one
    window where the table can be left lying: the entry has to go in before
    `resume_children`, because a readmitted child starts a drive job owned by
    this root's scope and the resume looks the root up. An unwind that does not
    take it back out leaves `start`'s fast path handing out a root with no task
    behind it, forever.

    `reached` is what stops this passing for the wrong reason: a mount that
    failed earlier would never have put anything in the table to leave.
    """
    reached: list[str] = []

    async def boom(self: object, parent: object, *, retry_limit: int) -> Sequence[str]:
        reached.append("resume_children")
        raise RuntimeError("the children could not be readmitted")

    async with supervised(tmp_path, monkeypatch) as supervisor:
        monkeypatch.setattr(SubagentService, "resume_children", boom)
        with pytest.raises(RuntimeError, match="could not be readmitted"):
            await supervisor.start("broken")

        assert reached == ["resume_children"], "the mount failed after the root was in the table"
        assert supervisor.roots == {}, "the half-built root did not stay in the table"
        # And the id is free: the lease went back with the context it was taken on.
        monkeypatch.undo()
        root = await supervisor.start("broken")
        assert root.ctx.active


async def test_shutdown_waits_for_a_mount_in_flight_and_admits_no_more(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`aclose` releases `self.roots`, and a mount is by definition not in it yet.

    Two windows, both opened by the mount being the supervisor's own task rather
    than its caller's. A root finishing *after* the release loop holds a session
    nothing flushes and a worktree nothing reclaims (F6) — the things teardown
    exists for. And a handler outlives the start of teardown, so a client can ask
    for a root while the ones already built are being unwound, which is work
    nothing would ever release.
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
            client.start_soon(supervisor.start, "in-flight")
            await entered.wait()
            client.cancel_scope.cancel()

        async with anyio.create_task_group() as closing:
            closing.start_soon(supervisor.aclose)
            await until(lambda: supervisor._closing, what="aclose to take charge")
            # Asked while the daemon is going: refused rather than built.
            with pytest.raises(RootStartAbandoned, match="shutting down"):
                await supervisor.start("too-late")
            release.set()

        assert supervisor.roots == {}, "the mount that finished under aclose was released"
        assert "in-flight" not in supervisor._mounting
        tasks.cancel_scope.cancel()


async def test_a_resumed_root_runs_the_work_its_log_still_holds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A prompt acked and flushed before the last process died, and never claimed:
    `Inbox` replays it, `status` reports it as `running`, and until this nothing
    rang for it, so the root sat "running" with nothing running until the next
    prompt or tick, which then ran both as one turn.

    Sabotage: drop the `has_pending` ring from `_start`, and the turn never comes.
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
    """**The half that makes the fix safe**, and the reason this is an index.

    `Supervisor.start` takes P5-03's lease, so "mount every stored session at
    boot" would claim every session on the machine and refuse the next `phern -p`
    over any of them with `session_already_active` — loud, immediate, and hitting
    sessions that have no schedule at all. Strictly worse than the silence it
    would be fixing. So only what the index names is woken.

    Both sessions are in one daemon on purpose: a test that only showed the plain
    one staying asleep would pass just as well against a daemon that wakes
    nothing, which is the state this row replaces.
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

        # And the untouched session is still openable, which is what unleased means.
        reopened = await supervisor.start("no-appointment")
        assert reopened.id == "no-appointment"


async def _lost_appointment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, profile: Profile | None = None
) -> int:
    """A root's interval schedule, written to its log, and an index that lost it.
    When it was made, so a pass can be run after it is due."""
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
    """S18: an index that lost its contents is rebuilt from the logs before the pass
    reads it, not left to correct itself one session at a time as each is opened —
    which, for a session nobody opens, is the silence P6-23 closed.

    With nothing mounted the read is a guess, so what it finds is woken and not
    vouched for; the root that wakes brings the store that can vouch.

    Sabotage: drop the rebuild from `rehydrate`, and the appointment never fires.
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
    """S18's rebuild read JSONL files under one directory, so a deployment that keeps
    its logs in Turso had none of them read — and the index was marked complete, so
    an appointment it had lost stayed lost.

    With nothing mounted the daemon can only guess, and the guess finds nothing.
    Once a root mounts, the rebuild reads through that root's store, whatever kind.

    Sabotage: have `_rebuild` read the guessed directory with a root mounted too,
    and the lost appointment never fires.
    """
    # Base's JSONL row disabled and the Turso one inserted, as a profile swaps them.
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
    """A rebuild that failed — on a read-only `$PH_HOME`, say — fails the same way
    on the next pass, and the daemon used to try every minute for as long as it did.
    It tries again only once something it depends on moved: a schedule made or
    canceled, one come due, or a store to read through.

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
    """An appointment due by the last pass was attempted then — and one it declined
    (`wake_within`) or could not mount is still overdue in the index. Planned from as
    it stands, it would wake the scheduler at once, every time: a busy loop. It waits
    for the next wake or change instead.

    Sabotage: drop the last-pass filter from `next_wake`, and the overdue entry is
    the next wake.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        supervisor.wake_within = 1.0
        ScheduleIndex(tmp_path).record("abandoned", next_at=5, now=0)

        assert await supervisor.wake_and_tick(now=100_000) == []

        assert "abandoned" not in supervisor.roots, "declined: confirmed too long ago"
        assert supervisor.next_wake(now=100_000) is None


async def _kept_appointment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> int:
    """A root's interval schedule, written to its log and to the index, and the root
    released: what a second daemon finds at boot. When it was made."""
    async with supervised(tmp_path, monkeypatch) as first:
        root = await first.start("appointed")
        root.ctx.require(SCHEDULE).create(
            root.session, Schedule(id="s", kind="interval", spec="1000", prompt="tick")
        )
        made = root.ctx.require(SCHEDULE).states(root.session)["s"].created_at
        await first._flush(root)
    return made


def _mount_refusing(monkeypatch: pytest.MonkeyPatch, *, times: int | None) -> list[int]:
    """`mounted` that refuses the first `times` mounts (every one, for `None`) the
    way a lease another process holds refuses them, then mounts as usual. Returns
    the list each attempt appends to."""
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
    """A wake the pass could not make is planned again, and the retry mounts the root
    with nothing else happening: no client, no schedule change, no other wake.

    The pass drops moments it already attempted (`next_wake`), which is right for
    an appointment it fired or declined and wrong for one it could not reach: a
    `phern -p` holding the session's lease at the appointed minute cost the
    appointment for as long as the daemon stayed quiet, while `booked()` kept the
    daemon up for it.

    Sabotage: drop `_wake_retry` from `next_wake`, and the retry never comes. Drop
    the attempt count assertion's floor (`PASS_FLOOR`) and a cadence would pass it.
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
    """Each pass that leaves an appointment behind plans the next a rung further up
    `wake_retry_delays`; the last rung repeats; a declined appointment is not a
    failure, so it plans no retry.

    Sabotage: plan every retry from the first rung, and the second pass plans
    `+2 s` where `+5 s` is asserted.
    """
    made = await _kept_appointment(tmp_path, monkeypatch)
    _mount_refusing(monkeypatch, times=None)
    async with supervised(tmp_path, monkeypatch) as second:
        second.wake_retry_delays = (2.0, 5.0)
        # Well past due, so the pass reaches the mount rather than finding
        # nothing due yet.
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
    """With no tick, nothing else bounds how often the scheduler runs: an interval of
    a millisecond would wake it a thousand times a second.

    Sabotage: drop `PASS_FLOOR` from `next_wake`, and the next wake is a millisecond
    away.
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
    """No tick. With nothing scheduled the scheduler has no deadline, and a schedule
    made on a root wakes it, by its log, to sleep until that one is due — which it
    then fires.

    **The wait is for that moment on the wall clock** (P12-00), not a delay on the
    monotonic one, which does not count a suspend: a scheduler that slept through a
    closed lid woke late by however long the machine had slept. The recorder pins
    the hand-off to `ph.wall_clock`, which `test_wall_clock` holds to
    `CLOCK_REALTIME`.

    Sabotage: drop the schedule watch from `_start`, and the sleeper never learns of
    the appointment. Put back `move_on_after((planned - now_ms()) / 1000)`, and
    nothing is armed on the wall clock.
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
    """**The scheduler's whole promise, and its whole scope.**

    pH keeps an appointment while `phern daemon` runs and picks up where it left off
    when it starts — however long it was down, coalesced by `claim` to one run per
    missed window. Bounding that by default would be a second policy on top of the
    one P5-06 already settled, and the OS already ships cron, anacron and systemd
    timers for anyone who wants a run to happen without a daemon at all.

    A year is well past any bound a default could reasonably have carried, which
    is what makes this an assertion about the *absence* of one.
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
    """The knob, for a deployment that would rather not resurrect a session
    somebody abandoned — a shape the OS tools do not have, because a schedule here
    is attached to a *conversation* and not to a crontab entry.

    Off by default; this is what turning it on does. Lifting it again on the same
    daemon is what makes the test say something: it proves the appointment was due
    and wakeable all along, and that the *bound* declined it.
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
    """A root and a line of sub-agents under it, left on this daemon's disk by an
    earlier process: `chain[0]` is the root, and each id after it was spawned by
    the one before.

    In the shape ph-rlm opens a child's log in — its spawner in `parentSession`,
    `origin: "subagent"`, and the root's family, so the whole line is one
    directory. Answers each file's bytes, for a test to say none of them moved.
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
        # Something past the header, so a second writer has a log to resume.
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
    """**P11-08's gate: a child's log has one writer, the mount of its root.**

    `session/attach` reaches `Supervisor.start`, which mounted whatever id it was
    handed — a sub-agent's included (defect 6). Its log is written by the mount of
    the root that spawned it, which readmits it whenever that root comes up, so a
    root of its own on the same file is a second writer: the one thing that can
    corrupt a log that is now the child's only record.

    Refused from the header, before anything is mounted or claimed — so every file
    is byte for byte what its root left — and the refusal names the **root**, the
    one id worth attaching instead, even for a grandchild whose spawner is itself
    a sub-agent.

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
        # The root it names is the one attach that is still served.
        assert (await supervisor.start("lead")).session.id == "lead"
