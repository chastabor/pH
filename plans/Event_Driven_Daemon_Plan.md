# Event-driven daemon: the sweep, the kernel clock, the invariant check and the socket watch

*Phase 12. Four timers become events and one timer is deleted. The scheduler went
first in `8cbb94a`; this phase finishes the job.*

## Context

pH has reversed a timed poll twice. The TUI used to redraw from a polling loop,
and now a draw is a debounce armed by an event (`tui/app.py:497`). The scheduler
used to tick every five seconds, and now it sleeps until the next appointment
or until something moves its plan (`Supervisor.keep_schedules`,
`supervisor.py:2032`). The rule behind both: **nothing in pH wakes up on a clock
to ask whether something changed.** A wait ends because the thing it waits on
happened, or because a deadline that can be computed in advance arrived.

An audit of every package's `src/` on 2026-09-30 found these left:

| # | Where | Cadence | What it asks |
|---|---|---|---|
| 1 | `server.py:1853` → `Supervisor.sweep` (`supervisor.py:2444`) | 60 s, forever | Has any root been quiet for `PASSIVATE_AFTER`? Has a keep-alive expired? |
| 2 | `Kernel._watch` (`ph_rlm/kernel/manager.py:827`) | 50 ms, while a cell runs | Settled? Canceled? Probe due? Abort grace expired? |
| 3 | `server.py:1875` → `Supervisor.verify_invariants` (`supervisor.py:2112`) | 5 min, forever | Does every cached projection still equal its fold? |
| 4 | `server.py:1867` → `DaemonServer.check_reachable` (`server.py:1474`) | 30 s, forever | Is the socket at our path still ours? |
| 5 | `server.py:1859` → `Supervisor.heartbeat` (`supervisor.py:2098`) | 5 min, forever | (writes `schedule/heartbeat` to every root with a live schedule) |
| 6 | `ph_runtime/lifecycle.py:85` (macOS only) | 1 s, guest lifetime | Has the host process died? |

Rows 1, 3, 4 and 5 all run through one helper, `_every` (`server.py:1726`). When
this phase is done, `_every` has no callers and is deleted.

Everything else the audit looked at is already event-driven or is a one-shot
deadline. That covers the TUI spinner (it exists only while a turn runs), the
crash and retry backoffs, guest output coalescing, every `move_on_after`, and
follow/attach. Bounded startup waits (`launch._await_socket`, the text-index
writer lock, the sandbox egress shim, Gemini upload readiness) are out of
scope here.

---

## Decided with the user (2026-09-30)

1. **Rows 1–3 become event-driven.** Every deadline is computed and slept until.
   Every change that can move a deadline wakes the sleeper. This is the shape
   `keep_schedules` already has.
2. **Row 4 becomes a filesystem watch: inotify on Linux, kqueue on macOS.** No new
   dependency. inotify goes through ctypes on libc, the way `ph_runtime.lifecycle`
   already reaches `prctl`. kqueue is in the standard library's `select`.
3. **Row 5 is deleted, not converted.** Health status belongs to an external
   monitoring process, not to the service's own log. If pH wants health checks,
   they will come from the OpenTelemetry sink pH manages (`session-telemetry-otel`,
   `ph/seams/telemetry_otel.py`) or from a health endpoint a monitor can reach.
   They will never come from a timed log record. This plan adds neither.
4. **No timer fallback anywhere.** If a watch cannot be armed, the daemon says so
   once and makes the question answerable on demand. It does not quietly go back
   to a cadence. *(Inferred from decision 3's rule; overturn it here if wrong.)*

## Decided with the user (2026-10-01)

5. **Wall-clock deadlines are in (P12-00).** A sleep on the monotonic clock does
   not count suspended time, so the scheduler from `8cbb94a` fires late by however
   long the laptop slept. P12-00 explains the mechanism and fixes it for every
   loop that sleeps until an epoch instant.
6. **The kernel probe stays, adjustable and opt-out (P12-03).** `probe_seconds` was
   already a `code-runtime-python` setting. It becomes `float | None`, and `None`
   turns the probe off, the way `passivate_after: None` turns release off. Opting
   out gives up the `phern doctor` loop gauge and the lost-`done` repair (P12-03
   says what each costs).
7. **The guest's parent watch moves to kqueue (P12-07)**, no longer optional.
8. **Order:** P12-06 first.

---

## What breaks

No backward-compatibility shims. Each of the following breaks outright:

* **Wire, protocol 6 → 7.** `DaemonStatusReply` (`payloads.py:438`) loses
  `sweep_every`, `heartbeat_every`, `watch_every` and `invariants_every`. It gains
  `next_release` (epoch ms or null), `socket_watch` (`"inotify" | "kqueue" |
  "unavailable: <reason>"`) and `verify_invariants` (bool). One bump, in the first
  row that touches the reply (P12-06). Later rows add to 7 before it ships. If
  0.6.0 was never cut, fold this into 6 instead.
* **`serve()` and `DaemonServer` arguments.** `sweep_every`, `heartbeat_every`,
  `watch_every` and `invariants_every` go away. `verify_invariants: bool` and
  `watch_socket: bool` replace the last two. Tests and `daemon_helpers.running`
  change with them.
* **`ph.cancel.POLL_SECONDS`** is removed from the module and from `__all__`.
* **`ScheduleService.heartbeat`, `schedule.HEARTBEAT` and the `schedule/heartbeat`
  type** are removed. **Old logs still open.** Every heartbeat was written
  `ignorable` (`session.py:575`), and `_readmit` (`session.py:929`) admits an
  unknown type that carries that flag. The old records stay in the log unread,
  and the TUI adapter drops types it does not know (`tui/adapter.py:146`). No
  log-format bump.
* **Invariant timing.** A cache drift is found at the next settle (a turn or
  command ending) instead of within five minutes. Only a writer can drift a cache,
  and writers run inside turns and commands. A drift injected by hand with no
  writer after it, as the current tests do, waits for the next settle.
* **Kernel probe timing and type.** The first ping moves from about 50 ms into a
  run to `probe_seconds` into it. A cell that finishes in under a second sends no
  ping at all, and a lost `done` is repaired after about 1 s instead of about
  50 ms. `Config.probe_seconds` becomes `float | None`. A profile that sets `0` or
  a negative number is refused at load instead of probing in a tight loop.
* **Socket watch coverage.** A bind mount over the runtime directory produces no
  inotify or kqueue event, though the 30 s `lstat` would have caught it. This is
  documented as a limit in `docs/dev-notes/linux-macos-differences.md`.

---

## The design

### The shape every converted loop takes

`keep_schedules` is the model, and every loop below copies it:

```python
while True:
    self._moved = moved = anyio.Event()        # before the pass, so a change it meets wakes the next
    try:
        await self._pass()
    except Exception:
        log.exception(...)
    planned = self._next(now=now_ms())          # None: nothing is due until something changes
    alarm = Alarm(planned)                      # P12-00: a wall-clock instant
    await first_of(stop, moved, alarm)
    if stop.is_set():
        return
    # `alarm.rang` says the clock ended the wait rather than an event
```

Each loop has three parts. A **pass** does the work and is idempotent, so a
spurious wake costs one pass. A **next** function computes the earliest deadline,
or `None`. A **notice** method sets `moved`, and every event that can move a
deadline calls it.

**The wiring is the risk.** A change with no notice behind it leaves a root
mounted until some unrelated event happens, and on a quiet daemon that may be
never. So **every term of every predicate gets a test** that changes only that
term and asserts the outcome arrives without a clock advancing past it. That
list is the replacement for the "not every term has an event behind it"
paragraph in `DaemonServer.holds` (`server.py:1547`), which this phase deletes.

### P12-00: wall-clock deadlines

**Why.** `anyio` sleeps on `loop.time()`, which is `time.monotonic()`. On Linux that
is `clock_gettime(CLOCK_MONOTONIC)`, which does not count time spent suspended. On
macOS under Python 3.12 it is `mach_absolute_time`, which also stops during sleep
(to be confirmed on the mac rig). So a daily schedule that is twenty hours away
when the lid closes overnight fires eight hours late. The old five-second tick
bounded that lateness to five seconds after waking, and `8cbb94a` traded the
bound away without saying so. A timer armed on the wall clock fires on resume,
and on Linux it is also told when the clock is stepped.

**Landed (2026-10-01)** as `ph/wall_clock.py`: `sleep_until(at)` and `Alarm(at)`.

* **`Alarm`, not a context manager.** The first design was a
  `deadline_scope(epoch_ms)` context manager. It would have run its body inside a
  task group, and anyio wraps any exception the body raises in an
  `ExceptionGroup`, so the scope would not have been transparent. An `Alarm` is a
  `Waitable` (an async `wait()`), raced with the loop's events through
  `first_of`, and `alarm.rang` answers what `move_on_after(...).cancelled_caught`
  used to. `first_of` was widened to take any `Waitable` here (`ph.cancel`),
  ahead of P12-03.
* **Linux:** `timerfd_create(CLOCK_REALTIME, O_NONBLOCK | O_CLOEXEC)`, armed with
  `TFD_TIMER_ABSTIME`, through ctypes (`os.timerfd_create` is 3.13+). **No
  `TFD_TIMER_CANCEL_ON_SET`.** An absolute realtime timer already follows a
  stepped clock, and the thing waited for is an instant, so there is nothing to
  re-plan. It is awaited through `anyio.wait_readable(fd)`, the cancel-safe free
  function.
* **macOS:** kqueue `EVFILT_TIMER`, `NOTE_ABSOLUTE | NOTE_USECONDS |
  NOTE_MACH_CONTINUOUS_TIME`. `<sys/event.h>` says that combination continues "to
  tick across sleep, still uses gettimeofday epoch". **Still to verify on the mac
  rig.** It is written, but not run.
* **Fallback:** another platform, or an `OSError` making the timer (`EMFILE`, a
  seccomp refusal), sleeps on the monotonic clock for the time remaining and logs
  why. Raising would have ended the scheduler's loop, and with it the daemon's
  task group.
* **Users:** `keep_schedules` now; P12-01 and P12-02 next. The kernel keeps
  monotonic time: its deadlines are intervals inside one running cell, and the
  cell slept too.
* **Tests:** `test_wall_clock.py` has eight tests. They cover the instant and not
  before; a past instant; whether the alarm rang; an alarm with no instant; the
  descriptor closed on both paths; the fallback; and, on Linux, the kernel's
  `/proc/self/fdinfo` showing `clockid: 0` and `settime flags: 01`.
  `test_the_scheduler_sleeps_until_something_is_due` also pins the scheduler's
  hand-off to the wall clock. Each was sabotage-checked. Suspend itself is a manual
  check (`linux-macos-differences.md` §9).

### P12-01: root release sleeps until a root's quiet window ends

`Supervisor.keep_releasing(stop)` replaces the sweep task. `serve` starts it when
`passivate_after is not None`.

* **Pass:** the existing `Supervisor.sweep()` (`supervisor.py:2444`), unchanged,
  so tests that call it directly keep working.
* **Next:** `next_release(now)`. For each mounted root that passes every
  non-time term of `passivatable` (status in `QUIET`, no subscribers, nothing
  working beneath, no live schedule), the moment is `quiet_since + window`.
  `quiet_since` is factored out of `Root.idle_for` (`supervisor.py:638`) so both
  read one rule: the last event's time, else the header's `created_at`. The result
  is the minimum, floored at the last pass plus `PASS_FLOOR` (`recovery.py:233`)
  so a release that raises cannot spin. A root blocked by any other term has no
  moment. The event that unblocks it supplies one.
* **Notices:** `Supervisor.notice_release()`. Its callers:

  | The change | Where it is heard |
  |---|---|
  | a turn starts or ends | the `lifetime` listener, `supervisor.py:1302`, which already calls `recheck_lifetime` |
  | a root is mounted | `start`, beside `self._moved.set()` at `supervisor.py:1198` |
  | the last watcher leaves | `Root.unsubscribe` (`:500`) and the dropped subscriber in `Root.publish` (`:525`), through a new `Root.on_unwatched` callback set in `_start` |
  | a root parks on a person, or stops waiting on one | `AskDesk` when `asks` goes empty ↔ non-empty (`frontend.py:303`, `:313`), through a new `AskDesk.on_waiting` callback |
  | a child settles | the `beneath` listener (`supervisor.py:1280`), **before** its `not root.subscribers` guard |
  | a schedule stops being live | `_watch_schedules` (`supervisor.py:2007`), on `CANCELED` and on `TICK` (a `once` that fired) |
  | the ladder ends | `Root.give_up` (`:694`); `failed` joins `QUIET` without the agent's status moving |

  The `AskDesk` callback also publishes the root's status and calls
  `recheck_lifetime`. That closes the first gap `holds()` documents (a parked
  root's `task` hold dropping silently), so the sidebar stops lagging by up to a
  sweep.
* **Doctor:** `DaemonStatusReply.next_release`, beside `next_wake`.
* **Decide here whether the loop becomes a helper.** This row writes the second
  copy of `keep_schedules`' loop, and P12-02 the third. The rules that copy would
  carry are a fresh `moved` before the pass, `stop` read before `alarm.rang`, and
  a floor on how soon the next pass may come. If a small planner (pass, next,
  notice, `planned`) can own them without bending either caller, it should, the
  way a core door keeps a durability rule.
* **Tests:** one per row of the notices table and one per `QUIET` status. Each
  uses `passivate_after=0`, blocks the root by that one term, clears it, and
  asserts the release with `settled(...)`, with no pass counted in between. One
  more test asserts that an idle daemon whose single root is inside its window
  makes exactly one pass (at boot) before the deadline. `test_daemon.py:1389`
  drops `sweep_every`.

### P12-02: the daemon's own lifetime deadlines

`DaemonServer.keep_lifetime(stop)` is the clock half of the deleted
`server.sweep()` (`server.py:1606`). `serve` starts it when `ephemeral or
keep_alive > 0`.

* **Next:** the earlier of `keep_alive_until` (when armed) and `started +
  SPAWN_TIMEOUT` (when `ephemeral and not served`, `server.py:1604`). These are
  the two lifetime terms that end on a clock.
* **Notices:** `_lifetime_moved`, set where those two change: the first frame of a
  connection (`server.py:481–487`) and `_handle`'s `finally` (`server.py:1722`).
* **Pass:** `check_lifetime()`.
* `server.sweep()` released roots before the exit so their flush ran on the
  ordinary path. Every *event-driven* exit today (the last client leaving) already
  goes straight to `stop`, and teardown's `aclose` releases roots within the
  shutdown budget. That path becomes the only one.
* **Tests:** `test_daemon_lifetime.py:477`, `:613`, `:646` and
  `test_tui_remote.py:912` drop `sweep_every=600.0`. That setting existed to stop
  the cadence covering for a missing event, and now no cadence exists. New tests:
  the keep-alive expires at its instant; an unspoken-to ephemeral daemon leaves at
  `SPAWN_TIMEOUT`; a client arriving disarms both.

### P12-03: the kernel clock sleeps until its next deadline

`Kernel._watch` (`manager.py:804`) stops ticking.

* `_ActiveRun` gains `started_at: float` and `moved: anyio.Event`, the latter
  replaced on each loop. `begin_abort()` (`:227`) sets `moved`. That matters,
  because `_serve_call` starts an abort from its own task and the grace deadline
  must be armed at once. `answered()` (`:247`) sets it when the probe leaves
  `stalled`, so the next probe is planned.
* **Next:** if aborting, `aborting_since + cancel_grace`. Otherwise it depends
  on the probe state: `idle` is `(probed_at or started_at) + probe_seconds`;
  `waiting` is `probed_at + probe_seconds` (to latch `stalled`); `stalled` has no
  probe deadline, and the pong arrives through `moved`.
* **Wait:** race `moved` against `token.wait()` (while not aborting) under
  `CancelScope(deadline=next)`. `ph.cancel.first_of` already takes any `Waitable`
  (P12-00), and `Cancellation` is one.
* After the wait the body is unchanged: the `settled` guard, `_probe`,
  `begin_abort` with `_interrupt` started, and `_kill_unresponsive` past the
  grace. `run()` still cancels the watcher when `_pump` returns.
* Remove `POLL_SECONDS` (`cancel.py:157`) and rewrite the docstrings that cite the
  50 ms tick: `_pump` (`:752`), `_watch` (`:810`) and `_ActiveRun.probed_at`
  (`:217`).
* **The probe that remains, and how to turn it off.** One `ping` each way every
  `probe_seconds` while a cell runs. It does two jobs:
  * a **gauge**: the round trip and the stall count `phern doctor` prints
    (`PythonCodeRuntime._loops`, `manager.py:1387`). Nothing acts on it.
  * a **repair**: a guest that is neither running the run nor owing it a `done`
    answers with one (`PingFrame.run`, `kernel/protocol.py:188`), which recovers
    a lost terminal frame. No event can replace this, because a `done` that was
    never sent produces no signal on either side.

  `Config.probe_seconds` (`manager.py:1700`) becomes `float | None`, threaded
  through `PythonCodeRuntime.probe_seconds` (`:1363`) and `Kernel.probe_seconds`
  (`:392`). `None` turns the probe off: there is no probe deadline, `_probe` is
  never called, and `_loops` reports "probe off" instead of a measurement. A
  value `<= 0` is refused when the config loads. The docstring says what opting
  out costs: an unattended run whose `done` is lost then waits for a person to
  press stop, and the abort ladder still ends it from there.
* **Tests:** a run that is never canceled wakes the watcher once per
  `probe_seconds` (counted); an abort begun by `_serve_call` still reaches the
  kill; a pong after a stall re-arms the next probe; cancel is noticed without
  delay. With `probe_seconds=None`, a run sends no ping and the watcher does not
  wake until cancel; a config of `0` is refused. The existing grace, kill and
  stall tests stay green unchanged.

### P12-04: invariants are checked when a root settles

* `Supervisor.verify_root(root)` is extracted from the per-root body of
  `verify_invariants` (`supervisor.py:2112`). `verify_invariants()` stays for
  tests and a future doctor question, and iterates `verify_root`.
* `Root.verified_seq: int | None`. A call returns at once when `root.session.seq`
  has not moved since the last check, so repeated settles with nothing new are
  free.
* **Called at the three moments a writer can have run and the root is quiet:**
  1. `_drive` after its flush (`supervisor.py:1676`), when the inbox has drained;
  2. `_acted` after the durable flush (`server.py:567`), when `root.status in QUIET`,
     which covers commands and mutations on an idle root;
  3. `_passivate` before `_unmount` (`supervisor.py:2494`), as a last look before
     release.

  It is not called on flushes in the middle of a turn. The settle that follows
  covers them.
* **Cost:** the same per-root refold as today (12 ms at 25k events, 161 ms at
  500k, on the loop), paid once per settle instead of every five minutes per live
  root. An idle daemon pays nothing.
* `invariants_every` → `verify_invariants: bool = True`. The test helper's
  default becomes `False` (`daemon_helpers.py:269`).
* **Tests:** `test_daemon_invariants.py:211` becomes "a drift is recorded when the
  next turn settles" (drift, prompt, assert). `:232` becomes the flag. New tests:
  an idle root is not re-verified (seq gate); a command on an idle root is
  verified; passivation verifies before release. The once-only and clearing tests
  keep their assertions.

### P12-05: the socket watch hears the filesystem

**`ph/path_watch.py`**, a new ph-core module beside `ph.lingering` (which owns
`socket_identity`). Platform selection uses static module-top imports. It
provides `EntryWatch(path)`, armed synchronously in `__init__` so `serve` can
arm it before the task group, immediately after it takes `identity`. Its
`changes()` is an async iterator that yields whenever `path`'s directory entry
may have changed, and ends once the directory itself is gone.

* **Linux:** `inotify_init1(IN_NONBLOCK | IN_CLOEXEC)` through ctypes on
  `ctypes.CDLL(None, use_errno=True)`. Watches:
  * the socket's directory, for `IN_CREATE | IN_DELETE | IN_MOVED_FROM |
    IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT`;
  * every ancestor up to `/`, for the self and unmount events only.

  It yields on an event naming the socket's basename, on any self, unmount or
  `IN_IGNORED` event, and on `IN_Q_OVERFLOW`. It awaits
  `anyio.wait_readable(fd)` and drains with `os.read`, parsing `struct
  inotify_event` with `struct.unpack_from("iIII", …)`.
* **macOS:** `select.kqueue()` with one `os.open(dir, os.O_EVTONLY)` descriptor
  per watched directory and `KQ_FILTER_VNODE`, flags `KQ_EV_ADD | KQ_EV_CLEAR`.
  The socket's directory gets fflags `KQ_NOTE_WRITE | KQ_NOTE_DELETE |
  KQ_NOTE_RENAME | KQ_NOTE_REVOKE`; ancestors get the last three. It awaits
  `anyio.wait_readable(kq.fileno())` and drains with `kq.control(None, n, 0)`.
  `NOTE_WRITE` does not name the entry, so any entry change in `$PH_RUNTIME` yields
  (for example, an atomic rewrite of `processes.jsonl`). The consumer's `lstat` is
  cheap enough to absorb that.
* **Anything else, or arming fails** (`ENOSPC` and `EMFILE` are the inotify
  limits): `WatchUnavailable(reason)`.
* **Share libc with `ph.wall_clock`.** It already holds a `ctypes.CDLL(None,
  use_errno=True)` and an errno-to-`OSError` helper (`_failed`). The second user
  is the moment to move both somewhere both modules import, rather than write
  them twice. `ph_runtime` keeps its own, since the guest stays dependency-free.

**Daemon:** `DaemonServer.keep_watching(stop)` replaces the `_every` task. It runs
`check_reachable()` once right after arming, which covers the window between
`identity` and the watch. Then it runs `async for _ in watch.changes(): if await
self.check_reachable(): break`. The latch is one-way, so the watch closes with
it. `check_reachable` itself does not change: the watch replaces only the clock.

**Unavailable:** one `log.error` naming the reason, `socket_watch =
"unavailable: <reason>"` in the status reply, and `status()` runs
`check_reachable()` itself, so `phern agents doctor` still gets a live answer when
it asks. There is no cadence.

* `watch_every` → `watch_socket: bool = True`. The docstring of `WATCH_EVERY`
  (`server.py:185`) is deleted, and `lingering.py:25` ("on a cadence") is
  reworded.
* **Tests:** `test_a_reaped_runtime_dir_reaches_every_root_as_a_record`
  (`test_daemon.py:1780`) drops `watch_every=0.05` and must pass on the event
  alone. New tests: *replaced* (unlink and bind a new socket at the path); an
  ancestor renamed; no `lstat` while nothing changes (spy); unavailable
  (monkeypatched arm failure, so the doctor answers and no task is started).
  `path_watch` gets unit tests per platform, with macOS run on the mac rig. All
  of these run outside the sandbox, which hides unix sockets.

### P12-06: delete the heartbeat

* **Delete:** `HEARTBEAT_EVERY` (`server.py:153`); `DaemonServer.heartbeat_every`
  (`:1358`); `serve(heartbeat_every=)` (`:1778`, `:1831`) and its task (`:1859`);
  `Supervisor.heartbeat` (`supervisor.py:2098`); `ScheduleService.heartbeat` and
  `HEARTBEAT` (`schedule.py:491`, `:73`); `schedule/heartbeat` in
  `KNOWN_SESSION_EVENT_TYPES`, `IGNORABLE_SESSION_EVENT_TYPES` and
  `WRITERS["ph.seams.schedule"]` (`known_event_types.py:118`, `:436`, `:578`);
  the TUI's `RECORDLESS` entry (`adapter.py:1194`) and trajectory handler
  (`trajectory.py:481`); `DaemonStatusReply.heartbeat_every` (`payloads.py:439`);
  and the doctor row (`agents.py:679`).
* **Reword:** `docs/seams/schedule.md:51` and `:119`; the `passivatable` docstring
  paragraph (`supervisor.py:2415`); `_live_schedules` (`:1768`, "the tick, the
  heartbeat and…").
* **Tests:** delete `test_a_heartbeat_records_that_something_is_still_watching`
  (`test_daemon.py:1717`); drop the type from `test_trajectory.py:153`; reword
  the `test_daemon.py:122` docstring. `test_log_writers.py` holds the vocabulary
  to its writers and will fail if a reference is missed.
* **What answers the heartbeat's question now:** "waiting for Wednesday or dead?"
  is a live question, and the daemon already answers it when asked.
  `daemon/status` carries `next_wake` (`server.py:1398`) and P12-01 adds
  `next_release`. A monitor that wants liveness asks the socket, or reads the
  OpenTelemetry export where that sink is configured. **No replacement record.**
* Lands first: it is pure deletion and shrinks what the other rows touch.

### P12-07: the guest's parent watch waits on kqueue

This is row 6, the answer to "why is it needed".

* **Why it exists.** The guest runs model-written cells, and it is started with
  `start_new_session=True` (`manager.py:498`), so a cell cannot reach the
  person's terminal. As a result no terminal or process-group signal reaches it.
  An orderly host exit tears it down (`Kernel._teardown`: shutdown frame, wait,
  `killpg`). A **hard** host death (SIGKILL, the OOM killer, a segfault) runs no
  teardown, and POSIX re-parents the guest to init or launchd rather than killing
  it.
* **Why the socket is not enough.** The guest's ordinary "host is gone" signal is
  the socket's EOF (`Channel.receive` returns `None`), and that read runs on the
  guest's event loop. While a cell is in synchronous Python (a tight loop, a long
  native call, `time.sleep(3600)`) the reader never runs, so the orphan keeps
  executing model code, holding the namespace's memory and anything the cell
  opened. A restarted daemon then starts a second guest beside it. `RLIMIT_CPU`
  eventually stops a CPU-bound orphan, but not one that is sleeping or blocked
  on I/O.
* **Per platform:** Linux has the kernel do it (`PR_SET_PDEATHSIG`, plus bwrap's
  `--die-with-parent`); Windows has the host's Job Object; macOS has neither, so
  a daemon thread checks `os.getppid()` every second. It is a thread because the
  thread still runs while the cell blocks the loop. The host logs the mechanism,
  and `manager.py:634` names the one-second window as a known gap.
* **The event version:** a `select.kqueue()` event with `KQ_FILTER_PROC` and
  `KQ_NOTE_EXIT` on the parent pid. The thread blocks in `kq.control([], 1, None)`
  until the parent exits, then calls `os._exit(0)`. `ESRCH` at registration means
  the parent is already gone, so it exits. It re-checks `os.getppid() == original`
  after registering, which closes the re-parent race. It stays a thread, for the
  reason above. Standard library only, so the guest stays dependency-free.
  `POLL_SECONDS` leaves `lifecycle.py`'s `__all__`, and the mechanism is reported
  as `kqueue-exit`, which also removes the one-second window.

### P12-08: docs and bookkeeping

* `DESIGN.md:1025`: passivation is "at `PASSIVATE_AFTER` of quiet", not "on a 60 s
  sweep".
* `docs/seams/schedule.md`: the scheduler's wake uses P12-00 and there is no
  heartbeat. `docs/seams/invariants.md`: when the daemon runs pollable
  checks.
* `docs/dev-notes/linux-macos-differences.md`: inotify vs kqueue, timerfd vs
  `EVFILT_TIMER`, `NOTE_EXIT`, and the overmount limit.
* `PROTOCOL_VERSION` docstring (`protocol.py:83`): the 7 entry.
* Delete `_every` (`server.py:1726`) and `agents.py`'s `_cadence` (`:184`) once
  nothing calls them. Replace the doctor rows (`agents.py:678–681`) with "next
  release", "socket watch: inotify | kqueue | unavailable (…)" and "invariants: on
  settle | off".
* Phase 12 rows go into `plans/Implementation_Plan.md`.

---

## Rows

| Row | What | Depends on | Gate |
|---|---|---|---|
| P12-06 | **Landed (2026-10-01).** Delete the heartbeat; protocol 7; `test_a_log_written_with_heartbeats_still_opens_with_its_schedule` pins old logs (sabotage-checked by dropping `ignorable`) | — | `test_log_writers`, `test_trajectory`, `test_schedule` |
| P12-00 | **Landed (2026-10-01).** `ph.wall_clock` (`sleep_until`, `Alarm`); `first_of` takes any `Waitable`; `keep_schedules` sleeps on the wall clock. macOS timer unverified on hardware | — | `test_wall_clock`, `test_the_scheduler_sleeps_until_something_is_due` |
| P12-01 | root release sleeps until a deadline | P12-00 | one test per notice row and per `QUIET` status |
| P12-02 | lifetime deadlines | P12-00 | `test_daemon_lifetime` without `sweep_every` |
| P12-03 | kernel clock; `probe_seconds: float \| None` | — | kernel wake count, cross-task abort, stall re-arm, probe off |
| P12-04 | invariants on settle | — | `test_daemon_invariants`, seq gate |
| P12-05 | socket watch on inotify and kqueue | — | the reaped-dir test with no cadence; replaced; ancestor |
| P12-07 | guest `NOTE_EXIT` | — | macOS kernel test: host SIGKILL during a blocking cell |
| P12-08 | docs, doctor rows, delete `_every` | all | `test_non_guarantees`, doc links |

## Reuse (do not rewrite)

* `keep_schedules`, `notice_schedules` and `next_wake` (`supervisor.py:2001–2056`)
  are the template for P12-01 and P12-02.
* `first_of` (`ph.cancel`), which takes any `Waitable` since P12-00, rather than a
  copy.
* `ph.wall_clock.Alarm` for every epoch-instant deadline (P12-01, P12-02).
* `Supervisor.sweep`, `passivatable` and `check_lifetime` are the passes, unchanged.
* `check_reachable` and `socket_identity` are the decision; only the trigger moves.
* `anyio.wait_readable` (the free function) for every fd: timerfd, inotify,
  kqueue.

## Non-goals

* A health endpoint or OpenTelemetry liveness metrics (decision 3 names where they
  would go).
* The bounded startup waits (`launch._await_socket`, the text-index writer lock,
  the egress shim, Gemini upload readiness).
* The LLM retry backoff not being raced against cancel (`agent_loop/driver.py:519`).
  It is a separate defect and gets its own row.
* Test-suite polling (`ph.testing.settled` and the `sleep` calls under `tests/`).

## Verification

* **Every row** ends with all four gates green: `ruff check`, `ruff format
  --check`, `./test.sh types` and `./test.sh test`. Daemon and socket tests run
  outside the sandbox.
* **Every new gate is sabotage-checked.** Delete the notice it holds (for example,
  the `AskDesk.on_waiting` call) and watch exactly that test fail.
* **No-wake check:** run the daemon idle for ten minutes with one passivatable root
  inside its window and one scheduled root, under `strace -f -e trace=epoll_wait`
  (Linux) or a wake counter. The process stays asleep apart from the
  deadlines it planned.

**Definition of done:**

* `_every` and `POLL_SECONDS` no longer exist.
* `grep -rn "anyio.sleep\|set_interval" packages/*/src` finds only one-shot
  backoffs, animation, and the bounded waits listed under non-goals.
* An idle daemon wakes only at deadlines it can name in `phern agents doctor`.
* No log carries a record written because time passed.
