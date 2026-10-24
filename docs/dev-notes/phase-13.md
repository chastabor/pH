# Phase 13 — Stable restart

**Status:** planned 2026-10-22; waves one and two landed the same day (P13-01, P13-02,
P13-08, P13-12; P13-03 to P13-07, P13-10), waves three and four on 2026-10-23 (P13-13,
P13-14; P13-09, P13-11). Fourteen rows in four waves, all landed. The kill-shaped
gates under "Gates" all landed by 2026-10-23, on two restart hosts.

**Gate:** `ruff` + `ruff format` + `mypy` across the tree, and `./test.sh test` green,
plus the SIGKILL gates listed under "Gates" once each wave lands.

pH records enough to recover everything, and recovery of the log itself is correct
and repeatable: an effect never outruns its record, repair is idempotent and
backdated, the lease is a kernel `flock` that a crash gives back, a crashed tree is
committed before it is removed, and a kernel namespace comes back from its
snapshots. What is missing is the half pi-durable's scheduler supplies: something
that, at boot, finds the roots that died busy, re-enters their interrupted work the
way children already are re-entered, and wakes a parent when a child it was told to
wait for reports back. The audit that found the gaps is summarized under "The gaps".

**The principle every row keeps:** phern reports facts and resumes work when asked,
and never restarts itself. An orchestrator — systemd, launchd, Kubernetes, or the
thin `phern service` client — owns process lifecycle, environment and backoff. phern
owes it three things: a cheap, truthful answer to "which sessions owe work", one verb
that resumes exactly those, and probes it can poll.

---

## The gaps

Found 2026-10-22 by reading the restart path end to end against pi-durable's design.

1. Nothing restarts a root at boot. `Supervisor.rehydrate` reads only the schedule
   index; a root mid-turn, with live children or an unclaimed prompt stays unmounted
   until a client attaches. Listed under `NON_GUARANTEES` as "restart".
2. A root's interrupted turn is never continued, so an autonomous goal dies: repair
   closes the turn, the claimed prompt is gone, `/autonomous` runs only on
   `agent/turn-stopping`, and the daemon never reads the goals seam.
3. A readmitted child's result does not wake its parent: `_deliver_notice` injects
   without a wake, and the supervisor rings only at start, prompt, credential and
   profile decision.
4. A child the sweep gives up on tells its parent nothing: `_end_child` passes no
   notice, and the parent was told "it reports back by message".
5. One damaged child log takes the whole family offline: `_stored_tree` has no
   per-child catch, and no operator tool repairs a log the reader refuses.
6. The schedule tick's write-ahead is only in memory; a kill between `claim` and the
   flush repeats the claim, and the index writer can advance first.
7. A credential hold does not propagate to children; a profile hold does.
8. Write failures at shutdown and at client acks are silent.
9. Smaller: a root's final answer rides on the post-run flush; the lock library's
   soft-lock fallback is not pinned off; a resumed root that runs no turn repairs
   again at every boot.

---

## The rows

| Item | Wave | Delivers | Where |
|---|---|---|---|
| P13-01 | 1 | Live-roots index: which roots a host holds busy and why | `ph.seams.live_index`, `Supervisor._note_live` |
| P13-02 | 1 | `sessions/owed` and `phern agents owed` | `verbs.SESSIONS_OWED`, `Supervisor.owed` |
| P13-03 | 2 | `sessions/resume`, `phern agents resume`, `phern daemon --resume-owed` | `Supervisor.resume_owed` |
| P13-04 | 2 | Continue a root's interrupted turn, bounded by a root ladder | `Supervisor._start`, `ROOT_RESUME_LIMIT` |
| P13-05 | 2 | A finishing child wakes an idle parent; a spent child says so | `beneath`, `_end_child` |
| P13-06 | 2 | A credential hold covers children | `Supervisor._start` |
| P13-07 | 2 | Durable schedule tick: append, flush, deliver, reindex | `Supervisor.tick`, `ScheduleService.claim` |
| P13-08 | 1 | Loud write failures: ops record, nonzero exit, `unwritten` in status | `write_on_unwind`, `Supervisor.unwritten` |
| P13-09 | 4 | Damaged logs isolated per child; `phern doctor sessions --repair` | `stored_tree`, `ph.persistence.jsonl` |
| P13-10 | 2 | Flush after a repairing resume; policy first on turn stopping | `Supervisor._start`, `checkpoint_policy` |
| P13-11 | 4 | Lease fallback refused; doctor names the lock mechanism | `ph.locks` |
| P13-12 | 1 | Probes over TCP and as exit codes; `sd_notify` | `ph_app.daemon.probes`, `verbs.DAEMON_PROBE` |
| P13-13 | 3 | `phern service` start/stop/restart/status/resume/watch | `ph_app.service` |
| P13-14 | 3 | Unit and manifest templates, docs, `NON_GUARANTEES` reworded | `contrib/`, this file |

**Probe semantics (P13-12).** *startup*: the unix listener is serving. *ready*:
startup, the socket still this daemon's, no unwritten root, and the lease directory
writable. P13-03 adds "the first resume pass finished".
*live*: a fresh heartbeat from the event loop, the socket still this daemon's, and
the scheduler's last pass inside its planned window. *status*: the daemon status and
lifetime replies plus the owed summary. Loopback only by default, read only, no
secrets in bodies. `READY=1` and `WATCHDOG=1` go to `NOTIFY_SOCKET` when systemd
sets it.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P13-01 | The live-roots index: holds per root, noted at every transition, kept across an orderly stop, rebuilt from the logs | `ph.seams.live_index`, `Supervisor._note_live`, `replay_pending` |
| P13-02 | `sessions/owed` and `phern agents owed [--rebuild] [--check]` | `verbs.SESSIONS_OWED`, `Supervisor.owed`, `JsonlArchive` |
| P13-08 | A failed write is an ops record, an `unwritten` list in `daemon/status`, readiness no, and exit 3 from `phern daemon` | `write_on_unwind`, `Supervisor._flush`, `cli.daemon` |
| P13-12 | `--probe-bind host:port`, `daemon/probe`, `phern agents probe`, the heartbeat, `sd_notify` | `ph_app.daemon.probes`, `ph_app.daemon.notify`, `DaemonServer.probe` |
| P13-03 | `sessions/resume`, `phern agents resume [IDS] [--all] [--force]`, `phern daemon --resume-owed`; readiness waits for the first pass | `Supervisor.resume_owed`, `DaemonServer.resumed`, `_resume_at_boot` |
| P13-04 | A cut-short turn nobody will send again is continued, bounded by the root's own ladder; `--continue-interrupted` widens it to a person's | `Supervisor._continue_interrupted`, `recovery.ROOT_RESUME_LIMIT`, `interrupted_resumes` |
| P13-05 | A waking delivery says `agent/woken`; a child's ending notice is one, and wakes an idle parent in the daemon and in rpc mode; a child the sweep gives up on tells its parent so | `ReactLoopAgent.send`, `_deliver_notice`, `SubagentService._end_child`, `RpcServer._drive` |
| P13-06 | A root held for a credential holds its children; the key that arrives readmits them | `Supervisor._start`, `credential_supplied` |
| P13-07 | A schedule fires through the seam's door: claim, deliver, one write, then the index, and the index only over a written log; a scheduled prompt carries the schedule's source | `ph.seams.schedule.fire_due`, `scheduled_source` |
| P13-10 | The repair a resume wrote is flushed before any work; the turn's answer is on disk before an autonomous gate runs, by the gate's own durable record | `Supervisor._start`, `test_goals` |
| P13-13 | `phern service start/stop/restart/status/resume/watch`: one verdict vocabulary (exit 0 ready, 1 down, 2 degraded), the daemon's options passed through as typed, a watcher on the scheduler's ladder; the agents client's spine shared (`reach`, `stop_daemon`); an owed row says `held` from the daemon; a spawned daemon that exits before listening is reported at once | `ph_app.service`, `agents.reach`, `OwedRoot.held`, `launch.phern_argv`, `launch._await_socket` |
| P13-14 | A system unit template per user, a user unit with the linger advice, an env file, a Kubernetes sketch with the three probes, and the `restart` non-guarantee reworded | `contrib/`, `NON_GUARANTEES` |
| P13-09 | A child log that will not read costs its own row and not the family: set aside as unreadable, counted as not live, written to by nothing — whether the tree read meets the damage or a readmission's strict read does; the live-roots index and the owed listing carry it as `damaged`, a root is listed for its damage alone, and a resume pass leaves that one to the repair; the strict reader's refusal says where it stopped (`LogDamaged`), `phern doctor sessions` asks it of every stored log, and `--repair` cuts a log back to the prefix that reads, under the lease, keeping the file beside it | `stored_tree`, `SubagentService.unreadable`, `SubagentService._isolate`, `LiveEntry.damaged`, `LiveEntry.listed`, `OwedRoot.left_by_a_pass`, `jsonl.LogDamaged`, `jsonl.damage_in`, `jsonl.truncate_damaged`, `lease.leased`, `cli.doctor_sessions` |
| P13-11 | The lease is `flock(2)` by name with the library's existence-lock fallback pinned off (`filelock>=3.32`, the first release verified to have the option); a filesystem without it refuses with `LockUnsupported`; `phern doctor` prints the mechanism, the lease directory, its filesystem and a lock taken there | `ph.locks`, `ph.paths.filesystem_of`, `cli.doctor` |

Tests: `packages/ph-core/tests/test_live_index.py`, `packages/phern/tests/test_daemon_live.py`,
`packages/phern/tests/test_daemon_resume.py`, `packages/phern/tests/test_service_cli.py`,
`packages/phern/tests/test_doctor_sessions.py`, and the rows added to `test_child_logs.py`,
`test_schedule.py`, `test_goals.py`, `test_modes.py`, `test_daemon_launch.py`,
`test_persistence.py`, `test_locks.py` and `tests/test_mass_restart.py`; the kill gates
in `tests/test_daemon_restart.py`.

---

## The things worth knowing

**The index is noted where things become durable.** Every hold reaches disk through
a flush — a turn's first request, a goal record, an acked prompt, a child's status in
its own log, the ladder's records — so the one hook is `session/durable` in the
root's mount, plus the mount itself, passivation and shutdown. The writer remembers
what it last wrote per root and `updated` is not compared, so a flush that moved no
hold costs the fold and nothing else; `updated` therefore reads as "owed since".
The same hook clears a root from `unwritten`, so any successful write of it does,
whichever path made it.

**A mounted root's entry is the process's, not its log's.** A rebuild reads every
stored root, and a root this daemon holds has a log that lags what the daemon knows
until the next barrier. So `owed` hands the rebuild its own entries for mounted roots
(`live`), and the file is surveyed once before and once after. An unvouched rebuild
— nothing mounted, so the store is a guess — is not repeated by every unasked
`owed`, the schedule index's `_tried` rule.

**The two indexes share one file format.** `ph.seams._json_index.JsonIndex` holds the
reading, the version check, the lock, the rewrite, the no-op-first rule and the
rebuild merge; `ScheduleIndex` and `LiveIndex` each add their name, version and entry,
and the schedule index keeps its claims. A format change is still not a migration:
a file of another version reads as empty and incomplete and is rebuilt once.

**A rebuild lists every stored log.** `SessionArchive.stored` takes `limit=None` for
all of them. The listing is flat across families, so under the 500 cap sub-agent
logs could crowd out the very root a rebuild after a crash exists to find.

**The read half of the JSONL backend is its own class.** Two readers need the
store's reads with nothing to write: the daemon rebuilding at boot with no root
mounted, and a test reading what a run left. Both built a `JsonlSessionStore` with
`ctx=None` and a type-ignore. `JsonlArchive` is that read half; the store extends it.

**The inbox's replay is a module function.** `holds_of` asks whether a stored log
holds a prompt no turn claimed, with no `Session` to build an `Inbox` on, and a
second replay there would be a second account of splice semantics. `replay_pending`
is the one fold, and `Inbox.__init__` is now written in terms of it.

**The probe accept loop ends on a closed listener, not a cancel.** Teardown closes
the TCP listener under its `serve`, which raises `ClosedResourceError` from `accept`;
the task treats that as the end of the job rather than letting it reach the group.

**A resume is a mount, and the pass only chooses which.** `resume_owed` reads the
owed listing and calls `start` per root; everything a resume does — the repair, the
flush of it, the continuation, the children, the ring — is in `_start`, so a client
attaching, a schedule waking and the resume pass cannot drift. The pass is one root at
a time because `_mount` serializes mounts anyway, and a root that will not mount is a
`refused` row with the reason, not a failed pass. A spent root is `spent` and left on
disk unless forced: mounting it rings nothing, and a resumer that runs every minute
would otherwise mount it every minute.

**The driver says when it was woken.** `followup`, `steer` and `interject` carried a
`wakeup` flag nothing in a host read; the daemon rang its own doorbell from `prompt`
and nowhere else, so a child's ending notice — delivered by `inject`, "without
waking" — left an idle parent "running" on a pending message forever. `send` now
emits `agent/woken` for a waking delivery and the notice is a `steer`. The daemon's
listener, registered as soon as the root exists, turns that into its ring, so what a
mount itself delivers rings too; `prompt` no longer rings on its own. rpc mode
delivers and then drives through one path (`RpcServer._drive`) that a wake takes as
well, under a per-session lock, so a prompt arriving during a woken turn joins it
rather than meeting "already has active work". Both hosts check the inbox before
they drive: a ring whose message a running turn already took would open an empty
turn.

**The resume record says which turn it cut.** `session/resumed` carries `turn` when
its repair closed one (`Resumption.turn`). `interrupted` alone also covers a repair
that only settled an intent left open between turns, and a resumer that walked back
to the last `turn/start` from there ran a finished turn again. Who the cut turn was
owed to is a core fold beside the record (`cut_turn_owner`: `person` or
`unattended`), walked back over one turn rather than the log, and the wording an
agent is told is shared with a readmitted child (`cut_short_note`). A turn cut
before it took a prompt is owed to nobody (`None`), whatever else holds: its prompt
is still in the inbox and the mount's ring runs it, and continuing the empty turn
as well ran a second turn for one prompt. It still climbs the root's ladder, so a
prompt whose turn dies before its claim at every boot is given up rather than rerun
for good. The tick gate found that one: a kill just after a schedule's write could
land with the woken turn's `turn/start` on disk and its claim not.

**The root's ladder is folded from records the resume already writes.** A
`session/resumed` that names a `turn` is a restart that cut one short, and an
`assistant/message` forgives the restarts before it — the child's rule
(`restarts_since_progress`) stated over the root's own log, walked back from the tail
to the last answer. Past `ROOT_RESUME_LIMIT`
the root is failed with the ladder's own record (`supervisor/failed`), so `status`,
the live-roots index and `resume_owed` all read it, and a successful turn later
clears it as it clears a crash ladder (`_drive` writes `recovered` for a failed
ladder with no attempts too). Which turns are continued unasked: a goal open, or a
turn owed to nobody at a keyboard — a schedule's, a child's notice, an earlier
continuation. A scheduled prompt is therefore delivered under the schedule's
own source (`scheduled_source`), where it used to be indistinguishable from the
person's.

**The live index's `failed` is the host's fact.** A rebuild from the logs read only
the hold records and wrote every root back as not failed, which would have handed a
resumer one more attempt per restart. `rebuild_live_index` requires the host's
`Ladder` — the record types and the fold — and the daemon passes `recovery.LADDER`.

**A schedule fires through the seam's door.** `fire_due` claims, delivers, writes
the log once and then tells the index, which the supervisor used to do by hand — and
the seam used to reindex inside `claim`, so the file could say "later" while the
tick was still in a buffer. One write is enough: the tick precedes the prompt in the
log, and the turn's first request is a fail-closed barrier that writes both, so no
run bills without its tick on disk. A write that fails leaves the index where it was
and marks the root unwritten. `docs/seams/schedule.md` has the crash table.

**The repair is written before any work.** A resume that synthesized closers held
them in the buffer until the next barrier, which a root that ran no turn never
reached, so every boot repaired the same turn; `_start` writes the root once, right
after the continuation or give-up is decided.

**The service client answers in three words.** Every `phern service` command exits
0 ready, 1 down, 2 degraded, and one fold decides which (`Observed.of`): the
readiness probe's own reasons, a root whose ladder is spent, a mounted root parked
on a person, a stored root that owes work and is not mounted. "Parked on a person"
is the daemon's fact, not the client's reading of a status word: an owed row carries
`held` from `Root.held`, since `Root.status` puts other words in front of a hold.
The verdict stays on the client: `ready` is "may be sent work" to a kubelet, and a
root only a person can move must not pull a pod out of service.

The client reaches the daemon through the agents client's spine. `agents.reach` is
one exchange or `Down` — a value, not a refusal, so `status` reports it, `stop`
finds the state it was asked for and `watch` acts on it — and `_ask` is `reach`
plus the exit. `agents.stop_daemon` is the shutdown exchange both `stop` commands
send; the daemon removes its socket before its connections close, so a closed
connection is a clear door and nothing polls the path. The two tables are shared
too. `start` is idempotent: a daemon already listening is joined and still asked to
resume. A daemon it spawns gets `--resume-owed`, so readiness — what a probe port
reports while `start` waits — covers the daemon's own pass rather than going yes with
roots still unmounted, the ordering rejected for `ExecStartPost` below. The daemon's
options are not spelled a second time; everything after `start`, `restart` or `watch`
that is not the command's own option goes to `phern daemon` as typed, and its
refusal of one is the daemon's.

**A spawned daemon that exits before it listens is said at once.** The launcher
waited the whole `SPAWN_TIMEOUT` for a daemon that had already ended on a usage
error — thirty seconds for one wrong flag, and then the wrong sentence. It keeps
the `Popen` and polls it beside the connect, and the exit status is what there is
to say: the daemon's words went to the null device with the rest of its output.
The interpreter-and-module prefix every self-start uses is the launcher's
(`phern_argv`), which `cli.reinvoke` and the service client both build on.

**The watcher waits on the scheduler's ladder.** `WAKE_RETRY_DELAYS` is a wait of
the same shape — something tried, failed, and not different next second — so a pass
that found the daemon ready waits its first rung, and each pass that found
otherwise climbs one. It says something only when the verdict or its reasons
change, so a daemon that stays ready for a week is one line. `--once` is one pass
with the verdict as the exit code, for a cron slot. The watcher takes no signal
handler, looping or not: it holds nothing a signal needs to unwind, and the daemon
it started is a process of its own.

**A damaged child log is a row, not a refusal.** `stored_tree` read every log beneath a
root and let the first refusal through, so one bad file refused the root, its siblings
were never readmitted, and nothing could repair the file because nothing could open the
family. The read now catches per child, and only a refusal of the log's content — a
`ValueError` from every backend, where a log gone missing still raises: the child is
unreadable, and in no parent's `children` — not live, so no sweep waits on it or
readmits it, and not ended, so no sweep writes a tombstone into a file it cannot read.
The tree read folds only the records a child's state is made of, so a bad envelope in a
record of another type passes it; the readmission then opens the log whole, is refused
with `LogDamaged`, and sets the child aside the same way (`_isolate`) rather than
ending it in a file that cannot be read or leaving it live to hold its root for good.
Both file it by parent (`SubagentService._unreadable`), and `unreadable(root)` is the
whole tree's, sorted — what the daemon's live entry reads `damaged` from, with no walk
at all while nothing is damaged. A rebuild reads the same `stored_tree`
(`stored_children`), so it sees what the tree read sees; damage only a strict read
meets shows once a mount has met it. A root is listed for its damage alone, holds or
not (`LiveEntry.listed`, which both writers ask): what a child mid-turn in a file nobody
can open owes is unknown, and a listing that hid it would hide the one state a repair
exists for. What a resume pass leaves on disk is one rule on the row
(`OwedRoot.left_by_a_pass`), asked after "already mounted" by the daemon's pass and by
the service client's verdict alike: `spent` without `force`, `damaged` for a root that
holds nothing else and was not named. The daemon reads a tree once per mount, so a
repair beneath a mounted root shows `damaged` until the root is next mounted; the
command says so.

**`doctor sessions` is the strict read, asked.** The reader is the one statement of
what a log must be: every refusal — a line that is not a record, an envelope this build
cannot validate, no header — is a `LogDamaged` (a `ValueError`, as every refusal was)
carrying the line, the reason and the bytes before it. To count those bytes the reader
walks the file in bytes and decodes each line, at the same cost as the text read it
replaced (35 ms either way on an 822 KB log; a skim unchanged at 2 ms), and an
unterminated last line that will not parse — cut anywhere, a multi-byte character
included — is the torn write it always was. `damage_in` runs that read and returns
where it stopped, so `phern doctor sessions` and a resume cannot disagree about whether
a file reads. `truncate_damaged` is a cut to the accepted prefix and nothing cleverer,
in core and under the session's lease (`lease.leased`, the door `claim_session` takes
too), so a log a daemon or a one-shot run is appending to raises `SessionBusy` however
the cut is reached: a writer appending behind it would put `seq` backwards. The whole
file is written durably to `<log>.damaged` first (`write_atomic`, numbered past an
earlier copy by `ph.paths.free_sibling`), then the log is truncated and synced. A child
a readmission set aside keeps the lease the open took until its root passivates, so
`--repair` leaves it while that daemon holds the root. `doctor` became a group whose
bare form is the old command (`invoke_without_command`), so `phern doctor --profile x`
is unchanged and `phern doctor sessions` mounts nothing.

**The lease is `flock(2)` or nothing.** The library's `FileLock` is whichever lock the
host can offer, and its unix lock switches to an *existence* lock — a file whose
presence is the claim — when a filesystem answers `flock` with "not supported", with a
warning at the moment nobody is reading. An existence lock is the opposite of a lease:
a killed process leaves it behind and the next daemon is refused every session it
held. `ph.locks` asks for `UnixFileLock` by name with `fallback_to_soft=False`, and
turns the errnos a filesystem without `flock` answers with into `LockUnsupported`,
naming the path — a type of its own, so no caller mistakes it for "already active".
`phern doctor` prints the mechanism, the lease directory, its filesystem from the
mount table (`ph.paths.filesystem_of`: `/proc/self/mountinfo`, the longest mount point
the canonical path is under) and the result of a lock taken and let go there for real,
above the profile mount, since the answer needs no profile and is wanted most when the
profile will not mount.

**`contrib/` is templates, not a package.** The system unit is a template instance
per unix user (`User=%i`, `RuntimeDirectory=phern-%i`, an env file under
`/etc/phern`), the user unit carries the linger advice `phern doctor` prints, and
both run `phern daemon --resume-owed` with `$PHERN_DAEMON_OPTS` appended and
`ExecStartPost=-phern agents owed` for the journal. The Kubernetes sketch is one
replica under `Recreate`, a volume for `$PH_HOME`, an `emptyDir` whose subdirectory
the daemon makes 0700 for `$PH_RUNTIME`, and the three probes on `--probe-bind
0.0.0.0:port`. `PATH` is set in the unit to where `uv tool install` puts `phern`,
because `ExecStart`'s first word may not be a variable and `ExecSearchPath` needs a
newer systemd than Ubuntu 22.04 ships.

**The daemon gates kill a daemon, and the core gates kill a mount.**
`tests/mass_restart_host.py` mounts a profile and its test restarts with the core's
resume, which is one root's mount. `tests/daemon_restart_host.py` runs `serve` itself,
in six modes that each put one shape of work in flight through the daemon's own
doors (a person's prompt through `session/prompt`, a schedule through
`schedule/create`), wait until the log and the live-roots index the next daemon
reads hold it, and print the ids. The next daemon is in the test's process, so it
can watch the roots, except for the second-kill gate, whose resuming daemon is the
host again (`resume`). The two tick modes pause the daemon inside `fire_due` rather
than racing it: `tick-claim` parks the delivery, and `tick-write` stops the loop
outright on the write's `session/durable`, before the index is told, with the
root's ring held, because the write yields to the loop and a woken turn's
`turn/start` would otherwise reach disk in one run and not the next. A kill between a turn's start and its claim is gated in process instead
(`test_daemon_resume`). Each gate's docstring names the sabotage that turns it red,
and each was run.

---

## What was traded

**P13-08 keeps the client's ack.** The plan refused a mutation whose write failed
with a `not_durable` code. `_acted`'s own argument stands: the act has happened, and
an error would tell the client the one false thing. Loud means the root joins the
daemon's `unwritten` set, readiness goes false, the log says so at error level, and
the exit code says so at shutdown. The retry a client sends after an error would have
run the act twice once the in-memory key was lost.

**The live index keeps no claim marks.** The schedule index marks a new appointment
on disk before its write because a lost appointment is silence. A lost live entry is
also silence, but the orchestrator has `rebuild` and the index carries `complete`;
the mark can come later if a rebuild ever proves too slow to lean on.

**The continuation is a message, not a record.** The plan had a supervisor record
per continuation beside the ladder. The ladder folds from `session/resumed`, which is
already written, and the continuation itself is a `user/message` tagged as the
supervisor's, which a transcript draws as a notice naming the attempt; a third record
would have said the same thing twice.

**No barrier on turn stopping.** The plan put a flush on `agent/turn-stopping`, first
in line, so a turn's answer was durable before the autonomous gates ran. A gate's own
record is a durable intent (`shell/command`), written whole before the gate starts,
which already carries the answer; the barrier was a second write on every turn for
nothing. `test_goals` pins that the answer is on disk before a gate runs.

**A scheduled prompt's source changed.** It was `{"kind": "user"}`, the person's words
as if just typed. It is the schedule seam's plugin source now, so a resumer can tell
it from a person's and a transcript shows which schedule sent it. A log reader that
drew every `user/message` as the person's own words now draws a schedule's as a
notice.

**`ExecStartPost` runs `owed`, not `resume`.** The plan had the unit resume through
`ExecStartPost=phern agents resume --all`. The daemon's own `--resume-owed` pass
does the same work with readiness no until it ends, which is what `Type=notify`
turns into "started"; a resume in `ExecStartPost` would have run after `READY=1`,
so the unit was active with its roots still unmounted. What the post step is for
now is the journal: the owed table once the pass is over, and with `-` so a refused
root is a line there rather than a failed start that `Restart=on-failure` loops on.

**Degraded includes "owed and not mounted".** The plan listed held, failed, damaged
and unwritten. A stored root that owes work and is mounted by nobody is the state
a resumer exists to end, and a `status` that called it ready would give an
orchestrator nothing to act on; it is a reason with the fix named, and the one
`watch` acts on without a restart. Damaged joined with P13-09, with
`phern doctor sessions --repair` as its named fix; a root listed for its damage alone
is not one `watch` or `service resume` tries to mount.

**A damaged root with no holds is still owed, and `--check` says so.** `phern agents
owed --check` exits 2 while any listed root is unmounted, and a damaged-only root stays
unmounted until a repair, so a cron line of `owed --check || resume --all` keeps
exiting 2 on it — which is right: the state needs an operator, and the table names the
command. A root whose *own* log will not read is not listed at all unless an earlier
host left its entry (a rebuild keeps what it cannot re-read); `phern doctor sessions`
finds it either way.

**Repair is a cut, not a splice.** Everything from the damaged line on goes with it,
kept only in the copy, and a resume then closes whatever turn the cut left open. A
reference-forked descendant whose `first_own_seq` lies past the cut can no longer
materialize its lineage; the "Session lineage" section of `phern doctor` names it, and
nothing here tries to be cleverer than the person reading that.

---

## Gates

All landed, on the two SIGKILL hosts: `tests/mass_restart_host.py` for the damaged
log, and `tests/daemon_restart_host.py` for the rest.

- A kill with a running child and an open goal, then `phern agents resume --all`
  brings the child back, continues the root, and the goal settles without a client
  — `tests/test_daemon_restart.py`.
- A kill while a readmitted child is still working, then the child finishes while the
  root is idle, and the root wakes and reads the notice — `tests/test_daemon_restart.py`.
- A kill between a tick's claim and its delivery produces exactly one run, and so
  does one between its write and the index — `tests/test_daemon_restart.py`.
- A child log with a malformed middle line leaves its root openable, listed damaged,
  and repairable by doctor — `tests/test_mass_restart.py`.
- Probe tests over the TCP port for each state, including readiness going false on
  an unwritten root — `tests/test_daemon_restart.py`.
- A second kill right after a resume adds no closers — `tests/test_daemon_restart.py`.
