# Holdings follow-ups — what the 2026-11-19 audit found and did not fix

*2026-11-19. Written with the "Holdings" standard (`DESIGN.md` §2.5) and its gate
(`tests/test_holdings.py`). Four read-only reviewers went over every package for the
class the gate cannot see: a start that fails or is canceled and leaves an entry, an
entry that goes another way while its teardown stays, and a write after an ending.*

**How far each item is checked.** "Read" means the steps were read in the source by
the reviewer and the sequence follows from them. Nothing below was reproduced with a
test. Items marked **(checked)** I re-read myself after the report; the rest are the
reviewers' reading alone and each needs a failing test before its fix. Line numbers
are for the tree as of this date.

Fixed the same day, and so not listed: `UtcpSeam.add`, the kernel runtime's scope
table and its build for an agent that is gone, `_Egress.mount` (failed probe, partial
start), the three claiming helpers and `ScopedTable.claim` (entry before teardown),
`WorkspaceSeam._track` (entry before teardown, tree nobody could own), and
`AgentRegistry.create` (a driver that fails).

## 1. Daemon: a root's lifetime (`phern/src/ph_app/daemon/`) — done 2026-11-19

All of D1 to D9 and the three small leaks, each with a test that failed first and
fails again under sabotage (`phern/tests/test_daemon_root_holding.py`,
`test_modes.py`, `test_uploads_google.py`). The shape: a root enters
`Supervisor.roots` as its build's last step and leaves by one door, `_unmount` (out
of the table, wake closed, released, then its task ended — after the release, so a
turn a shutdown cut stays open on disk); a restart is a mount on the supervisor's
task group; `_join` subscribes on every attach. Who is attached is kept per session
id since the follow-up below (`Supervisor.audiences`), which replaced moving the
watchers on a restart and finding a root mid-restart through `Supervisor.holder`.
DESIGN.md §5.4 says so.

## 2. Sub-agents (`ph-core/src/ph/seams/subagents.py`, `ph-rlm/src/ph_rlm/subagents.py`) — done 2026-11-19

S1 to S6 done, each with a test that failed first and fails under sabotage.
S1: one sweep of a parent's children at a time (`SubagentService._sweeps`, a `KeyedLocks`); reproduced
by holding the first sweep inside `readmit`, where the second's hold write met
`SessionForkError` and it tried to end the child the first was readmitting. S2: the
seam's `_admit` abandons the started child on any way out, under `releasing()`. S3:
`AgentRegistry.get` declines an agent whose scope is down (`scope_of` answers through
it); the rlm `rehydrate` check on `child.agent` is the guard that then refuses. S4: a
failed `rehydrate` disposes its agent and puts the settled answer back. S5: the rlm
`_admit` ends the agent and session on any way out of `_workspace`, and `_dispose`s a
child stopped while it was wired.

S6: `JobService.start` takes `on_settled`, called however a job ends, one canceled
before it ran included; the harness's "already running" mark and utcp's runs come
down there. And in `jobs.py`: a job stopped from outside while it waits gives back
the places it took, and `on_queued` does not fire for a job already canceled.

## 3. Workspace (`ph-core/src/ph/seams/workspace.py`) — done 2026-11-19

W1 (decided: stop the copy): the holding records a running copy (`_Copy`); its release
sets a stop flag the copy checks before each file (`workspace_provision.ProvisionStop`),
waits for the thread, and counts every target as provisioned; `_acquire` then finds its
holding gone and raises `InactiveScopeError` rather than handing out a released tree.
W2: a cancel inside the tier's acquire takes back what it made, under `releasing()`.

## 4. Tools and the loop (`ph-core/src/ph/tools/`) — done 2026-11-19

T1: `_run_group` answers its calls on any raise, an outside cancel included, and
`execute_tool_calls` answers the calls queued behind a group that raised. The answers
are synchronous appends, so no shield was needed. T2: `ToolRuntime._projecting` is
keyed by the agent and the call id, and a run takes down only its own entry.

## 5. Kernel (`ph-rlm/src/ph_rlm/kernel/manager.py`) — done 2026-11-19

K1: `Kernel.aclose` closes for good (`_closed`), and `run` refuses a closed kernel
rather than starting a guest no table holds; `start`'s failure paths stop the guest
with `_teardown`, which a restart may follow. K2: a boot or a run that does not settle
takes its guest down under `releasing()`, and `_teardown` is shielded with its kill,
`_alive = False`, journal entry (only once reaped) and socket close in a `finally`.

## 6. Egress — done 2026-11-19

A connection accepted as the proxy closes is closed unserved (`EgressProxy._closing`),
and `_BRIDGE_VERDICT` believes a bridge the probe proved for good and one it could
not for `BRIDGE_RETRY_SECONDS`, rather than for the life of the process.

## 7. Latent — done 2026-11-19

Each with a test that failed first and fails under sabotage, though no caller reached
it: `AgentRegistry._forget` takes out only its own agent (`create` still replaces the
entry of an agent made again for the same session, which tests rely on); a screen whose presenter raised is undrawn and its claim given back;
`present_as` refuses a second claim on a scope; code-graph cache claims are a stack,
each release taking back its own; `MirrorService.render` takes back a file written as
its session went. Tables now emptied: `workspace_git._pinned` (on release), the
sub-agent seam's empty `_live` sets, `token_meter._nodes` (not re-made for a session the
store let go), the sandbox command's refusal counts, `ContextService._notes`,
`HarnessService._merged` and its fold, the messaging limiter's buckets, and the text
index's guards (`ph.locks.KeyedLocks`, which the sub-agent sweeps use too). Row writes
taken back: the snapshot policy and the python skills' lists on `PythonCodeRuntime`.

## Behavior changes

- A root unmounted mid-turn (a restart, a passivation of a `waiting` root, a shutdown)
  has its task canceled after its release; the turn stays open on disk, as before.
- `Supervisor.restart` runs on the supervisor's task group; `start` waits for a mount
  or restart of the same id in flight; `start` refuses every id once `aclose` began.
- A send to a child that is ending is refused ("could not be woken") rather than
  acknowledged and lost.
- `ToolRuntime.present_as` refuses a second claim on one scope.
- A workspace released mid-provisioning stops the copy and `acquire` raises
  `InactiveScopeError`.
- A kernel closed for good refuses `run` with `KernelDied`.
- An rpc session whose children sweep failed is not served until a prompt sweeps again.

## After the plan: two of the skipped cleanups — done 2026-11-19

Both were listed as skipped by the `/simplify` pass and taken up afterwards.

**Per-session tables are a standard** (DESIGN.md §2.5, "Per-session tables").
`ph.session.SessionTable` takes the session as its scope: filing an entry asks the
session to release it when the store lets it go (`Session.when_disposed`, run by
`SessionStore.dispose`), and nothing is filed for a session that has gone. It needs
no context, and it is keyed by the session object. `SessionFoldCache` keeps its
entries in one. Ten hand-written `session/disposed` listeners became tables (goals,
schedule, sandbox refusals, the intent journal, the clm section map, readouts and
mirror, the token meter, the rlm context loader and harness, session telemetry);
three stay hand-written and say why in the gate (`tests/test_session_tables.py`).
Found on the way and fixed: `ph_stabilize.limits` kept its counts for every session
of its mount's life; every fold cache filed an entry again when read after its
session had gone; `UtcpSeam.warmed` kept a mark per root agent ever stepped (now a
kept `Releaser` per mark, dropped by the manual's `forget`). The gate's second half
runs a turn in every shipped profile that boots without a key, lets the session
go, reads it once more and searches what the process holds for anything keyed by
it.

**Who is attached is kept per session id** (DESIGN.md §5.4). `Supervisor.audiences`
holds an `Audience` per id: watchers, answering front ends, the composer's tray and
the ask count. A root reads its id's audience, so `_carry`, `Supervisor.holder` and
`_Mounting.replacing` are gone, and `AskDesk.leave` with them; a connection parts
through `Supervisor.part`, and a watcher leaves only through `Audience.drop`, which
replaced `Root.unsubscribe` and `Root.unwatched`. Tests in
`phern/tests/test_daemon_root_holding.py`.

### Behavior changes from these two

- `SessionFoldCache.forget` takes the session, not its id, and `SessionFoldCache.clear`
  and the `SessionLog` protocol are gone (the cache reads a `Session`).
- Gone, since each table now forgets by itself: `TokenMeter.forget`,
  `MirrorService.forget` and `stale`, `SectionMap.forget`, `Readouts.forget`,
  `HarnessService.forget`, `IntentJournal.forget`, `ScheduleService.forget_session`,
  `GoalService.forget_session`, and the pass-through `stale_folds` on `GoalService`,
  `ScheduleService`, `SubagentService` and `_Denials` (each row hands its cache's
  `stale` to `contribute_fold_cache` directly). `SubagentService.forget_session`
  no longer touches the fold cache.
- `Readouts.crossed` takes the session, not its id.
- `UtcpSeam.warmed` is a `dict` of each mark to the `Releaser` that takes it back.
- `Root.unsubscribe` and `Root.unwatched` are gone; `AskDesk.asked` is gone (the
  count is `Audience.asked`).
- Watchers of a restart that fails are told `failed` and stay attached to the id;
  they hear the next mount of it. They used to be dropped.
- Watchers hear nothing an unmounted root says after its unmount ends.
- The ask count goes on across a root started again while anybody is attached; it
  restarted at 1 with each new desk except across a restart.
