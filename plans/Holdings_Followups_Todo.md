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

## 1. Daemon: a root's lifetime (`phern/src/ph_app/daemon/`)

The largest group, and one cause runs through it: a `Root` is reachable by id from
the moment it enters `self.roots`, and is ended by `_unmount`, but neither moment is
the same for everybody who holds it.

| # | What | Where | Fix shape |
|---|---|---|---|
| D1 | `_unmount` does not end the root's own task when it is inside `_drive`. A `waiting` root that is passivated leaves its task parked for the life of the daemon. A root unmounted mid-turn (`/profile clear <row>` has no idle check; D4's race) runs on against a disposed context, then takes the retry ladder: `supervisor/retry` and `supervisor/failed` appended to the stale session, and `retrying`/`failed` frames published to the watchers of the live successor under the same id. | `supervisor.py` `_unmount` ~3070, `_drive` ~1950-1991, `restart` ~1855 | A cancel scope per root on `root.exits`; `_drive` asks `self.roots.get(root.id) is root` before the ladder; `restart` clears the old root's subscribers once copied. |
| D2 | `restart` runs on the connection handler's task. A client that disconnects mid-restart cancels it inside `_start`: the root is gone from `self.roots` with no `passivated` or `failed` notice, and the other watchers are stranded on the dead root. | `supervisor.py` `restart` ~1842-1854; `duplex.py` ~279; `server.py` ~510, ~831 | Run the restart on the supervisor's task group, as `start` runs mounts (`_mounting`, `_in_flight`). |
| D3 | Watchers are held by `Root` object and attachment by id, so a connection attached to a dead root is never subscribed to its successor: `_join` skips the subscribe when the id is already in `attached`. After a failed restart the TUI spins for status frames that never come. | `server.py` `_join` ~1067, `_part` ~1084; `supervisor.py` `publish` ~595 | Subscribe on every attach (a set add needs no guard), or keep `attached` by root object. |
| D4 | `passivate` decides, waits for `_starting`, then checks identity only. A client that attaches and prompts in the wait has its root recorded `passivated` and unmounted mid-turn. | `supervisor.py` ~3025-3060 | Ask `passivatable` again once the lock is held. |
| D5 | `start`'s fast path returns a root `_start` is still building: before `resume_children`, before `relay` is registered. A client attaching then misses root-log events, and a second sweep can overlap the first (S1). If the build then fails, the second caller holds a success reply for a dead root. | `supervisor.py` fast path ~1217, table write ~1405, `_resume_children` ~1449, `relay` ~1580 | Put the root in the table when it is whole, or have the fast path wait on `_mounting`. |
| D6 | `Root.ring` suppresses `WouldBlock` only, and a closed wake raises `ClosedResourceError`: in the `woken` listener during shutdown (a logged traceback), in `credential_supplied` (an error for a credential that was stored), and in `_decide_profile` outside its `try`. | `supervisor.py` ~567, ~1371, ~1827, ~1917 | Suppress `ClosedResourceError` in `ring`. |
| D7 | `aclose` releases roots one at a time and clears the table at the end, and the fast path precedes the `_closing` check: a handler in the window is handed a disposed root. | `supervisor.py` ~3121-3134 | Pop each root as it is released; check `_closing` first. |
| D8 | `restart` copies subscribers and front ends that left while the root was popped. | `supervisor.py` ~1855-1864; `server.py` ~1086 | Falls out of D1's and D3's fixes. |
| D9 | rpc mode sets `served.agent` before `resume_children`; if that raises, the children are never readmitted in that server. | `modes/rpc_mode.py` ~301-303 | Set the slot after the sweep. |

Small leaks in the same package: `adapters/google.py` `_pending` written only on the
readiness timeout (a cancel during `_ready` re-sends the bytes next step);
`rpc_mode.py` ~132 an unkeyed mount left by a failing `_open`; `rpc_mode.py` ~277 a
woken turn canceled at EOF still writes `session.status idle` after `shutdown`.

## 2. Sub-agents (`ph-core/src/ph/seams/subagents.py`, `ph-rlm/src/ph_rlm/subagents.py`)

| # | What | Where | Fix shape |
|---|---|---|---|
| S1 **(checked)** | `_resume` reads `state.run_id in self._runs`, then awaits before `_admit` writes the entry. Two sweeps of one parent that overlap both readmit the child; the loser's `open_child_log` fails on the live session and its `except Exception` ends the winner's child: a false failure notice to the parent, a log that reads error, running, done, and the child's own unfinished children revoked for good. The seam half is as read; that two sweeps do overlap rests on D5 and on two `credentials/store` requests in a row, neither of which is locked. | seam `_resume` ~2955-3020 | One sweep at a time per parent (a lock that waits, so the second sweep re-reads the state and a credential that arrived is not missed), and D5. |
| S2 | `SubagentService._admit` releases a started child only on `SubagentSpawnError`. A cancel at its `session_written`, or any other exception after `provider.start` returned, leaves the drive parked on `ready` holding its slots, an admission in the log with no ending, and no entry in `_runs` for `delete` to reach. A later sweep readmits it and the provider replaces `_children[run_id]` unconditionally. | seam `_admit` ~2413-2454; rlm ~347, ~558 | `except BaseException`: `_abandon` under `releasing()`, then re-raise. |
| S3 | A send to a child that is ending is acknowledged, then lost. The roster entry outlives the agent for the whole of its teardown (`agents.entry` is the second effect registered, so it runs second to last), `agents.get` returns the dying agent, `ensure_addressable` is skipped, and `steer` writes into an inbox that `dispose` has cleared or is about to. | `agent/registry.py` `get`; rlm `_quiesce` ~739-744; `messaging.py` ~328-354 | `get` declines an agent whose scope is down, as `scope_of` does; `rehydrate`'s "already running" check then needs its own guard. |
| S4 | `rehydrate` that fails or is canceled after `agents.create` leaves `child.agent` set with no drive: every later `rehydrate` answers "already running" and every later send lands in an inbox nobody drains. A kernel cancels binding calls when their cell settles, so a cell stopped mid-send is the trigger. | rlm `rehydrate` ~481-490 | `try`/`except BaseException` around the rest: dispose the agent, clear `child.agent`. |
| S5 | rlm `_admit` handles only `InactiveScopeError` from `_workspace`. A cancel there leaves the created agent and its open session until the parent's scope goes. | rlm `_admit` ~307-322 | The same door as S4. |
| S6 | A job canceled before it runs never calls `run`, and two holders remove their entry only inside it: the harness's `running` set (`/refine` then answers "already running" for that session id for the life of the mount) and `UtcpSeam.runs` (every later background call joins a run whose `ended` never sets, waits out `wait_s`, and no run is started again). Needs a kind cap, or a cancel between `start` and the body's first step. | `harness/__init__.py` ~260-282; `ph_utcp/__init__.py` `_start` ~677, `_running` ~723; `seams/jobs.py` body ~282-289 | The entry's removal belongs to the job's settling, whichever way: a settle callback on `JobService.start`, used by both. |

Latent in `jobs.py`: a foreign cancel while a body waits in `_take` leaks the tokens
already taken (nothing calls `bind`, so loop shutdown only); `on_queued` fires for an
already-canceled job and `record_waiting` guards only the tombstone.

## 3. Workspace (`ph-core/src/ph/seams/workspace.py`)

| # | What | Where | Fix shape |
|---|---|---|---|
| W1 | A workspace is tracked before it is provisioned, so its scope can release it mid-copy. The release reads `held.workspace` with `provisioned=()`: half-copied materials are taken for the agent's work and committed to its branch (S13's exposure, on the live path), and the worktree is removed under the copy thread. `_acquire` then resumes, appends `workspace/provisioned` after `workspace/disposed`, and returns a workspace whose tree is gone. | `_acquire` ~1256-1283; `_track`'s `release`; `workspace_git._release` ~643 | The holding says it is being provisioned and with what; the release waits for the copy and treats those materials as provisioned; `_acquire` asks `self._held.get(agent_id) is held` after each await and raises if not. |
| W2 | A cancel inside `provider.acquire` skips `_take_back`, which only `except Exception` reaches. The tree waits for the next `session/loaded` of that session. | `_acquire` ~1196-1231 | `except BaseException` with `_take_back` under `releasing()`. |

## 4. Tools and the loop (`ph-core/src/ph/tools/`)

| # | What | Where | Fix shape |
|---|---|---|---|
| T1 | A cancel from outside (`SIGINT` or `SIGTERM` on `phern -p`, `--mode json`, `--mode rpc`) leaves `tool/call` with no `tool/result`: the answers are written by ordinary code after the task group, which the cancel skips. The turn is then closed (`step/end`, `turn/end{aborted}`), so repair has nothing open to close on resume, and every later request carries a `tool_use` with no `tool_result`. The same hole when an exclusive call's group raises: later groups get no skipped answer. | `batch.py` ~165-274; `driver.py` ~389-426; `repair.py` ~294 | `answer_uncommitted` in a `finally` under `releasing()`, or repair answers an open call at `turn/end`. |
| T2 | `ToolRuntime._projecting` is keyed by call id, and ids minted for unnamed calls are the same for two agents at the same turn, step and index. | `registry.py` ~1424; `llm/assembler.py` ~44 | Key by agent and call id. |

## 5. Kernel (`ph-rlm/src/ph_rlm/kernel/manager.py`)

| # | What | Where | Fix shape |
|---|---|---|---|
| K1 | `Kernel` has no closed-for-good state. A `run` that already holds the reference when `close_namespace` pops it sees `not self._alive` and spawns a process nothing tracks. | ~715 | A closed flag that `run` refuses on. |
| K2 | No cleanup on a hard cancel: during boot (`_alive` set, never acked, so the next run skips the restore), mid-run (the guest runs on and answers "a program is already running"), and in `_teardown` after `_process` is detached (the kill, `_alive = False` and `journal.forget` are skipped). | ~527-560, ~713-743, ~1317-1352 | `finally` in each, under `releasing()`. |

## 6. Egress (`ph-core/src/ph/seams/sandbox_egress.py`, `sandbox_local.py`)

- `EgressProxy.aclose` cancels a snapshot of `_connections`; one accepted and not yet
  stepped adds itself after the `clear()`, and can call `on_denied` after the unmount.
- `_BRIDGE_VERDICT` keeps a transient `probe_egress` failure for the whole process.

## 7. Latent: the shape is there, no caller reaches it today

- `AgentRegistry._forget` pops by id with no identity check, and `create` overwrites
  `_agents[id]`. Reachable from S2's readmit chain.
- `TuiScreenRegistry._undraw` pops `drawn[screen_id]` by id; both callers discard the
  disposer. A presenter that raises in `register` leaves the claimed entry.
- `ToolRuntime.present_as`: a second claim overwrites `mode`, and its `clear` is not
  identity-checked.
- code-graph `cache_release` restores "previous" without an identity check: wrong for
  two in-process mounts unmounted out of order.
- `MirrorService.render` can re-create `_written[session.id]` and the file after
  `forget` ran on `session/disposed`.
- Tables that are never emptied: `workspace_git._pinned`, empty `_live` sets in the
  sub-agent seam, `token_meter._nodes` (re-created by a read after `forget`),
  `commands/sandbox.py` `_Denials.cache`, `ContextService._notes`,
  `HarnessService._merged`, messaging `_Limiter._buckets`, text-index `_guards`.
- Row-mount writes with no unmount: `PythonCodeRuntime.snapshots`, `runtime.skills`
  and `runtime.skill_modules` (a remount duplicates them).

## Order

D1 to D5 and S1 are one piece of work: who may hold a root, and when. T1 is separate
and is the one that damages a log a person will resume. S2 to S5 are four uses of one
door (a start that fails ends what it started, under a shield). W1 needs a decision
on what a release does while a copy is running.
