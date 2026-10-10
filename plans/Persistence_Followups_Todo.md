# Persistence follow-ups — what the 2026-10-09 cleanup rounds left standing

*2026-10-09. Follows `dfc0d5b`. The `/simplify` rounds over the spill store, the session
doors and the resume code skipped some findings as outside their diffs, and decided against
others. This document turns the ones worth doing into an ordered todo list, and records why
the rest are closed.*

**Status, 2026-10-09.** R1–R4 have landed (uncommitted), each with a sabotage-checked
gate, and the `/simplify` pass over them put the root's open behind one door
(`runtime.open_root`). R5, R6, R7 and R8 (found by that pass) are open.

## The goal every item is weighed against

The same as `plans/Contained_Log_Writes_Todo.md`'s: **several sub-agents are running, each
with its own session log. Everything stops at once. The daemon restarts, and every sub-agent
picks up where it left off.** These items are about what that restart costs. A stored log
should be found once and read once, and never on the event loop every root shares.

What a store search costs, measured on a store of tagged `<cwd-tag>-<root>` families
(F is the number of family directories):

| F | Search that misses (a new id) | Search that hits (a tagged root) | One `stat`, family known |
|---|---|---|---|
| 100 | 0.84 ms | 0.33 ms | ~6 µs |
| 1000 | 8.4 ms | 3.0 ms | ~6 µs |
| 5000 | 45 ms | 16.8 ms | ~6 µs |

A whole-log read takes about 12 ms for 2,000 events and 130 ms for 20,000. Reading a fork's
start (`recorded_start`, through `materialize`) took 14 ms and 135 ms at those sizes.

---

## The items, one by one

### R1 — The daemon reads a root's log on its event loop, then searches for it again

**Today.** `Supervisor._start` (`ph_app/daemon/supervisor.py:1230`) calls `recorded_start`
synchronously inside an async method. That is a store search, then a header read and a line
scan; for a fork or segment root it is a read of the whole lineage. Every root the daemon
serves waits behind it. Then `open_session` searches for the same log again, off the loop.

**Why it matters.** A mass restart is when the daemon starts many roots at once, so this is
when the stall is largest. `open_session`'s docstring (`ph/persistence/opening.py`) says a
root is searched for once, which is not true on the daemon.

**Fix.**
- Run `recorded_start` off the loop (`anyio.to_thread.run_sync`).
- Add `family` to `RecordedStart` (`ph_app/sessions.py:297`); the header it reads has it.
- Pass it on: `open_session(ctx, root_id, family=recorded.family or None, ...)`, so the
  resume's read is one `stat` instead of a second search.

**Gate.** A daemon start test that patches the per-id search to raise, asserting the
resume finds the log through the family; plus one that `recorded_start` does not run on
the loop thread.

### R2 — `mount_session` reads a session's start twice

**Today.** `mount_session` (`ph_app/runtime.py:111-115`) calls `recorded_start` to check
`.owner` and discards the rest. Then `session_profile(session_id, requested, recorded=None)`
(`ph_app/profiles.py:691`) calls `recorded_environment`, which calls `recorded_start` again.
`open_session` then searches a third time. This hits every `phern -p --session <id>` and
every rpc session opened by id.

**Fix.**
- Keep the first `RecordedStart` and pass `recorded=start.environment` into
  `session_profile`.
- With R1's `family` on `RecordedStart`, return it from `mount_session` so `run`
  (`runtime.py:191`) and rpc can pass it to `open_session` as well.

**Gate.** Count `recorded_start` calls across a `mount_session` for a stored id (one), and
reuse R1's no-search assertion for the open.

### R3 — `recorded_start` repeats work of its own

**Today.**
- `ph_app/sessions.py:340-341` checks `path.is_file()` right after `locate_session` found
  the file. `read_stored` dropped the same repeat in `dfc0d5b`.
- `sessions.py:396-397` reads a fork's lineage with
  `materialize(partial(read_stored, sessions_dir), header.id)`, without
  `family=header.family`, so it searches the store again for a log whose family it holds.

**Fix.** Drop the check (`locate_session` already said the file is there), and pass
`family=header.family` to `materialize`.

**Gate.** A fork-start test that patches the per-id search to raise after the first
locate.

### R4 — rpc mode mints ids before `open_session`, so new sessions are searched for

**Today.** `open_session` skips the read for an id it mints itself. rpc mode mints its own
instead (`ph_app/modes/rpc_mode.py:166` for `session/new`, `:208` for `session/prompt`),
because `_mount` keys `_served` by id before the open. It turns that knowledge into
`fresh=` for `mount_session` (`:107-116`) but not for `open_session`. So every id-less rpc
session pays a search that misses, for an id minted microseconds earlier.

**Fix.** Pass `None` to both doors, as print mode does, and key `_served` by `session.id`
after the open. That removes `_mount`'s `fresh` parameter too. Keep `session/new`'s current
behavior for a named id: it always opens, even when this server already serves the session.

**Not this.** A `fresh=`/`new=` keyword on `open_session`: a caller that passed it wrongly
would create a session over a stored log, which is P5-03's corruption.

**Gate.** An rpc test that `session/new` with no id never searches the store, and that a
later `session/prompt` naming the returned id finds the same served mount.

### R5 — The TUI and web front ends mint ids that the daemon then searches for

**Today.** The TUI (`ph_app/tui/app.py:398`) and the web launcher's shared tab id
(`ph_app/cli.py:303`) mint an id and hand it to the daemon. The daemon cannot tell the id is
new, so it searches twice: `read_start`'s search misses, so there is no family, and
`open_session` searches again. Since R1 both are off the loop.

**Needs a decision.** Skipping that search needs the attach protocol to say "new", or the
daemon to mint the id and hand it back. The web case is the hard one: every tab of a launch
shares one id (`serve.py`'s module docstring says why), so the id must exist before any tab
attaches. Either way it is a daemon-protocol change. After R1 the remaining cost is two
off-loop searches per new root, so this is the lowest-value item here.

### R6 — `AgentRegistry._forget`'s docstring contradicts its code

**Today.** `ph/agent/registry.py:178-189`. The docstring says "`emit` before the pop is
deliberate — a listener asking `agents.get` about the agent it is being told about should
still find it". The code pops first, then emits `agent/disposed`.

**Needs a decision.** If a listener of `agent/disposed` needs `agents.get`, the code is
wrong and the order should flip. If none does, the docstring is wrong. Check every
`agent/disposed` listener first. If one needs the lookup, this is a correctness bug: run
`/code-review` on it rather than fixing it as a docstring edit.

**Gate.** If the order flips: a listener that calls `agents.get` from `agent/disposed`
finds the agent, on both the explicit `dispose` path and a parent scope's cascade.

### R7 — Optional: measure the batched spill write's sequential fsyncs

**Today.** `write_atomic_all` (`ph/paths.py`) writes a batch's files one after another on
one thread, then syncs each directory once. Before `4461585`, the kernel snapshot wrote its
blobs concurrently, which can share a journal commit on ext4.

**Do it only if** snapshots of many large variables turn out slow. Measure a cell spilling
K variables at K = 1, 5 and 20, before and after. Most cells spill none or one, and for one
blob the batch is one thread hop, as `try_save` was.

### R8 — A resumed fork or segment still reads its whole lineage twice

**Today.** `recorded_start`'s fork branch (`ph_app/sessions.py`, `_environment_at`) folds
the environment through `materialize`, which validates every envelope of every ancestor,
to keep only the `profile/*` records. The open then runs `materialize` again. Measured at
F=1000: 21 ms + 17.5 ms at 2,000 events, 185 ms + 180 ms at 20,000. Root starts run one at
a time (`_starting`), so on a mass restart this adds straight to the wall time.

**Fix.** A filtered lineage walk: the line scan roots already get, walked down the chain
(the fork's own file, then each ancestor through `header.parent_session` and
`header.family`, keeping records below the boundary each child inherits at). A prototype
measured 2.97 ms at 20,000 events, about 60 times faster. `materialize` cannot be reused
with `types=` as it stands: it takes the boundary from a file's first event, which a
filtered read does not have, so the walk takes it from each header's `seed_length`
instead. It belongs beside `materialize` in `ph/persistence/lineage.py`, so the chain's
rules (depth bound, cycles) are kept in one place. The open's `materialize` stays the
one strict read, and still refuses a broken lineage.

**Gate.** A fork-of-a-fork start that folds the right environment without
`materialize` being called (patched to raise), against one that does.

---

## Closed, with the reason

- **`locate_under`'s bare-root `stat` before any listing stays.** The print and rpc hosts
  open roots with no cwd, so those are filed under their own id and one `stat` finds them.
  Dropping it would cost every bare root a listing. Stated in its docstring.
- **`exists(family=)` stays optional.** `SubagentService._write_child` asks about a child it
  has no state for, by id alone, through `to_thread.run_sync(store.exists, session_id)`.
- **The lineage survey keeps its `exists(ancestor, descendant)` callback.** Rows of
  `(id, parent, family)` would only move the family dict from `protocol.py` into
  `lineage.py`; a lambda stays either way while `family` is an optional keyword.
- **The survey does not cache repeat ancestor checks.** It runs only in `phern doctor`, and
  each check is one `stat`.
- **Turso still checks the file twice when a read finds it.** `_reading`'s `is_file()` must
  stay: the driver (pyturso 0.7.2) has no read-only open and creates a missing file.
- **A Turso database with no header row now reads as absent**, and `open_session` creates
  a session in it. The header is written in the same transaction as the first events, so
  only an already-broken file has none.
- **phern's store-less readers stay filesystem-shaped.** `recorded_start`,
  `session_summaries` and `profiles_cli` read JSONL files before anything is mounted;
  `sessions.py` says why, and a Turso deployment's root is brought to its log after it
  opens (`opened`).
- **`_resume` readmits children in order.** Its order decides which notices reach the
  parent's inbox first.

## Todo list

In order. Same rules as before: every gate sabotage-checked (put the old code back, the
gate fails), `./test.sh` green outside the sandbox, nothing committed by Claude.

- [x] **R1** — `Supervisor._start`: `recorded_start` off the loop; `RecordedStart.family`;
  `open_session(..., family=)`; correct `open_session`'s "searched for once".
  - *Landed.* `runtime.read_start` reads a session's start on a worker thread (nothing
    for an id about to be made). `RecordedStart.family` is the directory the log was
    found in. The daemon hands it to `_session_for`, which opens the root by path.
    `open_session`'s docstring says how a host passes a root's family.
  - *Gate:* phern `test_daemon.py::test_a_resumed_root_says_so_in_its_own_log`, which now also
    checks the start's thread and counts the searches.
  - *Sabotaged two ways*, and each failed its gate: the start read on the loop, and
    `family` dropped from `_session_for`'s open (the root searched for twice).
- [x] **R2** — `mount_session`: one `recorded_start`, its environment passed to
  `session_profile`, its family returned for the open.
  - *Landed.* `mount_session(exits, start, requested)` takes the `RecordedStart` its host
    read, id included, instead of reading it again, and composes the profile off the
    loop. `session_profile(requested, environment)` takes the environment only, so
    nothing below the host reads the log. The open is one door, `runtime.open_root(ctx,
    start)`: the id and the family from one reading, used by print mode, the daemon's
    `_session_for` and rpc's `_open`. rpc keeps the start on `_Served`, and once a
    session opens, its own id and family. The withdrawal path passes the family too.
  - *Gate:* phern `test_modes.py::test_a_one_shot_run_reads_a_stored_session_once`
    (one start read, one search, where there were two reads and three searches).
  - *Sabotaged:* `family` dropped from `prompted`'s open, and the store was searched
    twice.
- [x] **R3** — `recorded_start`: drop the repeated `is_file()`; read a fork's lineage with
  `family=header.family`.
  - *Landed*, with the family taken from the directory the fork was found in
    (`path.parent.name`), which is exact even for a header that will not parse.
  - *Gate:* phern `test_session_per_profile.py::test_a_fork_is_mounted_from_the_base_its_prefix_holds`,
    which now also counts the searches.
  - *Sabotaged:* `family` dropped from `_environment_at`'s `materialize`, and the fork
    was searched for twice.
- [x] **R4** — rpc mode: let `open_session` mint the id; key `_served` after the open; drop
  `_mount`'s `fresh`.
  - *Landed.* `_mount(None)` mounts a session about to be made; `_open` opens it, letting
    `open_session` mint the id, and files the mount under `session.id`. `session/new`
    and `session/prompt` both go through it; a named id still always opens on
    `session/new`, as before.
  - *Gate:* phern `test_modes.py::test_an_rpc_session_made_without_an_id_is_not_searched_for`,
    which also checks a prompt naming the returned id is served on the same mount, and
    `test_an_rpc_session_named_by_a_stored_id_is_searched_for_once`.
  - *Sabotaged two ways*, and each failed its gate: the id minted in `RpcServer` and
    passed in (the store searched), and a stored id opened by id alone (searched
    twice).
- [ ] **R6** — `AgentRegistry._forget`: audit the `agent/disposed` listeners, then fix
  the code or the docstring.
- [ ] **R8** — a filtered lineage walk beside `materialize`, for a fork's start.
- [ ] **R5** — TUI/web ids: decide the attach-protocol shape (needs a protocol bump).
- [ ] **R7** — optional: measure `write_atomic_all` against concurrent writes for large
  snapshots.
