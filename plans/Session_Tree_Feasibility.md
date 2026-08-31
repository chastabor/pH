# Session log as a tree — feasibility assessment

*Investigating: replace the session log's flat monotonic `seq` with a tree, so
another process can pick up at any point and start its own branch. Modelled on
`sources/tau`. The monotonic count is to be kept.*

Verified against source and by experiment. Citations are `file:line`.

---

## 0. Verdict

**Feasible, and far cheaper than a tree of per-event ids — because the metadata
to do it already exists.**

The design that works is **reference-forking**: a session file records *which
parent file* and *how many of its leading events to take*, and materialises its
lineage on load by walking parents to the root. Segmenting one logical session
across several files is the same mechanism with no divergence.

Measured facts behind that:

1. **`SessionHeader` already carries exactly what is needed** —
   `parent_session` (which file) and `seed_length` (how many leading events).
   Today those two fields *describe* a copy that already happened; under
   reference-forking they *define* the reference. Nothing new is added to the
   envelope.

2. **A forked child's own work already begins at `seq == seed_length`**, and its
   inherited prefix is byte-for-byte `parent[0:seed_length]`. Verified. So the
   materialised session is **identical to today's copied one** — this is a pure
   storage-layer change.

3. **Read amplification is 1.00×.** Each file stores only its own contiguous run,
   so the ranges pulled from each ancestor are *disjoint* and their union is
   exactly the materialised log. The cost is D file opens for chain depth D, not
   D × filesize. Verified by construction.

4. **Copy-forking is expensive and the measurement is stark.** Ten forks at the
   tip of a 348 KiB / 2 000-event session: **3 829 KiB on disk — 11× the base** —
   25.8 ms per fork, each fork writing 348 KiB to hold *two* events of its own
   work. Reference-forking makes each fork file ~0.3 KiB: a **~1 160× reduction**.

5. **No compound per-event id is needed.** `seq` stays dense and positional
   *within a lineage*; siblings forked from one boundary reuse the same numbers
   and never appear in the same list. Cross-lineage addressing is
   `(session_id, seq)` — both values already exist. **Turso's schema does not
   change**, and the 32/32 bit-packing question is moot.

6. **The safety property is the append-only structure itself.** A child depends on
   `parent[0:n]`, and a *prefix of an append-only log is immutable*. The parent
   can keep growing forever without invalidating any descendant. That is why one
   writer per file needs no lock, no coordination, and no conflict resolution —
   and it is precisely what tau's shared-file design cannot offer.

**What this supersedes.** An earlier draft of this document recommended adding a
compound `(agent, n)` entry id beside `seq`. Reference-forking is the better
design: it delivers O(1) forks, composable logs, and per-file single-writer
semantics **without touching event identity at all**. §5 is rewritten around it;
§4's blocker list is retained because it still describes what a *full* per-event
tree would cost, and is the reason not to build one.

**What is still genuinely new work:** two readers must become chain-aware, and
referential integrity becomes a real concern (§5.4). That is the whole of it.

## 1. How session log entries are stored today

### The envelope

`SessionEvent` (`ph/session/events.py:118-142`), a frozen slots dataclass:

| Field | Meaning |
|---|---|
| `type` | the event vocabulary (~70 known types) |
| `seq` | **always its log index** |
| `time` | epoch ms |
| `data` | frozen, losslessly-JSON payload |
| `ignorable` | may a different build skip this type |
| `source_event_seqs` | provenance: which earlier events this one derives from |
| `surface_op` | `"append"` or `SurfaceReplace(start, end)` — surface-eligible types only |

### Identity is position

`Session.append` mints `seq=len(self._log)` (`session.py:295`). `_readmit`
refuses any seed where `source.seq != index` (`session.py:404-408`). So identity,
position and log length are the same number by construction — which is invariant
A1, and is deliberate.

### Two derivations from one log

- `derive_messages()` — the model-visible history, projected from the **surface**
  (`session.py:340-355`).
- `transcript()` — every append-origin message, compaction or not.

The surface (`ph/session/surface.py`) is an ordered list of node seqs.
`surface_op: "append"` pushes; `SurfaceReplace(start, end)` swaps a slice of the
node list for one new node. The log never changes — invariant I4.

### Storage

- **JSONL**: one file per session, header line then one line per event, appended.
  No key.
- **Turso**: one database per session,
  `CREATE TABLE events (seq INTEGER PRIMARY KEY, wire TEXT NOT NULL)`
  (`turso.py:69`). In SQLite an `INTEGER PRIMARY KEY` *is* the rowid, so the
  table is clustered by seq and `SELECT wire FROM events ORDER BY seq`
  (`turso.py:181`) needs no sort. `INSERT OR REPLACE` makes writes idempotent.

### Branching today: a prefix copy

`SessionStore.fork(source, boundary)` (`store.py:159-230`) slices
`log[:boundary + 1]` and seeds a **new** session with a **copy**. The module
docstring states the model plainly (`store.py:8-11`):

> *"pH does not model branching as a message tree, it models it as
> `fork(source, boundary)` plus `seed_length`."*

Consequences: forking is O(prefix) in storage, and **the same logical event has
different identity in parent and child**. Ten forks of a 10 000-event session
copy 100 000 events.

---

## 2. What tau actually does

| | tau | pH |
|---|---|---|
| Identity | `uuid4().hex` | `seq` = log index |
| Structure | `parent_id: str \| None` | none (flat) |
| Order within a branch | the parent chain, reversed | `seq` |
| Order across branches | **JSONL file order** | n/a |
| Monotonic counter | **none anywhere** | `seq` |
| Derivation | `path_to_entry(entries, leaf_id)` — root→leaf walk | fold over the surface |
| Current branch | last `LeafEntry` in file order | n/a |
| Branch operation | append **one** `LeafEntry`; nothing copied | copy the prefix |
| Compaction | `replaces_entry_ids` — an **id-set** | `SurfaceReplace(start, end)` — a node-list slice |
| Concurrent writers | **no** | no |
| Read cost | full file read + parse + validate **per message** | incremental, seq-indexed |

### What is worth taking

**Branching as one entry.** `branch_to_entry` (`session.py:870-939`) appends a
single `LeafEntry(parent_id=target, entry_id=target)`, sets `_last_parent_id`, and
re-replays. Nothing is copied, nothing is deleted; the abandoned subtree stays as
a sibling. Editing an earlier user message is the same operation with
`target = entry.parent_id` and the old text pre-filled.

**Compaction by id-set is branch-safe by construction.** `_apply_compaction`
(`memory.py:107-129`) drops rows whose id is in `replaces_entry_ids`. A
compaction on one branch **cannot** affect a sibling, because the sibling's
root→leaf path never contains that `CompactionEntry`. pH's positional slice has
no such guarantee.

### What is worth refusing

**tau's concurrency story, because there isn't one.**

- The per-session `flock` (`storage.py:124-133`) is held for one
  `append`/`read_all` — never across a read-modify-write.
- `_last_parent_id` is cached in memory (`session.py:403`), and
  `_refresh_persisted_state` re-reads the file but replays with **the leaf the
  caller just wrote** (`session.py:3214-3217`) — a rival's entries are read and
  ignored.
- No conflict resolution, no sibling ordering. Two processes appending under one
  parent produce two silent sibling subtrees; the winner is whichever `leaf` line
  lands last in the file (`_latest_leaf_entry`, `session.py:3678-3682`).
- Print mode *avoids* the problem: it refuses to resume an existing id and
  creates the file exclusively (`cli.py:1161-1175`).

**tau's absence of an index.** `_persist_message` (`session.py:3117-3153`) calls
`_refresh_persisted_state` after **every** message — a full read, full JSONL
parse, full pydantic validation, plus an O(n) dangling-parent pass. A turn with N
tool calls costs ~2N+2 appends and ~2N+2 full reparses. There is no pagination
anywhere; `read_all` is the only read primitive. `append_batch` reads the whole
file and rewrites it.

**Your instinct to keep the monotonic count is well-founded** — it is exactly
what lets pH read incrementally where tau re-reads everything.

**tau's file-order dependence.** Leaf selection is a reversed file scan. That is
a *total-order* dependency hiding inside a pointer-based design, and it is
precisely the part that breaks with two writers.

---

## 3. The measurement that shrinks the problem

You expected folding to be the big issue. Measured:

```
surface nodes          : (0, 2, 4, 6)     <- already sparse vs log positions
after a replace        : (7, 6)
  monotonic in seq?    : False
  dense?               : False
derived                : ['SUMMARY', 'd']

replace naming a seq that exists in the log but is not a node:
  REFUSED -> "surface replace: start seq 1 not found in surface"
```

So, today, already:

- The node list is **sparse** — non-surface events (turn/step boundaries, chunks)
  are correctly absent.
- After any replacement it is **non-monotonic** — the new node's seq is larger
  than the nodes after it.
- `SurfaceReplace(start, end)` is resolved by `state.nodes.index(op.start)`
  (`surface.py:173,177`) — **membership and position in the node list, never a
  numeric range**. A seq that is a real log event but not a surface node is
  refused.
- `compaction.py:1242-1243` builds `shadowed_seqs = nodes[:cutoff]` — a slice of
  the node tuple, already sparse in seq space.

**Nothing anywhere enumerates `range(start, end)`.** `end - start == len(shadowed)`
is already false after any replacement.

**Conclusion: the derivation treats node ids as opaque, ordered by list
position.** Swapping the id type is mechanical there. The real work is elsewhere.

---

## 4. Blast radius

`seq` is three roles fused (`events.py:122`, `session.py:295`, `session.py:245`):

| | Role | Needs |
|---|---|---|
| **(1)** | event **identity** | to become a tree id |
| **(2)** | **position** in the flat list | a dense per-file ordinal |
| **(3)** | log-length **watermark** | a count |

208 `seq` references across 36 source files, 77 in tests. But the split matters:
**everything that only needs (2) or (3) survives untouched.**

### Hard — genuinely requires a dense ordinal

| # | Site | What it asserts |
|---|---|---|
| 1 | `session.py:295` | identity minted from position |
| 2 | `session.py:404-408` `_readmit` | `seq == index`, the gate every seed takes |
| 3 | `surface.py:224` + `:301` + `:272` + `:326-329` | `expected_seq` contiguity, three times; `_processed` is simultaneously count, index and expected seq |
| 4 | `session.py:353`; `surface.py:201`; `compaction.py` ×7 | node seqs dereferenced as `log[seq]` — **no `seq → event` map exists**; the largest single work item |
| 5 | `surface.py:154` | `source >= event.seq` — the one true arithmetic comparison ("sources must be earlier"). Companion: `compaction.py:738` |
| 6 | `store.py:210,217,230,243-279` | fork boundary is a positional int and the cut is a prefix slice — *this is the feature being added* |
| 7 | `repair.py:110,148,156,169` | closers minted by `last.seq + 1`, and must land exactly at `len(events)…` to pass #2 |
| 8 | `turso.py:69,146,181` | `seq INTEGER PRIMARY KEY` is the rowid **and** the read-back order |
| 9 | `ph_app/wire.py:156-176` `index_at_or_before` | "nearest preceding" transcript↔trajectory join — ambiguous across siblings by construction |
| 10 | `ph_app/agents.py:362-365,411,515` | live/replay dedupe by `>` on a total order |

### Mechanical — different key type only

- Every type declaration (~20 sites).
- **`SurfaceReplace` resolution** — membership + node-list position (§3).
- All `source_event_seqs` producers and consumers **except** `surface.py:154`.
  Verified: 7 files, 13 hits, no producer builds a contiguous range.
- All 8 `SessionFoldCache` users plus `folds.py` — they want log **length**.
- All `seed_length` slice consumers — counts.
- The whole daemon paging/cursor protocol. The cursor carries `session.seq`,
  which **is** `len(log)` — an offset, never an id (`protocol.py:161,179-180`).
  Only the field name misleads.
- `checkpoints()` dict key; `ref_for(session, agent, seq)` — which **already
  carries `agent_id`**, so a compound key maps naturally.
- `/revert <seq>` CLI parse/format (`revert.py:58` uses `isdigit()`).
- The `-1` "no seq" sentinel contract (~6 sites).
- `SESSION_FORMAT_VERSION` bump — **mandatory**: `wire.py:59` sets
  `extra="forbid"`, so any added envelope field is refused by an older reader.

### Free wins

- `Session.first_live_seq` — written, read only by tests.
- `ph_rlm/subagents.py:406` `targetSeq` — records a *child's* seq into the
  *parent's* log. **Zero consumers**, and already ill-defined. A compound id
  fixes it.
- `ph-runtime-guest` — zero `seq` references.

---

## 5. Recommended design — reference-forking

### 5.1 The mechanism

A session file holds **only its own contiguous run of events**, plus a header
saying where the rest comes from:

```
header: { id, parent_session, seed_length, ... }     # both fields already exist
events: seq = seed_length, seed_length+1, ...        # this file's own work
```

Materialising a session:

```
materialise(s) = materialise(s.parent)[0 : s.seed_length]  ++  s.own_events
```

Recursion ends at a root (`parent_session is None`, `seed_length is None`).

**Forking** is: write a header naming the parent and the boundary, then append
your own work starting at `seq == boundary + 1`. Nothing is copied.

**Segmenting** is the same call with the boundary at the parent's tip — a fork
with no divergence. That is the "break a log into multiple files" case, and it
needs no separate mechanism.

### 5.2 Why `seq` survives untouched

| Property | Under reference-forking |
|---|---|
| `seq == index` in a materialised log | **holds** — ranges are disjoint and consecutive |
| `_readmit`'s contiguity gate | **unchanged** |
| `SurfaceManager`, `derive_messages` | **unchanged** — they see a materialised list |
| All 8 `SessionFoldCache` users | **unchanged** — they want length |
| Daemon paging / cursors | **unchanged** — offsets into a materialised list |
| `index_at_or_before`, live/replay dedupe | **unchanged** — one lineage, total order |
| Repair's `last.seq + 1` | **unchanged** |
| **Turso schema** | **unchanged** — `seq INTEGER PRIMARY KEY` still dense per file |

`seq` is unique **per lineage**, not per tree. Two siblings forked at boundary
`b` both start their own work at `b+1`; they live in different files and are never
materialised into the same list. Verified.

Cross-lineage references — a parent citing a subagent's event — are
`(session_id, seq)`. Both halves already exist. This also repairs
`ph_rlm/subagents.py:406`'s `targetSeq`, which today writes a *child's* seq into
the *parent's* log with no session qualifier and no consumers.

### 5.3 What actually has to change

**Small.** Only two places read a session file directly:

| Site | Change |
|---|---|
`ph/persistence/jsonl.py:155` `JsonlSessionStore.read` | follow the chain and concatenate. **This is the Protocol method**, so resume, `stored_survivors`, the daemon and everything else inherit it for free |
`ph_app/tui/trajectory_app.py:67` | calls `read_session(path)` directly, bypassing the Protocol — needs the chain read or a switch to the Protocol |

Plus:

- `SessionStore.fork` stops slicing (`store.py:230`) and instead records the
  reference. `_fork_seed`'s validation — the open-turn refusal, the contiguity
  check — is still wanted; only the copy goes.
- `TursoSessionStore.read` gets the same chain walk.
- Materialisation wants a cache. `SessionFoldCache` is the right shape and
  already carries the purity requirement.

`ph_rlm/kernel/journal.py` and `ph_rlm/harness/state.py` also call
`read_records`, but on the kernel journal and the global harness log — **not
session logs**. Unaffected.

### 5.4 The two real costs

**(a) Referential integrity.** Today a session file is self-contained; under
reference-forking it is not. Delete or move an ancestor and every descendant
becomes unreadable.

Mitigations, in order of preference:

- **Refuse to remove a session that has descendants.** The predicate already
  exists: `descendants()` in `ph/seams/subagents.py:1014` walks
  `(id, parent)` pairs, and `stored()` already surfaces `parent` on every
  `StoredSession` from a header peek both backends already pay for. So the check
  is available without new state.
- **Materialise-on-detach** for archival: collapse a lineage into one
  self-contained file when it is exported or pruned. Reference-forking makes this
  an option, not an obligation.
- **A broken chain must fail loudly**, naming the missing ancestor. A silent
  partial read would reconstruct a *wrong* session, which is exactly what
  `_readmit`'s unknown-type refusal exists to prevent.

**(b) Chain depth.** One file open per ancestor. Bytes read are 1.00× the
materialised size, so the cost is syscalls, not I/O volume. Worth a bound
(refuse or auto-collapse past depth N) mostly to keep a pathological
fork-of-fork-of-fork chain from turning one read into hundreds of opens.

### 5.5 What this does *not* deliver, and does not need to

Reference-forking gives you branching, composable logs and single-writer files.
It does **not** give a per-event tree, and for the stated goals it does not need
to. Specifically it does not support **two writers extending the same lineage** —
but that is not the goal: a second process that wants to continue from a point
*branches*, which under this design is a new file and therefore no conflict at
all. The partial-order problem in §5.4 of the earlier draft disappears, because
every materialised log is one lineage with one writer and a total order.

The one thing worth taking from tau regardless is in §5.6.

### 5.6 Still worth taking from tau

**Make `surface_op` replace an id-set rather than a positional range.** tau's
`replaces_entry_ids` is branch-safe by construction. §3 shows pH's resolution is
*already* membership-based — `nodes.index(op.start)` — so this is a small change
that attaches a real guarantee, and `source_event_seqs` already carries the full
shadowed set. Independent of reference-forking, and worth doing first.

## 6. Turso, settled empirically

| Approach | Works | Keeps clustering |
|---|---|---|
| `seq INTEGER PRIMARY KEY` (today) | yes | yes |
| `PRIMARY KEY (agent, entry)` | **yes** | no — needs `ORDER BY agent, entry` |
| `… WITHOUT ROWID` | **no** — *"experimental feature. Enable with --experimental-without-rowid"* | — |
| Packed `(agent << 32) \| entry` | **yes**, and `>> 32` / `& 0xFFFFFFFF` work in SQL | yes |

`INSERT OR REPLACE` stays idempotent on both compound and packed keys — the
property that makes Turso resume-safe survives either way.

Your instinct was right, and sharper than expected: the clean answer
(`WITHOUT ROWID` for a clustered compound key) is **unavailable**, so packing is
the only way to keep clustering with a compound key. One constraint: `agent` must
stay below 2³¹ or the packed value overflows signed 64-bit (measured).

**But under §5.1 none of this is needed** — `seq` stays the key and the compound
id rides in `wire`.

---

## 7. Suggested sequencing

Each step is independently useful and independently revertible.

1. **Make `surface_op` replace an id-set** (§5.6). Self-contained, buys
   branch-safety before any branching exists, and resolution is already
   membership-based so the change is small.
2. **Make `read` chain-aware** in both backends, while `fork` still copies.
   A no-op on today's data — every session has `parent_session is None` or a
   fully-copied prefix — so it can land and be exercised before anything depends
   on it. Add the loud refusal for a missing ancestor here.
3. **Point the trajectory view at the Protocol** (`trajectory_app.py:67`), so
   only one code path reads a session.
4. **Stop copying in `fork`.** Keep `_fork_seed`'s validation, drop the slice,
   write the reference. This is the step that delivers O(1) forks; steps 2–3 make
   it a two-line change.
5. **Guard deletion of a session with descendants** (§5.4a), using the existing
   `descendants()` walk over `StoredSession.parent`.
6. **Segmentation**, if wanted — roll to a new file at a size or age bound. Same
   mechanism as step 4 with the boundary at the tip.

Steps 1–4 deliver composable logs and O(1) forks. Nothing here requires per-event
tree identity, a compound key, or a Turso migration.

## 8. Open questions

- **Does the trajectory view want the lineage or the segment?** Today it reads
  one file and shows a whole session. Under reference-forking those differ, and
  "show me this branch's history" versus "show me what this file added" are both
  legitimate views.
- **Depth bound.** Is there a maximum chain length before a lineage is collapsed
  into a self-contained file? Purely a syscall-cost question (§5.4b), not
  correctness.
- **Does `fork` still need the open-turn refusal?** `_fork_seed` refuses a
  boundary inside an open turn (`store.py:225`). Under reference-forking the
  same rule should hold — a lineage that begins mid-turn is not resumable — but
  it is now a validation on a *reference* rather than on a copy.
- **Retention.** An abandoned branch still costs a file, though now a small one.
  `ph workspaces gc` is the existing precedent for age-bounded collection of
  exactly this shape, and step 5 gives it the "has descendants" predicate.
- **Does anything outside this repo read a session file directly?**
  `daemon/client.py:5` names dsh as a consumer. A chained file is not readable by
  a naive reader, and `SESSION_FORMAT_VERSION` is the mechanism for saying so —
  though note reference-forking adds no envelope field, so only *forked* sessions
  change shape.
