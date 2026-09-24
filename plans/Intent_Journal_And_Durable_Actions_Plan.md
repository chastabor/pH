# Actions the log can vouch for — declared types, an intent journal, tools that answer, batches

*Five design changes from `reviews/10-intent-journal-proposal.md`, turned into rows.
Phase 10 — making an action atomic and idempotent through the session log, once, for
every component, instead of once per pair of records.*

## Context

Verified against the source at `bdb99a1`. Line numbers are at that commit.

**What the question was.** `reviews/09` found that the write side of an action —
intent, effect, outcome, and "did it happen?" — was reimplemented per pair of records,
and that the copies disagreed on durability, on dedupe, and on whether repair knew
about them. `reviews/10` proposed five changes; the smallest fixes from `reviews/09`
§7 landed first, in `39c9426` and `bdb99a1`. This plan is what is left, row by row.

**What exists today, and is kept.**

- **The read side is sound and centralized.** `SessionFoldCache`
  (`ph/session/folds.py:67`) caches a pure fold per session, keyed on `seq`;
  `ph.testing.folds.check_fold_laws` (`ph/testing/folds.py:119`) holds each fold to
  determinism, purity and batching invariance; `contribute_fold_cache`
  (`ph/seams/invariants.py:230`) makes each cache pollable at runtime.
- **One append is atomic in memory.** `Session._commit` validates against the surface
  before it pushes (`ph/session/session.py`), and `Session.append` (`:412`) now refuses a
  type the read door would refuse (F11).
- **Plugins can declare types.** `declare_log_type(name, owner=, ignorable=)`
  (`ph/session/known_event_types.py:434`) refuses a ph-core name, a second owner, and a
  changed ignorability. **No production caller yet**: every shipped type is still in
  ph-core's frozen set, whoever writes it.
- **Persistence is honest.** Both backends write what the log holds past a cursor
  (`_Progress.cursor`), so nothing appended after a listener unwinds is missed;
  `write_on_unwind` (`ph/persistence/protocol.py:331`) is registered by each backend's
  `claim`, after the lease, so the mount's last act writes every live log;
  `AgentRegistry.dispose` (`ph/agent/registry.py:173`) writes an agent's log as it lets
  go. The JSONL write is all-or-nothing, and a torn final line is dropped by the reader
  and settled by the writer (`_settle_tail`, `ph/persistence/jsonl.py:226`).
- **Durability is placed per site, by hand.** Each write-ahead record now reaches disk
  before the effect it guards, but each site spells its own barrier:
  `checkpoint_policy.before_tool_body` (`ph/persistence/checkpoint_policy.py:66`) for
  tool calls and nested dispatches (`ToolRuntime.restore_covers`,
  `ph/tools/registry.py:710`); `_mutate` (`ph_app/daemon/server.py:483`) for daemon
  verbs; `run_shell` (`ph_app/shell.py:141`) for `!!`; `ApprovalService.request`
  (`ph/seams/approval.py:331`) and `UserQuestions.ask` for asks; `UploadRegistry.handle_for`
  for uploads. `session_written(ctx, session)` (`ph/session/store.py:515`) is the one
  non-raising flush they share.

**What is still per pair.**

| Pair | Written by | Keyed by | Folded by | Settled on crash by |
|---|---|---|---|---|
| `tool/call` → `tool/result` | `batch._append_call` (`ph/tools/batch.py:279`) | `callId` | `interrupted_turn_closers` | repair, in an open turn only |
| `tool/code-dispatch-start` → `tool/code-dispatch` | `DispatchBridge._log_start` (`ph/tools/code_mode.py:331`) | `subCallId` | readers each (`/revert`, limits, TUI) | **nothing** |
| `approval/asked` → `approval/decided` | `ApprovalService` | `callId or toolName` | `pending_approvals` (`ph/seams/approval.py:277`) | `_settled_asks` (`ph/persistence/repair.py:89`), open turn only |
| `question/asked` → `question/answered` | `UserQuestions` | `askId` | `pending_questions` (`ph/seams/user_questions.py:144`) | `_settled_asks`, open turn only |
| `shell/command` → `shell/result` | `run_shell` | `seq` / `commandSeq` | nothing | **nothing** |
| `client/command` → *(no outcome record)* | `Root.once` (`ph_app/daemon/supervisor.py:465`) | `clientId:commandId` | `Root.commands`, folded at start | **nothing**; dedupes on the intent, not the outcome |
| `workspace/acquired` → `workspace/disposed` | `WorkspaceSeam` | `agentId` | `workspace_leaks`, `workspace_survivors` (`ph/seams/workspace.py:1717`) | `WorkspaceSeam.reconcile` (`:1268`), on open |

Repair knows four kinds — the tool pair, the step, the turn, and the two asks — and acts
only inside an open turn (`interrupted_turn_closers`, `ph/persistence/repair.py:134`, F13).
`ToolDefinition` declares `is_irreversible`, `effects_confined_to_workspace` and
`is_concurrency_safe`, but not what its effect *is* or whether it happened, so repair
tells the model to work it out (`_OUTCOME_UNKNOWN_TEXT`). **There is no MCP bridge in the
repository**; the "an MCP server adds its own" argument in `definition.py` is about
tools a deployment brings, and Part 4 is built for them rather than for a shipped
consumer.

**Two corrections to `reviews/10`, found while writing this.**

1. **The nested-dispatch barrier is `unless-restorable`, not `if-irreversible`.** It
   landed as `ToolRuntime.restore_covers` — the rule `/revert` lists by — because a
   barrier keyed on `is_irreversible` alone flushed nothing for a tool that declares
   nothing, and that is most tools.
2. **Part 5 has fewer producers than claimed.** `input_offload` awaits `store.commit`
   between `offload/input-spilled` and its `user/message` replace
   (`ph_stabilize/input_offload.py:168-180`), deliberately: the spill record must
   precede the blob. The tool-result offload has the same two-phase shape. Neither can
   be a synchronous batch, and neither needs one — spill's reserve → append → commit is
   already repaired at open. The real producers are compaction's summary
   (`ph_stabilize/compaction.py:1267`) and its argument truncation (`:940-968`).

---

## Decided with the user (2026-09-24): no backwards compatibility

**This phase ships with a version bump, and nothing written or spoken before it has to
keep working after it.** Logs written at format 1 are refused by `SessionHeader`'s version
check (`_current_format_only`, `ph/session/session.py`) rather than migrated, as format-0
logs were at the bump to 1. A front end and a daemon from either side of the bump are not
expected to talk; `PROTOCOL_VERSION` (`ph_app/protocol.py:82`) records the move, and its
own docstring already says what a skew costs and that restarting the daemon fixes it.

**What that changed in this plan, rechecked row by row:**

| Where | Shaped by compatibility? | Now |
|---|---|---|
| Decision 5, P10-15 — how a batch is marked | **Yes.** The marker event was chosen to avoid a format bump | **An envelope field, and format 2** |
| Decision 4, P10-10 — `MutationRepeated.outcome` | **Partly.** A new wire field would have needed a default for older clients | **Required**, and `PROTOCOL_VERSION` moves to 4 |
| P10-05 — where a pair's key is read from | **Yes, as stated.** "Every log already on disk folds the same way" | Keys stay in the payloads, **for a different reason**: they are data their readers already use, and an envelope copy would be a second carrier of one fact |
| P10-08, P10-10 — new payload fields (`interrupted`) and types (`client/command-settled`) | No | Unchanged; they land with the bump like everything else |
| Decision 7, P10-03 — types a satellite writes | No. The reader in question is a build of *this* version without the bundle installed, which stays a real deployment | Unchanged |
| `ignorable` on new informational types | No. It is the vocabulary's rule for records a reader may skip, not a concession to old builds | Unchanged |

Everything format- or wire-breaking — P10-15's envelope field, P10-10's reply field — lands
in the one release that carries the bump, so each number moves once.

## Decisions to take with the user

Each has a recommendation, which the rows below assume. A row that depends on one says
so, and none of them is started until the decision is recorded here.

1. **Where the journal lives.** *Recommended:* `ctx.intents`, provided by the `session`
   row beside `ctx.sessions`. Every profile that has a log gets it, and it is the store's
   own concern — its barrier is the store's flush. *Alternative:* its own row, which lets
   a profile drop it and leaves every kind's producer to test for its absence.
2. **The orphan policy vocabulary.** *Recommended:* `outcome-unknown` (repair writes the
   settled record saying so), `not-started` (repair writes it as not started),
   `owner-settles` (repair leaves it; the owning seam reconciles on open, as
   `WorkspaceSeam.reconcile` already does). A `Literal`, per the types-over-`Any` rule.
3. **The barrier vocabulary.** *Recommended:* `durable` (the journal flushes before
   returning the claim, fail-closed), `buffered` (no flush; accounting), and
   `tools-execute` for the two tool kinds, whose barrier stays the checkpoint policy's
   because it must run after every pre-execute gate — the journal declares it and does not
   place it.
4. **What a repeated daemon verb says when its first attempt never finished.**
   *Recommended:* `MutationRepeated` gains a **required** `outcome: Literal["settled",
   "unknown"]`. Today a retry after a crash mid-act is refused as repeated with no word that
   the act may not have happened; `unknown` lets a client ask the person instead of
   assuming. Required rather than defaulted, since daemon and front ends ship together and
   no older client needs a fallback. `PROTOCOL_VERSION` moves to 4, with its docstring
   entry.
5. **How a batch is marked on disk.** *Recommended (revised 2026-09-24):* a `batch`
   field on the **envelope** of every member, `{"first": <seq>, "count": <n>}`, and
   `SESSION_FORMAT_VERSION` 2. Every member describes its own batch, so the completeness
   check is exact anywhere in a log and not only at its tail; no extra event takes a seq or
   needs a place in the renderers' record-less sets; and a reader can tell which records
   belong together. The earlier recommendation — an ignorable `session/batch` marker event —
   was chosen only to avoid the bump, which is no longer a cost.

   **What this is not: a rollback.** No batch mechanism undoes a committed event; the log is
   append-only (I4). A batch that fails *in the process* — a member the surface refuses, or
   an exception in the block — never reaches the log, because `Session.batch()` (P10-14)
   pushes nothing until every member has validated. A flush that *fails* part-way is already
   taken back by `_append_and_sync`. The envelope field answers the one case left: a process
   that dies while the flush carrying a batch is being written. Undoing earlier steps of a
   multi-step action when a later one fails is not a batch at all — it is Part 3's settle
   records plus a tool that can reverse its own effect, which only a workspace restore can do
   today.
6. **Runtime writer checks for posture types (P10-02).** *Recommended:* ship only if the
   appending row can be identified exactly from `running_for()`; otherwise the static
   gate of P10-01 is the whole answer, and that is written beside `SandboxMode` and
   `approval_policy` (rule 6).
7. **Types a satellite writes stay in ph-core's set.** *Recommended:* yes. The reader
   that refuses an unknown required type is ph-core's `_readmit`, and it must know
   `todo/write` or `harness/refined` whether or not the bundle that writes them is
   installed. `declare_log_type` is for types no ph-core reader needs to understand.

---

## Phase 10 rows

| Row | What it lands | Depends on | Part |
|---|---|---|---|
| P10-01 | **Landed.** Writers of record for ph-core's vocabulary, and one AST gate across every package | — | 1 |
| P10-02 | **Spiked, not shipped.** *Spike:* posture types refuse a writer that is not their owner | P10-01, decision 6 | 1 |
| P10-03 | **Not started — its condition is unmet.** *Conditional:* a `ph.log_types` entry-point group, so a reader knows a package's types unmounted | the first type declared outside ph-core | 1 |
| P10-04 | **Landed.** A lineage writes a parent only when the child inherits its prefix | — | 2 |
| P10-05 | **Landed.** `ph.session.intents`: kinds declared once, the pure fold, the fold laws | P10-01 | 3 |
| P10-06 | **Landed.** `ctx.intents`: record, open, settle, claim; the barrier; the dedupe index; the invariant row | P10-05, decisions 1–3 | 3 |
| P10-07 | **Landed.** Repair settles every declared kind, in a turn or out of one | P10-05 | 3 |
| P10-08 | **Landed.** `shell/command` → `shell/result`: the first kind, end to end | P10-06, P10-07 | 3 |
| P10-09 | **Landed.** Approvals and questions become kinds; `_settled_asks` goes | P10-07 | 3 |
| P10-10 | **Landed.** Daemon verbs: `client/command` settles; a key lives as long as its effect (L1) | P10-06, decision 4 | 3 |
| P10-11 | **Landed** (dispatch; workspace kept on its own fold, see row). Code Mode dispatches and workspace trees as kinds | P10-06, P10-07 | 3 |
| P10-12 | **Landed.** Tools name their own effect: `idempotency_key`, and a key for a far side | P10-06 | 4 |
| P10-13 | **Landed.** Tools answer "did I happen?": `reconcile`, asked on resume | P10-07, P10-12 | 4 |
| P10-14 | **Landed.** `Session.batch()`: planned as a unit, published as a unit, adopted by compaction | — | 5 |
| P10-15 | **Landed.** Batch membership on the envelope; a batch reads back whole or not at all (format 2) | P10-14, decision 5 | 5 |
| P10-16 | **Landed.** Docs, non-guarantees, `Implementation_Plan.md` §4 and §6, `reviews/` status | all | — |

**Order.** P10-01, P10-04 and P10-14 are independent and small; any can go first. Then
P10-05 → P10-06 → P10-07 → **P10-08**, which is the row that proves the API on the
simplest pair before anything with a wire contract moves. P10-09 to P10-11 follow in any
order. Part 4 needs the journal; Part 5 does not. P10-16 is last.

---

## Part 1 — the vocabulary names its writers

## P10-01 — writers of record for ph-core's vocabulary, and one gate across every package

*(Landed. 74 types, each with at least one writer; the gate is its own module,
`packages/ph-core/tests/test_log_writers.py`, not `test_session_append.py`, since the walk
and its variable-site list are more than a vocabulary test. The table is keyed by the
module whose code calls `append` — lexical, not the running row — and is built from a
module → types map so each writer's list reads in one place. Three variable sites exist,
not the two guessed below: `ph.agent_loop.driver`'s `Inbox.append(target, …)` (not a log
append; listed as writing nothing), `ph.llm.media`'s `event_type`, and
`ph.testing.builders`' `workspace_log` scaffolding. The Continual Harness's records are
imported constants, which the walk resolves. Sabotaged four ways — a stray `sandbox/mode`,
an unknown type, a new variable site, and a writer that stopped writing — each failed the
gate that names it.)*

**Why.** `test_every_appended_type_is_a_known_event_type`
(`packages/ph-core/tests/test_session_append.py:178`) is a regex over *literal*
`.append("x/y"` calls in *ph-core*. It cannot see a constant (`COMMAND_ACCEPTED`,
`DISPOSED`, `TICK`), and it cannot see the four other packages that append —
`ph-stabilize`, `ph-rlm`, `phern`, and whatever comes next. Since F11 the write door
refuses an unknown type at runtime, so what is left for a gate is the question the
runtime cannot answer: **who is allowed to write each type.** Five namespaces are
written from more than one package today — the three surface types among them, which
compaction rewrites — and some types have two writers on purpose (`todo/write`, from two
`ph-stabilize` rows). A table that says so is the difference between "deliberately
shared" and "nobody noticed".

**What.** A `WRITERS: Mapping[str, frozenset[str]]` beside
`KNOWN_SESSION_EVENT_TYPES` — type to the module names that may append it — with each
multi-writer entry carrying its reason in a comment, the way the vocabulary already does.
`declare_log_type` already records one `owner` per plugin type; its table joins the same
check. The gate is `reviews/probes/writer_matrix.py` promoted into ph-core's test suite:
an AST walk of every shipped module in every package that resolves literals, module
constants and constants imported by name, and asserts that every append site's type is
known **and** that the appending module is among its writers.

**Files:** `ph/session/known_event_types.py`, `packages/ph-core/tests/test_session_append.py`
(the regex test is replaced, not kept beside), `tests/workspace_layout.py` (the package
walk the other workspace gates already use).

**Gates** (`test_session_append.py`):
`test_every_append_in_every_package_names_a_known_type` ·
`test_every_append_site_is_a_writer_of_record` ·
`test_every_writer_of_record_still_appends_the_type` (the reverse direction, so a
writer that stopped writing is removed rather than left to widen the table).
*Sabotage:* append `sandbox/mode` from `ph_stabilize.hitl` and the second gate names the
module and the type.

**Not enforced (rule 6):** a `session.append(kind, …)` whose `kind` is a variable is
invisible to the walk. The gate lists such sites and fails when one is added, so a new one
is a decision rather than a leak. Those that exist today are named in the test —
`ph/llm/media.py`'s `append(event_type, …)` for the two `attachment/*` records, and the
Continual Harness's outcome records passed by name.

## P10-02 — *spike:* posture types refuse a writer that is not their owner

*(Spiked; **not shipped**, per decision 6. The running row is not known at every
legitimate append: the daemon's preset verb (`_act_preset`,
`packages/phern/src/ph_app/daemon/server.py`) calls `apply_preset` on the connection's task
with no activation bound, so `Context.current_owner()` is `None` there and a check would
have to answer "allowed" when it cannot tell. And where a row *is* running, it is the wrong
one for the table: `sandbox/mode` from a preset is appended while `permission-presets` is
the caller, but the code that appends it is `ph.seams.sandbox`'s — P10-01's table is
lexical, and a runtime check would need a second, dynamic table held in step with it. The
static gate is the whole answer; it is written beside `SandboxSeam.logged_mode` and
`approval_policy`.)*

**Why.** F12. `sandbox/mode`, `approval/policy`, `approval/mode` and `permission/preset`
are read as the posture in force (`logged_mode`, `approval_policy`, `hitl`,
`PermissionPresetService`). A static gate (P10-01) catches a shipped writer; it does not
catch a third-party row that appends one at runtime.

**What.** For those four types only — a `frozenset` lookup on every append, then a
`running_for()` read on these alone — `Session.append` compares the running row's module
against `WRITERS`. **The spike is whether the running row is always known at the append.**
Repair appends outside any row; `admit` must not check (a replica owns nothing); a host
appends through `open_session`. If every legitimate path can be named exactly, ship it. If
any would need a fallback that answers "allowed" when it cannot tell, do not ship it —
record the static gate as the whole answer beside `SandboxMode` and `approval_policy`.

**Files:** `ph/session/session.py`, `ph/session/known_event_types.py`,
`packages/ph-core/tests/test_session_append.py`.

**Gates:** `test_a_posture_type_from_a_row_that_does_not_own_it_is_refused` ·
`test_the_owner_and_repair_may_write_posture` · `test_a_replica_admits_posture_unchecked`.

## P10-03 — *conditional:* readers learn a package's types without mounting it

**Why.** A type declared with `declare_log_type` is known only once its declaring module
is imported. `phern --mode trajectory` opens a stored log with nothing mounted, so a
*required* plugin type is refused there even when the package is installed.

**What.** An entry-point group, `ph.log_types`, naming the modules that declare types;
`is_known` imports them once, lazily, on the first miss. **Lands with the first type
declared outside ph-core**, and not before — per decision 7 there is none today, and a
mechanism with no user is the shape rule 6 exists to forbid.

**Gates:** `test_a_declared_type_is_known_to_a_reader_that_never_mounted_its_row`.

---

## Part 2 — honest persistence *(landed in `39c9426`, `bdb99a1`; one residual row)*

## P10-04 — a lineage writes a parent only when the child inherits its prefix

*(Landed. The test is `seed_length` truthy rather than `is not None`: a seed of zero
references nothing, so it needs no ancestor either. The fork ordering test is unchanged
and still holds; `test_a_segment_still_writes_its_parent_first` adds `roll`, the other
child that references a prefix. Sabotaged both ways — the old walk fails the subagent
test, a walk that never follows a parent fails both prefix tests.)*

**Why.** `SessionStore.lineage` (`ph/session/store.py:287`) follows `parent_session` for
every child, so every flush of a subagent child also flushes its parent. A subagent
child is created fresh — `durable_length == 0`, no `seed_length` — and does not need its
parent's prefix to be readable. Under the `rlm` profile the nested-dispatch barrier makes
that one extra parent fsync per child tool call that can reach past the tree. The
ancestors-first rule exists for reference forks and segments, whose file is unreadable
without the parent's.

**What.** `lineage` includes a parent only when the child's header records an inherited
prefix (`seed_length is not None`). `write_on_unwind` keeps its order, since it walks
`lineage` too.

**Files:** `ph/session/store.py`; tests beside `test_a_child_is_never_durable_before_the_prefix_it_references`
(`packages/ph-core/tests/test_persistence.py`).

**Gates:** `test_flushing_a_subagent_child_does_not_write_its_parent` ·
`test_a_reference_fork_still_writes_its_parent_first` (the existing ordering test,
re-asserted).

---

## Part 3 — the intent journal

## P10-05 — `ph.session.intents`: kinds declared once, the pure fold, the fold laws

*(Landed. **One fold, `fold_intents`** — per key, the latest intent opened under it and
its settle (`IntentRecord`), with `since=` as the `extend` — and `open_intents` and
`settled_record` are both read off it, from a log or from the journal's cached index, so
"open" and "settled" cannot drift apart. The index keeps settled keys too, which is what
P10-06's dedupe needs, and hands back the prefix's value unchanged when a slice holds
nothing of the kind, so a cache read per model step costs the slice. `closer` takes the
reason, `closer(opened, why)` with `why: Unsettled` (`"outcome-unknown" | "not-started"`),
because the journal writes settles too: `not-started` on a failed barrier,
`outcome-unknown` when a claimed body raises. `declared_intents()` is what repair will
walk; `is_declared(kind)` is the journal's check. `declare_intent` refuses one thing more
than listed — a type that both opens and settles. The rule is `pending_approvals'`
exactly, and `test_the_fold_agrees_with_the_folds_it_replaces` holds it to both the
approval and question folds over the same logs, which is P10-09's licence. The fold-law
gate lives in `test_intents.py` beside the fold rather than in `test_fold_laws.py`.
Sabotaged three ways — first-open-wins, `since` ignored, and a settle blind to a re-open
— each failed its gate.)*

**Why.** Seven pairs, seven folds, three settlers. Each existing fold is correct; what
is missing is one statement of "an intent this log opened and never settled" that repair,
the dedupe index and every reader share.

**What.** A module beside `folds.py`, with no service yet — declarations and pure
functions over a `Sequence[SessionEvent]`, callable on a stored log with nothing mounted,
for the reason `folds.py` gives for keeping folds off `Session`.

```python
IntentOrphan: TypeAlias = Literal["outcome-unknown", "not-started", "owner-settles"]
Barrier: TypeAlias = Literal["durable", "buffered", "tools-execute"]

@dataclass(frozen=True, slots=True)
class IntentKind:
    opened: str                                       # "shell/command"
    settled: str                                      # "shell/result"
    opened_key: Callable[[SessionEvent], str | None]  # read off the record, as each fold does today
    settled_key: Callable[[SessionEvent], str | None]
    orphan: IntentOrphan
    barrier: Barrier = "durable"
    closer: Callable[[SessionEvent], JsonObject] | None = None  # required unless owner-settles
    owner: str = ""

def declare_intent(kind: IntentKind) -> IntentKind: ...  # at import; both types must be known (P10-01)
def open_intents(events: Sequence[SessionEvent], kind: IntentKind) -> tuple[OpenIntent, ...]: ...
def settled_record(events: Sequence[SessionEvent], kind: IntentKind, key: str) -> SessionEvent | None: ...
```

**Keys stay where they are.** A kind reads its key off the records the way
`pending_approvals` (`callId or toolName`) and `pending_questions` (`askId`) read theirs
now, because those fields are data their readers already use — the TUI pairs dispatch cards
by `subCallId`, `/revert` lists by `parentCallId`. P10-15's format bump would allow an
envelope `intent` key as well, and it would be a second carrier of the same fact, which this
codebase refuses; so the bump does not make it better, and it is not done.

`declare_intent` refuses a second kind on one `opened` type, a kind whose types are not
in the vocabulary, and an `orphan` that is not `owner-settles` with no `closer`.

**Files:** `ph/session/intents.py` (new), `ph/session/__init__.py`,
`packages/ph-core/tests/test_intents.py` (new), `packages/ph-core/tests/test_fold_laws.py`.

**Gates:** `test_an_opened_intent_with_no_settle_is_open` ·
`test_a_settle_closes_only_its_own_key` · `test_a_kind_on_an_unknown_type_is_refused` ·
`test_open_intents_obeys_the_fold_laws` (`check_fold_laws`, over logs each migrated
kind's own producer wrote — the list grows with P10-08 to P10-11).

## P10-06 — `ctx.intents`: record, open, settle, claim, and the barrier

*(Landed, in `ph/session/journal.py` (new) rather than `intents.py`, which stays pure.
`IntentJournal` holds the store it is provided beside and uses `SessionStore.written` as
the barrier: `session_written` lives in `store.py`, which imports the journal to provide
it, so importing it back would be a cycle. Beyond the sketch: the journal **refuses an
undeclared kind** (repair settles only declared ones, so its orphans would never close)
and **an empty key** (it would pair every keyless record); `record` returns a `Claim`
rather than the event, so a `tools-execute` producer settles through the same method;
`settle` refuses a key that is not the claim's and an intent no longer open; `is_open`
and `forget` (on `session/disposed`) are public; `claim` settles
`{**closer(opened, "outcome-unknown"), "failed": true}` on any `BaseException`,
cancellation included, and a body that returns without settling leaves the intent open
for its owner. One `SessionFoldCache` per kind over `fold_intents`, reported as
`intent-fold-cache` with the kind named in each finding; the base-profile roster test
lists the row. Two extra gates: `test_a_settle_closes_only_its_own_intent_once`,
`test_what_the_journal_refuses_to_open`. Sabotaged four ways — no flush, no dedupe, a
claim that does not settle, a silent `stale` — each failed its gate.)*

**Why.** The seven sites each spell their own append, flush and failure. The journal is
where that spelling lives once, so a new component that must record before an effect gets
it right by calling one method.

**What.** `IntentJournal`, provided as `ctx.intents` by the `session` row (decision 1).

```python
class IntentJournal:
    def record(self, session: Session, kind: IntentKind, data: JsonObject) -> SessionEvent: ...
    async def open(self, session: Session, kind: IntentKind, data: JsonObject) -> Claim | Prior: ...
    def settle(self, session: Session, claim: Claim, data: JsonObject) -> SessionEvent: ...
    @asynccontextmanager
    async def claim(self, session, kind, data) -> AsyncIterator[Claim | Prior]: ...
    def pending(self, session: Session, kind: IntentKind) -> tuple[OpenIntent, ...]: ...
    def outcome(self, session: Session, kind: IntentKind, key: str) -> SessionEvent | None: ...
```

- **`open`** dedupes first: a key already opened in this log returns `Prior(opened,
  settled | None)` and nothing is appended. Then it appends `kind.opened`, and for a
  `durable` kind awaits the flush — fail-closed, as the checkpoint policy's barriers are:
  if the write fails, the claim is not handed out and the intent is settled `not-started`
  in memory, so the next flush writes an honest pair.
- **`record`** is `open` without the barrier and without dedupe, for the `tools-execute`
  kinds, whose flush the checkpoint policy places after the pre-execute gates.
- **`claim`** settles with `{"failed": true, …}` on an exception, the way `step/end` is
  written in a `finally`, so a caller cannot leave a pair open by raising.
- **The dedupe index and `pending`** are one `SessionFoldCache` over `open_intents`, with an
  `extend`, contributed through `contribute_fold_cache` as `intent-fold-cache`, so the daemon
  poll reports drift the way it does for the other six caches.
- **`session_written` is the barrier's flush.** It stays the one non-raising flush; `open`
  treats its `False` as the failure.

**Files:** `ph/session/intents.py`, `ph/session/store.py` (the `session` row's `apply`),
`ph/keys.py` (`INTENTS`), `ph/seams/invariants.py` (the row), `packages/ph-core/tests/test_intents.py`.

**Gates:** `test_open_is_on_disk_before_the_claim_is_handed_out` ·
`test_a_failed_barrier_hands_out_no_claim_and_closes_the_pair` ·
`test_a_second_open_of_one_key_returns_the_prior_and_appends_nothing` ·
`test_claim_settles_as_failed_when_the_body_raises` ·
`test_the_intent_cache_is_polled_as_an_invariant`.
*Sabotage:* remove the flush from `open` and the first gate reads the store and finds
nothing.

## P10-07 — repair settles every declared kind, in a turn or out of one

*(Landed. `_settled_intents` walks `declared_intents()` and writes each open intent's
settle with the kind's own `closer(opened, orphan)`. Skipped: `owner-settles`. A kind whose settle is **surface-eligible** is refused by
`declare_intent` itself (it was first skipped here; the refusal is the deeper place), since
a model-visible closer needs the surface metadata and provider rules only the turn repair
has. Each closer is probed against the kind's `settled_key` and a mismatch raises
`IntentError`: a closer that does not settle its own key would reopen on every resume and
grow the log each time. The asks stay inside the turn check until P10-09 moves them onto
kinds. Not enforced, and said in the module: a kind is settled only if its declaring
module is imported in the resuming process. Three extra gates: the unit form of the
out-of-turn test, `test_a_kind_settled_on_the_surface_is_refused` (now in `test_intents.py`),
`test_a_closer_that_does_not_settle_its_own_key_is_refused`. Sabotaged four ways — early
return for a balanced turn, no surface skip, no owner skip, no key probe — each failed
its gate; the owner-settles test first passed its sabotage, because its kind had no
closer, and now carries one.)*

**Why.** F13. `interrupted_turn_closers` returns `[]` when no turn is open, so an
orphaned `shell/command` — which is always outside a turn — is never settled, and every
future reader is told a command is running. The module argues that "a registry for two is
indirection with no second reader"; with P10-05 there are seven.

**What.** `interrupted_turn_closers` gains one pass over every declared kind that is not
`owner-settles` and is not the provider-facing tool pair, whose repair stays as it is —
it answers a different requirement (a `tool_use` block with no `tool_result` is rejected
by providers). The pass runs **whether or not a turn is open**. Closers keep today's
rules: seqs continue the log, the timestamp is the last real event's, and the order is
asks and intents first, then tool results, then `step/end`, then `turn/end`.
`session/resumed.closed` already counts them.

**Files:** `ph/persistence/repair.py`, `packages/ph-core/tests/test_repair.py`.

**Gates:** `test_an_orphan_outside_any_turn_is_settled_on_resume` ·
`test_an_owner_settles_kind_is_left_for_its_owner` ·
`test_a_balanced_log_still_resumes_with_no_closers` (a clean reopen must not grow the log)
· `test_closers_are_deterministic_and_backdated`.

## P10-08 — `shell/command` → `shell/result`: the first kind, end to end

*(Landed. `SHELL_COMMAND` in `ph.seams.shell` is keyed by the command's **own seq** —
nothing else about a command is unique — and settled by `commandSeq`, which the result
already carried, so no payload changed shape. The closer writes `{commandSeq, ok: false,
interrupted: why}` with `interrupted` the **reason** (`outcome-unknown` | `not-started`)
rather than a bare `true`, so a card can say which half is known. `run_shell` is a
`ctx.intents.claim`, its flush deleted; a barrier that fails now raises
`IntentNotDurable` rather than the backend's exception, and the command still does not
run. The rendering is in `shell_body` (`INTERRUPTED`), which both TUI front ends already
draw a card from; `trajectory.py` needed no change, since its generic line already prints
`interrupted=…`. Two things the plan did not foresee: **repair imports `ph.seams.shell`**,
so ph-core's kinds are declared wherever repair runs, and the module says the rest is not
enforced; and **P10-01's gate learned the journal** — its two `kind.opened` /
`kind.settled` sites are listed as writing nothing of their own, and the walk reads each
`IntentKind(opened=…, settled=…)` as a write by the declaring module, so
`WRITERS["shell/*"]` is `ph.seams.shell` now, not `ph_app.shell`. The test fixtures that
swap the kind table go through `ph.testing.isolated_intent_kinds`, which declares ph-core's
kinds into the real table first, so a first import under a swapped table cannot lose one. The resume gate is in `packages/ph-core/tests/test_repair.py`
(it needs no daemon), the drawing gate in `packages/phern/tests/test_tui_adapter.py`, fed
the closer repair actually writes; `test_the_shell_kind_obeys_the_fold_laws` joins the
fold-law list. Sabotaged three ways — a `buffered` barrier, a closer without
`commandSeq`, a renderer without the line — each failed its gate.)*

**Why.** The simplest real pair and the one with no settler today: no wire contract,
one writer (`run_shell`), one reader shape, and a claim in the vocabulary — "one that takes
the daemon down with it still shows in the log what was started" — that is only half true
until repair settles it. It proves the API before anything a client depends on moves.

**What.** `SHELL = declare_intent(...)` with `orphan="outcome-unknown"` and a closer
writing `shell/result {commandSeq, ok: false, interrupted: true, …}`. `run_shell` becomes a
`ctx.intents.claim(...)`; the explicit flush added in `39c9426` is deleted, because the
kind's barrier does it. The two TUI renderers and `trajectory.py` read `interrupted` on a
result.

**Files:** `ph_app/shell.py`, `ph/seams/shell.py` (the declaration: the seam owns the pair),
`ph_app/tui/adapter.py`, `ph_app/tui/trajectory.py`, `packages/phern/tests/test_daemon_shell.py`,
`packages/phern/tests/test_trajectory.py`.

**Gates:** `test_the_command_is_on_disk_before_it_runs` (the existing gate, now held by the
kind) · `test_a_command_the_daemon_died_during_is_settled_on_resume` ·
`test_an_interrupted_command_is_drawn_as_interrupted`.

## P10-09 — approvals and questions become kinds; `_settled_asks` goes

*(Landed. `APPROVAL_ASK` and `QUESTION_ASK` — `APPROVAL` is already the service key — both
`orphan="outcome-unknown"` as planned: the person may have answered on a screen whose
answer never reached the log. Each closer maps the reason to today's words, so no payload
changed: `not-started` (the barrier failed, nobody was asked) writes `unavailable` /
`resolution: failed`, and repair's `outcome-unknown` writes `INTERRUPTED, automatic` /
`interrupted: true`. `pending_approvals` and `pending_questions` are one comprehension
each over `open_intents`; `_settled_asks` is gone, and repair orders kinds **by type**, so
closers do not depend on import order. What the migration found, each now a gate:

- **Asks reuse keys.** An approval is keyed by tool name when it has no call id, and a
  question re-posed after a resume keeps its id, so `IntentKind` gained `dedupe` (default
  true; false for both asks) — `test_an_ask_under_a_key_asked_before_is_put_again`. And
  concurrent asks of one tool share a key, so the second open replaces the first in the
  fold, as it always did; the journal therefore holds a non-deduping kind's settle to its
  key alone rather than raising from the K7 `finally` (the ph-rlm concurrent-writes test
  found it).
- **A cancellation during the barrier write** closes the pair `not-started` and still
  propagates — `test_a_barrier_canceled_mid_write_still_closes_the_pair`.
- **A question canceled mid-answer stays pending**, so a passivated root re-poses it: the
  question seam uses `open` and an explicit settle, not `claim` (`test_ask_user` pins it).
  Approvals keep the K7 `finally`, settling through the journal.
- **Seams on a bare `Context`** get `intents_of(ctx)`: `ctx.intents`, or a store-less
  journal with no barrier — `session_written`'s reading of the same case.
  `IntentJournal.sessions` is optional for it.
- **Repair imports the declaring seams inside `_kinds()`, not at module top**:
  `ph.seams.shell` reaches `ph.orphans`, which imports `ph.persistence`, so the top-level
  import was a cycle for any process importing `ph.orphans` first (the ph-rlm lifecycle
  host). `test_importing_repair_declares_every_core_kind` probes a fresh interpreter that
  imports `ph.orphans` first; `test_repair_no_longer_knows_the_ask_shapes` holds that
  repair reads nothing of those seams.
- `IntentKind`'s three callables are classified `UNBOUND` in the registration-ownership
  gate: pure reads of a record, called with nothing mounted.

The six P5-13 settlement tests pass unchanged. Sabotaged four ways — an approval closer
that always writes `unavailable`, `dedupe=True` on each ask, an unhandled cancellation in
the barrier, a module-level declaring import — each failed its gate.)*

**Why.** The two asks already work like kinds. Making them kinds deletes the special
case in repair and moves the barrier added in `39c9426` into the declaration.

**What.** `APPROVAL` (`orphan="outcome-unknown"`, closer writing `outcome: INTERRUPTED,
automatic: true` — today's text) and `QUESTION` (closer writing `interrupted: true`).
`pending_approvals` and `pending_questions` stay as the seams' public folds, now one line
each over `open_intents`, and repair stops importing them. The K7 `finally` in
`ApprovalService.request` stays — it is the in-process half, and `claim` provides the
same shape.

**Files:** `ph/seams/approval.py`, `ph/seams/user_questions.py`, `ph/persistence/repair.py`,
`packages/ph-core/tests/test_repair.py`, `packages/ph-core/tests/test_seams.py`.

**Gates:** the six existing P5-13 settlement tests, unchanged ·
`test_repair_no_longer_knows_the_ask_shapes` (repair imports no seam).

## P10-10 — daemon verbs: `client/command` settles, and a key lives as long as its effect

*(Landed. `CLIENT_COMMAND` is declared in `ph_app.daemon.supervisor`; `Root.once`,
`remember`, `accepted` and `commands` are gone, and `_mutate` is a `ctx.intents.claim`
settled with `{command, outcome: "settled"}` after `act`. `client/command-settled` is in
the vocabulary, ignorable, and rendered by the auditor and not the transcript.
`MutationRepeated.outcome: RepeatOutcome` is required and `PROTOCOL_VERSION` is 4, with its
docstring entry. Three corrections to the sketch:

- **The kind is `buffered`, and the post-act flush in `_mutate` stays.** A `durable`
  barrier would flush the key *before* the act — the ordering P5-02 refused, since a
  crash before the act would then refuse a retry for work that never began — and the
  flush the plan said would go is F7's, *after* the act and before the reply, which no
  pre-act barrier replaces. So a key still reaches disk with its act's own records, and
  one that did so with no settle after it is the open intent repair settles `unknown`.
- **`claim` gives the act that raises a word too**: its key is settled `unknown`
  (`failed`), so the retry is told the outcome is unknown rather than a bare repeat —
  `test_an_act_that_raises_leaves_its_key_unknown`.
- **The keyword is `key_scope`, not `scope`**, which in this codebase means a `Context`
  lifetime — the registration-ownership gate refused `scope=` on the journal. `IntentScope`
  is `Literal["log", "process"]`; a `process` key carries `"scope": "process"` on its
  record and is no prior before `session.first_live_seq`. The `credentials/store` row
  passes it (`Mutation.key_scope`).

The TUI's remote slash command says so when a re-send's outcome is unknown
(`UNKNOWN_REPEAT`), and `test_daemon`'s vocabulary check reads the kind's types. Gates:
the existing same-key test now asserts `outcome == "settled"` for every row;
`test_a_retry_after_a_crash_mid_act_is_told_the_outcome_is_unknown` resumes a crashed log
in a second daemon over the same home; `test_a_credential_re_sent_after_a_restart_is_stored_again`;
`test_a_process_scoped_key_is_no_prior_to_the_next_process` (ph-core);
`test_a_re_sent_command_whose_outcome_is_unknown_says_so`; the payload test refuses a
repeat with no outcome. Sabotaged four ways — no process scope, an outcome that always
says `settled`, no dedupe, a `process` key that lives forever — each failed its gates.)*

**Why.** `Root.once` dedupes on the *intent*. A retry after a crash mid-act is refused as
repeated with no word that the act may not have happened. And L1: `credentials/store`'s
key is durable while its effect — a value in process memory — is not, so after a restart
a re-send is refused and the value never comes back.

**What.**
- `CLIENT_COMMAND` pairs `client/command` with a new ignorable `client/command-settled`,
  written after `act` returns. `Root.once` / `remember` / `accepted` / `commands` are
  replaced by `ctx.intents.open`, and the flush added to `_mutate` in `39c9426` goes with
  them.
- A retry finding a **settled** prior gets `MutationRepeated` as today. One finding an
  **open** prior — which, after P10-07, only happens within the process that opened it —
  gets `MutationRepeated(outcome="unknown")` (decision 4). One finding a prior that
  **repair** settled as unknown also gets `outcome="unknown"`, which is the case that
  matters.
- **A key's lifetime follows its effect.** `open(..., scope="process")` records
  `"scope": "process"` on the opened record, and the dedupe index ignores such keys from
  before `session.first_live_seq` (`ph/session/session.py:321`) — "the first seq appended
  in this process". `credentials/store` opens with it; every other verb does not.

**Files:** `ph_app/daemon/server.py` (`_mutate`, `MUTATIONS` at `:1075`),
`ph_app/daemon/supervisor.py`, `ph_app/payloads.py` (`MutationRepeated`, `:224`),
`ph_app/tui/remote.py`, `ph_app/daemon/client.py`, the vocabulary,
`packages/phern/tests/test_daemon_mutations.py`, `packages/phern/tests/test_payloads.py`,
`packages/ph-core/tests/test_wire_forms.py`.

**Gates:** `test_the_same_key_twice_acts_once_and_says_so` (existing, every row) ·
`test_a_retry_after_a_crash_mid_act_is_told_the_outcome_is_unknown` ·
`test_a_credential_re_sent_after_a_restart_is_stored_again` ·
`test_a_verb_is_on_disk_before_it_is_answered` (existing, now held by the kind).

## P10-11 — Code Mode dispatches and workspace trees as kinds

*(Landed for the dispatch; **the workspace pair is deliberately not migrated**.
`TOOL_DISPATCH` in `ph.tools.code_mode` — keyed by `subCallId`, `tools-execute`,
`outcome-unknown` — and `_log_start` is `ctx.intents.record`, its `Claim` held on the
bridge by sub-call id until `_log_settle` settles it through the journal. A dispatch
refused before it started (a denial, an approval that said no) has no intent, and its
settle record is appended alone as before. The closer copies the start's
`CodeDispatchRef` identity and writes `isError`, `interrupted` and a text body
(`DISPATCH_INTERRUPTED`), so the TUI's existing handler draws it with no adapter change.
Repair's `_kinds()` imports `code_mode` beside the three seams. The checkpoint policy's
nested branch is unchanged in code — "reading the kind's barrier" there would compare a
constant with itself — and its docstring names `TOOL_DISPATCH` as the kind whose barrier it
places.

**Why the workspace pair stays `workspace_survivors`'**: its openness is folded **from
`seed_length`** (a fork's inherited acquires are the parent's, not the child's) and only
over tiers with a fresh root; a key-pair fold from seq 0 would report a fork's parent's
live trees as the child's leaks, and `reclaim` deletes what that fold says. The docstring
of `workspace_leaks` already forbids a second implementation for that reason (A11), and
repair leaves an undeclared pair alone exactly as it leaves an `owner-settles` one — so
declaring it would add a second, wrong statement of openness and change nothing repair
does. `WorkspaceSeam.reconcile` stays the settler.

Gates: `test_an_orphaned_dispatch_is_settled_by_repair` (the crash-point log is the one
the dispatched tool's body sees, taken there), `test_an_orphaned_dispatch_is_drawn_settled`
(phern, fed repair's closer), `test_revert_still_lists_an_orphaned_dispatch_as_not_undone`;
the existing reconcile tests stand as they are. Sabotaged three ways — a closer that is
not an error, repair forgetting `code_mode`, a `durable` dispatch kind — each failed its
gates.)*

**Why.** An orphaned `tool/code-dispatch-start` is never settled, so after a crash the TUI
draws a dispatch card that is running forever, and each of its three readers keys the pair
its own way (`CodeDispatchRef` exists so they cannot drift; nothing folds them). The
workspace pair is already reconciled on open, by its owner, through `workspace_leaks`.

**What.**
- `TOOL_DISPATCH`: `barrier="tools-execute"`, `orphan="outcome-unknown"`, keyed by
  `subCallId`. `_log_start` becomes `ctx.intents.record`. The checkpoint policy's nested
  branch reads the kind's barrier and keeps `restore_covers` as the rule.
- `WORKSPACE`: `orphan="owner-settles"`. `workspace_leaks` becomes `pending(WORKSPACE)`,
  and `WorkspaceSeam.reconcile` stays the settler, because it alone may reclaim a tree and
  refuses one that belongs to a tier not mounted here.
- **Not migrated here:** `tool/call`, whose repair is the provider-facing special case
  P10-07 keeps; `workspace/checkpoint`, which is a fact rather than an intent — a restore
  point is valid whether or not its run finished.

**Files:** `ph/tools/code_mode.py`, `ph/persistence/checkpoint_policy.py`,
`ph/seams/workspace.py`, `ph_app/tui/adapter.py`,
`packages/ph-core/tests/test_code_mode.py`, `packages/ph-core/tests/test_workspace_reconcile.py`.

**Gates:** `test_an_orphaned_dispatch_is_settled_and_drawn_settled` ·
`test_revert_still_lists_an_orphaned_dispatch_as_not_undone` ·
`test_a_leak_is_a_pending_workspace_intent` (the existing reconcile tests, over the new fold).

---

## Part 4 — tools vouch for their own effects

## P10-12 — tools name their own effect: `idempotency_key`, and a key for a far side

*(Landed. `ToolDefinition.idempotency_key` (through `define_tool`) is called with the
arguments that will run, as plain JSON; `None` or a raise means unkeyed, the raise logged.
`TOOL_EFFECT` — declared in `ph.tools.registry` over two new **required** types,
`tool/effect` and `tool/effect-settled` — is opened in `ToolRuntime.dispatch`, **before**
the `tools/execute` waterfall, so the checkpoint policy's barrier carries it to disk before
the body; `batch.py` and `execute` both reach it through `dispatch`, so no batch change was
needed. The settle holds the result's content, `isError`, and an `outcome`:

- a prior settled `settled` is answered from the log — its content, `meta: {repeated,
  repeatOf}` — and the tool does not run;
- a prior that **failed** runs again with no note: the tool reported on its own far side;
- a prior open or repaired to `unknown` runs again under a new intent with
  `EFFECT_MAY_HAVE_HAPPENED` appended — the branch P10-13's `reconcile` will take first.

`ToolRunContext.idempotency_key` is `{session}/{call_id}` (a Code Mode dispatch's call id
already carries `:code:{n}`). `ph.testing.external_tool` returns a keyed tool and its
`FarSide` counter. No shipped tool declares a key, said on the field. Three repo gates
found three things beside the row: the auditor's type set, `as_bool` over `bool(...)` on a
JSON field, and `idempotency_key`'s classification as `UNBOUND` (a pure read of the
arguments). Gates in `test_tools_idempotency.py` — the four planned plus
`test_a_repeat_of_a_failed_effect_runs_again` and
`test_a_different_effect_is_a_different_key`. Sabotaged three ways — no answer from the
log, no note, a failure read as unknown — each failed its gate.)*

**Why.** DESIGN I2: "Outside state needs to be idempotent… the session may repeat those
actions." After `TOOL_OUTCOME_UNKNOWN` the model re-issues the call with a **new** call id,
so nothing keyed on the call can recognize the repeat. Only the effect can.

**What.**
- `ToolDefinition.idempotency_key: Callable[[Any], str | None] | None`, through
  `define_tool` (`ph/tools/definition.py:726`) beside `is_irreversible` (`:631`), and for
  its reason: only the producer knows what its effect is.
- The pipeline records each keyed call as a `TOOL_EFFECT` intent (key: tool name plus
  effect key, **session-scoped**). A call whose key has a settled prior returns the
  recorded result, marked `repeated` in `meta`, instead of running. One whose prior is open
  or was repaired to unknown goes to P10-13's `reconcile` if the tool has one, and
  otherwise runs with a note appended to its result that an earlier call with the same
  effect may have happened.
- `ToolRunContext.idempotency_key` (`ph/tools/definition.py:328`) is
  `f"{session.id}/{call_id}"` — for Code Mode `…/{call_id}:code:{n}`, already
  deterministic. It is for a tool's *own* retries against a far side that accepts keys (an
  HTTP `Idempotency-Key`, an SMTP `Message-ID`), and it is documented as that and nothing
  more: it does not survive the model re-issuing a call, which is what the effect key is
  for.
- **No shipped tool adopts the effect key in this row**, and that is stated on the field.
  No shipped tool has an external effect whose repeat is worth suppressing — `bash` is a
  different command every time — and suppressing a legitimate repeat is the failure. The
  gates use a test tool.

**Files:** `ph/tools/definition.py`, `ph/tools/registry.py` (`prepare`), `ph/tools/batch.py`,
`ph/testing/builders.py` (an `external_tool` with a counting far side),
`packages/ph-core/tests/test_tools_idempotency.py` (new).

**Gates:** `test_a_repeated_effect_returns_the_recorded_result` ·
`test_a_repeat_after_an_unknown_outcome_is_not_silently_suppressed` ·
`test_the_run_key_is_stable_across_a_tools_own_retries` ·
`test_a_tool_that_declares_no_key_runs_every_time`.

## P10-13 — tools answer "did I happen?": `reconcile`, asked on resume

*(Landed. `Done(value) | NotDone() | Unknown()` in `ph.tools.definition` (exported from
`ph.tools`); `ToolDefinition.reconcile` takes the call's arguments, its record **and the
session** — a third argument the sketch lacked, because `write` cannot find the file
without the agent's root, and at resume no agent exists to ask: the tool reads it from the
session's own open fresh-root workspace record (else the deployment root), the way
`root_for` would have resolved it. Repair gained `unresolved_calls(events)` (the started,
unresolved `tool/call` records of the open turn) and `CallOutcome`, handed in by call id:
`resume_session`'s `_reconciled` pass asks each tool at `DEPLOYMENT` scope, over a
read-only copy of the stored log built only when some tool can answer, and a raise or
`Unknown` is no answer. `Done` becomes a non-error `tool/result` of the tool's own
rendering with `meta: {reconciled: true}`; `NotDone` becomes a `TOOL_NOT_STARTED` result
whose text says the tool checked, rather than the "never recorded" text, which would be
false. `write` is exact or `Unknown`: done on identical bytes, not done on a missing or
different file. **And the pipeline asks too**, as P10-12 planned: a repeat of an effect
whose prior is unknown asks `reconcile` (bound as the registering row) before running —
`Done` answers the call from the tool's rendering (`meta: {reconciled, repeatOf}`) without
running, `NotDone` runs without the note. `ToolDefinition.reconcile` is classified `BOUND`
to `ToolRuntime._reconcile`, with the resume path's lack of a binding said there and on the
field. Gates: the four planned, plus `test_a_tool_that_can_tell_is_asked_before_the_retry_runs`
and `test_a_tool_that_says_it_did_not_happen_runs_without_the_note` in
`test_tools_idempotency.py` (`external_tool(reconciles=True)` asks its `FarSide`).
Sabotaged four ways — resume never asking, `write` unable to tell, a done result drawn as
an error, the pipeline never asking — each failed its gates.)*

**Why.** Repair's `_OUTCOME_UNKNOWN_TEXT` asks the *model* to "decide whether to retry
from the tool semantics", because the harness never asks the tool, which knows. A `write`
whose file holds exactly the content it was given has happened.

**What.**
- `ToolDefinition.reconcile: Callable[[Any, SessionEvent], Awaitable[Reconciled]] | None`,
  where `Reconciled` is `Done(value) | NotDone() | Unknown()`.
- `resume_session` (`ph/persistence/jsonl.py:684`) is already `async` and runs mounted.
  Before repair synthesizes a closer for a started, unresolved call, it looks the tool up at
  `DEPLOYMENT` scope:
  - `Done` becomes a non-error `tool/result` rendered by the tool's own `render`, with
    `meta.reconciled = true`;
  - `NotDone` becomes the `TOOL_NOT_STARTED` closer;
  - `Unknown`, or no tool, or an agent-scoped tool that resume cannot see, is today's text.
- **Repair stays pure.** The reconcile pass produces answers keyed by call id, and
  `interrupted_turn_closers` takes them as an argument, so the fold over a stored log with
  nothing mounted is unchanged.
- The reference adopter is `write`: done when the file's digest equals the call's content,
  not done when it does not exist or differs. It is covered by a restore, so the stakes are
  low and the check is exact.

**Files:** `ph/tools/definition.py`, `ph/persistence/jsonl.py`, `ph/persistence/repair.py`,
`ph/tools/builtin/fs_tools.py`, `packages/ph-core/tests/test_repair.py`.

**Gates:** `test_a_write_that_landed_is_reported_done_after_a_crash` ·
`test_a_write_that_did_not_land_is_reported_not_started` ·
`test_a_tool_that_cannot_answer_keeps_the_unknown_text` ·
`test_repair_is_still_a_pure_fold_over_a_stored_log`.

**Not enforced (rule 6):** an agent-scoped tool is invisible at resume, so it is never
reconciled. Said on `reconcile`.

---

## Part 5 — records that only mean something together

## P10-14 — `Session.batch()`: planned as a unit, published as a unit

*(Landed. `Session.batch()` yields a `SessionBatch` (exported from `ph.session`) whose
`append` stamps each member with its batch seq and the write door's refusals, so a later
member can cite an earlier one; `SurfaceManager.validate_batch` plans them on a scratch
`_FoldState`; `_commit` and the batch share one `_push`, which publishes each member with
the log ending at it. Two refusals the plan did not name: a log that **moved** while the
block was open (an `await` inside it let another task append) refuses the batch whole,
since its seqs now belong to someone else — "no `await`" is not enforced, only its
consequence is caught; and a batch appended to after its block closed raises rather than
stamping an event no commit will take. Not nested. Two extra gates for those:
`test_a_batch_whose_log_moved_is_refused_whole`,
`test_a_batch_is_one_at_a_time_and_closes_behind_itself`. The truncation gate's refusal
is the write door's (an unwritable payload on the second rewrite), since a truncation pass
cannot make the surface refuse its own rewrites. Sabotaged five ways — no validation,
planning against the live state, push-all-then-publish, no moved check, and truncation
back to one append at a time — each failed its gate.)*

**Why.** Compaction appends `compaction/summarized` and then the summary's `user/message`
replace, adjacent by construction (`ph_stabilize/compaction.py:1267`, "no `await` between
here and the replacement"). Argument truncation appends N `assistant/message` replaces and
then `compaction/args-truncated` (`:940-968`). Both are atomic against other tasks. Neither
is atomic against a later append in the group being refused by the surface: the earlier
ones stay, and the accounting record is missing, or describes a replacement that never
landed.

**What.**

```python
with session.batch() as batch:
    batch.append("compaction/summarized", accounting)
    batch.append("user/message", summary, SurfaceIntent(SurfaceReplace(...), sources))
```

On exit, every event is planned against a scratch copy of the surface's `_FoldState` —
`_plan` (`ph/session/surface.py:247`) is already pure against a state and a log — and only
if all pass are they pushed and published, in order, under one reentrancy guard. An
exception inside the block, or a refused plan, pushes nothing. The copy costs O(nodes) per
batch, and batches are rare. **This is the whole of in-process failure**: there is nothing to
roll back, because nothing was committed. A batch holds only synchronous appends — no
`await`, no effect — by construction.

**Files:** `ph/session/session.py`, `ph/session/surface.py`, `ph_stabilize/compaction.py`,
`packages/ph-core/tests/test_session_append.py`, `packages/ph-stabilize/tests/test_compaction.py`.

**Gates:** `test_a_refused_event_leaves_none_of_its_batch` ·
`test_observers_see_a_batch_in_order_after_it_validates` ·
`test_a_batch_obeys_the_fold_laws` ·
`test_a_truncation_pass_lands_whole_or_not_at_all`.

## P10-15 — batch membership on the envelope; a batch reads back whole or not at all (format 2)

*(Landed. `BatchRef(first, count)` is a `WireModel` (`count ≥ 2`, a `last` property),
exported from `ph.session`; `SessionEvent.batch` and `_EventWire.batch`, omitted when
absent. `Session.batch()` stamps members only when there are two or more and hands back
the committed events through `batch.events`. Membership is checked by `_within_batch` on
**every** `admit` — seed and replica alike, since members are contiguous whichever door
they come through — and only the seed refuses to **end** inside one, as planned. Two
additions: `read_session` drops an unfinished trailing batch only on an **unbounded** read
— a bounded one is a reference fork reading a prefix it cites, where dropping would
silently shorten what a child depends on — and `_fork_seed` refuses a boundary inside a
batch with a new `ForkRejection`, `OPEN_BATCH`, rather than letting the seed refuse it
less legibly. `_settle_batch` runs after the torn-line rule in `_settle_tail`, walking
back to the member whose seq is `first`; when that member is in a parent's file instead,
it leaves the tail for the seed to refuse. `SESSION_FORMAT_VERSION` is 2 with its
docstring entry; the format-version gate is parametrized over format 1 by name.
`tests/test_wire.py` gained a `BatchRef` sample. Gates: the planned set, with the replica
test in `test_session_admit.py` beside `test_an_event_that_breaks_a_batch_is_refused_by_admit`,
and the stamping/round-trip test in `test_session_append.py`. Sabotaged four ways — the
reader keeping half a batch, the writer keeping half, members never stamped, membership
unchecked — each failed its gates.)*

**Why.** A batch is appended synchronously, so one flush carries all of it, and Turso
commits a flush as one transaction. JSONL writes a flush with one `write`, and a process
that dies mid-write can cut between two lines of one batch — the reader then keeps
`compaction/summarized` and drops its replacement. Nothing in the process can answer that
case; P10-14 has already covered every failure that happens while the process is alive.

**What** (decision 5).
- **The envelope.** `SessionEvent.batch: BatchRef | None`, with `BatchRef(first: int,
  count: int)`; on the wire `{"first": n, "count": k}` beside `surfaceOp`, omitted when
  absent the way `ignorable` is (`to_wire`, `ph/session/events.py`). `_EventWire` gains the
  field — which is exactly why this is a bump: a format-1 reader refuses it as unknown.
  `Session.batch()` stamps every member with the same ref; a batch of one is not stamped,
  since there is nothing to keep together.
- **Acceptance.** `_readmit` checks membership is consistent: a member's seq lies in
  `[first, first + count)`, members are contiguous, and no unstamped event falls inside an
  open batch. A **seed** that ends inside a batch is refused — after the JSONL reader has
  dropped a torn tail, an unfinished batch can only be damage. A replica's `admit` is not
  held to completion at each step, because a daemon publishes members one frame at a time;
  the check is the seed path's alone.
- **JSONL.** `read_session` drops a trailing batch whose count is not met, after dropping any
  torn line, and says so in the log line it already writes. `_settle_tail` applies the same
  rule before the first append — it walks back line by line (`_line_start`) to the member
  whose seq is `first` and cuts the file there — so reader and writer agree, as they do for a
  torn line.
- **Turso** never meets an unfinished batch: a batch is never split across flushes, and a
  flush is one transaction. The seed check covers it anyway.
- **`SESSION_FORMAT_VERSION = 2`**, with its docstring entry: "2: the envelope carries batch
  membership." Format-1 logs are refused by the header check, not migrated.

**Files:** `ph/session/events.py`, `ph/session/session.py`, `ph/persistence/jsonl.py`,
`packages/ph-core/tests/test_session_event.py`, `packages/ph-core/tests/test_session_admit.py`,
`packages/ph-core/tests/test_persistence.py`.

**Gates:** `test_a_batch_cut_by_a_torn_write_is_dropped_whole` ·
`test_a_complete_batch_reads_back_intact` ·
`test_a_resumed_log_appends_behind_a_dropped_batch_cleanly` ·
`test_an_unfinished_batch_before_the_end_refuses_the_log` ·
`test_a_replica_admits_a_batch_one_member_at_a_time` ·
`test_every_envelope_field_maps_to_to_camel_of_its_name` (existing; now covers `batch`) ·
`test_a_wrong_format_version_is_refused` (existing; a format-1 log is refused by name).

**Not in this row:** drawing a batch as one unit. The TUI adapter and `trajectory.py` can
read membership now, and a compaction's accounting and summary would read better as one
row, but that is a presentation change with its own gate.

---

## P10-16 — docs, non-guarantees, bookkeeping

*(Landed. `plans/Implementation_Plan.md` §4 gained Phase 10 and §6 its row; `DESIGN.md`
§5.2's durability paragraph now says kinds declare their barrier, §7 I2's "where it is
imperfect" says what is offered and what is not, and §8 gained three rows — P10-02's
outcome, the workspace pair kept on its own fold, and P10-03's unmet condition — and
lost one that had gone false (`Session.first_live_seq` is read now, by the journal's
process-scoped keys). `reviews/09` and `reviews/10` carry status lines;
`docs/dev-notes/phase-10.md` records what was traded. Two `NON_GUARANTEES` rows, claimed
in `test_non_guarantees`: "an outside effect after a crash" and "facts across two logs".
The version constants moved with their docstring entries; **the package versions are not
bumped here** — that is the release's, with its note.)*

- **`plans/Implementation_Plan.md`:** §4 gains **Phase 10 — actions the log can vouch
  for**, with these rows as they land, and §6 gains its line (below).
- **`DESIGN.md`:** §7 I2's "Where it is imperfect" is rewritten to describe what is now
  offered and what is not. The durability paragraph in §5.2 is reduced to "kinds declare
  their barrier". §8 records P10-02's outcome.
- **`reviews/09`, `reviews/10`:** status lines.
- **`docs/dev-notes/phase-10.md`:** what was traded.
- **The version bump:** `SESSION_FORMAT_VERSION` 2 and `PROTOCOL_VERSION` 4, each with its
  docstring entry, released together; the release note says format-1 logs are refused rather
  than migrated, and that a daemon started before the upgrade must be restarted.
- **`NON_GUARANTEES`**, printed by both doctors: an external system with no key and no
  `reconcile` is still *outcome unknown* after a crash; cross-log facts (a child's log and
  its parent's roster, a message's send and its receipt) are eventual, not atomic.

---

## Reuse (do not rewrite)

| Need | Already there |
|---|---|
| A cached, drift-checked fold | `SessionFoldCache`, `check_fold_laws`, `contribute_fold_cache` |
| A non-raising flush | `session_written(ctx, session)` |
| The mount's last write | `write_on_unwind`, registered by `claim` |
| "Does a restore cover this call?" | `ToolRuntime.restore_covers` |
| Types a plugin owns | `declare_log_type`, `is_known`, `is_ignorable` |
| Planning a surface transition without mutating | `_plan` against a `_FoldState` |
| The torn-tail rule, reader and writer | `read_session`, `_settle_tail` |
| "Appended in this process" | `Session.first_live_seq` |
| Refusing a log from another format | `SessionHeader`'s version check, `SESSION_FORMAT_VERSION` |
| Two-phase blob writes, repaired at open | spill's reserve → append → commit — **not** a batch |
| The package walk and the writer matrix | `tests/workspace_layout.py`, `reviews/probes/writer_matrix.py` |
| Reading what a store holds, in a test | `ph.testing.stored_types` |

## Non-goals

- **Reading or migrating logs written before the bump.** Format-1 logs are refused by the
  header check (decided 2026-09-24).
- **Rolling back a committed event.** A batch is all-or-nothing *before* it is committed;
  nothing in this plan rewrites the log.
- **Two-phase commit across logs.** A child's log and its parent's roster stay eventual.
  The parent's record is the intent and the child's log is the reconcile source, which is
  the shape `subagent_roster` already has.
- **Exactly-once against a system with no idempotency key and no way to ask.** After a
  crash that is *outcome unknown*, and says so in repair's words.
- **Runtime writer authorization for every type.** P10-01 is static; P10-02 covers four
  types, if the spike allows.
- **Migrating `schedule/tick`, `subagent/admitted`, `goal/*` or `attachment/uploaded`.**
  The tick and its delivery are already co-flushed, and a lost tick costs a skipped run by
  design. The roster is already a complete fold. A goal's settlement is accounting. The
  upload record is a fact with no outcome to wait for, and it is already durable before
  the cache.
- **A general `tools/execute` barrier inside `ToolRuntime.dispatch`.** The checkpoint
  policy remains the one place a tool barrier is placed, until a second wrapper that
  appends before the body exists.

## Verification

Every row ends with the four CI gates green — `ruff check .`, `ruff format --check .`,
`mypy`, `pytest` — from the baseline of 3,323 passed and 8 opt-in skips at `bdb99a1`. Every
new gate is sabotage-checked by reverting the mechanism it holds and watching it fail.

**Crash injection** is how Part 3's gates are written. The act is held open on an
`anyio.Event`, the store is read through the Protocol while it is held, and that snapshot
is resumed in a second mount over the same file — the shape `test_subagents.py`'s
`_restart` and `test_daemon_shell.py`'s blocking command already use. A process is never
actually killed, so the gates stay portable and fast.

**Definition of done** (`Implementation_Plan.md` §6, Phase 10): every write-ahead record in
the vocabulary belongs to a declared kind; a crash between any kind's open and its settle
leaves the intent on disk, and the next open settles it or leaves it to its owner, whether
or not a turn was running; a daemon retry after a crash mid-act is told the outcome is
unknown rather than refused as repeated; a tool that declares `reconcile` is asked before
the model is told "unknown"; a compaction reads back whole or not at all; and every append
in every package names a type and a writer of record.
