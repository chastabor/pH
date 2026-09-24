# Phase 10 — Actions the log can vouch for

**Status:** fourteen rows landed, P10-02 spiked and not shipped, P10-03 conditional on
a type nobody has declared yet.

**Gate:** `ruff` + `ruff format` + `mypy --strict` across 461 source files + 3 401
tests (8 opt-in skips), green. Every new gate was sabotage-checked by reverting the
thing it guards and watching it fail; the plan records each sabotage.

`reviews/09` found the session log honest about what it held and quiet about what it
had promised. Seven places recorded an intent before an act and its outcome after —
an approval, a question, a `!!`, a tool call, a Code Mode dispatch, a daemon verb, a
workspace tree — and each spelled its own fold, its own flush and its own failure.
Three places settled the orphans a crash left, and two pairs nobody settled at all:
a `!!` the daemon died during read as running forever. This phase declares the pair
once (`ph.session.intents`), gives it one door (`ctx.intents`), and lets a tool
vouch for its own effect (`idempotency_key`, `reconcile`).

The recurring correction was **the plan's sketch meeting the code's ordering**. Four
times out of five, a barrier the sketch placed *before* an act was one the code had
deliberately placed *after* it, for a reason written beside it: the daemon verb's key
(a key made durable first refuses a retry for work that never began), a question a
passivated root must re-pose (so a cancellation must not settle it), the approval's
K7 `finally` (a cancel is `canceled`, not `interrupted`), F7's flush before a reply
(after the act, not before). Each row kept the code's ordering and said why in the
plan. The journal grew the knobs those orderings needed — `buffered`, `dedupe`,
`key_scope` — rather than the call sites bending to the sketch.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P10-01 | Writers of record for every type, held by an AST walk of every package | `known_event_types.WRITERS`, `test_log_writers.py` |
| P10-02 | *Spiked, not shipped* — a runtime posture-writer check | stated at `SandboxSeam.logged_mode`, `approval_policy` |
| P10-04 | A lineage flushes a parent only for a child that inherits its prefix | `SessionStore.lineage` |
| P10-05 | Kinds declared once; one fold | `ph/session/intents.py` |
| P10-06 | `ctx.intents`: open, record, settle, claim, `intent-fold-cache` | `ph/session/journal.py` |
| P10-07 | Repair settles every declared kind, in a turn or out | `persistence/repair.py` |
| P10-08 | `!!` as a kind | `seams/shell.py` `SHELL_COMMAND`, `ph_app/shell.py` |
| P10-09 | Approvals and questions as kinds | `APPROVAL_ASK`, `QUESTION_ASK` |
| P10-10 | Daemon verbs settle; repeats say `settled`/`unknown`; process-scoped keys | `CLIENT_COMMAND`, `MutationRepeated.outcome` |
| P10-11 | Code Mode dispatches as a kind | `tools/code_mode.py` `TOOL_DISPATCH` |
| P10-12 | Tools name their effect | `ToolDefinition.idempotency_key`, `TOOL_EFFECT` |
| P10-13 | Tools say whether a call happened | `ToolDefinition.reconcile`, `write` |
| P10-14 | Records planned and published as a unit | `Session.batch()` |
| P10-15 | Batch membership on the envelope; log format 2 | `BatchRef`, `read_session`, `_settle_batch` |
| P10-16 | Non-guarantees, `DESIGN.md`, these notes | `NON_GUARANTEES` |

---

## The things worth knowing

**One fold, not two.** P10-05 first shipped `open_intents` with its own loop and
`settled_record` with a backwards scan. P10-06's dedupe index needed settled keys
too, which `open_intents` drops, and a third loop would have been a third statement
of "open". So `fold_intents` — per key, the latest open and its settle — became the
only fold, and the other two read off it, from a log or from the cached index.

**Keys that repeat on purpose.** An approval is keyed by call id *or tool name*; a
question re-posed after a resume keeps its id. Deduping either would answer a second
ask from the log without showing it to anybody. `IntentKind.dedupe` is false for both,
and the journal then holds their settles to the key alone, because two concurrent asks
of one tool share a key the fold could never tell apart — refusing the first one's
settle turned that old ambiguity into an error raised from a `finally`. The ph-rlm
concurrent-writes test found it.

**A cycle the in-process tests could not see.** Repair imports the seams that declare
ph-core's kinds, so a resume anywhere settles them. At module top that was
`ph.orphans` → `ph.persistence` → `repair` → `ph.seams.shell` → `ph.seams.subprocess`
→ `ph.orphans`: invisible to every test, since the suite imports everything first,
and fatal to the ph-rlm lifecycle host, which imports `ph.orphans` first. The import
moved into `_kinds()`, and `test_importing_repair_declares_every_core_kind` now asks a
fresh interpreter that imports `ph.orphans` first.

**`scope=` means a lifetime here.** The process-scoped key was first spelled
`open(..., scope="process")`; the registration-ownership gate refused it, because in
this codebase `scope=` names a `Context` a registration lives on. It is `key_scope`.

**The workspace pair stayed where it was.** Its openness is folded from `seed_length`
(a fork's inherited trees are the parent's) and over fresh-root tiers only, and
`reclaim` deletes what that fold says. A key-pair fold from seq 0 would report a
fork's parent's live trees as the child's leaks. Declaring the kind would have added a
second, wrong statement of openness and changed nothing repair does.

**`write` is exact or `Unknown`.** It is done on identical bytes and not done on a
missing or different file, resolved against the agent's own root as its log records
it, since at resume no agent exists to ask. `NotDone` is a strong claim — the model
reads "retry it" — so every case the check cannot make exactly is `Unknown`.

**What a torn write can cut.** One flush writes a whole batch; a death mid-write can
cut between two of its lines. The reader drops an unfinished *trailing* batch and the
writer cuts the file back to its first member before appending — the torn-line rule,
one level up. A bounded read (a reference fork's prefix) does not drop anything, since
that would silently shorten what a child cites; a fork boundary inside a batch is
refused as `OPEN_BATCH` instead.

---

## What was traded

- **A runtime writer check for posture types (P10-02).** The daemon's preset verb
  reaches `set_mode` with no row running, so a check would have had to answer
  "allowed" when it could not tell. The static gate holds every shipped writer; a
  third-party row is not refused at runtime, and that is stated where the posture is
  read.
- **Two fsyncs per daemon verb.** A `durable` `CLIENT_COMMAND` would have made every
  crash mid-act reportable, at the price of a flush before every `session/prompt` and
  of P5-02's rule that a crash before the act re-runs it. It is `buffered`: the key
  travels with its act's records, and only a key that reached disk with no settle is
  reported `unknown`.
- **Settling a canceled question.** `claim` settles on any exit that is not a return;
  the question seam uses `open` and an explicit settle instead, so a root passivated
  while a person is looking at a question re-poses it on the next resume.
- **MCP tools.** None declares a key or a `reconcile`, and nothing here can declare
  one for them. After a crash they keep `TOOL_OUTCOME_UNKNOWN` — a `NON_GUARANTEES`
  row.
- **Cross-log atomicity.** A child's log and its parent's roster, a message sent and
  its receipt, are separate writes. `Session.batch()` is one log's; there is no batch
  across two — a `NON_GUARANTEES` row.

## The version bump

`SESSION_FORMAT_VERSION` 2 and `PROTOCOL_VERSION` 4 move together. A format-1 log is
refused by the header check, not migrated. A daemon started before the upgrade must be
restarted: a 0.4 client requires the `outcome` a 0.3 daemon's repeat does not carry.
The package versions are not bumped here; that is the release's.
