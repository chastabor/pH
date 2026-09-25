# Facts across two logs — the lower findings Phase 10 left standing

*2026-09-25. Follows 0.4.0 (`8e391d4`). Works through the "Lower" table of
`reviews/09-log-atomicity-and-divergence.md` §4, the rows L3–L9 that the revised
`reviews/architecture-map.html` still shows red, and turns the ones worth fixing into an
ordered todo list.*

**Status, 2026-09-25.** L6, L5, L4, L5b and L6b have landed (uncommitted), with the doc
corrections below. L5b was found while landing L5 (the resume sweep reached a root's own
children and stopped), and L6b turned out to need Code Mode dispatches asked on resume,
not only a `reconcile` on the send tool. L8 and L9 are closed with the reasons given; L3
and L7 are optional.

## The goal every item is weighed against

The same as `plans/Contained_Log_Writes_Todo.md`'s: **several sub-agents are running, each
with its own session log. Everything stops at once. The daemon restarts, and every
sub-agent picks up where it left off.** A restart reads each log separately, so what it
needs from a fact that spans two logs is that neither log tells it something false.

## Why these were left standing

Mostly, no reason was written down. Phase 10 was organized around the F findings and
took L1, L2 and L10 on the way; review 09 then said only "L2–L9 stand". The only recorded
reasoning is the plan's non-goal "Two-phase commit across logs" and the daemon's
`NON_GUARANTEES` row "facts across two logs". That row said a crash leaves one log
ahead of the other "until the next open reconciles what it can". For messages and for a
child's usage, nothing reconciled anything. The row now says what is true.

## The rule these items share

Two logs are written on two schedules, and no batch spans them. What can be arranged is
the **order**, so a crash keeps the harmless half:
- **The receiver before the sender is told** (L6): a crash can leave the send unknown,
  never falsely delivered. This is F1's rule, a child's log written before its parent
  hears `done`, applied to messages.
- **The authorization before the effect** (L4): a global refinement's approval is on
  disk before the edit it approved.
- **Where the order cannot be arranged, reconcile from the log that runs ahead** (L5):
  the child's log reaches disk before each of its requests, and the parent's copy of its
  usage waits for the parent's next flush, so on resume the parent learns from the child.

## The findings, one by one

### L6 — a message is durable in the sender's log before the receiver's splice

**Was.** `send` steers the message into the receiver's inbox. The inbox records it in the
receiver's log (`agent/inbox/spliced`), in memory, and the sender's receipt goes into the
sender's log. Each reached disk at its own agent's next flush. A receiver inside a long
tool call flushes last, so a crash could leave the sender's log saying `delivered` and the
receiver's log saying nothing: a message lost, and nobody told.

**Fixed.** See L6 in the todo list.

**L6b, the other half.** A crash after the receiver's write and before the sender's
result left the sender's call outcome unknown, so the model might send it again. A
`reconcile` on the send tool answers from the receiver's log. Landing it found that it
would never have been asked in the shipped profile:
- `rlm` runs `tools.mode: code`, so every send is a Code Mode dispatch inside a cell;
- resume asked tools only about top-level `tool/call` records, and closed an open
  dispatch `outcome-unknown` without asking;
- and the model reads only the cell's own result, which said nothing about what ran in
  it.

So L6b is four changes, not one: the message id derived from the call, `reconcile` on
the send, resume asking a dispatch's tool as it asks a top-level call's, and the cell's
result naming what the tools found. See L6b in the todo list.

### L5 — a child's usage in the parent's log can lag the child's own log

**The review's wording was wrong in both halves.** It said "a re-run child attributes
twice", and that DESIGN claims only the TUI reads the record.
- **Counting a re-run twice is right.** A child re-run after a crash made new requests
  and the provider was paid for them; the budget should count them.
- **The real gap runs the other way.** `ph_rlm.subagents._mirror` appends each
  `subagent/usage-attributed` to the parent's log in memory. The child's own log reaches
  disk before each of its requests (the checkpoint barrier), and a parent waiting on a
  long child makes no requests and so no flush. A crash drops every attribution since
  the parent's last flush while the child's log still holds the answers.
- **Two decisions read the record, not only the TUI:**
  - a goal's `max_tokens` budget (`goals._TOKEN_RECORDS`, as `children`), which then
    under-counts what the child spent;
  - the child's retry ladder: `restarts_since_progress` is `resumes` minus
    `resumesAtLastAnswer`, and each attribution is the answer that forgives the restarts
    before it. A child that answered in each run but whose attributions were lost looks
    stuck, and after `CHILD_RETRY_LIMIT` (3) such restarts is failed as exhausted while
    making progress.

**The fix.** Reconcile on resume, from the child's log, before the ladder decides.
- **Where.** `SubagentService.resume_children` (ph-core) decides `spent` from the parent's
  roster. `subagent/usage-attributed` is written by `ph_rlm.subagents`, its writer of
  record, which also owns `_mirror`. So the provider reconciles, through a method the
  sweep calls on the child's registered provider (the one `_readmitter` finds) before it
  reads `restarts_since_progress`. `WRITERS` is unchanged.
- **What.** For each row the sweep would put back on the ladder, read the child's stored
  log. For each `assistant/message` carrying usage whose seq is past the highest
  `targetSeq` the parent holds for that run, append the missing attribution, marked
  `origin: "reconciled"`. The sweep then folds the roster again, and `resumesAtLastAnswer`
  and the budget both see what the child did.
- **Only the rows that need it.** A child whose terminal `subagent/status` is on the
  parent's disk has every attribution before it on disk too, since they are one log in
  order. Only rows still `running` at the crash can be behind (a `queued` row was
  reconciled by the sweep that queued it), and those are the rows whose logs
  readmission reads anyway.
- **Not double-counting.** Only seqs past the parent's highest `targetSeq` are added, so
  a reconcile never repeats what the parent already has. A child whose own log lost its
  tail re-answers at the same seqs; its live `_mirror` records those as the new spend
  they are.
- **Gate.** Landed in `ph-rlm` `test_subagents.py`, over the in-process restart those
  tests already use (a snapshot of the logs, resumed by a fresh mount) rather than
  `tests/test_mass_restart.py`'s `SIGKILL`: the state a crash leaves is the same, and
  the parked child makes it exact. See L5 in the todo list.

### L5b — the resume sweep stops at a root's own children (landed)

**Found while landing L5.** `SubagentService.resume_children` has one caller,
`Supervisor._start`, for the root agent. `RLM_MAX_DEPTH` is 2, so a root's child can
delegate too, and after a restart:
- the root's own children are reconciled, laddered, held or readmitted;
- a readmitted child's *own* roster is never swept. Its children caught mid-turn keep a
  `running` row that nothing drives, readmits or fails, the child's model is shown them
  as running (`list_children`, the prompt), and their answers are not reconciled into the
  child's log. That is the restart goal failing one level down.

**The fix.** Sweep each readmitted child's roster the way the root's is swept, with the
same `retry_limit`, before that child takes its first step. `_readmit_one` starts the
child's drive job at once (`_attach`), and the seam can bound the child only after that, so
landed as a gate the drive waits on (`SubagentRun.ready`), opened once the child is
bounded and its own children swept. The limit is the host's (P6-32), so it travels with
the sweep rather than being chosen by the seam. See L5b in the todo list.
- **Gate.** A root whose child had a child running at the crash. After the restart the
  grandchild is readmitted or failed by the ladder, and its answers are reconciled into
  its parent's log. Sabotage by sweeping the root only.

**Why not fold every startup sweep into one.** Asked when L5 landed. Two kinds of sweep
run at startup, and they differ in scope and in when they must run:
- **Per log, when it is opened** (`open_session`): repair in `resume_session` (every
  open intent settled, an interrupted turn closed, tools asked to reconcile), then the
  `session/created` listeners (spill sweep, workspace reconcile, schedule reindex). Every
  log gets them, a root's or a child's, through the one door.
- **Per parent, over its roster** (`resume_children`): answers reconciled (L5), the
  ladder, credential holds, readmission. It compares two logs, and the ladder must see
  the reconciled answers before it decides.

`reconcile_answers` cannot move into the per-log sweep: a child's log is opened when it is
readmitted, which is after the ladder decided, and never for a child the ladder fails. So
it stays the roster sweep's first step. What L5b adds is that the roster sweep reaches
every level, not a new sweep.

**A child that fails in a live daemon needs none of this.** A root and all its children
run in the daemon's one process, so a child cannot lose its in-memory records without its
root's process losing them too. A child whose drive raises is marked `error`, its log is
written first (F1), and its parent is told. Its attributions are still in the parent's
memory. It is not restarted: the ladder runs only when a root resumes.

### L4 — the harness edit is durable ahead of its session records

**Today.** `HarnessService._commit` appends `harness/refined` to the session in memory,
or for a global edit appends to `$PH_HOME/harness/events.jsonl`, and then writes the
projection `harness_state.json` to disk.
- **Global.** The approval ask is on disk before anyone is asked (F8), but its
  `approval/decided` is in memory when the edit reaches the global log. A crash in
  between leaves a deployment-wide edit in force for every future session, and repair
  settles its approval `outcome-unknown`: an edit with no record of who allowed it.
- **Local.** The projection runs ahead of the log. Nothing reads the projection, and
  `stale_projections` reports the drift, so this half is cosmetic.
- **The global append does not fsync.** `write_text_under(..., append=True)` writes
  through the page cache, so the edit survives the daemon being killed but not
  necessarily a power loss.

**The fix.**
- In `_commit`, write the session (`session_written`) before the out-of-log write: the
  global append, and the projection. The session is the approving agent's, which for a
  global edit is the one `approval.request` wrote to.
- The global record names what authorized it: the approving session's id, with the
  refinement's id travelling on the ask as its `callId`, so the global log can be read
  against the decision that allowed it. (Planned as the ask's seq; the id is known before
  the ask and needs no lookup afterwards.)
- The global append fsyncs, and takes a failed write back the way `_append_and_sync`
  does, since it is a durable log outside the session store.
- Cost: one flush and one fsync per refinement. Refinements are rare, and a global one is
  approved by a person.
- **Gate.** A global refinement with a probe on the global file's append: at that moment
  the session's stored log holds `approval/decided` for the ask. The stored global record
  names its session and ask. A monkeypatched `os.fsync` records the global append.
  Sabotage by dropping the write, and by dropping the fsync.

### L3 — a settings file rewritten whole with no lock

**Today.** `SettingsService.set` changes the tree this process loaded once and rewrites
the whole file, atomically but with no lock, so a second process's change is lost, and
two `set`s racing in one process can land their writes in the wrong order.
- **Latent as written.** Nothing in shipped code calls `ctx.settings`; only its own tests
  do.
- **Live elsewhere.** `ph_app.tui.config.save_tui_settings` has the same shape for
  `$PH_HOME/tui.json`. Two TUIs open at once undo each other's `/view` and sidebar
  toggles. Only a preference is lost.

**The fix (optional, low).** Read the file again under a file lock (`ph.locks.file_lock`),
change only the key being set, and write atomically, for both files. **Gate:** two writers
each set a different key, and both keys survive; sabotaged by writing the cached tree.

### L7 — `/sandbox allow` is kept in a profile drop-in, not the log

**Today.** The allowances apply to the whole deployment, so a drop-in is the right store,
and every confined command is bounded by what is in force when it runs. What is missing is
history: a replayed log cannot say what a confined command could reach at the time. No
running agent sees a mismatch.

**The fix (optional; decide first).** An ignorable record of the allowances in force, when
a session starts or resumes and on each `/sandbox` change, written by the sandbox seam.
Worth it only if network access should be auditable from the logs alone.

### L8 — a worktree exists before any durable record names it (closed)

**Closed, with this reasoning.** The tree is created before `workspace/acquired` reaches
disk, but a crash between costs little:
- The git tier keys each tree by agent id, and `_add` reuses a tree it finds for that
  agent, so a readmitted agent gets its own tree back with its uncommitted work.
- `/workspaces` lists every checkout a tier still holds on disk (`Stray`), whether or not a
  log names it, so an orphan left by an agent never readmitted can be seen and removed.
- Recording the acquire durably first would cost an fsync per spawn to save a stray a
  person can already find.

### L9 — posture folds have a replay nothing compares (closed)

**Closed, with this reasoning.** Since `96f8140`, before the review, `_LatestFold.read` is
`fold_latest` with a cursor (`since=`): the live projection and the replay are one
function over a log that is only appended to (I4). A runtime comparison of the two would
test Python's list semantics. `test_session_append.py` pins the fold's rule: the newest
event it can parse, reading only what arrived since.

## Todo list

In order. Same rules as before: every gate sabotage-checked, the four gates green,
nothing committed by Claude.

- [x] **D — Doc corrections.**
  - DESIGN §6.3 and the `ph_rlm.subagents` module docstring said nothing but the TUI reads
    `subagent/usage-attributed`. They now name the goal budget and the retry ladder, and
    L5's gap.
  - `docs/seams/subagents.md`'s ladder table named a row field, `attempts`, that the roster
    does not have. It now names `resumes`, `resumesAtLastAnswer` and
    `restarts_since_progress`, and states L5's gap beside the ladder.
  - The `NON_GUARANTEES` row "facts across two logs", and DESIGN's matching paragraph
    under I2, say what holds after L6 and what L5 leaves open.
  - DESIGN §8's row on turns parked on a human named the deleted `pending_approvals` and
    `pending_questions` and a gate that no longer exists. It now states the behavior since
    P10-09 and its gate.
  - `ph.session.journal`'s module example checked `claim` for a `Prior`, which only
    `claim_once` returns since T3.
  - `reviews/09` and the map mark L4, L5 and L6 fixed, and L8 and L9 closed.
  - After L4 and L5 landed, the `NON_GUARANTEES` row, DESIGN and
    `docs/seams/subagents.md` say what they fixed and named L5b's limit, and after L5b
    and L6b they say what those fixed instead.
- [x] **L6 — The receiver's disk before the sender's receipt.**
  - *Landed.* `send` writes the receiver's log (`session_written`) after `steer` and
    before it returns the receipt. `written`, not `flush`, for `_mutate`'s reason: the
    message is already in the receiver's inbox, so a failed write must not tell the sender
    it was not sent.
  - *Gate:* `ph-rlm` `test_messaging.py::test_a_message_is_on_the_receivers_disk_before_the_sender_is_told`,
    which reads the receiver's log from the store as `send` returns.
  - *Sabotaged two ways*, and each failed its gate: no write (nothing of the receiver's
    log is on disk), and the write placed before `steer` (the stored log has no splice).
- [x] **L5 — Reconcile a child's usage into its parent's log on resume.**
  - *Landed.*
    - ph-core: `AttributingProvider.reconcile_answers(parent, run_id, *, session_id,
      through)`, an optional provider capability beside `ReadmittingProvider`.
      `resume_children` calls it for every `running` row first
      (`_reconcile_answers`), and folds the roster again only when answers were added.
      The roster row gains `lastAnswerSeq`, the child's seq of the latest attributed
      answer.
    - ph-rlm: `RlmChildProvider.reconcile_answers` reads the child's stored log and
      appends, in one batch, an attribution for each answer past `through`, marked
      `origin: "reconciled"`. `_attribution` is the one payload builder, shared with
      the live `_mirror`.
    - Why this is exact: the answers are appended before the `resumed` record this
      restart writes, in one log written in order, so after another crash both are on
      disk or neither is, and an answer is never credited to a later restart.
    - Known limit: a child whose own log lost its tail while the parent kept the
      attributions re-answers at the same seqs, and a reconcile after a second crash
      skips those seqs. This needs both logs to lose different halves across two
      crashes.
  - *Gates* in `ph-rlm` `test_subagents.py`:
    - `test_an_answer_only_the_childs_log_kept_is_counted_after_a_restart`: a child at
      `RETRIES` restarts answers, only its own log is written, and after the restart it
      is readmitted, not failed as exhausted, with the attribution in its parent's log;
    - `test_a_restart_attributes_nothing_the_parent_already_counted`.
  - *Sabotaged two ways*, and each failed its gate: no reconcile (the child is failed
    as "interrupted 3 times, so it will not be started again" though it answered), and
    reconciling from seq 0 (the answer counted twice).
- [x] **L4 — The session before the harness edit; the global record names its approval;
  the global append fsyncs.**
  - *Landed.*
    - `_commit` writes the session before anything outside it: for a global edit the
      approving agent's session, whose `approval/decided` allowed it, and a global edit
      whose approval cannot be written is refused. For a local edit, the session
      holding `harness/refined`, ahead of the projection.
    - The refinement's id is made before a global edit is asked about and travels as
      the ask's `callId`, so the decision names its record. `RefinementRecord` gains
      `approved_in`, the approving session's id. `rollback` does the same.
    - `ph.persistence.append_records` is `read_records`' durable writer, a public face
      on `_append_and_sync`: whole lines, `fsync`ed, a failed write taken back, and the
      directory synced on creation. The global harness log is appended through it.
  - *Gates:*
    - `ph-rlm` `test_harness.py`:
      `test_a_global_edit_is_written_only_after_the_approval_that_allowed_it`,
      `test_a_global_edit_whose_approval_cannot_be_written_is_refused`,
      `test_a_local_refinement_is_on_disk_before_its_projection`;
    - `ph-core` `test_persistence.py::test_a_record_log_outside_the_store_is_synced_and_takes_a_failed_write_back`.
  - *Sabotaged five ways*, and each failed its gate: no approving-session write, no
    `callId` on the ask, no local session write, the global append through a plain
    file append, and `append_records` as a plain append.
- [x] **L5b — Sweep a readmitted child's own children on resume.**
  - *Landed.*
    - `_readmit_children` sweeps each child it readmits (`_sweep_readmitted`, which is
      `resume_children` on the child) with the same `retry_limit`, which is now threaded
      through `_readmit_children`. `readmit_waiting(parent, *, retry_limit)` descends into
      each live child the same way, so a credential arriving releases a held grandchild.
      The supervisor passes `CHILD_RETRY_LIMIT`.
    - **The gate, `SubagentRun.ready`.** A provider starts a child's drive before the seam
      can bound it (`_enforce` needs the scope the provider builds), so a child could run
      ahead of its ceiling, and a readmitted one ahead of its own sweep. The drive now
      waits on the run's `ready` event. The seam sets it after `_enforce` for a fresh
      admission, and after the nested sweep for a readmission; a refused child is
      released with it unset, and its drive canceled without having run. This also closes
      the ordering `_admit`'s docstring described ("by the time it refuses there is a
      child driving").
  - *Gates* in `ph-rlm` `test_subagents.py`:
    - `test_a_grandchild_the_restart_interrupted_is_put_back_to_work_too`: readmitted
      under its own id, its restart counted in its parent's log, and the sweep's record
      ahead of the child's new turn;
    - `test_a_grandchild_held_for_its_key_is_released_when_the_key_arrives`.
  - *Sabotaged three ways*, and each failed its gate: no nested sweep (not readmitted),
    the readmitted child's gate opened at admission (its turn started ahead of the sweep),
    and `readmit_waiting` kept to one level (the grandchild stayed held with its key
    supplied).
- [x] **L6b — A send interrupted by a crash is checked against the receiver's log.**
  - *Landed.*
    - **The id.** `relay_message_id(run.idempotency_key)`, a UUID5 of `{session}/{call
      id}`, is the relayed message's own id as well as the one the receipt and the text
      name (before, the `Message` carried a second, random id). `create_message` takes
      it as `message_id`.
    - **`reconcile_send`.** Looks for that id in every session the send could have
      reached: the parent named in the sender's header, the children in its roster, and a
      child's siblings in its parent's roster, each read live or from the store. Found:
      `Done`, a receipt with `deliveryStatus: "recorded"`. Absent from every one: `NotDone`.
      Any unreadable log, or a root's sibling send (the other roots are not all live after
      a restart): `Unknown`.
    - **Dispatches asked on resume.** `_reconciled` in `ph.persistence.jsonl` asks the
      tool of each open intent of a kind that declares `reconciled`, as it asks each
      unresolved `tool/call`; `TOOL_DISPATCH` is that kind today.
      `ToolRuntime.reconciled` reads a dispatch record's arguments, which are an object
      where `tool/call`'s are a string.
    - **Repair writes the answer.** `IntentKind.reconciled`, a new optional builder, is
      how a kind writes the settle for an intent its tool says happened; not done is the
      kind's own closer as `not-started`, so only repair and the journal write that
      marker. Repair marks both `reconciled`. `interrupted_turn_closers` takes the
      answers as `intents=`, by opening type and key, and stays a pure fold.
    - **The model is told.** `IntentKind.within` names the call an intent ran inside
      (a dispatch's `rootCallId` and tool). A call left `TOOL_OUTCOME_UNKNOWN` lists the
      answered intents inside it: "`agent_message_send` happened: recorded to …",
      "`write` did not happen." Repair reads no kind's fields to do it.
  - *Gates:*
    - `ph-core` `test_repair.py`:
      `test_a_crashed_cells_dispatch_is_asked_on_resume_and_named_in_its_result` and
      `test_a_dispatch_its_tool_says_never_happened_reads_not_started`, over a crashed
      cell whose dispatch is a `write`;
    - `ph-rlm` `test_messaging.py`: `test_a_send_is_found_in_its_receivers_log_or_ruled_out`,
      `test_a_roots_send_to_a_sibling_can_be_found_but_not_ruled_out`, and
      `test_a_send_a_crashed_cell_made_is_reported_delivered_after_a_restart`, end to end
      over a snapshot resumed by a fresh deployment.
  - *Sabotaged six ways*, and each failed its gate: resume not asking dispatches, repair
    ignoring the answers, the cell's result not naming them, a random message id, the
    message keeping its own id, and the live roots taken for every root.
  - The messaging tests' `_send` now gives each send its own call id, as the harness
    does, since two sends under one call id would now relay one message id twice.
- [ ] **L3** *(optional, low)* — Re-read under a lock and write only the changed key, for
  `settings.json` and `tui.json`.
- [ ] **L7** *(decide first)* — Record the sandbox allowances in the log, if egress should be
  auditable from logs alone.
