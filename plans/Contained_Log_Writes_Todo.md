# Contained log writes — the redesigns Phase 10's cleanup skipped

*2026-09-24. Follows `plans/Intent_Journal_And_Durable_Actions_Plan.md` (Phase 10, landed,
uncommitted) and the `/simplify` pass over it, which skipped five redesigns as "changes to
approved behavior". This document weighs each against the goal below and turns them into
an ordered todo list. Nothing here is built yet.*

## The goal every item is weighed against

**Several sub-agents are running, each with its own session log. Everything stops at once
— the machine shuts down, the daemon is killed. The daemon restarts, and every sub-agent
picks up where it left off.**

What that restart does today, in order:

1. The daemon resumes the root on first use: `Supervisor._start` → `open_session` →
   `resume_session` (`ph/persistence/jsonl.py`). The root's log is repaired
   (`interrupted_turn_closers`), and each started, unresolved call's tool is asked to
   reconcile it.
2. `SubagentService.resume_children` (`ph/seams/subagents.py`) walks the root's roster.
   A child that was `running` goes back on the retry ladder, and one that was `queued` is
   driven again.
3. Each readmitted child's provider (`ph_rlm.subagents._child_session`) resumes the child's
   *own* log through `resume_session` when one reached disk, so repair and reconcile run
   again, once per child.

So the property that matters is: **in the process doing the resume, repair can settle
every intent any of those logs holds, and every consumer reads what happened the same way.**
It must hold whatever that process happens to have imported, and whatever profile it
mounted.

## The principle you set

> No direct appends. The journal holds the behavior that keeps the log durable, and letting
> another package write to the session log breaks that containment.

Today it is the other way round:
- About 80 two-argument `.append(` sites in 34 shipped modules write the log directly: 21
  modules in `ph`, 7 in `ph_stabilize`, 5 in `ph_rlm`, 1 in `ph_app`. A few are not log
  appends (`Inbox.append`).
- The writers-of-record gate (`test_log_writers.py`) checks those sites **statically**.
- Nothing stops code at runtime from writing any type to any `Session` it holds. That is
  F12, which P10-02 left unfixed.

---

## 1. Where intent kinds are declared: "the seam owns the pair", or leaf modules

**Today.** Each kind is declared in the seam that writes it: `SHELL_COMMAND` in
`ph/seams/shell.py`, `APPROVAL_ASK` in `approval.py`, `QUESTION_ASK` in
`user_questions.py`, `TOOL_DISPATCH` in `ph/tools/code_mode.py`, `TOOL_EFFECT` in
`ph/tools/registry.py`, `CLIENT_COMMAND` in `ph_app/daemon/supervisor.py`. Repair sees a
kind only if its module was imported. So `repair._kinds()` imports four named modules
itself, and only inside the function: at module top the import was a cycle
(`ph.orphans` → `ph.persistence` → `repair` → `ph.seams.shell` → `ph.orphans`), which only a
fresh-interpreter test catches.

**Benefits of moving the kinds into leaf modules** — one small, pure-data module per
package, importing nothing above `ph.session`, imported statically by whatever needs it:
- **Repair's correctness stops depending on import order.** Across a mass restart, every
  child log must be settled by whatever process resumes it. Today a kind whose module wasn't
  imported leaves its orphans open, and the only warning is a sentence in a docstring. With
  the core leaf imported at repair's top, and a vocabulary table of the types that open
  intents, repair either has the kind or refuses the resume by name.
- **The cycle hazard goes away.** The lazy import in `_kinds()` and the subprocess test
  that pins it are workarounds; a leaf module has nothing to cycle through.
- **A package's kinds are settled without mounting its rows.** The trajectory viewer
  imports `ph_app.kinds` statically and settles `CLIENT_COMMAND` with no daemon row mounted.
  A process that lacks the leaf refuses by name rather than skipping it.
- **One place to audit all of a package's pairs and closers**, instead of six seams.
- **It fits the one-door principle.** The kinds module *declares*, and the journal *writes*.
  "The declaring module is the writer of record" stops being a fiction: today
  `WRITERS["shell/*"]` names `ph.seams.shell`, a module that never appends.

**Costs of moving:**
- **The pair's definition moves away from the code that fills it.** The closers now reuse
  each seam's live payload builders (`ApprovalService._decided_data`,
  `UserQuestionService._answered_data`), so those builders must move into the leaf module
  as pure functions, or be duplicated.
- **Seam constants move too**, such as `INTERRUPTED` and the shell's `interrupted` reasons.
- **ph-core cannot host a package's kinds**, so this is one leaf per package, each imported
  by its package's own resuming modules. (Decided below to be a consolidation, not a cost.)

**Decided (2026-09-24): move.** One `kinds` leaf per package, and **no lazy imports for
kinds anywhere**. The pitfalls are the ones Phase 10 already hit:
- *import order decides behavior*: repair settles only what happens to be imported;
- *hidden cycles* that surface only in a process importing things in a different order
  (`ph.orphans` first);
- *failures that arrive late*, at the first resume rather than at import.

The cost listed above — one leaf per package — is not a cost here. It **consolidates what a
package brings with it** into one place a reader can open.

**How the kinds are found — statically, so the checker and the tests see it, and nothing is
discovered at runtime.** The point of the types is to catch a mistake at build time. A lazy
import, or an entry-point scan, moves the discovery back to runtime: a missing or misspelled
registration would first show up as an intent a restart left open. So:
- **Every kind is reached through a static, module-top import.**
  - ph-core's leaf, `ph/session/kinds.py`, imports only `ph.session.intents`,
    `ph.session.events` and `ph.json`. `ph.persistence.repair` and the seams import it at
    the top.
  - A package's leaf, `ph_app/kinds.py`, is imported at the top by the package's own modules
    that fill its pairs or resume its logs: the daemon's supervisor, the runtime,
    the trajectory viewer.
  - mypy sees every one of those imports.
  - **No entry-point group.**
- **ph-core's vocabulary says, statically, which types open an intent.** A table beside
  `KNOWN_SESSION_EVENT_TYPES` lists every opening type, `client/command` included. It lives
  there for decision 7's reason: ph-core's reader already knows every type a package writes.
  - That is what lets repair notice a kind it was never given. In a process that did not
    import `ph_app`, an open `client/command` is no longer silently left open: the resume is
    **refused, loudly, by name** ("this log holds open `client/command` intents and this
    process has no kind for them").
  - A *third-party* package's intent types are declared log types, not core ones, and a log
    carrying a required type the reader doesn't know is already refused at seed. So no
    process can quietly skip an intent it cannot settle.
- **Completeness is a gate, run before anything ships:**
  - every `IntentKind(...)` in shipped code lives in a package's `kinds.py`;
  - the vocabulary's opening-type table equals the union of every leaf's opening types, in
    both directions;
  - each package's resuming modules import their leaf at module top;
  - no function-level import of a kinds module exists anywhere.
- **Leaf purity is a gate too.** A leaf may import only the stdlib, `ph.session.intents`,
  `ph.session.events` and `ph.json`, so importing one can never cycle back through ph-core.
- **What a leaf holds.** The package's `IntentKind` declarations, their key and closer
  functions, and the pure payload builders both the closer and the seam's live settle call:
  - approval's decided payload;
  - the question's answered payload;
  - the shell's interrupted result;
  - the dispatch identity;
  - the effect settle.

  Seam constants those need move with them (`INTERRUPTED`, the shell's `interrupted`
  reasons), and the seams re-export them. After T6 the leaf is also where a package's log
  writers are minted: one module per package that says everything it brings to the log.
- **Decision 7 stands.** Types a satellite bundle writes (`todo/write`, `harness/refined`)
  stay in ph-core's vocabulary, because ph-core's reader must know them even where the bundle
  is not installed. A leaf declares a package's *kinds*, and any type ph-core's reader does
  not need.

## 2. Keys that aren't unique: `dedupe=False` and the relaxed `settle`

**Today.** An approval is keyed `callId or toolName`, and some real callers send no call
id: the Continual Harness asks `approval.request(tool_name="refine")`
(`ph_rlm/harness/service.py:330`). When two such asks run at once, the second open replaces
the first in the fold. So the journal:
- has a `dedupe` flag, false for both asks;
- relaxes `settle` to check only the key for those kinds;
- still needs a `raise RuntimeError` at three callers that can never fire, just to narrow
  `Claim | Prior`.

**Benefits of unique keys** — keying each intent by its opening record's own seq, as
`SHELL_COMMAND` already does, with the settle carrying `askSeq`:
- **No orphan hides after a restart.** A replaced ask is invisible to the fold, so repair
  never settles it, and the log stays half-written even though every fold reports it clean.
  With per-intent keys, every open ask is found and settled.
- **`settle` checks the same thing for every kind.** "Settle only your own intent, once"
  holds everywhere, and the relaxed branch goes.
- **Deduping becomes the caller's choice, not the kind's.** Split the door into `open`
  (always a new intent, returns `Claim`) and `open_once` (dedupe by key, returns
  `Claim | Prior`). The `dedupe` flag and the three unreachable raises go.
- **Concurrent asks can be told apart**, which the fold never could.

**Costs:**
- **Payloads change:** `approval/decided` and `question/answered` gain `askSeq`, while
  `callId`, `toolName` and `askId` stay for the readers that pair by them. Format 2 is still
  unreleased, so this needs no further version bump.
- **Every writer of those two settles** has to carry the seq: the seams, repair's closer and
  the "never" policy path.

**Recommendation: do it.** It is small, and it removes a correctness gap that matters most
in exactly the mass-restart case.

## 3. What a prior means: one outcome vocabulary, and a re-open policy per kind

**Today.** "This key was seen before" is interpreted three ways:
- questions re-open through `dedupe=False`;
- `TOOL_EFFECT` goes around dedupe with `record()` and decodes `outcome` strings;
- `CLIENT_COMMAND` refuses the call and reports `unknown`.

The closers say "not the act's own settle" four different ways: `interrupted: why`,
`interrupted: true`, `outcome: "unknown"`, `outcome: INTERRUPTED`. On top of that, `claim`
adds `failed: true` to mean *unknown*, while `TOOL_EFFECT`'s `outcome: "failed"` means *a
retry is safe*.

**A latent bug in this area (verified by reading):** two concurrent calls with the same
effect key. The second finds `Prior(settled=None)` and records a new intent. The first call's
`settle` then raises `IntentError`, because it is no longer the open intent, and it raises
outside `_dispatch`'s `try`. No shipped tool declares a key yet.

**Benefits of one vocabulary:**
- **Every consumer reads a restart's aftermath the same way.** After a mass restart most
  intents are settled by repair, and the daemon's repeat reply, the tool pipeline, the
  parent's roster and the TUI all ask "did this finish?". Every settle not written by the act
  would carry one marker, `outcome: unknown | not-started | failed` plus who wrote it, read by
  one function.
- **`Prior` knows what it is:** settled by the act, settled by a closer, **running in this
  process**, or **orphaned by a dead one**. `session.first_live_seq` already tells the last
  two apart. The restart case is exactly "orphaned".
- **Each kind declares what a prior means** (answer it, refuse it, or open again), instead
  of each caller decoding strings. That fixes the latent bug structurally: a prior running
  in this process is never re-opened underneath itself.

**Costs:**
- **The six kinds' settle payloads gain one common field.** Readers (TUI, trajectory,
  daemon repeat) switch to the one reader function.
- **It needs design before code.** Where the marker lives, and whether `outcome` is per kind
  or shared.

**Recommendation: do it, before items 2 and 4** so the vocabulary is settled once. Fix the
latent bug first as a small standalone change, since this item subsumes it later.

## 4. Credentials across a restart: check before anything starts; store nothing

**Decided (2026-09-24): pH does not store credentials.** Values stay where they are today:
the daemon's environment, and the in-memory `CredentialService._overrides` a client fills
through `credentials/store`. Keeping a credential across a restart is left to a future
**secure plugin**: a source slot on the credential seam that a keychain or vault row could
fill. Nothing is built for that now, beyond not closing the door.

**What a restart does today, and why that is the problem.**
- A resumed root or child whose provider's key is missing starts anyway.
- The first request then fails at the adapter edge (`resolve_secret` in
  `ph_app/adapters/_http.py` raises `MISSING_CREDENTIAL`).
- For a child, that failure spends its retry ladder and can end with the child failed, all
  for a key a person could have supplied in a second, had anything said it was missing.

**The check.** It follows the pattern of the other open-time sweeps: the spill store's
`sweep_session` reports "a blob the log names that is nowhere", and `WorkspaceSeam.reconcile`
checks the trees the log names. Both run on `session/created`, which is also the resume and
fork path. The credential check is the same shape, with one difference: **it gates, it does
not only report.**
- **Read the journal for what the session depends on.** Its `request/context` records say
  which providers it has been routing to. Its configured route covers a session that has
  not sent a request yet.
- **Map each provider through the profile to the credential it names.** Adapter rows name
  theirs under `apiKeyEnv`, which `credentials_named` already walks. Then ask
  `CredentialService.has()`.
  - Checked against *this* profile rather than a name copied into the log. The question is
    "can this session run here, now", and a profile that renamed the variable must be asked
    about the new name.
- **A session missing one does not start.**
  - The supervisor asks before it drives a resumed root's turn. `resume_children` asks
    before it readmits a child.
  - A held session waits in a `needs-credential` state that names what is missing, and is
    shown in `phern agents doctor`, `sessions/list` and the TUI.
  - Holding does not spend a retry.
  - A `credentials/store` of that name releases every session waiting on it.
- **Per session, not per daemon.** One missing key holds the roots and children that need
  it, and everything else resumes.
- **Said at startup too.** After a restart the daemon logs one line per missing credential,
  naming it and the sessions holding on it.
- **Recorded through the journal, never as a direct append:**
  - the hold is written as a record when it happens, and its release when the name arrives;
  - so a transcript shows why a child paused, and a second restart before the key arrives
    holds the child again rather than finding a mystery.
  - Names only, never values. `CredentialRef` is documented as safe to log (I-3).

**`credentials/store`'s key.** It can also go unkeyed, since storing one value twice is a
no-op. That removes `IntentScope`, `_lives`, the `"scope"` stamp and `Mutation.key_scope`,
all of which exist for this one verb. The cost is that it loses the uniform `repeated` reply,
and its callers move from `mutate` to `call`.

**Not covered yet, and said where it would be assumed:**
- **Credentials a *tool* needs.** An MCP server token or a search API key is named in its row
  too, but the log records which tools were called, not which credential each would ask for.
- First version: the provider route only.
- A tool that declares its credential joins the check when one does.

## 5. One door for every write

**Today.**
- Intent pairs go through `ctx.intents`, but everything else calls `Session.append`
  directly: chunks, messages, turn markers, accounting, posture.
- Two paths write a declared kind's types past the journal:
  - the approval "never" policy (`_record_asked` / `_record_decided`);
  - the Code Mode dispatch refused before it started (a bare `tool/code-dispatch`).
- The writers gate credits a declaring module with both of its kind's types, so it cannot
  tell a journal write from a bypass. That bypass is exactly the F8/F9 regression the
  journal exists to prevent.

**Benefits of one door**, as you framed it:
- **Durability rules have one home.** Barriers, batch membership, keys and the "on disk
  before the act" rule can't be skipped by a writer that never heard of them.
- **Writers of record are enforced at runtime, not only by the AST gate.** This closes F12
  properly, where P10-02 had to decline: the check no longer needs to know which row is
  running, because the right to write a type is an object only its owner holds.
- **Another package can write only the types it declared.** It can never write a core type,
  such as a posture record, or another package's types.
- **It is simpler to reason about for the restart goal:** every record in every child's log
  came through code that knows how repair will read it.

**Design, briefly:**
- `Session.append` becomes private.
- Every write goes through a **writer** minted by a declaration:
  - `declare_log_type`, and the core table's owners, hand each owning module a `LogWriter`
    for its types;
  - the journal holds the writers for declared kinds;
  - `LogWriter.append(session, type, data, surface)` refuses a type that is not its own.
- Replicas and seeds keep `admit`, since they mirror a log and do not write one.

**Costs:**
- **Migration:** about 34 modules across four packages. That includes the agent loop's hot
  path (`assistant/chunk` per streamed token), where the writer must stay one method call
  and a set lookup, backed by a benchmark gate.
- **Tests build logs with `session.append` in hundreds of places.** They need a test-only
  writer from `ph.testing` and a mechanical rewrite.
- **Python can't make a writer truly unforgeable.** Importing another module's writer is a
  deliberate, reviewable act, which the AST gate turns into a failure.
- **The two remaining pair bypasses need journal doors of their own:**
  - "open and settle together", one `Session.batch()`, for the "never" policy;
  - "settle with no open", for a dispatch refused before it started.

**Recommendation: do it, last.** It is the largest item, and items 1–3 decide what the
journal's doors are.

---

## Todo list

In order. Each item says what "done" means as a gate. Every gate is sabotage-checked, as in
Phase 10.

- [ ] **T0 — Fix the concurrent-effect settle** (item 3's latent bug; small, standalone).
  - The fix: a keyed call whose prior is still running *in this process* must not open a
    second intent underneath it.
  - *Gate:* two concurrent calls with one effect key both return; the first's settle does
    not raise, and the far side's count matches what ran.
- [ ] **T1 — Write the goal as a gate first.** An end-to-end mass-restart test:
  - a daemon with a root and several sub-agents mid-work (a `!!` running, a `write` in
    flight, an approval parked, a keyed effect, a queued child);
  - stopped without teardown, then restarted over the same `$PH_HOME`.
  - *Gate:* every child is readmitted or failed by the ladder; every intent in every log is
    settled or reconciled; no log is refused; a second restart appends nothing.
  - Written before T2–T6 so each tightens it; marked `xfail` where today's behavior falls
    short.
  - Include **L2**: child sessions are opened without the I-5 lease, and a mass restart is
    exactly when two writers could meet on one child log.
- [ ] **T2 — One outcome vocabulary and a re-open policy per kind** (item 3).
  - One marker on every settle the act did not write.
  - `Prior` exposes the outcome and whether it is *running here* or *orphaned*.
  - `IntentKind` declares what a prior means.
  - Readers (daemon repeat, tool pipeline, TUI, trajectory) use one reader function.
  - *Gate:* each kind's closer, `claim`'s failure settle and the journal's `not-started`
    settle read back through the one function as the right outcome.
- [ ] **T3 — Unique intent keys, and split `open`** (item 2).
  - Approvals and questions keyed by their opening seq; `askSeq` on their settles.
  - `open` (always new) and `open_once` (dedupe) replace the `dedupe` flag.
  - `settle` is strict for every kind.
  - *Gate:* two concurrent `tool_name="refine"` asks are settled separately; a crash with
    both parked settles both on resume; the three unreachable raises are gone.
- [ ] **T4 — Kinds as data, one leaf per package, found statically** (item 1, decided).
  - **The leaves.**
    - `ph/session/kinds.py` for ph-core (`SHELL_COMMAND`, `APPROVAL_ASK`, `QUESTION_ASK`,
      `TOOL_DISPATCH`, `TOOL_EFFECT`) and `ph_app/kinds.py` for phern (`CLIENT_COMMAND`), each
      with the pure payload builders its closers and live settles share.
    - Every consumer imports them at module top. No entry-point group.
  - **The vocabulary** gains its opening-type table. Repair refuses by name a resume whose log
    holds open intents of a kind this process never declared.
  - **Remove the eight function-level imports Phase 10 added in shipped code**:
    - `repair._kinds()`: two, replaced by the core leaf's static import;
    - `jsonl._reconciled`: four. `ToolRuntime.reconciled` takes the raw `tool/call` record
      and parses its own arguments, so persistence needs `ph.tools` only for type checking.
      `.repair` and `..keys` move to the top of the module;
    - `ph.testing.isolated_intent_kinds`: two, imported at top once the leaf exists.

    While there, drop `resume_session`'s two older function-level imports: `Session` is
    already imported at the top of the module, and `interrupted_turn_closers` can be.
  - `isolated_intent_kinds` stops needing to import anything first. The writers gate credits
    the leaf, not the seam. P10-03's idea — a package's own `declare_log_type` calls — lives in
    its leaf.
  - *Gates:*
    - **leaf purity**, sabotaged by adding a seam import to one leaf;
    - **completeness**: every `IntentKind(...)` is in a `kinds.py`; the opening-type table
      matches the leaves in both directions; each resuming module imports its leaf at top;
    - **no function-level import** of a kinds module, as an AST check;
    - **the loud refusal**: a log with an open `client/command`, resumed by a process that
      never imported `ph_app`, is refused by name;
    - T1's restart scenario still settles every kind.
- [ ] **T5 — Nothing resumes without its credentials** (item 4). No credential storage;
  leave a source slot for a future secure plugin.
  - A check on open/resume, shaped like the spill sweep:
    - read the session's routes from its journal (`request/context`, else its configured
      route);
    - map each through the profile's `apiKeyEnv` to a credential name;
    - ask `CredentialService.has()`.
  - A session missing one is held, not started:
    - the supervisor asks before driving a resumed root, and `resume_children` asks before
      readmitting a child;
    - it waits as `needs-credential`, shown in the doctor, `sessions/list` and the TUI;
    - its retry ladder is untouched;
    - a `credentials/store` of the name releases it.
  - Hold and release are journal records (names only). The daemon's startup summary lists
    each missing name and the sessions holding on it.
  - Unkey `credentials/store` and remove `IntentScope` with everything that exists only
    for it.
  - A `NON_GUARANTEES` row: a credential given over the wire is gone after a restart, and
    the sessions that need it wait for it, by name.
  - *Gate:*
    - T1's scenario, restarted with one provider's key absent: that root's children are held
      (not failed, no ladder spent), every other session resumes, and the doctor names the
      key;
    - supplying it releases them;
    - a second restart before supplying it holds them again and grows nothing.
- [ ] **T6 — One door for every write** (item 5, your principle). In stages:
  - **T6a:** the `LogWriter` design; the journal holds kind writers and gains the two
    missing doors ("open and settle together", "settle with no open").
  - **T6b:** migrate ph-core's 21 modules, with a hot-path benchmark on `assistant/chunk`.
  - **T6c:** migrate ph-stabilize, ph-rlm and phern.
  - **T6d:** a `ph.testing` writer and the mechanical test rewrite.
  - **T6e:** enforcement:
    - `Session.append` private;
    - the AST gate becomes "nothing outside `ph.session` appends, and a writer is used only
      in its owner module";
    - the runtime refuses a type that is not the writer's own, which closes F12 and
      replaces P10-02's "not shipped".
  - *Gate:* T6e's two gates, sabotage-checked. A third-party row appending `sandbox/mode`
    is refused at runtime. The `WRITERS` table is derived from the writers minted, not
    maintained by hand.

## Next todo list, after this one

Two areas the cleanup pass also flagged, to be turned into their own list once the one
above is done:

1. **Public API used only by tests**:
   - `IntentJournal.pending` and `.outcome`;
   - `settled_record` and `is_declared` in `ph.session.__all__`;
   - `ToolRunContext.idempotency_key`;
   - the seams' `pending_approvals` / `pending_questions` now that repair no longer calls
     them;
   - anything T6 leaves public that only a fixture reaches.

   For each, keep it (a real consumer is coming), make it private, or delete it.
2. **Test consolidation**:
   - one shared AST walker for `test_log_writers.py` and
     `test_importing_repair_declares_every_core_kind`, on top of
     `tests/workspace_layout.py`;
   - drop the per-package `*_in_the_vocabulary` tests (phern, ph-stabilize, ph-rlm) that the
     cross-package gate now covers;
   - derive the test intent kinds from the real ones;
   - retire `test_the_fold_agrees_with_the_folds_it_replaces`, which now compares the fold
     with itself.
3. **Function-level imports across the codebase.**
   - T4 removes the ten on the kinds and resume path.
   - That leaves about 54 elsewhere: 36 in `ph` at the time of counting, then `ph_app`,
     `ph_text_index`, `ph_code_graph`, `ph_rlm` and `ph_runtime`.
   - Each one either:
     - moves to module top;
     - becomes a leaf split, where it was dodging a cycle;
     - or stays as a named, justified exception, such as an optional heavy dependency a
       process may not have installed.
   - An AST gate holds the list, so a new one is a decision, not a habit. The same principle
     as T4: a mistake the checker or a test can catch should not wait for a process that
     happens to import things in a different order.
