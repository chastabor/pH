# Contained log writes — the redesigns Phase 10's cleanup skipped

*2026-09-24. Follows `plans/Intent_Journal_And_Durable_Actions_Plan.md` (Phase 10,
committed in `ccbdcd9`) and the `/simplify` pass over it, which skipped five redesigns as
"changes to approved behavior". This document weighs each against the goal below and turns
them into an ordered todo list.*

**Status, 2026-09-25. Closed.** Every row has landed and is committed: T0–T6 in `69d6a81`,
and L2's fix, N1–N3 and the fixes after them in `0b604fd`. They ship in 0.4.0 with daemon
protocol 4 and session log format 2. Left open on purpose, each stated where it applies:
- **A settled child's lease is held until its root unwinds**, so
  `phern -p --session <child>` is refused while the root is mounted. Releasing it earlier
  would mean disposing the child's session when it settles (T1's L2 note).
- **Credentials a tool needs are not checked on resume**, only the provider route's
  (`NON_GUARANTEES`, "credentials across a restart").
- **Keeping a credential across a restart is left to a future secure plugin.**
  `CredentialRef.source` is the slot; nothing is stored (item 4).
- **Two cleanups the last `/simplify` pass skipped:** probing only the modules that sit
  in an import cycle, and the older hand-written wait loops in `test_daemon_asks.py`.

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

- [x] **T0 — Fix the concurrent-effect settle** (item 3's latent bug; small, standalone).
  - The fix: a keyed call whose prior is still running *in this process* must not open a
    second intent underneath it.
  - *Gate:* two concurrent calls with one effect key both return; the first's settle does
    not raise, and the far side's count matches what ran.
  - *Landed.*
    - A prior opened at or after `session.first_live_seq` and not settled is **running
      here**. The second call is refused with `TOOL_EFFECT_IN_FLIGHT`, naming the running
      call, rather than run or opened underneath it. After the first returns, a retry is
      answered from the log.
    - A cancellation with the body entered now settles the effect `unknown` (a
      `BaseException` around `_dispatch`). Otherwise the intent would read as running here
      and refuse every repeat until a restart.
    - Gates, in `test_tools_idempotency.py`:
      - `test_two_concurrent_calls_with_one_effect_run_it_once`;
      - `test_a_canceled_keyed_call_leaves_its_effect_unknown_not_running`.
    - Both are bounded by `fail_after`: the first sabotage, with no in-flight check,
      *hangs* an unbounded test, because the second call waits on the release the test only
      sets after it. Sabotaged both ways; each failed its gate.
- [x] **T1 — Write the goal as a gate first.** An end-to-end mass-restart test:
  - a daemon with a root and several sub-agents mid-work (a `!!` running, a `write` in
    flight, an approval parked, a keyed effect, a queued child);
  - stopped without teardown, then restarted over the same `$PH_HOME`.
  - *Gate:* every child is readmitted or failed by the ladder; every intent in every log is
    settled or reconciled; no log is refused; a second restart appends nothing.
  - Written before T2–T6 so each tightens it; marked `xfail` where today's behavior falls
    short.
  - Include **L2**: child sessions are opened without the I-5 lease, and a mass restart is
    exactly when two writers could meet on one child log.
  - *Landed* — `tests/test_mass_restart.py`, driven by `tests/mass_restart_host.py`.
    - The host is a real process that the test **`SIGKILL`s**, so no teardown runs. It
      leaves the listed work in flight:
      - the root's `write` landed but held in `tools/post-execute`;
      - a `!!` intent;
      - an approval on an answerer that never answers;
      - a keyed effect mid-body;
      - two children under a concurrency of one.
    - A fresh mount then does `Supervisor._start`'s two calls: `resume_session`, then
      `resume_children`.
    - **Today's code already passes the whole goal.** Both children finish; every root
      intent is settled or reconciled (the `write` reports done off the file); every log
      reads back with nothing open; a second restart closes nothing.
    - L2 is a strict `xfail`: a readmitted child's log is not leased by the process that
      resumed it.
    - **L2 closed after T6.**
      - `open_session` moved from `ph_app.runtime` into `ph.persistence.opening`.
        ph-rlm sits below phern and could not reach the door, which is why children
        were never claimed. It gained `meta` for the child's header, and after the
        /simplify pass `meta` is the only way in: the supervisor passes
        `meta={"cwd": …}` where it used to pass `cwd=`.
      - `_child_session` now opens every child through it, fresh or resumed. The
        lease is held on the mount's scope (`ctx.root`) like the root's, since a
        child's session lives in the deployment's store for as long as the root is
        mounted, and the mount's last act writes it before the lease goes.
        `open_session` claims on `ctx.root` itself, so no caller can hand the lease
        a shorter life.
      - One scope now holds a lease per child, and each claim registered a last write
        that walked every live session: a root with 200 children spent 472ms
        unwinding, against 260ms with one walk. Each claim still registers, since
        only the newest registration sits above the newest lease, but the first to
        run writes everything and the rest return (`write_on_unwind`'s `_OWED`).
        `test_persistence.py::test_a_scope_holding_many_leases_writes_each_log_once_while_it_is_held`
        holds both halves: dropping the check writes each log once per claim, and
        registering once gives the second lease back before its log is written.
      - Not changed: a settled child's lease is held until the root unwinds, so
        `phern -p --session <child>` is refused while its root is mounted. Releasing
        it at `_quiesce` would mean disposing the child's session there.
      - A readmission refused by another holder is settled "could not be resumed".
      - The xfail is removed. `test_subagents.py` gains
        `test_a_live_childs_log_refuses_a_second_opener`.
      - Its in-process restart tests now resume from a snapshot of the logs: the
        first harness, still alive, rightly holds its children's leases.
      - Sabotaged by opening the child directly again; both tests failed.
      - Two collisions I first listed (a daemon restarting inside the old one's
        grace period, a second daemon sharing `$PH_HOME`) were already covered by
        the root's lease, since children are readmitted only by whoever starts the
        root. The real gap was a second opener naming the child's own id.
    - Sabotaged by emptying repair's intent pass; two of the three gates failed.
- [x] **T2 — One outcome vocabulary and a re-open policy per kind** (item 3).
  - One marker on every settle the act did not write.
  - `Prior` exposes the outcome and whether it is *running here* or *orphaned*.
  - `IntentKind` declares what a prior means.
  - Readers (daemon repeat, tool pipeline, TUI, trajectory) use one reader function.
  - *Gate:* each kind's closer, `claim`'s failure settle and the journal's `not-started`
    settle read back through the one function as the right outcome.
  - *Landed.*
    - **The marker.** `"unsettled": {"why": <Unsettled>, "by": "repair" | "process"}` is
      merged into every settle the act did not write, by repair and by the journal (a failed
      barrier, a raising `claim`), and by the pipeline for a canceled keyed call. The
      closers stop writing their own:
      - the shell's `interrupted: why`;
      - the dispatch's `interrupted: why`;
      - the effect's and the daemon verb's `outcome: "unknown"`;
      - `claim`'s `failed: true`.

      A kind's domain fields stay: an approval's `outcome`, a question's `resolution`,
      which their own readers use. (A question's `interrupted` stayed too, until the
      post-T6 cleanup found no reader and dropped it.) `by` is `repair` or `process` — the
      distinction that matters after a restart, a dead process against a live one.
    - **The reader.** `outcome_of(kind, settle)` returns
      `done | failed | outcome-unknown | not-started`, with `failed` read by the kind's new
      `IntentKind.failed` (the effect's `isError`). `unsettled_why(data)` serves readers
      that hold only a payload (`shell_body`).
    - **`Prior`** is its own dataclass again, carrying:
      - `outcome`;
      - `here`, meaning opened at or after `first_live_seq`;
      - `running_here` (and `orphaned`, dropped in the post-T6 cleanup: no caller).
    - **The re-open policy.** `IntentKind.reopen` is the set of outcomes after which `open`
      opens the key again instead of answering with the prior. `TOOL_EFFECT` reopens on
      `failed` and `not-started`; every other kind answers every prior.
    - The daemon's `_repeat_outcome` and the pipeline's `_open_effect` read `Prior` rather
      than decoding strings.
    - Gates in `test_repair.py`:
      - `test_every_core_kind_has_a_sample`;
      - `test_what_repair_writes_reads_back_as_outcome_unknown_for_every_kind`;
      - `test_what_the_journal_writes_reads_back_through_the_same_function`;
      - `test_an_effect_that_reported_failure_reads_back_as_failed`.

      Plus phern's `test_a_verb_repair_closed_reads_back_as_unknown`. Existing tests moved
      onto the one reader.
    - Sabotaged three ways — repair not stamping, the reader ignoring the marker, the
      journal's not-started unmarked — each failed its gates.
- [x] **T3 — Unique intent keys, and split `open`** (item 2).
  - Approvals and questions keyed by their opening seq; `askSeq` on their settles.
  - `open` (always new) and `open_once` (dedupe) replace the `dedupe` flag.
  - `settle` is strict for every kind.
  - *Gate:* two concurrent `tool_name="refine"` asks are settled separately; a crash with
    both parked settles both on resume; the three unreachable raises are gone.
  - *Landed.*
    - **Keys.** `APPROVAL_ASK` and `QUESTION_ASK` key an ask by `str(event.seq)` and its
      settle by the new `askSeq` field, which every writer of `approval/decided` and
      `question/answered` now carries: the live settle, the `never`-policy pair, both
      kinds' closers, and the not-started settle. `callId`, `toolName` and `askId` stay
      in the payloads; no reader pairs by them (the TUI rows each record alone).
    - **Two opens.** `IntentJournal.open(session, kind, data) -> Claim` is always a new
      intent. `open_once(..., key_scope=) -> Claim | Prior` keeps the dedupe, the re-open
      policy and the process scope. Likewise `claim` and `claim_once`. `IntentKind.dedupe`
      is gone. Callers:
      - the approval and question seams, and phern's `run_shell`, use `open` / `claim`;
      - the tool pipeline's `_open_effect` uses `open_once`;
      - the daemon's `_mutate` uses `claim_once`.
    - **Strict settle.** `settle` refuses any claim that is not the open intent under its
      key, for every kind. The relaxed rule existed only because two asks of one tool
      shared a key.
    - **The three raises.** The `RuntimeError`s in `ApprovalService.request`,
      `UserQuestionService.ask` and `run_shell`, which narrowed `Claim | Prior` to `Claim`,
      are deleted. `open`'s return type now makes them unreachable, and mypy checks it.
    - Gates:
      - `test_seams.py::test_two_asks_of_one_tool_at_once_are_settled_separately`: two
        `refine` asks with no call id, answered in reverse order; each decision's
        `askSeq` names its own ask;
      - `test_repair.py::test_asks_parked_under_one_name_are_each_settled_on_resume`: two
        `refine` approvals and one question asked twice under one id, taken from the log
        while all four are parked; repair settles each once, by its own seq.

      Sabotaged by restoring the name keys (call id or tool name; `askId`): both gates
      failed.
    - `test_the_fold_agrees_with_the_folds_it_replaces` is retired. The folds it compared
      used the name keys. Hand-built logs in `test_repair.py` and `test_seams.py` now
      carry `askSeq`.
- [x] **T4 — Kinds as data, one leaf per package, found statically** (item 1, decided).
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
  - *Landed.*
    - **The leaves.**
      - `ph/session/kinds.py` holds the five core kinds, their key functions and
        closers, and the builders the closers and live settles share:
        `approval_decided`, `question_answered`, `dispatch_identity`, `effect_settle`
        and `command_seq`.
      - Constants moved with them: `INTERRUPTED`, `AskResolution`,
        `DISPATCH_INTERRUPTED`. The approval and question seams re-export theirs.
      - `ph_app/kinds.py` holds `CLIENT_COMMAND` and `command_settled`.
      - The seams, the registry, Code Mode, the daemon and the shell import their
        kinds from the leaves. No seam re-exports a kind.
    - **Each package imports its own leaf from its `__init__`** (`ph.session`,
      `ph_app`). This is stronger than the plan's "each resuming module imports it":
      loading any part of a package declares its kinds, so a new resume path cannot
      miss them. Repair gets ph-core's through `ph.session` and imports no leaf itself.
    - **The vocabulary** gains `INTENT_PAIRS`, which maps each opening type to its
      settling type and its leaf.
      - A log with an open intent of a kind the process never declared raises
        `UndeclaredIntentError` (exported from `ph.persistence`), naming the type and
        the leaf. The log is left as it was.
      - Rule 6, in repair's docstring: without the kind its records cannot be paired,
        so they are counted. That is exact, since T3 made each settle close one intent
        and nothing writes a settle with no opening. (T6's `settle_unopened` door was
        the one exception; the post-T6 cleanup found its only caller was dead and
        removed both.)
      - A settled verb resumes normally.
    - **The ten function-level imports are gone:**
      - `repair._kinds()`'s two;
      - `_reconciled`'s four;
      - `resume_session`'s two;
      - `isolated_intent_kinds`' two.

      `ToolRuntime.reconciled(record, session, *, scope)` now reads the tool's name
      and arguments off the record itself, and returns `None` for `Unknown`, so
      persistence needs `ph.tools` only for its types.
    - `isolated_intent_kinds(core=True)` now starts from exactly
      `ph.session.kinds.KINDS`, whatever the interpreter has imported. That table is
      what "a process without the app" holds.
    - **`WRITERS` credits the leaves.**
      - `ph.seams.shell`, `ph.seams.user_questions` and `ph.tools.registry` are no
        longer writers.
      - `ph.seams.approval` keeps the `never`-policy pair.
      - `ph.tools.code_mode` keeps only its refused-dispatch `tool/code-dispatch`,
        both of which are direct appends for T6.
    - **One behavior change:** repair's effect settle is built by `effect_settle` and so
      gains `"content": []`, the same shape as the pipeline's cancel settle.
    - No `declare_log_type` ships, so none moved. `ph_app.kinds` notes that the leaf is
      where one would go.
    - **Gates:**
      - `test_intent_kinds.py`:
        - both leaves found;
        - every `IntentKind(...)` in a leaf;
        - `INTENT_PAIRS` equal to the leaves' pairs;
        - leaf purity;
        - each package imports its leaf at top;
        - no kinds import below module top, in shipped code or any suite;
        - no function-level import on the resume path;
        - the refusal, in-process through `resume_session`;
        - a settled verb resumes.
      - `test_repair.py::test_a_fresh_process_declares_ph_cores_kinds_and_refuses_the_rest`:
        a fresh interpreter that imports `ph.orphans` first. It has exactly the core
        leaf's kinds and not `ph_app`, and refuses an open `client/command` by name.
        This replaces `test_importing_repair_declares_every_core_kind`.
      - `test_code_mode.py::test_the_leaf_spells_the_dispatch_identity_as_the_model_does`:
        `DISPATCH_REF_KEYS` equals `CodeDispatchRef`'s wire keys.
      - T1 still passes.
    - **Sabotaged seven ways**, and each failed its gate:
      - a `ph.wire` import in the core leaf;
      - the `client/command` row dropped from `INTENT_PAIRS`;
      - a kind declared in `ph.seams.shell`;
      - `ph_app/__init__` not importing its leaf;
      - `isolated_intent_kinds` importing the leaf inside the function;
      - `_refuse_undeclared` skipped;
      - a dispatch key respelled.
- [x] **T5 — Nothing resumes without its credentials** (item 4). No credential storage;
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
  - *Landed.*
    - **Which name a route needs.**
      - `ResolvedModel.credential` is the name an adapter resolves at its edge.
        `resolved(..., credential=)` is required, so no adapter can forget it; the
        three shipped adapters pass their `api_key_env`.
      - `ph.seams.credentials.missing_credential(ctx, provider, model)` asks the
        mounted adapter for the name and `has()` for the value, and never sees one.
      - A route with no adapter is not reported as missing a credential.
    - **Checked against the route the session will run on, not `request/context`.**
      That is a deviation from the plan's wording.
      - The log's routes are the ones it has used. After a profile edit they need
        not be the one a resume runs, and the question is "can it run here, now".
      - A daemon root runs on its agent's options (the supervisor's route).
      - A child runs on `child_route(request)`, "its selector, else its parent's".
        That rule is now stated once in the seam and used by ph-rlm's
        `_resolve_model`.
    - **The hold is a journal kind.**
      - `CREDENTIAL_WAIT` (`credential/needed` → `credential/supplied`, in
        `ph.session.kinds` and `INTENT_PAIRS`) is `owner-settles`: repair leaves it.
        It is keyed `holder:name`, where the holder is a run id or `session`.
      - Its records hold names only and are ignorable.
      - `record_wait(ctx, session, holder, name | None)` brings a holder's holds in
        line with what it waits for now. It settles the others, and opens one
        through `open_once` unless one is already open, so asking again appends
        nothing. `reopen={"done"}` lets a released name be needed again.
      - The journal's new `open_claim(session, kind, key)` hands the owner of an
        `owner-settles` kind the claim on an intent it did not open in this call.
        It refuses any other kind.
      - `credential_waits(events)` is the fold every reader uses.
    - **Children** (`SubagentService._readmit_children`).
      - Before a queued child is readmitted, its route is checked. A missing name
        records the hold and skips the child, which stays `queued`: live, no start
        counted, and so no rung of the ladder spent.
      - `readmit_waiting(parent)` runs the same sweep again when a credential
        arrives.
    - **Roots** (`Supervisor`).
      - `_start` checks the root's own route before its children, and sets
        `Root.needs_credential`. `_run` does not drive a held root, so a prompt
        waits in the inbox.
      - `status` reads `needs-credential`, which is in `QUIET` for `waiting`'s
        reason.
      - The `credentials/store` handler calls `credential_supplied(root)`. That
        re-checks the root and rings its inbox if it was released, then readmits
        the children waiting on the name.
    - **Said where a person looks:**
      - one daemon log line per missing name when a root comes back;
      - a `credentials awaited` section in `phern agents doctor` (`Supervisor.awaited`);
      - `sessions/list`'s status;
      - TUI transcript rows and trajectory records for both types.

      A held child's sidebar row still reads `queued`; its transcript row says what
      it waits for.
    - **`credentials/store` is unkeyed.**
      - It is a `METHODS` row, and the TUI sends it with `call`.
      - `IntentScope`, `key_scope`, `_lives`, the `"scope"` stamp and
        `Mutation.key_scope` are gone.
      - `StoreCredentialParams` refuses `clientId`/`commandId`. Protocol 4 is not
        released yet (0.3.0 shipped protocol 3, and nothing has been pushed since
        `bdb99a1`), so the change rides on 4 and needs no number of its own. The
        version-4 note in `ph_app/protocol.py` records that a 0.3.x client, which
        stamps a key, is refused.
    - Nothing is stored. `CredentialRef.source` is already the slot a future
      secure plugin would fill.
    - A `NON_GUARANTEES` row, "credentials across a restart": the value is gone,
      the work waits for it by name, and tool credentials are not checked.
    - **Gates:**
      - `tests/test_mass_restart.py`, T1's `SIGKILL` with the sub-agents on a
        `keyed` route and the key absent:
        - `test_a_sub_agent_whose_key_a_restart_lost_is_held_then_released`: both
          children held, `starts` unchanged, every other intent settled, and
          supplying the key runs both to done;
        - `test_a_second_restart_before_the_key_arrives_holds_again_and_grows_nothing`:
          the second resume appends exactly `session/end-seed` and
          `session/resumed`.
      - `test_daemon_mutations.py`:
        - `test_a_root_whose_key_is_missing_is_held_and_the_key_releases_it`
          covers the status, the doctor section, `sessions/list`, the prompt
          waiting, and release;
        - the re-send test, rewritten for an unkeyed verb.
      - `test_credential_holds.py`: seven unit tests.
      - `test_adapters.py::test_every_route_names_the_credential_its_adapter_resolves`.
      - `test_repair.py::test_repair_leaves_a_credential_hold_to_its_owner`.
    - **Sabotaged five ways**, and each failed its gate:
      - the store keyed again;
      - children never held;
      - a hold opened on every ask;
      - a held root driven;
      - an adapter naming the default variable.
- [x] **T6 — One door for every write** (item 5, your principle). In stages:
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
  - *Landed.*
    - **T6a, the writer** (`ph/session/writers.py`).
      - `LogWriter.append(log, type, data, surface)` takes a session or an open batch,
        and refuses a type its owner is not the writer of record for (`LogWriteError`).
      - `log_writer(__name__)` mints a module's own writer from its `_WRITTEN_BY` row
        plus the types it declared. It refuses when the owner named is not the
        calling module, read off the frame.
      - `scaffolding_writer()` writes any type, for `ph.testing` only.
      - `Session.append` and `SessionBatch.append` are now `_append`.
      - `admit` is unchanged.
    - **Kinds carry their leaf's writer.** `IntentKind.writer` (keyword-only) replaces
      `owner`, which is now a property. `declare_intent` refuses a writer that does not
      own both types, and the journal writes every pair through it.
    - **The two missing doors:**
      - `IntentJournal.open_settled(session, kind, data, settle)` writes both halves
        in one batch. The approval `never` path uses it; `_record_asked` and
        `_record_decided` are gone.
      - `settle_unopened(session, kind, data)` was for Code Mode's refused dispatch.
        **Removed after T6:** the pipeline writes the start record for every call its
        gate decides, a refusal included (P7-15), so the branch that called it was
        dead — a probe raising there reached no test in any suite. With no settle
        lacking an opening, repair's count for an undeclared kind is exact.
        `test_a_pre_execute_denial_of_a_sub_call_also_fails_the_run` now pins that a
        refused dispatch is an opened-and-settled pair.
      - `WRITERS` loses `ph.seams.approval`'s ask pair and `ph.tools.code_mode`
        entirely.
    - **T6b and T6c: 72 sites in 31 modules**, across ph-core, ph-stabilize, ph-rlm
      and phern, moved to `_LOG.append(…)` by a codemod. `Inbox.append` is left
      alone.
      - Hot path: the writer measures 1.0-1.05× the bare append on `assistant/chunk`.
      - `test_the_writer_costs_the_hot_path_almost_nothing` bounds it at 1.2×,
        alternating the two measurements within each round.
    - **T6d, tests.** `ph.testing.log_event(log, type, data, surface)` goes through
      `SCAFFOLDING`. About 475 test sites were rewritten mechanically, and the star-arg
      ones by hand. Test kinds carry `writer=SCAFFOLDING`. It defers unknown types to
      the session's own F11 refusal, so F11 stays testable.
    - **T6e, enforcement** (`test_log_writers.py`, rewritten).
      - Statically, over every package:
        - every write is the module's own writer, minted at module top and used
          there alone;
        - no import of another module's writer;
        - no `LogWriter(...)` construction;
        - no `scaffolding_writer` outside `ph.testing`;
        - no `._append(` outside `ph.session`.
      - The table equals what the writers write, in both directions.
      - At runtime:
        - a module's writer refuses `sandbox/mode`;
        - `log_writer("ph.seams.sandbox")` is refused from another module;
        - the scaffolding writer is refused outside `ph.testing`;
        - `Session` has no `append`;
        - a declared type is written by its owner and by nobody else.
    - **"`WRITERS` derived from the writers minted" is realized as *checked against*
      them, not *generated from* them.** If mint calls named their own types, any
      module could claim `sandbox/mode`. So `_WRITTEN_BY` stays the authority, since
      the writer is minted from it. The gate derives every module's writes from its
      writer's call sites and kind declarations, and holds the two equal.
    - F12 is closed. The F12/P10-02 caveats beside `SandboxSeam.logged_mode` and
      `approval_policy`, and DESIGN's known-gaps row, now state the residual
      deliberate bypass (rule 6).
    - **Sabotaged eight ways**, and each failed its gate:
      - the ownership check dropped;
      - a module borrowing `ph.seams.sandbox`'s writer;
      - a grant nothing writes;
      - any module minting any writer;
      - `declare_intent` without the writer check;
      - a seam calling `_append` past its writer;
      - the writer rebuilding its type set per call (1.23-1.32×, where the first
        bound of 1.3× missed it and was tightened);
      - and, in T6d, the `record` name colliding with locals, which mypy caught and
        which led to the `log_event` rename.

## Second todo list: what the cleanup passes flagged

*2026-09-24, after T0–T6. The three areas the first list deferred, audited and turned into
rows. Same rules: in order, every gate sabotage-checked, the four gates green, nothing
committed by Claude.*

- [x] **N1 — Public API only tests use.** Audited over Phase 10's and T0–T6's surface: for
  each name, keep it (name the consumer), make it private, or delete it.
  - **Delete**, each with no shipped caller, the tests reaching the same state another
    way:
    - `IntentJournal.outcome`, and `settled_record`, whose only shipped caller it was.
      Tests read `fold_intents(...)[key].settled`.
    - `pending_approvals` / `PendingApproval` and `pending_questions` / `PendingQuestion`.
      Repair stopped calling them in P10-09, and nothing else ever did. Tests ask
      `open_intents(events, APPROVAL_ASK)`, the one fold.
    - `credential_waits`, which T5's cleanup left with no shipped caller: the supervisor
      reads `waiting_for`, off the journal's index. Tests do the same.
  - **Make private:**
    - `IntentJournal.is_open`: only the journal calls it.
    - `record_wait`: only `hold_for_credential` calls it. Its tests move onto
      `hold_for_credential`, driving the fake route's credential name, which exercises
      the real path.
  - **Stop re-exporting** `is_declared` from `ph.session`. The journal keeps it; tests ask
    `kind in declared_intents()`.
  - **Keep:**
    - `ToolRunContext.idempotency_key`: the tool author's half of P10-12, the key a tool
      hands a far side (`Idempotency-Key`). It is documented contract on the object every
      tool body receives.
    - `missing_credential`: the credential seam's own question, "can this route run here,
      now", beside `has`.
    - `LogWriteError` and `UndeclaredIntentError`: the exceptions public doors raise.
    - `EFFECT_MAY_HAVE_HAPPENED`: a sentence tests quote rather than restate.
  - Stale docstrings that name a deleted fold or its keying go with it
    (`daemon/frontend.py`'s "keyed by the same string `pending_approvals` uses").
  - *Gate:* mypy and the suite, since a caller left behind fails to type-check. Nothing
    to sabotage for a deletion; each private rename is checked by an import from outside
    failing mypy.
  - *Landed*, as listed. Deleted: `IntentJournal.outcome`, `settled_record`, both
    `pending_*` folds with their dataclasses, and `credential_waits`. Private:
    `IntentJournal._is_open`, `_record_wait`. `is_declared` is no longer in
    `ph.session.__all__`.
    - Tests now read `open_intents(events, APPROVAL_ASK)` and `fold_intents(...)[key]`.
      Thin `_asked` / `_questioned` helpers in `test_seams.py` and `test_repair.py` wrap
      the one fold.
    - The hold tests go through `hold_for_credential`, with the fake route's name and the
      environment driving each case.
    - The docstrings that promised "re-asking on resume" (the approval seam, the ask desk,
      the question seam, the vocabulary) now say what happens: repair settles the ask.
    - Found on the way, and not fixed here: the daemon's `AskDesk` names an ask on the
      wire by call id or tool name, so two concurrent `refine` asks share a desk entry.
      That is T3's collision, in memory.
      - *Fixed after N3.* Each root's desk names its own asks, `ask-<n>` from its own
        count (`AskDesk.asked`), and coordinates with no other root. The call id and
        the question's `askId` still travel inside the frame.
        - The name is unique within its root only. A front end attached to two roots
          files an ask by the frame's `sessionId` and `askId` together:
          `ph_app.payloads.AskKey`, which both frames that name an ask hand over as
          `key`, and which `ModalHost` now takes.
        - A root started again counts from 1 again. That is safe because
          `passivatable` never releases a root with a front end attached, and the TUI
          takes its own modals down when it loses the daemon.
        - What the shared entry broke:
          - the first ask answered removed the other's entry, so the root read as idle
            while still parked on a person, and a front end attaching then was not
            posed the open ask;
          - the TUI files modals by this name, so `ask.settled` for one withdrew the
            other's modal, and that modal's cancel value went back as the other ask's
            answer.
        - `test_daemon_asks.py::test_the_wire_ask_id_is_the_one_the_log_wrote` argued
          for the old naming by a re-pose on resume that repair replaced, and is
          retired. `test_two_asks_under_one_name_are_two_asks_on_the_wire` replaces it:
          two `refine` approvals, and two questions under one `askId`.
        - Sabotaged by restoring each old name (`call_id or tool_name`, the question's
          own `ask_id`): each fails its own case, on the root no longer reading as
          waiting.
        - Found on the way, in the TUI: `withdraw_ask` took down the wrong modal when
          two were stacked. Textual's `dismiss` gives this screen's waiter its result
          but pops whichever screen is on top, so withdrawing the lower one popped the
          upper one, and that ask never heard back. `PhModal.withdraw` now dismisses a
          modal on top, and marks one under another to dismiss itself when it reaches
          the top.
          - Gate: `test_tui_pilot.py::test_an_ask_is_withdrawn_by_its_root_and_its_name`
            stacks two roots' `ask-1`s and withdraws the lower one.
          - Sabotaged by filing modals by `ask_id` alone, and by withdrawing with a
            plain `dismiss`: both fail it.
- [x] **N2 — Test consolidation.**
  - **The per-package "is in the vocabulary" tests** (ph-rlm's `test_vocabulary.py`,
    phern's `test_every_type_this_package_writes_is_in_the_vocabulary`, and five in
    ph-stabilize) each assert two things.
    - That a type is *known*, which `test_log_writers.py`'s cross-package walk now covers
      for every write in every package. Those halves go, along with the files that held
      nothing else.
    - That a type is *ignorable* or *required*: a per-type decision nothing else pins,
      argued in each test's docstring. Those stay, renamed for what they hold.
    - `known_event_types`' module docstring still says the walker "sees only ph-core". It
      is corrected.
  - **Derive the test intent kinds from the real ones.** `test_intents.py`'s `APPROVAL`,
    `QUESTION`, `DURABLE` and `BUFFERED` claim `approval/*` and `question/*` under the
    call-id-or-tool-name key that T3 removed, so they test a keying rule nothing ships.
    - They become `dataclasses.replace` of real field-keyed kinds (`TOOL_EFFECT`,
      `TOOL_DISPATCH`) with only the property under test changed (barrier, orphan). So
      their keys, closers and payload shapes are the shipped ones.
    - `test_repair.py`'s `COMMAND` (a made-up `command/run` pair) becomes the real
      `SHELL_COMMAND`, which is the between-turns kind it stands in for.
  - *Gate:* sabotage the cross-package walk's known-type check, and confirm the deleted
    per-package halves are not needed to catch a package writing an unknown type.
  - *Landed.*
    - **The vocabulary tests.** ph-rlm's `test_vocabulary.py` and phern's
      known-types-only test are gone.
      - ph-stabilize's five keep their ignorability pins, each renamed for what it holds
        (`..._is_required_not_ignorable`, `..._are_ignorable`), with the reason in the
        docstring.
      - `known_event_types`' module docstring now says the walk covers every package.
      - Sabotaged by giving a ph-stabilize write a misspelled type: the cross-package
        walk failed on its own (`test_every_write_in_every_package_names_a_known_type`,
        and the writers table).
    - **The test kinds.**
      - `test_intents.py`'s fold tests run on the shipped `TOOL_EFFECT` and
        `TOOL_DISPATCH` records.
      - Its refusal tests use `replace(TOOL_EFFECT, …)` with the one field under test
        changed.
      - Its journal kinds are `DURABLE = replace(TOOL_EFFECT, barrier="durable")` and
        `BUFFERED = replace(TOOL_DISPATCH, barrier="buffered")`.
      - The keyless-record test moved to `SHELL_COMMAND`, whose settle key reads `None`
        without a `commandSeq`. The effect and dispatch keys read `""` instead, which
        the journal refuses at the write.
      - `test_repair.py`'s `command/run` pair became `SHELL_COMMAND`, with its variants
        (`owner-settles`, a closer that settles the wrong key) declared into their own
        isolated table.
      - Sabotaged by changing the shipped effect kind's settle key: five of these tests
        failed, where the made-up kinds could not have noticed.
- [x] **N3 — Function-level imports across the codebase.** 55 in shipped code today:
  26 in `ph`, 8 in `ph_app`, 8 in `ph_text_index`, 6 in `ph_code_graph`, 4 in
  `ph_runtime` and 3 in `ph_rlm`. Each is one of three things:
  - **Moves to module top.** Standard library and in-package imports with no stated
    reason, checked for cycles by a fresh-interpreter import of every module.
  - **A leaf split**, where it was dodging a cycle.
  - **A named exception, with its reason:**
    - an optional or heavy dependency a process may not have installed (tree-sitter,
      turso, croniter, opentelemetry, tiktoken, numpy, sentence-transformers, turbovec,
      dill);
    - a platform-only module (`pwd`, `resource`);
    - the CLI's deliberate late loads, which `test_app_layering.py` already argues (the
      TUI, the web server, the daemon);
    - `ph.testing`'s `pytest`.
  - *Gate:*
    - a new AST test holds the exception table, keyed by module, enclosing function and
      imported module, each with its reason;
    - a function-level import not in the table fails, and so does a table entry the code
      no longer has;
    - a fresh-interpreter import of every shipped module, in two orders, proves the moves
      opened no cycle;
    - sabotaged by adding a function-level import, and by moving an exception's import to
      the top where a cycle or an optional dependency makes it wrong.
  - *Landed.*
    - **Moved to the top: 24.** None was dodging a cycle, so no leaf split was needed.
      - Standard library and declared dependencies:
        - `os` (`ph_code_graph._extract`);
        - `importlib.util` (the `ph_code_graph` and `ph_text_index` rows,
          `ph_rlm.kernel.venv`), `importlib.metadata`, and `importlib`
          (`ph_runtime.runner`);
        - `time` (`ph_runtime.limits`);
        - `subprocess` (`ph.seams.subprocess`, as `_sp`);
        - `anyio` (`changes`, the builders);
        - pydantic's `ValidationError` (`json_schema`).
      - In-package:
        - `ph.seams.changes`' `git` and `jj`;
        - `code_runtime_stub`'s `CodeRunFailure`;
        - `skills`' `json_schema` import, and `json_schema`'s `validation_errors`;
        - `ph.testing`'s builders, backends and `FAKE_OPTIONS`;
        - `ph_rlm.harness.service`'s `CodeRunRequest`.
      - `croniter` moved at first and was moved back by the /simplify pass after it.
        It is a declared dependency, but about 20ms to import, and `ph_app.cli` loads
        `ph.seams.schedule`, so every `phern` start paid for it. `schedule`'s own
        docstrings said so; nothing enforced it, and `test_app_layering.py` now lists it.
    - **Stayed inside a function: 30**, each with its reason in a comment above it and
      `# noqa: PLC0415` on it:
      - not declared, and checked for before use: tree-sitter ×4, opentelemetry ×5,
        tiktoken, sentence-transformers;
      - heavy, and needed only where an index is built or searched: numpy ×4, turbovec;
      - loaded on every `phern` start otherwise: `croniter` ×2;
      - turso's native driver, `pwd`, and the CLI's eight late loads;
      - `ph.testing`'s `pytest`, and `ph_runtime.snapshot`'s `dill`.
      - `resource` was already a guarded module-top import.
    - **The gate is ruff's `PLC0415`**, selected in `pyproject.toml` for shipped code
      (a negated per-file ignore leaves the suites out). The first version was a
      hand-built AST walk with an exceptions table keyed by module, function and
      import. The /simplify pass replaced it: the rule finds exactly the same
      imports, the reason sits beside each one, and `RUF100` flags a `noqa` that
      stops suppressing anything, so both directions still hold.
      `test_intent_kinds.py::test_the_resume_path_holds_no_function_level_import` is
      retired with it; the rule covers the resume path along with everything else.
    - **The cycle check changed from the plan.**
      - Two orders were too weak. Importing every module first to last and then last to
        first both passed a real cycle: with `changes` imported at the top of
        `workspace_jj`, another module had always loaded `changes` first.
      - `tests/test_import_cycles.py::test_every_module_imports_first` imports each
        shipped module as the first one a process reaches, dropping every workspace
        module from `sys.modules` before each. It runs in eight parallel processes over
        `workspace_layout.parsed_modules()`'s names, in about 6s (25s serially).
    - **Sabotaged:**
      - an `import json` inside a function in `ph.seams.fs` fails `ruff check`
        (`PLC0415`);
      - `tiktoken` moved to the top of `token_meter` with its `noqa` left on fails it
        too (`RUF100`, unused `PLC0415`);
      - `croniter` at the top of `schedule` fails `test_app_layering.py`;
      - the `workspace_jj` cycle above fails the probe, naming `ph.seams.workspace_jj`
        and `ph.testing.jj`.
