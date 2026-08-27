# pH Implementation Plan

**Status:** v1.0 — 2026-08-26
**Companion to:** [Python_Harness_Port_Plan.md](Python_Harness_Port_Plan.md) (the specification — 21 decisions D1–D21, 13 closed questions Q1–Q13, §10 roadmap) · [Prime_Agent_Feature_Map.md](Prime_Agent_Feature_Map.md) (the governance analysis) · [DeepSeek_to_Prime_Intellect_Integration.md](DeepSeek_to_Prime_Intellect_Integration.md) (the originating thesis, annotated)

---

## 0. What this document is

The port plan is the *specification*: it says what pH is and why each decision fell where it did. This is the *work plan*: it says what to build, in what order, and what proves each piece is done.

It is organised in two directions on purpose:

- **§2 is feature-first.** Every safety and stability property pH promises is a row with an ID, the failure it prevents, the mechanism, the phase, and the test that gates it. This is the checklist a reviewer uses to ask "is the harness actually safe in the way the docs claim?"
- **§4 is phase-first.** Work items small enough to become tickets, each carrying the feature IDs it delivers and its gate. This is what an implementer uses.

The two are joined by ID: a §2 row names the §4 items that build it; a §4 item names the §2 rows it delivers. **A feature with no work item, or a work item with no gate, is a defect in this document.**

Where the port plan wins on *what*, this document wins on *when* and *how proven*. Where they disagree on *what*, the port plan wins.

---

## 1. The invariants everything else serves

Eight statements. Every §2 row protects one of them; every §4 gate tests one of them. Five come from dsh (port plan §2); three were added by this design.

| # | Invariant | Origin | Why it is load-bearing |
|---|---|---|---|
| **I1** | **Everything is a plugin; there is no privileged core to patch.** The loop, the adapter, the registry and the log are rows in a profile. | dsh | Swapping a provider changes the product without forking consumers. Every safety feature below is a row, which is why it can be audited, disabled, or replaced. |
| **I2** | **Registrations *and acquired resources* are effects that unwind.** Every `ctx.on`, `ctx.tools.register`, and every child process, worktree, temp path or lock returns a disposer torn down with its scope. | dsh, extended §4.9 | Cleanup is structural, not remembered. No plugin holds an artifact outside the seam. |
| **I3** | **Model-visible means logged.** Any content reaching a model request is reconstructable from `Session.events` via `derive_messages()`; a runtime invariant asserts it. | dsh | The session log *is* the trace. Anything the model saw can be audited, replayed, and offloaded. |
| **I4** | **The log is append-only; the surface is what changes.** Compaction, pruning and offloading append events whose `surfaceOp` replaces nodes in the derivation; history is never rewritten. | dsh | One mechanism for compaction, offload, rollback and prefix-cache stability, and the reason a checkpointer could not substitute (port plan §8). |
| **I5** | **Seams have three roles** — Definition, Provider, Consumer. | dsh | Containment tiers, runtimes and persistence backends are provider swaps, not redesigns. |
| **I6** | **Every projection equals its fold.** `harness_state.json` equals the `harness/*` fold; `kernel-state.dill` equals the `kernel/snapshot` chain; a runtime invariant asserts both. | D14, D17 | The log is the single source of truth; files are for humans. Fork inherits state *as of the boundary*, which a file cannot express. |
| **I7** | **The knowledge layer may only reference capability that already exists.** `/refine` writes procedure, never capability; an unresolvable `reference` is rejected. | Q13 | Keeps the model from expanding its own governed surface. Forces it to *ask* for a plugin rather than assert one. |
| **I8** | **Containment is not interception, and no document may blur them.** Bindings are governed per call; model-authored raw Python is bounded by tier, never gated. | D20, Q10 | The one place the harness cannot enforce, stated so a deployment does not write a `deny` row and believe it is enforced. |

---

## 2. The safety and stability surface

Every row: **ID · feature · what failure it prevents · mechanism · phase · gate**. Gates are named tests; the §4 item that owns each gate is listed under **Built by**.

### 2.A Log and state integrity *(I3, I4, I6)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| A1 | Append-only `SessionEvent` log | silent history rewriting; unreconstructable sessions | `append()` snapshots data losslessly, assigns `seq = len(log)`, stamps `time`, deep-freezes, validates `surfaceOp`; listener failure cannot un-append | 0 | `seq == len(log)` property; JSON losslessness (BigInt/NaN/-0/cycles rejected); frozen after append |
| A2 | `derive_messages()` is the only path to model context | content reaching the model that is not in the log | per-node projection cache keyed on `replaceGeneration`; the loop asserts `messages == derive_messages()` on every request | 0 | invariant fires on a fake request that bypasses derivation |
| A3 | Surface `replace` for compaction and offload | destroying history to save context | surface events carry `surfaceOp: append \| {op: replace, start, end}`; shadowed nodes leave the derivation, never the log | 0/4 | `replace` op leaves `log` intact and `derive_messages()` shows the summary; originals in `conversation_history/` |
| A4 | Checkpoint-policy flush barriers | losing work between a model request and a crash | `session/flush` (parallel) before each model request, before top-level tool dispatch, at step end | 1 | crash injected after each barrier; resume shows everything before it |
| A5 | Crash repair | corrupt open turns on resume | `interruptedTurnClosers` synthesizes `tool/result {TOOL_NOT_STARTED \| TOOL_OUTCOME_UNKNOWN}` + `step/end` + `turn/end{interrupted}` | 1 | open-turn fixtures resume cleanly; synthetic results match dsh's vocabulary |
| A6 | Fork only at closed-turn boundaries | forking mid-turn inconsistency | `ctx.sessions.fork(source, boundary)` rejects `OPEN_TURN`; `session/end-seed` marks the seed; `header.seedLength` records lineage | 0 | fork inside an open turn is refused; fork at a boundary replays identically |
| A7 | Runtime state lives in the log (D17) | cross-call state invisible to the log — dsh's stated reason for refusing a persistent kernel | `kernel/snapshot {kind: snap \| patch \| clear \| recipe}` per variable, digest-skipped when unchanged, HMAC-SHA256 tag bound to **session id**, blobs > 64 KiB to `ctx.spill_store` | 3 | fork at a boundary restores *that* namespace, not the parent's latest; tampered or cross-session blob fails verification, is not unpickled |
| A8 | `recipe` snapshots for harness-loaded corpora (Q4) | an over-cap variable silently vanishing after restart | `{loader, sources, digest}` recorded instead of dropping; re-resolved on restore with rebuilt/unavailable notices | 3 | over-cap corpus rehydrates on digest match; changed sources produce the rebuilt notice; never an undefined name |
| A9 | Harness state is a fold (D14, Q5) | two-writer races; prompt content not in the log | local: `harness/*` events; global: `$PH_HOME/harness/events.jsonl` under `filelock`; `harness_state.json` written after apply, **never read back** | 3 | delete the file, re-derive, byte-identical; two concurrent sessions do not corrupt the global log; incremental cache does not re-fold from zero |
| A10 | Write-ahead ordering for blob-bearing events (§4.9) | orphaned blobs after a crash between blob write and event append | append the event carrying digest + locator **first**, write the blob **second**; missing blob → recoverable `kernel/restored {failed}` | 3 | `SIGKILL` mid-snapshot leaves no unreferenced blob after the next session open |
| A11 | "Every projection equals its fold" invariant (I6) | projection drift | runtime invariant plugin compares file to fold for harness and kernel state | 6 | invariant fires on a hand-edited projection file |
| A12 | Prefix-cache stability | silent cache-hit regressions billed rather than caught | static sections precede `context()` snapshots; `request/header` changes are explicit events; a recorded session asserts consecutive requests share the predicted prefix | 1 | prefix-stability test on a recorded session |

### 2.B Tool pipeline governance *(I1, I3)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| B1 | The full pipeline on every tool call | any action skipping policy | `tools/pre-execute` → monotonic guards → `ctx.approval` on `ask` → `tools/execute` → body → `tools/post-execute` → normalize → `finalize_content` → `tools/result` → `tool/result` event | 1 | dsh's ordering invariants; a listener that vetoes stops everything downstream |
| B2 | Monotonic guards | a later listener re-permitting a denial | `ctx.tools.guard()` returns a reason or `None`; a denial cannot be turned back | 1 | guard denial survives a later `allow` listener |
| B3 | Approval fails closed | proceeding on an unanswered or unavailable approval | `request()` → `allowed-once \| rejected \| cancelled \| unavailable`; only `allowed-once` proceeds; `approval/asked` then `approval/decided` logged | 1 | `unavailable` and `cancelled` both deny; pending approval is re-asked on resume because `asked` without `decided` is visible in the log |
| B4 | `tool/call` logged **before** execution | an action with no durable record | the event is appended at pipeline entry with losslessly-snapshotted arguments | 1 | a tool body that crashes still leaves its `tool/call` |
| B5 | Lossless result normalization | a throwing tool corrupting the loop | throws become `is_error` results; only `content/error/meta` persist | 1 | a raising tool body yields a structured `is_error` result and the turn continues |
| B6 | Execution-mode barriers | concurrency-unsafe tools overlapping | `execution_mode: parallel \| exclusive`; exclusive calls drain the pool and bar later calls; results commit in model order | 1 | an exclusive call never overlaps; results commit in model order regardless of completion order |
| B7 | Scoped tool visibility | one agent seeing another's tools | `ctx.tools.register` on `agent.ctx` shadows by name; `ctx.tools.restrict(filter)` masks per scope; `schemas(scope)` reports what that scope sees | 1 | a restricted-away global reads as absent to that agent and present to others |

### 2.C Code Mode governance *(C1–C3, D18, Q9)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| C1 | One transport, bindings re-enter the pipeline | one cell = one governance evaluation regardless of what it does | `present_as("code")`; every `await tools.<name>(...)` is a `call` frame dispatched through **B1** as a sub-call with the outer token as `parent` | 3 | a cell calling `tools.edit` fires `fs/write-intent` and produces `tool/code-dispatch-start`/`tool/code-dispatch` |
| C2 | Per-dispatch durable records | forty writes recorded as one stdout blob | `tool/code-dispatch-start {id: <parent>:code:<n>}` at entry; `tool/code-dispatch {content, is_error}` at settle; log-only, offloadable via `tools/code-dispatch-log` | 3 | 3 binding calls → 3 dispatch pairs; `ToolCallLimit` ticks 3 times |
| C3 | Denial fails the run (Q9) | a program catching a denial and routing around it | first `deny` or rejected `ask` settles the run as `CodeRunFailure {kind: "denied"}`; run-scoped abort fires; queue drains inside the open turn. A *failed* call keeps `ToolCallError` | 3 | denied `tools.edit` fails the whole run, file unwritten; a timeout raises `ToolCallError` inside the program |
| C4 | Per-cell dispatch budgets | one approved cell issuing unbounded governed calls | `max_dispatches_per_run = 256`, `max_subagent_spawns_per_run = 32`, enforced in the bridge as `CodeRunFailure` | 3 | a runaway loop fails at the budget with the budget named |
| C5 | Per-binding offload | oversized results melted into stdout | each dispatch result passes `tools/post-execute`; the spill policy replaces it *individually* with preview + locator | 3/4 | an oversized `tools.read` result is spilled while its siblings stay inline |
| C6 | `UNKNOWN_TOOL` for model-direct calls under `code` | policy bypass via a native call the prompt did not offer | resolved at execution creation, before `tools/pre-execute`; denial text names the route back | 1 | a native `edit` call under `mode: code` is refused before any listener runs |
| C7 | Family boundary as a monotonic guard | a child messaging outside parent/siblings/children | `ctx.tools.guard` on `agent_message.send` and on `ctx.subagents.followup` | 3 | send to a non-family target refused; a later listener cannot re-permit it |
| C8 | Message rate limits | a child flooding a sibling | `tools/pre-execute` policy: 16 384 chars, 20 pending/session, token bucket 3 / 1 s per sender→target | 3 | the 21st pending message queues; the 4th in a second is delayed |
| C9 | `access` policy-capped at pre-execute | a child obtaining a writable repo by asking | the `rlm.run` binding's `access` kwarg is subject to a `tools/pre-execute` policy; a deployment can forbid `write` for children without prompt changes | 3 | with the policy row, `access="write"` is denied and the spawn fails the run |
| C10 | The program is a hostile peer | forged frames reaching the host | every inbound fd-3 frame is shape-validated and **rebuilt**; extra fields dropped; non-numeric ids never echoed; junk → `None`, never a raise in the handler | 3 | fuzzed frames: none raise in the host; none forge a reply id |

### 2.D Runtime safety *(D19)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| D1 | Out-of-process CPython | model code crashing or monkeypatching the harness | `code-runtime-python`: a subprocess with fd 3 as the framed channel; nothing shared but the pipe | 3 | `os._exit(1)` in a cell → `kernel/reset`, host unaffected, next run succeeds |
| D2 | No magics | `%%bash` as an unregistered shell tool (feature map item 4) | pH's runtime executes plain Python via `PyCF_ALLOW_TOP_LEVEL_AWAIT`; there is no IPython to supply magics | 3 | `%%bash` in a cell is a `SyntaxError` with the SDK's `tools.bash` guidance |
| D3 | Resource limits | a runaway cell consuming the host | `boot {cpu_seconds, address_space_bytes, max_log_bytes, max_value_bytes}` applied in the child before `boot-ack` (`RLIMIT_CPU`, `RLIMIT_AS`; Job Object limits on Windows) | 3 | a `while True` cell hits `cpu_seconds`; a memory bomb hits `address_space_bytes`; both settle as `CodeRunFailure` |
| D4 | Output caps with a shared marker | unbounded stdout into context | each stream capped at `max_log_bytes` (65 536) with a byte-identical truncation marker on both sides | 3 | marker text equals on host and guest (mirror test) |
| D5 | Cancellation without deadlock | an in-flight run holding the channel | `cancel` frame + `SIGINT`; fd 3 is not the channel the run occupies, so no control-channel workaround | 3 | `cancel` aborts a running cell; the next `run` succeeds |
| D6 | Persistence obligation asserted at registration | a persistent provider that forgets to snapshot | `persistence: "namespace"` requires `kernel/snapshot` emission; checked at `ctx.provide`, not at runtime | 1 | registering a namespace provider with no snapshot hook fails at registration |
| D7 | Protocol drift guard | host and guest disagreeing on a frame shape | versioned vocabulary; `boot-ack` from a mismatched guest refused; cross-implementation e2e asserts constants and each frame's required/optional field set | 3 | mirror test spawns the real child and diffs field sets |
| D8 | Runtime venv is cache | a rebuildable multi-GB venv in backups | lives at `$PH_CACHE/runtime-venv`; no `ipykernel`/`jupyter_client`/`nest_asyncio`; bootstrap marker detects staleness | 3 | deleting `$PH_CACHE` costs a rebuild and nothing else |

### 2.E Containment *(D20, D21, Q9–Q11, I8)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| E1 | Three-tier ladder, honestly described | a tier *name* overstating what it does | `advisory` → `worktree` → `sandbox`; §4.8's table states per tier what is bounded, what is **not**, and which property is bought (isolation+revertibility vs confinement) | 4 | docs test: no tier is described as bounding writes it does not bound |
| E2 | Per-agent worktree (D21) | fan-out children trampling one tree | `workspace-git-worktree`: `git worktree add` on `ph/<session>/<agent>`; `ctx.fs` root **and** `ctx.subprocess` cwd resolve to `workspace.root` | 4 | two children on separate branches; parent merges both diffs without collision |
| E3 | `access: write \| read` with honest kinds | a caller assuming "read-only" a tier cannot enforce | `worktree` + `read` → `worktree-ephemeral` (writable, discarded, never merged; `repo_writable: True`); `sandbox` + `read` → `readonly-scratch` (`repo_writable: False`) | 4/6 | an ephemeral child's dirty worktree is discarded; the parent's branch is untouched; `repo_writable` reports the truth per tier |
| E4 | `access` defaults to `read` for children (Q11) | a writing child that should not have written | the recoverable default; the RLM prompt states it; one default across all front-ends | 3 | omitted `access` yields a research-shaped child |
| E5 | `scratch` always writable | read-only that is safe and useless | `<agent artifacts>/scratch/`, on every kind and tier, survives disposal as an artifact | 4 | a `readonly-scratch` child writes notes to `scratch` and nowhere else |
| E6 | Default write scope = worktree + scratch (Q9) | an approval queue interrupting every cell | `SandboxExecutionPolicy {mode: workspace-write, workspace_root: <worktree>}` + a `permissions-fs` row; prompt only for writes outside | 4 | a cell writing inside its worktree prompts zero times; one write outside prompts once |
| E7 | Per-run checkpoint and `/revert` (Q9) | an unrecoverable partial state after a denied run | `git add -A && git write-tree` under `refs/ph/<session>/<agent>/pre-run/<seq>` before a mutating run; `workspace/checkpoint` event; `/revert <seq>` restores tracked + untracked-not-ignored, never `.gitignore`d paths or `scratch` | 4 | a denied run reverts exactly; pre-denial writes gone; ignored paths untouched |
| E8 | `containment.strict` (Q10) | a host without a sandbox silently running an `advisory` agent that believes it is confined | refuses to start unless tier is `sandbox` **and** `enforcement: full`; `partial` is a refusal, not a downgrade — dsh's `SANDBOX_UNAVAILABLE` posture at profile start | 4 | strict on a host with no backend refuses, naming the missing one |
| E9 | Permission rows self-report reach (Q10) | a `deny` row believed to cover raw Python | row validation and `ph doctor` both say "applies to tool calls; raw `open()`/`subprocess` is not covered" when no confining provider is mounted | 4 | the message appears without a sandbox and disappears with one |
| E10 | `ph doctor` reports the *effective* tier | configured `worktree` silently degrading to `shared` on a non-repo cwd | provider declines, loader falls back with a logged notice, doctor prints effective tier + per-agent `kind` + `repo_writable` | 4 | non-repo cwd → doctor says `advisory`, not `worktree` |
| E11 | `readonly-scratch` enforced by the sandbox (Q6) | a research child writing the repo | `ctx.sandbox.confine()` with `workspace_root: <scratch>` — repo readable, only scratch writable, at the kernel | 6 | an absolute-path `open("/repo/x", "w")` is refused; scratch write succeeds |
| E12 | Build-tool redirection env | `pytest` failing under a read-only repo | `TMPDIR`, `PYTHONPYCACHEPREFIX`, `PYTEST_ADDOPTS=-p no:cacheprovider --basetemp=<scratch>/pytest`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`, `GIT_CONFIG_GLOBAL` → scratch; documented best-effort | 4 | `pytest` runs green inside a `readonly-scratch` child, writing nothing outside scratch |
| E13 | The `sandbox` tier is the only confinement (Q10) | `worktree` mistaken for a boundary | an absolute-path raw write escapes `worktree` and is refused under `sandbox`, asserted so the tier table cannot regress | 4/6 | the paired assertion runs in CI |

### 2.F Resource lifecycle *(§4.9, I2)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| F1 | All artifacts acquired via `ctx.effect()` | a plugin holding a handle outside the seam | child processes, worktrees, temp paths, locks all registered as effects over `AsyncExitStack`; LIFO, children first | 1 | agent-scope disposal releases every artifact the agent took; a plugin holding one outside the seam fails a lint |
| F2 | Graceful shutdown | `ph` quitting with live children | `atexit`; `SIGTERM`/`SIGINT` → orderly `dispose()` with a 10 s grace then self-`SIGKILL` | 1 | `SIGTERM` unwinds the scope within the grace period |
| F3 | Die-with-parent, per platform | orphaned runtime children after a host `SIGKILL` | Linux `PR_SET_PDEATHSIG`; macOS `os.getppid()` poll in the guest; Windows Job Object `KILL_ON_JOB_CLOSE` | 3 | `SIGKILL` the host on each platform; no surviving child |
| F4 | Zombie reaping | exited children never reaped by a live parent | `await proc.wait()` in a `finally` on every spawn path | 3 | a normally-exited run leaves no zombie |
| F5 | Process-level orphan journal | strays from a session nobody reopens | `$PH_RUNTIME/processes.jsonl`, `fsync` on append (pid, start time, argv digest, session id); swept at **every** pH start | 3 | a journalled stray from a prior hard kill is killed at next start; a reused pid with a different start time is not |
| F6 | Paired events reconciled at session open | a leaked worktree | `workspace/acquired` without `disposed` is detected on open | 4 | a crash between acquire and dispose is reconciled on the next open |
| F7 | Blob GC at session open | unreferenced spill blobs accumulating | sweep `ctx.spill_store` for blobs no event references | 3 | orphaned blob removed on next open |
| F8 | Ephemeral scratch is `TemporaryDirectory` in an effect | GC-timed cleanup | `mkdtemp` (0700, unguessable) wrapped in `ctx.effect()`; known to be layer 1 (its cleanup is a `weakref.finalize`) | 1 | disposal removes it before GC would |
| F9 | `$PH_RUNTIME` is **not** scratch | randomising the socket and journal path | well-known location, contents outlive the writer; tier-3 `/tmp/ph-$UID` checked for dir/uid/0700/not-symlink | 0 | wrong-owner `/tmp/ph-$UID` refuses to start |

### 2.G Context stability *(Deep Agents features on dsh waterfalls — D12)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| G1 | Todo planning | goal drift in a long recursive loop | `tool-todo` + `todo/write` event + prompt section; opt-in row | 4 | `write_todos` parallel-call error; reminder text matches |
| G2 | Large-result offload | context exhaustion by one tool result | `tools/post-execute` at ≥ 20 000 tokens (× 4 chars) → spill + preview (head 5 / tail 5 / 1 000-char clip); excluded tools untouched | 4 | replaces at 80 001 chars, not 80 000 |
| G3 | Input offload | a pasted blob exhausting context | ≥ 50 000 tokens on a human message → spill + preview | 4 | threshold test |
| G4 | Threshold summarization | context overflow mid-task | `compaction-summarize` on `agent/pre-step` (pressure) and `agent/request-error` (overflow): trigger `0.85` fraction / keep `0.10`; window unknown → 170 000 tokens / keep 6 messages; cutoff never splits a call/result pair; summary is a surface `replace` | 4 | triggers at 0.85 with a known window; never splits a pair; log intact |
| G5 | Model / tool call limits | infinite loops | `ModelCallLimit` / `ToolCallLimit` as `agent/pre-step` reject and `tools/pre-execute` deny; end / continue / error behaviours; breaker on repeated identical failures | 4 | each behaviour tested; the breaker trips on N identical failures |
| G6 | Human-in-the-loop | destructive actions without a human | `tools/pre-execute` `ask` → `ctx.approval`; decisions `approve \| edit \| reject \| respond`; modes `MANUAL \| AUTO \| YOLO`; destructive classifier for `run_code` and mutating bindings | 4 | edit and respond flows via pilot tests; a destructive cell prompts in `MANUAL` |
| G7 | Filesystem permissions | writes outside an allowed set | `permissions-fs`: `{operations, paths, mode: allow \| deny \| interrupt}` first-match-wins; recursive `delete` fails closed | 4 | first-match tests; recursive delete with a possibly-matching descendant refused |
| G8 | Memory | re-learning across sessions | `memory-agents-md` from `$PH_HOME/AGENTS.md` + `<project>/AGENTS.md`; placed **after** caching so edits do not bust the static prefix | 4 | prefix test unchanged by a memory edit |
| G9 | Progressive skills | prompt bloat from every skill's body | `skills-progressive`: catalog in prompt, body on demand; limits name 1–64, description ≤ 1 024, file ≤ 10 MiB | 4 | limits enforced; body absent until requested |
| G10 | Compaction is RLM-aware | REPL variables lost across compaction | the summary prompt lists live runtime variables from the snapshot manifest; runtime state itself is untouched | 4 | variables survive compaction; summary names them |

### 2.H Knowledge-layer safety *(Q13, I7)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| H1 | Reference resolved before apply | `/refine` teaching the model to call something that does not exist | `reference.import`/`callable` resolved via `ctx.code_runtime` in a silent cell; unresolvable → edit rejected, failure on the `harness/refined` event | 3 | an entry naming a missing import is rejected with the failure recorded |
| H2 | Call patterns rendered as bindings | `/refine` steering the model onto the raw-namespace path | `call_pattern` rendered as `await tools.<name>(...)` wherever a binding exists | 3 | a skill entry for `edit` renders the binding form |
| H3 | `scope: global` approval-gated | the model editing every future session including other projects | a global edit is a `tools/pre-execute` `ask` on the `refine` tool | 3 | a global edit prompts; a local one does not |
| H4 | No promotion command | refinement output treated as proto-tools | `ph tool scaffold` explicitly rejected; `ph harness report` is analytics only | 6 | (design gate — the command does not exist) |
| H5 | `base_system_prompt` immutable | `/refine` rewriting the doctrine | validator rejects the id | 3 | rejected with prime-agent's error text |
| H6 | Rollback by event id | an unrecoverable bad refinement | `/refine --rollback <id>` builds the inverse from the event's before/after snapshots | 3 | apply + rollback returns the fold to its prior state |
| H7 | Auto-refine gated and cooled | runaway self-modification | every 25 turns or after compaction; cheap LLM review gate; 20-minute cooldown; `session_before_refine` waterfall veto | 3 | cooldown respected; veto honoured |

### 2.I Process, path, wire and credential security *(Q1, Q2, Q7)*

| ID | Feature | Prevents | Mechanism | Phase | Gate |
|---|---|---|---|---|---|
| I-1 | `$PH_RUNTIME` resolution order | a socket or PID journal in a synced dotdir | `$XDG_RUNTIME_DIR/ph` → per-user `$TMPDIR/ph` → `/tmp/ph-$UID` (checked) | 0 | `ph doctor` prints the resolved tier; the tier-3 check refuses a bad dir |
| I-2 | One daemon per user (Q7) | one root OOMing another user's agents | documented line; isolation between users is the operator's layer | 5 | doctor reports the worker model; docs state the non-guarantees |
| I-3 | Credentials never as values | a secret reaching a child, a log or a container | `ctx.credentials` passes `CredentialRef`; adapters resolve at the edge | 1 | no credential value appears in any event, frame or child env |
| I-4 | Subprocess env scrubbed | secrets inherited by model-run code | parent env scrubbed of `*KEY*`/`*SECRET*`/`*TOKEN*`/`*PASSWORD*` before spawn | 1 | a planted `FOO_API_KEY` is absent in the child |
| I-5 | Session leases | two writers on one JSONL | in-process lock per root; `filelock` on the canonical path against a *second daemon* | 5 | concurrent open → `session_already_active` |
| I-6 | Lingering detection | a daemon that outlives logout but loses its socket | `ph doctor` reports `$XDG_RUNTIME_DIR` derivation and `loginctl` linger state, naming `enable-linger` when a daemon is configured without it | 5 | simulated session end → clear "socket path removed" diagnostic |
| I-7 | Wire casing: declare, never derive (Q2) | field-name drift between Python and JSON | `WireModel` base with `alias_generator=to_camel` + `populate_by_name=True`; `SessionEvent.to_wire()` pinned to `to_camel(field)` by test | 0 | round-trip property; no field reaches the wire un-aliased |
| I-8 | Config is data, not code (D9) | `eval` on a user-controlled config | YAML rows with `${env:VAR:-default}` only; no `!!js`/`!!py` | 0 | a tag that would evaluate is rejected |

---

## 3. Non-goals — stated so nobody assumes otherwise

Each is documented in the plan, the first-run notice, and where relevant in `ph doctor`. **A reviewer finding any of these implied as covered should file it as a defect.**

| # | pH does **not** | Because | Bounded instead by |
|---|---|---|---|
| N1 | intercept model-authored raw Python (`open()`, `subprocess.run()`) | a deny-list needs a registered name; the model is authoring its own tool (§4.8, §6a) | the containment ladder — `worktree` for location, `sandbox` for confinement |
| N2 | treat `worktree` as confinement | an absolute-path write never consults cwd | only `sandbox` refuses it; the tier table says so |
| N3 | undo the world with `/revert` | git restores the tree, not a published package or a dropped table | `/revert` lists the run's non-filesystem dispatches so irreversible actions are visible |
| N4 | replay raw writes | they are bounded by the worktree but not recorded | the checkpoint is the recovery mechanism; replay is explanation |
| N5 | cap per-root memory or contain crashes between roots in the daemon | one `anyio` task per root (Q7) | one daemon per user; process-per-root is a later provider swap |
| N6 | manage containers | environment decision, operator's layer (Q12) | a "running pH in a container" page; fd 3 crosses the boundary as an inherited descriptor |
| N7 | run any cleanup on `SIGKILL` | no finalizer does, on any platform | the crash-recovery layer: paired events + the orphan journal |
| N8 | provide provenance for raw reads | a raw `open()` has no `tool/call` | bindings, which the SDK advertises as the path of least resistance |
| N9 | reconstruct Code Mode intermediate values from the log | dsh's own stated limitation | `kernel/snapshot` + `recipe` keep *harness-created* bulk state reconstructable |
| N10 | enforce a `deny` row without a confining provider | see N1 | rows self-report their reach; `containment.strict` refuses to start until confinement is real |

---

## 4. Work breakdown

IDs are `P<phase>-<nn>`. **Delivers** names §2 rows; **Gate** is the test that closes the item. **Depends** lists blocking items only. Items within a phase are in dependency order; unlisted pairs are parallelisable.

### Phase 0 — Spike the core *(1–2 weeks)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P0-01 | Repo scaffold: `uv` workspace with `ph-core`, `ph-app`, `ph-rlm`, `ph-stabilize`; `ruff` (line 100), `mypy --strict` on `ph-core`, `pytest` + `anyio`; CI matrix Linux / macOS / Windows; `[project.entry-points."ph.plugins"]` group reserved | — | I1 | CI green on an empty package set on all three OSes |
| P0-02 | `ph.cordis` Context: `plugin()`, `inject()`, `provide()`, `__getattr__` most-specific-wins, `effect()` returning a disposer, `scope()`, `dispose()` LIFO children-first | P0-01 | I1, I2 | disposal unwinds every effect in reverse; a disposed scope's services are gone |
| P0-03 | `ph.cordis` dispatch: `emit`, `waterfall` (outermost-first, `next`, veto-by-return), `parallel` (`allSettled` + `AggregateError`), `serial` (bail on non-null), `bail`; `prepend`, `global`, scope filtering | P0-02 | I1 | waterfall veto stops built-in behaviour; `next()` ordering; parallel aggregates rejections |
| P0-04 | Event registry: `events.declare(name, mode, payload)`; `ctx.<mode>` raises on mismatch; producer/consumer matrix export | P0-03 | I1 | dispatching an event under the wrong mode raises |
| P0-05 | Loader: YAML rows in file order, id-addressed patches (whole-config replace, insert), `${env:VAR:-default}`, `disabled:` predicates, entry-point discovery, inject-driven activation/deactivation, **no code evaluation** | P0-04 | I1, I-8 | `--dump-config` shows composed rows; a `!!js`-style tag is rejected; a plugin activates only when its `inject` keys are provided |
| P0-06 | `WireModel` base (`alias_generator=to_camel`, `populate_by_name=True`); round-trip property test; "no un-aliased field" assertion | P0-01 | I-7 | property passes over every model; a model without the base fails the assertion |
| P0-07 | `SessionEvent` frozen `dataclass(slots=True)`: `type, seq, time, data, ignorable?, sourceEventSeqs?, surfaceOp?`; `to_wire()`/`from_wire()`; test pinning each mapping to `to_camel(field)` | P0-06 | A1, I-7 | pin test; wire form equals dsh's envelope byte-for-byte on a fixture |
| P0-08 | `ph.session.append()`: lossless snapshot (`snapshotJsonValue` rejecting BigInt/undefined/-0/NaN/Map/Date/instances/cycles), `seq = len(log)`, `time`, deep-freeze, `SurfaceManager.validateNext`, push, publish `session/event` with listener-failure isolation | P0-07 | A1 | `seq == len(log)` property; losslessness rejections; a raising listener does not un-append |
| P0-09 | `SurfaceManager`: surface types require `surfaceOp`; `replace {start, end}` shadows nodes; `ignorable` skipped by unknown readers; unknown non-ignorable type refuses the log | P0-08 | A3 | replace leaves the log intact; unknown type refuses load |
| P0-10 | `derive_messages()` with per-node cache keyed on `replaceGeneration`; `request_header()` / `request_context()` incremental folds | P0-09 | A2 | cache invalidates only on generation change; folds are incremental |
| P0-11 | Seed / fork / resume: `session/end-seed`, `header.seedLength`, `fork(source, boundary)` rejecting `OPEN_TURN` | P0-10 | A6 | fork inside an open turn refused; fork at boundary replays identically |
| P0-12 | `ph.llm`: `Message`/`ContentBlock`/`StreamChunk` types, `BlockAssembler`, fake adapter emitting text/thinking/tool-call start/delta/end | P0-07 | — | assembler reconstructs a recorded stream |
| P0-13 | Minimal `ReactLoopAgent`: `turn/start`, inbox claim (`agent/inbox/claimed`), `system-prompt/assemble`, `agent/pre-step` (`reject \| enter`), `step/start`, `user/message`, `agent/request` → `llm/stream` → `assistant/chunk*` → `assistant/message`, `step/end`, `agent/turn-stopping`, `turn/end`; no tools | P0-10, P0-12 | A2 | the lifecycle events appear in order on a fake run |
| P0-14 | **Runtime invariant: `messages == derive_messages()`** on every request | P0-13 | A2, I3 | fires on a deliberately bypassed request |
| P0-15 | JSONL persistence provider + `session/flush` (parallel) + buffered async writer keeping `append` sync and I/O-free | P0-08 | A1 | flush drains; `append` never awaits I/O |
| P0-16 | Path roots: `$PH_HOME` / `$PH_CACHE` / `$PH_RUNTIME` resolution (XDG → per-user `$TMPDIR` → `/tmp/ph-$UID` with dir/uid/0700/not-symlink check); Windows mapping; `ph doctor` prints all three | P0-01 | I-1, F9 | wrong-owner tier-3 dir refuses to start; doctor output on each OS |
| P0-17 | `ph -p "…"` print mode; `--dump-config`; `dev-notes/phase-0.md` | P0-13, P0-15, P0-16 | — | a one-shot Q&A against the fake adapter writes an inspectable JSONL |

### Phase 1 — Core parity *(3–4 weeks)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P1-01 | `ph.tools`: `ToolDefinition` (mandatory `output` + `render`), `define_tool`, `register` global/scoped with shadowing, `restrict(filter)` with allow/deny intersection, `schemas(scope)`, `tools/change` | P0-05 | B7 | restricted-away global absent to that scope, present to others |
| P1-02 | Pipeline: `tools/pre-execute` → guards → `ask` → `tools/execute` (around, signal-only replacement) → body → `tools/post-execute` (`accept \| block`) → normalize → `finalize_content` → `tools/result` → `tool/result`; `tool/call` appended at entry | P1-01 | B1, B2, B4, B5 | dsh ordering invariants; guard denial survives later listeners; crashing body leaves `tool/call` and yields `is_error` |
| P1-03 | Execution modes: `is_concurrency_safe` classification, bounded rolling pool, exclusive barriers, model-order commit | P1-02 | B6 | exclusive never overlaps; commit order fixed |
| P1-04 | Presentation: `tools.mode: native \| code \| both`, `ctx.tools.present_as()` per agent, reserved `run_code`, `tools:sdk` section renderer (Python + TypeScript), `tools:code-only` rule, `UNKNOWN_TOOL` before policy | P1-02 | C6 | native `edit` under `code` refused before any listener; SDK block renders per language |
| P1-05 | Dispatch bridge (Code Mode `run_code` body): lossless-JSON sub-call snapshot, per-run pool honouring B6, `parent` token, full pipeline re-entry, `tool/code-dispatch-start`/`tool/code-dispatch`, `tools/code-dispatch-log` waterfall, run-scoped abort, drain-before-return; **denied sub-call settles the run as `CodeRunFailure {kind: "denied"}`**, failed sub-call raises `ToolCallError` | P1-03, P1-04 | C1, C2, C3, C5 | (tested end-to-end in P3-21; here: against `code-runtime-worker-thread` semantics with a stub runtime) |
| P1-06 | `ctx.code_runtime` seam **definition only**: `CodeRuntime` Protocol with `language`, `isolation`, `persistence`; `CodeRunRequest {program, bindings, namespace, signal}`; portable name rules; **registration assertion that `persistence: "namespace"` emits `kernel/snapshot`** | P0-02 | D6 | registering a namespace provider without the hook fails at `provide()` |
| P1-07 | System prompt: `section` / `context()` / `tools` / `variable`, ordering, `request/header` epoch fold and change detection | P0-13 | A12 | header appended only on change; static sections precede `context()` |
| P1-08 | `ctx.approval`: `request()` → `allowed-once \| rejected \| cancelled \| unavailable`; `approval/asked`/`decided`; `approval/request` waterfall to answerers; `ApprovalPolicy` from last `approval/policy`; fail-closed; re-ask on resume | P1-02 | B3 | `unavailable`/`cancelled` deny; asked-without-decided re-asks on resume |
| P1-09 | `ctx.user_questions`; `ctx.commands` (human slash commands, no model turn) | P1-08 | — | a command dispatches without a `turn/*` |
| P1-10 | `agent/request-error` waterfall + retry plugin (backoff, canonical context-overflow detection) | P0-13 | — | overflow classified; retry bounded |
| P1-11 | `checkpoint-policy`: flush before each model request, before top-level tool dispatch, at step end | P0-15, P1-02 | A4 | crash after each barrier; resume shows everything before it |
| P1-12 | Crash repair: `interruptedTurnClosers` on load | P0-11 | A5 | open-turn fixtures; synthetic vocabulary matches dsh |
| P1-13 | `ctx.token_meter`: provider usage authoritative, `tiktoken`/`len/4` estimation for pressure, `measure()` per node | P0-13 | — | baseline switches from estimate to usage after first response |
| P1-14 | `ctx.session_telemetry`: `SessionTelemetryRecord`, `session-telemetry/record` redaction waterfall, JSONL sink; first `assistant/chunk` only per step | P0-08 | — | redaction listener runs before any sink |
| P1-15 | `llm-tau-ai` adapter: OpenAI-compatible (DeepSeek incl. `reasoning_content`), Anthropic; `ctx.llm.register_adapter` | P0-12 | — | real-API smoke (skipped without key) |
| P1-16 | `ctx.credentials`: `CredentialRef` only; resolution at the adapter edge | P1-15 | I-3 | no value in any event, frame or child env on a planted-secret run |
| P1-17 | `ctx.fs` seam + `fs-local`: read/write/edit/glob/grep; `fs/write-intent`, `fs/edit-intent` waterfalls; `fs/observed`; read-before-edit policy row | P1-02 | — | write-intent fires before the write; veto prevents it |
| P1-18 | `ctx.subprocess`: explicit `SubprocessSpawnSpec {argv, cwd, stdio, grace_ms, env?}`, parent-env scrub, offset-based readers with spill, `await proc.wait()` in `finally`, spawned via `ctx.effect()` | P0-02 | F1, F4, I-4 | planted `FOO_API_KEY` absent in child; disposal terminates and reaps |
| P1-19 | `ctx.shell` (bash over subprocess); `ctx.sandbox` seam + `sandbox_policy` (policy-only provider; `confine()` throws `SANDBOX_UNAVAILABLE` rather than pass-through); `permission_presets` | P1-18 | — | requesting confinement with no backend throws, never runs unconfined |
| P1-20 | **§4.9 resource ownership**: `AsyncExitStack` behind `ctx.effect()`; `atexit`; `SIGTERM`/`SIGINT` → orderly dispose with 10 s grace then self-`SIGKILL`; `TemporaryDirectory` helper wrapped in an effect; lint forbidding `Popen`/`mkdtemp` outside the seam | P0-02, P1-18 | F1, F2, F8 | `SIGTERM` unwinds within grace; lint catches a raw `Popen` |
| P1-21 | `ctx.spill_store`: `save_text()` → `SpillRef {locator, bytes, retrieval_hint}` | P0-15 | — | round-trip |
| P1-22 | Tools: `bash`, `read`, `write`, `edit`, `glob`, `grep` with `present_call`/`present_result` cards; `agent-instructions` (AGENTS.md discovery) | P1-17, P1-19 | — | each tool's card renders from durable result alone |
| P1-23 | `ctx.jobs`, `ctx.settings`, `ctx.skills` seam (capability-layer packages) | P0-02 | — | job cancel/done hooks |
| P1-24 | `llm-replay` adapter; **prefix-stability test** over a recorded session | P1-15 | A12 | replay reproduces `derive_messages()`; prefix assertion passes |
| P1-25 | `json` / `transcript` / `rpc` modes (camelCase output), `headless` profile; `dev-notes/phase-1.md` | P1-22 | I-7 | RPC round-trip with the dsh Python SDK client shape |

### Phase 2 — TUI *(3 weeks)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P2-01 | `PHTuiApp` + adapter boundary + state; streaming transcript on `session/event` + `agent/*` | P1-25 | — | resume rebuilds transcript from `session.events` |
| P2-02 | `PromptInput`, sidebar, autocomplete, slash commands via `ctx.commands` | P2-01 | — | pilot keypress tests |
| P2-03 | Pickers: model / session / tree / theme / login | P2-01 | — | pilot tests through each |
| P2-04 | **Approval modal** (`allowed-once \| rejected` + reason), **ask-user modal**, permission-preset switcher, plan-mode review | P1-08, P2-01 | B3, G6 | approval round-trips through `approval/asked`/`decided`; `push_screen(callback)` never awaited in a handler |
| P2-05 | Themes/keybindings from `$PH_HOME/tui.json`; terminal title; notifications; project trust | P2-01 | — | snapshot tests |
| P2-06 | Textual discipline: `Content.from_markup("$var")`, never f-string markup, `notify(markup=False)`, off-pump continuations | P2-01 | — | bracketed user text renders literally |
| P2-07 | `tui` profile; `dev-notes/phase-2.md` | P2-05 | — | snapshot tests for streaming / tool card / error / compaction marker |

### Phase 3 — RLM bundle *(4–5 weeks; parallel with Phase 4)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P3-01 | fd-3 protocol vocabulary, both sides: `boot`/`boot-ack`/`run`/`call`/`reply`/`log`/`display`/`snapshot`/`restore`/`cancel`/`done`/`shutdown` as `TypedDict`s (guest) and `WireModel`s (host), camelCase; **version field**; `PROTOCOL_FD = 3`; shared truncation-marker function | P1-06 | D4, D7 | mirror test: constants and each frame's required/optional field set equal across sides |
| P3-02 | Host frame codec: `validateChildFrame` shape-validate and **rebuild**; drop extra fields; never echo a non-numeric id; junk → `None`; `hasUnsafeIntegerToken` on raw text | P3-01 | C10 | fuzzing: no raise, no forged reply |
| P3-03 | Guest runner (`ph_runtime.runner`): read `boot`, apply `RLIMIT_CPU`/`RLIMIT_AS` (Job Object on Windows), `boot-ack`; loop on `run`: compile with `PyCF_ALLOW_TOP_LEVEL_AWAIT`, exec as async body against persistent `globals()`, stream `log` with caps, `done {value?, error?}`; `display`; handle `cancel` (`SIGINT` → `KeyboardInterrupt` in the cell); `shutdown` | P3-01 | D2, D3, D4, D5 | top-level `await`/`return`; variable persists across runs; `%%bash` → `SyntaxError`; cpu/memory bombs → `CodeRunFailure`; `cancel` then next `run` succeeds |
| P3-04 | Guest die-with-parent: `PR_SET_PDEATHSIG` (Linux), `getppid()` poll (macOS), Job Object `KILL_ON_JOB_CLOSE` (Windows) | P3-03 | F3 | host `SIGKILL` on each OS → no surviving child |
| P3-05 | Host provider `code-runtime-python`: `language="python"`, `isolation="process"`, `persistence="namespace"`; spawn with `stdio=[pipe]*4` **through `ctx.effect()`**; `boot` with caps + namespaces + namespace id; `run` dispatch; `call` → **P1-05 bridge** → `reply`; `cancel` on abort; `shutdown` + 5 s kill; `await wait()` in `finally`; `<runtime_reset>` prefix after a kill | P3-02, P3-03, P1-05, P1-20 | D1, C1, F1, F4 | `os._exit(1)` in a cell → reset notice, host unaffected; disposal reaps |
| P3-06 | Orphan journal: `$PH_RUNTIME/processes.jsonl` `fsync`ed append (pid, start time, argv digest, session); startup sweep with start-time check | P3-05, P0-16 | F5 | journalled stray killed at next start; reused pid with other start time spared |
| P3-07 | Runtime venv builder at `$PH_CACHE/runtime-venv`: `uv venv --seed`, `ph-runtime-guest`, `dill`, `--editable <skill>` per Python skill; `PH_RUNTIME_PYTHON` override; staleness marker | P3-03 | D8 | deleting `$PH_CACHE` costs only a rebuild |
| P3-08 | `ph_runtime` guest package: `tools.*`, `rlm(...)`, `rlm.find_models/list_subagents/delete_subagent/replies`, `agent_message.*`, `agent_observe.*` proxies each marshalling one `call`; `wrap_skill_module`; bootstrap (`NO_COLOR`, namespaces into globals, skill import with "unavailable" stubs) | P3-05 | C1 | each proxy produces exactly one `call` frame with the declared shape |
| P3-09 | `rlm-presentation`: `present_as("code")` on the RLM preset; `ipython` alias for `run_code`; result text `stdout\nstderr\nresult\ntraceback`; `IpythonToolDetails`; streaming via `on_update` | P1-04, P3-05 | C1, C6 | model sees one callable; SDK block lists the four namespaces |
| P3-10 | `rlm-bindings`: the `tools` / `rlm` / `agent_message` / `agent_observe` namespaces as `CodeBindingNamespace`s; **budgets** `max_dispatches_per_run=256`, `max_subagent_spawns_per_run=32` | P3-08, P3-09 | C4, C9 | runaway loop fails at the budget; `access="write"` denied under a policy row |
| P3-11 | `rlm-subagent-provider` (`ctx.subagents` provider `rlm-child`): kwarg validation, depth gate (`RLM_MAX_DEPTH=2`), model preflight with no fallback, `rlm/child-admitted` **logged first**, handle returned immediately, detached `anyio` task, `[task from parent]`, `rlm/child-status`, usage attribution `rlm/child-usage-attributed`, terminal notices, `rlm/child-deleted` tombstone; `access` kwarg (default `read`) passed to `ctx.workspace.acquire` | P3-10 | E4 | spawn returns before child completes; depth error text; 8-child fan-out reconciles usage; omitted `access` → read |
| P3-12 | `rlm-messaging`: family roster + **monotonic guard** on `agent_message.send` and `followup`; rate-limit pre-execute policy; steer/queue delivery with `deliveryStatus`; received-message rendering verbatim | P3-10 | C7, C8 | non-family send refused and not re-permittable; 21st pending queues |
| P3-13 | `rlm-registry`: roster folded from `rlm/child-*` events; tombstones; passivation/rehydration (in-process variant) | P3-11 | — | roster survives restart and compaction by construction |
| P3-14 | `rlm-prompt`: doctrine (ported, minus the raw-call-form line), child doctrine, **workspace section** (`Workspace: <root> (read-only, enforced \| isolated \| writable)`, `Writable scratch:`, `Branch:`), `context()` snapshot; documents `rlm(..., access=)` and the `read` default | P3-09 | E4 | prompt reflects the kind the child actually got |
| P3-15 | `rlm-kernel-snapshot` (D17): per-variable `dill` + digest, 16 MiB/256 MiB caps, 1.5 s debounce; `snap`/`patch` (`bsdiff4`)/`clear`; **`recipe`** for harness-loaded over-cap variables; HMAC bound to session id; blobs > 64 KiB to spill; **write-ahead ordering**; re-anchor at `max_chain=32`; `restore` with `{restored, rebuilt, unavailable, failed}`; blob GC at session open; `bytes appended per cell` benchmark | P3-05, P1-21 | A7, A8, A10, F7 | fork at boundary restores that namespace; tampered blob refused; `SIGKILL` mid-snapshot leaves no orphan blob; recipe rehydrates with correct notice; benchmark decides patch vs snap-only |
| P3-16 | `rlm-harness`: **fold** over `harness/*` with incremental cache per `(scope, last_seq)`; global `$PH_HOME/harness/events.jsonl` under `filelock`; `harness_state.json` written after apply, never read; `/refine` as `ctx.jobs` background job at turn end; planner prompt; **validation + H1 (resolve via `ctx.code_runtime`), H2 (render `call_pattern` as binding), H3 (`global` → `ask`), H5**; apply with before/after snapshots; `harness/refined`; `/refine --rollback <id>`; auto-refine (25 turns / post-compaction, review gate, 20-min cooldown, `session_before_refine` veto) | P3-05, P1-08, P1-23 | A9, H1, H2, H3, H5, H6, H7 | delete-and-re-derive byte-identical; concurrent global writes safe; missing import rejected with failure on event; `edit` entry renders binding form; global prompts; rollback restores fold |
| P3-17 | `rlm-context-loader` (off by default): sources → corpus; `tools.context_search/chunks/head` **bindings**; corpus registered as a `recipe` variable; metadata-only prompt section; `rlm-stable` threshold ≥ 200 000 chars | P3-10, P3-15 | A8, C5 | queries produce dispatch records and are individually offloadable; corpus rehydrates after restart |
| P3-18 | `rlm-skills-python`: `SKILL.md` + package discovery, editable install into the runtime venv, catalog section | P3-07 | — | a skill's `run()` is callable in a cell |
| P3-19 | `CodeCellWidget` (program + streams + result, one row per dispatch) + subagent panel | P2-01, P3-09 | — | renders from durable events alone |
| P3-20 | `rlm` profile bundle; `dev-notes/phase-3.md` | P3-09 … P3-18 | — | profile boots; smoke run |
| P3-21 | **Governance gate (a)–(e)** as a named test module | P3-10, P3-12 | C1–C4, C7 | all five pass — if not, the fold did not land |
| P3-22 | **Runtime conformance suite**: one test per frame type, one per binding namespace; fuzzed frames; persistence across runs; `restore` after kill; `recipe` rehydration; `cancel` recovery; `%%bash` `SyntaxError`; lifecycle (no zombie; `SIGKILL` → no child on 3 OSes; journal cleans strays; mid-snapshot kill → no orphan blob) | P3-05, P3-06, P3-15 | D1–D5, F3–F5, F7 | full suite green on Linux, macOS, Windows |
| P3-23 | Trajectory-fixture replay: prime-agent fixtures under the new surface; diff turn counts and tool-call shapes; expected diffs = `access` default + SDK block | P3-20 | — | report checked in; unexpected diffs triaged |

### Phase 4 — Stabilization bundle *(4–5 weeks; parallel with Phase 3)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P4-01 | `tool-todo` + `todo/write` + prompt section (opt-in row) | P1-22 | G1 | parallel-call error; reminder text |
| P4-02 | `tool-result-offload` + `input-offload`: thresholds, previews, excluded tools, `conversation_history/` | P1-21 | G2, G3, C5 | 80 001 vs 80 000; excluded untouched |
| P4-03 | `compaction-summarize` as `CompactionEngine` + `command-compact`; RLM-aware variable listing | P1-13, P1-10 | A3, G4, G10 | 0.85 / 170k triggers; never splits a pair; log intact; variables survive |
| P4-04 | `limits`: `ModelCallLimit`, `ToolCallLimit`, breaker, child caps | P1-02 | G5 | end/continue/error; breaker trips |
| P4-05 | `hitl`: `approve \| edit \| reject \| respond`, `MANUAL \| AUTO \| YOLO`, destructive classifier for `run_code` and mutating bindings | P1-08, P2-04 | G6 | pilot flows; destructive cell prompts in MANUAL |
| P4-06 | `permissions-fs`: first-match rows, recursive-delete fail-closed, `ls/glob/grep` post-filter; **reach self-report** when no confining provider is mounted | P1-17 | G7, E9 | first-match tests; the reach message toggles with a sandbox |
| P4-07 | `ctx.workspace` seam + `workspace-shared` (default; returns cwd; `scratch` at `<session artifacts>/scratch/`) | P0-02 | E5 | mounting the seam changes nothing until a profile opts in |
| P4-08 | `workspace-git-worktree`: `acquire(access)`; `worktree` (branch `ph/<session>/<agent>`) / `worktree-ephemeral`; `repo_writable`; `ctx.fs` root + `ctx.subprocess` cwd → `workspace.root`; redirection env; non-repo → decline + fallback + notice; dispose policy (keep dirty, remove clean, **discard ephemeral even if dirty**); `workspace/acquired`/`disposed` events | P4-07, P1-17, P1-18 | E2, E3, E5, E12 | two children merge without collision; ephemeral discarded; `pytest` green in a read child writing only to scratch; non-repo → `advisory` in doctor |
| P4-09 | `workspace/checkpoint` + `/revert <seq>`: `git write-tree` under `refs/ph/<session>/<agent>/pre-run/<seq>` before a mutating run (write-ahead: event first); restore tracked + untracked-not-ignored; never ignored paths or scratch; `/revert` lists non-filesystem dispatches | P4-08, P1-05 | E7, N3 | denied run reverts exactly; ignored paths untouched; irreversible dispatches listed |
| P4-10 | Default write scope: `SandboxExecutionPolicy {workspace-write, <worktree>}` + `permissions-fs` row for the `worktree` tier | P4-08, P4-06 | E6 | inside-worktree cell prompts zero times; outside prompts once |
| P4-11 | Containment tier selector (`containment.tier`, per-profile defaults `rlm: advisory`, child: `worktree`, `rlm-stable: worktree`) + **`containment.strict`** (refuse unless `sandbox` + `enforcement: full`) | P4-08, P1-19 | E1, E8 | strict on a backend-less host refuses, naming the backend |
| P4-12 | `ph doctor`: effective tier, per-agent `kind` + `repo_writable`, permission reach, worker model | P4-11 | E9, E10 | outputs match the three-column tier table |
| P4-13 | `memory-agents-md` (after caching), `skills-progressive`, `subagent-task` | P1-07, P1-23 | G8, G9 | prefix unchanged by a memory edit; skill body absent until requested |
| P4-14 | Paired-event reconciliation at session open (`workspace/acquired` without `disposed`) | P4-08 | F6 | crash between acquire and dispose reconciled on next open |
| P4-15 | `rlm-stable` profile; context-usage footer + todo sidebar; `dev-notes/phase-4.md` | P4-01 … P4-14 | — | profile boots |
| P4-16 | **Containment test module**: `worktree` bounds relative writes; **absolute-path `open()` escapes `worktree`** (asserted, so E1 cannot regress — the `sandbox` half lands in P6-06); ephemeral discard; scratch; redirection env; doctor effective tier | P4-08 … P4-12 | E1, E13 | suite green |

### Phase 5 — Long-running *(3–4 weeks)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P5-01 | Daemon supervisor: `$PH_RUNTIME/daemon.sock` (named pipe on Windows), JSONL framing, **one `anyio` task per root**, worker addressed by id | P1-25, P0-16 | I-2 | attach/detach; TUI close leaves the root running |
| P5-02 | Protocol: dsh SDK shape (`initialize`, `session/prompt`, `session.event`/`status`) + capabilities (`daemon_hello`, command envelopes, `{generation, sequence}` cursors, 512 KiB snapshot chunks); mutating commands journaled by `clientId+commandId` | P5-01 | — | reattach preserves streaming position; duplicate command is idempotent |
| P5-03 | Leases: in-process per root; `filelock` on canonical path against a second daemon | P5-01 | I-5 | concurrent open → `session_already_active` |
| P5-04 | Root-task crash recovery: retries 250 ms / 1 s / 5 s then failed; the root's tree restored from `workspace/checkpoint` | P5-01, P4-09 | — | injected crash recovers; third failure reports |
| P5-05 | Passivation sweeper (90 min default, `"off"`; requires no clients/heartbeats/cron/active descendants) + rehydration | P5-01, P3-13 | — | round-trip |
| P5-06 | Scheduler (`ctx.schedule`): `once \| cron \| interval` via `croniter`, ticks claimed before delivery, missed ticks coalesce; heartbeats every 5 m | P5-01, P1-23 | — | coalescing test |
| P5-07 | Goals (`ctx.goals`) and `/autonomous` **as tools and commands** (C2 demotion): `GoalState`, budgets `{max_continuations: 3, max_turns: 12, max_tokens: 80_000, timeout: 30 min}`, shell quality gates with worktree fingerprint | P5-06, P4-08 | — | budget exhaustion → `budget_limited`; unchanged failed gate not re-run |
| P5-08 | SQLite persistence provider + full-text session search | P0-15 | — | same `SessionPersistence` tests pass on both backends |
| P5-09 | OTel telemetry sink | P1-14 | — | redaction still precedes export |
| P5-10 | `ph agents \| attach \| send \| schedule \| status \| doctor \| shutdown` | P5-02 | — | each round-trips |
| P5-11 | **Lingering detection**: doctor reports `$XDG_RUNTIME_DIR` derivation + `loginctl` state, names `enable-linger`; "socket path removed" diagnostic | P5-01 | I-6 | simulated session end → clear diagnostic, not a silent failure |
| P5-12 | Non-guarantees documented (no per-root memory cap, no rolling restart, no cross-root crash containment; one daemon per user); `dev-notes/phase-5.md` | P5-01 | I-2, N5 | doctor prints the worker model; docs reviewed |

### Phase 6 — Hardening & docs *(ongoing)*

| ID | Work item | Depends | Delivers | Gate |
|---|---|---|---|---|
| P6-01 | Invariants registry + runtime invariant plugins (session, loop, tools, scope) incl. **"every projection equals its fold"** | P3-15, P3-16 | A11, I6 | hand-edited projection trips the invariant |
| P6-02 | `ph events` producer/consumer matrix; config catalog; 100 % coverage gate on `ph-core` | P0-04 | — | CI gates |
| P6-03 | Benchmark: prefix-cache hit rate and tokens/turn, recorded RLM session with vs without stabilization | P4-15, P3-23 | — | report checked in |
| P6-04 | **`sandbox-local`**: `bwrap` → Landlock (Linux), Seatbelt (macOS); `confine()` for the runtime and shell; `enforcement: full \| partial`; denial signatures | P1-19 | E11 | absolute-path write refused under `sandbox` (closes E13's second half) |
| P6-05 | `readonly-scratch` kind via `workspace-write` rooted at scratch; **degradation path**: `partial` → refusal under strict / reported downgrade otherwise; `access="read"` → `worktree-ephemeral` where no enforcing backend | P6-04, P4-08 | E3, E11 | read child cannot write the repo; degradation reported in doctor |
| P6-06 | Docs test: no tier described as bounding writes it does not bound | P6-04 | E1 | CI |
| P6-07 | "Running pH in a container" page (what to mount; fd 3 crosses as an inherited descriptor; `CredentialRef`; tiers layer inside) | P6-04 | N6 | reviewed |
| P6-08 | `code-runtime-quickjs` (D16) — **only if** a profile needs sandboxed code mode | P1-06 | — | conformance suite passes against it |
| P6-09 | `ph harness report` — analytics only ("these skill entries wrap the same import") | P3-16 | H4 | no scaffold command exists |
| P6-10 | Cookbook: adding a plugin / tool / adapter / seam; every seam page | all | — | every seam has a page |
| P6-11 | Windows in CI for every lifecycle and path test | P3-22, P0-16 | F3, I-1 | matrix green |

---

## 5. Engineering rules that hold across every phase

1. **Declare, never derive.** Aliases are fixed at class definition (`WireModel`); a name is never reconstructed from a wire string. Same for event modes (`events.declare`) and tool outputs (mandatory `output`).
2. **Log first, act second.** `tool/call` before execution; the snapshot event before the blob; `rlm/child-admitted` before the handle returns; `workspace/checkpoint` before the mutating run.
3. **Every artifact through `ctx.effect()`.** A lint forbids `subprocess.Popen`, `tempfile.mkdtemp`, `git worktree add` and lock acquisition outside the seam.
4. **The program and the child are hostile.** Every inbound fd-3 frame is rebuilt; every binding argument is snapshotted as lossless JSON; the guest trusts the host, the host trusts nothing.
5. **Fail closed at the seam.** `SANDBOX_UNAVAILABLE` rather than unconfined; `unavailable` approval denies; a `partial` backend under `strict` refuses; an unresolvable `reference` is rejected.
6. **State what is not enforced, next to where it would be assumed.** The tier table, the permission row's validation, `ph doctor`, `/revert`'s output and the first-run notice all carry their own caveat. A caveat only in the docs is a defect.
7. **Constants come from Appendix D, exposed as row config.** No hidden `??` defaults inside a `run()`.
8. **Every phase ends with its gate module green on three OSes** and a `dev-notes/phase-N.md` recording what was traded.

---

## 6. Definition of done, per phase

| Phase | Done when |
|---|---|
| 0 | P0-14's invariant fires on a bypass; the wire round-trip property passes; `ph doctor` prints three resolved roots on all three OSes; a print-mode run writes a JSONL that dsh tooling reads |
| 1 | dsh's pipeline ordering invariants pass; crash-repair fixtures resume; the prefix-stability test passes on a recorded session; registering a namespace provider without a snapshot hook fails at `provide()`; `SIGTERM` unwinds within grace |
| 2 | pilot tests drive every modal; resume rebuilds the transcript from events alone |
| 3 | **P3-21 (governance gate) and P3-22 (runtime conformance) both green on three OSes**; fork restores the boundary namespace; harness re-derives byte-identical; the fixture-replay report is checked in with its diffs triaged |
| 4 | P4-16 green; a denied run reverts exactly; an inside-worktree cell prompts zero times; `containment.strict` refuses on a backend-less host; doctor's tier output matches the table |
| 5 | reattach preserves position; concurrent open is refused; lingering state reported; non-guarantees documented and asserted |
| 6 | 100 % coverage on `ph-core`; the docs test passes; `sandbox` refuses an absolute-path write; every seam has a page; Windows in the matrix |
