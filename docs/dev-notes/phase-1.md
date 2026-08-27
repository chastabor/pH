# Phase 1 — Core parity

**Status:** complete · **Gate:** `ruff` + `ruff format` + `mypy --strict` on `ph-core` + 370 tests, green.

Phase 0 proved the event-sourced core was portable. Phase 1 had to prove
something harder: that **every action a model takes passes through one governed
pipeline**, and that the pipeline's ordering, failure modes and durability
barriers are the ones dsh actually implements rather than the ones a summary
table describes.

---

## What landed

| Item | Delivered | Where |
|---|---|---|
| P1-01 | `ToolDefinition` with mandatory `output`+`render`, `define_tool`, global/scoped registration with shadowing, `restrict`, `schemas(scope)`, `tools/change` | `ph/tools/{definition,registry}.py` |
| P1-02 | The pipeline: `tools/pre-execute` → approval → guards → `tools/execute` → body → `tools/post-execute` → normalize → `finalize_content` → `tools/result` | `ph/tools/registry.py` |
| P1-03 | Execution modes, bounded rolling pool, exclusive barriers, model-order commit | `ph/tools/batch.py` |
| P1-04 | `tools.mode`, `present_as`, reserved `run_code`, `tools:sdk` renderers (Python + TypeScript), `UNKNOWN_TOOL` before policy | `ph/tools/{registry,sdk,code_mode}.py` |
| P1-05 | Dispatch bridge: per-call pipeline re-entry, `tool/code-dispatch-start`/`-dispatch`, budgets, denial fails the run | `ph/tools/code_mode.py` |
| P1-06 | `ctx.code_runtime` seam definition + the persistence obligation, checked at registration | `ph/seams/code_runtime.py` |
| P1-07 | Prompt `tools` providers receive the target scope; header fold and change detection | `ph/system_prompt/assembly.py`, `ph/tools/prompt.py` |
| P1-08 | `ctx.approval`, four outcomes, fail-closed, pending-on-resume derived from the log | `ph/seams/approval.py` |
| P1-09 | `ctx.commands` (no model turn), `ctx.user_questions` | `ph/seams/{commands,user_questions}.py` |
| P1-10 | Retry with backoff; context-overflow deliberately **not** retried | `ph/llm/retry.py` |
| P1-11 | Three checkpoint barriers, fail-closed at the model and tool boundaries | `ph/persistence/checkpoint_policy.py` |
| P1-12 | Crash repair with dsh's exact synthetic vocabulary | `ph/persistence/repair.py` |
| P1-13 | `ctx.token_meter`: usage authoritative, estimate for pressure | `ph/seams/token_meter.py` |
| P1-14 | `ctx.session_telemetry`: redaction before any sink; first chunk per step only | `ph/seams/telemetry.py` |
| P1-15 | OpenAI-compatible (DeepSeek `reasoning_content`) and Anthropic adapters | `ph_app/adapters/` |
| P1-16 | `ctx.credentials`: `CredentialRef` travels, values resolve at the adapter edge | `ph/seams/credentials.py` |
| P1-17 | `ctx.fs` + `fs-local`, write/edit intent gates, read-before-edit as its own row | `ph/seams/fs.py` |
| P1-18 | `ctx.subprocess`: explicit spec, scrubbed env, reap in `finally`, spawned via `ctx.effect()` | `ph/seams/subprocess.py` |
| P1-19 | `ctx.shell`, `ctx.sandbox` (refuses rather than passing through), permission presets | `ph/seams/{shell,sandbox,permission_presets}.py` |
| P1-20 | Lifecycle disposal on exit/signal; `temporary_directory`; the acquisition lint | `ph/resources.py`, `tests/test_resources.py` |
| P1-21 | `ctx.spill_store` with digest naming and a retrieval hint | `ph/seams/spill.py` |
| P1-22 | `read`/`write`/`edit`/`glob`/`grep`/`bash` with cards; AGENTS.md discovery | `ph/tools/builtin/` |
| P1-23 | `ctx.jobs`, `ctx.settings`, `ctx.skills` | `ph/seams/{jobs,settings,skills}.py` |
| P1-24 | `llm-replay` + the prefix-stability assertion | `ph/testing/replay_adapter.py`, `tests/test_prefix_stability.py` |
| P1-25 | `json` / `transcript` / `rpc` modes, provider profiles | `ph_app/modes/`, `ph_app/profiles/` |

**Definition of done, met:** the pipeline ordering tests pass; crash-repair
fixtures resume into a provider-valid transcript; the prefix-stability test
passes over a recording; registering a `persistence: "namespace"` runtime with no
snapshot promise fails at registration; `SIGTERM` unwinds within the grace
period.

---

## Decisions taken inside Phase 1

### 1. Guards run **after** approval — the pH plans' tables are wrong

Both `Python_Harness_Port_Plan.md` §4.4 and `Implementation_Plan.md` row B1
summarise the order as `pre-execute → guards → approval`. dsh does the opposite,
in both its documentation and its code:

```ts
const gate = await waterfall('tools/pre-execute', exec, () => ({kind:'allow'}))
const askResolution = gate.kind === 'ask' ? await serviceAsk(exec, gate) : {decision: gate}
const denialReason = decision.kind === 'allow' ? this.guardReason(exec) : decision.reason
```

pH follows dsh, because the port plan's own §4.4 says "Pipeline exactly as
`docs/tool-execution-pipeline.md`" and then paraphrases it incorrectly — the
normative sentence is the reference, not the summary.

It also happens to be the better order, and the reason is worth keeping: a guard
is **deny-only and runs last**, so it is the final word *even over a human's
explicit approval*. That is what "monotonic" buys. Policy that must not be
overridable stays a guard; policy that is a judgement call is a listener. Under
the summary's order a guard could never refuse something a human had approved,
which is exactly the case a hard policy exists for.

A `pre-execute` denial still skips the guards: nothing is left to decide, and
asking a guard to confirm a denial only creates an opportunity to re-permit it.

### 2. A failure declares its kind; nothing infers it

The bridge has to tell "policy refused" from "the tool broke", because they take
opposite paths (C3: a refusal ends the run, a failure is the program's to
handle). Inferring it from the absence of an error code worked until a tool body
raised a plain `RuntimeError` — indistinguishable, and the test caught it. A
second draft kept a set of denial codes in the bridge, which would have needed a
new entry for every Phase 4 and 6 gate — a missed one being a refusal a program
could `except` its way past, silently.

So `ToolFailure.kind: "denied" | "failed" | "aborted"` is set **by the
producer**: `prepare()` builds denials, `HarnessError.denies` marks the error
classes that are refusals (`ToolNotFoundError`, `SandboxError`), and cancellation
is `aborted`. The kind travels into the `tool/result` event as `failureKind`, so
a Phase 2 card gets the same fact without re-deriving it.

### 3. tau_ai was not wrapped, and D6's premise is why

D6 says to wrap `tau_ai` providers, on the stated grounds that it "already emits
a provider-neutral text/thinking/tool-call **start/delta/end** stream". Reading
`tau_ai/_provider_events.py`, it does not: `ProviderToolCallEvent` carries a
**complete** `ToolCall`, and there is no incremental `arguments` delta.

`assistant/chunk` promises token-level replay fidelity, so wrapping it would
have silently made that promise false for every tool call — and the log is the
trace (§8). It would also have made `ph-app` depend on `tau_agent` and
`tau_coding`, a second agent framework, for a mapping we would write anyway.

pH therefore ships native adapters over `httpx`: one for the OpenAI-compatible
wire (covering DeepSeek, including `reasoning_content`) and one for Anthropic.
Both stream real `tool-call-delta` chunks. The tau-modelled **TUI** (D7) is
unaffected and still lands in Phase 2.

### 4. Two mappings that are easy to get wrong, and one that is not symmetric

* **Thinking is a `reasoning` block, never appended to text.** DeepSeek's
  `reasoning_content` and Anthropic's `thinking` blocks both map across
  separately. Folding them into visible text would make the transcript claim
  the model *said* what it was only considering.
* **Usage counts are disjoint (D15), and the two wires differ.** DeepSeek folds
  cache hits into `prompt_tokens`, so `prompt_cache_hit_tokens` is subtracted
  out; Anthropic already reports them separately, so they map directly. Getting
  this backwards over-reports every cached turn — silently, and only on the
  invoice.

### 5. Frozen log data is thawed at the one layer every model shares

A latent Phase 0 bug, found by a Phase 1 test. Pydantic coerces a
`MappingProxyType` into a declared `dict` field, but a field typed `Any` keeps
what it was given — so a JSON array read back from the log stays a **tuple**.
`ToolSchema.parameters` is `Any`, so `required: ("path",)` compared unequal to
`required: ["path"]`, `header_equals` returned `False` for two identical headers,
and a `request/header` was appended on **every step** — invalidating the cached
prefix each turn (A12).

The first fix was a `from_log()` helper call sites had to remember. That is a
per-callsite discipline, and the next `Any`-typed field (`ImageBlock.attachment`,
`ModelSource.replay_state`) would have hit it again. `WireModel` now thaws a
frozen input in a `mode="before"` validator, so every model is robust to log data
at one layer and no call site has to know. The cost is one `isinstance` per
validation and a copy only when the input actually was frozen.

### 6. `ToolModel`: the one declared exemption from camelCase

Q2 exempts tool *parameter* names, because they are Python identifiers in the
generated SDK — under Code Mode the model writes `await tools.edit(old_text=…)`.
Rather than leaving that as a convention the wire test would flag, tool schemas
now subclass `ToolModel`, which the test knows about. The exemption is visible at
every declaration site, and nothing else can claim it by accident.

### 7. "At step end" is barrier 1 wearing a different name

A4 asks for three barriers, one of them "at step end". dsh flushes on
`agent/pre-step`, and the first draft did too — which meant **two fsyncs per
step, back to back**: the pre-step flush, then `step/start`/`user/message`/
`request/header` appended, then the `llm/stream` flush. The second is never a
no-op, and the first buys nothing the second does not.

So on the request path there is one barrier. The only step end it never reaches
is a pre-step **reject** (no request follows), and that is the one case flushed
after the decision. Two barriers remain **fail-closed**: if the flush raises,
neither the adapter nor the tool body is invoked, since a side effect whose
record could not be written is worse than one that did not happen.

### 8. `CancelToken` rather than a bare cancel scope

The pipeline has to distinguish "aborted before dispatch" (the call had no
effect) from "aborted" (the body ran) at points where no `await` is pending. A
scope only *acts* on cancellation; a token can be **asked**. A child token is
cancelled by itself or by any ancestor, which makes dsh's "the registry fuses
every replacement with the captured caller signal" structural instead of
something each `tools/execute` wrapper must remember.

### 9. A signal handler cannot block the loop it interrupted

The first `install_lifecycle` tried to run disposal synchronously from the
handler. That deadlocks: the handler runs on the main thread, *interrupting* the
event loop it needs in order to finish. It now schedules the teardown as a task
(keeping a reference, so it cannot be collected mid-flight) and returns; the task
unwinds and then re-raises the signal with the default handler. Past the grace
period pH stops trusting its own teardown — a shutdown path that can hang will.

---

## Deliberately deferred

| Deferred | To | Why |
|---|---|---|
| A real `ctx.code_runtime` provider | P3-05 (D19) | Phase 1 owns the *governance*; the fd-3 CPython subprocess is Phase 3. The stub runtime exercises the bridge |
| `sandbox-local` (`bwrap`/Landlock/Seatbelt) | P6-04 | Until then `confine()` refuses, which is the honest answer — and only this tier bounds an absolute-path write (N2, E13) |
| `ctx.workspace` | P4-07/P4-08 | `ctx.fs` roots at cwd; per-agent worktrees are Phase 4 |
| Compaction, offload, limits, HITL modes, todo | Phase 4 | The seams they attach to (`tools/post-execute`, `agent/pre-step`, `ctx.approval`, `ctx.spill_store`) all exist |
| `session-query-sqlite`, OTel sink | P5-08/P5-09 | One `SessionPersistence` contract; a backend is a row swap |
| `adapter_defaults` re-resolution | later | `_request_proposal` is in place; no adapter reports defaults yet |
| MCP tools | Phase 6 | The raw-JSON-Schema path in `ToolOutput`/`ToolDefinition` exists for them |

### 10. Things the `/simplify` pass moved to the right layer

* **`register_transport`** — the Code Mode row claims `run_code` through a
  registry method that owns the invariant, not by writing into a private layer.
  It takes the ordinary path, so `tools/change` fires and disposal cleans up.
* **`ToolRunContext.definition`** replaces a token-keyed side table that leaked
  on every path that ended before `finish()`.
* **`ToolExecutionInput.scope`/`session`** — the loop states its agent's scope
  and log; the registry stopped inferring them from `getattr(agent, "ctx")`,
  which would have degraded a shape mismatch to "global policy only" silently.
* **`CodeBinding.counts_as_spawn`** — the spawn budget is a property the
  namespace author declares, not a match on the string `"rlm"` (which could
  never fire: binding names may not contain dots, and the bridge sees tool
  names).
* **One SDK-renderer registry**, on `ctx.code_runtime`. A disposed renderer
  leaves an absence that fails loudly, not a silent fallback whose different text
  would invalidate the cached prefix.
* **`tools-timeout`** — `ToolDefinition.timeout_ms` had no enforcement; a
  declared budget with nothing behind it is the type telling a lie. It is a row,
  as in dsh, so the policy is swappable.
* **`max_parallel_tool_calls`** is agent-loop config (dsh's
  `agentLoop.config.maxParallelToolCalls`), not a Code Mode knob repurposed.
* **One `httpx.AsyncClient` per adapter**, not per request — a fresh TCP+TLS
  handshake per model call was the largest wall-clock waste in the diff.

## Known sharp edges

* **The JSON-Schema subset is a subset.** `unsupported_keywords()` reports what
  is not enforced; a caller taking a foreign schema (MCP, Phase 6) should refuse
  one it cannot fully check rather than assume validation covered it.
* **`ctx.jobs` runs inline until a host binds a task group.** Correct for a
  one-shot CLI; the daemon (Phase 5) binds one.
* **`token_meter.baseline()` has no caller yet.** When compaction wires it in
  (Phase 4), the pre-usage estimate re-tokenizes the whole history per call;
  cache per message id then.
