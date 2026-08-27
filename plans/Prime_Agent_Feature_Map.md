# Prime Agent → pH Feature Map, and What Does Not Fit

**Status:** analysis / v1.6 — 2026-08-26 — **All thirteen §12 questions decided; Q1's `$PH_RUNTIME` resolution order and its `logind` lingering caveat recorded.** (v1.5: Q2 — camelCase on every JSON boundary, snake_case in Python, tool parameters exempt.) (v1.4: Q1 — `ph` everywhere; hybrid paths.) (v1.3: Q4 — context access is a binding, the corpus is a `recipe` snapshot, and logged/provenance/replay are three separate properties.) (v1.2: Q7 — one task per root; child lifecycle is spawn flags plus a journal.) (v1.1: all architecture-shaping questions closed.) (v1.0: Q13 — capability layer and knowledge layer are two axes, not one ladder.) (v0.9: Q10 — the containment ladder is two properties; deny rows self-report their reach; `containment.strict`.) (v0.8: Q9 closed — denial fails the run, worktree is the default write scope, per-run checkpoints. v0.7: Q5 closed.) (v0.6: containers ruled out of scope — pH's ladder is advisory → worktree → sandbox, Q12 closed.) (v0.5: §5 reversed — pH implements its own code runtime; item 4 solved, item 9 dissolved. v0.4: C1–C3 folded, §6a, Q11 closed)
**Companion to:** [Python_Harness_Port_Plan.md](Python_Harness_Port_Plan.md) (the specification, which carries the decisions this document argued for) · [Implementation_Plan.md](Implementation_Plan.md) (the work plan: every safety and stability feature as a gated work item)
**Question answered:** does prime-agent's design bypass the stability features of `deepseek-harness` and LangChain Deep Agents, and if so, what is the dependency-correct mapping onto pH?

**Answer in one line:** yes, it bypasses them — but ZeroMQ is not the mechanism. The mechanism is `export type ToolName = "ipython"`.

> **Reading order.** This document is the *evidence and rationale*; the port plan is the *specification*. §4 (C1–C3) and §5 (transport) are now decided, not proposed — each subsection carries a **Landed in** pointer to the plan section that owns it. §6 (what does not fit), §6a (the tool-authoring deviation) and §7 (the inventory) remain the reference material the plan cites back to. Where the two disagree, **the port plan wins** and this document should be corrected.

### Where each conclusion landed in the port plan

| This document | Port plan v0.3 |
|---|---|
| §0 single-tool finding | §1.3 (recorded at source), §0 fact 5 |
| §1 no HTTP server; reject `AsyncSubAgentMiddleware` | §1.4, §8 "Non-blocking delegation" row, D11 |
| §2 the bypass, quantified | §0 fact 5, §6.1 "What C1–C3 changed" |
| §3 dsh Code Mode is the missing half | §4.4 (dispatch bridge), D18 |
| **C1** kernel as `CodeRuntime` provider | **§4.7** (seam gains `namespace`/`persistence`), **§6.2** |
| **C2** host requests → bindings | **§6.3** (`rlm-bindings`, `rlm-kernel-compat`), D18 |
| **C3** raw side effects → bindings | **§6.3**, §6.8 (skills split), §6.4/§6.5 (spawn + family guard) |
| **§5** pH builds its own code runtime (D19 **reversed**) | **D19** (rewritten), **§6.2** (`code-runtime-python`), §6.3, §6.8, §12 Q8 |
| §6 items 3–4 (raw Python, `%%bash`) | §6.2 "Trust and the enforcement boundary", §11 risk row, §12 Q10 |
| **§6a** the tool-authoring deviation + containment tiers | **§4.8** (the `ctx.workspace` seam, the three-tier ladder, the `access` kinds), **D20**/**D21**; tier and `access` defaults settled in **§12 Q11 (closed)**; containers ruled out of scope in **§12 Q12 (closed)** |
| §6 item 5 (`harness_state.json`) | **D14** (rewritten: events are the fold, the file is a projection), §6.6, **§12 Q5 (closed)** |
| §6 item 6 (`kernel-state.dill`) | D17, §6.6 |
| §6 item 7 (trace upload) | §7 verdict **D**; not ported |
| §6 item 8 (no approval layer) | §7.6, §4.7 `approval`/`permission_presets` |
| §8 governance test | §10 Phase 3 exit criteria (a)–(g) |

---

## 0. The finding that reframes the question

`packages/coding-agent/src/core/tools/index.ts`:

```ts
export type ToolName = "ipython";

export function createAllToolDefinitions(cwd: string, options?: ToolsOptions): Record<ToolName, ToolDef> {
	return { ipython: createIpythonToolDefinition(cwd, options?.ipython) };
}
```

`bash.ts` (452 lines) and `edit.ts` (533 lines) sit in the same directory and are **not** in that record. The extension API still carries `ReplayBuiltInToolName = "bash" | "edit"` with the comment *"Replay renderer to use for removed built-ins in saved transcripts."* Prime Agent shipped those tools, then **deliberately removed them** and collapsed the entire model-facing surface to one callable.

Every stability feature in `deepseek-harness` and in Deep Agents is a hook on a **tool boundary**. Collapse the tool surface to one entry and you collapse the governance surface to one evaluation per cell. ZeroMQ is downstream of that decision, not the cause of it: the comm channel exists *because* a model that can only call `ipython` still needs to reach the host, and the only route left is from inside the kernel.

So the concern is correct and the target is one level up from where it was aimed.

---

## 1. The HTTP-server constraint

**Prime Agent needs no HTTP server, and neither does the design proposed here.** Verified across `packages/*/src`:

| Channel | Transport | Server? |
|---|---|---|
| Daemon supervisor ↔ clients | unix socket, JSONL (`daemon-supervisor.ts` → `server.listen(this.socketPath)`) | unix socket only |
| Kernel fork-server | unix socket (`kernel/fork-server.ts:203`) | unix socket only |
| RPC / ACP / JSON modes | newline-delimited JSON on stdin/stdout | none |
| Host ↔ IPython kernel | ZeroMQ over loopback TCP (`jupyter_client`), HMAC-signed | not a server in any operational sense — a child process's IPC |
| MCP integrations | `stdio` first-class (`mcp-manager.ts:151`); streamable-HTTP is an **outbound client** to a remote server | none |

The only `http.createServer` calls in the whole repo are **ephemeral OAuth callback listeners** — `packages/ai/src/utils/oauth/anthropic.ts` (`CALLBACK_PORT`), `openai-codex.ts` (port 1455), `mcp/oauth.ts`. Each binds localhost, receives one browser redirect, and closes. That is a login flow, not a running service, and it only exists for providers you choose to authenticate against.

**The HTTP-server risk comes from Deep Agents, not from Prime Agent.** `deepagents/middleware/async_subagents.py` — the *only* Deep Agents subagent form that returns a handle instead of blocking, i.e. the only one with RLM's semantics — drives a remote Agent Protocol server through `langgraph_sdk`:

```python
thread = await client.threads.create()
run = await client.runs.create(thread_id=thread["thread_id"], ...)
```

That needs LangGraph Platform or a self-hosted Agent Protocol server, with its own run queue. Ordinary `SubAgentMiddleware` subagents are in-process compiled subgraphs that **block** until done.

**Decision: do not adopt `AsyncSubAgentMiddleware`.** pH's `ctx.subagents` provider (§6.4, D11) already returns an admission handle immediately from an `anyio` task in the agent's scope, with replies re-entering through the parent inbox. Same semantics, no server, no queue. This is the one place where the existing plan is structurally *lighter* than Deep Agents, and it should stay that way.

One item to decide explicitly rather than inherit: `agent-traces.ts` (968 lines) uploads session JSONL to `${baseUrl}/api/v1/agent-traces/sessions/<id>` on Prime Intellect's API. It is an outbound client, not a server, but it is **vendor data egress** and should be dropped in favour of `ctx.session_telemetry` + OTel (§8). See §4 item 7.

---

## 2. The bypass, quantified

Prime Agent is not hookless. Its extension API (`core/extensions/types.ts`) exposes ~30 events including veto-capable `tool_call` / `tool_result` handlers, `before_provider_request`, `session_before_compact`, `session_before_refine`, `context`. The seams exist. **They fire once per cell.**

A single `ipython` call that writes 40 files, runs 6 shell commands, spawns 8 RLM children and returns 200 KB is, to every listener, one `tool/call` carrying `{code: "<program>"}` and one `tool/result` carrying a text blob.

| Governance point | dsh / Deep Agents mechanism | What one `ipython` cell does to it |
|---|---|---|
| Permission / allow-deny | `tools/pre-execute` → allow \| deny \| ask | evaluated against a Python **program string**. There is no allow-list for arbitrary code. |
| Human-in-the-loop approval | `ctx.approval.request()`, `interrupt_on` | one prompt for "run this code", never per side effect |
| Tool-call limits | `ToolCallLimit`, monotonic guards | counts **1**, regardless of what the cell did |
| Large-result offload | `tools/post-execute` ≥ threshold → spill + locator | offloads the concatenated cell output; the individual results were already merged into stdout and are unrecoverable |
| Filesystem write intent | `fs/write-intent`, `fs/edit-intent` | **never fires** — the `edit` skill calls `pathlib.Path.write_text()` in the kernel |
| Sandbox confinement | `ctx.sandbox.confine()` | wraps the kernel launch argv; nothing inside the cell |
| Per-tool timeout / retry / metrics | `tools/execute` around-wrapper | one budget for the whole program |
| Durable per-action record | `tool/call` + `tool/result` events | one pair. Forty file writes are stdout text inside one blob. |
| Prefix-cache stability | `request/header`, surface `replace` | unaffected — this one survives |

Proof that the MIME side channel is a notification and not a gate — `skills/edit/src/edit/__init__.py`:

```python
filepath.write_text(content.replace(old_str, new_str, 1), encoding="utf-8")
resolved_path = str(filepath.resolve())
_emit_diff(resolved_path, old_str, new_str, start_line)   # display_data, AFTER the write
```

The host learns a file changed **after it changed**. There is no point at which anything can say no.

And the `host.request` comm bridge is a *second* ungoverned path: comm frames dispatched on `data["type"]` to host handlers, touching the tool pipeline at no point. The port plan logs only `rlm/host-request {type, payload_digest, ok}` as `ignorable: true` — deliberately not model-visible, therefore not offloadable, not approvable, not countable.

**Prime Agent has no approval or permission subsystem at all.** Grepping `approval|permission` across `packages/coding-agent/src/core/` hits only `auth-guidance.ts`, `prime-inference-auth.ts` and `sdk.ts` — all about API credentials. There is nothing to bypass because nothing was built. That is a gap to fill from dsh + Deep Agents, not a mismatch to reconcile.

---

## 3. `deepseek-harness` already solved this, and already rejected the kernel for exactly this reason

From `packages/core/tools/README.md`:

> **`run_code` state is fresh per run** — a persistent REPL-style kernel is rejected for the MVP (**cross-call state would be invisible to the log**).

That is dsh's own authors stating your concern as their reason for not shipping a kernel. And the mechanism they shipped instead is precisely the missing half of Prime Agent:

> Each lossless-JSON binding call **re-enters the complete tool pipeline** under the native scheduling contract … with logged correlation to the outer call.
> Each started sub-call logs a `tool/code-dispatch-start` event (deterministic id `<parent>:code:<n>`) … and settles with one `tool/code-dispatch` event carrying the complete model-facing `content`/`isError` outcome.
> the `tools/code-dispatch-log` waterfall lets the spill policy replace an oversized `tool/code-dispatch` content with a preview + locator.

So:

|  | dsh **Code Mode** | Prime Agent **`ipython`** |
|---|---|---|
| Model-facing surface | one callable (`run_code`) | one callable (`ipython`) |
| In-code capability access | `await tools.name(args)` **bindings** | raw Python + `host.request` comm |
| Governance of in-code actions | full pipeline per binding call, logged as `tool/code-dispatch` | none |
| Cross-call state | **none** ("fresh per run") | **persistent kernel namespace** |
| Transport to the runtime | fd-3 JSON-lines (`code-runtime-python`), host validates and **rebuilds every inbound frame as hostile** | ZeroMQ / Jupyter comm |

**Each has the half the other lacks.** pH is the only place both halves can exist at once, because D17 (kernel state as `kernel/snapshot` events in the append-only log) removes the exact objection — "invisible to the log" — that made dsh refuse persistence.

Worth noting for the transport question: `packages/code-runtime/code-runtime-python/` already runs **CPython out of process with no ZeroMQ**, over JSON-lines on fd 3, with the host treating every inbound frame as hostile. A non-ZMQ Python code runtime is not hypothetical; half of it is already written in the codebase being ported.

---

## 4. The three structural changes (accepted; folded into the port plan v0.3)

### C1 — The kernel becomes a `CodeRuntime` provider; `ipython` becomes Code Mode with a persistent namespace

> **Landed in** port plan §4.7 (the seam definition, promoted into `ph-core`), §4.4 (presentation mode + the dispatch bridge), §6.2 (the provider), D18. It is the **only** core change in the fold; everything else is a plugin row.

`ctx.code_runtime` already has the right contract: `CodeRunRequest {program, bindings: CodeBindingNamespace[]}` → `CodeRunResult {value?, logs, error?}`, with the Consumer (`dsh-tools` Code Mode) owning SDK generation and tool dispatch. **One contract change** is needed, and it is the one dsh explicitly deferred:

```
CodeRuntime.run(request, *, namespace: str | None = None)
```

`namespace=None` keeps today's fresh-per-run semantics; a key selects a persistent namespace. The seam's "no state survives between runs" clause becomes provider-declared (`persistence: 'none' | 'namespace'`), and a persistent provider **must** emit `kernel/snapshot` events (D17) — the log-visibility requirement is enforced by the seam, not by convention.

The model still sees one callable, so Prime Agent's ergonomics are unchanged. `presentAs: code` is already per-agent in dsh, so the RLM preset selects it without affecting other presets.

*As folded:* the plan names the namespace key (the **agent id**, so a kernel is scoped exactly like the agent's tools, inbox and log), keeps the transport's model-facing name as `ipython` via a presentation alias, and adds a risk the argument above understates — the generated `tools:sdk` block displaces prompt text the RLM doctrine relied on, so Phase 3 replays prime-agent's trajectory fixtures under the new surface before the profile is declared done (§11).

### C2 — Every `host.request` type becomes a binding namespace, not an RPC catalogue

> **Landed in** port plan §6.3 (`rlm-bindings` + the four namespaces, `rlm-kernel-compat`), §6.1 (rows), §6.5 (the family boundary as a monotonic `ctx.tools.guard` rather than a handler check).

| Prime Agent `host.request` | pH binding | Pipeline it now re-enters |
|---|---|---|
| `rlm.run` | `await rlm(prompt, ...)` → `tools.spawn_subagent` binding | pre-execute (depth + model policy) → `ctx.subagents` provider → `tool/code-dispatch` logged |
| `rlm.list_subagents` / `.delete_subagent` / `.find_models` | `tools.list_subagents` / `delete_subagent` / `find_models` | full pipeline; deletion becomes an approvable action |
| `agent_message.send` / `.list_agents` | `tools.send_agent_message` / `list_agents` | family-boundary guard becomes a `ctx.tools.guard`, rate limit a pre-execute policy, receipt a real `tool/result` |
| `agent_observe.*` | `tools.observe_*` | offloadable like any other large read |
| `goal.*`, `compact.*`, `refine.*`, `rlm_heartbeat.*`, `mcp.refresh` | **not bindings** — ordinary tools + `/goal`, `/compact`, `/refine`, `/heartbeat` commands | native tool calls; the model was never calling these from inside a loop |
| `model.info` | prompt `context()` section | not a call at all |
| diff / attachment / agent-message MIME side channels | **replaced** by C3 bindings | the write goes through the binding; the display frame becomes presentation only |

Net: the comm channel carries a **fixed, small** contract instead of an open-ended RPC that grows with every capability the harness gains.

### C3 — Every raw side effect inside the cell becomes a binding call

> **Landed in** port plan §6.3 (the `tools` namespace), §6.8 (which bundled skills are replaced), §6.2 (the enforcement boundary and the non-goal). The plan adds a per-cell dispatch budget the argument below omits — `max_dispatches_per_run` 256, `max_subagent_spawns_per_run` 32 — because governance is per call but attention is per turn.

- `edit` skill → `await tools.edit(path, old_str, new_str)` → `fs/write-intent` + permission + approval + a real diff card + a durable `tool/code-dispatch` record.
- `%%bash` → steer to `await tools.bash(cmd)` → sandbox provider + permission + timeout policy + `tool/call` record.
- Read/glob/grep/websearch → bindings, so results are individually offloadable instead of being melted into stdout.

This is also the answer to a limitation the current plan carries: under the RLM preset the model has *only* `ipython`, so it must reimplement search and file reading in Python. Bindings give it the harness's real tools, governed.

---

## 5. The code runtime: pH builds its own (decided — port plan **D19**, reversed)

> **Landed in** port plan D19 (rewritten), §6.2 (`code-runtime-python`), §6.3 (the compat shim shrinks to a translation layer), §6.8 (reuse verdicts), §12 Q8.

**Earlier versions of this document framed §5 as a port choice — adopt prime-agent's Jupyter kernel, or adopt dsh's fd-3 protocol. That framing was wrong, and it kept producing the same answer for the wrong reason.** Both options were "which upstream do we copy?", and the argument for the Jupyter kernel was almost entirely *"`prime-agent-runtime` and its bundled skills run unmodified"* — i.e. the value of not writing something, purchased with a permanent dependency on another project's execution model.

The decision is now: **pH implements its own `ctx.code_runtime` provider — `code-runtime-python` — a CPython subprocess speaking a pH-owned JSON-lines protocol on fd 3, with a persistent namespace.** dsh's `code-runtime-python` is the *reference for the protocol shape*, not code to vendor; prime-agent's kernel is the *reference for RLM semantics*, not a component to depend on. Neither is a substrate.

### Why this is the right call and not merely a preference

Under C1–C3 the runtime's job is small and fully specified by pH's own seams: **run one program against a set of host-provided bindings, against a namespace that persists between runs, emitting `kernel/snapshot` events so that state stays in the log.** That is a contract pH defines. An IPython kernel is a large, general-purpose interactive environment that happens to be able to satisfy it — and the parts we do not need (kernel specs, connection files, HMAC session keys, ZeroMQ, comm targets, `nest_asyncio`, the control-channel deadlock workaround) are exactly the parts that generate the awkwardness catalogued in §6.

The clinching point is that **the Jupyter features we would be paying for are the ones C3 is trying to remove.** `%%bash` is item 4 — the model authoring its own shell tool. It is a hole *because* IPython supplies the magic. Build our own runtime and the hole does not need closing; it never opens.

### What IPython gave us, and pH's own answer

| IPython / ipykernel | `code-runtime-python` |
|---|---|
| top-level `await` (`autoawait`) | compile with `PyCF_ALLOW_TOP_LEVEL_AWAIT`, execute as an async function body against the persistent globals dict — dsh already does exactly this for its fresh-per-run case |
| persistent namespace | the child holds one `globals()` dict across `run` frames; identity is the agent id (C1) |
| `%%bash` | **`await tools.bash(...)`** — a governed binding. The magic *was* the bypass |
| `%cd`, `%env` | plain `os.chdir` / `os.environ` in the namespace, or bindings where policy should see them |
| `display_data` MIME bundles | a `display` frame on fd 3 → `present_result` render intents, typed by pH rather than parsed out of a Jupyter bundle |
| `execute_result` repr | `done.value` (already lossless JSON) plus a repr fallback through `log` |
| interrupt on the control channel | a `cancel` frame plus `SIGINT`; **no deadlock**, because fd 3 is not the channel the run occupies — the workaround prime-agent needs disappears with its cause |
| kernel spec, connection file, HMAC key, ZeroMQ | none — fd 3 is inherited at spawn; no port, no auth material, no discovery |
| `nest_asyncio` | **moot.** One loop in the child, program is an async body. §6 item 9 dissolves rather than being configured |
| tab completion, `?`/`??`, `%debug`, matplotlib inline | **dropped.** There is no human at this REPL |

### The protocol, as an extension of a known-good shape

dsh's fd-3 vocabulary is small and already hardened against a hostile child (`validateChildFrame` shape-validates and **rebuilds** every inbound frame): host→child `boot` / `run` / `reply`, child→host `boot-ack` / `call` / `log` / `done`. pH's version keeps that vocabulary and makes three additions, each required by a pH seam rather than by taste:

| Addition | Frames | Required by |
|---|---|---|
| Persistence | `boot` carries a namespace id; the child loops `run → … → done` instead of exiting after one run; new `shutdown` | C1 (`persistence: "namespace"`) |
| State in the log | child→host `snapshot`, host→child `restore` | D17 (`kernel/snapshot` events) |
| Rich results | child→host `display` | replaces `display_data`; feeds `present_result` |
| Cancellation | host→child `cancel` | turn abort, tool timeout |

`call` / `reply` are unchanged and are the whole of C2/C3: every binding the model invokes is one `call` frame, answered by one `reply`, having passed the full tool pipeline in between.

### The honest costs — this is not free

- **We are now writing and maintaining a code runtime.** The child-side runner is small (order 300–500 LOC: frame codec, the exec-as-async-body loop, resource limits, output caps, the snapshot hook), but the host side plus a real conformance suite is the larger half. dsh wrote the fresh-per-run version — in TypeScript, so it is a reference, not a dependency.
- **`prime-agent-runtime` no longer runs unmodified**, because `host_request` needs `ipykernel.Comm`. This is the argument that carried the old decision, and it is genuinely lost. But note what C2 already concluded: that package's comm path *is* the ungoverned route. We reimplement the **programming model** — `await rlm(...)`, `await agent_message.send(...)` — as bindings, which C2 required anyway. What we lose is not the model's ergonomics; it is the convenience of not writing a guest-side module.
- **Prime-agent's own test suite stops being a free acceptance gate.** pH needs its own conformance tests per binding and per frame type. §6.8's reuse table already replaced five of the bundled skills; the rest (`websearch`, `attach_image`, MCP integrations) are small and portable.
- **Cells written verbatim against prime-agent will not run**, and no compat shim can fully hide that. The `access` default (§6a) had already broken verbatim parity, so this is a widening of an accepted break rather than a new one.
- **IPython's debugging affordances go.** `%debug`, `pdb` post-mortem, introspection. Real, and mostly irrelevant to a model-driven REPL — but it will be missed the first time a cell misbehaves in a way the traceback does not explain.

### What it buys, beyond removing ZeroMQ

- **§6 item 4 (`%%bash`) is solved by construction**, not "stated but unresolved". Shell access is a binding or it is `subprocess.run()` — and the latter is item 3, not a separate hole.
- **§6 item 9 (`nest_asyncio`) dissolves**, along with its row config, its risk row and its Phase 3 criterion.
- **The control-channel reply workaround disappears with its cause.** No shell-channel deadlock means no "reply on control" special case, which was the single most fragile mechanism in the ported design.
- **fd 3 crosses a container boundary as an inherited descriptor**, so an operator who runs pH inside their own container needs nothing from pH: no published port, no shared network namespace, no auth material. This was flagged as an argument the old D19 had not weighed, and it is what makes §6a's "containers are the operator's layer" decision cost-free.
- **The kernel venv shrinks to almost nothing** — no `ipykernel`, no `jupyter_client`, no `nest_asyncio`. Startup is a plain Python subprocess, so the fork-server fast-start prime-agent needs may be unnecessary.
- **One protocol across every pH code runtime**, so `code-runtime-quickjs` (D16) and any future backend share a codec, a hostile-input validator, and a test suite.

## 6. What does **not** fit — the honest list

Ordered by how hard the mismatch is. Each item carries where the port plan handles it. **After §5's reversal, item 3 is the only one still *stated* rather than solved** — item 4 is closed by construction (no magics in pH's runtime) and item 9 dissolved with the ipykernel loop.

**1. `ipython` as the sole model-facing tool. — Incompatible; resolved by C1** *(plan §6.2, §6.3, §4.4)*.
Governance in both dsh and Deep Agents keys on tool boundaries. One tool = one boundary per cell. Not a bug in Prime Agent; a deliberate removal of `bash` and `edit`. pH keeps the one-callable ergonomics via Code Mode, where the callable is a *transport* whose bindings are individually governed.

**2. The `host.request` catalogue as an open RPC. — Incompatible; resolved by C2** *(plan §6.3; the compat shim keeps `prime-agent-runtime` working without keeping the bypass, and plan §12 Q8 now sets the shim's retirement at Phase 5)*.
It is a parallel capability channel that no seam observes, logged only as an `ignorable` digest. It also violates "everything is a plugin, every capability is a seam": each new host capability adds a request type instead of a provider.

**3. Raw filesystem and process side effects from kernel Python. — Irreducible by interception; bounded by containment. See §6a for the root cause and the two mechanisms** *(plan §6.2 "Trust and the enforcement boundary", §4.8, §11 risk row, §12 Q10–Q12)*.
**Root cause, stated in §6a:** this is the model *authoring its own file-write tool*, and a deny-list needs a registered name to match. Not a weakly-enforced control — an unavailable one.
`pathlib.Path("x").write_text(...)`, `subprocess.run(...)`, `os.remove(...)` inside a cell cannot be intercepted by a host-side waterfall. An `os.open`/`pathlib` audit hook in the bootstrap is **advisory** — model code can remove it. Two honest postures:
   - *Accept it* (Prime Agent's posture): the kernel runs with the user's permissions; bindings are the encouraged path, not the enforced one.
   - *Confine the process*: `bwrap`/Landlock around the kernel argv via `ctx.sandbox.confine()`, so the blast radius is bounded even though individual calls are not gated.
   *As folded:* the plan takes both — bindings are the path of least resistance (the SDK block advertises them, and `rlm-bindings` binds the name `edit` to the binding so ported cells hit the governed path by default), and `ctx.sandbox.confine()` on the kernel argv is named as the enforcement boundary, with approval on the transport until Phase 6 delivers a confining provider. The bootstrap audit hook is documented as telemetry, never as a control. Plan §12 Q10 asks how loudly to state this; the recommendation there is docs + first-run notice for v1, plus refusing `danger-full-access` without an explicit acknowledgement once the sandbox lands. **Do not claim per-call governance for a cell that can bypass it.**

   *Escalation ladder (§6a, plan §4.8) — and it is **two properties, not three points on one axis**:* **advisory** (bindings preferred, audit hook as telemetry; bounds nothing) → **worktree** (per-agent `git worktree`; `ctx.fs` root and `ctx.subprocess` cwd both resolve there, so it bounds tool-mediated and **relative-path** raw writes — but `open("/etc/passwd", "w")` never consults cwd, so it buys *collision isolation and revertibility*, **not** confinement) → **same-world sandbox** (`bwrap`/Landlock/Seatbelt on the runtime argv; the only tier that refuses an absolute-path raw write). **The ladder ends there**: containerization is the operator's layer, outside pH (plan §12 Q12, closed). None adds interception. **Describing `worktree` as a blast-radius boundary is the same category error this section warns operators about** — the plan's own tier table made it until §12 Q10 corrected it. Defaults are settled (plan §12 Q11, closed): `advisory` for root agents, `worktree` for every child, `access="read"` when a spawn does not say otherwise.

**4. `%%bash` as an IPython magic. — RESOLVED by §5: pH's runtime has no magics, so the hole never opens** *(plan §6.2, §6.3: shell access is the `tools.bash` binding)*.
In prime-agent it reaches the shell without touching the bash tool — no sandbox provider, no permission evaluation, no timeout policy, no `tool/call` record — and under a *ported* Jupyter kernel the best available answer was to steer the prompt and redefine the magic, neither of which stops a determined cell. Building pH's own `code-runtime-python` removes the mechanism rather than mitigating it: there are no magics to redefine, so the only shell routes are `await tools.bash(...)` (governed) and `subprocess.run(...)` (which is item 3, not a second hole). **This is the clearest single argument for §5's decision** — the feature we would have been paying an IPython dependency for is one C3 exists to remove.

**5. `harness_state.json` as mutable state written by both the kernel and the host. — RESOLVED: the second writer is gone, so the file becomes a projection** *(plan D14 rewritten, §6.6, §12 Q5 **closed**)*.
Prime Agent has two writers — the host applying `/refine`, and the kernel calling `rlm.harness.*` — reconciled by an mtime-guarded reload. That is a race workaround, not a resolution, and it violates "model-visible means logged" because the file feeds a prompt section.

**Why the second writer existed:** prime-agent's host is TypeScript, so guest-side code had no way to reach host state except through the file. **D19 removes that runtime and C2 routes model-side harness access through a binding**, leaving exactly one writer — the host. The mtime-guarded reload is therefore *deleted rather than fixed*, and the conflict rule Q5 asked for never has to be chosen.

What pH does instead: **local** state folds this session's `harness/refined` / `harness/rolled-back` events; **global** state folds its own append-only log at `~/.ph/harness/events.jsonl` — a log rather than a file so "state is a fold over an append-only log" holds at *both* scopes, since a global file beside local events would restore two authorities one level up. `harness_state.json` is written after applying, for humans and `ph trace`, and **nothing reads it back to decide anything**; a runtime invariant asserts the projection equals the fold. Two consequences: the fold must be incrementally cached per `(scope, last_seq)` like `derive_messages()`, or prompt assembly re-folds every turn; and `fork(source, boundary)` now inherits the harness *as of the boundary*, which is correct and which a file could not express.

**6. `kernel-state.dill` as a side file. — Resolved by D17** *(plan §6.6; C1 makes it an obligation the seam asserts at registration rather than a convention a provider may forget)*.
dsh's stated reason for refusing a persistent kernel. Also silently wrong under `ctx.sessions.fork(source, boundary)`: a fork would receive the parent's *current* namespace, not the namespace as of the boundary.

**7. `agent-traces.ts` uploading sessions to Prime Intellect's API. — Does not fit; drop** *(plan §8 tracing row; verdict **D** in §7 below)*.
`${baseUrl}/api/v1/agent-traces/sessions/<id>`, gated on a global sharing opt-in. pH's equivalent is `ctx.session_telemetry` through the `session-telemetry/record` redaction waterfall to local sinks (JSONL mirror, OTLP). Keep `/traces` as a local export command if it is wanted; do not port the uploader.

**8. No approval / permission / HITL subsystem. — Nothing to port; build from dsh + Deep Agents** *(plan §4.7, §7.6; Phase 4)*.
`ctx.approval`, `permission-presets`, `user-approval`, `fs-sandbox`, `sandbox-policy` on the dsh side; `HumanInTheLoopMiddleware` / `interrupt_on` and `permissions.py` on the Deep Agents side. This is additive work, not reconciliation — and it is only *reachable* once C1–C3 create the boundaries it hooks.

**9. `nest_asyncio.apply()` in the kernel bootstrap. — DISSOLVED by §5. Retained below as the reasoning that led there** *(plan §6.2; the row config, the §11 risk row and the Phase 3 criterion (h) it created are all withdrawn)*.

*Status (v0.5).* pH's own runtime executes the program as an async function body on the child's single event loop — there is no ipykernel loop to re-enter, so `nest_asyncio` has no reason to exist and neither does the `(a)+(c)` configuration. **What survives is (c)'s insight, now structural rather than optional:** binding dispatch resolves on the host side of fd 3, so harness code was never on the guest's loop to begin with. The analysis below stands as the record of how the question was worked out, and of why versioned checkpoints were not the answer.

*Correction (v0.2).* v0.1 said the RLM design depends on re-entrancy. It does not. Checked at source:

- `prime-agent-runtime` uses `asyncio.get_running_loop()` + `loop.call_soon_threadsafe()` and **never re-enters the loop**; `host_request` is a plain coroutine awaiting a future the comm callback resolves. The single `asyncio.run()` in the package (`rlm/skill.py:37`) is a **console-script entry point** that runs in its own process, not in a kernel.
- IPython ≥ 7 executes top-level `await` natively (`autoawait` / `run_cell_async`), so `await rlm(...)` in a cell needs nothing extra.
- Prime Agent's own bootstrap wraps it in `try: … except Exception: pass` — it is best-effort cover, not a dependency.

What it actually defends against is **model-written** `asyncio.run(...)` or `loop.run_until_complete(...)` inside a cell (a very common LLM habit) and third-party libraries doing sync-over-async. That is a real ergonomics need, but a much narrower claim than "the architecture requires it."

**Why versioned checkpoints do not fix this** (asked directly, recorded because it is easy to conflate): re-entrancy is a *concurrency-correctness* problem — callbacks running out of order, `current_task()` returning the wrong task, deadlocks in primitives that assume non-reentrant execution, exception context leaking across nesting levels. LangGraph's counted per-super-step channel versions, and D17's `kernel/snapshot` chain, are *durability and replay* mechanisms: they answer "we lost state on a crash", not "the scheduler ran something twice". They cannot prevent a re-entrancy bug. What they do is **bound its blast radius** — a wedged or corrupted kernel is killed, restarted, and replayed from the last snapshot with the `<ipython_kernel_reset>` notice, and `checkpoint-policy` guarantees a flush barrier before the model request that preceded it. That containment is already specified (D17, §6.6 in the port plan); it is the right mitigation and the wrong *fix*.

**What Deep Agents does about this exact problem is structural, not a checkpoint.** `langchain-quickjs` runs the guest VM on its own OS thread (`ThreadWorker`) and marshals guest→host calls back with `asyncio.run_coroutine_threadsafe(_call(), outer_loop)`. The host loop is **never re-entered**, so `nest_asyncio` has no reason to exist there. That fix is available because the guest is a separate VM with its own execution stack — the option CPython-in-a-kernel does not have, since guest and host share one loop.

**Three options for pH, now that C2 exists:**

| | Behaviour | Cost |
|---|---|---|
| (a) **Do not apply it**; rely on IPython autoawait | model code calling `asyncio.run()` gets `RuntimeError: asyncio.run() cannot be called from a running event loop` — a clear, self-correctable error the generated SDK block can pre-empt by showing `await` usage | some plausible model code fails on first attempt |
| (b) Apply it (prime-agent parity) | maximum compatibility with cells written against prime-agent | inherits the re-entrancy risk for *all* code, harness included |
| (c) **Serve binding dispatch off the kernel loop** | because C2 routes every host call through the binding bridge, pH controls that side: dispatch from a dedicated executor rather than the kernel's loop, so re-entrancy risk is scoped to model code alone and never touches harness code — the Deep Agents ThreadWorker insight applied to the half we own | one more thread-hop per binding call |

**Decided (v0.2, now superseded): (a) + (c)** — do not monkeypatch by default, keep harness dispatch off the shared loop, expose `nest_asyncio: true` as row config. **Superseded by §5 (v0.5):** with pH's own runtime there is no shared loop and no monkeypatch to configure. A model cell calling `asyncio.run()` still gets Python's native `RuntimeError`, and the generated SDK block still pre-empts it by showing `await` usage — so (a)'s ergonomics answer survives; only its *mechanism* was tied to IPython.

> **Landed in** port plan §6.2 — as the *absence* of the problem. The `nest_asyncio` row config, the "Event-loop re-entrancy in the kernel" risk row, Phase 3 criterion (h) and the Appendix D entry are withdrawn by D19's reversal; the SDK block's `await`-usage guidance stays.

**10. Deep Agents' `AsyncSubAgentMiddleware`. — Does not fit the no-HTTP-server constraint; do not adopt** *(plan §8 "Non-blocking delegation" row, D11)*.
Covered in §1. pH's `ctx.subagents` + `anyio` provides the same admission-handle semantics in-process.

**11. Prime Agent's extension API as a fixed hook list. — Fits as a source of hook *names*, not as an architecture** *(plan §7 inventory, verdict **A** "names only")*.
~30 named events with veto-capable results is a good inventory to check pH's waterfall coverage against, but it is a closed enum registered on one object, not a plugin tree with scoped shadowing and reversible effects. Port the coverage; keep pH's dispatch model.

Everything else maps cleanly — see §7.

---

## 6a. The deviation behind items 3 and 4: model-authored tools vs. configuration-registered tools

Items 3 and 4 are not two independent holes. They are **one deviation with two symptoms**, and naming it correctly changes what "fixing" means.

`pathlib.Path.write_text()` in a cell *is* an unregistered file-write tool. `%%bash` *is* an unregistered shell tool. Prime Agent's RLM does not merely *permit* the model to author tools — authoring them is the design. dsh holds the opposite theory:

| | **deepseek-harness** | **Prime Agent RLM** |
|---|---|---|
| Who decides a tool exists | the **deployment** — a YAML row → a plugin module → `ctx.tools.register(definition)` | the **model**, at inference time, by writing a function in a cell |
| Schema | mandatory (pydantic → JSON Schema) plus a mandatory canonical `output` declaration | none |
| Scope | global or agent-scope, with shadowing and `ctx.tools.restrict()` masks | a name in the kernel namespace |
| Presentation | `present_call` / `present_result` render intents (generic/terminal/diff/search/read/web cards) | whatever it prints to stdout |
| Governance | `tools/pre-execute` → guards → approval → `tools/execute` → `tools/post-execute` | none |
| Lifetime | a reversible effect, disposed with its plugin scope | the namespace, until kernel reset |
| Durable record | `tool/call` + `tool/result` events | a substring of one program |
| Restrictable by config | yes | **no — there is no name to deny** |

The last row is the load-bearing one. A permission row can deny `edit`; it cannot deny `open(path, "w")`, because a deny-list needs a registered name to match. **Interception-based governance is structurally unavailable for the authored surface** — not weakly enforced, unavailable.

### Consequence 1 — C3 provides alternatives; it cannot make authoring illegal

This is why items 3 and 4 stay on the "does not fit" list after the fold. C3's bindings make the governed path the *convenient* path (the SDK block advertises it, `rlm-bindings` binds the name `edit` to the binding, the prompt steers to it). None of that stops a model from writing `open()`. Any document or config that implies otherwise is wrong, and a deployment that writes a `deny` permission row believing it is enforced has been misled.

### Consequence 1b — "skill" is not a middle ground between the two theories; it is on a different axis entirely

The tempting reconciliation is *let the model promote useful authored code into something registered*. **That is a category error, and this document made it before correcting it** (plan §12 Q13). Prime Agent's refinement taxonomy is `HarnessKind = Literal["prompt", "memory", "skill", "subagent"]`, and its own guidance reads *"repeated delegation roles should become subagent specs, repeated procedures should become skills, durable facts/preferences should become memories, narrow behavioral policies should become prompt addendums"* — **nothing becomes a tool**. There is no rung above `skill`, because skills are not immature tools.

There are **two layers, not one ladder**: dsh plugins change the **capability layer** (*what actions exist*, developer authority, code, schema, pipeline); `/refine` changes the **knowledge layer** (*how this agent should work*, inference-time authority, prose plus a pointer, no code). A `skill` entry is a note saying "the procedure for X is: call `foo(...)` with these arguments" — knowledge **about** a capability. The layers touch at one point, and it is a **pointer, not a promotion**.

Three things wear the word "skill", and only the middle one is knowledge-layer. Verified at source:

| | **Tool row** (dsh) | **Installed Python skill** (prime-agent) | **Harness skill entry** (`/refine`) |
|---|---|---|---|
| What it is | a registered callable | a directory: `SKILL.md` + importable package, `uv pip install --editable` into the kernel venv, pre-imported by the bootstrap | a JSON record in `harness_state.json`: prose + `reference {type:"python", import, callable, call_pattern}` + `arguments` |
| Who creates it | the deployment (YAML row) | the deployment or user (install) | **the model** |
| Contains executable code | yes | yes | **no — it points at code that must already exist** |
| Schema | mandatory, validated | prose in `SKILL.md` | `arguments`, **unvalidated** |
| Where it surfaces | tool schemas / the generated SDK block | prompt catalog **and** the kernel namespace | the prompt's "Continual Harness State" |
| Lifetime | its plugin scope | until uninstalled | persistent + versioned; **`scope: global` = every future session, all projects** |
| Undo | unload the row | uninstall | `/refine --rollback <id>` |

So a `/refine`-authored skill is **not** "a temporary tool the agent made." It is durable, and it is not a tool: `refinement.ts:692-711` checks only that `import` and `callable` are non-empty *strings*. Nothing is compiled, installed or registered.

That is exactly why the line sits where it does (plan §12 Q13, closed): **a tool row is a deployment claim about capability** — "this agent may do X", with a schema the model is told to trust and a pipeline enforcing policy on it — whereas **a skill entry is a claim about procedure** — "when you need X, this is how" — which grants nothing. A wrong skill entry wastes tokens and rolls back; a wrong tool row is a standing capability.

**The invariant that keeps the layers apart:** *the knowledge layer may only reference capability that already exists.* `/refine` cannot conjure capability, so an entry pointing at an unresolvable import is the knowledge layer attempting to — which is why gap 1 below is enforcement rather than hygiene, and why pH does **not** ship a `ph tool scaffold` promotion command. If a capability is missing a developer writes a plugin; if a procedure keeps being re-derived `/refine` records it. Different problems, different fixes, no path between them.

Three gaps this exposed, all specified in plan §6.6:

1. **The reference is never verified upstream** — `/refine` can write a confident prompt section instructing the model to call something that does not exist. pH resolves it through `ctx.code_runtime` at apply time and rejects the edit, **enforcing the layer invariant**: the model must *ask* for a capability rather than assert one.
2. **Prime-agent's refinement prompt tells the model to write raw-namespace call forms** (*"Include the RLM-native call form `await <skill_import>(...)`"*) — which under C1–C3 is the **ungoverned** path. Left alone, `/refine` would have the model author prompt text steering itself off the bindings. pH renders `call_pattern` as `await tools.<name>(...)` where a binding exists.
3. **`scope: global` is the real hazard, not temporariness** — a global entry reaches every future session including other projects, so pH routes it through `ctx.approval` rather than merely requiring it be asked for.

### Consequence 2 — the boundary moves from per-call interception to per-agent containment

If the authored surface cannot be intercepted, the enforceable question changes from *"may this call proceed?"* to *"what can this agent reach at all?"* Two mechanisms answer it, and they compose:

#### Containers are the operator's layer, not pH's — decided (plan §12 Q12, closed)

The instinct to reach for `ctx.sandbox` here is wrong, and dsh says so explicitly (`packages/sandbox/sandbox/README.md`):

> **Same-world confinement only.** A backend shares the host's filesystem and kernel (`bwrap`, Landlock, Seatbelt) … **Containers, microVMs, and remote executors are NOT backends of this seam — they replace the Service Providers for whole capability seams (`ctx.shell`, `ctx.fs`) as environment-coherent groups.**

That sentence is the whole argument, and it points further than "use a different seam": containerization is an **environment** decision, not a harness feature. **pH's ladder therefore stops at `sandbox`, and the operator runs pH inside a container they build and manage.**

Taking containers *into* pH would have cost a container-runtime dependency and capability probe; an image-derivation problem, since the `ipython` contract demands *"the target project's own environment for project imports, tests, scripts, CLIs"* and a generic image cannot supply it; a coherent four-provider group (`fs`/`subprocess`/`shell`/`code_runtime` swapped together, because a half-containerized environment is incoherent — the harness would read files the child cannot write); and a materially degraded macOS path where containers run in a VM. None of that buys anything an operator-managed container does not already give.

The layering that replaces it is simpler, and each layer is owned by whoever can actually enforce it:

| Layer | Owner | Bounds |
|---|---|---|
| container / VM / remote host | **the operator, independently of pH** | everything pH can reach |
| `ctx.sandbox.confine()` on the runtime argv | pH (`sandbox` tier) | the sandbox mode's writable roots |
| `ctx.workspace` per-agent worktree | pH (`worktree` tier) | that agent's own tree |
| bindings through the tool pipeline | pH (C2/C3) | per call, for the governed surface |

Running pH inside an operator's container needs **nothing from pH**, and D19 is why: the runtime's only channel is an inherited descriptor (fd 3), so no port is published, no network namespace shared, no auth material exists; `ctx.credentials` passes `CredentialRef` and never values, so secrets stay outside by construction; and both pH tiers keep working unchanged inside. What pH owes is **documentation, not machinery** — a "running pH in a container" page (Phase 6).

**One consequence to state plainly:** `readonly-scratch` now has exactly one enforcing implementation — `ctx.sandbox` `workspace-write` rooted at `scratch`. Where the backend is weak (macOS Seatbelt) or absent, `access="read"` degrades to `worktree-ephemeral`, and `ph doctor` must report the effective guarantee rather than the requested one.

#### Git worktrees — and dsh already named this gap for itself

dsh's own multi-agent package lists it as a known limitation (`packages/experimental/agent-team/README.md`):

> **One process and one shared checkout** — members share cwd and observe edits immediately; **this package provides no worktree**, remote member, merge, or filesystem lock.
> **Advisory write scopes** — Bash, formatters, code generators, and direct external writers **can bypass filesystem version checks**; Leads must coordinate ownership and review the final diff.

That is the same two problems, reached independently: no per-agent checkout, and advisory-only write scoping that shell escapes. So per-agent git workspaces are not a foreign graft onto dsh — they fill a gap dsh documented against its own feature.

Shape:

- **One `git worktree` per agent scope** (not a clone — shares the object store, so creation is cheap), on a branch `ph/<session-id>/<agent-id>`, created when the agent's scope is created and torn down with it.
- **`ctx.fs` root and `ctx.subprocess` cwd both point at it**, so bindings *and* authored code land in the same place. This is what makes it bound item 3 rather than merely observe it.
- **`workspace/created` / `workspace/disposed` session events**, so the workspace is replayable and a `fork(source, boundary)` can reconstruct which tree a turn ran against.
- **Disposal policy**: keep if the worktree has changes (the user inspects and merges); remove if unchanged.
- **Children merge back through git, not through a shared tree.** This answers a governance question the roster does not: an RLM child cannot silently corrupt its parent's working tree, and the parent reviews a diff instead of trusting a sibling's writes. It also removes the fan-out hazard in §6.4 — eight children writing the same repo concurrently.

#### Read-only is a tier claim, not a workspace kind

A research child — "read this codebase and report" — should not hold a writable checkout. But **"read-only" is an enforcement claim, and the `worktree` tier cannot make one**: a worktree is a full checkout the child can write. So `access="read"` has to resolve differently per tier, and the seam must *report* which guarantee was obtained rather than let a caller assume:

| Tier | `access="read"` yields | Honest label |
|---|---|---|
| `advisory` | session cwd + scratch; deny-write rows reach bindings only | **not read-only** |
| `worktree` | full checkout, writable, **unconditionally discarded, never merged** | **isolated, not read-only** |
| `sandbox` | `ctx.sandbox` `workspace-write` rooted at `scratch` — repo readable, only scratch writable, enforced by `bwrap`/Landlock/Seatbelt | **read-only, enforced** (the only enforcing tier; degrades to `worktree-ephemeral` where the backend is weak or absent) |

`SandboxExecutionPolicy` needs no extension for this — dsh defines it as *"the complete per-call mode + workspace root"*, so pointing `workspaceRoot` at the scratch dir instead of the repo **is** "read the repo, write only here". Policy rides the call, so a research child and an implementer child can be confined differently at the same instant.

**Defaults — settled (port plan §12 Q11, closed).** `advisory` in the `rlm` profile (root agents edit the user's cwd, as today), **`worktree` for every RLM child**, `worktree` in `rlm-stable`. A child's kind follows the `access` it was spawned with, and **`access` defaults to `"read"`** — the cautious default is the recoverable one: a research child that turns out to need writes is re-spawned at the cost of one turn, while a writing child that should not have written costs a review of every diff it produced. `ph doctor` reports the *effective* tier, plus each agent's `kind` and `repo_writable`, so "is this research child actually prevented from writing?" is answerable without reading config.

Two consequences of that default, both specified in plan §6.3 rather than left implicit:

- **`access` is a real extension to prime-agent's `rlm()` contract**, which validates kwargs strictly and fails loudly on unknown ones (§6.0). Deliberate, and documented in the RLM prompt along with the profile's default.
- **The default is uniform across front-ends, which costs prime-agent parity.** `rlm-kernel-compat` applies the same `"read"` default as the SDK path — a per-path default (parity over the comm channel, cautious over the SDK) would make identical-looking spawns behave differently depending on which front-end issued them, the trap C2 exists to remove. So unmodified `prime-agent-runtime` cells that spawn *writing* children change behaviour under pH. One expected diff, in one place, surfaced by Phase 3's trajectory-fixture replay as a fixture update rather than a regression.

Two practical consequences the plan carries (§4.8):

- **Every workspace has an always-writable `scratch`**, on every kind and tier. A research child still needs somewhere for notes, extracted data, and a reproduction script — read-only without scratch is safe and useless.
- **Read-only repo vs. running the project's tests is a real tension.** `pytest` writes `.pytest_cache/` and `__pycache__/` into the tree. The provider redirects (`TMPDIR`, `PYTHONPYCACHEPREFIX`, `PYTEST_ADDOPTS=--basetemp=…`, `PIP_CACHE_DIR`, `UV_CACHE_DIR`) into `scratch`, documented as best-effort. A toolchain that insists on writing beside sources is a signal to spawn with `access="write"`, not to weaken the tier.

### What this does and does not buy

**Buys:** item 3 becomes *bounded* instead of *unbounded* (item 4 is gone entirely — see §5). A raw `write_text` still bypasses the binding, but it can only reach the agent's own worktree, or the sandbox policy's writable roots. Blast radius becomes a deployment decision — and an operator who wants a harder outer bound puts pH in a container, which pH neither needs to know about nor manage.

**Does not buy:** per-call governance, on any tier. A cell that writes 40 files inside its worktree still produces no `fs/write-intent` and no `tool/code-dispatch` records for those writes. **Containment is not interception, and the docs must not blur them.** The worktree tier also does nothing for a session whose cwd is not a git repository, and nothing for writes *outside* the worktree — that is what the `sandbox` tier bounds, and beyond it the operator's container.

---

## 7. Full feature inventory

Verdict key: **A** = adopt as-is onto an existing seam · **R** = re-express through the tool pipeline (C1–C3) · **D** = drop · **N** = new work in pH.

The **R** rows are what C1–C3 changed; they are specified in port plan §6.1–§6.3 and §6.8. The **N** rows are additive work the plan schedules in Phase 4 (approval/permission/HITL, todo) and Phase 6 (kernel sandbox) — they are only *reachable* once the R rows create the boundaries they hook.

| Prime Agent feature | Source | pH home | Deep Agents contribution | Verdict |
|---|---|---|---|---|
| `ipython` tool | `tools/ipython.ts` | Code Mode `run_code` over `ctx.code_runtime` | PTC binding style (`_ptc.py`) | **R** |
| Kernel manager (ZMQ) | `kernel/*.ts` | **`code-runtime-python`: pH's own CPython subprocess + fd-3 protocol** (§5) | dsh's `code-runtime-python` protocol shape as reference | **N** |
| Kernel venv / bootstrap | `kernel/bootstrap.ts` | provider config + `uv`-built venv, **minus `ipykernel`/`jupyter_client`/`nest_asyncio`** | — | **R** |
| Fork-server fast start | `kernel/fork-server.ts` | likely unnecessary — a plain subprocess starts fast (§5) | — | **D** (revisit if measured) |
| Kernel state snapshot | `kernel/state-snapshot.ts` | `kernel/snapshot` **events** (D17) | `_snapshot.py` patch chain + HMAC | **R** |
| `host.request` bridge | `rlm-runtime.ts`, `rlm/__init__.py` | `call`/`reply` frames → `CodeBindingNamespace` (C2); pH-authored guest module replaces `rlm/__init__.py` | — | **R** |
| `rlm()` spawn + handle | `agent-session.ts` | `ctx.subagents` provider `rlm-child` | — | **A** |
| `agent_message.*` | `agent-messages.ts` | binding + `ctx.tools.guard` for family boundary | — | **R** |
| `agent_observe.*` | `agent-observe.ts` | binding; results offloadable | `tools/post-execute` offload | **R** |
| Subagent ledger / roster | `rlm-*.jsonl` | folded from `rlm/child-*` session events | — | **A** |
| Continual Harness / `/refine` | `refinement/`, `harness.py` | `rlm-harness` + `harness/refined` events | `MemoryMiddleware` ordering vs prompt cache | **A** |
| Python skills (`SKILL.md` + package) | `skills.ts`, `skills/` | `ctx.skills` + kernel-venv editable installs | `SkillsMiddleware` progressive disclosure | **A** |
| MCP integrations | `mcp/`, `mcp.py` | `ctx.tools` MCP provider; **stdio preferred** | — | **A** |
| Goals + budgets | `goals.ts` | `ctx.goals` + `goal-round-driver` | — | **A** |
| Heartbeats / cron | `cron-jobs.ts` | `ctx.schedule` + `ctx.jobs` | — | **A** |
| `/autonomous` budgets | `autonomous.ts` | budget policy on `agent/pre-step` | `ModelCallLimit` / `ToolCallLimit` | **A** |
| Compaction | `compaction/` | `ctx.compaction` + surface `replace` | `SummarizationMiddleware` fractions | **A** |
| Context tree / `/context` | `context-tree.ts` | `ctx.token_meter` per-node + child links | — | **A** |
| Side question (`/side`) | `side-question.ts` | `ctx.subagents` one-shot, result injected | — | **A** |
| Steering / follow-up lanes | `session-action-store.ts` | agent inbox + `agent/pre-step` | — | **A** |
| Prompt admission | `prompt-admission.ts` | turn/inbox claim path | — | **A** |
| Session JSONL / fork / tree | `session-manager.ts` | `SessionEvent` log + `fork(source, boundary)` | — | **A** |
| Session leases | `session-lease.ts` | `filelock` on canonical path | — | **A** |
| Daemon + workers | `modes/daemon/` | `ph-rlm-daemon`, unix socket JSONL | — | **A** (Phase 5) |
| RPC / ACP / JSON modes | `modes/` | `--mode` renderers over `session/event` | — | **A** |
| Extensions API | `extensions/types.ts` | coverage checklist for pH waterfalls | Deep Agents middleware stack | **A** (names only) |
| Output truncation (2000 lines / 50 KB) | `truncate.ts` | `tools/post-execute` caps | `TOO_LARGE_TOOL_MSG` + spill | **A** |
| Orphan-process journal | `orphan-process-journal.ts` | `ctx.subprocess` reaping | — | **A** |
| Model registry / resolver / thinking | `model-*.ts` | `ctx.llm` adapter seam | — | **A** |
| Trace upload | `agent-traces.ts` | `ctx.session_telemetry` + OTel | `TracePolicy` redaction | **D** |
| Approval / permission / HITL | — (absent) | `ctx.approval`, `permission-presets`, `fs-sandbox` | `HumanInTheLoopMiddleware`, `permissions.py` | **N** |
| Todo planning | — (absent) | `tool-todo` + `todo/write` | `TodoListMiddleware` prompts | **N** |
| Kernel sandbox (same-world) | — (absent, explicitly) | `ctx.sandbox` bwrap/Landlock provider | — | **N** |
| Per-agent git workspace | — (absent; dsh's `agent-team` names the gap) | `ctx.workspace` seam + `workspace-git-worktree`; `access: write\|read` → `worktree` / `worktree-ephemeral` / `readonly-scratch`; always-writable `scratch`; `workspace/*` events | — | **N** |
| Container tier | — (absent) | **out of scope — the operator's layer** (plan §12 Q12, closed); pH ships a "running pH in a container" doc, no machinery | `langchain_daytona` / `langchain_modal` are the same posture: an environment someone else manages | **D** |
| Capability/knowledge layer boundary | Continual Harness `skill` entries (`reference.type == "python"`) | **two layers, no ladder** (§12 Q13): plugins change capability, `/refine` changes knowledge, and a `skill` entry may only *reference* capability that already exists — resolved at apply time, `call_pattern` rendered as a binding, `scope: global` approval-gated. **No promotion command** | `SkillsMiddleware`'s source layering (base→user→project→team) as a knowledge-scoping model | **N** |

---

## 8. What the fold changed in the port plan (v0.2 → v0.3)

All five consequences this document predicted are now specified. Recorded here so the two documents can be diffed rather than re-derived.

| # | Predicted | As folded (port plan v0.3) |
|---|---|---|
| 1 | §6.2/§6.3 change shape; D18 becomes load-bearing | Done. §6.1 replaces three rows (`tool-ipython`, `rlm-host-bridge`, `rlm-tool-bindings` → `rlm-presentation`, `rlm-bindings`, `rlm-kernel-compat`); §6.3 rewritten around the four binding namespaces; D18 restated as C1–C3 with the v0.2 "bindings *beside* the bridge" reading recorded as the rejected alternative |
| 2 | The seam gains one field and one obligation | Done, and slightly larger than predicted: **two** fields (`namespace` on `CodeRunRequest`, `persistence` on the provider), with the `kernel/snapshot` obligation **asserted at registration** and checked by a runtime invariant. `code_runtime` is promoted into the §4.7 core seam list; §4.4 gains the presentation-mode and dispatch-bridge paragraphs |
| 3 | Open question 8 gets sharper; skills split | Done. Q8 is now *when to retire the compat shim* (recommendation: Phase 5), not *whether to stay compatible* — the shim closes the ungoverned path without a fork. §6.8 carries the per-skill table: keep `agent_message`, `agent_observe`, `attach_image`, `websearch`, MCP; replace `edit`, `compact`, `goal`, `refine`, `rlm_heartbeat` |
| 4 | Phase 3 exit criteria gain a governance test | Done, and expanded from one test to seven — (a)–(g) in §10, including that a `deny` row must make a `tools.edit` cell *fail* rather than write, that three binding calls tick `ToolCallLimit` three times, and that unmodified `rlm/__init__.py` over the comm target produces the **same** `tool/code-dispatch` record as the SDK path |
| 5 | A documented non-goal | Done in three places: §6.2 "Trust and the enforcement boundary", a §11 risk row, and §12 Q10 on how loudly to state it |

**Two things the fold added that this document did not anticipate:**

- **A per-cell dispatch budget.** Governance is per call; attention is per turn. One approved cell can issue unbounded *governed* calls, so `max_dispatches_per_run` (256, following Deep Agents' `_DEFAULT_MAX_PTC_CALLS`) and `max_subagent_spawns_per_run` (32, following `_MAX_TASK_CALLS_PER_THREAD`) are enforced in the bridge and reported as a `CodeRunFailure`.
- **An ergonomics risk.** Code Mode's generated `tools:sdk` block displaces prompt text prime-agent's RLM doctrine relied on, and a model tuned on prime-agent may write `call_skill(...)` wrappers or attempt native calls. Mitigation: the `tools:code-only` rule plus the `UNKNOWN_TOOL` denial names the route back, prime-agent's anti-wrapper line stays in `rlm-prompt`, and Phase 3 replays prime-agent's trajectory fixtures under the new surface, diffing turn counts and tool-call shapes, before the profile is declared done.

**Still open** (port plan §12): **none.** Every question in the section is decided.

**Closed since v1.4:** **Q2** — camelCase at every JSON boundary (`json`/RPC/ACP output, the session JSONL, the fd-3 runtime frames), snake_case in Python, via a shared pydantic `WireModel` base carrying `alias_generator=to_camel` + `populate_by_name=True`. The decision reversed this plan's earlier recommendation and turned out to be **more** faithful to dsh, not less: §1.1 records dsh's envelope as already camelCase (`sourceEventSeqs`, `surfaceOp`), so D2's "byte-compatible in spirit" becomes byte-compatible in fact and dsh tooling reads a pH log directly — no `--format pi` renderer needed. Two details recorded so they are not rediscovered: **declare aliases, never derive them** (pydantic maps by the alias fixed at class definition; `to_camel`→`to_snake` happens to round-trip for every field in the plan against 2.13.4, but relying on that would be fragile at acronyms and digits), and **tool parameter names are exempt** — they are Python identifiers in the generated Code Mode SDK, so `await tools.edit(old_str=…)` must not become `oldStr`, which would also diverge from prime-agent's own tool signatures and silently camelize a plugin author's pydantic fields in the schema the model sees.

**Closed since v1.3:** **Q1** — the name is `ph` for the project, the command, the distribution prefix (`ph-core`/`ph-app`/`ph-rlm`/`ph-stabilize`) and the `ph.plugins` entry-point group, fixed now because that group becomes a compatibility surface the moment a third party ships a plugin. Prime-agent path interop is not pursued: after D19 there is no shared runtime venv and no `prime-agent-runtime` to find, so `ph session import` covers reading a prime-agent JSONL instead.

Paths are **hybrid**, because a single dotdir had accumulated four lifecycles: `$PH_HOME` (`~/.ph`) for state and config, `$PH_CACHE` for the rebuildable runtime venv, and `$PH_RUNTIME` for the daemon socket and the orphan journal. **`$PH_RUNTIME` resolves in a strict order, and only the last tier is pH's problem:**

| Order | Path | Properties | Check pH performs |
|---|---|---|---|
| 1 | **`$XDG_RUNTIME_DIR/ph`** (typically `/run/user/$UID`) | tmpfs, mode `0700`, owned by the user, removed at session end — all provided by `logind` | **none** |
| 2 | per-user `$TMPDIR/ph` (macOS `/var/folders/…`) | already per-user and `0700` | ownership assertion |
| 3 | `/tmp/ph-$UID` | predictable path in a **world-writable** directory | directory, current uid, mode `0700`, not a symlink — refuse to start otherwise |

The security paragraph applies to **tier 3 alone**. Tier 1 is the design, and pH checks nothing there because the OS already owns the guarantees — the same posture `tmux` and `ssh-agent` take, except pH prefers the OS-owned directory whenever one exists rather than defaulting to `/tmp`. `$PH_RUNTIME` also takes a **subdirectory** rather than a bare `ph.sock`, since `$XDG_RUNTIME_DIR` is shared across the user's applications and pH stores `processes.jsonl` alongside the socket.

Why the split is load-bearing rather than tidy: a `~/.ph` inside Dropbox or iCloud would sync a socket and another machine's PIDs, which makes plan §4.9's orphan journal *wrong* rather than merely useless — and `$PH_RUNTIME` being wiped on reboot is **correct**, since PIDs do not survive one and become dangerous once reused.

**One interaction the resolution order exposes**, verified against a live systemd host (`Linger=no`, `KillUserProcesses=no` — both defaults): `logind` removes `/run/user/$UID` when the user's **last session** ends, while `KillUserProcesses=no` lets processes survive it. So a pH daemon outlives a full logout *as a process* but loses its socket path, and clients cannot reconnect. Closing a terminal is not a session end, so Phase 5's actual goal — "agents that keep running when the terminal closes" — is unaffected; but *surviving logout* requires `loginctl enable-linger $USER`. `ph doctor` reports the resolved `$PH_RUNTIME`, whether it is XDG-derived, and whether lingering is enabled, naming the command when a daemon is configured without it (plan §6.7, Phase 5).

**Closed since v1.2:** **Q4** — `rlm-context-loader` stays **off by default**, but for a changed reason and in a changed shape. C3 gave the model governed `tools.read`/`grep`/`glob`, so pre-loading a corpus is no longer a headline feature but a specialist tool for **non-file corpora** and repeated queries over one fixed corpus. Access is a **binding** (`await tools.context_search(...)`), not a bare namespace variable — a variable produces no `call` frame, hence no per-query provenance and no `tools/post-execute` offload, which is §6a's pattern one level up: bulk data reaching the model through a channel no seam observes. And the corpus is recorded as a new **`recipe`** snapshot kind (D17) — `{loader, sources, digest}` — so an over-cap corpus rehydrates on restart instead of being silently dropped and leaving the model an undefined name.

Q4 also forced apart three properties that had been travelling together, now tabulated in plan §8: **model-visible means logged** (invariant 3 — satisfied by *every* design here, including plain raw-Python reads, since cell stdout *is* the tool result; **prime-agent parity does not violate dsh's logging principle**), **provenance** (which file, which query — supplied by bindings, absent from raw reads, and a documented non-goal per §6a), and **reconstructability** (broken only by *unrecorded large state*, which `recipe` fixes). Two corollaries: recording the code is **not** recording the result — `tool/call` already logs every program losslessly, but re-running one depends on the filesystem, network and clock, which is exactly why D17 snapshots state instead of replaying cells; and dsh accepts the general limitation for its own Code Mode (*"the canonical typed values cannot be reconstructed from session replay"*), so pH's addition is only to stop it silently swallowing harness-created bulk state.

**Closed since v1.1:** **Q7** — one `anyio` task per root, not a worker process per root, with the daemon protocol addressing a worker by id so process-per-root stays a provider swap. D19 had already moved the crash-prone code out (every agent has a runtime subprocess, so the risky children exist either way), Q9 made a crash cheap to recover from, and in-process leases are simpler than `filelock`. The non-guarantees are stated rather than implied — no per-root memory cap, no rolling restart, no crash containment between roots — with the multi-tenancy line the one Q12 already drew: **one daemon per user**, isolation between users is the operator's layer.

Q7 also forced a correction worth recording, since the intuition it contradicts is common: **the OS does not kill children when their parent dies.** POSIX re-parents them to PID 1, and `atexit` never runs under `SIGKILL`, so a hard-killed host leaves live children — *orphans*. Zombies are the opposite failure: a child that exited while the parent is **alive** and did not reap it. Dying-with-parent is per-platform, and **Windows is the easiest of the three** (a Job Object with `KILL_ON_JOB_CLOSE`, and no Unix-style zombies), while Linux needs `PR_SET_PDEATHSIG` and macOS has no equivalent at all — which is why prime-agent's `fork-server-script.ts` polls `os.getppid()` in a loop. pH needs the same handling regardless of Q7's answer, since D19 gives every agent a runtime child: about 50 lines plus an orphan journal, and **no broker, queue or supervisor protocol**.

**Closed since v1.0:** **Q3** and **Q6**, both by decisions taken for other reasons — a useful signal that the architecture settled rather than that the questions were dodged. Q3 (runtime venv Python version) existed only to host `prime-agent-runtime` at ≥ 3.11 alongside a ≥ 3.12 host; **D19 removed that package**, so the venv matches the host and `PH_RUNTIME_PYTHON` covers a deployment whose skills need otherwise — and since host and runtime are separate processes over fd 3, matching was always a toolchain preference rather than a constraint. Q6 (is a `bwrap`/Landlock provider in scope for v1?) was answered between **Q10** (approval-gating is not sufficient *as a claim*, so v1 reports reach and offers `containment.strict` while enforcement becomes the default later), **Q12** (no container tier, so Phase 6's confinement work is `bwrap`/Landlock/Seatbelt alone) and **Q11** (`worktree` by default for children, giving v1 real isolation without waiting on confinement).

**Closed since v0.9:** **Q13** — retitled from "tool-authoring promotion ladder" to **the capability/knowledge layer boundary**, because the ladder was this plan's own category error. `HarnessKind` is `prompt | memory | skill | subagent` and prime-agent's refinement guidance ends there — nothing becomes a tool. Plugins change *what actions exist* (developer authority, code, schema, pipeline); `/refine` changes *how the agent works* (inference-time authority, prose plus a pointer, no code). They touch at one point — a `skill` entry's `reference` — and the invariant **"the knowledge layer may only reference capability that already exists"** keeps that a pointer rather than a promotion. Consequences: a `ph tool scaffold` command is explicitly rejected; surfacing "these entries wrap the same import" is analytics, not a pipeline; and Deep Agents' `SkillsMiddleware` layering is a knowledge-scoping model that belongs beside the harness scopes, never near tool rows.

**Closed since v0.8:** **Q10** — how loudly to state the raw-Python non-goal. The finding that decided it: the most likely thing to mislead an operator is a **tier name**, not a missing paragraph — and the plan's own §4.8 table claimed `worktree` "bounds an authored write to the agent's own git worktree", which it does not, since an absolute-path `open()` never consults cwd. That table now states per tier what *is* bounded, what is **not**, and which property is bought. On top of it: permission rows are honoured always and **self-report their reach** ("applies to tool calls; raw `open()`/`subprocess` in a code cell is not covered"), `ph doctor` prints the effective tier in the same three columns, and `containment.strict: true` lets an operator refuse to start unless the tier is `sandbox` with `enforcement: full` — dsh's fail-closed `SANDBOX_UNAVAILABLE` posture lifted from per-call to profile start. Enforcement becomes the default in Phase 6. The earlier option (b) — gate `danger-full-access` behind an acknowledgement — was **dropped rather than deferred**: implying the other sandbox modes are enforced would have strengthened the false belief it was meant to correct.

**Closed since v0.7:** **Q9** — denial semantics for a binding call inside a program. Three parts: a denied call **fails the whole run** (`CodeRunFailure {kind: "denied"}`) instead of raising a catchable `ToolCallError` the program can route around; the `worktree` tier's **default write scope is the agent's own tree plus scratch**, so approvals become rare and meaningful instead of a queue interrupting every cell; and a per-run `workspace/checkpoint` (a `git write-tree` under a hidden ref) makes a failed run **revertible exactly**. The v0.3 phrasing — "make mutating tools *also* natively callable via `mode: both`" — turned out not to be expressible: dsh states that *"within one agent no tool can be native-only while another is code-only"*, so `both` would make `read`/`grep` dual-callable too and defeat the batching PTC exists for. Presentation stays `mode: code`. Two limits are stated rather than implied: git restores the **tree, not the world** (a published package or a dropped table is not undone), and replay reconstructs only the **governed** prefix, since raw `pathlib`/`subprocess` writes are bounded by the worktree but not recorded — so the checkpoint, not replay, is the recovery mechanism.

**Closed since v0.4:** **Q8** (nothing to stay compatible with — D19), **Q12** (containers are the operator's layer; pH's ladder ends at `sandbox`), and **Q5** (the log is the source of truth; `harness_state.json` is a projection — D14 rewritten). All three closed the same way: D19 removed a compatibility obligation, and the design got smaller rather than needing a rule.

**Closed since v0.3:** **Q11** — containment-tier and child-`access` defaults (`advisory` root / `worktree` child / `access="read"`), together with the two §6.3 obligations it created: the RLM prompt states the default, and the compat shim applies one default across both front-ends rather than preserving prime-agent parity over the comm channel.

**One correction this document owed the plan, now folded (v0.2).** §6 item 9 previously called `nest_asyncio` an unresolved consequence of transport A. Verified at source, that was wrong in both directions: `prime-agent-runtime` never re-enters the loop (`get_running_loop()` + `call_soon_threadsafe()`; its only `asyncio.run()` is a separate-process console script), IPython ≥ 7 handles top-level `await` natively, and prime-agent's own bootstrap applies it under `try/except: pass`. It is defensive cover for *model-written* `asyncio.run(...)`, not an architectural dependency. C2 then supplied a fix the v0.1 analysis did not anticipate — serve binding dispatch off the kernel loop so harness code is never subject to re-entrancy. **Both are now specified in port plan §6.2** with row config, a Phase 3 exit criterion (h), a §11 risk row and an Appendix D entry; item 9 is a configuration choice, no longer an open liability.
