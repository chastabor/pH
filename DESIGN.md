# pH — Design

*A plugin-composed Python agent harness.*

This document describes what pH **is**, verified against the source rather than
against the plans. Where the code and the plans disagree, the code is reported
and the divergence is named. Where a mechanism is declared but not implemented,
that is stated next to where a reader would otherwise assume it — which is the
codebase's own §5 rule 6: *"State what is not enforced, next to where it would
be assumed. A caveat only in the docs is a defect."*

Citations are `file:line` at the time of writing.

---

## 1. The overall shape

pH is an agent harness with **no privileged core**. The agent loop, the model
adapter, the tool registry and the session log are not framework internals — they
are rows in a YAML profile, mounted as plugins into a dependency-injection tree,
and any of them can be replaced without forking anything else.

**Two axes organise everything below, and dsh's notes name them: space and
time.** *Space* is the topology — the `Context` tree, what is mounted, which
service each plugin resolved, which realm an agent runs in. cordis owns it, and
`ph doctor` reports it. *Time* is the session log — the immutable, append-only
order in which state changed and results arrived. The persistence rows own it,
and `ph --mode trajectory` reads it. The agent loop sits between the two: it
reads the log to build a request (I3) and acts through the topology to run a
tool. Neither axis is allowed to stand in for the other, which is I4 in one
sentence — the log is never rewritten, and what the model sees is a projection
of it.

```
┌──────────────────────────────────────────────────────────────────────────┐
│ ph-app          CLI (Typer) · TUI (Textual) · daemon · 6 output modes    │
│                 consumes session/event + agent/* ; drives ctx.agents     │
├──────────────────────────────────────────────────────────────────────────┤
│ plugin bundles  (YAML rows; each row = one plugin module + config)       │
│   ph-core       llm · session · tools · system-prompt · agent · loop     │
│                 persistence · 27 capability seams · builtin tools        │
│   ph-rlm        code-runtime-python · bindings · subagent provider       │
│                 messaging · registry · harness (/refine) · presentation  │
│   ph-stabilize  todo · result-offload · compaction · limits · hitl       │
│                 memory · skills-progressive                              │
├──────────────────────────────────────────────────────────────────────────┤
│ ph.cordis       Context · services · effects · scopes/isolation          │
│                 Profile → Mount (rows → plugin tree) · patches · entry pts │
│                 4 dispatch modes · event declaration registry            │
└──────────────────────────────────────────────────────────────────────────┘
                              ┊ fd 3, framed protocol v2
┌──────────────────────────────────────────────────────────────────────────┐
│ ph-runtime-guest   the guest half of the code runtime, in its own venv   │
│                    imports neither ph-core nor ph-rlm, by construction   │
└──────────────────────────────────────────────────────────────────────────┘
```

### Scale

| | |
|---|---|
| Packages | 5 (`ph-core`, `ph-app`, `ph-rlm`, `ph-stabilize`, `ph-runtime-guest`) |
| Source | ~48 000 lines |
| Tests | 1 796 (one skipped without a provider key) |
| Registered plugins | 90 across the `ph.plugins` entry-point group |
| Installable bundles | 2 (`rlm`, `stabilize`) via `ph.bundles` |
| Base profile | 55 rows (`ph/bundles/base.yaml`) |
| Capability seams | 27 service keys across 26 modules |
| Declared bus events | 36 |
| Python | ≥ 3.12 |

### The five packages

**`ph-core`** — cordis, the session log, the LLM vocabulary, the tool pipeline,
the agent loop, and every capability seam. Nothing here knows about a terminal.

**`ph-app`** — the `ph` command, the Textual TUI, the daemon and its wire
protocol, and the provider adapters (Anthropic, OpenAI-compatible). Depends on
`ph-core` and on neither bundle: bundles reach it through the `ph.bundles`
entry-point group, so `ph-app` need not import `ph-rlm` to offer the `rlm`
profile (`ph_app/profiles.py`).

**`ph-rlm`** — Prime Agent's design as plugins. The model's surface is Code Mode:
it writes Python, and *bindings* re-enter the governed tool pipeline as
`tool/code-dispatch`. Ships the subagent provider, agent messaging, and the
Continual Harness (`/refine`).

**`ph-stabilize`** — Deep Agents' features as plugins: a todo tool as a cognitive
anchor, result offloading, threshold compaction, call limits, human-in-the-loop.

**`ph-runtime-guest`** — the guest half of the Python code runtime. It runs in
`$PH_CACHE/runtime-venv`, in a subprocess spawned per agent, and speaks to the
host over one framed channel on **fd 3** (`ph_runtime/protocol.py`).
`PROTOCOL_VERSION = 2`; fd 0/1/2 stay the program's own so a cell's `print` and a
grandchild's output need not be untangled from frames. **It imports neither
`ph-core` nor `ph-rlm`** — verified, zero such imports — because "the process
boundary exists so that model code cannot reach the harness, and importing the
harness would put it back inside" (`ph_runtime/__init__.py`).

---

## 2. cordis — the plugin substrate

`ph.cordis` is a port of a TypeScript meta-framework. It provides a `Context`
tree that is simultaneously a service locator, an event bus, and a lifetime
manager. Understanding it is most of understanding pH.

### 2.1 A plugin is four things

```python
@plugin("session", inject=["llm"], config=SessionConfig)
async def apply(ctx: Context, config: SessionConfig) -> None: ...
```

A name, a list of injected service keys, an optional pydantic config model, and
an `apply` body (`cordis/plugin.py`). The shape is duck-typed by
`normalize_plugin` (`plugin.py`), not enforced by a base class — a
decorated function, a module, or any object carrying those attributes all work.

### 2.2 Rows compose into a profile

A profile is an ordered list of YAML documents. Each document holds **rows** (a
row mounts one plugin with one config) or **patches** addressing an existing row
by id (`cordis/loader.py`).

A patch may `insert`, `remove`, replace `config`, set `disabled`, or set
`isolate` (§2.7). A patch replaces a row's **whole** config rather than merging
into it, deliberately: "a row's effective value is always one layer's, readable
in one place" (`loader.py`).

**Three layers, and the third is reachable from the command line.** dsh's notes
name them — bundle, profile, patch — and for six phases pH had the third only as
a file, `$PH_HOME/profiles/<name>.yaml`. `--patch` on `ph`, `ph doctor` and
`ph events` takes a patch entry as YAML and composes it last, as the `cli`
layer, so `--dump-config` prints `layer: cli` beside what it changed and
`doctor`'s topology says `by cli`. It is the *same grammar* as a document,
deliberately: a second spelling for "change this row" is how a flag and a file
come to accept different things. And it goes through `safe_yaml_load`, so the
code-tag refusal that guards a file guards the flag — the argument that a
`!!js`-style value in a profile is refused at parse time (D9, I-8) would be worth
nothing if the command line were a way around it.

YAML is read by `SafeRowLoader`, which strips every non-scalar implicit
conversion — refusing timestamps, sexagesimals, and any unknown `!tag`
(`loader.py`). Config is data, not code. `${env:VAR:-default}` is the only
interpolation.

### 2.3 Activation is service-driven, never file-ordered

`Profile.mount(ctx)` mounts every enabled row in file order — **and nothing runs**
(`loader.py`). Execution begins at `Context.reconcile()`, which runs to a
fixpoint because activating one plugin may provide the service another was
waiting on (`context.py`).

A plugin activates when every key in its `inject` list resolves. Load order is
therefore *expressed through service requirements*, never through position in a
file. Three consequences:

- Losing a service **deactivates** the fork but leaves it mounted, so
  re-providing the key reactivates it on a later reconcile
  (`context.py`, and `ForkScope` in the same module).
- An `apply` that raises is deactivated before the exception propagates
  (`context.py`).
- Oscillation is bounded: 64 rounds, then `RuntimeError` naming the cycle
  (`context.py`).

`running(scope)` is bound around each activation and released on the way out, so
one row's registrations cannot land on the previous row's scope
(`context.py`).

After reconciliation, `await ctx.serial("profile/mounted")` fires — the one
moment a profile is whole and nothing has run. A row uses it either to **refuse
the deployment** (a strict containment posture with no sandbox backend) or to
**collect what the profile turned out to contain** (`loader.py`).

**dsh calls this unit a *fiber*.** pH's spelling is `ForkScope` — the mounted
plugin — over `_Dependent`, the reactive half: the keys it waits on, the
activation callback, and the scope it owns while active. The mechanism is the
same, a wrapper that watches its dependencies and activates when all are
present. The word does not appear in this codebase, which is said here so a
reader arriving from dsh can find the thing it names.

`Mount.inactive()` reports row ids whose plugin is not active — an unmet
`inject` key (`loader.py`). `Mount.topology()` renders that per row —
`active`, `activating`, `waiting on <key>`, `unwound · waiting on <key>`,
`unmounted`, or `disabled · by <layer>` — with what each injects and which layer
put it there, into `ph doctor`'s **Topology** section: the live half that
`--dump-config` deliberately does not show. dsh's rule is that the dump must show what *is* running, because once
structure comes from configuration the static code no longer says; for one
round pH had the answer in `inactive()` and nothing called it.

The section reaches the report as **a row like any other** (`seams/topology.py`,
`id: topology` in `base.yaml`), which is why `Profile.mount` provides the `Mount` as
`ctx.mount`: `ph.seams` may import `ph.cordis` and never the reverse, so cordis
cannot file the report itself. A hand-appended section cost a misdiagnosis — a
raise in the reader came out of `doctor`'s broad catch as *"profile does not
mount"*, about a profile that had mounted — and put this one section out of reach
of `ph agents doctor`, which relays `DiagnosticsRegistry.report()`'s shape
verbatim.

### 2.4 Services and realms

`ctx.provide(key, service)` claims `ctx.<key>` within a **provisioning realm**
(`context.py`). A second claim on a held key raises
`ServiceConflictError` naming both the realm and the incumbent owner. Resolution
walks the scope chain most-specific-first (`context.py`), so a child
scope can shadow a service for itself and its descendants.

`provide` returns a disposer, registered as an effect of the calling scope, and
the disposer is **identity-checked** — it removes the service only if the current
occupant is still the one it registered.

### 2.5 Effects: everything unwinds

```python
await ctx.effect(enter, label="worktree")
```

`Context.effect` acquires an artifact and registers its release in one step, so
a failure between the two cannot leave the artifact unregistered
(`context.py`). Its docstring states the rule: *"Every external artifact
an agent takes — a child process, a worktree, a temp path, a lock — is acquired
through here, so cleanup is structural rather than remembered."*

`Context.dispose()` unwinds **children first (reverse order), then own effects
LIFO** (`context.py`). Each disposer's exception is caught and logged
so one bad teardown cannot strand the rest.

> **That ordering is load-bearing and is a real constraint on plugin authors.**
> An effect registered *about* a child runs **after** that child's scope is
> already gone. The subagent provider's teardown is exactly this shape, and its
> docstring bounds what that path may do: the roster, the parent's log and the
> tombstone are live; anything needing the *child's* scope — flushing its session
> through its own services, snapshotting its workspace — is not
> (`ph_rlm/subagents.py`).

Process-level shutdown is the other half. `ph.resources.install_lifecycle`
disposes the root on `atexit` and on `SIGTERM`/`SIGINT` with a grace period, then
**self-`SIGKILL`s**, "because a shutdown path that can hang is a shutdown path
that will" (`resources.py`). It states its own limit: `SIGKILL` runs
nothing on any platform, which is why the crash-recovery layer (paired events,
reconciliation) exists separately.

> **Not currently wired.** `install_lifecycle` has no production caller —
> `ph daemon` calls bare `anyio.run(...)`. Signals are handled only by default
> cancellation, and the daemon's shielded 10-second `finally` is what makes that
> survivable. See §8.

### 2.6 Dispatch: four modes, and every event is declared

Each mode is a method of the same name on `Context` (`cordis/context.py`).

| Mode | Semantics |
|---|---|
| `emit` | Sync, return values ignored. A coroutine is scheduled, not awaited. `contained=True` logs a failing listener and continues |
| `serial` | Awaits listeners in registration order until one bails |
| `parallel` | All listeners concurrently, all awaited; failures collected into an `ExceptionGroup` |
| `waterfall` | Around-middleware. Listeners run outermost-first and receive `(*args, next)`; returning without calling `next()` vetoes the rest of the chain, `inner` included |

> Four modes, where P0-03 ported five and P6-25 still counts five. The fifth
> was a synchronous `bail` — `serial` without the `await` — that no package
> ever declared an event in and that dsh does not have. Dropped rather than
> kept for symmetry; a mode nothing uses is vocabulary a reader has to learn for
> no reason. The one rule it carried, that `0` bails because a decision object
> is never falsy by accident, lives on in `is_bailed` and is held by `serial`.
> The drop is recorded against P0-03 in the plan, where the port history lives.

**Every event must be declared before it can be listened to or dispatched.**
`EventRegistry.declare(name, mode, payload, owner, doc)` (`cordis/events.py`)
fixes an event's dispatch mode as part of its contract: re-declaring with a
different mode raises, dispatching through the wrong mode raises
(`events.py`), and listening to an undeclared name raises
(`events.py`). `note_consumer` records who listens, so `ph events` renders
a producer/consumer matrix from the registry rather than a hand-kept list
(`events.py`).

**The consumer half needs a mount, and did not work until P6-02.** A `declare`
runs at import, so producers are knowable from the import alone; `ctx.on` runs
when a row *activates*, so who listens is a property of the profile rather than
of the code — which is why `ph events` now takes `--profile` and mounts one.
Before that it printed a producer matrix under a producer/consumer heading:
`note_consumer` guards on `if module`, `Context._module` was only ever set by
`Context.scope`, and nothing in the mount path set it, so every scope inherited
the root's empty string and no listener was recorded in any profile. `ForkScope`
now stamps the plugin's own `apply.__module__` onto the activation scope.

The 36 declared events, by mode: 17 `emit`, 16 `waterfall`, 2 `serial`, 1
`parallel`. The `waterfall` majority is the design working — most extension
points in pH are *policy interceptions* (`tools/pre-execute`, `agent/request`,
`fs/write-intent`, `approval/request`, `system-prompt/assemble`), and a
waterfall is the only mode where a plugin can veto.

### 2.7 Scopes, isolation, and visibility

Every agent gets its own `Context` created with `isolated=True`
(`context.py`), which makes it an **isolation boundary**. The key line:

```python
self._isolation = self if isolated else (parent._isolation if parent is not None else None)
```
— fixed at construction, "so `reaches()` is a lookup rather than a walk"
(`context.py`).

Two derived questions answer everything about visibility:

- **`isolation_chain()`** (`context.py`) — this context's isolation
  scopes, most specific first, ending in `None` (the global layer). A scoped
  registry walks it to resolve a name: the innermost scope that registered one
  wins.
- **`reaches(target)`** (`context.py`) — *the one visibility rule*, shared
  by event dispatch and by every scoped registry:
  ```python
  return self._isolation is None or self._isolation.is_ancestor_of(target)
  ```
  A global registration reaches everything; an agent-scoped one reaches that
  agent alone.

A plugin's *activation* scope is deliberately **transparent** — it is not
`isolated`, so it answers `isolation == None` and its registrations are global
(`context.py`). Isolation is for agents, not for rows.

**A realm can be asked for from YAML, which is dsh's `isolate.fs`.** A row that
says `isolate: [fs]` runs in a scope that is an isolation boundary *and* its own
provisioning realm — `scope()`, the same thing an agent gets — and the loader
mounts a second copy of the `fs` row inside it first, so that copy's `provide`
lands in the realm. The isolating row and everything beneath it then resolve
the private `ctx.fs`; every other row keeps the shared one, because nothing was
redirected — `_provision` walks up from the realm and meets the nearer provision
first. The mapping form, `isolate: {fs: {root: /sealed}}`, is the case the
feature exists for: a private copy with identical config is a second instance
and nothing more.

Two things about the spelling are deliberate. It names **row ids, not service
keys**: a row does not declare what it provides, so `fs` is the row whose
`apply` provides `ctx.fs`, and that is the name a compose-time check can
verify — an id that is not a row, a row isolating itself, or a private copy of a
row a later layer disabled are all refused before `--dump-config` prints. And
the private copies are settled by their own `reconcile()` before the isolating
row mounts, **and then checked**. `has(key)` is true from the realm the moment
root provides the key, so the isolating row is ready at once — against the
shared instance — and only registration order makes the private copy win. The
reconcile fixes that order for a copy that is ready; it does nothing for one
whose own `inject` is unmet, which would stay waiting while the row activated
against root's service and the realm silently fell through to exactly what it
was meant to replace. So a private copy that did not activate is a **refusal**
naming the key it lacks: what it needs has to be provided by a row above the
realm. A first draft claimed the reconcile alone covered this; the test that
showed otherwise was the one not yet written. `Mount.topology` lists each realm
and each private copy under `<row>/<source>`.

What a realm does **not** do is change hands while an agent is running in it —
the provider a realm holds is the one it was mounted with. That is the one half
of dsh's isolation story pH has not built, and §8 records it as a deliberate
deferral rather than a gap to be discovered.

### 2.8 `Boundary` and `DEPLOYMENT`

```python
Boundary: TypeAlias = "Context | Deployment"  # context.py:301
DEPLOYMENT = Deployment()  # context.py:298
```

`Boundary` is deliberately **not** `| None`. The reason is a defect that recurred
four times under four different names (`context.py`):

| | The `None` that widened |
|---|---|
| P6-12 | `owner_for(None)` → the seam, so a registration outlived its row |
| P6-24 | `_scope_of(agent)` → `None` → the mount, so an agent-scoped file screen applied to nobody |
| P6-31 | `held_by`'s `None` → the *unrestricted* set, so an unreadable parent handed a child everything the deployment holds |
| P6-31 | `_enforce` skipping its containment check on the same `None` |

One root cause: `scope: Context | None = None` conflated *"I did not state a
boundary"* with *"I mean the deployment"*, and the convenient default was
`scope or self.ctx` — the mount, the widest boundary there is.

The fix has two halves. **Give "everything" a name**: `DEPLOYMENT` is greppable,
where passing the mount `Context` explicitly would not be "since `ctx` is in
scope everywhere and reaches for itself" (`context.py`). **Take the
default off**: enforcement is two layers — mypy for typed references, and a
runtime `TypeError: missing 1 required positional argument` at mount for the
sites that reach a seam through `ctx.<seam>` (which is `Any`).

**What `DEPLOYMENT` means, precisely.** It resolves the *mount's* isolation
chain — widest along the **restriction** axis only. It is **not** a union over
agents: a tool registered on one agent's scope is invisible under it
(`context.py`, `tools/registry.py`, pinned at
`tests/test_tools_registry.py`). "A true is-it-taken-*anywhere* audit is a
per-layer question no single boundary answers."

`boundary_of(scope, mount)` (`context.py`) is the **one** narrowing site,
because `scope is DEPLOYMENT` does not narrow the union — identity against a
value is not a type guard.

**Where the sweep deliberately stops.** A *registration* takes a `Context`, never
a `Boundary`, and so do the two dispatch-time resolvers `owner_for` and
`layer_for`. Their `None` resolves through the `_ACTIVATING` contextvar to the
running row — the **narrowest** correct answer, the opposite of the `None` this
mechanism deletes (`context.py`).

Explicitly rejected: `None` meaning *no access*. That "trades a silent-wide
failure for a silent-narrow one: an empty tool set or an empty prompt degrades a
model quietly rather than erroring, and nobody notices" (`context.py`).

---

## 3. Seams

> **Invariant 5.** *Seams have three roles.* **Definition** (a `Protocol` + a
> service key), **Provider** (a plugin registering an implementation),
> **Consumer** (usually a tool). Swapping a provider changes the product without
> forking consumers.

That is the design. The implementation is richer, and the difference matters.

### 3.1 Not every seam has a provider

Of **27 service keys** across `ph/seams/`, only **six** hold a provider slot. The
rest are contribution tables or plain services. A reader who expects the triad
everywhere will misread two thirds of the directory.

**Provider seams** — a Protocol, a registration method, a claim, a fallback rule:

| Key | Protocol | Registration | Claim | No provider ⇒ |
|---|---|---|---|---|
| `workspace` | `WorkspaceProvider`, `ReclaimingProvider` | `register_provider` | `claim_slot` | **falls back** to `SharedWorkspaceProvider` |
| `sandbox` | `SandboxProvider` | `register_provider` | `claim_slot` | **refuses** — `confine()` raises; never returns `argv` unchanged |
| `compaction` | `CompactionEngine` | `register` | `claim_slot` | `compact_if_needed` no-ops; `compact_now` raises |
| `code_runtime` | `CodeRuntime` | `register` | `claim_slot` | `require()` raises |
| `fs` | *(a callable slot)* | `rebase` | `claim_slot` | falls back to the process root |
| `subagents` | `SubagentProvider`, `RehydratableProvider` | `register_provider(name, …)` | `claim_key` — **many, by name** | `resolve()` returns `None` + a warning naming what *is* mounted |

The five `claim_slot` seams each carry a sibling `<attr>_by: Running | None`
field recording *who* registered — verified: `fs`, `sandbox`, `compaction`,
`code_runtime`, `workspace`. `subagents` deliberately uses `claim_key` instead,
because "run a child" has genuinely different answers in one deployment
(`seams/subagents.py`).

**Contribution tables** — many registrations, no single holder: `commands`,
`skills`, `diagnostics`, `tui_screens`, `tui_status`, `session_telemetry`, and
compaction *notes*.

**Waterfall registrations** — `approval` and `user_questions` register answerers
as `ctx.on(...)` listeners, deliberately: "sugar over `ctx.on`; there is one
routing mechanism" (`seams/approval.py`).

> **These two now have a transport, and it changed the seam rather than sitting
> beside it** (P5-13). An answerer used to be a screen in this process; under the
> daemon it is a socket away, and `AskDesk` registers one answerer that fans the
> ask to *every* attached front end — first answer wins, later ones are refused
> `ask_settled`. Two consequences are visible in the seam itself. `register_answerer`
> takes a **`reachable`** callable, because whether anyone is there to ask is no
> longer knowable from the fact that a listener registered: the desk is reachable
> only while a front end is attached, and `user_questions.ask` appends nothing at
> all when nothing is `attended` (P7-09). And an answer has to be **serialisable** —
> `Edited` and `Responded` are frozen dataclasses, so `answer_to_wire` is the
> inverse of the decoder the seam already had. With nobody attached, an approval
> does *not* resolve: the ask parks, which is what the log already described —
> `approval/asked` with no `approval/decided` **is** the pending state.

**Service-only** — no third-party registration surface: `attachments`,
`containment`, `credentials`, `goals`, `jobs`, `permission_presets`, `schedule`,
`settings`, `shell`, `spill_store`, `subagent_presets`, `subprocess`,
`token_meter`.

> **One provider seam lives outside `ph/seams/`.** `LlmAdapter` is a `Protocol`
> but **not** `@runtime_checkable`, and `LlmRuntime.register_adapter` uses no
> `claim_*` helper — it appends a handle by hand and takes no `scope=`
> (`llm/adapter.py`). This is documented in place as the case that
> escaped the ownership sweep.

### 3.2 The three claiming helpers

`ph/seams/_registry.py` exists because this was written by hand six times and the
release step drifted: "some copies checked identity before removing, some did
not. A disposer that removes whatever *currently* occupies the slot would tear
down a successor registered after its own owner was replaced" (`_registry.py`).

| Helper | Guarantees | Release |
|---|---|---|
| `claim_key` | **Refuses a conflict** — a held key raises | `del` only if the value is still *identically* the one registered |
| `claim_entry` | Appends; no conflict | Removes **by identity**, never `==` — because two rows contributing an equal entry would otherwise have one disposer take the other's |
| `claim_slot` | **Single holder** — a second raises. Also sets the derived `<attr>_by` field | Clears **both** fields, identity-checked |

`claim_slot` derives the `_by` field name rather than taking it as a parameter,
so a holder that omits the field fails loudly at registration (`slots=True` plus
`setattr` on an undeclared name) — `_registry.py`.

Unwinding is *not* these helpers' job; they delegate to `Context.add_disposer`.
Ownership is decided by the caller through `owner_for`/`running_for`.

### 3.3 How a provider is actually verified — four layers

**1. Static typing.** Five Protocols (`WorkspaceProvider`, `SandboxProvider`,
`CompactionEngine`, `CodeRuntime`, `SubagentProvider`) are checked by mypy and
never by `isinstance`. Both `WorkspaceProvider` and `SandboxProvider` say why
they are typed rather than duck-typed: a drifted method would fail *inside the
seam's `except`* and be reported to the operator as `shared` or *unconfined* —
"the one direction this seam must never fail in silently".

**2. Runtime `isinstance`, for capability probes only.** Exactly two Protocols
are ever `isinstance`-checked, and both are *second* Protocols asking "can this
provider also do X":

- `ReclaimingProvider` — can the mounted tier release a tree it did not create?
  (`seams/workspace.py`)
- `RehydratableProvider` — can this provider re-attach a runtime to a settled
  child? (`seams/subagents.py`)

Both are a Protocol rather than a `getattr` probe on purpose: "a provider whose
method is misnamed or has the wrong arity would then fail silently as 'cannot
rehydrate'".

**3. Registration-time refusal.** One seam validates the provider object itself:
a `CodeRuntime` declaring `persistence == "namespace"` without
`declares_kernel_snapshots is True` raises `PersistenceObligationError`
(`seams/code_runtime.py`). A runtime that promises to survive must
promise to snapshot.

**4. Tree-walking gate tests.** `tests/test_registration_ownership.py` (~1 460
lines) walks every module under `ph` via `pkgutil.walk_packages` and requires
that **every scoped method and every provider slot is classified in exactly one
table**. A new seam cannot join unchecked. Highlights:

- `test_every_scoped_method_is_accounted_for` — every `scope=`-taking
  method is in `RECIPES`, `NOT_A_LIFETIME`, or `NOT_EXERCISED`.
- `test_a_registration_is_an_effect_of_the_row_that_made_it` — the
  behavioural gate: row registers, row unmounts, **the seam's own context did not
  grow an effect**.
- `test_a_boundary_parameter_never_has_a_default` — §2.8's durable half,
  covering both parameters and dataclass fields.
- `test_the_classification_is_a_check_and_not_a_promise` — source-text
  falsifiability, so a table entry cannot claim something the code does not do.
- `test_no_module_is_skipped_for_a_bad_reason` — a module that fails to
  import must not silently shrink every other walk.

Sibling gates: `test_cordis_events.py` (every dispatched event is declared),
`test_wire.py` (every model uses the shared wire base and camel aliases),
`test_seams.py` (per-seam failure-mode contracts), `test_layering.py` (import
layering).

### 3.4 Decline is not failure pattern

`WorkspaceSeam.acquire` distinguishes **three** outcomes, and the distinction is
load-bearing because half the directories a person runs pH in are not git
repositories (`seams/workspace.py`):

| Outcome | Signal | Recorded `DeclineReason` |
|---|---|---|
| Decline **with** a reason | `raise WorkspaceDeclined(reason, detail)` | the provider's own — `not-a-repository`, `branch-in-use`, `path-exists` |
| Decline **without** one | `return None` | **`None`** — no reason is *fabricated* |
| Fail | any other exception | `provider-failed` |

None is fatal; all three fall back to `shared` and record what happened on
`workspace/acquired`. `DeclineReason` is a `Literal`, "a code rather than prose",
because `ph doctor` prints it and "a durable event carrying an English sentence
is unparseable by the consumer that has to branch on it"
(`seams/workspace.py`).

The refusal to invent a reason is the interesting part: fabricating one "would
reintroduce one level down the very confusion this field exists to remove"
(`workspace.py`).

Parallel shapes elsewhere: `ApprovalService.request` never raises and returns
`"unavailable"` — **fail-closed**; `SubagentService.resolve` returns `None` plus a
warning naming what is mounted; `FsService.root_for` catches a raising resolver
and returns the process root.

---

## 4. The `ph` command surface

### 4.1 Commands

```
ph [--print P] [--profile P] [--provider P] [--model M] [--session ID]
   [--mode MODE] [--attach PATH ...] [--resume ID] [--dump-config]
   [--no-spawn]                                  # tui/web: refuse to start a daemon
   [--keep-daemon]                               # tui/web: start a service one, not ephemeral
   [--host H] [--port N] [--open]                # web: bind, and the token URL

ph doctor      [--profile]
ph daemon      [--profile] [--provider] [--model] [--passivate-after off|MIN]
               [--ephemeral]                     # exit once nothing needs it
ph events      [--json]

ph agents                                     # list running roots
ph agents send      <session> <prompt>
ph agents attach    <session> [--since N] [--until-idle] [--all]
ph agents schedule  <session> [--prompt P] [--at MS|--every MS|--cron X|--cancel ID]
ph agents status    <session>
ph agents doctor
ph agents shutdown

ph workspaces gc [--profile] [--older-than DAYS] [--remove] [--session ID]
```

`ph doctor` prints the three path roots, platform, daemon socket lifetime,
non-guarantees, available profiles — then **mounts** the profile and prints every
`ctx.diagnostics` section. If the profile refuses to start (a strict containment
posture with no sandbox backend), that is the most important thing it can say, so
it reports a sentence and exits 1 rather than a traceback (`cli.py`).

`ph agents` is the client half of the daemon. Every command goes through one
`_ask()` spine, which is what keeps "no daemon is running" one sentence rather
than seven, and which distinguishes an **absent** socket (nothing was started)
from a **present but refusing** one (something crashed and left its path behind)
— opposite next steps (`agents.py`).

`ph workspaces gc` **reports by default; removing is the flag** — the opposite
way round from most `gc`, because the person who most needs it is the one who
just found the disk full and does not yet know what these directories are
(`workspaces.py`).

Everything exits **1** on refusal, except argument errors under
`--mode trajectory` and unreadable attachments, which exit **2**.

### 4.2 Output modes

| Mode | What it is |
|---|---|
| `text` | Assistant text, plus a stderr line with session id, event count, log path |
| `json` | **Not a rendering** — the session log's *own* camelCase envelopes, emitted as each commits, so a pipe consumer and the stored JSONL parse one format |
| `transcript` | Reads `session.transcript()`, **not** `derive_messages()`, so compaction does not erase what the human saw |
| `rpc` | JSON-RPC over stdio. Takes no `--print` — the peer drives |
| `tui` | Textual, imported lazily. **A daemon client** (P5-14): it attaches, and closing it detaches rather than ending the turn. Silently starts an ephemeral daemon when no socket answers — `--keep-daemon` makes the one it starts a service instead, and `--no-spawn` refuses to start one at all |
| `web` | The same `tui` in a browser tab, over `textual-serve` — one subprocess per tab, so each tab is one more front end on the one daemon, and **all tabs of a launch share one session** (the multiplex: private composers, one log). Drop a file on the page to attach it. Needs the `ph-app[web]` extra; `--mode web` without it prints the install line. Binds `127.0.0.1` unless `--host` says otherwise, and every request needs the per-launch token from the URL |
| `trajectory` | **Mounts nothing** — no agent, provider, answerers, or plugins. A fold over a stored file |

### 4.3 Slash commands

The `commands` seam (`seams/commands.py`) registers in-session `/verbs`. A
dispatch appends `command/run` then `command/done` **even when the body raises**,
and never opens a `turn/*` — because the human decided it, not the model
(`commands.py`).

| Command | Row | In profiles |
|---|---|---|
| `/autonomous <objective> [-- <gate>; …]` · `/autonomous stop` | `autonomous` | headless, tui, rlm, rlm-stable |
| `/revert <seq>` | `workspace-revert` | rlm, rlm-stable |
| `/workspaces [list \| merge \| remove …]` | `workspace-commands` | rlm, rlm-stable |
| `/refine [--global] [--show] [--rollback <id>]` | `rlm-harness` | rlm, rlm-stable |
| `/compact [what you are about to work on]` | `command-compact` | rlm-stable |

The TUI registers its own verbs at runtime rather than through a profile row —
`/commands`, `/model`, `/theme`, `/sessions`, `/permissions`, `/login`,
`/thinking`, `/tools`, `/sidebar`, `/quit` — each reachable three ways (slash
command, Textual action, key binding), so adding one is a table row plus a method
(`tui/commands.py`). A contributed screen gets the same three routes;
`/trajectory` (F2) is the one that ships.

---

## 5. Stages of a process

### 5.1 Mount

```
profile name ──► profile_or_exit ──► Profile.from_documents ──► compose_rows
                 (resolve + read + compose, once)                     │
                                          Context() ◄─────────────────┘
                                              │
                            ctx.plugin(row) ×N   (nothing runs yet)
                                              │
                                       ctx.reconcile()   ← fixpoint
                                              │
                                 ctx.serial("profile/mounted")
```

`mounted()` (`ph_app/runtime.py`) wraps this and guarantees
`await ctx.drain(); await ctx.dispose()` in an unconditional `finally`.
`prompted()` adds the run: create session → ingest attachments → create agent →
`followup` → `run()` → **flush**. Attachments are ingested *before* the agent
exists so an unreadable file fails the command rather than a turn.

### 5.2 Running

The per-turn contract, from `agent_loop/driver.py`:

```
turn/start
  ├ inbox.claim               → agent/inbox/claimed
  ├ system_prompt.assemble    → system-prompt/assemble
  ├ agent/pre-step            → reject | enter(messages)
  │    reject → turn/end{blocked}
  ├ step/start
  │    user/message*          (claimed batch, surface: append)
  │    agent/request          → LlmCallConfig
  │    request/header         (appended only when it changed)
  │    request/context        (appended only when the route changed)
  │    llm/stream             → assistant/chunk* → assistant/message
  │    agent/request-error    → retry | None
  ├ step/end
  ├ agent/turn-stopping       (a listener objects by steering)
turn/end
```

Two properties the loop holds itself to (`driver.py`): every request's
`messages` **is** `session.derive_messages()` (§7.3), and `max-tokens` is
**sticky for the turn**, so a later completed step cannot report a truncated
answer as clean.

A driver that let an exception escape would take the process down with one bad
turn, so `run()` contains everything except cancellation
(`driver.py`).

Three inbox targets, differing in *when* and *whether they wake*
(`agent/inbox.py`):

| Call | Lands at | Wakes an idle agent |
|---|---|---|
| `followup` | next **turn** | yes |
| `steer` | next **step** | yes |
| `inject` | next **step** | **no** |

Durability is not the loop's job: `session-checkpoint-policy` flushes before
every model request, before every top-level tool body, and after a rejected
`agent/pre-step` (`persistence/checkpoint_policy.py`).

### 5.3 Rehydrate

```
store.read(id) ──► interrupted_turn_closers(events) ──► Session(seed=[…, …])
                                                              │
                                      sessions.adopt ─────────┤
                                                              │
                                       append session/resumed ┘
```

`resume_session` (`persistence/jsonl.py`) reads through the **Protocol**,
not a filename, so a database backend resumes identically.

**Repair runs on the seed, before publication**, so a resumed session is
provider-valid the first time anything reads it — an open turn reaching
`derive_messages()` would otherwise be rejected by the provider before anyone
noticed it was unclosed. `interrupted_turn_closers` (`persistence/repair.py`)
emits, in order: one synthetic `tool/result` per unresolved call (`is_error`,
code `TOOL_NOT_STARTED` if the call was never stamped, else
`TOOL_OUTCOME_UNKNOWN`), a `step/end` if a step was open, and `turn/end` with
`{"reason": {"kind": "interrupted"}}`. Timestamps are the **last real event's**,
never `now()`.

`interrupted` in the `session/resumed` payload is the honest signal for "this
crashed" versus "this was reopened": a clean stop synthesizes no closers.

Because `adopt` emits `session/created`, the resume path meets the **same**
listeners a fresh session does — which is how workspace reconciliation runs on
resume without a resume-only hook (`seams/workspace.py`).

**Seed acceptance is one gate for every path** (fork, resume, replay, import):
`_readmit` requires `seq == index`, contiguous from 0, and refuses unknown
non-ignorable types (`session/session.py`).

**Subagent rehydration** is separate and narrower. `RehydratableProvider.rehydrate`
re-attaches a runtime to a *settled* child so it can be addressed again. The one
implementation refuses if the session or the **parent** is not live — checked
*before* anything is built, because the earlier order orphaned an unbounded agent
— and then **re-applies the stored grant, because a fresh scope means a fresh
ceiling** (`ph_rlm/subagents.py`). A deleted child is never rehydrated;
the tombstone is the record.

> ### A session resumes repeatedly
>
> **What a resume owes the store is not what the store already has.** A resume
> seeds the stored events, then adds two things nobody wrote: the repair closers,
> and the `session/end-seed` the constructor appends. Both are in the log before
> the store is asked to track it.
>
> `JsonlSessionStore.track` used to infer durability from `path.exists()` and
> queue nothing when the file was already there — which is right for a fresh
> session (empty log) and a fork (new file), and wrong for a resume: it discarded
> exactly the events that were owed, leaving a **gap in the seq space** that
> `_readmit` refuses. The session resumed once and could never be opened again.
> Measured: a daemon root survived two lifetimes, because `_session_for` resumes
> on every start *and* every wake from passivation.
>
> `TursoSessionStore` was unaffected — it upserts by `seq`, so it queues its whole
> log unconditionally. The two backends disagreed about a Protocol-level
> guarantee, and the appending one was wrong.
>
> **Now** `Session.durable_length` states what a store already holds, set by
> `resume_session` — the only place holding both the events it read and the log it
> built — and `track` queues `events[durable_length:]`. One rule covers all three
> cases. Nothing is renumbered: seqs and timestamps are preserved verbatim, which
> is what keeps repair's deliberately backdated closers backdated. A side effect
> worth having: **the repair is now durable**, so a stored log stops reading as
> crashed after the first reopen.
>
> Covered in the parity suite (both backends, three reopens) and in
> `test_repair.py` (the real `resume_session`, which the parity tests only
> simulate). Each resume adds two events, as Turso always did.

### 5.4 Shutdown — every distinct reason

| Reason | Trigger | Event appended | What survives |
|---|---|---|---|
| **Clean dispose** | mode completion, or wire `shutdown` | **none** — there is no `supervisor/shutdown` type | whatever the caller flushed |
| **Passivation** | idle ≥ `PASSIVATE_AFTER` (90 min) on a 60 s sweep | `supervisor/passivated {idleMs}`, **write-ahead** | the JSONL: journal, schedules, ladder state, all re-folded on next start |
| **Retry ladder** | any `Exception` from a root's task | `supervisor/retry` before each attempt; `supervisor/failed` on give-up; `supervisor/recovered` on success | the root stays **mounted** and still accepts wakes |
| **Daemon unreachable** | socket `(st_dev, st_ino)` changed | `supervisor/unreachable`, to **every** root, each flushed | everything — **roots keep working** |
| **Session lease (I-5)** | a second daemon opens a held log | **none** — an error frame, `session_already_active` | the first daemon is unaffected |
| **Agent cancellation** | `AgentCancelCause` | `turn/end{aborted}` | a partial `assistant/message` with `interrupted: true`; durable `tool/result` pairs for skipped calls |
| **Subagent release** | parent teardown, or model `delete()` | `subagent/status{cancelled}` + `subagent/deleted` | the child's **log**, always — it is a tombstone, not a deletion |
| **Limits / breaker** | a configured ceiling | `limits/exceeded` or `limits/breaker-tripped` | everything — none of these stop a process |

Details worth having:

**Passivation** measures idleness **from the log's own last event**, so the clock
survives a restart. It keeps a root alive for any of five reasons: status ≠ idle
(a root mid-ladder is never released), live subscribers, inside the window, a
live subagent, or a live schedule (`supervisor.py`). A passivated root
is resumable through the ordinary `start()` path — but while passivated,
`session/status` and friends return `no_such_session`, because `_root()` looks
only in the live map.

**The retry ladder** is three delays `(0.25, 1.0, 5.0)`; the fourth consecutive
crash gives up. Its state is **folded from the log, not remembered** — and
`turn/end` is deliberately not consulted, because doing so once produced 165
retries in two seconds with no give-up (`daemon/recovery.py`). **Turn
failures are explicitly out of scope**: the driver contains its own, retry
policy has already handled transient provider errors, and re-running a failed
turn would find an empty inbox and produce a trivially-successful empty one.
Between attempts the worktree is rolled back best-effort, and the boolean is
recorded as `restored` so the transcript never implies a rollback that did not
happen.

**Unreachability** is latched once and never cleared, and compares
`(st_dev, st_ino)` rather than existence — giving two distinct outcomes,
`removed` (logind reaped the runtime dir) and `replaced`. It is appended to every
root **and each is flushed**, because the way out has stopped being reliable and
the likely next action is `kill`. It is *not* reported as `failed`: that would put
the recovery ladder to work climbing over a socket.

**The lease is daemon-against-daemon and no further.** A `ph -p --session x` run
against a daemon-held session still opens it, because the lease is not in the
store (`supervisor.py`). `thread_local=False` is load-bearing and its
absence is silent: filelock's re-entrancy counter is thread-local, so a lease
taken on a worker thread releases nothing.

**Cancellation vocabulary.** `AgentCancelCause.kind` is
`user | parent | hook | disposed | legacy` — but only **three** are ever
constructed: `user` (TUI interrupt, which keeps the inbox), `parent` (subagent
release), `disposed` (driver disposal, which clears it). `hook` and `legacy` are
declared and dead. `TurnEndReason.kind` is
`completed | aborted | blocked | error | max-tokens | interrupted`, of which
`interrupted` is never constructed as a dataclass — it reaches logs only as the
wire payload repair writes on resume.

**Limits ship off.** All five ceilings default to unlimited and no profile sets
them, because "a limit nobody chose is a limit that fires on someone's longest
legitimate turn" (`ph_stabilize/bundle.yaml`). The **breaker** is the
exception: on by default at 5 consecutive failures per tool, reset by any
success, and it refuses *one tool call* rather than ending a turn.

**There is no global turn or request timeout**, no max-steps in the driver, no
health check, no max-roots cap, and no per-root memory cap. `NON_GUARANTEES`
(`supervisor.py`) states this rather than leaving it to be discovered, and
`ph doctor` prints it — including that **after a restart roots are not
re-mounted, so a schedule does not fire until something touches its root**.

---

## 6. Parent and child

### 6.1 The hierarchy is a `Context` tree

```python
owner = getattr(parent, "ctx", None)
base = owner if isinstance(owner, Context) else self.ctx
scope = base.scope(f"agent:{session.id}")
```
— `agent/registry.py`. That is the whole mechanism. `parent=` makes the
child's scope a **child node of the parent's `Context`**; without it the scope
hangs off the registry.

Three consequences follow structurally, with no bookkeeping:

- **Visibility inherits** — the child's `isolation_chain()` is
  `[child, parent, None]`, so the parent's layer is consulted when resolving the
  child's tools.
- **Disposal cascades** — `Context.dispose` unwinds `_children` first.
- **Containment is one-way** — `parent.ctx.reaches(child.ctx)` is true and the
  reverse is false (pinned at `tests/test_subagent_grant.py`).

The roster entry is itself a disposer of the scope it describes
(`registry.py`), so a parent's cascade cannot leave a live-looking handle
behind.

In the log, the same relationship is `SessionHeader.parent_session`,
`origin: "subagent"`, and `delegation_depth`. An agent's id **is** its session's
id, so the link needs no side index.

### 6.2 Orchestration

**Admission is non-blocking.** `start()` resolves once the child is *admitted* —
session created, admission logged, task detached — not once it has answered
(`seams/subagents.py`).

The spawn path, in order (`ph_rlm/subagents.py`): depth gate → name and
model resolution (**no model fallback**) → child session with the header meta →
`agents.create(..., parent=parent)` → workspace → `SubagentRun` → append
`subagent/admitted` → register the parent-scope effect → `followup` the task.

**Status is a fold, not a field.** `subagent_roster(session)` folds
`admitted | status | deleted` from the parent's own log
(`seams/subagents.py`). Admission seeds `queued` because `to_wire()`
deliberately omits status and the first `subagent/status` comes from a detached
task — without the seed, a reader between the two sees a child with no status.
Deletion writes a **tombstone**, not a status: a parent asking what happened to
the child it revoked deserves an answer other than silence.

`SETTLED_STATUSES` is a real constant for a real reason: a hand-written copy was
once wrong in three of its four members and pinned every parent forever
(`seams/subagents.py`). `child_is_live` checks the tombstone first and
treats **an unrecognised status as live** — failing toward keeping a child alive.

**Depth** is `RLM_MAX_DEPTH = 2` ("depth 0 delegates, depth 1 delegates, depth 2
does the work"), read from the **typed** header field rather than the wire alias,
because a rename would return 0 and 0 *opens* the gate
(`ph_rlm/subagents.py`).

**Quiescing** drops the session observer, releases the jobs entry, and disposes
the agent — because the agent scope owns the child's kernel subprocess, and
keeping it leaked one CPython per delegation. The terminal `result` is kept, so a
late `result()` still answers.

### 6.3 Communication

**The reach rule (C7, "nuclear family")** — `family_reach` in `seams/subagents.py`:

```python
if sender_id == target_id:
    return True  # self
if target_id == sender_parent:
    return True  # the parent
if target_parent == sender_id:
    return True  # a direct child
return sender_parent == target_parent  # a sibling, roots included
```

A grandparent is out of reach. Two roots are siblings. `reachable_family` is
*derived from* this function so the guard that refuses a send and the roster that
tells the model who it may address cannot disagree.

**The boundary is a `ctx.tools.guard`** — deny-only, runs last, and cannot be
re-permitted by any later listener (`ph_rlm/messaging.py`). The rate
limit is deliberately **not** a guard, because under Code Mode a *denial* ends
the whole cell, and being rate-limited should not.

Delivery is always `steer`, landing at the target's next step. A settled target
is woken through `ensure_addressable` first. Pending messages are capped per
session, and bodies at 16 KiB.

**Usage is attributed upward.** Each child `assistant/message` appends
`subagent/usage-attributed` to the **parent's** log.

> It is an **additive record for readers**, not an input to any measurement: the
> meter does not read this event — `TokenMeter.last_usage` scans only
> `assistant/message` in the log it is given (`seams/token_meter.py`), and
> a child's messages are in the *child's* log, so the parent's measurement is
> already correct without it. Its only consumer today is the TUI panel. The
> producing module claimed the meter "can subtract" a child's tokens; that wording
> is now corrected at the source.

**`descendants()` is deliberately not `reachable_family`.** Descent is transitive
and covers grandchildren; the messaging family is one hop and includes siblings.
Borrowing the messaging rule for a filesystem question "would widen a filesystem
question with an answer computed for a different one. That is the shape of
privilege escalation I7 names, arrived at by reuse rather than by intent"
(`seams/subagents.py`).

### 6.4 Resources

Everything a child takes is an effect of the child's scope, so it is released
with the child: the workspace (`scope=child_agent.ctx`), the code-runtime kernel
subprocess (`scope.effect(enter, label=f"code-runtime:{namespace}")`), jobs, and
the child itself as an effect of the *parent's* scope.

Since the scope tree nests, that parent-scope effect is **no longer what stops
the child** — the tree shape is. What it does now is write the tombstone and
update the roster. And because `dispose` unwinds children before its own effects,
that handler runs when the child's scope is *already gone*, which bounds what it
may do (§2.5).

### 6.5 Capability: a child never holds more than its parent at time of initialization

This is the security content of the hierarchy, and it is enforced at the **seam**,
not left to providers (`check_grant` and `_enforce` in `seams/subagents.py`):

```
resolve_preset  →  _delegating_boundary  →  held_by  →  check_grant
                                                            │
                       provider.start(request)  ◄────────────┘
                                    │
                              _enforce(grant, run, held, boundary)
```

- **`_delegating_boundary`** resolves the ceiling's frame. A parent whose `.ctx`
  is unreadable is **refused** — not defaulted to the deployment. That defect was
  real: an unreadable parent produced a 7-tool ceiling where its own was 6.
- **`check_grant`** refuses a spawn naming any skill or tool the parent does not
  hold — **refused, not silently intersected**, because a `reviewer` child missing
  its review skill does the job wrong and reports success.
- **`_enforce`** verifies `boundary.reaches(run.scope)` — the child's scope must
  be inside the parent's — and applies the grant. A provider that hands back no
  scope is refused, but **only when the spawn actually narrows**, so a deployment
  where nothing is restricted keeps working with any provider.
- **`Grant.apply`** narrows **by restriction, never by registration**. A scope's
  own registration is unmaskable by its own filter, so registering on a child
  would hand it something its parent cannot see. "Filters only intersect, so they
  are the only instrument a spawn is allowed to use."

The restriction algebra is one type, `NameFilter(allow, deny)`
(`seams/_restriction.py`). `None` means "no opinion" in both directions,
"which is what makes intersection the whole composition rule: a filter can only
ever remove a name another filter allowed, never restore one another removed."

The resolution rule in `ToolRuntime._build_view` (`tools/registry.py`):

> **A restriction reaches everything outside the scope that wrote it, and nothing
> inside.** A layer is never filtered by its own restriction — an agent's own
> registration cannot be masked out from under it — while an ancestor's is, which
> is what "a child holds a subset of its parent" means.

And its corollary, stated in place: *"'Inside' means this layer, not this
subtree. A tool registered on a descendant's scope is not reachable by an
ancestor's filter either, so a grandchild can hold what its granting parent
cannot see. Deliberate, and the reason a spawn may only ever `restrict`."*

**A child's capability is fixed at admission.** `Grant` materializes the
allow-list rather than relying on the chain, because "the chain answers 'no more
than the parent **holds**', and this answers 'no more than the parent held
**then**'. Only the second is stable enough to read a transcript against."
Pinned: a tool the parent gains *later* is visible to the parent and not to the
child (`tests/test_subagent_grant.py`).

### 6.6 The containment ladder

| Tier | Kind handed out | `repo_writable` | Bounds an absolute-path `open()`? |
|---|---|---|---|
| `advisory` | `shared` | `True` | no — honoured by *saying so* |
| `worktree` | `worktree` / `worktree-ephemeral` | `True` for both | no |
| `worktree` | `overlay` / `overlay-ephemeral` | `True` for both | no — a **peer** of the row above, not a rung between it and `sandbox` |
| `sandbox` | `readonly-scratch` | `False` | yes, at the kernel |

**Two boundaries wear the word "sandbox", and they are not the same one.**
`ctx.sandbox` bounds *every command the harness wraps* (P6-04's `sandbox-local`,
which claims its slot only after a probe writes a canary outside the workspace and
finds the host copy unchanged). The `sandbox` **workspace tier** is what an agent
is *handed*: `workspace-readonly-scratch` (P6-05) roots the workspace inside the
agent's own scratch and reports `repo_writable=False`, so the repository is
readable and writable nowhere. The second depends on the first and says so — the
row claims the rung only when a backend is mounted and enforcing, because
`repo_writable=False` without one is E1's failure in the field E1 is about.

`ph doctor` prints both, which is why `describe()` carries a separate **"confined
commands"** row: a reader must not have to reconcile "bounds: nothing" in the
workspace section with a kernel that is in fact refusing the write.

The rule, verbatim (the module docstring of `seams/workspace.py`):

> **`repo_writable` records which guarantee was obtained, never which was
> requested.** A caller asking `access="read"` gets the strongest kind the
> mounted tier can actually provide; `False` means a tier is enforcing it. Any
> wording here, in `ph doctor`, or in a config comment that blurs request and
> guarantee is a defect (§12 Q10).

The reasoning that used to sit in that docstring — an absolute-path `open()`
never consults a cwd, so the `worktree` tier bounds nothing about `/etc/passwd`
— now lives in `tests/test_workspace.py` and is measured at the rung itself in
`test_containment_ladder.py`, per §9's rule.

`project_access(kind)` maps the kind back to what was granted *of the project*: a
`worktree-ephemeral` child may write its checkout freely and merges nothing, so
what it was granted of the project is `read`. This — not the request — is what
lands in `granted_access` on `subagent/admitted`.

Both halves of tier selection can only **lower** the answer: a chosen `advisory`
declines a registered provider, and an absent provider cannot deliver whatever
was chosen (`seams/workspace.py`).

> **The ladder is complete, and this paragraph has been wrong twice.** It once
> read "no tier in pH bounds an absolute-path write, and `containment.strict` has
> nothing to satisfy it" — closed by P6-04's `sandbox-local`, verified against a
> real kernel. It then read that the top *workspace* rung was
> vocabulary-complete and implementation-absent — closed by P6-05's
> `workspace-readonly-scratch`, whose own gate writes to the repository from a
> confined child and finds it untouched. Every kind in `WorkspaceKind` now has a
> producer, and `effective_tier` can return every rung in `ContainmentTier`.
>
> What remains unwritten is one *backend*: `sandbox-local` ships `bwrap` and
> Seatbelt, both verified against a real kernel (P6-04, P6-40), and Landlock does
> not exist. A host with neither has no `sandbox` rung — the row declines rather
> than claiming one, which is the whole of why the claim is trustworthy where it
> is made.

> **`DowngradeReason` has exactly one member** — `workspace-not-mounted` — and one
> producer. A *tier-driven* narrowing (asking `write`, getting
> `worktree-ephemeral`) records **no** downgrade reason; it shows up only as
> `granted_access`.

---

## 7. The five invariants, and where they affect

The port plan (`Python_Harness_Port_Plan.md` §2) lists five invariants carried
from DeepSeek Harness. They are worth reading as *properties the architecture
makes hard to violate*, not as rules anyone remembers.

> **Three numbering systems exist in `plans/`, and they collide.** Disambiguate
> before citing:
>
> | Register | Where | Content |
> |---|---|---|
> | Items **1–5** | `Python_Harness_Port_Plan.md` §2 | the five below |
> | **I1–I8** | `Implementation_Plan.md` | those five (origin `dsh`) **plus three added by pH**: I6 every-projection-equals-its-fold, I7 knowledge-layer-may-only-reference-existing-capability, I8 containment-is-not-interception |
> | **I-1 – I-9** | `Implementation_Plan.md` | a *separate* hardening register — runtime-dir resolution, credentials, session leases, wire casing |
>
> So code citing "I7" means the knowledge-layer rule; code citing "I-5" means
> session leases. They are not the same series. Note also that I7's *canonical*
> wording is about the knowledge layer; the "delegation must not be privilege
> escalation" phrasing used throughout the subagent code is the code's gloss on
> it, not the table's text.

### I1 — Everything is a plugin; there is no privileged core to patch

**Mechanism.** 80 plugins behind one entry-point group. The agent loop
(`agent-loop`), the adapters, the tool registry, the session store and the
persistence backend are all rows. A profile is a list of them; a patch addresses
one by id.

**Where it bites.** Two shipped session backends (JSONL, Turso) satisfy one
`SessionPersistence` Protocol, and four consumers that used to reach for
`store.root` and rebuild a filename now ask `locate()` — which is allowed to
answer `None` (`persistence/protocol.py`). That is I1 paying for itself: a
second backend was addable without breaking four call sites.

**Where it is imperfect.** `ph-app` is not a plugin — the CLI, the TUI and the
daemon are a host. The invariant is about the *harness*, not the shell around it.

### I2 — Registrations *and acquired resources* are effects that unwind

**Mechanism.** `Context.add_disposer` / `Context.effect` (§2.5), the three
claiming helpers (§3.2), and `Running.add_disposer` for the intersection case (a
registration whose lifetime is *both* the registering row's and the agent
layer's).

**How ownership is decided.** `owner_for(scope)` answers "whose lifetime does a
registration made *now* belong to", in three cases: an explicit `scope=` wins; the
**activation scope** when a row's `apply` is running (this is the fix P6-12
landed); otherwise this context (`context.py`). The third branch is a
widening in I2's sense, and it is the one branch that *fails open* — so it
**warns**, because "a silent fallback would make the one path that still outlives
its owner both invisible and unmeasurable".

**Where it bites hardest.** The resource half. A child's worktree, its kernel
subprocess, its jobs and its lease are all effects of scopes, so a `dispose()`
anywhere in the tree releases everything beneath it without a cleanup list. The
ordering rule (children before own effects) is what makes tombstone-style
handlers work at all, and is why P6-28's retention had to become
retain-by-default-and-withdraw rather than retain-on-failure.

**How it is enforced, not just intended.** `tests/test_registration_ownership.py`
walks every module and requires every scoped method and provider slot to be
classified; the behavioural gate mounts a row, registers through it, unmounts,
and asserts **the seam's own context did not grow an effect**. A lint additionally
refuses `subprocess.Popen` and `tempfile.mkdtemp` outside the seams, "because the
fiftieth plugin author will not have read §4.9" (`resources.py`).

**Where it is imperfect.** Outside state needs to be idempotent. Should a change
take place in an external database, or an email get sent out, those cannot be
undone as the session may repeat those actions after the harness has unwound.
The invariant is about the *harness*, not the external state.

### I3 — Model-visible means logged

**Mechanism.** `Session.derive_messages()` (`session/session.py`) projects
the ordered *surface* into the LLM history. Anything reaching a model request must
be exactly that projection.

**Enforced at runtime, not only in tests.** `agent-loop-invariant`
(`agent_loop/invariant.py`) registers a `llm/stream` listener that compares
`request.messages` against `session.derive_messages()` — *identically*, not
"equivalently", not "a superset" — and raises `ModelVisibleNotLoggedError` naming
both counts. It is **prepended**, so it sees what the loop built before any
middleware can rewrite it, and it holds a request when it is session-bound and
names no other purpose.

**Why the equality is strict.** "That equality is what makes the session log a
complete trace: anything the model saw can be audited, replayed and offloaded
only if the log is the sole source it came from."

**The opt-out is a closed set.** A request declares a `purpose`; `refine` was
added to it deliberately, "because opting out of the model-visible-means-logged
invariant has to be a declaration the union can enumerate".

**Where it bites.** Compaction, offloading and context injection cannot take a
shortcut through the request builder — they must append events, or the assertion
fires on the next step.

**And enforcement is a property of the profile, which `ctx.invariants` is what
says.** Every invariant here is enforced by a *row*, and every row is optional —
so "pH enforces I3" was never true of pH, only of a deployment that mounted
`agent-loop-invariant`, and nothing said which those were. P6-01's registry
(`seams/invariants.py`) makes the enforced set enumerable and gives `ph doctor` a
section for it, so a deployment promising less than this document is
distinguishable from one promising all of it. It draws one distinction the
report depends on: an invariant enforced *inline* (I3, which refuses on the path
it governs) is reported as **enforced inline** and never as "holds", because
there is no state between requests to poll and a check answering `[]` there
would read as verified while verifying nothing. The pollable ones — I6's
`harness_state.json`, the surface fold, the tool view cache, and I2's scope tree
— answer about now.

### I4 — The log is append-only; the surface is what changes

**Mechanism.** `SurfaceManager` (`session/surface.py`) keeps an ordered
list of surface *node* seqs. A surface-eligible event
(`user/message | assistant/message | tool/result`) **must** declare a
`surface_op`: either `append`, or `replace(start, end)` which swaps a range of
nodes for one new node. The log keeps every event; only the *derivation* changes.

**The validation is a plan, not a mutation.** `validate_next(event)` plans the
transition without committing it, so "a rejected append leaves the surface
untouched — a partially mutated surface would be unrecoverable"
(`surface.py`). `Session.append` validates **before** the push
(`session.py`).

Four rules the planner enforces:

- **Contiguity** — `event.seq` must equal the expected index.
- **Eligibility** — a non-eligible type may carry neither `surface_op` nor
  `source_event_seqs`; an eligible one must carry a `surface_op`.
- **Provenance** — a replacement must cite **every** shadowed node in
  `source_event_seqs`, with no duplicates, all earlier than itself
  (`surface.py`).
- **Tool-result narrowing** — a `tool/result` replacement may rewrite exactly one
  node and may change **only content**, never the call id or the error identity,
  so "a 'spill' cannot silently rewrite what the model is told happened"
  (`surface.py`).

**And the fold is reproducible offline.** `fold_surface(events)` replays a stored
log through the same rules, so an external reconstructor must reach the same
nodes (`surface.py`).

**Areas affected.** `transcript()` and `derive_messages()` deliberately differ:
the transcript keeps what compaction shadowed, because a human scrolling back
should still see what they said. Two projections, one log.

Nothing in `Session` can rewrite an event: `data` is frozen through
`freeze_json_value`, `seq` is assigned from the log length, and a reentrant
append during publication raises rather than assigning a seq inside another
event's publication.

#### Readers of the log have one shape

Because the log is append-only, `session.seq` is an exact invalidation key: if
the log has not grown, no fold over it can have changed. Every reader in pH is
therefore the same three things — a **state**, a **step** that folds one more
event into it, and a **canonical replay** over the whole prefix that the steps must
agree with — and `SessionFoldCache` (`session/folds.py`) is that shape with the
cache attached: `compute` is the replay, `extend` the step, and the key is `seq`.
Six rows hold their projection in one: the subagent roster, goals, schedules,
the harness state, the limits, the sandbox refusals.

Three readers live *inside* `Session` and are its own, and `Session.stale()` checks
exactly those three against their replays (I6): the **events snapshot** against the
log's length, the **surface order** (`SurfaceManager`) against `fold_surface`, and
**`derive_messages`** against a fresh derivation over the canonical nodes. A fourth
kind is there and is *not* checked — the latest-of-type folds (`_LatestFold`, behind
`request_header()`, `request_context()` and `latest()`), whose replay `fold_latest`
exists but is compared nowhere. Named as the known exception rather than counted
among the checked, because the claim this section makes is precisely that a reader
has a replay its steps are held to.
They are inside rather than cached beside for the reason `folds.py` gives for
keeping everyone else *out*: a fold attached to a live session is monotonic in that
log and cannot answer for an earlier prefix, and forking at a boundary and viewing a
stored file both need exactly that.

#### "Must agree with" is checked, ahead of time and at runtime

The shape above rests on a contract `folds.py` states and could not enforce: the
cached function must be a **pure fold of the prefix**, and `extend` resumed from a
prefix must equal the replay of the whole. A function that also reads the clock,
the filesystem or a mutable table changes its answer without the log growing, and
`seq` cannot see it. That is the same *shape* of assurance as the freeze fast path
refused in §8 — correct only insofar as it is tested — at lower stakes: a wrong
fold serves a stale projection, where a wrong freeze admits unvalidated data into
the log. It is now checked from both ends.

**Ahead of time, in tests** (`ph.testing.folds`). The laws only mean something to
a process that holds the whole log and therefore every prefix of it, which is the
process that appended it — never the deployment reading one back. `prefix_of`
rebuilds a prefix as a real `Session` by re-admitting its events, and
`check_fold_laws` runs three laws over the prefixes from `first_live_seq` up,
every one of them until the log is long and evenly sampled past that:

- **Determinism** — two replays of one log agree, and a replica admitting the same
  events under the same header agrees with them. This is what catches a fold
  reading *process* state rather than log state.
- **No side effects** — the fold does not grow the log, and a profiler watches its
  C calls for the clock, the filesystem and random sources. A **heuristic by name**,
  so it catches the common impurities and cannot prove their absence; an
  environment read reaches no C function it knows.
- **Batching invariance** — extending from any prefix to any later one equals the
  replay of the later one; walking the prefixes one step at a time, which is what a
  cache read after every append does, never leaves the replay; and extending over
  an empty slice changes nothing. The first step that diverges is named with its
  seq and event type, because that event is the one the two paths read differently.

All six consumers are held to these, over logs their own services wrote — the
third law to the four that carry an `extend`, since schedules and the subagent
roster have no step to hold to their replay. (Deterministic and
batching-invariant are the same two laws LangGraph asks of a `DeltaChannel`
reducer, and enforces in a docstring; pH runs them.)

**At runtime, as an invariant** (`SessionFoldCache.stale`, one row per cache:
`goal-fold-cache`, `schedule-fold-cache`, `subagent-fold-cache`,
`sandbox-fold-cache` in `ph-base`, the limits and harness rows with their own
bundles). The check lives on the cache class for `ToolRegistry.stale_views`'
reason — `_entries` is its own secret, and a check written from outside is one a
rename disables silently — and each owning row declares it through
`contribute_fold_cache`, so a profile that drops a seam drops the claim with it.
It compares **only entries whose key still matches**: an entry the log has
outgrown is the ordinary state of a cache between reads, not drift, and a cached
session no longer live is skipped because there is no log left to fold. What is
left is exactly the pair `seq` cannot see — a fold that answered from outside the
log, and a *reader* that mutated the value it was handed and so poisoned the entry
for everyone after it.

**A row each rather than one row for all six**, for the reason `skills-invariant`
gives about not folding itself into `tools-invariant`: different folds with
different failure stories, and a report naming *which* cache drifted is what a
person can act on.

Neither end closes `_LatestFold`, which remains the named exception above: it is
inside `Session`, not a `SessionFoldCache`, and `fold_latest` is still compared
nowhere.

One reader is deliberately outside this: `TuiEventAdapter` has a state and a step,
but **no canonical replay to be checked against**, and cannot have one. Two of the
inputs its steps consume are not in the log — `Frame.view` is a card the daemon
rendered beside the event, and `Frame.live` selects between two deliberately
different foldings (a replay reads `assistant/message` and ignores the chunks a
live fold streams). `replay()` is therefore a *different* fold, not a check on
`apply`, and the same fact is why the adapter stays off `SessionFoldCache`, whose
one requirement is a pure fold of the log prefix. `build_trajectory` is a full fold
per open for a different reason: `fork_boundaries` runs over the whole log, and an
incremental version would have to reproduce it over a slice.

**The TUI over a socket is a reader too, and it is now the same reader.** A remote
front end keeps a real `Session` and `admit`s each wire event into it as it arrives
— the daemon's own `seq` and `time`, held to the seed path's contiguity rule and the
surface's validation, published to observers through the tail `append` uses. It
used to keep a list and rebuild `Session(seed=…)` from it on every screen open,
re-validating the whole log and growing a `session/end-seed` marker the daemon's
log does not have. The generation from the `session/new` reply keys the
mirror at construction, so `cursor_of` gives the same answer on both ends,
`stale()` runs on the client, and any `SessionFoldCache` consumer can be pointed at
the mirror unchanged. Because the mirror is kept incrementally, a frame it cannot
take desynchronises it silently where the rebuild used to refuse loudly — so
`diverged` is a fact the screen path asks about rather than a counter that only
quietens a log line.

### I5 — Seams have three roles

**Mechanism and its limits: §3.** The design statement is Definition + Provider +
Consumer. The implementation adds two things the statement does not:

1. **Not every seam has a provider.** 27 keys, 6 provider slots. The rest are
   contribution tables, waterfall answerers, or plain services.
2. **Verification is four layers deep** (§3.3) — static Protocols, two runtime
   capability probes, one registration-time obligation check, and tree-walking
   gate tests that refuse an unclassified seam.

**Areas affected — the benefits.** Containment tiers, code runtimes,
compaction engines and persistence backends are provider swaps. The `worktree`
tier is a row; removing it degrades every agent to `shared` with a recorded
reason rather than breaking a consumer. The subagent seam holds *many* providers
by name because "run a child" legitimately has several answers in one deployment.

**And one benefit dsh's notes state that this document had not: the prefix
cache.** A consumer addresses a capability by its *signature* — `read(path=…)`,
`ctx.fs.root_for(agent)` — so a provider swap changes what is behind the path
without changing the path, and the bytes the model has already seen stay
byte-identical (A12). That is why swapping a workspace tier or a filesystem
backend does not invalidate a cached prefix, and why "same interface, different
storage" is a cost argument and not only an architectural one. P6-03 measured
the property it protects: without stabilization, every request's cacheable
prefix is its predecessor in full — 74.9 % of everything sent — and it is
compaction's surface `replace`, not any provider swap, that breaks it.

**Whether a provider *bills* it as a prefix is a second question, and on
Anthropic it is opt-in** (P6-13). The OpenAI-compatible routes cache implicitly;
Anthropic's does nothing without `cache_control` markers, so until they were sent
every stability decision above paid off on one wire and nowhere on the other.
`adapters/anthropic.py` spends the four markers a request may carry on `tools`,
`system`, and **two quantized message checkpoints** — quantized because a marker
on the newest message sits at a different index in every request, so it is
written once and read never. `tests/test_prompt_cache.py` holds that as a number
rather than a shape: the second request of a session reads a prefix, and so does
P4-03's replayed summarize call.

**Media is the same argument at a different layer.** A route declares what it
takes (`accepts`), how big (`max_attachment_bytes`), how many pixels it will
*accept* and how many it can actually *use* — and `media-degrade` answers all of
it above every adapter, so a text-only route silently receiving an image is not a
shape this tree can express. `ph.llm.dimensions` reads an image's size out of its
header with **no dependency at all** — no Pillow, no ImageMagick, no `ffmpeg` —
which is what turned the pixel ceiling from a knob wired to nothing into a
refusal and a warning (P7-03). Over `max_image_edge` nothing is sent; over
`usable_image_edge` everything works and `attachment/oversized` says the surplus
is being uploaded every turn and discarded.

**`ctx.uploads` is the one cache in the tree that is deliberately not in the
log.** A provider's file API takes bytes once and returns a handle; that handle is
a *prediction* with an expiry, and an append-only log cannot retract one. So the
`(provider, digest) → handle` map lives under `$PH_CACHE`, one file per entry,
reconstructible from blobs the store still holds — while the *fact* that bytes
left the machine for a named provider is appended as `attachment/uploaded`,
because that is what a person auditing their data comes for. A handle the
provider stops honouring is detected mid-request, forgotten, and raised as
`FILE_EXPIRED`, which `llm-retry` already treats as transient: the retry rebuilds
against a fresh upload rather than losing the turn.

**The decline/fail distinction (§3.4) is part of this invariant.** A seam must
be able to say *"I could not serve this, and here is the code for why"* without
that being an error — otherwise every optional tier becomes a startup failure.

---

## 8. Known gaps

Stated here rather than left to be discovered, per the codebase's own rule.

| Gap | Status |
|---|---|
| `sandbox-local` ships `bwrap` and Seatbelt, both verified against a real kernel — the blind Seatbelt profile was wrong in four places a Mac found (P6-40). Landlock is unwritten | P6-04, P6-40; Landlock open |
| `install_lifecycle` (signal handling, grace period, self-`SIGKILL`) has **no production caller**; `ph daemon` calls bare `anyio.run` | unwired |
| `AgentCancelCause.kind` declares `hook` and `legacy`; neither is ever constructed | dead vocabulary |
| `TurnEndReason(kind="interrupted")` is never constructed as a dataclass — it reaches logs only as repair's wire payload | dead vocabulary |
| `SubagentRun.dispose` has no production caller; a model `delete()` leaves the parent-scope effect registered (it no-ops via re-entry) | dead handle |
| `Session.first_live_seq` is written and read only by a test | unused |
| `ph attachments gc` is cited as precedent in two docstrings but **does not exist** | doc drift |
| `DowngradeReason` has one member and one producer; tier-driven narrowing records none (§6.6) | incomplete |
| `_enforce`'s containment refusal (a scope outside the parent's) has **no test**; the no-scope branch does | untested |
| Tool-call limit with `exit: "error"` — the "one failed `tool/result`, not a turn stop" reading is traced, not tested | untested |
| Kernel-namespace rehydration: `kernel/snapshot` / `kernel/restored` exist as types but nothing wires them into the resume path | unwired |
| **A staged attachment is not durable.** `session/stage` puts a reference on the root's shared tray so an upload becomes a chip in every attached composer — composer state, shared but transient, deliberately not in the log. A daemon restart loses the tray with the blob still in the store, so a person who dropped a file and then reopened has to drop it again | by decision, P7-06 |
| **No chunked upload.** `attachment/put` carries the whole file in one frame, capped at 5 MiB; a larger one is refused by name rather than truncated. Both doors say so — the browser gets a sentence, `/attach` gets `attachment_too_large` | P7-06, stated |
| **A screen a third-party row contributes is invisible to a remote front end.** `ScreenDefinition.build()` cannot travel, so `screens/list` is intersected with what the client can draw and everything else is silently not offered. Every screen pH ships is drawable; a row's own is not | P5-15, gated by `test_a_screen_this_build_cannot_draw_is_not_offered`; P7-07 closes it |
| **`repaired()` closes a turn parked on a human as interrupted.** The ask is in the log (`approval/asked` with no decision) and `pending_approvals`/`pending_questions` fold it, but nothing reads that fold on *resume* — only `AskDesk.join` re-poses, across an attach. Leaving the turn open is unsafe until something does, because the model's `tool_use` block is still unanswered — it rides the *assistant message*, and a message carrying one with no matching `tool_result` is a log several providers reject (`tool/call` is not surface-eligible and no provider sees it; since P7-15 it is written after the gate, so a parked turn has none and repairs as `TOOL_NOT_STARTED`) | P5-13, deferred; gated by `test_a_turn_parked_on_a_human_is_closed_as_interrupted` so it cannot go quietly false |
| **`freeze_json_value` re-copies an already-frozen tree, and the fast path is refused on purpose.** Every container is rebuilt on every pass, so a re-admitted tree pays a second structural copy — 0.95 µs for a streamed chunk, **232 µs for a 500-node tool result**, scaling with node count rather than bytes. Four paths pay it: `Session.admit` per wire event on a remote front end, `Session(seed=…)`, `resume_session` on every rehydrate, and `SessionStore.fork`. A validation-only pre-walk that returns the input when it is already frozen measured at about half the cost. It is **not built**, because this function is where A1 is enforced: a second route through the gate is correct only insofar as its "already frozen?" predicate is, nothing in the type system tells a `MappingProxyType` over a frozen tree from one over a live dict, and a predicate like that can only be validated by enumerating hostile shapes in tests — the wrong kind of guarantee for a gate. Waiting on a design that is **structurally** safe: a frozen tree that carries its own proof (a distinct wrapper the walker recognises by identity), or a freeze idempotent by construction | deferred by decision; measured and documented in `session/json.py`, P6-44 |
| `LlmRuntime.register_adapter` uses no claiming helper and takes no `scope=` — the one provider slot outside the ownership sweep | documented in place |
| **Per-service isolation is implemented** (`isolate:` on a row, §2.7 — dsh's `isolate.fs`), but a realm's provider cannot be **swapped mid-session**: dsh's example — "we start processing sensitive data, so we swap the filesystem to read-only and the agent still sees the same filesystem" — has no pH spelling. `fs.rebase` is a `claim_slot`, so the *root* can change under a stable `ctx.fs`; read-only is a `permissions-fs` rule or the `readonly-scratch` workspace kind, both fixed at mount. The mechanism a swap needs (`claim_slot` releasing to a new claimant) exists; no row drives it and nothing has asked for it. **Deferred by decision, not by omission**: it will be built when a use case shows up, and the use case will decide whether the answer is a provider swap, a rule, or a new realm | deferred until a use case |

---

## 9. Reading the source

- **Start at** `packages/ph-core/src/ph/cordis/context.py`. Everything else is
  built on the `Context` tree.
- **Then** `ph/bundles/base.yaml` — 55 rows is the whole default harness, in
  order, with comments explaining each.
- **Application code states the contract and the invariants; tests state why.**
  A docstring under `src/` says what a thing does, what it refuses, and which
  invariants it is holding — and stops there. The reasoning this codebase records
  at length — the measurement, the defect that motivated a design, the
  alternative that was rejected and on what grounds — lives in the test that
  holds the behaviour to account, next to an assertion that fails if the
  reasoning stops being true. **The index into that reasoning is the filename,
  not a pointer**: `seams/workspace_git.py` is held by `tests/test_workspace_git.py`,
  and so on throughout. A `src/` docstring does not name its test file, because a
  pointer is one more place for the two to drift apart while the convention
  already answers the question.
- **So read the tests for the argument.** "The first draft did X, which broke Y"
  is still the most valuable sentence in the codebase; it is now in
  `packages/*/tests/`, where X is a test that fails.
- **Cite a file and a symbol, never a line number.** The same drift argument, one
  step further: a path survives every edit to the file, a function or class name
  survives everything but a rename, and `file.py:1093-1110` is wrong the next time
  anybody inserts a paragraph above it. This document carried 118 such citations
  and the plans another 79; enough of them had rotted — pointing at blank lines,
  at the middle of an unrelated docstring, at a `return None` — that the numbers
  were removed wholesale rather than repaired. Where the surrounding prose does
  not already name what is being pointed at, name it: `family_reach` in
  `seams/subagents.py` still finds it after the function has moved twice.
- **`ph events`** prints the live producer/consumer matrix.
- **`ph doctor --profile <name>`** mounts a profile and reports what it actually
  composed — the effective containment tier, what the file rules reach, what runs
  model code.
