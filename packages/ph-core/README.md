# ph-core

*Everything a pH profile is made of: the plugin framework, the append-only log,
the ReAct loop, the capability seams — and the seventy-four rows that provide
them.*

There is no privileged core to patch (invariant I1). What this distribution
ships is the *vocabulary* a profile is written in — a context that holds
services, an event bus, scopes that unwind what they created — plus one row per
thing pH can do, each addressable by id from a YAML document. The agent loop is
a row. The session store is a row. The filesystem is a row. So is the tool
registry that decides whether a tool call is even allowed to happen.

```bash
ph --profile headless -p "hello"                 # ph-base + the fake adapter
ph config --profile base                         # every knob every row accepts
ph doctor --profile base                         # what actually activated
```

`ph-core` is a dependency of every other package in this workspace and depends
on none of them.

## What is in here

| module | what it owns |
|---|---|
| `ph.cordis` | the plugin meta-framework subset (D1): `Context`, service keys, the four event dispatch modes, scopes whose disposal unwinds every registration (I2), and the YAML profile loader |
| `ph.session` | the append-only event log, the derived model surface (`derive_messages`), the human `transcript()`, and the folds over both |
| `ph.llm` | the provider-neutral vocabulary, the stream assembler, the `ctx.llm` seam, plus the `fake` and `replay` adapters |
| `ph.agent` / `ph.agent_loop` | the agent handle and its inbox; the ReAct driver, mounted as a row like anything else |
| `ph.tools` | the registry, the governed pipeline (`tools/pre-execute`, `tools/post-execute`, guards), the batch scheduler, and Code Mode's transport and generated SDK |
| `ph.seams` | one module per capability seam: the Protocol, the local provider, and what it refuses |
| `ph.system_prompt` | prompt assembly — cached `section`s, post-cache `context()` snapshots, and `AGENTS.md` discovery |
| `ph.persistence` | JSONL and Turso backends, checkpoints, leases, lineage and crash repair |
| `ph.commands` | the slash commands ph-core itself owns |
| `ph.bundles` | `base.yaml`, `headless.yaml`, and the `ph.bundles` entry-point group other distributions register into |
| `ph.paths` | the three roots (`$PH_HOME`, `$PH_CACHE`, `$PH_RUNTIME`) and their resolution rules |
| `ph.testing` | builders, stubs and fixtures a test stands a profile up with. Nothing shipped imports it |

## The two bundles it ships

**`ph-base`** (`src/ph/bundles/base.yaml`) is the shared core of every pH
profile: the log, the loop, the tool registry, every capability seam with its
local provider, the built-in tools, durability, resilience and the runtime
invariant rows. **`ph-headless`** adds one row — the scripted `llm-fake`
adapter — so a one-shot or a scenario test can script a conversation without
touching code.

Both are *paths*, not entry points, because `ph-app` can import them directly.
Every other bundle in this workspace is **discovered** through the `ph.bundles`
entry-point group, which is what lets `ph-app` compose the `rlm` profile without
depending on `ph-rlm`:

```toml
[project.entry-points."ph.bundles"]
rlm = "ph_rlm:BUNDLE"
```

## The tools the model gets

Registered by rows, so a profile decides which of them exist at all.

| row | tools | note |
|---|---|---|
| `tool-fs` | `read`, `write`, `edit`, `glob`, `grep` | thin shells over `ctx.fs`, so the policy gates and the workspace root apply to any second editing tool too |
| `tool-bash` | `bash` | goes through `ctx.shell`, which is what `sandbox-local` confines |
| `tool-attach` | `attach` | puts an image, audio file, video or PDF from the workspace in front of the model's own eyes. **Registers nothing without an attachment store** |
| `tool-ask-user` | `ask_user` | **ships disabled**: a question with nobody to answer it spends a turn on nothing. `tui.yaml` arms it |
| `subagent-task` | `task` | blocking delegation. **Registers nothing until a `ctx.subagents` provider is mounted** — a tool named in every prompt and refused on every call teaches the model a capability the deployment does not have |
| `skills-progressive` | `skill` | the catalog goes in the prompt; a body only when the model asks for it by name (G9) |

## The commands a person can type

These are **commands, not tools**: a person asks the harness directly, it costs
no model turn, and the log records `command/run`/`command/done` rather than the
model having decided something the user decided.

| command | row | what it does |
|---|---|---|
| `/sandbox` | `sandbox-commands` | show what confined commands may reach, and change it without a restart |
| `/workspaces` | `workspace-commands` | list, export, merge or remove the branches agents left behind in this session |
| `/revert` | `workspace-revert` | restore this agent's workspace to a per-run checkpoint |
| `/autonomous` | `autonomous` | work toward a goal until its gates pass or a budget stops it |

## Adjusting it from a profile

Three layers, applied in this order — and a patch replaces the targeted row's
**whole** config rather than merging into it, so a row's effective value is
always one layer's and readable in one place:

1. the shipped documents — `ph-base`, then whatever the profile layers;
2. your overlay, `$PH_HOME/profiles/<name>.yaml`;
3. drop-ins pH wrote on your behalf, `$PH_HOME/profiles/<name>.d/*.yaml`, in
   name order (this is where `/sandbox allow …` keeps its decisions);
4. `--patch`, this run only, same grammar as a profile document.

A patch entry is `{id: …, config: {…}}` to reconfigure, `{id: …, disabled:
false}` to arm a row a bundle ships off, `{id: …, remove: true}` to drop one, or
`{insert: [...]}` to add one. **A list in row config is replaced, not merged** —
an overlay restating one field of a route must restate the whole route entry.

```bash
ph --patch '{id: fs, config: {root: /tmp/scratch}}' --profile tui --mode tui
ph --patch '{id: tool-ask-user, disabled: false}' --profile headless -p "..."
ph --dump-config --profile llama          # the composition, before anything runs
ph config --row containment --profile tui # one row's knobs, defaults and what the profile set
```

### The knobs most deployments touch

| row | config | default |
|---|---|---|
| `agent-loop` | `maxParallelToolCalls` | `10` |
| `tools` | `mode: native \| code` | `native` |
| `tools-code-mode` | `maxDispatchesPerRun`, `maxSubagentSpawnsPerRun`, `maxParallelSubCalls` | `256`, `32`, `10` |
| `fs-local` | `root`, `ignore` | the session cwd |
| `sandbox-policy` | `defaultMode: read-only \| workspace-write` | `read-only` (`tui.yaml` sets `workspace-write`) |
| `sandbox-allow` | `paths`, `network.{mode,hosts}` | an allowlist over the package indexes and documentation hosts in `DEFAULT_HOSTS` |
| `containment` | `tier`, `childTier`, `strict` | **unset** — which is not `advisory`: unset means nobody chose, so a tier provider a profile layered is used |
| `workspace-lifecycle` | `access`, `provision` | `write` |
| `jobs-local` | `concurrency` (per job kind) | unset |
| `llm-retry` | `maxAttempts`, `baseDelayMs`, `maxDelayMs` | `3`, `500`, `20000` |
| `session-telemetry` | `enabled`, `path` | `enabled: false` in `ph-base` |
| `subagent-presets` | `presets` | empty — a menu, never a grant: selecting a preset never widens what the parent itself holds |
| `skills-progressive` | `paths` | **empty on purpose** — scanning a well-known directory would make "install a skill" mean "drop a file somewhere" (I7) |
| `autonomous` | `maxContinuations`, `maxTurns`, `maxTokens`, `timeoutMs` | `3`, `12`, `80000`, `1800000` |
| `subprocess-local` | `scrub`, `keep`, `maxOutputBytes` | `8388608` |
| `tool-attach` | `maxBytes` | `33554432` |

**An entry carrying both `id:` and `name:` is a row, not a patch** — that is the
rule that decides which of the two a document line is. So swapping a *provider*
is a removal and an insertion, not a rename; consumers never learn which one
answered either way (I5):

```yaml
# $PH_HOME/profiles/tui.yaml — keep the log in Turso instead of JSONL
- id: session-persistence
  remove: true
- insert:
    - id: session-persistence
      name: session-persistence-turso
```

Giving the existing id a new `name:` in place looks like it should work and does
not: it appends a *second* row, and the mount then refuses with
`service "session_persistence" is already provided`.

## Seams that ship with no provider, deliberately

`ph-base` mounts the *definition* of several seams and no backend, because these
have genuinely different answers per deployment and a harness that shipped one
would have made the choice for you:

- **`code_runtime`** — nothing runs model-written code until `ph-rlm` mounts
  `code-runtime-python` (or a profile mounts `code-runtime-stub`).
- **`subagents`** — no child-agent provider, which is why `subagent-task`
  registers no tool in `ph-base`.
- **`compaction`** — the seam records and replaces; *when* and *what to say* are
  `ph-stabilize`'s `compaction-summarize`.
- **`uploads`** — mounted with no uploader; each adapter row registers its own,
  so a profile with no file API sends every byte inline.

A profile that layers nothing here simply never compacts, never delegates and
never runs code. That is the plain harness, and it is a supported posture.

## Limitations, and things that are deliberate

- **No Textual, Rich, Typer, `ph_app`, aiohttp, textual-serve or jinja2
  imports** — `tests/test_layering.py` fails the build on any of them. A front
  end is a consumer of this package, never the other way round.
- **No real model wire ships here.** `llm-fake` and `llm-replay` are for wiring
  work and tests; Anthropic, Google and the OpenAI-compatible route are rows in
  `ph-app`, so a headless deployment that wants a real provider layers one of
  its profiles.
- **`sandbox-local` is Linux (bwrap) and macOS (Seatbelt).** It probes both
  claims at mount — a write outside the workspace must fail, and a `CONNECT`
  through the egress shim must reach the proxy and be refused — and on a host
  that cannot confine it says so in `ph doctor` and declines rather than
  pretending.
- **`permissions-fs`-style path rules are not here** — they are `ph-stabilize`,
  and they bound *seam-mediated* access only. A model-authored `open(path, "w")`
  inside a code cell never fires an intent; what bounds that is the sandbox rung
  below the rules (E9, N1).
- **Nothing watches the filesystem.** `AGENTS.md` is re-read as a post-cache
  snapshot each turn, which is what makes an edit take effect in the turn after
  it; nothing else polls.
- **Row order carries no load semantics.** Activation is service-availability
  driven, so a row waits for whatever it injects regardless of where it sits.
  The grouping in `base.yaml` is for readers.

## Tests

`tests/` — 75 modules, the largest suite in the workspace. Three worth knowing
about, because they enforce rules rather than behaviour:

- `test_layering.py` — the forbidden-import rule above;
- `test_keys.py` — every `ctx.provide(...)` in this package has a typed key in
  `ph.keys`, and every key has a provider, held against each other in both
  directions;
- the invariant rows (`agent-loop-invariant`, `session-invariant`,
  `tools-invariant`, `skills-invariant`, `scope-invariant`) are checked *at
  runtime*, not only in CI — a harness that can only prove these in a test
  cannot prove them about your session, and `ph doctor` reports which hold.
