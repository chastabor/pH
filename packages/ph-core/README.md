# ph-core

*Everything that a pH profile is made of: the plugin framework, the append-only
log, the ReAct loop, the capability seams, and the seventy-four rows that
provide them.*

There is no privileged core to patch (invariant I1). This distribution ships the
*vocabulary* that a profile is written in: a context that holds services, an
event bus, and scopes that unwind what they created. It also ships one row for
each thing that pH can do. A YAML document addresses each row by id. The agent
loop is a row. The session store is a row. The filesystem is a row. The tool
registry, which decides whether a tool call is allowed to happen at all, is also
a row.

```bash
phern --profile headless -p "hello"                 # ph-base, the stabilize bundle, and the fake adapter
phern config --profile base                         # every setting that every row accepts
phern doctor --profile base                         # the rows that activated
```

`ph-core` is a dependency of every other package in this workspace except
`ph-runtime-guest`, and it depends on none of them.

## What is in here

| Module | What it owns |
|---|---|
| `ph.cordis` | The subset of the plugin meta-framework (D1): `Context`, service keys, the four event dispatch modes, scopes whose disposal unwinds every registration (I2), and the YAML profile loader |
| `ph.session` | The append-only event log, the derived model surface (`derive_messages`), the human `transcript()`, and the folds over both |
| `ph.llm` | The provider-neutral vocabulary, the stream assembler, the `ctx.llm` seam, and the `fake` and `replay` adapters |
| `ph.agent` / `ph.agent_loop` | The agent handle and its inbox. The ReAct driver, which is mounted as a row like everything else |
| `ph.tools` | The registry, the governed pipeline (`tools/pre-execute`, `tools/post-execute`, guards), the batch scheduler, and the transport and generated SDK of Code Mode |
| `ph.seams` | One module for each capability seam: the Protocol, the local provider, and what the seam refuses |
| `ph.system_prompt` | Prompt assembly: cached `section`s, post-cache `context()` snapshots, and `AGENTS.md` discovery |
| `ph.persistence` | The JSONL and Turso backends, checkpoints, leases, lineage and crash repair |
| `ph.commands` | The slash commands that ph-core itself owns |
| `ph.bundles` | `base.yaml`, `headless.yaml`, and the `ph.bundles` entry-point group that other distributions register into |
| `ph.paths` | The three roots (`$PH_HOME`, `$PH_CACHE`, `$PH_RUNTIME`) and their resolution rules |
| `ph.testing` | The builders, stubs and fixtures that a test uses to stand up a profile. Nothing shipped imports it |

## The two bundles it ships

**`ph-base`** (`src/ph/bundles/base.yaml`) is the shared core of every pH
profile: the log, the loop, the tool registry, every capability seam with its
local provider, the built-in tools, durability, resilience and the runtime
invariant rows. **`ph-headless`** adds one row, the scripted `llm-fake` adapter.
A one-shot run or a scenario test can then script a conversation without a
change to code.

Both are *paths*, not entry points, because `ph_app` can import them directly.
Every other bundle in this workspace is **discovered** through the `ph.bundles`
entry-point group. That group lets `ph_app` compose the `rlm` profile without
importing `ph_rlm`:

```toml
[project.entry-points."ph.bundles"]
rlm = "ph_rlm:BUNDLE"
```

## The tools the model gets

Rows register these tools, so a profile decides which of them exist at all.

| Row | Tools | Note |
|---|---|---|
| `tool-fs` | `read`, `write`, `edit`, `glob`, `grep` | Thin shells over `ctx.fs`, so the policy gates and the workspace root also apply to any second editing tool |
| `tool-bash` | `bash` | Goes through `ctx.shell`, which is what `sandbox-local` confines |
| `tool-attach` | `attach` | Shows the model an image, audio file, video or PDF from the workspace. **Registers nothing without an attachment store** |
| `tool-ask-user` | `ask_user` | **Ships disabled**: a question with nobody to answer it spends a turn on nothing. `tui.yaml` arms it |
| `subagent-task` | `task` | Blocking delegation. **Registers nothing until a `ctx.subagents` provider is mounted.** A tool that is named in every prompt and refused on every call teaches the model a capability that the deployment does not have |
| `skills-progressive` | `skill` | The catalog goes in the prompt, and a body goes there only when the model asks for it by name (G9) |

## The commands a user can type

These are **commands, not tools**. A user asks the harness directly, and the
request costs no model turn. The log records `command/run` and `command/done`.
It does not record that the model decided something that the user decided.

| Command | Row | What it does |
|---|---|---|
| `/sandbox` | `sandbox-commands` | Shows what confined commands may reach, and changes it without a restart |
| `/workspaces` | `workspace-commands` | Lists, exports, merges or removes the branches that agents left behind in this session |
| `/revert` | `workspace-revert` | Restores the workspace of this agent to a per-run checkpoint |
| `/autonomous` | `autonomous` | Works toward a goal until its gates pass or a budget stops it |

## Adjusting it from a profile

There are five layers, applied in this order. A patch replaces the **whole**
config of the row that it targets and does not merge into it. The effective
value of a row therefore always comes from one layer, and can be read in one
place:

1. the shipped documents: `ph-base`, then whatever the profile layers;
2. the rows of the host, `rows:` in `$PH_HOME/daemon.yaml`, and the flags of the
   daemon itself;
3. your named profile, `$PH_HOME/profiles/<name>.yaml`: it `extends:` a shipped
   profile and holds the `rows:` that differ from it;
4. the drop-ins that `/sandbox` wrote before S4,
   `$PH_HOME/profiles/<name>.d/*.yaml`. They are read until
   `phern profiles fold` folds them. `/sandbox` now records an override in the
   log of the session instead;
5. `--patch`, for this run only, in the same grammar as a profile document.

Each row declares what it `affects`: `environment`, `deployment` or
`presentation`. A layer that a user writes sets one kind. Layer 2 sets
deployment rows, and layers 3–5 set environment rows. A row that is set in the
wrong layer is refused by name. No profile sets presentation rows. A front end
hides what it does not draw.

A patch entry takes one of these forms:

- `{id: …, config: {…}}` reconfigures a row;
- `{id: …, disabled: false}` arms a row that a bundle ships off;
- `{id: …, remove: true}` drops a row;
- `{insert: [...]}` adds a row.

**A list in row config is replaced, not merged.** An overlay that restates one
field of a route must restate the whole route entry.

```bash
phern --patch '{id: fs, config: {root: /tmp/scratch}}' --profile tui --mode tui
phern --patch '{id: tool-ask-user, disabled: false}' --profile headless -p "..."
phern --dump-config --profile llama          # the composition, before anything runs
phern config --row containment --profile tui # one row: its settings, defaults, and profile values
```

### The settings that most deployments change

| Row | Config | Default |
|---|---|---|
| `agent-loop` | `maxParallelToolCalls` | `10` |
| `tools` | `mode: native \| code` | `native` |
| `tools-code-mode` | `maxDispatchesPerRun`, `maxSubagentSpawnsPerRun`, `maxParallelSubCalls` | `256`, `32`, `10` |
| `fs-local` | `root`, `ignore` | The session cwd |
| `sandbox-policy` | `defaultMode: read-only \| workspace-write \| danger-full-access` | `read-only` (`tui.yaml` sets `workspace-write`) |
| `sandbox-allow` | `paths`, `network.{mode,hosts}` | An allowlist over the package indexes and documentation hosts in `DEFAULT_HOSTS` |
| `containment` | `tier`, `childTier`, `strict` | **Unset**, which is not `advisory`. Unset means that nobody chose, so a tier provider that a profile layered is used |
| `workspace-lifecycle` | `access`, `provision` | `write` |
| `jobs-local` | `concurrency` (for each job kind) | Unset |
| `llm-retry` | `maxAttempts`, `baseDelayMs`, `maxDelayMs`, `countAllRetries` | `3`, `500`, `20000`, `false`. With `false`, only the retries of `llm-retry` itself spend its budget. `true` counts the retries of every row, including those of compaction |
| `session-telemetry` | `enabled`, `path` | `enabled: false` in `ph-base` |
| `subagent-presets` | `presets` | Empty. A preset is a menu, never a grant: to select a preset never widens what the parent itself holds |
| `skills-progressive` | `paths` | **Empty on purpose.** A scan of a well-known directory would make "install a skill" mean "drop a file somewhere" (I7) |
| `autonomous` | `maxContinuations`, `maxTurns`, `maxTokens`, `timeoutMs`, `tokenSources` | `3`, `12`, `80000`, `1800000`, `[own]`. Add `compaction`, `decision` and `children` to charge summaries, decision-model questions and subagents to `maxTokens` |
| `subprocess-local` | `scrub`, `keep`, `maxOutputBytes` | `8388608` |
| `tool-attach` | `maxBytes` | `33554432` |

**An entry that carries both `id:` and `name:` is a row, not a patch.** That
rule decides which of the two a document entry is. To swap a *provider* is
therefore a removal and an insertion, not a rename. In either case, consumers
never learn which provider answered (I5):

```yaml
# $PH_HOME/daemon.yaml: keep the log in Turso and not in JSONL (a host row)
rows:
  - id: session-persistence
    remove: true
  - insert:
      - id: session-persistence
        name: session-persistence-turso
```

To give the existing id a new `name:` in place looks correct, but it does not
work. The entry declares a *second* row with the same id, and the composition
refuses it with `row id "session-persistence" is already declared`.

## Seams that ship with no provider, deliberately

`ph-base` mounts the *definition* of several seams and no backend. These seams
have different answers in different deployments, and a harness that shipped a
backend would have made the choice for you:

- **`code_runtime`**: nothing runs model-written code until `ph-rlm` mounts
  `code-runtime-python` (or a profile mounts `code-runtime-stub`).
- **`subagents`**: there is no child-agent provider, which is why
  `subagent-task` registers no tool in `ph-base`.
- **`compaction`**: the seam records and replaces. `compaction-summarize` in
  `ph-stabilize` decides *when* to compact and *what to say*. Every profile that
  `phern` offers layers that bundle when it is installed, so a *profile*
  compacts. `ph-base` composed alone still does not.
- **`uploads`**: the row is mounted with no uploader. Each adapter row registers
  its own, so a profile with no file API sends every byte inline.

A profile that layers nothing here never compacts, never delegates and never
runs code. That is the plain harness, and it is a supported posture.

## Limitations, and things that are deliberate

- **No Textual, Rich, Typer, `ph_app`, aiohttp, textual-serve or jinja2
  imports.** `tests/test_layering.py` fails the build on any of them. A front
  end is a consumer of this package, and never the reverse.
- **No real model wire ships here.** `llm-fake` and `llm-replay` are for wiring
  work and tests. Anthropic, Google and the OpenAI-compatible route are rows in
  `phern`, so a headless deployment that wants a real provider layers one of the
  profiles of `phern`.
- **`sandbox-local` is Linux (bwrap) and macOS (Seatbelt).** It probes both
  claims at mount: a write outside the workspace must fail, and a `CONNECT`
  through the egress bridge must reach the proxy and be refused. On a host that
  cannot confine, it says so in `phern doctor` and declines, and does not
  pretend.
- **`permissions-fs`-style path rules are not here.** They are in
  `ph-stabilize`, and they bound *seam-mediated* access only. A model-authored
  `open(path, "w")` inside a code cell never fires an intent. The sandbox rung
  below the rules bounds that (E9, N1).
- **No row watches the filesystem.** `AGENTS.md` is re-read as a post-cache
  snapshot each turn, so an edit takes effect in the next turn. Nothing else
  polls.
- **Row order has no effect on loading.** Activation depends on which services
  are available, so a row waits for whatever it injects, wherever it sits. The
  grouping in `base.yaml` is for readers.

## Tests

`tests/` has 104 modules. It is the largest suite in the workspace. Three things
are worth knowing about, because they enforce rules and not behavior:

- `test_layering.py`: the forbidden-import rule above;
- `test_keys.py`: every `ctx.provide(...)` in this package has a typed key in
  `ph.keys`, and every key has a provider. The test checks both directions;
- the invariant rows (`agent-loop-invariant`, `session-invariant`,
  `tools-invariant`, `skills-invariant`, `scope-invariant`) are checked *at
  runtime*, not only in CI. A harness that can prove these only in a test cannot
  prove them about your session, and `phern doctor` reports which hold.
