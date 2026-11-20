# ph-rlm

*Code Mode: the model writes Python, and every call that the program makes is a
governed, logged tool call.*

This package implements the design of prime-agent on the seams of pH: the RLM
loop, non-blocking delegation, the nuclear-family boundary, the Continual
Harness, and the doctrine prompts. The whole package follows one rule (§6.8):
take the *semantics*, not the runtime. `ph_rlm.kernel` is the CPython subprocess
of pH itself, and [`ph-runtime-guest`](../ph-runtime-guest/) is its other half.

```bash
phern --profile rlm --provider llama --model <model> --mode tui
phern --profile rlm-stable --provider llama --model <model> --mode tui   # gates on
```

The package registers a **bundle**. `phern` composes the `rlm` profiles through
the `ph.bundles` entry-point group, and `ph_app` does not import this package.
An install without this distribution is not offered those profiles.

## What the model sees

The model sees one callable. Under `tools.mode: code`, the registry presents the
reserved transport `run_code` as **`python`**, described in the words of pH:
plain Python in an interpreter that belongs to the agent and persists across
calls. It is not IPython or Jupyter, so there are no magics. A shell is reached
as `await tools.bash(...)`. The result keeps the section order of prime-agent
(`stdout / stderr / result / traceback`), except that pH interleaves the two
output streams in arrival order. Everything else arrives in the `globals()` of
the cell as namespaces. Each call on a namespace is one `call` frame that leaves
the kernel and goes back through the whole tool pipeline:

```python
files = await tools.glob(pattern="src/**/*.py")  # every registered tool
child = await rlm.run("review this diff", name="reviewer")
await agent_message.send("done", receiver_role="parent")
answer = await websearch(query="…")  # a Python skill
```

That is the reason for the package. Prime-agent reached the host over an
`ipykernel.Comm`. No `tools/pre-execute` listener, no approval and no call limit
ever observed that channel. There is no such channel here. The governed path is
not a convention. It is the only path that exists. One cell that makes forty
dispatches produces forty `tool/code-dispatch` records, forty permission
evaluations, and forty independent offload decisions. One oversized `tools.read`
is therefore spilled, and its siblings pass through unchanged.

## The rows

Each row is a listener on a seam that already exists. The bundle
(`src/ph_rlm/bundle.yaml`) turns the rows on.

| Row | What it adds |
|---|---|
| `code-runtime-python` | The runtime: one CPython child per agent, fd 3 as the framed channel, and resource limits applied in the child |
| `rlm-presentation` | Renames the transport to `python`, and states how a settled cell reads |
| `rlm-bindings` | The `rlm` namespace: `rlm_run`, `rlm_wait`, `rlm_list_subagents`, `rlm_delete_subagent` |
| `rlm-subagent-provider` | `rlm()` as a `ctx.subagents` provider. Admission is logged, and the handle returns before the child answers |
| `rlm-messaging` | `agent_message` and `agent_observe`: `agent_message_send`, `agent_message_list_agents`, `agent_observe_list`, `agent_observe_get` |
| `rlm-prompt` | The doctrine, plus the volatile facts (depth, cwd, family, workspace tier) as a `context()` snapshot that is outside the cached prefix |
| `rlm-harness` | The Continual Harness and `/refine` |
| `rlm-harness-invariant` | Asserts that `harness_state.json` still equals the `harness/*` fold |
| `rlm-kernel-snapshot` | The per-variable `kernel/snapshot` events that justify `persistence: namespace` |
| `rlm-skills-python` | A skill directory that is also an installable package: installed into the kernel venv and imported at boot |
| `rlm-context-loader` | A corpus that the model can query: `context_search`, `context_chunks`, `context_head`. **Ships disabled** |

The bundle also does the following:

- It sets `tools.mode: code`.
- It sets `jobs.concurrency.subagent` to 8.
- It puts the containment ladder at `advisory` for the root agent and `worktree`
  for its children.
- It layers the git-worktree tier, with a restore point before each code run
  (`workspace-checkpoint`), `/workspaces` and `/revert`.
- It **disables `subagent-task`**. `rlm_run` is the same capability, in the
  shape that the rest of this bundle is designed around. Two ways to delegate in
  one prompt make the model guess which one the tools were built for.

## The one command

`/refine` refines the Continual Harness, or rolls a refinement back
(`[--global] [--show] [--rollback <id>] [instructions]`).

It is a **command, not a tool**, and that is deliberate. The user asks for a
refinement, and a route through a model turn would put the model in the log as
the one that decided it. The harness state is a fold over `harness/refined`
events and not a file. For that reason, a fork inherits the harness as of its
boundary, and a rollback can be derived from the event that made the change.
`harness_state.json` is written for humans and never read back. That
arrangement lets a projection drift, which is why the invariant row exists.

Four checks run before anything durable is written, and each refusal is recorded
*on* the event:

- The reference must resolve. It is probed in the runtime that the model uses.
- The call pattern is **rendered, never accepted** from the model.
- A `scope: global` entry goes through `ctx.approval`.
- The base doctrine is not editable.

## Including it, and adjusting it

`rlm` is `tui` plus this bundle. `rlm-stable` adds `ph-stabilize` as a required
bundle (`rlm` layers it only as an optional one) and turns on one row that each
bundle ships off: `tool-todo` and `rlm-context-loader`. To layer this bundle
onto something else, name the bundle. To change one row, patch it:

```bash
phern --patch '{id: code-runtime-python, config: {python: host}}' --profile rlm -p "…"
```

```yaml
# $PH_HOME/profiles/rlm.yaml: a longer per-cell CPU budget and a skill directory
- id: code-runtime-python
  config:
    python: managed
    cpuSeconds: 120
    addressSpaceBytes: 2147483648
    maxLogBytes: 65536
    maxValueBytes: 65536
    maxSnapshotBytes: 16777216
    skills: ["acme-websearch"]

- id: rlm-skills-python
  config:
    paths: ["~/.ph/skills", "./.ph/skills"]     # last source wins, by name
```

| Row | Config | Default |
|---|---|---|
| `code-runtime-python` | `python` (`managed` \| `host`), `interpreter`, `cpuSeconds`, `idleCpuSeconds`, `addressSpaceBytes`, `maxLogBytes`, `maxValueBytes`, `maxSnapshotBytes`, `bootTimeoutSeconds`, `shutdownGraceSeconds`, `cancelGraceSeconds`, `probeSeconds`, `skills` | `managed`, unset, 30 s, 2 s, 2 GiB, 64 KiB, 64 KiB, 16 MiB, 30 s, 5 s, 2 s, 1 s, none |
| `rlm-subagent-provider` | `maxDepth`, `maxConcurrent`, `answerPreviewChars` | `2`. `4` in the bundle (the row default is `null`, which means no cap). `240` |
| `rlm-messaging` | `maxMessageChars`, `maxPending`, `rateCapacity`, `rateRefillSeconds`, `observeMaxMessages` | `16384`, `20`, `3`, `1.0`, `40` |
| `rlm-harness` | `autoRefine`, `turnsBetweenRefinements`, `cooldownMinutes`, `maxPerKind`, `maxRefinements`, `conversationChars`, `maxTokens` | on, `25`, `20`, `12`, `5`, `80000`, `32000` |
| `rlm-kernel-snapshot` | `inlineBlobMax` | `65536` |
| `rlm-bindings` | `provider` | `rlm-child` |
| `rlm-skills-python` | `paths` | Empty. No skills, so the row costs nothing until a deployment configures one (I7) |
| `rlm-context-loader` | `corpus`, `sources`, `minChars`, `maxMatches` | `context`, none, `0` (`rlm-stable` sets `200000`), `200` |
| `tools-code-mode` (ph-core) | `maxDispatchesPerRun`, `maxSubagentSpawnsPerRun`, `maxParallelSubCalls` | `256`, `32`, `10` in this bundle |

The table only lists these three settings of `code-runtime-python`, so here is
what each one does:

- **`idleCpuSeconds`** is the CPU time that a namespace may use while no cell is
  running. Only a runaway thread or detached task reaches this limit, and the
  guest then ends its own process.
- **`probeSeconds`** is how often the guest loop of an open run is probed, or
  `null` for never. It is a measurement interval, not a deadline. `phern doctor`
  reports the round trip as how far behind that loop runs.

Three other settings need more explanation before you change them:

**`python`** decides what model code can reach:

- `managed` builds `$PH_CACHE/runtime-venv`. The venv holds `ph-runtime-guest`,
  `dill` and the Python skills, *and nothing else*.
- `host` is the interpreter that pH itself runs on. It is fast and needs no `uv`
  and no network, and the test suite uses it. It also puts `ph-core`, pydantic,
  and Textual on the `sys.path` of the child. That reaches no live objects,
  because a different process shares nothing. But it is a wider surface, which
  is why it is not the default.
- `$PH_RUNTIME_PYTHON` (or `interpreter:`) names an interpreter directly, for a
  deployment whose skills need a particular build.

**`cpuSeconds` is per cell, not per kernel.** `RLIMIT_CPU` is cumulative over a
process, and this process is persistent. The limit is therefore armed again at
each run, from the CPU time that the process has already used. A breach raises
an exception that derives from `BaseException`, so a cell cannot get past it
with `except Exception`.

**`maxDepth` and the two concurrency caps set the fan-out posture.** The two
caps are `maxConcurrent` for each parent and `jobs.concurrency.subagent` for the
host. A child beyond `maxDepth` is refused. Children past `maxConcurrent`
**queue in admission order** and are not refused. That is why the default of the
row itself is `null`, and the bundle, not the row, chooses a number.

## Limitations, and things that are deliberate

- **`rlm.run` does not return an answer.** It returns an admission handle. The
  reply of the child arrives on a later turn as an ordinary inbox message. A
  model that waits for the answer waits forever, so the doctrine states this as
  a rule and not a hint. A program whose next step needs the result calls
  `rlm.wait`, which waits for one child to end and returns the value or the
  answer of that child.
- **A denial ends the run.** `RunStopped` derives from `BaseException`, so a
  program cannot catch a refusal and go around it, for example with a retry on a
  different path or a fallback to `subprocess`. A *failure* (`ToolFailed`) is
  for the program to handle. A refusal is not (C3). The same applies to a budget
  (C4).
- **The namespace does not survive a dead kernel.** A kernel process that dies
  is replaced. The next run gets a fresh kernel, and its output starts with a
  reset notice. The session is degraded, but the harness is not dead.
- **`kernel/snapshot` is per variable, not per namespace**, and `patch` is
  deliberately not implemented. `dill` output is not byte-stable across
  processes in the way that a QuickJS heap image is, so per-variable digests are
  what keeps log growth linear. The HMAC tag on a blob is *provenance, not
  secrecy*. It stops a blob from another session from being unpickled into this
  session. It is not a defense against a hostile writer on the filesystem.
- **`find_models` is absent.** It would need a model catalog on `ctx.llm`, which
  does not exist. A discovery call that could only answer "I don't know" is
  worse than no call. A child with no `model` inherits the model of its parent.
- **The message rate limit is backpressure, not policy.** The tool body raises
  it, so the program can handle it and retry. Under C3, a *denial* would cost
  the model its whole program over four messages in a second. The **family
  boundary** is the opposite. It is a `ctx.tools.guard`: it is deny-only, it
  runs last, and no later listener can permit what it denied.
- **Delivery is always steer.** A message reaches the target at its next *step*,
  not its next turn. A busy target reports `queued` and not `delivered`. Those
  are different facts, and a sender can act on the difference.
- **Orphans are journaled, not left to chance.** `SIGKILL` runs no cleanup, and
  POSIX re-parents children to PID 1. Every spawn is therefore journaled, and
  every pH start sweeps the journal. A stray is killed only when its start token
  still matches the pid. Where the token cannot be read, the stray is **reported
  and not killed**. A correct "there may be a stray" is better than a confident
  kill of some other process.

## Tests

`tests/` has 26 modules. Three of them need a note:

- `test_protocol_mirror.py` is the contract between this package and
  `ph-runtime-guest`. The two halves of the fd-3 protocol are written twice on
  purpose, because the guest must not import the harness. The test compares the
  required and optional field set of every frame, and it compares the
  truncation marker byte for byte. `PROTOCOL_VERSION` and three more constants
  are declared one time, in the guest, and the test checks that the host imports
  them.
- `test_governance_gate.py` runs real cells in a real kernel against the
  **shipped** profile, not against hand-picked rows. It tests the claim that the
  whole package depends on: one cell is one tool call, but forty writes are
  forty governance evaluations.
- `test_conformance.py` reverses the usual arrangement. It enumerates the
  vocabulary of the protocol itself and the namespaces of the mounted registry
  itself, so a frame type with no test is a failure and not a silent gap.
