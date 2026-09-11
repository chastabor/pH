# ph-rlm

*Code Mode: the model writes Python, and every call that program makes is a
governed, logged tool call.*

Prime Agent's design — the RLM loop, non-blocking delegation, the nuclear-family
boundary, the Continual Harness, the doctrine prompts — implemented on pH's
seams. The rule the whole package follows (§6.8) is: take the *semantics*, not
the runtime. `ph_rlm.kernel` is pH's own CPython subprocess and
[`ph-runtime-guest`](../ph-runtime-guest/) is its other half.

```bash
ph --profile rlm --provider llama --model <model> --mode tui
ph --profile rlm-stable --provider llama --model <model> --mode tui   # gates on
```

The package registers a **bundle**, so `ph-app` composes the `rlm` profiles
without depending on this distribution — and an install without it is simply not
offered them.

## What the model sees

One callable. Under `tools.mode: code` the registry presents the reserved
transport `run_code` as **`ipython`**, with prime-agent's wording ported
verbatim and its result layout (`stdout / stderr / result / traceback`) kept, so
a model trained against that surface finds the surface it knows. Everything else
arrives in the cell's `globals()` as namespaces whose every call is one `call`
frame out of the kernel and back through the whole tool pipeline:

```python
files = await tools.glob(pattern="src/**/*.py")  # every registered tool
child = await rlm.run("review this diff", name="reviewer")
await agent_message.send("done", receiver_role="parent")
answer = await websearch(query="…")  # a Python skill
```

That is the whole argument for the package. Prime Agent reached the host over an
`ipykernel.Comm`, which no `tools/pre-execute` listener, no approval and no call
limit ever observed. There is no such channel here: the governed path is not a
convention, it is the only path that exists. One cell making forty dispatches
produces forty `tool/code-dispatch` records, forty permission evaluations, and
forty independent offload decisions — so one oversized `tools.read` is spilled
while its siblings pass through untouched.

## The rows

Each is a listener on a seam that already exists; the bundle
(`src/ph_rlm/bundle.yaml`) is what turns them on.

| row | what it adds |
|---|---|
| `code-runtime-python` | the runtime: one CPython child per agent, fd 3 as the framed channel, resource limits applied in the child |
| `rlm-presentation` | the transport renamed to `ipython`, and how a settled cell reads |
| `rlm-bindings` | the `rlm` namespace — `rlm_run`, `rlm_list_subagents`, `rlm_delete_subagent` |
| `rlm-subagent-provider` | `rlm()` as a `ctx.subagents` provider: admission logged, the handle returns before the child answers |
| `rlm-messaging` | `agent_message` and `agent_observe` — `agent_message_send`, `agent_message_list_agents`, `agent_observe_list`, `agent_observe_get` |
| `rlm-prompt` | the doctrine, plus the volatile facts (depth, cwd, family, workspace tier) as a post-cache `context()` snapshot |
| `rlm-harness` | the Continual Harness and `/refine` |
| `rlm-harness-invariant` | asserts `harness_state.json` still equals the `harness/*` fold |
| `rlm-kernel-snapshot` | the per-variable `kernel/snapshot` events that earn `persistence: namespace` |
| `rlm-skills-python` | a skill directory that is also an installable package, installed into the kernel venv and imported at boot |
| `rlm-context-loader` | a queryable corpus — `context_search`, `context_chunks`, `context_head`. **Ships disabled** |

The bundle also sets `tools.mode: code`, raises `jobs.concurrency.subagent` to
8, puts the containment ladder at `advisory` for the person and `worktree` for
their children, layers the git-worktree tier with `/workspaces` and `/revert` —
and **disables `subagent-task`**, because `rlm_run` is the same capability in
the shape the rest of this bundle is designed around, and two ways to delegate
in one prompt makes the model guess which one the tools were built for.

## The one command

`/refine` — refine the Continual Harness, or roll a refinement back
(`[--global] [--show] [--rollback <id>] [instructions]`).

A **command, not a tool**, and deliberately so: a refinement is something the
human asks for, and routing it through a model turn would put the model in the
log as having decided it. Harness state is a fold over `harness/refined` events
rather than a file, which is what makes a fork inherit the harness as of its
boundary and a rollback derivable from the event that made the change.
`harness_state.json` is written for humans and never read back — which is
exactly the arrangement that lets a projection drift, hence the invariant row.

Four checks run before anything durable is written, and each refusal is recorded
*on* the event: the reference must resolve (probed in the runtime the model
actually uses); the call pattern is **rendered, never accepted** from the model;
a `scope: global` entry goes through `ctx.approval`; and the base doctrine is
not editable.

## Including it, and adjusting it

`rlm` is `tui` plus this bundle. `rlm-stable` adds `ph-stabilize` and arms the
rows both bundles ship off. To layer it onto something else, name the bundle —
or patch a single row:

```bash
ph --patch '{id: code-runtime-python, config: {python: host}}' --profile rlm -p "…"
```

```yaml
# $PH_HOME/profiles/rlm.yaml — a longer per-cell CPU budget and a skill directory
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

| row | config | default |
|---|---|---|
| `code-runtime-python` | `python` (`managed` \| `host`), `interpreter`, `cpuSeconds`, `addressSpaceBytes`, `maxLogBytes`, `maxValueBytes`, `maxSnapshotBytes`, `bootTimeoutSeconds`, `shutdownGraceSeconds`, `cancelGraceSeconds`, `skills`, `sweepOrphans` | `managed`, 30 s, 2 GiB, 64 KiB, 64 KiB, 16 MiB, 30 s, 5 s, 2 s, none, on |
| `rlm-subagent-provider` | `maxDepth`, `maxConcurrent`, `answerPreviewChars` | `2`, `4` in the bundle (`null` — no cap — as the row default), `240` |
| `rlm-messaging` | `maxMessageChars`, `maxPending`, `rateCapacity`, `rateRefillSeconds`, `observeMaxMessages` | `16384`, `20`, `3`, `1.0`, `40` |
| `rlm-harness` | `autoRefine`, `turnsBetweenRefinements`, `cooldownMinutes`, `maxPerKind`, `maxRefinements`, `conversationChars`, `maxTokens` | on, `25`, `20`, `12`, `5`, `80000`, `32000` |
| `rlm-kernel-snapshot` | `inlineBlobMax` | `65536` |
| `rlm-bindings` | `provider` | `rlm-child` |
| `rlm-skills-python` | `paths` | empty — no skills, so the row costs nothing until a deployment configures one (I7) |
| `rlm-context-loader` | `corpus`, `sources`, `minChars`, `maxMatches` | `context`, none, `0` (`rlm-stable` sets `200000`), `200` |
| `tools-code-mode` (ph-core) | `maxDispatchesPerRun`, `maxSubagentSpawnsPerRun`, `maxParallelSubCalls` | `256`, `32`, `10` in this bundle |

Three knobs are worth understanding before changing them:

**`python`** decides what model code can reach. `managed` builds
`$PH_CACHE/runtime-venv` holding `ph-runtime-guest`, `dill` and the Python
skills *and nothing else*; `host` is the interpreter pH itself runs on — fast,
needs no `uv` and no network, which is why the suite uses it, and also what puts
`ph-core`, pydantic and Textual on the child's `sys.path`. That reaches no live
objects (a different process shares nothing) but it is a wider surface, and it
is why it is not the default. `$PH_RUNTIME_PYTHON` or `interpreter:` is the
third answer, for a deployment whose skills need a particular build.

**`cpuSeconds` is per cell, not per kernel.** `RLIMIT_CPU` is cumulative over a
process and this process is persistent, so the limit is re-armed at each run
from the CPU already consumed. Exceeding it raises from `BaseException`, so a
cell cannot `except Exception` its way past it.

**`maxDepth` and the two concurrency caps are the fan-out posture.** A child
beyond `maxDepth` is refused; children past `maxConcurrent` **queue in admission
order** rather than being refused, which is why the row's own default is `null`
and the bundle — not the row — picks a number.

## Limitations, and things that are deliberate

- **`rlm.run` does not return an answer.** It returns an admission handle; the
  child's reply arrives on a later turn as an ordinary inbox message. A model
  that waits for the answer waits forever, which is why the doctrine states it
  as a rule rather than a hint.
- **A denial ends the run.** `RunStopped` derives from `BaseException` so a
  program cannot catch a refusal and route around it — retry with a different
  path, fall back to `subprocess`. A *failure* (`ToolFailed`) is the program's
  to handle; a refusal is not (C3). The same applies to a budget (C4).
- **The namespace does not survive a dead kernel.** A child that dies is
  replaced and the next run gets a fresh kernel prefixed with a reset notice —
  a degraded session rather than a dead harness.
- **`kernel/snapshot` is per variable, not per namespace**, and `patch` is
  deliberately unimplemented: `dill` output is not byte-stable across processes
  the way a QuickJS heap image is, so per-variable digesting is what actually
  keeps log growth linear. The HMAC tag on a blob is *provenance, not secrecy* —
  it stops a blob from another session being unpickled into this one; it is not
  a defence against a hostile filesystem writer.
- **`find_models` is absent.** It would need a model catalogue on `ctx.llm`,
  which does not exist; a discovery call that could only answer "I don't know"
  is worse than none. A child with no `model` inherits its parent's.
- **The message rate limit is backpressure, not policy.** It raises from the
  tool body — the program's to handle and retry — because under C3 a *denial*
  would cost the model its whole program over four messages in a second. The
  **family boundary** is the opposite: a `ctx.tools.guard`, deny-only, run last,
  and not re-permittable by any later listener.
- **Delivery is always steer.** A message reaches the target at its next *step*,
  not its next turn. A busy target reports `queued` rather than `delivered`,
  because those are different facts and a sender can act on the difference.
- **Orphans are journalled, not hoped away.** `SIGKILL` runs no cleanup and
  POSIX re-parents children to PID 1, so every spawn is journalled and `fsync`ed
  and every pH start sweeps. A stray is killed only when its start token still
  matches the pid; where the token cannot be read, it is **reported and not
  killed** — an honest "there may be a stray" beats a confident kill of
  something else.

## Tests

`tests/` — 26 modules. `test_protocol_mirror.py` is the contract between this
package and `ph-runtime-guest`: the two halves of the fd-3 protocol are written
twice on purpose (the guest must not import the harness), and that test compares
`PROTOCOL_VERSION`, every frame's required and optional field set, and the
truncation marker byte for byte. `test_governance_gate.py` runs real cells in a real kernel against
the **shipped** profile — not hand-picked rows — to pin the claim the whole
package rests on: one cell is one tool call, but forty writes are forty
governance evaluations. `test_conformance.py` inverts the usual arrangement and
enumerates the protocol's own vocabulary and the mounted registry's own
namespaces, so an untested frame type is a failure rather than a silence.
