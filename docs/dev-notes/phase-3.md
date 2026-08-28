# Phase 3 — The RLM bundle

**Status:** in progress. The runtime is complete and green; the RLM semantics layer is next.

**Gate so far:** `ruff` + `ruff format` + `mypy --strict` on `ph-core`, `ph-app`, `ph-rlm` and `ph-runtime-guest` + 559 tests (114 new), green.

Phase 2 gave pH a person to talk to. Phase 3 gives it Prime Agent's design — the
recursive loop, non-blocking admission, the nuclear-family boundary, the Continual
Harness — implemented on pH's seams. The rule the whole phase follows is §6.8's:
**take prime-agent's semantics, not its runtime.**

D19 is what makes that a rule rather than a preference. Prime Agent reaches its
host through an `ipykernel.Comm`, and pH's runtime has no such thing — so
`rlm/__init__.py` cannot run here at all. What C2 asked of a compat shim (that
every capability be reached through the governed path) is now true by
construction: the only guest→host channel is a `call` frame.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P3-01 | The fd-3 vocabulary, both sides, with a version field and a shared truncation marker | `ph_runtime/protocol.py`, `ph_rlm/kernel/protocol.py` |
| P3-02 | Host frame codec: shape-validate and **rebuild**, drop extra fields, never echo a non-numeric id, unsafe integers refused | `ph_rlm/kernel/codec.py` |
| P3-03 | Guest runner: `RLIMIT` application, the run loop, top-level `await`/`return`, capped streaming output, `display`, `cancel`, `shutdown` | `ph_runtime/{runner,cell,channel,limits}.py` |
| P3-04 | Die-with-parent per platform | `ph_runtime/lifecycle.py` |
| P3-05 | The provider: spawn through `ctx.effect()`, `boot`, `run`, `call`→pipeline→`reply`, reap in `finally`, reset notice after a kill | `ph_rlm/kernel/manager.py` |
| P3-06 | Orphan journal, `fsync`ed, swept at every start, start-token guarded | `ph_rlm/kernel/journal.py` |
| P3-07 | Runtime venv resolution and staleness | `ph_rlm/kernel/venv.py` |
| P3-08 | The guest package: binding proxies, skill wrapping, bootstrap | `ph_runtime/{proxies,skill,errors}.py` |
| P3-09 | `rlm-presentation`: the transport presented as `ipython`, prime-agent's description verbatim, its result layout, `IpythonToolDetails` | `ph_rlm/presentation.py`, `ph/tools/definition.py`, `ph/tools/registry.py` |
| P3-15 | `kernel/snapshot` and `kernel/restored`: per-variable, tagged, spilled, folded | `ph_rlm/snapshot.py` |
| P3-10 | The extension point, then the `rlm` namespace over it | `ph/tools/{code_mode,registry}.py`, `ph_rlm/bindings.py` |
| P3-11 | The `ctx.subagents` seam **and** the `rlm-child` provider: non-blocking admission, depth gate, status mirroring, usage attribution, terminal notices, tombstones | `ph/seams/subagents.py`, `ph_rlm/subagents.py` |
| P3-12 | `rlm-messaging`: the family guard, steer delivery, the `agent_message` and `agent_observe` namespaces | `ph_rlm/messaging.py`, `ph/seams/subagents.py` |
| P3-13 | Rehydration on address: a settled child is woken by a send | `ph/seams/subagents.py`, `ph_rlm/subagents.py` |
| P3-14 | `rlm-prompt`: doctrine, child doctrine, conditional delegation section, volatile facts as a `context()` | `ph_rlm/prompt.py` |
| P3-21 | The governance gate: C1-C4 and C7 against the shipped profile | `ph-rlm/tests/test_governance_gate.py` |
| P3-16 | The Continual Harness: the fold at both scopes, validation (H1/H2/H3/H5), apply, rollback (H6), the prompt section, `/refine`, the planner and auto-refine (H7) | `ph_rlm/harness/{state,service,planner,auto,__init__}.py` |
| P3-17 | `rlm-context-loader`: a corpus queried through three registered tools, metadata-only prompt section, the recipe as `context/loaded` | `ph_rlm/context_loader.py` |
| P3-18 | `rlm-skills-python`: `SKILL.md` discovery, editable install, import names to the kernel at boot, catalog section | `ph_rlm/skills.py`, `ph_runtime/runner.py` |

**Still to come:** P3-19, P3-20 and P3-22…P3-25 — the TUI code cell and subagent
panel, the profile bundle and its smoke run, the runtime conformance gate, the
fixture replay, and the trajectory view. P3-13's remaining half — passivation
across a *restart*, where the child's session is gone and has to come off disk —
is the daemon's (Phase 5); the in-process half landed here.

**P3-09's gate is met as of P3-12.** "SDK block lists the four namespaces" needed
all four to exist: `tools` from Code Mode itself, then `rlm`, `agent_message` and
`agent_observe` as claims by three separate rows. `test_bundle.py` asserts it
against the shipped profile, and also that none of the governed tool names is
offered a second time.

---

## Decisions taken inside Phase 3 so far

### 1. The protocol is written twice, and a mirror test is the only thing joining them

`ph_runtime.protocol` and `ph_rlm.kernel.protocol` do not import each other.
They cannot: the guest runs in `$PH_CACHE/runtime-venv`, and making it import
the host package would put the harness inside the process boundary that exists
to keep the harness out.

So `test_protocol_mirror.py` compares the version, the two frame vocabularies,
every frame's required and optional field set, and the truncation marker byte for
byte. It earned its keep on its first run: `protocol` is required on the guest's
`boot` but the host derived it as *optional*, because the field has a default.
The fix was to derive from the dump semantics instead — `encode` uses
`exclude_none=True`, so a field is omitted exactly when its value is `None`, and
a field with a non-`None` default is therefore always **sent**.

A related rule fell out of the same place: **the host owns every default.** `boot`
carries all five limits as required fields, so the guest has nothing to guess.
Two sets of defaults would mean two answers to "what is the log cap", and which
one applied would depend on which side was older.

### 2. `PH_RUNTIME_FD` exists because `pass_fds` does not renumber

`PROTOCOL_FD = 3` is the default, but the host passes the descriptor's *number*
in an environment variable rather than moving the descriptor to 3.
`subprocess.pass_fds` keeps an fd at the number it has in the parent, and
re-numbering it in the child needs a `preexec_fn` — which is unsafe in a threaded
parent. The channel itself is one end of a `socketpair`, because a pipe is
one-way and the guest has to both answer the host and call it.

### 3. A cell is wrapped in an `async def`, and every name it binds is declared `global`

Three requirements pull against each other: top-level `await`, top-level
`return`, and names that persist between cells. `PyCF_ALLOW_TOP_LEVEL_AWAIT`
gives the first but not the second — a bare `return` is a syntax error in module
code. Wrapping the body in an `async def` gives both, and then loses the third,
because a name assigned inside a function is local to it.

So `ph_runtime.cell` computes the set of names the cell's top level binds and
emits a `global` declaration for them. That is a real AST walk, including the
cases that are easy to forget: `import`, `with ... as`, `for`, `except ... as`,
`match` captures, walrus, and `del` (which raises for a name that is in globals
but not local). A trailing expression becomes the cell's value, so the
`stdout / stderr / result / traceback` result text reads the way prime-agent's
did.

The flag the plan named is therefore unnecessary, and is not used.

### 4. `RLIMIT_CPU` is cumulative, which a persistent kernel cannot use directly

Set once to `cpu_seconds`, it would give the whole kernel one budget for its whole
life — so the fortieth cell in a session would die on a limit the first cell
nearly spent. It is re-armed at each run from the CPU already consumed, which
turns the cumulative counter into the per-cell budget the caller means.
`CpuBudgetExceeded` derives from `BaseException` for the same reason a denial
does (C3): a cell that could `except Exception` past the limit would make the
limit advisory.

This is also the only mechanism that can interrupt a cell spinning in Python —
see §6.

### 5. A run reads its own frames, and owns the tasks that serve them

The first design was a long-lived reader task per kernel. It cannot work: a
background task needs a task group, a group entered when the kernel starts is
exited when the kernel closes, and those are different tasks — which anyio
refuses, correctly. The symptom was a `ClosedResourceError` from a cancelled
output drain surfacing *instead of* the failure being reported.

So `run()` opens a task group for the duration of one program: it reads frames
inline, starts one task per concurrent binding call, and drains the child's
stdout and stderr in the same group. Everything is entered and exited by one
task. `done` is therefore the last frame of a run, which is why the guest
snapshots **before** settling — and that ordering has a second benefit worth
naming: the namespace is durable before the model is told the cell finished,
which is the same rule as the checkpoint barriers (A4).

Two smaller things came out of this restructuring:

* **Writes are serialized.** `wait_writable` refuses two waiters on one socket,
  and there are genuinely three writers — the run loop, and a reply task per
  concurrent binding call. Contention arrived as `BusyResourceError`, my `_send`
  read it as the child having exited, and eight concurrent `tools.slow` calls
  were enough to "kill" a perfectly healthy kernel.
* **A restart is a fresh incarnation.** Every per-incarnation field is reset in
  `start()`. Reusing the previous boot event made the second start return without
  waiting for a `boot-ack`.

### 6. Cancellation: the plan's `SIGINT` was fatal as specified

The design was "the `cancel` frame plus `SIGINT`", the signal being for a cell
spinning in Python that never yields. But Python's **default** `SIGINT` handler
raises `KeyboardInterrupt` into whatever frame is executing, and when a cell is
`await`ing, that frame is `asyncio`'s own — so the signal killed the entire guest
rather than the cell. That is the common case, not the rare one: the test that
found it cancels a cell awaiting `asyncio.sleep`.

`SIGINT` is now installed with `loop.add_signal_handler`, which delivers it as an
ordinary loop callback and can never land inside library internals. Three
mechanisms then cover three genuinely different situations:

| The cell is… | What reaches it |
|---|---|
| awaiting a `reply` or a `sleep` | the `cancel` frame, or the `SIGINT` callback — both cancel the cell's task |
| spinning in Python | neither: the loop is starved. `SIGXCPU` from the per-run CPU budget *does* land, because the cell is executing bytecode |
| spinning, and cancelled by the user before the budget | nothing cooperative works, so the host escalates to `SIGKILL` after `cancel_grace` and restarts |

The third row is a deliberate loss: the namespace goes, and the result says so.
A wedged kernel that holds the turn open until it times out is worse.

### 7. The guest is a hostile peer, and the codec is where that is enforced

Model code holds fd 3 and can write anything onto it. Every inbound frame is
**rebuilt** from `INBOUND`, so a forged field cannot reach a handler that reads
`frame.get(...)`; a non-numeric id is dropped *before* there is an id to echo,
which is the ordering that matters — a codec that coerced `"1"` to `1` would let
the child choose which pending call a reply lands on. Nothing malformed raises,
because a decoder that raises is a decoder the child can crash the host with on
demand.

dsh's `hasUnsafeIntegerToken` is ported as a `json.loads(parse_int=...)` hook
rather than a text scan, which is strictly better: digits inside a string are not
mistaken for a number, and a number nested in an object is still checked. The
rule matters because pH's log is JSON that dsh's TypeScript tooling reads (Q2),
and past 2^53 a JS reader loses precision silently.

There is an end-to-end test for this where the *cell itself* writes forged frames
onto its own descriptor.

### 8. A model reads its own traceback, not pH's

The first cut reported `traceback.format_exc()`, which meant a failing cell
showed the model `ph_runtime/runner.py` frames. Those are identical on every
failure and they invite the model to debug the harness instead of its cell. The
traceback is now trimmed to frames at or after `<cell>`, and a `SyntaxError`
reports its message rather than a traceback at all — because the useful sentence
for a magic is `MAGIC_HINT`, which names `await tools.bash(...)`.

The magic *was* the bypass (feature map item 4): one shell command per cell that
no `tools/pre-execute` listener, no approval and no sandbox `confine()` ever saw.
Removing the mechanism closes the hole, so the error explains the governed route
instead of apologising for a missing feature.

### 9. Kernels are owned by the agent's scope, not by an event

`agent/disposed` was the obvious hook, but `emit` schedules an async listener
*without awaiting it* — so release would happen eventually rather than as part of
unwinding. The provider now notes each agent's scope from `agent/created` and
registers the kernel's disposal as an effect **on that scope**, which makes the
child process an artifact of the agent exactly like every other acquired resource
(F1). The provider also holds a root-level effect, so nothing leaks either way.

### 10. `patch` is in the vocabulary and deliberately not emitted

D17 allows a `bsdiff4` delta chain against an anchor **and** says to benchmark
first, because `dill` output is not byte-stable across processes the way a
QuickJS heap image is — memo ordering and `id()`-derived bytes move even when the
value does not. The plan's own fallback is "snap-only, still log-resident".

Per-variable digesting is what actually keeps growth linear, and that is what
shipped: an unchanged 200 MiB DataFrame emits nothing, because its digest did not
move. A delta chain over unstable bytes would add a re-anchoring policy and a
second failure mode for a gain nobody has measured. `SNAPSHOT_KINDS` names
`patch` so the vocabulary is ready when someone measures it.

### 11. The snapshot tag is provenance, not secrecy

HMAC-SHA256 keyed by the session id, so a blob from another session or a mangled
file fails verification instead of being unpickled — and `dill.loads` on
arbitrary bytes executes arbitrary code, so "instead of being unpickled" is the
point. Anyone who can write the log can write the tag too; this is not a defence
against a hostile filesystem writer. It is a defence against the mistake that
actually happens: a payload restored into the wrong session, or a half-written
file read as sound.

The event is appended **before** the blob is written (§4.9's write-ahead
ordering). A death between the two leaves an event naming a blob that is not
there, which `kernel/restored` reports as a failed variable; the opposite
ordering would leave a blob nothing references, which nothing would ever find.
The orphan case is swept at session open (F7).

### 12. A crash does not silently reconstitute the namespace

`_rehydrate` restores a fresh kernel from the log — the resume path — but skips
it after a kill. Those payloads describe a namespace the model has already been
told is empty, and restoring half of it behind the model's back would be worse
than the empty namespace the reset notice announces: the model would find some
names present and others missing with no way to tell which.

---

## Fixed along the way

* **`Session.append` could not mark a record `ignorable`.** The field existed on
  `SessionEvent` and nothing set it, so no event pH wrote carried it. Kernel
  state is exactly what it is for: an older pH reading a log this build wrote
  should skip these records, not refuse the log outright. Ignorability is a
  property of the *type*, not of the call, so `append` takes no `ignorable=`
  argument: it stamps `event_type in IGNORABLE_SESSION_EVENT_TYPES`. Two call
  sites appending the same type can therefore not disagree about it.
* **`SpillStore` was text-only.** A `dill` payload is bytes, and base64 through
  `save_text` would cost a third of its size on disk for nothing. `save_bytes`,
  `load_bytes` and `sweep` (for F7) are new.
* **The restore deadlocked on its own lock.** `_rehydrate` runs inside `run()`,
  which already holds the run lock; `anyio.Lock` is not reentrant, so going
  through a locking wrapper raised rather than waiting — and the symptom was a
  restored namespace surfacing as a failed cell. There is now one `_restore`,
  private and lock-free, and its docstring says why: a public `restore()` was an
  entry point nothing outside the class could safely use.
* **The reset notice was consumed by the wrong run.** The cell that called
  `os._exit` knows what it did; the *next* one is the one facing an empty
  namespace with no idea why.
* **The Phase 2 TUI vocabulary test caught the two new event types**, which is
  what it is for. `kernel/snapshot` is `RECORDLESS` — there is one per changed
  variable per cell, and rendering them would bury the conversation in its own
  bookkeeping. `kernel/restored` *is* rendered, but only when a variable failed
  to come back, because a clean restore is not news and a missing name is.

---

## Known gaps in what has landed

* **The model is not yet told about a failed restore.** The event is written and
  the TUI renders it, but "tell the model `{restored, failed}`" needs a
  model-facing notice, which belongs with the presentation layer (P3-09).
* **`python: "managed"` is the default and is not exercised by tests.** Building
  it shells out to `uv` and reaches the network. Every *decision* around it —
  staleness, refusals, where it is found — is tested; the build itself is not.
  The suite runs `python: "host"`, which is faster and wider: it puts `ph-core`,
  pydantic and Textual on the child's `sys.path`. That reaches no live objects,
  but it is why `host` is not the default.
* **Windows is unimplemented, not merely untested.** `pass_fds` is POSIX-only,
  and the Job Object work (D3's limits, F3's `KILL_ON_JOB_CLOSE`) has no code
  yet. The guest reports which die-with-parent mechanism it armed in `boot-ack`,
  and `_await_boot_ack` writes it at `INFO` alongside the applied limits, so a
  log says what was actually in force rather than what was requested.
* **The `display` frame has no producer *or* consumer.** The frame is defined on
  both sides, decoded, collected into `_ActiveRun.displays`, and now carried out
  through `CodeRunResult` and onto the transport's tool value — but the guest has
  no `display()` in the cell namespace, so nothing can send one, and nothing
  renders one until P3-19's code cell. The plumbing was finished at P3-09 because
  writing `IpythonToolDetails.attachments` is what surfaced that it went nowhere;
  the producer belongs with the widget that would show it.

  A related defect found the same way: `run_code` was dropping **`truncated`** as
  well, so a capped cell's card claimed the output was complete. That one had a
  real producer already, and now has a test driven through an actual capped
  stream — the hand-built-dict test beside it is what missed it.
* **The runtime is effectively asyncio-only.** `anyio.wait_readable` is
  backend-agnostic, but the guest uses `asyncio` directly (it must: it cannot
  depend on anyio), and D3 already fixes asyncio for the whole app because
  Textual requires it.

---

## P3-09, and the two core changes it needed

### An alias would not have worked

The plan called `ipython` "a presentation-level alias for `run_code`". Writing it
revealed that an alias is the wrong shape, because five separate places compare
against *a* transport name:

* `register` refuses the name, so nothing can occupy it;
* `create_execution` refuses a native call under `mode: code` unless the name is
  the transport (C6);
* the denial's route-back text tells the model which name to call instead;
* the `tools` namespace skips the transport, so a program cannot re-enter it;
* the code-only rule in the prompt names it.

Two names means five places deciding which one is authoritative, and any
disagreement is a model told to call something that does not resolve. So the
transport is **renamed in place** instead: `ctx.tools.present_transport` claims a
`TransportPresentation` on a layer exactly like `present_as` claims a mode,
`_build_view` swaps the definition's name and description as it resolves, and
`_View.transport_name` is the one answer all five now read. `run_code` stops
being visible; it does not become a second door.

`TransportPresentation` carries name, description, `output` and the two
presentation hooks — and deliberately *not* `parameters` or `execute`. A profile
that could replace the argument schema or the governed body would have replaced
Code Mode rather than renamed it, and the type says so by omission.

The name is unshadowable in both directions, which took two checks rather than
one: presenting over an existing tool is refused, and registering a tool under
the presented name is refused. Either check alone leaves the other order open.

### The result text keeps prime-agent's order, not its stream split

Prime Agent concatenates `stdout + "\n" + stderr + "\n" + result + "\n" +
traceback`. pH frames each write as it happens, so the two streams arrive
interleaved — and relocating stderr to the end would *misread* a cell that
prints, warns, then prints again. What is ported is the section order, which is
what the model parses; what is dropped is the split, which was an artifact of
buffering two pipes. Absent sections are dropped rather than left blank, because
three empty lines above a traceback is a puzzle rather than information.

One thing worth being explicit about: **a cell that raises is not `is_error`.**
A traceback is the model's to read and act on — that is the whole point of a
scratchpad — so `is_error` stays reserved for a refused or aborted tool *call*.
The details payload carries `status: "error"` so the card can still colour it.

### `CodeRunResult` was dropping the display frames

`_ActiveRun.displays` collected them and `CodeRunResult` had nowhere to put them,
so every `display` the runtime decoded was discarded at the seam. They are on the
result now, and deliberately not folded into `logs`: `logs` is for the model and
a base64 PNG in the model's text costs a fortune and says nothing. The card shows
a count.

### What is deferred, and why

**Streaming.** The plan lists `on_update` → `ToolExecutionUpdate`. There is no
such seam yet and its only consumer is P3-19's `CodeCellWidget`. Adding the hook
now would mean a contract with nothing to hold it honest, so it lands with the
widget.

**`present_as("code")` in the row.** The plan lists it under `rlm-presentation`.
It is not there: the mode is the bundle's `tools.mode`, and `present_as` is a
single-cell claim — two rows claiming it means the first disposal clears what the
second still wants. The row's docstring says so, so the next reader does not add
it back.

---

## P3-10's extension point, landed early

`tools-code-mode` hard-coded `bindings=(tools_namespace,)`. There was no way for
a row to contribute the `rlm`, `agent_message` or `agent_observe` namespaces, so
this had to land before P3-10 could start.

A row claims `ctx.tools.register_code_namespace(name, factory)` — keyed and
scoped like a tool registration, resolved into the same memoized view. The
property worth keeping from the first draft: **the run and the SDK prompt
section ask the factories the same question**, one `CodeBindingsRequest {scope,
bridge}`, the prompt with `bridge=None` — the same "describing, not bound"
convention `CodeBinding.dispatch` already used — so the block cannot list a
namespace the program could not reach, or omit one it can.

This landed first as a `tools/code-bindings` *waterfall*, and the same-day
cleanup pass replaced it: contribution-by-anonymous-listener meant a name
conflict could only be detected per run, so a deployment with two rows claiming
`rlm` booted green and then failed every cell with a `RuntimeError` the model
had to read. As a keyed claim the conflict fails at mount, like every other
name conflict in the codebase, and `tools` itself is unclaimable.

Two consequences that P3-11 and P3-12 will rely on, found by writing the tests
rather than by reading the plan:

1. **A binding's program-facing name is not the tool it dispatches to.**
   `rlm.run(...)` dispatches the tool `spawn_child`. A namespace cannot claim a
   bare global name like `run`, and does not need to: the SDK renders the
   namespaced form, `tool/code-dispatch-start` names the governed tool, and both
   are right. The first version of the test got this wrong by giving the
   contributed binding a plain closure — which passed, and proved nothing,
   because it never touched the bridge and therefore never met a budget.
2. **A contributed namespace goes through the same `DispatchBridge`**, so C2's
   records and C4's budgets (`counts_as_spawn` included) apply with no extra
   code. A namespace dispatching around the bridge would be bypassing the
   containment argument, not taking a shortcut — which is why the budget test
   exists at this level and not only for `tools`.

### `PromptSection` text may now be async

A prompt section that has to ask a seam a question could not, because `resolve()`
was synchronous — while `assemble()` around it was already a coroutine. The
alternative was for the SDK section to answer from a stale copy of the namespace
list, which is precisely the disagreement the waterfall exists to prevent.
`PromptText = str | Callable[[Context], str | Awaitable[str]]`, one `maybe_await`
in `resolve`. P3-14's workspace section and its `context()` snapshot need the
same widening.

---

## The cleanup pass, and what it turned up

A `/simplify` pass over the Phase 3 diff was meant to find duplication. Three of
its findings were not cosmetic, and they are recorded here because each one is a
claim this phase makes elsewhere in this document.

### C3 was described, not enforced

`§C3` says a denied dispatch fails the whole run. The host raised
`CodeRunFailure` from `_serve_call` and reported the tool call as failed — and
the cell kept running. A program only has to write

```python
try:
    await tools.write_file(path="/etc/passwd", content="...")
except BaseException:
    pass
Path("/etc/passwd").write_text("...")
```

to see the difference: the tool call is refused, the tool call is reported
refused, and the file is written anyway. Demonstrated, then fixed — the failure
branch in `_serve_call` now fires the run-scoped abort ladder (`cancel` frame →
`SIGINT` → `SIGKILL` after `cancel_grace`), which is the ladder C3's "run-scoped
abort fires" was always naming. `test_a_refused_cell_is_stopped_before_it_can_write_anyway`
asserts the file is *not* there and that the kernel survives to serve the next
cell.

The lesson is narrower than "we had a bug": a safety-surface ID that reads as a
description of behaviour is not the same as a test that the behaviour happens,
and the containment argument is exactly where that gap is expensive. C10 says
the program is a hostile peer; a hostile peer catches your exception.

### The host could be OOM-ed by the guest it contains

`_recv_line` accumulated into an unbounded buffer, so `os.write(3, b"x" * 10**10)`
from inside a cell grows the *host's* heap. Containment is not interception (I8),
but it does mean the contained side cannot decide how much of the container it
gets: `MAX_FRAME_BYTES = 64 MiB`, and a frame over it is a fault that kills the
kernel rather than the app. The same read loop also stopped re-scanning bytes it
had already scanned for a newline — a `_scanned` offset with a tail-only `find`,
which is why a 10 MiB `display` payload is no longer quadratic.

### `sweep` deleted other sessions' blobs

`KernelSnapshotPolicy.sweep(session)` walked every namespace the process had
seen and kept only the locators *that one session* referenced. For any other
session in the same process the reference set was empty, so everything it owned
looked unreferenced and was collected — a resumable session silently losing its
spilled namespace. Both halves now come from the same log: `namespaces_in(session)`
supplies the directories to visit, `referenced_locators(session)` the locators to
keep, and `test_a_sweep_leaves_another_sessions_blobs_alone` pins it.

Reconciliation between a log and a filesystem is only sound when both sides of
the comparison are derived from the same log. This is A11 (projection equals
fold) wearing different clothes.

### What the pass found that was merely duplication

* **The guest's 13 protocol `TypedDict`s had already drifted from `FRAME_FIELDS`.**
  `__required_keys__` said `namespaceId` was required where `FRAME_FIELDS` said
  optional, because `from __future__ import annotations` turns `NotRequired` into
  a string that `TypedDict` never interprets. Two declarations of one protocol,
  one of them silently inert — deleted, leaving `FRAME_FIELDS` as the only
  declaration, with the two load-bearing rationales moved onto its entries.
* **The spill naming rule lived in two places.** `ph-rlm`'s `_spill_path`
  re-derived the digest-and-sanitize rule that `SpillStore` also implements; a
  divergence would have written blobs the store could not find. `locator_for()`
  is now the single home of it, and the write-ahead ordering in §4.9 depends on
  the derived path being the real one.
* **Plugins kept `agent/created` side tables to answer "which session owns this
  agent?"**, shadowing the registry that already knew. `AgentRegistry.get()` and
  `.list()` are new; `ph_rlm.snapshot` dropped its `_sessions` dict.
* **Nine call sites spelled out `ToolExecutionInput(...)` to run one tool.**
  `ph.testing.run_tool` is new, and the three `ph-rlm` test modules now share one
  set of profile rows through `conftest.mounted_runtime` — a row set that drifts
  between modules tests three things while appearing to test one.

### CI was type-checking one package out of five

`[tool.mypy]` declared `packages = ["ph"]`, so `uv run mypy` checked 77 files in
`ph-core` and none in `ph_app`, `ph_rlm`, `ph_runtime` or `ph_stabilize` — the
whole of Phase 3 included. All four were already strict-clean, which is the only
reason this reads as a config fix rather than a backlog: extending the list cost
nothing and now covers 136 files. `packages = [...]` rather than `.` because the
test trees hold several modules named `conftest`, which mypy refuses as
duplicates.

Worth noting *why* it mattered here specifically: the protocol is deliberately
written twice (decision 1), and one of the two copies was the unchecked one.

### Measured, because "faster" is not a claim

The efficiency findings were real and the numbers are worth keeping:

| | before | after |
|---|---|---|
| cell touching an unchanged 1M-int list | 511 ms | 19.5 ms |
| cell creating a 10 MiB blob | 588 ms | 95.3 ms |
| cell printing 10,000 lines | 2100–2600 ms | 20.7 ms |
| cell whose trailing expression is huge | 77 ms wasted | bounded 251-char repr |

The first two are one change: the guest's snapshotter now tries C-`pickle`
before `dill` and holds an identity fast path for immutables, so an unchanged
variable is not re-serialized to be re-digested. The third is output coalescing
(8 KiB / 50 ms) instead of a frame per `print`. The fourth is `iterencode` with
a byte cap instead of encoding the whole value to find out it was too big.

### Skipped deliberately

* **The full D6 inversion** — making `code-runtime-python` inject
  `kernel_snapshots` rather than discovering it — because
  `test_runtime_integration.py` deliberately mounts the runtime *without* the
  snapshot row, and that separation is what lets the runtime be tested on its
  own. Noted as a follow-up rather than argued with.
* **Spawning through `ctx.subprocess`.** The seam has no `pass_fds` and no stdin
  control, and fd 3 is the whole mechanism.
* **Giving `CancelToken` an awaitable.** That is a `ph-core` change for one
  consumer; the polling is confined to one method and says so.

---

## The second cleanup pass (over P3-09 itself)

Four review agents over the P3-09/P3-10 diff. What changed, beyond the
waterfall-to-claim reshape above:

* **`reset` became a field instead of a sniff.** `cell_details` recovered "the
  kernel restarted" by `logs.startswith(RESET_NOTICE)` — parsing back a fact the
  kernel had held as a boolean and thrown into prose. Same failure class as the
  `truncated`/`displays` drop this diff had already fixed, plus one worse
  property: a cell whose first output was the literal marker text forged the
  flag. `CodeRunResult.reset` now carries it; the notice stays in `logs` for the
  model; a test proves both the real flag and the forgery's failure.
* **The transport's value is typed.** `run_code` returned a six-key dict read
  back through `.get()` with defaults — which is exactly why the dropped keys
  never failed a test. `CodeCellValue(WireModel)` is now constructed from the
  `CodeRunResult` and is the transport's `ToolOutput.schema`, so `render()`
  validates what it renders and a new result field is a visible type decision.
* **`register` stopped building a view per registration.** Its presented-name
  check called `view()`, whose cache the same registration invalidates one line
  later — O(N²) throwaway schema construction across a mount. It walks the layer
  cells instead.
* **The occupancy rule got a backstop where it can actually be enforced.** The
  claim-time checks in `register`/`present_transport` are scope-local snapshots:
  an agent-scoped presentation plus a later parent-scope registration slipped
  past both, and `_build_view` silently clobbered the tool. The view build —
  the one place every view resolves — now fails loudly on the contradiction,
  and the transport cell is a real claim (double-present raises; disposal is
  identity-checked).
* **The model is never told a name it has not seen.** The transport's own
  argument error said `run_code` under a profile whose model knows only
  `ipython` (`ToolCallError(run.name, ...)` now), and calling `run_code` under a
  rename says *presented as "ipython"* instead of the false "not enabled".
* **Dead weight dropped:** `CodeBindingsRequest` lost three fields nothing read
  (two duplicated the bridge); `TransportPresentation.timeout_ms` (a knob
  nothing turned); `_View.transport_name`'s misleading default; a stray
  re-export. Tests: three rlm tests re-proving core-level mechanics deleted, the
  hand-rolled `ToolExecutionInput` incantations replaced with `run_tool`
  (which exists precisely to end them), manual section-joining replaced with
  `render_prompt`, and the card views stopped materializing every line of a
  program to show one.
* **Skipped deliberately:** parallelizing prompt-section resolution (one async
  provider exists; the loop's lost atomicity is instead documented on
  `PromptText`), caching the described namespace tuple per registry generation
  (µs-scale), and a meta-threading `simple_views` variant (worth extracting on
  the next customer, not the first).

---

## P3-11: delegation, and three things the plan could not have known

### `ctx.subagents` was never scheduled

The port plan lists it among the capability seams; no Phase 0/1 row delivers it
and no code provided it. So the seam definition landed here with its first
provider — which is the right pairing anyway, because a seam with no consumer is
a guess about what a consumer will need.

Two contract points are stated in the seam rather than left to the provider:
**`start()` returns at admission, never with the answer** (the non-blocking
fan-out an RLM parent is built on), and **`result()` exists separately** for the
caller that genuinely blocks — a generic `task` tool returning the child's last
text. One provider serves both, which is only possible because the answer is
*reachable* without being what admission hands back.

`family_reach()` also lives in the seam, unused so far. It is C7's rule, and
P3-12's guard must not be able to disagree with the roster that displays it — one
rule, one implementation, even though only one of the two callers exists yet.

### `ctx.jobs` would have made admission blocking

`JobService.start` documents that a host which never binds a task group runs the
job **inline** — deliberately, so a headless one-shot stays honest instead of
silently dropping work. For a subagent that is exactly wrong: `start()` would not
return until the child had finished, destroying the one property the provider
exists to have. Every test would have passed while measuring nothing.

The fix was already in the codebase, unreachable: `Context._spawn` puts a
coroutine in the pool `drain()` awaits, with failures logged at their own
boundary. It was private and named for `emit` listeners. It is now
`ctx.detach(coro, label=...)`, and `_spawn` is three lines that call it. Nothing
about the mechanism changed — only that a second legitimate caller can reach it.

### The "no fallback" model rule has nothing to check yet

The plan specifies a model preflight with no fallback: an explicit selector must
resolve or the spawn fails. But `ctx.llm` has no model *catalogue* — only
`resolve_model(provider, model)`, which returns an empty `ResolvedModel` rather
than raising for an unknown name. So what P3-11 can enforce is the *route*: the
provider must have a registered adapter, checked at admission. An unknown model
name still fails, at the child's first request rather than at spawn.

The part that matters is intact and tested: **nothing substitutes a different
model.** An explicit selector is passed through untouched. `rlm.find_models` will
have to bring the catalogue with it, and it gets the earlier check for free when
it does.

### A `mappingproxy` is not a `dict`

The usage-attribution observer read `event.data.get("usage")` and guarded it with
`isinstance(usage, dict)`. A committed event's data is frozen into
`MappingProxyType`, which is a `Mapping` and is **not** a dict instance — so the
guard was always false and nothing was ever attributed. `code_mode.py` carries a
comment about exactly this trap for tool arguments; the same trap, one file over.
Caught by a test that asserted the attribution existed rather than that the code
ran.

### A namespace now owns the tools it presents

`rlm_run` was reachable two ways from a cell: `rlm.run(...)` and
`tools.rlm_run(...)`. Both governed, so not a safety hole — but two SDK routes to
one capability, and the second one has no prompt text explaining it.
`register_code_namespace(..., owns=(...))` drops the owned names from the `tools`
listing while leaving them dispatchable and policy-addressable. Found by an
assertion that was written expecting the property and did not get it.

### Also fixed: the CPU re-arm was flooring the wrong way

`arm_cpu_budget` computed `used = int(cpu_seconds_used())`. Flooring the CPU
*already spent* hands the next cell less than its budget: a bomb that burned
1.9 s floors to 1, so `cpu_seconds=1` leaves 0.1 s and the *next* trivial cell
dies on the previous one's spend. `math.ceil`. This is why
`test_a_cpu_bomb_hits_its_budget_and_the_kernel_survives` was intermittent under
load — a flake with a real bug behind it.

### The TUI vocabulary test earned its keep again

Four new event types, and the Phase 2 test that walks `KNOWN_SESSION_EVENT_TYPES`
against the adapter's tables refused them all until each was classified.
`subagent/status` and `subagent/usage-attributed` are `RECORDLESS` — they
belong to the subagent panel (P3-19), and eight children ticking through
`queued → running → done` would push the conversation off screen.
`rlm/child-admitted` and `rlm/child-deleted` *render*: a spawn is the moment work
left the conversation, and reading the transcript later without it makes the
child's reply arrive from nowhere. The same split decides ignorability — status
and usage are skippable by an older build; an admission is not, because skipping
it shows the parent the wrong family.

---

## The cleanup pass over P3-11

Four review agents. Three findings were structural, and one of them was a
resource leak.

### A settled child leaked a CPython subprocess

`code-runtime:<namespace>` is an effect of the *agent's scope* — that is the
Phase 3 design working correctly, and it is exactly why this went wrong. The
provider created child agents and disposed them nowhere: `delete()` cancelled the
agent but never called `ctx.agents.dispose`, which is the only thing that unwinds
an agent's scope. So every delegation left a live child process holding its whole
namespace, for the host's lifetime, whether the child finished or was revoked.

The fix is the invariant the rest of the package already follows: a child is
acquired through **`parent.ctx.effect()`**, so a disposed parent unwinds its
children (I2) and `delete()` becomes "release it early" rather than a second
cleanup path that has to remember everything. Settlement also calls `_quiesce`,
which drops the observer, disposes the agent and releases the session while
keeping the terminal `SubagentResult` — so a caller awaiting `result()` after the
child is gone still gets its answer. Two new tests pin both halves.

The session observer had the same shape of bug: `observe()` returns a disposer
and the return value was discarded, so a cancelled child kept attributing usage
to its parent forever.

### `rlm/child-*` was the wrong name for a generic seam's contract

ph-core declares the vocabulary and **ph-app** consumes it — and ph-app depends on
ph-core only, never on ph-rlm. So a second `ctx.subagents` provider would have had
to emit `rlm/…` (lying about its identity in an append-only log) or be invisible
to the TUI. The repo had already settled this the other way for `kernel/snapshot`,
which only ph-rlm emits and which is named neutrally.

Renamed to `subagent/admitted | status | deleted | usage-attributed`, with the
producer named in the payload. Worth doing now and effectively impossible later:
these are on-disk logs, and an unknown *non-ignorable* type makes a log
unreadable rather than degraded.

The roster fold moved with the names, from ph-rlm into the seam. P3-19's subagent
panel lives in ph-app and could not have imported `ph_rlm.child_roster` — it would
have grown a second copy of the fold, which is the "two projections that
disagree" A11 exists to forbid.

### `owns=` was a second list that could drift from the first

`register_code_namespace(..., owns=(...))` had the namespace declare which tools
it presents, as free text validated against nothing. In `bindings.py` the same
fact was stated twice — once as the `specs` tuple that builds the bindings, once
as `owns` — and a typo in `owns` silently suppressed nothing while a missing entry
silently double-listed. Generically it was worse: `owns=("bash",)` would have
deleted `bash` from every SDK block in scope, from a plugin that does not own it.

Inverted: `CodeBinding.presents` names the governed tool a binding is the face of,
so **the suppression list *is* the binding list**. `owns`, `_CodeNamespace` and
`_View.namespaced_tools` all disappeared. `_namespaces` builds the contributed
namespaces first, collects their `presents`, then builds `tools` — the ordering
change that makes the derivation possible.

The three hand-rolled copies of "wrap a governed tool as a namespaced binding"
became one `governed_binding()` in `code_mode.py`, which is also where the
load-bearing subtlety now lives: the binding handed to the bridge carries the
*tool's* name, so the dispatch record names the capability rather than the alias.

### `ctx.jobs` gained the path that made `detach` unnecessary for callers

`JobService.bind()` has zero callers, so the inline `await body()` branch was the
only one that had ever executed — a placeholder, not a shipped behaviour. It now
falls back to `ctx.detach`, which means `start()` returns immediately either way
and a job gets an id, a cancel and `job/started`/`job/settled` for free. The
provider uses `ctx.jobs.start` and `detach` goes back to being the primitive
underneath rather than a second public mechanism for the same concern.

### Reuse the codebase already had

* `delegation_depth` read `header.to_wire()["delegationDepth"]`. `SessionHeader`
  has a typed `delegation_depth` field — and reconstructing the wire alias by
  hand meant a rename would return 0, which *opens* the depth gate.
* `_last_assistant_text` hand-rolled the message projection and the text join.
  `derive_event_message` + `text_of` own those rules.
* The child's `AgentOptions` was rebuilt field by field, silently dropping
  `temperature`. `dataclasses.replace` keeps every field the parent had.
* `summary[:120]` → `CONTEXT_SUMMARY_MAX_CHARS`, which is what already caps the
  field.
* `access: str` plus a hand-check plus a `type: ignore` → `access: Access`, so
  pydantic validates it and the model sees `read|write` in the schema.

### Dead surface removed

`SubagentRequest.metadata`; `SubagentResult.replied` and its `WireModel` base (it
crosses no JSON boundary, and being a `WireModel` forced a hand-maintained sample
into `test_wire.py`); `SubagentRun.status` (the roster folds status — a copy on
the handle is a second source of truth frozen at the last in-process update);
`SubagentProvider.capabilities` (no consumer, and the vocabulary a second provider
needs is not guessable from the first); `SubagentService.provider()`;
`Config.default_access` and its shipped `bundle.yaml` key (inert: the request
default lives on the seam); `mark_replied` and the `replied` branch (P3-12 brings
both, and today the notice is unconditional because there is nothing to condition
it on). One prose `accessNote` in a durable event became a
`downgrade_reason` code with `downgrade_text()` rendering the sentence once, for
the model and the card both.

Also: `subagent_roster` now tests the event type *first* — 1.58 ms → 0.34 ms per
scan on a 20k-event log, measured, because the old loop read `data["runId"]` off
every `assistant/chunk` before discovering it was not a roster event.

### Skipped

* **An incremental cached fold for the roster.** The real fix (8 spawns in one
  cell is still 8 scans), but it needs a general `session.fold(key, types, step)`
  in ph-core — `_LatestFold` only does latest-of-one-type — and that is a seam
  addition beyond this diff. The filter-first win is 4.7× of it for two lines.
* **`family_reach` shipped unused.** The argument to land it with P3-12 is fair;
  the counter-argument that the guard and the roster must not be able to disagree
  is why it is in the seam. Left as-is.
* **`TokenUsage` validation** on the attributed payload, and inlining `_spawn`.

---

## P3-12: one boundary, two mechanisms

The interesting decision was not the family rule — that was already written down
as C7 — but **which pipeline mechanism each half of "messaging policy" gets**,
and the answer turned out to be two different ones.

### The family boundary is a guard; the rate limit deliberately is not

The plan specifies both as policy: the boundary as a `ctx.tools.guard`, the rate
limit as a `tools/pre-execute` listener "because a rate limit is a policy a
deployment may tune while the family boundary is not". Implementing it exposed
what that sentence costs under C3: **a `pre-execute` `Deny` ends the whole cell.**
So a token bucket refusing the fourth message in a second would destroy the
model's program — for backpressure, whose correct response is "wait and send
again".

So the two halves get opposite mechanisms:

* **the boundary** is a guard — deny-only, runs last, and unre-permittable by any
  later listener, so there is no ordering in which a permissive row lets a send
  out of the family. A test mounts exactly such a row and confirms it cannot.
* **the rate limit** raises `ToolCallError` from the tool body — the program's to
  handle. The distinction is visible in `error.kind` (`denied` vs `failed`), which
  is precisely what Code Mode's bridge branches on, so this is not a convention:
  it is the same field that decides whether the cell survives.

That asymmetry is the whole design, and it only became visible because C3 gives a
denial teeth that prime-agent's comm-channel check never had.

### `reachable_family` derives from `family_reach`

`family_reach` shipped unused at P3-11 with the argument that the guard and the
roster must not disagree. P3-12 is the consumer, and it needed *enumeration* (who
may I address?) not just the predicate (may I address X?). Rather than write a
second traversal, `reachable_family` calls the predicate for every candidate — and
a test asserts the two agree for every pair in a five-agent family, which is the
property the argument was actually about.

An agent's id is its session's id, so the parent link is
`SessionHeader.parent_session` and no side index is needed. Sessions are passed in
rather than read from a store, so the rule answers on a resumed log with no live
agents.

### Two roots are siblings, which cost two tests to learn

The rule makes root agents siblings of each other — deliberately, so two
top-level agents in one deployment can talk. Two of my tests were written as if
an unrelated root were out of reach; both had to move a generation down to a
*grandchild* to test the boundary at all. Worth recording because the same
mistake is available to anyone reading `family_reach` quickly.

### Three tests were racing a child's completion

A fake-adapter child settles inside the very `await` that a send to it needs, so
"send to a sibling child" and "a replied child gets no notice" were both
timing-dependent. The limit tests now use two root agents, which are deterministic
because nothing disposes them; the reply-suppression claim split into the two
facts it actually is — that a send *records* the reply (asserted in
`test_messaging.py`, deterministic because the record survives settlement) and
that the provider suppresses the notice (asserted in `test_subagents.py`, where
`mark_replied` can be called before the child settles).

The third failure was not a test bug: **a settled child cannot be steered**,
because `_quiesce` disposed its agent. That is the correct behaviour today and the
plan's answer is passivation/rehydration (P3-13), so the refusal names the
`agent_observe` route that does work, and a test pins it as the current contract
rather than a silent gap.

### Fixed along the way

* **`mark_replied` was keyed wrong.** `_children` is keyed by run id; a sender
  knows its own *agent* id. It now looks up by `run.session_id` rather than
  relying on how a child session id happens to be composed.
* **`_release` raised during parent teardown.** `self.ctx.subagents.forget(...)`
  reaches the service by attribute, and on the unwind path the provision is
  already gone — `ServiceNotFoundError` inside a disposer, which aborts the rest
  of the unwind. `ctx.get("subagents")` instead. Introduced by the previous
  cleanup pass and caught by the first test that disposed a parent with a live
  child.

---

## P3-13: rehydration, and the field name that hid the bug

The roster and the fold had already shipped with P3-11/P3-12. What P3-13 owed was
the thing P3-12's own test had pinned as a gap: **addressing a settled child**.

Settlement releases the child's *agent* — which is what holds an inbox — while
its session, its log and its roster row all survive. So rehydration is the
provider re-creating an agent against the same session, and the `Inbox` rebuilds
itself from that log's `agent/inbox/spliced` records, so anything queued before
settlement is still there. `ctx.subagents.rehydrate(run_id)` asks the owning
provider; a provider that cannot do it simply does not implement the method and
the caller gets `False` rather than an exception to interpret.

Two decisions inside it:

* **`rehydrated` is a distinct status, not a second `running`.** A parent reading
  the roster should be able to tell a child still on its original task from one
  woken up to answer a question.
* **A revoked child is not revived.** The tombstone is the parent's record that it
  revoked the child; quietly reviving it behind that record would make the record
  false. `rehydrate` refuses, and a send to it still fails.

### `provider_name` meant two different things

The first version of `SubagentService.rehydrate` looked up the provider with
`run.provider_name` — which is the **LLM** provider (`"fake"`, `"deepseek"`), not
the subagent provider (`"rlm-child"`). So the lookup silently missed, `rehydrate`
returned `False`, and the P3-12 test that asserted "a settled child cannot be
steered" *kept passing* — reporting the feature as absent when it was merely
misrouted.

Renamed to `model_provider`, with `provider` added and stamped by the service at
`start` (the service is what knows which name the caller asked for). Worth
recording because the failure mode was a green test, not a red one: a field whose
name fits two meanings will eventually be read as the wrong one, and the test
that should have caught it was written against the old behaviour.

The shared attach path (`_attach`) came out of this too — a fresh admission and a
rehydration now wire the usage mirror, the job and the status through one
function, so the two cannot attach different things.

---

## P3-14: the doctrine, and what it refuses to say

The port is mostly text, and the interesting parts are the omissions.

**Prime Agent's "RLM-native call contract" paragraph is dropped.** It told the
model that installed skills are pre-imported modules and not to invent wrappers
like `call_skill(...)` — advice that existed because prime-agent had no generated
surface listing. `tools:sdk` *is* that listing, generated from the registry, so
keeping the prose would mean two descriptions of one surface with the
hand-written one going stale first. A test asserts the paragraph is absent *and*
that the generated block is present, so the substitution is pinned rather than
implied.

**The delegation section is conditional on the surface being in the resolved
view**, not on config alone. A deployment that restricted the `rlm` namespace away
would otherwise be advertising calls this agent will be denied — which costs a
turn and teaches the model nothing.

**The volatile facts are a `context()`.** Depth, cwd, the family and the roster
all move between turns; in a cached `section` each change would re-bill the whole
prefix (A12). A test asserts `# Session` appears in the snapshot and *not* in the
prompt, which is the only way to catch that regression — it is otherwise
invisible until an invoice.

### The depth limit was quoted from the wrong place

The snapshot reported `RLM_MAX_DEPTH`, the module constant, while the provider
enforced `config.max_depth`. A deployment that set `maxDepth: 1` would have told
its children they had a level left. Now read from the provider that enforces it,
which is the only source that cannot disagree with itself. Caught by the one test
that configured a non-default limit.

### The workspace section ships partial, on purpose

`ctx.workspace` is Phase 4 (D21), so the section cannot say which tier is in
force. It says *no tier is mounted, and a child's `access="write"` is recorded but
not granted* — which is the same warning the full section exists to give: a child
told nothing about its workspace attempts writes and reads the failures as its own
bug. Omitting the section until the tier exists would have left exactly the gap
the section is for.

---

## The cleanup pass over P3-13/P3-14

Four review agents. Two findings were defects, one was a 69× regression in the
per-step hot path, and the rest were the duplication they were looking for.

### The one durable-log defect

`SubagentRun.provider` was stamped by the *service* after `provider.start()`
returned — but the provider appends `subagent/admitted` with `run.to_wire()` from
*inside* `start()`. So every admission event, and every roster row folded from
one, recorded `"provider": ""`. The field put there so Phase 5 could answer "who
wakes this child?" from a replayed log was empty in every log.

Renamed to `owner`, taken off the wire entirely, and documented as what it is: a
routing stamp, not a fact about the child. Two reviewers found this
independently, from opposite directions — one from the ordering, one from
grepping for readers of the new wire key.

### Prompt assembly was 69× more expensive than it needed to be

`facts()` folded `subagent_roster` once for the children line and once more per
family member (through `_name`, which walked to the member's parent and folded
*that* log). Every child of one parent has the same parent, so those were N+1
identical folds of the same log — per model step, inside `_pre_step`, before the
request goes out. Measured by the reviewer at **2.27 ms/step** for a parent of 8
on a 20k-event log, and quadratic across a fan-out: N² + 1 folds per family
round, 58 ms at 16 children.

Two changes: `SubagentService.roster(session)` caches the fold keyed on
`session.seq` (an exact invalidation key, because the log is append-only — one
entry per session, replaced rather than accumulated), and `facts()` asks for the
fold once. Re-measured after: **0.033 ms/step** on the same shape. The same cache
also removes ~3.8 ms from every `agent_message.send`, which was folding the
parent's log 16 times to resolve one `receiver_name`.

### `rehydrated` was the wrong shape for a status

The roster folds status last-write-wins, so a woken child that was *actively
working* read as `rehydrated` rather than `running` — losing "it is busy" for
every future consumer that branches on `"running"` (the P3-19 panel, `ph trace`,
`_render_roster`). It is now `{status: "running", cause: "rehydrated"}`: the
lifecycle stays a lifecycle and the provenance rides beside it. Free to change
because nothing branches on `SubagentStatus` yet — which is exactly why it was
worth changing now.

### Two contracts got declared instead of probed

`SubagentService.rehydrate` discovered the provider's method by
`getattr(provider, "rehydrate", None)`, so a provider whose method was misnamed
or mis-arity'd would fail *silently* as "cannot rehydrate" — the same failure
class that had just cost this package a green-test day. Now a
`RehydratableProvider` Protocol and an `isinstance` check.

And `PromptText` providers now receive the whole `AssembleContext` rather than
just the scope. The child doctrine needed the agent and was fishing it out of
`scope.get("agent")` — a bundle in another package knowing how
`ph.agent.registry` provisions it, and silently rendering nothing whenever an
assembly ran outside an agent scope. `AssembleContext` already carried both; it
was one parameter away.

### The workspace line now asks instead of asserting

It was a hardcoded sentence claiming no tier is mounted — which would keep being
emitted after D21 landed one, *and the test pinning it would have defended the
falsehood*. `facts()` now reads `ctx.get("workspace")` and renders the
not-mounted text only in its absence; a second test provides a stub workspace and
asserts the line changes. (A reviewer suggested reusing `downgrade_text` for the
sentence; tried and reverted — that sentence is about a *child's grant*, and
making one string mean two facts read badly in both places.)

### Also fixed

* **The delegation section keyed on the wrong thing.** It checked whether the
  `rlm` namespace was registered; the SDK block checks per-tool *visibility*. A
  deployment denying `rlm_run` alone would have dropped it from the listing while
  the doctrine kept teaching it. Now the same check the listing makes.
* **A roster row had no status between admission and the first status event** —
  a window a detached job makes real, and one the prompt renders in. The fold
  defaults to `queued`.
* **`_name`/`_display` were the same lookup in two modules.** One
  `roster_name` (and `SubagentService.name_of`, which folds through the cache)
  next to the fold that owns the fact.
* **The depth limit was quoted from two places.** `RlmChildProvider.depth_limit`
  is now the one value, asked of the enforcer.
* **The prompt's own delegation bullets re-described the SDK surface** — the same
  argument this file already makes about prime-agent's dropped paragraph, applied
  to a list I had written myself. Trimmed to the two rules the generated
  signatures cannot state; a test asserts the removed calls stay out.
* **Empty sections are dropped at the seam**, so "empty means absent" is stated
  once rather than in two renderers, and a consumer enumerating section names
  does not see phantoms.
* **`_status`/`_mirror` stopped taking `parent_session`** now that `_Child` holds
  it — two sources for one log is how a tombstone lands somewhere else.
* Test cleanups: `join_context_sections` instead of a hand-rolled join with the
  *wrong separator* (so every assertion had been against a string the model never
  sees); additive `extra_rows` instead of an override that re-listed the defaults;
  the delegation rows moved to `conftest`; substring echoes after a
  whole-constant assert deleted; `assert "parent " in ...` (which passed only
  because the session was named `parent`) made specific; the revoked-child test
  now exercises both refusal doors and the message it asserts.

### Skipped

* **An incremental roster fold** (`session.fold(...)` in ph-core) — strictly
  better than the cache, and still the right eventual shape, but a seam addition
  beyond this diff. The cache gets the measured win.
* **`by_session` as public API** — folded into `ensure_addressable` instead, so
  the three-hop wake path is stated once.
* **An incremental roster fold attached to `Session`** — investigated and
  *rejected*, for a better reason than the one I first gave. See below.

### And one skip that was retired instead

I first recorded the per-wake `Job` accretion as "inherent to P3-13's design".
Half right: the *rate* is inherent (a wake per message, where every other planned
job is one per long-lived thing), but the unbounded table was a Phase 1 default
nobody had picked a bound for. Asked directly whether jobs should be tied to
sessions, the answer turned out to be neither a session nor a number:

**A job is now an effect of the scope that owns it (I2)** — `start` takes a
`scope=` like every other registration in the codebase, and disposing that scope
cancels the job and drops its entry. A subagent's drive job belongs to the
*delegation* (the parent's scope), a `/refine` pass will belong to its session, a
daemon sweeper to the process. The bound is structural instead of a retention
number, which is the same reason `ctx.effect` exists at all.

Two things this had to distinguish, and the reason is specific:

* **abandoned** — the owner went away with work still running: cancel, then
  forget.
* **released** (`jobs.forget`) — the owner knows the work is done: forget,
  cancel nothing.

Without the second, a subagent drive job would report `cancelled` for work that
completed, because `_drive`'s own last act disposes the child. The presence of the
entry in `_jobs` *is* the flag that tells the two apart — `forget` pops it before
deregistering, so the abandon path sees it gone and no-ops. That also settled
where the job's owner is: the **parent's** scope, not the child's, since a job
whose body disposes its own owner would abandon itself.

Two questions I answered "no" to along the way, recorded because they are
reasonable and the reasons are not obvious:

* **Clear jobs when a new session starts.** `sessions.create()` is not a rare
  top-level event here — it fires once per subagent, 32 per cell at the shipped
  budget — so clearing then would kill the drive jobs of every sibling still
  working.
* **Log jobs to the session.** A job is mechanism, not a fact about the
  conversation, and it is not model-visible. Every job's *meaningful* outcome is
  already logged by its owner in domain terms (`subagent/status`,
  `harness/refined`); a generic `job/*` record would put scheduler bookkeeping in
  the conversation log beside it.

---

## Why folds are not attached to `Session`

I skipped "an incremental `session.fold(...)` in ph-core" with the weak reason
"beyond this diff". Looking at it properly, it is the wrong shape, and the reason
is a property worth writing down.

`Session.latest` *is* an incremental fold attached to the log, and it works
because "the current policy" is only ever asked of the live log. The plugin folds
are not like that:

* `fold_namespace` is what makes `ctx.sessions.fork(source, boundary)`
  reconstruct a namespace **as of the boundary** — that is the entire argument
  for D17 over a side file, and `test_snapshot.py` pins it by folding a
  hand-made slice of the log.
* P3-24's trajectory view projects a **stored** log with nothing mounted.

A fold attached to a live `Session` is monotonic in that log: there is no way to
ask it about an earlier prefix. Attaching these folds would have traded the
property for the speed — and the property is why the design is a fold in the
first place.

So the fold stays a pure function of *a* log, and the cache is a separate thing
the consumer owns. That is what `SessionFoldCache` is: one value per session,
keyed on `session.seq` — an exact invalidation key, because an append-only log
that has not grown cannot have changed any fold over it (A1). Entries are
replaced rather than accumulated, so it is bounded by live sessions, and
`session/disposed` drops the last projection of a session nobody can reach.

Two things this settled beyond tidying:

* **`SubagentService`'s bespoke `_rosters` dict became an instance of the
  pattern**, so the previous pass's measured win (2.27 ms → 0.033 ms per model
  step) is now expressed once rather than as a one-off in a seam.
* **P3-16 already prescribes this exact shape** — "fold over `harness/*` with
  incremental cache per `(scope, last_seq)`" — so the second consumer was
  certain, and it now has the helper and, more importantly, the stated
  requirement rather than a copy of the mechanism.

The requirement is the part worth having documented: **the cached function must
be a pure fold of the prefix.** One that also reads the clock, the filesystem or
a mutable table can change its answer without the log growing, and the cache
cannot notice. Nothing enforces it, so it is said where a second implementer will
read it.

### The other two folds were left alone, deliberately

`pending_approvals` has **no production caller** (tests only), and
`fold_namespace` runs once per kernel start rather than per step. Neither is hot,
and wrapping a cold fold in a cache adds a stale-value question to code that had
none.

---

## P3-21: the gate found three things, which is what a gate is for

Every one of the five claims already had unit tests. The gate exists because
those mount hand-picked rows, and the claim the containment argument rests on is
about the profile a *user* gets — so it mounts `ph-base` + `rlm/bundle.yaml`
through the loader and runs real cells in a real kernel. Doing that surfaced
three things no unit test had.

### `CodeRunFailure` never declared itself a denial

`HarnessError.denies` is what `registry.py` reads to set `ToolFailure.kind`, and
`ToolFailure.kind`'s own docstring calls it "the fact every consumer branches on
and none may infer". `CodeRunFailure` did not set it — so a cell refused by policy
reached the model, the log and any future UI as **`failed`**, indistinguishable
from a cell that timed out. The plan's gate (a) names `CodeRunFailure {kind:
"denied"}` explicitly; the code produced `failed`.

Fixing it needed one change of shape: `denies` was a `ClassVar`, and
`CodeRunFailure` is one type with three kinds — a denial *is* policy refusing, a
budget is a cap, an abort is cancellation. So `denies` became a class attribute an
instance may set, and `CodeRunFailure.__init__` sets it from its kind.

### `cancel_grace` was the one grace a deployment could not tune

`boot_timeout_seconds` and `shutdown_grace_seconds` are row config;
`cancel_grace` was a `Kernel` field with a hardcoded 2.0 that the provider never
passed. It is the window in which a cell that swallowed a refusal can still finish
straight-line synchronous work — the most consequential of the three — and it was
the only one unreachable. Found because the gate could not make its own assertion
deterministic without it: the shipped 2.0 raced the test cell's own `sleep(2)`.

### The abort ladder's reach is narrower than my earlier note implied

I wrote, after the first cleanup pass, that C3's enforcement means "the file is no
longer written". True of that test — whose docstring is explicit that "the `sleep`
is the window" — but not a general claim. The ladder (`cancel` frame → `SIGINT` →
`SIGKILL` after the grace) stops a cell **at its next yield point, or kills it if
it never reaches one**. It cannot preempt straight-line synchronous Python: two
`pathlib` statements immediately after a caught refusal run before the guest's
loop regains control.

That residue is a stated non-goal (§11, Q10) — raw `pathlib`/`subprocess` cannot
be gated per call, and the sandbox tier is the boundary for it, not the dispatch
bridge. The gate now says so in the one place someone will look for it, rather
than asserting a stronger claim that happens to hold for one grace value.

### Two of the five test a mechanism, not a consumer

(b)'s `ToolCallLimit` and (c)'s shipped spill policy are both Phase 4 rows. What
Phase 3 owes is that a per-dispatch boundary *exists* for them to attach to, so
the gate mounts listeners that stand in for them — a counter for (b), an offload
for (c) — and asserts the boundary: three governed evaluations rather than one,
and one oversized dispatch reshaped while its sibling stays inline. Marked as
such rather than left to look like the policy ships.

### One finding left as a follow-up

`FsDenied` is a `PermissionError`, not a `HarnessError`, so an `fs/write-intent`
veto also reports as `failed` — the same class of gap as `CodeRunFailure`'s, one
seam over. The gate records the current behaviour instead of asserting a
distinction the code does not draw. The row that makes it matter is
`permissions-fs` (Phase 4), which is where the fix belongs: changing the base
class now would touch every `except PermissionError` for a distinction nothing
yet consumes.

---

## The cleanup pass over P3-21

Four review agents over a diff that was mostly one test module. They found two
defects in the gate itself, one in the code it was testing, and a 3× runtime win.

### The gate was silently not testing what it claimed

`_documents()` appended `[HOST_INTERPRETER, *overlay]` as raw loader patches, and
**a patch replaces a row's whole config**. Both fragments targeted
`code-runtime-python`, so the `cancelGraceSeconds: 0.5` overlay dropped
`python: host` and `sweepOrphans: False` outright — meaning the C3 test, the most
important one in the module, was building a `uv` venv and re-enabling the orphan
sweep, in a module whose comment said it "needs no `uv` and no network". It
passed, so nothing reported it.

The fix is the fixture owning the merge: `shipped_profile({row_id: {key: value}})`
merges fragments per row and emits one patch each, so a test cannot un-pin the
interpreter by tuning an unrelated knob. Verified by composing the real documents
before and after.

### 72% of the gate's runtime was `os.walk` over the repo

`tools.glob(pattern='*.nothing')` with no `path` resolves against `FsService.root`,
which falls back to `Path.cwd()` — the checkout, 11,279 files, **~400 ms per
dispatch**. Two tests issued seven of them: 2.9 s of the module's 4.0 s, and
runtime that scaled with whatever was sitting in the developer's working tree. One
`path=str(tmp_path)` argument: **4.0 s → 1.08 s**, and the whole-suite tax from
+11% to about +3%.

The reviewer also checked the `time.sleep(5)` I had assumed was the cost: cell
wall time is `cancel_grace + ~60 ms` flat in the sleep length, because the cell is
`SIGKILL`ed at expiry. 5 s is free margin. The `cancelGraceSeconds: 0.5` override
is itself a 1.5 s *saving* against the shipped default — the row config paid for
itself in the test that motivated it.

### `denies` was a boolean carrying a three-valued fact

`FailureKind` is `denied|failed|aborted`; `CodeRunFailure.kind` is
`denied|budget|aborted`; and the bridge between them was
`"denied" if error.denies else "failed"`. So `budget → failed` (right) and
**`aborted → failed` (wrong)** — unreachable today, but the Literal advertised a
kind whose consumer contract was already broken.

Replaced with `HarnessError.failure_kind: FailureKind`, declared by the error and
read directly. The three-way mapping is now one dict in `CodeRunFailure`, and
`registry._failure` has no mapping at all. `denies` is gone rather than kept
alongside.

### `FsDenied` — the deferral was too conservative

I had recorded "an `fs/write-intent` veto reports as `failed`" as a Phase 4
follow-up. The reviewer pointed out the precedent is one file over —
`SandboxError` already sets this — and verified the blast radius: `FsDenied` is
caught nowhere in `src/`, and the only production `except PermissionError` is an
unrelated `os.kill` probe. `class FsDenied(HarnessError, PermissionError)` with
`failure_kind = "denied"` linearizes cleanly and keeps every catcher working.

So the hole C3 closes for `tools/pre-execute` is now closed for the fs seam too —
in the seam whose module docstring calls itself a gate rather than a report.

### Two gate claims were testing the wrong thing

* **(c) was on a log-only seam.** `tools/code-dispatch-log` reshapes the durable
  record and nothing else, so a listener there leaves the *program* holding the
  full bytes — and C5 is about what reaches the model's context. It also has no
  consumer in `src/` at all, so the gate was filling the seam's missing third
  role and authoring both halves of its own contract. Moved to
  `tools/post-execute`, where the Phase 4 row attaches and where an
  `Accept(has_value=True)` changes what the cell receives. The test now asserts
  the program saw `["[spilled]", "small"]` — the actual C5 fact — plus that the
  seam fired per dispatch.

  Writing it surfaced a real P4-02 design question worth recording: a value
  replacement re-renders content through the tool's own output schema, so a
  *generic* spill policy cannot invent a replacement value without knowing the
  tool. Noted in the test rather than solved.

* **(d) was refusing for the wrong reason.** It sent to a name that did not
  exist, so `resolve` took the empty-roster branch — a guard denial, but not the
  out-of-family one C7 is about. It now spawns a child and a grandchild and
  addresses the grandchild: genuinely one generation too far.

* **(b) no longer needs a stand-in.** The dispatch budget is a *shipped* counter
  of exactly the boundary `ToolCallLimit` will count, so setting it to 3 and
  issuing a fourth proves the per-dispatch boundary without the test authoring
  the policy it checks.

### The bundle-inventory test could not see the most load-bearing row

`test_the_gate_runs_against_the_shipped_bundle` parsed `bundle.yaml` and checked
six `name` literals. It could not see `disabled:`, could not see a row that fails
to activate, silently omitted two of the eight named rows — and could not check
`- id: tools / config: {mode: code}` at all, because a patch has no `name`.
Flipping that to `mode: native` deletes C6 from the shipped profile and **all five
behavioural tests still pass**, because they all reach tools through the
transport.

Replaced with one observable property of the *mounted* profile: a top-level native
`write` must be refused with the transport named in the route back. That single
call fails if the mode changed, if the transport was renamed, or if
`rlm-presentation` left the bundle — and needs no inventory to go stale on a
rename. `Loader.inactive() == []` moved to `test_bundle.py`, where it catches the
never-activated case the name list could not.

### Consolidation

`shipped_profile` in `conftest.py` replaced the gate's hand-rolled mount *and*
five `Context()` + try/finally blocks in `test_bundle.py`, so "the gate mounts
what the bundle test mounts" is now true by construction. `run_ipython_cell`,
`dispatch_names` and `settled_dispatches` moved to `runtime_helpers`, removing a
fifth hand-written `ToolExecutionInput` and four copies of the dispatch-name
comprehension. The root `mount` fixture grew one `profile=` keyword rather than
the suite growing a second lifecycle.

And a gap the reviewer found while arguing about the grace threading: **nothing
tested the `Config → PythonCodeRuntime → Kernel` handoff**, which is how
`cancel_grace` came to be a config field that did nothing. `test_bundle.py` now
asserts two configured graces arrive and an untouched one keeps its default.

### The abort residue moved to the code that owns it

It was documented in a test docstring while `code_mode.py` still claimed a denial
"bounds partial state to one cell". The limit now lives on
`Kernel.cancel_grace` — the ladder's only tuning point — `code_mode.py`'s C3
bullet says "about one cell" and points there, and the gate references it instead
of restating it. Three prose copies became one.

### Skipped

* **Grouping the three graces into a `KernelTimings` value object.** Real
  duplication (five coordinated edits per knob, and `KernelLimits`' own docstring
  argues against exactly this shape), but it churns three construction sites to
  save six lines, and the new handoff test now catches the failure mode that made
  it urgent. Revisit when a fourth timing knob arrives.
* **Module-scoped mounts in the gate.** Ceiling was ~0.3 s and only five of seven
  tests could share; three register process-wide mutating listeners, in the one
  module whose job is asserting what policy does. Fixing the glob root was 6× the
  saving with none of the coupling.

---

## P3-16: the harness as a fold, and what writing the gates found

Increment 1 is everything except the planner: the fold at both scopes
(`state.py`), validation and apply (`service.py`), the prompt section, `/refine`
and its rollback (`__init__.py`). A caller supplies a `RefinementProposal`; the
LLM that writes one is increment 2, and it is a thin layer over
`HarnessService.apply` rather than a second path into the state.

### Three bugs the tests found, in order of how quietly they failed

**`_local` was not a field at all.** `_local = field(default_factory=…)` with no
annotation is a plain class attribute, and `slots=True` keeps it as one — so
`self._local` was a `dataclasses.Field` object and the first `local()` call would
have raised. It type-checked, linted and imported cleanly; the first test that
folded anything caught it.

**Every global refinement was refused, however the human answered.** `_approved`
compared the outcome against `"allowed"`, and `ApprovalOutcome` has no such
member — it is `allowed-once | rejected | cancelled | unavailable`. The failure
direction is the safe one, which is exactly why nothing else would have noticed:
a fail-closed gate that always fails closed looks like a working gate until
someone tries to use it. Only the test that asserted a global edit *lands* could
see it.

**`harness/refined` was unrendered.** Adding the type to
`KNOWN_SESSION_EVENT_TYPES` tripped the TUI adapter's vocabulary equality — the
P2-01 invariant doing its job across a package boundary. It is a row, not an
auditor's record: `/refine` is only one way here, the planner refines at turn end
with no command to show for it, and a user who cannot see the harness change has
no way to know why the next turn behaves differently. Rejections go on the same
row, because an edit the harness *refused* is the interesting half.

### Four divergences from the plan, and why

**One event type, not two.** The plan folds `harness/refined` and
`harness/rolled-back`. A rollback is an inverse *applied*, and the refined record
already carries `rollbackOf`, so a second type would be a second fold case that
has to agree with the first about one operation.

**A rollback is the inverse record, not an inverse proposal.** The first version
built inverse `HarnessEdit`s and re-ran them through `apply`. That restored the
title and content and left the entry at `v3` — so `after.entries ==
before.entries` failed, which is the plan's own gate ("rollback restores fold").
Re-proposing cannot satisfy it: `apply` constructs entries and bumps versions by
construction. The inverse of a record is that record reversed with `before` and
`after` swapped, which is both shorter and actually an inverse; the churn stays in
the refinement history, which an append-only log keeps anyway. It is also
deliberately not revalidated — that state passed validation when it was written,
and refusing to undo a bad refinement because the world has since moved would
trap a user in the state they are trying to leave. `apply`'s `rollback_of`
parameter went away with it, and both paths now commit through one `_commit`, so
a rollback and an apply cannot come to differ in what they leave behind.

**H3's gate is an approval request, not a `tools/pre-execute` `ask`.** The plan
describes a global edit as an `ask` "on the `refine` tool". There is no such
tool — precisely because `/refine` is a command — so no `pre-execute` waterfall
ever fires for it. `ctx.approval.request(tool_name="refine")` asks the same
question through the same answerers, records both halves in the log, and fails
closed with nowhere to ask. Registering a model-callable `refine` tool to satisfy
the letter would hand the model the one operation the design says is the human's.

**Kinds are `note | procedure | skill`.** Prime Agent has
`prompt | memory | skill | subagent`. A `note` rendered in the prompt section *is*
a prompt patch with the doctrine off-limits (H5), so `prompt` and `memory` collapse
into one; a `subagent` entry would be a preset, and a preset is capability, which
Q13's own boundary puts outside the knowledge layer.

### Two things the design got for free, and one it paid for

The fork test is three lines: apply, note the boundary seq, apply again, then
`ctx.sessions.fork(session, boundary)` and fold. Nothing was copied and no state
was handed over — that is the whole D14 claim, and a file could not express it.
Likewise the delete-and-re-derive gate is `fold_session(session).to_wire() ==
local(session).to_wire()`, asserted against a *cold* fold so the cache cannot be
what makes them agree.

What it paid for is H1. Resolving a reference in the runtime the model actually
uses means the probe boots a kernel — one per deployment, in a `harness-probe`
namespace of its own so `_m`/`_c` never land in the namespace the model is using
(they would otherwise show up in its snapshots). Checking the import against the
host process instead would have been free and would have proved something about
the harness's own `sys.path` rather than the model's. The whole module still runs
in 1.4 s.

One thing H2 is worth naming precisely: `render_call_pattern` matches on the
callable's *name*, so an entry pointing at stdlib `glob.glob` renders
`await tools.glob(...)`. That is a real ambiguity, and it fails in the safe
direction — the only way it can be wrong is by steering the model toward the
governed path, never away from it.

### Fixed while writing it

The projection was written to one shared `$PH_HOME/harness/harness_state.json`,
but it is a projection of *local layered over global* — so two sessions would
overwrite each other's, and P6-01's "the file equals the fold" invariant would
flap in any deployment running more than one. It is now one file per session
(`projection_path`), which is the shape the plan's `<session artifacts>/harness/`
describes; artifact roots do not exist yet, so a directory named for the session
under the harness root stands in.

It was also written with `path.write_text`, which truncates: a human can read the
file while a refinement applies, and under the daemon two sessions project at
once. It now stages and renames. And `_probe` reached past `ctx.code_runtime` to
`.provider.run(...)` when the seam has its own `run` — the seam checks the
provider exists in one place and the harness should not be the second.

---

## P3-16, increment 2: the planner, and where the trigger state went

Increment 1 could apply a `RefinementProposal`. This is where one comes from:
`planner.py` (the two model calls) and `auto.py` (when they run unasked).

### Two calls, deliberately different sizes

The **review gate** is cheap and answers one question — is there anything here
worth recording — because auto-refine fires on a turn count and most turns teach
nothing. The **planner** is the expensive one and runs only when something says
yes, or when a human typed `/refine`. A user-triggered pass skips the gate
entirely: asking a model whether to do what the human just asked for spends a
call to second-guess them.

Neither is a turn. Both name `purpose="refine"`, which is what keeps them out of
`is_loop_request` and therefore out of the "model-visible means logged"
invariant — correctly, because their prompt is *about* the conversation rather
than part of it. That meant adding `"refine"` to `GenerateOptions.purpose`, which
is a closed union on purpose: opting out of the invariant has to be a declaration
this file can enumerate, not a flag anyone can leave at its default.

The planner reads `derive_messages()`, not the raw log — so a compacted range
reaches it as its summary, and it cannot learn from history the agent has already
been told to forget.

### The trigger state is a fold, and it has no clock

Prime Agent keeps a turn counter and a last-run timestamp beside the session. All
four of its numbers are ported — 25 turns, post-compaction, a cheap gate, a
20-minute cooldown — but every one of the inputs is now read off the log by
`due()`, in a single backwards pass that stops at the most recent
`harness/refine-considered` or `harness/refined`.

The part worth naming: **the clock is the log's**. `due()` compares against the
timestamp of the event that triggered the check rather than `time.time()`. That
event was appended moments ago, so nothing is lost — and the whole decision
becomes a pure function of a prefix, which answers the same on a resumed session,
on a fork, and in a test with no clock to freeze. The cooldown tests are three
fabricated events and no monkeypatching.

Compaction is a trigger in its own right because it is the one moment the
conversation gets *shorter*: what the summary dropped is exactly what the harness
should have kept, and after it nobody can read it. It is detected as
`is_replacement_surface_event`, which already exists — so this works today rather
than waiting for Phase 4's compaction row to name itself.

### `harness/refine-considered`, and why a non-event is an event

Every outcome that is not a refinement appends one: a declined review, an empty
proposal, a planner that returned prose, a veto. It is what advances the
cooldown, so a broken planner or a quiet conversation costs one cheap call rather
than one per turn for the rest of the session.

It is **ignorable** — a reader that skips it gets one extra review pass, not a
harness the session does not have, which is exactly the line
`IGNORABLE_SESSION_EVENT_TYPES` draws — and in the TUI it renders **only when the
trigger was `user`**. Automatic passes saying "nothing changed" every
twenty-five turns are the auditor's business; a `/refine` the human typed is the
opposite, because without that row the command would answer "refining in the
background" and never come back. `_on_kernel_restored` had already established
the shape: a handler that renders only when there is news.

### Two things that only showed up once it was wired together

**A vetoed `/refine` answered with silence.** The veto ran inside the job, wrote
nothing, and returned a string to a caller that had already replied. It now
records a consideration like any other non-refinement, so the reason reaches the
transcript. "A veto costs nothing" narrowed to "a veto costs no tokens", which is
the claim that was actually load-bearing.

**Finished passes accreted in `ctx.jobs`.** A refine job is owned by the agent's
scope, so nothing dropped its entry until the agent went — one per pass, for the
life of a long session. `jobs.forget` exists for precisely this ("released", not
"abandoned"), and the seam's own docstring names `/refine` as its example; the
body now calls it in a `finally`.

### `/refine --show`

Increment 1's placeholder — a status line saying the planner was not mounted yet —
had to go. What replaced it is `--show`, which prints the harness **unbounded**,
unlike the prompt section: the entries the per-kind bound elides are exactly the
ones a human cannot see any other way.

### Known gap

**A subagent's own session auto-refines.** A child runs a real loop, so 25 turns
in one child triggers a local pass over a harness nothing reads after the child
settles. The clean fix needs the harness row to know a session is a delegation,
and `rlm-harness` deliberately mounts without the delegation rows. Cheap in
practice (children rarely run 25 turns) and worth revisiting when P3-19's panel
gives the two rows a reason to know about each other.

---

## The cleanup pass over P3-16

Four reviewers (reuse, simplification, efficiency, altitude), findings deduped.
Every reviewer independently flagged `_collect_text`; the interesting fixes are
the two caching ones.

### The "incremental" fold wasn't

`SessionFoldCache` keys on `session.seq`, and `seq` bumps on every event —
chunks included — so the per-step prompt read missed the key on essentially
every model step and refolded the local harness from zero: a full log scan plus
one pydantic validation per historical refinement, to reproduce a state that
changes at most once per 25 turns. The plan's own line ("incremental … so a long
session does not re-fold from zero on every prompt assembly") said what this was
supposed to be; the cache alone was not it.

The fix is at the seam, not in the plugin: `SessionFoldCache` gained an optional
`extend(previous, session, from_seq)` — a miss now folds only `events[from_seq:]`
(exact, because `seq` *is* the log length). The requirement is stated where
`compute`'s was: extending the fold of a prefix must equal folding the whole
log. The harness's `extend_session` scans the new slice, hands back the *same
object* when it holds no refinement, and folds onto a copy when it does — and
`fold_session` stays a pure function a fork slice can use. The roster fold gets
the same option for free and keeps its full-refold behaviour until it opts in.

Same shape one level up: `state()` merged local over global on every read,
re-copying every entry and the entire refinement history for a byte-identical
answer. It now memoizes on the *identity* of the two folds — each cache returns
the same object until its log actually gained a refinement, so `is` on the pair
is an exact "nothing changed" test.

### Applied

- `_collect_text` deleted; the planner pushes chunks through `BlockAssembler` —
  "the single canonical assembly algorithm" — and takes `text_of(blocks())`.
  All four reviewers converged on this one: a second assembly fold beside the
  loop's could disagree with the transcript about what a reply said.
- `on_session_event` resolves the agent with `ctx.agents.get(session.id)`
  (the loop agent's id *is* the session id) instead of a linear scan over
  `agents.list()` with a duck-typed identity test — the registry's docstring
  warns against exactly that side table.
- `start` takes the `running` flag down if `ctx.jobs.start` itself fails;
  before, a failed start left the session answering "already running" forever.
- `_conversation` renders newest-first and stops at the char budget instead of
  rendering the whole derivation and keeping the tail.
- Dead `slugify` fallbacks in `apply`/`_entry` removed: `validate` stamps the id
  on every accepted edit, and the dead copies derived it *differently*
  (`edit.title` vs `edit.title or edit.content`) — a latent divergence.
- The proposal's `rationale`/`expected_outcome` are now recorded on
  `harness/refined`: the prompt demanded them, the models parsed them, and then
  nothing kept them — the docstring's "the proposal that produced it is on that
  event" was false for exactly those fields.
- One rendering identity: `KINDS` (derived from `HarnessKind` via `get_args`),
  `entry_label`, `refinement_line` in `state.py`, shared by the prompt section
  and the planner overview — the planner says "use its exact id" about text
  that must therefore match what the section shows.
- `RefineRequest` built once at the two entry points and threaded as one object;
  `refine`/`start`/`_considered` had five parallel parameters re-spelling it.
- `_probe` no longer peeks at `runtime.provider` — the seam's `require()` owns
  that refusal (the increment-1 note said this rule; the peek had survived).
- `_append_global` appends through `ph.paths.write_text_under` (the telemetry
  seam's idiom) inside the lock; `rederive` (a one-line alias of `fold_session`)
  deleted; package `__all__` trimmed to the consumed surface; `HARNESS_ROW`
  moved to the tests' conftest beside the other row constants; the TUI test uses
  the file's own `_replay`; two stray `bool(…)` wraps and an identity
  comprehension dropped; one stat call instead of `exists()` + `stat()`.

### Skipped, with reasons

- **A `ctx.llm` "one auxiliary call → text" helper** (altitude): warranted the
  moment compaction or session-title lands — both are the same shape — but today
  it would have one caller. The `BlockAssembler` swap removes the drift risk now.
- **Jobs-seam `key=` dedup replacing the `running` set** (altitude): the right
  eventual home — the seam's docstring names `/refine` as its example — but a
  core seam change beyond this diff. The desync edge it protected against is
  fixed locally.
- **A shared ph-core `slug()`** (reuse): the twin lives inline in the subagents
  seam, outside this diff; extracting it moves settled code for two callers.
- **Suffix-reading the global log / folding the in-hand record into the cache**
  (efficiency): growth is human-approval-gated, so the file stays small; the
  refold happens only when the size actually changed.
- **`due()`'s backwards scan** (efficiency): structurally bounded — every pass
  that fires writes a marker (`refined` or `refine-considered`), so the scan
  range is capped by the trigger cadence itself, and it runs once per turn, not
  per event.
- **Batching H1 probes / caching the review-vs-plan double render** (efficiency):
  proposals carry few skill edits, and both sit next to an LLM call that costs
  orders of magnitude more.
- **`metadata` on entries** (simplification): the plan's edit schema names it;
  dropping it would refuse proposals the plan says are valid.
- **Closure mount → class** (efficiency): the closure-bodied `apply()` is the
  house pattern for every row; nothing large is uselessly pinned.

---

## P3-17: the corpus, and the event the plan filed in the wrong place

`rlm-context-loader` is "prompt-as-a-variable" — a body of text the agent can
consult without any of it entering the prompt. Three things about the port.

**It is three ordinary tools, not a namespace.** The plan writes
`tools.context_search/chunks/head`, and the reason that spelling is achievable
with no namespace machinery is that Code Mode builds the `tools` namespace *from
the registry*: registering a tool called `context_search` is what makes
`await tools.context_search(...)` exist. So the row registers three tools and is
done — and gets per-query dispatch records and per-query offload for free, which
is the entire argument Q4 makes for the binding form over a bare
`context.search(...)` on a kernel variable.

**The recipe is `context/loaded`, not `kernel/snapshot {kind: "recipe"}`.** The
plan files the corpus under the snapshot chain. That is right for a *kernel
variable* the harness built and which exceeds the per-variable cap — but this
corpus never enters the kernel, precisely because access is a binding. Recording
it as a namespace variable would have `fold_namespace` describing a namespace
that does not contain it, and `materialize` trying to restore something with no
blob. The recipe *semantics* are what matter and they are kept exactly:
`{loader, sources, digest}` over the resolved source set, a matching digest is
silent, a changed one tells the model the corpus was rebuilt, an unreadable
source says so. `kernel/snapshot {kind: "recipe"}` still has no producer, and now
has a documented reason to wait for the case it was designed for.

**A `Document` stores line offsets, not lines.** Splitting a corpus large enough
to be worth this row would hold every line as its own Python string for the life
of the process. A line is a slice; `_end_of` is the one place that knows a
trailing newline ends the last line rather than beginning another.

The prompt contribution is a `section`, not a `context()` — the opposite of the
harness's. The corpus is resolved once at mount and cannot change mid-session, so
its text is fixed and belongs in the cached prefix (A12). The recipe is appended
on first assembly rather than at mount, because that is when the corpus becomes
model-visible, and I3 says model-visible means logged.

## P3-18: the skills row, and the dead code it turned out to need

`install_skills` was written in P3-08 and **nothing ever called it**. The `boot`
frame had no `skills` field, so the guest's import-and-bind path — including
`UnavailableSkill`, the stub that explains itself — had been shipped, tested at
the unit level, and was unreachable. P3-18 is what connects it, and three edits
outside the new row were needed:

- The `boot` frame carries `skills` on **both** protocol twins. The mirror test
  is what caught the second one missing, which is the entire reason it exists
  (D7). It is required rather than optional, like `namespaces`: an empty list is
  a deployment with no Python skills, which is a fact, not an absence.
- `install_skills` is now called from the bootstrap, *before* `_protected` is
  computed, so skill names join the harness's surface rather than the cell's
  state — snapshotting one would try to pickle a module. Its own duplicate
  `_protected` assignment went away with that.
- The venv builder decides editability, in one place with the reason: the
  staleness marker digests the requirement *specs*, not their contents, so a
  non-editable local skill would keep serving its build-time bytes after every
  subsequent edit.

**Two lists, deliberately.** `runtime.skills` is what to install (once, per venv
build) and `runtime.skill_modules` is what to import (at every kernel boot). A
distribution named `acme-websearch` imports as `acme_websearch`, and the import
name is read from `[project] name` in `pyproject.toml` rather than guessed from
the directory — a package whose name differs from its folder is the normal case.

**Validation refuses rather than half-accepts.** A skill whose frontmatter `name`
disagrees with its directory is dropped with a logged reason: registering it
would put one string in the catalog and leave the body under another on disk,
which surfaces later as a confused model reporting a bug.

The gate — *a skill's `run()` is callable in a cell* — runs against a real
kernel with the package on `PYTHONPATH`, so it drives discovery → boot frame →
`wrap_skill_module` without building a venv (which shells out to `uv` and reaches
the network). What it proves is the wiring the venv path shares.

One thing worth recording about the failure mode: calling an unimportable skill
raises *inside the cell*, so the traceback is the cell's result rather than a
tool failure. That is right — a skill that did not import is the deployment's
problem to fix, not a refusal the model should route around — and it is what the
test asserts.

---

## The cleanup pass over P3-17/P3-18

Four reviewers (reuse, simplification, efficiency, altitude). The fixes worth
explaining; the mechanical ones are listed at the end.

### The offsets design was being defeated by its own callers

`Document` keeps line *offsets* precisely so a large corpus is never split into
per-line strings — and then `Corpus.chunk` joined the entire corpus into a fresh
string and `splitlines()` it **on every page request**, making a full walk
O(corpus²/size), and `Corpus.search` called `line(number)` in a Python loop,
re-slicing every line per query. Both now use the machinery as designed: the
joined view is built once (`cached_property` — `Corpus` dropped `slots` for a
`__dict__`, one instance per mount) and a chunk is arithmetic over its offsets;
search hands the whole text to `pattern.finditer` (the regex engine scans in C,
with `re.MULTILINE` so a model's `^`/`$` keep their per-line meaning) and maps
hits to lines by bisecting the offsets. At most `limit` matches are ever built —
the `ceiling` parameter died with the discard-buffer it existed to bound, and
`max_matches` is now simply the cap on `limit`. `Document.of` also stopped
enumerating half a gigabyte of characters in the interpreter (`str.find` loop).

And a real bug the rename fix uncovered: threading `name` into `_resolve` was
shadowed by the per-document `name` loop variable, so every corpus was named
after its last document. The prompt test caught it on the first re-run.

### The per-step costs

`announce` ran inside the prompt section on every model step and scanned the
whole log backwards for `context/loaded` — the event it looks for is appended
near the *start* of a session, so the scan was effectively O(log) per step.
`session.latest(LOADED)` is the incrementally-folded helper built for exactly
this shape (its docstring says so), and the service now also memoizes the note
per session and the rendered manifest per note: the log stays the durable
answer, the memo just stops re-asking a question that cannot change while the
mount lives. `note_for` — dead outside one test — went away.

### The protocol version finally earned its keep

`boot` gained a required field and `PROTOCOL_VERSION` said "bumped when a
frame's field set changes" — and had not been. Now 1 → 2 on both twins. The
managed venv was covered incidentally (the staleness marker digests the skills
specs), but **override mode** — the mode that exists for deployments whose
skills need a particular interpreter — had no staleness story: a stale guest
there would pass the version check and silently boot with no skills bound,
handing the model bare `NameError`s instead of the `UnavailableSkill` stub. The
bump converts that silent degrade into the loud boot refusal D7 specifies.
`BootFrame.skills` also lost its default (the host twin could construct a frame
the guest twin's required-set would refuse), and `to_boot`'s parameter followed.

### One YAML policy, one name format

Frontmatter now parses through `ph.cordis.safe_yaml_load` — the codebase's one
hardened loader, which refuses unknown tags *and* implicit date coercion, so
`description: 2024-01-01` stays the string the author wrote. That deleted
ph-rlm's day-old direct `pyyaml` dependency. And `NAME_PATTERN` moved to
`ph.seams.skills` beside `NAME_MAX` (interpolated, so 64 is written once), with
`register()` now enforcing the charset — two readers of `SKILL.md` will share
the format, and the seam is where a shared format lives. `_read` also reads only
the first 64 KiB of a `SKILL.md`: the fields it validates are capped at a
kilobyte, and the body — up to 10 MiB — staying on disk is the G9 point of the
row.

### The catalog was giving an unfollowable instruction

"Read its `SKILL.md` with `tools.read`" — with no path rendered anywhere. Each
entry now names its file, and the sentence stopped naming the tool, so Phase 4's
progressive disclosure can supersede the route without this row's prompt text
becoming a lie.

### Mechanical

One `Corpus` construction instead of two (name passed into `_resolve`); the
`context/loaded` payload trimmed to the recipe its own docstring defines
(`{corpus, loader, sources, digest, note}`); uniform `output=Model, render=fn`
across the three tools with the chunk lambda named; `SEARCH_TOOL` out of
`__all__`; the gate test's `if False` await replaced by an *actually awaited*
async `run()` (which is the §6.8 convention the test claims to pin); the
oversized-skill test uses a sparse `os.truncate` since the size gate stats
before reading; `expanduser()` bound once in the venv's editable branch; the
guest's redundant `list()` copy dropped; `FAKE_OPTIONS` imported at module top.

### Skipped, with reasons

- **Parallel source reads in `_resolve`** (efficiency): one-off mount cost;
  task-group machinery for the rare many-file corpus isn't earned yet.
- **`Corpus.document` linear scan** (efficiency): per `context_head` call over
  typically few documents; the reviewer's own caveat.
- **`_read`'s description pre-check duplicating `register()`** (altitude): kept
  deliberately — the pre-check skips-with-a-log where `register()` would raise
  and take the whole mount down with one bad skill.
