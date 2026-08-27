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
| P3-15 | `kernel/snapshot` and `kernel/restored`: per-variable, tagged, spilled, folded | `ph_rlm/snapshot.py` |

**Still to come:** P3-09…P3-14 and P3-16…P3-25 — the presentation alias, the
binding namespaces, the subagent provider, messaging and the registry, the RLM
prompt, the Continual Harness, the context loader, Python skills, the TUI code
cell, the profile bundle, the two conformance gates, the fixture replay, and the
trajectory view.

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
* **No `display` consumer.** The frame is defined, decoded and collected into
  `_ActiveRun.displays`, and nothing renders it until P3-19's code cell.
* **The runtime is effectively asyncio-only.** `anyio.wait_readable` is
  backend-agnostic, but the guest uses `asyncio` directly (it must: it cannot
  depend on anyio), and D3 already fixes asyncio for the whole app because
  Textual requires it.

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
