# ph-runtime-guest

*The other side of the process boundary: the interpreter that runs the code that
the model writes.*

The Python code runtime of pH has two halves in two venvs. The **host** half is
[`ph_rlm.kernel`](../ph-rlm/), which starts the other half and governs it. This
package is the **guest** half. It runs inside `$PH_CACHE/runtime-venv`, in a
subprocess that the host starts for each agent. It reaches the host over one
newline-delimited JSON channel on **fd 3**.

The guest imports neither `ph-core` nor `ph-rlm`, and that is its purpose. The
process boundary exists so that code written by a model cannot reach the
harness. An import of the harness would put the harness back inside the
boundary. The only dependency is `dill`, which is imported lazily. A venv
without `dill` still runs cells, and snapshots only the values that the standard
pickler accepts.

Two modules are copies of ph-core code. A test keeps each copy equal to its
source:

- **`_json.py`** holds the narrowings that this package needs from `ph.json`.
  There is one today, `as_str`. It is copied and not imported for the reason
  above: `ph.json` ships in the ph-core wheel, and this package exists to not
  depend on that wheel. `test_protocol_mirror.py` compares each definition with
  the definition in ph-core, character for character. A copy that drifts
  therefore fails in that test and not in a guest.
- **`_characters.py`** is the whole of `ph.characters`: the rule that a cut
  follows to fall between characters. The guest needs it to cap what a cell
  prints in bytes without breaking a character. The same test compares
  everything below its docstring, including the tables.

The same test file also holds the third copy in this package,
`truncation_marker`, equal to the one in `ph.text`.

```bash
python -m ph_runtime          # how the host starts it; not a command you type
```

**This package registers no rows.** It has no `ph.plugins` entry point and no
`ph.bundles` entry point. It is never mounted, and it appears in no profile.
Everything about it is controlled from the host side. See *Configuring it*
below.

## What it provides

| Module | What it owns |
|---|---|
| `channel` | Newline-delimited JSON over one duplex descriptor. `json.dumps` never emits a literal newline, so one line is exactly one frame |
| `protocol` | The fd-3 frame vocabulary, guest side. It is written twice on purpose, and `FRAME_FIELDS` is the *only* declaration of it here |
| `runner` | The run loop: one process, many cells, one namespace, runs serialized |
| `cell` | Compiles one cell so that top-level `await`, top-level `return`, and persistence across cells all work together |
| `proxies` | What the model sees in `globals()`: namespaces in which every call is a governed dispatch |
| `skill` | Python skills as pre-imported callables |
| `limits` | `RLIMIT_CPU`, address space, and log caps, applied in the child before it reports ready |
| `lifecycle` | Ends the guest when its parent ends, on each platform |
| `snapshot` | Serialization of each variable for `kernel/snapshot` |
| `errors` | The two failures that a cell can see, and how they differ |

### Compiling a cell

Three requirements conflict, and `cell.py` exists to reconcile them:

1. Top-level `await` needs the program to be a coroutine on the one event loop
   of the child.
2. Top-level `return` is a syntax error in module code, even with
   `PyCF_ALLOW_TOP_LEVEL_AWAIT`. The body is therefore wrapped in an
   `async def`, which also satisfies the first requirement.
3. A name that is assigned inside a function is *local* to it. A simple wrapper
   would lose every variable that the cell defined, and the persistent namespace
   would stop working.

The wrapper therefore declares as `global` every name that the cell binds at its
top level. The set is computed from the AST and includes the cases that are easy
to forget: `import`, `with … as`, `for`, `except … as`, walrus, `del`, and
function and class definitions. A trailing expression becomes the value of the
cell, as in a REPL.

### What the model sees

The programming model is the model of prime-agent, deliberately and exactly:

```python
files = await tools.glob(pattern="src/**/*.py")
child = await rlm.run("review this diff", name="reviewer")
await agent_message.send("done", receiver_role="parent")
answer = await websearch(query="…")  # a skill: module-as-callable
```

Namespaces are built from what the **`boot` frame declared**, not from a list in
this package. A namespace that a plugin adds therefore reaches the cell without
a change to this package (I1, I7). Unknown attributes and unknown keyword
arguments fail with an error that lists the available names. The reader is a
model. With a silent `None`, a cell would use a turn to discover that a
capability that the model invented does not exist. For the same reason, a skill
that fails to import binds a **stub that explains itself**, and does not leave
the name undefined.

### Cancellation: three mechanisms for three situations

One task reads the channel, and a cell runs in another task. If one task did
both, a cell that awaits `tools.read(...)` would be blocked on a reply that its
own loop was meant to deliver. That is the classic control-channel deadlock, and
it is the reason why fd 3 is not the channel that a run occupies (D5). Three
mechanisms cancel a cell:

- **The cell awaits a reply or a sleep.** The `cancel` frame, or the `SIGINT`
  callback, cancels the task of the cell. `SIGINT` is installed with
  `loop.add_signal_handler`, **not** with the default handler. The default
  handler raises `KeyboardInterrupt` into whatever frame is executing. When a
  cell is `await`ing, that frame belongs to `asyncio`, so the signal killed the
  whole guest and not the cell. That was the common case, not a rare one.
- **The cell spins in Python.** The loop is starved, so neither of those
  arrives. `SIGXCPU` from the per-run CPU budget lands in the frame of the cell,
  because the cell is executing bytecode. That is also why the other mechanisms
  cannot reach it.
- **Neither works in time.** The host escalates to `SIGKILL` and restarts the
  guest. The namespace is lost and the model is told so, which is better than a
  kernel that is stuck.

### Ending with the parent

The operating system does not do this by default. The death of a parent does not
kill its children (POSIX re-parents them to PID 1), and `atexit` never runs under
`SIGKILL`. Each platform needs its own mechanism:

- **Linux:** the guest sets `prctl(PR_SET_PDEATHSIG, SIGKILL)`.
- **macOS:** there is no equivalent. A daemon thread waits on a kqueue exit
  event for the parent (`EVFILT_PROC` / `NOTE_EXIT`) and calls `os._exit` when
  the event arrives.
- **Windows:** this is the job of the host, with a Job Object that has
  `KILL_ON_JOB_CLOSE`. The host does not create one yet, so on Windows the
  orphan journal is the only protection.

The orphan journal of the host is the backstop on all three platforms.

## Configuring it

Nothing here is configured directly. Every setting belongs to the
`code-runtime-python` row of `ph-rlm`. That row starts this package and sends
the `boot` frame that sets its parameters. The **host owns every default**,
because the protocol has two definitions across the two sides, and each value
must have exactly one owner. `boot` carries every limit as a required field. A
guest therefore has nothing to guess, and a changed default cannot have two
meanings at the same time.

```yaml
- id: code-runtime-python
  config:
    python: managed          # which interpreter runs model code
    cpuSeconds: 30           # per cell, not per kernel: RLIMIT_CPU is armed again for each run
    addressSpaceBytes: 2147483648
    maxLogBytes: 65536       # per stream
    maxValueBytes: 65536     # the value that a cell returns
    maxSnapshotBytes: 16777216   # per snapshotted variable
    skills: ["acme-websearch"]
```

The `python` setting selects the interpreter:

- **`python: managed`** *creates* the venv that this package lives in. It runs
  `uv venv --seed` at `$PH_CACHE/runtime-venv`, then `uv pip install
  ph-runtime-guest` plus each configured skill. When pH runs from a checkout,
  the install uses the source directory of the checkout. A local skill directory
  is installed **editable**, because the staleness marker digests the specs and
  not their contents.
- **`python: host`** runs the guest on the interpreter of pH itself. It is fast
  and needs no `uv` and no network, and the test suite uses it. It also puts
  `ph-core`, pydantic, and Textual on the `sys.path` of the child. That is a
  wider surface, which is why it is not the default.
- **`$PH_RUNTIME_PYTHON`** (or `interpreter:`) names an interpreter directly.

Staleness is decided by a marker file (`.ph-runtime.json`), not by a heuristic.
The file records the protocol version, the version of this package, and the
skill set. Any difference causes a rebuild. Without the marker, a guest that is
one protocol version behind would be found only as a refused `boot` at the first
cell of a session (D7).

See [the README of `ph-rlm`](../ph-rlm/README.md#including-it-and-adjusting-it)
for the full row table.

## Limitations, and deliberate choices

- **The frames are written twice, and this package imports nothing from the
  host.** The twin is `ph_rlm.kernel.protocol`. Five constants are declared one
  time, here, and the host imports them: `PROTOCOL_VERSION`, `PROTOCOL_FD`,
  `FD_ENV`, `FRAME_BYTES_ENV`, and `NAMESPACE_ENV`.
  `packages/ph-rlm/tests/test_protocol_mirror.py` keeps the rest equal. It
  checks that the host imports those five constants. It
  compares the required and optional field set of every frame, and it compares
  the truncation marker byte for byte. **That test is the contract.** The frame
  tables have no shared module to fall back on.
- **A refusal cannot be caught.** `ToolFailed` is for the program to handle: a
  timeout, a bad argument, or a missing file. To catch it and try something else
  is correct. `RunStopped` derives from `BaseException`, so `except Exception`
  cannot catch it. A program that can catch a refusal can go around it (C3). A
  CPU budget breach raises in the same way.
- **Runs are serialized.** There is one namespace, and one program runs at a
  time.
- **Snapshots are per variable, and some things are never captured.** An
  unchanged 200 MiB DataFrame emits nothing, because its digest did not change.
  A value that is certainly immutable is compared by *identity*. This is
  deliberately not extended to tuples or frozensets: `t = ([1],)` keeps its
  identity while its contents change, and a fast path that is wrong one time is
  worse than no fast path. The binding proxies, the imported skill modules, and
  everything else that the bootstrap put in `globals()` are excluded. They are
  rebuilt from the `boot` frame, so pickling them would store a stale copy of
  the harness's own surface.
- **The C pickler is tried first.** `dill` is the fallback for what the C
  pickler refuses: a function, a class, or a lambda that a cell defined. Each
  object takes the same path every time, so digests stay stable. `dill.loads`
  reads standard pickle bytes unchanged in both cases.
- **This package must stay nearly free of dependencies.** Every dependency in
  the runtime venv is another import that model code can reach, and another
  thing that can break a session in a way that the host cannot repair.

## Tests

Most of the tests for this package are with the host half, in
`packages/ph-rlm/tests/`: the mirror test above, `test_kernel.py` (the process
boundary), `test_codec.py` (the decoder, fuzzed), `test_lifecycle.py`,
`test_snapshot.py`, `test_venv.py`, and `test_runtime_integration.py`. The guest
is tested from the host side because what matters is that the two halves agree,
and only a test that holds both halves can show that.

This package also has unit tests of its own in `tests/`. They cover behavior
that the host-side tests reach only by accident, such as the binding forms of a
cell, the settlement of a run, and the quiet paths of the snapshotter.
