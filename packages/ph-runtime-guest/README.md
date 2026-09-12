# ph-runtime-guest

*The other side of the process boundary: the interpreter the model's code
actually runs in.*

pH's Python code runtime is two halves in two venvs. The **host** half is
[`ph_rlm.kernel`](../ph-rlm/), which spawns and governs; this is the **guest** —
it runs inside `$PH_CACHE/runtime-venv`, in a subprocess the host starts per
agent, and reaches the host over one newline-delimited JSON channel on **fd 3**.

It imports neither `ph-core` nor `ph-rlm`, and that is the whole point: the
process boundary exists so that model-written code cannot reach the harness, and
importing the harness would put it back inside. Its only dependency is `dill`,
imported lazily — a venv without it still runs cells and forgoes snapshots.

One module bends that without breaking it. `_json.py` holds the narrowings this package needs out of `ph.json` — one today, `as_str` — copied rather than imported for the reason above: that module ships in the ph-core wheel, and depending on the wheel is what this package exists not to do. `test_protocol_mirror.py` compares each definition against ph-core's character for character — it holds this package's other copy, `truncation_marker`, to `ph.text`'s in the same file — so a copy that drifts fails there rather than in a guest.

```bash
python -m ph_runtime          # how the host spawns it; not a command you type
```

**This package registers no rows.** It has no `ph.plugins` and no `ph.bundles`
entry point, is never mounted, and appears in no profile. Everything about it is
governed from the other side — see *Configuring it* below.

## What it provides

| module | what it owns |
|---|---|
| `channel` | newline-delimited JSON over one duplex descriptor. `json.dumps` never emits a literal newline, so a line is exactly a frame |
| `protocol` | the fd-3 frame vocabulary, guest side — written twice on purpose, and `FRAME_FIELDS` is the *only* declaration of it here |
| `runner` | the run loop: one process, many cells, one namespace, runs serialized |
| `cell` | compiling one cell so that top-level `await`, top-level `return` and cross-cell persistence all hold at once |
| `proxies` | what the model sees in `globals()`: namespaces whose every call is a governed dispatch |
| `skill` | Python skills as pre-imported callables |
| `limits` | `RLIMIT_CPU`, address space and log caps, applied in the child before it reports ready |
| `lifecycle` | dying with the parent, per platform |
| `snapshot` | per-variable serialization for `kernel/snapshot` |
| `errors` | the two failures a cell can see, and the line between them |

### Compiling a cell

Three requirements pull against each other, and reconciling them is the only
interesting thing in `cell.py`. Top-level `await` needs the program to be a
coroutine on the child's one loop. Top-level `return` is a syntax error in
module code even with `PyCF_ALLOW_TOP_LEVEL_AWAIT`, so the body is wrapped in an
`async def` — which makes the first fall out for free. But a name assigned
inside a function is *local* to it, so the naive wrapping would lose every
variable the cell defined and quietly undo the persistent namespace. So the
wrapper declares every name the cell binds at its top level as `global`,
computed from the AST including the cases that are easy to forget — `import`,
`with … as`, `for`, `except … as`, walrus, `del`, and function and class
definitions. A trailing expression becomes the cell's value, as in a REPL.

### What the model sees

The programming model is prime-agent's, deliberately and exactly:

```python
files = await tools.glob(pattern="src/**/*.py")
child = await rlm.run("review this diff", name="reviewer")
await agent_message.send("done", receiver_role="parent")
answer = await websearch(query="…")  # a skill: module-as-callable
```

Namespaces are built from what the **`boot` frame declared**, not from a list in
this package, so a namespace a plugin adds reaches the cell without this module
changing (I1, I7). Unknown attributes and unknown keyword arguments fail loudly
with the available names in the message — the reader is a model, and a silent
`None` is how a cell spends a turn discovering a capability it invented does not
exist. A skill that fails to import binds a **stub that explains itself** rather
than leaving the name undefined, for the same reason.

### Cancellation, three mechanisms for three situations

The channel is read by one task and a cell runs in another — otherwise a cell
awaiting `tools.read(...)` would be blocked on a reply its own loop was supposed
to deliver, which is the classic control-channel deadlock and the reason fd 3 is
not the channel a run occupies (D5). On top of that:

- awaiting a reply or a sleep → the `cancel` frame, or the `SIGINT` callback,
  cancels the cell's task. `SIGINT` is installed with
  `loop.add_signal_handler`, **not** the default handler: that one raises
  `KeyboardInterrupt` into whatever frame is executing, and when a cell is
  `await`ing that frame is `asyncio`'s own — so the signal killed the entire
  guest instead of the cell, in the common case rather than a rare one;
- spinning in Python → the loop is starved, so neither arrives. `SIGXCPU` from
  the per-run CPU budget lands in the cell's own frame, because it is executing
  bytecode, which is exactly why it is unreachable by the others;
- neither works in time → the host escalates to `SIGKILL` and restarts. The
  namespace is lost and the model is told so, which beats a wedged kernel.

### Dying with the parent

The OS does not do what one would hope: a parent's death does not kill its
children (POSIX re-parents them to PID 1) and `atexit` never runs under
`SIGKILL`. Each platform needs its own mechanism and only one is in the guest's
gift — **Linux** sets `prctl(PR_SET_PDEATHSIG, SIGKILL)` here; **macOS** has no
equivalent, so a daemon thread watches `os.getppid()` and `os._exit`s when it
changes; **Windows** is the host's job, a Job Object with `KILL_ON_JOB_CLOSE`.
The host's orphan journal is the backstop for all three.

## Configuring it

Nothing here is configured directly. Every knob belongs to `ph-rlm`'s
`code-runtime-python` row, which spawns this package and sends the `boot` frame
that parameterises it — and the **host owns every default**, because there are
two definitions of the protocol across the two sides and exactly one owner of
each value. `boot` carries every limit as a required field, so a guest has
nothing to guess and a changed default cannot mean two things at once.

```yaml
- id: code-runtime-python
  config:
    python: managed          # which interpreter runs model code
    cpuSeconds: 30           # per cell, not per kernel — RLIMIT_CPU is re-armed each run
    addressSpaceBytes: 2147483648
    maxLogBytes: 65536       # per stream
    maxValueBytes: 65536     # per snapshotted variable
    maxSnapshotBytes: 16777216
    skills: ["acme-websearch"]
```

`python: managed` is what *creates* the venv this package lives in: `uv venv
--seed` at `$PH_CACHE/runtime-venv`, then `uv pip install ph-runtime-guest` (the
checkout's own source directory when pH is running from one) plus each
configured skill — a local skill directory **editable**, because the staleness
marker digests the specs rather than their contents. `python: host` runs the
guest on pH's own interpreter instead: fast, needs no `uv` and no network, and
what the test suite uses — and also what puts `ph-core`, pydantic and Textual on
the child's `sys.path`, which is a wider surface and the reason it is not the
default. `$PH_RUNTIME_PYTHON` (or `interpreter:`) is the third answer.

Staleness is a marker file (`.ph-runtime.json`), not a heuristic: it records the
protocol version, this package's version and the skill set, and any difference
rebuilds. A guest one protocol behind would otherwise be discovered as a refused
`boot` at the first cell of somebody's session (D7).

See [`ph-rlm`'s README](../ph-rlm/README.md#including-it-and-adjusting-it) for
the full row table.

## Limitations, and things that are deliberate

- **The protocol is written twice and neither side imports the other.**
  `ph_rlm.kernel.protocol` is the twin. What keeps them honest is
  `packages/ph-rlm/tests/test_protocol_mirror.py`, which compares
  `PROTOCOL_VERSION`, every frame's required and optional field set, and the
  truncation marker byte for byte. **That test is the contract**; there is no
  shared module to fall back on.
- **A refusal is not catchable.** `ToolFailed` is the program's to handle — a
  timeout, a bad argument, a missing file — and catching it to try something
  else is exactly right. `RunStopped` derives from `BaseException` so
  `except Exception` cannot swallow it, because a program that can catch a
  refusal can route around it (C3). A CPU budget raises the same way.
- **Runs are serialized.** One namespace, one program at a time.
- **Snapshots are per variable, and some things are never captured.** An
  unchanged 200 MiB DataFrame emits nothing because its digest did not move. A
  definitively-immutable value is compared by *identity* — deliberately not
  extended to tuples or frozensets, since `t = ([1],)` keeps its identity while
  its contents change, and a fast path that is wrong once is worse than none.
  The binding proxies, the imported skill modules and anything else the
  bootstrap put in `globals()` are excluded: they are rebuilt from the `boot`
  frame, so pickling them would store a stale copy of the harness's own surface.
- **The C pickler is tried first**, with `dill` as the fallback for what it
  refuses (a cell-defined function, class or lambda). Each object takes the same
  path every time, so digests stay stable — and `dill.loads` reads standard
  pickle bytes unchanged either way.
- **This package must stay nearly dependency-free.** Every dependency in the
  runtime venv is another import model code can reach and another thing that can
  break a session the host cannot repair.

## Tests

Its suite lives with the host half, in `packages/ph-rlm/tests/` — the mirror
test above, plus `test_kernel.py` (the process boundary), `test_codec.py` (the
decoder, fuzzed), `test_lifecycle.py`, `test_snapshot.py`, `test_venv.py` and
`test_runtime_integration.py`. Testing the guest from the host's side is the
honest arrangement: what matters is that the two halves agree, and only a test
that holds both can say so.
