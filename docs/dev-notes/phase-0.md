# Phase 0 — Spike the core

**Status:** complete · **Gate:** `ruff` + `ruff format` + `mypy --strict` on `ph-core` + 180 tests, green.

What Phase 0 had to prove is narrow and specific: that dsh's event-sourced core
*is portable to Python without giving up the properties it exists for*. Everything
below is either a delivered work item or a trade recorded so a later phase does
not have to rediscover it.

---

## What landed

| Item | Delivered | Where |
|---|---|---|
| P0-01 | `uv` workspace (`ph-core`, `ph-app`, `ph-rlm`, `ph-stabilize`), ruff (line 100), `mypy --strict` on `ph-core`, pytest + anyio, CI on Linux/macOS/Windows × 3.12/3.13, `ph.plugins` entry-point group reserved | `pyproject.toml`, `.github/workflows/ci.yml` |
| P0-02 | `Context`: `plugin`/`inject`/`provide`/`__getattr__`/`effect`/`scope`/`dispose` | `ph/cordis/context.py` |
| P0-03 | `emit` · `bail` · `serial` · `parallel` · `waterfall`; `prepend`, `global_`, scope filtering | `ph/cordis/context.py` |
| P0-04 | `events.declare(name, mode)`; wrong-mode dispatch raises; producer/consumer matrix | `ph/cordis/events.py`, `ph events` |
| P0-05 | YAML rows in file order, id-addressed patches, `${env:VAR:-default}`, `disabled:` predicates, entry-point discovery, **no code evaluation** | `ph/cordis/loader.py` |
| P0-06 | `WireModel` (`alias_generator=to_camel`, `populate_by_name=True`), round-trip property, "no un-aliased field" assertion | `ph/wire.py`, `tests/test_wire.py` |
| P0-07 | `SessionEvent` frozen `dataclass(slots=True)`, `to_wire`/`from_wire`, pin test | `ph/session/events.py` |
| P0-08 | `append()`: lossless snapshot, `seq = len(log)`, freeze, surface validation, contained publication | `ph/session/session.py` |
| P0-09 | `SurfaceManager`, `fold_surface`, tool-result rewrite restriction | `ph/session/surface.py` |
| P0-10 | `derive_messages()` with a per-node cache keyed on `replace_generation`; header/context folds | `ph/session/session.py`, `ph/session/request_header.py` |
| P0-11 | Seed / fork / resume; `session/end-seed`, `header.seedLength`, `OPEN_TURN` refusal | `ph/session/store.py` |
| P0-12 | `Message`/`ContentBlock`/`StreamChunk`, `BlockAssembler`, fake adapter | `ph/llm/`, `ph/testing/` |
| P0-13 | `ReactLoopAgent` — the full lifecycle minus tools | `ph/agent_loop/driver.py` |
| P0-14 | Runtime invariant `messages == derive_messages()` | `ph/agent_loop/invariant.py` |
| P0-15 | JSONL persistence, `session/flush` (parallel), buffered writer, checkpoint policy | `ph/persistence/` |
| P0-16 | `$PH_HOME` / `$PH_CACHE` / `$PH_RUNTIME` with the three-tier resolution and the `/tmp` ownership check | `ph/paths.py` |
| P0-17 | `ph -p`, `--dump-config`, `ph doctor`, `ph events` | `ph_app/` |

**Definition of done, met:** P0-14's invariant fires on a bypassed request; the
wire round-trip property passes; `ph doctor` prints three resolved roots; a
print-mode run writes a JSONL whose envelopes are dsh's, field for field.

---

## Decisions taken inside Phase 0

These are places where the port plan named a *what* and Python forced a *how*.
Each is a divergence from the TypeScript original that a later reader would
otherwise have to reverse-engineer.

### 1. Activation is explicitly settled, not scheduled

Cordis activates a plugin when its injected services appear, on a microtask.
Python has no synchronous await, and a plugin's `apply` is a coroutine, so pH
splits the two halves: `ctx.plugin(...)` mounts, `await ctx.reconcile()` settles
to a fixpoint (activating what became ready, deactivating what stopped being).

The gain is not merely mechanical. A test or a loader now knows *exactly* when
the plugin tree has settled, and a `provide` inside one plugin's `apply` that
unblocks another is resolved in the same call rather than "eventually".

### 2. Three kinds of context, one visibility rule

This one was found by a failing test and is worth stating plainly, because
getting it wrong is silent.

| kind | built by | provides into | its listeners reach |
|---|---|---|---|
| root | `Context()` | itself | everything |
| activation scope | `reconcile()`, for a plugin | the realm the row was mounted in | everything |
| isolated scope | `ctx.scope()`, for an agent | itself | that agent alone |

A row under the root is globally visible, which is what makes rows composable;
an agent's registration shadows the global one for that agent only. One rule,
`Context.reaches(target)`, decides both event dispatch and every scoped registry
(today the prompt sections; in Phase 1, the tool registry), and the isolation
root is fixed at construction so the rule is a lookup, not a chain walk.

The first draft filtered by plain ancestry, which made a *plugin* invisible to
every agent — the persistence backend silently stopped hearing `session/event`.
Ancestry is the wrong question; isolation is the right one.

The registry, not the driver, owns an agent's scope: `ctx.agents.create()`
builds it, provides the handle into it as `agent`, and disposes it. A second
driver row inherits all of that rather than reproducing it.

### 3. `next(*replacement)` in the waterfall

Cordis waterfall listeners rewrite by mutating a shared payload object. pH's
payloads are frozen values (`GenerateOptions`, `SessionEvent`), so that door is
closed by design. Instead `next()` optionally takes replacement arguments.

This is a strict superset of cordis's behaviour and it is more legible: a
listener that rewrites a request says so at the call site, rather than by
mutating an argument three frames up. Phase 1's retry and replay adapters both
need it.

### 4. Freezing, in a language with no `Object.freeze`

`append()` validates and detaches the payload (`snapshot_json_value`), then
converts it to a read-only view: mappings become `MappingProxyType`, sequences
become tuples. `event.data["x"] = 1` raises.

The consequence to remember: **frozen data must be thawed before it re-enters
the snapshotter**, because `snapshot_json_value` deliberately refuses tuples (a
tuple would silently come back a list). The fork/seed path does exactly that,
and `SessionEvent.to_wire()` thaws on the way out.

The rejection list is stricter than dsh's in one place: integers outside
±(2⁵³−1) are refused. Python's JSON survives them; a JavaScript reader does
not — and Q2's whole point is that dsh tooling reads pH logs.

### 5. `append` is synchronous and I/O-free — and a test says so

`test_append_is_synchronous_and_io_free` asserts it against the source. That
looks paranoid until you notice what depends on it: every observer on the
post-commit feed would otherwise be running behind disk latency, and the
checkpoint policy would have nothing left to decide.

### 6. Config is data — enforced at parse time

The loader uses a narrowed `SafeLoader` that refuses **every** tag, and drops
the implicit timestamp resolver so a value that looks like a date stays the
string the author wrote. dsh's `!!js` idiom is the one thing deliberately not
ported (D9), and the refusal happens when the file is read, not when the value
is used.

### 7. Only `true` is a valid `ignorable`

`{"ignorable": false}` is refused on read. Absent means required; a writer
setting it to `false` has misunderstood the field rather than been thorough,
and accepting it would make two encodings of one meaning.

### 8. A loop request is one that names no other purpose

The I3 invariant needs to know which requests it governs. The first draft put a
`loop_request: bool` on `GenerateOptions` — set by the loop, honoured by the
check — which made the invariant opt-in by the party it polices: a middleware
that rebuilt the request and left the flag at its default passed untouched.

Now there is one field. A request is a loop request when it is session-bound
and `purpose` is unset; compaction and session-title calls (Phase 1) name their
purpose and are exempt. Nothing outside the loop can opt a conversation request
out by omission — it would have to claim a purpose it does not have.

### 9. Acceptance happens where every seed passes, not in one backend

The known-types refusal — an unrecognized *required* event makes the log
unreadable rather than partially readable — sat in the JSONL reader. Every
other seed path (fork, `SessionStore.create(seed=…)`, the Phase 1 replay
adapter, `ph session import`) would have reconstructed a wrong session without
complaint. It now lives in `Session.__init__`'s seed admission, and a test
scans every `append(` call site in `ph-core` against the known set so a type
this build can write but would refuse to read cannot ship.

### 10. Two projections, both named

`derive_messages()` is what the model sees: the surface, with compacted ranges
shadowed. `transcript()` is what the human saw: every append-origin message,
compaction or not. Print mode had reimplemented the second one privately; the
Phase 2 TUI and the `json`/`transcript` modes would each have done so again and
drifted the moment compaction landed.

---

## Deliberately deferred

Not gaps — scope. Each is scheduled and named so nothing here reads as an
oversight.

| Deferred | To | Why |
|---|---|---|
| Tool registry and the execution pipeline | P1-01…P1-04 | Phase 0's loop ends a turn at the assistant message by design ("no tools") |
| Crash repair (`interruptedTurnClosers`) | P1-12 | Needs the tool vocabulary it synthesizes results in |
| The other two flush barriers (before tool dispatch, at step end) | P1-11 | Both are tool-pipeline positions |
| SQLite persistence | P5-08 | The `SessionPersistence` contract is already one Protocol; a backend is a row swap |
| `adapter_defaults` re-resolution | P1-15 | The plumbing is in place (`_request_proposal`); no adapter reports defaults yet |
| Token metering | P1-13 | The fake adapter reports usage; nothing consumes it |
| `ph session import` | later | `populate_by_name` already makes every reader tolerant of both casings |
| A registry for third-party session event types | Phase 1+ | `KNOWN_SESSION_EVENT_TYPES` is a frozenset; a plugin appending its own type must mark it `ignorable` until the vocabulary is declared the way events are |

## Known sharp edges

* **`emit` with an async listener** schedules a task and does not await it,
  which is cordis's behaviour. `ctx.drain()` exists for shutdown. Anything that
  must be awaited belongs on `parallel` or `serial`, not `emit`.
* **`ctx.reconcile()` is not reentrant** — calling it from inside a plugin's
  `apply` would re-enter the fixpoint loop. Nothing does today.
* **A resumed session's turn counter** comes from the last `turn/start` in the
  log. Correct for resume; when Phase 1 adds crash repair, the synthesized
  closers must land before the counter is read.
