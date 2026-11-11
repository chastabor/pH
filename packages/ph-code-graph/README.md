# ph-code-graph

*`code_index` and `code_graph`: ask a codebase about its own shape, and get back
a `path:start-end` you can `read`.*

Two tools, a command that runs them, and a [UTCP](https://utcp.io) manual that
describes them. No Node, no Rust toolchain, no submodule — tree-sitter through a
Python wheel, the graph in stdlib `sqlite3`.

```bash
ph-code-graph call code_index --paths packages/ph-core --format text
ph-code-graph call code_graph --mode impact --query claim_slot --format text
```

The same two tools reach a model three ways: inside a [phern](https://github.com/chastabor/pH)
session, typed at a shell, or registered with any other harness that speaks
UTCP. This package needs phern for the first only.

## Installing

Python 3.12 or newer, on Linux or macOS.

**With phern, nothing to do.** `uv tool install phern` brings this package with
it, and the `rlm-indexed` profile mounts it.

**By itself**, for a shell or for another harness:

```bash
uv tool install ph-code-graph        # or: pip install ph-code-graph
ph-code-graph install                # fetch the grammars now, not during a call
ph-code-graph status
```

That is the `ph-code-graph` command and nothing of phern's: `ph-core` (the
registry the tools are written against), `ph-utcp` and the two UTCP libraries it
names, tree-sitter and its language pack: 24 packages and about 30 MB, measured
in an empty environment. No model and no torch. The grammars are fetched from
GitHub once, by `ph-code-graph install` or by the first index that can reach it,
and nothing needs the network after that.

**The command arrives in 0.11.** `ph-code-graph` 0.10.0 is the plugin
that ran inside a phern session; it has no command and no manual.

**From a checkout**, `uv sync` puts the command in the workspace's environment
(`uv run ph-code-graph ...`), and `uv tool install ./packages/ph-code-graph`
puts it on `PATH`.

## Three ways in

### In a phern session

The `code-graph` row registers this package's manual with `ph-utcp`, and the two
tools appear under their own names. A profile layers the `utcp` bundle and then
this one; `rlm-indexed` does:

```bash
phern --profile rlm-indexed -p "index packages/ph-core, then tell me what breaks if I change claim_slot"
```

On another profile, add the two rows:

```bash
phern --profile llama --provider llama --model <model> \
   --patch '{insert: [{id: utcp, name: utcp}, {id: code-graph, name: code-graph}]}' \
   -p "what calls claim_slot?"
```

Or in a profile document:

```yaml
- id: utcp
  name: utcp
- id: code-graph
  name: code-graph
  config:
    languages: [python]
```

The index is built for you: a host is started at the agent's first step and
indexes the row's `index` scope in the background (see "The two tools").
`/code-graph status` and `/code-graph install` ask the command about its
grammars, and `phern doctor` reports where the index is.

**Run `/code-graph install` once, before the first session.** The grammars come
from GitHub, and a confined agent cannot reach it: with an empty grammar cache
its index skips every file, and the note that tells the agent the run ended
names each as skipped for want of a parser.

### From a command line

To try a query by hand, or to see why one came back empty. Run it **in the tree
you are asking about**: the workspace is the working directory, or `--root`.

```bash
ph-code-graph call code_index --paths packages/ph-core --format text
ph-code-graph call code_graph --mode callers --query claim_slot --format text
ph-code-graph tools      # the tools and their flags
ph-code-graph status     # grammars, and what the index holds
```

| command | what it does |
|---|---|
| `call TOOL [--field value ...]` | run one tool; each of its fields is a flag, or pass them all as `--args-json '{...}'` |
| `tools` | the tools and their flags |
| `status` | which grammars are ready, where the index is and what it holds |
| `install` | fetch every grammar now |
| `manual` | the UTCP manual; `manual --registration` is what a harness registers |
| `serve --socket PATH` | stay up and answer `call`s on a unix socket |

**A call prints one JSON object** and nothing else: `{"ok": true, "value": ...,
"text": ...}`, or `{"ok": false, "error": {"code": ..., "message": ...}}`.
`value` is the tool's result as data and `text` is the same result written for a
model to read; `--format text` prints only that. The exit code is 0, or 1 for a
failure, and the failure is in the object either way.

**Settings** are the session row's own, by row id, read from three places with
the later winning: `$PH_HOME/tools/ph-code-graph.yaml`, then `$PH_TOOL_CONFIG`
(JSON), then `--config FILE`:

```yaml
# ~/.ph/tools/ph-code-graph.yaml
code-graph:
  languages: [python]
  index:
    paths: [src]
    exclude: ["**/tests/**"]
```

**Where things go** is said by five variables, none of which has to be set:

| variable | default | what the command keeps there |
|---|---|---|
| `PH_HOME` | `~/.ph` | `tools/ph-code-graph.yaml`, the settings above |
| `PH_CACHE` | `$XDG_CACHE_HOME/ph`, so `~/.cache/ph` | the grammars, the log, and the index when nothing else says where |
| `PH_TOOL_STATE` | unset | the directory for the index; `--state DIR` is the same |
| `PH_TOOL_CONFIG` | unset | row settings as one JSON object, for a caller that writes no file |
| `PH_TOOL_HOST` | unset | the socket of a `serve` to ask instead of running the call here |

See "Where things live" for each path.

### From another harness, over UTCP

Install the package where the harness runs its commands, then register what this
prints:

```bash
ph-code-graph manual --registration
```

```json
{"name": "ph_code_graph",
 "call_template_type": "cli",
 "commands": [{"command": "/path/to/python -m ph_code_graph manual"}],
 "inherit_env_vars": ["PATH", "HOME", "...", "PH_HOME", "PH_CACHE", "PH_TOOL_STATE", "PH_TOOL_HOST"]}
```

The harness runs that command once to read the manual, and from then on runs the
command each tool names. With the Python UTCP client, which this package
installs:

```python
import asyncio, json, subprocess

from utcp.utcp_client import UtcpClient


async def main() -> None:
    registration = json.loads(
        subprocess.run(
            ["ph-code-graph", "manual", "--registration"], check=True, capture_output=True
        ).stdout
    )
    client = await UtcpClient.create(config={"manual_call_templates": [registration]})
    await client.call_tool("ph_code_graph.code_index", {"paths": ["src"]})
    found = await client.call_tool(
        "ph_code_graph.code_graph", {"mode": "callers", "query": "claim_slot"}
    )
    print(found["text"] if found["ok"] else found["error"])


asyncio.run(main())
```

What a harness needs to know:

- **The tools are `ph_code_graph.code_index` and `ph_code_graph.code_graph`.** A
  client prefixes each tool with the manual's name.
- **The registration names this install.** Its command is the absolute path of
  the interpreter the package is installed in, so it needs no `PATH`, and it is
  printed again after the package moves.
- **The workspace is the directory the command is started in.** Start the
  harness in the tree, or have it run its commands there. A `working_dir` on
  the registered template covers reading the manual and not the tool calls, so
  it does not move the workspace.
- **Give the model `text`.** Each answer is the envelope above, parsed: `text`
  is what a model in a phern session reads, and `value` is for code. A failure
  is `ok: false` with a `code`; the UTCP command-line protocol ignores exit
  codes, so the envelope is the only place it is said.
- **Optional arguments may be left out.** The Python client logs "Missing
  argument" for each one and the call runs.
- **Nothing watches the tree.** Have the model call `code_index` again after it
  edits; a file that has not changed is not parsed again.
- **One index a tree.** Without `PH_TOOL_STATE` the index is kept under
  `$PH_CACHE`, keyed by the workspace. A harness whose agents work in separate
  trees needs nothing more; one that wants an index per agent sets
  `PH_TOOL_STATE` for each.
- **In a sandbox, three places are written**: the index's directory,
  `$PH_CACHE/tree-sitter` and `$PH_CACHE/logs`. Nothing in the tree is. The
  grammars come from GitHub the first time, so run `ph-code-graph install`
  outside the sandbox; after that the tools need no network.
- **A host is optional.** A call takes about 0.3 s by itself. A harness that
  wants indexing to go on in the background starts
  `ph-code-graph serve --socket PATH --warm code_index` in the tree and sets
  `PH_TOOL_HOST=PATH` on its calls: `code_index` then answers
  `state: indexing` after `wait_s` and the run goes on. A call that finds
  nobody on the socket runs by itself. Keep the path short; a unix socket's is
  limited to about a hundred characters.
- **The model's instructions ship with the package**, as
  `ph_code_graph/skills/code-graph/SKILL.md`: index first, ask the question you
  have, then read the narrow thing.

phern's own `ph-utcp` row is one such harness, and what it adds is described
next.

### Where the tools run, and what bounds them

In the `ph-code-graph` command's own process — started, in a pH session, by
`ctx.shell` for the agent that called the tool: in that agent's workspace, under
that agent's sandbox. So a child agent in its own worktree indexes that worktree,
and the harness process never loads tree-sitter.

**`permissions-fs` does not screen these tools' reads.** Its rules govern calls
through a session's `ctx.fs`, and a command is not one. What bounds the command
is the sandbox, as it bounds `bash`.

**Read-only bars the repository, not the index.** These tools change no file of
the tree; they build data to probe it. So the index is kept in the calling
agent's scratch — `$PH_HOME/scratch/<session>/<agent>/code-graph/` — which is
writable under every sandbox mode, while the tree is writable only under
`workspace-write`. An agent that may not write a tree can still index it, and
cannot write it by doing so. The grammar cache and the command's log are shared
by every agent, under `$PH_CACHE`, and the row grants those two paths.

## The two tools

**`code_index(paths?, glob?, exclude?, forget?)`** parses each file and records
what it defines and references.

What it looks at when a call does not say is the row's `index` block:

```yaml
- id: code-graph
  name: code-graph
  config:
    index:
      paths: [packages/ph-core]        # directories or files; default ["."]
      include: ["**/*.py"]             # default ["**/*"] — the language filter is the real one
      exclude: ["**/tests/**", "**/vendor/**"]
      on_mount: true                   # index this when a root session starts; default true
    languages: [python]
    wait_s: 15                         # how long code_index waits before answering "indexing"
```

**It starts unasked, and does not hold a call up.** In a session one
`ph-code-graph serve` stays up for each tree, kept by `ph-utcp` for the agent
whose tree it is. When a root session's agent takes its first step it is started
and indexes the `index` scope in the background; the footer shows the count
moving and the agent is told when it ends. A `code_index` call waits `wait_s`
for its run and then answers `state: indexing` with how far it has got, and
`code_graph` answers meanwhile from what is indexed so far and says so.
`on_mount: false` leaves indexing to `code_index` calls.

**A child builds its own graph only when it has a tree it writes for keeps.** A
child in its parent's checkout, or one granted only to read the project — every
child of a read-only fan-out — asks its parent's graph through its parent's
host, and its results say which tree their paths belong to. A child with a
worktree or an overlay of its own has files that diverge, so it gets its own, at
its first call.

`paths` is how the row is pointed at one package of a monorepo, so a bare
`code_index` stays under `max_files` instead of walking everything. `exclude` is
read against the workspace root, so `**/tests/**` means the same thing under
every entry of `paths`, and a directory it covers is never entered. A call's
`glob` replaces `include` for that call; a call's `exclude` is *added* to the
row's, so a call can narrow a deployment's scope and cannot lift an exclusion
out of it. A file named outright in `paths` is taken whatever is excluded.

**Nothing watches the filesystem.** There is no daemon and no file watcher: the
index changes when `code_index` runs, and at no other moment. So after your own
edits, run it again — which is cheap by design, and cheap enough to be a habit.

**Incremental in two layers**, and the order matters:

1. *Does this file need opening?* Asked of **git or jj**, which already hold a
   content-addressed tree — `ph.seams.changes`. A file the version control
   vouches for is never read. Measured on `packages/ph-core/src/ph`: a re-index
   of 137 unchanged files went from 0.10 s to **0.03 s**, with zero reads and
   zero hashes, on a `tree_state` costing 7 ms for the whole 509-file repo.
2. *Is what I opened different?* Asked of the file's **sha256**, exactly as
   before. Never an mtime — a checkout or a rebase moves timestamps without
   moving content, and re-indexing because git touched them is a cost nobody
   attributes correctly.

Layer 1 only decides what to open, so the guarantee layer 2 makes is untouched:
a skip is always justified by somebody's content hash, never by a clock. A tree
with no version control loses layer 1 and nothing else.

**`code_graph(mode, query?, …)`** answers six questions:

| `mode` | question | what comes back |
|---|---|---|
| `search` | "something about workspace tiers" | symbols ranked by FTS5 over name + docstring |
| `define` | "where exactly is `claim_slot`" | every definition of that name |
| `callers` | "what calls this" | the calling symbol **and the calling line** |
| `callees` | "what does this call" | the definitions it reaches |
| `impact` | "what breaks if I change this" | transitive callers, ring by ring |
| `entities` | "what is big here" | definitions by span, biggest first |

Every result carries `path:start-line-end-line`. That is the point of the
package: the model finds what matters in one call, then `read`s exactly that.

## Using it from the RLM

Nothing to do. Under Code Mode the model is handed one callable and a generated
SDK listing, and *every* registered tool is in that listing — a tool that
arrives through a manual included — so these arrive as
`await tools.code_index(...)` and `await tools.code_graph(...)` with no Code
Mode work in this package at all. The `rlm-indexed` profile is `rlm-stable` plus
this bundle and `ph-text-index`'s:

```bash
phern --profile rlm-indexed --provider llama --model <model> --mode tui
```

This package registers a `ph.bundles` entry point, which is what lets `ph-app`
compose that profile without depending on this distribution — and what makes an
install missing it see no `rlm-indexed` rather than one that fails at mount.

## It is a name-based graph, and it says so

A reference records the *name* it used, and `callers`/`callees` join on that
name. Two `register` methods in two classes are one name to this index.

Every answer that could be ambiguous carries `definitions` — how many places
that name is defined — and the rendered text says so outright when it is above
one, so the model sees the ambiguity instead of being handed one of the
possibilities. `mode=define` is how it disambiguates.

Resolving properly means an import graph, scope, and per-language type
inference. In the Rust/Node CodeGraph that is 29,708 lines of TypeScript, and
this package deliberately did not port it. Name-based matching answers most of
what an agent actually asks and reports where it cannot — which is a better
trade than a resolved graph for one language.

## Why Python-native, and not a wrapper

The obvious plan was to submodule the Rust/Node CodeGraph and wrap its kernel
with maturin. The investigation said no, for three reasons in increasing order
of importance:

1. **`codegraph-kernel` is a `#[napi]` crate** — `crate-type = ["cdylib"]`
   against `napi`/`napi-derive`. maturin builds PyO3/cffi extensions; wrapping
   it means forking the crate to rewrite its boundary.
2. **It is only an extractor.** Its whole export surface is `extract_file`,
   `contract_info`, `grammar_info` and two `cfnptr` helpers — "tree-sitter
   parse+extract with one JS boundary crossing per file", in its own words. It
   also has a complete TS/WASM fallback, so it is an accelerator, not the core.
3. **The intelligence is TypeScript**: `resolution/` 29,708 lines, `extraction/`
   25,612, `mcp/` 13,940, `graph/` 6,023, `db/` 5,475 — 112,612 against the
   kernel's 25,242. Wrapping the kernel gets a parser and leaves the product
   behind.

And the parser is the part Python already has. `tree-sitter-language-pack` is a
3.7 MB wheel with 26 languages it calls bundled (python, ts/tsx/js, rust, go,
java, csharp, c, cpp, ruby, php, swift, kotlin, scala, dart, lua, r, and more)
and 371 available. **None of them works offline on a fresh cache.** Measured
2026-11-11 with 1.16.2 and 1.25.0: with no network and an empty cache no parser
is to be had, python's included; with the network the bundled set arrives as one
archive from GitHub the first time a parser is asked for. After that the cache
is all it needs.

## Why `sqlite3` and not `pyturso`

pH's session log runs on turso, so this is the exception and the reason is
measured. Against a real 40 MB CodeGraph database (11,396 nodes, 36,830 edges),
`pyturso` served indexed lookups, joins and aggregates correctly, and then:

- **FTS5 is absent** — a `CREATE VIRTUAL TABLE … USING fts5` is invisible to
  it, shadow tables included. That is `search`.
- **`Recursive CTEs are not yet supported`** — that is `impact`.
- **`json_extract` silently returns NULL** where SQLite returns the value:
  `0.95` against `None`, same rows, same expression. A wrong answer with no
  error is worse than a missing feature.

Two of the six modes here are exactly the two turso cannot serve. Stdlib
`sqlite3` (SQLite 3.45, FTS5 compiled in) costs no dependency and does all six.
If turso grows both, `_store.py` changes one import.

## Extraction: two passes, and why

- **`process()`**, the pack's intelligence layer — definitions with kinds, the
  file's imports, docstrings and comments. What it will not give is
  *references*: `SymbolInfo` says a function exists, never who calls it.
- **the tags query**, tree-sitter's own `tags.scm` per language — the data
  GitHub's code navigation is built on. `@definition.*` and, crucially,
  `@reference.call`.

Measured over `packages/ph-core/src/ph`: 136 files and 1.58 MiB through
`process()` in 219 ms, through the tags query in 108 ms. Indexing this
repository's core takes 2.0 s end to end including the SQLite writes, and
0.1 s when nothing has changed. Parsing twice is worth more than the parse it
saves.

Four details that are easy to get wrong, each handled once in `_extract` and
each pinned by a test:

- **`ProcessConfig` spans are 0-based** while tree-sitter points are 0-based and
  every line pH shows a model is 1-based. Normalized on the way out, so nothing
  downstream has to remember which pass a number came from. Getting this wrong
  is an off-by-one pointer, which is what makes a model stop trusting a tool.
- **`SymbolInfo.doc` is always `None`.** The pack does not populate it —
  measured against a file full of docstrings. Trusting it would have shipped an
  index with an empty `doc` column and a `search` mode matching names only. The
  prose is on two other channels: `result.docstrings`, which carries the pack's
  own `associated_item`, and `result.comments` for doc comments (Rust `///`,
  TypeScript block comments, Go `//`). Those have no association, so the rule is
  positional and narrow — the comment's last line must be *immediately* above
  the definition's first.
- **A comment node's `span.end_line` can be one past the line it occupies**,
  because Rust's `///` node includes its trailing newline. Trusting it missed
  the definition directly below *and* matched one a blank line away. The line
  count comes from the comment's own text instead.
- **TypeScript's tags query is half the story.** Its grammar extends
  javascript's and upstream splits the queries to match, so `class`, `function`
  and `call` live in the *javascript* query — a `.ts` file matched against the
  typescript query alone yields two tags and no calls. `INHERITS` fixes that.

## Per-language caveats

Both are pinned by tests, so they are known shapes rather than surprises:

- **C has definitions but no call references.** Its tags query ships no
  `@reference.*` captures at all, so `callers`/`callees` are empty for C while
  `search`, `define` and `entities` work.
- **A bare Ruby send is invisible.** `helper` without parentheses parses as an
  identifier rather than a `call`, so only `helper()` becomes an edge.

Only `reference.call` is stored. The tags queries also emit `reference.type`,
`reference.class` and more; each is a real relationship, but folding them in
would make `callers` return every place that merely *mentions* a type.

## Where things live

- **the index, in a session**: `graph.db` in the scratch of the agent whose tree
  it is, `$PH_HOME/scratch/<session>/<agent>/code-graph/`: the calling agent, or
  the ancestor whose tree it shares. A child in a worktree of its own indexes its
  own tree, so two agents on different trees never answer from each other's line
  numbers. A resumed session has the same session and agent ids, so
  the same index: `code_index` then re-parses only what changed. The cost is that
  a new session builds its own rather than finding the last one's.
- **the index, from a shell**: `--state DIR` puts it in `DIR`, and so does
  `$PH_TOOL_STATE` — the variable `ph-utcp` sets per call, and another harness
  can set. With neither it is
  `$PH_CACHE/code-graph/<digest of the workspace root>/graph.db`: the cache root
  because it is rebuildable from the source, and keyed by root so two checkouts
  do not answer each other's questions. That directory records its workspace,
  and one whose workspace has been gone a week is removed on the next index.
- **the grammars**: `$PH_CACHE/tree-sitter`, set by the engine row when the
  command mounts it.
- **the writer lock**: `graph.db.lock` beside the database. One indexing run at a
  time per index, across every process on the workspace; a second run waits for
  the first and then finds most of the tree current.
- **the command's log**: `$PH_CACHE/logs/ph-code-graph.log`. A call prints one
  JSON object and nothing on stderr, so this is where a traceback goes.

That second one is not housekeeping. The pack keeps every grammar it uses in a
writable cache — its bundled ones too, which it downloads there on first use —
and it **fails hard** when that directory cannot be created. Its own default is
`$XDG_CACHE_HOME/…`, which is right on a laptop and wrong in a container with a
read-only `HOME`; both measured. So the row hands over the root pH already
designates for rebuildable artifacts, and refuses to mount with one sentence —
not a `pathlib` traceback — when even that is unwritable. From the command that
sentence is the call's answer, with the code `MOUNT_REFUSED`.

`TREE_SITTER_LANGUAGE_PACK_CACHE_DIR` still wins if an operator set it: their
spelling is what makes the variable mean anything.

Two consequences worth expecting. The **first** index on a cold grammar cache
pays for fetching the grammar: measured on `packages/ph-core/src/ph`, 9 s cold
against 2 s warm, and 0.1 s when nothing changed. And a command that cannot
reach GitHub — a confined agent's, a machine with no network — indexes nothing
until the cache is filled: each file is skipped, with a reason that names
`/code-graph install`.

Provision it on purpose rather than during someone's turn:

```
/code-graph status     # how many grammars are ready, and where they live
/code-graph install    # load each one now, naming any that will not
```

A **command**, not a tool — a person asks the harness to provision, and it costs
no model turn. In a session it runs `ph-code-graph status` or `install` and shows
what that said; from a shell, type those directly. `phern doctor` reports the
index's path, whether it is built and whether the manual was discovered, without
starting the command. `install` fetches all eighteen grammars in about three
seconds, and it runs outside the sandbox, because it needs the network.

## The skill

The package installs a `code-graph` skill, so an RLM under progressive
disclosure sees one line in its catalog and can read the page when it needs
it — index first, ask the question you actually have, then `read` the narrow
thing; and that a name-based graph reports ambiguity rather than resolving it.

Registered by the row rather than found by a directory scan, so it arrives
exactly when the tools do and leaves with them — `skills-progressive` ships an
empty `paths` on purpose, because a skill is something a distribution installs
deliberately (I7).

## Tests

`tests/test_code_graph.py` runs the real parser and the real SQLite index.
Three things it pins deliberately:

- **line spans against the file** — every span is checked by slicing the actual
  source at it, not against an expected number, because a pointer that is off
  by one still looks plausible in a diff;
- **the enclosing-symbol rule** — a call inside a nested closure belongs to the
  closure, not the 40-line function around it;
- **incrementality** — a second index pass over an unchanged tree parses nothing
  and a touched-but-unmodified file stays unchanged.

`tests/test_code_graph_cli.py` starts real processes, because what it holds only
exists between them: the command answers with the values the row answers with,
two processes indexing one tree report counts that add up, and a real UTCP
client registers the manual and calls a tool with a list argument and with
optional arguments left out.

`tests/test_code_graph_row.py` holds the row a session mounts: the tools keep
their names, values, rendered text and error codes; an agent in its own tree
indexes that tree; a confined agent that may not write its tree still indexes
it; and a fresh interpreter that mounts only the session's rows never loads
tree-sitter.
