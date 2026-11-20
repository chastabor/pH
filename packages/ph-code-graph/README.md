# ph-code-graph

*`code_index` and `code_graph`: ask a codebase about its own structure, and get
back a `path:start-end` that you can `read`.*

This package has two tools, a command that runs them, and a
[UTCP](https://utcp.io) manual that describes them. It needs no Node, no Rust
toolchain, and no submodule. It uses tree-sitter through a Python wheel, and
keeps the graph in stdlib `sqlite3`.

```bash
ph-code-graph call code_index --paths packages/ph-core --format text
ph-code-graph call code_graph --mode impact --query claim_slot --format text
```

The same two tools reach a model in three ways: inside a
[phern](https://github.com/chastabor/pH) session, typed at a shell, or registered
with any other harness that supports UTCP. Only the first way needs phern.

## Installing

The package needs Python 3.12 or newer, on Linux or macOS.

**With phern, there is nothing to do.** `uv tool install phern` installs this
package too, and the `rlm-indexed` profile mounts it.

**Without phern**, for a shell or for another harness:

```bash
uv tool install ph-code-graph        # or: pip install ph-code-graph
ph-code-graph install                # fetch the grammars now, not during a call
ph-code-graph status
```

That installs the `ph-code-graph` command and nothing from phern. The
dependencies are `ph-core` (the registry that the tools are written against),
`ph-utcp` and the two UTCP libraries that it names, and tree-sitter with its
language pack. That is 24 packages and about 30 MB, measured in an empty
environment. There is no model and no torch. The grammars are fetched from
GitHub one time, by `ph-code-graph install` or by the first index that can reach
GitHub. After that, nothing needs the network.

**The command arrives in 0.11.** `ph-code-graph` 0.10.0 is the plugin that ran
inside a phern session. It has no command and no manual.

**From a checkout**, `uv sync` puts the command in the environment of the
workspace (`uv run ph-code-graph ...`), and
`uv tool install ./packages/ph-code-graph` puts it on `PATH`.

## Three ways in

### In a phern session

The `code-graph` row registers the manual of this package with `ph-utcp`, and the
two tools appear under their own names. A profile layers the `utcp` bundle and
then this bundle. `rlm-indexed` does that:

```bash
phern --profile rlm-indexed -p "index packages/ph-core, then tell me what breaks if I change claim_slot"
```

On another profile, add the two rows:

```bash
phern --profile llama --provider llama --model <model> \
   --patch '{insert: [{id: utcp, name: utcp}, {id: code-graph, name: code-graph}]}' \
   -p "what calls claim_slot?"
```

Or add them in a profile document:

```yaml
- id: utcp
  name: utcp
- id: code-graph
  name: code-graph
  config:
    languages: [python]
```

The index is built for you. Before the first step of the agent, an indexing run
starts in the background over the `index` scope of the row (see "The two
tools"). `/code-graph status` and `/code-graph install` ask the command about its
grammars, and `phern doctor` reports where the index is.

**Run `/code-graph install` one time, before the first session.** The grammars
come from GitHub, and a confined agent cannot reach GitHub. With an empty grammar
cache, the index of such an agent skips every file. The note that tells the agent
that the run ended then names each file as skipped because no parser was
available.

### From a command line

Use the command line to try a query by hand, or to see why a query returned
nothing. Run the command **in the tree that you ask about**. The workspace is the
working directory, or the value of `--root`.

```bash
ph-code-graph call code_index --paths packages/ph-core --format text
ph-code-graph call code_graph --mode callers --query claim_slot --format text
ph-code-graph tools      # the tools and their flags
ph-code-graph status     # grammars, and what the index holds
```

| Command | What it does |
|---|---|
| `call TOOL [--field value ...]` | Runs one tool. Each field of the tool is a flag, or pass all fields as `--args-json '{...}'` |
| `tools` | Lists the tools and their flags |
| `status` | Shows which grammars are ready, where the index is, and what it holds |
| `install` | Fetches every grammar now |
| `manual` | Prints the UTCP manual. `manual --registration` prints what a harness registers |

**A call prints one JSON object** and nothing else: `{"ok": true, "value": ...,
"text": ...}`, or `{"ok": false, "error": {"code": ..., "message": ...}}`.
`value` is the result of the tool as data. `text` is the same result, written
for a model to read. `--format text` prints only `text`. The exit code is 0, or
1 for a failure. A failure is also in the object in both cases.

**Every call is one command.** No process stays running between calls, and no
process listens on a socket. Two flags of `call` are for a run that takes a long
time:

- **`--progress`** reports the progress of the run on stderr as JSON Lines, one
  JSON object on each line. While the run works, it writes
  `{"state":"working","done":4,"total":8,"what":"files"}` at most every 0.5 s.
  It then writes exactly one last line:
  - `"state":"done"`, with `changed` and a `note`.
  - `"state":"busy"`, when another run held the index and this run did nothing.
  - `"state":"failed"`, with a `code` and a `failure`.

  If the stream ends with no last line, the command was killed or it died.
  stdout does not change: it carries one object.
- **`--wait SECONDS`** is how long the call waits for another run that holds the
  index. It overrides the `wait_s` of the row. `--wait 0` answers immediately.

**Settings** are the settings of the session row, under the row id. They are read
from three places, and a later place overrides an earlier one:
`$PH_HOME/tools/ph-code-graph.yaml`, then `$PH_TOOL_CONFIG` (JSON), then
`--config FILE`:

```yaml
# ~/.ph/tools/ph-code-graph.yaml
code-graph:
  languages: [python]
  index:
    paths: [src]
    exclude: ["**/tests/**"]
```

**Four variables control where things go.** None of them has to be set:

| Variable | Default | What the command keeps there |
|---|---|---|
| `PH_HOME` | `~/.ph` | `tools/ph-code-graph.yaml`, the settings above |
| `PH_CACHE` | `$XDG_CACHE_HOME/ph`, which is usually `~/.cache/ph` | The grammars, the log, and the index when nothing else says where |
| `PH_TOOL_STATE` | unset | The directory for the index. `--state DIR` does the same |
| `PH_TOOL_CONFIG` | unset | The row settings as one JSON object, for a caller that writes no file |

See "Where things live" for each path.

### From another harness, over UTCP

Install the package where the harness runs its commands. Then register what this
command prints:

```bash
ph-code-graph manual --registration
```

```json
{"name": "ph_code_graph",
 "call_template_type": "cli",
 "commands": [{"command": "/path/to/python -P -m ph_code_graph manual"}],
 "inherit_env_vars": ["PATH", "HOME", "...", "PH_HOME", "PH_CACHE", "PH_TOOL_STATE"]}
```

The harness runs that command one time to read the manual. After that, it runs
the command that each tool names. This example uses the Python UTCP client,
which this package installs:

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
  client puts the name of the manual before each tool name.
- **The registration names this install.** Its command is the absolute path of
  the interpreter in which the package is installed, so it needs no `PATH`.
  Print the registration again after the package moves. `-P` keeps the working
  directory off the module path, so the command imports no file from the tree.
- **The workspace is the directory in which the command starts.** Start the
  harness in the tree, or make it run its commands there. A `working_dir` on the
  registered template applies to reading the manual and not to the tool calls,
  so it does not move the workspace.
- **Give the model `text`.** Each answer is the envelope above, parsed. `text` is
  what a model in a phern session reads, and `value` is for code. A failure is
  `ok: false` with a `code`. The UTCP command-line protocol ignores exit codes,
  so the envelope is the only place that reports a failure.
- **Optional arguments can be left out.** The Python client logs "Missing
  argument" for each one, and the call still runs.
- **Nothing watches the tree.** Make the model call `code_index` again after it
  edits. A file that has not changed is not parsed again.
- **One index for each tree.** Without `PH_TOOL_STATE`, the index is kept under
  `$PH_CACHE`, keyed by the workspace. A harness whose agents work in separate
  trees needs nothing more. A harness that wants an index for each agent sets
  `PH_TOOL_STATE` for each agent.
- **In a sandbox, the command writes three places**: the directory of the index,
  `$PH_CACHE/tree-sitter`, and `$PH_CACHE/logs`. It writes nothing in the tree.
  The grammars come from GitHub the first time, so run `ph-code-graph install`
  outside the sandbox. After that the tools need no network.
- **One indexing run at a time.** A `code_index` call is one command, and it
  finishes before it answers. A call that is made while another run holds the
  index waits `wait_s` for that run. It then answers `state: indexing`: another
  run is building the graph, and this call did nothing. `code_graph` never
  waits. It answers from what is indexed so far, with `indexing: true` in its
  value.
- **Progress is for a harness that starts the command itself.** A standard UTCP
  client merges stderr into stdout, so no template of the manual carries
  `--progress`. A harness that wants indexing to run in the background runs
  `ph-code-graph call code_index --progress` in the tree and reads stdout and
  stderr separately. Its model can call `code_graph` during the run.
- **The instructions for the model ship with the package**, as
  `ph_code_graph/skills/code-graph/SKILL.md`: index first, ask the question that
  you have, then read the narrow result.

The `ph-utcp` row of phern is one such harness. The next section describes what
it adds.

### Where the tools run, and what limits them

The tools run in the process of the `ph-code-graph` command. In a pH session,
`ctx.shell` starts that command for the agent that called the tool, in the
workspace of that agent and under its sandbox. A child agent in its own worktree
therefore indexes that worktree, and the harness process never loads
tree-sitter.

**`permissions-fs` does not screen the reads of these tools.** Its rules govern
calls through the `ctx.fs` of a session, and a command is not such a call. The
sandbox limits the command, as it limits `bash`.

**A read-only session protects the repository, not the index.** These tools
change no file in the tree. They build data that is used to query the tree. The
index is therefore kept in the scratch directory of the calling agent,
`$PH_HOME/scratch/<session>/<agent>/code-graph/`. That directory is writable in
every sandbox mode, and the tree is writable only in `workspace-write`. An agent
that may not write a tree can still index it, and cannot write the tree by doing
so. Every agent shares the grammar cache and the log of the command, under
`$PH_CACHE`, and the row grants those two paths.

## The two tools

**`code_index(paths?, glob?, exclude?, forget?)`** parses each file and records
what it defines and references.

When a call does not say what to index, the `index` block of the row decides:

```yaml
- id: code-graph
  name: code-graph
  config:
    index:
      paths: [packages/ph-core]        # directories or files; default ["."]
      include: ["**/*.py"]             # default ["**/*"]; the language filter is the real filter
      exclude: ["**/tests/**", "**/vendor/**"]
      on_mount: true                   # index this when a root session starts; default true
    languages: [python]
    wait_s: 15                         # how long code_index waits before it answers "indexing"
```

**Indexing starts without a request, and it does not delay a call.** Before the
agent of a root session takes its first step, `ph-utcp` starts one `code_index`
run over the `index` scope, in the background. The run is a job of that agent,
under its sandbox. The footer shows the count as it changes. The agent is told
when the run ends, if the run changed the graph. `on_mount: false` leaves
indexing to `code_index` calls.

A `code_index` call starts a run, or joins the same request if it is already in
progress, and waits `wait_s` for it:

- If the run ends in time, its result is the answer to the call.
- If not, the call answers `state: indexing`, with a line that gives the
  progress of the run. The run continues, and the agent is told when it ends.

**One run at a time for each index.** A run holds the lock of the index while it
works. `state: indexing` means that another run holds the index and this call
did nothing. The totals beside it are what the graph holds so far. A different
request that is made during a run is not queued, so call `code_index` again
after the run ends. `code_graph` takes no lock and never waits. It answers from
what is indexed so far, and the `indexing` field of its value is `true` while a
run holds the index.

**A child agent builds its own graph only when it has a tree that it writes to
permanently.** A child that works in the checkout of its parent, or that can
only read the project, asks the graph of its parent. That includes every child
of a read-only fan-out. The results say which tree their paths belong to. A
child with its own worktree or overlay has files that diverge, so it gets its
own graph at its first call.

How the scope settings work:

- **`paths`** points the row at one package of a monorepo. A bare `code_index`
  then stays under `max_files` and does not walk everything. A file that is
  named directly in `paths` is indexed whatever is excluded.
- **`exclude`** is read against the workspace root, so `**/tests/**` means the
  same thing under every entry of `paths`. A directory that it covers is never
  entered.
- **The `glob` of a call** replaces `include` for that call.
- **The `exclude` of a call** is *added* to the `exclude` of the row. A call can
  narrow the scope of a deployment, but it cannot remove one of its exclusions.

**Nothing watches the filesystem.** There is no daemon and no file watcher. The
index changes when `code_index` runs, and at no other time. After your own
edits, run it again. That is cheap by design, so it can be a habit.

**Indexing is incremental in two layers**, and the order matters:

1. *Does this file need to be opened?* The answer comes from **git or jj**,
   which already hold a content-addressed tree (`ph.seams.changes`). A file that
   the version control vouches for is never read. Measured on
   `packages/ph-core/src/ph`: a re-index of 137 unchanged files went from 0.10 s
   to **0.03 s**, with zero reads and zero hashes. The `tree_state` call cost
   7 ms for the whole repository of 509 files.
2. *Is the opened content different?* The answer comes from the **sha256** of
   the file, exactly as before. It never comes from an mtime. A checkout or a
   rebase changes timestamps without changing content, and nobody attributes the
   cost correctly when files are indexed again because git touched them.

Layer 1 decides only what to open, so the guarantee of layer 2 is unchanged. A
skip is always justified by a content hash and never by a clock. A tree with no
version control loses layer 1 and nothing else.

**`code_graph(mode, query?, …)`** answers six questions:

| `mode` | Question | What comes back |
|---|---|---|
| `search` | "something about workspace tiers" | Symbols ranked by FTS5 over name and docstring |
| `define` | "where exactly is `claim_slot`" | Every definition of that name |
| `callers` | "what calls this" | The calling symbol **and the calling line** |
| `callees` | "what does this call" | The definitions that it reaches |
| `impact` | "what breaks if I change this" | Transitive callers, ring by ring |
| `entities` | "what is big here" | Definitions by span, biggest first |

Every result carries `path:start-line-end-line`. That is the purpose of the
package: the model finds what matters in one call, and then reads exactly that.

## Using it from the RLM

There is nothing to do. In Code Mode the model gets one callable and a generated
SDK listing. *Every* registered tool is in that listing, including a tool that
arrives through a manual. These tools therefore arrive as
`await tools.code_index(...)` and `await tools.code_graph(...)`, with no Code
Mode work in this package. The `rlm-indexed` profile is `rlm-stable` plus this
bundle and the bundle of `ph-text-index`:

```bash
phern --profile rlm-indexed --provider llama --model <model> --mode tui
```

This package registers a `ph.bundles` entry point. That lets `phern` compose
the profile without importing this package. It also means that an install
without this distribution sees no `rlm-indexed`, and does not see one that
fails at mount.

## It is a name-based graph, and it says so

A reference records the *name* that it used, and `callers` and `callees` join on
that name. Two `register` methods in two classes are one name to this index.

Every answer that could be ambiguous carries `definitions`, the number of places
that define the name. When that number is above one, the rendered text says so
directly. The model then sees the ambiguity, and is not given one of the
possibilities without notice. `mode=define` lists the definitions. It does not
say which definition an edge belongs to. No name-based index can say that.

A resolved graph needs an import graph, scope, and type inference for each
language. In the Rust/Node CodeGraph that is 29,708 lines of TypeScript, and
this package deliberately did not port it. Name-based matching answers most of
what an agent asks, and reports where it cannot. That is a better trade than a
resolved graph for one language.

## Why Python-native, and not a wrapper

The obvious plan was to add the Rust/Node CodeGraph as a submodule and wrap its
kernel with maturin. The investigation rejected that plan, for three reasons in
increasing order of importance:

1. **`codegraph-kernel` is a `#[napi]` crate**: `crate-type = ["cdylib"]`
   against `napi` and `napi-derive`. maturin builds PyO3 and cffi extensions. A
   wrapper would need a fork of the crate with a rewritten boundary.
2. **It is only an extractor.** Its whole export surface is `extract_file`,
   `contract_info`, `grammar_info`, and two `cfnptr` helpers. In its own words:
   "tree-sitter parse+extract with one JS boundary crossing per file". It also
   has a complete TS/WASM fallback, so it is an accelerator and not the core.
3. **The intelligence is TypeScript**: `resolution/` has 29,708 lines,
   `extraction/` 25,612, `mcp/` 13,940, `graph/` 6,023, and `db/` 5,475. That is
   112,612 lines against 25,242 in the kernel. A wrapper around the kernel gives
   a parser and leaves the product behind.

Python already has the parser. `tree-sitter-language-pack` is a 3.7 MB wheel. It
calls 26 languages bundled (python, ts/tsx/js, rust, go, java, csharp, c, cpp,
ruby, php, swift, kotlin, scala, dart, lua, r, and more) and has 371 available.
**None of them works offline on a fresh cache.** Measured 2026-11-11 with 1.16.2
and 1.25.0: with no network and an empty cache, no parser can be obtained,
including the python parser. With the network, the bundled set arrives as one
archive from GitHub the first time a parser is requested. After that, the pack
needs only the cache.

## Why `sqlite3` and not `pyturso`

The session log of pH runs on turso, so this package is the exception, and the
reason is measured. Against a real 40 MB CodeGraph database (11,396 nodes,
36,830 edges), `pyturso` served indexed lookups, joins, and aggregates
correctly. It then failed in three ways:

- **FTS5 is absent.** `pyturso` does not see a `CREATE VIRTUAL TABLE … USING
  fts5`, or its shadow tables. `search` needs FTS5.
- **`Recursive CTEs are not yet supported`.** `impact` needs a recursive CTE.
- **`json_extract` silently returns NULL** where SQLite returns the value:
  `0.95` against `None`, with the same rows and the same expression. A wrong
  answer with no error is worse than a missing feature.

Two of the six modes here are exactly the two that turso cannot serve. Stdlib
`sqlite3` (SQLite 3.45, with FTS5 compiled in) adds no dependency and serves all
six. If turso gets both features, `_store.py` changes one import.

## Extraction: two passes, and why

- **`process()`** is the intelligence layer of the pack. It gives definitions
  with kinds, the imports of the file, docstrings, and comments. It does not
  give *references*: `SymbolInfo` says that a function exists, never who calls
  it.
- **The tags query** is the `tags.scm` of tree-sitter for each language. GitHub
  builds its code navigation on the same data. It gives `@definition.*` and,
  most important, `@reference.call`.

Measured over `packages/ph-core/src/ph`: 136 files and 1.58 MiB take 219 ms
through `process()` and 108 ms through the tags query. An index of the core of
this repository takes 2.0 s end to end, including the SQLite writes, and 0.1 s
when nothing has changed. The second parse gives more value than it costs.

Four details are easy to get wrong. Each is handled one time in `_extract`, and
each has a test:

- **`ProcessConfig` spans are 0-based.** Tree-sitter points are also 0-based,
  and every line that pH shows a model is 1-based. The numbers are normalized on
  the way out, so nothing downstream has to remember which pass a number came
  from. An error here gives a pointer that is off by one, which makes a model
  stop trusting a tool.
- **`SymbolInfo.doc` is always `None`.** The pack does not set it (measured
  against a file full of docstrings). Code that trusted it would have shipped an
  index with an empty `doc` column and a `search` mode that matched names only.
  The prose is on two other channels. `result.docstrings` carries the pack's own
  `associated_item`. `result.comments` carries doc comments (Rust `///`,
  TypeScript block comments, Go `//`). Those comments have no association, so
  the rule is positional and narrow: the last line of the comment must be
  *immediately* above the first line of the definition.
- **The `span.end_line` of a comment node can be one past the line that the
  comment occupies**, because the `///` node of Rust includes its trailing
  newline. With `end_line`, the rule missed the definition directly below *and*
  matched one a blank line away. The line count now comes from the text of the
  comment.
- **The tags query of TypeScript is only part of the query.** Its grammar
  extends the javascript grammar, and upstream splits the queries in the same
  way. `class`, `function`, and `call` are in the *javascript* query. A `.ts`
  file that is matched against the typescript query alone yields two tags and no
  calls. `INHERITS` fixes that.

## Limits for each language

Tests cover both of these, so they are known shapes and not surprises:

- **C has definitions but no call references.** Its tags query ships no
  `@reference.*` captures. `callers` and `callees` are therefore empty for C,
  while `search`, `define`, and `entities` work.
- **A bare Ruby send is not seen.** `helper` without parentheses parses as an
  identifier and not as a `call`, so only `helper()` becomes an edge.

Only `reference.call` is stored. The tags queries also emit `reference.type`,
`reference.class`, and more. Each is a real relationship, but with them
`callers` would return every place that only *mentions* a type.

## Where things live

- **The index, in a session**: `graph.db` in the scratch directory of the agent
  that owns the tree, `$PH_HOME/scratch/<session>/<agent>/code-graph/`. The
  owner is the calling agent, or the ancestor whose tree it shares. A child in
  its own worktree indexes its own tree, so two agents on different trees never
  answer from the line numbers of the other. A resumed session has the same
  session id and agent id, and therefore the same index. `code_index` then
  parses again only what changed. The cost is that a new session builds its own
  index and does not find the index of the last session.
- **The index, from a shell**: `--state DIR` puts it in `DIR`, and so does
  `$PH_TOOL_STATE`. `ph-utcp` sets that variable for each call, and another
  harness can set it. With neither, the index is
  `$PH_CACHE/code-graph/<digest of the workspace root>/graph.db`. It is under
  the cache root because it can be rebuilt from the source. It is keyed by the
  root so that two checkouts do not answer the questions of each other. That
  directory records its workspace. When the workspace has been gone for a week,
  the next index removes the directory.
- **The grammars**: `$PH_CACHE/tree-sitter`. The engine row sets this when the
  command mounts it.
- **The writer lock**: `graph.db.lock` beside the database. It allows one
  indexing run at a time for each index, across every process on the workspace.
  A second run waits `wait_s` for the first. It then finds most of the tree
  current, or answers `state: indexing`.
- **The log of the command**: `$PH_CACHE/logs/ph-code-graph.log`. A call prints
  one JSON object, and prints nothing on stderr except the progress lines that
  `--progress` asks for. A traceback therefore goes to the log.

The location of the grammars is important. The pack keeps every grammar that it
uses in a writable cache. That includes its bundled grammars, which it downloads
there on first use. The pack **fails hard** when it cannot create that
directory. Its own default is `$XDG_CACHE_HOME/…`, which is correct on a laptop
and wrong in a container with a read-only `HOME` (both measured). The row
therefore gives the pack the root that pH already uses for rebuildable
artifacts. When even that root is not writable, the row refuses to mount with
one sentence, not with a `pathlib` traceback. From the command, that sentence is
the answer of the call, with the code `MOUNT_REFUSED`.

`TREE_SITTER_LANGUAGE_PACK_CACHE_DIR` still wins if an operator set it. Their
value is what keeps the variable meaningful.

Expect two consequences:

- The **first** index on a cold grammar cache pays for the grammar fetch.
  Measured on `packages/ph-core/src/ph`: 9 s cold against 2 s warm, and 0.1 s
  when nothing changed.
- A command that cannot reach GitHub indexes nothing until the cache is filled.
  That applies to the command of a confined agent, and to a machine with no
  network. Each file is skipped, with a reason that names `/code-graph install`.

Provision the grammars deliberately, not during the turn of a user:

```
/code-graph status     # how many grammars are ready, and where they are
/code-graph install    # load each one now, and name any that will not load
```

This is a **command**, not a tool. A person asks the harness to provision, and
that costs no model turn. In a session, it runs `ph-code-graph status` or
`ph-code-graph install` and shows the output. From a shell, type those commands
directly. `phern doctor` reports the path of the index, whether the index is
built, and whether the manual was discovered. It does not start the command.
`install` fetches all eighteen grammars in about three seconds. It runs outside
the sandbox, because it needs the network.

## The skill

The package installs a `code-graph` skill. An RLM under progressive disclosure
sees one line in its catalog and can read the page when it needs it. The page
says: index first, ask the question that you have, then `read` the narrow
result. It also says that a name-based graph reports ambiguity and does not
resolve it.

The row registers the skill. A directory scan does not find it. The skill
therefore arrives exactly when the tools do, and is removed with them.
`skills-progressive` ships an empty `paths` on purpose, because a distribution
installs a skill deliberately (I7).

## Tests

`tests/test_code_graph.py` runs the real parser and the real SQLite index. It
deliberately tests three things:

- **Line spans against the file.** Each span is checked by slicing the real
  source at the span, not against an expected number. A pointer that is off by
  one still looks plausible in a diff.
- **The enclosing-symbol rule.** A call inside a nested closure belongs to the
  closure, not to the 40-line function around it.
- **Incrementality.** A second index pass over an unchanged tree parses nothing,
  and a file that was touched but not modified stays unchanged.

`tests/test_code_graph_cli.py` starts real processes, because its subject exists
only between processes:

- The command answers with the values that the row answers with.
- Two processes that index one tree report counts that add up.
- A real UTCP client registers the manual and calls a tool, with a list argument
  and with optional arguments left out.

`tests/test_code_graph_row.py` covers the row that a session mounts:

- The tools keep their names, values, rendered text, and error codes.
- An agent in its own tree indexes that tree.
- A confined agent that may not write its tree still indexes it.
- A fresh interpreter that mounts only the rows of the session never loads
  tree-sitter.
