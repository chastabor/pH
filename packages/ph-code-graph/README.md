# ph-code-graph

*`code_index` and `code_graph`: ask a codebase about its own shape, and get back
a `path:start-end` you can `read`.*

One row, `code-graph`, registering two tools. No Node, no Rust toolchain, no
submodule — tree-sitter through a Python wheel, the graph in stdlib `sqlite3`.

```bash
ph --patch '{insert: [{id: code-graph, name: code-graph}]}' --profile llama \
   -p "index packages/ph-core, then tell me what breaks if I change claim_slot"
```

## The two tools

**`code_index(paths, glob?, forget?)`** parses each file through `ctx.fs` and
records what it defines and references.

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
SDK listing, and *every* registered tool is in that listing — so these arrive as
`await tools.code_index(...)` and `await tools.code_graph(...)` with no Code
Mode work in this package at all. The `rlm-indexed` profile is `rlm-stable` plus
this bundle and `ph-text-index`'s:

```bash
ph --profile rlm-indexed --provider llama --model <model> --mode tui
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

And the parser is the part Python already has. `tree-sitter-language-pack` 1.16
ships **26 languages compiled into a 3.7 MB wheel, working offline** (python,
ts/tsx/js, rust, go, java, csharp, c, cpp, ruby, php, swift, kotlin, scala,
dart, lua, r, and more), with 371 available and the long tail fetched on
demand — which this row never triggers implicitly.

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
  every line pH shows a model is 1-based. Normalised on the way out, so nothing
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

- **the index**: `$PH_CACHE/code-graph/<digest of the workspace root>.db` — the
  cache root because it is rebuildable from the source, and keyed by root so two
  checkouts do not answer each other's questions. `path:` in the row's config
  overrides it.
- **the grammars**: `$PH_CACHE/tree-sitter`, set by the row at mount.

That second one is not housekeeping. The pack materialises even its *bundled*
grammars into a writable cache on first use — the wheel ships them as an
archive, not as loadable libraries — and it **fails hard** when that directory
cannot be created, rather than falling back to the wheel. Its own default is
`$XDG_CACHE_HOME/…`, which is right on a laptop and wrong in a container with a
read-only `HOME`; both measured. So the row hands over the root pH already
designates for rebuildable artifacts, and refuses at mount with a `MountRefusal`
sentence — not a `pathlib` traceback — when even that is unwritable.

`TREE_SITTER_LANGUAGE_PACK_CACHE_DIR` still wins if an operator set it: their
spelling is what makes the variable mean anything.

One consequence worth expecting: the **first** index on a cold grammar cache
pays for materialising the grammar. Measured on `packages/ph-core/src/ph`, that
is 9 s cold against 2 s warm, and 0.1 s when nothing changed.

Provision it on purpose rather than during someone's turn:

```
/code-graph status     # how many grammars are ready, and where they live
/code-graph install    # load each one now, naming any that will not
```

A **command**, not a tool — a person asks the harness to provision, and it costs
no model turn. `ph doctor` reports the same path and the index's state without
mounting an agent. It matters less here than for `ph-text-index`, whose model is
a download rather than an unpack, but the question should have one answer per
plugin asked the same way.

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
