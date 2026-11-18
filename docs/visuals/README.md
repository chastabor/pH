# pH visuals

Diagrams of pH that open in a browser, with no server and no network. Each page
is one picture and a drill-down: **click a box or an arrow** to open what is
behind it, or press **Details** (bottom left) for everything the page can open.

A box that is a row opens to its row entry, where it ships and whether it ships
switched off, and every option it takes. An arrow opens to how it is carried (a
bus event and its dispatch mode, a service call, a frame on fd 3, a request to a
server) and the structure of what is sent.

| page | shows |
|---|---|
| [overview.html](overview.html) | how the nine packages work together, and which profile layers which |

Across packages:

| page | shows |
|---|---|
| [dispatch.html](dispatch.html) | how the parts talk: emit, serial, parallel and waterfall, every declared event, and what is not the bus |
| [invariants.html](invariants.html) | the five invariants, the mechanism behind each, what holds it, and where it is imperfect |
| [durability.html](durability.html) | what is written where, what a restart reads back, and what is lost |
| [security.html](security.html) | scopes, isolation and visibility, what a sub-agent inherits, the containment ladder, and the limits |
| [self-steering.html](self-steering.html) | a self-steering skill from the ask to the last step, and how to switch it on |

One per package:

| page | shows |
|---|---|
| [ph-core.html](ph-core.html) | the loop, the log and the seams |
| [phern.html](phern.html) | from a command to a model call |
| [ph-rlm.html](ph-rlm.html) | how a code cell runs |
| [ph-runtime-guest.html](ph-runtime-guest.html) | one cell run over fd 3, frame by frame |
| [ph-stabilize.html](ph-stabilize.html) | each row and the event or seam it hooks |
| [ph-clm.html](ph-clm.html) | how a context edit lands |
| [ph-text-index.html](ph-text-index.html) | files to search hits |
| [ph-code-graph.html](ph-code-graph.html) | files to answers |
| [ph-utcp.html](ph-utcp.html) | how a manual becomes a tool |

`page.html?drill=node:<id>`, `edge:<id>`, `topic:<id>` or `index` opens a page
with that panel already showing.

## Changing one

The pages are generated; edit the sources and build.

```bash
uv run python docs/visuals/builder/build.py                # every page
uv run python docs/visuals/builder/build.py ph-stabilize   # one
uv run python docs/visuals/builder/build.py --check        # resolve the details, render nothing
uv run python docs/visuals/builder/build.py --facts        # read the code again first
```

A page has two sources:

* `src/<page>.<type>.json` is the diagram, written for
  [Archify](https://github.com/tt-a1i/archify), which lays it out and renders
  the page. The build looks for it at `$ARCHIFY_HOME`, else
  `~/.claude/skills/archify`.
* `src/details/<page>.yaml` is the drill-down. Its `nodes:` and `edges:` are
  keyed by the ids in the diagram, and the build refuses an id the diagram does
  not have.

**Most of a details file is a name, and the build looks the rest up in the
code**, so those parts say what the code says on the day the page is built. What
is written by hand is the prose around them, and that goes stale like prose.

### The details file

```yaml
intro: |                       # shown at the top of the page's index
  Prose.
topics:                        # things that belong to the page, not to one box
  - id: durability
    title: What survives a restart
    summary: Prose.
    blocks: [...]
nodes:
  <node id>:
    title: skill-steps         # optional; the diagram's label when left out
    summary: Prose.
    blocks: [...]
edges:
  <edge id>:
    transport: bus             # bus | service | frame | socket | http | process | log | file | call
    mode: waterfall            # for a bus event: emit | serial | parallel | waterfall
    summary: Prose.
    blocks: [...]
```

A block is one of these, with an optional `title:`, a `note:` shown under it,
and for a `list` a `tone:` of `limit`, `secure` or `durable`.

| block | you write | the build adds |
|---|---|---|
| `row: skill-steps` | a registered row's name | its module, what it injects, which shipped file mounts it and whether it is switched off there, how to switch it on, and its option table from its config model |
| `event: agent/pre-step` | a declared event; for one declared with no payload type, `args:` rows of `[argument, type, what it is]` | its dispatch mode, who declares it, which rows listen in which profiles, and its payload's fields |
| `events: waterfall` | a mode, a namespace or `all` | a table of every such event |
| `bundle: ph_stabilize/bundle.yaml` | the end of a shipped YAML file's path | a table of every row that file mounts: on or off there, and its option names |
| `struct: ph.llm.adapter:DecisionRequest` | `module:Class`; `fields: [a, b]` to show only those; `methods: true` or `methods: [a, b]` to list methods beside the fields | the class's summary, and each field's type, default and docstring; its public methods where it has no fields (a Protocol, a service) or where asked |
| `frame: done` | a guest protocol frame | who sends it and its required and optional fields; add `example:` |
| `frames: all` | `all`, `host` or `guest` | a table of every frame that side sends |
| `record: todo/write` | a log record type | who writes it, whether an older reader may skip it, what telemetry ships; add `example:` |
| `code: \|` | text, and `lang:` | |
| `text: \|` | paragraphs, `- ` lists and `1.` lists | |
| `list: [...]` | items | |
| `table: {columns: [...], rows: [[...]]}` | cells | |
| `links: [{to: ..., label: ...}]` | targets | |

Prose takes `` `code` ``, `**bold**` and `[links](target)`. A target is
`node:<id>`, `edge:<id>` or `topic:<id>` on the same page, `<page>#node:<id>` on
another, a page name, or a URL.

The build prints the nodes and edges a details file leaves out.
