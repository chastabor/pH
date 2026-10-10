# ph-clm

*The model edits its own context: tombstone, replace and rewrite sections of what
it sees, with every original kept in the log.*

After [Context Language Models][clm]: a model keeps the context it works best with
when it may edit that context itself. In pH an edit never rewrites the log. Each
one is a surface `replace` — the mechanism compaction and paste offload already
use — so the model's view changes, the log keeps the originals, and the person's
transcript still shows the conversation they had.

## Sections

The model's context is cut into **sections** where no tool call is outstanding: a
user message, an assistant reply, or an assistant message with every result it
asked for. Each is named by the seq of its first event (`S412`), which never
changes, so an id read on one call is good on the next.

## Tools

| tool | does |
|---|---|
| `context_sections` | the map: each section's id, kind, size, and how much follows it |
| `context_tombstone` | replace a run of sections with a one-line marker saying why |
| `context_replace` | replace a run of sections with text the model wrote — a summary, or a rewrite |
| `context_rewrite` | change one passage inside one section's tool output or reply |
| `context_recall` | read back the originals a revised section stands for |
| `context_diff` | a unified diff of a section against the originals it stands for |

Every edit reports what it costs: a provider's prefix cache cannot serve anything
after the first change, so the sections *after* an edit are read again once.

## The context file

The `clm-mirror` row writes the model's context to a file in the agent's workspace
scratch before each request, one `[[SECTION S<id>]]` block per section. The model
edits it with whatever it already uses: `edit`, `sed`, Python in a cell. When the call
that touched the file finishes, its changes land through the same editor the tools
use, in one batch, before the call's result. The result carries the receipt.

| in the file | becomes |
|---|---|
| a section deleted | a tombstone |
| a tool result's text changed | that result rewritten in place |
| a step's result lines removed | the step replaced by the text left |
| a message's text changed | a replacement (a reply is rewritten in place) |
| no section lines left | one replacement for everything editable |

A reorder, a damaged header, an edit to a protected section, or a file written before
the context was last revised is refused. The call's result says why, and the file is
rewritten.

```bash
phern --profile rlm --patch '{insert: [{id: clm-context, name: clm-context}]}'
```

[clm]: https://arxiv.org/abs/2609.37725
