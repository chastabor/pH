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

```bash
phern --profile rlm --patch '{insert: [{id: clm-context, name: clm-context}]}'
```

[clm]: https://arxiv.org/abs/2609.37725
