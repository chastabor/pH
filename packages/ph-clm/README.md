# ph-clm

*The model edits its own context. It can tombstone, replace, and rewrite sections
of what it sees. The log keeps every original.*

This package follows [Context Language Models][clm]. The idea of that paper is
that a model keeps the context it works best with when it can edit that context
itself. In pH, an edit never rewrites the log. Each edit is a surface `replace`,
which is the mechanism that compaction and paste offload already use. The view of
the model changes, the log keeps the originals, and the transcript still shows
the user the conversation that they had.

## Sections

The context of the model is divided into **sections**. A boundary is a point
where no tool call waits for a result. A section is therefore a user message, an
assistant reply, or an assistant message together with every tool result that it
asked for. The id of a section is the sequence number (seq) of its first event,
for example `S412`. That number never changes, so an id that the model reads on
one call is still valid on the next call.

## Tools

| Tool | What it does |
|---|---|
| `context_sections` | Shows the map: the id, kind, and size of each section, and how many tokens follow it |
| `context_tombstone` | Replaces a run of sections with a one-line marker that gives the reason |
| `context_replace` | Replaces a run of sections with text that the model wrote: a summary or a rewrite |
| `context_rewrite` | Changes one passage inside the tool output or the reply of one section |
| `context_restore` | Undoes the latest edit of a section, one level. A rewrite is undone exactly. A removed or replaced run comes back as the text that it held |
| `context_recall` | Reads back the originals that a revised section stands for |
| `context_diff` | Shows a unified diff of a section against the originals that it stands for, after the edits that led there |

Every edit reports its cost. The prefix cache of a provider cannot serve anything
after the first change. The provider therefore reads the sections *after* an edit
again, one time.

## Readouts

A readout is one line at the end of a tool result. When the same call edited the
context file, the receipt of that edit follows it. A readout appears when a result
takes the context past a threshold, which is a share of the context window. The
line gives the size of the context and says how to free space.

- **Each threshold is reported one time.** It is reported again only after an
  edit brings the context back below it.
- **The thresholds are the `remindAt` setting of the `clm-context` row.** The
  default is `[0.5, 0.75]`, and `[]` turns readouts off.
- **With compaction mounted, the last reminder moves.** It comes `lastMargin`
  (0.2) below the compaction trigger. With the default trigger of 0.85, that is
  65%. The model then has room to act before compaction writes a summary for it.
- **In Code Mode, the readout shows the full call**, including the keywords.

## The context file

Before each request, the `clm-mirror` row writes the context of the model to a
file in the scratch directory of the agent's workspace. The file has one
`[[SECTION S<id>]]` block for each section. The model edits the file with the
tools that it already uses: `edit`, `sed`, or Python in a cell. When the call
that changed the file finishes, the row applies the changes through the same
editor that the tools use. The changes land in one batch, before the result of
the call. The result carries the receipt.

| Change in the file | Result |
|---|---|
| A section is deleted | A tombstone |
| The text of a tool result is changed | That result is rewritten in place |
| The result lines of a step are removed | The step is replaced by the text that is left |
| The text of a message is changed | A replacement (a reply is rewritten in place) |
| No section lines are left | One replacement for everything editable |

The header line of a revised section says what the section stands for. The text
under the header is only the part that the model can edit.

The row refuses four kinds of change: a reorder, a damaged header, an edit to a
protected section, and a file that was written before the last revision of the
context. The result of the call gives the reason, and the row writes the file
again.

The file holds a whole conversation, so the row keeps it private and short-lived:

- The directory has mode 0700 and the file has mode 0600.
- The file is deleted when the session leaves the store.
- If a link is put in place of the file, the row replaces the link and never
  reads through it.

The row applies only an edit that a tool call makes. If the file changes outside
a call (in an editor, or by another process), the row records the change as
declined and writes the file again before the next request.

To start a session with this bundle:

```bash
phern --profile rlm-clm
```

[clm]: https://arxiv.org/abs/2609.37725
