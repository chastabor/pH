# `ctx.fs` — filesystem access with a gate before every access

**Module:** `ph/seams/fs.py` · **Row:** `fs-local` · **Consumers:** `tool-fs`,
`tool-attach`, `ctx.shell`, `ctx.attachments`, `ctx.workspace`, `rlm-context-loader`

Every read, write and edit passes through a waterfall *before* it touches the
disk. That ordering is the whole value: a policy plugin that ran after the write
would be a reporter, not a gate.

## What a consumer may assume

That a path it hands over is resolved against **this agent's** root, that policy
has been asked before anything happened, and that a refusal arrives as a
`FsDenied` — a `HarnessError` whose `failure_kind` is `denied`, so it survives to
a consumer as a *denial* rather than an ordinary failure. Under Code Mode that
difference decides whether a model's program ends (C3) or catches the veto and
routes around it.

Two arguments are required and mean different things. **Pass both.**

| | |
|---|---|
| `scope` | the *policy* boundary — which screens apply, which intent listeners the gate reaches |
| `agent` | the *physical* key — whose worktree the path resolves in, where an approval prompt is routed |

Nothing checks them against each other, deliberately: `agent=child,
scope=parent.ctx` gets the child's worktree under the parent's rules, which is
legal by construction and invisible at the call site. `scope` is required on the
five model-facing methods because there is no honest fallback for "which boundary
is this" once an agent is in play (P6-24, P6-32).

## The surface

```text
await ctx.fs.read(path, *, scope, offset=0, limit=2_000, agent=None, session=None)  # -> FileSlice
await ctx.fs.read_bytes(path, *, scope, max_bytes=None, agent=None, session=None)   # -> bytes
await ctx.fs.write(path, content, *, scope, agent=None, session=None)               # -> Path
await ctx.fs.edit(path, old, new, *, scope, replace_all=False, agent=None, session=None)  # -> int
await ctx.fs.glob(pattern, *, scope, root=None, limit=1_000, agent=None)            # -> list[str]
await ctx.fs.grep(pattern, *, scope, root=None, glob="**/*", limit=200, agent=None) # -> list[GrepMatch]
```

`read` returns a **line window** (`FileSlice`: text, offset, lines, total_lines,
truncated) — the model asks for the next one by offset. `read_bytes` is the whole
file and exists for one reason: media (P7-01). It is a separate method rather
than a flag because `offset`, `limit`, `total_lines` and `truncated` are all
statements about *lines*, and a `bytes`-or-`FileSlice` return would push that
branch into every caller. Its `max_bytes` is answered from the file's own size
before it is opened, so refusing a 2 GB video costs a `stat`, and raises
`FileTooLarge` — a **failure**, not a denial: no policy refused anything, the
caller named a bound.

Path helpers, all pure:

* `resolve(path, agent=)` — relative against the agent's root; **absolute passes
  through**, because refusing it here would be a confinement claim this layer
  cannot make (N2).
* `named(path, agent=)` — how a path is *written down*: relative to the agent's
  root. Every path that reaches the model or the log takes this form, because a
  read echoing `/tmp/ph-w-7/src/x.py` puts the machine and the run into the
  conversation and moves the provider's cached prefix (A11/A12) for a difference
  the conversation cannot see. A path *outside* the workspace keeps its absolute
  form — it is not a name the workspace can express.
* `root_for(agent=)` — the agent's cwd, what `bash` and `glob` run against.

## Two contribution points, and they answer different questions

This is the distinction most policy rows get wrong.

**`ctx.on("fs/<read|write|edit>-intent", listener)` — *may I open it?***
A waterfall, awaited, one path at a time. Veto by returning a reason string; the
seam raises `FsDenied(reason)`. The intent carries `path`, `scope`, `agent` and,
for writes, the content.

**`ctx.fs.screen(decide, scope=)` — *may I be told it exists?***
A synchronous predicate over every path `glob`/`grep` considers, answering
`"yield" | "skip" | "prune"`. It runs *during* the walk rather than over the
results, because `grep` reads the files it visits: post-filtering matches would
return no rows having already read every byte of the file the rule protected.
`prune` refuses to enter a directory at all, which is why the callback takes
directories too.

**Filtered is not refused.** A path a screen hides is still readable through
`read` unless a rule also vetoes `fs/read-intent`. `permissions-fs` registers
*both* rather than deriving one from the other, and a deployment reading
`ignore:` as access control has misread it — the ignore list keeps noise out of a
listing and protects nothing.

`screen`'s `scope` is **required**, where every sibling registration defaults it:
P6-18 made this same field the input to `Context.reaches`, so omitting it would
not mean "clean up with the mount", it would mean *screen every agent* — the
widest policy in the seam, chosen by forgetting.

## Events

| event | mode | |
|---|---|---|
| `fs/read-intent` | waterfall | before a read; a veto prevents it |
| `fs/write-intent` | waterfall | before a whole-file write reaches disk |
| `fs/edit-intent` | waterfall | before an in-place edit reaches disk |
| `fs/changed` | emit | a path was written or edited; consumers refresh |
| `fs/observed` | *session event* | this file was read — appended to the log |

`fs/observed` is what lets read-before-edit be a policy row rather than a
hard-coded rule. The `fs-read-before-edit` row refuses an edit to a file this
session has not read since it last changed; it is its own row so a deployment can
drop it, and so it applies to *every* editing tool rather than the one that
remembered to check.

## Per-agent roots

`rebase(resolver, scope=)` claims a **slot** — not a list — answering "where do
this agent's relative paths resolve" (D21). Two answers to that is a
contradiction, and the first-match-wins reading a list would need is one nobody
could configure. `None` from the resolver means the agent has no workspace and
`root` stands; a resolver that *raises* also falls back rather than failing the
call, because an agent whose workspace lookup broke must still be able to read a
file.

The seam does not consult `ctx.workspace` itself — that would make `ctx.fs` know
which seam owns agent state when what it needs is one path. `workspace-lifecycle`
wires the two, and a deployment mounting no such row keeps exactly today's
behaviour.

## The honest scope (N1)

**This bounds *tool-mediated* access.** Model-authored `open(path, "w")` inside a
code cell is unreachable from here by construction — a deny-list needs a
registered name. That is what the containment ladder is for, and no wording
should suggest otherwise. `permissions-fs` says the same sentence in its own
words when no confining provider is mounted:

> `permissions-fs` applies to tool calls through `ctx.fs`; raw
> `open()`/`subprocess` from inside a code cell is not covered. Mount a sandbox
> provider to bound it.

## The row

```yaml
- id: fs
  name: fs-local
  config:
    root: /work          # optional; a deployment saying "always work here"
    ignore: [dist, .git] # directory *names*, pruned from walks
```

The root is resolved most-specific-first: `config.root`, then a `project_root`
provided by whoever mounted (a daemon holds one composition across sessions that
live in different repositories — P5-14), then `Path.cwd()`.

`ignore` is `list[str] | None`, and the three-way distinction is load-bearing:
`None` keeps the built-in list, `[]` turns pruning off entirely, and a list
replaces it. Bare final components, matched anywhere — `"dist"`, not
`"build/dist"`, which would silently never fire.

## Providing a different backend

There is no provider Protocol here: `fs-local` publishes a concrete `FsService`,
and a remote or virtual filesystem would replace the row rather than register into
it. What *is* pluggable is policy — screens and intent listeners — which is where
every shipped extension attaches:

```python
@plugin("my-fs-policy", inject=["fs"])
async def apply(ctx: Context, config: Config) -> None:
    ctx.on("fs/read-intent", refuse_secrets)  # may I open it
    ctx.fs.screen(hide_secrets, scope=ctx)  # may I be told it exists
```

`permissions_fs.py` in `ph-stabilize` is the worked example, and registers exactly
those four things.

## What it does not enforce

* Absolute paths are not refused (N2) — the `worktree` tier bounds a *relative*
  write, and confinement is `ctx.sandbox`'s claim to make.
* A screen that raises is treated as a refusal for that path (`prune` for a
  directory, `skip` for a file) and logged; it does not fail the walk.
* Symlinks are not resolved away before the gate: what the intent carries is what
  `resolve` produced.

## See also

[Adding a seam](../cookbook/adding-a-seam.md) · [Adding a tool](../cookbook/adding-a-tool.md)
· `test_fs.py`, `test_permissions_fs.py` (ph-stabilize)
