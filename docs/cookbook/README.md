# Cookbook — extending pH

Four recipes, in the order most people need them:

1. [Adding a plugin](adding-a-plugin.md) — the unit everything else is packaged
   in. Read this first; the other three assume it.
2. [Adding a tool](adding-a-tool.md) — a capability the model can call.
3. [Adding an adapter](adding-an-adapter.md) — a new provider wire.
4. [Adding a seam](adding-a-seam.md) — a new *kind* of capability, with
   pluggable providers.

## The shape all four share

A **plugin** is a name, a list of injected service keys, an optional config
model, and an `apply(ctx, config)` body. A **row** is one line of a profile
naming a plugin and its config. Mounting a profile runs each row's `apply` in
file order.

`apply` does exactly one thing: it *registers*. A tool, an adapter, a listener, a
prompt section, a service — each registration returns a disposer that the calling
scope owns, so unloading the row unregisters everything it added (invariant I2).
Nothing you write needs an `unload()`.

```python
@plugin("tool-clock", inject=["tools"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    ctx.tools.register(define_tool(...))  # disposer owned by this scope
```

`inject` is not documentation — it **gates activation**. A row whose injected
keys are not provided by anything in the profile never runs, which is how
`subagent-task` ships in `ph-base` and registers nothing until a profile layers a
`ctx.subagents` provider. A tool advertised in every prompt and refused on every
call has taught the model a capability the deployment does not have.

## Four rules that will bite you if you learn them by discovering them

They are `plans/Implementation_Plan.md` §5, and they are not style:

* **Declare, never derive.** A wire alias is fixed at class definition; an event
  mode at `events.declare`; a tool's output schema at `define_tool`. Nothing
  reconstructs a name from a string later.
* **Log first, act second.** `tool/call` is appended before the tool runs; the
  snapshot event before the blob. A crash between the two must leave evidence
  that the attempt happened.
* **Every artifact through `ctx.effect()`.** Child processes, temp directories,
  locks, worktrees. A lint fails the build on a raw `subprocess.Popen` or
  `tempfile.mkdtemp` outside the seams — see `test_resources.py`, which is the
  enforcement rather than a convention.
* **Fail closed at the seam.** `SANDBOX_UNAVAILABLE` rather than unconfined; an
  unavailable approval denies; an unresolvable credential reference is refused.
  Where you cannot fail closed, say so next to where a reader would assume
  otherwise — a caveat only in the docs is a defect.

## Where the rest of the answer lives

Every seam's module docstring argues its own design at length, and is the
authority for that seam. These pages tell you which module to open and what the
shape is; they do not restate it. The [seam reference](../seams/) indexes them.
