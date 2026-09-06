# `ctx.tools` — the registry, the visibility rules, and the pipeline

**Module:** `ph/tools/registry.py` · **Row:** `tools` · **Consumers:** the agent
loop, Code Mode, every policy row, every front end

Not in `ph/seams/` — `ctx.tools` is core rather than a pluggable capability:
there is one registry, and what varies is what is registered *into* it. Three
separable things live in one module because they share one traversal.

To *write* a tool, read [Adding a tool](../cookbook/adding-a-tool.md). This page
is about the registry a tool lands in and the pipeline that runs it.

## Registration and shadowing (B7)

```text
ctx.tools.register(definition, *, scope=None)   -> Disposer
ctx.tools.restrict(NameFilter(...), *, scope=None)
ctx.tools.guard(guard, *, scope=None)
ctx.tools.present_as(mode, *, scope=None)
```

`scope=None` registers globally — the deployment's tool. A `scope` registers for
that agent and unwinds with it, shadowing a global tool of the same name.

**Visibility intersects, and only downward.** `NameFilter(allow=…, deny=…)` with
`None` meaning "no opinion" in both directions: a filter can only remove a name
another filter allowed, never restore one another removed. Restrictions apply to
**global** names only, so a restriction can never silence a tool an agent
registered for itself — an agent's own capability is not something a deployment
policy can take away by accident.

Reading the surface, always for a stated scope:

```text
ctx.tools.names(scope=...)      ctx.tools.schemas(scope=...)     # what the model is offered
ctx.tools.get(name, scope=...)  ctx.tools.mode_for(scope)
ctx.tools.view(scope)                                            # visible + transport + namespaces
```

`tools/change` is emitted when the visible set changes; consumers re-read
`schemas()` rather than caching.

## The pipeline (B1–B5)

One call, in order:

```text
tool/call appended                      # log first, act second
  tools/pre-execute   waterfall  -> allow | deny | ask
  approval on `ask`                     # ctx.approval; only allowed-once proceeds
  guards              deny-only, last, final
  tools/execute       around, signal-only replacement
  the body
  tools/post-execute  waterfall  -> accept | block
  normalize -> finalize_content
  tools/result        emit
tool/result appended
```

Four properties worth relying on:

* **`tool/call` is appended before the body runs.** A crash mid-tool leaves
  evidence the attempt happened, and replay sees a call for every result.
* **A crashing body cannot take the loop down.** Any exception is normalized into
  a structured `is_error` result carrying a `FailureKind` — `denied`, `failed` or
  `aborted`. Which one is not cosmetic: Code Mode ends the whole run on `denied`
  (C3) and lets the program handle `failed`.
* **Guards run *after* approval and are the final word.** A guard is deny-only
  and monotonic, so it overrides even an explicit human approval. Policy that
  must not be reorderable belongs in a guard rather than a `pre-execute`
  listener. (The plans' summary tables list these two the other way round; that
  is a transcription slip, not a second design.)
* **`tools/execute` is *around* the body** and may replace the cancel signal —
  timeouts, retries, metrics — but not the arguments. Substitution is approval's
  (`Edited`), applied by `prepare()` and recorded as `substituted`.

## Presentation: native, code, or both

```text
PresentationMode = "native" | "code" | "both"
```

Under `code`, the model is offered **one** callable — the transport, `run_code`
— plus a generated SDK listing, and a native call to any other name is refused
before any listener runs (C6, `UNKNOWN_TOOL` before policy). `present_as(mode,
scope=)` sets it per agent over the deployment default; `register_transport` and
`present_transport` let a profile rename the transport, which is why nothing may
hard-code `run_code`.

`register_code_namespace(name, factory, scope=)` contributes a namespace beyond
`tools` — `rlm`, `agent_message` — each binding a governed call that re-enters
this pipeline.

## Ownership

A registration records **who** made it (P6-12, P6-29), and the tool's `execute`
runs re-bound to that row rather than to whoever happened to be dispatching. You
get this by registering through `register()`; you lose it by stashing a
definition and calling it yourself.

`scope` is a `Context` for registrations (it must be owned by something that can
unwind) and a `Boundary` for reads (it is only a question about reach).

## Events

| event | mode | |
|---|---|---|
| `tools/pre-execute` | waterfall | allow, deny or ask — hooks, permissions, sandbox |
| `tools/execute` | waterfall (around) | timeouts, retries, metrics; may replace the signal |
| `tools/post-execute` | waterfall | accept, replace a projection, or block with feedback |
| `tools/result` | emit | the frozen authoritative outcome of one call |
| `tools/change` | emit | the visible set changed; re-read `schemas()` |
| `tool/call`, `tool/result` | *session events* | the durable record |

## Extending it

Most extensions are a listener, not a tool:

```python
@plugin("my-policy", inject=["tools"])
async def apply(ctx: Context, config: Config) -> None:
    async def gate(execution, next_):
        if execution.name == "bash" and _looks_destructive(execution.arguments):
            return Ask()          # -> ctx.approval
        return await next_()

    ctx.on("tools/pre-execute", gate)
    ctx.tools.guard(refuse_after_budget)          # deny-only, final
    ctx.tools.restrict(NameFilter(deny={"bash"}), scope=child.ctx)
```

Read the frozen argument tree rather than its JSON rendering. `ph-stabilize`'s
`destructive.py` records why: a classifier over `dumps(arguments)` had every
pattern stop firing the moment a payload was on a second line, because escaping
turned a newline into the characters `\` and `n` and `\b` cannot match between
`n` and a letter. It parses the structure now.

## What it does not do

* It does not decide what needs approval — that is a policy row returning `ask`.
* It does not bound what model-authored code does inside a cell; a deny-list
  needs a registered name (N1). That is the containment ladder's job.
* It does not persist registrations. The tool set is a property of the mounted
  profile plus whatever agents registered for themselves, rebuilt every mount.

## See also

[Adding a tool](../cookbook/adding-a-tool.md) · [`ctx.approval`](approval.md) ·
[`ctx.fs`](fs.md) · `test_tools_registry.py`, `test_tools_pipeline.py`,
`test_tools_batch.py`, `test_code_mode.py`
