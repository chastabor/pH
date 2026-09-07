# Adding a tool

A tool is a capability the model can call. Registering one puts it in the prompt,
through the pipeline, into the log and onto a card — so the declaration carries
more than a signature.

Worked examples in the tree, smallest first: `ph/tools/builtin/ask_user.py`,
`attach_tool.py`, `fs_tools.py`.

## The shape

```python
from pydantic import Field
from ph.cordis import Context, plugin
from ph.tools.definition import (
    ToolModel,
    ToolOutput,
    ToolRunContext,
    define_tool,
    text_content,
)
from ph.tools.presentation import simple_views


class ReadArgs(ToolModel):
    path: str = Field(description="Path to read. Relative paths resolve against the workspace.")
    limit: int = Field(2_000, ge=1, le=20_000, description="Maximum lines to return.")


class ReadValue(ToolModel):
    path: str
    text: str
    truncated: bool


@plugin("tool-read", inject=["tools", "fs"])
async def apply(ctx: Context, config: Any) -> None:
    async def read(args: ReadArgs, run: ToolRunContext) -> Any:
        window = await ctx.fs.read(
            args.path,
            limit=args.limit,
            agent=run.agent,
            scope=run.scope,
            session=run.session,
        )
        return window.model_dump()

    ctx.tools.register(
        define_tool(
            "read",
            "Read a file, or a window of one. Prefer this over shelling out to cat.",
            parameters=ReadArgs,
            output=ToolOutput(schema=ReadValue, render=_render),
            execute=read,
            is_concurrency_safe=True,
            self_limits=True,
            effects_confined_to_workspace=True,
            **simple_views("read", "Read", "path"),
        )
    )
```

`parameters` as a pydantic model means the body receives a **validated instance**
— a tool never hand-checks its own input. A raw schema dict is accepted for
declarations pH did not author (MCP, a subagent's shape), and then the tool owns
validation.

## `output` is mandatory, and this is the part people get wrong

`output` is not a return annotation. It is two things:

* **`schema`** — the canonical structured value, which is what the durable record
  holds;
* **`render(args, value) -> content`** — a *pure* projection from the validated
  arguments and the value to what the model reads.

They are separate because a replayed session must render identically to the live
one: the card is drawn from `content` plus `meta` with no access to the live call.
A tool that formatted its text inside `execute` and returned a string would make
the transcript unreproducible.

Return the structured value from `execute`; let `render` turn it into prose.

## Declaring what the tool *is*

These flags are read by policy rows that must not need a list of your tool names:

| flag | what it says | who reads it |
|---|---|---|
| `is_concurrency_safe` | two of these may overlap in one batch | the scheduler (B6) |
| `is_irreversible` | a per-call predicate over the arguments | the approval gate (P6-16) |
| `effects_confined_to_workspace` | every effect is a file inside the tree | `/revert` (N3) |
| `self_limits` | bounds its own output and offers paging | the offload row (G2) |
| `arguments_disposable` | the model need not re-read the arguments | compaction |
| `timeout_ms` | a bound on the body | the pipeline |

They are declared by the tool rather than matched by name elsewhere because tools
are *registered plugins*: a deployment renames them and an MCP server adds its
own, so a name list in another package cannot know. `is_irreversible` is a
predicate rather than a flag because one `bash` call reads a file and the next
drops a table.

## Failing

Raise. The pipeline normalizes any exception into a structured `is_error` result
(B5) — a tool cannot take the loop down.

What matters is *which kind*, because consumers branch on it and Code Mode does
something different for each:

* **`failed`** — the ordinary case. Raise `HarnessError(message, code)`, or
  anything else. A program under Code Mode may catch it and continue.
* **`denied`** — policy refused. Ends the whole Code Mode run (C3), so it must
  not be catchable by model-authored code. Raise something whose
  `failure_kind = "denied"` — `FsDenied` and `SandboxError` already are.
* **`aborted`** — cancellation. Call `run.raise_if_cancelled()` in a long body.

Getting this wrong is invisible in tests that only read the message: a denial
reported as a failure lets a program route around a policy veto.

## What the body gets

`run: ToolRunContext` carries the execution's identity and its two boundaries:

* `run.agent` — the *physical* key (which workspace, where to route a prompt);
* `run.scope` — the *policy* boundary (which rules apply). **Pass both onward**;
  they are two values and nothing checks them against each other.
* `run.session`, `run.call_id`, `run.signal`
* `run.defer_context(message)` — attach a message to the conversation *after*
  this call's result, preserving call/result adjacency. This is how a tool
  contributes content the result itself cannot carry — see `attach_tool.py`,
  where the media rides a context message because both provider wires flatten a
  tool result to text.
* `run.conclude_turn()` — mark a successful result as terminal.

## Presentation

`**simple_views(card, title, key)` covers the common case — a fixed title over one
salient argument. Card kinds are `generic | terminal | diff | search | read | web`;
a front end must know how to draw each, so this is a closed set rather than a
free string.

## Checklist

- [ ] `parameters` is a model, so the body gets a validated object
- [ ] `render` is pure — no I/O, no clock, no access to the live call
- [ ] the flags above are declared, not left to default silently
- [ ] denials raise something whose `failure_kind` is `denied`
- [ ] `run.agent` and `run.scope` are both threaded to whatever you call
- [ ] the tool is registered by a row, and `inject` names the seams it uses
