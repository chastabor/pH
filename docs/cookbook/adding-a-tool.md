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
async def apply(ctx: Context, config: None) -> None:
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
            effect_free=True,
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
| `effect_free` | the call changes nothing at all | `/revert` (N3), the checkpoint policy (S4) |
| `effects_confined_to_workspace` | every effect is a file inside the tree — a flag, or a per-call predicate over the arguments and the agent | `/revert` (N3), the checkpoint policy (S4) |
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
* **`aborted`** — cancellation. Call `run.raise_if_canceled()` in a long body.

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

## Publishing the tool to other harnesses

A tool registered by a row can be offered through
[UTCP](https://utcp.io) without a second implementation. `ph.toolcli` builds a
command line from the package's rows:

```python
from ph.toolcli import ToolCli, main

CLI = ToolCli(
    program="ph-notes",
    manual="ph_notes",
    summary="Search the notes.",
    version="0.10.0",
    rows=({"id": "notes", "name": "notes-engine"},),
)


def run() -> None:
    main(CLI)
```

Point a `[project.scripts]` entry at `run` and the package has:

* `ph-notes call <tool> --field value …` — mounts the rows in its own process
  and runs the tool **through the registry**, so the body, the validation and
  the `render` are the ones the row declared. It prints one JSON object,
  `{ok, value, text}` or `{ok, error: {code, message}}`, and nothing on stderr.
* `ph-notes manual` — the UTCP manual, generated from the tools' schemas.
  `ph-notes manual --registration` prints what another harness registers.
* `ph-notes tools`, and `status`/`install` when the package has a provisioning
  command.

Three things the UTCP command-line protocol decides, which `ph.toolcli` handles
and a tool's author should know:

* **Arguments arrive one at a time, as text.** Top-level fields become flags, and
  each is decoded by the tool's own schema. Keep the fields a model fills in at
  the top level; a deeply nested argument object travels as one JSON string.
* **A failure has to be in the answer.** The client ignores the exit code and
  merges stderr into the result. Raise a `HarnessError` with a code as you
  would in a session; it arrives as `error.code`.
* **The environment is cut down.** A variable the tool reads has to be named in
  `ToolCli.inherit_env`, or the manual's commands will not see it.

To reach the same tool from a pH session, the session's row registers the manual
with [`ph-utcp`](../../packages/ph-utcp/README.md) instead of registering the
tool: `await ctx.require(UTCP).add(Manual(template=manual_call_template(CLI),
envelope=True, names="bare", …))`. The tool then runs in the command's process,
in the calling agent's workspace and under its sandbox. `ph-code-graph` is the
worked example: `ph_code_graph/row.py` is the session's row,
`ph_code_graph/engine.py` is the row the command mounts, and `_config.py` holds
what both need — the row id and where the engine writes. Keep the session's row
out of the package's `__init__`: `python -m <package>` imports it before every
call. Set `module=` on the `ToolCli` so the manual runs the interpreter that
printed it, and hand the row's config across with `ph.toolcli.config_env`.

What a manual cannot say — that two calls may overlap, that a call has no
effect, how a front end draws the card — goes in the `Manual`'s `tools` table.
Every default there is the careful one.

## Checklist

- [ ] `parameters` is a model, so the body gets a validated object
- [ ] `render` is pure — no I/O, no clock, no access to the live call
- [ ] the flags above are declared, not left to default silently
- [ ] denials raise something whose `failure_kind` is `denied`
- [ ] `run.agent` and `run.scope` are both threaded to whatever you call
- [ ] the tool is registered by a row, and `inject` names the seams it uses
