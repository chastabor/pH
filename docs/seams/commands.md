# `ctx.commands` — human slash commands that spend no model turn

**Module:** `ph/seams/commands.py` · **Row:** `commands` · **Consumers:** the TUI
composer, the daemon protocol, `/compact`, `/revert`, `/workspaces`, `/autonomous`

## A command is not a tool

`/compact`, `/revert`, `/refine` are things the **human** asks the harness to do.
Routing them through a model turn would be both slower and **dishonest**: the log
would show the model deciding something the user decided.

So a command dispatches directly, records `command/run` and `command/done`, and
**never opens a `turn/*`**.

| | tool | command |
|---|---|---|
| who decides | the model | the person |
| costs | a model turn | nothing |
| in the log | `tool/call`, `tool/result` inside a turn | `command/run`, `command/done`, no turn |
| declared to | the provider, as a schema | the front end, as a listing |

If the model should be able to invoke it, it is a tool. If a person types it, it
is a command. A capability that genuinely needs both gets one of each, sharing an
implementation.

## The surface

```text
ctx.commands.register(definition, *, scope=None)   -> Disposer
ctx.commands.list(...)                              # what a composer offers
ctx.commands.get(name)
await ctx.commands.dispatch(name, argument, ...)
```

A `CommandDefinition` is four fields:

| field | |
|---|---|
| `name` | what the person types, without the slash |
| `summary` | one line, shown in the composer's listing |
| `argument_hint` | the author's one line for the common case |
| `run` | the body — takes the raw argument string |

The argument arrives as **text**, not a parsed structure. A command is a thing a
person types, and inventing a schema for it would make the composer's autocomplete
and the command's parser two places that have to agree about the same string.

## Registering one

```python
@plugin("my-command", inject=["commands"])
async def apply(ctx: Context, config: Config) -> None:
    ctx.commands.register(
        CommandDefinition(
            name="mycmd",
            summary="Do the thing this deployment needs.",
            argument_hint="[path]",
            run=partial(_run, ctx),
        )
    )
```

Registered per key, so two rows claiming `/compact` is a conflict named at
registration rather than a silent last-wins.

`declarable` is what turns a definition into the schema a **non-Textual** client
receives — the daemon sends the command listing over the wire so a browser tab
and the terminal offer the same set (P7-11). A command that carried a Textual
widget in its definition could not travel; that is why `run` takes and returns
plain data.

## What it does not do

* It does not ask the model anything. If the body needs the model, it starts a
  turn explicitly — `/compact` does, and the log shows that turn as what it is.
* It does not gate itself. A command that should ask first calls
  [`ctx.approval`](approval.md); `/revert` does exactly that.
* It does not parse. One string in, and the body owns the meaning.

## See also

[`ctx.user_questions`](user_questions.md) · [`ctx.tools`](tools.md) ·
`test_commands_workspaces.py`, `test_seams.py`
