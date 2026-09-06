# `ctx.tui_status` — a live reading in the footer, contributed by a row

**Module:** `ph/seams/tui_status.py` · **Row:** `tui-status` · **Consumers:** the
TUI footer, the daemon's `session.status` projection

The sibling of [`ctx.tui_screens`](tui_screens.md), for the other thing a row
wants from a front end: not a whole screen, but **one short reading** beside the
model name and the context gauge.

The gauge is the shape being generalised — *"the number a user needs to see
coming is the one where the harness will act"* — and a limit is exactly that
number for a different mechanism.

## A reading, not a notice

Both are worth having, and they answer different questions:

| | says | when |
|---|---|---|
| a **notice** in the transcript | why something *happened* | after the fact |
| a **reading** in the footer | how close you are to it happening | before |

A budget that only announces itself on the step it stops you is a budget you
cannot plan around.

## Semantic level, never a colour

```text
StatusReading = text | level
```

`level` says a reading is `warning`; **what that looks like is the front end's
business** — the same rule `ContextForm` and `CardKind` are held to, and the
reason ph-core can own this seam without knowing what a terminal is.

## It must stay cheap

A field is read on **every redraw**, which for a running agent is every spinner
frame. A contributor that stats a tree or spawns a subprocess would take the
footer down with it.

That is the whole reason [`ctx.diagnostics`](diagnostics.md) is a second seam
rather than a parameter on this one: a diagnostic answers *"what is this
deployment"* once, at a person's request, so it **may** be expensive.

A contributor that raises is dropped and logged rather than failing the redraw.

## The surface

```text
ctx.tui_status.register(field, *, scope=None)   -> Disposer
ctx.tui_status.readings(session)                # -> list[StatusReading]
```

`readings` takes the **session**, so a field is a projection of the log — which
is what lets the daemon compute the same footer for a browser tab that the
terminal draws, from the same fold.

Readings ride the `session.status` notification, pushed when the agent moves —
because that is when they can have changed, rather than on the TUI's 30 Hz tick.

## What it does not do

* It does not notify. Something that should interrupt is a notice or a modal.
* It does not colour, size, or place anything.
* It does not poll. A reading is computed when the front end asks.

## See also

[`ctx.tui_screens`](tui_screens.md) · [`ctx.diagnostics`](diagnostics.md) ·
[`ctx.token_meter`](token_meter.md) · `test_daemon_projections.py`
