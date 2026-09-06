# `ctx.tui_screens` — the front end's registration seam

**Module:** `ph/seams/tui_screens.py` · **Row:** `tui-screens` · **Consumers:**
the TUI's screen stack, the trajectory view

A row contributes a **screen** the way it already contributes a tool, a command
or a prompt section.

Ported from dsh's slot service, whose own comment names the property worth
copying: *"the registration rides the slot service's effect wrapper, so plugin
unload removes the tab."* That is invariant I2 applied to the front end, and
`claim_key` already provides it.

## Three things deliberately narrower than dsh

**One slot, not a hierarchy.** dsh separates `conversation.view`,
`conversation.composer`, `settings.section` and more. pH's TUI has one extension
point worth opening today, and *a hierarchy with a single member is a hierarchy
nobody can check*. A second registrant is what should motivate a second slot.

**A screen is built, not injected.** dsh's `inject` returns a props bag for a
component the shell renders; `build(session)` returns the front end's own screen
object — and it is given the **session** rather than the harness, so a screen
stays a projection of the log rather than a view onto live services.

**In `ph-core`, though only a TUI can use it.** A seam is a seam: a headless run
that mounts a row registering a screen nothing draws needs no special case for
it.

## The surface

```text
ctx.tui_screens.register(definition, *, scope=None)   -> Disposer
ctx.tui_screens.list()      ctx.tui_screens.get(key)
ctx.tui_screens.present_with(...)
```

A `ScreenDefinition` is `key`, `id`, `label`, `order`, `build`.

Sorted by **order then id**, so two rows that both say `order=50` still produce a
stable listing rather than one that depends on mount order.

## What a non-Textual client gets

`build()` returns a Textual object, which cannot travel — so a third-party row's
screen is currently **invisible to a browser tab or a `ph agents` client**, and
`ctx.tui_screens`' own gate says so.

The half that would fix it is a declarative body — a row contributing a panel as
*data* rather than a `build()` — which is P7-07's, together with a projection
layer of pH's own. Recorded here because a seam that looks like it serves every
front end and serves one is exactly the overstatement E1 forbids.

## What it does not do

* It does not draw. `build` returns the front end's object; ph-core never learns
  what a terminal is.
* It does not route. Which screen is on top is the app's business.
* It does not survive a restart — a screen is a registration, rebuilt every
  mount.

## See also

[`ctx.tui_status`](tui_status.md) · [`ctx.diagnostics`](diagnostics.md) ·
`test_tui_screens.py`, `dev-notes/screen-registry-design.md`
