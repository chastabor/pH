# `ctx.settings` — durable user preferences, read as data

**Module:** `ph/seams/settings.py` · **Row:** `settings-local` · **Consumers:**
the TUI (theme, keybindings), any row with a user-facing preference

## Why this is not a profile row

| | a **row** | a **setting** |
|---|---|---|
| chosen by | the deployment | the user |
| versioned with | the code | nothing — it outlives profiles |
| changed by | editing a profile | the person, at runtime |

Conflating them means one of two bad outcomes: a user's edit gets clobbered by an
upgrade, or a deployment cannot change a default. Keeping them apart is what lets
both move independently.

The rule of thumb: if a *deployment* would want to set it for everyone, it is row
config. If a *person* would want it to follow them between projects, it is a
setting.

## The surface

```text
ctx.settings.get("tui.theme", default=None)     # dotted key
await ctx.settings.set("tui.theme", "dark")     # writes and persists
ctx.settings.load()                              # the whole document
ctx.settings.path                                # where it lives
```

Dotted keys are read and written through nested dicts, so `tui.theme` and
`tui.keybindings` share a namespace without any row owning the parent.

`get` returns the default for a missing key at **any** depth — a missing
intermediate is the same answer as a missing leaf, because from a caller's point
of view "nobody has set this" is one condition.

## A corrupt file must not stop the harness starting

`load` catches a decode error and answers `{}`.

Defaults are **always a valid answer for a preference**, so a settings file
somebody hand-edited into invalid JSON costs the person their customisations and
not their session. That is the opposite trade from a *profile*, where a malformed
document is refused loudly at startup — a deployment's composition is not
something to guess at, and a preference is.

## What it does not do

* **It is not configuration.** A row that reads a setting to decide policy has
  put a security decision where a user's convenience file can reach it. Policy is
  row config; preferences are here.
* **It is not per-session.** One document per user, under `$PH_HOME`. Something
  that varies per session belongs in that session's log.
* **It does not validate.** Values are read as data; a consumer that needs a
  shape validates on read and falls back to its default. There is no schema,
  because a schema here would be a second declaration of things rows already
  declare.
* **It does not notify.** Nothing watches the file; a change is seen on the next
  read.

## See also

[`ctx.permission_presets`](permission_presets.md) — a *posture* is recorded in
the log rather than here, because it changes what a turn is allowed to do ·
`test_seams.py`
