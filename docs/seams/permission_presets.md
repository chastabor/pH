# `ctx.permission_presets` — one name for a sandbox mode *and* an approval policy

**Module:** `ph/seams/permission_presets.py` · **Row:** `permission-presets` ·
**Consumers:** the TUI's posture picker, `hitl` (`ph-stabilize`)

A user thinks in **postures** — "read-only", "let it work in here", "I know what
I'm doing" — not in two independent knobs.

## The table

| preset | sandbox mode | approval | |
|---|---|---|---|
| `read-only` | `read-only` | `ask` | Reads freely; every write or command asks first. |
| `workspace-write` | `workspace-write` | `ask` | Writes inside the workspace without asking; anything outside asks. |
| `danger-full-access` | `danger-full-access` | `never` | No confinement and no prompts. **The name is the warning.** |

Two knobs, one name. Setting them independently is still possible and is what a
deployment does in a profile; a *person* switching posture mid-session picks a
preset, because the combinations that make sense are few and the ones that do not
are confusing — "confined but never asks" is a posture nobody wants and every
two-knob UI offers.

## The choice is recorded

`apply_preset` appends `permission/preset`, so **the log says which posture a turn
ran under** — which is the question anyone reviewing a session asks first.

That also makes it survive a resume: the posture is read from the log rather than
held in memory, so a session reopened tomorrow is in the posture the person chose
rather than the profile's default. `hitl` reads it for the same reason, and an
earlier bug is why it is stated here — a preset that changed only the in-memory
value silently reverted `interrupt_on` on the next mount.

## The surface

```text
ctx.permission_presets.list()                    # -> the presets a UI offers
ctx.permission_presets.resolve(name)             # -> PermissionPreset | None
ctx.permission_presets.active(session)           # what is in force now
ctx.permission_presets.apply_preset(session, name)
```

A `PermissionPreset` carries `name`, `summary`, `sandbox_mode`, `approval_policy`.

`PresetName` is a closed `Literal` — the same rule `ApprovalOutcome`, `CardKind`
and `WorkspaceKind` are held to, so a fourth posture fails to type-check at every
site that has to handle it rather than appearing as an unhandled string.

## What a preset does *not* promise

**It names a mode; it does not create a backend.** `read-only` sets the sandbox
*mode*, and whether that mode is enforced depends on whether a provider is
mounted and what its `enforcement` is — see [`ctx.sandbox`](sandbox.md) and
[`ctx.containment`](containment.md).

A deployment with no sandbox backend that selects `read-only` gets the approval
half in force and the confinement half not. That is stated rather than hidden,
because a posture that *looks* confining and is not is the single failure E1
exists to prevent.

## See also

[`ctx.approval`](approval.md) · [`ctx.sandbox`](sandbox.md) ·
[`ctx.containment`](containment.md) · `test_seams.py`, `test_hitl.py`
(ph-stabilize)
