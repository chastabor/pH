# `ctx.permission_presets` — one name for a sandbox mode *and* an approval policy

**Module:** `ph/seams/permission_presets.py` · **Row:** `permission-presets` ·
**Consumers:** the TUI's posture picker

A user thinks in **postures** — "read-only", "let it work in here", "I know what
I'm doing" — not in two independent knobs.

## The table

| preset | sandbox mode | approval | |
|---|---|---|---|
| `read-only` | `read-only` | `ask` | Reads freely; every write or command asks first. |
| `workspace-write` | `workspace-write` | `ask` | Writes inside the workspace without asking; anything outside asks. |
| `danger-full-access` | `danger-full-access` | `never` | No confinement and no prompts: a call that would have asked is refused instead. **The name is the warning.** |

**`never` does not mean "approve everything".** Under it `ctx.approval` answers
each ask at once as `rejected`, and records the pair (`approval/asked`,
`approval/decided` with `automatic: true`) so the log says why. So in
`danger-full-access` nothing is confined and nobody is prompted, and whatever a
row still sends to an ask does not run: a `hitl` rule that matches, a
`permissions-fs` rule set to `interrupt`. The model is told the call was
rejected. A deployment that wants those calls to run takes the rule away, or
sets `hitl`'s `mode` to `yolo`, rather than relying on the preset.

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
rather than the profile's default.

`hitl` (`ph-stabilize`) does not read the preset. Its posture is its own record,
`approval/mode` (`manual`, `auto`, `yolo`), kept apart on purpose: a preset
writes `approval/policy`, and a posture riding that record would be reset by
every switch of preset. The two meet only at `ctx.approval`, where `hitl`'s ask
is answered under whatever policy the preset set.

## The surface

```text
ctx.permission_presets.schemas(session)          # -> the presets a UI offers, live one marked
ctx.permission_presets.resolve(session)          # -> the PermissionPreset in force
ctx.permission_presets.posture_reading(session)  # -> the footer's `<name> accepted`
ctx.permission_presets.apply_preset(name, session)
```

A `PermissionPreset` carries `name`, `summary`, `sandbox_mode`, `approval_policy`.
A `PresetSchema` is the half of that a front end draws — `name`, `summary` and
whether it is `active` — because a picker on the other side of a socket cannot
reach `PRESETS`, and a client that hardcoded the three would be a second
statement of what this deployment offers.

**`schemas` rather than a list plus a separately-tracked name.** The TUI folded
`permission/preset` events to decide which row to mark, so a client that
attached to a session somebody had already switched marked nothing at all. The
seam knows without being told, and `presets/list` carries the answer.

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
