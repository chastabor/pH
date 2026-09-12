"""`ctx.permission_presets` — one name for a sandbox mode *and* an approval policy.

A user thinks in postures ("read-only", "let it work in here", "I know what I'm
doing"), not in two independent knobs. A preset maps one name onto both, and
records the choice as `permission/preset` so the log says which posture a turn
ran under — which is the question anyone reviewing a session asks first.

@module ph.seams.permission_presets
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..cordis import Context, plugin
from ..json import as_str
from ..keys import APPROVAL, PERMISSION_PRESETS, SANDBOX, TUI_STATUS
from ..session import Session
from ..wire import WireModel
from ._registry import contribute_item
from .approval import ApprovalPolicy
from .sandbox import SandboxMode
from .tui_status import StatusField, StatusReading

__all__ = [
    "PRESETS",
    "PRESET_NAMES",
    "PermissionPreset",
    "PermissionPresetService",
    "PresetName",
    "PresetSchema",
    "apply",
]

PresetName: TypeAlias = Literal["read-only", "workspace-write", "danger-full-access"]


@dataclass(frozen=True, slots=True)
class PermissionPreset:
    """One posture: what may be written, and whether pH asks."""

    name: PresetName
    sandbox_mode: SandboxMode
    approval_policy: ApprovalPolicy
    summary: str


PRESETS: dict[PresetName, PermissionPreset] = {
    "read-only": PermissionPreset(
        name="read-only",
        sandbox_mode="read-only",
        approval_policy="ask",
        summary="Reads freely; every write or command asks first.",
    ),
    "workspace-write": PermissionPreset(
        name="workspace-write",
        sandbox_mode="workspace-write",
        approval_policy="ask",
        summary="Writes inside the workspace without asking; anything outside asks.",
    ),
    "danger-full-access": PermissionPreset(
        name="danger-full-access",
        sandbox_mode="danger-full-access",
        approval_policy="never",
        summary="No confinement and no prompts. The name is the warning.",
    ),
}


class PresetSchema(WireModel):
    """One posture as a front end draws it: its name, what it means, and whether
    it is the one in force.

    `CommandSchema`'s shape and its reason — a picker on the other side of a
    socket cannot reach `PRESETS`, and a client that hardcoded the three from
    this module would be a second statement of what a deployment offers. The
    active flag is here rather than beside the list because "which one" is the
    question a picker is asking, and an answer split across two fields is two
    things a reader has to line up.
    """

    name: str
    summary: str
    active: bool = False


PRESET_NAMES: Mapping[str, PresetName] = {name: name for name in PRESETS}

_SCHEMAS: Mapping[str, tuple[PresetSchema, ...]] = {
    active: tuple(
        PresetSchema(name=one.name, summary=one.summary, active=one.name == active)
        for one in PRESETS.values()
    )
    for active in PRESETS
}
"""Every posture list there is — one per posture that could be active.

Nine immutable models built once, for `_POSTURE_READINGS`' reason two lines
down: there are three presets, so "which list does a picker get" has exactly
three answers and none of them needs rebuilding per call."""

_POSTURE_READINGS: dict[str, StatusReading] = {
    name: StatusReading(text=f"{name} accepted") for name in PRESETS
}
"""One reading per preset, built once — there are three, and the field is read
on every footer refresh."""
"""Every preset name by its own spelling.

A person's pick arrives as a `str` — off a picker, off a wire — and this is the
membership test that also *narrows* it, so the read site needs no `cast`. Like
`ph.seams.workspace`'s `_WORKSPACE_KINDS`, and stricter: keyed off `PRESETS`
rather than `get_args(PresetName)`, so a name the alias allows but no row
implements is narrowed away here instead of raising a `KeyError` downstream."""


@dataclass(slots=True)
class PermissionPresetService:
    """The service published as `ctx.permission_presets`."""

    ctx: Context
    active: PresetName = "read-only"

    def apply_preset(self, name: PresetName, session: Session | None = None) -> PermissionPreset:
        """Switch posture, recording it where a reviewer will look."""
        preset = PRESETS[name]
        self.active = name
        if session is not None:
            session.append("permission/preset", {"preset": name})
            sandbox = self.ctx.get(SANDBOX)
            if sandbox is not None:
                sandbox.set_mode(session, preset.sandbox_mode)
            approval = self.ctx.get(APPROVAL)
            if approval is not None:
                approval.set_policy(session, preset.approval_policy)
        return preset

    def schemas(self, session: Session | None = None) -> tuple[PresetSchema, ...]:
        """Every posture, with the live one marked — what a picker draws.

        Resolved rather than remembered, which is the whole point: the TUI folded
        `permission/preset` events to decide what to mark, so a client attaching
        to a session somebody had already switched marked nothing at all. The
        seam knows without being told.
        """
        return _SCHEMAS[self.resolve(session).name]

    def posture_reading(self, session: Session) -> StatusReading:
        """`read-only accepted` — what runs without anybody being asked.

        The verb is load-bearing. All three presets are *named* for the sandbox
        mode they set, so the bare word reads equally as the posture and as the
        mode, and the two are different questions: this one is about prompting,
        the sandbox reading is about what the kernel permits.

        A bound method over a memo rather than a closure building an f-string,
        because there are three postures and this is read on every refresh.
        """
        return _POSTURE_READINGS[self.resolve(session).name]

    def resolve(self, session: Session | None = None) -> PermissionPreset:
        if session is not None:
            event = session.latest("permission/preset")
            # Through the lookup, which is the membership test *and* the
            # narrowing — the `in PRESETS` / `cast` pair it replaced did the
            # first and needed the second to say so.
            name = PRESET_NAMES.get(as_str(event.data.get("preset"))) if event else None
            if name is not None:
                return PRESETS[name]
        return PRESETS[self.active]


@plugin("permission-presets")
async def apply(ctx: Context, config: None) -> None:
    """Mount the permission-preset mapping, and the posture it can state."""
    service = PermissionPresetService(ctx=ctx)
    ctx.provide(PERMISSION_PRESETS, service)
    contribute_item(
        ctx,
        TUI_STATUS,
        StatusField(id="posture", read=service.posture_reading, order=10),
        label="permission-presets(status)",
    )
