"""`ctx.permission_presets` — one name for a sandbox mode *and* an approval policy.

A user thinks in postures ("read-only", "let it work in here", "I know what I'm
doing"), not in two independent knobs. A preset maps one name onto both, and
records the choice as `permission/preset` so the log says which posture a turn
ran under — which is the question anyone reviewing a session asks first.

@module ph.seams.permission_presets
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

from ..cordis import Context, plugin
from ..session import Session
from .approval import ApprovalPolicy
from .sandbox import SandboxMode

__all__ = ["PRESETS", "PermissionPreset", "PermissionPresetService", "PresetName", "apply"]

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


@dataclass(slots=True)
class PermissionPresetService:
    """The service published as `ctx.permission_presets`."""

    ctx: Context
    active: PresetName = "read-only"

    def list(self) -> list[PermissionPreset]:
        return list(PRESETS.values())

    def apply_preset(self, name: PresetName, session: Session | None = None) -> PermissionPreset:
        """Switch posture, recording it where a reviewer will look."""
        preset = PRESETS[name]
        self.active = name
        if session is not None:
            session.append("permission/preset", {"preset": name})
            sandbox = self.ctx.get("sandbox")
            if sandbox is not None:
                sandbox.set_mode(session, preset.sandbox_mode)
            approval = self.ctx.get("approval")
            if approval is not None:
                approval.set_policy(session, preset.approval_policy)
        return preset

    def resolve(self, session: Session | None = None) -> PermissionPreset:
        if session is not None:
            event = session.latest("permission/preset")
            if event is not None and event.data.get("preset") in PRESETS:
                return PRESETS[cast(PresetName, event.data["preset"])]
        return PRESETS[self.active]


@plugin("permission-presets")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the permission-preset mapping."""
    ctx.provide("permission_presets", PermissionPresetService(ctx=ctx))
