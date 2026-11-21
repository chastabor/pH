"""`ctx.permission_presets`: one name for a sandbox mode *and* an approval policy.

A user thinks in postures ("read-only", "let it work in here", "I know what I'm
doing"), and not in two independent settings. A preset maps one name onto both.
It records the choice as `permission/preset`, so the log says which posture a
turn ran under. That is the first question of anyone who reviews a session.

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
from ..session.writers import log_writer
from ..wire import WireModel
from ._registry import contribute_item
from .approval import ApprovalPolicy
from .sandbox import SandboxMode
from .tui_status import StatusField, StatusReading

_LOG = log_writer(__name__)

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
        summary="No confinement and no prompts: a call that would ask is refused.",
    ),
}


class PresetSchema(WireModel):
    """One posture as a front end draws it: its name, what it means, and whether
    it is the one in force.

    It has the shape of `CommandSchema`, for the same reason. A picker on the
    other side of a socket cannot reach `PRESETS`, and a client that hardcoded
    the three presets of this module would be a second statement of what a
    deployment offers. The active flag is here and not beside the list, because
    "which one" is the question that a picker asks. An answer that is split
    across two fields is two things that a reader has to line up.
    """

    name: str
    summary: str
    active: bool = False


PRESET_NAMES: Mapping[str, PresetName] = {name: name for name in PRESETS}
"""Every preset name, keyed by its own spelling.

The pick of a user arrives as a `str`, from a picker or from a wire. This
lookup is the membership test that also *narrows* the value, so the read site
needs no `cast`. It is like `_WORKSPACE_KINDS` in `ph.seams.workspace`, and
stricter. It is keyed by `PRESETS` and not by `get_args(PresetName)`. So a name
that the alias allows but no entry of `PRESETS` implements is narrowed away
here, and does not raise a `KeyError` downstream."""

_SCHEMAS: Mapping[str, tuple[PresetSchema, ...]] = {
    active: tuple(
        PresetSchema(name=one.name, summary=one.summary, active=one.name == active)
        for one in PRESETS.values()
    )
    for active in PRESETS
}
"""Every list of postures that exists: one list for each posture that could be
active.

These are nine immutable models that are built one time, for the same reason
as `_POSTURE_READINGS` below. There are three presets, so "which list does a
picker get" has exactly three answers, and none of them needs to be built again
for each call."""

_POSTURE_READINGS: dict[str, StatusReading] = {
    name: StatusReading(text=f"{name} accepted") for name in PRESETS
}
"""One reading for each preset, built one time. There are three presets, and
the field is read on every refresh of the footer."""


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
            sandbox, approval = self.ctx.get(SANDBOX), self.ctx.get(APPROVAL)
            # **One posture, one batch** (S14). A torn tail that kept the preset
            # and lost its sandbox mode was read as one preset and ran as
            # another.
            with session.batch() as batch:
                _LOG.append(batch, "permission/preset", {"preset": name})
                if sandbox is not None:
                    sandbox.set_mode(batch, preset.sandbox_mode)
                if approval is not None:
                    approval.set_policy(batch, preset.approval_policy)
        return preset

    def schemas(self, session: Session | None = None) -> tuple[PresetSchema, ...]:
        """Every posture, with the live one marked. This is what a picker draws.

        The answer is resolved and not remembered, and that is the purpose. The
        TUI used to fold `permission/preset` events to decide what to mark. A
        client that attached to a session that somebody had already switched
        then marked nothing at all. The seam knows without being told.
        """
        return _SCHEMAS[self.resolve(session).name]

    def posture_reading(self, session: Session) -> StatusReading:
        """`read-only accepted`: what runs without anybody being asked.

        The verb matters. All three presets are *named* for the sandbox mode
        that they set, so the bare word reads equally as the posture and as the
        mode. The two are different questions. This reading is about prompting,
        and the sandbox reading is about what the kernel permits.

        This is a bound method over a memo, and not a closure that builds an
        f-string, because there are three postures and this is read on every
        refresh.
        """
        return _POSTURE_READINGS[self.resolve(session).name]

    def resolve(self, session: Session | None = None) -> PermissionPreset:
        if session is not None:
            event = session.latest("permission/preset")
            # The read goes through the lookup, which is the membership test
            # *and* the narrowing. The `in PRESETS` test and the `cast` that it
            # replaced did the first, and needed the second to say so.
            name = PRESET_NAMES.get(as_str(event.data.get("preset"))) if event else None
            if name is not None:
                return PRESETS[name]
        return PRESETS[self.active]


@plugin("permission-presets", affects="environment")
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
