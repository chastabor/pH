"""`ctx.tui_screens`: the registration seam of the front end (P4-17).

A row contributes a *screen* in the same way that it already contributes a tool,
a command or a prompt section. The seam is ported from the slot service of dsh
(`packages/client/ui-slots`). The ~35 `ui-*` packages of dsh register into named
slots with `ctx.slots.register({name, id, order, label, inject})`. A comment in
that service names the property that is worth copying: *"the registration rides
the slot service's effect wrapper, so plugin unload removes the tab."* That is
invariant I2 applied to the front end, and `claim_key` already provides it.

Three things are deliberately narrower than in dsh.

**One slot, not a hierarchy.** dsh separates `conversation.view`,
`conversation.composer`, `settings.section` and more. The TUI of pH has one
extension point that is worth opening today, and a hierarchy with a single
member is a hierarchy that nobody can check. A second kind of extension point
is what should motivate a second slot.

**A screen is built, not injected.** The `inject` of dsh returns a props bag for
a component that the shell renders. `build(session)` returns the screen object
of the front end itself. It is given the *session* and not the harness, so a
screen stays a projection of a log. That lets the same screen open over a live
chat and over a file (the property of P3-25, kept).

**What `build` returns is `Any` here, on purpose.** ph-core may not import
Textual (P0-01, enforced by `test_layering.py`), so the front end states the
type of a screen. The seam is in core anyway, because a *plugin* registers a
screen, and a plugin depends on ph-core alone.

A registration *does* something through the event that it emits. The daemon
listens to `screens/change` and publishes the list to every attached client,
and the TUI builds the slash command and the key of each screen from that list
(`ph_app.tui.remote`). A row that unloads releases its registration, the event
fires again, and the command and the key go with it. So the screen of a row,
its command and its key are one lifetime, and not three that somebody keeps in
step by hand.

The seam once also took an in-process *presenter* (`present_with`), from the
time when the TUI ran in the same process as the harness. The harness moved
into the daemon (P5-14), the presenter on the front-end side went with the
move, and this half was removed later because nothing attached to it.

@module ph.seams.tui_screens
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any, TypeAlias

from ..cordis import Context, Disposer, events, plugin
from ..keys import TUI_SCREENS
from ..wire import WireModel, declarable
from ._names import require_slug
from ._registry import claim_key

events.declare(
    "screens/change",
    "emit",
    owner="ph.seams.tui_screens",
    doc="The screen set changed; consumers re-read list().",
)

__all__ = [
    "ScreenDefinition",
    "ScreenFactory",
    "ScreenSchema",
    "TuiScreenRegistry",
    "apply",
]

ID_MAX = 32
"""How long a screen id may be. The *format* belongs to `_names`. It is shared
with the name of a skill because an id is the same kind of token: something that
a person types, here as `/<id>`. The bound is not shared. It is half the bound
of a skill name, because an id must fit a key hint and a palette row beside its
label, and a skill name stands alone in a catalog.

Only the bound is here, deliberately. A compiled pattern beside it would be a
second statement of the format that nothing consults, so nothing could hold it
equal to the first."""

ScreenFactory: TypeAlias = Callable[[Any], Any]
"""`build(session) -> screen`. The return value is an object of the front end."""


class ScreenSchema(WireModel):
    """What a palette needs to list a screen: everything except how to draw it.

    `build` is a callable, and it stays where the session that it draws from is.
    The declarative body that a browser could render is the work of P7-07. Until
    then this is what travels. It is derived from the definition and not written
    out at the daemon edge, so a field that is added there arrives here (P7-11).
    """

    id: str
    label: str
    order: int = 100
    key: str | None = None


@dataclass(frozen=True, slots=True)
class ScreenDefinition:
    """One screen a plugin contributes to the front end."""

    id: str
    """The addressable name. It becomes `/<id>` and the binding id that a keymap remaps."""
    label: str
    """What the palette and the picker call it."""
    build: ScreenFactory
    """`build(session) -> screen`. It is called each time the screen is opened, so
    the screen shows a fold of the log as it is now, and not as it was."""
    order: int = 100
    """Sort order, as dsh orders slot entries. Lower is earlier."""
    key: str | None = None
    """A default key, when the screen wants one. The binding carries `id` as its
    binding id, so a user rebinds it like any other binding."""

    def schema(self) -> ScreenSchema:
        """The wire-facing half. `build` is the one thing that cannot travel."""
        return ScreenSchema.model_validate(declarable(self))


@dataclass(slots=True)
class TuiScreenRegistry:
    """The service published as `ctx.tui_screens`."""

    ctx: Context
    _entries: dict[str, ScreenDefinition] = field(default_factory=dict)

    def register(self, screen: ScreenDefinition, *, scope: Context | None = None) -> Disposer:
        """Contribute a screen.

        The ordinary case needs no `scope=` (P6-12, P6-25). A registration made
        from the `apply` of a row, or from a listener that the row wrote,
        already unwinds with the row. Pass `scope=` to register on the lifetime
        of *someone else*. That is its only meaning now.

        The registration emits `screens/change` when it is made and again when
        it is released, after the entry has left the table (`claim_key`).
        """
        require_slug(screen.id, maximum=ID_MAX, kind="screen id")
        changed = partial(self.ctx.emit, "screens/change")
        release = claim_key(
            self.ctx.owner_for(scope),
            self._entries,
            screen.id,
            screen,
            label="screen",
            then=changed,
        )
        changed()
        return release

    def list(self) -> list[ScreenDefinition]:
        """Every screen, by `order` then `id`. A palette shows them in this order."""
        return sorted(self._entries.values(), key=lambda screen: (screen.order, screen.id))


@plugin("tui-screens", affects="presentation")
async def apply(ctx: Context, config: None) -> None:
    """Mount the screen registry.

    The row is in `ph-base` and not in a front-end profile. The seam follows the
    same rule as every other seam. A headless run that mounts a row that
    registers a screen that nothing draws needs no special case for it.
    """
    ctx.provide(TUI_SCREENS, TuiScreenRegistry(ctx=ctx))
