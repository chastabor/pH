"""`ctx.tui_screens` — the front end's registration seam (P4-17).

A row contributes a *screen* the way it already contributes a tool, a command
or a prompt section. Ported from dsh's slot service (`packages/client/ui-slots`),
whose ~35 `ui-*` packages register into named slots with
`ctx.slots.register({name, id, order, label, inject})` and whose own comment
names the property worth copying: *"the registration rides the slot service's
effect wrapper, so plugin unload removes the tab."* That is invariant I2 applied
to the front end, and `claim_key` already provides it.

Three things are deliberately narrower than dsh.

**One slot, not a hierarchy.** dsh separates `conversation.view`,
`conversation.composer`, `settings.section` and more. pH's TUI has one extension
point worth opening today, and a hierarchy with a single member is a hierarchy
nobody can check. A second registrant is what should motivate a second slot.

**A screen is built, not injected.** dsh's `inject` returns a props bag for a
component the shell renders; `build(session)` returns the front end's own screen
object. It is given the *session* rather than the harness so a screen stays a
projection of a log — which is what lets the same screen open over a live chat
and over a file (P3-25's property, kept).

**What `build` returns is `Any` here, on purpose.** ph-core may not import
Textual (P0-01, enforced by `test_layering.py`), so the type of a screen is the
front end's to state. The seam lives in core anyway because a *plugin* is what
registers one, and a plugin depends on ph-core alone.

Registering also has to *do* something, and that is what a presenter is for. A
front end attaches with `present_with`, and each presentation it makes unwinds
with the **registration's** scope — so a row's screen, its slash command and its
key are one lifetime, not three that have to be kept in step by hand.

@module ph.seams.tui_screens
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

from ..cordis import Context, Disposer, plugin
from ._registry import claim_key

__all__ = [
    "ID_PATTERN",
    "ScreenDefinition",
    "ScreenFactory",
    "ScreenPresenter",
    "TuiScreenRegistry",
    "apply",
]

ID_MAX = 32
ID_PATTERN = re.compile(rf"^[a-z0-9-]{{1,{ID_MAX}}}$")
"""The one format of a screen id.

Bounded here rather than shared with `ph.seams.skills.NAME_PATTERN`, which
happens to have the same shape for a different reason: an id becomes a
`/<id>` command, so what constrains it is the command grammar — a name with a
space in it would be a command whose argument is part of its name."""

ScreenFactory: TypeAlias = Callable[[Any], Any]
"""`build(session) -> screen`. The return is the front end's own object."""

ScreenPresenter: TypeAlias = Callable[["ScreenDefinition"], "Disposer | None"]
"""What a front end does with a screen — a verb, a key, a palette entry. The
disposer it returns undoes all of that."""


@dataclass(frozen=True, slots=True)
class ScreenDefinition:
    """One screen a plugin contributes to the front end."""

    id: str
    """Addressable name. Becomes `/<id>` and the binding id a keymap remaps."""
    label: str
    """What the palette and the picker call it."""
    build: ScreenFactory
    """`build(session) -> screen`, called each time the screen is opened, so
    what it shows is a fold of the log as it stands rather than as it stood."""
    order: int = 100
    """Sort order, as dsh orders slot entries. Lower is earlier."""
    key: str | None = None
    """A default key, when the screen wants one. The binding carries `id` as its
    binding id, so a user rebinds it like any other."""


@dataclass(frozen=True, slots=True)
class _Entry:
    """A registration and the scope that owns it.

    The owner is kept because a presentation made *later* — by a front end that
    attached after this row mounted — still has to unwind when this row does.
    """

    screen: ScreenDefinition
    owner: Context


@dataclass(slots=True)
class _FrontEnd:
    """One attached presenter and everything it has drawn.

    Keyed by screen id, not appended to: a row that unloads has already had its
    presentation released through its own scope, and a list would keep the spent
    release forever — one per registration for the life of the front end.
    """

    present: ScreenPresenter
    drawn: dict[str, Disposer] = field(default_factory=dict)

    def undraw(self) -> None:
        for release in reversed(list(self.drawn.values())):
            release()
        self.drawn.clear()


@dataclass(slots=True)
class TuiScreenRegistry:
    """The service published as `ctx.tui_screens`."""

    ctx: Context
    _entries: dict[str, _Entry] = field(default_factory=dict)
    _front_ends: list[_FrontEnd] = field(default_factory=list)

    def register(self, screen: ScreenDefinition, *, scope: Context | None = None) -> Disposer:
        """Contribute a screen.

        **Pass `scope=ctx` from a row's `apply`.** The default is this service's
        own context, which is the right owner for a screen the harness itself
        contributes and the wrong one for a row's: without it the row could
        unload and leave its screen, its command and its key behind.
        """
        if ID_PATTERN.match(screen.id) is None:
            raise ValueError(
                f'"{screen.id}" is not a screen id: 1..{ID_MAX} of lowercase [a-z0-9-]'
            )
        owner = scope or self.ctx
        entry = _Entry(screen=screen, owner=owner)
        release = claim_key(owner, self._entries, screen.id, entry, label="screen")
        for front_end in self._front_ends:
            self._draw(front_end, entry)
        return release

    def present_with(self, present: ScreenPresenter, *, scope: Context | None = None) -> Disposer:
        """Attach a front end: every screen registered now, and every later one.

        Returns a disposer that both detaches the presenter and undoes what it
        drew — so a front end that goes away while the rows stay does not leave
        commands pointing at it.
        """
        front_end = _FrontEnd(present=present)
        self._front_ends.append(front_end)
        for entry in self._ordered():
            self._draw(front_end, entry)

        def detach() -> None:
            if front_end in self._front_ends:
                self._front_ends.remove(front_end)
            front_end.undraw()

        return (scope or self.ctx).add_disposer(detach, label="tui-screens(front-end)")

    def list(self) -> list[ScreenDefinition]:
        """Every screen, by `order` then `id` — the order a palette shows them."""
        return [entry.screen for entry in self._ordered()]

    def get(self, screen_id: str) -> ScreenDefinition | None:
        entry = self._entries.get(screen_id)
        return None if entry is None else entry.screen

    # ------------------------------------------------------------ internals --

    def _ordered(self) -> Sequence[_Entry]:
        """Registrations in slot order.

        `Sequence`, not `list`: `list` is a method on this class, so inside the
        body it no longer names the builtin as far as an annotation is
        concerned.
        """
        return sorted(
            self._entries.values(), key=lambda entry: (entry.screen.order, entry.screen.id)
        )

    def _draw(self, front_end: _FrontEnd, entry: _Entry) -> None:
        """Present one screen to one front end, owned by the registration.

        `add_disposer` hands back an idempotent release, which is what lets the
        same teardown belong to two lifetimes at once: the row unloading runs it
        through the scope, the front end detaching runs it through `undraw`, and
        whichever happens second does nothing.
        """
        undo = front_end.present(entry.screen)
        if undo is None:
            return
        label = f"screen-view({entry.screen.id})"
        front_end.drawn[entry.screen.id] = entry.owner.add_disposer(undo, label=label)


@plugin("tui-screens")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the screen registry.

    In `ph-base` rather than in a front-end profile: the seam is the same rule
    every other seam follows, and a headless run that mounts a row registering a
    screen nothing draws needs no special case for it.
    """
    ctx.provide("tui_screens", TuiScreenRegistry(ctx=ctx))
