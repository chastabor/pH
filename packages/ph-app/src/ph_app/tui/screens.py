"""What a registered screen *becomes* in this front end, and how it talks back.

`ctx.tui_screens` (P4-17) is the seam; this is pH's presenter for it. One
`ScreenDefinition` yields, with no further wiring:

* a slash command, so `/<id>` is in the palette, completes at the prompt, and
  records `command/run` like every other verb — built by `commands.py`'s
  `action_command`, the one spelling of "a command whose body is an action";
* a key, when the screen asked for one — carrying the screen's id as its
  binding id, which is what `App.set_keymap` remaps, so a plugin's key obeys
  `tui.json` exactly as a built-in's does;
* `escape` back to the chat, because it is a pushed screen rather than a second
  app.

All of it unwinds with the *registration*: the presenter hands its teardown to
the seam, which owns it on the registering row's scope (I2). Unloading the row
takes the verb and the key with it, and there is no second list to keep in step.

**Cross-navigation is two halves and one join.** `RevealSeq` is a screen asking
its host to show the transcript row for a log seq; `Revealing` is a screen
accepting the same number on the way in. `RevealHost` is the mirror a screen
checks before it offers the action at all — structurally, so any shell that can
answer gets the behaviour and a screen never counts the screen stack to guess
where it is. All three read `source_seq`/`seq`, which the log already carries.

@module ph_app.tui.screens
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from textual.binding import Binding
from textual.message import Message

from ph.cordis import Context, Disposer
from ph.seams.tui_screens import ScreenDefinition

from .commands import action_command

__all__ = ["RevealHost", "RevealSeq", "Revealing", "open_screen_action", "present_screens"]


class RevealSeq(Message):
    """A screen asking its host to show the transcript row for a log seq.

    A message rather than a dismissal value: the shell pushes screens it knows
    nothing about, and a return type it had to interpret would make every
    screen's meaning the app's business. `screen_id` travels with it so the host
    can name whoever asked without knowing any of them by name.
    """

    def __init__(self, seq: int, *, screen_id: str = "") -> None:
        super().__init__()
        self.seq = seq
        self.screen_id = screen_id


@runtime_checkable
class Revealing(Protocol):
    """A screen that can open positioned at a log seq.

    Optional, and checked structurally: a screen that has nothing to position
    simply does not have the method, and the shell opens it as it is.
    """

    def reveal(self, seq: int) -> None:
        """Select whatever this screen shows for `seq`, before it mounts."""
        ...


@runtime_checkable
class RevealHost(Protocol):
    """An app that answers `RevealSeq` — the mirror of `Revealing`.

    A screen asks its host structurally rather than counting the screen stack,
    because stack depth answers "how did I get here", which is not the question:
    `--mode trajectory` has no transcript to reveal into however it was mounted,
    and a shell that can reveal can do so from any depth.
    """

    def on_reveal_seq(self, message: RevealSeq) -> None: ...


def open_screen_action(screen_id: str) -> str:
    """The Textual action string that opens one screen.

    One spelling, used by the binding and by the slash command, so a key and a
    verb cannot come to mean different things.
    """
    return f"open_screen({screen_id!r})"


def present_screens(ctx: Context, app: Any) -> Disposer | None:
    """Attach this app to `ctx.tui_screens`. `None` when the seam is absent."""
    registry = ctx.get("tui_screens")
    if registry is None:
        return None
    detach: Disposer = registry.present_with(_Presenter(ctx=ctx, app=app))
    return detach


@dataclass(frozen=True, slots=True)
class _Presenter:
    """One screen in, a verb and a key out.

    A dataclass rather than a closure because it is *stored* — the seam holds it
    for as long as this front end is attached, and an object that outlives its
    call is clearer holding named fields than free variables.
    """

    ctx: Context
    app: Any

    def __call__(self, screen: ScreenDefinition) -> Disposer:
        action = open_screen_action(screen.id)
        command: Disposer = self.ctx.commands.register(
            action_command(self.app, screen.id, screen.label, action)
        )
        if not screen.key:
            return command
        binding = Binding(screen.key, action, screen.label, priority=True, id=screen.id, show=False)
        key = self.app.add_binding(binding)

        def undo() -> None:
            key()
            command()

        return undo
