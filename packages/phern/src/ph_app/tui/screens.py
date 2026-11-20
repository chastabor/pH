"""What a registered screen *becomes* in this front end, and how it talks back.

`ctx.tui_screens` (P4-17) is the seam, and this front end presents it. One
`ScreenDefinition` yields these, with no further wiring:

* A slash command, so `/<id>` is in the palette and completes at the prompt.
  `action_command` in `commands.py` builds it. That function is the one place
  that builds "a command whose body is an action".
* A key, when the screen asked for one. The key carries the id of the screen as
  its binding id, which is what `App.set_keymap` remaps. The key of a plugin
  therefore obeys `tui.json` exactly as a built-in key does.
* `escape` back to the chat, because it is a pushed screen and not a second
  app.

All of it unwinds with the *registration*. The seam owns the registration on
the scope of the registering row (I2). When the row unloads, the daemon
publishes the new list as `session.screens`, and `ph_app.tui.remote` releases
the old key bindings and builds the verbs and the keys again. There is no
second list to keep in step.

**The front end builds the verb and the key, and this module has no
presenter.** There used to be one. `present_screens` attached an in-process app
to `ctx.tui_screens` and turned each registration into a command and a binding.
The harness moved into the daemon (P5-14), so there is no in-process front end
left to attach: a client is *told* over the wire which screens exist. The
presenter, its `_Presenter` and `register_tui_commands` went with the move.
`ph_app.tui.remote` builds the two routes from the list on the wire.

**Cross-navigation is two halves and one join.** `RevealSeq` is a screen that
asks its host to show the transcript row for a log seq. `Revealing` is a screen
that accepts the same number on the way in. `RevealHost` is the mirror: a screen
checks for it before it offers the action at all. The check is structural, so
any shell that can answer gets the behavior, and a screen never counts the
screen stack to guess where it is. All three read `source_seq`/`seq`, which the
log already carries.

@module ph_app.tui.screens
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any, Protocol, runtime_checkable

from textual.binding import Binding
from textual.message import Message

__all__ = ["AppSurface", "RevealHost", "RevealSeq", "Revealing", "open_screen_action"]


class RevealSeq(Message):
    """A screen asks its host to show the transcript row for a log seq.

    This is a message and not a dismissal value. The shell pushes screens that
    it knows nothing about. A return type that it had to interpret would make
    the meaning of every screen the business of the app. `screen_id` travels
    with the message so that the host can name the screen that asked, without
    knowing any screen by name.
    """

    def __init__(self, seq: int, *, screen_id: str = "") -> None:
        super().__init__()
        self.seq = seq
        self.screen_id = screen_id


class AppSurface(Protocol):
    """The three things that a front end asks of the app that it is attached to.

    **A Protocol, and not `PHTuiApp`.** To name the class would close an import
    cycle, because `app` imports `remote.attach_session`. That alone does not
    rule the class out: `from __future__ import annotations` plus
    `TYPE_CHECKING` removes exactly that problem, and `daemon/projections.py`
    does so for `Root`. What rules the class out is that it would be *wrong*.
    These three are the *surface* of the app, and not the session machinery of
    pH. To demand the concrete class would make the headless doubles illegal
    for no gain. A front end driven with no terminal is a shape that this layer
    supports on purpose.

    **It is here and not in `frontend.py`**, where it started, for the reason
    that the docstring of that file gives. The HTML renderer of P7-07 consumes
    `FrontSession`, so a Textual `Binding` in the imports of `frontend.py` is
    exactly the coupling that the file is written to avoid. This module is
    already the Textual-shaped half of the front end. It already holds
    `RevealHost`, the sibling protocol for the same object.

    It replaced `app: Any`, which had spread to `attach_surfaces`,
    `_wire_screens`, `local_commands`, `action_command` and `_RunAction.app`.
    Those were six annotations that said nothing, and a double satisfied all of
    them by implementing one member and never reaching the other two.
    """

    def run_worker(self, work: Coroutine[Any, Any, Any]) -> object:
        """Own a coroutine that a sync caller started, so that it dies with the app.

        The parameters of Textual are not restated. A structural match needs
        only what is called. `exclusive=False`, the one parameter that this
        used to carry, is the default that it was passing anyway, in three
        places that had to agree.

        The return type is `object`, and not the `Worker` of Textual. No caller
        reads what comes back, here or in `run_action` below, so to name the
        class would import it for a value that nobody holds. A return type that
        this surface does not promise is also one that an implementation is
        free to widen."""
        ...

    def add_binding(self, binding: Binding) -> Callable[[], None]:
        """Bind a key on the live app. The returned function removes it again."""
        ...

    async def run_action(self, action: str) -> object:
        """Dispatch a Textual action. This is the body of every local slash command."""
        ...


@runtime_checkable
class Revealing(Protocol):
    """A screen that can open positioned at a log seq.

    This is optional, and checked structurally. A screen that has nothing to
    position does not have the method, and the shell opens it as it is.
    """

    def reveal(self, seq: int) -> None:
        """Select whatever this screen shows for `seq`, before it mounts."""
        ...


@runtime_checkable
class RevealHost(Protocol):
    """An app that answers `RevealSeq`. It is the mirror of `Revealing`.

    A screen asks its host structurally and does not count the screen stack.
    Stack depth answers "how did I get here", which is not the question.
    `--mode trajectory` has no transcript to reveal into, however it was
    mounted, and a shell that can reveal can do so from any depth.
    """

    def on_reveal_seq(self, message: RevealSeq) -> None: ...


def open_screen_action(screen_id: str) -> str:
    """The Textual action string that opens one screen.

    The binding and the slash command both use this one string, so a key and a
    verb cannot come to mean different things. It carries the `app.` namespace
    because the key is bound on the base screen of the chat
    (`commands.screen_bindings` says why) and the action is on the app.
    """
    return f"app.open_screen({screen_id!r})"
