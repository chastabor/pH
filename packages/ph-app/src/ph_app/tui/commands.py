"""The TUI's verbs: one table, three routes.

Every front-end action a person can name is a `TuiVerb`, and one verb is
reachable three ways — as a slash command registered into `ctx.commands` (so the
palette lists it, the prompt completes it, and `command/run` records it), as a
Textual action on the app, and, when it has a `key`, as a binding remapped from
`tui.json`. Adding a verb is adding a row here plus an `action_<name>` method;
nothing re-dispatches on the name string.

This table is the *built-in* source of verbs. A screen contributed through
`ctx.tui_screens` gets the same three routes from `screens.py`, which builds
them per registration rather than per table row — because those come and go with
the plugin that registered them, and this table does not.

The commands are registered on the *root* context, so they unwind with it (I2).

@module ph_app.tui.commands
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from textual.binding import Binding, BindingType

from ph.cordis import Context
from ph.seams.commands import CommandDefinition

from .config import TuiKeybindings

__all__ = [
    "TUI_VERBS",
    "TuiVerb",
    "action_command",
    "app_bindings",
    "local_commands",
    "register_tui_commands",
]


@dataclass(frozen=True, slots=True)
class TuiVerb:
    """One front-end action."""

    name: str
    """The slash command: `/<name>`."""
    summary: str
    action: str
    """The Textual action: `PHTuiApp.action_<action>`."""
    key: str | None = None
    """The `TuiKeybindings` field that binds it, if any. Doubles as the binding
    id, which is what `App.set_keymap` remaps."""
    argument_hint: str = ""
    """What follows the name, when the verb takes something — shown in the
    palette row, and the reason `_RunAction` forwards the typed argument."""


TUI_VERBS: tuple[TuiVerb, ...] = (
    TuiVerb("commands", "Browse every command.", "open_commands", "command_palette"),
    TuiVerb("model", "Choose the provider and model.", "open_models", "model_picker"),
    TuiVerb("theme", "Choose a colour theme.", "open_themes", "theme_picker"),
    TuiVerb("sessions", "Reopen a stored session.", "open_sessions", "session_picker"),
    TuiVerb(
        "permissions", "Change what pH may do without asking.", "open_presets", "permission_picker"
    ),
    TuiVerb("login", "Provide a provider credential for this process.", "open_login"),
    TuiVerb(
        "thinking", "Show or hide the model's reasoning.", "toggle_thinking", "toggle_thinking"
    ),
    TuiVerb("tools", "Show or hide tool results.", "toggle_tool_results", "toggle_tool_results"),
    TuiVerb("sidebar", "Show or hide the sidebar.", "toggle_sidebar", "toggle_sidebar"),
    TuiVerb("attach", "Attach files to the next prompt.", "attach", argument_hint="<path> …"),
    TuiVerb("quit", "Leave pH.", "quit", "quit"),
)


def app_bindings(keys: TuiKeybindings) -> list[BindingType]:
    """The app-level bindings, one per keyed verb.

    `priority=True` so they are checked before the focused widget: the prompt is
    a `TextArea`, which binds `ctrl+k`, `ctrl+y` and others for editing, and a
    non-priority binding would lose to it — or, worse, fire *as well as* it.
    """
    bindings: list[BindingType] = [
        Binding(getattr(keys, verb.key), verb.action, verb.summary, id=verb.key, priority=True)
        for verb in TUI_VERBS
        if verb.key is not None
    ]
    return bindings


def local_commands(app: Any) -> list[CommandDefinition]:
    """The table as definitions, each dispatching into `app`.

    Two front ends want this list: the in-process one registers it into
    `ctx.commands`, and the socket one has no registry to register into and
    offers it beside the daemon's. One constructor, so a verb behaves the same
    whichever end owns the harness."""
    return [
        action_command(app, verb.name, verb.summary, verb.action, verb.argument_hint)
        for verb in TUI_VERBS
    ]


def register_tui_commands(ctx: Context, app: Any) -> list[Callable[[], Any]]:
    """Register one slash command per verb. Returns their disposers."""
    return [ctx.commands.register(definition) for definition in local_commands(app)]


def action_command(
    app: Any, name: str, summary: str, action: str, argument_hint: str = ""
) -> CommandDefinition:
    """A slash command whose whole body is one Textual action.

    The one spelling of that, because there are two sources of verbs — this
    table and `screens.py`'s registered screens — and a command-body contract
    that had to be honoured in both places is one that will be honoured in one.
    A body is dispatched on the message pump, so an action that opens a screen
    does so with a callback and returns, the same constraint every key handler
    already lives under.
    """
    return CommandDefinition(
        name=name, summary=summary, run=_RunAction(app, action), argument_hint=argument_hint
    )


@dataclass(frozen=True, slots=True)
class _RunAction:
    """The body itself. A dataclass because it outlives the call that made it
    and is stored in a registry — the fields it needs are the fields it holds."""

    app: Any
    action: str

    async def __call__(self, argument: str, _context: Any) -> None:
        # Forwarded as a Python literal, which is what Textual's action parser
        # reads (`ast.literal_eval`), so a path with spaces round-trips. An
        # action that takes no argument is called bare, as before.
        if argument:
            await self.app.run_action(f"{self.action}({argument!r})")
        else:
            await self.app.run_action(self.action)
