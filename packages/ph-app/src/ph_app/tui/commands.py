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

from dataclasses import dataclass
from typing import Any

from textual.binding import Binding, BindingType

from ph.seams.commands import CommandDefinition

from .config import TuiKeybindings

__all__ = [
    "TUI_VERBS",
    "VIEWABLE",
    "VIEW_KEYS",
    "VIEW_USAGE",
    "TuiVerb",
    "action_command",
    "app_bindings",
    "local_commands",
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


VIEWS: dict[str, tuple[str, bool]] = {
    "tools": ("show_tools", False),
    "skills": ("show_skills", False),
    "results": ("show_tool_results", True),
    "thinking": ("show_thinking", True),
}
"""What `/view <name>` flips: the `TuiSettings` field, and whether the transcript
has to be rebuilt for it.

A table because the same four names were being written three times — here, in
the usage line, and in `action_view`'s branch chain — and three lists of one
thing drift: the usage line could offer a word the chain then rejected. The
`bool` is the one fact that differs between them: `results` and `thinking` are
what the transcript is *built* from, while the two panels are read at draw time."""

PANELS: tuple[str, ...] = ("tools", "skills")
"""The two `/view all` means, and the two `VIEWABLE` names before the rest."""

VIEWABLE: tuple[str, ...] = (
    *PANELS,
    "all",
    # Iterated rather than set-subtracted: a set has no order, so the usage line
    # would have been free to rename itself between runs.
    *(name for name in VIEWS if name not in PANELS),
    "sidebar",
)
"""**Every** word `/view` takes, which is the table plus the two that are not
settings fields.

`all` is the pair of panels rather than everything here — it is asked by
somebody looking at the sidebar — and `sidebar` is this window's own, persisting
nowhere. Both are in this tuple even so, because a word the verb accepts and
this list omits is a word the usage line does not offer and a key can still
fire: the first draft left `sidebar` out and `ctrl+b`'s binding pointed at a
`/view` spelling no list admitted."""

VIEW_USAGE = f"usage: /view {' | '.join(VIEWABLE)}"

TUI_VERBS: tuple[TuiVerb, ...] = (
    TuiVerb("commands", "Browse every command.", "open_commands", "command_palette"),
    TuiVerb("model", "Choose the provider and model.", "open_models", "model_picker"),
    TuiVerb("theme", "Choose a colour theme.", "open_themes", "theme_picker"),
    TuiVerb("sessions", "Reopen a stored session.", "open_sessions", "session_picker"),
    TuiVerb(
        "permissions", "Change what pH may do without asking.", "open_presets", "permission_picker"
    ),
    TuiVerb("login", "Provide a provider credential for this process.", "open_login"),
    TuiVerb("view", "Show or hide part of the view.", "view", argument_hint=" | ".join(VIEWABLE)),
    TuiVerb("tools", "List what the model may call.", "list_tools"),
    TuiVerb("skills", "List the skills installed here.", "list_skills"),
    TuiVerb("attach", "Attach files to the next prompt.", "attach", argument_hint="<path> …"),
    TuiVerb("quit", "Leave pH.", "quit", "quit"),
)

VIEW_KEYS: tuple[tuple[str, str, str], ...] = (
    ("toggle_tool_results", "results", "Show or hide tool results."),
    ("toggle_thinking", "thinking", "Show or hide the model's reasoning."),
    ("toggle_sidebar", "sidebar", "Show or hide the sidebar."),
)
"""Keys that fire `/view <word>`: the binding id, the word, and what it says.

Its own table rather than three rows in `TUI_VERBS`, because they are not verbs
— a row there is a slash command, and `/view results` is already reachable as an
argument to `/view`. They exist only to keep the binding **ids**: `ctrl+o` has
bound "show or hide tool results" since before there was a `/view`, `tui.json`
remaps by id, and a person who rebound `toggle_tool_results` keeps their key
across the rename. Listing them here costs one table; listing them there cost
every consumer of `TUI_VERBS` a filter, and two tests had written one."""


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
    bindings.extend(
        Binding(getattr(keys, key), f"view({word!r})", summary, id=key, priority=True)
        for key, word, summary in VIEW_KEYS
    )
    return bindings


def local_commands(app: Any) -> list[CommandDefinition]:
    """The table as definitions, each dispatching into `app`.

    Handed back rather than registered: after P5-14 there is no in-process
    registry to register into — a socket client offers these beside the
    daemon's, which is what `DaemonSession.commands` merges."""
    return [
        action_command(app, verb.name, verb.summary, verb.action, verb.argument_hint)
        for verb in TUI_VERBS
    ]


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
