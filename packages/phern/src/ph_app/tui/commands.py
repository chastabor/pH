"""The verbs of the TUI: one table, three routes.

Every front-end action that a user can name is a `TuiVerb`. One verb is
reachable in three ways:

- As a slash command. `local_commands` builds one `CommandDefinition` for each
  row, and `PHTuiApp.commands` lists them, so the palette lists the verb and
  the prompt completes it.
- As a Textual action on the app.
- As a binding that is remapped from `tui.json`, when the verb has a `key`.

To add a verb, add a row here and an `action_<name>` method. Nothing
re-dispatches on the name string.

This table is the *built-in* source of verbs. A screen contributed through
`ctx.tui_screens` gets the same three routes from `ph_app.tui.remote`, which
builds them for each screen that the daemon lists and not for each table row.
Those screens come and go with the plugin that registered them, and this table
does not.

The commands are not registered anywhere. `PHTuiApp` builds them one time and
holds them for as long as the app lives. `local_commands` says why.

@module ph_app.tui.commands
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import Literal, cast

from textual.binding import Binding, BindingType

from ph.cordis import maybe_await
from ph.seams.commands import CommandContext, CommandDefinition, parse_command_line

from .config import TuiKeybindings, TuiSettings
from .screens import AppSurface

__all__ = [
    "TUI_VERBS",
    "VIEWABLE",
    "VIEW_KEYS",
    "VIEW_USAGE",
    "TuiVerb",
    "ViewToggle",
    "action_command",
    "app_bindings",
    "local_commands",
    "run_line",
    "screen_bindings",
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
    """The `TuiKeybindings` field that binds it, if any. The field name is also
    the binding id, which is what `App.set_keymap` remaps."""
    argument_hint: str = ""
    """What follows the name, when the verb takes an argument. The palette row
    shows it, and it is the reason why `_RunAction` forwards the typed argument."""
    work: Literal["replace", "queue"] | None = None
    """How the body reaches the harness, or `None` if it never leaves the app (H7).

    A verb that makes a wire call must not run on the message pump of Textual.
    `verb_work` reads this field to decide two things: whether the verb
    schedules a worker at all, and whether a second press supersedes the first
    (`replace`) or runs beside it (`queue`). A picker replaces, because a second
    press means "I meant this one". A verb with an effect queues: two `/attach`
    calls are two files, and to cancel the first in order to honor the second
    drops one.

    It is declared here because this is where a verb already says what it is.
    Both facts used to be copied by hand into a test file, so a new slow verb
    answered the worker question in two places. `ToolDefinition.self_limits`
    makes the same argument about name lists in another package.
    """
    behind_dialogs: bool = False
    """Whether its key works while a dialog or another screen is on top. That is
    true for quit and for nothing else. `app_bindings` binds such a verb on the
    app, which the key chain of every screen reaches. Every other keyed verb is
    bound on the base screen of the chat (`screen_bindings`), which is in the
    chain of no other screen. A dialog therefore silences it by being on top,
    and not by a check that somebody must keep current."""


@dataclass(frozen=True, slots=True)
class ViewToggle:
    """One thing that `/view` can show or hide.

    `flip` is a function and not a field *name*, and that is the purpose. The
    first version held `("show_tools", False)`. It reached the field with
    `getattr` and passed a `dict[str, Any]` splat to `replace`. Those are three
    places that the type checker cannot follow: a renamed field passed the
    check and failed at the keystroke. Written as a call, a rename is a type
    error at import.
    """

    flip: Callable[[TuiSettings], TuiSettings]
    rebuilds: bool
    """Whether the transcript must be rebuilt for it.

    This is the one fact that differs between the toggles. The transcript is
    *built* from `results` and `thinking`, and the two panels are read at draw
    time."""


VIEWS: dict[str, ViewToggle] = {
    "tools": ViewToggle(lambda s: replace(s, show_tools=not s.show_tools), False),
    "skills": ViewToggle(lambda s: replace(s, show_skills=not s.show_skills), False),
    "results": ViewToggle(lambda s: replace(s, show_tool_results=not s.show_tool_results), True),
    "thinking": ViewToggle(lambda s: replace(s, show_thinking=not s.show_thinking), True),
}
"""What `/view <name>` flips.

This is a table because the same four names were written three times: here, in
the usage line, and in the branch chain of `action_view`. Three lists of one
thing drift apart: the usage line could offer a word that the chain then
rejected."""

PANELS: tuple[str, ...] = ("tools", "skills")
"""The two panels that `/view all` means. They are also the first two names in
`VIEWABLE`."""

VIEWABLE: tuple[str, ...] = (
    *PANELS,
    "all",
    # This iterates and does not subtract sets. A set has no order, so the
    # usage line could have changed between runs.
    *(name for name in VIEWS if name not in PANELS),
    "sidebar",
)
"""**Every** word that `/view` takes: the table, plus the two words that are not
settings fields.

`all` means the pair of panels and not everything here, because the user who
asks for it is looking at the sidebar. `sidebar` belongs to this window and
persists nowhere. Both are in this tuple even so. A word that the verb accepts
and this list omits is a word that the usage line does not offer and that a key
can still fire. The first draft left `sidebar` out, and the binding of `ctrl+b`
pointed at a `/view` spelling that no list admitted."""

VIEW_USAGE = f"usage: /view {' | '.join(VIEWABLE)}"

TUI_VERBS: tuple[TuiVerb, ...] = (
    TuiVerb("commands", "Browse every command.", "open_commands", "command_palette"),
    TuiVerb("model", "Choose the provider and model.", "open_models", "model_picker"),
    TuiVerb("theme", "Choose a color theme.", "open_themes", "theme_picker"),
    TuiVerb(
        "sessions", "Reopen a stored session.", "open_sessions", "session_picker", work="replace"
    ),
    TuiVerb("history", "Search the prompts you have sent.", "open_history", "history_search"),
    TuiVerb(
        "permissions",
        "Change what pH may do without asking.",
        "open_presets",
        "permission_picker",
        work="replace",
    ),
    TuiVerb(
        "login",
        "Provide a provider credential for this process.",
        "open_login",
        work="replace",
    ),
    TuiVerb("view", "Show or hide part of the view.", "view", argument_hint=" | ".join(VIEWABLE)),
    TuiVerb("tools", "List what the model may call.", "list_tools"),
    TuiVerb("skills", "List the skills installed here.", "list_skills"),
    TuiVerb(
        "attach",
        "Attach files to the next prompt.",
        "attach",
        argument_hint="<path> …",
        work="queue",
    ),
    TuiVerb("quit", "Leave pH.", "quit", "quit", behind_dialogs=True),
)

VIEW_KEYS: tuple[tuple[str, str, str], ...] = (
    ("toggle_tool_results", "results", "Show or hide tool results."),
    ("toggle_thinking", "thinking", "Show or hide the model's reasoning."),
    ("toggle_sidebar", "sidebar", "Show or hide the sidebar."),
)
"""Keys that fire `/view <word>`: the binding id, the word, and the description.

They have their own table and are not three rows in `TUI_VERBS`, because they
are not verbs. A row there is a slash command, and `/view results` is already
reachable as an argument to `/view`. They exist only to keep the binding
**ids**. `ctrl+o` has bound "show or hide tool results" since before there was a
`/view`, and `tui.json` remaps by id. A user who rebound `toggle_tool_results`
therefore keeps their key across the rename. To list them here costs one table.
To list them in `TUI_VERBS` cost every consumer of that table a filter, and two
tests had written one."""


def app_bindings(keys: TuiKeybindings) -> list[BindingType]:
    """The keys that work behind a dialog, for the app: the verbs marked
    `behind_dialogs`, which is only quit. `screen_bindings` returns every other
    key."""
    bound = keys.as_map()
    return [
        Binding(bound[verb.key], verb.action, verb.summary, id=verb.key, priority=True)
        for verb in TUI_VERBS
        if verb.key is not None and verb.behind_dialogs
    ]


def screen_bindings(keys: TuiKeybindings) -> list[BindingType]:
    """The keys of the chat, for its base screen: every other keyed verb, and the
    view toggles.

    **They are on the base screen and not on the app**, and that keeps them
    silent behind a dialog. Textual walks the chain of the *active* screen for
    a key, and the base screen is in the chain of no other screen. The app is in
    every chain. A key that was bound there had to be refused by name while a
    dialog was up. That gate also refused the Tab of Textual itself, and it
    passed any key whose action it had not been told about. The actions carry
    the `app.` namespace: the screen holds the key, and the app holds the verb.

    `priority=True` makes Textual check them before the focused widget. The
    prompt is a `TextArea`, which binds `ctrl+k`, `ctrl+y` and others for
    editing. A binding without priority would lose to it or, worse, fire *as
    well as* it.
    """
    # This uses `as_map()` and not `getattr(keys, …)`. The class already exposes
    # the mapping from binding id to key that `set_keymap` takes, so this reads
    # it in the way that Textual does and does not add a second way in.
    #
    # **This is not type safety.** A `dict[str, str]` subscript is as opaque to
    # the type checker as `getattr`, and a renamed field arrives here as a
    # `KeyError` and not as an `AttributeError`. Type safety would need a shared
    # `Literal` for the binding ids across `TuiKeybindings`, `TuiVerb.key` and
    # `VIEW_KEYS`. That `as_map()` exists shows that the dataclass is used as a
    # mapping anyway.
    bound = keys.as_map()
    bindings: list[BindingType] = [
        Binding(bound[verb.key], f"app.{verb.action}", verb.summary, id=verb.key, priority=True)
        for verb in TUI_VERBS
        if verb.key is not None and not verb.behind_dialogs
    ]
    bindings.extend(
        Binding(bound[key], f"app.view({word!r})", summary, id=key, priority=True)
        for key, word, summary in VIEW_KEYS
    )
    return bindings


def local_commands(app: AppSurface) -> list[CommandDefinition]:
    """The table as definitions. Each one dispatches into `app`.

    The definitions are returned and not registered. After P5-14 there is no
    in-process registry to register into. A socket client offers these beside
    the commands of the daemon, and `PHTuiApp.commands` merges the two."""
    return [
        action_command(app, verb.name, verb.summary, verb.action, verb.argument_hint)
        for verb in TUI_VERBS
    ]


async def run_line(definitions: Iterable[CommandDefinition], line: str) -> str | None:
    """Dispatch a `/name` line to the definition in `definitions` that names it.

    The list is `PHTuiApp.commands`: the verbs of the terminal, then the
    commands that its front end offers, which are one for each screen and those
    of the daemon. `parse_command_line` is the split that the registry itself
    uses, so a line tokenizes here in the same way as on the daemon. A name
    that no definition claims raises a `KeyError`, whose text is what the
    prompt shows.
    """
    name, argument = parse_command_line(line)
    for definition in definitions:
        if definition.name == name:
            # A client holds no `Context`, so it cannot build a
            # `CommandContext`. Neither body that is reachable here reads one.
            # The `run` of a local definition dispatches a Textual action. The
            # `run` of a remote definition sends the line back to the daemon,
            # which builds the real `CommandContext`.
            shown = await maybe_await(definition.run(argument, cast("CommandContext", None)))
            return shown or None
    raise KeyError(f'unknown command "/{name}"')


def action_command(
    app: AppSurface, name: str, summary: str, action: str, argument_hint: str = ""
) -> CommandDefinition:
    """A slash command whose whole body is one Textual action.

    This is the one place that builds such a command, because there are two
    sources of verbs: this table, and the screens that `ph_app.tui.remote`
    wires. A contract for a command body that had to be honored in both places
    would be honored in one. A body is dispatched on the message pump. An
    action that opens a screen therefore does so with a callback and returns,
    which is the same constraint that every key handler already has.
    """
    return CommandDefinition(
        name=name, summary=summary, run=_RunAction(app, action), argument_hint=argument_hint
    )


@dataclass(frozen=True, slots=True)
class _RunAction:
    """The body itself. It is a dataclass because it outlives the call that
    made it, inside the `CommandDefinition` that holds it. The fields that it
    needs are the fields that it holds."""

    app: AppSurface
    action: str

    async def __call__(self, argument: str, _context: object) -> None:
        # The argument is forwarded as a Python literal, which is what the
        # action parser of Textual reads (`ast.literal_eval`), so a path with
        # spaces survives the round trip. An action that takes no argument is
        # called bare, as before.
        if argument:
            await self.app.run_action(f"{self.action}({argument!r})")
        else:
            await self.app.run_action(self.action)
