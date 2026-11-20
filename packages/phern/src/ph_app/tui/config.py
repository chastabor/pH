"""`$PH_HOME/tui.json`: keybindings and front-end preferences.

There are two rules, and both are borrowed deliberately.

**Never hard-code a key check** (the rule of prime-agent, adopted). Every
binding is a named field with a default, and the field names are also the
Textual binding ids. One `App.set_keymap(keybindings.as_map())` therefore
rebinds the whole app, with its screens and modals. A widget that compared
`event.key == "escape"` directly would silently ignore the setting of the user.

**A broken settings file must not stop the TUI from starting.** A preference
file is not a source of truth for anything that the harness needs. If it fails
to parse, the TUI starts on defaults and says so. A refusal to start because of
a typo in a sidebar position would be a worse failure than the typo.

**The theme is not here** (P9-02). It is in
`$PH_HOME/themes/theme-profile.yaml`, which `/theme` writes and nothing else
does. It used to be a field on `TuiSettings`, and `save_tui_settings` is the
reason why it could not stay. That function persists the *whole* document, so a
`/view thinking` toggle would rewrite a theme that it had no opinion about, and
two files would claim one fact. A `theme` key that is left over from before
this change is ignored on read by the rule below, and it is gone on the next
write. That is the whole upgrade.

@module ph_app.tui.config
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

from ph.documents import read_document
from ph.json import as_bool, as_seq, as_str
from ph.paths import write_atomic

__all__ = [
    "SidebarPosition",
    "TuiKeybindings",
    "TuiSettings",
    "TurnNotification",
    "load_tui_settings",
    "save_tui_settings",
    "tui_settings_from_json",
    "tui_settings_path",
]

log = logging.getLogger("ph_app.tui.config")

TurnNotification: TypeAlias = Literal["off", "bell"]
SidebarPosition: TypeAlias = Literal["left", "right", "off"]


@dataclass(frozen=True, slots=True)
class TuiKeybindings:
    """The configurable keys. Each field name is also the Textual binding id that
    it maps."""

    cancel: str = "escape"
    submit: str = "enter"
    queue_follow_up: str = "alt+enter"
    command_palette: str = "ctrl+k"
    session_picker: str = "ctrl+r"
    model_picker: str = "ctrl+p"
    theme_picker: str = "ctrl+y"
    permission_picker: str = "ctrl+g"
    accept_completion: str = "tab"
    completion_next: str = "down"
    completion_previous: str = "up"
    history_previous: str = "up"
    history_next: str = "down"
    """The keys that walk the prompt history. They are the same two keys that the
    completion list uses, and that is deliberate.

    The two never contend, and `PromptInput._decide` is where that is arranged.
    It is stated there and not repeated here, because a claim about the branch
    order of another module goes stale silently. The fields are split even so,
    because they are two *bindings*. Somebody who moves history onto `alt+up`,
    to free the arrows for editing, must be able to say that, and a single field
    would move both."""
    history_search: str = "alt+r"
    """The key that searches the prompt history. `ctrl+r` is the session picker,
    which is older. The shell reflex has to give way to the binding that was
    already there."""
    toggle_thinking: str = "ctrl+t"
    toggle_tool_results: str = "ctrl+o"
    toggle_sidebar: str = "ctrl+b"
    quit: str = "ctrl+d"
    trajectory_quit: str = "q"
    """The key that leaves the trajectory viewer (H7).

    It is a field because it is a binding, and the rule of this class is that
    every binding id is a field. It is bound under an id of its own, so that the
    keymap of `quit` could not move it. Without a field it would have been the
    single key in the app that nobody could remap, which is the same rule broken
    in the other direction."""
    extra: Mapping[str, str] = field(default_factory=dict)
    """Keys for binding ids that this build has no field for: the key of a plugin
    screen, under its screen id (P4-17).

    They are kept and not dropped, on purpose. A screen that a row contributes
    is remapped in `tui.json` exactly like a built-in screen, because
    `set_keymap` rebinds by binding id and does not care where the id came from.
    An id that nothing binds costs nothing."""

    def as_map(self) -> dict[str, str]:
        """Binding id → key: the shape `App.set_keymap` takes."""
        named = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        return {**named, **self.extra}


@dataclass(frozen=True, slots=True)
class TuiSettings:
    """Everything that the front end remembers between runs."""

    keybindings: TuiKeybindings = field(default_factory=TuiKeybindings)
    sidebar: SidebarPosition = "right"
    turn_notification: TurnNotification = "bell"
    show_thinking: bool = True
    show_tool_results: bool = True
    show_tools: bool = True
    daemon_keep_alive: str = "0"
    """How long a daemon that this UI starts stays after the terminal closes.

    It is a duration in the spelling of `--keep-alive` (`"30s"`, `"5m"`). The
    default is `"0"`, which means leave at once. The setting is here and not on
    the daemon because it is a *preference*. The person who switches between two
    terminals is the one who knows whether it is worth avoiding a re-mount
    between them. It reaches the daemon through the argv of `spawn_command`,
    which is the only route that a client-side preference has
    (`cli._spawned_keep_alive`)."""
    show_skills: bool = True
    """Whether the tools panel and the skills panel of the sidebar are drawn.

    They are visible by default, because a person should not have to go looking
    for what fills the context window. Hiding a panel hides its header too,
    because a 32-column panel has no room for a heading over nothing."""
    hidden_screens: tuple[str, ...] = ()
    """Screen ids that this terminal does not offer: no verb, no palette entry and
    no key.

    The setting is here, and a screen is not hidden by removing a row from a
    profile, because a screen is presentation (decision 23). Every named profile
    mounts the screens that pH ships, and a session profile that removed one
    would be refused. Screens are hidden for each terminal, so a second front
    end on the same daemon still offers what this one hides."""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["keybindings"] = self.keybindings.as_map()
        return data


def tui_settings_path(home: Path) -> Path:
    return home / "tui.json"


def _coerce(value: object, allowed: tuple[str, ...], fallback: str) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def tui_settings_from_json(data: object) -> TuiSettings:
    """Build settings from parsed JSON, and ignore anything unrecognized.

    This is tolerant on purpose. An older pH wrote fewer keys and a newer pH
    writes more, and neither should make the other refuse to start.
    """
    if not isinstance(data, dict):
        return TuiSettings()
    defaults = TuiKeybindings()
    raw_keys = data.get("keybindings")
    keys = defaults
    if isinstance(raw_keys, dict):
        named = {f.name for f in fields(defaults)} - {"extra"}
        usable = {
            name: value
            for name, value in raw_keys.items()
            if isinstance(name, str) and isinstance(value, str) and value
        }
        # What is left over is a binding id that this build has no field for:
        # the id of a contributed screen. If it were dropped, exactly one class
        # of key could not be rebound, and this file exists to prevent that.
        keys = replace(
            defaults,
            **{name: value for name, value in usable.items() if name in named},
            extra={name: value for name, value in usable.items() if name not in named},
        )
    return TuiSettings(
        keybindings=keys,
        sidebar=_coerce(data.get("sidebar"), ("left", "right", "off"), "right"),  # type: ignore[arg-type]
        turn_notification=_coerce(data.get("turn_notification"), ("off", "bell"), "bell"),  # type: ignore[arg-type]
        show_thinking=as_bool(data.get("show_thinking"), True),
        show_tools=as_bool(data.get("show_tools"), True),
        show_skills=as_bool(data.get("show_skills"), True),
        daemon_keep_alive=as_str(data.get("daemon_keep_alive"), "0"),
        show_tool_results=as_bool(data.get("show_tool_results"), True),
        # The strings of an array, and nothing from any other value. This is the
        # tolerance of this reader, applied to each entry.
        hidden_screens=tuple(
            one for one in as_seq(data.get("hidden_screens")) if isinstance(one, str) and one
        ),
    )


def load_tui_settings(home: Path) -> TuiSettings:
    """Read `$PH_HOME/tui.json`, or return defaults.

    `read_document` covers every case: an absent file is quiet, an unreadable
    file is logged, and both answer `None`. `tui_settings_from_json` already
    reads a value that is not a mapping as "no settings", so the two compose
    into one line.
    """
    return tui_settings_from_json(read_document(tui_settings_path(home)))


def save_tui_settings(home: Path, settings: TuiSettings) -> None:
    """Write `$PH_HOME/tui.json`. A `/view` toggle or a sidebar move is saved here."""
    # The write is atomic (O2). A second front end reads this file while this
    # one writes it, and half a settings document parses as no document.
    write_atomic(tui_settings_path(home), json.dumps(settings.to_json(), indent=2) + "\n")
