"""`$PH_HOME/tui.json` — keybindings and front-end preferences.

Two rules, both borrowed deliberately.

**Never hard-code a key check** (prime-agent's rule, adopted). Every binding is
a named field with a default, and the field names double as Textual binding ids,
so one `App.set_keymap(keybindings.as_map())` rebinds the whole app — screens
and modals included. A widget that compared `event.key == "escape"` directly
would silently ignore the user's setting.

**A broken settings file must not stop the TUI starting.** A preference file is
not a source of truth for anything the harness needs; if it fails to parse, the
TUI launches on defaults and says so. Refusing to launch over a typo'd sidebar
position would be a worse failure than the typo.

**The theme is not here** (P9-02). It lives in `$PH_HOME/themes/theme-profile.yaml`,
which `/theme` writes and nothing else does. It used to be a field on
`TuiSettings`, and the reason it could not stay is `save_tui_settings`: it
persists the *whole* document, so a `/view thinking` toggle would rewrite a theme
it had no opinion about, and two files would claim one fact. A `theme` key left
over from before this change is ignored on read by the rule below and gone on the
next write — which is the whole of the upgrade.

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
from ph.json import as_bool
from ph.paths import write_text_under

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
    """Configurable keys. Each field name is also the Textual binding id it maps."""

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
    """The same two keys the completion list uses, and deliberately so.

    They never contend, and `PromptInput._decide` is where that is arranged —
    stated there rather than restated here, since a claim about another module's
    branch order goes stale silently. Split fields even so, because they are two
    *bindings*: somebody who moves history onto `alt+up` to free the arrows for
    editing must be able to say that, and a single field would move both."""
    history_search: str = "alt+r"
    """`ctrl+r` is the session picker, which is older; the shell reflex has to
    give way to the binding that was already there."""
    toggle_thinking: str = "ctrl+t"
    toggle_tool_results: str = "ctrl+o"
    toggle_sidebar: str = "ctrl+b"
    quit: str = "ctrl+d"
    extra: Mapping[str, str] = field(default_factory=dict)
    """Binding ids this build has no field for — a plugin screen's key, under
    its screen id (P4-17).

    Kept rather than dropped, and that is the whole point: a screen contributed
    by a row is remapped in `tui.json` exactly like a built-in, because
    `set_keymap` rebinds by binding id and does not care where the id came
    from. An id nothing binds costs nothing."""

    def as_map(self) -> dict[str, str]:
        """Binding id → key: the shape `App.set_keymap` takes."""
        named = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        return {**named, **self.extra}


@dataclass(frozen=True, slots=True)
class TuiSettings:
    """Everything the front-end remembers between runs."""

    keybindings: TuiKeybindings = field(default_factory=TuiKeybindings)
    sidebar: SidebarPosition = "right"
    turn_notification: TurnNotification = "bell"
    show_thinking: bool = True
    show_tool_results: bool = True
    show_tools: bool = True
    show_skills: bool = True
    """Whether the sidebar's tools and skills panels are drawn.

    Visible by default, because what fills the context window is the thing a
    person should not have to go looking for. Hiding one hides its header too —
    a 32-column panel cannot afford a heading over nothing."""

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["keybindings"] = self.keybindings.as_map()
        return data


def tui_settings_path(home: Path) -> Path:
    return home / "tui.json"


def _coerce(value: object, allowed: tuple[str, ...], fallback: str) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def tui_settings_from_json(data: object) -> TuiSettings:
    """Build settings from parsed JSON, ignoring anything unrecognized.

    Tolerant on purpose: an older pH wrote fewer keys, a newer one writes more,
    and neither should make the other refuse to start.
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
        # What is left over is a binding id this build has no field for — a
        # contributed screen's. Dropping it would make exactly one class of key
        # unrebindable, which is the rule this file exists to prevent.
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
        show_tool_results=as_bool(data.get("show_tool_results"), True),
    )


def load_tui_settings(home: Path) -> TuiSettings:
    """Read `$PH_HOME/tui.json`, or return defaults.

    `read_document` carries the whole ladder — absent is quiet, unreadable is
    logged, both answer `None` — and `tui_settings_from_json` already reads a
    non-mapping as "no settings", so the two compose into one line.
    """
    return tui_settings_from_json(read_document(tui_settings_path(home)))


def save_tui_settings(home: Path, settings: TuiSettings) -> None:
    """Write `$PH_HOME/tui.json`. A `/view` toggle or a sidebar move lands here."""
    write_text_under(tui_settings_path(home), json.dumps(settings.to_json(), indent=2) + "\n")
