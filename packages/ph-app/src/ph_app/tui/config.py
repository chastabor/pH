"""`$PH_HOME/tui.json` — keybindings, theme, and front-end preferences.

Two rules, both borrowed deliberately.

**Never hard-code a key check** (prime-agent's rule, adopted). Every binding is
a named field with a default, and the field names double as Textual binding ids,
so one `App.set_keymap(keybindings.as_map())` rebinds the whole app — screens
and modals included. A widget that compared `event.key == "escape"` directly
would silently ignore the user's setting.

**A broken settings file must not stop the TUI starting.** A preference file is
not a source of truth for anything the harness needs; if it fails to parse, the
TUI launches on defaults and says so. Refusing to launch over a typo'd theme
name would be a worse failure than the typo.

@module ph_app.tui.config
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Literal, TypeAlias

from ph.paths import write_text_under

from .themes import DEFAULT_THEME

__all__ = [
    "DEFAULT_THEME",
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
    toggle_thinking: str = "ctrl+t"
    toggle_tool_results: str = "ctrl+o"
    toggle_sidebar: str = "ctrl+b"
    quit: str = "ctrl+d"

    def as_map(self) -> dict[str, str]:
        """Binding id → key: the shape `App.set_keymap` takes."""
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass(frozen=True, slots=True)
class TuiSettings:
    """Everything the front-end remembers between runs."""

    keybindings: TuiKeybindings = field(default_factory=TuiKeybindings)
    theme: str = DEFAULT_THEME
    sidebar: SidebarPosition = "right"
    turn_notification: TurnNotification = "bell"
    show_thinking: bool = True
    show_tool_results: bool = True

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["keybindings"] = self.keybindings.as_map()
        return data


def tui_settings_path(home: Path) -> Path:
    return home / "tui.json"


def _coerce(value: Any, allowed: tuple[str, ...], fallback: str) -> str:
    return value if isinstance(value, str) and value in allowed else fallback


def tui_settings_from_json(data: Any) -> TuiSettings:
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
        overrides = {
            f.name: raw_keys[f.name]
            for f in fields(defaults)
            if isinstance(raw_keys.get(f.name), str) and raw_keys[f.name]
        }
        keys = replace(defaults, **overrides)
    return TuiSettings(
        keybindings=keys,
        theme=data["theme"] if isinstance(data.get("theme"), str) else DEFAULT_THEME,
        sidebar=_coerce(data.get("sidebar"), ("left", "right", "off"), "right"),  # type: ignore[arg-type]
        turn_notification=_coerce(data.get("turn_notification"), ("off", "bell"), "bell"),  # type: ignore[arg-type]
        show_thinking=bool(data.get("show_thinking", True)),
        show_tool_results=bool(data.get("show_tool_results", True)),
    )


def load_tui_settings(home: Path) -> TuiSettings:
    """Read `$PH_HOME/tui.json`, or return defaults."""
    path = tui_settings_path(home)
    try:
        return tui_settings_from_json(json.loads(path.read_text(encoding="utf-8")))
    except FileNotFoundError:
        return TuiSettings()
    except (json.JSONDecodeError, OSError) as error:
        log.warning("ph_app.tui: %s is unreadable (%s); using defaults", path, error)
        return TuiSettings()


def save_tui_settings(home: Path, settings: TuiSettings) -> None:
    """Write `$PH_HOME/tui.json`. A toggle or a theme pick lands here."""
    write_text_under(tui_settings_path(home), json.dumps(settings.to_json(), indent=2) + "\n")
