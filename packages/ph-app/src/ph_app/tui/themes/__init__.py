"""Themes are data, not code.

Built-in themes ship as JSON next to this module and load through the **same
parser** as a user's own, from `$PH_HOME/themes/*.json`. That matters: if the
built-ins took a private path, a user theme could hit a validation rule the
shipped ones never exercise, and the bug would only appear on someone else's
machine.

A `TuiTheme` resolves to a Textual `Theme` plus `$ph-*` CSS variables, so a
widget names a role (`$ph-tool-error`) rather than a colour. Re-theming is then a
data change, and a widget cannot quietly hard-code a hex value that survives it.

Design ported from tau's `tau_coding.tui.themes` (see `docs/dev-notes/phase-2.md`
for why pH re-implements rather than imports).

@module ph_app.tui.themes
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, fields
from functools import cache
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any

from textual.app import App
from textual.theme import Theme

__all__ = [
    "BUILTIN_THEME_NAMES",
    "DEFAULT_THEME",
    "ThemeCatalog",
    "ThemeError",
    "TuiTheme",
    "fallback_variables",
    "load_catalog",
    "load_user_themes",
    "parse_theme",
    "theme_file",
]

log = logging.getLogger("ph_app.tui.themes")

DEFAULT_THEME = "ph-dark"


class ThemeError(ValueError):
    """A theme definition is missing a role or names an unusable colour."""


@dataclass(frozen=True, slots=True)
class TuiTheme:
    """Every colour role the pH TUI draws with.

    Deliberately a closed set. A widget that needs a colour the theme does not
    name should be asking for a *role* — one a designer can re-point — not
    reaching for a literal.
    """

    name: str
    dark: bool
    background: str
    foreground: str
    surface: str
    panel: str
    muted: str
    border: str
    accent: str
    success: str
    error: str
    warning: str
    user_text: str
    assistant_text: str
    thinking_text: str
    tool_success: str
    tool_error: str
    highlight_background: str
    highlight_text: str

    def to_textual(self) -> Theme:
        """The Textual theme, with every role also exposed as a `$ph-*` variable."""
        return Theme(
            name=self.name,
            dark=self.dark,
            background=self.background,
            foreground=self.foreground,
            surface=self.surface,
            panel=self.panel,
            primary=self.accent,
            secondary=self.muted,
            accent=self.accent,
            success=self.success,
            error=self.error,
            warning=self.warning,
            variables={f"ph-{key.replace('_', '-')}": value for key, value in self.roles().items()},
        )

    def roles(self) -> dict[str, str]:
        """Every colour role as a flat mapping, for CSS variables and tests."""
        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name not in ("name", "dark")
        }


_REQUIRED = frozenset(field.name for field in fields(TuiTheme)) - {"name"}


def parse_theme(name: str, data: Any, origin: str = "theme") -> TuiTheme:
    """Build a theme from decoded JSON, refusing anything it cannot render.

    `origin` names the file in the error, because the message a user sees when
    a theme will not load should say which theme.
    """
    if not isinstance(data, dict):
        raise ThemeError(f"{origin}: a theme must be a JSON object")
    missing = sorted(_REQUIRED - set(data))
    if missing:
        raise ThemeError(f"{origin}: theme is missing {', '.join(missing)}")
    unknown = sorted(set(data) - _REQUIRED - {"name"})
    if unknown:
        # Refused rather than ignored: a typo'd role would otherwise leave the
        # real one at its default and look like a rendering bug.
        raise ThemeError(f"{origin}: theme has unknown roles {', '.join(unknown)}")
    dark = data["dark"]
    if not isinstance(dark, bool):
        raise ThemeError(f"{origin}: 'dark' must be a boolean")
    return TuiTheme(name=name, **{key: value for key, value in data.items() if key != "name"})


def theme_file(name: str) -> Traversable:
    """The packaged JSON for a built-in theme — a starting point for a user's own."""
    return files(__package__) / f"{name}.json"


@cache
def _builtins() -> dict[str, TuiTheme]:
    themes: dict[str, TuiTheme] = {}
    for resource in files(__package__).iterdir():
        if not resource.name.endswith(".json"):
            continue
        name = resource.name.removesuffix(".json")
        data = json.loads(resource.read_text(encoding="utf-8"))
        themes[name] = parse_theme(name, data, resource.name)
    return themes


BUILTIN_THEME_NAMES: tuple[str, ...] = tuple(sorted(_builtins()))
"""Discovered from the shipped files, so a fourth JSON is a fourth theme."""


def load_user_themes(home: Path) -> dict[str, TuiTheme]:
    """Themes from `$PH_HOME/themes/*.json`.

    One bad file does not stop the TUI starting: it is skipped with its reason,
    because a theme is a preference and refusing to launch over one is worse
    than launching in the default.
    """
    directory = home / "themes"
    found: dict[str, TuiTheme] = {}
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*.json")):
        try:
            found[path.stem] = parse_theme(
                path.stem, json.loads(path.read_text(encoding="utf-8")), str(path)
            )
        except (ThemeError, json.JSONDecodeError, OSError):
            continue
    return found


@dataclass(frozen=True, slots=True)
class ThemeCatalog:
    """Every theme one run can offer, read once.

    The app builds a catalog at start and hands it to whatever lists, resolves
    or describes a theme. Before this existed each of those re-read
    `$PH_HOME/themes`, so startup scanned the directory once per theme name.
    """

    themes: Mapping[str, TuiTheme]
    user: frozenset[str]
    """Names that came from `$PH_HOME/themes`. A user file shadowing a built-in
    is listed once, as the user's, because that is the one that loads."""

    @property
    def names(self) -> list[str]:
        return sorted(self.themes)

    def resolve(self, name: str) -> TuiTheme:
        """The theme by name, falling back to the default when it is unknown.

        A theme that vanished (a deleted user file, a renamed built-in) is a
        cosmetic problem; falling back keeps the session usable.
        """
        theme = self.themes.get(name)
        if theme is None:
            log.warning("ph_app.tui: unknown theme %r; using %s", name, DEFAULT_THEME)
            return self.themes[DEFAULT_THEME]
        return theme

    def install(self, app: App[Any]) -> None:
        """Register every theme with a Textual app."""
        for theme in self.themes.values():
            app.register_theme(theme.to_textual())


def load_catalog(home: Path | None = None) -> ThemeCatalog:
    """Built-ins plus the user's, one directory scan."""
    user = load_user_themes(home) if home is not None else {}
    return ThemeCatalog(themes={**_builtins(), **user}, user=frozenset(user))


def fallback_variables() -> dict[str, str]:
    """The `$ph-*` variables every stylesheet can rely on: the default theme's.

    Textual parses CSS before a theme is chosen, and a `$ph-*` that resolves
    nowhere is a hard parse failure rather than a default colour. An app that
    returns this from `get_theme_variable_defaults` makes every role resolvable
    always — which also means switching to one of Textual's own themes degrades
    the colours instead of crashing.
    """
    return dict(_builtins()[DEFAULT_THEME].to_textual().variables or {})
