"""Themes are data, not code.

Built-in themes ship beside this module and load through the **same parser** as
a theme of the user, which is read from `$PH_HOME/themes/`. That matters. If
the built-in themes took a private path, a user theme could hit a validation
rule that the shipped themes never exercise, and the bug would appear only on
the machine of somebody else. The shipped set is deliberately **mixed**: three
JSON, four YAML. So neither notation is a branch that only the file of somebody
else takes.

A `TuiTheme` resolves to a Textual `Theme` plus `$ph-*` CSS variables, so a
widget names a role (`$ph-tool-error`) and not a color. A change of theme is
then a data change, and a widget cannot hard-code a hex value that would
survive it unnoticed.

**This module also decides which theme is in force.**
`$PH_HOME/themes/theme-profile.yaml` is the one file that the preference is
read from and written to. `/theme` creates the file on the first pick. Until
something does, `DEFAULT_THEME` applies and the user has nothing to set up. The
preference is deliberately *not* a key in `tui.json`. `save_tui_settings`
persists that whole document, so a theme left there would be rewritten by every
unrelated preference change, and two files would claim one fact.

The design is ported from tau's `tau_coding.tui.themes` (see
`docs/dev-notes/phase-2.md` for why pH implements it again and does not import
it). The `vars` palette and the parser that reports every fault are the half of
that port that P9-01 did. The profile is P9-02.

@module ph_app.tui.themes
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, replace
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from textual.app import App
from textual.color import Color, ColorParseError
from textual.theme import Theme

from ph.documents import DOCUMENT_FAULTS, decode_document, read_document
from ph.json import as_seq, as_str
from ph.paths import write_atomic

__all__ = [
    "BUILTIN_THEME_NAMES",
    "COLOR_ROLES",
    "DEFAULT_THEME",
    "PROFILE_FILENAME",
    "THEME_SUFFIXES",
    "ThemeCatalog",
    "ThemeError",
    "ThemeProfile",
    "TuiTheme",
    "choose_theme",
    "fallback_variables",
    "load_catalog",
    "load_theme_profile",
    "load_user_themes",
    "parse_theme",
    "save_theme_profile",
    "theme_document",
    "theme_profile_path",
    "themes_dir",
]

log = logging.getLogger("ph_app.tui.themes")

DEFAULT_THEME = "ph-dark"

THEME_SUFFIXES: tuple[str, ...] = (".yaml", ".yml", ".json")
"""The notations that a theme may be written in, **in precedence order**.

A user with `mocha.yaml` beside a `mocha.json` has made their own ambiguity.
The benefit of this tuple is that the ambiguity resolves the same way on every
run, and not by the order in which the directory happened to list the files.
"""

PROFILE_FILENAME = "theme-profile.yaml"
"""The preference file. It is stored among the themes and is not a theme.

It is defined beside `THEME_SUFFIXES` for two reasons. The two together are
what a filename in that directory *means*. And `_documents`, which runs at
import through `BUILTIN_THEME_NAMES`, has to know the exclusion before the
profile section below is reached.

The file is in `$PH_HOME/themes/` and not in `$PH_HOME/profiles/`. The profile
loader lists that second directory for `--profile` names. A theme document
there would offer itself as a plugin profile and be refused at compose time,
which is a confusing failure for a file that is not a plugin profile.
"""


class ThemeError(ValueError):
    """A theme definition is missing a role or names an unusable color."""


@dataclass(frozen=True, slots=True)
class TuiTheme:
    """Every color role that the pH TUI draws with.

    The set is deliberately closed. A widget that needs a color that the theme
    does not name should ask for a *role*, which a designer can point at another
    color. It should not use a literal.
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
        """Every color role as a flat mapping, for CSS variables and tests."""
        return {role: getattr(self, role) for role in COLOR_ROLES}


COLOR_ROLES: tuple[str, ...] = tuple(
    field.name for field in fields(TuiTheme) if field.name not in ("name", "dark")
)
"""The roles that carry a color: everything but the name of the theme and `dark`.

The tuple is derived and not listed, so a role added to `TuiTheme` is parsed,
validated and exposed as a `$ph-*` variable with no other edit.
"""

_REQUIRED = frozenset(COLOR_ROLES) | {"dark"}


# ------------------------------------------------------------------ parsing --


def parse_theme(name: str, data: object, origin: str = "theme") -> TuiTheme:
    """Build a theme from a decoded document, and refuse anything that it cannot render.

    `origin` names the file in the error. The message that a user sees when a
    theme does not load should say which theme.

    **Every fault is reported at once.** This used to raise on the first fault.
    For a hand-written theme of seventeen roles over a fourteen-color palette,
    that is seventeen cycles of edit and retry. The refusals themselves are
    unchanged. An unknown role is still refused and not ignored, because a
    mistyped role name would otherwise leave the real role at its default and
    look like a rendering bug.
    """
    if not isinstance(data, dict):
        raise ThemeError(f"{origin}: a theme must be a mapping")
    palette, problems = _palette(data.get("vars"))
    keys = set(data)
    problems.extend(f"unknown role {key!r}" for key in sorted(keys - _REQUIRED - {"name", "vars"}))
    problems.extend(f"missing {key}" for key in sorted(_REQUIRED - keys))
    # Defaulted, not guarded on presence. An absent `dark` is already reported
    # as missing above, and `False` keeps it from being reported twice.
    dark = data.get("dark", False)
    if not isinstance(dark, bool):
        problems.append("'dark' must be a boolean")
    roles: dict[str, str] = {}
    for role in COLOR_ROLES:
        if role not in data:
            continue  # already reported as missing
        raw = as_str(data[role]).strip()
        if not raw:
            problems.append(f"{role} must be a non-empty string")
            continue
        # **Whole-value substitution, not the token-wise pass of tau.** Every
        # role here is a single color. A role of tau may be a Rich style string
        # (`bold #061a1a on #a7f3f0`). That is why tau has to substitute for
        # each token, and why it must then refuse a var named after a Rich
        # keyword (`on`, `bold`, …) that would corrupt a style string. pH has no
        # style strings and therefore no such hazard, so it does not carry the
        # rule.
        value = palette.get(raw, raw)
        fault = _color_problem(value)
        if fault is not None:
            problems.append(f"{role} {fault}")
            continue
        roles[role] = value
    if problems:
        raise ThemeError(f"{origin}: " + "; ".join(problems))
    return TuiTheme(name=name, dark=bool(dark), **roles)


def _palette(raw: object) -> tuple[dict[str, str], list[str]]:
    """Return the optional `vars` block: a name for each color, used by the roles below it.

    The block is one level deep on purpose: a var may not refer to another var.
    A palette is a list of the colors that a theme is built from. Resolution
    order is a feature that nobody asked for, and every reader would then have
    to keep it in mind.

    The function returns its faults and does not append to a list that it was
    handed. `_color_problem` beside it already has that shape. Two helpers in
    one pass that report the same kind of thing in two different ways are one
    way too many.
    """
    problems: list[str] = []
    if raw is None:
        return {}, problems
    if not isinstance(raw, dict):
        return {}, ["'vars' must be a mapping of name to color"]
    palette: dict[str, str] = {}
    for key, value in raw.items():
        name = as_str(key).strip()
        color = as_str(value).strip()
        if not name:
            problems.append(f"vars name must be a non-empty string: {key!r}")
        elif not color or _color_problem(color) is not None:
            problems.append(f"vars.{name} must be a color: {value!r}")
        else:
            palette[name] = color
    return palette, problems


def _color_problem(value: str) -> str | None:
    """Return `None` when Textual can read this color, and otherwise a description
    of the fault for the user.

    The color is checked **at load**, not at paint. Otherwise a theme that
    carries `#ff00gg` installs cleanly and takes the screen down on the frame
    that first draws the role. At that point the traceback names a widget and
    not the file.
    """
    try:
        Color.parse(value)
    except ColorParseError:
        return f"is not a color Textual reads: {value!r}"
    return None


SHIPPED_DIR = Path(__file__).parent
"""Where the built-in documents are.

This is a plain `Path`, which is the idiom of this repo for packaged data.
`ph_app.profiles` uses it for the profile YAML one directory up, and each
bundle uses it for its `bundle.yaml`. It also lets the shipped set and the set
of the user share `_documents` below. An `importlib.resources` `Traversable`
has a `name` and no `suffix`/`stem`/`glob`, so reading it cost a private
implementation of all three.
"""


def _documents(directory: Path) -> dict[str, Path]:
    """Return every theme document in `directory`, one for each name. The notation
    with the highest precedence wins.

    This is the precedence pass, written one time for both the shipped set and
    the set of the user. The suffixes are iterated in reverse, so that a later
    suffix, which has higher precedence, overwrites. A user with `mine.yaml`
    beside `mine.json` has made their own ambiguity. The benefit is that it
    resolves the same way on every run, and not by the order in which the
    filesystem returns the files.

    The theme **profile** is in this directory too and is not a theme, so it is
    skipped by name. If it were left to the parser, it would be refused and
    skipped. That is the same outcome by accident, with a log line that accuses
    a perfectly good file.
    """
    found: dict[str, Path] = {}
    for suffix in reversed(THEME_SUFFIXES):
        for path in sorted(directory.glob(f"*{suffix}")):
            if path.name != PROFILE_FILENAME:
                found[path.stem] = path
    return found


def theme_document(name: str) -> object:
    """A built-in theme as decoded data, whichever notation it ships in."""
    path = _documents(SHIPPED_DIR).get(name)
    if path is None:
        raise FileNotFoundError(SHIPPED_DIR / f"{name}{THEME_SUFFIXES[0]}")
    return decode_document(path)


@cache
def _builtins() -> dict[str, TuiTheme]:
    return {
        name: parse_theme(name, decode_document(path), path.name)
        for name, path in sorted(_documents(SHIPPED_DIR).items())
    }


BUILTIN_THEME_NAMES: tuple[str, ...] = tuple(sorted(_builtins()))
"""Discovered from the shipped files, so an eighth document is an eighth theme."""


_UNREADABLE: tuple[type[Exception], ...] = (ThemeError, *DOCUMENT_FAULTS)
"""Everything that a theme file can fail with: the set of ph-core for reading
one, plus the error of this module for parsing it. The answer to all of them is
the same: skip this file, keep the others, and say which file and why."""


def load_user_themes(home: Path) -> dict[str, TuiTheme]:
    """Load the themes from `$PH_HOME/themes/*.{yaml,yml,json}`.

    One bad file does not stop the TUI from starting. The file is skipped and
    its reason is logged. A theme is a preference, and refusing to launch over
    one is worse than launching in the default.
    """
    directory = themes_dir(home)
    if not directory.is_dir():
        return {}
    found: dict[str, TuiTheme] = {}
    for name, path in sorted(_documents(directory).items()):
        try:
            found[name] = parse_theme(name, decode_document(path), str(path))
        except _UNREADABLE as error:
            log.warning("ph_app.tui: skipping theme %s (%s)", path, error)
    return found


# ------------------------------------------------------------------ catalog --


@dataclass(frozen=True, slots=True)
class ThemeCatalog:
    """Every theme that one run can offer, read one time.

    The app builds a catalog at start and hands it to whatever lists, resolves
    or describes a theme. Before the catalog existed, each of those read
    `$PH_HOME/themes` again, so startup scanned the directory one time for each
    theme name.

    **Not enforced (§5 rule 6): a theme file is read at start and never again.**
    Editing one while pH is running changes nothing until the next launch, and
    there is no watcher. That is the cost of the paragraph above and not an
    oversight. `/theme` reads nothing again either: it picks from this catalog.
    A user who iterates on a palette restarts, and nothing more is needed.
    """

    themes: Mapping[str, TuiTheme]
    user: frozenset[str]
    """Names that came from `$PH_HOME/themes`. A user file that shadows a
    built-in theme is listed one time, as the user's, because that is the file
    that loads."""

    @property
    def names(self) -> list[str]:
        return sorted(self.themes)

    def resolve(self, name: str) -> TuiTheme:
        """Return the theme by name, or the default when the name is unknown.

        A theme that vanished (a deleted user file, a renamed built-in) is a
        cosmetic problem. Falling back keeps the session usable.
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
    """Load the built-in themes plus those of the user, in one directory scan."""
    user = load_user_themes(home) if home is not None else {}
    return ThemeCatalog(themes={**_builtins(), **user}, user=frozenset(user))


def fallback_variables() -> dict[str, str]:
    """Return the `$ph-*` variables that every stylesheet can rely on: those of
    the default theme.

    Textual parses CSS before a theme is chosen, and a `$ph-*` that resolves
    nowhere is a hard parse failure and not a default color. An app that returns
    this from `get_theme_variable_defaults` makes every role always resolvable.
    A switch to one of Textual's own themes then degrades the colors and does
    not crash.
    """
    return dict(_builtins()[DEFAULT_THEME].to_textual().variables or {})


# ------------------------------------------------------------------ profile --

_PROFILE_HEADER = """\
# pH theme profile — which theme the TUI opens in.
#
# Written by `/theme`; edit it by hand if you prefer, and delete it to go back to
# the default. The themes themselves are the other files in this directory, as
# `.yaml` or `.json`.
"""


@dataclass(frozen=True, slots=True)
class ThemeProfile:
    """The theme preference of the user: which theme, and how `/theme` lists them.

    Every field is optional, and an absent file is an empty one. That is all
    that a first run needs: `chosen` answers `DEFAULT_THEME`, and nothing has to
    be configured before pH starts.
    """

    default: str = ""
    """The theme to open in. `/theme` writes this, and nothing else does."""
    order: tuple[str, ...] = ()
    """Names to list first in `/theme`, in this order. A name that no longer
    resolves is not listed. An ordering is a convenience, and refusing the file
    over a theme that somebody deleted would cost them the rest of it."""

    @property
    def chosen(self) -> str:
        """The theme to open in: this profile's own, else the default."""
        return self.default or DEFAULT_THEME

    def ordered(self, names: Sequence[str]) -> list[str]:
        """`names`, with this profile's order first and everything else after it."""
        return [one for one in self.order if one in names] + [
            one for one in names if one not in self.order
        ]


def themes_dir(home: Path) -> Path:
    """Return `$PH_HOME/themes`, which holds the themes and the profile that names one."""
    return home / "themes"


def theme_profile_path(home: Path) -> Path:
    return themes_dir(home) / PROFILE_FILENAME


def load_theme_profile(home: Path) -> ThemeProfile:
    """Read the profile, or return an empty one.

    A missing file is the ordinary first run. An unreadable file costs the
    customization and not the session. `tui.json` already follows that rule, for
    the same reason: defaults are always a valid answer for a preference.

    **A key that this build has no field for is skipped.** That is the tolerance
    rule of `tui_settings_from_json`, applied to the other preference file. It
    makes a future field a pure addition: a profile written by a newer pH still
    loads here, and one written here still loads there. The light/dark pair
    that a terminal-appearance switch would want was carried here and removed
    again, precisely because it can be added later in that way.
    """
    path = theme_profile_path(home)
    data = read_document(path)
    if data is None:
        # The file is absent on a first run, or unreadable after a bad edit by
        # hand. `read_document` has already logged an unreadable file, and the
        # answer to both is the default.
        return ThemeProfile()
    if not isinstance(data, dict):
        log.warning("ph_app.tui: %s is not a mapping; using the default theme", path)
        return ThemeProfile()
    return ThemeProfile(
        default=_name(data.get("default")),
        order=tuple(name for one in as_seq(data.get("order")) if (name := _name(one))),
    )


def _name(value: object) -> str:
    """Return a theme name, or `""` for anything that is not one.

    `as_str` is the narrowing function of ph-core and the spelling that this
    codebase uses. `tui/state.py` folds the log with the same function. It
    answers `""` for a field of the wrong shape, where `str(value)` would give
    a plausible-looking `"None"`.
    """
    return as_str(value).strip()


def save_theme_profile(home: Path, profile: ThemeProfile) -> None:
    """Write `$PH_HOME/themes/theme-profile.yaml`. A `/theme` pick is written here.

    The document is dumped and not formatted by hand. A theme name is a file
    stem and may carry anything that a filesystem allows, and `safe_dump` knows
    when that needs quoting.
    """
    # One rule for both keys: drop what is unset, so that a field that nobody
    # touched leaves no trace in a file that a person reads.
    written: dict[str, object] = {"default": profile.default, "order": list(profile.order)}
    document = {key: value for key, value in written.items() if value}
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True) if document else ""
    # Written atomically (O2), for the same reason as in `save_tui_settings`.
    # The profile is a whole document, and a reader that meets half of it falls
    # back to the default theme and not the one that somebody chose.
    write_atomic(theme_profile_path(home), _PROFILE_HEADER + body)


def choose_theme(home: Path, profile: ThemeProfile, name: str) -> ThemeProfile:
    """Record `name` as the theme to open in, and return the new profile.

    This is the one place where a pick is written, so the caller does not have
    to know that a pick is a `replace` plus a dump. A write failure costs the
    memory, not the change: the returned profile is adopted either way.
    `PHTuiApp._save` makes the same trade for `tui.json`.
    """
    updated = replace(profile, default=name)
    try:
        save_theme_profile(home, updated)
    except OSError:
        log.warning("ph_app.tui: could not write %s", theme_profile_path(home), exc_info=True)
    return updated
