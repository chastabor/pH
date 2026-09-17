"""Themes are data, not code.

Built-in themes ship beside this module and load through the **same parser** as a
user's own, from `$PH_HOME/themes/`. That matters: if the built-ins took a private
path, a user theme could hit a validation rule the shipped ones never exercise, and
the bug would only appear on someone else's machine. The shipped set is
deliberately **mixed** — three JSON, four YAML — so neither notation is a branch
only somebody else's file takes.

A `TuiTheme` resolves to a Textual `Theme` plus `$ph-*` CSS variables, so a widget
names a role (`$ph-tool-error`) rather than a color. Re-theming is then a data
change, and a widget cannot quietly hard-code a hex value that survives it.

**Which theme is in force is this module's other job.**
`$PH_HOME/themes/theme-profile.yaml` is the one reader and the one writer of that
preference: `/theme` creates it on the first pick, and until something does,
`DEFAULT_THEME` applies and there is nothing for a person to set up. It is
deliberately *not* a key in `tui.json` — `save_tui_settings` persists that whole
document, so a theme left there would be rewritten by every unrelated preference
change and two files would claim one fact.

Design ported from tau's `tau_coding.tui.themes` (see `docs/dev-notes/phase-2.md`
for why pH re-implements rather than imports). The `vars` palette and the
report-every-fault parser are P9-01's half of that port; the profile is P9-02.

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
from ph.paths import write_text_under

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
"""The notations a theme may be written in, **in precedence order**.

A person with `mocha.yaml` beside a `mocha.json` has made their own ambiguity;
what this tuple buys is that it resolves the same way every run, rather than on
whatever order the directory happened to hand back.
"""

PROFILE_FILENAME = "theme-profile.yaml"
"""The preference file, which lives among the themes and is not one.

Beside `THEME_SUFFIXES` because the two together are what a filename in that
directory *means*, and because `_documents` — which runs at import, through
`BUILTIN_THEME_NAMES` — has to know the exclusion before the profile section
below is reached.

In `$PH_HOME/themes/` rather than `$PH_HOME/profiles/`: that directory is what
the profile loader lists for `--profile` names, so a theme document there would
offer itself as a plugin profile and be refused at compose time — a confusing
failure for a file that is not one.
"""


class ThemeError(ValueError):
    """A theme definition is missing a role or names an unusable color."""


@dataclass(frozen=True, slots=True)
class TuiTheme:
    """Every color role the pH TUI draws with.

    Deliberately a closed set. A widget that needs a color the theme does not
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
        """Every color role as a flat mapping, for CSS variables and tests."""
        return {role: getattr(self, role) for role in COLOR_ROLES}


COLOR_ROLES: tuple[str, ...] = tuple(
    field.name for field in fields(TuiTheme) if field.name not in ("name", "dark")
)
"""The roles that carry a color — everything but the theme's name and `dark`.

Derived rather than listed, so a role added to `TuiTheme` is parsed, validated and
exposed as a `$ph-*` variable with nothing else edited.
"""

_REQUIRED = frozenset(COLOR_ROLES) | {"dark"}


# ------------------------------------------------------------------ parsing --


def parse_theme(name: str, data: object, origin: str = "theme") -> TuiTheme:
    """Build a theme from a decoded document, refusing anything it cannot render.

    `origin` names the file in the error, because the message a user sees when a
    theme will not load should say which theme.

    **Every fault at once.** This used to raise on the first one, which against a
    hand-written theme of seventeen roles over a fourteen-color palette is
    seventeen edit-and-retry cycles. The refusals themselves are unchanged: an
    unknown role is still refused rather than ignored, because a typo'd role name
    would otherwise leave the real one at its default and read as a rendering bug.
    """
    if not isinstance(data, dict):
        raise ThemeError(f"{origin}: a theme must be a mapping")
    palette, problems = _palette(data.get("vars"))
    keys = set(data)
    problems.extend(f"unknown role {key!r}" for key in sorted(keys - _REQUIRED - {"name", "vars"}))
    problems.extend(f"missing {key}" for key in sorted(_REQUIRED - keys))
    # Defaulted rather than guarded on presence: an absent `dark` is already
    # reported as missing above, and `False` keeps it from being reported twice.
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
        # **Whole-value substitution, not tau's token-wise pass.** Every role here
        # is a single color, where tau's may be a Rich style string
        # (`bold #061a1a on #a7f3f0`) — which is why tau has to substitute per
        # token, and why it must then refuse a var named after a Rich keyword
        # (`on`, `bold`, …) that would corrupt one. pH has neither the style
        # strings nor, therefore, that hazard, so it does not carry the rule.
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
    """The optional `vars` block: a name per color, used by the roles below it.

    One level deep on purpose — a var may not refer to another var. A palette is
    a list of the colors a theme is built from, and resolution order is a
    feature nobody asked for and every reader would then have to hold in mind.

    Answers with its faults rather than appending to a list it was handed, which
    is the shape `_color_problem` beside it already has: two helpers in one pass
    reporting the same kind of thing two different ways is one way too many.
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
    """`None` when Textual can read this color, else how it reads to a person.

    Checked **at load**, not at paint. A theme carrying `#ff00gg` otherwise
    installs cleanly and takes the screen down on the frame that first draws the
    role — at which point the traceback names a widget rather than the file.
    """
    try:
        Color.parse(value)
    except ColorParseError:
        return f"is not a color Textual reads: {value!r}"
    return None


SHIPPED_DIR = Path(__file__).parent
"""Where the built-in documents live.

A plain `Path`, which is this repo's idiom for packaged data — `ph_app.profiles`
does it for the profile YAML one directory up, and each bundle does it for its
`bundle.yaml`. It is also what lets the shipped set and the user's share
`_documents` below: an `importlib.resources` `Traversable` has a `name` and no
`suffix`/`stem`/`glob`, so reading it cost a private re-implementation of all
three.
"""


def _documents(directory: Path) -> dict[str, Path]:
    """Every theme document in `directory`, one per name, newest notation winning.

    The precedence pass, written once for both the shipped set and the user's.
    Iterated in reverse so a later (higher-precedence) suffix overwrites: a person
    with `mine.yaml` beside `mine.json` has made their own ambiguity, and what
    this buys is that it resolves the same way every run rather than on the order
    the filesystem hands them back.

    The theme **profile** lives in this directory too and is not a theme, so it is
    passed over by name. Left to the parser it would be refused and skipped — the
    same outcome by accident, and a log line accusing a perfectly good file.
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
"""Everything a theme file can fail with: ph-core's set for reading one, plus
this module's own for parsing it. The answer to all of them is the same — skip
this file, keep the others, say which and why."""


def load_user_themes(home: Path) -> dict[str, TuiTheme]:
    """Themes from `$PH_HOME/themes/*.{yaml,yml,json}`.

    One bad file does not stop the TUI starting: it is skipped with its reason,
    because a theme is a preference and refusing to launch over one is worse than
    launching in the default.
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
    nowhere is a hard parse failure rather than a default color. An app that
    returns this from `get_theme_variable_defaults` makes every role resolvable
    always — which also means switching to one of Textual's own themes degrades
    the colors instead of crashing.
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
    """The person's theme preference: which one, and how `/theme` lists them.

    Every field is optional and an absent file is an empty one, which is the whole
    of the first-run story — `chosen` answers `DEFAULT_THEME` and nothing has to be
    configured before pH will start.
    """

    default: str = ""
    """The theme to open in. `/theme` writes this, and nothing else does."""
    order: tuple[str, ...] = ()
    """Names to list first in `/theme`, in this order. A name that no longer
    resolves is simply not listed — an ordering is a convenience, and refusing
    the file over a theme somebody deleted would cost them the rest of it."""

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
    """`$PH_HOME/themes` — the themes and the profile that names one, together."""
    return home / "themes"


def theme_profile_path(home: Path) -> Path:
    return themes_dir(home) / PROFILE_FILENAME


def load_theme_profile(home: Path) -> ThemeProfile:
    """Read the profile, or answer an empty one.

    A missing file is the ordinary first run. An unreadable one costs the
    customization and not the session — the rule `tui.json` already lives by, and
    for the same reason: defaults are always a valid answer for a preference.

    **A key this build has no field for is read past**, which is
    `tui_settings_from_json`'s tolerance rule applied to the other preference
    file. That is what makes a future field a pure addition — the light/dark pair
    a terminal-appearance switch would want was carried here and removed again
    precisely because it can be: a profile written by a newer pH still loads here,
    and one written here still loads there.
    """
    path = theme_profile_path(home)
    data = read_document(path)
    if data is None:
        # Absent on a first run, unreadable after a bad hand-edit —
        # `read_document` has said which, and the answer to both is the default.
        return ThemeProfile()
    if not isinstance(data, dict):
        log.warning("ph_app.tui: %s is not a mapping; using the default theme", path)
        return ThemeProfile()
    return ThemeProfile(
        default=_name(data.get("default")),
        order=tuple(name for one in as_seq(data.get("order")) if (name := _name(one))),
    )


def _name(value: object) -> str:
    """A theme name, or `""` for anything that is not one.

    `as_str` is ph-core's narrowing and the house spelling — the same one
    `tui/state.py` folds the log with — and it answers `""` for a mis-shaped
    field rather than `str(value)`'s plausible-looking `"None"`.
    """
    return as_str(value).strip()


def save_theme_profile(home: Path, profile: ThemeProfile) -> None:
    """Write `$PH_HOME/themes/theme-profile.yaml`. A `/theme` pick lands here.

    Dumped rather than formatted by hand: a theme name is a file stem and may
    carry anything a filesystem allows, and `safe_dump` is what knows when that
    needs quoting.
    """
    # One rule for both keys: drop what is unset, so a field nobody touched
    # leaves no trace in a file a person reads.
    written: dict[str, object] = {"default": profile.default, "order": list(profile.order)}
    document = {key: value for key, value in written.items() if value}
    body = yaml.safe_dump(document, sort_keys=False, allow_unicode=True) if document else ""
    write_text_under(theme_profile_path(home), _PROFILE_HEADER + body)


def choose_theme(home: Path, profile: ThemeProfile, name: str) -> ThemeProfile:
    """Record `name` as the theme to open in, and hand back the new profile.

    The one place a pick is written, so the caller does not have to know that a
    pick is a `replace` plus a dump. A write failure costs the memory, not the
    change: the returned profile is adopted either way, which is the same trade
    `PHTuiApp._save` makes for `tui.json`.
    """
    updated = replace(profile, default=name)
    try:
        save_theme_profile(home, updated)
    except OSError:
        log.warning("ph_app.tui: could not write %s", theme_profile_path(home), exc_info=True)
    return updated
