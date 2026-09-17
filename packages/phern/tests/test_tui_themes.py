"""Themes and the theme profile — the two files a person may hand-write.

Neither needs a running app, and that is the point of testing them here: a theme
that will not parse must fail at *load*, with a message naming the file and every
fault in it, rather than on the frame that first draws the role it broke.

Two rules carry most of this suite.

**The built-ins take no private path.** They load through the same parser as a
user's own, so the shipped set is deliberately mixed — three JSON, four YAML — and
`test_every_shipped_theme_parses` is what holds that honest. A notation only
somebody else's file takes is a notation nobody here tests.

**A preference file must not be able to stop the TUI.** A broken theme is skipped,
a broken profile costs the ordering, and in both cases pH opens in the default. The
opposite trade from a *profile document*, which is refused loudly — a deployment's
composition is not something to guess at, and a preference is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from ph_app.tui.config import load_tui_settings, save_tui_settings, tui_settings_path
from ph_app.tui.modals.pickers import theme_choices
from ph_app.tui.themes import (
    BUILTIN_THEME_NAMES,
    COLOR_ROLES,
    DEFAULT_THEME,
    PROFILE_FILENAME,
    ThemeError,
    ThemeProfile,
    choose_theme,
    load_catalog,
    load_theme_profile,
    load_user_themes,
    parse_theme,
    save_theme_profile,
    theme_document,
    theme_profile_path,
)

# --------------------------------------------------------------------- themes --


def test_every_builtin_theme_converts_to_textual() -> None:
    catalog = load_catalog()
    assert set(catalog.names) == set(BUILTIN_THEME_NAMES)
    for name in BUILTIN_THEME_NAMES:
        converted = catalog.themes[name].to_textual()
        assert converted.name == name
        # Every role reaches the stylesheet as a `$ph-*` variable, which is what
        # the widgets are written against.
        assert converted.variables
        assert all(key.startswith("ph-") for key in converted.variables)


def test_every_shipped_theme_parses_and_carries_every_role() -> None:
    """The gate the mixed shipped set exists for.

    Four of the seven are YAML with a `vars` palette, so this walks the notation
    *and* the substitution that a user's own file will take. A color Textual
    cannot read is a `ThemeError` from `load_catalog`, which is what makes this one
    assertion cover every hex digit in all seven files.

    Sabotage: change one palette entry to `#ff00gg` and the catalog refuses to
    build.
    """
    catalog = load_catalog()
    assert len(catalog.names) >= 7
    for name in catalog.names:
        roles = catalog.themes[name].roles()
        assert set(roles) == set(COLOR_ROLES)
        # Substituted, not left as a palette name: every value is a color.
        assert all(value and not value.isidentifier() for value in roles.values())


def test_a_yaml_theme_and_its_json_twin_parse_identically(tmp_path: Path) -> None:
    """One parser under two notations.

    Sabotage: drop the suffix dispatch in `_decode` and the YAML file never loads.
    """
    source = _roles()
    _write_theme(tmp_path, "twin-json.json", source)
    _write_theme(tmp_path, "twin-yaml.yaml", source)
    found = load_user_themes(tmp_path)
    assert found["twin-json"].roles() == found["twin-yaml"].roles()
    assert found["twin-yaml"].dark is True


def test_a_palette_name_substitutes_into_every_role() -> None:
    """`vars` is the whole reason a catppuccin file is legible.

    Sabotage: drop the substitution and every role is refused as "not a color",
    which is the failure this reads as when the palette is not applied.
    """
    theme = parse_theme(
        "palette",
        {"dark": True, "vars": {"ink": "#123456"}, **{role: "ink" for role in COLOR_ROLES}},
    )
    assert set(theme.roles().values()) == {"#123456"}


def test_a_var_that_is_not_a_color_is_refused() -> None:
    with pytest.raises(ThemeError, match=r"vars\.ink"):
        parse_theme(
            "bad-palette",
            {"dark": True, "vars": {"ink": "teal-ish"}, **{role: "ink" for role in COLOR_ROLES}},
        )


def test_a_theme_with_three_faults_names_all_three() -> None:
    """Every problem in one message.

    Seventeen roles over a fourteen-color palette is seventeen edit-and-retry
    cycles when the parser stops at the first fault.

    This also carries the **at load, not at paint** claim: `#ff00gg` is refused
    here, by `parse_theme`, rather than on the frame that first draws the role —
    at which point the traceback would name a widget. It had its own test, which
    was deleted as redundant: measured, removing it changed neither the covered
    lines nor the partial branches of `ph_app.tui.themes`, because this assertion
    exercises the same path with the same input.

    Sabotage: restore the first-fault `raise` and two of these three assertions
    fail.
    """
    document = {role: "#000000" for role in COLOR_ROLES}
    document["dark"] = "yes"
    document["acccent"] = "#fff"
    document["highlight_text"] = "#ff00gg"
    with pytest.raises(ThemeError) as raised:
        parse_theme("broken", document, "broken.yaml")
    message = str(raised.value)
    assert message.startswith("broken.yaml:")
    assert "unknown role 'acccent'" in message
    assert "'dark' must be a boolean" in message
    assert "highlight_text is not a color" in message


def test_yaml_wins_over_a_json_of_the_same_name(tmp_path: Path) -> None:
    """A person's own ambiguity, resolved the same way every run.

    Sabotage: iterate the directory once instead of once per suffix, and which
    file wins depends on the order the filesystem hands them back.
    """
    _write_theme(tmp_path, "mine.json", {**_roles(), "accent": "#111111"})
    _write_theme(tmp_path, "mine.yaml", {**_roles(), "accent": "#222222"})
    assert load_user_themes(tmp_path)["mine"].accent == "#222222"


def test_a_user_theme_shadows_a_builtin(tmp_path: Path) -> None:
    _write_theme(tmp_path, "ph-dark.json", {**_roles(), "accent": "#ff00ff"})
    catalog = load_catalog(tmp_path)
    assert catalog.themes["ph-dark"].accent == "#ff00ff"
    # Listed once, as the user's — that is the one that loads.
    assert catalog.names.count("ph-dark") == 1
    assert "ph-dark" in catalog.user


def test_a_theme_missing_a_role_is_refused(tmp_path: Path) -> None:
    _write_theme(tmp_path, "broken.json", {"dark": True, "background": "#000"})
    # Skipped rather than fatal: one bad file must not cost the user the others.
    assert "broken" not in load_user_themes(tmp_path)


def test_a_theme_with_an_unknown_role_is_refused() -> None:
    # A typo'd role would otherwise leave the real one at its default, which
    # reads as a rendering bug rather than a bad theme file.
    with pytest.raises(ThemeError):
        parse_theme("typo", {**_roles(), "acccent": "#fff"})


def test_an_unknown_theme_falls_back(tmp_path: Path) -> None:
    profile = ThemeProfile(default="does-not-exist")
    assert load_catalog(tmp_path).resolve(profile.chosen).name == DEFAULT_THEME


def _roles() -> dict[str, object]:
    document = theme_document("ph-dark")
    assert isinstance(document, dict)
    return dict(document)


def _write_theme(tmp_path: Path, filename: str, document: dict[str, object]) -> None:
    """One theme into `$PH_HOME/themes`, in whichever notation the name asks for.

    Four tests were writing this out by hand and had drifted on the `encoding=`
    kwarg between copies — which is exactly the kind of difference that makes two
    otherwise-identical setups read as if they meant something different.
    """
    directory = tmp_path / "themes"
    directory.mkdir(exist_ok=True)
    body = json.dumps(document) if filename.endswith(".json") else yaml.safe_dump(document)
    (directory / filename).write_text(body, encoding="utf-8")


# -------------------------------------------------------------- the profile --


def test_a_first_run_with_no_profile_uses_the_default_theme(tmp_path: Path) -> None:
    """Nothing to set up before pH will start, which is the whole of P9-02's ask.

    Sabotage: make `load_theme_profile` raise on a missing file.
    """
    assert not theme_profile_path(tmp_path).exists()
    assert load_theme_profile(tmp_path).chosen == DEFAULT_THEME


def test_the_first_pick_writes_the_profile(tmp_path: Path) -> None:
    """The file the docs name is the file that holds the answer.

    Sabotage: write `tui.json` instead, and the YAML never appears.
    """
    profile = choose_theme(tmp_path, load_theme_profile(tmp_path), "catppuccin-mocha")
    assert profile.default == "catppuccin-mocha"
    path = theme_profile_path(tmp_path)
    assert path.name == PROFILE_FILENAME
    assert "catppuccin-mocha" in path.read_text(encoding="utf-8")
    # Read back by the next launch, which is the claim that matters.
    assert load_theme_profile(tmp_path).chosen == "catppuccin-mocha"


def test_an_unrelated_preference_change_does_not_write_a_theme(tmp_path: Path) -> None:
    """The reason the field had to leave `TuiSettings` at all.

    `save_tui_settings` persists the whole document, so a theme left on it would
    be rewritten by a `/view` toggle — two files claiming one fact, the losing one
    still being maintained.

    Sabotage: put `theme` back on `TuiSettings`, and this fails on the next
    unrelated toggle.
    """
    choose_theme(tmp_path, ThemeProfile(), "catppuccin-latte")
    save_tui_settings(tmp_path, load_tui_settings(tmp_path))
    written = json.loads(tui_settings_path(tmp_path).read_text(encoding="utf-8"))
    assert "theme" not in written
    assert load_theme_profile(tmp_path).chosen == "catppuccin-latte"


def test_an_unreadable_profile_costs_the_ordering_and_not_the_session(tmp_path: Path) -> None:
    path = theme_profile_path(tmp_path)
    path.parent.mkdir(parents=True)
    path.write_text("default: [this is not a name\n", encoding="utf-8")
    assert load_theme_profile(tmp_path).chosen == DEFAULT_THEME


def test_a_profile_naming_a_theme_that_is_gone_still_opens(tmp_path: Path) -> None:
    """A deleted theme is cosmetic; the catalog falls back and the session runs."""
    save_theme_profile(tmp_path, ThemeProfile(default="deleted-theme"))
    catalog = load_catalog(tmp_path)
    assert catalog.resolve(load_theme_profile(tmp_path).chosen).name == DEFAULT_THEME


def test_the_profile_is_not_read_as_a_theme(tmp_path: Path) -> None:
    """It lives among the themes and is not one.

    Sabotage: drop the filename guard in `load_user_themes` — the parser refuses
    it anyway, so the test that catches this is the *log line* accusing a
    perfectly good file, which is why the guard is by name.
    """
    save_theme_profile(tmp_path, ThemeProfile(default="ph-light"))
    assert PROFILE_FILENAME.removesuffix(".yaml") not in load_user_themes(tmp_path)


def test_the_profile_order_leads_the_picker(tmp_path: Path) -> None:
    """Position is the only thing ordering says, so this asserts order.

    Sabotage: drop `profile.ordered` from `theme_choices` and the list is
    alphabetical again.
    """
    catalog = load_catalog(tmp_path)
    profile = ThemeProfile(default="ph-dark", order=("ph-light", "catppuccin-mocha", "gone"))
    rows = theme_choices("ph-dark", catalog, profile)
    assert [row.value for row in rows][:2] == ["ph-light", "catppuccin-mocha"]
    # Every theme is still listed, and exactly once — an ordering is not a filter.
    assert sorted(row.value for row in rows) == catalog.names
    assert next(row for row in rows if row.value == "ph-dark").marked


def test_exactly_one_row_is_the_default(tmp_path: Path) -> None:
    """`default` names what opens next launch; the dot names what you are seeing.

    The two differ while the picker previews, which is the reason they are two
    things. Sabotage: label the `order` rows instead, and three rows claim a word
    the profile records once.
    """
    catalog = load_catalog(tmp_path)
    profile = ThemeProfile(default="ph-light", order=("catppuccin-mocha", "ph-light"))
    # `active` is mid-preview, two rows away from the stored default.
    rows = theme_choices("catppuccin-mocha", catalog, profile)
    defaults = [row.value for row in rows if "default" in row.detail]
    assert defaults == ["ph-light"]
    assert next(row for row in rows if row.value == "catppuccin-mocha").marked

    # And on a first run, with no profile at all, it is the built-in default.
    bare = theme_choices(DEFAULT_THEME, catalog, ThemeProfile())
    assert [row.value for row in bare if "default" in row.detail] == [DEFAULT_THEME]
