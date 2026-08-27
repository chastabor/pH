"""Themes, settings, completion, and session listing — the parts with no app.

Kept separate from the pilot tests because these are the pieces a broken
terminal must not be able to hide: a settings file that stops the TUI starting
is worse than a settings file pH ignores, and the only way to know which one
happens is to test it without a running app.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ph_app.tui.autocomplete import (
    PathCompleter,
    build_completion_state,
    parse_completion_token,
)
from ph_app.tui.config import DEFAULT_THEME, load_tui_settings, tui_settings_from_json
from ph_app.tui.modals.login import credential_choices, credential_names
from ph_app.tui.modals.pickers import session_choices
from ph_app.tui.sessions import session_summaries
from ph_app.tui.themes import (
    BUILTIN_THEME_NAMES,
    ThemeError,
    load_catalog,
    load_user_themes,
    parse_theme,
    theme_file,
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


def test_a_user_theme_shadows_a_builtin(tmp_path: Path) -> None:
    source = _roles()
    source["accent"] = "#ff00ff"
    directory = tmp_path / "themes"
    directory.mkdir()
    (directory / "ph-dark.json").write_text(json.dumps(source))
    catalog = load_catalog(tmp_path)
    assert catalog.themes["ph-dark"].accent == "#ff00ff"
    # Listed once, as the user's — that is the one that loads.
    assert catalog.names.count("ph-dark") == 1
    assert "ph-dark" in catalog.user


def test_a_theme_missing_a_role_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "themes"
    directory.mkdir()
    (directory / "broken.json").write_text(json.dumps({"dark": True, "background": "#000"}))
    # Skipped rather than fatal: one bad file must not cost the user the others.
    assert "broken" not in load_user_themes(tmp_path)


def test_a_theme_with_an_unknown_role_is_refused() -> None:
    # A typo'd role would otherwise leave the real one at its default, which
    # reads as a rendering bug rather than a bad theme file.
    with pytest.raises(ThemeError):
        parse_theme("typo", {**_roles(), "acccent": "#fff"})


def test_an_unknown_theme_falls_back(tmp_path: Path) -> None:
    settings = tui_settings_from_json({"theme": "does-not-exist"})
    assert load_catalog(tmp_path).resolve(settings.theme).name == DEFAULT_THEME


def _roles() -> dict[str, object]:
    data: dict[str, object] = json.loads(theme_file("ph-dark").read_text(encoding="utf-8"))
    return data


# ------------------------------------------------------------------- settings --


def test_settings_survive_an_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / "tui.json").write_text("{ not json")
    settings = load_tui_settings(tmp_path)
    assert settings.theme == DEFAULT_THEME


def test_an_unknown_key_in_settings_is_ignored() -> None:
    settings = tui_settings_from_json(
        {"theme": "ph-light", "somethingFromANewerPH": 3, "keybindings": {"quit": "ctrl+d"}}
    )
    assert settings.theme == "ph-light"
    assert settings.keybindings.quit == "ctrl+d"
    # Unnamed bindings keep their defaults rather than becoming unset.
    assert settings.keybindings.cancel


# --------------------------------------------------------------- autocomplete --


def test_a_slash_completes_only_at_the_start_of_the_prompt() -> None:
    assert parse_completion_token("/comp") is not None
    assert parse_completion_token("see docs/plan") is None
    assert parse_completion_token("look at /etc") is None


def test_completion_replaces_only_the_token() -> None:
    state = build_completion_state("/comp", commands=[("compact", "Compact the context")])
    assert [item.label for item in state.items] == ["/compact"]
    assert state.replace(state.items[0]) == "/compact "


def test_a_trailing_space_closes_the_popup() -> None:
    assert build_completion_state("/compact ", commands=[("compact", "")]).items == ()


def test_path_completion_lists_one_directory(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "readme.md").touch()
    (tmp_path / ".hidden").touch()
    found = PathCompleter(root=str(tmp_path))("")
    assert list(found) == ["src/", "readme.md"], "directories first, hidden entries out"


def test_path_completion_filters_by_stem(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").touch()
    (tmp_path / "src" / "base.py").touch()
    assert list(PathCompleter(root=str(tmp_path))("src/ap")) == ["src/app.py"]


def test_path_completion_survives_a_missing_directory(tmp_path: Path) -> None:
    assert PathCompleter(root=str(tmp_path))("nope/deeper") == ()


# -------------------------------------------------------------------- sessions --


def _write_session(directory: Path, session_id: str, title: str, **header: object) -> None:
    record = {"id": session_id, "createdAt": 0, "cwd": "/work", **header}
    (directory / f"{session_id}.jsonl").write_text(
        "\n".join(
            [
                json.dumps({"type": "session/header", "header": record}),
                json.dumps({"type": "turn/start", "data": {"turn": 1}}),
                json.dumps(
                    {
                        "type": "user/message",
                        "data": {"content": [{"type": "text", "text": f"{title}\nmore"}]},
                    }
                ),
            ]
        )
        + "\n"
    )


def test_session_summaries_title_from_the_first_user_message(tmp_path: Path) -> None:
    _write_session(tmp_path, "s1", "fix the parser")
    [summary] = session_summaries(tmp_path)
    assert summary.session_id == "s1"
    assert summary.title == "fix the parser"
    assert summary.cwd == "/work"
    assert summary.parent is None


def test_session_summaries_skip_a_missing_directory(tmp_path: Path) -> None:
    assert session_summaries(tmp_path / "nowhere") == []


def test_forked_sessions_are_listed_under_their_parent(tmp_path: Path) -> None:
    """The header records the fork; the picker shows it as one."""
    _write_session(tmp_path, "root", "the original")
    _write_session(tmp_path, "branch", "a fork", parentSession="root")
    labels = [choice.label for choice in session_choices(tmp_path)]
    assert labels == ["the original", "  ↳ a fork"]


# ------------------------------------------------------------- credentials --


def test_credential_names_come_from_the_composed_rows() -> None:
    """Read from the configuration, not from a list kept in the front-end (I7).

    The key is nested inside a provider profile in the real rows, so the walk
    has to go all the way down rather than checking the row's top level.
    """
    rows = [
        {"id": "llm", "config": None},
        {"id": "llm-anthropic", "config": {"apiKeyEnv": "ANTHROPIC_API_KEY"}},
        {
            "id": "llm-openai-compatible",
            "config": {
                "profiles": [
                    {"provider": "deepseek", "apiKeyEnv": "DEEPSEEK_API_KEY"},
                    {"provider": "other", "apiKeyEnv": "OTHER_KEY"},
                ]
            },
        },
        {"id": "duplicate", "config": {"apiKeyEnv": "ANTHROPIC_API_KEY"}},
    ]
    assert credential_names(rows) == [
        "ANTHROPIC_API_KEY",
        "DEEPSEEK_API_KEY",
        "OTHER_KEY",
    ]


def test_credential_choices_mark_what_is_already_set(monkeypatch: pytest.MonkeyPatch) -> None:
    from ph.cordis import Context
    from ph.seams.credentials import CredentialService

    monkeypatch.setenv("PH_PRESENT_KEY", "x")
    credentials = CredentialService(ctx=Context())
    rows = [
        {"id": "a", "config": {"apiKeyEnv": "PH_PRESENT_KEY"}},
        {"id": "b", "config": {"apiKeyEnv": "PH_ABSENT_KEY"}},
    ]
    marked = {choice.value: choice.marked for choice in credential_choices(rows, credentials)}
    assert marked == {"PH_PRESENT_KEY": True, "PH_ABSENT_KEY": False}
