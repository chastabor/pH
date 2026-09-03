"""Themes, settings, completion, and session listing — the parts with no app.

Kept separate from the pilot tests because these are the pieces a broken
terminal must not be able to hide: a settings file that stops the TUI starting
is worse than a settings file pH ignores, and the only way to know which one
happens is to test it without a running app.

## Two measurements behind the session picker and the trajectory filter

**`_inherited_title` opens the ancestor rather than searching for it.** An ancestor
is a sibling inside the session's own family directory, so the path is a join.
Resolving each one by id instead scanned every family in the store: at **200
families whose newest rows were segment tips, 111.8 ms and 200 store scans against
18.6 ms — 6x**, running synchronously on the UI thread.

**`Query` is compiled once per rebuild, not per record.** `refresh_rows` calls the
predicate once per record, so parsing inside it re-lexed the query and re-parsed
every selector for every row on every keystroke — making the *plain free-text*
path, the common one, **2.5x** more expensive than before `type:` existed, because
the tag scan ran whether or not anyone had typed a tag. A malformed term also
constructed and raised a `SelectorError` per record: thousands of exceptions caught
and dropped.

**The segment-chain title cache.** A segmented run is one chain, so every row below
walks the same ancestors. Measured on a **60-segment chain: 550 opens and 39.9 ms
against 60 and 4.6 ms** — synchronously, on the UI thread.

**The trajectory filter coalesces keystrokes.** `DataTable` virtualizes *rendering*,
not row construction: a refill measures every cell of every row, so a
thousand-record log costs **~150 ms per keystroke and typing four characters cost
625 ms**. Row surgery is worse, because removing the rows that drop out reindexes
the table each time — so the fix is the frequency, not the primitive.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ph.persistence.jsonl import session_path
from ph.testing import write_reference_fork
from ph_app.sessions import session_summaries
from ph_app.tui.autocomplete import (
    PathCompleter,
    build_completion_state,
    parse_completion_token,
)
from ph_app.tui.config import (
    DEFAULT_THEME,
    load_tui_settings,
    save_tui_settings,
    tui_settings_from_json,
)
from ph_app.tui.modals.login import credential_choices, credential_names
from ph_app.tui.modals.pickers import session_choices
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


def test_a_binding_this_build_has_no_field_for_is_kept(tmp_path: Path) -> None:
    """A screen contributed by a row is rebound in `tui.json` like a built-in.

    Its binding id is the screen's, and this build has no field for it — so
    dropping the unrecognized key would make exactly one class of key
    unrebindable, which is the rule the settings file exists to prevent.
    """
    settings = tui_settings_from_json(
        {"keybindings": {"quit": "ctrl+q", "trajectory": "ctrl+j", "malformed": 7}}
    )
    keymap = settings.keybindings.as_map()

    assert keymap["quit"] == "ctrl+q"
    assert keymap["trajectory"] == "ctrl+j"
    assert "malformed" not in keymap, "only strings are keys"

    # And it survives a save, so a rebound plugin key is not lost on the next
    # toggle the app writes.
    save_tui_settings(tmp_path, settings)
    assert load_tui_settings(tmp_path).keybindings.as_map()["trajectory"] == "ctrl+j"


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
    family = str(header.get("family") or header.get("parentSession") or session_id)
    record = {"id": session_id, "createdAt": 0, "cwd": "/work", "family": family, **header}
    path = session_path(directory, session_id, family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
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


def test_a_reference_forked_child_takes_its_title_from_the_lineage(tmp_path: Path) -> None:
    """A child whose log begins mid-conversation still names itself.

    Its own file holds no `user/message` — that is what storing a reference
    means — so without the walk this row renders blank and a person picking a
    session is choosing between ids. `StoredSession`'s own comment refuses
    exactly that outcome, in the same words, for the listing one layer down.
    """
    _write_session(tmp_path, "root", "the original")
    write_reference_fork(tmp_path, "branch", "root", boundary=4)

    titles = {summary.session_id: summary.title for summary in session_summaries(tmp_path)}
    assert titles == {"root": "the original", "branch": "the original"}


def test_a_child_whose_ancestor_is_gone_still_gets_a_row(tmp_path: Path) -> None:
    """The reader refuses a broken chain; the picker must not.

    A person whose parent log was deleted needs to *see* the orphan in order to
    do anything about it, so a missing ancestor costs a title here rather than
    the row — and never an exception out of a directory scan.
    """
    write_reference_fork(tmp_path, "orphan", "deleted", boundary=4)

    [summary] = session_summaries(tmp_path)
    assert summary.session_id == "orphan"
    assert summary.title == ""
    assert summary.parent == "deleted", "and it still says what it is missing"


def test_a_segment_cycle_still_lists_every_session(tmp_path: Path) -> None:
    """A corrupt header must not empty the picker.

    Two segments naming each other have no chain head, so nothing represents
    them; hanging each under the other then puts both inside the children graph
    with no root above them, and the render walk — which starts from roots —
    emits nothing at all. Every row vanishes because two are broken, against
    this view's own rule that losing a session is worse than misplacing one.
    """
    write_reference_fork(tmp_path, "s0", "s1", boundary=4, kind="segment", family="s0")
    write_reference_fork(tmp_path, "s1", "s0", boundary=4, kind="segment", family="s0")

    assert sorted(choice.value for choice in session_choices(session_summaries(tmp_path))) == [
        "s0",
        "s1",
    ]


def test_forked_sessions_are_listed_under_their_parent(tmp_path: Path) -> None:
    """The header records the fork; the picker shows it as one."""
    _write_session(tmp_path, "root", "the original")
    _write_session(tmp_path, "branch", "a fork", parentSession="root")
    labels = [choice.label for choice in session_choices(session_summaries(tmp_path))]
    assert labels == ["the original", "  ↳ a fork"]


def test_a_rolled_session_is_one_row_with_no_indent(tmp_path: Path) -> None:
    """**A segment is the same conversation in a new file, so it is one row.**

    Structurally a roll is a fork at the tip, so before `kind` existed nothing on
    disk told them apart and this rendered as a staircase — three nested rows
    carrying one inherited title between them, for one conversation. The row that
    survives is the **tip**: the newest file, and the one a resume should open.
    """
    _write_session(tmp_path, "s0", "the conversation")
    write_reference_fork(tmp_path, "s1", "s0", boundary=4, kind="segment")
    write_reference_fork(tmp_path, "s2", "s1", boundary=8, kind="segment", family="s0")

    rows = session_choices(session_summaries(tmp_path))
    assert [choice.label for choice in rows] == ["the conversation"]
    assert [choice.value for choice in rows] == ["s2"], "the tip is what a resume opens"


def test_a_fork_still_indents_under_its_parent(tmp_path: Path) -> None:
    """The other half of the same rule: a branch *is* a second conversation."""
    _write_session(tmp_path, "s0", "the original")
    write_reference_fork(tmp_path, "b1", "s0", boundary=4, kind="fork")

    assert [choice.label for choice in session_choices(session_summaries(tmp_path))] == [
        "the original",
        "  ↳ the original",
    ]


def test_a_branch_of_a_rolled_session_lands_under_the_surviving_row(tmp_path: Path) -> None:
    """The case that makes contraction more than cosmetic.

    Forking an early segment leaves a child whose parent is a file the list no
    longer shows. Mapping it through the contraction puts it under the row that
    stands for that conversation; without that it would be orphaned to the root
    and read as an unrelated session.
    """
    _write_session(tmp_path, "s0", "the conversation")
    write_reference_fork(tmp_path, "s1", "s0", boundary=4, kind="segment")
    write_reference_fork(tmp_path, "b1", "s0", boundary=2, kind="fork")

    rows = session_choices(session_summaries(tmp_path))
    assert [(choice.value, choice.label) for choice in rows] == [
        ("s1", "the conversation"),
        ("b1", "  ↳ the conversation"),
    ]


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

    def held(name: str) -> bool:
        return credentials.has(credentials.reference(name))

    marked = {choice.value: choice.marked for choice in credential_choices(rows, held)}
    assert marked == {"PH_PRESENT_KEY": True, "PH_ABSENT_KEY": False}
