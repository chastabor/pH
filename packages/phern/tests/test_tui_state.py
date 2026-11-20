"""Settings, completion and session listing: the parts with no app.

Themes moved to `test_tui_themes.py` when they gained a second notation and a
profile file of their own (P9-01, P9-02). This file holds everything else that
can be asserted without a terminal.

These tests are kept separate from the pilot tests because these are the pieces
that a broken terminal must not be able to hide. A settings file that stops the
TUI from starting is worse than a settings file that pH ignores. The only way to
know which one happens is to test it without a running app.

## Four measurements behind the session picker and the trajectory filter

**`_inherited_title` opens the ancestor and does not search for it.** An
ancestor is a sibling inside the family directory of the session itself, so the
path is a join. To resolve each one by id scanned every family in the store. At
**200 families whose newest rows were segment tips, that was 111.8 ms and 200
store scans against 18.6 ms, which is 6x**, and it ran synchronously on the UI
thread.

**`Query` is compiled one time for each rebuild, and not for each record.**
`refresh_rows` calls the predicate one time for each record. Parsing inside the
predicate therefore lexed the query again and parsed every selector again, for
every row, on every keystroke. That made the *plain free-text* path, which is
the common one, **2.5x** more expensive than before `type:` existed, because the
tag scan ran whether or not anyone had typed a tag. A malformed term also
constructed and raised a `SelectorError` for each record: thousands of
exceptions that were caught and dropped.

**The segment-chain title cache.** A segmented run is one chain, so every row
below walks the same ancestors. Measured on a **60-segment chain: 550 opens and
39.9 ms against 60 and 4.6 ms**, synchronously, on the UI thread.

**The trajectory filter coalesces keystrokes.** `DataTable` virtualizes
*rendering* and not row construction. A refill measures every cell of every row,
so a thousand-record log costs **~150 ms for each keystroke, and typing four
characters cost 625 ms**. Row surgery is worse, because removing the rows that
drop out reindexes the table each time. So the fix is the frequency and not the
primitive.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import ph_app.sessions as S
from ph.persistence.jsonl import session_path
from ph.seams.models import ModelEntry, ModelRoute
from ph.session import SessionHeader
from ph.testing import write_reference_fork, write_stored_log
from ph_app.sessions import SessionSummary, session_summaries
from ph_app.tui.autocomplete import (
    PathCompleter,
    build_completion_state,
    parse_completion_token,
)
from ph_app.tui.config import (
    TuiSettings,
    load_tui_settings,
    save_tui_settings,
    tui_settings_from_json,
)
from ph_app.tui.modals.login import credential_choices
from ph_app.tui.modals.pickers import model_choices, session_choices
from ph_app.tui.state import ChatItem, PromptRecord, TuiState

# ------------------------------------------------------------------- settings --


def test_settings_survive_an_unreadable_file(tmp_path: Path) -> None:
    (tmp_path / "tui.json").write_text("{ not json")
    settings = load_tui_settings(tmp_path)
    assert settings == TuiSettings(), "defaults, rather than a refusal to start"


def test_an_unknown_key_in_settings_is_ignored() -> None:
    """Including `theme`, which this file wrote until P9-02 moved it.

    The tolerance rule *is* the upgrade path. A key that is left over from a
    build that had the field is read past, and it is gone the next time anything
    writes the file.
    """
    settings = tui_settings_from_json(
        {"theme": "ph-light", "somethingFromANewerPH": 3, "keybindings": {"quit": "ctrl+d"}}
    )
    assert settings == TuiSettings(keybindings=settings.keybindings)
    assert settings.keybindings.quit == "ctrl+d"
    # A binding that is not named keeps its default and does not become unset.
    assert settings.keybindings.cancel


def test_a_binding_this_build_has_no_field_for_is_kept(tmp_path: Path) -> None:
    """A screen that a row contributes is rebound in `tui.json` like a built-in one.

    The id of its binding is the id of the screen, and this build has no field
    for it. To drop the unrecognized key would therefore make exactly one class
    of key impossible to rebind, and the settings file exists to prevent that.
    """
    settings = tui_settings_from_json(
        {"keybindings": {"quit": "ctrl+q", "trajectory": "ctrl+j", "malformed": 7}}
    )
    keymap = settings.keybindings.as_map()

    assert keymap["quit"] == "ctrl+q"
    assert keymap["trajectory"] == "ctrl+j"
    assert "malformed" not in keymap, "only strings are keys"

    # The binding also survives a save, so a rebound plugin key is not lost the
    # next time the app writes a toggle.
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
    # The family comes from the model, which tags it with the cwd since format 1.
    # A fixture that wrote `family = session_id` would therefore write into a
    # directory that no filtered listing looks in.
    record = {"id": session_id, "createdAt": 0, "cwd": "/work", **header}
    if record.get("cwd") is None:
        record.pop("cwd", None)
    if "family" not in record and record.get("parentSession"):
        record["family"] = str(record["parentSession"])
    family = SessionHeader.model_validate(record).family
    record["family"] = family
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


def test_browse_filtered_by_cwd_finds_a_session_older_than_the_limit(tmp_path: Path) -> None:
    """This is why the filter is a parameter of the fold and not a comprehension.

    `limit` takes the newest rows. A filter on what comes *back* would let the
    session of this directory fall out of that window behind newer work
    elsewhere. A repo that somebody last touched a month ago would then list as
    empty on a busy machine.

    Sabotage: filter the result instead of the scan — `[s for s in
    session_summaries(d) if s.cwd == "/work"]` — and this finds nothing.
    """
    _write_session(tmp_path, "wanted", "the one worked on long ago", cwd="/work")
    for index in range(60):
        _write_session(tmp_path, f"other{index}", "newer work elsewhere", cwd="/elsewhere")
    # The mtimes are explicit. The writes above are fast enough to land in the
    # same tick, and this fold sorts on mtime. "wanted" is the oldest by a clear
    # margin.
    for path in tmp_path.rglob("*.jsonl"):
        os.utime(path, (0, 1_000) if path.stem == "wanted" else (0, 2_000_000))

    found = session_summaries(tmp_path, cwd="/work")
    assert [summary.session_id for summary in found] == ["wanted"]
    # It is also past the window that an unfiltered listing shows.
    assert "wanted" not in {summary.session_id for summary in session_summaries(tmp_path)}


def test_a_filtered_listing_opens_no_other_directorys_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The purpose of the tag on the lineage directory (format 1).

    Before the tag, the answer to "which sessions belong to this repo" needed
    the header of every log in the store to be opened. Now the tag on the
    *directory name* gives the answer, so the files of another repo are never
    read. The test counts the opens, because the whole value of the change is
    the reads that do not happen.

    Sabotage: drop the `tag=` argument in `session_summaries` and every log is
    opened again.
    """
    _write_session(tmp_path, "ours", "here", cwd="/work")
    for index in range(20):
        _write_session(tmp_path, f"theirs{index}", "elsewhere", cwd=f"/other/{index}")

    opened: list[str] = []
    real = S._summarize

    def counting(
        path: Path, modified: float | None, size: int | None, *, cwd: str | None = None
    ) -> SessionSummary | None:
        opened.append(path.stem)
        return real(path, modified, size, cwd=cwd)

    monkeypatch.setattr(S, "_summarize", counting)
    found = session_summaries(tmp_path, cwd="/work")

    assert [summary.session_id for summary in found] == ["ours"]
    assert opened == ["ours"], "no other directory's log was opened"


def test_a_session_with_no_cwd_belongs_to_no_directory(tmp_path: Path) -> None:
    """An unfiltered listing finds a lineage with no tag, and no filter finds it.

    This is the correct answer, and a placeholder tag is not. A session that
    never had a working directory should not appear in a search for one.
    """
    _write_session(tmp_path, "rootless", "no cwd", cwd=None)
    assert [s.session_id for s in session_summaries(tmp_path)] == ["rootless"]
    assert session_summaries(tmp_path, cwd="/work") == []


def test_an_unfiltered_browse_is_unchanged(tmp_path: Path) -> None:
    """`cwd=None` is every session, which is what `/sessions` asks for."""
    _write_session(tmp_path, "here", "one", cwd="/work")
    _write_session(tmp_path, "there", "two", cwd="/elsewhere")
    assert {summary.session_id for summary in session_summaries(tmp_path)} == {"here", "there"}
    assert {summary.session_id for summary in session_summaries(tmp_path, cwd=None)} == {
        "here",
        "there",
    }


def test_an_empty_cwd_is_a_filter_and_not_the_absence_of_one(tmp_path: Path) -> None:
    """`None` is "no filter". An empty string is a directory like any other.

    No session recorded an empty working directory, so the filter finds nothing.
    The daemon turns the `""` of the wire into `None` before it asks
    (`_sessions_browse`), so a client that sends `""` still gets every session.

    Sabotage: add `cwd = cwd or None` as the first line of `session_summaries`,
    and the empty string lists both sessions.
    """
    _write_session(tmp_path, "here", "one", cwd="/work")
    _write_session(tmp_path, "rootless", "no cwd", cwd=None)

    assert session_summaries(tmp_path, cwd="") == []


def test_the_limit_counts_matching_sessions(tmp_path: Path) -> None:
    """Filtering first means `limit` bounds the answer, not the scan."""
    for index in range(4):
        _write_session(tmp_path, f"mine{index}", "ours", cwd="/work")
        _write_session(tmp_path, f"theirs{index}", "not ours", cwd="/elsewhere")
    found = session_summaries(tmp_path, limit=3, cwd="/work")
    assert len(found) == 3
    assert all(summary.cwd == "/work" for summary in found)


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
    """A child whose log begins in the middle of a conversation still has a title.

    Its own file holds no `user/message`, which is what storing a reference
    means. Without the walk, this row renders blank, and a person who picks a
    session chooses between ids. The comment of `StoredSession` refuses exactly
    that outcome, in the same words, for the listing one layer down.
    """
    _write_session(tmp_path, "root", "the original")
    write_reference_fork(tmp_path, "branch", "root", boundary=4)

    titles = {summary.session_id: summary.title for summary in session_summaries(tmp_path)}
    assert titles == {"root": "the original", "branch": "the original"}


def test_an_ancestor_is_read_for_its_title_and_is_given_no_row(tmp_path: Path) -> None:
    """The walk reads the top of an ancestor and states no file facts for it.

    A row carries the time and the size of its file, and the walk takes no
    `stat`. An ancestor was once read with a time of `0.0` and a size of `0`,
    which a real file can have. Now its time and size are `None`, and the walk
    answers `None` when no ancestor has a title.

    Sabotage: read the ancestor with `_summarize(path, 0.0, 0)` in
    `_inherited_title`, and `known` holds a time that no `stat` backs.
    """
    # No cwd, so the family of the root is its bare id, which is where the fork
    # helper writes the branch.
    _write_session(tmp_path, "root", "the original", cwd=None)
    write_reference_fork(tmp_path, "branch", "root", boundary=4)
    rows = {summary.session_id: summary for summary in session_summaries(tmp_path)}
    branch = rows["branch"]

    known: dict[str, SessionSummary] = {}
    assert S._inherited_title(tmp_path, branch, known) == "the original"
    above = known["root"]
    assert (above.modified, above.size) == (None, None), "no stat backs an ancestor"
    assert (above.title, above.parent, above.cwd) == ("the original", None, None)

    orphan = branch.model_copy(update={"parent": "deleted"})
    assert S._inherited_title(tmp_path, orphan, {}) is None


def test_a_child_whose_ancestor_is_gone_still_gets_a_row(tmp_path: Path) -> None:
    """The reader refuses a broken chain, and the picker must not.

    A person whose parent log was deleted needs to *see* the orphan before they
    can do anything about it. So a missing ancestor costs a title here and not
    the row, and it never raises an exception out of a directory scan.
    """
    write_reference_fork(tmp_path, "orphan", "deleted", boundary=4)

    [summary] = session_summaries(tmp_path)
    assert summary.session_id == "orphan"
    assert summary.title is None
    assert summary.parent == "deleted", "and it still says what it is missing"


def test_a_segment_cycle_still_lists_every_session(tmp_path: Path) -> None:
    """A corrupt header must not empty the picker.

    Two segments that name each other have no chain head, so nothing represents
    them. Hanging each under the other then puts both inside the children graph
    with no root above them. The render walk starts from roots, so it emits
    nothing at all. Every row vanishes because two are broken. That is against
    the rule of this view itself, that losing a session is worse than
    misplacing one.
    """
    write_reference_fork(tmp_path, "s0", "s1", boundary=4, kind="segment", family="s0")
    write_reference_fork(tmp_path, "s1", "s0", boundary=4, kind="segment", family="s0")

    assert sorted(choice.value for choice in session_choices(session_summaries(tmp_path))) == [
        "s0",
        "s1",
    ]


def test_forked_sessions_are_listed_under_their_parent(tmp_path: Path) -> None:
    """The header records the fork, and the picker shows it as one."""
    _write_session(tmp_path, "root", "the original")
    _write_session(tmp_path, "branch", "a fork", parentSession="root")
    labels = [choice.label for choice in session_choices(session_summaries(tmp_path))]
    assert labels == ["the original", "  ↳ a fork"]


def test_a_rolled_session_is_one_row_with_no_indent(tmp_path: Path) -> None:
    """**A segment is the same conversation in a new file, so it is one row.**

    Structurally a roll is a fork at the tip. Before `kind` existed, nothing on
    disk told them apart, and this rendered as a staircase: three nested rows
    with one inherited title between them, for one conversation. The row that
    remains is the **tip**: the newest file, and the one that a resume should
    open.
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

    A fork of an early segment leaves a child whose parent is a file that the
    list no longer shows. Mapping the child through the contraction puts it
    under the row that stands for that conversation. Without the mapping it
    would be orphaned to the root and read as an unrelated session.
    """
    _write_session(tmp_path, "s0", "the conversation")
    write_reference_fork(tmp_path, "s1", "s0", boundary=4, kind="segment")
    write_reference_fork(tmp_path, "b1", "s0", boundary=2, kind="fork")

    rows = session_choices(session_summaries(tmp_path))
    assert [(choice.value, choice.label) for choice in rows] == [
        ("s1", "the conversation"),
        ("b1", "  ↳ the conversation"),
    ]


def test_the_picker_shows_a_child_as_a_child(tmp_path: Path) -> None:
    """**P11-08: the log of a sub-agent is listed under its parent, and is not a
    way in.**

    Its header names the agent that spawned it in `parentSession`, which is the
    same link that the header of a fork has. So the picker drew it as a fork: a
    second conversation that a person could carry on. Choosing it attached it as
    a root, which put a second writer on a log that the mount of its root
    writes. The `origin` of the header is what tells the two apart, so it is
    carried on the summary. The row says which one it is: nested under its
    parent, marked as a sub-agent, and refused with the way to read it instead.

    Sabotage: draw a sub-agent's row as a fork's, and it takes the fork's arrow and
    answers with its id.
    """
    _write_session(tmp_path, "lead", "the original")
    _write_session(tmp_path, "branch", "a fork", parentSession="lead", kind="fork")
    _write_session(
        tmp_path,
        "helper",
        "the delegated task",
        parentSession="lead",
        origin="subagent",
        delegationDepth=1,
    )

    summaries = session_summaries(tmp_path)
    assert {one.session_id: one.origin for one in summaries} == {
        "lead": None,
        "branch": None,
        "helper": "subagent",
    }
    rows = session_choices(summaries)
    assert rows[0].value == "lead", "both hang under the root they came from"
    by_id = {row.value: row for row in rows}
    assert by_id["branch"].label == "  ↳ a fork"
    assert by_id["helper"].label == "  ⤷ the delegated task", "a child is not drawn as a fork"
    assert by_id["helper"].detail.startswith("sub-agent · ")
    # Seen, not taken: a fork is a session of its own, a child is not.
    assert by_id["branch"].unavailable == ""
    assert "sub-agent of lead" in by_id["helper"].unavailable
    assert "phern --mode trajectory --session helper" in by_id["helper"].unavailable


# ------------------------------------------------------------- credentials --


def test_credential_choices_mark_what_is_already_set() -> None:
    """The rows come directly from the answer of the daemon, in the order that it
    gave them.

    The daemon now does the walk that used to produce these names
    (`credentials_of`). What is left here is the rendering and the order. The
    order is the row order of the profile that the root is mounted on, and it is
    the reason why the answer is a mapping and not a set.
    """
    choices = credential_choices({"PH_PRESENT_KEY": True, "PH_ABSENT_KEY": False})

    assert [one.value for one in choices] == ["PH_PRESENT_KEY", "PH_ABSENT_KEY"]
    assert [one.marked for one in choices] == [True, False]
    assert [one.detail for one in choices] == ["set", "not set"]


def test_a_row_with_no_log_position_is_not_in_the_prompt_history() -> None:
    """A history record is a position in the log, and the picker reveals it.

    `ChatItem.seq` is `None` for a row that the log has no position for. Such a
    row was listed with the seq `-1`. The picker keys its rows by seq, so two
    such rows shared one key, and the transcript could not reveal either. The
    history now holds only rows that the log has.

    Sabotage: drop `item.seq is None` from `prompt_history`, and the type check
    refuses the record before this test runs.
    """
    state = TuiState()
    state.add(ChatItem(key="u1", role="user", text="first", seq=3, turn=1))
    state.add(ChatItem(key="u2", role="user", text="not in the log", turn=2))

    assert state.prompt_history() == [PromptRecord(text="first", seq=3, turn=1)]


# ------------------------------------------- what is absent, in the pickers --


def test_a_stored_root_with_nothing_typed_states_no_title_and_no_kind(tmp_path: Path) -> None:
    """A root whose log holds only its header has no title, and a root has no
    kind.

    The row leaves both out of its frame. A picker then draws the session id
    for the title by a test for `None`.

    Sabotage: pass `title or ""` in `_summarize`, and the row carries a title
    of `""`.
    """
    header = SessionHeader(id="bare", created_at=0, cwd="/work")
    write_stored_log(tmp_path, header, [])
    family = header.family

    [summary] = session_summaries(tmp_path)

    assert (summary.title, summary.kind, summary.parent) == (None, None, None)
    wire = summary.to_wire()
    assert "title" not in wire and "kind" not in wire
    assert (wire["cwd"], wire["family"]) == ("/work", family)
    [row] = session_choices([summary])
    assert row.label == "bare"


def test_a_live_row_with_no_file_shows_its_state() -> None:
    """A live root that the disk has not seen has no time and no size to show.

    Its row shows where the root is, which is what a person chooses on. The row
    used to show the day in 1970 and the size of zero that stood in for the
    file.

    Sabotage: test only `summary.state == "running"` in `_session_row`, and the
    row is built from a time and a size that are not there.
    """
    live = SessionSummary(session_id="live", state="idle", cwd="/repos/thing")

    [row] = session_choices([live])

    assert (row.label, row.detail) == ("live", "idle · /repos/thing")
    assert live.when is None


def test_a_route_with_no_listed_name_gets_a_row_of_its_own() -> None:
    """The model picker marks the route in force.

    A route that the profile does not list has no key, so no listed row can be
    marked for it. It gets a row of its own at the top. A listed route does not,
    and neither does a session with no route.

    Sabotage: drop the `active_key is None` test from `model_choices`, and a
    listed route in force is listed twice.
    """
    listed = [ModelEntry(key="main", route=ModelRoute(provider="fake", model="fake-1"))]

    unlisted = model_choices(listed, None, "fake/fake-9")
    assert [(row.value, row.marked) for row in unlisted] == [
        ("fake/fake-9", True),
        ("main", False),
    ]
    assert [(row.value, row.marked) for row in model_choices(listed, "main", "fake/fake-1")] == [
        ("main", True)
    ]
    assert [(row.value, row.marked) for row in model_choices(listed, None, None)] == [
        ("main", False)
    ]
