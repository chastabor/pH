"""The auditor's view (P3-25), driven as a person drives it.

The row's gates, one test each: *opens a stored log with nothing mounted;
search finds a tool call by name and by argument; fork offered only at a
closed-turn boundary and refused elsewhere; the fork's prefix is byte-identical
to the source; pilot tests drive the table, the details panel and the fork.*

The first is the one the whole shape was chosen for. A view that could only show
the session you are already in reaches none of the logs an auditor actually
wants — the crashed run, the child's transcript, the fixture from a replay
report — so "nothing mounted" is asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from textual.coordinate import Coordinate
from textual.widgets import DataTable
from tui_helpers import until

from ph.cordis import Context
from ph.persistence.jsonl import session_path
from ph.session import Session, SurfaceIntent
from ph.testing import assistant_payload, user_payload
from ph_app.tui.trajectory import build_trajectory
from ph_app.tui.trajectory_app import TrajectoryApp, load_records
from ph_app.tui.widgets.trajectory import FORK_MARK, Query, search_index

pytestmark = pytest.mark.anyio


def _log(session: Session) -> Session:
    """One turn with a tool call, so there is something to search and fork at."""
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user_payload("read a.py"), SurfaceIntent("append"))
    session.append("step/start", {"turn": 1, "step": 0})
    session.append("assistant/message", assistant_payload("reading", "a1"), SurfaceIntent("append"))
    session.append(
        "tool/call", {"callId": "c1", "name": "read", "arguments": '{"path": "src/a.py"}'}
    )
    session.append("step/end", {"turn": 1, "step": 0})
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    return session


async def _stored(ctx: Context, session_id: str = "audit") -> Path:
    """Write a real log to disk through the persistence row."""
    session = _log(ctx.sessions.create(session_id))
    await ctx.sessions.flush(session)
    return session_path(ctx.session_persistence.root, session.id)


# ----------------------------------------------------- nothing mounted --


async def test_it_opens_a_stored_log_with_nothing_mounted(mount: Any) -> None:
    """The gate the shape exists for.

    The records are read from a *file* — no context, no agent, no provider, no
    answerers — which is what makes a crashed run and a subagent's log readable
    at all. The mounted harness here only writes the log; nothing of it is
    passed to the view.
    """
    ctx: Context = await mount()
    path = await _stored(ctx)
    await ctx.dispose()

    session_id, records = load_records(str(path))

    assert session_id == "audit"
    assert [record.kind for record in records] == [
        "event",
        "user",
        "message",
        "tool",
        "event",
        "event",  # the seed boundary a stored log records
    ]
    app = TrajectoryApp(records, session_id=session_id)
    assert app.trajectory.sessions is None, "a file-backed view has no store, and says so"


def test_a_missing_log_is_named_as_the_person_typed_it(tmp_path: Path) -> None:
    """A path that does not exist is reported as *that path*.

    Falling back to "maybe it is a session id" appended `.jsonl` to what was
    already a filename and reported `/nope.jsonl.jsonl` missing — a file nobody
    named.
    """
    with pytest.raises(FileNotFoundError, match=r"/nope\.jsonl$"):
        load_records("/nope.jsonl")
    with pytest.raises(FileNotFoundError, match=r"absent\.jsonl$"):
        load_records("absent", home=tmp_path)


async def test_a_log_is_readable_by_id_or_by_path(mount: Any, tmp_path: Path) -> None:
    """By path, so a fixture or a copy someone sent is readable without being
    installed into `$PH_HOME` first."""
    ctx: Context = await mount()
    path = await _stored(ctx)

    by_path, _ = load_records(str(path))
    by_id, _ = load_records("audit", home=Path(ctx.session_persistence.root).parent)
    assert by_path == by_id == "audit"


# ------------------------------------------------------------- search --


def test_search_finds_a_tool_call_by_name_and_by_argument() -> None:
    """The row's gate. Both, because a name alone would not find the call you
    remember by the file it touched."""
    records = build_trajectory(_log(Session("search")))
    (tool,) = [record for record in records if record.kind == "tool"]
    index = search_index(tool)

    assert Query.compile("read").matches(tool, index)
    assert Query.compile("src/a.py").matches(tool, index)
    assert Query.compile("read src/a.py").matches(tool, index), "every term, not any"
    assert not Query.compile("write").matches(tool, index)


def test_search_covers_the_source_not_only_the_text() -> None:
    """ "Which records came from this plugin" is the question the transcript
    cannot answer, and the reason the index carries attribution."""
    records = build_trajectory(_log(Session("sources")))
    (message,) = [record for record in records if record.kind == "message"]

    assert Query.compile("model").matches(message, search_index(message))


# --------------------------------------------------------------- fork --


def test_a_fork_is_marked_only_at_a_closed_turn() -> None:
    """A6. The table *shows* where the action is available rather than letting a
    person aim anywhere and be refused after the fact."""
    records = build_trajectory(_log(Session("fork")))
    marked = [record for record in records if record.fork_point]

    assert [record.title for record in marked] == ["turn end"]


async def test_the_fork_key_refuses_off_a_boundary_and_says_why(mount: Any) -> None:
    """A key that silently does nothing is worse than one that is not offered."""
    ctx: Context = await mount()
    session = _log(ctx.sessions.create("forking"))
    records = build_trajectory(session)
    app = TrajectoryApp(records, session_id=session.id, sessions=ctx.sessions)

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.trajectory.panel
        # The cursor starts on record #1 — a `turn start`, not a boundary.
        assert panel.selected() is not None
        assert not panel.selected().fork_point
        await pilot.press("f")
        # The store's own words, with its own code — this view no longer states
        # A6 in parallel with the layer that enforces it.
        assert "open turn" in app.trajectory.notice
        assert f"#{app.trajectory.panel.selected().index}" in app.trajectory.notice


async def test_a_fork_at_a_boundary_produces_a_byte_identical_prefix(mount: Any) -> None:
    """The row's gate, and the reason forking is the view's headline action.

    Driven through the key, so what is tested is the action a person performs.
    """
    ctx: Context = await mount()
    session = _log(ctx.sessions.create("source"))
    records = build_trajectory(session)
    boundary = next(record for record in records if record.fork_point)
    app = TrajectoryApp(records, session_id=session.id, sessions=ctx.sessions)

    child = None
    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.trajectory.panel.query_one("#trajectory-table")
        table.move_cursor(row=app.trajectory.panel.visible_records.index(boundary))
        await pilot.pause()
        assert app.trajectory.panel.selected() is boundary
        # Through the action the key invokes, and taking the *session* back:
        # recovering the child by string-parsing a notification made the
        # user-facing wording part of the test's contract.
        child = app.trajectory.action_fork()

    assert child is not None
    assert "forked at" in app.trajectory.notice

    # Byte-identical, through the wire form the log is written in.
    source_prefix = [
        event.to_wire(thaw=False) for event in session.events[: boundary.source_seq + 1]
    ]
    child_prefix = [event.to_wire(thaw=False) for event in child.events[: len(source_prefix)]]
    assert child_prefix == source_prefix


async def test_forking_without_a_harness_says_so() -> None:
    """Advertising a key that cannot work is the failure the plan names, so the
    file-backed view reports rather than pretending."""
    records = build_trajectory(_log(Session("fileonly")))
    app = TrajectoryApp(records, session_id="fileonly")

    async with app.run_test() as pilot:
        await pilot.pause()
        boundary = next(record for record in records if record.fork_point)
        app.trajectory.panel.query_one("#trajectory-table").move_cursor(
            row=app.trajectory.panel.visible_records.index(boundary)
        )
        await pilot.pause()
        await pilot.press("f")

    assert "needs a running harness" in app.trajectory.notice


# ------------------------------------------------- the table and panel --


async def test_the_table_and_the_details_panel_follow_the_cursor() -> None:
    """The pilot gate: the two halves a person actually reads."""
    records = build_trajectory(_log(Session("panel")))
    app = TrajectoryApp(records, session_id="panel")

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.trajectory.panel
        assert panel.query_one("#trajectory-table").row_count == len(records)
        # The header states what the log holds, including where forks may aim.
        header = app.query_one("#trajectory-header").render()
        assert "1 fork point(s)" in str(header)

        tool = next(record for record in records if record.kind == "tool")
        panel.query_one("#trajectory-table").move_cursor(row=panel.visible_records.index(tool))
        await pilot.pause()
        body = str(panel.query_one(".details-body").render())
        assert "src/a.py" in body, "the panel did not follow the cursor"


async def test_filtering_narrows_the_table_and_keeps_a_selection() -> None:
    records = build_trajectory(_log(Session("filter")))
    app = TrajectoryApp(records, session_id="filter")

    async with app.run_test() as pilot:
        await pilot.pause()
        panel = app.trajectory.panel
        panel.query_one("#trajectory-filter").value = "read"
        await until(pilot, lambda: len(panel.visible_records) < len(records))

        assert all("read" in search_index(record) for record in panel.visible_records)
        assert panel.selected() is not None, "a narrowed table still has a selection"

        panel.query_one("#trajectory-filter").value = "nothing-matches-this"
        await until(pilot, lambda: not panel.visible_records)
        assert panel.selected() is None


async def test_the_fork_mark_reaches_the_table(mount: Any) -> None:
    """The mark is the affordance; without it the key is a guess.

    Asserted on the rendered cell rather than on the record, because the first
    version checked that a module constant was non-empty and would have stayed
    green if `refresh_rows` stopped emitting it.
    """
    session = _log(Session("mark"))
    records = build_trajectory(session)
    app = TrajectoryApp(records, session_id="mark")

    async with app.run_test() as pilot:
        await pilot.pause()
        table = app.trajectory.panel.query_one("#trajectory-table", DataTable)
        marked = {
            str(table.get_cell_at(Coordinate(row, 0)))
            for row, record in enumerate(app.trajectory.panel.visible_records)
            if record.fork_point
        }
        unmarked = {
            str(table.get_cell_at(Coordinate(row, 0)))
            for row, record in enumerate(app.trajectory.panel.visible_records)
            if not record.fork_point
        }

    assert marked and all(cell.startswith(FORK_MARK) for cell in marked)
    assert not any(cell.startswith(FORK_MARK) for cell in unmarked)


# ------------------------------------------------------- P6-33: type filter --


def test_every_record_carries_the_log_type_it_came_from() -> None:
    """`kind` is the view's eight-value vocabulary; `type` is the log's 63-value
    one, and forty-odd harness types share the single kind `event`. A reader
    wanting "just the retained worktrees" can only ask in `type`."""
    records = build_trajectory(_log(Session("typed")))

    assert all(record.type for record in records), "a record with no type cannot be filtered"
    by_kind = {record.kind for record in records}
    by_type = {record.type for record in records}
    assert len(by_type) >= len(by_kind), "the log vocabulary is the finer of the two"


def test_a_type_term_selects_a_namespace_and_free_text_still_works() -> None:
    """Two predicates because there are two questions: "mentions read" is a
    substring search; "is a tool record" is a namespace, and asking for one by
    substring is how `tool` comes to select `tools/`."""
    records = build_trajectory(_log(Session("filter")))
    picked = [
        record
        for record in records
        if Query.compile("type:tool").matches(record, search_index(record))
    ]

    assert picked, "the fixture has tool records"
    assert all(record.type.startswith("tool/") for record in picked)
    # Free text is unchanged, and the two compose.
    both = [
        record
        for record in records
        if Query.compile("type:tool read").matches(record, search_index(record))
    ]
    assert both and all(record.type.startswith("tool/") for record in both)


def test_several_type_terms_union_rather_than_intersect() -> None:
    """Two namespaces mean *either*, matching `--type` on the command line.

    A record cannot be in two namespaces at once, so intersecting them would
    always answer empty — which is what the first version did, and what nothing
    here noticed: the mutation that restores the intersection was the one case in
    twenty-one that no test caught. Free text still ANDs, and the type half is
    ANDed with it; only the type terms among themselves are a union.
    """
    records = build_trajectory(_log(Session("union")))
    picked = [
        record
        for record in records
        if Query.compile("type:tool type:user").matches(record, search_index(record))
    ]

    roots = {record.type.split("/")[0] for record in picked}
    assert roots == {"tool", "user"}, "either namespace, not both at once"
    assert len(picked) > 1, "an intersection would have answered with nothing"

    # Still ANDed with free text: `src` is in the tool call's arguments and not
    # in the user message, so the union narrows to one.
    narrowed = [
        record
        for record in records
        if Query.compile("type:tool type:user src").matches(record, search_index(record))
    ]
    assert [record.type for record in narrowed] == ["tool/call"]


def test_a_half_typed_or_foreign_type_term_matches_nothing_rather_than_raising() -> None:
    """This runs per keystroke: `type:w` on the way to `type:workspace` must not
    throw. The refusal that explains itself belongs on a command line, where the
    input is complete."""
    records = build_trajectory(_log(Session("typing")))
    record = records[0]
    index = search_index(record)

    assert Query.compile("type:").matches(record, index) is False, "incomplete, not a raise"
    assert Query.compile("type:bus:tools").matches(record, index) is False, "wrong vocabulary"
    # A compiled query is reusable across every record — that is the point.
    compiled = Query.compile("type:turn")
    assert [r.type for r in records if compiled.matches(r, search_index(r))]


def test_a_handler_that_adds_nothing_does_not_stamp_a_previous_record() -> None:
    """`on_request_header` returns without adding when the payload will not parse.

    The fold used to write `records[-1].fork_point` unconditionally, so that
    event's fork status landed on an unrelated earlier record — and would have
    raised `IndexError` had it been the first record-producing event. The fix is
    a slice of what the handler actually added, which is correct for zero, one
    and many.
    """
    session = Session("broken-header")
    session.append("turn/start", {"turn": 1})
    # A `request/header` whose payload `parse_request_header` refuses.
    session.append("request/header", {"header": {"nonsense": True}, "reason": "initial"})
    records = build_trajectory(session)

    assert [record.type for record in records] == ["turn/start"], "the bad header made no record"
    assert records[0].type == "turn/start", "and did not restamp the turn record"


def test_a_leading_unparsable_header_does_not_crash_the_fold() -> None:
    """The `IndexError` case: nothing has been added yet when the handler bails."""
    session = Session("header-first")
    session.append("request/header", {"header": {"nonsense": True}, "reason": "initial"})

    assert build_trajectory(session) == []
