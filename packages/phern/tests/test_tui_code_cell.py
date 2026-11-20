"""The code cell and the subagent panel (P3-19).

Both consume records that existed before they did. `CellDetails` was written at
P3-09 with nothing to draw it. The records of the children were in the
`RECORDLESS` set of the adapter, which named *this* row as the reason why they
were classified and not rendered.

The claim under both is the claim of P2-01, one layer down: everything drawn
comes from the settled record. So a replayed cell is the cell that ran, and the
panel of a resumed session is the family that the parent left. Each session owns
its log (Phase 11), so the daemon reads that family from the log of each child
and sends it as `session.children` (P11-09).

## Why the cell card does not print the dispatch count

The collapsible below the card already reports the count, from the
`tool/code-dispatch-start` fold that owns those rows. Two projections of one
number in one widget can disagree, which A11 forbids. The first draft of this
card did exactly that: a snapshot showed **"3 governed calls" over a section
titled "1 governed call"**.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import cast

import pytest
from daemon_helpers import spawned, supervised, until
from tui_helpers import StubClient, StubHost

from ph.seams.subagents import ADMITTED, DELETED, STATUS
from ph.session import Session, SessionHeader, SurfaceIntent
from ph.session.kinds import SESSION_HOLDER, credential_hold
from ph.testing import assistant_payload, log_event
from ph_app.daemon.client import DaemonClient
from ph_app.daemon.projections import family_of
from ph_app.payloads import ChildRow, SessionChildrenNotice
from ph_app.tui.adapter import TuiEventAdapter
from ph_app.tui.remote import DaemonSession
from ph_app.tui.state import ChatItem, ToolCard, TuiState
from ph_app.tui.widgets.status import (
    NO_WORK_SEEN,
    _todo_line,
    children_heading,
    render_subagents,
    shorten_path,
)
from ph_app.tui.widgets.transcript import (
    CodeCellWidget,
    ToolCardWidget,
    TranscriptView,
    _cell_facts,
)

pytestmark = pytest.mark.anyio


def cell() -> ToolCard:
    return ToolCard(
        call_id="c1",
        name="python",
        arguments='{"program": "x = 1\\nprint(x)"}',
        title="python",
        subtitle="2 lines",
        card="terminal",
        input_text="x = 1\nprint(x)",
        settled=True,
        body="1",
    )


# ------------------------------------------------------------- the facts --


def test_only_facts_that_are_true_are_shown() -> None:
    """A cell that dispatched nothing and truncated nothing has nothing to
    report. A line of `0 governed calls · not truncated` would report it
    anyway."""
    assert _cell_facts({}) == ""
    assert _cell_facts({"dispatches": 0, "truncated": False, "reset": False}) == ""


def test_the_facts_line_carries_what_the_collapsible_does_not() -> None:
    facts = _cell_facts({"dispatches": 40, "attachments": 2, "truncated": True, "reset": True})
    assert "2 attachments" in facts
    assert "output truncated" in facts
    assert "kernel restarted" in facts
    # The dispatch *count* belongs to the collapsible, from the fold that owns
    # those rows. Rendering it here too put two projections of one number in
    # one widget, and they could disagree (A11).
    assert "40" not in facts


def test_one_of_something_is_not_pluralized() -> None:
    assert _cell_facts({"attachments": 1}) == "1 attachment"
    assert _cell_facts({"attachments": 2}) == "2 attachments"


def test_a_field_this_build_does_not_know_is_ignored() -> None:
    """The details are read as a mapping, and not as the model of ph-rlm. The
    module `ph_app` does not import the bundle, so a tool can enrich its own
    card without the transcript learning its schema."""
    assert _cell_facts({"attachments": 1, "somethingNewer": "ignored"}) == "1 attachment"


# ------------------------------------------------------------ the widget --


def test_the_terminal_kind_gets_the_cell_widget() -> None:
    """The card kind selects the widget. A tool that declares `terminal`
    therefore gets this rendering, and the view does not know which tool it
    was."""
    view = TranscriptView()
    assert isinstance(view._build(ChatItem(key="t1", role="tool", tool=cell())), CodeCellWidget)

    generic = ToolCard(call_id="c2", name="read", arguments="{}")
    plain = view._build(ChatItem(key="t2", role="tool", tool=generic))
    assert isinstance(plain, ToolCardWidget)
    assert not isinstance(plain, CodeCellWidget)


def test_the_program_is_what_the_call_view_offered() -> None:
    """The program is `ToolCallView.body`, not the raw arguments. The JSON that
    the model emitted may not parse, and a widget must not be the thing that
    discovers that."""
    widget = CodeCellWidget(ChatItem(key="t1", role="tool", tool=cell()))
    assert widget._program() == "x = 1\nprint(x)"

    broken = CodeCellWidget(
        ChatItem(key="t2", role="tool", tool=ToolCard(call_id="c", name="python", arguments="{"))
    )
    assert broken._program() == ""


def test_the_cell_redraws_when_its_program_or_facts_change() -> None:
    """A memo stops every settled row from running its layout again on each
    frame. The memo must include the two things that this widget adds, or a
    streamed cell would freeze at its first snapshot."""
    card = cell()
    widget = CodeCellWidget(ChatItem(key="t1", role="tool", tool=card))
    before = widget._snapshot()

    card.details = {"attachments": 3}
    assert widget._snapshot() != before

    after_facts = widget._snapshot()
    card.input_text = "x = 2"
    assert widget._snapshot() != after_facts


# ------------------------------------------------------------- the panel --
#
# The daemon feeds the panel (P11-09). The log of a root holds no record of its
# children, so the panel draws `session.children`, which the daemon reads from
# the log of each child.


def _front() -> DaemonSession:
    """A front end on root `lead` with no socket behind it. A test hands it the
    frames of the daemon directly, through `dispatch`, the one entry that real
    frames take."""
    state = TuiState()
    return DaemonSession(
        client=cast(DaemonClient, StubClient()),
        session_id="lead",
        state=state,
        adapter=TuiEventAdapter(state=state),
        host=StubHost(),
    )


def _row(run_id: str, name: str = "", *, parent: str = "lead") -> ChildRow:
    """One row as the daemon sends it: admitted, and not yet started."""
    return ChildRow(
        run_id=run_id,
        session_id=f"{parent}-{run_id}",
        parent_id=parent,
        name=name or run_id,
        model="fake-1",
    )


def _drawn(*rows: ChildRow) -> TuiState:
    """The panel after one `session.children` frame carrying `rows`."""
    front = _front()
    front.dispatch(
        SessionChildrenNotice.METHOD,
        SessionChildrenNotice(session_id="lead", children=list(rows)).to_wire(),
    )
    return front.state


def _answer(child: Session) -> None:
    """One answer in a child's own log, with the four-term usage it spent."""
    log_event(
        child,
        "assistant/message",
        {
            **assistant_payload("found it", "m1"),
            "usage": {
                "inputTokens": 400,
                "outputTokens": 200,
                "cacheReadTokens": 100,
                "cacheWriteTokens": 50,
            },
        },
        SurfaceIntent("append", ()),
    )


async def test_the_panel_is_the_daemons_children_field_for_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A11, from the daemon: the panel is the log of each child, as the daemon read it.

    The rows are compared field by field against the `ChildState` of the seam,
    which is the fold that the resume sweep, passivation and the budgets read.
    They are not compared on two columns, because the divergence that this
    guards against is in a *field*. An early panel folded `cause` in its own
    way, and a comparison of the status alone did not see it. Every state that a
    row can be in is here: woken, errored, held for a key one level down, and
    revoked.

    The test is end to end without a socket. The watcher of the root *is* the
    `dispatch` of the front end, so the frame that the panel draws is the frame
    that the daemon built.

    Sabotage: leave `cause` out of `ChildRow.of`, and the woken child disagrees.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("lead")
        front = _front()
        root.subscribe(front.dispatch)

        scout = spawned(root, "r1", name="scout")
        log_event(scout, STATUS, {"status": "done"})
        log_event(scout, STATUS, {"status": "running", "cause": "rehydrated"})
        _answer(scout)
        nested = spawned(root, "g1", name="nested", under=scout)
        log_event(nested, "credential/needed", credential_hold(SESSION_HOLDER, "EXAMPLE_KEY"))
        recon = spawned(root, "r2", name="recon")
        log_event(recon, STATUS, {"status": "error", "detail": "boom"})
        revoked = spawned(root, "r3", name="revoked")
        with revoked.batch() as batch:
            log_event(batch, STATUS, {"status": "canceled"})
            log_event(batch, DELETED, {"reason": "user"})

        await until(
            lambda: (
                front.state.subagents.get(revoked.id, None) is not None
                and front.state.subagents[revoked.id].child.deleted
            ),
            what="the revocation to reach the panel",
        )
        family = family_of(root)
        drawn = list(front.state.subagents.values())
        assert [row.session_id for row in drawn] == [state.session_id for state in family]
        for row, state in zip(drawn, family, strict=True):
            for name in ChildRow.model_fields:
                assert getattr(row.child, name) == getattr(state, name), (
                    f"{state.session_id}: the panel's {name} is not the child's own"
                )
        assert [row.depth for row in drawn] == [0, 1, 0, 0]
        assert drawn[0].child.cause == "rehydrated", "the woken child is what this compares"


async def test_attributed_usage_is_summed_per_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What a child spent is the usage of its own answers, from its own log (Phase 11).

    The log of its parent used to carry a copy of the usage of each answer, and
    a crash could leave that copy behind the log of the child. The copy is gone,
    and so is the lag. All four terms are summed, as `/autonomous` charges the
    `children` of a goal (P2 review). A sum of input and output alone left the
    panel of a cache-heavy child in disagreement with the budget. The budget is
    `TokenUsage.total`, by which the fold of the seam charges each answer.

    Sabotage: leave `tokens` out of `ChildRow.of`, and the panel reads nothing spent.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("lead")
        front = _front()
        root.subscribe(front.dispatch)

        scout = spawned(root, "r1", name="scout")
        for _ in range(3):
            _answer(scout)

        await until(
            lambda: [row.child.tokens for row in front.state.subagents.values()] == [2_250],
            what="the child's own spend to reach the panel",
        )
        assert render_subagents(front.state) == "○ scout fake-1 2.2k"


async def test_a_revoked_child_stays_listed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A tombstone, not a removal. A parent that asks what happened to the child
    that it revoked needs an answer, and a row that vanishes is not one.

    The test starts from the revocation in the log of the child, as the daemon
    pushes it. The row stays, marked, and the heading stops counting it.

    Sabotage: skip deleted children in `family_of`, and the row vanishes.
    """
    async with supervised(tmp_path, monkeypatch) as supervisor:
        root = await supervisor.start("lead")
        front = _front()
        root.subscribe(front.dispatch)

        scout = spawned(root, "r1", name="scout")
        with scout.batch() as batch:
            log_event(batch, STATUS, {"status": "canceled"})
            log_event(batch, DELETED, {"reason": "user"})

        await until(
            lambda: any(row.child.deleted for row in front.state.subagents.values()),
            what="the revocation to reach the panel",
        )
        assert render_subagents(front.state).startswith("⊘ scout")
        assert children_heading(front.state) == "children"


def test_a_child_admitted_but_not_yet_started_reads_as_queued() -> None:
    """The first status comes from a detached job. A reader that arrives between
    admission and that record must not see a child with no status at all."""
    (row,) = _drawn(_row("r1", "scout")).subagents.values()
    assert (row.child.status, row.glyph) == ("queued", "○")


def test_a_woken_child_still_reads_as_running() -> None:
    """The `cause` of P3-13: rehydration is the reason why the child runs, and
    not a status of its own. A consumer that branches on `running` must still
    see it."""
    state = _drawn(
        _row("r1", "scout").model_copy(update={"status": "running", "cause": "rehydrated"})
    )
    (row,) = state.subagents.values()
    assert (row.child.status, row.child.cause) == ("running", "rehydrated")
    assert "rehydrated" in render_subagents(state)


def test_a_grandchild_is_drawn_under_the_child_that_spawned_it() -> None:
    """The family is a tree, and `parentId` is the edge. A fan-out that
    delegated again reads as nested, and not as more siblings."""
    state = _drawn(
        _row("r1", "scout"),
        _row("g1", "nested", parent="lead-r1"),
        _row("r2", "recon"),
    )
    assert render_subagents(state).splitlines() == [
        "○ scout fake-1",
        "  ○ nested fake-1",
        "○ recon fake-1",
    ]


def test_a_held_child_says_which_key_it_needs() -> None:
    """`queued` alone reads exactly like a child that waits for a slot, and only
    one of the two needs a person (T5)."""
    state = _drawn(_row("r1", "scout").model_copy(update={"awaiting": "EXAMPLE_KEY"}))
    assert render_subagents(state) == "○ scout needs EXAMPLE_KEY"


def test_a_family_frame_replaces_the_panel_whole() -> None:
    """Each frame is the whole family, so a row that the daemon stops listing
    leaves the panel. If frames were merged, the row would stay drawn for
    good."""
    front = _front()
    for rows in ([_row("r1"), _row("r2")], [_row("r2")]):
        front.dispatch(
            SessionChildrenNotice.METHOD,
            SessionChildrenNotice(session_id="lead", children=rows).to_wire(),
        )
    assert list(front.state.subagents) == ["lead-r2"]


class _Answering(StubClient):
    """A client whose `session/children` answer is `rows`. When `front` is given,
    a frame that carries `newer` lands while the call is in flight, and it
    overtakes the answer."""

    def __init__(
        self, rows: list[ChildRow], *, front: DaemonSession | None = None, newer: list[ChildRow]
    ) -> None:
        super().__init__()
        self.rows, self.front, self.newer = rows, front, newer

    async def call(self, verb: object, params: object = None) -> SessionChildrenNotice:
        if self.front is not None:
            self.front.dispatch(
                SessionChildrenNotice.METHOD,
                SessionChildrenNotice(session_id="lead", children=self.newer).to_wire(),
            )
        return SessionChildrenNotice(session_id="lead", children=self.rows)


async def test_a_frame_in_flight_is_not_overwritten_by_the_list_asked_before_it() -> None:
    """`refresh_children` reads the family one time at attach. A change that
    lands while it asks arrives as a frame. The answer must not be drawn over
    that frame, because the answer may be the list from before the change.

    Sabotage: draw the answer whatever arrived meanwhile, and the stale list wins.
    """
    front = _front()
    front.client = cast(DaemonClient, _Answering([_row("r1")], front=front, newer=[_row("r2")]))
    await front.refresh_children()
    assert list(front.state.subagents) == ["lead-r2"], "the list from before the change won"

    # With nothing in flight, the answer is the panel.
    front.client = cast(DaemonClient, _Answering([_row("r3")], newer=[]))
    await front.refresh_children()
    assert list(front.state.subagents) == ["lead-r3"]


def test_delegation_records_produce_no_transcript_rows() -> None:
    """The records of a sub-agent belong to its own log, and a transcript draws none of them.

    The log of a root holds none of them any more (Phase 11). So the "Delegated
    to …" and "Revoked child …" rows that they drew are gone with them. The
    spawn and the revocation are tool calls of the root itself, and their cards
    already say so. The family belongs to the panel, and it comes from the
    daemon. So a replay of the log of a child moves neither the transcript nor
    the panel. The log of a child is read in the trajectory view, which renders
    these records (`test_trajectory`).
    """
    session = Session("lead-r1", header=SessionHeader(id="lead-r1", created_at=1))
    log_event(session, ADMITTED, {"runId": "r1", "name": "scout", "model": "fake-1"})
    log_event(session, STATUS, {"status": "running"})
    log_event(session, STATUS, {"status": "done"})
    log_event(session, DELETED, {"reason": "user"})

    state = TuiEventAdapter().replay(session)

    assert state.items == []
    assert state.subagents == {}


def test_an_empty_family_renders_nothing() -> None:
    assert render_subagents(TuiState()) == ""
    assert children_heading(TuiState()) == "children"


def test_the_panel_heading_counts_the_fan_out() -> None:
    """Eight children are a list that somebody must tally by eye. This is the tally.

    **"pending", not "queued"**: the status bar already says "queued" for the
    *own* prompts of the user that wait on a busy agent. One word for two counts
    on one screen is worse than a synonym. The vocabulary of the children does
    not change: their logs still say `queued`, and every other reader still
    folds it.
    """
    rows = [_row(f"r{index}", f"scout-{index}") for index in range(5)]
    rows[0] = rows[0].model_copy(update={"status": "running"})
    rows[1] = rows[1].model_copy(update={"status": "running"})
    rows[4] = rows[4].model_copy(update={"status": "done"})

    state = _drawn(*rows)

    assert children_heading(state) == "children · 2 running, 2 pending"
    assert "queued" not in children_heading(state), "that word is the person's prompts"


def test_the_heading_counts_only_what_is_still_going() -> None:
    """A settled or revoked child stays *listed*, because a parent that asks
    what happened to it needs an answer. But "how busy is this fan-out" is a
    question about the other children."""
    state = _drawn(
        _row("r1", "scout").model_copy(update={"status": "done"}),
        _row("r2", "revoked").model_copy(update={"deleted": True, "deleted_reason": "user"}),
    )

    assert children_heading(state) == "children"
    assert len(render_subagents(state).splitlines()) == 2, "both stay on the panel"


# ------------------------------------------------------------- the todo panel --


def test_a_tick_with_work_behind_it_and_one_without_look_different() -> None:
    """The half of the receipt of P7-16 that the user sees.

    `tool-todo` attaches `worked` when it writes the list. The number is counted
    from what the harness saw run, and the model never supplies it. So a
    completed entry with a zero in it is a claim with nothing behind it. The
    tool card says so for one call. This panel is the plan that a user watches
    for the whole session, and the difference is worth seeing there.

    The number is read as a *field* and is not derived again. The module
    `ph_app` imports `ph-core` and does not import the bundle that owns the
    tool. A copy of "what counts as work" on this side is exactly the drift
    that the boundary exists to prevent.

    Sabotage: render every completed entry the same and a model that ticked a box
    without doing anything is indistinguishable from one that did the work.
    """
    worked = _todo_line({"content": "port the row", "status": "completed", "worked": 3})
    bare = _todo_line({"content": "decide the approach", "status": "completed", "worked": 0})

    assert worked == "● port the row"
    assert bare == f"● decide the approach{NO_WORK_SEEN}"


def test_only_a_completion_carries_the_receipt() -> None:
    """An unfinished entry has nothing to be evidence *for* yet.

    By definition a pending step has run no tools. To mark it "no work seen"
    would read as an accusation about work that nobody claimed.
    """
    assert _todo_line({"content": "gate it", "status": "pending"}) == "○ gate it"
    assert _todo_line({"content": "wire it", "status": "in_progress"}) == "◐ wire it"


def test_a_shortened_path_keeps_its_end_whole() -> None:
    """The sidebar keeps the end of a long path, which is the part that names
    it, and it cuts between characters. A filesystem can store a name
    decomposed: "é" as two code points, as macOS writes it. Such a name does not
    lose the letter that its accent belongs to.

    Sabotage: cut with `path[-(width - 1):]` again and the shortened path starts
    with a bare combining accent.
    """
    path = "/srv/" + "nested/" * 6 + "re\u0301sume\u0301.md"

    for width in range(8, 24):
        short = shorten_path(path, width)
        assert short.startswith("…") and len(short) <= width
        assert path.endswith(short[1:]) and not unicodedata.combining(short[1])


def test_a_child_with_no_name_and_no_model_is_drawn_by_its_run_id() -> None:
    """A queued child shows the model that it waits to run on, and a child goes
    by its name.

    A row with neither is drawn by its run id, with no model after it.

    Sabotage: format `child.model` into the line without the `None` test in
    `render_subagents`, and the row reads `○ r1 None`.
    """
    bare = ChildRow(run_id="r1", session_id="lead-r1", parent_id="lead")

    assert render_subagents(_drawn(bare)) == "○ r1"
