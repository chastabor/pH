"""The code cell and the subagent panel (P3-19).

Both are consumers of records that existed before they did: `IpythonToolDetails`
was written at P3-09 with nothing to draw it, and the children's records sat in
the adapter's `RECORDLESS` set naming *this* row as the reason they were
classified rather than rendered.

The claim under both is the P2-01 one, one layer down: everything drawn comes
from the settled record, so a replayed cell is the cell that ran and a resumed
session's panel is the family the parent left — which, since each session owns its
log (Phase 11), is read from each child's own log by the daemon and sent as
`session.children` (P11-09).

## Why the cell card does not print the dispatch count

The collapsible below it already reports the count, from the
`tool/code-dispatch-start` fold that owns those rows. Two projections of one number
in one widget can disagree — which is what A11 forbids, and what the first draft of
this card did: a snapshot showing **"3 governed calls" over a section titled "1
governed call"**.
"""

from __future__ import annotations

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
        name="ipython",
        arguments='{"program": "x = 1\\nprint(x)"}',
        title="ipython",
        subtitle="2 lines",
        card="terminal",
        input_text="x = 1\nprint(x)",
        settled=True,
        body="1",
    )


# ------------------------------------------------------------- the facts --


def test_only_facts_that_are_true_are_shown() -> None:
    """A cell that dispatched nothing and truncated nothing has nothing to say,
    and a line of `0 governed calls · not truncated` would say it anyway."""
    assert _cell_facts({}) == ""
    assert _cell_facts({"dispatches": 0, "truncated": False, "reset": False}) == ""


def test_the_facts_line_carries_what_the_collapsible_does_not() -> None:
    facts = _cell_facts({"dispatches": 40, "attachments": 2, "truncated": True, "reset": True})
    assert "2 attachments" in facts
    assert "output truncated" in facts
    assert "kernel restarted" in facts
    # The dispatch *count* is the collapsible's, from the fold that owns those
    # rows: rendering it here too put two projections of one number in one
    # widget, able to disagree (A11).
    assert "40" not in facts


def test_one_of_something_is_not_pluralized() -> None:
    assert _cell_facts({"attachments": 1}) == "1 attachment"
    assert _cell_facts({"attachments": 2}) == "2 attachments"


def test_a_field_this_build_does_not_know_is_ignored() -> None:
    """Read as a mapping, not as ph-rlm's model — ph-app does not depend on the
    bundle, so a tool can enrich its own card without the transcript learning
    its schema."""
    assert _cell_facts({"attachments": 1, "somethingNewer": "ignored"}) == "1 attachment"


# ------------------------------------------------------------ the widget --


def test_the_terminal_kind_gets_the_cell_widget() -> None:
    """The card kind is what selects the widget, so a tool declaring `terminal`
    gets this rendering without the view knowing which tool it was."""
    view = TranscriptView()
    assert isinstance(view._build(ChatItem(key="t1", role="tool", tool=cell())), CodeCellWidget)

    generic = ToolCard(call_id="c2", name="read", arguments="{}")
    plain = view._build(ChatItem(key="t2", role="tool", tool=generic))
    assert isinstance(plain, ToolCardWidget)
    assert not isinstance(plain, CodeCellWidget)


def test_the_program_is_what_the_call_view_offered() -> None:
    """`ToolCallView.body`, not the raw arguments: the JSON the model emitted may
    not even parse, and a widget must not be the thing that discovers that."""
    widget = CodeCellWidget(ChatItem(key="t1", role="tool", tool=cell()))
    assert widget._program() == "x = 1\nprint(x)"

    broken = CodeCellWidget(
        ChatItem(key="t2", role="tool", tool=ToolCard(call_id="c", name="ipython", arguments="{"))
    )
    assert broken._program() == ""


def test_the_cell_redraws_when_its_program_or_facts_change() -> None:
    """The memo that stops every settled row re-laying-out per frame has to
    include the two things this widget adds, or a streamed cell would freeze at
    its first snapshot."""
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
# Fed by the daemon (P11-09): a root's log holds no record of its children, so the
# panel draws `session.children`, which the daemon reads from each child's own log.


def _front() -> DaemonSession:
    """A front end on root `lead` with no socket behind it: a test hands it the
    daemon's frames directly, through the one door real frames take (`dispatch`)."""
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
    """A11, from the daemon: the panel is each child's own log, as the daemon read it.

    Compared field by field against the seam's `ChildState` — the fold the resume
    sweep, passivation and the budgets read — rather than on two columns, because
    the divergence this guards against is a *field*: an early panel folded `cause`
    its own way, and a status-only comparison passed straight over it. Every state
    a row can be in is here: woken, errored, held for a key a level down, revoked.

    End to end without a socket: the root's watcher *is* the front end's
    `dispatch`, so the frame the panel draws is the one the daemon built.

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
    """What a child spent is its own answers' usage, from its own log (Phase 11).

    Its parent's log used to carry a copy of each answer's usage, which a crash
    could leave behind the child's own; the copy is gone, and so is the lag. All
    four terms, as `/autonomous` charges a goal's `children` (P2 review): summing
    input and output alone left a cache-heavy child's panel disagreeing with the
    budget — `TokenUsage.total`, which the seam's fold charges each answer by.

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
    """A tombstone, not a removal: a parent asking what happened to the child it
    revoked deserves an answer other than the row vanishing.

    From the revocation in the child's own log, as the daemon pushes it — the row
    stays, marked, and the heading stops counting it.

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
    """The first status comes from a detached job, so a reader between admission
    and that record must not see a child with no status at all."""
    (row,) = _drawn(_row("r1", "scout")).subagents.values()
    assert (row.child.status, row.glyph) == ("queued", "○")


def test_a_woken_child_still_reads_as_running() -> None:
    """P3-13's `cause`: rehydration is why it is running, not a status of its
    own — a consumer branching on `running` must still see it."""
    state = _drawn(
        _row("r1", "scout").model_copy(update={"status": "running", "cause": "rehydrated"})
    )
    (row,) = state.subagents.values()
    assert (row.child.status, row.child.cause) == ("running", "rehydrated")
    assert "rehydrated" in render_subagents(state)


def test_a_grandchild_is_drawn_under_the_child_that_spawned_it() -> None:
    """The family is a tree, and `parentId` is the edge: a fan-out that delegated
    again reads as nested rather than as more siblings."""
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
    """`queued` alone reads exactly like a child waiting for a slot, and only one of
    the two needs a person (T5)."""
    state = _drawn(_row("r1", "scout").model_copy(update={"awaiting": "EXAMPLE_KEY"}))
    assert render_subagents(state) == "○ scout needs EXAMPLE_KEY"


def test_a_family_frame_replaces_the_panel_whole() -> None:
    """Each frame is the whole family, so a row the daemon stops listing leaves the
    panel — merged, it would stay drawn for good."""
    front = _front()
    for rows in ([_row("r1"), _row("r2")], [_row("r2")]):
        front.dispatch(
            SessionChildrenNotice.METHOD,
            SessionChildrenNotice(session_id="lead", children=rows).to_wire(),
        )
    assert list(front.state.subagents) == ["lead-r2"]


class _Answering(StubClient):
    """A client whose `session/children` answer is `rows` — overtaken, when `front` is
    given, by a frame carrying `newer` that lands while the call is in flight."""

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
    """`refresh_children` reads the family once at attach — and a change that lands
    while it is asking arrives as a frame the answer must not be drawn over, since
    that answer may be the list from before the change.

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
    """A sub-agent's records are its own log's, and a transcript draws none of them.

    A root's log holds none of them any more (Phase 11), so the "Delegated to …" and
    "Revoked child …" rows they drew are gone with them: the spawn and the revocation
    are the root's own tool calls, whose cards already say so. The family is the
    panel's, from the daemon — so replaying a child's own log moves neither the
    transcript nor the panel. A child's log is read in the trajectory view, which
    renders them (`test_trajectory`).
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
    """Eight children is a list somebody has to tally by eye; this is the tally.

    **"pending", not "queued"**: the status bar already says "queued" for the
    person's *own* prompts waiting on a busy agent, and one word for two counts
    on one screen is worse than a synonym. The children's vocabulary is untouched —
    their logs still say `queued` and every other reader still folds it.
    """
    rows = [_row(f"r{index}", f"scout-{index}") for index in range(5)]
    rows[0] = rows[0].model_copy(update={"status": "running"})
    rows[1] = rows[1].model_copy(update={"status": "running"})
    rows[4] = rows[4].model_copy(update={"status": "done"})

    state = _drawn(*rows)

    assert children_heading(state) == "children · 2 running, 2 pending"
    assert "queued" not in children_heading(state), "that word is the person's prompts"


def test_the_heading_counts_only_what_is_still_going() -> None:
    """A settled or revoked child stays *listed* — a parent asking what happened
    to it deserves an answer — but "how busy is this fan-out" is about the rest."""
    state = _drawn(
        _row("r1", "scout").model_copy(update={"status": "done"}),
        _row("r2", "revoked").model_copy(update={"deleted": True, "deleted_reason": "user"}),
    )

    assert children_heading(state) == "children"
    assert len(render_subagents(state).splitlines()) == 2, "both stay on the panel"


# ------------------------------------------------------------- the todo panel --


def test_a_tick_with_work_behind_it_and_one_without_look_different() -> None:
    """The person-facing half of P7-16's receipt.

    `worked` is attached by `tool-todo` when it writes the list — counted from
    what the harness saw run, never supplied by the model — so a completed entry
    with a zero in it is a claim with nothing behind it. The tool card says so
    for one call; this panel is the plan a person watches all session, which is
    where the difference is worth seeing.

    Read as a *field*, not re-derived: `ph-app` depends on `ph-core` and not on
    the bundle that owns the tool, so a copy of "what counts as work" on this
    side is exactly the drift that boundary exists to prevent.

    Sabotage: render every completed entry the same and a model that ticked a box
    without doing anything is indistinguishable from one that did the work.
    """
    worked = _todo_line({"content": "port the row", "status": "completed", "worked": 3})
    bare = _todo_line({"content": "decide the approach", "status": "completed", "worked": 0})

    assert worked == "● port the row"
    assert bare == f"● decide the approach{NO_WORK_SEEN}"


def test_only_a_completion_carries_the_receipt() -> None:
    """An unfinished entry has nothing to be evidence *for* yet.

    A pending step has run no tools by definition, and marking it "no work seen"
    would read as an accusation about work that was never claimed.
    """
    assert _todo_line({"content": "gate it", "status": "pending"}) == "○ gate it"
    assert _todo_line({"content": "wire it", "status": "in_progress"}) == "◐ wire it"
