"""The code cell and the subagent panel (P3-19).

Both are consumers of records that existed before they did: `IpythonToolDetails`
was written at P3-09 with nothing to draw it, and `subagent/status` /
`subagent/usage-attributed` sat in the adapter's `RECORDLESS` set naming *this*
row as the reason they were classified rather than rendered.

The claim under both is the P2-01 one, one layer down: everything drawn comes
from the settled record, so a replayed cell is the cell that ran and a resumed
session's panel is the family the parent left.

## Why the cell card does not print the dispatch count

The collapsible below it already reports the count, from the
`tool/code-dispatch-start` fold that owns those rows. Two projections of one number
in one widget can disagree — which is what A11 forbids, and what the first draft of
this card did: a snapshot showing **"3 governed calls" over a section titled "1
governed call"**.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.seams.subagents import ADMITTED, DELETED, STATUS, USAGE, subagent_roster
from ph.session import Session
from ph_app.tui.adapter import TuiEventAdapter
from ph_app.tui.state import ChatItem, SubagentRow, ToolCard, TuiState
from ph_app.tui.widgets.status import NO_WORK_SEEN, _todo_line, render_subagents
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


def _admitted(run_id: str, name: str) -> dict[str, Any]:
    return {"runId": run_id, "name": name, "model": "fake-1", "grantedAccess": "read"}


def test_the_panel_is_the_seams_fold_field_for_field() -> None:
    """A11: the panel and the roster the model reads are one projection.

    Compared field by field rather than on two columns, because the divergence
    this guards against is a *field* — the first draft folded `cause` its own
    way, and a status-only comparison passed straight over it.
    """
    session = Session("panel")
    session.append(ADMITTED, _admitted("r1", "scout"))
    session.append(ADMITTED, _admitted("r2", "recon"))
    session.append(STATUS, {"runId": "r1", "status": "running", "cause": "rehydrated"})
    session.append(STATUS, {"runId": "r1", "status": "done"})
    session.append(STATUS, {"runId": "r2", "status": "done"})
    session.append(DELETED, {"runId": "r2", "reason": "user"})

    state = TuiEventAdapter().replay(session)
    roster = subagent_roster(session)

    assert state.roster == roster, "the panel folded something the seam did not"
    assert list(state.subagents) == list(roster)
    for run_id, entry in roster.items():
        row = state.subagents[run_id]
        assert row.status == entry["status"]
        assert row.name == entry["name"]
        assert row.model == entry["model"]
        assert row.cause == str(entry.get("cause") or "")
        assert row.deleted == bool(entry.get("deleted"))


def test_a_child_admitted_but_not_yet_started_reads_as_queued() -> None:
    """The first `subagent/status` comes from a detached job, so a reader between
    admission and that event must not see a child with no status at all."""
    session = Session("queued")
    session.append(ADMITTED, _admitted("r1", "scout"))

    (row,) = TuiEventAdapter().replay(session).subagents.values()
    assert (row.status, row.glyph) == ("queued", "○")


def test_a_woken_child_still_reads_as_running() -> None:
    """P3-13's `cause`: rehydration is why it is running, not a status of its
    own — a consumer branching on `running` must still see it."""
    session = Session("woken")
    session.append(ADMITTED, _admitted("r1", "scout"))
    session.append(STATUS, {"runId": "r1", "status": "done"})
    session.append(STATUS, {"runId": "r1", "status": "running", "cause": "rehydrated"})

    state = TuiEventAdapter().replay(session)
    (row,) = state.subagents.values()
    assert (row.status, row.cause) == ("running", "rehydrated")
    assert "rehydrated" in render_subagents(state)


def test_attributed_usage_is_summed_per_child() -> None:
    """The one field the seam's fold does not carry, so the panel adds it."""
    session = Session("usage")
    session.append(ADMITTED, _admitted("r1", "scout"))
    for _ in range(3):
        session.append(
            USAGE, {"runId": "r1", "childUsage": {"inputTokens": 400, "outputTokens": 200}}
        )

    (row,) = TuiEventAdapter().replay(session).subagents.values()
    assert row.tokens == 1_800


def test_delegation_records_produce_no_transcript_rows() -> None:
    """Status and usage are the panel's, not the conversation's: eight children
    ticking through `queued → running → done` would push it off screen."""
    session = Session("quiet")
    session.append(ADMITTED, _admitted("r1", "scout"))
    session.append(STATUS, {"runId": "r1", "status": "running"})
    session.append(USAGE, {"runId": "r1", "childUsage": {"inputTokens": 1}})

    state = TuiEventAdapter().replay(session)
    # One row for the admission — a spawn is a decision — and nothing for the
    # two that followed, which the panel drew instead.
    assert len(state.items) == 1
    assert "Delegated to scout" in state.items[0].text
    assert state.subagents["r1"].status == "running"


def test_an_empty_family_renders_nothing() -> None:
    assert render_subagents(TuiState()) == ""


def test_a_revoked_child_stays_listed() -> None:
    """A tombstone, not a removal: a parent asking what happened to the child it
    revoked deserves an answer other than the row vanishing."""
    state = TuiState()
    state.subagents["r1"] = SubagentRow(run_id="r1", name="scout", status="done", deleted=True)
    assert render_subagents(state).startswith("⊘ scout")


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
