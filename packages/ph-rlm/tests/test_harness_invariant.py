"""P6-01's gate — a hand-edited `harness_state.json` trips the invariant (I6, A11).

This is the row's named gate, and the projection is the right thing to aim it at
because it is the one file in pH that **nothing reads back**. `write_projection`
says so in one line: written for humans. A file with no reader is a file whose
drift nothing can notice, which makes it both the safest projection to have and
the only one where an invariant is doing work rather than restating a type.

**The two drifts it catches are different failures wearing one shape.** A file
that fell behind its fold is a person reading a stale account of what the agent
learned. A file somebody *wrote* is worse: the deployment now has two carriers
for one fact (A11), and the log has quietly stopped being the source of truth
(I6). Neither announces itself, and the second looks exactly like the first until
someone forks a session and the two disagree.
"""

from __future__ import annotations

from typing import Any

import pytest
from conftest import INVARIANT_ROW, Harnessed, note_edit

from ph.testing import report_section
from ph_rlm.harness import RefinementProposal
from ph_rlm.harness.invariant import violations

pytestmark = pytest.mark.anyio


async def _refined(harnessed: Harnessed) -> tuple[Any, Any, Any]:
    """A session whose harness has one entry and a written projection, and where it is."""
    ctx, session, agent = await harnessed(INVARIANT_ROW)
    await ctx.harness.apply(
        RefinementProposal(summary="one", edits=[note_edit("thing")]), session=session, agent=agent
    )
    path = ctx.harness.projection_path(session)
    assert path.exists(), "nothing was projected to check"
    return ctx, session, path


async def test_a_hand_edited_projection_trips_the_invariant(harnessed: Harnessed) -> None:
    """The gate. Somebody edited the file, and the harness noticed.

    Edited rather than deleted, because deletion is the *benign* case this must
    not confuse with drift: an absent projection loses nothing, since the state is
    the log. What an edit means is that something now believes the file is state
    — and the next fold will silently overwrite whatever it believed.
    """
    ctx, _session, path = await _refined(harnessed)

    assert violations(ctx) == [], "a freshly written projection disagreed with its own fold"
    assert report_section(ctx, "Invariants")["harness-projection"].startswith("holds ·"), (
        "the row is mounted but never reached the report"
    )

    path.write_text(path.read_text(encoding="utf-8").replace('"thing"', '"edited"'), "utf-8")

    (found,) = violations(ctx)
    assert str(path) in found and "does not equal the fold" in found
    assert report_section(ctx, "Invariants")["harness-projection"].startswith("VIOLATED ·")


async def test_a_second_refinement_leaves_the_projection_equal_to_the_fold(
    harnessed: Harnessed,
) -> None:
    """The other drift, and the path that has to keep not producing it.

    A stale projection is what a path that changes the state and forgets to
    re-project leaves behind, and it is the drift that happens without anybody
    meaning it — no hand edit, no second writer, just a fold that moved. So the
    property worth holding is on `apply` rather than on the file: every path that
    moves the fold re-projects, and the invariant is what says so afterwards
    instead of a reader having to trust it.

    Two refinements rather than one, because the first writes a projection where
    there was none — a path could pass that by creating the file and still never
    update one.
    """
    ctx, session, _path = await _refined(harnessed)

    await ctx.harness.apply(
        RefinementProposal(summary="two", edits=[note_edit("second")]), session=session, agent=None
    )

    assert ctx.harness.state(session).entry("note", "second") is not None, "the fold did not move"
    assert violations(ctx) == [], "the fold moved and the projection was left behind"


async def test_a_missing_projection_is_not_a_violation(harnessed: Harnessed) -> None:
    """Absence is not drift, and reporting it as drift would ruin the signal.

    Nothing requires a session to have projected — `write_projection` is called
    where a human might look, not on every refinement — so a deployment that
    simply never wrote one would report a violation on every `ph doctor`. An
    alarm that fires loudest where the feature is used least is one people learn
    to ignore, which costs the alarm that matters.
    """
    ctx, session, path = await _refined(harnessed)
    path.unlink()

    assert violations(ctx) == []
    assert ctx.harness.state(session).entry("note", "thing") is not None, (
        "deleting the projection lost state, so it was never only a projection"
    )
