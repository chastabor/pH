"""The gate of P6-01: a hand-edited `harness_state.json` trips the invariant (I6, A11).

This is the named gate of the row. The projection is the right target for it,
because it is the one file in pH that **nothing reads back**. `write_projection`
says so in one line: written for humans. Nothing can notice the drift of a file
that has no reader. That makes it the safest projection to have. It is also the
only one where an invariant does real work and does not restate a type.

**The two drifts that it catches are different failures that look the same.** A
file that fell behind its fold means that a person reads a stale account of what
the agent learned. A file that somebody *wrote* is worse. The deployment then
has two carriers for one fact (A11), and the log has quietly stopped being the
source of truth (I6). Neither failure announces itself. The second looks exactly
like the first until someone forks a session and the two disagree.
"""

from __future__ import annotations

from typing import Any

import pytest
from rlm_fixtures import INVARIANT_ROW, Harnessed, note_edit

from ph.testing import report_section
from ph_rlm.harness import RefinementProposal
from ph_rlm.harness.invariant import violations
from ph_rlm.keys import HARNESS

pytestmark = pytest.mark.anyio


async def _refined(harnessed: Harnessed) -> tuple[Any, Any, Any, Any]:
    """A session whose harness has one entry and a written projection, the agent
    of the session, and the path of the projection."""
    ctx, session, agent = await harnessed(INVARIANT_ROW)
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="one", edits=[note_edit("thing")]), agent=agent
    )
    path = ctx.require(HARNESS).projection_path(session)
    assert path.exists(), "nothing was projected to check"
    return ctx, session, agent, path


async def test_a_hand_edited_projection_trips_the_invariant(harnessed: Harnessed) -> None:
    """The gate. Somebody edited the file, and the harness noticed.

    The file is edited and not deleted, because deletion is the *benign* case
    that this must not confuse with drift. An absent projection loses nothing,
    because the state is the log. An edit means that something now believes
    that the file is state, and the next fold will silently overwrite whatever
    it believed.
    """
    ctx, _session, _agent, path = await _refined(harnessed)

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
    """The other drift, and the path that must keep not producing it.

    A path that changes the state and forgets to project again leaves a stale
    projection behind. This is the drift that happens without anybody intending
    it: no hand edit, no second writer, only a fold that moved. So the property
    to hold is on `apply` and not on the file. Every path that moves the fold
    projects again. The invariant reports that afterwards, so a reader does not
    have to trust it.

    The test uses two refinements and not one, because the first writes a
    projection where there was none. A path could pass that test by creating
    the file and still never update one.
    """
    ctx, session, agent, _path = await _refined(harnessed)

    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="two", edits=[note_edit("second")]), agent=agent
    )

    assert ctx.require(HARNESS).state(session).entry("note", "second") is not None, (
        "the fold did not move"
    )
    assert violations(ctx) == [], "the fold moved and the projection was left behind"


async def test_a_missing_projection_is_not_a_violation(harnessed: Harnessed) -> None:
    """Absence is not drift, and reporting it as drift would ruin the signal.

    Nothing requires a session to have projected. `HarnessService._commit`
    calls `write_projection` when it records a refinement or a rollback, so a
    session that has never refined has no file. If a missing file were drift, a
    deployment that never wrote one would report a violation on every
    `phern doctor`. An alarm that fires loudest where the feature is used least
    is an alarm that people learn to ignore, and then the alarm that matters is
    lost.
    """
    ctx, session, _agent, path = await _refined(harnessed)
    path.unlink()

    assert violations(ctx) == []
    assert ctx.require(HARNESS).state(session).entry("note", "thing") is not None, (
        "deleting the projection lost state, so it was never only a projection"
    )
