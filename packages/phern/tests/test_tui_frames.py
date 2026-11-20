"""Drawing is coalesced and event-triggered, and costs nothing while idle.

The terminal used to poll a dirty flag thirty times a second. That is cheap for
one terminal and not for `phern --mode web`, which runs **one Textual subprocess
for each browser tab**: ten idle tabs are ten processes that wake thirty times a
second and find nothing changed. So the first change schedules one draw, and
every change until that draw is included in it.

**Most of this file is about the hazard that the change creates.** A poll
forgives a missed notification. A mutation that forgot to mark the view dirty
was redrawn within 33 ms by the next tick anyway. Nothing watches now, so the
same omission is a pane that stays wrong until the user types something, and
that is hard to diagnose from a bug report. `state_changed` is therefore the
single entry point, and `test_every_state_mutation_schedules_a_draw` keeps it
the single entry point.
"""

from __future__ import annotations

import ast
import pathlib

import pytest
from daemon_helpers import Daemon
from tui_helpers import MakeApp, root_of, running, until

import ph_app.tui.app
from ph.keys import TUI_STATUS
from ph.seams.tui_status import StatusField, StatusReading
from ph.session import Session

pytestmark = pytest.mark.anyio


async def test_a_burst_of_changes_draws_once(make_tui_app: MakeApp) -> None:
    """The coalescing: the reason why a draw is scheduled and not done at once.

    A streaming turn commits an `assistant/chunk` every few tokens, and
    `view.sync` reconciles a widget list. A draw for each event would therefore
    spend more time on layout than on rendering. The *presence* of the timer is
    the mechanism: the second change through the tenth find a draw already
    pending.
    """
    async with running(make_tui_app()) as (app, pilot):
        await until(pilot, lambda: app._draw_timer is None)
        draws = 0
        original = app._draw

        async def counted() -> None:
            nonlocal draws
            draws += 1
            await original()

        app._draw = counted  # type: ignore[method-assign]

        for _ in range(10):
            app.state_changed()

        await pilot.pause(0.2)
        assert draws == 1, "ten changes in one frame must draw once"


async def test_nothing_is_drawn_while_nothing_happens(make_tui_app: MakeApp) -> None:
    """Idle costs nothing, which is the purpose of the change.

    The test *counts draws* and does not inspect the timer field. A poll that
    was restored as `set_interval(FRAME_INTERVAL, self._draw)` leaves
    `_draw_timer` empty. It would pass a check on that field while it did
    exactly what this test exists to forbid. The observable property is the
    claim.

    Sabotage: `set_interval(FRAME_INTERVAL, self._draw)` at mount — six draws in
    the quiet fifth of a second below.
    """
    async with running(make_tui_app()) as (app, pilot):
        await until(pilot, lambda: app._draw_timer is None)
        status = app._status
        assert status is not None
        drawn = 0
        original = status.show

        def counted(*args: object, **kwargs: object) -> None:
            nonlocal drawn
            drawn += 1
            original(*args, **kwargs)  # type: ignore[arg-type]

        # The count is on the widget and not on `app._draw`. A timer captured
        # the bound method when it was scheduled. Replacing the attribute of the
        # app afterwards would therefore count nothing, and it would pass
        # against the very poll that this test forbids.
        status.show = counted  # type: ignore[method-assign]

        await pilot.pause(0.2)

        assert drawn == 0, f"redrew the status line {drawn} times with nothing to draw"
        assert app._draw_timer is None, "a draw is pending with nothing to draw"
        assert app._spinner is None, "the frame clock is running with no turn to time"


async def test_the_spinner_runs_only_while_a_turn_does(make_tui_app: MakeApp) -> None:
    """The spinner is the one thing that needs a clock, and only while it is wanted.

    A spinner advances on wall-clock time and not on anything that arrives, so
    it cannot be event-driven. But it is the *only* such thing. So it starts
    when a turn starts and stops when a turn ends, and it does not run for the
    life of the app.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        await until(pilot, lambda: app._spinner is None)

        front.state.status = "running"
        app.state_changed()
        await until(pilot, lambda: app._spinner is not None)

        front.state.status = "idle"
        app.state_changed()
        await until(pilot, lambda: app._spinner is None)


async def test_a_turn_that_ends_unannounced_still_stops_the_spinner(
    make_tui_app: MakeApp,
) -> None:
    """A second safeguard, against the usual failure of an animation loop.

    `_draw` is what normally stops the spinner. Suppose that a turn ends without
    a draw: the status changed and nothing marked the view dirty. The frame
    clock would otherwise run for the life of the app. That is exactly the idle
    cost that this change exists to remove, brought back by an omission.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        front.state.status = "running"
        app.state_changed()
        await until(pilot, lambda: app._spinner is not None)

        # No `state_changed`: the omission under test.
        front.state.status = "idle"

        await until(pilot, lambda: app._spinner is None)


async def test_the_footer_is_folded_on_change_and_not_per_frame(
    make_tui_app: MakeApp, tui_daemon: Daemon
) -> None:
    """A reading is a fold of the log, so a frame must not compute one again.

    Every registered status field runs over the whole session to produce the
    footer. Doing that thirty times a second while a turn streamed was the most
    expensive thing on the frame path, and between appends it could only return
    the same answer.

    Sabotage: call `front.status_readings()` from `_advance`, and the count
    climbs with the frame rate.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        folds = 0

        def counted(session: Session) -> StatusReading:
            nonlocal folds
            folds += 1
            return StatusReading(text="probe")

        # The count is in a registered field, and the accessor is not replaced.
        # The front end is a dataclass with slots, so its methods cannot be
        # replaced. The field is also where the cost is: every field runs over
        # the whole session on every fold.
        root_of(tui_daemon).ctx.require(TUI_STATUS).register(StatusField(id="probe", read=counted))
        front.state.status = "running"
        app.state_changed()
        await until(pilot, lambda: app._spinner is not None)
        after_draw = folds

        # Long enough for several spinner frames at 1/30 s.
        await pilot.pause(0.25)

        assert folds == after_draw, "a spinner frame re-folded the log"


def _writes_dirty(node: ast.AST) -> bool:
    """Whether `node` assigns to an attribute named `_dirty`, in any form: `=`,
    `|=`, an annotated assignment, or one side of a tuple assignment."""
    if isinstance(node, ast.Assign):
        targets = node.targets
    elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
        targets = [node.target]
    else:
        return False
    return any(
        isinstance(leaf, ast.Attribute) and leaf.attr == "_dirty"
        for target in targets
        for leaf in ast.walk(target)
    )


DIRTY_WRITERS = {"__init__", "state_changed", "_draw"}
"""The three functions in `app.py` that may write `_dirty`:

- `__init__` seeds it. The seed cannot schedule a draw, because `set_timer`
  needs a running app.
- `state_changed` adds surfaces to it and schedules the draw.
- `_draw` clears it, which is what a draw is.
"""


def test_every_state_mutation_schedules_a_draw() -> None:
    """Only `__init__`, `state_changed` and `_draw` may write `self._dirty`.

    This is the gate for the hazard that the change introduced. While a poll
    watched the flag, setting the flag *was* the notification. Now the timer is
    the notification. A mutation that sets the flag without scheduling a draw
    leaves a pane stale until something unrelated redraws it. That is invisible
    in review and hard to diagnose from a bug report.

    **The gate matches every form of assignment, and it must find the writes
    that it allows.** It used to match only `self._dirty = True`. The flag then
    became a set of `Surface` values, written as `= Surface.ALL` and
    `|= surfaces`. No code had the old shape any more, so the gate matched
    nothing and passed whatever `app.py` did. The comparison below is therefore
    an equality: a writer that is not allowed fails it, and so does a gate that
    no longer finds the three writers that exist.

    Sabotage: write `self._dirty |= Surface.FOOTER` in any other method of
    `app.py`.
    """
    source = pathlib.Path(ph_app.tui.app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app.py")
    writers = {
        function.name
        for function in ast.walk(tree)
        if isinstance(function, (ast.FunctionDef, ast.AsyncFunctionDef))
        and any(_writes_dirty(node) for node in ast.walk(function))
    }
    assert writers == DIRTY_WRITERS, (
        f"these write the flag without scheduling a draw: {sorted(writers - DIRTY_WRITERS)}; "
        f"the gate no longer finds a write in: {sorted(DIRTY_WRITERS - writers)}"
    )
