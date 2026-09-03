"""Drawing is coalesced and event-triggered, and costs nothing while idle.

The terminal used to poll a dirty flag thirty times a second. That is cheap for
one terminal and not for `ph --mode web`, which runs **one Textual subprocess per
browser tab**: ten idle tabs is ten processes waking thirty times a second to
find nothing changed. So the first change schedules one draw and every change
until then rides it.

**The hazard the change creates is what most of this file is about.** A poll
forgives a missed notification — a mutation that forgot to mark the view dirty
was redrawn within 33 ms by the next tick anyway. Nothing watches now, so the
same omission is a pane that stays wrong until the person types something, which
is a miserable thing to diagnose from a bug report. `state_changed` is therefore
the single entry point, and `test_every_state_mutation_schedules_a_draw` is what
keeps it that way.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Callable
from typing import Any

import pytest
from tui_helpers import root_of, running, until

import ph_app.tui.app
from ph.seams.tui_status import StatusField, StatusReading
from ph_app.tui.app import PHTuiApp

pytestmark = pytest.mark.anyio

MakeApp = Callable[..., PHTuiApp]


async def test_a_burst_of_changes_draws_once(make_tui_app: MakeApp) -> None:
    """The coalescing, which is the whole reason a draw is scheduled and not done.

    A streaming turn commits an `assistant/chunk` every few tokens and
    `view.sync` reconciles a widget list, so drawing per event spends more time
    laying out than rendering. The timer's *presence* is the mechanism: the
    second change through the tenth finds one already pending.
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
    """Idle costs nothing — the point of the change.

    Asserted by *counting draws*, not by inspecting the timer field: a poll
    restored as `set_interval(FRAME_INTERVAL, self._draw)` leaves `_draw_timer`
    empty and would sail past a check on it, while doing exactly the thing this
    test exists to forbid. The observable property is the claim.

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

        # Counted on the widget rather than on `app._draw`: a timer captured the
        # bound method when it was scheduled, so replacing the app's attribute
        # afterwards would count nothing and pass against the very poll this
        # forbids.
        status.show = counted  # type: ignore[method-assign]

        await pilot.pause(0.2)

        assert drawn == 0, f"redrew the status line {drawn} times with nothing to draw"
        assert app._draw_timer is None, "a draw is pending with nothing to draw"
        assert app._spinner is None, "the frame clock is running with no turn to time"


async def test_the_spinner_runs_only_while_a_turn_does(make_tui_app: MakeApp) -> None:
    """The one thing that genuinely wants a clock, and only while it is wanted.

    A spinner advances on wall-clock time rather than on anything arriving, so it
    cannot be event-driven — but it is the *only* such thing, so it starts when a
    turn starts and stops when one ends rather than running for the life of the
    app.
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
    """Belt and braces, and the failure mode of every animation loop ever written.

    `_draw` is what normally stops it. If a turn ends without one — the status
    changed and nothing marked the view dirty — the frame clock would otherwise
    spin for the life of the app, which is exactly the idle cost this change
    exists to remove, reintroduced by an omission.
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
    make_tui_app: MakeApp, tui_daemon: Any
) -> None:
    """A reading is a fold of the log, so a frame must not recompute one.

    Every registered status field runs over the whole session to produce the
    footer. Doing that thirty times a second while a turn streams was the most
    expensive thing on the frame path, and it could only ever return the same
    answer between appends.

    Sabotage: call `front.status_readings()` from `_advance` instead of reading
    the cache, and the count climbs with the frame rate.
    """
    async with running(make_tui_app()) as (app, pilot):
        front = app.front
        assert front is not None
        folds = 0

        def counted(session: object) -> StatusReading:
            nonlocal folds
            folds += 1
            return StatusReading(text="probe")

        # Counted in a registered field rather than by replacing the accessor:
        # The front end is frozen, and this is where the cost actually is —
        # every field runs over the whole session on every fold.
        root_of(tui_daemon).ctx.tui_status.register(StatusField(id="probe", read=counted))
        front.state.status = "running"
        app.state_changed()
        await until(pilot, lambda: app._spinner is not None)
        after_draw = folds

        # Long enough for several spinner frames at 1/30 s.
        await pilot.pause(0.25)

        assert folds == after_draw, "a spinner frame re-folded the log"


def test_every_state_mutation_schedules_a_draw() -> None:
    """`self._dirty = True` may appear only where it cannot mean anything else.

    This is the gate for the hazard the change introduced. While a poll watched
    the flag, setting it *was* the notification; now the timer is, and a mutation
    that sets the flag without scheduling leaves a pane stale until something
    unrelated redraws it — which is invisible in review and awful in a bug
    report.

    Two assignments are legitimate and both are checked by name rather than
    counted: the `__init__` seed, which cannot schedule because `set_timer` needs
    a running app, and the one inside `state_changed` itself.

    Sabotage: write `self._dirty = True` anywhere else in `app.py`.
    """
    source = pathlib.Path(ph_app.tui.app.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source, filename="app.py")
    functions = {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    offenders = [
        f"{name}:{node.lineno}"
        for name, body in functions.items()
        if name not in ("__init__", "state_changed")
        for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        # `= True` only: clearing the flag is what a draw *is*, and `_draw` does
        # it on the line before it draws.
        and isinstance(node.value, ast.Constant)
        and node.value.value is True
        and any(
            isinstance(target, ast.Attribute) and target.attr == "_dirty" for target in node.targets
        )
    ]
    assert offenders == [], f"these set the flag without scheduling a draw: {offenders}"
