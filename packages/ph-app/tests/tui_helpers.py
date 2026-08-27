"""The two waits every pilot test needs, written once.

`running` opens an app and waits for the harness to mount — which happens in a
worker, so a single `pause()` is not enough. `until` polls a predicate through
the pilot instead of sleeping, so a test waits exactly as long as the app takes
and fails loudly rather than hanging when it never does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

from textual.pilot import Pilot

from ph_app.tui.app import PHTuiApp


async def until(pilot: Pilot[object], predicate: Callable[[], bool], *, tries: int = 400) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("the app never reached the expected state")


@asynccontextmanager
async def running(app: PHTuiApp) -> AsyncIterator[tuple[PHTuiApp, Pilot[object]]]:
    async with app.run_test() as pilot:
        await pilot.pause()
        await until(pilot, lambda: app.front is not None)
        yield app, pilot


def turn_done(app: PHTuiApp) -> Callable[[], bool]:
    """True once a turn has run to completion and been recorded."""

    def check() -> bool:
        front = app.front
        return (
            front is not None
            and front.state.status == "idle"
            and any(event.type == "turn/end" for event in front.session.events)
        )

    return check
