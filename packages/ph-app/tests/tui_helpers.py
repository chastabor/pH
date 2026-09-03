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

from ph.seams.approval import ApprovalRequest
from ph.seams.user_questions import UserQuestion
from ph_app.tui.app import PHTuiApp


async def until(pilot: Pilot[object], predicate: Callable[[], bool], *, tries: int = 400) -> None:
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("the app never reached the expected state")


@asynccontextmanager
async def running(
    app: PHTuiApp, *, size: tuple[int, int] = (80, 24)
) -> AsyncIterator[tuple[PHTuiApp, Pilot[object]]]:
    """`size` for a test that needs the transcript to actually scroll: the
    default is Textual's own, tall enough to hold a short conversation whole.

    Not `None`, which `run_test` reads as "detect the real terminal" — a test
    whose assertions depend on the viewport must not depend on the window it
    happened to run in.
    """
    async with app.run_test(size=size) as pilot:
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


class StubHost:
    """A `ModalHost` that answers every prompt without drawing anything.

    One double for both front ends — the in-process `HarnessSession` and the
    socket `DaemonSession` — so a member added to `ModalHost` is added here once.
    """

    def __init__(self) -> None:
        self.approvals: list[ApprovalRequest] = []
        self.questions: list[UserQuestion] = []
        self.redraws = 0

    async def ask_approval(self, request: ApprovalRequest) -> tuple[str, str]:
        self.approvals.append(request)
        return "allowed-once", ""

    async def ask_question(self, question: UserQuestion) -> str | None:
        self.questions.append(question)
        return "42"

    def state_changed(self) -> None:
        self.redraws += 1
