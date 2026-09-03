"""The two waits every pilot test needs, written once.

`running` opens an app and waits for it to attach to a daemon — which happens
in a worker, so a single `pause()` is not enough. `until` polls a predicate through
the pilot instead of sleeping, so a test waits exactly as long as the app takes
and fails loudly rather than hanging when it never does.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from textual.pilot import Pilot

from ph.seams.approval import ApprovalRequest
from ph.seams.user_questions import UserQuestion
from ph_app.tui.app import PHTuiApp


def tui_app(
    *, home: Path, project: Path | None = None, trusted: bool = True, **overrides: Any
) -> PHTuiApp:
    """One `PHTuiApp` wired for a test, built in the one place.

    Two callers: the async `make_tui_app` fixture, and the snapshot suite, which
    cannot use a fixture because `snap_compare` drives its own event loop. They
    were two copies of the app's construction contract, and this diff had to edit
    both when it added `daemon_argv`.

    `spawn=False`: a test must never start a supervisor. The fixture's socket is
    already answering, so `ensure_daemon` returns without spawning, and a
    regression that tried fails as a named `DaemonAbsent`.

    `trusted` writes the file the *daemon* reads — `session/new` is what enforces
    trust — so trusting here is what lets the mount happen at all. `project` is
    set before it, because trust is keyed on that path.
    """
    options: dict[str, Any] = {
        "session_id": "pilot",
        "home": home,
        "daemon_argv": (),
        "spawn": False,
    }
    options.update(overrides)
    app = PHTuiApp(**options)
    if project is not None:
        # `PHTuiApp.project` is `Path.cwd()`, and the sidebar prints it. A
        # snapshot taken here therefore embedded the developer's checkout and
        # could never match CI, whose cwd differs.
        app.project = project
    if trusted:
        app.trust.trust(app.project)
    return app


def root_of(daemon: Any, session_id: str = "pilot") -> Any:
    """The daemon-side root a pilot app is attached to.

    What `app.front.ctx` used to be. It is deliberately not reachable *through*
    the front end any more — a socket client has no `ctx`, and the AST gate in
    `test_tui_screens.py` holds the terminal to that — so a test whose subject is
    the harness asks the daemon it is attached to instead.

    A name for the default session id; the lookup is `_Daemon.held`, which owns
    the supervisor's internals.
    """
    return daemon.held(session_id)


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

    The one `ModalHost` double in the tree, so a member added to that protocol is
    added here once.
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
