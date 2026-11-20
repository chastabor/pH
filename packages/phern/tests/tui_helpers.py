"""The two waits that every pilot test needs, written one time.

`running` opens an app and waits for it to attach to a daemon. The attach
happens in a worker, so a single `pause()` is not enough. `until` polls a
predicate through the pilot and does not sleep. A test therefore waits exactly
as long as the app takes, and it fails loudly and does not hang when the app
never reaches the state.

The module also holds the doubles (`StubHost`, `StubApp`, `StubClient`,
`WorkingApp`), the builders of a log to revise, and `TranscriptApp`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

import anyio
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.pilot import Pilot

from ph.llm.types import ContextForm, Message, PluginSource, create_user_message
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.user_questions import UserQuestion
from ph.session import Session, SessionEvent, SurfaceIntent, editable_message, rewrite
from ph.testing import (
    assistant_payload,
    log_event,
    revision_record,
    tool_result_payload,
    user_payload,
)
from ph_app.daemon.supervisor import Root
from ph_app.payloads import AskKey, ProfileDecision
from ph_app.tui.app import PHTuiApp
from ph_app.tui.state import ChatItem, Surface, TuiState
from ph_app.tui.themes import DEFAULT_THEME, fallback_variables, load_catalog
from ph_app.tui.widgets.transcript import TranscriptView

if TYPE_CHECKING:
    # This import is for an annotation only, and `daemon_helpers` has a cost:
    # importing it composes a profile from disk at module scope. The three
    # modules that use these helpers without the daemon fixture should not pay
    # that cost for a name that only the type checker reads.
    from daemon_helpers import Daemon


MakeApp = Callable[..., PHTuiApp]
"""`make_tui_app(...)` → an app wired for a test, from `app_fixtures`.

The type is beside the helpers and not in each suite. Three suites had written
it out, which is the sprawl that the docstring of `ph.testing.MountProfile`
argues against one package up. It is not in `ph.testing`, because `PHTuiApp`
belongs to `ph_app` and ph-core may not name it.
"""


def tui_app(
    *,
    home: Path,
    project: Path | None = None,
    trusted: bool = True,
    **overrides: object,
) -> PHTuiApp:
    """One `PHTuiApp` wired for a test, built in one place.

    There are two callers: the async `make_tui_app` fixture, and the snapshot
    suite, which cannot use a fixture because `snap_compare` drives its own
    event loop. They were two copies of the construction contract of the app,
    and the change that added `daemon_argv` had to edit both.

    `spawn=False`: a test must never start a supervisor. The socket of the
    fixture is already answering, so `ensure_daemon` returns without spawning.
    A regression that tried to spawn fails as a named `DaemonAbsent`.

    `trusted` writes the file that the *daemon* reads. `session/new` is what
    enforces trust, so the mount can happen only because the directory is
    trusted here. `project` is set before the trust is written, because trust is
    keyed on that path.
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
        # snapshot that was taken here therefore embedded the checkout of the
        # developer, and it could never match CI, whose cwd differs.
        app.project = project
    if trusted:
        app.trust.trust(app.project)
    return app


def root_of(daemon: Daemon, session_id: str = "pilot") -> Root:
    """The root on the daemon side that a pilot app is attached to.

    This is what `app.front.ctx` used to be. The root is deliberately not
    reachable *through* the front end any more. A socket client has no `ctx`,
    and the AST gate in `test_tui_screens.py` holds the terminal to that. So a
    test whose subject is the harness asks the daemon that the app is attached
    to.

    This function is a name for the default session id. The lookup is
    `_Daemon.held`, which owns the internals of the supervisor.
    """
    return daemon.held(session_id)


async def until[T](pilot: Pilot[T], predicate: Callable[[], bool], *, tries: int = 400) -> None:
    """Pause until the predicate holds, or give up with an `AssertionError`.

    The function is generic in the type parameter of the pilot, and it does not
    take `Pilot[object]`. `Pilot` is invariant, so a `Pilot[None]` is not a
    `Pilot[object]`. `run_test()` gives a `Pilot[None]` to an app that returns
    nothing, and every caller of that shape was refused."""
    for _ in range(tries):
        if predicate():
            return
        await pilot.pause()
    raise AssertionError("the app never reached the expected state")


@asynccontextmanager
async def running(
    app: PHTuiApp, *, size: tuple[int, int] = (80, 24)
) -> AsyncIterator[tuple[PHTuiApp, Pilot[str | None]]]:
    """`size` is for a test that needs the transcript to scroll. The default is
    the default of Textual, which is tall enough to hold a short conversation
    whole.

    The default is not `None`, which `run_test` reads as "detect the real
    terminal". A test whose assertions depend on the viewport must not depend on
    the window that it ran in.
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

    This is the one `ModalHost` double in the tree, so a member that is added to
    that protocol is added here one time.
    """

    def __init__(self) -> None:
        self.approvals: list[ApprovalRequest] = []
        self.questions: list[UserQuestion] = []
        self.withdrawn: list[AskKey] = []
        self.redraws = 0

    async def ask_approval(
        self, request: ApprovalRequest, *, ask: AskKey | None = None
    ) -> tuple[ApprovalAnswer, str]:
        # The type is `ApprovalAnswer` and not `tuple[str, str]`. `ModalHost`
        # promises the narrowed answer. It is a Protocol so that a double whose
        # member drifted fails here and not at the first modal.
        self.approvals.append(request)
        return "allowed-once", ""

    async def ask_question(
        self, question: UserQuestion, *, ask: AskKey | None = None
    ) -> str | None:
        self.questions.append(question)
        return "42"

    async def ask_profile(
        self, name: str, listing: str, *, ask: AskKey | None = None
    ) -> ProfileDecision:
        return "later"

    def withdraw_ask(self, ask: AskKey, *, reason: str = "") -> None:
        self.withdrawn.append(ask)

    def state_changed(self, surfaces: Surface = Surface.ALL) -> None:
        self.redraws += 1


class StubApp:
    """An `AppSurface` for a front end that is driven with no terminal.

    This is the one `AppSurface` double in the tree, for the same reason as
    `StubHost` and beside it: a member that is added to that protocol is added
    here one time.

    It has all three members, and not only the one that a given test reaches. A
    double that stubs only what the current path calls stops being a double when
    the path changes, and it does so silently. That has happened: this class
    replaced `object()` and a four-line class with a single method, and both of
    those passed because `_front` returned `Any`.
    """

    def __init__(self) -> None:
        self.ran: list[str] = []

    def run_worker(self, work: Coroutine[Any, Any, Any]) -> None:
        # The work is closed and not started. There is no app to own it here,
        # and a coroutine that is left unawaited warns.
        work.close()

    def add_binding(self, binding: Binding) -> Callable[[], Any]:
        return lambda: None

    async def run_action(self, action: str) -> None:
        self.ran.append(action)


class StubClient:
    """A `DaemonClient` for a front end with no daemon behind it.

    It has **all four members that `DaemonSession` reaches**, and not only the
    one that a given test needs. The docstring of `StubApp` states the rule and
    the reason: a double that stubs only the current path stops being a double
    when the path changes, and it does so silently. The calls answer with
    nothing and record nothing, because no test asks what was sent.
    """

    def __init__(self) -> None:
        self.closed = anyio.Event()

    async def prompt(self, session_id: str, text: str) -> None:
        return None

    async def call(self, verb: Any, params: Any = None) -> Any:  # noqa: ANN401
        return None

    async def mutate(self, *arguments: Any, **options: Any) -> Any:  # noqa: ANN401
        return None


class WorkingApp(StubApp):
    """A `StubApp` that *runs* the work that it is given.

    `StubApp` closes the coroutine. That is right for a test that asserts on
    what was asked for, and wrong for a test whose subject is the effect of the
    work. A `queue()` whose frame is never sent cannot be counted at the other
    end.

    The task group belongs to the caller. So the `async with` that owns the
    group bounds the work, and the work does not outlive the test that started
    it.
    """

    def __init__(self, tasks: anyio.abc.TaskGroup) -> None:
        super().__init__()
        self.tasks = tasks

    def run_worker(self, work: Coroutine[Any, Any, Any]) -> None:
        async def run() -> None:
            await work

        self.tasks.start_soon(run)


# ------------------------------------------------------- a log to revise --


def stand_in(text: str, *, form: ContextForm = "compaction", summary: str | None = None) -> Message:
    """Text that a plugin wrote in place of conversation: a summary (`compaction`,
    a stand-in) or, with another form, the preview of a relocated paste."""
    return create_user_message(
        content=[{"type": "text", "text": text}],
        source=PluginSource(plugin="probe", form=form, summary=summary),
    )


def revisable(output: str) -> tuple[Session, SessionEvent, SessionEvent, SessionEvent]:
    """user · assistant(call) · tool/result printing `output`, for a test to revise."""
    session = Session("diffs")
    asked = log_event(
        session, "user/message", user_payload("read a.md", "m1"), SurfaceIntent("append")
    )
    call = {"type": "tool-call", "id": "c1", "name": "read", "arguments": '{"path": "a.md"}'}
    replied = log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=[{"type": "text", "text": "reading a.md"}, call]),
        SurfaceIntent("append"),
    )
    result = log_event(
        session, "tool/result", tool_result_payload(output, "m3", "c1"), SurfaceIntent("append")
    )
    return session, asked, replied, result


def rewritten(session: Session, result: SessionEvent, text: str) -> SessionEvent:
    """`result` with its output rewritten in place to `text`, as the overflow clip
    rewrites one, with its record."""
    message = editable_message(result)
    message["content"][0]["content"] = [{"type": "text", "text": text}]
    return rewrite(session, result, message, revision_record("offload/spilled", {"callId": "c1"}))


class TranscriptApp(App[None]):
    """A transcript and nothing else, with its rows fed to it directly.

    The snapshot tests draw it and the gutter tests measure it. With no model
    behind it there is no timing, no token order and no session id in the frame.
    So a diff, or a position, points at the rows.
    """

    CSS = """
    Screen { background: $ph-background; color: $ph-foreground; }
    """

    def __init__(self, items: list[ChatItem], theme: str = DEFAULT_THEME) -> None:
        super().__init__()
        self.items = items
        self._theme_name = theme

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def compose(self) -> ComposeResult:
        yield TranscriptView(id="transcript")

    async def on_mount(self) -> None:
        load_catalog().install(self)
        self.theme = self._theme_name
        await self.query_one(TranscriptView).sync(self.items)


def rows_of(*items: ChatItem) -> list[ChatItem]:
    """The rows that a `TuiState` which holds `items` shows. They go through the
    state, so the rows of a test are folded and filtered as the rows of the app
    are."""
    state = TuiState()
    for item in items:
        state.add(item)
    return state.visible_items()
