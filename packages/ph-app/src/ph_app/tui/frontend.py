"""What the terminal needs from a harness, and what draws its modals.

Two protocols and no implementation. `ph_app.tui.remote.DaemonSession` is the
one that ships: after P5-14 the harness runs in the daemon, so a front end is a
protocol client and every member here is answered over a socket. The in-process
`HarnessSession` that used to live below was deleted with 2c — it mounted pH in
the terminal, which is precisely the thing that made closing the terminal end
the turn.

Kept as a protocol rather than folded into its one implementation because the
*second* one is the point: P7-07's HTML renderer consumes this, the AST gate in
`test_tui_screens.py` holds the terminal to reading nothing else, and every
member is shaped around what the screen needs rather than which service supplies
it — `providers()` and not `ctx.llm` — which is what made the move possible at
all.

Two decisions worth reading before changing this file.

**The transcript is rebuilt from `session.events`.** Not from
`derive_messages()`. The derivation is the *model's* view and compaction shadows
what it replaced; rebuilding a human's transcript from it would delete
conversation the person already read (P2-01).

**A modal is awaited in a worker, never on the message pump.**
`push_screen_wait` is illegal anywhere else, and the daemon's asks arrive on a
read loop. `ModalHost` exists so that stays a property of the caller rather than
a rule someone has to remember.

@module ph_app.tui.frontend
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Protocol

from ph.llm.types import AttachmentRef
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import Session

from ..sessions import SessionSummary
from .state import TuiState

__all__ = ["FrontSession", "ModalHost"]


class ModalHost(Protocol):
    """What the front-end needs from whatever is drawing it."""

    async def ask_approval(self, request: ApprovalRequest) -> tuple[ApprovalAnswer, str]:
        """Put the approval modal up and wait. Must be called from a worker."""
        ...

    async def ask_question(self, question: UserQuestion) -> str | None:
        """Put the ask-user modal up and wait. Must be called from a worker."""
        ...

    def state_changed(self) -> None:
        """The state was mutated; redraw when convenient."""
        ...


class FrontSession(Protocol):
    """What the terminal needs from a harness — wherever that harness is.

    **The terminal may not reach past this.** `PHTuiApp` used to read
    `front.ctx` in nine places — the command registry, the screen registry, the
    llm routes, the persistence store, the permission presets, the credentials —
    and every one of them is a service lookup that exists only when the harness
    is in *this process*. Once a front end can be a socket client (P5-14) those
    reads have no answer, so each becomes a method here and each implementation
    answers it however it can: in-process by resolving the seam, remotely by
    asking the daemon.

    So the protocol is deliberately shaped around **what the screen needs**
    rather than which service supplies it: `providers()` and not `ctx.llm`,
    `credential_held(name)` and not `ctx.credentials`. A member that handed back
    a service would be a `ctx` reach-through wearing a different name, and the
    remote implementation could not satisfy it.

    `session` stays, and is not a leak: both implementations have a real
    `Session` — in-process it is the live one, and a remote front end rebuilds
    one from the snapshot pages it already needs for the transcript. A screen is
    a fold of a log (`ScreenDefinition.build(session)`), which is exactly what
    survives the move.
    """

    state: TuiState
    config_rows: tuple[Any, ...]

    @property
    def session(self) -> Session:
        """The log this front end can fold — read-only, because a remote one
        rebuilds it from the snapshot pages rather than holding the live one."""
        ...

    @property
    def session_id(self) -> str:
        """The id on its own, because it is read on every redraw and `session` is
        not free everywhere: a remote front end rebuilds its `Session` from the
        log it has, and a sidebar asking for `.session.id` thirty times a second
        would rebuild it thirty times a second during a streaming turn."""
        ...

    def status_readings(self) -> list[StatusReading]: ...
    async def submit(self, text: str) -> None: ...

    async def attach(self, paths: Sequence[str]) -> list[AttachmentRef]:
        """Stage these files for the next prompt; return everything now staged.

        **The front end reads the bytes**, wherever it is: in process that is
        this machine's filesystem, and over a socket it is the client's, which is
        the only thing a browser tab could do and the only reading I-9 lets a
        person's own attach do. The daemon never learns a path.
        """
        ...

    async def run_command(self, line: str) -> str | None: ...

    async def shell(self, command: str) -> None:
        """Run the person's own shell command in the session's workspace.

        Nothing is returned: the command and its output reach every attached
        front end as `shell/*` events, so the person who typed it reads it back
        off the same log as everybody else — one rendering path, not two."""
        ...

    def queue(self, text: str) -> None: ...
    def cancel(self) -> None: ...
    async def flush(self) -> None: ...
    async def close(self) -> None: ...

    def attach_surfaces(self, app: Any) -> list[Callable[[], Any]]:
        """Register this front end's own verbs and screens; return their disposers."""
        ...

    def commands(self) -> list[Any]:
        """Every slash command a person may run here."""
        ...

    def screen(self, screen_id: str) -> Any:
        """One registered screen definition, or `None`."""
        ...

    def providers(self) -> list[Any]:
        """The model routes this deployment can reach."""
        ...

    async def browse_sessions(self) -> list[SessionSummary]:
        """Every session a person could open here — stored and live, in one list.

        Folded by the harness, which is the only place that can see both halves:
        the logs are its disk and which roots are mounted is its own state. A
        front end reads no session file, so one that is not on the harness's
        machine — or has no filesystem — is offered the same list.

        Replaced `sessions_directory()`, which handed back a *path* for the
        client to walk. That worked only while the two shared a machine, and it
        let them disagree about which `$PH_HOME` they meant.
        """
        ...

    async def refresh_credentials(self, names: Sequence[str]) -> None:
        """Re-read which of these the harness holds.

        On the protocol because the login screen has to *await* it before it can
        draw: `credential_held` is synchronous, so somebody has to have asked.
        Left off it, the remote implementation answered `False` for every
        credential and the picker showed a deployment with none set.
        """
        ...

    def set_preset(self, name: str) -> None:
        """Switch the permission preset. The service records it; the log carries it."""
        ...

    def credential_held(self, name: str) -> bool:
        """Whether this credential is already available to the harness."""
        ...

    def store_credential(self, name: str, value: str) -> bool:
        """Provide a secret for this session. `False` when there is nowhere to put it."""
        ...
