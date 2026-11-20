"""What the terminal needs from a harness, and what draws its modals.

This module holds two protocols and no implementation.
`ph_app.tui.remote.DaemonSession` is the implementation that ships. After
P5-14 the harness runs in the daemon, so a front end is a protocol client, and
every member here is answered over a socket. The in-process `HarnessSession`
that used to be below was deleted with 2c. It mounted pH in the terminal, and
that is exactly what made closing the terminal end the turn.

`FrontSession` stays a protocol, and is not folded into its one implementation,
because the *second* implementation is the purpose:

- The HTML renderer of P7-07 consumes this protocol.
- The AST gate in `test_tui_screens.py` holds the terminal to reading nothing
  else.
- Every member is shaped around what the screen needs, and not around which
  service supplies it: `models()` and not `ctx.llm`. That is what made the move
  possible at all.

Read these two decisions before you change this file.

**The transcript is rebuilt from `session.events`**, and not from
`derive_messages()`. The derivation is the view of the *model*, and compaction
shadows what it replaced. A transcript for a human that was rebuilt from it
would delete conversation that the user already read (P2-01).

**A modal is awaited in a worker, never on the message pump.**
`push_screen_wait` is illegal anywhere else, and the asks of the daemon arrive
on a read loop. `ModalHost` exists so that this stays a property of the caller,
and is not a rule that someone has to remember.

@module ph_app.tui.frontend
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Protocol

from ph.llm.types import AttachmentRef
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.commands import CommandDefinition
from ph.seams.models import ModelChoice, ModelEntry
from ph.seams.permission_presets import PresetName, PresetSchema
from ph.seams.tui_screens import ScreenDefinition
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import Session

from ..payloads import AskKey, ProfileDecision, RootDescription
from ..sessions import SessionSummary
from .screens import AppSurface
from .state import Surface, TuiState

__all__ = ["FrontSession", "ModalHost"]


class ModalHost(Protocol):
    """What the front end needs from whatever draws it."""

    async def ask_approval(
        self, request: ApprovalRequest, *, ask: AskKey | None = None
    ) -> tuple[ApprovalAnswer, str]:
        """Put the approval modal up and wait. Must be called from a worker.

        `ask` is the key by which `withdraw_ask` takes the modal down. It is
        optional so that an in-process caller with no wire ask can leave it out.
        """
        ...

    async def ask_question(
        self, question: UserQuestion, *, ask: AskKey | None = None
    ) -> str | None:
        """Put the ask-user modal up and wait. Must be called from a worker."""
        ...

    async def ask_profile(
        self, name: str, listing: str, *, ask: AskKey | None = None
    ) -> ProfileDecision:
        """Show how the named profile of a session changed, and ask which version
        the session runs on (S6). Must be called from a worker."""
        ...

    def withdraw_ask(self, ask: AskKey, *, reason: str = "") -> None:
        """Take down a modal that can no longer be answered here (H1).

        The daemon asks *every* attached front end and keeps the first answer.
        It then publishes `ask.settled` to the others. Nothing used to consume
        that notice. The modal of the second terminal stayed up, and the user
        who answered it was told nothing. Their answer went to a daemon that
        had already decided, and it was discarded in silence. The modal that
        they had just dismissed was their only evidence that anything happened.

        This is sync, and safe to call for an ask that this host has never seen:
        a client that was not asked still receives the notice. `reason` is what
        the user is told. An empty `reason` says nothing, for a caller that has
        a better sentence of its own.
        """
        ...

    def state_changed(self, surfaces: Surface = Surface.ALL) -> None:
        """The state was mutated. Redraw when convenient."""
        ...


class FrontSession(Protocol):
    """What the terminal needs from a harness, wherever that harness is.

    **The terminal may not reach past this protocol.** `PHTuiApp` used to read
    `front.ctx` in nine places: the command registry, the screen registry, the
    llm routes, the persistence store, the permission presets, the credentials.
    Each one is a service lookup that exists only when the harness is in *this
    process*. When a front end can be a socket client (P5-14), those reads have
    no answer. So each one becomes a method here, and each implementation
    answers it in the way that it can: in process by resolving the seam,
    remotely by asking the daemon.

    The protocol is therefore deliberately shaped around **what the screen
    needs**, and not around which service supplies it: `models()` and not
    `ctx.llm`, `refresh_credentials()` and not `ctx.credentials`. A member that
    returned a service would be a `ctx` reach-through under a different name,
    and the remote implementation could not satisfy it.

    `session` stays, and it is not a leak. Both implementations have a real
    `Session`. In process it is the live one. A remote front end keeps a mirror
    of the log, which it builds from the events that it already needs for the
    transcript. A screen is a fold of a log (`ScreenDefinition.build(session)`),
    and that is exactly what survives the move.
    """

    state: TuiState

    @property
    def session(self) -> Session:
        """The log that this front end can fold. It is read-only, because a remote
        front end holds a mirror of the log and not the live one."""
        ...

    @property
    def session_id(self) -> str:
        """The id on its own, because it is read on every redraw and `session` is
        not free everywhere. A remote front end used to rebuild its `Session`
        from the log on every read. A sidebar that asked for `.session.id`
        thirty times a second then rebuilt it thirty times a second during a
        streaming turn. `DaemonSession` now keeps its mirror incrementally, and
        the id stays separate so that the sidebar never asks for the whole log."""
        ...

    def status_readings(self) -> list[StatusReading]: ...
    async def submit(self, text: str) -> None: ...

    async def attach(self, paths: Sequence[str]) -> list[AttachmentRef]:
        """Stage these files for the next prompt, and return everything now staged.

        **The front end reads the bytes**, wherever it is. In process, that is
        the filesystem of this machine. Over a socket, it is the filesystem of
        the client. That is the only thing that a browser tab could do, and the
        only reading that I-9 allows for an attach by the user. The daemon
        never learns a path.
        """
        ...

    async def shell(self, command: str, *, surface: bool = False) -> None:
        """Run a shell command of the user in the workspace of the session.

        `surface` is true for `!` and false for `!!`. `ph_app.shell` explains
        the difference.

        Nothing is returned. The command and its output reach every attached
        front end as `shell/*` events. The user who typed the command therefore
        reads it from the same log as everybody else: one rendering path, and
        not two."""
        ...

    def queue(self, text: str) -> None: ...
    def cancel(self) -> None: ...
    async def close(self) -> None: ...

    def attach_surfaces(self, app: AppSurface) -> list[Callable[[], None]]:
        """Attach the verbs and screens of this front end to `app`. Return their
        disposers."""
        ...

    def commands(self) -> list[CommandDefinition]:
        """The slash commands that this front end offers: one for each screen that
        it can draw, and those of the daemon. The verbs of the terminal belong
        to the app (`PHTuiApp.commands`), so they answer with no front end at
        all.

        The return type is definitions and not `Any`. The palette reads `name`,
        `argument_hint` and `summary` from them. The picker that renders them
        used to take `Iterable[Any]` to accept the answer. A renamed field on
        the seam then reached the terminal as an `AttributeError` at the moment
        when somebody pressed `/`. This whole layer is typed so that such a
        failure moves forward to import time.
        """
        ...

    def screen(self, screen_id: str) -> ScreenDefinition | None:
        """One registered screen definition, or `None`."""
        ...

    @property
    def diverged(self) -> bool:
        """Whether the view of the log that this front end holds has stopped
        matching the log of the harness.

        It is on the protocol and is not probed for. Everything that the
        terminal reads from its harness must be something that a *remote*
        harness can answer, which is the rule that
        `test_the_terminal_never_reaches_past_the_front_session` enforces. A
        `getattr(front, "diverged", False)` passed that test precisely because
        it is not an attribute access.
        """
        ...

    def models(self) -> list[ModelEntry]:
        """The models this session's profile lists, the default first."""
        ...

    @property
    def arrival(self) -> str | None:
        """What the user should know about this session as they arrive, told one
        time: a named profile that moved past the version that the session runs
        on (S6). `None` when there is nothing to tell."""
        ...

    async def choose_model(self, choice: ModelChoice) -> RootDescription:
        """Run this session on `choice` from its next request, or raise the refusal."""
        ...

    async def browse_sessions(self, *, cwd: str | None = None) -> list[SessionSummary]:
        """Every session that a user could open here, stored and live, in one list.

        `cwd` narrows the list to the sessions whose header recorded that
        directory. `None` means everything, which is what `/sessions` asks for.
        The filter runs on the harness, before its own row limit, so a
        directory that was worked in long ago is still found on a machine with
        newer work elsewhere.

        The harness folds the list, because only the harness can see both
        halves: the logs are on its disk, and which roots are mounted is its
        own state. A front end reads no session file. A front end that is not
        on the machine of the harness, or that has no filesystem, is therefore
        offered the same list.

        This replaced `sessions_directory()`, which returned a *path* for the
        client to walk. That worked only while the two shared a machine, and it
        let them disagree about which `$PH_HOME` they meant.
        """
        ...

    async def refresh_credentials(self) -> Mapping[str, bool]:
        """Read again which credentials the harness names, and which of them it
        holds.

        This is on the protocol because the login screen must *await* it before
        it can draw. When it was not on the protocol, the only question was a
        synchronous one about a single name. The remote implementation answered
        that with `False` for every credential, and the picker showed a
        deployment with none set.

        It takes no names, because to work out which credentials exist is the
        job of the harness: it composed the profile. The front end used to hold
        that profile, as a `config_rows` member on this protocol, only so that
        a modal could walk it for `apiKeyEnv` keys. That was a service artifact
        that reached a screen through a member shaped like the need of a screen.

        It returns the whole map, because the picker draws every row at the
        same time.
        """
        ...

    def set_preset(self, name: PresetName) -> None:
        """Switch the permission preset. The service records it, and the log carries it."""
        ...

    async def presets(self) -> list[PresetSchema]:
        """The permission postures, with the live one marked."""
        ...

    def store_credential(self, name: str, value: str) -> None:
        """Provide a secret for this session. The harness reports a refusal in its
        own log, because the answer arrives after this returns."""
        ...
