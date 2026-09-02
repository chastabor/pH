"""The bridge between the harness and the screen.

Kept apart from `PHTuiApp` on purpose. Everything interesting about a front-end
is here — which seams it answers, how a prompt becomes a turn, what a resume
rebuilds from — and all of it is testable against a `ModalHost` stub with no
terminal in the loop. The app is then only widgets and keys.

Three decisions worth reading before changing this file.

**The transcript is rebuilt from `session.events`.** Not from
`derive_messages()`. The derivation is the *model's* view and compaction shadows
what it replaced; rebuilding a human's transcript from it would delete
conversation the person already read (P2-01).

**A modal is awaited in a worker, never on the message pump.** The answerers
run inside `agent.run()`, which the app drives in a Textual worker, so
`push_screen_wait` is legal there. `ModalHost` exists so that stays a property
of the caller rather than a rule someone has to remember.

**pH records the approval; the front-end only decides.** `ApprovalService`
appends `approval/asked` and `approval/decided` around the waterfall. A modal
that logged its own answer would give one decision two authors.

@module ph_app.tui.frontend
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any, Protocol

from ph.agent.types import AgentCancelCause, AgentOptions
from ph.cordis import Context, Profile
from ph.llm.types import create_user_message
from ph.seams.approval import ApprovalAnswer, ApprovalRequest
from ph.seams.tui_status import StatusReading
from ph.seams.user_questions import UserQuestion
from ph.session import Session, SessionEvent

from ..runtime import mounted
from .adapter import TuiEventAdapter
from .state import TuiState

__all__ = ["HarnessSession", "ModalHost", "open_harness"]

log = logging.getLogger("ph_app.tui.frontend")


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


@dataclass(slots=True)
class HarnessSession:
    """A mounted pH, one session, one agent — plus the screen's view of it."""

    ctx: Context
    session: Session
    agent: Any
    state: TuiState
    adapter: TuiEventAdapter
    host: ModalHost
    config_rows: tuple[Any, ...] = ()
    """The composed configuration, as `Profile.dump()` reports it. Kept so the
    front-end can read what the profile declared — which credentials exist, for
    instance — instead of maintaining a second list of what pH supports."""
    _stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    _disposers: list[Callable[[], Any]] = field(default_factory=list)

    def status_readings(self) -> list[StatusReading]:
        """What the rows want the footer to say (P4-04's budget, and later ones).

        Here rather than in `app.py`, which is "the terminal": resolving a seam
        and asking it for a projection is harness-shaped work, and doing it
        inline in the frame callback put a service lookup on every redraw with
        nowhere to test it from.
        """
        registry = self.ctx.get("tui_status")
        return [] if registry is None else registry.readings(self.session)

    # ----------------------------------------------------------------- turns --

    async def submit(self, text: str) -> None:
        """Run one prompt to idle.

        Called from a worker: the answerers this session registered will push
        modals, and they can only do that from inside one. `state.status` is set
        here because this is the one place a turn is known to be running.
        """
        self.state.status = "running"
        self.host.state_changed()
        try:
            await self.agent.prompt(text)
        finally:
            self.state.status = "idle"
            self.host.state_changed()

    async def run_command(self, line: str) -> str | None:
        """Dispatch a `/name` line. Spends no model turn, and says so in the log.

        A command is the human's verb, not the model's: routing it through a
        prompt would both cost a turn and make the log claim the model decided
        something the person decided.
        """
        shown: str | None = await self.ctx.commands.dispatch(
            line, scope=self.agent.ctx, session=self.session, agent=self.agent
        )
        return shown

    def queue(self, text: str) -> None:
        """Add to the inbox without waiting for idle — a follow-up mid-turn.

        `followup` rather than `steer`: a person typing while the model works is
        adding to the conversation, not interrupting this step. Steering is a
        separate, explicit act.
        """
        self.agent.followup(_user_text(text))
        self.state.queued += 1
        self.host.state_changed()

    def cancel(self) -> None:
        """Interrupt the running turn, keeping what the user already queued."""
        self.agent.cancel(AgentCancelCause(kind="user"), keep_inbox=True)

    async def flush(self) -> None:
        await self.ctx.sessions.flush(self.session)

    # ------------------------------------------------------------- lifecycle --

    async def close(self) -> None:
        """Flush the log, unregister the front-end, unmount the harness.

        Flush is part of closing rather than a separate call the app must
        remember to pair with it: disposal does not flush, and a session lost
        on exit is the worst possible way to learn that.
        """
        await self.flush()
        for dispose in reversed(self._disposers):
            try:
                dispose()
            except Exception:
                log.debug("ph_app.tui: a front-end disposer failed", exc_info=True)
        self._disposers.clear()
        await self._stack.aclose()


async def open_harness(
    profile: Profile,
    *,
    host: ModalHost,
    provider: str,
    model: str,
    session_id: str | None = None,
    resume: str | None = None,
) -> HarnessSession:
    """Mount a profile, attach the front-end, and return the live session.

    `resume` replays a stored session into the transcript before the first
    prompt. The replay reads `session.events` — see the module docstring.
    """
    stack = AsyncExitStack()
    ctx = await stack.enter_async_context(mounted(profile))

    if resume is not None:
        from ph.persistence.jsonl import resume_session

        session = await resume_session(ctx, resume)
    else:
        session = ctx.sessions.create(session_id)

    state = TuiState()
    adapter = TuiEventAdapter(state=state, tools=ctx.get("tools"))
    if resume is not None:
        adapter.replay(session)

    agent = ctx.agents.create(session, AgentOptions(provider=provider, model=model))
    state.provider = provider
    state.model = model

    front = HarnessSession(
        ctx=ctx,
        session=session,
        agent=agent,
        state=state,
        adapter=adapter,
        host=host,
        config_rows=tuple(ctx.mount.profile.dump()),
        _stack=stack,
    )
    front._disposers.extend(_attach(ctx, front))
    return front


def _attach(ctx: Context, front: HarnessSession) -> list[Callable[[], Any]]:
    """Register the three listeners a front-end owns, and return their disposers."""

    def observe(_session: Session, event: SessionEvent) -> None:
        try:
            front.adapter.apply(event)
        except Exception:
            log.exception("ph_app.tui: the adapter refused an event")
        front.host.state_changed()

    async def answer_approval(request: ApprovalRequest, _next: Any = None) -> ApprovalAnswer:
        outcome, reason = await front.host.ask_approval(request)
        if reason:
            # "No, use the existing helper" redirects a turn where a bare
            # refusal only stops it. It is delivered as what it is — user input
            # at the next step boundary — rather than as a new event type: the
            # log's vocabulary is fixed (`KNOWN_SESSION_EVENT_TYPES`), and a
            # front-end inventing a type writes a log this build cannot read.
            front.agent.steer(_user_text(reason))
        return outcome

    async def answer_question(question: UserQuestion, _next: Any = None) -> str | None:
        return await front.host.ask_question(question)

    # The session's own feed, not the store-wide `session/event` bus: a
    # subagent's events belong to its own transcript, and subscribing here
    # means never receiving them rather than receiving and discarding them.
    disposers: list[Callable[[], Any]] = [front.session.observe(observe)]
    approval = ctx.get("approval")
    if approval is not None:
        disposers.append(approval.register_answerer(answer_approval))
    questions = ctx.get("user_questions")
    if questions is not None:
        disposers.append(questions.register_answerer(answer_question))
    return disposers


def _user_text(text: str) -> Any:
    return create_user_message(content=[{"type": "text", "text": text}], source={"kind": "user"})
