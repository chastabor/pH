"""Who the daemon asks when a turn needs a person (P5-13).

Under the daemon there was nobody to ask. `ApprovalService` runs its waterfall
with an `inner` that answers `unavailable` when no answerer took the prompt, and
no answerer was ever registered on a root. A gated tool call under
`phern daemon` was therefore **denied**. No person said no: nobody could be
asked. Both doctors still reported the deployment as healthy. This module is the
piece that was missing.

**One desk for each root, and many front ends.** Attach is not exclusive.
Several UIs may watch one session: a terminal, a browser tab,
`phern agents attach`. Each has its own private composer, and the log is shared.
So an ask goes to *every* attached front end, the **first answer wins**, and the
others are told `ask.settled`. pH still records exactly one decision, because pH
records it. A front end only decides.

**Nobody attached is not a denial.** The ask stays here and the answerer waits
on it. `join` poses it again to whoever arrives. If a gate became a denial
because a terminal closed, the actual answer of the human would be lost.

The ask is posed again **from memory, so that lasts only as long as the daemon
does**. That is all that it promises, and it is no longer a silent edge. A
restart cannot carry the ask across, because a coroutine inside a turn was
waiting on it, and both are gone. The restart **settles** the ask instead.
Repair writes the `approval/decided` that the crash never reached (P5-13). The
log then holds no ask that nobody can answer, and the transcript says that the
person was still being asked when the harness stopped.

So the guarantee is bounded and stated, not absent. Within one daemon, an ask
outlives every front end and waits for whoever arrives. An ask does not survive
a restart. It is closed truthfully, nothing ran, and the work resumes in the way
that every interrupted turn does: the model reads `TOOL_NOT_STARTED` and asks
again, and the ask goes to whoever is attached by then.

**A slow front end does not hold up the others.** Each delivery is a task of its
own. `_Connection.ask` queues into a bounded outbox and waits for room there, so
a client that cannot keep up delays only its own delivery. The ask stays pending
for the others. A front end that cannot be told `ask.settled` stops being asked
(`_answer`).

@module ph_app.daemon.frontend
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import anyio
from anyio.abc import TaskGroup

from ph.cordis import Disposer
from ph.keys import APPROVAL, USER_QUESTIONS
from ph.llm.types import user_text
from ph.seams.approval import ApprovalAnswer, ApprovalRequest, answer_from_wire
from ph.seams.user_questions import UserQuestion
from ph.session_profile import ProfileChange, listing

if TYPE_CHECKING:
    # `supervisor` imports this module for `AskDesk`. The import in the other
    # direction therefore exists only for the checker.
    from .supervisor import Root

from ..payloads import (
    ApprovalAsk,
    ApprovalAskReply,
    AskSettledNotice,
    ProfileAsk,
    ProfileAskReply,
    ProfileDecision,
    QuestionAsk,
    QuestionAskReply,
)
from ..protocol import DaemonError

__all__ = ["AskDesk"]

log = logging.getLogger("ph_app.daemon.frontend")


class FrontEnd(Protocol):
    """A connection that has said it can answer for a person."""

    async def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...
    def notify(self, method: str, params: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class PendingAsk:
    """One question in flight, and the answer that the first front end to reply
    leaves.

    `tasks` is the group of the fan-out itself. It is kept so that a front end
    which arrives *while* the question is open can also be asked. That makes
    "nobody was attached" a delay and not an outcome.
    """

    ask_id: str
    method: str
    params: dict[str, Any]
    answered: anyio.Event
    answer: dict[str, Any] | None = None
    tasks: TaskGroup | None = None


@dataclass(slots=True)
class AskDesk:
    """The answerers a root registers, and the front ends they reach."""

    root: Root
    asks: dict[str, PendingAsk] = field(default_factory=dict)

    @property
    def front_ends(self) -> set[FrontEnd]:
        """The attached connections that can answer. They belong to the id, not to
        this desk (`Audience.front_ends`). A root that is started again has a new
        desk, and whoever answered for the last desk answers for this one."""
        return self.root.audience.front_ends

    # ------------------------------------------------------------- wiring --

    def attach(self) -> list[Disposer]:
        """Register both answerers on the context of the root. Returns their
        disposers.

        `register_answerer` is sugar over `ctx.on(...)`, so the answerers unwind
        with the scope that the root gives them. That is the scope of the root
        itself, because this runs outside the `apply` of any row.
        """
        disposers: list[Disposer] = []
        approval = self.root.ctx.get(APPROVAL)
        if approval is not None:
            disposers.append(approval.register_answerer(self.answer_approval))
        questions = self.root.ctx.get(USER_QUESTIONS)
        if questions is not None:
            # This registration passes `reachable`, and the registration on the
            # approval seam does not, because the two failure modes differ. An
            # approval that cannot be answered must fail closed and *deny*. A
            # question that cannot be answered must not be asked at all. A daemon
            # with no front end attached is exactly the case that the seam
            # declines to log, and this probe is how the seam finds out.
            disposers.append(
                questions.register_answerer(
                    self.answer_question, reachable=lambda: bool(self.front_ends)
                )
            )
        return disposers

    def join(self, who: FrontEnd) -> None:
        """Add a front end that can answer. Anything already pending goes to it
        too.

        Because the ask is posed again on arrival, "nobody was attached" is a
        delay and not an outcome. The turn waited, the person arrived, and the
        question is still the one that the model asked.
        """
        if who in self.front_ends:
            # **A second attach is an expected call.** `session/attach` is how a
            # client resumes, and `_attach` already guards `subscribe` against
            # it. Without the same guard here, a re-attach starts a *second*
            # delivery of every open ask to the same connection. That is a second
            # frame, a second slot of the in-flight budget of that client, and a
            # second modal in front of one person, whose two answers then race
            # for one question.
            return
        self.front_ends.add(who)
        for pending in list(self.asks.values()):
            if pending.tasks is not None:
                pending.tasks.start_soon(self._deliver, who, pending)

    @property
    def waiting(self) -> bool:
        """Whether this root is waiting for a person.

        `Root.status` reads this, and that lets release take the root. A root
        that waits for a human is idle in every sense that matters. To hold a
        process for somebody who closed their laptop is the failure that P5-05
        exists to prevent.
        """
        return bool(self.asks)

    # ---------------------------------------------------------- answerers --

    async def answer_approval(
        self,
        request: ApprovalRequest,
        _next: object = None,
    ) -> ApprovalAnswer:
        """The answerer of `ctx.approval`, over the socket.

        The desk names the ask on the wire (`asked`). The call id does not name
        it, because a `refine` ask has no call id. The call id and the tool name
        still travel, in `request`.
        """
        result = ApprovalAskReply.model_validate(
            await self._ask(
                ApprovalAsk(session_id=self.root.id, ask_id=self._name(), request=request)
            )
        )
        if reason := result.reason:
            # "No, use the existing helper" redirects a turn, where a bare refusal
            # only stops it. The reason is delivered as what it is, user input at
            # the next step boundary, and not as a new event type. The vocabulary
            # of the log is fixed, and a front end that invents an event type
            # writes a log that this build cannot read.
            self.root.agent.steer(user_text(reason))
        return answer_from_wire(result.answer)

    async def answer_question(
        self,
        question: UserQuestion,
        _next: object = None,
    ) -> str | None:
        """The answerer of `ctx.user_questions`, over the socket.

        The desk names the ask on the wire (`asked`), as it does for an approval.
        The `ask_id` of the question itself, which is the `askId` of the log,
        still travels, in `question`.
        """
        result = QuestionAskReply.model_validate(
            await self._ask(
                QuestionAsk(session_id=self.root.id, ask_id=self._name(), question=question)
            )
        )
        return result.answer

    async def decide_profile(self, change: ProfileChange) -> ProfileDecision:
        """Ask whether this root takes the new version of its named profile (S6).

        Unlike the two methods above, this is not the answerer of a seam: nothing
        in the session asked. The root was held at its start because a profile
        that belongs to a person moved since the root began. So the root waits
        here before anything runs, for whoever attaches, as every ask does. The
        supervisor records the answer.
        """
        result = ProfileAskReply.model_validate(
            await self._ask(
                ProfileAsk(
                    session_id=self.root.id,
                    ask_id=self._name(),
                    name=change.now.name,
                    listing="\n".join(listing(change)),
                )
            )
        )
        return result.decision

    def settle(self, method: str, answer: dict[str, Any]) -> None:
        """Answer every open ask of `method` from outside the desk, and remove it
        from the screen of every front end.

        This is for a question that something other than a person answered:
        `session/adopt` settles the profile question on which a root was held
        (S6). Every front end is told `ask.settled`, as the front ends that did
        not give the answer are for any other ask. So no modal stays up that
        asks about a decision that is already made.
        """
        for pending in list(self.asks.values()):
            if pending.method == method and not pending.answered.is_set():
                self._answer(pending, answer)

    def _answer(
        self, pending: PendingAsk, answer: dict[str, Any], *, by: FrontEnd | None = None
    ) -> None:
        """Take `answer` as the one decision. Tell every front end except the one
        that gave it that the question is settled. A front end that cannot be
        told stops being asked."""
        pending.answer = answer
        pending.answered.set()
        settled = AskSettledNotice(session_id=self.root.id, ask_id=pending.ask_id)
        for other in list(self.front_ends):
            if other is by:
                continue
            try:
                other.notify(settled.METHOD, settled.to_wire())
            except Exception:
                self.front_ends.discard(other)

    # ------------------------------------------------------------ the ask --

    def _name(self) -> str:
        audience = self.root.audience
        audience.asked += 1
        return f"ask-{audience.asked}"

    async def _ask(self, ask: ApprovalAsk | QuestionAsk | ProfileAsk) -> dict[str, Any]:
        """Put one question to every front end and wait for the first answer.

        The fan-out has its own task group and does not borrow the group of the
        root. Every delivery belongs to *this* question and ends with it. To
        cancel the group when the answer arrives is therefore exactly the right
        lifetime. The other front ends are no longer waited on, and
        `ask.settled` tells them why.
        """
        pending = PendingAsk(
            ask_id=ask.ask_id,
            method=ask.METHOD,
            params=ask.to_wire(),
            answered=anyio.Event(),
        )
        self.asks[ask.ask_id] = pending
        # `Root.status` reads `waiting` from here, and the status of the agent
        # itself does not move. So the root reports the change itself: while it
        # waits for a person, it may be released and no longer holds the daemon.
        self.root.recheck()
        try:
            async with anyio.create_task_group() as tasks:
                pending.tasks = tasks
                for who in list(self.front_ends):
                    tasks.start_soon(self._deliver, who, pending)
                await pending.answered.wait()
                tasks.cancel_scope.cancel()
        finally:
            pending.tasks = None
            self.asks.pop(ask.ask_id, None)
            self.root.recheck()
        return pending.answer or {}

    async def _deliver(self, who: FrontEnd, pending: PendingAsk) -> None:
        try:
            answer = await who.ask(pending.method, pending.params)
        except DaemonError as refused:
            if refused.reason == "invalid_params":
                # The front end could not *read* this ask: its build does not
                # know the shape. That says nothing about the next ask of a
                # different kind. The reason is named and not inferred from the
                # message, which is what `InvalidParams` is for. The front end
                # stays joined. The warning is loud enough to find, because a
                # front end that silently answers nothing is how the modals of a
                # person stop appearing.
                log.warning(
                    "ph_app.daemon: a front end could not read %s — %s", pending.method, refused
                )
                return
            log.debug("ph_app.daemon: a front end refused %s — %s", pending.method, refused)
            self.front_ends.discard(who)
            return
        except Exception:
            # A front end that cannot answer stops being asked: a dead socket, or
            # a client that refused the method. The question stays pending for
            # whoever else is here, or for whoever arrives.
            log.debug("ph_app.daemon: a front end could not answer %s", pending.method)
            self.front_ends.discard(who)
            return
        if pending.answered.is_set():
            # Somebody was faster. One question has one decision. This answer is
            # discarded and not recorded, because pH appends the decision that it
            # acted on, and a second one would make the log claim two.
            return
        self._answer(pending, answer, by=who)
