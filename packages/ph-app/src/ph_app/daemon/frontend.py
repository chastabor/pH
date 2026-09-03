"""Who the daemon asks when a turn needs a person (P5-13).

Under the daemon there was nobody to ask. `ApprovalService` runs its waterfall
with an `inner` that answers `unavailable` when no answerer took the prompt, and
no answerer was ever registered on a root — so a gated tool call under
`ph daemon` was **denied**, not because a person said no but because nobody could
be asked, and both doctors reported the deployment as healthy. This is the piece
that was missing.

**One desk per root, many front ends.** Attach is not exclusive: several UIs may
watch one session — a terminal, a browser tab, `ph agents attach` — and each has
its own private composer while the log is shared. So an ask goes to *every*
attached front end, the **first answer wins**, and the rest are told
`ask.settled`. pH still records exactly one decision, because pH is what records
it: a front end only decides.

**Nobody attached is not a denial.** The ask sits here and the answerer waits on
it, and `join` re-poses it to whoever turns up. A gate that became a denial
because a terminal closed would lose the human's actual answer.

Not enforced (§5 rule 6): that re-posing is **in-memory, and so lasts as long as
the daemon does**. The log holds the question either way — an `approval/asked`
with no `approval/decided` *is* the pending state, and `pending_approvals` folds
it — but nothing reads that fold yet, so an ask does not survive a restart. The
turn does not resume and re-ask by itself; P5-13's repair half is what closes
that, and until it lands this is a delay bounded by the process rather than by
the log.

**A slow front end is dropped, not waited on.** `_Connection.ask` queues into a
bounded outbox; a client that cannot keep up raises rather than blocking the
turn, and it simply stops being asked. The ask stays pending for the others.

@module ph_app.daemon.frontend
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import anyio

from ph.llm.types import user_text
from ph.seams.approval import ApprovalAnswer, ApprovalRequest, answer_from_wire
from ph.seams.user_questions import UserQuestion

__all__ = ["AskDesk"]

log = logging.getLogger("ph_app.daemon.frontend")


class FrontEnd(Protocol):
    """A connection that has said it can answer for a person."""

    async def ask(self, method: str, params: dict[str, Any]) -> dict[str, Any]: ...
    def notify(self, method: str, params: dict[str, Any]) -> None: ...


@dataclass(slots=True)
class PendingAsk:
    """One question in flight, and the answer whoever gets there first leaves.

    `tasks` is the fan-out's own group, kept so a front end that arrives *while*
    the question is open can be posted to as well — which is what makes "nobody
    was attached" a delay rather than an outcome.
    """

    ask_id: str
    method: str
    params: dict[str, Any]
    answered: anyio.Event
    answer: dict[str, Any] | None = None
    tasks: Any = None


@dataclass(slots=True)
class AskDesk:
    """The answerers a root registers, and the front ends they reach."""

    root: Any
    front_ends: set[FrontEnd] = field(default_factory=set)
    asks: dict[str, PendingAsk] = field(default_factory=dict)

    # ------------------------------------------------------------- wiring --

    def attach(self) -> list[Callable[[], Any]]:
        """Register both answerers on the root's context. Returns their disposers.

        `register_answerer` is sugar over `ctx.on(...)`, so these unwind with
        whatever scope the root gives them — which is the root's own, since this
        runs outside any row's `apply`.
        """
        disposers: list[Callable[[], Any]] = []
        approval = self.root.ctx.get("approval")
        if approval is not None:
            disposers.append(approval.register_answerer(self.answer_approval))
        questions = self.root.ctx.get("user_questions")
        if questions is not None:
            # `reachable`, unlike the approval seam's registration, because the
            # two failure modes differ: an unanswerable approval must fail closed
            # and *deny*, while an unanswerable question must not be asked at
            # all. A daemon with no front end attached is exactly the case the
            # seam declines to log, and this probe is how it finds out.
            disposers.append(
                questions.register_answerer(
                    self.answer_question, reachable=lambda: bool(self.front_ends)
                )
            )
        return disposers

    def join(self, who: FrontEnd) -> None:
        """A front end that can answer. Anything already pending goes to it too.

        Re-posing on arrival is what makes "nobody was attached" a delay rather
        than an outcome: the turn parked, the person turned up, and the question
        is still the one the model asked.
        """
        if who in self.front_ends:
            # **Attaching twice is an expected call** — `session/attach` is how a
            # client resumes, and `_attach` already guards `subscribe` against it.
            # Without the same guard here a re-attach starts a *second* delivery
            # of every open ask to the same connection: a second frame, a second
            # slot of that client's in-flight budget, and a second modal in front
            # of one person, whose two answers then race for one question.
            return
        self.front_ends.add(who)
        for pending in list(self.asks.values()):
            if pending.tasks is not None:
                pending.tasks.start_soon(self._deliver, who, pending)

    def leave(self, who: FrontEnd) -> None:
        self.front_ends.discard(who)

    @property
    def waiting(self) -> bool:
        """Whether this root is parked on a person.

        Read by `Root.status`, which is what lets the sweep release it: a root
        waiting on a human is idle in every sense that matters, and holding a
        process for somebody who closed their laptop is the failure P5-05 exists
        to prevent.
        """
        return bool(self.asks)

    # ---------------------------------------------------------- answerers --

    async def answer_approval(self, request: ApprovalRequest, _next: Any = None) -> ApprovalAnswer:
        """`ctx.approval`'s answerer, over the socket.

        The ask is keyed by the same string `pending_approvals` uses — the call
        id, or the tool name when there is none — so a question re-posed after a
        restart is recognisably the one the log is still holding open.
        """
        result = await self._ask(
            "approval/ask",
            request.call_id or request.tool_name,
            {"request": request.to_wire()},
        )
        reason = str(result.get("reason") or "")
        if reason:
            # "No, use the existing helper" redirects a turn where a bare refusal
            # only stops it. Delivered as what it is — user input at the next step
            # boundary — rather than as a new event type, because the log's
            # vocabulary is fixed and a front end inventing one writes a log this
            # build cannot read.
            self.root.agent.steer(user_text(reason))
        return answer_from_wire(result.get("answer"))

    async def answer_question(self, question: UserQuestion, _next: Any = None) -> str | None:
        """`ctx.user_questions`' answerer, over the socket.

        Keyed by the question's own `ask_id`, which `UserQuestionService.ask`
        fills in before any answerer sees it — so the frame a front end answers
        and the record a resume would re-pose from are the same string rather
        than two that happen to line up. Minting a fallback here would be a
        second, weaker id scheme (a counter that restarts at 1 after a resume)
        for a value the seam already mints so that it cannot collide.
        """
        result = await self._ask(
            "question/ask", str(question.ask_id), {"question": question.to_wire()}
        )
        answer = result.get("answer")
        return answer if isinstance(answer, str) else None

    # ------------------------------------------------------------ the ask --

    async def _ask(self, method: str, ask_id: str, params: dict[str, Any]) -> dict[str, Any]:
        """Put one question to every front end and wait for the first answer.

        The fan-out has a task group of its own rather than borrowing the root's:
        every delivery belongs to *this* question and ends with it, so cancelling
        the group when the answer lands is exactly the right lifetime — the other
        front ends stop being waited on, and `ask.settled` tells them why.
        """
        pending = PendingAsk(
            ask_id=ask_id,
            method=method,
            params={"sessionId": self.root.id, "askId": ask_id, **params},
            answered=anyio.Event(),
        )
        self.asks[ask_id] = pending
        try:
            async with anyio.create_task_group() as tasks:
                pending.tasks = tasks
                for who in list(self.front_ends):
                    tasks.start_soon(self._deliver, who, pending)
                await pending.answered.wait()
                tasks.cancel_scope.cancel()
        finally:
            pending.tasks = None
            self.asks.pop(ask_id, None)
        return pending.answer or {}

    async def _deliver(self, who: FrontEnd, pending: PendingAsk) -> None:
        try:
            answer = await who.ask(pending.method, pending.params)
        except Exception:
            # A front end that cannot answer — a dead socket, a client that
            # refused the method — stops being asked. The question stays pending
            # for whoever else is here, or for whoever arrives.
            log.debug("ph_app.daemon: a front end could not answer %s", pending.method)
            self.front_ends.discard(who)
            return
        if pending.answered.is_set():
            # Somebody was faster. One question, one decision — and this answer
            # is discarded rather than recorded, because pH appends the decision
            # it acted on and a second one would be a log claiming two.
            return
        pending.answer = answer
        pending.answered.set()
        settled = {"sessionId": self.root.id, "askId": pending.ask_id}
        for other in list(self.front_ends):
            if other is who:
                continue
            try:
                other.notify("ask.settled", settled)
            except Exception:
                self.front_ends.discard(other)
