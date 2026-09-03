"""P7-09 — `ask_user`, and a question the log keeps only when it was really asked.

`ctx.user_questions` shipped with a definition, a front end and a modal and no
caller. Two things follow from giving it one, and they are what this file holds.

**A question that reached a person is durable.** Both halves are appended, the
ask *before* the waterfall runs, so a crash while somebody was deciding leaves
the question in the log rather than losing it — the shape `pending_approvals`
already has, and `pending_questions` is the fold that reads it.

**A question that reached nobody never happened.** The seam's own failure mode is
"no answer" rather than a denial, so an unattended ask resolves instantly; if it
still appended, every `/autonomous` turn inside an interactive profile would
write a question-and-refusal pair into the log for an exchange that did not
occur, and the log would claim a person was asked and declined. That is a false
claim, not a tidiness problem, which is why the check is here and not in the UI.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest

from ph.cordis import DEPLOYMENT
from ph.llm.types import text_of
from ph.seams.user_questions import UserQuestion, pending_questions
from ph.testing import FAKE_OPTIONS, run_tool
from ph.tools.builtin.ask_user import UNATTENDED

pytestmark = pytest.mark.anyio

ROW: dict[str, Any] = {"id": "tool-ask-user", "disabled": False}
"""A patch of the row `ph-base` already carries, disarmed. Addressing the id
rather than inserting one keeps this the same row `tui.yaml` arms."""


def _agent(ctx: Any, session: Any) -> Any:
    return ctx.agents.create(session, FAKE_OPTIONS)


def _answering(answer: str | None, seen: list[UserQuestion]) -> Any:
    async def answerer(question: UserQuestion, _next: Any = None) -> str | None:
        seen.append(question)
        return answer

    return answerer


async def _ask(
    ctx: Any, session: Any, header: str | None = None, options: list[str] | None = None
) -> Any:
    """One `ask_user` call through the real pipeline.

    Positional, because `start_soon` takes no keywords and the cancellation gate
    below needs to launch this as a task."""
    arguments: dict[str, Any] = {"question": "which port?"}
    if header is not None:
        arguments["header"] = header
    if options is not None:
        arguments["options"] = options
    return await run_tool(ctx, "ask_user", arguments, agent=_agent(ctx, session), session=session)


# ------------------------------------------------------------------ the row --


async def test_the_row_is_disarmed_in_the_base_bundle(mount: Any) -> None:
    """Nothing is offered to a model that has nobody to ask.

    Composed without the patch, so this is `ph-base` + `headless` exactly as an
    unattended run gets it: no schema in the prompt, no turn spent calling it,
    nothing in the log. Sabotage: drop `disabled: true` from `base.yaml`, and a
    headless deployment starts paying prompt tokens to describe a tool whose only
    possible answer is "nobody is there".
    """
    ctx = await mount()

    assert ctx.tools.get("ask_user", scope=DEPLOYMENT) is None


async def test_arming_the_row_registers_the_tool(mount: Any) -> None:
    ctx = await mount(ROW)

    assert ctx.tools.get("ask_user", scope=DEPLOYMENT) is not None


# ------------------------------------------------------------ the exchange --


async def test_the_model_can_ask_the_person_a_question_and_read_the_answer(
    mount: Any,
) -> None:
    """The whole path, end to end: tool → seam → answerer → result → log."""
    ctx = await mount(ROW)
    seen: list[UserQuestion] = []
    ctx.user_questions.register_answerer(_answering("8080", seen))
    session = ctx.sessions.create("asked")

    result = await _ask(ctx, session, "Port", ["8080", "9000"])

    assert text_of(result.content) == "8080"
    assert [one.question for one in seen] == ["which port?"]
    assert [one.options for one in seen] == [["8080", "9000"]]
    types = [event.type for event in session.events]
    assert types.count("question/asked") == 1
    assert types.count("question/answered") == 1
    answered = session.latest("question/answered")
    assert answered is not None and answered.data.get("answer") == "8080"


async def test_the_question_is_logged_before_the_person_answers(mount: Any) -> None:
    """§5 rule 2, and the reason the two events are two.

    The answerer reads the log from inside its own call — which is the only
    moment the ordering is observable — and must already see the ask. One event
    appended on completion would lose every question a crash interrupted, and
    those are exactly the ones a resume needs to put back.

    Sabotage: append both after the waterfall, and `during` is empty.
    """
    ctx = await mount(ROW)
    session = ctx.sessions.create("ordered")
    during: list[str] = []

    async def answerer(question: UserQuestion, _next: Any = None) -> str:
        during.extend(event.type for event in session.events if event.type.startswith("question/"))
        return "yes"

    ctx.user_questions.register_answerer(answerer)

    await _ask(ctx, session)

    assert during == ["question/asked"], "the ask must be committed before it is put"


async def test_a_question_nobody_is_there_to_answer_writes_nothing_to_the_log(
    mount: Any,
) -> None:
    """Never asked is not the same as asked and declined.

    No answerer at all, which is what a sandboxed `/autonomous` run inside an
    interactive profile looks like: the row is mounted and there is no screen.
    The model gets a sentence it can act on and the log stays silent.

    Sabotage: append unconditionally, and two events appear for an exchange that
    never happened — a log that says a person was asked and said no.
    """
    ctx = await mount(ROW)
    session = ctx.sessions.create("alone")

    result = await _ask(ctx, session)

    assert text_of(result.content) == UNATTENDED
    assert [event.type for event in session.events if event.type.startswith("question/")] == []


async def test_an_answerer_that_cannot_reach_anyone_is_the_same_as_none(
    mount: Any,
) -> None:
    """Registered but unreachable — the daemon with every front end closed.

    The distinction the `reachable` probe exists for: a transport-shaped answerer
    is always registered, and whether it can find a person changes minute to
    minute, so the question is asked at ask time rather than at registration.

    Sabotage: consult `reachable` when the answerer registers, and a UI that
    attached once keeps this session logging questions into an empty room.
    """
    ctx = await mount(ROW)
    session = ctx.sessions.create("unreachable")
    attached: list[str] = []
    ctx.user_questions.register_answerer(_answering("42", []), reachable=lambda: bool(attached))

    away = await _ask(ctx, session)
    attached.append("a front end")
    back = await _ask(ctx, session)

    assert text_of(away.content) == UNATTENDED
    assert text_of(back.content) == "42"
    assert [event.type for event in session.events].count("question/asked") == 1


async def test_a_declined_question_is_recorded_and_stops_being_pending(
    mount: Any,
) -> None:
    """Somebody was there and said nothing — which is an answer to record.

    The other half of the rule above: this one *was* delivered, so both events
    are written even though there is no answer in the second. Leaving it out
    would make a question a person dismissed indistinguishable from one a crash
    interrupted, and a resume would put it back forever.
    """
    ctx = await mount(ROW)
    session = ctx.sessions.create("declined")
    ctx.user_questions.register_answerer(_answering(None, []))

    result = await _ask(ctx, session)

    assert text_of(result.content) == UNATTENDED
    answered = session.latest("question/answered")
    assert answered is not None and answered.data.get("declined") is True
    assert pending_questions(session) == []


# ------------------------------------------------------------- the pending --


async def test_a_question_cancelled_mid_answer_stays_pending(mount: Any) -> None:
    """The pending state is the log, not a table somebody remembered to keep.

    A real interruption, not a doctored log: the turn is cancelled while a person
    is still looking at the question — which is precisely what an ephemeral
    daemon passivating a root parked on a human does. The ask is already
    committed, the answer half never runs, and the fold reports the question.

    It also pins the shape of the `except Exception` in `ask`: cancellation is
    **not** an answerer failing. Widening that to `BaseException` would record
    `declined: true` for a question nobody ever dismissed, and the fold would
    then say a released root had been answered.
    """
    ctx = await mount(ROW)
    session = ctx.sessions.create("interrupted")
    posed = anyio.Event()

    async def never(question: UserQuestion, _next: Any = None) -> str:
        posed.set()
        await anyio.sleep(30)
        return "too late"

    ctx.user_questions.register_answerer(never)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(_ask, ctx, session, "Port", ["8080"])
        await posed.wait()
        tasks.cancel_scope.cancel()

    pending = pending_questions(session)

    assert [one.question.question for one in pending] == ["which port?"]
    assert pending[0].question.options == ["8080"]
    assert pending[0].question.header == "Port"
    assert session.latest("question/answered") is None, "nobody answered, so nothing says they did"


async def test_the_ask_id_is_the_call_id_the_rest_of_the_log_already_uses(
    mount: Any,
) -> None:
    """One string joins the four records of one exchange.

    `tool/call`, `tool/result` and both `question/*` events all carry the call
    id, so the exchange is reconstructible without a join table — and, more to
    the point, a minted counter cannot restart at 1 after a resume and answer a
    question the log is still holding open.
    """
    ctx = await mount(ROW)
    ctx.user_questions.register_answerer(_answering("8080", []))
    session = ctx.sessions.create("keyed")

    await run_tool(
        ctx,
        "ask_user",
        {"question": "which port?"},
        agent=_agent(ctx, session),
        session=session,
        call_id="call-abc",
    )

    asked = session.latest("question/asked")
    assert asked is not None and asked.data.get("askId") == "call-abc"
    assert [one.data.get("askId") for one in session.events if one.type == "question/answered"] == [
        "call-abc"
    ]
