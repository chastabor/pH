"""P5-13 — the daemon can ask a person, and every attached front end is asked.

**The defect this closes is silent.** `ApprovalService` fails closed: with no
answerer its waterfall returns `unavailable` and the call is denied. Under
`ph daemon` no answerer was ever registered, so a gated tool call was denied —
not because anyone said no, but because nobody could be asked — while both
doctors reported the deployment as healthy. Nothing in the suite could see it,
because every approval test drove the *in-process* front end.

So these run against a real socket, and they assert on the **root's own log**,
which is where the decision has to land: pH records the approval, a front end
only decides. A test that checked the client's copy would pass for a design that
let a UI write its own answer into somebody else's session.

Attaching is not exclusive here and that is the point (the multiplex decision):
several UIs may watch one session, so an ask goes to all of them, the first
answer wins, and the rest are told.
"""

from __future__ import annotations

from typing import Any

import anyio
import pytest
from daemon_helpers import running

from ph.seams.user_questions import UserQuestion
from ph.testing import StubAgent

pytestmark = pytest.mark.anyio


async def _root(daemon: Any, session_id: str = "asked") -> Any:
    """One live root, started the way `session/attach` starts one."""
    return await daemon.server.supervisor.start(session_id)


async def _ask(root: Any) -> Any:
    """Fire one approval through the seam, exactly as a gated tool does."""
    return await root.ctx.approval.request(
        agent=StubAgent(ctx=root.ctx, session=root.session), tool_name="write", call_id="c1"
    )


async def _front_end(daemon: Any, root: Any, handler: Any, method: str = "approval/ask") -> Any:
    """One client that declares `asks`, answers `method`, and is attached.

    The four lines this replaces appeared at every front end in the file, which
    made the one line that actually differs — the handler — the hardest to see.
    """
    client = await daemon.client("asks")
    client.handlers[method] = handler
    await client.call("session/attach", sessionId=root.id)
    return client


def _answering(answer: str, seen: list[dict[str, Any]]) -> Any:
    async def handler(params: dict[str, Any]) -> dict[str, Any]:
        seen.append(params)
        return {"answer": answer}

    return handler


async def test_a_gated_call_under_the_daemon_reaches_a_person(tmp_path: Any) -> None:
    """The whole point: an approval over the socket is answered, not denied.

    Before this, the same call came back `unavailable` — a denial nobody made.
    The decision is read from the root's log rather than from the client, because
    the log is what pH acted on.
    """
    async with running(tmp_path) as daemon:
        seen: list[dict[str, Any]] = []
        root = await _root(daemon)
        await _front_end(daemon, root, _answering("allowed-once", seen))

        outcome = await _ask(root)

        assert outcome == "allowed-once"
        assert [one["request"]["toolName"] for one in seen] == ["write"]
        decided = [one for one in root.session.events if one.type == "approval/decided"]
        assert len(decided) == 1, "pH records the decision, once"


async def test_every_attached_front_end_is_asked_and_the_first_answer_wins(
    tmp_path: Any,
) -> None:
    """Multiplexed, not leased — and still exactly one decision in the log.

    Both UIs see the question, because both people are looking at it. Only one
    answer is acted on, and the other is discarded rather than recorded: a log
    claiming two decisions for one call would be a log that cannot be replayed.
    """
    async with running(tmp_path) as daemon:
        fast: list[dict[str, Any]] = []
        slow: list[dict[str, Any]] = []
        settled: list[tuple[str, dict[str, Any]]] = []
        first = await daemon.client("asks")
        second = await daemon.client("asks", on_notify=lambda m, p: settled.append((m, p)))
        first.handlers["approval/ask"] = _answering("allowed-once", fast)

        async def dawdle(params: dict[str, Any]) -> dict[str, Any]:
            slow.append(params)
            await anyio.sleep(30)
            return {"answer": "rejected"}

        second.handlers["approval/ask"] = dawdle
        root = await _root(daemon)
        for client in (first, second):
            await client.call("session/attach", sessionId=root.id)

        outcome = await _ask(root)

        assert outcome == "allowed-once"
        assert len(fast) == 1 and len(slow) == 1, "both were asked"
        assert [one.type for one in root.session.events].count("approval/decided") == 1
        with anyio.fail_after(5):
            while not [one for one in settled if one[0] == "ask.settled"]:
                await anyio.sleep(0.01)


async def test_a_watcher_that_is_not_a_front_end_is_never_asked(tmp_path: Any) -> None:
    """`ph agents attach` follows a log; it cannot answer for anyone.

    Answering is opt-in for that reason — the `asks` capability, declared once at
    `initialize` — and the sabotage is instructive: make attaching imply it and
    this session parks forever behind a follower that has no way to show a modal.
    """
    async with running(tmp_path) as daemon:
        asked: list[dict[str, Any]] = []
        watcher = await daemon.client()  # declares nothing
        watcher.handlers["approval/ask"] = _answering("allowed-once", asked)
        root = await _root(daemon)
        await watcher.call("session/attach", sessionId=root.id)
        await _front_end(daemon, root, _answering("rejected", []))

        outcome = await _ask(root)

        assert outcome == "rejected", "the follower's answer must not have been taken"
        assert asked == [], "and it was never asked"


async def test_an_ask_with_nobody_attached_waits_for_whoever_arrives(
    tmp_path: Any,
) -> None:
    """Nobody attached is a delay, not a denial (P5-13).

    The turn parks, the person turns up, and the question is still the one the
    model asked. Resolving on "no front end" instead would answer for them —
    which is exactly the `unavailable` denial this row exists to end.
    """
    async with running(tmp_path) as daemon:
        root = await _root(daemon)
        outcome: list[Any] = []

        async with anyio.create_task_group() as tasks:

            async def approve() -> None:
                outcome.append(await _ask(root))

            tasks.start_soon(approve)
            # Long enough that a design which resolved on "nobody there" would
            # already have done so; the assertion below is what proves it did not.
            await anyio.sleep(0.05)
            assert outcome == [], "an unattended ask must not resolve itself"
            assert root.status == "waiting", "and the root says it is parked on a person"

            await _front_end(daemon, root, _answering("allowed-once", []))

        assert outcome == ["allowed-once"]


async def test_a_root_parked_on_a_person_may_be_released(tmp_path: Any) -> None:
    """`waiting` is releasable, and that is the whole reason it exists.

    A turn suspended on an approval reports `running` from the agent, because it
    genuinely is mid-turn — but the only thing it waits for is a human, and
    calling that busy holds a whole process for somebody who closed their laptop.
    So `waiting` joins `idle` in `passivatable`, and the daemon may stop.

    What that costs today is worth stating beside the claim: the ask is held in
    memory, so stopping loses it. The log keeps the question — `approval/asked`
    with no `approval/decided` — but nothing reads that fold yet, so the turn does
    not resume and re-ask by itself. P5-13's repair half is what closes it.
    """
    async with running(tmp_path) as daemon:
        root = await _root(daemon)
        supervisor = daemon.server.supervisor

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_ask, root)
            await anyio.sleep(0.05)

            assert root.status == "waiting"
            assert supervisor.passivatable(root, now=root.idle_for(0) + 10**9, after=0.0)
            tasks.cancel_scope.cancel()


async def test_a_front_end_that_vanishes_mid_ask_is_dropped_not_answered_for(
    tmp_path: Any,
) -> None:
    """A dead socket is not a decision, and this is where that nearly went wrong.

    `_Connection.ask` used to return `{}` for a connection that had gone away.
    Downstream that reads as a successful answer carrying no fields, decodes to
    `unavailable`, and **denies the call** — reintroducing, from a new direction,
    exactly the silent denial with nobody behind it that this whole row exists to
    end. It raises now, the desk drops that front end, and the question survives
    for whoever attaches next.
    """
    async with running(tmp_path) as daemon:
        root = await _root(daemon)
        outcome: list[Any] = []

        async def never(params: dict[str, Any]) -> dict[str, Any]:
            await anyio.sleep(30)
            return {"answer": "rejected"}

        async with anyio.create_task_group() as tasks:
            leaver = await _front_end(daemon, root, never)

            async def approve() -> None:
                outcome.append(await _ask(root))

            tasks.start_soon(approve)
            await anyio.sleep(0.05)
            await leaver.aclose()
            await anyio.sleep(0.05)

            assert outcome == [], "a vanished client must not have answered for anyone"
            assert root.status == "waiting", "the question is still open"

            await _front_end(daemon, root, _answering("allowed-once", []))

        assert outcome == ["allowed-once"]


async def test_answering_is_declared_once_for_a_connection_not_per_attach(
    tmp_path: Any,
) -> None:
    """One `initialize` covers every session this client goes on to watch.

    Whether a UI can put a modal in front of a person is a fact about the UI, not
    about which root it happens to be looking at, and the flag it replaced could
    say yes for one and no for another — two answers to a question with one true
    answer. So the declaration comes before any attach and applies to all of
    them, which is also what lets a client attach to a second session without
    remembering what it claimed for the first.

    Sabotage: read the capability off the attach frame instead, and the second
    session is never asked.
    """
    async with running(tmp_path) as daemon:
        seen: list[dict[str, Any]] = []
        client = await daemon.client("asks")
        client.handlers["approval/ask"] = _answering("allowed-once", seen)
        one, two = await _root(daemon, "one"), await _root(daemon, "two")
        for root in (one, two):
            await client.call("session/attach", sessionId=root.id)

        # With a deadline, because the sabotage's failure is a *park*: an ask
        # with no front end waits by design (P5-13), so a test that only
        # asserted the answer would hang instead of failing.
        with anyio.fail_after(5):
            assert await _ask(one) == "allowed-once"
            assert await _ask(two) == "allowed-once"
        assert len(seen) == 2, "both roots reached the same declared front end"


# --------------------------------------------------------------- questions --


async def _ask_question(root: Any) -> Any:
    """One question through the seam, exactly as `ask_user` puts it."""
    return await root.ctx.user_questions.ask(
        UserQuestion(question="which port?", ask_id="call-1"), session=root.session
    )


async def test_a_question_over_the_socket_reaches_a_front_end(tmp_path: Any) -> None:
    """The second ask direction, and the one that had no producer until P7-09.

    Same desk, same fan-out, different failure mode — so it is worth its own
    exercise over a real socket rather than being assumed from the approval
    tests: the answer travels back as a string, and the log keeps both halves.
    """
    async with running(tmp_path) as daemon:
        seen: list[dict[str, Any]] = []

        async def answer(params: dict[str, Any]) -> dict[str, Any]:
            seen.append(params)
            return {"answer": "8080"}

        root = await _root(daemon, "asked-a-question")
        await _front_end(daemon, root, answer, "question/ask")

        assert await _ask_question(root) == "8080"
        assert [one["question"]["question"] for one in seen] == ["which port?"]
        types = [one.type for one in root.session.events]
        assert types.count("question/asked") == 1
        assert types.count("question/answered") == 1


async def test_the_wire_ask_id_is_the_one_the_log_wrote(tmp_path: Any) -> None:
    """One identity for one question, not two that happen to agree.

    The front end answers a frame keyed by `askId`, and a resume would re-pose
    from a log record keyed by `askId`. If the desk minted its own counter those
    are different strings, and the re-posed question could never be recognised as
    the one already open — which is the whole point of holding it in the log.
    """
    async with running(tmp_path) as daemon:
        posed: list[dict[str, Any]] = []

        async def answer(params: dict[str, Any]) -> dict[str, Any]:
            posed.append(params)
            return {"answer": "8080"}

        root = await _root(daemon, "keyed")
        await _front_end(daemon, root, answer, "question/ask")

        await _ask_question(root)

        asked = root.session.latest("question/asked")
        assert asked is not None
        assert [one["askId"] for one in posed] == [asked.data["askId"]] == ["call-1"]


async def test_a_daemon_with_no_front_end_does_not_log_a_question(tmp_path: Any) -> None:
    """Attached-but-watching is not attended, and neither is nobody at all.

    A root's desk registers its question answerer the moment the root starts, so
    without the reachability probe the seam would see a registered answerer,
    believe somebody was there, append the ask, and park the turn on a fan-out
    with no recipients — forever, since a question is not supposed to fail
    closed. Both halves are asserted: nothing logged, and no parking.

    Sabotage: register the answerer without `reachable`.
    """
    async with running(tmp_path) as daemon:
        root = await _root(daemon, "nobody-attending")
        watcher = await daemon.client()  # declares nothing; can only follow
        await watcher.call("session/attach", sessionId=root.id)

        with anyio.fail_after(5):
            assert await _ask_question(root) is None

        assert [one.type for one in root.session.events if one.type.startswith("question/")] == []
        assert root.status != "waiting", "an unattended question must not park a root"


async def test_re_attaching_while_a_question_is_open_does_not_ask_twice(
    tmp_path: Any,
) -> None:
    """Attaching again is an ordinary call, not a second front end.

    `session/attach` is how a client resumes, and `_attach` has always guarded
    `subscribe` against being called twice — the desk needed the same guard.
    Without it a re-attach starts a second delivery of every *open* ask to the
    same connection: a second frame, a second slot of that client's in-flight
    budget, and a second modal in front of one person, whose two answers then
    race for a question that can only have one.

    Sabotage: drop the `who in self.front_ends` early return in `AskDesk.join`,
    and `posed` holds two copies of the same `askId`.
    """
    async with running(tmp_path) as daemon:
        posed: list[dict[str, Any]] = []
        answer = anyio.Event()

        async def wait(params: dict[str, Any]) -> dict[str, Any]:
            posed.append(params)
            await answer.wait()
            return {"answer": "allowed-once"}

        root = await _root(daemon, "re-attached")
        client = await _front_end(daemon, root, wait)

        async with anyio.create_task_group() as tasks:
            tasks.start_soon(_ask, root)
            with anyio.fail_after(5):
                while not posed:
                    await anyio.sleep(0.01)

            await client.call("session/attach", sessionId=root.id)
            await anyio.sleep(0.05)
            answer.set()

        assert [one["askId"] for one in posed] == ["c1"], "one person, one question, once"
