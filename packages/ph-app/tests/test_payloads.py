"""P8-08 — what the daemon emits, held to one shape per name.

P8-07 typed what the daemon *receives*; this is the other direction. The tests
here pin the two claims the models make that prose was making before:

* **`session.status` is one type, not three dicts.** Three producers reach one
  reader — a bare `passivated`, an `announce` with the footer, and the attach
  reply with the route as well — and `StatusFacts` is that union with every
  field optional, `None` meaning "this frame does not say".
* **A notice carries the name it travels under.** `Root.publish` reads `METHOD`
  off the payload, so a command list cannot go out as `session.status`.

The wire spellings are pinned too: these models are the daemon's half of a
contract a browser tab and `ph agents attach` both read, and a rename that
compiled would otherwise reach a client as a silently missing field.
"""

from __future__ import annotations

import pytest

from ph.seams.approval import ApprovalRequest
from ph.seams.tui_status import StatusReading
from ph_app import payloads
from ph_app.daemon.duplex import answering
from ph_app.payloads import (
    NOTICES,
    ApprovalAsk,
    ApprovalAskReply,
    AttachReply,
    MutationRepeated,
    QuestionAsk,
    RootDescription,
    SessionNotice,
    SessionStatusNotice,
    SnapshotPage,
    notice_of,
)
from ph_app.protocol import Cursor, InvalidParams

pytestmark = pytest.mark.anyio

CURSOR = Cursor(generation="1700000000000", sequence=4)

NOTICE_TYPES = [*NOTICES.values(), ApprovalAsk, QuestionAsk]
"""Every payload that travels under a name of its own.

Derived: the hand-written list had already drifted — it omitted `QuestionAsk`,
so the one ask whose `METHOD` nothing else checked was the one this file exists
to check."""


@pytest.mark.parametrize("notice", NOTICE_TYPES, ids=lambda one: one.__name__)
def test_every_notice_declares_the_name_it_travels_under(notice: type[SessionNotice]) -> None:
    """`Root.publish` reads this rather than taking it as a second argument —
    the two used to have to agree at eight call sites with nothing checking."""
    assert notice.METHOD, f"{notice.__name__} would publish under an empty method"


def test_the_method_is_a_class_fact_and_never_reaches_the_wire() -> None:
    """A `ClassVar`, so it is not a field: a client reading the payload sees the
    fields and learns the name from the frame's own `method`, as JSON-RPC says."""
    assert SessionStatusNotice.METHOD == "session.status"
    assert "method" not in SessionStatusNotice(session_id="s", status="idle").to_wire()


def test_the_three_status_shapes_are_one_type() -> None:
    """The union `tui/remote.py` was absorbing in prose. Each producer states
    only what it knows, and every one of them validates."""
    bare = SessionStatusNotice(session_id="s", status="passivated")
    assert bare.to_wire() == {"sessionId": "s", "status": "passivated"}
    assert bare.readings is None and bare.provider is None

    announced = SessionStatusNotice(
        session_id="s",
        status="idle",
        last_turn="completed",
        readings=[StatusReading(text="12k", level="warning")],
    )
    assert announced.to_wire()["lastTurn"] == "completed"
    assert announced.readings is not None and len(announced.readings) == 1

    attached = AttachReply(
        session_id="s",
        status="idle",
        watchers=1,
        cursor=CURSOR,
        provider="llama",
        model="m",
        readings=[StatusReading(text="12k", level="warning")],
    )
    assert attached.facts().provider == "llama", "the reply states the route"


def test_none_means_not_stated_rather_than_cleared() -> None:
    """The rule every reader was spelling by hand as `params.get(x) or self.x`.
    A frame that omits the route must not blank a footer that has one."""
    bare = SessionStatusNotice(session_id="s", status="retrying")
    assert bare.provider is None and bare.model is None and bare.readings is None
    # And an *empty* footer is a real answer, distinguishable from an absent one.
    emptied = SessionStatusNotice(session_id="s", status="idle", readings=[])
    assert emptied.readings == [], "an empty list is 'no readings', not 'unstated'"


def test_a_description_carries_its_cursor_as_a_cursor() -> None:
    """Nested model, so `describe()` cannot hand out a shape `resume_at` refuses
    — and `to_wire` recurses, which `dict(model)` does not."""
    described = RootDescription(session_id="s", status="idle", watchers=0, cursor=CURSOR)
    assert described.to_wire()["cursor"] == {"generation": "1700000000000", "sequence": 4}
    # `dict(model)` keeps the nested model **live** — which is a trap toward the
    # wire (one handler returned `dict(describe())` and every reply carrying a
    # cursor failed to serialize) and exactly right toward another model, where
    # it is what stops `model_dump()` round-tripping the `Cursor` through a dict.
    # `test_protocol` pins the cursor's own spelling; this pins the difference.
    assert isinstance(dict(described)["cursor"], Cursor)
    assert MutationRepeated(**dict(described)).cursor == CURSOR


def test_a_repeat_is_a_description_that_says_so() -> None:
    """One shape for every verb, so a client branches on one field."""
    described = RootDescription(session_id="s", status="idle", watchers=0, cursor=CURSOR)
    repeated = MutationRepeated(**described.model_dump())
    assert repeated.to_wire()["repeated"] is True
    assert repeated.session_id == described.session_id


def test_the_snapshot_page_spells_from_without_being_named_it() -> None:
    """`from` is a Python keyword, so the field cannot be called that — and the
    directional aliases keep the constructor reachable by its field name."""
    page = SnapshotPage(session_id="s", cursor=CURSOR, started_at=7, more=True)
    assert page.to_wire()["from"] == 7
    assert SnapshotPage.model_validate({"sessionId": "s", "cursor": CURSOR.to_wire(), "from": 9})


def test_a_field_the_notice_does_not_take_is_refused() -> None:
    """`extra="forbid"` reaches the emitted half too: a producer inventing a
    field is a failing test rather than one a reader silently drops."""
    with pytest.raises(ValueError):
        SessionStatusNotice.model_validate({"sessionId": "s", "status": "idle", "extra": 1})


# ------------------------------------------------------- the notice family --


def test_every_notice_is_reachable_by_its_method() -> None:
    """`NOTICES` is the client's half of the daemon's `METHODS`, and the two
    halves of a vocabulary have to be checked against each other or one grows.

    Read off the module rather than restating the list: a seventh notice added
    with a `METHOD` and forgotten in the table fails here, which is the only
    place that can notice — the daemon would publish it and every reader would
    silently drop it as unknown. No `key == METHOD` assertion, because `NOTICES`
    is built by that comprehension: it could only fail if someone rewrote the
    table as a literal, and a test that can only fail on a refactor of itself is
    a tautology wearing a check's clothes.
    """
    assert set(NOTICES.values()) == {one for one in _notice_classes() if one.METHOD}


def test_an_ask_cannot_be_registered_as_a_notice() -> None:
    """The exclusion is the **type**, not an author's memory.

    `SessionAsk` and `SessionNotice` are siblings under `SessionScoped`, so
    `Mapping[str, type[SessionNotice]]` structurally cannot hold an ask — where
    a denylist naming the two asks had to be edited for a third, and would have
    failed pointing at the wrong fix when it wasn't. A reader that found
    `approval/ask` among the notices would treat a question as an event and
    never answer it."""
    assert not issubclass(ApprovalAsk, SessionNotice)
    assert not issubclass(QuestionAsk, SessionNotice)
    assert ApprovalAsk.METHOD not in NOTICES and QuestionAsk.METHOD not in NOTICES
    # And the ask half still names itself: `answering` reads `METHOD` for the
    # refusal, so an ask without one would refuse under an empty method.
    assert ApprovalAsk.METHOD and QuestionAsk.METHOD
    assert "ask_id" in ApprovalAsk.model_fields and "ask_id" in QuestionAsk.model_fields


def test_an_unknown_method_reads_as_nothing_rather_than_raising() -> None:
    """A daemon newer than this client sends a notice this build has no model
    for. That is a feature the client does not have, not an error."""
    assert notice_of("session.invented-later", {"sessionId": "s"}) is None


def test_a_known_method_comes_back_as_its_own_type() -> None:
    read = notice_of(SessionStatusNotice.METHOD, {"sessionId": "s", "status": "idle"})
    assert isinstance(read, SessionStatusNotice)
    assert read.status == "idle"


def _notice_classes() -> set[type[SessionNotice]]:
    """Every notice declared in `ph_app.payloads`, read off the module.

    Off the module rather than `SessionNotice.__subclasses__()`: that walks a
    *process-global* registry, so a subclass defined in some other test would
    join it and make this assertion depend on what else the run imported — the
    order-dependent-failure shape issue 58 already documents once.
    """
    return {
        one
        for one in vars(payloads).values()
        if isinstance(one, type) and issubclass(one, SessionNotice) and one is not SessionNotice
    }


# --------------------------------------------------------- the typed door --


async def test_a_malformed_ask_is_refused_by_name_not_by_traceback() -> None:
    """`answering`'s refusal path, which nothing exercised.

    A handler that could not read its ask used to let pydantic's
    `ValidationError` escape, and that carries no `code` — so `respond` sent it
    back as an unnamed `-32000` whose message was a multi-line pydantic dump,
    and `DaemonError.reason` arrived empty at the one end that needs to branch
    on it. Going through `parse_params` makes the client's refusal the same
    shape as the daemon's, in both directions.

    The daemon's half of the bargain is `AskDesk._deliver`: `invalid_params`
    keeps the front end joined, because a client that cannot read *this* ask can
    still answer the next one of a different kind.
    """
    answered: list[ApprovalAsk] = []

    async def answer(ask: ApprovalAsk) -> ApprovalAskReply:
        answered.append(ask)
        return ApprovalAskReply(answer="allow")

    handler = answering(ApprovalAsk, answer)

    with pytest.raises(InvalidParams) as refused:
        await handler({"sessionId": "s"})  # no askId, no request

    assert refused.value.code == "invalid_params"
    assert ApprovalAsk.METHOD in str(refused.value), "the refusal names the method"
    assert "askId" in str(refused.value), "and the field that was missing"
    assert not answered, "a body must not run on an ask it could not read"


async def test_a_well_formed_ask_reaches_the_body_as_its_model() -> None:
    """And the reply goes back as a model — `respond` dumps it at the one point
    a result becomes a frame, so the handler does not spell `.to_wire()`."""
    seen: list[ApprovalAsk] = []

    async def answer(ask: ApprovalAsk) -> ApprovalAskReply:
        seen.append(ask)
        return ApprovalAskReply(answer="allow", reason="because")

    request = ApprovalRequest(tool_name="bash", call_id="c1")
    reply = await answering(ApprovalAsk, answer)(
        {"sessionId": "s", "askId": "a1", "request": request.to_wire()}
    )

    assert seen and seen[0].ask_id == "a1" and seen[0].request.tool_name == "bash"
    assert isinstance(reply, ApprovalAskReply), "a model, not a dict — respond dumps it"
