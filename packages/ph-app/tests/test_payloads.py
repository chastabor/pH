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

from ph.seams.tui_status import StatusReading
from ph_app.payloads import (
    ApprovalAsk,
    AskSettledNotice,
    AttachReply,
    MutationRepeated,
    RootDescription,
    SessionCommandsNotice,
    SessionEventNotice,
    SessionNotice,
    SessionScreensNotice,
    SessionStagedNotice,
    SessionStatusNotice,
    SnapshotPage,
)
from ph_app.protocol import Cursor

CURSOR = Cursor(generation="1700000000000", sequence=4)

NOTICES = [
    SessionEventNotice,
    SessionStatusNotice,
    SessionCommandsNotice,
    SessionScreensNotice,
    SessionStagedNotice,
    AskSettledNotice,
    ApprovalAsk,
]


@pytest.mark.parametrize("notice", NOTICES, ids=lambda one: one.__name__)
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
