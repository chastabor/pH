"""P10-05 — `ph.session.intents`: one statement of "opened and never settled".

The fold has to agree with the ones it replaces before anything moves onto it, so
the approval and question folds are asked the same questions here, over the same
logs: P10-09 deletes them, and this is what says it may.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import cast

import pytest

from ph.cordis import Context
from ph.json import JsonObject, as_str
from ph.keys import INTENTS, INVARIANTS, SESSION_PERSISTENCE, SESSIONS
from ph.session import (
    IntentError,
    IntentKind,
    IntentRecord,
    OpenIntent,
    Session,
    SessionEvent,
    Unsettled,
    declare_intent,
    declared_intents,
    fold_intents,
    open_intents,
    outcome_of,
    settled_record,
)
from ph.session.journal import Claim, IntentJournal, IntentNotDurable, Prior
from ph.session.kinds import SHELL_COMMAND
from ph.session.store import SessionStore
from ph.session.writers import log_writer
from ph.testing import (
    SCAFFOLDING,
    MountProfile,
    check_fold_laws,
    isolated_intent_kinds,
    log_event,
    not_none,
)

pytestmark = pytest.mark.anyio


def _approval_key(event: SessionEvent) -> str | None:
    return as_str(event.data.get("callId") or event.data.get("toolName"))


def _ask_key(event: SessionEvent) -> str | None:
    return as_str(event.data.get("askId"))


def _denied(opened: SessionEvent, why: Unsettled) -> JsonObject:
    return {**opened.data, "decision": "deny", "why": why}


APPROVAL = IntentKind(
    opened="approval/asked",
    settled="approval/decided",
    opened_key=_approval_key,
    settled_key=_approval_key,
    orphan="not-started",
    closer=_denied,
    writer=SCAFFOLDING,
)
"""The approval pair under the rule `pending_approvals` states — not the kind
P10-09 declares, which is that row's to write."""

QUESTION = IntentKind(
    opened="question/asked",
    settled="question/answered",
    opened_key=_ask_key,
    settled_key=_ask_key,
    orphan="not-started",
    closer=_denied,
    writer=SCAFFOLDING,
)


@pytest.fixture
def kinds() -> Iterator[None]:
    """An empty registry this test can declare into without leaking — `vocabulary`'s
    reason, one table over."""
    with isolated_intent_kinds(core=False):
        yield


def _approvals() -> Session:
    """Asks and decisions interleaved: a re-ask of a live key, a decision for one
    key among several, a decision nobody asked for, and a key read off the tool
    name because the call id is missing."""
    session = Session("s")
    log_event(session, "approval/asked", {"callId": "c1", "toolName": "write"})
    log_event(session, "approval/asked", {"callId": "c2", "toolName": "bash"})
    log_event(session, "approval/decided", {"callId": "c1", "decision": "allow"})
    log_event(session, "approval/asked", {"callId": "c2", "toolName": "bash", "reason": "again"})
    log_event(session, "approval/decided", {"callId": "c9", "decision": "deny"})
    log_event(session, "approval/asked", {"toolName": "fetch"})
    log_event(session, "approval/asked", {"callId": "c1", "toolName": "write"})
    return session


def _questions() -> Session:
    session = Session("s")
    for ask in ("q1", "q2", "q3"):
        log_event(session, "question/asked", {"askId": ask, "question": f"{ask}?"})
    log_event(session, "question/answered", {"askId": "q2", "answer": "yes"})
    return session


def test_an_opened_intent_with_no_settle_is_open() -> None:
    session = Session("s")
    log_event(session, "approval/asked", {"callId": "c1", "toolName": "write"})

    (intent,) = open_intents(session.events, APPROVAL)
    assert intent == OpenIntent(key="c1", opened=session.events[0])
    assert settled_record(session.events, APPROVAL, "c1") is None


def test_a_settle_closes_only_its_own_key() -> None:
    """And the rest of the rule: a re-ask replaces the live one, a settle for a
    key never opened is nothing, and the result is in opening order."""
    session = _approvals()
    found = open_intents(session.events, APPROVAL)

    assert [(intent.key, intent.opened.seq) for intent in found] == [
        ("c2", 3),
        ("fetch", 5),
        ("c1", 6),
    ]


def test_the_settle_is_the_latest_opens_or_none() -> None:
    session = _approvals()
    events = session.events

    assert settled_record(events, APPROVAL, "c1") is None, "c1 was asked again after"
    assert settled_record(events, APPROVAL, "c9") is None, "a settle nobody asked is no intent"
    assert settled_record(events, APPROVAL, "never") is None
    log_event(session, "approval/decided", {"callId": "c1", "decision": "deny"})
    assert settled_record(session.events, APPROVAL, "c1") is session.events[-1]


def test_a_record_with_no_key_is_not_part_of_the_pair() -> None:
    def keyed(event: SessionEvent) -> str | None:
        value = event.data.get("callId")
        return value if isinstance(value, str) else None

    kind = IntentKind(
        opened="approval/asked",
        settled="approval/decided",
        opened_key=keyed,
        settled_key=keyed,
        orphan="owner-settles",
        writer=SCAFFOLDING,
    )
    session = Session("s")
    log_event(session, "approval/asked", {"toolName": "write"})
    log_event(session, "approval/decided", {"toolName": "write"})
    assert open_intents(session.events, kind) == ()


Index = Mapping[str, IntentRecord]


def _laws(session: Session, kind: IntentKind) -> list[str]:
    def compute(log: Session) -> Index:
        return fold_intents(log.events, kind)

    def extend(previous: Index, log: Session, start: int) -> Index:
        return fold_intents(log.events_from(start), kind, since=previous)

    return check_fold_laws(session, compute, extend)


def test_the_intent_index_obeys_the_fold_laws() -> None:
    """Over the logs above until P10-08 to P10-11 add each migrated kind's own
    producer's; `since` is the `extend` the journal's cache uses."""
    assert _laws(_approvals(), APPROVAL) == []
    assert _laws(_questions(), QUESTION) == []


def test_extending_over_nothing_of_the_kind_hands_back_the_same_index() -> None:
    """What makes the journal's cache cheap per model step: a slice of chunks
    costs the slice, not a copy of every key the index holds."""
    session = _approvals()
    index = fold_intents(session.events, APPROVAL)
    log_event(session, "turn/start", {"turn": 1})
    assert fold_intents(session.events_from(session.seq - 1), APPROVAL, since=index) is index


@pytest.mark.usefixtures("kinds")
def test_a_kind_on_an_unknown_type_is_refused() -> None:
    with pytest.raises(IntentError, match='"quantum/entangle" is not a session event type'):
        declare_intent(
            IntentKind(
                opened="quantum/entangle",
                settled="approval/decided",
                opened_key=_approval_key,
                settled_key=_approval_key,
                orphan="owner-settles",
                writer=SCAFFOLDING,
            )
        )
    assert declared_intents() == ()


@pytest.mark.usefixtures("kinds")
def test_a_kind_repair_cannot_settle_is_refused() -> None:
    """`not-started` and `outcome-unknown` are settled by repair, which has to
    write something; `owner-settles` is left to its owner and needs nothing."""
    unsettleable = IntentKind(
        opened="approval/asked",
        settled="approval/decided",
        opened_key=_approval_key,
        settled_key=_approval_key,
        orphan="outcome-unknown",
        writer=SCAFFOLDING,
    )
    with pytest.raises(IntentError, match="needs a closer"):
        declare_intent(unsettleable)
    with pytest.raises(IntentError, match="cannot both open and settle"):
        declare_intent(
            IntentKind(
                opened="approval/asked",
                settled="approval/asked",
                opened_key=_approval_key,
                settled_key=_approval_key,
                orphan="owner-settles",
                writer=SCAFFOLDING,
            )
        )


@pytest.mark.usefixtures("kinds")
def test_a_kind_settled_on_the_surface_is_refused() -> None:
    """The tool pair's settle is a `tool/result` a provider validates. The journal
    writes no surface placement and repair's closer would answer one call twice,
    so such a pair is the turn repair's and cannot be declared as a kind."""
    with pytest.raises(IntentError, match='"tool/result" is model-visible'):
        declare_intent(
            IntentKind(
                opened="tool/call",
                settled="tool/result",
                opened_key=_approval_key,
                settled_key=_approval_key,
                orphan="owner-settles",
                writer=SCAFFOLDING,
            )
        )


@pytest.mark.usefixtures("kinds")
def test_one_type_opens_one_kind() -> None:
    assert declare_intent(APPROVAL) is APPROVAL
    assert declare_intent(APPROVAL) is APPROVAL, "a module imported twice refuses nothing"
    rival = IntentKind(
        opened="approval/asked",
        settled="question/answered",
        opened_key=_approval_key,
        settled_key=_ask_key,
        orphan="owner-settles",
        writer=SCAFFOLDING,
    )
    with pytest.raises(IntentError, match=r"already opens an intent declared by 'ph\.testing'"):
        declare_intent(rival)
    assert declared_intents() == (APPROVAL,)


@pytest.mark.usefixtures("kinds")
def test_a_kind_whose_writer_does_not_own_its_types_is_refused() -> None:
    """T6. The journal writes a kind's pair through the kind's writer, so a kind is
    declared only by a module that is the writer of record for both of its types —
    this module is for neither, and could otherwise write a posture record through
    a kind it made up.

    Sabotage: drop the writer check from `declare_intent`, and this declares.
    """
    with pytest.raises(IntentError, match='is not a writer of record for "sandbox/mode"'):
        declare_intent(
            IntentKind(
                opened="sandbox/mode",
                settled="sandbox/denied",
                opened_key=_approval_key,
                settled_key=_approval_key,
                orphan="owner-settles",
                writer=log_writer(__name__),
            )
        )
    assert declared_intents() == ()


# ------------------------------------------------------------ the journal --
# P10-06. `ctx.intents`, mounted the way a deployment mounts it: by the
# `session` row, over a store a real backend writes.

DURABLE = IntentKind(
    opened="approval/asked",
    settled="approval/decided",
    opened_key=_approval_key,
    settled_key=_approval_key,
    orphan="not-started",
    closer=_denied,
    writer=SCAFFOLDING,
)
BUFFERED = replace(
    DURABLE,
    opened="question/asked",
    settled="question/answered",
    opened_key=_ask_key,
    settled_key=_ask_key,
    barrier="buffered",
)
ASK = {"callId": "c1", "toolName": "write"}


async def _journal(mount: MountProfile) -> tuple[Context, IntentJournal, Session]:
    ctx = await mount()
    declare_intent(DURABLE)
    declare_intent(BUFFERED)
    return ctx, ctx.require(INTENTS), ctx.require(SESSIONS).create("journaled")


@pytest.mark.usefixtures("kinds")
async def test_open_is_on_disk_before_the_claim_is_handed_out(mount: MountProfile) -> None:
    """The barrier. *Sabotage:* drop the flush from `open` and the store read
    below finds no `approval/asked` — the claim was handed out for an act the log
    could not yet say was about to happen."""
    ctx, journal, session = await _journal(mount)

    held = await journal.open(session, DURABLE, ASK)

    assert isinstance(held, Claim) and held.key == "c1"
    _, stored = ctx.require(SESSION_PERSISTENCE).read(session.id)
    assert [event.type for event in stored] == ["approval/asked"]


@pytest.mark.usefixtures("kinds")
async def test_a_failed_barrier_hands_out_no_claim_and_closes_the_pair(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Fail-closed, and honest: no claim, and the pair settled `not-started` in
    memory, so the next flush that works writes what happened rather than an
    orphan repair would have to guess about."""
    _, journal, session = await _journal(mount)

    async def never(_store: SessionStore, _session: Session) -> bool:
        return False

    monkeypatch.setattr(SessionStore, "written", never)
    with pytest.raises(IntentNotDurable, match="could not be written"):
        await journal.open(session, DURABLE, ASK)

    assert [event.type for event in session.events] == ["approval/asked", "approval/decided"]
    assert session.events[-1].data["why"] == "not-started"
    assert journal.pending(session, DURABLE) == ()


@pytest.mark.usefixtures("kinds")
async def test_a_barrier_canceled_mid_write_still_closes_the_pair(
    mount: MountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    """K7's rule, in the journal: a cancellation is not an `Exception`, and an
    ask canceled while its record was being written was never put to anybody —
    so it closes `not-started`, and the cancellation still propagates."""
    _, journal, session = await _journal(mount)

    class Stopped(BaseException):
        pass

    async def stopped(_store: SessionStore, _session: Session) -> bool:
        raise Stopped

    monkeypatch.setattr(SessionStore, "written", stopped)
    with pytest.raises(Stopped):
        await journal.open(session, DURABLE, ASK)

    assert [event.type for event in session.events] == ["approval/asked", "approval/decided"]
    assert session.events[-1].data["why"] == "not-started"


@pytest.mark.usefixtures("kinds")
async def test_a_second_open_of_one_key_returns_the_prior_and_appends_nothing(
    mount: MountProfile,
) -> None:
    _, journal, session = await _journal(mount)
    first = await journal.open_once(session, DURABLE, ASK)
    assert isinstance(first, Claim)

    running = await journal.open_once(session, DURABLE, ASK)
    assert isinstance(running, Prior) and running.opened == first.opened
    assert (running.settled, running.outcome, running.running_here) == (None, None, True)
    settled = journal.settle(session, first, {"callId": "c1", "decision": "allow"})
    done = await journal.open_once(session, DURABLE, ASK)
    assert isinstance(done, Prior) and (done.settled, done.outcome) == (settled, "done")
    assert session.seq == 2, "a prior appends nothing"
    assert journal.outcome(session, DURABLE, "c1") is settled


@pytest.mark.usefixtures("kinds")
async def test_claim_settles_as_failed_when_the_body_raises(mount: MountProfile) -> None:
    """`step/end` in a `finally`, as a method: a caller cannot leave a pair open
    by raising — and a body that settled before it raised is not settled twice."""
    _, journal, session = await _journal(mount)

    with pytest.raises(LookupError):
        async with journal.claim(session, BUFFERED, {"askId": "q1"}) as held:
            assert isinstance(held, Claim)
            raise LookupError("the act failed")
    settle = not_none(journal.outcome(session, BUFFERED, "q1"))
    assert outcome_of(BUFFERED, settle) == "outcome-unknown"
    assert settle.data["unsettled"] == {"why": "outcome-unknown", "by": "process"}

    with pytest.raises(LookupError):
        async with journal.claim(session, BUFFERED, {"askId": "q2"}) as held:
            assert isinstance(held, Claim)
            journal.settle(session, held, {"askId": "q2", "answer": "yes"})
            raise LookupError("after the settle")
    assert [event.type for event in session.events].count("question/answered") == 2
    assert outcome_of(BUFFERED, not_none(journal.outcome(session, BUFFERED, "q2"))) == "done"


@pytest.mark.usefixtures("kinds")
async def test_a_settle_closes_only_its_own_intent_once(mount: MountProfile) -> None:
    _, journal, session = await _journal(mount)
    held = await journal.open(session, DURABLE, ASK)
    assert isinstance(held, Claim)

    with pytest.raises(IntentError, match="cannot close the intent opened as 'c1'"):
        journal.settle(session, held, {"callId": "c2", "decision": "allow"})
    journal.settle(session, held, {"callId": "c1", "decision": "allow"})
    with pytest.raises(IntentError, match="is not open"):
        journal.settle(session, held, {"callId": "c1", "decision": "deny"})


@pytest.mark.usefixtures("kinds")
async def test_what_the_journal_refuses_to_open(mount: MountProfile) -> None:
    """A kind repair does not know, a record whose key cannot be read, and a
    durable kind opened without its barrier — each refused before any append."""
    _, journal, session = await _journal(mount)
    stranger = replace(DURABLE, reopen=frozenset({"failed"}))

    with pytest.raises(IntentError, match="not a declared intent kind"):
        await journal.open(session, stranger, ASK)
    with pytest.raises(IntentError, match="must carry its key"):
        await journal.open(session, BUFFERED, {"question": "no id"})
    with pytest.raises(IntentError, match="is durable; open it with `open`"):
        journal.record(session, DURABLE, ASK)
    assert session.seq == 0
    assert journal.record(session, BUFFERED, {"askId": "q1"}).key == "q1"


@pytest.mark.usefixtures("kinds")
async def test_the_intent_cache_is_polled_as_an_invariant(mount: MountProfile) -> None:
    """The index is a `SessionFoldCache` like the six others, and reported by
    its own row, naming the kind that drifted."""
    ctx, journal, session = await _journal(mount)
    await journal.open(session, DURABLE, ASK)
    assert ctx.require(INVARIANTS).verify() == []

    index = cast(dict[str, IntentRecord], journal._index(session, DURABLE))
    index["ghost"] = IntentRecord(opened=session.events[0], settled=None)

    (violation,) = ctx.require(INVARIANTS).verify()
    assert violation.invariant == "intent-fold-cache"
    assert "approval/asked" in violation.detail and "journaled" in violation.detail


async def test_the_shell_kind_obeys_the_fold_laws(mount: MountProfile) -> None:
    """P10-08's kind, over a log its own journal calls wrote: settled commands,
    and one still running."""
    ctx = await mount()
    journal = ctx.require(INTENTS)
    session = ctx.require(SESSIONS).create("shell-laws")
    for command in ("make", "make test"):
        held = await journal.open(session, SHELL_COMMAND, {"command": command, "surface": False})
        assert isinstance(held, Claim)
        journal.settle(session, held, {"commandSeq": held.opened.seq, "ok": True})
    await journal.open(session, SHELL_COMMAND, {"command": "sleep 9", "surface": False})

    assert _laws(session, SHELL_COMMAND) == []
    assert [
        intent.opened.data["command"] for intent in journal.pending(session, SHELL_COMMAND)
    ] == ["sleep 9"]
