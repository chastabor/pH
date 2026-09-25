"""P10-05 — `ph.session.intents`: one statement of "opened and never settled".

Over the shipped kinds (N2): the fold's rule is asked of `TOOL_EFFECT` and
`TOOL_DISPATCH`, the field-keyed pairs, and the journal's doors of those kinds with one
property changed by `dataclasses.replace` — so a test kind's keys, closer and payloads
cannot drift from the ones the harness writes.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import cast

import pytest

from ph.cordis import Context
from ph.json import JsonObject
from ph.keys import INTENTS, INVARIANTS, SESSION_PERSISTENCE, SESSIONS
from ph.session import (
    IntentError,
    IntentKind,
    IntentRecord,
    OpenIntent,
    Session,
    declare_intent,
    declared_intents,
    fold_intents,
    open_intents,
    outcome_of,
    unsettled_why,
)
from ph.session.journal import Claim, IntentJournal, IntentNotDurable, Prior
from ph.session.kinds import SHELL_COMMAND, TOOL_DISPATCH, TOOL_EFFECT, effect_settle
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
from ph.tools.code_mode import CodeDispatchLog, CodeDispatchRef

pytestmark = pytest.mark.anyio


def _effect(key: str, call_id: str = "c1") -> JsonObject:
    """A `tool/effect` opening, as the pipeline writes one."""
    return {"key": key, "tool": key.partition(":")[0], "callId": call_id}


def _effect_done(key: str, call_id: str = "c1") -> JsonObject:
    """Its settle, by the kind's own builder."""
    return effect_settle(key, call_id, is_error=False, content=())


def _ref(sub_call_id: str) -> CodeDispatchRef:
    root = sub_call_id.partition(":")[0]
    return CodeDispatchRef(
        root_call_id=root, parent_call_id=root, sub_call_id=sub_call_id, name="bash"
    )


def _dispatch(sub_call_id: str) -> JsonObject:
    """A Code Mode dispatch's identity, as its start record carries it."""
    return _ref(sub_call_id).to_wire()


def _dispatch_done(sub_call_id: str) -> JsonObject:
    """Its settle, as `DispatchBridge` writes one."""
    record = CodeDispatchLog(**_ref(sub_call_id).__dict__, is_error=False)
    return {**record.to_wire(), "content": []}


@pytest.fixture
def kinds() -> Iterator[None]:
    """An empty registry this test can declare into without leaking — `vocabulary`'s
    reason, one table over."""
    with isolated_intent_kinds(core=False):
        yield


def _effects() -> Session:
    """Openings and settles of the real field-keyed kind, interleaved: an effect opened
    again while live, a settle for one key among several, and a settle nothing
    opened."""
    session = Session("s")
    log_event(session, "tool/effect", _effect("send:m1"))
    log_event(session, "tool/effect", _effect("send:m2", "c2"))
    log_event(session, "tool/effect-settled", _effect_done("send:m1"))
    log_event(session, "tool/effect", _effect("send:m2", "c3"))
    log_event(session, "tool/effect-settled", _effect_done("send:m9", "c9"))
    log_event(session, "tool/effect", _effect("fetch:page", "c4"))
    log_event(session, "tool/effect", _effect("send:m1", "c5"))
    return session


def _dispatches() -> Session:
    session = Session("s")
    for n in range(3):
        log_event(session, "tool/code-dispatch-start", _dispatch(f"r1:code:{n}"))
    log_event(session, "tool/code-dispatch", _dispatch_done("r1:code:1"))
    return session


def test_an_opened_intent_with_no_settle_is_open() -> None:
    session = Session("s")
    log_event(session, "tool/effect", _effect("send:m1"))

    (intent,) = open_intents(session.events, TOOL_EFFECT)
    assert intent == OpenIntent(key="send:m1", opened=session.events[0])
    assert fold_intents(session.events, TOOL_EFFECT)["send:m1"].settled is None


def test_a_settle_closes_only_its_own_key() -> None:
    """And the rest of the rule: an opening replaces the live one under its key, a
    settle for a key never opened is nothing, and the result is in opening order."""
    session = _effects()
    found = open_intents(session.events, TOOL_EFFECT)

    assert [(intent.key, intent.opened.seq) for intent in found] == [
        ("send:m2", 3),
        ("fetch:page", 5),
        ("send:m1", 6),
    ]


def test_the_settle_is_the_latest_opens_or_none() -> None:
    session = _effects()
    events = session.events

    folded = fold_intents(events, TOOL_EFFECT)
    assert folded["send:m1"].settled is None, "send:m1 was opened again after"
    assert "send:m9" not in folded, "a settle nothing opened is no intent"
    assert "never" not in folded
    log_event(session, "tool/effect-settled", _effect_done("send:m1", "c5"))
    assert fold_intents(session.events, TOOL_EFFECT)["send:m1"].settled is session.events[-1]


def test_a_record_with_no_key_is_not_part_of_the_pair() -> None:
    """A `shell/result` with no `commandSeq` names no command, so it closes none."""
    session = Session("s")
    log_event(session, "shell/command", {"command": "make", "surface": False})
    log_event(session, "shell/result", {"ok": True})

    (intent,) = open_intents(session.events, SHELL_COMMAND)
    assert intent.opened.data["command"] == "make"


Index = Mapping[str, IntentRecord]


def _laws(session: Session, kind: IntentKind) -> list[str]:
    def compute(log: Session) -> Index:
        return fold_intents(log.events, kind)

    def extend(previous: Index, log: Session, start: int) -> Index:
        return fold_intents(log.events_from(start), kind, since=previous)

    return check_fold_laws(session, compute, extend)


def test_the_intent_index_obeys_the_fold_laws() -> None:
    """Over the logs above, of the real kinds; `since` is the `extend` the journal's
    cache uses."""
    assert _laws(_effects(), TOOL_EFFECT) == []
    assert _laws(_dispatches(), TOOL_DISPATCH) == []


def test_extending_over_nothing_of_the_kind_hands_back_the_same_index() -> None:
    """What makes the journal's cache cheap per model step: a slice of chunks
    costs the slice, not a copy of every key the index holds."""
    session = _effects()
    index = fold_intents(session.events, TOOL_EFFECT)
    log_event(session, "turn/start", {"turn": 1})
    assert fold_intents(session.events_from(session.seq - 1), TOOL_EFFECT, since=index) is index


@pytest.mark.usefixtures("kinds")
def test_a_kind_on_an_unknown_type_is_refused() -> None:
    with pytest.raises(IntentError, match='"quantum/entangle" is not a session event type'):
        declare_intent(replace(TOOL_EFFECT, opened="quantum/entangle"))
    assert declared_intents() == ()


@pytest.mark.usefixtures("kinds")
def test_a_kind_repair_cannot_settle_is_refused() -> None:
    """`not-started` and `outcome-unknown` are settled by repair, which has to
    write something; `owner-settles` is left to its owner and needs nothing."""
    with pytest.raises(IntentError, match="needs a closer"):
        declare_intent(replace(TOOL_EFFECT, closer=None))
    with pytest.raises(IntentError, match="cannot both open and settle"):
        declare_intent(replace(TOOL_EFFECT, settled="tool/effect"))


@pytest.mark.usefixtures("kinds")
def test_a_kind_settled_on_the_surface_is_refused() -> None:
    """The tool pair's settle is a `tool/result` a provider validates. The journal
    writes no surface placement and repair's closer would answer one call twice,
    so such a pair is the turn repair's and cannot be declared as a kind. (By the
    scaffolding writer, which owns both types, so the surface rule is what refuses.)"""
    with pytest.raises(IntentError, match='"tool/result" is model-visible'):
        declare_intent(
            replace(TOOL_EFFECT, opened="tool/call", settled="tool/result", writer=SCAFFOLDING)
        )


@pytest.mark.usefixtures("kinds")
def test_one_type_opens_one_kind() -> None:
    assert declare_intent(TOOL_EFFECT) is TOOL_EFFECT
    assert declare_intent(TOOL_EFFECT) is TOOL_EFFECT, "a module imported twice refuses nothing"
    rival = replace(TOOL_EFFECT, settled="tool/code-dispatch")
    with pytest.raises(
        IntentError, match=r"already opens an intent declared by 'ph\.session\.kinds'"
    ):
        declare_intent(rival)
    assert declared_intents() == (TOOL_EFFECT,)


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
            replace(
                TOOL_EFFECT,
                opened="sandbox/mode",
                settled="sandbox/denied",
                writer=log_writer(__name__),
            )
        )
    assert declared_intents() == ()


# ------------------------------------------------------------ the journal --
# P10-06. `ctx.intents`, mounted the way a deployment mounts it: by the
# `session` row, over a store a real backend writes.

DURABLE = replace(TOOL_EFFECT, barrier="durable", reopen=frozenset())
"""The shipped effect kind, made durable: its keys, closer and payloads are the real
ones, and only the barrier under test differs."""
BUFFERED = replace(TOOL_DISPATCH, barrier="buffered")
"""The shipped dispatch kind, buffered, for the same reason."""
EFFECT = _effect("send:m1")


async def _journal(mount: MountProfile) -> tuple[Context, IntentJournal, Session]:
    ctx = await mount()
    declare_intent(DURABLE)
    declare_intent(BUFFERED)
    return ctx, ctx.require(INTENTS), ctx.require(SESSIONS).create("journaled")


@pytest.mark.usefixtures("kinds")
async def test_open_is_on_disk_before_the_claim_is_handed_out(mount: MountProfile) -> None:
    """The barrier. *Sabotage:* drop the flush from `open` and the store read
    below finds no `tool/effect` — the claim was handed out for an act the log
    could not yet say was about to happen."""
    ctx, journal, session = await _journal(mount)

    held = await journal.open(session, DURABLE, EFFECT)

    assert isinstance(held, Claim) and held.key == "send:m1"
    _, stored = ctx.require(SESSION_PERSISTENCE).read(session.id)
    assert [event.type for event in stored] == ["tool/effect"]


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
        await journal.open(session, DURABLE, EFFECT)

    assert [event.type for event in session.events] == ["tool/effect", "tool/effect-settled"]
    assert unsettled_why(session.events[-1].data) == "not-started"
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
        await journal.open(session, DURABLE, EFFECT)

    assert [event.type for event in session.events] == ["tool/effect", "tool/effect-settled"]
    assert unsettled_why(session.events[-1].data) == "not-started"


@pytest.mark.usefixtures("kinds")
async def test_a_second_open_of_one_key_returns_the_prior_and_appends_nothing(
    mount: MountProfile,
) -> None:
    _, journal, session = await _journal(mount)
    first = await journal.open_once(session, DURABLE, EFFECT)
    assert isinstance(first, Claim)

    running = await journal.open_once(session, DURABLE, EFFECT)
    assert isinstance(running, Prior) and running.opened == first.opened
    assert (running.settled, running.outcome, running.running_here) == (None, None, True)
    settled = journal.settle(session, first, _effect_done("send:m1"))
    done = await journal.open_once(session, DURABLE, EFFECT)
    assert isinstance(done, Prior) and (done.settled, done.outcome) == (settled, "done")
    assert session.seq == 2, "a prior appends nothing"
    assert fold_intents(session.events, DURABLE)["send:m1"].settled is settled


@pytest.mark.usefixtures("kinds")
async def test_claim_settles_as_failed_when_the_body_raises(mount: MountProfile) -> None:
    """`step/end` in a `finally`, as a method: a caller cannot leave a pair open
    by raising — and a body that settled before it raised is not settled twice."""
    _, journal, session = await _journal(mount)

    with pytest.raises(LookupError):
        async with journal.claim(session, BUFFERED, _dispatch("r1:code:0")) as held:
            assert isinstance(held, Claim)
            raise LookupError("the act failed")
    settle = not_none(fold_intents(session.events, BUFFERED)["r1:code:0"].settled)
    assert outcome_of(BUFFERED, settle) == "outcome-unknown"
    assert settle.data["unsettled"] == {"why": "outcome-unknown", "by": "process"}

    with pytest.raises(LookupError):
        async with journal.claim(session, BUFFERED, _dispatch("r1:code:1")) as held:
            assert isinstance(held, Claim)
            journal.settle(session, held, _dispatch_done("r1:code:1"))
            raise LookupError("after the settle")
    assert [event.type for event in session.events].count("tool/code-dispatch") == 2
    second = not_none(fold_intents(session.events, BUFFERED)["r1:code:1"].settled)
    assert outcome_of(BUFFERED, second) == "done"


@pytest.mark.usefixtures("kinds")
async def test_a_settle_closes_only_its_own_intent_once(mount: MountProfile) -> None:
    _, journal, session = await _journal(mount)
    held = await journal.open(session, DURABLE, EFFECT)
    assert isinstance(held, Claim)

    with pytest.raises(IntentError, match="cannot close the intent opened as 'send:m1'"):
        journal.settle(session, held, _effect_done("send:m2"))
    journal.settle(session, held, _effect_done("send:m1"))
    with pytest.raises(IntentError, match="is not open"):
        journal.settle(session, held, _effect_done("send:m1"))


@pytest.mark.usefixtures("kinds")
async def test_what_the_journal_refuses_to_open(mount: MountProfile) -> None:
    """A kind repair does not know, a record whose key cannot be read, and a
    durable kind opened without its barrier — each refused before any append."""
    _, journal, session = await _journal(mount)
    stranger = replace(DURABLE, reopen=frozenset({"failed"}))

    with pytest.raises(IntentError, match="not a declared intent kind"):
        await journal.open(session, stranger, EFFECT)
    with pytest.raises(IntentError, match="must carry its key"):
        await journal.open(session, BUFFERED, {"name": "bash"})
    with pytest.raises(IntentError, match="is durable; open it with `open`"):
        journal.record(session, DURABLE, EFFECT)
    assert session.seq == 0
    assert journal.record(session, BUFFERED, _dispatch("r1:code:0")).key == "r1:code:0"


@pytest.mark.usefixtures("kinds")
async def test_the_intent_cache_is_polled_as_an_invariant(mount: MountProfile) -> None:
    """The index is a `SessionFoldCache` like the six others, and reported by
    its own row, naming the kind that drifted."""
    ctx, journal, session = await _journal(mount)
    await journal.open(session, DURABLE, EFFECT)
    assert ctx.require(INVARIANTS).verify() == []

    index = cast(dict[str, IntentRecord], journal._index(session, DURABLE))
    index["ghost"] = IntentRecord(opened=session.events[0], settled=None)

    (violation,) = ctx.require(INVARIANTS).verify()
    assert violation.invariant == "intent-fold-cache"
    assert "tool/effect" in violation.detail and "journaled" in violation.detail


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
