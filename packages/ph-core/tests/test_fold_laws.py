"""The fold laws are checkable, and every fold ph-core caches passes them.

`ph.session.folds` says nothing there can check that a cached function is a pure
fold of the prefix. `ph.testing.folds` checks it from the process that appended
the log. The first half of this file shows each law catching the mistake it is
for — a second implementation of the fold that drifts, a clock read, hidden
state, a fold that writes — and the second half holds every `SessionFoldCache`
consumer in this package to all of them, on logs their own services wrote.
"""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Any

import pytest

from ph.commands.sandbox import denial_count, extend_denial_count
from ph.seams.goals import GATE, Goal, GoalService, extend_goals, goals
from ph.seams.sandbox import DENIED
from ph.seams.schedule import Schedule, ScheduleService, schedules
from ph.seams.subagents import ADMITTED, DELETED, STATUS, USAGE, subagent_roster
from ph.session import Session, SessionFoldCache, SessionHeader, SurfaceIntent, now_ms
from ph.testing import (
    VerifyingFoldCache,
    assert_fold_laws,
    assistant_payload,
    check_fold_laws,
    prefix_of,
)

# ------------------------------------------------------------------ the laws --


def _turns(log: Any) -> int:  # noqa: ANN401
    return sum(1 for event in log.events if event.type == "turn/start")


def _more_turns(previous: int, log: Any, from_seq: int) -> int:  # noqa: ANN401
    return previous + sum(1 for event in log.events_from(from_seq) if event.type == "turn/start")


def _fork(parent: Session, child_id: str = "child") -> Session:
    """A child seeded with the whole of `parent`, carrying the fork boundary.

    `seed_length` is the half that matters here: `goals` and `schedules` both fold
    from it, so a child without one folds its parent's history as its own.
    """
    header = SessionHeader(id=child_id, created_at=now_ms(), seed_length=parent.seq)
    return Session(child_id, seed=list(parent.events), header=header)


def _turn_log(turns: int = 3) -> Session:
    session = Session("s")
    for turn in range(1, turns + 1):
        session.append("turn/start", {"turn": turn})
        session.append("assistant/chunk", {"text": "…"})
        session.append("turn/end", {"turn": turn, "reason": {"kind": "completed"}})
    return session


def test_a_lawful_fold_has_no_findings() -> None:
    assert check_fold_laws(_turn_log(), _turns, _more_turns) == []
    assert check_fold_laws(_turn_log(), _turns) == []


def test_a_prefix_is_the_same_events_under_the_same_header_and_nothing_more() -> None:
    """Seeding would add a `session/end-seed`; a prefix must not."""
    session = _turn_log()
    prefix = prefix_of(session, 4)

    assert prefix.events == session.events[:4]
    assert prefix.header == session.header
    assert prefix.seq == 4


def test_a_second_implementation_that_drifts_is_caught_at_the_event_that_breaks_it() -> None:
    """The finding names the event, because that is the lesson: which type the two
    paths handle differently."""

    def counts_ends_too(previous: int, log: Any, from_seq: int) -> int:  # noqa: ANN401
        return previous + sum(
            1 for event in log.events_from(from_seq) if event.type in {"turn/start", "turn/end"}
        )

    findings = check_fold_laws(_turn_log(), _turns, counts_ends_too)

    walked = [one for one in findings if one.startswith("extending step by step")]
    assert len(walked) == 1
    assert "seq 2 (turn/end)" in walked[0], walked[0]


@pytest.mark.parametrize(
    ("read", "named"),
    [
        (time.time, "the clock through time.time"),
        (lambda: Path("/").exists(), "the filesystem through posix.stat"),
        (random.random, "a random source through Random.random"),
    ],
)
def test_a_fold_that_reads_outside_the_log_is_caught(read: Any, named: str) -> None:  # noqa: ANN401
    """Deterministic in its answer, impure in how it got there — which is exactly
    the fold the cache's invalidation key cannot see."""

    def peeks(log: Any) -> int:  # noqa: ANN401
        read()
        return _turns(log)

    findings = check_fold_laws(_turn_log(), peeks)
    assert findings == [f"compute reads outside the log: {named}"]


def test_a_fold_with_hidden_state_is_caught() -> None:
    calls = [0]

    def remembers(log: Any) -> int:  # noqa: ANN401
        calls[0] += 1
        return calls[0]

    findings = check_fold_laws(_turn_log(), remembers)
    assert any(one.startswith("compute is not deterministic") for one in findings), findings


def test_a_fold_that_writes_to_the_log_is_refused_before_anything_else() -> None:
    def appends(log: Any) -> int:  # noqa: ANN401
        log.append("turn/start", {"turn": 99})
        return _turns(log)

    findings = check_fold_laws(_turn_log(), appends)
    assert findings == ["the fold appended to the log: seq went from 9 to 10"]


def test_extending_over_nothing_must_change_nothing() -> None:
    def always_one_more(previous: int, log: Any, from_seq: int) -> int:  # noqa: ANN401
        return _more_turns(previous, log, from_seq) + 1

    findings = check_fold_laws(_turn_log(), _turns, always_one_more)
    assert any(one.startswith("extend over an empty slice") for one in findings), findings


def test_a_reader_that_mutates_what_it_was_handed_is_caught_by_the_verifying_cache() -> None:
    """The one failure no law over the fold can see: the fold was right, and a
    consumer changed the answer in place for everyone who reads it next."""
    cache: VerifyingFoldCache[dict[str, int]] = VerifyingFoldCache(
        lambda log: {"turns": _turns(log)}
    )
    session = _turn_log()

    handed = cache.read(session)
    handed["turns"] = 99

    with pytest.raises(AssertionError, match=r"cached fold .* does not equal the cold fold"):
        cache.read(session)


def test_prefixes_begin_where_the_cache_could_first_have_read() -> None:
    """A forked session was never at seq 0 in this process, so a fold whose cold
    path starts at the seed boundary — `goals` and `schedules` both do — is lawful
    as far as any cache can tell, and only a check told to start at 0 disagrees."""
    child = _fork(_turn_log(2))
    child.append("turn/start", {"turn": 3})

    def own_turns(log: Any) -> int:  # noqa: ANN401
        since = log.header.seed_length or 0
        return sum(1 for event in log.events_from(since) if event.type == "turn/start")

    assert check_fold_laws(child, own_turns, _more_turns) == []
    assert check_fold_laws(child, own_turns, _more_turns, start=0) != []


def test_a_fold_that_reads_process_state_is_caught_as_reading_more_than_the_log() -> None:
    """`first_live_seq` is this process's relationship to the log, not a fact
    about it; a replica of the same events starts at 0, and the fold must agree."""
    parent = _turn_log(2)
    child = Session("child", seed=list(parent.events))
    child.append("turn/start", {"turn": 3})

    def live_turns(log: Any) -> int:  # noqa: ANN401
        return sum(1 for event in log.events_from(log.first_live_seq) if event.type == "turn/start")

    findings = check_fold_laws(child, live_turns)
    assert any(one.startswith("compute reads more than the log") for one in findings), findings


# ------------------------------------------------------------------ the poll --


def test_the_poll_says_nothing_about_a_cache_that_is_holding() -> None:
    cache: SessionFoldCache[int] = SessionFoldCache(_turns)
    session = _turn_log()
    cache.read(session)

    assert cache.stale([session]) == []


def test_the_poll_catches_a_reader_that_mutated_what_it_was_handed() -> None:
    """The failure the invalidation key structurally cannot see: the log did not
    grow, so the entry still looks current, and every later reader gets the
    poisoned value."""
    cache: SessionFoldCache[dict[str, int]] = SessionFoldCache(lambda log: {"turns": _turns(log)})
    session = _turn_log()

    cache.read(session)["turns"] = 99

    (found,) = cache.stale([session])
    assert found == (
        f"session s: the value cached at seq {session.seq} does not equal the fold of its log"
    )


def test_the_poll_catches_a_fold_that_answered_from_outside_the_log() -> None:
    """A clock-reading fold changes its answer without the log growing. The cache
    cannot notice; a poll that refolds can."""
    answers = iter([1, 2])
    cache: SessionFoldCache[int] = SessionFoldCache(lambda _log: next(answers))
    session = _turn_log()
    cache.read(session)

    assert len(cache.stale([session])) == 1


def test_an_entry_the_log_has_outgrown_is_not_drift() -> None:
    """The ordinary state of a cache between reads: `read` folds the new slice
    before serving it, so reporting this would be an alarm on every append."""
    cache: SessionFoldCache[int] = SessionFoldCache(_turns, extend=_more_turns)
    session = _turn_log()
    cache.read(session)
    session.append("turn/start", {"turn": 99})

    assert cache.stale([session]) == []


def test_the_poll_can_only_speak_for_sessions_it_is_handed() -> None:
    """A cached session nobody can reach has no log left to fold, so it is skipped
    rather than reported — an alarm about a projection of nothing is one people
    learn to ignore."""
    cache: SessionFoldCache[dict[str, int]] = SessionFoldCache(lambda log: {"turns": _turns(log)})
    session = _turn_log()
    cache.read(session)["turns"] = 99

    assert cache.stale([]) == []
    assert len(cache.stale([session])) == 1


# ------------------------------------------------ the folds this package caches --


def _goal_log(session: Session) -> Session:
    service = GoalService()
    goal = service.set(session, Goal(id="g1", objective="make the tests pass", gates=["pytest"]))
    session.append("turn/start", {"turn": 1})
    service.continued(session, goal.id)
    session.append("turn/end", {"turn": 1, "reason": {"kind": "completed"}})
    session.append(
        "assistant/message",
        {
            **assistant_payload("done", "m1"),
            "usage": {"inputTokens": 900, "outputTokens": 100},
        },
        SurfaceIntent("append"),
    )
    session.append(GATE, {"id": goal.id, "gate": "pytest", "tree": "abc", "passed": False})
    session.append(GATE, {"id": goal.id, "gate": "pytest", "tree": "def", "passed": True})
    service.settle(session, goal.id, "achieved")
    service.set(session, Goal(id="g2", objective="then tidy up"))
    session.append("turn/end", {"turn": 2, "reason": {"kind": "completed"}})
    return session


def test_the_goal_fold_obeys_the_laws() -> None:
    assert_fold_laws(_goal_log(Session("goals")), goals, extend_goals)


def test_the_goal_fold_obeys_the_laws_on_a_fork() -> None:
    """The cold path starts at the seed boundary and the incremental one at the
    last read; on a fork those are different numbers, and must not matter."""
    child = _goal_log(_fork(_goal_log(Session("parent"))))

    assert_fold_laws(child, goals, extend_goals)


def test_the_schedule_fold_obeys_the_laws() -> None:
    session, service = Session("sched"), ScheduleService()
    service.create(session, Schedule(id="s1", kind="interval", spec="60000", prompt="go"))
    created = schedules(session)["s1"].created_at
    service.create(session, Schedule(id="s2", kind="once", spec=str(created + 1), prompt="go"))
    service.claim(session, now=created + 120_000)
    service.cancel(session, "s1")

    assert_fold_laws(session, schedules)


def test_the_subagent_roster_obeys_the_laws() -> None:
    session = Session("parent")
    for run_id, name in (("r1", "scout"), ("r2", "recon")):
        session.append(
            ADMITTED, {"runId": run_id, "name": name, "model": "fake-1", "grantedAccess": "read"}
        )
    session.append(STATUS, {"runId": "r1", "status": "running"})
    session.append(USAGE, {"runId": "r1", "childUsage": {"inputTokens": 400, "outputTokens": 2}})
    session.append(STATUS, {"runId": "r1", "status": "running", "cause": "resumed"})
    session.append(STATUS, {"runId": "r1", "status": "done"})
    session.append(STATUS, {"runId": "r2", "status": "done"})
    session.append(DELETED, {"runId": "r2", "reason": "user"})

    assert_fold_laws(session, subagent_roster)


def test_the_sandbox_refusal_count_obeys_the_laws() -> None:
    session = Session("boxed")
    session.append("turn/start", {"turn": 1})
    for host in ("example.com", "example.org"):
        session.append(DENIED, {"kind": "network", "via": "proxy", "message": host, "host": host})
        session.append("assistant/chunk", {"text": "…"})

    assert_fold_laws(session, denial_count, extend_denial_count)
