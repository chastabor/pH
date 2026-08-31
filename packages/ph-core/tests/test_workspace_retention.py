"""P6-28 — the three things a directory on disk can mean, and who may remove it.

`git worktree list` reports a deliberate keep, a dirty-tree keep and a leak
identically. The log can tell them apart, and this is the fold that does — plus
the two halves that make the retention policy affordable: a parent enumerating
what its whole family left, and a collector that removes only what nobody came
back for.

The fold is tested against hand-written logs because that is what it will meet:
events read off disk by a process that was not running when they were written.
The collector's verdicts are tested against a `WorkspaceSeam` with no provider,
because the verdicts are a *rule* and the removal is a tier's business — the two
failing separately is the point of the split.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from ph.persistence.protocol import StoredSession
from ph.seams.subagents import descendants, reachable_family
from ph.seams.workspace import (
    WorkspaceRecord,
    family_survivors,
    stored_survivors,
    workspace_leaks,
    workspace_survivors,
)
from ph.session import Session, SessionHeader
from ph.testing import (
    workspace_acquired,
    workspace_seam,
)
from ph.testing import (
    workspace_disposed as _disposed,
)
from ph.testing import (
    workspace_log as _log,
)
from ph.testing import (
    workspace_retained as _retained,
)

pytestmark = pytest.mark.anyio

DAY = 86400.0


def _acquired(
    agent: str, root: str, kind: str = "worktree-ephemeral"
) -> tuple[str, dict[str, Any]]:
    """`workspace_acquired`, defaulted to the kind this module is about."""
    return workspace_acquired(agent, root, kind=kind)


def _record(root: Path, *, session_id: str = "s", **extra: Any) -> WorkspaceRecord:
    """A closed record with a reason — which is what `outcome == "retained"` is.

    Stated as the two facts rather than as the outcome, because the outcome is
    derived from them: a helper that could set them apart would be able to build
    a record the fold cannot produce.
    """
    return WorkspaceRecord(
        agent_id="a",
        kind="worktree-ephemeral",
        root=root,
        reason="error",
        closed=True,
        session_id=session_id,
        **extra,
    )


# ------------------------------------------------------------------- the fold --


def test_the_three_survivors_are_told_apart() -> None:
    """The claim the row is built on, as one assertion over one log.

    All three leave a directory; nothing outside the log distinguishes them.
    A deliberate keep says why, a dirty-tree keep is `kept` with no reason, and
    a leak has no closing event at all.
    """
    session = _log(
        _acquired("kept", "/trees/kept"),
        _disposed("kept", kept=True),
        _acquired("said-why", "/trees/said-why"),
        _disposed("said-why", kept=True, retained="cancelled at parent-teardown"),
        _acquired("leaked", "/trees/leaked"),
    )

    found = {one.agent_id: one for one in workspace_survivors(session)}

    assert found["kept"].outcome == "kept"
    assert found["kept"].reason == ""
    assert found["said-why"].outcome == "retained"
    assert found["said-why"].reason == "cancelled at parent-teardown"
    assert found["leaked"].outcome == "leaked"


def test_a_removed_tree_is_not_a_survivor() -> None:
    """`kept: false` is the policy saying the directory is gone.

    The fold's whole value is that it describes what is *on disk*; a record for a
    tree the disposal policy removed would have the collector chasing paths that
    do not exist and `ph doctor` reporting a pile nobody has.
    """
    session = _log(_acquired("a", "/trees/a"), _disposed("a", kept=False))

    assert workspace_survivors(session) == []


def test_a_retention_survives_the_crash_it_was_written_for() -> None:
    """Why the mark is its own event rather than only a field on the closing half.

    A retention is decided *because* a run went wrong, and the most complete way
    for one to go wrong is for the process to die — which writes no `disposed` at
    all. A mark that lived only in memory would be lost by exactly the failure it
    exists to survive.

    The record is then both things at once, which is why `closed` is a separate
    field: reconciliation still owes this pair a closing event, and it is
    `reclaim` — not the fold — that knows a reason means leave the tree alone.
    """
    session = _log(_acquired("a", "/trees/a"), _retained("a", "the child was cancelled"))

    (record,) = workspace_survivors(session)

    assert record.outcome == "retained"
    assert record.reason == "the child was cancelled"
    assert record.closed is False
    assert [one.agent_id for one in workspace_leaks(session)] == ["a"], (
        "an unclosed pair is still owed its closing event, retained or not"
    )


def test_a_clean_settle_withdraws_the_mark() -> None:
    """The other half of retain-by-default.

    The shipped policy marks the discardable kind at acquire, because the window
    to mark closes with the child's scope. So a clean finish is a caller saying
    "never mind", and it has to be as durable as the mark it withdraws —
    otherwise a successful child pins a checkout forever and the kind's promise
    is inverted for every outcome rather than for the ones that need evidence.
    """
    session = _log(
        _acquired("a", "/trees/a"),
        _retained("a", "the child has not settled cleanly"),
        _retained("a", ""),
    )

    (record,) = workspace_survivors(session)

    assert record.outcome == "leaked", "a withdrawn mark leaves an ordinary open pair"
    assert record.reason == ""


def test_a_reason_beats_a_disposal_that_says_otherwise() -> None:
    """A closing half that forgot the reason does not lose the decision.

    The orderly release repeats it, so the two agree; a reconciled close writes
    `kept` and no reason at all. Reading the pair as authoritative there would
    have reconciliation quietly downgrade every retained tree it closed.
    """
    session = _log(
        _acquired("a", "/trees/a"),
        _retained("a", "error"),
        _disposed("a", kept=True, reconciled=True),
    )

    (record,) = workspace_survivors(session)

    assert (record.outcome, record.reason, record.closed) == ("retained", "error", True)


def test_a_mark_for_a_tree_this_session_never_took_is_ignored() -> None:
    """A `retained` with no open acquire names somebody else's directory.

    Reachable through a seeded fork: the parent's events are below `seed_length`,
    so the acquire is invisible here while a later mark is not. Answering would
    be this session claiming a tree it does not hold.
    """
    session = _log(_retained("a", "error"))

    assert workspace_survivors(session) == []


def test_each_tree_an_agent_left_is_its_own_record() -> None:
    """Closed records are a list where open ones are a dict, and this is why.

    Re-acquire after release is ordinary, so among *open* records the last one
    per agent is the live one. But each closed record is a distinct directory
    that was really left behind, and keying those by agent would hide every tree
    but the last for an agent that took three.
    """
    session = _log(
        _acquired("a", "/trees/first"),
        _disposed("a", kept=True, retained="one"),
        _acquired("a", "/trees/second"),
        _disposed("a", kept=True, retained="two"),
    )

    assert [str(one.root) for one in workspace_survivors(session)] == [
        "/trees/first",
        "/trees/second",
    ]


def test_the_fold_starts_after_the_seed() -> None:
    """A fork seeds the child with the parent's transcript.

    Folding from the beginning reports the parent's still-held worktrees as the
    child's, and the collector would then offer to remove a tree an agent is
    actively working in — worse than the accumulation it is fixing.
    """
    parent = _log(_acquired("p", "/trees/p"))
    child = Session(
        "c",
        seed=list(parent.events),
        header=SessionHeader(id="c", created_at=1, seed_length=len(parent.events)),
    )
    child.append(*_acquired("c", "/trees/c"))

    assert [one.agent_id for one in workspace_survivors(child)] == ["c"]


# ------------------------------------------------------------------ discovery --


def _family(*links: tuple[str, str | None]) -> list[Session]:
    return [
        Session(child, header=SessionHeader(id=child, created_at=1, parent_session=parent))
        for child, parent in links
    ]


def _lineage(sessions: list[Session]) -> list[tuple[str, str | None]]:
    """What `descendants` actually walks: `(id, parent)`, not whole sessions.

    Spelled here rather than passing sessions, because taking pairs is what lets
    the collector narrow a *listing* before it reads a single log — and a test
    that fed it sessions would not be exercising the shape every caller uses.
    """
    return [(session.id, session.header.parent_session) for session in sessions]


def test_descent_is_transitive_where_the_messaging_family_is_not() -> None:
    """The reuse this deliberately does not make (I7).

    `reachable_family` answers "who may this agent address" — the C7 nuclear
    family, which includes siblings and the parent. `descendants` answers "whose
    leftovers are mine to account for", and the two must not be the same set: a
    sibling's worktree is not this agent's to enumerate, still less to collect,
    and borrowing the messaging rule would widen a filesystem question with an
    answer computed for a different one.

    Transitive for the reason read the other way: a grandchild that failed is
    evidence its grandparent is the only live party left to look at, because the
    child that spawned it settled too.
    """
    sessions = _family(("root", None), ("kid", "root"), ("grandkid", "kid"), ("other", None))

    assert descendants(_lineage(sessions), "root") == ["root", "kid", "grandkid"]
    assert reachable_family(sessions, "root") == {
        "root": "self",
        "kid": "child",
        "other": "sibling",
    }
    assert "grandkid" not in reachable_family(sessions, "root"), (
        "one hop is right for a message and wrong for an inventory"
    )
    assert "other" not in descendants(_lineage(sessions), "root"), (
        "roots are siblings for messaging and strangers for evidence"
    )


def test_a_cycle_in_the_headers_does_not_hang_the_walk() -> None:
    """A corrupted or hand-edited header claiming its own ancestor as a parent."""
    sessions = _family(("a", "b"), ("b", "a"))

    assert sorted(descendants(_lineage(sessions), "a")) == ["a", "b"]


def test_a_parent_reads_what_its_family_left_without_opening_a_child_log() -> None:
    """The fold P6-28 exists to make possible.

    `workspace_survivors` answers for one session, and a child's workspace events
    are in the *child's* log — so a parent asking "what did my children leave"
    was reduced to opening each transcript by hand and knowing which fields to
    read. Ordered parent-first, then by descent, because that is the order the
    answer is read in.
    """
    parent = _log(_acquired("p", "/trees/p"), _disposed("p", kept=True), session_id="p")
    kid = Session("kid", header=SessionHeader(id="kid", created_at=1, parent_session="p"))
    kid.append(*_acquired("k", "/trees/k"))
    kid.append(*_disposed("k", kept=True, retained="error"))
    stranger = _log(_acquired("x", "/trees/x"), _disposed("x", kept=True), session_id="x")

    found = family_survivors([parent, kid, stranger], "p")

    assert [str(one.root) for one in found] == ["/trees/p", "/trees/k"]
    assert [one.session_id for one in found] == ["p", "kid"], (
        "a record has to name the log it came from; the collector reads back through it"
    )


def test_an_unlisted_agent_yields_nothing_rather_than_raising() -> None:
    """The collector reads what a backend chose to list, and a truncated listing
    is an ordinary answer rather than an error."""
    assert family_survivors(_family(("a", None)), "not-listed") == []


# ----------------------------------------------------------------- collection --


def test_only_retained_trees_are_ever_collectable(tmp_path: Path) -> None:
    """The boundary that is the whole safety argument.

    A `kept` tree is a dirty checkout the disposal policy left for a person to
    inspect and merge — that is their work, and `/workspaces remove` is the
    deliberate way to end it. A `leaked` tree belongs to `reconcile`, which is
    the only thing that can tell "the process died" from "the process is
    running". What this row created, and therefore all it may collect, is the
    pile of evidence a *policy* retained without anybody asking for it one tree
    at a time.
    """
    tmp_path.mkdir(exist_ok=True)
    session = _log(
        _acquired("kept", str(tmp_path)),
        _disposed("kept", kept=True),
        _acquired("said-why", str(tmp_path)),
        _disposed("said-why", kept=True, retained="error"),
        _acquired("leaked", str(tmp_path)),
        # Retained *and* unclosed: marked while the agent was live, by a process
        # that then died or is still running. Reconciliation owes this pair its
        # closing event and only it can tell those two apart, so an outcome test
        # alone would have the collector removing a tree from under a live agent.
        _acquired("still-open", str(tmp_path)),
        _retained("still-open", "error"),
    )

    rows = workspace_seam(tmp_path / "scratch").collectable(
        workspace_survivors(session), older_than=DAY, now=10 * DAY, touched={"s": 0.0}
    )

    assert [(one.record.agent_id, one.verdict) for one in rows] == [("said-why", "collect")]


async def test_the_three_refusals(tmp_path: Path) -> None:
    """`held`, `recent` and `gone`, each of which a person may want to argue with.

    A verdict per record rather than a filtered list, because "nothing to
    collect" and "three trees, all still held by a live agent" are very
    different answers to `ph workspaces gc`.

    `held` comes from a real acquire rather than from a set the caller hands in.
    A second, id-keyed "the caller says this session is live" was drafted and
    removed unused — nothing wired one in, and an unexercised branch inside the
    one function that may delete a directory is worse than the gap it reached
    for. The tree above is refused because the seam is *holding* it.
    """
    seam = workspace_seam(tmp_path / "scratch")
    held, fresh, old, missing = (tmp_path / name for name in ("held", "fresh", "old", "gone"))
    for tree in (held, fresh, old):
        tree.mkdir()
    await seam.acquire(session_id="live", agent_id="a", base=held)

    rows = seam.collectable(
        [
            _record(held, session_id="live"),
            _record(fresh, session_id="fresh"),
            _record(missing, session_id="old"),
            _record(old, session_id="old"),
        ],
        older_than=7 * DAY,
        now=100 * DAY,
        touched={"live": 0.0, "fresh": 99 * DAY, "old": 0.0},
    )

    assert [one.verdict for one in rows] == ["held", "recent", "gone", "collect"]


def test_a_tree_that_cannot_be_dated_is_refused(tmp_path: Path) -> None:
    """ "I could not date this" and "this is old" are different answers, and only
    one of them may delete a checkout."""
    (tmp_path / "here").mkdir()

    (row,) = workspace_seam(tmp_path / "scratch").collectable(
        [_record(tmp_path / "here", session_id="unlisted")],
        older_than=DAY,
        now=100 * DAY,
        touched={},
    )

    assert row.verdict == "recent"


async def test_a_live_workspace_is_refused_by_root_not_by_name(tmp_path: Path) -> None:
    """Matched by root path for `live()`'s reason.

    `sanitize_ref` is lossy, so an agent id that does not sanitize to itself
    would read as unheld and lose the refusal that protects it. The record here
    names an id nothing holds, and the *path* is what saves it.
    """
    seam = workspace_seam(tmp_path / "scratch")
    (tmp_path / "held").mkdir()
    await seam.acquire(session_id="s", agent_id="agent/with slashes", base=tmp_path / "held")

    (row,) = seam.collectable(
        [_record(tmp_path / "held", session_id="s")],
        older_than=DAY,
        now=100 * DAY,
        touched={"s": 0.0},
    )

    assert row.verdict == "held"


async def test_collecting_without_a_reclaiming_tier_removes_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A profile whose tier cannot reclaim reports rather than guesses.

    The same refusal `reconcile` makes, and for the sharper reason: removing a
    directory on the strength of a record written by a configuration we are not
    running is the one way this could destroy the work it exists to protect.

    The **sentence** is asserted, not just the empty answer. Silence and "no
    mounted tier can collect these" are the same return value, and only one of
    them tells an operator staring at a full disk why `gc` did nothing.
    """
    (tmp_path / "here").mkdir()
    seam = workspace_seam(tmp_path / "scratch")

    rows = seam.collectable(
        [_record(tmp_path / "here")], older_than=DAY, now=100 * DAY, touched={"s": 0.0}
    )

    assert [one.verdict for one in rows] == ["collect"]
    with caplog.at_level(logging.WARNING, logger="ph.seams.workspace"):
        assert await seam.collect(rows) == []
    assert "no mounted tier can collect" in caplog.text
    assert str(tmp_path / "here") in caplog.text, "the refusal has to name what it left"
    assert (tmp_path / "here").exists()


# --------------------------------------------------------------- the store fold --


class _Store:
    """Enough of `SessionPersistence` to be folded, including one bad log."""

    def __init__(self, sessions: list[Session], *, unreadable: str = "") -> None:
        self.sessions = {session.id: session for session in sessions}
        self.unreadable = unreadable

    def stored(self, *, limit: int = 50) -> list[Any]:
        return [
            StoredSession(session_id=one, modified=float(index))
            for index, one in enumerate(self.sessions)
        ][:limit]

    def read(self, session_id: str) -> tuple[SessionHeader, list[Any]]:
        if session_id == self.unreadable:
            raise ValueError("a half-written log")
        session = self.sessions[session_id]
        return session.header, list(session.events)


def test_the_store_fold_skips_a_log_it_cannot_read() -> None:
    """The direction is the safe one in both consumers.

    An unreadable log contributes no records, so `ph doctor` under-counts and the
    collector removes nothing — which is what you want from a half-written file.
    """
    good = _log(
        _acquired("a", "/trees/a"),
        _disposed("a", kept=True, retained="error"),
        session_id="good",
    )
    bad = _log(_acquired("b", "/trees/b"), session_id="bad")

    survivors, touched = stored_survivors(_Store([good, bad], unreadable="bad"))

    assert [one.session_id for one in survivors] == ["good"]
    assert sorted(touched) == ["bad", "good"], "a log that would not read was still listed"
