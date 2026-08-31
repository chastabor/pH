"""P6-28 — `ph workspaces gc`, and the count that makes the pile visible.

Retention buys a parent the evidence of a child that failed; what it sells is an
unbounded set of checkouts, one per child that ever ended badly, on a disk nobody
is watching. These are the two things that close that trade: a command that
collects what nobody came back for, and a `ph doctor` row that says how many are
being kept even when the answer is none.

Driven through the CLI against a real JSONL store, because what is under test is
precisely the cross-*session* half: the fold's own rules are pinned in ph-core
against hand-written logs, and what a command adds is reading logs it did not
write, out of a process that was not running when they were.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from typer.testing import CliRunner

from ph.persistence import session_path
from ph.session import Session, SessionHeader
from ph_app.cli import app

runner = CliRunner()


def _log(
    sessions: Path, session_id: str, *events: tuple[str, dict[str, Any]], parent: str = ""
) -> Path:
    """One stored session on disk, written before this process starts.

    Through the real envelope rather than a hand-rolled dict: the format is ours,
    and a fixture that spelled it out separately would pin this test to a shape
    the backend has moved on from. What the test supplies that a live run cannot
    is the *provenance* — these logs were not written by the process that reads
    them, which is the only state the collector ever meets.
    """
    path = session_path(sessions, session_id)
    header = SessionHeader(id=session_id, created_at=1, parent_session=parent or None)
    session = Session(session_id, header=header)
    for kind, data in events:
        session.append(kind, data)
    lines = [{"type": "session/header", "header": session.header.to_wire()}]
    lines += [event.to_wire() for event in session.events]
    path.write_text("".join(f"{json.dumps(line)}\n" for line in lines), encoding="utf-8")
    return path


def _retained_tree(
    tmp_path: Path, sessions: Path, session_id: str, reason: str, *, parent: str = ""
) -> Path:
    """A stored session that left one retained tree, and the directory itself."""
    tree = tmp_path / "trees" / session_id
    tree.mkdir(parents=True)
    _log(
        sessions,
        session_id,
        ("workspace/acquired", {"agentId": "a", "kind": "worktree-ephemeral", "root": str(tree)}),
        ("workspace/disposed", {"agentId": "a", "kept": True, "retained": reason}),
        parent=parent,
    )
    return tree


def test_doctor_reports_the_pile_even_when_it_is_empty(tmp_path: Path, roots: Path) -> None:
    """Rule 6: state what is not enforced next to where it would be assumed.

    The assumption a reader makes in the absence of a row is that nothing is
    accumulating, which is exactly the assumption this row exists to check. The
    per-agent rows cannot carry it — `doctor` mounts a profile with no agents, so
    every retained tree is by definition one nobody holds any more.
    """

    result = runner.invoke(app, ["doctor", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "retained trees" in result.stdout
    # The whole phrase: bare "none" also appears two rows up, where the tier
    # reports itself as absent — so the weaker assertion passed against a build
    # that printed no count at all.
    assert "none, across the" in result.stdout


def test_doctor_counts_what_is_being_kept(tmp_path: Path, roots: Path) -> None:
    """The number is a floor rather than a census, and it says so.

    A diagnostic that read a thousand transcripts to print one number is one
    people stop running, so it is bounded by whatever the store lists.
    """
    _retained_tree(tmp_path, roots, "one", "the child was cancelled")
    _retained_tree(tmp_path, roots, "two", "the child failed")

    result = runner.invoke(app, ["doctor", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "2 across 2 of the 2 most recent session(s)" in result.stdout
    assert "ph workspaces gc" in result.stdout, "the count is only useful beside the way out"


def test_gc_reports_before_it_removes(tmp_path: Path, roots: Path) -> None:
    """Reporting is the default and removing is the flag, the opposite way round
    from most `gc`.

    Every tree here was kept *because a run went wrong*, and the person who most
    needs this command is the one who just found the disk full and does not yet
    know what these directories are. They should learn that by typing the obvious
    thing.
    """
    tree = _retained_tree(tmp_path, roots, "one", "the child was cancelled")

    result = runner.invoke(app, ["workspaces", "gc", "--profile", "headless", "--older-than", "0"])

    assert result.exit_code == 0, result.output
    assert "collect" in result.stdout
    assert "the child was cancelled" in result.stdout, "a reason is why a person can decide"
    assert "--remove" in result.stdout
    assert tree.exists(), "the default run removed a checkout nobody asked it to"


def test_gc_refuses_a_tree_inside_the_age_bound(tmp_path: Path, roots: Path) -> None:
    """The bound is what stops last night's failure being collected this morning."""
    tree = _retained_tree(tmp_path, roots, "one", "the child was cancelled")

    result = runner.invoke(app, ["workspaces", "gc", "--profile", "headless", "--remove"])

    assert result.exit_code == 0, result.output
    assert "recent" in result.stdout
    assert "removed 0 of 0" in result.stdout
    assert tree.exists()


def test_gc_collects_only_what_the_tier_can_end(tmp_path: Path, roots: Path) -> None:
    """`headless` mounts `workspace-shared`, which cannot reclaim anything.

    The same refusal `reconcile` makes, and for the sharper reason: removing a
    directory on the strength of a record written by a configuration we are not
    running is the one way this could destroy the work it exists to protect. A
    tree is *cleared* for collection here and still survives, which is the split
    between the rule and the removal doing its job.
    """
    tree = _retained_tree(tmp_path, roots, "one", "the child was cancelled")

    result = runner.invoke(
        app, ["workspaces", "gc", "--profile", "headless", "--older-than", "0", "--remove"]
    )

    assert result.exit_code == 0, result.output
    assert "removed 0 of 1" in result.stdout
    assert tree.exists()


def test_gc_says_so_when_there_is_nothing_retained(tmp_path: Path, roots: Path) -> None:
    """ "Nothing to collect" and "three trees, all still held" are very different
    answers, and a person who typed this deserves to be told which."""
    _log(roots, "quiet", ("session/end-seed", {}))

    result = runner.invoke(app, ["workspaces", "gc", "--profile", "headless"])

    assert result.exit_code == 0, result.output
    assert "no retained trees" in result.stdout


def test_gc_refuses_an_unknown_profile_with_the_same_code(tmp_path: Path, roots: Path) -> None:
    """One resolver, one exit code: `_documents` is shared with `doctor` so the
    two cannot come to disagree about what an unknown `--profile` costs."""

    result = runner.invoke(app, ["workspaces", "gc", "--profile", "nonesuch"])

    assert result.exit_code == 2


def test_gc_narrows_to_one_session_and_the_children_it_spawned(tmp_path: Path, roots: Path) -> None:
    """The enumeration half of P6-28 as a person asks it.

    A child's workspace events are in the *child's* log, so "what did that run
    and its children leave" was reduced to opening each transcript by hand and
    knowing which fields to read. **Descent, not the messaging family**: a
    sibling's checkout is not this run's to collect, and reusing the C7 reach
    rule here would widen a filesystem question with an answer computed for a
    different one.
    """
    _retained_tree(tmp_path, roots, "parent", "the parent kept one")
    _retained_tree(tmp_path, roots, "kid", "the child was cancelled", parent="parent")
    _retained_tree(tmp_path, roots, "stranger", "someone else's run")

    result = runner.invoke(
        app, ["workspaces", "gc", "--profile", "headless", "--session", "parent"]
    )

    assert result.exit_code == 0, result.output
    assert "the parent kept one" in result.stdout
    assert "the child was cancelled" in result.stdout, "a child's log was not folded in"
    assert "someone else" not in result.stdout, "a stranger's tree was offered for collection"
    assert "2 within the age bound" in result.stdout, "the counts describe the narrowed set"


def test_gc_says_which_family_had_nothing(tmp_path: Path, roots: Path) -> None:
    """ "No retained trees" and "no retained trees *under this run*" are different
    answers to a person who narrowed the question themselves."""
    _retained_tree(tmp_path, roots, "elsewhere", "the child was cancelled")
    _log(roots, "quiet", ("session/end-seed", {}))

    result = runner.invoke(app, ["workspaces", "gc", "--profile", "headless", "--session", "quiet"])

    assert result.exit_code == 0, result.output
    assert "no retained trees under quiet" in result.stdout
    # The denominator is *not* narrowed with the filter: it says how much of the
    # store was read, and one that shrank would claim a family's trees had come
    # from every log on disk.
    assert "across the 2 most recent session(s)" in result.stdout
