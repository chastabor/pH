"""P4-14 — the crash half of the workspace pair (F6).

`ctx.effect` unwinds a workspace when the process exits; a process that *dies*
unwinds nothing. What is left behind looks, to `git worktree list`, exactly like
a tree the disposal policy kept on purpose for review — so the filesystem cannot
tell a leak from a feature, and `/workspaces` cannot either.

**The log can.** A `disposed` with `kept: true` is the policy deciding; no
`disposed` at all is a process that died holding the tree. That asymmetry is the
whole of F6, and it is why the pair is written by the seam rather than by each
provider: a pair only reconciles if one place owns both halves.

The fold is tested against hand-written logs because that is what it will meet —
events read off disk by a process that was not running when they were written —
and the gate is tested against a real repository, because "the tree is gone"
is a claim about git rather than about our arithmetic.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ph.seams.workspace import workspace_leaks
from ph.session import Session
from ph.testing import WORKTREE_ROWS, git, needs_git, worktree_agent

pytestmark = pytest.mark.anyio


def _log(*events: tuple[str, dict[str, Any]]) -> Session:
    session = Session("s")
    for kind, data in events:
        session.append(kind, data)
    return session


def _acquired(
    agent: str, root: str, kind: str = "worktree", ref: str = "ph/s/a"
) -> tuple[str, dict[str, Any]]:
    return ("workspace/acquired", {"agentId": agent, "kind": kind, "root": root, "ref": ref})


def _disposed(agent: str, kept: bool = True) -> tuple[str, dict[str, Any]]:
    return ("workspace/disposed", {"agentId": agent, "kept": kept})


async def _reopen(mount: Any, base: Path, session: Session) -> tuple[Any, Session]:
    """The next open: a second process, the same log, and the drain that makes
    the detached listener observable.

    The `fs` root is pointed at the repository because that is what a person
    resuming in their project has, and the three facts here — the seed, the
    header, the drain — are the ones a reconciliation test must get right.
    """
    ctx = await mount(*WORKTREE_ROWS, {"id": "fs", "config": {"root": str(base)}})
    revived = ctx.sessions.adopt(
        Session(session.id, seed=list(session.events), header=session.header)
    )
    await ctx.drain()
    return ctx, revived


# ---------------------------------------------------------------------- fold --


def test_an_unclosed_acquire_is_a_leak() -> None:
    session = _log(_acquired("a", "/trees/a"))

    (leak,) = workspace_leaks(session)

    assert leak.agent_id == "a"
    assert leak.root == Path("/trees/a")
    assert leak.ref == "ph/s/a"


def test_a_closed_pair_is_not_a_leak() -> None:
    """Including `kept: true` — the disposal policy keeping a dirty tree for
    review is the *feature*, and reconciliation that reclaimed it would delete
    the work the policy was protecting."""
    session = _log(_acquired("a", "/trees/a"), _disposed("a", kept=True))

    assert workspace_leaks(session) == []


def test_a_shared_workspace_leaks_no_directory() -> None:
    """An unclosed pair on `shared` records a crash and no stray: its root *is*
    the base, so there is nothing to reclaim and the person's own checkout is
    the last thing this row should touch."""
    session = _log(_acquired("a", "/project", kind="shared"))

    assert workspace_leaks(session) == []


def test_re_acquiring_after_release_leaks_only_the_live_one() -> None:
    """An agent that released and took another tree is ordinary. Only the last
    unclosed acquire is outstanding — a list would report the released one too."""
    session = _log(
        _acquired("a", "/trees/first"),
        _disposed("a"),
        _acquired("a", "/trees/second"),
    )

    (leak,) = workspace_leaks(session)

    assert leak.root == Path("/trees/second")


def test_each_agent_is_folded_separately() -> None:
    """A fan-out is the case this exists for: one child's clean exit says
    nothing about its siblings."""
    session = _log(
        _acquired("a", "/trees/a"),
        _acquired("b", "/trees/b"),
        _acquired("c", "/trees/c"),
        _disposed("b"),
    )

    assert sorted(one.agent_id for one in workspace_leaks(session)) == ["a", "c"]


# ---------------------------------------------------------------------- gate --


@needs_git
async def test_a_crash_between_acquire_and_dispose_is_reconciled_on_the_next_open(
    mount: Any, tmp_path: Path
) -> None:
    """P4-14's gate, against a real repository.

    The crash is simulated the only honest way: acquire a real worktree, write
    the `acquired` event, and then throw the process away *without* unwinding —
    which is what `dispose()` not running means. A second mount then opens the
    same log and must find the tree gone.
    """
    _ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    base = tmp_path / "repo"
    leaked = workspace.root
    # The process dies here: no scope disposal, so no `workspace/disposed`. The
    # fold is checked against a *real* acquire payload here, so the hand-written
    # logs above cannot silently drift from what the seam writes.
    assert workspace_leaks(session) != []

    reopened, _revived = await _reopen(mount, base, session)

    assert not leaked.exists(), "the leaked worktree survived the next open"
    _code, out, _ = await git(reopened, base, "worktree", "list", "--porcelain")
    assert str(leaked) not in out, "git still has the worktree registered"


@needs_git
async def test_a_leak_whose_tree_is_already_gone_still_closes_its_pair(
    mount: Any, tmp_path: Path
) -> None:
    """Nothing to reclaim is not nothing to record.

    A leak reported and left open is one reported at every future open, and a
    record nobody can act on twice is noise that hides the one that matters.
    """
    _ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    shutil.rmtree(workspace.root)

    _reopened, revived = await _reopen(mount, tmp_path / "repo", session)

    assert workspace_leaks(revived) == []
    closing = [event for event in revived.events if event.type == "workspace/disposed"]
    assert closing and closing[-1].data["reconciled"] is True
    assert closing[-1].data["kept"] is False, "a tree that is gone was reported as kept"


@needs_git
async def test_a_dirty_leak_is_kept_rather_than_discarded(mount: Any, tmp_path: Path) -> None:
    """A crash is not a reason to throw away work. Reconciliation runs the same
    disposal policy an orderly release runs, so a tree with the agent's work in
    it survives — reconciliation that discarded more than a normal exit would
    make crashing *worse* than the leak it is fixing.
    """
    _ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    (workspace.root / "unfinished.txt").write_text("half a thought\n", encoding="utf-8")

    _reopened, revived = await _reopen(mount, tmp_path / "repo", session)

    assert (workspace.root / "unfinished.txt").exists(), "reconciliation discarded the work"
    closing = [event for event in revived.events if event.type == "workspace/disposed"]
    assert closing[-1].data["kept"] is True


@needs_git
async def test_forking_does_not_reclaim_the_parents_live_worktree(
    mount: Any, tmp_path: Path
) -> None:
    """The defect this row shipped for one commit, and the reason the fold starts
    at `seed_length`.

    `sessions.fork` seeds the child with the parent's transcript and publishes it
    through `session/created` like any other session — so a fold over the whole
    log reports the parent's **still-held** worktree as the child's leak, and
    reconciliation removes a tree an agent is actively working in. Two things now
    stop it: the fold reads only what this session acquired, and the seam skips
    what it is still holding.
    """
    ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)

    child = ctx.sessions.fork(session)
    await ctx.drain()

    assert workspace_leaks(child) == [], "the fork folded its parent's live worktree as a leak"
    assert workspace.root.is_dir(), "forking reclaimed the parent's live worktree"


@needs_git
async def test_a_held_workspace_is_never_reconciled(mount: Any, tmp_path: Path) -> None:
    """Belt and braces, on the seam's own knowledge. `live()` exists so
    `/workspaces` can ask "is this tree anybody's" before offering to delete a
    directory; a reconciler with a second, weaker answer to that question is how
    the two come to disagree about a tree someone is working in."""
    ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    assert workspace_leaks(session) != []

    await ctx.workspace.reconcile(session)

    assert workspace.root.is_dir(), "the seam reclaimed a workspace it still holds"


async def test_a_leak_no_mounted_tier_can_reclaim_is_left_alone(
    mount: Any, tmp_path: Path, caplog: Any
) -> None:
    """Reported, not removed. The tree belongs to a tier this profile does not
    have, and deleting a directory on the strength of a record written by a
    configuration we are not running is the one way this row could destroy the
    work it exists to protect."""
    tree = tmp_path / "trees" / "a"
    tree.mkdir(parents=True)
    ctx = await mount()
    session = Session("crashed")
    session.append(
        "workspace/acquired",
        {"agentId": "a", "kind": "worktree", "root": str(tree), "ref": "ph/s/a"},
    )

    with caplog.at_level("WARNING"):
        ctx.sessions.adopt(session)
        await ctx.drain()

    assert tree.exists(), "a tree no mounted tier owns was removed anyway"
    assert "no mounted tier can reclaim" in caplog.text
