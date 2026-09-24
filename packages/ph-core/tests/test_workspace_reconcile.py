"""P4-14 — the crash half of the workspace pair (F6).

`ctx.effect` unwinds a workspace when the process exits; a process that *dies*
unwinds nothing. What is left behind looks, to `git worktree list`, exactly like
a tree a live agent is working in — so the filesystem cannot tell a leak from a
running run, and `/workspaces` cannot either.

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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph.keys import SESSIONS, WORKSPACE
from ph.seams.workspace import WorkspaceRecord, workspace_leaks
from ph.session import Session
from ph.testing import (
    MountProfile,
    log_event,
)
from ph.testing import (
    workspace_acquired as _acquired,
)
from ph.testing import (
    workspace_disposed as _disposed,
)
from ph.testing import (
    workspace_log as _log,
)
from ph.testing.git import WORKTREE_ROWS, git, worktree_agent

pytestmark = pytest.mark.anyio


async def _reopen(mount: MountProfile, base: Path, session: Session) -> tuple[Any, Session]:
    """The next open: a second process, the same log, and the drain that makes
    the detached listener observable.

    The `fs` root is pointed at the repository because that is what a person
    resuming in their project has, and the three facts here — the seed, the
    header, the drain — are the ones a reconciliation test must get right.
    """
    ctx = await mount(*WORKTREE_ROWS, {"id": "fs", "config": {"root": str(base)}})
    revived = ctx.require(SESSIONS).adopt(
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


@pytest.mark.needs_git
async def test_a_crash_between_acquire_and_dispose_is_reconciled_on_the_next_open(
    mount: MountProfile, tmp_path: Path
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


@pytest.mark.needs_git
async def test_a_leak_whose_tree_is_already_gone_still_closes_its_pair(
    mount: MountProfile, tmp_path: Path
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


@pytest.mark.needs_git
async def test_a_dirty_leak_reaches_the_branch_rather_than_being_discarded(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A crash is not a reason to throw away work. Reconciliation runs the same
    disposal policy an orderly release runs, so what the agent had written when the
    process died is committed to its branch — reconciliation that discarded more
    than a normal exit would make crashing *worse* than the leak it is fixing.

    **This is the half of the design that makes crashing cheap.** The checkout is a
    resource the agent borrowed and reconciliation takes it back; the branch is the
    artifact and reconciliation is what puts the last of the work on it. A tree left
    on disk instead would be an orphan holding the only copy of that work — which is
    the arrangement where a *second* crash, or a `rm -rf` of a temp directory,
    costs the day the first crash did not.
    """
    _ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    (workspace.root / "unfinished.txt").write_text("half a thought\n", encoding="utf-8")

    reopened, revived = await _reopen(mount, tmp_path / "repo", session)

    assert not workspace.root.exists(), "reconciliation left the checkout behind"
    code, out, _ = await git(reopened, tmp_path / "repo", "show", f"{workspace.ref}:unfinished.txt")
    assert code == 0 and out == "half a thought\n", "reconciliation discarded the work"
    closing = [event for event in revived.events if event.type == "workspace/disposed"]
    assert closing[-1].data["kept"] is True


@pytest.mark.needs_git
async def test_forking_does_not_reclaim_the_parents_live_worktree(
    mount: MountProfile, tmp_path: Path
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

    child = ctx.require(SESSIONS).fork(session)
    await ctx.drain()

    assert workspace_leaks(child) == [], "the fork folded its parent's live worktree as a leak"
    assert workspace.root.is_dir(), "forking reclaimed the parent's live worktree"


@pytest.mark.needs_git
async def test_a_held_workspace_is_never_reconciled(mount: MountProfile, tmp_path: Path) -> None:
    """Belt and braces, on the seam's own knowledge. `live()` exists so
    `/workspaces` can ask "is this tree anybody's" before offering to delete a
    directory; a reconciler with a second, weaker answer to that question is how
    the two come to disagree about a tree someone is working in."""
    ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    assert workspace_leaks(session) != []

    await ctx.require(WORKSPACE).reconcile(session)

    assert workspace.root.is_dir(), "the seam reclaimed a workspace it still holds"


async def test_a_leak_no_mounted_tier_can_reclaim_is_left_alone(
    mount: MountProfile, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Reported, not removed. The tree belongs to a tier this profile does not
    have, and deleting a directory on the strength of a record written by a
    configuration we are not running is the one way this row could destroy the
    work it exists to protect."""
    tree = tmp_path / "trees" / "a"
    tree.mkdir(parents=True)
    ctx = await mount()
    session = Session("crashed")
    log_event(
        session,
        "workspace/acquired",
        {"agentId": "a", "kind": "worktree", "root": str(tree), "ref": "ph/s/a"},
    )

    with caplog.at_level("WARNING"):
        ctx.require(SESSIONS).adopt(session)
        await ctx.drain()

    assert tree.exists(), "a tree no mounted tier owns was removed anyway"
    assert "no mounted tier can reclaim" in caplog.text


# -------------------------------------------- reconciling while an agent arrives --


@dataclass(slots=True)
class _GatedReclaim:
    """The mounted tier, with a reclaim this test can hold open.

    A wrapper rather than a fake provider: what J6 is about is the window
    *inside* a real teardown — `git worktree remove --force` takes a few
    milliseconds and an `acquire` that lands in them gets a directory being
    deleted — so the teardown has to be the real one. `__getattr__` forwards
    everything else, which is what keeps this from being a second implementation
    of the tier drifting from the first.
    """

    inner: Any
    entered: anyio.Event = field(default_factory=anyio.Event)
    may_finish: anyio.Event = field(default_factory=anyio.Event)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self.inner, name)

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        self.entered.set()
        await self.may_finish.wait()
        return bool(await self.inner.reclaim(record))


@pytest.mark.needs_git
async def test_an_acquire_waits_for_the_reclaim_that_is_deleting_its_tree(
    mount: MountProfile, tmp_path: Path
) -> None:
    """J6 — reconciliation is detached, and the first `acquire` after a resume
    asks for the same agent.

    `emit` schedules an async listener and does not wait, so the sweep that
    tears down a crash's leaked trees runs *beside* the lifecycle that is
    bringing the session back. The provider derives a root from the session and
    agent id, so the tree the reclaim is deleting is the tree the acquire is
    about to reuse: the agent came up in a directory that vanished under it, or
    `worktree remove` failed halfway and left a registration pointing at a
    checkout the agent was already writing.

    Asserted as a wait rather than as a race: the reclaim is held open, and an
    `acquire` for the same agent must still be unfinished. Without the guard it
    returns immediately, holding a root the teardown is mid-way through.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    seam = ctx.require(WORKSPACE)
    gated = _GatedReclaim(inner=seam.provider)
    seam.provider = gated
    # The crash: the tree is leaked, and this process no longer holds it.
    leak = workspace_leaks(session)
    assert [one.agent_id for one in leak] == [agent.id]
    await seam.dispose(agent.id)
    log_event(
        session,
        "workspace/acquired",
        {"agentId": agent.id, "kind": "worktree", "root": str(workspace.root), "ref": "ph/s1/a"},
    )

    taken: list[Path] = []

    async def take() -> None:
        again = await seam.acquire(
            session_id="s1", agent_id=agent.id, base=tmp_path / "repo", access="write"
        )
        taken.append(again.root)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(seam.reconcile, session)
        with anyio.fail_after(5):
            await gated.entered.wait()
        tasks.start_soon(take)
        await anyio.sleep(0.05)

        assert taken == [], "an acquire took a tree the reclaim was still deleting"
        gated.may_finish.set()

    assert taken and taken[0].is_dir(), "and once the reclaim is done the agent gets its tree"
