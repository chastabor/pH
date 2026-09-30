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
from typing import Any, Literal

import anyio
import pytest

from ph.keys import SESSIONS, WORKSPACE
from ph.seams.workspace import (
    ContainmentTier,
    WorkspaceAccess,
    WorkspaceRecord,
    WorkspaceSite,
    workspace_leaks,
    workspace_survivors,
)
from ph.session import Session
from ph.testing import (
    MountProfile,
    log_event,
    prefix_of,
    stored_events,
)
from ph.testing import (
    workspace_acquired as _acquired,
)
from ph.testing import (
    workspace_acquiring as _acquiring,
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


def test_a_tree_recorded_before_it_was_made_is_a_leak() -> None:
    """S12. `acquiring` is written before the tier acts, so one with nothing after it
    is a crash inside `worktree add` or an overlay's mount — and the root it names is
    the one place that tree can be."""
    session = _log(_acquiring("a", "/trees/a", kind="overlay", ref=""))

    (leak,) = workspace_leaks(session)

    assert (leak.root, leak.kind) == (Path("/trees/a"), "overlay")


def test_the_finished_tree_answers_the_one_it_was_about_to_be() -> None:
    session = _log(_acquiring("a", "/trees/a"), _acquired("a", "/trees/a"), _disposed("a"))

    assert workspace_leaks(session) == []


def test_a_tier_that_declined_leaves_nothing_of_its_own_open() -> None:
    """The tier named a root and then declined — not a repository, a failed mount — so
    the seam fell back to `shared`. Nothing was made where the record said, and a leak
    reported there would send reconcile after a tree that never existed."""
    session = _log(_acquiring("a", "/trees/a"), _acquired("a", "/project", kind="shared"))

    assert workspace_leaks(session) == []


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
async def test_a_crash_inside_the_tiers_own_act_is_reconciled(
    mount: MountProfile, tmp_path: Path
) -> None:
    """S12, the window `acquired` could not close: the tree exists, and the process
    died before the tier handed it back. The log ends at `acquiring`, which is what
    the crash leaves — and the next open reclaims the tree it names."""
    _ctx, session, _agent, workspace = await worktree_agent(mount, tmp_path)
    cut = next(i for i, one in enumerate(session.events) if one.type == "workspace/acquiring")
    crashed = prefix_of(session, cut + 1)

    _reopened, revived = await _reopen(mount, tmp_path / "repo", crashed)

    assert not workspace.root.exists(), "a tree the log said was coming survived the open"
    assert workspace_leaks(revived) == []


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
async def test_a_reconciled_close_is_on_disk_with_the_reclaim_it_records(
    mount: MountProfile, tmp_path: Path
) -> None:
    """S12: the reclaim is the act, and its record now reaches disk with it.

    A second open after a crash here would reclaim again, find the tree gone, and
    close the pair a second time as `kept: false` — about a branch the first
    reclaim had just committed the work to.
    """
    _ctx, session, _agent, _workspace = await worktree_agent(mount, tmp_path)

    reopened, revived = await _reopen(mount, tmp_path / "repo", session)

    closing = [
        one.data for one in stored_events(reopened, revived.id) if one.type == "workspace/disposed"
    ]
    assert [one.get("reconciled") for one in closing] == [True]


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


@dataclass(slots=True)
class _HalfMade:
    """A tier that names its site, makes part of the tree there, and then raises —
    an overlay mounted, then its base not recorded."""

    root: Path
    tier: ContainmentTier = "worktree"
    reclaimed: list[WorkspaceRecord] = field(default_factory=list)

    def locate(self, *, session_id: str, agent_id: str, access: WorkspaceAccess) -> WorkspaceSite:
        return WorkspaceSite(root=self.root / agent_id, kind="worktree", ref=f"ph/{session_id}/a")

    async def acquire(self, **kwargs: Any) -> None:  # noqa: ANN401
        (self.root / str(kwargs["agent_id"])).mkdir(parents=True)
        raise RuntimeError("mounted, then fell over")

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        self.reclaimed.append(record)
        return False


async def test_a_tier_that_raised_part_way_is_taken_back_at_once(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A tier that raised falls back to `shared`, and the `shared` acquire supersedes
    the `acquiring` record — so nothing named the root any more, and whatever the
    tier had made there (a live mount) was left for nobody to find. The seam
    reclaims the recorded site then and there, as reconcile would after a crash.

    Sabotage: drop `_take_back` from the seam's `except Exception`, and nothing is
    reclaimed.
    """
    ctx = await mount()
    tier = _HalfMade(root=tmp_path / "trees")
    ctx.require(WORKSPACE).register_provider(tier)
    session = ctx.require(SESSIONS).create("s")

    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s", agent_id="a", base=tmp_path, session=session
    )

    assert workspace.kind == "shared"
    assert [(one.root, one.kind) for one in tier.reclaimed] == [
        (tmp_path / "trees" / "a", "worktree")
    ]
    assert workspace_survivors(session) == []


# -------------------------------------------- reconciling while an agent arrives --


@dataclass(slots=True)
class _Gated:
    """The mounted tier, with one of its acts — `acquire` or `reclaim` — held open
    until the test lets it finish, and every reclaim counted.

    A wrapper rather than a fake provider: what J6 is about is the window *inside*
    a real act — `git worktree remove --force` takes a few milliseconds, and an
    `acquire` that lands in them gets a directory being deleted — so the act has to
    be the real one. `__getattr__` forwards everything else, which is what keeps
    this from being a second implementation of the tier drifting from the first.
    """

    inner: Any
    held: Literal["acquire", "reclaim"]
    entered: anyio.Event = field(default_factory=anyio.Event)
    may_finish: anyio.Event = field(default_factory=anyio.Event)
    reclaimed: list[WorkspaceRecord] = field(default_factory=list)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self.inner, name)

    async def _hold(self, act: Literal["acquire", "reclaim"]) -> None:
        if act == self.held:
            self.entered.set()
            await self.may_finish.wait()

    async def acquire(self, **arguments: Any) -> Any:  # noqa: ANN401
        await self._hold("acquire")
        return await self.inner.acquire(**arguments)

    async def reclaim(self, record: WorkspaceRecord) -> bool:
        self.reclaimed.append(record)
        await self._hold("reclaim")
        return bool(await self.inner.reclaim(record))


@dataclass(slots=True)
class _Leaked:
    """An agent whose worktree a crash leaked, over a tier with one act gated, and
    an acquire of that agent's tree again, to run beside a reconcile."""

    session: Session
    seam: Any
    gated: _Gated
    agent_id: str
    base: Path
    taken: list[Path] = field(default_factory=list)

    async def take(self) -> None:
        again = await self.seam.acquire(
            session_id="s1", agent_id=self.agent_id, base=self.base, access="write"
        )
        self.taken.append(again.root)


async def _leaked(
    mount: MountProfile, tmp_path: Path, held: Literal["acquire", "reclaim"]
) -> _Leaked:
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    seam = ctx.require(WORKSPACE)
    gated = _Gated(inner=seam.provider, held=held)
    seam.provider = gated
    assert [one.agent_id for one in workspace_leaks(session)] == [agent.id]
    # The crash: the tree is leaked, and this process no longer holds it.
    await seam.dispose(agent.id)
    log_event(session, *_acquired(agent.id, str(workspace.root), ref="ph/s1/a"))
    return _Leaked(session, seam, gated, agent.id, tmp_path / "repo")


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
    leaked = await _leaked(mount, tmp_path, "reclaim")

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(leaked.seam.reconcile, leaked.session)
        with anyio.fail_after(5):
            await leaked.gated.entered.wait()
        tasks.start_soon(leaked.take)
        await anyio.sleep(0.05)

        assert leaked.taken == [], "an acquire took a tree the reclaim was still deleting"
        leaked.gated.may_finish.set()

    assert leaked.taken and leaked.taken[0].is_dir(), (
        "and once the reclaim is done the agent gets its tree"
    )


@pytest.mark.needs_git
async def test_a_reclaim_leaves_the_tree_an_acquire_is_handing_over(
    mount: MountProfile, tmp_path: Path
) -> None:
    """J6, the other direction. An `acquire` already past its wait awaits the scratch
    directory, its `acquiring` flush and the tier before it holds the tree, and a
    reconcile landing in that gap found no holder: it reclaimed the tree the acquire
    was building or reusing, and the agent came up in a directory being deleted.

    Asserted with the tier's acquire held open, so the gap is the window.

    Sabotage: take no claim in `acquire`, and the reconcile reclaims it.
    """
    leaked = await _leaked(mount, tmp_path, "acquire")

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(leaked.take)
        with anyio.fail_after(5):
            await leaked.gated.entered.wait()
        await leaked.seam.reconcile(leaked.session)

        assert leaked.gated.reclaimed == [], "the reconcile reclaimed a tree an acquire was taking"
        leaked.gated.may_finish.set()

    assert leaked.taken and leaked.taken[0].is_dir(), "and the agent gets its tree whole"


@pytest.mark.needs_git
async def test_a_reclaim_asks_again_for_an_acquire_begun_after_its_fold(
    mount: MountProfile, tmp_path: Path
) -> None:
    """J6, the last line. A reconcile folds its leaks and only then starts a reclaim
    per tree, so an acquire can take its claim in between: the fold never saw it,
    and the reclaim's own check, just before the removal, is what does.

    Started in that order, the reconcile folds before the acquire runs, and its
    reclaim runs after the acquire has taken its claim.

    Sabotage: drop the claim from `_reclaim`'s check, and it reclaims the tree.
    """
    leaked = await _leaked(mount, tmp_path, "acquire")

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(leaked.seam.reconcile, leaked.session)
        tasks.start_soon(leaked.take)
        with anyio.fail_after(5):
            await leaked.gated.entered.wait()
        await anyio.sleep(0.05)

        assert leaked.gated.reclaimed == [], "the reclaim took a tree an acquire had claimed"
        leaked.gated.may_finish.set()

    assert leaked.taken and leaked.taken[0].is_dir()
