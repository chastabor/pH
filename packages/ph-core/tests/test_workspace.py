"""P4-07 — `ctx.workspace`: where an agent's writes land, stated honestly (D21, E5).

Two things are worth pinning here and the rest follows from them.

**`repo_writable` is a guarantee, not a request.** The whole seam exists because
"read-only" is an enforcement claim and most tiers cannot make one: a `worktree`
child may write freely, and only nobody reads it. A caller that inferred safety
from `access="read"` would be reasoning about a boundary that is not there, so
the shared provider answers a read request with `repo_writable=True` and says
`shared` — the honest answer, and the one §4.8's table gives.

**`acquire` never fails.** An agent needs a working directory; "no workspace" is
not a state the lifecycle can be in. A provider that cannot serve a request
declines and the seam falls back, because turning "this directory is not a git
repository" into "pH will not start" is the wrong trade for a containment tier
nobody has to use.

**A workspace is an effect of the scope that took it (I2).** Disposing the scope
releases it and writes the closing event, so the leak the durable pair detects at
session open is a *crash*, not an ordinary error path.

**Why this is a seam and not a tool.** A permission row can deny `edit`, because
a deny-list needs a registered name to match — it cannot deny `open(path, "w")`.
The workspace is what `ctx.fs`'s root and `ctx.subprocess`'s cwd resolve to, so a
tier bounds *authored* code rather than merely observing the calls a model makes
by name. The other half of that argument is measured in
`test_containment_ladder.py`: an absolute-path `open()` never consults a cwd, so
the `worktree` rung bounds nothing about `/etc/passwd` and only `sandbox` can
refuse it at the kernel.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.seams.workspace import (
    ContainmentTier,
    Workspace,
    WorkspaceKind,
    discards_writes,
    fresh_root,
    project_access,
    restorable,
    workspace_of,
    workspace_policy,
    writable_roots,
)
from ph.session import Session
from ph.testing import workspace_seam

pytestmark = pytest.mark.anyio


class _Provider:
    """A tier that answers, or declines, or breaks — whichever the test needs."""

    tier: ContainmentTier = "worktree"

    def __init__(self, answer: Workspace | None = None, raises: bool = False) -> None:
        self.answer = answer
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    async def acquire(self, **kwargs: Any) -> Workspace | None:
        self.calls.append(kwargs)
        if self.raises:
            raise RuntimeError("git is not installed")
        return self.answer


def _worktree(
    tmp_path: Path,
    *,
    kind: WorkspaceKind = "worktree",
    ref: str | None = None,
    release: Callable[[Workspace], Awaitable[bool]] | None = None,
) -> Workspace:
    """What a tier hands back: a value, with its own teardown and nothing else.

    `scratch` is the seam's, so a provider under test takes the path it was given
    rather than inventing one — which is the property that keeps the layout from
    drifting per tier.
    """
    return Workspace(
        root=tmp_path,
        scratch=tmp_path / "scratch",
        kind=kind,
        repo_writable=True,
        ref=ref,
        release=release,
    )


# --------------------------------------------------------------- the shared --


async def test_the_shared_provider_hands_back_the_directory_it_was_given(
    tmp_path: Path,
) -> None:
    """Today's behaviour, at zero cost — which is what lets the seam live in
    `ph-base` without changing what any profile does."""
    seam = workspace_seam(tmp_path / "scratch")
    base = tmp_path / "repo"
    base.mkdir()

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=base)

    assert workspace.root == base
    assert workspace.kind == "shared"
    assert workspace.repo_writable is True
    assert workspace.ref is None
    assert workspace.env == {}


async def test_scratch_exists_and_lives_outside_the_workspace(tmp_path: Path) -> None:
    """E5. It is created, not merely named — a research child told it has
    somewhere to write should not have to `mkdir` it first.

    Outside `root` on purpose: it survives disposal as a session artifact, so a
    child whose worktree is thrown away still leaves behind what it produced.
    """
    seam = workspace_seam(tmp_path / "scratch")
    base = tmp_path / "repo"
    base.mkdir()

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=base)

    assert workspace.scratch.is_dir()
    assert base not in workspace.scratch.parents
    # Per session *and* per agent, because two children of one session writing
    # notes into one directory is the collision the seam exists to avoid.
    other = await seam.acquire(session_id="s1", agent_id="a2", base=base)
    assert other.scratch != workspace.scratch


async def test_a_read_request_is_answered_honestly_rather_than_flattered(
    tmp_path: Path,
) -> None:
    """§4.8's table, at the `advisory` tier: `access="read"` yields `shared` with
    `repo_writable=True`.

    Nothing here enforces anything, so reporting `False` would be a lie a caller
    would then act on — the exact failure `repo_writable` exists to prevent.
    """
    seam = workspace_seam(tmp_path / "scratch")
    (tmp_path / "repo").mkdir()

    workspace = await seam.acquire(
        session_id="s1", agent_id="a1", base=tmp_path / "repo", access="read"
    )

    assert workspace.kind == "shared"
    assert workspace.repo_writable is True


# ------------------------------------------------------------- the fallback --


async def test_a_provider_that_declines_falls_back_rather_than_failing(
    tmp_path: Path,
) -> None:
    """`workspace-git-worktree` handed a directory that is not a repository.

    Declining is normal and must not be fatal: a containment tier is a
    deployment's choice, and "your cwd is not a git repo" turning into "pH will
    not start" is the wrong trade. The kind reported is `shared`, which is what
    tells an operator the tier is not in force.
    """
    seam = workspace_seam(tmp_path / "scratch")
    seam.register_provider(_Provider(answer=None))
    (tmp_path / "repo").mkdir()

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path / "repo")

    assert workspace.kind == "shared"


async def test_a_provider_that_raises_falls_back_too(tmp_path: Path) -> None:
    """A tier that broke is a tier that is not in force, and the difference
    between declining and crashing is the provider's problem, not the agent's."""
    seam = workspace_seam(tmp_path / "scratch")
    seam.register_provider(_Provider(raises=True))
    (tmp_path / "repo").mkdir()

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path / "repo")

    assert workspace.kind == "shared"


async def test_the_provider_gets_the_whole_request(tmp_path: Path) -> None:
    """`access` reaches the tier, because it is the tier that decides what kind
    a read request resolves to."""
    seam = workspace_seam(tmp_path / "scratch")
    provider = _Provider(answer=_worktree(tmp_path, kind="worktree-ephemeral"))
    seam.register_provider(provider)

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path, access="read")

    assert provider.calls[0]["access"] == "read"
    # And the scratch directory, already created: a tier is handed the layout
    # rather than deriving it, so `readonly-scratch` and `worktree` cannot drift
    # apart on where a child is told it may write.
    assert provider.calls[0]["scratch"].is_dir()
    assert workspace.kind == "worktree-ephemeral"
    # Isolated, not read-only: writes happen and reach nobody. A seam that
    # reported `False` here would be describing a guarantee no tier made.
    assert workspace.repo_writable is True


async def test_the_effective_tier_is_reported_not_the_configured_one(tmp_path: Path) -> None:
    """What `ph doctor` has to print. Configured-vs-effective is the whole
    distinction: a `worktree` row over a non-repository declines on every
    acquire, and a doctor reading config would report containment nobody has."""
    seam = workspace_seam(tmp_path / "scratch")
    assert seam.effective_tier(child=False) == "advisory"

    seam.register_provider(_Provider(answer=None))
    assert seam.effective_tier(child=False) == "worktree"


# ---------------------------------------------------------------- the pair --


async def test_acquire_and_dispose_bracket_each_other_in_the_log(tmp_path: Path) -> None:
    """The pair is what makes a leak detectable at session open (§4.9).

    Both halves are written by the seam rather than by the provider, because a
    pair only reconciles if one place owns both — a provider that forgot the
    second would leave every workspace looking leaked.
    """
    seam = workspace_seam(tmp_path / "scratch")
    session = Session("s1")
    (tmp_path / "repo").mkdir()

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path / "repo", session=session)
    acquired = [event for event in session.events if event.type == "workspace/acquired"]
    assert [event.data["kind"] for event in acquired] == ["shared"]
    assert acquired[0].data["repoWritable"] is True
    assert not [event for event in session.events if event.type == "workspace/disposed"]

    await seam.dispose("a1")
    (disposed,) = [event for event in session.events if event.type == "workspace/disposed"]
    assert disposed.data["agentId"] == "a1"
    assert disposed.data["kept"] is True


async def test_disposal_runs_the_providers_teardown_and_forgets_the_agent(
    tmp_path: Path,
) -> None:
    """`of()` is how the prompt line and `ph doctor` ask what an agent holds, so
    an entry outliving its agent would report a workspace that is gone."""
    released: list[str] = []

    async def release(_workspace: Workspace) -> bool:
        released.append("released")
        return True

    seam = workspace_seam(tmp_path / "scratch")
    seam.register_provider(_Provider(answer=_worktree(tmp_path, release=release)))

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path)
    assert seam.of("a1") is not None
    assert seam.of("a2") is None

    await seam.dispose("a1")
    assert released == ["released"]
    assert seam.of("a1") is None

    # Twice is a no-op: the disposer deregisters itself, so an explicit release
    # followed by the owning scope unwinding cannot tear the same worktree down
    # twice or write a second `disposed`.
    await seam.dispose("a1")
    assert released == ["released"]


async def test_disposing_the_owning_scope_releases_the_workspace(tmp_path: Path) -> None:
    """I2: a workspace is an effect of the scope that took it.

    Without this the durable pair would report a leak for every ordinary error
    path, when what it exists to catch is a crash that runs nothing (§4.9).
    """
    released: list[str] = []

    async def release(_workspace: Workspace) -> bool:
        released.append("released")
        return True

    seam = workspace_seam(tmp_path / "scratch")
    seam.register_provider(_Provider(answer=_worktree(tmp_path, release=release)))
    session = Session("s1")
    agent_scope = seam.ctx.scope("agent")

    await seam.acquire(
        session_id="s1", agent_id="a1", base=tmp_path, session=session, scope=agent_scope
    )
    await agent_scope.dispose()

    assert released == ["released"]
    assert seam.of("a1") is None
    assert [event.type for event in session.events if event.type.startswith("workspace/")] == [
        "workspace/acquired",
        "workspace/disposed",
    ]


async def test_a_discarded_workspace_says_so(tmp_path: Path) -> None:
    """`kept` is the teardown's answer, not a field set at acquire time.

    P4-08's policy is "keep dirty, remove clean, **discard ephemeral even if
    dirty**" — a decision made while releasing, so a value frozen at acquire
    could not carry it and `kind` cannot be asked instead. The record is what
    lets a reader tell "nothing changed, so it was removed" from "these writes
    were thrown away by design".
    """

    async def release(_workspace: Workspace) -> bool:
        return False

    seam = workspace_seam(tmp_path / "scratch")
    session = Session("s1")
    seam.register_provider(
        _Provider(answer=_worktree(tmp_path, kind="worktree-ephemeral", release=release))
    )

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path, session=session)
    await seam.dispose("a1")

    (disposed,) = [event for event in session.events if event.type == "workspace/disposed"]
    assert disposed.data["kept"] is False


async def test_a_broken_teardown_still_forgets_the_agent(tmp_path: Path) -> None:
    """Otherwise one failed `git worktree remove` leaves the seam claiming an
    agent holds a workspace for the life of the process, and every reader of
    `of()` — the prompt, the doctor — repeats it."""

    async def release(_workspace: Workspace) -> bool:
        raise RuntimeError("worktree is locked")

    seam = workspace_seam(tmp_path / "scratch")
    seam.register_provider(_Provider(answer=_worktree(tmp_path, release=release)))
    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path)

    with pytest.raises(RuntimeError):
        await seam.dispose("a1")
    assert seam.of("a1") is None


async def test_re_acquiring_does_not_let_the_old_handle_evict_the_new_one(
    tmp_path: Path,
) -> None:
    """The `_registry` rule, applied to the live map: a disposer removes what it
    put there, not whatever currently occupies the key.

    An unconditional removal would leave `of()` answering `None` for a live
    workspace, which the prompt line and `ph doctor` would then repeat.
    """
    seam = workspace_seam(tmp_path / "scratch")
    first_scope = seam.ctx.scope("first")

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path, scope=first_scope)
    second = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path)
    await first_scope.dispose()

    assert seam.of("a1") is second


async def test_a_ref_rides_both_halves_when_the_kind_has_one(tmp_path: Path) -> None:
    """So a reader can say which branch a turn ran against without inspecting
    the repository — which is the point of the events being durable."""
    seam = workspace_seam(tmp_path / "scratch")
    session = Session("s1")
    seam.register_provider(_Provider(answer=_worktree(tmp_path, ref="ph/s1/a1")))

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path, session=session)
    await seam.dispose("a1")

    refs = [
        event.data.get("ref")
        for event in session.events
        if event.type in {"workspace/acquired", "workspace/disposed"}
    ]
    assert refs == ["ph/s1/a1", "ph/s1/a1"]


async def test_acquiring_without_a_session_records_nothing_and_still_works(
    tmp_path: Path,
) -> None:
    """A workspace taken outside a session — a doctor probe, a test — is a real
    case, and it must not be the one that raises."""
    released: list[str] = []

    async def release(_workspace: Workspace) -> bool:
        released.append("released")
        return True

    seam = workspace_seam(tmp_path / "scratch")
    seam.register_provider(_Provider(answer=_worktree(tmp_path, release=release)))

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path)
    await seam.dispose("a1")

    assert released == ["released"]
    assert seam.of("a1") is None


async def test_mounting_the_seam_changes_nothing(mount: Any) -> None:
    """P4-07's gate. `ph-base` layers this row, so the claim has to hold for
    every profile: the seam is present, no tier is in force, and nothing has
    been acquired until the agent lifecycle does it (P4-08)."""
    ctx = await mount()

    assert ctx.workspace.effective_tier(child=False) == "advisory"
    assert ctx.workspace.of("any-agent") is None


# ------------------------------------------------------- the kind vocabulary --


def test_every_kind_is_classified_the_same_way_by_all_four_predicates() -> None:
    """**The whole `WorkspaceKind` vocabulary, in one table.**

    `mypy` already refuses a `match` that forgets a kind — that is why these four
    are exhaustive matches rather than the membership tests they started as. What
    it cannot catch is a kind added to the *wrong arm*, and each of these
    predicates is one a mistake in is expensive:

    * `project_access` is what a spawn records as `granted_access` and what
      `ph doctor` prints per agent, so a wrong answer misreports containment.
    * `fresh_root` gates provisioning. A `True` for `shared` would copy `.env`
      onto itself — destroying the file the provisioning exists to provide.
    * `discards_writes` is what the retention policy keys on, so a wrong `False`
      silently loses the evidence of a failed child.
    * `restorable` gates `/revert`, and it is the one that nearly went wrong:
      it was `kind not in ("worktree", "worktree-ephemeral")` written inline,
      correct but invisible to exhaustiveness, so a new kind fell outside it and
      `/revert` answered "no restore points in this session" — true, useless, and
      indistinguishable from a run that simply had not checkpointed yet.

    Before this, three of the four asserted only the two `overlay` kinds.
    """
    table: dict[WorkspaceKind, tuple[str, bool, bool, bool]] = {
        # kind                   access  fresh  discards  restorable
        "shared": ("write", False, False, False),
        "worktree": ("write", True, False, True),
        "worktree-ephemeral": ("read", True, True, True),
        "readonly-scratch": ("read", True, False, False),
        "overlay": ("write", True, False, False),
        "overlay-ephemeral": ("read", True, True, False),
    }

    assert set(table) == set(WorkspaceKind.__args__), "a kind was added without a row here"
    for kind, (access, fresh, discards, restore) in table.items():
        assert (project_access(kind), fresh_root(kind), discards_writes(kind)) == (
            access,
            fresh,
            discards,
        ), kind
        assert restorable(kind) is restore, kind


def test_only_the_ephemeral_kinds_lose_evidence_so_only_they_are_retained() -> None:
    """Why `discards_writes` is the predicate the retention policy keys on.

    The kinds that answer `True` discard because reaching nobody is their entire
    promise, which makes them the only ones whose evidence an *ordinary* release
    can lose — and therefore the only ones a policy has any business retaining
    without being asked one tree at a time. Every other kind either keeps a dirty
    tree for review or never had writes of its own.
    """
    kinds: tuple[WorkspaceKind, ...] = WorkspaceKind.__args__

    discarding = {kind for kind in kinds if discards_writes(kind)}
    assert discarding == {"worktree-ephemeral", "overlay-ephemeral"}
    assert all(project_access(kind) == "read" for kind in discarding), (
        "a kind that throws its writes away granted nothing of the project"
    )


def test_an_overlay_grants_write_because_its_delta_outlives_release() -> None:
    """`overlay` sits with `worktree`, not with `worktree-ephemeral`.

    It resembles the ephemeral checkout — writable, and the host sees nothing —
    but its delta survives release and can be exported onto a branch, so what the
    holder wrote *can* reach the project. Granting write is not a promise that
    anybody ran the export, exactly as it is not a promise anybody ran a merge.
    """
    assert project_access("overlay") == "write" == project_access("worktree")
    assert discards_writes("overlay") is False
    assert project_access("overlay-ephemeral") == "read"
    assert discards_writes("overlay-ephemeral") is True


# ------------------------------------------------------- the writable bounds --


def test_the_prompted_boundary_and_the_enforced_one_are_one_definition(
    tmp_path: Path,
) -> None:
    """**`workspace_policy` is derived from `writable_roots`, not a second copy.**

    Two consumers need this set and a third arrives with P6-05:
    `permissions-fs`'s default rule prompts about what falls outside it, and
    `ctx.shell` hands the same set to a backend to *enforce*. Two spellings that
    drifted would be a tier whose name promises what its policy does not do —
    which is the defect §4.8's table exists to prevent. A test asserting the two
    agree catches drift after the fact; deriving one from the other prevents it,
    and this is what pins the derivation.
    """
    workspace = _worktree(tmp_path)
    policy = workspace_policy(workspace)

    named = {policy.workspace_root, *policy.writable_extra}
    assert named == {str(path) for path in writable_roots(workspace)}


def test_scratch_is_writable_without_being_asked_about(tmp_path: Path) -> None:
    """`scratch` is in the set, always — and it is the reason the set exists.

    It sits outside the worktree by design (E5) and is the one place a read-only
    or ephemeral agent is *told* it may write. A boundary naming only `root` would
    prompt on exactly the writes the design invites, and would confine away the
    redirected `TMPDIR` that makes a read-only repo usable rather than merely
    safe.
    """
    workspace = _worktree(tmp_path, kind="readonly-scratch")

    assert workspace.scratch in writable_roots(workspace)
    assert workspace.root in writable_roots(workspace)


# ------------------------------------------------------------ asking the seam --


async def test_the_workspace_question_is_fail_soft_for_a_seam_nobody_mounted(
    tmp_path: Path,
) -> None:
    """**`workspace_of` answers `None` rather than raising.** Five callers ask it.

    The prompt line, `bash`, the kernel, the spawn path and the fs resolver all
    need "where does this agent write", and the copies they each had disagreed
    about two things: whether an agent with no `id` means `None` or a lookup with
    an empty key, and whether a raising seam is fatal. Both are settled here,
    fail-soft, because a profile that layers no workspace row is an ordinary
    deployment and a caller asking during teardown is an ordinary moment — not
    reasons for a prompt line to take the process down.
    """
    seam = workspace_seam(tmp_path / "scratch")
    ctx = Context()

    assert workspace_of(ctx, "agent") is None, "no seam mounted"
    assert workspace_of(ctx, None) is None, "no agent"

    ctx.provide("workspace", seam)
    assert workspace_of(ctx, "never-acquired") is None
    assert workspace_of(ctx, object()) is None, "an agent with no id is not a lookup key"

    acquired = await seam.acquire(session_id="s", agent_id="a", base=tmp_path)
    assert workspace_of(ctx, "a") is acquired
