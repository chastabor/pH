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
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.seams.workspace import (
    ContainmentTier,
    SharedWorkspaceProvider,
    Workspace,
    WorkspaceKind,
    WorkspaceSeam,
)
from ph.session import Session

pytestmark = pytest.mark.anyio


def _seam(tmp_path: Path) -> WorkspaceSeam:
    return WorkspaceSeam(
        ctx=Context(), shared=SharedWorkspaceProvider(), scratch_root=tmp_path / "scratch"
    )


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
    seam = _seam(tmp_path)
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
    seam = _seam(tmp_path)
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
    seam = _seam(tmp_path)
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
    seam = _seam(tmp_path)
    seam.register_provider(_Provider(answer=None))
    (tmp_path / "repo").mkdir()

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path / "repo")

    assert workspace.kind == "shared"


async def test_a_provider_that_raises_falls_back_too(tmp_path: Path) -> None:
    """A tier that broke is a tier that is not in force, and the difference
    between declining and crashing is the provider's problem, not the agent's."""
    seam = _seam(tmp_path)
    seam.register_provider(_Provider(raises=True))
    (tmp_path / "repo").mkdir()

    workspace = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path / "repo")

    assert workspace.kind == "shared"


async def test_the_provider_gets_the_whole_request(tmp_path: Path) -> None:
    """`access` reaches the tier, because it is the tier that decides what kind
    a read request resolves to."""
    seam = _seam(tmp_path)
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
    seam = _seam(tmp_path)
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
    seam = _seam(tmp_path)
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

    seam = _seam(tmp_path)
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

    seam = _seam(tmp_path)
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

    seam = _seam(tmp_path)
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

    seam = _seam(tmp_path)
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
    seam = _seam(tmp_path)
    first_scope = seam.ctx.scope("first")

    await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path, scope=first_scope)
    second = await seam.acquire(session_id="s1", agent_id="a1", base=tmp_path)
    await first_scope.dispose()

    assert seam.of("a1") is second


async def test_a_ref_rides_both_halves_when_the_kind_has_one(tmp_path: Path) -> None:
    """So a reader can say which branch a turn ran against without inspecting
    the repository — which is the point of the events being durable."""
    seam = _seam(tmp_path)
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

    seam = _seam(tmp_path)
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
