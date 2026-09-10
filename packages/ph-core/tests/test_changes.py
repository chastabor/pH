"""`ph.seams.changes` — asking git or jj what changed, against real repositories.

Gates: *a clean file is vouched for without being read; a dirty one never is;
the workspace provider decides which backend answers.*

## Why real repositories and not a stub

The whole module is an assertion about what two external tools report, and every
interesting property is one a stub would encode rather than test. Two of them
were found this way and would have been invisible otherwise:

* `git ls-files -s` reports the **index**, so a modified-unstaged file still
  carries its old blob id — which is why `suspect` exists at all, and why a stub
  returning "the current hash" would have made the whole `suspect` mechanism
  look redundant.
* `jj` snapshots the working copy on *any* command, so `@` moves with no commit
  and one diff covers uncommitted work. A stub would have had to be told that,
  and would then have been testing the telling.

## Why the safe direction is asserted explicitly

`vouches_for` is conservative in one direction only: a false `False` costs a
read the caller was going to do anyway, while a false `True` leaves a stale
index that nothing later detects. So the tests below assert the *negative* cases
as hard as the positive ones — an unknown backend, an unreadable token, a
renamed path, a file dirty in the working tree.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.keys import WORKSPACE
from ph.seams.changes import (
    Backend,
    TreeState,
    VersionedProvider,
    backend_for,
    tree_state,
)
from ph.testing import MountProfile
from ph.testing.git import git, git_repo
from ph.testing.jj import jj_repo

pytestmark = pytest.mark.anyio


# ------------------------------------------------------------- the predicate ----


def test_no_backend_vouches_for_nothing() -> None:
    """The empty state a failed or absent backend returns.

    A consumer that ignored `TreeState` entirely would still be correct, just
    slower — which is the property that makes this safe in front of an indexer.
    """
    empty = TreeState()

    assert not empty.vouches_for("a.py", "somedigest")
    assert not empty.vouches_for("a.py", "")
    assert empty.id_for("a.py") == ""


def test_a_suspect_path_is_never_vouched_for() -> None:
    """Whatever else is known about it. Both backends put paths here."""
    state = TreeState(backend="git", ids={"a.py": "abc"}, suspect=frozenset({"a.py"}))

    assert not state.vouches_for("a.py", "abc"), "a dirty path must be re-read"


def test_git_compares_the_stored_id_against_the_index() -> None:
    state = TreeState(backend="git", ids={"a.py": "abc", "b.py": "def"})

    assert state.vouches_for("a.py", "abc")
    assert not state.vouches_for("a.py", "stale"), "a moved id must be re-read"
    assert not state.vouches_for("a.py", ""), "no stored id is no proof"
    assert not state.vouches_for("gone.py", "abc"), "a path git does not list"


def test_jj_rests_on_the_diff_rather_than_per_file_ids() -> None:
    """Not being in the changed set *is* the answer — see the module docstring."""
    state = TreeState(backend="jj", token="abc123", diffed=True, suspect=frozenset({"b.py"}))

    assert state.vouches_for("a.py", ""), "jj needs no stored id"
    assert not state.vouches_for("b.py", "")
    # A state that never diffed proves nothing, even holding a good token —
    # which is exactly the first-run case, and why `diffed` is its own field.
    assert not TreeState(backend="jj", token="abc123", diffed=False).vouches_for("a.py", "")


# ------------------------------------------------------------- which backend ----


class _Tier:
    """A workspace provider that declares a backend, as the two real tiers do."""

    def __init__(self, vcs: Backend) -> None:
        self.vcs = vcs


class _Untiered:
    """A provider with no `vcs` — `shared`, `readonly-scratch`, an overlay."""


class _Seam:
    """The shape `backend_for` reads: the seam, holding a claimed provider slot.

    Modelled rather than mounted because the real `WorkspaceSeam` needs a tier
    row and a repository to hand one out, and the question here is only which of
    two answers `backend_for` prefers.
    """

    def __init__(self, provider: Any = None) -> None:
        self.provider = provider


def test_a_provider_that_declares_a_backend_is_the_authority(tmp_path: Path) -> None:
    """**Not `Workspace.kind`**, which is `worktree` for git *and* jj.

    And not the directory: a jj-backed workspace checked out inside a git repo
    must answer jj, which is the case a filesystem probe alone gets wrong.
    """
    ctx = Context()
    ctx.provide("workspace", _Seam(_Tier("jj")))
    (tmp_path / ".git").mkdir()

    assert backend_for(ctx, tmp_path) == "jj", "the provider's word beat the marker"


def test_a_provider_declaring_nothing_falls_through_to_the_root(tmp_path: Path) -> None:
    """The `shared` tier — the default profile, and therefore the common case.

    A filter that stood down here would do nothing in the profile most people
    run, which is why the probe is not an afterthought.
    """
    ctx = Context()
    ctx.provide("workspace", _Seam(_Untiered()))
    (tmp_path / ".git").mkdir()

    assert backend_for(ctx, tmp_path) == "git"


def test_the_probe_walks_upward_and_prefers_jj(tmp_path: Path) -> None:
    """A colocated repository has both, and jj is the one that snapshots."""
    ctx = Context()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".jj").mkdir()
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)

    assert backend_for(ctx, nested) == "jj"


def test_no_repository_anywhere_is_no_backend(tmp_path: Path) -> None:
    ctx = Context()
    # `tmp_path` is under pytest's own root, which is not a repository — assert
    # that rather than assuming it, so a checkout under one cannot pass this
    # vacuously.
    assert not any((one / ".git").exists() for one in (tmp_path, *tmp_path.parents))

    assert backend_for(ctx, tmp_path) == ""


def test_the_protocol_matches_a_provider_that_declares_the_attribute() -> None:
    """`isinstance`, not `hasattr` — the rule `ReclaimingProvider` states."""
    assert isinstance(_Tier("git"), VersionedProvider)
    assert not isinstance(_Untiered(), VersionedProvider)


# --------------------------------------------------------------- against git ----


@pytest.mark.needs_git
async def test_git_vouches_for_a_committed_file(mount: MountProfile, tmp_path: Path) -> None:
    """The win: a clean file is proved unchanged without being opened."""
    ctx = await mount()
    root = await git_repo(ctx, tmp_path / "repo")
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add a")

    state = await tree_state(ctx, root)

    assert state.backend == "git"
    stored = state.id_for("a.py")
    assert stored, "git listed no id for a committed file"
    assert state.vouches_for("a.py", stored)


@pytest.mark.needs_git
async def test_git_refuses_to_vouch_for_a_modified_file(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The reason `suspect` exists.**

    `ls-files -s` reports the *index*, so a modified-unstaged file still carries
    its committed blob id — comparing ids alone would call it unchanged and the
    caller would never re-read it.
    """
    ctx = await mount()
    root = await git_repo(ctx, tmp_path / "repo")
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add a")
    first = await tree_state(ctx, root)
    stored = first.id_for("a.py")

    (root / "a.py").write_text("a = 999\n", encoding="utf-8")
    after = await tree_state(ctx, root)

    assert after.id_for("a.py") == stored, "the index still reports the old blob"
    assert "a.py" in after.suspect, "status is what catches it"
    assert not after.vouches_for("a.py", stored)


@pytest.mark.needs_git
async def test_git_refuses_to_vouch_for_an_untracked_file(
    mount: MountProfile, tmp_path: Path
) -> None:
    ctx = await mount()
    root = await git_repo(ctx, tmp_path / "repo")
    (root / "fresh.py").write_text("f = 1\n", encoding="utf-8")

    state = await tree_state(ctx, root)

    assert "fresh.py" in state.suspect
    assert not state.vouches_for("fresh.py", "")


@pytest.mark.needs_git
async def test_git_treats_both_ends_of_a_rename_as_suspect(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A caller may hold a record under either name, so neither is vouched for."""
    ctx = await mount()
    root = await git_repo(ctx, tmp_path / "repo")
    (root / "old.py").write_text("x = 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "add old")
    await git(ctx, root, "mv", "old.py", "new.py")

    state = await tree_state(ctx, root)

    assert "old.py" in state.suspect and "new.py" in state.suspect


@pytest.mark.needs_git
async def test_git_answers_in_the_asked_about_trees_spelling(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**A subdirectory is not the repository, and the two git commands disagree.**

    `ls-files` names paths from the directory it runs in; `status --porcelain`
    names them from the repository root. So a workspace below the repo root got
    `ids` and `suspect` in different vocabularies, and the same string named two
    different files — `README.md` meaning the package's to one command and the
    repository's to the other. `suspect` then marked a path `ids` never had, and
    a modified file kept its stale blob id and was vouched for.

    Every assertion here is about *spelling*, because that is what was wrong; the
    modified-file case above passes either way when the tree is the repo root.
    """
    ctx = await mount()
    root = await git_repo(ctx, tmp_path / "repo")
    inner = root / "pkg"
    inner.mkdir()
    (root / "README.md").write_text("the repository's\n", encoding="utf-8")
    (inner / "README.md").write_text("the package's\n", encoding="utf-8")
    (inner / "kept.py").write_text("k = 1\n", encoding="utf-8")
    await git(ctx, root, "add", "-A")
    await git(ctx, root, "commit", "-m", "both readmes")

    # Modify the *outer* one. Asked about `pkg`, nothing has changed.
    (root / "README.md").write_text("edited outside the tree\n", encoding="utf-8")
    state = await tree_state(ctx, inner)

    assert set(state.ids) == {"README.md", "kept.py"}, "keys are relative to `pkg`"
    assert not state.suspect, "the edit is outside this tree, so it names nothing here"
    assert state.vouches_for("README.md", state.id_for("README.md")), (
        "pkg/README.md is untouched — the outer edit must not make it suspect"
    )

    # Now modify the *inner* one, which must be caught in that same spelling.
    (inner / "README.md").write_text("edited inside\n", encoding="utf-8")
    after = await tree_state(ctx, inner)

    assert "README.md" in after.suspect
    assert not after.vouches_for("README.md", after.id_for("README.md"))
    assert after.vouches_for("kept.py", after.id_for("kept.py")), "its neighbour is still clean"


# ---------------------------------------------------------------- against jj ----


@pytest.mark.needs_jj
async def test_jj_reports_a_token_and_vouches_for_nothing_on_a_first_run(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Nothing to diff from, so nothing is proved — but the token is returned.

    The caller stores it and the *next* run is the cheap one.
    """
    ctx = await mount()
    root = await jj_repo(ctx, tmp_path / "repo")
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")

    state = await tree_state(ctx, root, since="")

    assert state.backend == "jj"
    assert state.token, "the token a first run exists to hand back"
    assert not state.diffed
    assert not state.vouches_for("a.py", ""), "a first run proves nothing"


@pytest.mark.needs_jj
async def test_jj_vouches_for_everything_it_did_not_diff(
    mount: MountProfile, tmp_path: Path
) -> None:
    """One call, and it covers **uncommitted** work — jj snapshots on any command."""
    ctx = await mount()
    root = await jj_repo(ctx, tmp_path / "repo")
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")
    (root / "b.py").write_text("b = 2\n", encoding="utf-8")
    first = await tree_state(ctx, root)
    token = first.token
    assert token, "jj gave no working-copy id"

    # Edited and never committed. jj still sees it.
    (root / "a.py").write_text("a = 999\n", encoding="utf-8")
    after = await tree_state(ctx, root, since=token)

    assert "a.py" in after.suspect, "the edited file must be re-read"
    assert after.vouches_for("b.py", ""), "the untouched file is proved unchanged"
    assert not after.vouches_for("a.py", "")
    assert after.token and after.token != token, "the snapshot moved"


@pytest.mark.needs_jj
async def test_jj_vouches_for_nothing_when_the_token_is_unusable(
    mount: MountProfile, tmp_path: Path
) -> None:
    """A token predating a rebuilt repository is not an error — just a re-read.

    The caller reads everything once and stores a token that works, which is
    strictly better than a traceback out of an optimisation.
    """
    ctx = await mount()
    root = await jj_repo(ctx, tmp_path / "repo")
    (root / "a.py").write_text("a = 1\n", encoding="utf-8")

    state = await tree_state(ctx, root, since="0" * 40)

    assert state.backend == "jj"
    assert state.token, "a usable token is still handed back"
    assert not state.vouches_for("a.py", "")


@pytest.mark.needs_jj
async def test_a_jj_tree_is_reported_as_jj_not_git(mount: MountProfile, tmp_path: Path) -> None:
    """`jj git init` leaves a `.git` too, so the order in `_probed` is load-bearing."""
    ctx = await mount()
    root = await jj_repo(ctx, tmp_path / "repo")

    assert backend_for(ctx, root) == "jj"


# ----------------------------------------------------- through an overlay ----


OVERLAY_ROW = {"insert": [{"id": "workspace-agentfs", "name": "workspace-agentfs"}]}
"""The overlay tier, mounted rather than hand-assembled — as `test_workspace_agentfs`
mounts it, and for that module's reason: the overlay root then comes from `$PH_HOME`,
which the `mount` fixture already points at `tmp_path`."""


@pytest.mark.needs_git
async def test_an_overlay_answers_git_from_inside_its_own_mount(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The third backend that should not exist.**

    An AgentFS overlay keeps a log of everything it has layered over the base, and
    it is unreadable while the overlay is mounted — the delta database takes one
    writer, so `agentfs diff` answers `Locking error` and a read-only `sqlite3`
    answers `database is locked`. An indexer runs *while* an agent is using the
    workspace, so that log is never available to it.

    What is available is the base's `.git`, served through the mountpoint like
    everything else. So the probe finds git, git runs inside the overlay, and the
    delta layer's writes arrive as the uncommitted changes they are — the shape
    `suspect` already existed for. This asserts that end to end, because the
    alternative is a plausible-looking `vcs = "agentfs"` on the provider that would
    override the probe and read nothing.
    """
    ctx = await mount(OVERLAY_ROW)
    if ctx.require(WORKSPACE).provider is None:
        pytest.skip("no working overlay on this host")
    base = await git_repo(ctx, tmp_path / "repo")
    (base / "a.py").write_text("a = 1\n", encoding="utf-8")
    (base / "b.py").write_text("b = 2\n", encoding="utf-8")
    await git(ctx, base, "add", "-A")
    await git(ctx, base, "commit", "-qm", "base")

    workspace = await ctx.require(WORKSPACE).acquire(
        session_id="s", agent_id="a1", base=base, access="write"
    )
    assert workspace is not None and workspace.kind == "overlay"
    try:
        # A write the agent makes: it lands in the delta layer, not in the tree.
        (workspace.root / "a.py").write_text("a = 999\n", encoding="utf-8")

        state = await tree_state(ctx, workspace.root)

        assert state.backend == "git", "the mountpoint serves the base's .git"
        assert "a.py" in state.suspect, "the overlay's own write must be re-read"
        vouched = state.id_for("b.py")
        assert vouched, "the base's blob ids are readable through the mount"
        assert state.vouches_for("b.py", vouched), "an untouched file is proved unchanged"
        assert (base / "a.py").read_text(encoding="utf-8") == "a = 1\n", (
            "reading through the overlay must not weaken what it isolates"
        )
    finally:
        if workspace.release is not None:
            await workspace.release(workspace)


# ------------------------------------------------------------- the safe path ----


async def test_a_directory_that_is_not_a_repository_answers_empty(
    mount: MountProfile, tmp_path: Path
) -> None:
    """No backend, no exception, and a state that proves nothing."""
    ctx = await mount()
    plain = tmp_path / "plain"
    plain.mkdir()

    state = await tree_state(ctx, plain)

    assert state.backend == ""
    assert not state.vouches_for("anything.py", "whatever")
