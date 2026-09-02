"""P6-21 — a copy-on-write overlay as a second workspace provider.

Gate: *with the backend absent the row declines and the tier is never claimed;
with it present, two agents writing the same path do not collide.*

The vocabulary half runs everywhere. The real half runs only where an overlay
actually isolates, which is decided by the same probe the row ships — not by
"is the binary installed", because `agentfs run --experimental-sandbox` exits 0
and writes straight through to the host.

## What was measured on a working host, and what each number decided

**Isolation, verified.** A write through the overlay is visible to the agent and
the host copy is unchanged; two agents over one base each read back their own
version of the same path.

**It does not confine the process, which is why `tier` is `worktree` and not a
rung above it.** A command whose cwd is *inside* the mount wrote to an absolute
path *outside* it and the write landed on the host for real.

**Why `mount` and not `exec` or `run`.** `run` adds user and mount namespaces —
the mode that would actually confine — and is unavailable on any host with
`kernel.apparmor_restrict_unprivileged_userns=1`, the Ubuntu 23.10+ default.
`exec` mounts and runs in one shot, which would suit `acquire` exactly, and fails
with `connection pool timeout: no connections available` on a brand-new agent with
nothing mounted — reproduced on tmpfs *and* on ZFS, so it is neither
backing-store sensitivity nor contention. `mount` works on both, returns in
**~60 ms** leaving a live FUSE mount, and unmounts with `fusermount3 -u`.

**Acquire cost against the tier it is a peer of.** An overlay acquire is flat at
**~90-190 ms** regardless of tree size, where `git worktree add` is linear:
**37 ms at 1 000 files, 300-425 ms at 11 000**. What the overlay pays for that is
read speed — FUSE reads measured **30x native**.

**The `agentfs run` trap, recorded because it cost a wrong conclusion.** `agentfs
run` prints its session banner *even when the sandbox fails to start*, so a banner
plus exit 0 is not evidence the command ran. The actual failure was `Failed to
make mounts private: Permission denied`, and the host file was unchanged because
**the command never ran** — not because it was contained. Never read an exit code
as proof of confinement (E13).

**The exact version this was measured against.** With `agentfs v0.6.4` installed,
`agentfs run --experimental-sandbox` exits 0, prints no error, and writes straight
through to the host — it sandboxes only its own mount. A probe checking the exit
code would have claimed the tier while the agent had none.

**The symlink that was silently dropped.** The diff-row pattern first matched
`[fd]`, and AgentFS types a symlink `l` — so every symlink an agent created fell
out of the changeset, silently, because a row that does not match is a row that is
not there. `_apply` now refuses a type letter it does not understand.

**Why `_record_base` is gated on a `stat`.** On a base that is not a repository the
`git` spawn fails just as slowly as it succeeds (**~2 ms either way**), so the
`stat` buys a process per agent on the one branch where the spawn can return
nothing.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from ph.seams.containment import TIERS
from ph.seams.workspace import (
    Workspace,
    WorkspaceKind,
    WorkspaceRecord,
    discards_writes,
    fresh_root,
    project_access,
    restorable,
)
from ph.seams.workspace_agentfs import (
    ExportRefused,
    export_overlay,
    fs_id,
    is_mount,
    probe_overlay,
    store_for,
)
from ph.seams.workspace_git import tree_hash
from ph.testing import FAKE_OPTIONS, StubWorkspaceProvider, git, git_repo, needs_git

pytestmark = pytest.mark.anyio

SESSION = "x"
"""The session every export test works under, named once."""

ROW = {"insert": [{"id": "workspace-agentfs", "name": "workspace-agentfs"}]}
"""The row under test, mounted rather than hand-assembled — so a typo in the
entry point fails here rather than in someone's profile, and the overlay root
comes from `$PH_HOME`, which the `mount` fixture already points at `tmp_path`."""


# ------------------------------------------------------------- vocabulary --


def test_the_overlay_kinds_split_exactly_as_the_worktree_kinds_do() -> None:
    """E3. A writer's delta survives release and can be exported onto a branch,
    so what it wrote can reach the project; a reader's is thrown away. That is
    `worktree` against `worktree-ephemeral`, one mechanism over."""
    assert project_access("overlay") == "write"
    assert project_access("overlay-ephemeral") == "read"


def test_only_the_ephemeral_overlay_discards_its_writes() -> None:
    """P6-28. Keeping the delta is what makes a deliberate export possible at
    all — a kind that threw it away at release would have nothing to export."""
    assert discards_writes("overlay") is False
    assert discards_writes("overlay-ephemeral") is True


def test_an_overlay_is_a_fresh_root_so_materials_are_provisioned() -> None:
    """Its root is a mountpoint, not the base, so `.env` and friends have to be
    put there — the same reason every kind but `shared` provisions."""
    assert fresh_root("overlay") is True
    assert fresh_root("overlay-ephemeral") is True


def test_an_overlay_has_no_restore_mechanism() -> None:
    """**P6-20's gate, as a predicate `/revert` can refuse on.**

    This was a membership test inside `workspace_git`, invisible to the
    exhaustiveness every other question about a kind gets — so a kind added to
    the Literal fell outside it silently and `/revert` answered "no restore
    points in this session", which reads as "not yet" for a workspace that can
    never have one.
    """
    assert restorable("worktree") and restorable("worktree-ephemeral")
    assert not restorable("overlay") and not restorable("overlay-ephemeral")


async def test_revert_refuses_an_overlay_and_still_lists_for_a_worktree(
    mount: Any, tmp_path: Path
) -> None:
    """**The other half of P6-20's gate: the sentence a person actually reads.**

    The predicate above was asserted and the branch that consumes it was not, so
    the refusal could have regressed to the empty listing with `restorable`
    still answering `False` and nothing failing.

    The distinction is the whole point. "no restore points in this session" is
    *true* of an overlay, and it reads as **not yet** — so a person waits for one
    to appear when no mechanism will ever produce one. Naming the kind says the
    mechanism is absent rather than the points.

    Both kinds are driven through one command, because a branch that refused
    *everything* would satisfy the overlay half while breaking `/revert`
    outright, and that is exactly the reading a single-case test would pass.
    """
    revert_row = {"insert": [{"id": "workspace-revert", "name": "workspace-revert"}]}
    kinds: dict[str, tuple[WorkspaceKind, WorkspaceKind]] = {
        "ov": ("overlay", "overlay-ephemeral"),
        "wt": ("worktree", "worktree-ephemeral"),
    }
    shown: dict[str, str] = {}
    for name, pair in kinds.items():
        ctx = await mount(revert_row)
        ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / name, kinds=pair))
        session = ctx.sessions.create(f"s-{name}")
        agent = ctx.agents.create(session, FAKE_OPTIONS)
        workspace = await ctx.workspace.acquire(
            session_id=session.id, agent_id=agent.id, base=tmp_path, access="write", session=session
        )
        assert workspace.kind == pair[0]
        shown[name] = str(await ctx.commands.dispatch("/revert", session=session, agent=agent))

    assert shown["ov"].startswith("refusing: a overlay workspace has no restore mechanism"), shown[
        "ov"
    ]
    assert shown["wt"] == "no restore points in this session", (
        "a restorable kind with no checkpoints yet gets the listing, not the refusal"
    )


async def test_doctor_states_what_an_overlay_bounds_not_what_its_rung_sells(
    mount: Any, tmp_path: Path
) -> None:
    """**E1, in the one place a person looks to check it.**

    `TIERS` is keyed by rung, so every provider at `worktree` inherited "buys:
    collision isolation and revertibility (fan-out safety, per-run checkpoints,
    /revert)". An overlay has none of that second half — `write-tree` against a
    FUSE mountpoint has nothing to hash — so `ph doctor` was advertising a
    mechanism the mounted tier does not have. The columns belong to whoever
    occupies the rung, not to its name.
    """
    ctx = await mount(ROW)  # `containment` is in the base profile already
    if ctx.workspace.provider is None:
        pytest.skip("no working overlay on this host")

    rows = dict(ctx.containment.describe())

    assert "/revert" in TIERS["worktree"].buys, "the stock row is what this must not print"
    assert "no /revert" in rows["buys"], rows["buys"]
    assert "delta layer" in rows["does NOT bound"], rows["does NOT bound"]
    assert rows["bounds"] == TIERS["worktree"].bounds, "what it bounds really is the same"


async def test_git_checkpointing_declines_an_overlay(mount: Any, tmp_path: Path) -> None:
    """**The gate, asserted by behaviour rather than by reading the source.**

    `workspace_git` captures with `write-tree` against a `GIT_INDEX_FILE`, which
    is meaningless over a FUSE mountpoint — a checkpoint that silently captured
    nothing would make `/revert` restore an agent to a state it was never in.

    The first version of this counted a string in `workspace_git.py` through a
    cwd-relative path, which passes for a module that keeps the wording and
    changes the behaviour, fails for one that refactors the gate while behaving
    correctly, and errors outright when pytest runs from anywhere but the repo
    root. `tree_hash` answers the same question and is the thing that matters.
    """
    ctx = await mount()
    for kind in ("overlay", "overlay-ephemeral"):
        workspace = Workspace(
            root=tmp_path, scratch=tmp_path / "scratch", kind=kind, repo_writable=True
        )
        assert await tree_hash(ctx, workspace) is None, f"{kind} has no git tree to capture"


def test_an_agentfs_id_stays_inside_the_alphabet_agentfs_accepts() -> None:
    """`Invalid agent ID` is an error, not a warning, and session ids carry
    characters it refuses — so the mapping happens before the first acquire."""
    assert fs_id("20260901T101500-ab12", "agent-0") == "20260901T101500-ab12-agent-0"
    assert fs_id("s/1", "a:2") == "s_1-a_2"


async def test_the_seam_handles_an_overlay_without_agentfs_installed(
    mount: Any, tmp_path: Path
) -> None:
    """**The seam-contract half, which the row asked for and only half landed.**

    Everything downstream of `kind` — what the retention policy keeps, what
    `project_access` records as granted, which tier answers — is decided by the
    seam from the provider's *answer*, not by AgentFS. So it runs with the stub,
    on a host with no overlay at all. Without this the entire downstream half
    was exercised only where the binary happens to work, which is precisely the
    CI shape this row's own gate names.
    """
    ctx = await mount()
    ctx.workspace.register_provider(
        StubWorkspaceProvider(root=tmp_path / "ov", kinds=("overlay", "overlay-ephemeral"))
    )

    writer = await ctx.workspace.acquire(
        session_id="s", agent_id="w", base=tmp_path, access="write"
    )
    reader = await ctx.workspace.acquire(session_id="s", agent_id="r", base=tmp_path, access="read")

    assert writer is not None and reader is not None
    assert (writer.kind, reader.kind) == ("overlay", "overlay-ephemeral")
    assert discards_writes(reader.kind) and not discards_writes(writer.kind)
    assert project_access(writer.kind) == "write"


# -------------------------------------------------------- provider contract --


async def test_a_writer_keeps_its_delta_and_a_reader_does_not(mount: Any, tmp_path: Path) -> None:
    """The two kinds, from the one call that decides between them."""
    ctx = await _overlaid(mount, tmp_path)
    base = tmp_path / "tree"
    base.mkdir()
    (base / "f.txt").write_text("host", encoding="utf-8")

    writer = await ctx.workspace.acquire(session_id="s3", agent_id="w", base=base, access="write")
    reader = await ctx.workspace.acquire(session_id="s3", agent_id="r", base=base, access="read")
    assert writer is not None and reader is not None
    assert writer.kind == "overlay"
    assert reader.kind == "overlay-ephemeral"


async def test_the_row_declines_rather_than_claiming_a_tier_it_cannot_deliver(
    mount: Any, tmp_path: Path
) -> None:
    """**The structural half of this row, and it is easy to get backwards.**

    `register_provider` is `claim_slot`: exclusive. A row that took the slot and
    then declined every acquire would not fall back to the worktree tier — it
    would fall back to `shared`, i.e. advisory — so a deployment that asked for
    more isolation would silently get none. Registration is therefore what the
    probe gates, and a host where the overlay does not isolate must leave the
    slot empty.
    """
    ctx = await mount(ROW)
    isolates = (await probe_overlay(ctx, tmp_path / "probe")).isolates

    assert (ctx.workspace.provider is not None) is isolates, (
        "the slot is claimed exactly when the overlay was proved to isolate"
    )
    if not isolates:
        assert ctx.workspace.effective_tier(child=True) == "advisory"


async def test_the_probe_says_why_when_it_declines(mount: Any, tmp_path: Path) -> None:
    """ "Why am I on worktrees" is asked of the tool, not of the source."""
    ctx = await mount(ROW)
    result = await probe_overlay(ctx, tmp_path / "probe")

    assert result.because, "a decline with no reason is indistinguishable from no row"
    if shutil.which("agentfs") is None:
        assert result.isolates is False
        assert "not installed" in result.because


# ------------------------------------------------- the real thing, or skipped --


async def _overlaid(mount: Any, tmp_path: Path) -> Any:
    """A mounted row over a host where the overlay actually isolates, or a skip."""
    ctx = await mount(ROW)
    if ctx.workspace.provider is None:
        # `apply` already probed and kept the answer; re-running it here paid a
        # second init+mount per skipped test purely to write the message.
        pytest.skip("no working overlay on this host")
    return ctx


async def test_two_agents_get_isolated_views_of_one_tree(mount: Any, tmp_path: Path) -> None:
    """**The row's own acceptance criterion**: two agents writing the same path
    do not collide, and the host tree keeps its own copy.

    Verified by hand before the row was written — A reads `A-version`, B reads
    `B-version`, the host keeps its original — and pinned here so a change to the
    mount lifecycle cannot quietly turn the overlay into a pass-through.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = tmp_path / "tree"
    base.mkdir()
    (base / "shared.txt").write_text("host-original", encoding="utf-8")

    first = await ctx.workspace.acquire(session_id="s1", agent_id="a1", base=base, access="read")
    second = await ctx.workspace.acquire(session_id="s1", agent_id="a2", base=base, access="read")
    assert first is not None and second is not None
    assert first.kind == "overlay-ephemeral" and second.kind == "overlay-ephemeral"

    (first.root / "shared.txt").write_text("A-version", encoding="utf-8")
    (second.root / "shared.txt").write_text("B-version", encoding="utf-8")

    assert (first.root / "shared.txt").read_text(encoding="utf-8") == "A-version"
    assert (second.root / "shared.txt").read_text(encoding="utf-8") == "B-version"
    assert (base / "shared.txt").read_text(encoding="utf-8") == "host-original", (
        "the host tree is the read-only base; neither agent's write reaches it"
    )


async def test_an_overlay_shows_files_a_checkout_would_not(mount: Any, tmp_path: Path) -> None:
    """**The functional case for this row, as against the performance one.**

    `git worktree add` hands an agent tracked files at a commit: no build output,
    no `.env`, no `node_modules`. An overlay presents the tree as it actually is,
    which is the difference that survives whatever P6-22 measures.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = tmp_path / "tree"
    base.mkdir()
    (base / "ignored-artifact").write_text("built", encoding="utf-8")

    workspace = await ctx.workspace.acquire(
        session_id="s2", agent_id="a1", base=base, access="read"
    )
    assert workspace is not None
    assert (workspace.root / "ignored-artifact").read_text(encoding="utf-8") == "built"


# --------------------------------------------------------------- the export --


async def _worked(ctx: Any, base: Path, agent_id: str, edit: str) -> None:
    """One writer's overlay, edited and released — the state an export starts from."""
    workspace = await ctx.workspace.acquire(
        session_id=SESSION, agent_id=agent_id, base=base, access="write"
    )
    assert workspace is not None and workspace.kind == "overlay"
    (workspace.root / "shared.py").write_text(edit, encoding="utf-8")
    (workspace.root / f"{agent_id}-new.py").write_text("added\n", encoding="utf-8")
    assert workspace.release is not None
    await workspace.release(workspace)


async def _export(ctx: Any, agent_id: str) -> str:
    """Export one agent's overlay onto its branch, deriving what the row derives.

    The session name, the store layout, the AgentFS id and the branch name were
    each re-spelled at every call site; all four come from one place now, so a
    change to any of them moves the tests with it. The repository is not among
    them — it was written beside the delta at acquire, which is what lets a
    `WorkspaceRecord` reach the export without carrying the tree.
    """
    return await export_overlay(
        ctx,
        store=store_for(ctx.workspace.provider.root, SESSION, agent_id),
        identifier=fs_id(SESSION, agent_id),
        ref=f"ph/{SESSION}/{agent_id}",
    )


@needs_git
async def test_an_export_lands_on_a_branch_rooted_at_what_the_agent_saw(
    mount: Any, tmp_path: Path
) -> None:
    """**The reason this is a branch and not a copy.**

    Rooted at the commit the agent *saw*, a merge is a true three-way one: work
    that touched a different file than the branch moved on to merges clean, and
    nothing of the meantime is lost. Rooted at whatever `HEAD` happened to be,
    the same export would present the agent's tree as though it had been written
    against a state it never had.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = await git_repo(ctx, tmp_path / "repo")
    (base / "shared.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    (base / "other.py").write_text("untouched\n", encoding="utf-8")
    await git(ctx, base, "add", "-A")
    await git(ctx, base, "commit", "-qm", "base")

    await _worked(ctx, base, "a1", "def a():\n    return 999\n")

    # The project moves on meanwhile, touching a different file.
    (base / "other.py").write_text("changed on main\n", encoding="utf-8")
    await git(ctx, base, "commit", "-qam", "main moved")

    ref = await _export(ctx, "a1")
    code, _, _ = await git(ctx, base, "merge", "--no-edit", ref)

    assert code == 0, "independent edits must merge without a conflict"
    assert (base / "shared.py").read_text(encoding="utf-8") == "def a():\n    return 999\n"
    assert (base / "other.py").read_text(encoding="utf-8") == "changed on main\n"
    assert (base / "a1-new.py").is_file(), "an added file reaches the project too"


@needs_git
async def test_git_finds_the_conflict_so_the_export_does_not_have_to(
    mount: Any, tmp_path: Path
) -> None:
    """**Why the branch, rather than a hand-rolled refusal.**

    The first design compared the target against a recorded state and refused if
    it had moved — a worse `git merge` written by hand, which cannot tell an edit
    to a different function from an edit to the same line. It would refuse work
    that merges cleanly and accept work that silently loses somebody's. Handing
    the question to a three-way merge answers both cases correctly, and the
    operator gets conflict markers rather than a diagnosis.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = await git_repo(ctx, tmp_path / "repo")
    (base / "shared.py").write_text("def a():\n    return 1\n", encoding="utf-8")
    await git(ctx, base, "add", "-A")
    await git(ctx, base, "commit", "-qm", "base")

    await _worked(ctx, base, "a2", "def a():\n    return 999\n")

    (base / "shared.py").write_text("def a():\n    return 42\n", encoding="utf-8")
    await git(ctx, base, "commit", "-qam", "the same line, on main")

    ref = await _export(ctx, "a2")
    code, _, _ = await git(ctx, base, "merge", "--no-edit", ref)

    assert code != 0, "a real conflict must not merge silently"
    assert "<<<<<<<" in (base / "shared.py").read_text(encoding="utf-8")


@needs_git
async def test_an_export_carries_symlinks_rather_than_flattening_them(
    mount: Any, tmp_path: Path
) -> None:
    """**Two defects met here, and the silent one was worse.**

    `agentfs diff` types a symlink `l`, and the changeset regex matched `[fd]`
    only — so every link an agent made was dropped from the export with no error,
    because a row that does not match is a row that is not there. And the copy
    used `shutil.copy2`'s default, which writes a *copy of the target* for a live
    link and raises outright for a dangling one.

    A dangling link is in the test on purpose: it is the case that turns a
    flattening bug into a crash, and the one a repository legitimately contains
    (a link into a build directory nobody has built yet).
    """
    ctx = await _overlaid(mount, tmp_path)
    base = await git_repo(ctx, tmp_path / "repo")

    workspace = await ctx.workspace.acquire(
        session_id=SESSION, agent_id="a5", base=base, access="write"
    )
    assert workspace is not None
    (workspace.root / "target.txt").write_text("real", encoding="utf-8")
    (workspace.root / "link.txt").symlink_to("target.txt")
    (workspace.root / "dangling.txt").symlink_to("nowhere.txt")
    assert workspace.release is not None
    await workspace.release(workspace)

    await git(ctx, base, "merge", "--no-edit", await _export(ctx, "a5"))

    assert (base / "link.txt").is_symlink(), "a link must arrive as a link"
    assert (base / "dangling.txt").is_symlink(), "and a dangling one must arrive at all"
    _, out, _ = await git(ctx, base, "ls-files", "-s", "link.txt")
    assert out.startswith("120000"), "git records it as a symlink, not a regular file"


@needs_git
async def test_an_export_refuses_rather_than_reusing_a_branch(mount: Any, tmp_path: Path) -> None:
    """A second export onto a taken name would discard whatever is on it."""
    ctx = await _overlaid(mount, tmp_path)
    base = await git_repo(ctx, tmp_path / "repo")
    await _worked(ctx, base, "a3", "one\n")

    await _export(ctx, "a3")
    with pytest.raises(ExportRefused) as caught:
        await _export(ctx, "a3")
    assert caught.value.reason == "branch-exists"


async def test_an_export_refuses_when_there_was_no_commit_to_root_at(
    mount: Any, tmp_path: Path
) -> None:
    """An overlay over a plain directory has no base commit, so it has no branch.

    The refusal is named, because "your work is in a database and there is no way
    out" is the one outcome an operator must not have to guess at.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = tmp_path / "plain"
    base.mkdir()
    await _worked(ctx, base, "a4", "one\n")

    with pytest.raises(ExportRefused) as caught:
        await _export(ctx, "a4")
    assert caught.value.reason == "no-base-commit"


def _record(ctx: Any, agent_id: str, kind: str = "overlay-ephemeral") -> WorkspaceRecord:
    """The durable pair as reconciliation would find it, with no live object."""
    store = store_for(ctx.workspace.provider.root, SESSION, agent_id)
    return WorkspaceRecord(session_id=SESSION, agent_id=agent_id, kind=kind, root=store / "mnt")


async def test_a_crashed_agents_overlay_is_reclaimed(mount: Any, tmp_path: Path) -> None:
    """**F6, and the leak here is worse than a leftover directory.**

    A crash skips `release` entirely, and a live FUSE mount outlives the process
    that made it — so what is left behind is a mount table entry and a server,
    not a folder. Before this the seam found the pair, asked
    `isinstance(provider, ReclaimingProvider)`, got `False`, and logged "no
    mounted tier can reclaim" — leaving the pair open for every future session to
    report again and `ph workspaces gc` collecting nothing.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = tmp_path / "tree"
    base.mkdir()
    workspace = await ctx.workspace.acquire(
        session_id=SESSION, agent_id="c1", base=base, access="read"
    )
    assert workspace is not None and await is_mount(workspace.root)

    # No release runs: this is the state a killed process leaves behind.
    kept = await ctx.workspace.provider.reclaim(_record(ctx, "c1"))

    assert kept is False, "an ephemeral overlay is discarded exactly as its release would"
    assert not await is_mount(workspace.root), "and the mount is gone, not merely forgotten"


@needs_git
async def test_the_seam_exports_an_overlay_without_knowing_which_tier_it_is(
    mount: Any, tmp_path: Path
) -> None:
    """**One verb, both tiers**, which is the point of the Protocol.

    `/workspaces export` asks `ctx.workspace`, never the provider — a worktree
    answers with the branch it has been committing to all along and an overlay
    assembles one out of its delta first. A caller that had to know the
    difference is the `if provider is agentfs` branch this exists to prevent.
    """
    ctx = await _overlaid(mount, tmp_path)
    base = await git_repo(ctx, tmp_path / "repo")
    await _worked(ctx, base, "a6", "one\n")

    ref = await ctx.workspace.export(_record(ctx, "a6", kind="overlay"))

    assert ref == f"ph/{SESSION}/a6"
    code, _, _ = await git(ctx, base, "rev-parse", "--verify", f"refs/heads/{ref}")
    assert code == 0, "the export left a real branch behind"
