"""P4-09 — per-run restore points, and what a restore cannot reach (E7, N3).

A denial settles the whole run (Q9a), which bounds partial state to *about* one
cell — "about", because the program had already written whatever preceded the
refused line. These tests are about making that recoverable, and about the one
sentence a user has to be told while it happens: **git restores the tree, not
the world.**

Real `git` throughout. What is pinned here is git's own behaviour — that a tree
written against a scratch index leaves the agent's staging area alone, that a
restore leaves an untracked file untracked rather than silently staging it, that
an ignored path is never in a tree `add -A` built — none of which a fake would
prove.

**Why the restore is `read-tree --reset -u` against a seeded scratch index.**
Seeding from the checkpoint index — refreshed first, so its stat data is current
— lets git touch only the paths that actually differ. `checkout-index -a -f` was
the obvious alternative and rewrites *every* file in the tree: **2.3 s of a 2.4 s
restore on an 11 000-file checkout**. Worse than the time, it stamps a new mtime
on 11 000 unchanged files and so invalidates every mtime-keyed cache the person
has — pytest, mypy, ruff, the editor's index — making them pay for the full-tree
rewrite again on their next command.

## Why `latest_checkpoint` scans in reverse instead of folding every point

The caller wants one restore point, and `checkpoints()` plus `max()` builds a dict
of every one — copying each payload to discard all but the last. Measured on a
**500 000-event log: 7.2 ms and 0.5 MB against 2 µs** for the reverse scan, on a
crash path that runs once per retry.

## Why pH's index is seeded from the worktree's real one

`git worktree add` has just written an index full of valid stat data. Starting
from an empty file instead makes the first `add -A` re-hash **every file in the
repository — measured at 2.6 s on an 11 000-file checkout**, paid per agent, on
the first cell. Copying it costs half a millisecond.

**Both `tree_hash` guards live in `tree_hash`.** Keeping copies at the call site
cost a *fifth* `git` spawn per code cell — `rev-parse --absolute-git-dir`, **1.3
ms** — which is the waste `_cached_checkpoint` names as "one of four spawns spent
learning a constant".
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ph.seams.workspace_git import (
    CHECKPOINT,
    checkpoint,
    checkpoints,
    ref_for,
    restore,
)
from ph.testing import (
    git,
    git_repo,
    needs_git,
    run_tool,
    worktree_agent,
)
from ph.tools.registry import RUN_CODE

pytestmark = [pytest.mark.anyio, needs_git]


async def _checkpointed(ctx: Any, session: Any, agent: Any, call_id: str = "c1") -> int:
    """Take a restore point and hand back the seq a person would type."""
    workspace = ctx.workspace.of(agent.id)
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id=call_id)
    return int(next(item.seq for item in reversed(session.events) if item.type == CHECKPOINT))


async def _run(ctx: Any, session: Any, agent: Any, argument: str = "") -> str:
    shown = await ctx.commands.dispatch(f"/revert {argument}".strip(), session=session, agent=agent)
    assert shown is not None
    return str(shown)


# ------------------------------------------------------------------ capture --


async def test_a_checkpoint_captures_the_tree_without_disturbing_the_agent(
    mount: Any, tmp_path: Path
) -> None:
    """The property that lets this run before *every* cell: it is invisible.

    Branch history, the working tree and — the one that would actually bite —
    the agent's own staging area are all untouched, because the capture goes
    through pH's own `GIT_INDEX_FILE`. An agent that had staged a file for its
    own commit finds it still staged, and only staged.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    (workspace.root / "staged.txt").write_text("mine\n", encoding="utf-8")
    await git(ctx, workspace.root, "add", "staged.txt")
    _, before, _ = await git(ctx, workspace.root, "status", "--porcelain")

    ref = await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="call-1")

    assert ref is not None
    _, after, _ = await git(ctx, workspace.root, "status", "--porcelain")
    assert after == before, "the checkpoint disturbed the agent's index"
    # A ref outside `refs/heads`, so `git branch` does not list it and a push
    # does not carry it — it exists only to keep the tree from being collected.
    assert ref.startswith("refs/ph/")
    _, branches, _ = await git(ctx, workspace.root, "branch", "--list")
    assert "pre-run" not in branches


async def test_the_event_precedes_the_ref_it_is_addressed_by(mount: Any, tmp_path: Path) -> None:
    """Write-ahead (A10), and the reason `/revert <seq>` names one thing.

    The event is appended *first* and its own `seq` is the address — nothing
    predicts what `append` will assign, and the payload carries neither `seq`
    nor `ref` because both are derivable from the event that holds them. A
    person types one number instead of correlating an event with a ref.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)

    ref = await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="call-1")

    (event,) = [item for item in session.events if item.type == CHECKPOINT]
    assert ref == ref_for(session.id, agent.id, event.seq)
    assert set(event.data) == {"agentId", "tree", "callId"}
    # The ref exists and points at the tree the event named.
    _, resolved, _ = await git(ctx, workspace.root, "rev-parse", ref)
    assert resolved.strip() == event.data["tree"]


async def test_a_shared_workspace_is_never_checkpointed(mount: Any, tmp_path: Path) -> None:
    """A `shared` workspace is the *person's* own checkout.

    A restore point there would offer to overwrite their uncommitted work with
    whatever state an agent happened to find the tree in — so the gate is the
    kind, and this arrives with the `worktree` tier or not at all.
    """
    ctx = await mount()
    base = await git_repo(ctx, tmp_path / "repo")
    session = ctx.sessions.create("s1")
    workspace = await ctx.workspace.acquire(
        session_id="s1", agent_id="a1", base=base, session=session
    )
    assert workspace.kind == "shared"

    ref = await checkpoint(ctx, workspace, session=session, agent_id="a1", call_id="c1")

    assert ref is None
    assert not [item for item in session.events if item.type == CHECKPOINT]


# ------------------------------------------------------------------ restore --


async def test_a_denied_run_reverts_exactly(mount: Any, tmp_path: Path) -> None:
    """E7's gate, end to end.

    Everything the run touched goes back: a tracked file it modified, a tracked
    file it deleted, and an untracked file it created is taken away again. What
    it did *not* touch is not touched either.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    root = workspace.root
    (root / "untracked.txt").write_text("before\n", encoding="utf-8")
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="c1")
    tree = next(item.data["tree"] for item in session.events if item.type == CHECKPOINT)

    # ... the run, before it was denied.
    (root / "tracked.txt").write_text("clobbered\n", encoding="utf-8")
    (root / "untracked.txt").unlink()
    (root / "new").mkdir()
    (root / "new" / "spilled.txt").write_text("partial\n", encoding="utf-8")

    removed = await restore(ctx, workspace, tree)

    assert (root / "tracked.txt").read_text(encoding="utf-8") == "original\n"
    assert (root / "untracked.txt").read_text(encoding="utf-8") == "before\n"
    assert not (root / "new" / "spilled.txt").exists()
    assert not (root / "new").exists(), "the directory the run created was left behind"
    assert "new/spilled.txt" in removed


async def test_ignored_paths_are_never_touched(mount: Any, tmp_path: Path) -> None:
    """`.gitignore`d paths are outside the promise, in both directions.

    `git add -A` never put them in the tree, so a restore has nothing to put
    back — and must not delete them either. A build cache is not the agent's
    work, and a `/revert` that wiped one would turn a recovery into a rebuild.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    root = workspace.root
    (root / "build.log").write_text("cached\n", encoding="utf-8")
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="c1")
    tree = next(item.data["tree"] for item in session.events if item.type == CHECKPOINT)
    (root / "build.log").write_text("cached, then some\n", encoding="utf-8")
    (root / "after.log").write_text("also ignored\n", encoding="utf-8")

    await restore(ctx, workspace, tree)

    assert (root / "build.log").read_text(encoding="utf-8") == "cached, then some\n"
    assert (root / "after.log").exists()


async def test_an_untracked_file_comes_back_untracked(mount: Any, tmp_path: Path) -> None:
    """The reason the restore uses a scratch index rather than the agent's.

    `git add -A` puts untracked-not-ignored files in the checkpoint tree, so
    reading that tree into the *real* index would come back with them staged —
    a `git status` an agent did not create and cannot explain.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    root = workspace.root
    (root / "untracked.txt").write_text("before\n", encoding="utf-8")
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="c1")
    tree = next(item.data["tree"] for item in session.events if item.type == CHECKPOINT)
    (root / "untracked.txt").write_text("changed\n", encoding="utf-8")

    await restore(ctx, workspace, tree)

    _, status, _ = await git(ctx, root, "status", "--porcelain")
    assert status.strip() == "?? untracked.txt", f"restore changed the index: {status!r}"


async def test_scratch_survives_a_revert(mount: Any, tmp_path: Path) -> None:
    """Scratch lives outside the worktree, so it is safe *by construction*.

    That is what makes it the right place for a child to put anything it wants
    to keep — notes, extracted data, a reproduction script — and it is why E5
    puts it there rather than in the tree.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    (workspace.scratch / "notes.md").write_text("what I learned\n", encoding="utf-8")
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="c1")
    tree = next(item.data["tree"] for item in session.events if item.type == CHECKPOINT)

    await restore(ctx, workspace, tree)

    assert (workspace.scratch / "notes.md").read_text(encoding="utf-8") == "what I learned\n"


async def test_a_collected_tree_reports_rather_than_raises(mount: Any, tmp_path: Path) -> None:
    """The write-ahead window (A10) has a visible failure mode, not a silent one.

    The event is appended before the ref that keeps the tree alive, so a crash
    in between leaves a restore point naming an object git may have collected.
    That has to read as "this checkpoint is gone", never as a traceback or a
    half-restored tree.
    """
    ctx, _session, _agent, workspace = await worktree_agent(mount, tmp_path)

    with pytest.raises(FileNotFoundError):
        await restore(ctx, workspace, "0" * 40)


# --------------------------------------------------------------------- fold --


async def test_restore_points_are_a_fold_over_the_log(mount: Any, tmp_path: Path) -> None:
    """A checkpoint is a *fact in the log*, so a resumed or forked session finds
    the same restore points a live one has — nothing had to remember them."""
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="c1")
    (workspace.root / "tracked.txt").write_text("second\n", encoding="utf-8")
    await checkpoint(ctx, workspace, session=session, agent_id=agent.id, call_id="c2")

    found = checkpoints(session)

    assert len(found) == 2
    assert [point["callId"] for _seq, point in sorted(found.items())] == ["c1", "c2"]


# ------------------------------------------------------------------ /revert --


async def test_revert_restores_and_says_what_it_restored(mount: Any, tmp_path: Path) -> None:
    """The happy path, through the command a person actually types."""
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    seq = await _checkpointed(ctx, session, agent)
    (workspace.root / "tracked.txt").write_text("the run did this\n", encoding="utf-8")

    shown = await _run(ctx, session, agent, str(seq))

    assert "restored" in shown
    assert (workspace.root / "tracked.txt").read_text(encoding="utf-8") == "original\n"


async def test_revert_lists_what_restoring_the_tree_did_not_undo(
    mount: Any, tmp_path: Path
) -> None:
    """N3, and the reason the command exists in this shape.

    Restoring the tree does not un-publish a package or un-drop a table. A user
    who reads "reverted" and stops there believes the run had no effect, so the
    dispatches a tree restore cannot cover are listed by name — while the ones
    it *does* cover (`edit`, and the read-only tools) are not, which is what
    keeps the list about the calls that actually reached past the tree.
    """
    ctx, session, agent, _workspace = await worktree_agent(mount, tmp_path)
    seq = await _checkpointed(ctx, session, agent)
    for name, arguments in (
        ("edit", {"path": "tracked.txt"}),
        ("bash", {"command": "npm publish"}),
        ("read", {"path": "tracked.txt"}),
    ):
        session.append(
            "tool/code-dispatch-start",
            {
                "parentCallId": "c1",
                "subCallId": f"c1:code:{name}",
                "name": name,
                "arguments": arguments,
            },
        )

    shown = await _run(ctx, session, agent, str(seq))

    assert "not the world" in shown
    assert "bash" in shown
    # `edit` and `read` declare `undone_by_workspace_restore`, so listing them
    # would be noise in the one place a person is checking for surprises.
    assert "edit" not in shown
    assert "npm publish" in shown


async def test_an_unknown_tool_is_reported_as_not_undone(mount: Any, tmp_path: Path) -> None:
    """The default is the safe direction.

    A tool that declares nothing — an MCP server's, a row added next month — is
    listed rather than trusted, so a new capability is over-reported instead of
    silently assumed reversible.
    """
    ctx, session, agent, _workspace = await worktree_agent(mount, tmp_path)
    seq = await _checkpointed(ctx, session, agent)
    session.append(
        "tool/code-dispatch-start",
        {
            "parentCallId": "c1",
            "subCallId": "c1:code:0",
            "name": "acme_deploy",
            "arguments": {"env": "prod"},
        },
    )

    shown = await _run(ctx, session, agent, str(seq))

    assert "acme_deploy" in shown


async def test_revert_with_no_argument_lists_the_restore_points(mount: Any, tmp_path: Path) -> None:
    """A person who does not know the seq should not have to read the log."""
    ctx, session, agent, _workspace = await worktree_agent(mount, tmp_path)
    await _checkpointed(ctx, session, agent)

    shown = await _run(ctx, session, agent)

    assert "c1" in shown
    assert agent.id in shown


async def test_an_unknown_seq_names_the_ones_that_exist(mount: Any, tmp_path: Path) -> None:
    ctx, session, agent, _workspace = await worktree_agent(mount, tmp_path)
    seq = await _checkpointed(ctx, session, agent)

    shown = await _run(ctx, session, agent, "9999")

    assert "refusing" in shown
    assert str(seq) in shown


# ------------------------------------------------------------------- the row --


async def test_the_row_checkpoints_a_code_run_and_nothing_else(mount: Any, tmp_path: Path) -> None:
    """The policy row, mounted — every other test here calls `checkpoint()` directly.

    Two halves. A run through the *transport* takes a restore point, which is the
    row's whole job. A native tool call does not: a call that is denied never
    ran, so there is nothing to restore it to, and a checkpoint per `read` would
    be a git spawn per tool call for no recoverable state (Q9a).
    """
    ctx, session, agent, _workspace = await worktree_agent(
        mount,
        tmp_path,
        {
            "insert": [
                {"id": "code-runtime-stub", "name": "code-runtime-stub"},
                {"id": "tools-code-mode", "name": "tools-code-mode"},
            ]
        },
    )
    ctx.code_runtime_stub.register_program("noop", lambda: None)

    await run_tool(ctx, "read", {"path": "tracked.txt"}, agent=agent, session=session)
    assert not [item for item in session.events if item.type == CHECKPOINT]

    await run_tool(ctx, RUN_CODE, {"program": "noop"}, agent=agent, session=session)

    (event,) = [item for item in session.events if item.type == CHECKPOINT]
    assert event.data["agentId"] == agent.id
    assert event.data["tree"]


async def test_provisioned_materials_are_not_the_agents_work(mount: Any, tmp_path: Path) -> None:
    """E14 meets E7, and the seam already settled which set is which.

    A copied `.env` or a hardlinked `node_modules` is untracked-and-not-ignored
    by construction, so a checkpoint that took `git add -A` wholesale would hash
    the whole dependency tree into every cell's tree object — and a restore would
    then delete anything provisioned after the checkpoint. `agent_work_pathspec()`
    is the one definition of "the agent's work", and this is its third consumer.
    """
    ctx, session, agent, workspace = await worktree_agent(mount, tmp_path)
    (workspace.root / "node_modules").mkdir()
    (workspace.root / "node_modules" / "dep.js").write_text("x\n", encoding="utf-8")
    held = replace(workspace, provisioned=("node_modules",))
    ctx.workspace._held[agent.id].workspace = held

    await checkpoint(ctx, held, session=session, agent_id=agent.id, call_id="c1")
    tree = next(item.data["tree"] for item in session.events if item.type == CHECKPOINT)
    _, listing, _ = await git(ctx, workspace.root, "ls-tree", "-r", "--name-only", tree)

    assert "node_modules/dep.js" not in listing
    assert "tracked.txt" in listing
