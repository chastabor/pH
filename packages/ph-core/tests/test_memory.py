"""P4-13 — `memory-agents-md`: what the user wrote, where the cache survives it (G8).

Phase 1 found these files already. What this row owns is *placement*, and the
two tests that matter here are the two properties placement buys.

**An edit lands in the next turn**, because the provider is asked per assembly
rather than read once at mount — which is what makes a file called memory worth
editing at all.

**And it lands after the cached prefix**, so the edit costs one snapshot rather
than every token before it. That half is asserted where the prefix test lives
(`test_prefix_stability.py`), because the claim is about the *request sequence*
and that file already owns the machinery for reading one.

Discovery order is nearest-first, then the user's own file — a subdirectory's
instructions are more specific than the repository's, and every level is kept
rather than the nearest one winning, because `AGENTS.md` is additive by
convention.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from ph.cordis import DEPLOYMENT
from ph.paths import write_text_under
from ph.system_prompt.assembly import AssembleContext
from ph.system_prompt.memory import (
    CACHE_MAX,
    FILENAME,
    MAX_BYTES,
    MemoryFiles,
    discover,
    render,
)
from ph.testing import FAKE_OPTIONS as FAKE
from ph.testing import StubWorkspaceProvider

pytestmark = pytest.mark.anyio


def _write(path: Path, text: str) -> Path:
    write_text_under(path, text)
    return path


async def _assembled(ctx: Any, agent: Any = None) -> str:
    """The memory snapshot as this profile would render it, or `""`.

    Filtered by name rather than using `join_context_sections`, because what is
    under test is *this* row's contribution and a profile contributes others.
    """
    assembly = await ctx.system_prompt.assemble(ctx, agent=agent)
    return "\n\n".join(section.text for section in assembly.contexts if section.name == "memory")


# ----------------------------------------------------------------- discovery --


def test_files_are_found_nearest_first_then_the_user(tmp_path: Path) -> None:
    """Both levels, in the order the model reads them."""
    _write(tmp_path / "project" / "AGENTS.md", "repo rules")
    _write(tmp_path / "project" / "pkg" / "AGENTS.md", "package rules")
    _write(tmp_path / "home" / "AGENTS.md", "user rules")

    found = discover(tmp_path / "project" / "pkg", home=tmp_path / "home")

    assert [item.text for item in found] == ["package rules", "repo rules", "user rules"]
    assert [item.scope for item in found] == ["project", "project", "user"]


def test_a_file_that_is_also_the_users_is_not_read_twice(tmp_path: Path) -> None:
    """`$PH_HOME` under the workspace is an ordinary layout for a single-user
    box, and the same file twice is the model reading one instruction as two."""
    _write(tmp_path / "AGENTS.md", "once")

    found = discover(tmp_path, home=tmp_path)

    assert [item.text for item in found] == ["once"]


def test_an_oversized_file_is_truncated_rather_than_refused(tmp_path: Path) -> None:
    """A refusal would drop instructions the user believes are in force; the cap
    is about what a *request* can carry, so the first 64 KiB is the honest
    answer to "how much of this fits"."""
    _write(tmp_path / "AGENTS.md", "x" * (MAX_BYTES + 500))

    (found,) = discover(tmp_path, home=tmp_path / "nowhere")

    assert len(found.text) == MAX_BYTES


def test_nothing_found_renders_nothing(tmp_path: Path) -> None:
    """`""` is how a `context()` opts out for an assembly — a deployment with no
    memory contributes no snapshot, not an empty heading."""
    assert render(discover(tmp_path, home=tmp_path)) == ""


# ------------------------------------------------------------------- the row --


async def test_the_row_contributes_a_context_not_a_section(mount: Any, tmp_path: Path) -> None:
    """G8 in one assertion. A `section` is the cached prefix; putting a file
    whose purpose is to be edited there bills every earlier token for the edit.
    """
    _write(tmp_path / "AGENTS.md", "Prefer explicit code.")
    ctx = await mount()

    assembly = await ctx.system_prompt.assemble(DEPLOYMENT)

    assert "Prefer explicit code." in await _assembled(ctx)
    assert not any("Prefer explicit code." in text for _, text in assembly.sections)


async def test_an_edit_is_visible_without_a_restart(mount: Any, tmp_path: Path) -> None:
    """The Phase-1 row read at mount, so editing memory did nothing until the
    process restarted. Being a provider is what fixes that; the cache below is
    what keeps it from costing a read per turn."""
    memory = _write(tmp_path / "AGENTS.md", "Prefer tabs.")
    ctx = await mount()
    assert "Prefer tabs." in await _assembled(ctx)

    _write(memory, "Prefer spaces, and say why.")

    assert "Prefer spaces, and say why." in await _assembled(ctx)


async def test_an_unchanged_file_is_not_re_read(mount: Any, tmp_path: Path) -> None:
    """`assemble` runs once per model *step*, so "memory is live" must not mean
    "every step reads 64 KiB from disk".

    The assertion is **zero** reads, which is what makes it a test of the
    ordering rather than of the cache: a version that reads first and compares
    signatures afterwards saves the render and none of the I/O, and would pass
    any assertion weaker than this one.
    """
    _write(tmp_path / "AGENTS.md", "stable")
    ctx = await mount()
    memory = MemoryFiles(ctx=ctx, home=tmp_path)
    request = AssembleContext(scope=ctx)
    first = memory.text(request)

    opens = 0
    original = Path.open

    def counted(self: Path, *args: Any, **kwargs: Any) -> Any:
        nonlocal opens
        if self.name == FILENAME:
            opens += 1
        return original(self, *args, **kwargs)

    Path.open = counted  # type: ignore[method-assign]
    try:
        assert memory.text(request) == first
    finally:
        Path.open = original  # type: ignore[method-assign]

    assert opens == 0, "the file was read to answer a question the signature had"


async def test_an_edit_is_read_again(mount: Any, tmp_path: Path) -> None:
    """The other half: the signature is mtime and size, so a changed file is
    re-read exactly when it changed."""
    memory_file = _write(tmp_path / "AGENTS.md", "before")
    ctx = await mount()
    memory = MemoryFiles(ctx=ctx, home=tmp_path)
    request = AssembleContext(scope=ctx)
    assert "before" in memory.text(request)

    _write(memory_file, "after, and longer than before")

    assert "after, and longer than before" in memory.text(request)


async def test_the_cache_does_not_grow_without_bound(mount: Any, tmp_path: Path) -> None:
    """The key is a workspace root, and a daemon fans out one per child — so
    without a bound the process retains a rendered snapshot for every worktree
    it ever saw. Driven through `text()`, because a test that re-implemented the
    eviction beside it would pass with the bound deleted."""
    ctx = await mount()
    memory = MemoryFiles(ctx=ctx, home=tmp_path)
    roots = [tmp_path / f"tree-{index}" for index in range(CACHE_MAX + 2)]
    for root in roots:
        _write(root / "AGENTS.md", f"rules for {root.name}")

    with patch.object(MemoryFiles, "root", lambda _self, agent: agent):
        for root in roots:
            memory.text(AssembleContext(scope=ctx, agent=root))

    assert len(memory._cache) <= CACHE_MAX


async def test_an_agent_reads_the_memory_of_the_tree_it_is_in(mount: Any, tmp_path: Path) -> None:
    """Per agent, not per process (D21). A child working in a worktree that has
    its own `AGENTS.md` is being told the rules of the tree it is in — a root
    read once at mount would hand every agent the process's own directory."""
    _write(tmp_path / "AGENTS.md", "the parent tree")
    ctx = await mount()
    ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    workspace = await ctx.workspace.acquire(
        session_id=session.id, agent_id=agent.id, base=tmp_path, session=session
    )
    _write(workspace.root / "AGENTS.md", "the tree this agent got")

    text = await _assembled(ctx, agent)

    assert "the tree this agent got" in text
    # And the tree it branched from, because a repository's rules still apply
    # inside a worktree of it.
    assert "the parent tree" in text
