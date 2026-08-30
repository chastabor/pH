"""P1-17 — the filesystem seam, and the gate that runs *before* the write.

Gate: *write-intent fires before the write; a veto prevents it.*

"Before" is the entire claim. A hook that ran after the write would be a
reporter, which is precisely the failure the feature map records in
prime-agent's `edit` skill — the diff is emitted after the file changed, so
"there is no point at which anything can say no".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import Context
from ph.seams.fs import (
    Config as FsConfig,
)
from ph.seams.fs import (
    EditIntent,
    FsDenied,
    FsService,
    WalkDecision,
    WriteIntent,
    apply,
    matches_glob,
    read_before_edit,
)
from ph.session import Session
from ph.testing import StubAgent

pytestmark = pytest.mark.anyio


def _fs(tmp_path: Path) -> tuple[Context, FsService]:
    """A bare service, with no screens registered.

    Bare on purpose: most of this file is about the write gate, and a walk that
    prunes nothing is the simplest thing that can answer a `glob`. The ignore
    list is `fs-local`'s config since P6-19 rather than knowledge inside `_walk`,
    so the test that is about *that* mounts the row — see `_mounted` below.
    """
    root = Context()
    service = FsService(ctx=root, root=tmp_path)
    root.provide("fs", service)
    return root, service


async def _mounted(tmp_path: Path, **config: Any) -> FsService:
    """`fs-local` as a profile composes it, screens included.

    The distinction matters since P6-19: pruning `node_modules` is a *screen the
    row registers*, not a branch inside the walk, so a test that hand-built the
    service and expected pruning would be asserting against a mechanism that no
    longer exists there.
    """
    root = Context()
    await apply(root, FsConfig(root=str(tmp_path), **config))
    service: FsService = root.fs
    return service


async def test_write_intent_fires_before_the_write_and_a_veto_prevents_it(
    tmp_path: Path,
) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "guarded.txt"
    observed: list[bool] = []

    async def deny(intent: WriteIntent, next_: Any) -> str:
        # The file must not exist yet when policy runs.
        observed.append(intent.path.exists())
        return "writes to this path are not allowed"

    root.on("fs/write-intent", deny)
    with pytest.raises(FsDenied, match="not allowed"):
        await fs.write("guarded.txt", "content")
    assert observed == [False]
    assert not target.exists()


async def test_an_allowed_write_reaches_disk_and_announces_itself(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    changed: list[Path] = []
    root.on("fs/changed", lambda path: changed.append(path))
    written = await fs.write("notes/deep.txt", "hello")
    assert written.read_text() == "hello"
    assert changed == [written]


async def test_reads_record_an_observation(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("one\ntwo\nthree\n")
    session = Session("s")
    window = await fs.read("a.txt", session=session)
    assert window.total_lines == 3
    assert [event.type for event in session.events] == ["fs/observed"]
    assert fs.observed_mtime("a.txt") is not None


async def test_a_read_window_says_how_to_get_the_next_one(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "long.txt").write_text("\n".join(str(n) for n in range(100)))
    window = await fs.read("long.txt", offset=10, limit=5)
    assert window.text.splitlines() == ["10", "11", "12", "13", "14"]
    assert window.truncated
    assert window.offset == 10


async def test_edit_requires_a_unique_match_unless_told_otherwise(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    target = tmp_path / "dup.txt"
    target.write_text("x\nx\n")
    with pytest.raises(ValueError, match="2 occurrences"):
        await fs.edit("dup.txt", "x", "y")
    assert await fs.edit("dup.txt", "x", "y", replace_all=True) == 2
    assert target.read_text() == "y\ny\n"


async def test_editing_absent_text_is_an_error(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    with pytest.raises(ValueError, match="no occurrence"):
        await fs.edit("a.txt", "goodbye", "hi")


async def test_read_before_edit_refuses_an_unread_file(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    (tmp_path / "a.txt").write_text("hello")
    await read_before_edit(root, None)

    with pytest.raises(FsDenied, match=r"read .* before editing"):
        await fs.edit("a.txt", "hello", "goodbye")

    await fs.read("a.txt")
    assert await fs.edit("a.txt", "hello", "goodbye") == 1


async def test_read_before_edit_refuses_a_file_that_changed_underneath(
    tmp_path: Path,
) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("hello")
    await read_before_edit(root, None)
    await fs.read("a.txt")

    import os
    import time

    time.sleep(0.01)
    target.write_text("changed by someone else")
    os.utime(target, (target.stat().st_atime, target.stat().st_mtime + 10))

    with pytest.raises(FsDenied, match="changed on disk"):
        await fs.edit("a.txt", "changed", "x")


async def test_edit_intent_sees_the_replacement_before_it_lands(tmp_path: Path) -> None:
    root, fs = _fs(tmp_path)
    target = tmp_path / "a.txt"
    target.write_text("before")
    seen: list[EditIntent] = []

    async def observe(intent: EditIntent, next_: Any) -> Any:
        seen.append(intent)
        assert intent.path.read_text() == "before"
        return await next_()

    root.on("fs/edit-intent", observe)
    await fs.edit("a.txt", "before", "after")
    assert seen[0].new_text == "after"
    assert target.read_text() == "after"


async def test_glob_and_grep_skip_the_usual_noise(tmp_path: Path) -> None:
    """The shipped default, now asserted through the row that registers it."""
    fs = await _mounted(tmp_path)
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "keep.py").write_text("import os\nTARGET = 1\n")
    (tmp_path / "node_modules").mkdir()
    (tmp_path / "node_modules" / "skip.py").write_text("TARGET = 2\n")

    found = await fs.glob("**/*.py")
    assert [Path(path).name for path in found] == ["keep.py"]

    matches = await fs.grep("TARGET")
    assert [match.path for match in matches] == [str(tmp_path / "src" / "keep.py")]
    assert matches[0].line == 2


async def test_an_invalid_regular_expression_is_reported_not_raised_raw(
    tmp_path: Path,
) -> None:
    _root, fs = _fs(tmp_path)
    with pytest.raises(ValueError, match="invalid regular expression"):
        await fs.grep("(unclosed")


async def test_relative_paths_resolve_against_the_root(tmp_path: Path) -> None:
    _root, fs = _fs(tmp_path)
    assert fs.resolve("a/b.txt") == tmp_path / "a" / "b.txt"
    # An absolute path passes through: refusing it here would be a confinement
    # claim this layer cannot make (N2).
    assert fs.resolve("/etc/hosts") == Path("/etc/hosts")


# --- P6-17/18/19: the walk, its screens, and who they reach -------------------


async def test_the_ignore_list_is_deployment_config_not_a_module_constant(
    tmp_path: Path,
) -> None:
    """P6-19's second half: two ways to drop a path collapsed into one.

    `_IGNORED_PARTS` was a `frozenset` consulted inline by `_walk`, so a
    repository with a `vendor/` had no way to add one and a deployment that
    wanted to look inside `.git` had no way to say so. It is a screen `fs-local`
    registers like any other, which is also what gives the `.gitignore` row
    somewhere to land instead of arriving as mechanism three.
    """
    for name in ("node_modules", "vendor"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "f.py").write_text("x = 1\n")

    added = await _mounted(tmp_path, ignore=["vendor"])
    assert [Path(p).parent.name for p in await added.glob("**/*.py")] == ["node_modules"]

    # `[]` is not `None`: one turns pruning off, the other keeps the default.
    off = await _mounted(tmp_path, ignore=[])
    assert sorted(Path(p).parent.name for p in await off.glob("**/*.py")) == [
        "node_modules",
        "vendor",
    ]


async def test_a_screen_can_refuse_a_tree_not_just_the_files_in_it(
    tmp_path: Path,
) -> None:
    """P6-19's first half, and the cost that motivated it.

    `hide` could skip a file and could not refuse a directory, so `deny
    secrets/**` concealed every file in `secrets/` *after* entering it and paying
    a predicate call each. `prune` is the walk answering the question it already
    knew how to answer — it has pruned directories before descent since it was
    written, just never on anybody's behalf.
    """
    (tmp_path / "secrets").mkdir()
    for index in range(5):
        (tmp_path / "secrets" / f"key{index}.txt").write_text("shh")
    (tmp_path / "open.txt").write_text("fine")

    root, fs = _fs(tmp_path)
    asked: list[tuple[str, bool]] = []

    def screen(_path: str, name: str, _agent: Any, is_dir: bool) -> WalkDecision:
        asked.append((name, is_dir))
        return "prune" if is_dir and name == "secrets" else "yield"

    fs.screen(screen, scope=root)
    assert [Path(p).name for p in await fs.glob("**/*.txt")] == ["open.txt"]
    # Asked once about the directory, and never about the five files inside it —
    # which is the difference between refusing a tree and refusing a tree's
    # contents one at a time.
    assert ("secrets", True) in asked
    assert not [name for name, is_dir in asked if name.startswith("key")]


async def test_a_screen_that_raises_refuses_at_the_widest_setting(tmp_path: Path) -> None:
    """Fail closed, and fail closed *early*.

    A policy row whose matcher broke must not become an open door — the approval
    seam's reading. For a directory that means `prune` rather than `skip`: a
    broken screen that let the walk descend would be asked once per file inside,
    each time failing the same way.
    """
    (tmp_path / "tree").mkdir()
    (tmp_path / "tree" / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")

    root, fs = _fs(tmp_path)

    def broken(_path: str, _name: str, _agent: Any, _is_dir: bool) -> WalkDecision:
        raise RuntimeError("this matcher is broken")

    fs.screen(broken, scope=root)
    assert await fs.glob("**/*.txt") == []


async def test_a_screen_only_sees_the_agents_its_scope_reaches(tmp_path: Path) -> None:
    """P6-18's enumeration half.

    `hide` kept a flat list of bare callables, so `Context.reaches` had nothing
    to filter on and an agent-scoped `permissions-fs` concealed from *every*
    agent's `glob` — including agents whose deployment never mounted it. Three
    registries already obey this rule; this one did not.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)
    mine = root.scope("agent")
    theirs = root.scope("agent")

    fs.screen(lambda _p, _n, _a, is_dir: "yield" if is_dir else "skip", scope=mine)

    assert await fs.glob("**/*.txt", agent=StubAgent(mine)) == []
    assert len(await fs.glob("**/*.txt", agent=StubAgent(theirs))) == 1, "not this agent's rule"
    assert len(await fs.glob("**/*.txt")) == 1, "nor the process's"


async def test_an_agent_scoped_gate_is_asked_about_that_agents_reads(
    tmp_path: Path,
) -> None:
    """P6-18's veto half, which failed in the opposite direction.

    `FsService._gate` waterfalled with no `scope=`, so `collect` asked whether
    each listener reached the *mount* — and an agent-scoped listener reaches
    only its own agent, so it never fired at all. The sibling gate `ToolRuntime`
    built has passed `scope=execution.scope` since it was written.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)
    mine = root.scope("agent")

    async def deny(_intent: Any, _next: Any) -> str:
        return "this agent may not read"

    mine.on("fs/read-intent", deny)

    with pytest.raises(FsDenied, match="may not read"):
        await fs.read("a.txt", agent=StubAgent(mine))
    # And nobody else's, which is the property the scope buys.
    assert await fs.read("a.txt") is not None


async def test_a_global_row_still_reaches_every_agent(tmp_path: Path) -> None:
    """The half that must not have changed.

    A global registration reaches everything, so scoping the dispatch is a fix
    for agent-scoped rows and a no-op for the way every shipped profile mounts
    `permissions-fs` today. Asserted because "nothing changes for a global row"
    is the claim that made the change safe to land.
    """
    (tmp_path / "a.txt").write_text("a")
    root, fs = _fs(tmp_path)

    async def deny(_intent: Any, _next: Any) -> str:
        return "nobody may read"

    root.on("fs/read-intent", deny)

    with pytest.raises(FsDenied, match="nobody"):
        await fs.read("a.txt", agent=StubAgent(root.scope("agent")))
    with pytest.raises(FsDenied, match="nobody"):
        await fs.read("a.txt")


async def test_the_walk_yields_the_same_paths_a_relative_to_walk_did(tmp_path: Path) -> None:
    """P6-17's correctness gate: a slice, and `Path.relative_to`, agree.

    The optimisation replaced `path.relative_to(base).as_posix()` — 23.5 µs per
    file, 62% of a 430 ms walk — with a prefix slice of the string `os.walk`
    already built. The speedup is only worth having if the answer is identical,
    so this holds the two against each other over a tree with the shapes that
    break naive slicing: nesting, dots in names, and a directory whose name is a
    prefix of a sibling's.
    """
    for relative in (
        "a.py",
        "pkg/b.py",
        "pkg/deep/c.py",
        "pkg.tests/d.py",
        "pkgx/e.py",
        "dotted.name.py",
    ):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("x = 1\n")

    _root, fs = _fs(tmp_path)
    for pattern in ("**/*", "**/*.py", "*.py", "pkg/**/*.py", "pkg*/*.py"):
        found = await fs.glob(pattern)
        expected = sorted(
            str(path)
            for path in tmp_path.glob(pattern)
            if path.is_file() and matches_glob(path.relative_to(tmp_path).as_posix(), pattern)
        )
        assert sorted(found) == expected, pattern
