"""P4-08b — making a fresh workspace usable, and refusing to leave the tree (E14).

Two claims carry this file.

**A worktree is not a usable checkout until its ignored materials arrive.**
`git worktree add` carries no `.env` and no dependency directory, so a child in
its own tree cannot run the project and reads the failure as its own bug. The
materials are copied, never rebuilt — no installer runs, so provisioning is
offline and cannot fail because a registry is down.

**Nothing outside the tree, in either direction.** Every guard here is a
*refusal*, and the reason they are tested one by one is that the tool this was
read off (`sources/wtp`) applies its own containment check only to relative
paths — an absolute `from` skips it entirely. That is coherent when the config
is the developer's own and the threat is a typo. It is not available to a
harness whose premise is that the thing reading the repository may be hostile.

## What each provision mode costs

Measured on a synthetic dependency tree (**2 000 packages, 20 000 files, 22 MB**):
`copy` **0.674 s**, `hardlink` **0.188 s**, `symlink` **0.003 s** — against tens of
seconds plus a network round-trip for the installer they replace.

That spread is why the default is the *isolated* one and the cheap ones are opt-in
with their sharing named: `hardlink` is mutated in place by a build step that
rewrites rather than replaces, and `symlink` is shared and mutable outright.

## Why the `FICLONE` probe is latched per device pair

On a filesystem that cannot clone, the attempt costs an open, an open, a failing
`ioctl`, an unlink and two closes **per file** — measured at **+50% wall time over a
20 000-file tree**, paid per child in a fan-out. The answer never varies within one
device pair, and a source and destination on different devices can only ever return
`EXDEV`, which two `stat` calls settle without attempting anything.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from ph.seams.workspace_provision import (
    ProvisionEntry,
    ProvisionRefused,
    provision,
    resolve_entry,
)

pytestmark = pytest.mark.anyio


@pytest.fixture
def trees(tmp_path: Path) -> tuple[Path, Path]:
    """`(base, root)` — a project and a fresh workspace branched from it."""
    base, root = tmp_path / "project", tmp_path / "worktree"
    base.mkdir()
    root.mkdir()
    (base / ".env").write_text("SECRET=1\n", encoding="utf-8")
    deps = base / "node_modules" / "left-pad"
    deps.mkdir(parents=True)
    (deps / "index.js").write_text("module.exports = 1\n", encoding="utf-8")
    return base, root


# ------------------------------------------------------------------ refusals --


@pytest.mark.parametrize(
    "raw",
    ["/etc/passwd", "~/.ssh/id_rsa", ""],
    ids=["absolute", "home-relative", "empty"],
)
@pytest.mark.parametrize("field", ["source", "dest"])
def test_a_path_that_is_not_relative_is_refused_at_config_load(field: str, raw: str) -> None:
    """The syntactic half of the guard fires where the **operator** sees it.

    This half needs no `base` and no `root`, so running it at use-time sent a
    profile typo down the failure channel to the *model's* prompt line, once per
    agent, forever, while the person who wrote it saw nothing. As a validator it
    fires at profile-config load and inside `discover_provisioning`, which is the
    same enforcement point `extra="forbid"` already uses to make the data-only
    claim real rather than aspirational.
    """
    fields = {"source": ".env", field: raw}

    with pytest.raises(ValidationError):
        ProvisionEntry(**fields)


@pytest.mark.parametrize(
    ("entry", "because"),
    [
        (ProvisionEntry(source="../outside.txt"), "a parent traversal leaves the project"),
        (
            ProvisionEntry(source=".env", dest="../../escape"),
            "a destination may not leave the root",
        ),
        (ProvisionEntry(source=".git/config"), "git's own state is not a material"),
        (ProvisionEntry(source=".env", dest=".git/hooks/pre-commit"), "a planted hook is code"),
    ],
)
def test_a_path_that_leaves_its_base_is_refused(
    trees: tuple[Path, Path], entry: ProvisionEntry, because: str
) -> None:
    """The *contextual* half — whether these relative paths, resolved, are still
    inside the trees they belong to. It needs `base` and `root`, so it can only
    be answered here.

    The `.git` pair is the one that is not merely about reading and writing:
    a `dest` under `.git/hooks/` is **code that runs on the agent's next
    commit**, which is the execution this whole design refused when it dropped
    the installer.
    """
    base, root = trees

    with pytest.raises(ProvisionRefused):
        resolve_entry(entry, base=base, root=root)


async def test_a_symlink_cannot_be_used_to_read_around_the_guard(
    trees: tuple[Path, Path], tmp_path: Path
) -> None:
    """The containment check runs on the *resolved* path, which is the point.

    A check on the joined-but-unresolved path would accept `link/passwd` for any
    `link` pointing outside — the classic traversal, and the reason `resolve()`
    comes before the comparison rather than after it.
    """
    base, root = trees
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "secret.txt").write_text("no", encoding="utf-8")
    (base / "link").symlink_to(tmp_path / "outside")

    with pytest.raises(ProvisionRefused):
        resolve_entry(ProvisionEntry(source="link/secret.txt"), base=base, root=root)


async def test_a_refusal_is_reported_and_the_rest_still_arrive(trees: tuple[Path, Path]) -> None:
    """Per-entry, never fatal. A harness that will not start because one line of
    a repository's optional config is wrong is worse than a missing `.env`."""
    base, root = trees

    report = await provision(
        [ProvisionEntry(source="../escape"), ProvisionEntry(source=".env")],
        base=base,
        root=root,
    )

    assert report.provisioned == (".env",)
    assert len(report.failed) == 1
    assert (root / ".env").read_text(encoding="utf-8") == "SECRET=1\n"


# --------------------------------------------------------------------- modes --


async def test_copy_brings_the_gitignored_material_a_checkout_lacks(
    trees: tuple[Path, Path],
) -> None:
    """E14's whole reason: `git worktree add` carries no ignored file, so the
    child has no `.env` and cannot run the project."""
    base, root = trees

    report = await provision([ProvisionEntry(source=".env")], base=base, root=root)

    assert (root / ".env").read_text(encoding="utf-8") == "SECRET=1\n"
    assert report.failed == ()
    # Isolated: the child's copy is its own, which is what `copy` buys over the
    # two cheaper modes.
    (root / ".env").write_text("SECRET=2\n", encoding="utf-8")
    assert (base / ".env").read_text(encoding="utf-8") == "SECRET=1\n"


async def test_copy_recurses_and_never_follows_a_symlink_out_of_the_tree(
    trees: tuple[Path, Path], tmp_path: Path
) -> None:
    """A dependency directory is full of symlinks (`.bin/` is nothing else), and
    copying *through* one turns a small tree into whatever it points at — or
    reaches outside the project entirely. They are recreated, not followed."""
    base, root = trees
    (tmp_path / "elsewhere").mkdir(exist_ok=True)
    (tmp_path / "elsewhere" / "big").write_text("x" * 100, encoding="utf-8")
    (base / "node_modules" / ".bin").mkdir()
    (base / "node_modules" / ".bin" / "tool").symlink_to(tmp_path / "elsewhere" / "big")

    await provision([ProvisionEntry(source="node_modules")], base=base, root=root)

    copied = root / "node_modules" / "left-pad" / "index.js"
    assert copied.read_text(encoding="utf-8") == "module.exports = 1\n"
    assert (root / "node_modules" / ".bin" / "tool").is_symlink()


async def test_a_symlinked_directory_survives_the_copy(
    trees: tuple[Path, Path], tmp_path: Path
) -> None:
    """The regression this row's first draft shipped, and the reason it uses
    `shutil.copytree` rather than a hand-rolled walk.

    `os.walk(followlinks=False)` puts a symlinked *directory* in `dirs` and never
    descends into it, so a loop that only recreates entries drawn from `files`
    drops it **entirely** — no copy, no link, nothing. That is most of a pnpm
    `node_modules` (every `<pkg> -> ../.pnpm/...`) and every `.venv/lib64 -> lib`
    — i.e. exactly the trees this row exists to bring across.
    """
    base, root = trees
    (base / "node_modules" / ".pnpm").mkdir()
    (base / "node_modules" / ".pnpm" / "real.js").write_text("x\n", encoding="utf-8")
    (base / "node_modules" / "linked-pkg").symlink_to(
        base / "node_modules" / ".pnpm", target_is_directory=True
    )

    report = await provision([ProvisionEntry(source="node_modules")], base=base, root=root)

    assert report.failed == ()
    assert (root / "node_modules" / "linked-pkg").is_symlink()
    assert (root / "node_modules" / ".pnpm" / "real.js").exists()


async def test_hardlink_shares_the_bytes_and_says_so(trees: tuple[Path, Path]) -> None:
    """Near-free, and safe for the replace-and-rename every package manager
    uses — pnpm's whole store is a hardlink farm. What it is *not* safe for is a
    build step that writes in place, which is why it is opt-in."""
    base, root = trees

    await provision([ProvisionEntry(source="node_modules", mode="hardlink")], base=base, root=root)

    source = base / "node_modules" / "left-pad" / "index.js"
    linked = root / "node_modules" / "left-pad" / "index.js"
    assert linked.stat().st_ino == source.stat().st_ino


async def test_symlink_is_shared_and_refuses_to_clobber(trees: tuple[Path, Path]) -> None:
    """Instant and zero-space, and a child writing through it reaches the
    parent's tree — which hands back some of the collision isolation the tier
    exists to buy, so the mode says so and stays opt-in.

    Refusing an existing destination is the second half: quietly turning a
    checked-in file into a link to somewhere else is not something a config line
    should be able to do.
    """
    base, root = trees

    report = await provision(
        [ProvisionEntry(source="node_modules", mode="symlink")], base=base, root=root
    )

    assert report.provisioned == ("node_modules",)
    assert (root / "node_modules").is_symlink()

    again = await provision(
        [ProvisionEntry(source="node_modules", mode="symlink")], base=base, root=root
    )
    assert again.provisioned == ()
    assert len(again.failed) == 1


async def test_a_missing_optional_source_is_not_a_failure(trees: tuple[Path, Path]) -> None:
    """Most projects have no `.env`; a fresh clone certainly does not. A list
    that reported a failure for every material a project happens not to use
    would put noise on every agent's prompt line."""
    base, root = trees

    report = await provision(
        [
            ProvisionEntry(source="nothing-here"),
            ProvisionEntry(source="also-missing", optional=False),
        ],
        base=base,
        root=root,
    )

    assert report.failed == ("also-missing: also-missing is not in the project",)


# ------------------------------------------------------------------- exclude --


async def test_the_report_names_what_was_put_there(trees: tuple[Path, Path]) -> None:
    """The list that keeps this row from defeating the tier it serves.

    A copied `.env` the project does not gitignore is an untracked file, so
    `git status` calls the tree dirty and `git add -A` puts it on the branch —
    which is a credential in the project's history, not just a checkout nobody
    wanted. `provisioned` is what disposal subtracts, and `test_workspace_git.py`
    pins the end of that.

    **Not `info/exclude`**, which is the obvious wrong answer and was the first
    one tried: git resolves `info/exclude` against the *common* directory even
    from inside a linked worktree — `git rev-parse --git-path info/exclude`
    returns the repository's own file — so there is no per-worktree exclude to
    write, and the only one that works would hide these paths in every other
    worktree and in the user's own tree.
    """
    base, root = trees

    report = await provision(
        [
            ProvisionEntry(source=".env"),
            ProvisionEntry(source="node_modules", mode="symlink"),
            ProvisionEntry(source="missing"),
        ],
        base=base,
        root=root,
    )

    assert report.provisioned == (".env", "node_modules")


async def test_a_workspace_that_is_not_a_checkout_still_gets_its_materials(
    trees: tuple[Path, Path],
) -> None:
    """`readonly-scratch` (P6-05) has a fresh root and no `.git` at all. The
    exclude step is the part that does not apply; the materials are the part
    that does, and failing the whole thing over the first would be backwards."""
    base, root = trees

    report = await provision([ProvisionEntry(source=".env")], base=base, root=root)

    assert report.provisioned == (".env",)
    assert (root / ".env").exists()


async def test_provisioning_opens_no_socket(
    trees: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The gate that makes the "no installer" decision enforceable rather than
    aspirational: a registry outage cannot stop an agent from starting, because
    nothing here reaches a registry."""
    import socket

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("provisioning must not touch the network")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    base, root = trees

    report = await provision(
        [
            ProvisionEntry(source=".env"),
            ProvisionEntry(source="node_modules", mode="hardlink"),
        ],
        base=base,
        root=root,
    )

    assert report.failed == ()
    assert os.path.exists(root / "node_modules" / "left-pad" / "index.js")
