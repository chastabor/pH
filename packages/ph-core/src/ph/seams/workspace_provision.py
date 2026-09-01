"""Making a fresh workspace *usable*: the materials, never an installer (E14).

`git worktree add` carries no ignored file. A child handed its own checkout has
no `.env`, no `node_modules`, no `.venv`, no `vendor/` — so "run the tests" fails
for a reason the model cannot see and reads as its own bug, which is the same
failure E12's redirection env prevents from the *output* side. This is the input
side: the materials are already sitting in the parent's tree, so they are copied
rather than rebuilt.

**No command runs here, and that is the design.** The tool this is read off
(`sources/wtp`) has a third hook type that shells out — `npm ci`, `make
db:setup` — and the reasoning against it is lifespan, not danger. A person's
branch amortizes a thirty-second install over weeks; an agent's worktree may
live for one turn, and in a fan-out that install is paid per child, before each
child's first step. It also converts a local, deterministic, offline operation
into a network-dependent one: `git worktree add` cannot fail because a registry
is down, and an install can. Two properties fall out of refusing it. There is no
provisioning process, so F1's scrub of `*KEY*`/`*SECRET*`/`*TOKEN*` needs no
exception. And a *repo-discovered* list is data — a `copy` entry cannot execute
— so cloning a repository and starting pH runs nothing that repository authored.

**Nothing outside the tree, in either direction, ever.** Every path is relative,
resolves against its own base (`source` against the parent's tree, `dest`
against the new root), and is re-checked *after* resolution, which is what makes
symlinked traversal fail rather than escape. wtp does the opposite and its code
is explicit about it — `ensureWithinBase` is called only
`if !filepath.IsAbs(hook.From)` — so `from: ~/.ssh/id_rsa` is a supported mode
there. That is coherent for a tool whose config is the developer's own and whose
threat is a typo; it is not available to a harness whose premise is that the
thing reading the repository may be adversarial. The symlink case is why the
rule is absolute rather than advisory: a symlink *inside* the writable root
pointing outside it is the classic escape, and under a sandbox it would fail
only because the backend catches a hole this layer dug.

@module ph.seams.workspace_provision
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

import anyio
from pydantic import field_validator

from ..wire import WireModel

__all__ = [
    "ProvisionEntry",
    "ProvisionMode",
    "ProvisionRefused",
    "ProvisionReport",
    "provision",
    "resolve_entry",
]

log = logging.getLogger("ph.seams.workspace_provision")

ProvisionMode: TypeAlias = Literal["copy", "hardlink", "symlink"]
"""How a material reaches the new workspace. Three costs, three guarantees.

`copy` is **isolated** and the default. `hardlink` is near-free and safe for the
replace-and-rename pattern every package manager uses — pnpm's whole store is a
hardlink farm — but a build step that writes *in place* (a native addon,
`patch-package`) mutates the parent's copy through the shared inode, so it is
opt-in with that named. `symlink` is instant and zero-space and **shared and
mutable**: a child writing through it reaches the parent's tree, handing back some
of the collision isolation the tier exists to buy.

What each one costs: `tests/test_workspace_provision.py`.
"""

_FICLONE = 0x40049409
"""Linux `FICLONE`: make `dst` a copy-on-write clone of `src`.

Isolation at symlink cost, where the filesystem has it (btrfs, XFS, and OpenZFS
only with block cloning enabled). Tried first and allowed to fail — an `ioctl`
is cheaper than asking whether it will work, and every failure mode is the same
answer: fall back to a real copy.
"""

_GIT_DIR = ".git"
"""Refused in both directions, and not merely as tidiness: a `dest` under `.git`
is a planted hook, i.e. code that runs on the agent's next commit — the exact
execution this module exists to not have."""


class ProvisionRefused(Exception):
    """An entry was rejected before anything was touched.

    Refusal is per *entry* and never fatal: a workspace with a missing `.env` is
    worse than one without, but a harness that will not start because one line of
    a config is wrong is worse than both.
    """


class ProvisionEntry(WireModel):
    """One material, and how it should arrive.

    `source`/`dest` where `wtp` says `from`/`to`. Not a preference: this repo
    pins every wire alias to `wire_alias(field_name)` (Q2 — "declare aliases,
    never derive them"), and an explicit `alias="from"` would be the one field in
    the tree with a name the convention does not generate. `from` is also a
    Python keyword, so the Python side could never have matched it anyway.
    """

    source: str
    """Relative to the *base* — the tree the workspace was branched from, which
    for a child is its parent's. Never absolute."""
    dest: str | None = None
    """Relative to the new workspace root. Defaults to `source`, which is what it
    is nearly always wanted to be."""
    mode: ProvisionMode = "copy"
    optional: bool = True
    """A source that is not there is usually a project that does not have one —
    no `.env` in a fresh clone — so the default is to say so and carry on. Set
    `false` for a material the agent genuinely cannot work without."""

    @field_validator("source", "dest")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        """Refuse an absolute path *here*, where the operator finds out.

        Half of the guard needs no `base` and no `root`, and running that half at
        use-time sent an operator's YAML typo down the failure channel to the
        **model's prompt line**, once per agent, forever — while the person who
        wrote it saw nothing. `provision_failures` reaching the agent is right
        for "this project has no `.env`" and wrong for "your config is
        malformed". So the syntactic half fires at profile-config load and
        inside `discover_provisioning`'s `model_validate`, which is the same
        enforcement point `extra="forbid"` already uses to make the data-only
        claim real rather than aspirational.
        """
        if value is None:
            return None
        if not value or value.startswith("~") or Path(value).is_absolute():
            raise ValueError(
                f"{value!r} must be relative to the workspace; provisioning may not "
                "name anything outside it"
            )
        return value

    @property
    def target(self) -> str:
        """Where this lands, named once — `dest` is nearly always `source`."""
        return self.dest or self.source


@dataclass(frozen=True, slots=True)
class ProvisionReport:
    """What arrived, and what did not.

    `failed` is the load-bearing half: it rides `workspace/provisioned` and
    reaches the agent's own prompt line, because the one party that has to know
    `.env` is missing is the agent about to wonder why the tests fail.
    """

    provisioned: tuple[str, ...] = ()
    """Every path that now exists in the workspace because we put it there.

        Carried to disposal, where it is what keeps this row from defeating the
        tier it serves: a copied `.env` the project does not gitignore is an
        untracked file, so `git status` calls the tree dirty, so the keep-dirty
        policy keeps it — and "remove a clean worktree" stops meaning anything.

        **Not** solved with `info/exclude`, and that is worth writing down
        because it is the obvious wrong answer. Git resolves `info/exclude`
        against the *common* directory even from inside a linked worktree
        (`git rev-parse --git-path info/exclude` proves it), so there is no
        per-worktree exclude file to write: the only one that works is the
        repository's own, which would hide these paths in every other worktree
        and in the user's own tree. The provisioned list rides to `_dirty`
    instead and becomes `:(exclude)` pathspecs there, so the rule reaches exactly
    the one decision it is about and mutates nothing."""
    failed: tuple[str, ...] = ()


def resolve_entry(entry: ProvisionEntry, *, base: Path, root: Path) -> tuple[Path, Path]:
    """`(source, dest)` as absolute paths, or `ProvisionRefused`.

    Only the *contextual* half of the guard lives here — whether these two
    relative paths, resolved, are still inside the trees they belong to. The
    syntactic half (absolute, `~`, empty) is a `ProvisionEntry` validator, so it
    fires where the operator can see it rather than once per agent forever.

    Resolution is `Path.resolve()` — symlinks followed — and the containment
    check happens on the *result*. Checking the joined-but-unresolved path would
    accept `link/passwd` wherever `link` points outside, which is the whole
    escape.
    """
    source_path = _contained(base, entry.source, label="source")
    dest_path = _contained(root, entry.target, label="dest")
    if source_path == dest_path:
        raise ProvisionRefused(f"source and dest are the same path: {source_path}")
    return source_path, dest_path


def _contained(base: Path, relative: str, *, label: str) -> Path:
    """Resolve `relative` against `base` and refuse anything that leaves it.

    `label` is the *wire* key (`source`/`dest`), so a person can grep the message
    against the line of YAML that caused it.
    """
    root = base.resolve()
    resolved = (base / relative).resolve()
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        raise ProvisionRefused(
            f"{label}: {relative!r} resolves to {resolved}, outside {root}"
        ) from None
    if _GIT_DIR in parts:
        raise ProvisionRefused(
            f"{label}: {relative!r} reaches into {_GIT_DIR}/, which is git's own state "
            "(a planted hook is code that runs on the next commit)"
        )
    return resolved


async def provision(
    entries: Sequence[ProvisionEntry], *, base: Path, root: Path
) -> ProvisionReport:
    """Put every material in place, and report rather than raise.

    Off the event loop wholesale: a `node_modules` walk is thousands of syscalls
    and the loop has an agent waiting on it.
    """
    if not entries:
        return ProvisionReport()
    return await anyio.to_thread.run_sync(lambda: _provision_sync(entries, base, root))


def _provision_sync(entries: Sequence[ProvisionEntry], base: Path, root: Path) -> ProvisionReport:
    provisioned: list[str] = []
    failed: list[str] = []
    clone = _Cloner()
    for entry in entries:
        try:
            source, dest = resolve_entry(entry, base=base, root=root)
            if not source.exists():
                if not entry.optional:
                    failed.append(f"{entry.target}: {entry.source} is not in the project")
                continue
            _materialize(source, dest, entry.mode, clone)
        except ProvisionRefused as refusal:
            failed.append(f"{entry.target}: {refusal}")
            log.warning("ph.seams.workspace_provision: refused %s (%s)", entry.target, refusal)
        except OSError as error:
            failed.append(f"{entry.target}: {error.strerror or error}")
            log.warning("ph.seams.workspace_provision: %s failed", entry.target, exc_info=True)
        else:
            provisioned.append(entry.target)
    return ProvisionReport(provisioned=tuple(provisioned), failed=tuple(failed))


def _materialize(source: Path, dest: Path, mode: ProvisionMode, clone: _Cloner) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        # Refusing to clobber rather than replacing: a destination that already
        # exists is a checked-in file, and quietly turning it into a link to
        # somewhere else is not something a config line should be able to do.
        if dest.exists() or dest.is_symlink():
            raise ProvisionRefused(f"{dest} already exists")
        dest.symlink_to(source, target_is_directory=source.is_dir())
        return
    if dest.exists() or dest.is_symlink():
        _remove(dest)
    copy = os.link if mode == "hardlink" else clone.copy
    if source.is_dir():
        # `shutil.copytree`, not a hand-rolled walk. `symlinks=True` is the
        # load-bearing argument: `os.walk` puts a symlinked *directory* in
        # `dirs` and never descends, so a walk that only recreates entries from
        # `files` drops it entirely — which is most of a pnpm `node_modules`
        # and every `.venv/lib64 -> lib`. `copy_function` is also exactly the
        # per-file dispatch the walk was hand-rolled for, and `scandir` answers
        # "is this a symlink" from the directory entry instead of a fresh stat
        # per file. `shutil.Error` subclasses `OSError`, so the caller's handler
        # is unchanged.
        shutil.copytree(source, dest, symlinks=True, copy_function=copy, dirs_exist_ok=True)
    else:
        copy(source, dest)


@dataclass(slots=True)
class _Cloner:
    """Copy-on-write where the filesystem has it, asked **once** per pair of them.

    `FICLONE` costs an open, an open, a failing `ioctl`, an unlink and two closes
    *per file* on a filesystem that cannot clone — measured at +50% wall time
    over a 20 000-file tree, paid per child in a fan-out. The answer never varies
    within one device pair, so it is latched; and a source and destination on
    different devices can only ever return `EXDEV`, which two `stat` calls settle
    without attempting anything.
    """

    _supported: dict[tuple[int, int], bool] = field(default_factory=dict)

    def copy(self, source: str | Path, dest: str | Path) -> None:
        src, dst = Path(source), Path(dest)
        if self._can_clone(src, dst) and self._clone(src, dst):
            return
        shutil.copy2(src, dst, follow_symlinks=False)

    def _can_clone(self, source: Path, dest: Path) -> bool:
        if not hasattr(fcntl, "ioctl"):  # pragma: no cover - Windows
            return False
        try:
            pair = (source.stat().st_dev, dest.parent.stat().st_dev)
        except OSError:
            return False
        return self._supported.get(pair, True)

    def _clone(self, source: Path, dest: Path) -> bool:
        """One `FICLONE` attempt; `False` means "fall back", and latches why."""
        try:
            pair = (source.stat().st_dev, dest.parent.stat().st_dev)
        except OSError:
            return False
        src_fd = os.open(source, os.O_RDONLY)
        try:
            dst_fd = os.open(dest, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                fcntl.ioctl(dst_fd, _FICLONE, src_fd)
            except OSError:
                self._supported[pair] = False
                dest.unlink(missing_ok=True)
                return False
            finally:
                os.close(dst_fd)
        finally:
            os.close(src_fd)
        self._supported[pair] = True
        shutil.copystat(source, dest)
        return True


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
