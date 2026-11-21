"""Make a fresh workspace *usable*: copy the materials, never run an installer (E14).

`git worktree add` carries no ignored file. A child that is given its own
checkout has no `.env`, no `node_modules`, no `.venv` and no `vendor/`. So "run
the tests" fails for a reason that the model cannot see, and the model reads the
failure as its own bug. The redirection env of E12 prevents the same failure
from the *output* side. This module is the input side: the materials are
already in the tree of the parent, so they are copied and not rebuilt.

**No command runs here, and that is the design.** The tool that this design is
taken from (`sources/wtp`) has a third hook type that runs a shell command, such
as `npm ci` or `make db:setup`. The argument against it is lifespan, not danger.
The branch of a person amortizes a thirty-second install over weeks. The
worktree of an agent may live for one turn, and in a fan-out each child pays
for that install before its first step. An install also turns a local,
deterministic, offline operation into one that depends on the network: `git
worktree add` cannot fail because a registry is down, and an install can. Two
properties follow from the refusal. There is no provisioning process, so the
scrub of `*KEY*`/`*SECRET*`/`*TOKEN*` in F1 needs no exception. And a list that
is *discovered in the repository* is data, because a `copy` entry cannot
execute. So to clone a repository and start pH runs nothing that the repository
authored.

**Nothing outside the tree, in either direction, ever.** Every path is relative
and resolves against its own base: `source` against the tree of the parent, and
`dest` against the new root. Each path is checked again *after* resolution,
which makes a traversal through a symlink fail and not escape. wtp does the
opposite, and its code is explicit about it: `ensureWithinBase` is called only
`if !filepath.IsAbs(hook.From)`, so `from: ~/.ssh/id_rsa` is a supported mode
there. That is coherent for a tool whose config belongs to the developer and
whose threat is a typo. It is not available to a harness whose premise is that
the thing reading the repository may be adversarial. The symlink case is why
the rule is absolute and not advisory. A symlink *inside* the writable root
that points outside it is the classic escape. Under a sandbox it would fail
only because the backend catches a hole that this layer made.

@module ph.seams.workspace_provision
"""

from __future__ import annotations

import fcntl
import logging
import os
import shutil
import threading
from collections.abc import Callable, Sequence
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
    "ProvisionStop",
    "provision",
    "resolve_entry",
]

log = logging.getLogger("ph.seams.workspace_provision")

ProvisionMode: TypeAlias = Literal["copy", "hardlink", "symlink"]
"""How a material reaches the new workspace. Three modes, each with its cost and guarantee.

`copy` is **isolated** and is the default. `hardlink` is nearly free. It is safe
for the replace-and-rename pattern that every package manager uses (the whole
store of pnpm is a hardlink farm). But a build step that writes *in place* (a
native addon, `patch-package`) changes the copy of the parent through the shared
inode. So `hardlink` is opt-in, with that risk named. `symlink` is instant and
takes no space, and it is **shared and mutable**: a child that writes through it
reaches the tree of the parent. That gives back some of the collision isolation
that the tier exists to provide.
"""

_FICLONE = 0x40049409
"""Linux `FICLONE`: make `dst` a copy-on-write clone of `src`.

This gives isolation at the cost of a symlink, where the filesystem supports it
(btrfs, XFS, and OpenZFS only with block cloning enabled). It is tried first
and allowed to fail. An `ioctl` is cheaper than a check of whether it will
work, and every failure mode has the same answer: fall back to a real copy.
"""

_GIT_DIR = ".git"
"""Refused in both directions, and not only for tidiness. A `dest` under `.git`
is a planted hook: code that runs on the next commit of the agent. That is
exactly the execution that this module exists to prevent."""


class ProvisionRefused(Exception):
    """An entry was rejected before anything was touched.

    A refusal applies to one *entry* and is never fatal. A workspace that lacks
    its `.env` is worse than one that has it, but a harness that does not start
    because one line of a config is wrong is worse than both.
    """


class ProvisionEntry(WireModel):
    """One material, and how it arrives.

    The fields are `source` and `dest` where `wtp` says `from` and `to`. This is
    not a preference. This repo holds every wire alias equal to
    `wire_alias(field_name)`, because the rule of Q2 is to declare aliases and
    never derive them. An explicit `alias="from"` would be the one field in the
    tree with a name that the convention does not generate. `from` is also a
    Python keyword, so the Python side could never have matched it.
    """

    source: str
    """A path relative to the *base*: the tree that the workspace was branched
    from, which for a child is the tree of its parent. Never absolute."""
    dest: str | None = None
    """A path relative to the new workspace root. It defaults to `source`, which
    is nearly always the wanted value."""
    mode: ProvisionMode = "copy"
    optional: bool = True
    """Whether a missing source is skipped without a failure. A source that is
    not there usually means a project that does not have one (no `.env` in a
    fresh clone), so the default is to skip it and continue. Set `false` for a
    material that the agent cannot work without."""

    @field_validator("source", "dest")
    @classmethod
    def _relative(cls, value: str | None) -> str | None:
        """Refuse an absolute path *here*, where the operator sees the refusal.

        Half of the guard needs no `base` and no `root`. When that half ran at
        the time of use, a typo in the YAML of an operator went down the failure
        channel to the **prompt line of the model**, one time per agent,
        forever. The person who wrote the typo saw nothing. It is right that
        `provision_failures` reaches the agent for "this project has no `.env`",
        and wrong for "your config is malformed". So the syntactic half runs
        when the profile config is loaded, and inside the `model_validate` of
        `discover_provisioning`. That is the same enforcement point that
        `extra="forbid"` already uses to make the data-only claim real and not
        only intended.
        """
        if value is None:
            return None
        # A `..` that climbs out is the same typo in relative form, and it can be
        # answered without a tree as well: `a/../..` leaves whatever it is joined
        # to. What still needs `base` and `root` is a *symlink* that leads out,
        # and `_contained` catches that.
        climbs_out = Path(os.path.normpath(value)).parts[:1] == ("..",)
        if not value or value.startswith("~") or Path(value).is_absolute() or climbs_out:
            raise ValueError(
                f"{value!r} must be relative to the workspace; provisioning may not "
                "name anything outside it"
            )
        return value

    @property
    def target(self) -> str:
        """Where this entry lands, named in one place. `dest` is nearly always `source`."""
        return self.dest or self.source


@dataclass(frozen=True, slots=True)
class ProvisionReport:
    """What arrived, and what did not.

    `failed` is the half that matters most. It is carried on
    `workspace/provisioned` and reaches the own prompt line of the agent. The
    one party that has to know that `.env` is missing is the agent that is
    about to wonder why the tests fail.
    """

    provisioned: tuple[str, ...] = ()
    """Every path that now exists in the workspace because the seam put it there.

    The list is carried to disposal. There it keeps this row from defeating the
    tier that it serves. Because disposal commits, it also keeps a copied secret
    out of git. A `.env` that the project does not gitignore is an untracked
    file: `git status` calls the tree dirty, and `git add -A` would put the file
    on the branch that somebody merges.

    This is **not** solved with `info/exclude`. That is worth writing down,
    because it is the obvious wrong answer. git resolves `info/exclude` against
    the *common* directory even from inside a linked worktree (`git rev-parse
    --git-path info/exclude` proves it). So there is no per-worktree
    `info/exclude` to write. The only one that works is the file of the
    repository, which would hide these paths in every other worktree and in the
    own tree of the user. The provisioned list goes to `_dirty` instead and
    becomes `:(exclude)` pathspecs there. The rule then reaches exactly the one
    decision that it is about, and it mutates nothing. Staging reads the same
    list through a per-workspace `core.excludesFile`
    (`workspace_git._excludes`)."""
    failed: tuple[str, ...] = ()


def resolve_entry(entry: ProvisionEntry, *, base: Path, root: Path) -> tuple[Path, Path]:
    """Return `(source, dest)` as absolute paths, or raise `ProvisionRefused`.

    Only the *contextual* half of the guard is here: whether these two relative
    paths, when resolved, are still inside the trees that they belong to. The
    syntactic half (absolute, `~`, empty, or a `..` that climbs out) is a
    `ProvisionEntry` validator. It therefore runs where the operator can see it,
    and not one time per agent forever.

    Resolution is `Path.resolve()`, which follows symlinks, and the containment
    check is made on the *result*. A check of the joined but unresolved path
    would accept `link/passwd` wherever `link` points outside, and that is the
    whole escape.
    """
    source_path = _contained(base, entry.source, label="source")
    dest_path = _contained(root, entry.target, label="dest", leaf=False)
    if source_path == dest_path:
        raise ProvisionRefused(f"source and dest are the same path: {source_path}")
    return source_path, dest_path


def _contained(base: Path, relative: str, *, label: str, leaf: bool = True) -> Path:
    """Resolve `relative` against `base` and refuse anything that leaves it.

    `label` is the *wire* key (`source`/`dest`), so a person can grep the
    message against the line of YAML that caused it.

    `leaf=False` is for a destination. Its parent is resolved and its own name
    is kept, because the question is where the entry is, not where a link that
    is already there points. When the leaf was followed, a link that
    provisioning had made read as outside the tree, and every later pass
    reported it as a failure. Nothing is written through a link at the leaf:
    `_materialize` replaces it or leaves it alone.
    """
    root = base.resolve()
    joined = base / relative
    resolved = (
        joined.resolve() if leaf or not joined.name else joined.parent.resolve() / joined.name
    )
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


ProvisionStop: TypeAlias = threading.Event
"""Asks a copy that is under way to stop before its next file (W1). It is set
from the event loop and read on the own thread of the copy, so it is a
thread-safe flag and not an `anyio.Event`."""


class _Stopped(Exception):
    """Raised on the thread of the copy when its `ProvisionStop` is set.
    `provision` ends there and reports what it finished."""


async def provision(
    entries: Sequence[ProvisionEntry],
    *,
    base: Path,
    root: Path,
    fill: bool = False,
    stop: ProvisionStop | None = None,
) -> ProvisionReport:
    """Put every material in place. Report a failure, and do not raise it.

    The whole job runs off the event loop. A walk of `node_modules` is thousands
    of syscalls, and the loop has an agent waiting on it.

    `fill` is for a tree that the tier reused (`Workspace.reused`). Only what is
    missing is put there, down to a file inside a copied directory, and nothing
    present is replaced. The tree was provisioned when it was made, or part-way
    by a process that crashed during the copy, and the agent may have edited it
    since.

    `stop` ends the copy before its next file after it is set (W1). The thread
    cannot be canceled. So a release that must not take the tree from under the
    copy asks it to stop here, and waits at most one file for it to return. What
    the copy did not reach is in neither half of the report.
    """
    if not entries:
        return ProvisionReport()
    return await anyio.to_thread.run_sync(lambda: _provision_sync(entries, base, root, fill, stop))


def _provision_sync(
    entries: Sequence[ProvisionEntry],
    base: Path,
    root: Path,
    fill: bool,
    stop: ProvisionStop | None = None,
) -> ProvisionReport:
    provisioned: list[str] = []
    failed: list[str] = []
    clone = _Cloner()
    for entry in entries:
        if stop is not None and stop.is_set():
            break
        try:
            source, dest = resolve_entry(entry, base=base, root=root)
            if not source.exists():
                if not entry.optional:
                    failed.append(f"{entry.target}: {entry.source} is not in the project")
                continue
            _materialize(source, dest, entry.mode, clone, fill=fill, stop=stop)
        except _Stopped:
            break
        except ProvisionRefused as refusal:
            failed.append(f"{entry.target}: {refusal}")
            log.warning("ph.seams.workspace_provision: refused %s (%s)", entry.target, refusal)
        except OSError as error:
            failed.append(f"{entry.target}: {error.strerror or error}")
            log.warning("ph.seams.workspace_provision: %s failed", entry.target, exc_info=True)
        else:
            provisioned.append(entry.target)
    return ProvisionReport(provisioned=tuple(provisioned), failed=tuple(failed))


def _stoppable(
    copy: Callable[[str | Path, str | Path], None], stop: ProvisionStop
) -> Callable[[str | Path, str | Path], None]:
    """Wrap `copy` so that it refuses to start a file after `stop` is set.
    `copytree` calls it per file, so a stop takes effect between two files and
    not after the whole tree."""

    def checked(source: str | Path, dest: str | Path) -> None:
        if stop.is_set():
            raise _Stopped
        copy(source, dest)

    return checked


def _materialize(
    source: Path,
    dest: Path,
    mode: ProvisionMode,
    clone: _Cloner,
    *,
    fill: bool,
    stop: ProvisionStop | None = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        # A link already to this source is in place, whoever made it.
        if dest.is_symlink() and Path(os.readlink(dest)) == source:
            return
        # Refuse to clobber, and do not replace. A destination that already
        # exists is a checked-in file. A config line must not be able to turn it
        # silently into a link to somewhere else.
        if dest.exists() or dest.is_symlink():
            raise ProvisionRefused(f"{dest} already exists")
        dest.symlink_to(source, target_is_directory=source.is_dir())
        return
    if dest.exists() or dest.is_symlink():
        if not fill:
            _remove(dest)
        elif not (_real_dir(source) and _real_dir(dest)):
            return
    copy: Callable[[str | Path, str | Path], None] = os.link if mode == "hardlink" else clone.copy
    if stop is not None:
        copy = _stoppable(copy, stop)
    if source.is_dir():
        # `shutil.copytree`, not a hand-written walk. `symlinks=True` is the
        # argument that matters. `os.walk` puts a symlinked *directory* in
        # `dirs` and never descends, so a walk that recreates entries only from
        # `files` drops it entirely. That is most of a pnpm `node_modules` and
        # every `.venv/lib64 -> lib`. `copy_function` is also exactly the
        # per-file dispatch that the walk was written for. And `scandir` answers
        # "is this a symlink" from the directory entry, without a fresh stat per
        # file. `shutil.Error` subclasses `OSError`, so the handler of the
        # caller is unchanged.
        shutil.copytree(
            source,
            dest,
            symlinks=True,
            copy_function=copy,
            dirs_exist_ok=True,
            ignore=_present(source, dest) if fill else None,
        )
    else:
        copy(source, dest)


def _real_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _present(source: Path, dest: Path) -> Callable[[str, list[str]], set[str]]:
    """The `ignore` of `copytree` for a fill: every name that is already at its
    destination. The exception is a directory on both sides, which is descended
    into for what it lacks."""

    def ignore(directory: str, names: list[str]) -> set[str]:
        here = Path(directory)
        there = dest / here.relative_to(source)
        return {
            name
            for name in names
            if os.path.lexists(there / name)
            and not (_real_dir(here / name) and _real_dir(there / name))
        }

    return ignore


@dataclass(slots=True)
class _Cloner:
    """Copy-on-write where the filesystem has it, asked **one time** per pair of devices.

    On a filesystem that cannot clone, `FICLONE` costs two opens, a failing
    `ioctl`, an unlink and two closes *per file*, and a fan-out pays that per
    child. The answer never varies within one device pair, so it is latched. A
    source and a destination on different devices can only return `EXDEV`.
    After the first attempt has latched that answer, two `stat` calls settle
    the pair without another attempt.
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
        """One `FICLONE` attempt. `False` means "fall back", and the answer is latched."""
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
