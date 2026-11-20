"""Caches kept per workspace, and the sweep that removes a cache whose workspace is gone.

The code graph and the text index each keep one index for each workspace tree.
The index is under `$PH_CACHE`, in a directory named by a digest of the path of
the tree. A worktree is a workspace of its own, so every agent that ran in a
worktree left an index behind. Nothing removed those indexes. A digest cannot be
turned back into its path, so nothing could tell the index of a live tree from
the index of a dead one.

`use` therefore records in each entry the workspace that the entry serves
(`WORKSPACE_FILE`), dated by its last use. At most one time in each
`SWEEP_EVERY`, it removes the entries whose workspace no longer exists and which
nothing has used for `STALE_AFTER`. **The age protects a tree that is absent
only for now**, such as an unmounted drive or a checkout that is being moved.
Without the age, the index of such a tree would cost a full re-index when the
tree came back. An entry with no record is left alone, because it cannot be told
from an entry that is being made.

The removal of a stale entry loses nothing that cannot be rebuilt, by the
definition of `$PH_CACHE` (Q1): an index is derived from its tree, and the tree
is gone.

@module ph.workspace_caches
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from pathlib import Path
from time import time

from .paths import write_atomic

__all__ = ["STALE_AFTER", "SWEEP_EVERY", "WORKSPACE_FILE", "digest", "use"]

log = logging.getLogger("ph.workspace_caches")

WORKSPACE_FILE = "workspace"
"""The file in an entry that names the workspace it serves, dated by its last use."""

SWEPT_FILE = ".swept"
"""The file under the base of a cache that is dated by the last sweep."""

STALE_AFTER = 7 * 86_400.0
"""How long an entry whose workspace is gone is kept after its last use. A week
is longer than a drive stays unplugged or a checkout stays half-moved. It is
short compared with the gigabytes that a busy worktree profile leaves behind."""

SWEEP_EVERY = 86_400.0
"""How often a cache is swept. A daily sweep against a guard of a week loses
nothing, and it spares every index call a walk of every entry."""


def digest(text: str) -> str:
    """The 16 hex digits that name an entry. A digest is used because a path or a
    model id has slashes in it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def use(base: Path, workspace: Path, *, now: float | None = None) -> Path:
    """Return the entry of `workspace` under `base`, and record it as in use. If
    the cache is due a sweep, also remove the entries of workspaces that are
    gone.

    The record is written one time and dated on every later use. The name of
    the entry is the digest of what the record holds, so a write that finds the
    record there has nothing to change. The function blocks: one or two stat
    calls, and a walk of `base` when a sweep is due.
    """
    entry = base / digest(str(workspace))
    write_atomic(entry / WORKSPACE_FILE, str(workspace), skip_if_present=True, durable=False)
    when = time() if now is None else now
    swept = base / SWEPT_FILE
    if when - _mtime(swept) >= SWEEP_EVERY:
        swept.touch()
        _remove_stale(base, when)
    return entry


def _remove_stale(base: Path, now: float) -> None:
    """Remove every entry under `base` whose workspace is gone and which nothing has
    used for `STALE_AFTER`. It never removes the entry of the caller, which `use`
    has just dated."""
    removed = 0
    for entry in base.iterdir():
        marker = entry / WORKSPACE_FILE
        if now - _mtime(marker) < STALE_AFTER:
            continue
        try:
            recorded = marker.read_text(encoding="utf-8")
        except OSError:
            continue
        if recorded and not Path(recorded).exists():
            shutil.rmtree(entry, ignore_errors=True)
            removed += not entry.exists()
    if removed:
        log.info("ph.workspace_caches: removed %d index(es) under %s", removed, base)


def _mtime(path: Path) -> float:
    """When `path` was last written, or the epoch when it does not exist. For a
    sweep, the epoch means long ago. For an entry, it means no use at all:
    `_remove_stale` then fails to read the marker of that entry and leaves the
    entry alone."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
