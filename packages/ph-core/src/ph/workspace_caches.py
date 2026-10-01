"""Caches kept per workspace, and the sweep that removes one whose workspace is gone.

The code graph and the text index each keep an index per workspace tree, under
`$PH_CACHE`, in a directory named for a digest of the tree's path. A worktree is a
workspace of its own, so every agent that ran in one left an index behind, and
nothing removed them: a digest cannot be turned back into its path, so nothing
could tell a live tree's index from a dead one's.

So `use` records in each entry the workspace it serves (`WORKSPACE_FILE`), dated
by its last use, and at most once per `SWEEP_EVERY` removes the entries whose
workspace no longer exists and which nothing has used for `STALE_AFTER`. **The age
is the guard for a tree that is only absent for now** — an unmounted drive, a
checkout being moved — whose index would otherwise cost a full re-index the next
time it came back. An entry with no record is left alone: it cannot be told from
one being made.

Removing a stale entry loses nothing that is not rebuildable, by the definition of
`$PH_CACHE` (Q1): an index is derived from its tree, and the tree is gone.

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
"""The file under a cache's base dated by its last sweep."""

STALE_AFTER = 7 * 86_400.0
"""How long an entry whose workspace is gone is kept from its last use. A week:
longer than a drive stays unplugged or a checkout stays half-moved, and short
against the gigabytes a busy worktree profile leaves behind."""

SWEEP_EVERY = 86_400.0
"""How often a cache is swept. Daily, against a week's guard, loses nothing, and
spares every index call a walk of every entry."""


def digest(text: str) -> str:
    """The 16 hex an entry is named by: a path or a model id has slashes in it."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def use(base: Path, workspace: Path, *, now: float | None = None) -> Path:
    """`workspace`'s entry under `base`, recorded as in use, and the entries of
    workspaces that are gone removed if the cache is due a sweep.

    The record is written once and dated on every use after it — the entry's
    name is the digest of what it records, so a write that finds it there has
    nothing to change. Blocking: a stat or two, and a walk of `base` when due.
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
    used for `STALE_AFTER` — never the caller's, which `use` has just dated."""
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
    """When `path` was last written, or the epoch when it does not exist — which
    reads as long ago for a sweep and as no use at all for an entry, whose marker
    `_remove_stale` then fails to read and leaves alone."""
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0
