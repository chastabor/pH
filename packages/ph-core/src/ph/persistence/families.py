"""Where a lineage's logs sit on disk: `<root>/<family>/<name><suffix>`.

**One statement of the layout, for backends that agree about nothing else.** JSONL
and Turso disagree about how a log is *encoded* and must not disagree about where
it *is*. Two implementations whose docstrings have to assert they agree ("JSONL's
rule exactly") are not a mechanism, and the two copies had already drifted before
either shipped.

Suffix-parameterized rather than shared by inheritance, because that is the only
thing the two backends actually differ by here.

@module ph.persistence.families
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["family_dirs", "locate_under", "logs_under", "path_under"]


def path_under(root: Path, family: str, name: str, suffix: str) -> Path:
    """Where a log is **written**. A pure function of what the writer holds."""
    return root / family / f"{name}{suffix}"


def family_dirs(root: Path, *, tag: str = "") -> list[str]:
    """The lineage directories under `root`, or nothing if it cannot be read.

    A missing or unreadable sessions root is an empty store, not an error: every
    caller here is answering "what is on record", and a listing that raised would
    turn an empty deployment into a crash.

    `tag` keeps only the lineages worked in one directory — the `<cwd-tag>-` a
    root's `family` carries (format 1). **This is where a filtered listing gets
    cheap**: a repo's sessions are skipped here, without a `scandir` of their
    files, without a `stat` on any of them, and without opening one. `""` is
    every lineage, which is what an unfiltered listing asks for.

    A *tagless* directory — a session created with no cwd — is not matched by any
    tag, which is the honest answer: it belongs to no working directory.

    **A dotted directory is not a lineage**, and saying so here is what keeps it
    from being one. `lease.LEASES` is `.leases` under this same root, and an
    unfiltered listing collected it as a family — harmless only because the
    suffix filter downstream finds no logs in it, which is the kind of accident
    that stops being one when somebody adds a second dot-directory.
    """
    prefix = f"{tag}-" if tag else ""
    try:
        with os.scandir(root) as entries:
            return [
                entry.path
                for entry in entries
                if entry.is_dir()
                and not entry.name.startswith(".")
                and (not prefix or entry.name.startswith(prefix))
            ]
    except OSError:
        return []


def logs_under(root: Path, suffix: str, *, tag: str = "") -> list[tuple[Path, os.stat_result]]:
    """Every stored log, **newest first**, one family directory at a time.

    `tag` narrows it to one working directory's lineages — see `family_dirs`,
    which is where the skipping happens and why it costs nothing per file.

    One level deep and no deeper: a family is flat inside, so this is not a walk
    and cannot wander into a workspace someone parked in the sessions root.

    Sorted here because all three callers sorted it identically the moment they
    got it, and a helper that hands back an order nobody wants is a helper that
    is really two.
    """
    found: list[tuple[Path, os.stat_result]] = []
    for family in family_dirs(root, tag=tag):
        try:
            with os.scandir(family) as logs:
                found.extend(
                    (Path(entry.path), entry.stat())
                    for entry in logs
                    if entry.name.endswith(suffix) and entry.is_file()
                )
        except OSError:  # pragma: no cover - a directory that vanished mid-scan
            continue
    found.sort(key=lambda pair: pair[1].st_mtime, reverse=True)
    return found


def locate_under(root: Path, name: str, suffix: str) -> Path | None:
    """The log for one id, wherever it sits. `None` if there is none.

    **The cost of the family layout, stated in one place.** An id alone does not
    determine a path, so a read that has only an id has to look: a root is answered in
    one `stat` — its family is its own id, so it names its own directory — and
    anything else falls through to a scan that is O(families).

    That scan is worth avoiding, and callers holding a family avoid it entirely
    through `path_under`.
    """
    own = path_under(root, name, name, suffix)
    if own.is_file():
        return own
    wanted = f"{name}{suffix}"
    for family in family_dirs(root):
        candidate = Path(family) / wanted
        if candidate.is_file():
            return candidate
    return None
