"""Where a lineage's logs sit on disk: `<root>/<family>/<name><suffix>`.

**One statement of the layout, for backends that agree about nothing else.** JSONL
and Turso disagree about how a log is *encoded* and must not disagree about where
it *is*. Two implementations whose docstrings have to assert they agree ("JSONL's
rule exactly") are not a mechanism — what they had already drifted on is recorded
in `tests/test_persistence_backends.py`.

Suffix-parameterised rather than shared by inheritance, because that is the only
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


def family_dirs(root: Path) -> list[str]:
    """The lineage directories under `root`, or nothing if it cannot be read.

    A missing or unreadable sessions root is an empty store, not an error: every
    caller here is answering "what is on record", and a listing that raised would
    turn an empty deployment into a crash.
    """
    try:
        with os.scandir(root) as entries:
            return [entry.path for entry in entries if entry.is_dir()]
    except OSError:
        return []


def logs_under(root: Path, suffix: str) -> list[tuple[Path, os.stat_result]]:
    """Every stored log, **newest first**, one family directory at a time.

    One level deep and no deeper: a family is flat inside, so this is not a walk
    and cannot wander into a workspace someone parked in the sessions root.

    Sorted here because all three callers sorted it identically the moment they
    got it, and a helper that hands back an order nobody wants is a helper that
    is really two.
    """
    found: list[tuple[Path, os.stat_result]] = []
    for family in family_dirs(root):
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
    determine a path, so a read that has only an id has to look: a root is
    answered in one `stat` — its family is its own id, so it names its own
    directory — and anything else falls through to a scan that is O(families).

    That scan is worth avoiding, and callers holding a family avoid it entirely
    through `path_under`. Measured at 200 families: 5.6 us for the root fast
    path, 543 us for a scan that finds, 1.14 ms for one that does not.
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
