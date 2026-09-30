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
from collections.abc import Iterable
from pathlib import Path

from ..session.store import child_session_id

__all__ = [
    "children_under",
    "descendants",
    "family_dirs",
    "locate_under",
    "logs_under",
    "path_under",
]


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


def children_under(
    root: Path, family: str, parent_id: str, suffix: str
) -> list[tuple[Path, os.stat_result]]:
    """The logs in one family that **may** be beneath `parent_id`: one `scandir`.

    Two facts narrow the store to this, and neither is the answer:
    - **A child is filed in its parent's family.** `SessionStore.create` inherits
      it, so one directory holds every candidate and no other directory holds any.
    - **A child's id starts with its parent's id and a dash**
      (`<parent>-child-<hex>`), so only the names that do are `stat`ed.

    The prefix only narrows. A grandchild shares it, which is what lets one scan find
    a whole tree, and so can a fork someone named after its source, so the header
    peek each backend makes afterwards decides (`descendants_among`).

    Not `logs_under` with a filter. The listing `stat`s every log in the store
    and then cuts at a limit, and a parent's children fall below that cut once
    the store is big enough (`SURVEY_LIMIT`). That is fine for a picker, but a
    resume ladder, a budget or a cap needs every child. This stays inside one
    directory, so its cost follows the size of the family and not the store.

    An empty `family` is refused. `path_under` would resolve it to the sessions
    root, which holds family directories and no logs, so the caller would get
    "no children" when it should get an error. A family nobody has written yet
    is a directory that does not exist, and that is an empty answer.

    **Only that one is.** `family_dirs` turns any `OSError` into an empty store,
    and a listing can afford that. Here "no children" tells a resume sweep there
    is nothing to settle, so a directory that exists and cannot be read raises.
    A log deleted between the scan and its `stat` is skipped: it is no longer
    anyone's child.
    """
    if not family:
        raise ValueError(f"listing {parent_id!r}'s children needs its family")
    prefix = child_session_id(parent_id, "")
    found: list[tuple[Path, os.stat_result]] = []
    try:
        with os.scandir(root / family) as logs:
            for entry in logs:
                if not (entry.name.startswith(prefix) and entry.name.endswith(suffix)):
                    continue
                try:
                    if entry.is_file():
                        found.append((Path(entry.path), entry.stat()))
                except FileNotFoundError:
                    continue
    except FileNotFoundError:
        return []
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


def descendants(lineage: Iterable[tuple[str, str | None]], agent_id: str) -> list[str]:
    """`agent_id` and everything spawned beneath it, transitively (P6-28).

    **Not `reachable_family`, and the difference is the point.** That answers "who
    may this agent *address*" — the C7 nuclear family, including siblings and the
    parent — which is the right rule for a message. This answers "whose leftovers are
    mine to account for": a sibling's worktree is not this agent's to enumerate,
    still less to collect, and borrowing the messaging rule would widen a filesystem
    question with an answer computed for a different one (I7).

    Transitive where `reachable_family` is one hop: a grandchild that failed is
    evidence its grandparent is the only live party left to look at, because the
    child that spawned it settled too.

    An agent's id is its session's id, so the links are `delegating_parent` — the
    agent that spawned each one — and nothing needs a side index. Not
    `parent_session`: a fork names the log it was cut from there, and a fork's
    trees are its own, not its source's leftovers. **`(id, parent)` pairs rather than
    `Session` objects**, which is what lets the workspace collector and the store's
    `descendants_among` answer this from a *listing* — a family is narrowed before a
    single log is read rather than after all of them are.

    Breadth-first, and cycle-safe by construction: `seen` is tested before descent,
    so a log claiming its own ancestor as a child costs a wasted lookup rather than a
    hang.
    """
    children: dict[str, list[str]] = {}
    known = set()
    for session_id, parent in lineage:
        known.add(session_id)
        if parent:
            children.setdefault(parent, []).append(session_id)
    if agent_id not in known:
        return []
    found = [agent_id]
    seen = {agent_id}
    for current in found:
        for child in children.get(current, ()):
            if child not in seen:
                seen.add(child)
                found.append(child)
    return found
