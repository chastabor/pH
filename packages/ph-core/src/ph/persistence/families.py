"""Where the logs of a lineage sit on disk: `<root>/<family>/<name><suffix>`.

**One statement of the layout, for backends that agree about nothing else.**
JSONL and Turso disagree about how a log is *encoded*. They must not disagree
about where it *is*. Two implementations whose docstrings have to assert that
they follow the same rule are not a mechanism. The two copies had already
drifted before either backend shipped.

The functions take the suffix as a parameter and are not shared by inheritance,
because the suffix is the only difference between the two backends here.

@module ph.persistence.families
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from pathlib import Path

from ..session.session import CWD_TAG_LENGTH
from ..session.store import child_session_id

__all__ = [
    "children_under",
    "descendants",
    "family_dirs",
    "locate_under",
    "logs_under",
    "path_under",
    "stored_under",
]


def path_under(root: Path, family: str, name: str, suffix: str) -> Path:
    """Where a log is **written**. A pure function of what the writer holds."""
    return root / family / f"{name}{suffix}"


def family_dirs(root: Path, *, tag: str = "") -> list[str]:
    """The lineage directories under `root`, or nothing if it cannot be read.

    A sessions root that is missing or unreadable is an empty store, not an
    error. Every caller here answers "what is on record", and a listing that
    raised would turn an empty deployment into a crash.

    `tag` keeps only the lineages that were worked in one directory: those whose
    name starts with the `<cwd-tag>-` that the `family` of a root carries
    (format 1). **A filtered listing becomes cheap here.** The sessions of
    every other repo are skipped here, without a `scandir` of their files,
    without a `stat` on any of them, and without opening one. `""` means every
    lineage, which is what an unfiltered listing asks for.

    A *tagless* directory, which belongs to a session created with no cwd, is
    matched by no tag. That is the correct answer: the session belongs to no
    working directory.

    **A dotted directory is not a lineage**, and this function is where that is
    decided. `lease.LEASES` is `.leases` under this same root, and an unfiltered
    listing collected it as a family. That was harmless only because the suffix
    filter downstream finds no logs in it. Such an accident stops being harmless
    when somebody adds a second dot-directory.
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

    `tag` narrows the result to the lineages of one working directory. See
    `family_dirs`, which does the skipping and explains why it costs nothing for
    each file.

    The scan is one level deep and no deeper. A family is flat inside, so this
    is not a walk, and it cannot enter a workspace that someone left in the
    sessions root.

    The result is sorted here because all three callers sorted it identically as
    soon as they got it. A helper that returns an order that nobody wants is
    really two helpers.
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

    **The cost of the family layout is stated in this one place.** An id alone
    does not determine a path, so a read that has only an id has to look. A
    caller that holds a family skips the search entirely, through `path_under`.

    **The directories of its lineage are tried first.** A family is
    `<cwd-tag>-<root>` or a bare `<root>` (`family_for`). Every id beneath a
    root begins with the id of the root and a dash (`child_session_id`). So the
    name of a directory says whether it holds a root or any of its descendants.
    Those directories are tried first, with one `stat` each. Every other
    directory is tried only after them. A fork is found there, because a fork
    is a new id filed in the family of its source. Before, only a bare root was
    found without that scan. Since format 1 a root with a cwd is not bare, so
    nearly every lookup by id `stat`ed every family.

    **The own path of a bare root is tried before any listing**, and it is not
    tried again in the listing. A root opened with no cwd is filed under its own
    id, and one `stat` finds it without a `scandir` of the store. The print and
    rpc hosts open their roots that way. For a root with a cwd that `stat`
    misses, which costs one `stat`. Dropping it would cost every bare root a
    listing.
    """
    own = path_under(root, name, name, suffix)
    if own.is_file():
        return own
    wanted = f"{name}{suffix}"
    families = [one for one in family_dirs(root) if os.path.basename(one) != name]
    likely = [one for one in families if _holds_lineage_of(os.path.basename(one), name)]
    for family in [*likely, *(one for one in families if one not in likely)]:
        candidate = Path(family) / wanted
        if candidate.is_file():
            return candidate
    return None


def stored_under(root: Path, name: str, suffix: str, family: str | None = None) -> bool:
    """Whether a log for `name` is on disk. When `family` is given, only that family
    is checked, with one `stat`. When it is not, the log is looked for anywhere
    under `root` (`locate_under`). This is `exists` for both backends, which
    differ only in `suffix`."""
    if family:
        return path_under(root, family, name, suffix).is_file()
    return locate_under(root, name, suffix) is not None


def _holds_lineage_of(family: str, name: str) -> bool:
    """Whether the name of a family directory says that it holds the lineage of
    `name`: its root (the name, less a cwd tag) is `name` or the id of an
    ancestor of `name`. This is only an ordering. `locate_under` looks everywhere
    else after."""
    tag, dash, rest = family.partition("-")
    roots = {family, rest} if dash and len(tag) == CWD_TAG_LENGTH else {family}
    return any(name == root or name.startswith(f"{root}-") for root in roots)


def descendants(lineage: Iterable[tuple[str, str | None]], agent_id: str) -> list[str]:
    """`agent_id` and everything spawned beneath it, transitively (P6-28).

    **This is not `reachable_family`, and the difference matters.**
    `reachable_family` answers "who may this agent *address*": the C7 nuclear
    family, with the siblings and the parent. That is the right rule for a
    message. This function answers "whose leftovers must this agent account
    for". The worktree of a sibling is not for this agent to enumerate, and
    still less to collect. The messaging rule would widen a filesystem question
    with an answer that was computed for a different question (I7).

    This function is transitive where `reachable_family` is one hop. A
    grandchild that failed is evidence that its grandparent is the only live
    party left to look at, because the child that spawned it settled too.

    The id of an agent is the id of its session. So the links are
    `delegating_parent`, the agent that spawned each one, and nothing needs a
    side index. The link is not `parent_session`: a fork names there the log
    that it was cut from, and the trees of a fork are its own, not the leftovers
    of its source. **The input is `(id, parent)` pairs and not `Session`
    objects.** So the workspace collector and `protocol.descendants_among` can
    answer this from a *listing*. A family is narrowed before a single log is
    read, and not after all of them are read.

    The walk is breadth-first and cycle-safe by construction. `seen` is tested
    before descent, so a log that claims its own ancestor as a child costs a
    wasted lookup and not a hang.
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
