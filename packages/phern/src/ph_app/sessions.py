"""List stored sessions cheaply enough to open a picker on them.

A session log is append-only and can be large. The picker needs three things:
when the session was touched, what it was about, and how big it is. To get them
by reading every event of every session would make the picker slow exactly where
a user has many sessions to choose between. So this module reads the header line
and stops at the first `user/message`. Both are at the top of the file, and the
first thing that the user typed is the best title that a session has.

**This module is outside `tui/` because the reader moved.** Before P5-14 the
terminal was the harness, so when it listed its logs it read its own files. Now
the daemon holds the logs. The daemon answers `sessions/browse`, and this is the
fold that it answers with. A client reads no session file at all. That makes a
front end possible that is not on the machine of the daemon. It also stops a
client and a daemon from disagreeing about which `$PH_HOME` they meant.

Rule 6, which the move did not change: the fold is **filesystem-shaped**. It
walks `<sessions>/<family>/*.jsonl` and reads the headers. A backend that keeps
sessions in any other form contributes no stored rows. Turso keeps one database
for each session, so the walk finds no `.jsonl` file in its directory. A store
that is not on a filesystem answers `directory()` with `None`.
`SessionPersistence.stored()` is the listing that works for every backend. It
deliberately carries neither `title` nor `size`, because those mean different
things in each backend. A picker without titles is what this module exists to
avoid.

@module ph_app.sessions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from ph.json import as_obj
from ph.persistence import MAX_DEPTH, LineageError, materialize_some
from ph.persistence.jsonl import (
    HEADER_LINE_TYPE,
    family_log,
    locate_session,
    session_logs,
    skim_stored,
)
from ph.session import SessionHeader, SessionKind, cwd_tag
from ph.session_profile import LoggedEnvironment, fold_environment
from ph.text import one_line
from ph.wire import WireModel

from .wire import named, text_of_wire

__all__ = [
    "Owner",
    "RecordedStart",
    "SessionSummary",
    "not_a_root",
    "recorded_environment",
    "recorded_start",
    "session_summaries",
    "stored_on",
]

TITLE_SCAN_LIMIT = 40
"""How many events to look through for a title before giving up. A session that
opens with forty non-user events has no title worth waiting for."""


class SessionSummary(WireModel):
    """Enough about a stored session to choose it from a list.

    This is a `WireModel` because it now crosses a socket. `sessions/browse`
    folds these on the daemon, and `to_wire()` and `model_validate` carry them.
    A field that is added here therefore reaches every front end with no edit
    at either edge (P7-11)."""

    session_id: str
    modified: float | None = None
    """When the log file was last written, in epoch seconds, or `None` for a live
    root that the disk has not seen yet. The log of such a root is still in a
    write buffer, so there is no file to take a time from."""

    size: int | None = None
    """The size of the log file in bytes. It comes from the same `stat` as
    `modified`, and it is `None` in the same case."""

    title: str | None = None
    """The first line that the user typed, or `None` when the top of the log has
    no `user/message`."""

    cwd: str | None = None
    """The working directory that the header recorded, or `None` when it recorded
    none."""

    parent: str | None = None
    """The session that this one was forked from, when the header says so. For
    the log of a sub-agent (`origin`), it is the agent that spawned it."""

    kind: SessionKind | None = None
    """`"fork"` or `"segment"`, or `None` for a root, which has no parent to
    qualify."""

    origin: Literal["subagent"] | None = None
    """`"subagent"` for the log of a sub-agent, as its header says (P11-08).

    `parent` alone cannot tell a child from a fork, because both name the
    session that they came from. A fork is a session of its own that a user can
    continue. The log of a child is written by the mount of its root and is
    never attached. So the picker must know which of the two a row is before it
    offers the row.
    """

    state: str = "stored"
    """Whether a daemon holds this session now. A stored session is `stored`. A
    session that a daemon holds carries the status of its root, such as
    `running` (`daemon.projections.browse_of` sets it).

    The distinction is new with P5-14. Before P5-14, the only session that could
    be live was the one that this terminal hosted. To pick a running session is
    to join a turn that may be in flight somewhere else, and the row should say
    so.

    **A stored session has one state, and not two.** A second state,
    `passivated`, for a root that the sweep released, was planned and dropped.
    To tell it from `stored`, the fold would read the *tail* of every log for a
    `supervisor/passivated` record, and this module exists because reading
    whole logs makes the picker slow where a user has many. The user then does
    nothing differently: picking either one mounts the session from its log. So
    the cost would have paid for a label and for no decision.
    """

    family: str | None = None
    """The lineage directory that this log is in, or `None` when no header was
    read. Every ancestor is a sibling in that directory.

    It is read from the same one-line header peek as `parent`. It turns the
    ancestor walk below from a search of the whole store into an open.
    """

    @property
    def when(self) -> str | None:
        """`modified` as local time, or `None` when the disk has not seen the log."""
        if self.modified is None:
            return None
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")


def session_summaries(
    sessions_dir: Path, *, limit: int = 50, cwd: str | None = None
) -> list[SessionSummary]:
    """Stored sessions, most recently touched first. Unreadable files are skipped.

    `cwd` keeps only the sessions whose header recorded that working directory.
    `None` keeps every session.

    There is one `stat` for each file: the `stat` of the directory entry, which
    is used for both the sort and the summary. `session_logs` walks one level of
    family directories, which is where every log is, and returns the logs newest
    first.

    **`cwd` filters during the scan and not after it.** That difference is the
    reason why `cwd` is a parameter here and not a comprehension at the call
    site. `limit` takes the newest rows. A filter on the result would let the
    sessions of a directory fall out of that window behind newer work elsewhere,
    so a repo that somebody last touched a month ago would list as empty on a
    busy machine. With the filter first, the limit counts *matching* sessions.

    **The cost was measured, because it is on the path to the first prompt.** A
    filtered scan reads headers until it has `limit` matches. It does not stop
    at `limit` files. So for a directory with fewer than `limit` sessions, which
    is nearly every directory, it reads the header of every log that the walk
    returns. The cwd tag narrows that walk to the lineages of one working
    directory (see the comment below). Over 500 logs the scan is 500 header
    reads against 50. `_summarize` returns for a log that does not match as early as
    it can, before the title scan, which is the expensive half. That keeps each
    such log to about 12 µs: 8.6 ms → 5.8 ms at 500 logs, and 88 ms → 62 ms at
    5 000.

    The bound is therefore every log that the walk returns, and the store has no
    retention: every fork and every compaction segment is another file. If that
    number ever stops being a few hundred, the answer is an index for each
    directory and not a faster scan. A scan *cap* is the one thing that the
    answer must not be, because "a repo touched a month ago still lists" is the
    property that this filter exists to provide.

    **Not enforced (§5 rule 6): `cwd` matches the string that a header recorded,
    and not the directory that the string names.** Nothing is resolved or
    canonicalized on either side. One checkout that is reached through two paths
    (a symlink, `/tmp` against `/private/tmp`, a mount under another name)
    therefore lists as two directories with two sets of sessions. To resolve
    the paths would be the wrong fix, and not a missing one. The header records
    where a session *said* it was working, and a picker that silently merged two
    paths would show sessions whose workspaces were provisioned somewhere else.
    `/sessions` lists the store without the filter, and that is how the user
    finds the other set.
    """
    # **The tag filters whole lineages before a single file is touched.** See
    # `family_dirs`. What reaches the loop below is already the work of this
    # directory, so the header check in `_summarize` confirms a 24-bit match and
    # does not sift the store. The walk spells "every lineage" as a tag that is
    # left out.
    found = (
        session_logs(sessions_dir) if cwd is None else session_logs(sessions_dir, tag=cwd_tag(cwd))
    )
    summaries: list[SessionSummary] = []
    for path, stat in found:
        if len(summaries) >= limit:
            break
        summary = _summarize(path, stat.st_mtime, stat.st_size, cwd=cwd)
        if summary is not None:
            summaries.append(summary)
    # A child that was forked by reference holds only its own events. The
    # `user/message` that names the conversation is in an ancestor, so the row
    # of the child would render blank. That is the "hex ids with nothing
    # failing" outcome that the comment in `StoredSession` refuses. This runs
    # after the scan, so that an ancestor that was already read here is not read
    # twice. That keeps the claim above of one `stat` for each file true for
    # every session that has its own title.
    known = {summary.session_id: summary for summary in summaries}
    for index, summary in enumerate(summaries):
        if summary.title is None and summary.parent is not None:
            inherited = _inherited_title(sessions_dir, summary, known)
            if inherited is not None:
                summaries[index] = summary.model_copy(update={"title": inherited})
    return summaries


def _inherited_title(
    sessions_dir: Path, of: SessionSummary, known: dict[str, SessionSummary]
) -> str | None:
    """The title of the nearest ancestor, for a child whose log starts in the
    middle of a conversation. `None` when no ancestor that can be read has one.

    `known` holds every row of the scan, and each ancestor that this walk
    opened. An ancestor has no `modified` and no `size`: the walk reads its
    title and its parent, and nothing took a `stat` of its file. It is never
    returned as a row.

    **The ancestor is opened, and not searched for.** An ancestor is a sibling
    inside the family directory of this session, so its path is a join. To
    resolve each ancestor by id would scan every family in the store,
    synchronously on the event loop of the daemon.

    The walk is bounded by the same `MAX_DEPTH` that the reader walks. The bound
    also ends a hand-edited cycle: a repeat costs up to 64 dict lookups, and no
    second collection is needed to track it. The walk stops at the first
    ancestor that cannot be read, because a picker owes a row and not an
    exception.
    """
    parent = of.parent
    family = of.family if of.family is not None else of.session_id
    for _ in range(MAX_DEPTH):
        if parent is None:
            return None
        above = known.get(parent)
        if above is None:
            above = _summarize(family_log(sessions_dir / family, parent), None, None)
            if above is None:
                return None
            # **The peek is cached back.** A segmented run is one chain, and
            # every row below this one walks the same ancestors. Without the
            # cache each row would open and parse the same files again,
            # synchronously, on the event loop of the daemon.
            known[parent] = above
        if above.title is not None or above.parent is None:
            return above.title
        parent = above.parent
    return None


def _summarize(
    path: Path, modified: float | None, size: int | None, *, cwd: str | None = None
) -> SessionSummary | None:
    """This log as a row, read from the top of the file, or `None` when the file
    cannot be read or `cwd` rules the log out. `cwd` is the filter of
    `session_summaries`, and `None` rules nothing out.

    `modified` and `size` come from the `stat` of the scan, and are `None` for an
    ancestor that `_inherited_title` opened without one. What the top of the
    log does not say stays `None` on the row.

    The filter is **here** and not on the result. A log that this scan will not
    keep then costs one header line and not a title scan. `session_summaries`
    reads every log that its walk returns whenever the matching set is smaller
    than its limit, and the title scan is the expensive half of a row that it
    is about to discard.
    """
    title: str | None = None
    recorded: str | None = None
    kind: SessionKind | None = None
    family: str | None = None
    parent: str | None = None
    origin: Literal["subagent"] | None = None
    try:
        with path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if index > TITLE_SCAN_LIMIT:
                    break
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    # A torn tail is a repair problem for Phase 1 and not for the
                    # picker. A session that will not parse still deserves a row.
                    break
                if record.get("type") == HEADER_LINE_TYPE:
                    header = _header(record.get("header"))
                    if header is not None:
                        recorded = header.cwd
                        if cwd is not None and recorded != cwd:
                            return None  # not this directory's; stop before the title
                        parent = header.parent_session
                        kind = header.kind
                        origin = header.origin
                        family = header.family
                    continue
                if record.get("type") == "user/message":
                    text = text_of_wire(as_obj(record.get("data")).get("content")).strip()
                    title = one_line(text.splitlines()[0], 72) if text else None
                    break
    except OSError:
        return None
    return SessionSummary(
        session_id=path.stem,
        modified=modified,
        size=size,
        title=title,
        cwd=recorded,
        kind=kind,
        origin=origin,
        family=family,
        parent=parent,
    )


def _header(raw: object) -> SessionHeader | None:
    """The header as pH wrote it, or nothing. A header pH cannot validate is not one."""
    try:
        return SessionHeader.model_validate(raw)
    except ValidationError:
        return None


@dataclass(frozen=True, slots=True)
class Owner:
    """Whose the log of a sub-agent is (P11-08). A log that has an `Owner` is not
    opened as a root."""

    root: str | None
    """The root whose mount writes the log, or `None` when the header of the
    child names no spawner. It is the root and not the spawner, so that a
    grandchild names the one id that is worth attaching to (`_owning_root`)."""


@dataclass(frozen=True, slots=True)
class RecordedStart:
    """What the log of a stored session says before anything is mounted for it:
    where it was worked in, and the environment that it runs in (S5, S6)."""

    session_id: str | None = None
    """The session that this is the start of, or `None` for a session that is
    about to be made."""
    cwd: str | None = None
    """The working directory that its header recorded, or `None` when there is
    no log here or its header recorded none."""
    environment: LoggedEnvironment = field(default_factory=LoggedEnvironment)
    family: str | None = None
    """The family directory in which its log was found. The log is opened there
    by path (`runtime.open_root`) and is not searched for again. `None` when
    there is no log here."""
    owner: Owner | None = None
    """Whose this log is, when it is the log of a sub-agent (P11-08). `None` for
    a session of its own: a root, a fork or a segment.

    A start refuses on this field. It is read from the header, before anything
    is mounted. The log of a child has one writer: the mount of the root that
    spawned it, which readmits the child whenever that root comes up."""


def not_a_root(session_id: str, owner: str | None) -> str:
    """Why the log of `session_id` is not opened as a root (P11-08). It is the log
    of a sub-agent, written by the mount of root `owner`. `owner` is `None` when
    the header names no spawner (`Owner.root`). The daemon, `phern -p` and rpc
    all refuse with this one sentence."""
    root = f"root {owner}" if owner is not None else "the root that spawned it"
    return (
        f"{session_id} is a sub-agent's log, written by the mount of {root}; open that "
        f"root instead, or read this one with `phern --mode trajectory --session {session_id}`"
    )


def recorded_start(sessions_dir: Path, session_id: str) -> RecordedStart:
    """Where a stored session was worked in, its environment, and whose it is
    when it is the log of a sub-agent. One locate gives all three, for the start
    of a root.

    **It reads without mounting anything**, and that is why it exists. The
    profile of a root must be mounted *with* its working directory: the fs seam
    fixes its root when the row is applied, and `workspace-lifecycle` reads it
    there to discover provisioning. The profile must also be mounted *from* the
    environment of its log. So both are needed before there is a `Context` to
    ask a store for them.

    It is filesystem-shaped, like `session_summaries` above, for the reason
    that the module docstring gives. A backend that keeps sessions elsewhere
    answers empty. The root then mounts where the profile of the deployment
    says, and it is brought to its log after it opens (`opened`).
    """
    path = locate_session(sessions_dir, session_id)
    if path is None:
        return RecordedStart(session_id=session_id)
    header, environment = _started(sessions_dir, path)
    return RecordedStart(
        session_id=session_id,
        cwd=header.cwd if header is not None else None,
        environment=environment,
        family=path.parent.name,
        owner=(
            Owner(root=_owning_root(sessions_dir, header))
            if header is not None and header.is_subagent
            else None
        ),
    )


def _owning_root(sessions_dir: Path, header: SessionHeader) -> str | None:
    """The top of the delegation line of a sub-agent: the spawner of its spawner,
    and so on. `None` when the header names no spawner.

    One header line is read for each step, and each log is **opened and not
    searched for**. The whole line shares one family directory, so every spawner
    is a sibling there. `_inherited_title` walks the ancestors of a fork in the
    same way. The walk is bounded by the `MAX_DEPTH` of the reader, which also
    ends a hand-edited cycle. It stops at the first spawner that cannot be read,
    and names the last one that it could read.
    """
    # An empty id names nobody, as an absent one does.
    owner = named(header.delegating_parent)
    if owner is None:
        return None
    family = sessions_dir / header.family
    for _ in range(MAX_DEPTH):
        above = _header_line(family_log(family, owner))
        if above is None or not above.delegating_parent:
            break
        owner = above.delegating_parent
    return owner


def recorded_environment(sessions_dir: Path, session_id: str) -> LoggedEnvironment:
    """What the log of a stored session says about its environment: its base, the
    overrides in force, and a version adopted or declined since. For a session
    with no log here, the result is an empty `LoggedEnvironment`."""
    return recorded_start(sessions_dir, session_id).environment


_PROFILE = "profile/"
"""The prefix of every record a session's environment is folded from."""


def _started(sessions_dir: Path, path: Path) -> tuple[SessionHeader | None, LoggedEnvironment]:
    """The header of a stored log and the environment that its lineage records,
    folded by the rule of `ph.session_profile`. When the lineage cannot be read,
    the result is the header alone and no environment. The strict read of the
    open is what refuses such a log.

    Every log goes through the lineage walk (`materialize_some`). For a root,
    the walk is one file. The file of a fork continues the file of its root from
    `seed_length`, and the base is in that prefix. Each file is skimmed
    (`skim_stored`): only its profile lines are parsed, and an ancestor is read
    only up to the point where its child was cut.
    """
    try:
        header, events = materialize_some(
            partial(skim_stored, sessions_dir, kinds=_PROFILE), path.stem, family=path.parent.name
        )
    except (LineageError, OSError, ValueError):
        return _header_line(path), LoggedEnvironment()
    return header, fold_environment((event.seq, event.type, event.data) for event in events)


def stored_on(sessions_dir: Path, name: str) -> list[tuple[str, LoggedEnvironment]]:
    """Every stored session whose base is the named profile `name`, newest first
    (S6).

    That is a root, a fork or a segment. A fork continues the base of its root
    with the prefix, and it is a session of its own to start. The log of a
    sub-agent is passed over at its header (`origin`), because a child runs on
    the mount of its root and has no base of its own.
    """
    found: list[tuple[str, LoggedEnvironment]] = []
    for path, _stat in session_logs(sessions_dir):
        header = _header_line(path)
        if header is None or header.is_subagent:
            continue
        _header, env = _started(sessions_dir, path)
        if env.base is not None and env.base.name == name:
            found.append((header.id, env))
    return found


def _header_line(path: Path) -> SessionHeader | None:
    """The log's header, from its first line. `None` when it has none this build reads."""
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                text = line.strip()
                if not text:
                    continue
                try:
                    record = json.loads(text)
                except json.JSONDecodeError:
                    return None
                if record.get("type") == HEADER_LINE_TYPE:
                    return _header(record.get("header"))
                return None
    except OSError:
        return None
    return None
