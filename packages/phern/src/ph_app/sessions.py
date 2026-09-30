"""Listing stored sessions cheaply enough to open a picker on.

A session log is append-only and can be large. The picker needs three things —
when, what it was about, how big — and getting them by reading every event of
every session would make the picker slow exactly where a user has many sessions
to choose between. So this reads the header line and stops at the first
`user/message`: both live at the top of the file, and the first thing the person
typed is the best title a session has.

**Out of `tui/` because the reader moved.** Before P5-14 the terminal was the
harness, so listing its own logs was reading its own files. Now the daemon holds
them: it answers `sessions/browse`, and this is the fold it answers with. A
client reads no session file at all — which is what makes a front end that is
not on the daemon's machine possible at all, and what stops a client and a
daemon disagreeing about which `$PH_HOME` they meant.

Rule 6, unchanged by the move: the fold is **filesystem-shaped**. It walks
`<sessions>/<family>/*.jsonl` and peeks their headers, so a backend that keeps
sessions anywhere else — Turso's one-database-per-session — answers `directory()`
with `None` and contributes no stored rows. `SessionPersistence.stored()` is the
backend-agnostic listing and deliberately carries neither `title` nor `size`,
because those mean different things per backend; a picker without titles is the
thing this module exists to avoid.

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

from ph.json import JsonObject, as_int, as_obj
from ph.persistence import MAX_DEPTH, LineageError, materialize
from ph.persistence.jsonl import (
    HEADER_LINE_TYPE,
    family_log,
    locate_session,
    read_stored,
    session_logs,
)
from ph.session import SessionHeader, cwd_tag
from ph.session_profile import LoggedEnvironment, fold_environment
from ph.wire import WireModel

from .wire import text_of_wire

__all__ = [
    "RecordedStart",
    "SessionSummary",
    "recorded_environment",
    "recorded_start",
    "session_summaries",
    "stored_on",
]

TITLE_SCAN_LIMIT = 40
"""Events to look through for a title before giving up. A session that opens
with forty non-user events has no title worth waiting for."""


class SessionSummary(WireModel):
    """Enough about a stored session to choose it from a list.

    A `WireModel` because it crosses a socket now: `sessions/browse` folds these
    on the daemon and `to_wire()`/`model_validate` carry them, so a field added
    here reaches every front end with nothing edited at either edge (P7-11)."""

    session_id: str
    modified: float
    size: int
    title: str = ""
    cwd: str = ""
    parent: str | None = None
    """The session this one was forked from, when the header says so — or, for a
    sub-agent's log (`origin`), the agent that spawned it."""

    kind: str = ""
    """`"fork"`, `"segment"`, or empty for a root — which has no parent to qualify."""

    origin: Literal["subagent"] | None = None
    """`"subagent"` for a sub-agent's log, as its header says (P11-08).

    `parent` alone cannot tell a child from a fork: both name the session they came
    from. A fork is a session of its own a person can carry on; a child's log is
    written by its root's mount and is never attached, so the picker has to know
    which one a row is before it offers it.
    """

    state: str = "stored"
    """`running` or `stored` — whether a daemon is holding this session now.

    The distinction is new with P5-14: before it, the only session that could be
    live was the one this terminal was hosting. Picking a running one means
    joining a turn that may be in flight somewhere else, which is worth saying
    on the row.

    **Two states, not three.** A `passivated` one — a root the sweep released —
    was planned and dropped: telling it from `stored` means reading the *tail* of
    every log for a `supervisor/passivated` record, and this module exists
    precisely because reading whole logs makes the picker slow where a person has
    many. Nothing the person then does differs: picking either mounts the session
    from its log. So the cost bought a label and no decision.
    """

    family: str = ""
    """The lineage directory this log sits in — every ancestor is a sibling in it.

    Read from the same one-line header peek as `parent`, and it is what turns the
    ancestor walk below from a store-wide search into an open.
    """

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")


def session_summaries(
    sessions_dir: Path, *, limit: int = 50, cwd: str = ""
) -> list[SessionSummary]:
    """Stored sessions, most recently touched first. Unreadable files are skipped.

    One `stat` per file: the directory entry's, reused for both the sort and
    the summary. `session_logs` walks one level of family directories, which is
    where every log lives, and hands them back newest-first.

    **`cwd` filters during the scan, not after it**, and the difference is the
    whole reason it is a parameter here rather than a comprehension at the call
    site. `limit` takes the newest rows; filtering what comes back would let a
    directory's own sessions fall off that window behind newer work elsewhere, so
    a repo somebody last touched a month ago would list as empty on a busy
    machine. Filtering first means the limit counts *matching* sessions.

    **The cost, measured, because it is on the path to the first prompt.** A
    filtered scan reads headers until it has `limit` matches rather than stopping
    at `limit` files — so a directory with fewer than `limit` sessions, which is
    nearly every directory, walks the whole store. At 500 logs that is 500 header
    reads against 50, and `_summarize` gives back the non-matching ones as early
    as it can (before the title scan, which is the expensive half) to keep it to
    ~12 µs each: 8.6 ms → 5.8 ms at 500 logs, 88 ms → 62 ms at 5 000.

    The bound is therefore the whole store, and the store has no retention — every
    fork and every compaction segment is another file. If that ever stops being
    a few hundred, the answer is a per-directory index rather than a faster scan;
    a scan *cap* is the one thing it must not be, since "a repo touched a month
    ago still lists" is the property this filter exists to provide.

    **Not enforced (§5 rule 6): `cwd` matches the string a header recorded, not
    the directory it names.** Nothing is resolved or canonicalized on either
    side, so one checkout reached through two paths — a symlink, `/tmp` against
    `/private/tmp`, a mount under another name — lists as two directories with
    two sets of sessions. Resolving would be the wrong fix rather than a missing
    one: the header records where a session *said* it was working, and a picker
    that silently merged two paths would show sessions whose workspaces were
    provisioned somewhere else. `/sessions` lists the store without the filter,
    which is how the other set is found.
    """
    # **The tag filters whole lineages before a single file is touched** — see
    # `family_dirs`. What reaches the loop below is already this directory's
    # work, so the header check in `_summarize` is confirming a 24-bit match
    # rather than sifting the store.
    found = sorted(
        session_logs(sessions_dir, tag=cwd_tag(cwd) if cwd else ""),
        key=lambda pair: pair[1].st_mtime,
        reverse=True,
    )
    summaries: list[SessionSummary] = []
    for path, stat in found:
        if len(summaries) >= limit:
            break
        summary = _summarize(path, stat.st_mtime, stat.st_size, cwd=cwd)
        if summary is not None:
            summaries.append(summary)
    # A reference-forked child holds only its own events, so the `user/message`
    # that names the conversation lives in an ancestor and its row would render
    # blank — the "hex ids with nothing failing" outcome `StoredSession`'s own
    # comment refuses. Done after the scan so an ancestor already read here is
    # not read twice, which is what keeps the one-stat-per-file claim above true
    # for every session that has its own title.
    known = {summary.session_id: summary for summary in summaries}
    for index, summary in enumerate(summaries):
        if not summary.title and summary.parent is not None:
            summaries[index] = summary.model_copy(
                update={"title": _inherited_title(sessions_dir, summary, known)}
            )
    return summaries


def _inherited_title(
    sessions_dir: Path, of: SessionSummary, known: dict[str, SessionSummary]
) -> str:
    """The nearest ancestor's title, for a child whose log starts mid-conversation.

    **Opened, not searched for.** An ancestor is a sibling inside this session's own
    family directory, so the path is a join; resolving each one by id instead scans
    every family in the store, synchronously on the UI thread.

    Bounded by the same `MAX_DEPTH` the reader walks, which is also what terminates a
    hand-edited cycle — a repeat costs up to 64 dict lookups rather than a second
    collection to track it. It stops at the first ancestor that will not read: a picker
    owes a row, not an exception.
    """
    parent = of.parent
    family = of.family or of.session_id
    for _ in range(MAX_DEPTH):
        if parent is None:
            return ""
        summary = known.get(parent)
        if summary is None:
            summary = _summarize(family_log(sessions_dir / family, parent), 0.0, 0)
            if summary is None:
                return ""
            # **Cached back**, because a segmented run is one chain: every row
            # below this one walks the same ancestors, so without this each row
            # re-opens and re-parses the same files — synchronously, on the UI
            # thread.
            known[parent] = summary
        if summary.title or summary.parent is None:
            return summary.title
        parent = summary.parent
    return ""


def _summarize(path: Path, modified: float, size: int, *, cwd: str = "") -> SessionSummary | None:
    """This log as a row, or `None` — including when `cwd` rules it out.

    The filter is **here**, rather than on the result, so a log this scan is not
    going to keep costs one header line instead of a title scan: `session_summaries`
    walks the whole store whenever the matching set is smaller than its limit, and
    the title scan is the expensive half of a row it is about to discard.
    """
    title = ""
    recorded = ""
    kind = ""
    family = ""
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
                    # A torn tail is Phase 1's repair problem, not the picker's;
                    # a session that will not parse still deserves a row.
                    break
                if record.get("type") == HEADER_LINE_TYPE:
                    header = _header(record.get("header"))
                    if header is not None:
                        recorded = header.cwd or ""
                        if cwd and recorded != cwd:
                            return None  # not this directory's; stop before the title
                        parent = header.parent_session
                        kind = header.kind or ""
                        origin = header.origin
                        family = header.family
                    continue
                if record.get("type") == "user/message":
                    text = text_of_wire(as_obj(record.get("data")).get("content")).strip()
                    title = text.splitlines()[0][:72] if text else ""
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
class RecordedStart:
    """What a stored session's log says before anything is mounted for it: where it
    was worked in, and the environment it runs in (S5, S6)."""

    cwd: str = ""
    environment: LoggedEnvironment = field(default_factory=LoggedEnvironment)
    owner: str | None = None
    """The root whose mount writes this log, when it is a sub-agent's (P11-08), and
    `None` for a session of its own: a root, a fork or a segment. `""` for a child
    whose header names no spawner.

    What a start refuses on, and from the header, before anything is mounted: a
    child's log has one writer, the mount of the root that spawned it, which
    readmits it whenever that root comes up. The root rather than the spawner, so a
    grandchild names the one id worth attaching instead (`_owning_root`)."""


def not_a_root(session_id: str, owner: str) -> str:
    """Why `session_id`'s log is not opened as a root (P11-08): it is a sub-agent's,
    written by the mount of root `owner` — `""` when its header names no spawner.
    The one sentence the daemon, `phern -p` and rpc all refuse with."""
    root = f"root {owner}" if owner else "the root that spawned it"
    return (
        f"{session_id} is a sub-agent's log, written by the mount of {root}; open that "
        f"root instead, or read this one with `phern --mode trajectory --session {session_id}`"
    )


def recorded_start(sessions_dir: Path, session_id: str) -> RecordedStart:
    """Where a stored session was worked in, its environment, and whose it is when it
    is a sub-agent's — one locate, for a root's start.

    **Read without mounting anything**, which is the whole reason it exists: a
    root's profile has to be mounted *with* its working directory — the fs seam
    fixes its root at row-apply time and `workspace-lifecycle` reads it there to
    discover provisioning — and mounted *from* its log's environment, so both are
    needed before there is a `Context` to ask a store for them.

    Filesystem-shaped, like `session_summaries` above and for the same reason
    stated there: a backend that keeps sessions elsewhere answers empty, the root
    mounts where the deployment's profile says, and it is brought to its log after
    it opens (`opened`).
    """
    path = locate_session(sessions_dir, session_id)
    if path is None or not path.is_file():
        return RecordedStart()
    header = _header_line(path)
    return RecordedStart(
        cwd=(header.cwd or "") if header is not None else "",
        environment=_environment_at(sessions_dir, path, header),
        owner=(
            _owning_root(sessions_dir, header)
            if header is not None and header.is_subagent
            else None
        ),
    )


def _owning_root(sessions_dir: Path, header: SessionHeader) -> str:
    """The top of a sub-agent's delegation line: its spawner's spawner, and so on.

    One header line per step, **opened rather than searched for**: the whole line
    shares one family directory, so every spawner is a sibling there, as
    `_inherited_title` walks a fork's ancestors. Bounded by the reader's
    `MAX_DEPTH`, which also ends a hand-edited cycle, and it stops at the first
    spawner that will not read, naming the last one it could.
    """
    owner = header.delegating_parent or ""
    family = sessions_dir / header.family
    for _ in range(MAX_DEPTH):
        above = _header_line(family_log(family, owner)) if owner else None
        if above is None or above.delegating_parent is None:
            break
        owner = above.delegating_parent
    return owner


def recorded_environment(sessions_dir: Path, session_id: str) -> LoggedEnvironment:
    """What a stored session's log says about its environment: its base, the
    overrides in force, and a version adopted or declined since — or an empty
    `LoggedEnvironment` for one with no log here."""
    return recorded_start(sessions_dir, session_id).environment


def _environment_at(
    sessions_dir: Path, path: Path, header: SessionHeader | None
) -> LoggedEnvironment:
    """The log at `path`, folded by `ph.session_profile`'s own rule.

    A line scan of the one file, decoding only the lines that name a profile record,
    because the resume that follows parses every envelope anyway — for a log that
    holds its own history, which is every root. A fork's file continues its root's
    from `seed_length`, and the base is in that prefix, so a fork is read through
    the store's own lineage walk (`materialize`). An unfinished last line is
    skipped, as the reader does; a torn last batch costs nothing here, since the
    only batch of these records is a base with the clears of overrides that no
    longer change anything.
    """
    if header is not None and header.seed_length:
        try:
            _header, events = materialize(partial(read_stored, sessions_dir), header.id)
        except (LineageError, OSError, ValueError):
            return LoggedEnvironment()
        return fold_environment((event.seq, event.type, event.data) for event in events)
    records: list[tuple[int, str, JsonObject]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if '"profile/' not in line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                kind, data = record.get("type"), record.get("data")
                if isinstance(kind, str) and isinstance(data, dict):
                    records.append((as_int(record.get("seq")), kind, data))
    except OSError:
        return LoggedEnvironment()
    return fold_environment(records)


def stored_on(sessions_dir: Path, name: str) -> list[tuple[str, LoggedEnvironment]]:
    """Every stored session whose base is the named profile `name`, newest first (S6).

    A root, a fork or a segment: a fork continues its root's base with the prefix,
    and is a session of its own to start. A sub-agent's log is passed over at its
    header (`origin`), since a child runs on its root's and has no base of its own.
    """
    found: list[tuple[str, LoggedEnvironment]] = []
    for path, _stat in session_logs(sessions_dir):
        header = _header_line(path)
        if header is None or header.is_subagent:
            continue
        env = _environment_at(sessions_dir, path, header)
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
