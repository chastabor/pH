"""Listing stored sessions cheaply enough to open a picker on.

A session log is append-only and can be large. The picker needs three things —
when, what it was about, how big — and getting them by reading every event of
every session would make the picker slow exactly where a user has many sessions
to choose between. So this reads the header line and stops at the first
`user/message`: both live at the top of the file, and the first thing the person
typed is the best title a session has.

@module ph_app.tui.sessions
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from ph.persistence import MAX_DEPTH
from ph.persistence.jsonl import HEADER_LINE_TYPE, family_log, session_logs
from ph.session import SessionHeader

from ..wire import obj, text_of_wire

__all__ = ["SessionSummary", "session_summaries"]

TITLE_SCAN_LIMIT = 40
"""Events to look through for a title before giving up. A session that opens
with forty non-user events has no title worth waiting for."""


@dataclass(frozen=True, slots=True)
class SessionSummary:
    """Enough about a stored session to choose it from a list."""

    session_id: str
    modified: float
    size: int
    title: str = ""
    cwd: str = ""
    parent: str | None = None
    """The session this one was forked from, when the header says so."""

    kind: str = ""
    """`"fork"`, `"segment"`, or empty for a root — which has no parent to qualify."""

    family: str = ""
    """The lineage directory this log sits in — every ancestor is a sibling in it.

    Read from the same one-line header peek as `parent`, and it is what turns the
    ancestor walk below from a store-wide search into an open.
    """

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")


def session_summaries(sessions_dir: Path, *, limit: int = 50) -> list[SessionSummary]:
    """Stored sessions, most recently touched first. Unreadable files are skipped.

    One `stat` per file: the directory entry's, reused for both the sort and
    the summary. `session_logs` walks one level of family directories, which is
    where every log lives, and hands them back newest-first.
    """
    found = sorted(session_logs(sessions_dir), key=lambda pair: pair[1].st_mtime, reverse=True)
    summaries: list[SessionSummary] = []
    for path, stat in found[:limit]:
        summary = _summarize(path, stat.st_mtime, stat.st_size)
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
            summaries[index] = replace(
                summary, title=_inherited_title(sessions_dir, summary, known)
            )
    return summaries


def _inherited_title(
    sessions_dir: Path, of: SessionSummary, known: dict[str, SessionSummary]
) -> str:
    """The nearest ancestor's title, for a child whose log starts mid-conversation.

    **Opened, not searched for.** An ancestor is a sibling inside this session's
    own family directory, so the path is a join. Resolving each one by id instead
    scanned every family in the store: measured at 200 families whose newest rows
    were segment tips, 111.8 ms and 200 store scans against 18.6 ms — 6x, and it
    runs synchronously on the UI thread.

    Bounded by the same `MAX_DEPTH` the reader walks, which is also what
    terminates a hand-edited cycle — a repeat costs up to 64 dict lookups rather
    than a second collection to track it, the same budget `materialise` already
    spends on every chained read. It stops at the first ancestor that will not
    read: a picker owes a row, not an exception.
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
            # below this one walks the same ancestors, and without this each of
            # fifty rows re-opens and re-parses the same files. Measured on a
            # 60-segment chain: 550 opens and 39.9 ms against 60 and 4.6 ms —
            # and this runs synchronously on the UI thread.
            known[parent] = summary
        if summary.title or summary.parent is None:
            return summary.title
        parent = summary.parent
    return ""


def _summarize(path: Path, modified: float, size: int) -> SessionSummary | None:
    title = ""
    cwd = ""
    kind = ""
    family = ""
    parent: str | None = None
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
                        cwd = header.cwd or ""
                        parent = header.parent_session
                        kind = header.kind or ""
                        family = header.family
                    continue
                if record.get("type") == "user/message":
                    text = text_of_wire(obj(record.get("data")).get("content")).strip()
                    title = text.splitlines()[0][:72] if text else ""
                    break
    except OSError:
        return None
    return SessionSummary(
        session_id=path.stem,
        modified=modified,
        size=size,
        title=title,
        cwd=cwd,
        kind=kind,
        family=family,
        parent=parent,
    )


def _header(raw: object) -> SessionHeader | None:
    """The header as pH wrote it, or nothing. A header pH cannot validate is not one."""
    try:
        return SessionHeader.model_validate(raw)
    except ValidationError:
        return None
