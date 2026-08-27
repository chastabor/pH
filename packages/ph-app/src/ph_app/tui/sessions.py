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
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from pydantic import ValidationError

from ph.persistence.jsonl import HEADER_LINE_TYPE
from ph.session import SessionHeader

from .wire import obj, text_of_wire

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

    @property
    def when(self) -> str:
        return datetime.fromtimestamp(self.modified).strftime("%Y-%m-%d %H:%M")


def session_summaries(sessions_dir: Path, *, limit: int = 50) -> list[SessionSummary]:
    """Stored sessions, most recently touched first. Unreadable files are skipped.

    One `stat` per file: the directory entry's, reused for both the sort and
    the summary.
    """
    try:
        with os.scandir(sessions_dir) as entries:
            found = [
                (entry, entry.stat())
                for entry in entries
                if entry.name.endswith(".jsonl") and entry.is_file()
            ]
    except OSError:
        return []
    found.sort(key=lambda pair: pair[1].st_mtime, reverse=True)
    summaries: list[SessionSummary] = []
    for entry, stat in found[:limit]:
        summary = _summarize(Path(entry.path), stat.st_mtime, stat.st_size)
        if summary is not None:
            summaries.append(summary)
    return summaries


def _summarize(path: Path, modified: float, size: int) -> SessionSummary | None:
    title = ""
    cwd = ""
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
                    continue
                if record.get("type") == "user/message":
                    text = text_of_wire(obj(record.get("data")).get("content")).strip()
                    title = text.splitlines()[0][:72] if text else ""
                    break
    except OSError:
        return None
    return SessionSummary(
        session_id=path.stem, modified=modified, size=size, title=title, cwd=cwd, parent=parent
    )


def _header(raw: object) -> SessionHeader | None:
    """The header as pH wrote it, or nothing. A header pH cannot validate is not one."""
    try:
        return SessionHeader.model_validate(raw)
    except ValidationError:
        return None
