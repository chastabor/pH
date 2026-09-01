"""`ph.persistence` — session storage backends, checkpoints, and crash repair."""

from __future__ import annotations

from .jsonl import (
    JsonlSessionStore,
    read_records,
    read_session,
    resume_session,
    resumption_of,
    session_path,
)
from .lineage import MAX_DEPTH, LineageError, ReadOne, materialise
from .repair import TOOL_NOT_STARTED, TOOL_OUTCOME_UNKNOWN, interrupted_turn_closers, repaired

__all__ = [
    "MAX_DEPTH",
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "JsonlSessionStore",
    "LineageError",
    "ReadOne",
    "interrupted_turn_closers",
    "materialise",
    "read_records",
    "read_session",
    "repaired",
    "resume_session",
    "resumption_of",
    "session_path",
]
