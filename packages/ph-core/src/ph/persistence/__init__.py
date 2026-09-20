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
from .lease import SessionBusy, claim_session
from .lineage import MAX_DEPTH, LineageError, ReadOne, lineage_faults, materialize
from .protocol import ClaimingStore
from .repair import TOOL_NOT_STARTED, TOOL_OUTCOME_UNKNOWN, interrupted_turn_closers, repaired

__all__ = [
    "MAX_DEPTH",
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "ClaimingStore",
    "JsonlSessionStore",
    "LineageError",
    "ReadOne",
    "SessionBusy",
    "claim_session",
    "interrupted_turn_closers",
    "lineage_faults",
    "materialize",
    "read_records",
    "read_session",
    "repaired",
    "resume_session",
    "resumption_of",
    "session_path",
]
