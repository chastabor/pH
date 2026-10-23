"""`ph.persistence` — session storage backends, checkpoints, and crash repair."""

from __future__ import annotations

from .jsonl import (
    JsonlArchive,
    JsonlSessionStore,
    LogDamage,
    LogDamaged,
    append_records,
    damage_in,
    damaged_logs,
    read_records,
    read_session,
    records_in,
    session_path,
    truncate_damaged,
)
from .lease import SessionBusy, claim_session
from .lineage import (
    MAX_DEPTH,
    LineageError,
    ReadOne,
    ReadSome,
    lineage_faults,
    materialize,
    materialize_some,
)
from .opening import open_session, stored_session
from .protocol import ClaimingStore, NoStoredSession, read_if_stored
from .repair import (
    TOOL_NOT_STARTED,
    TOOL_OUTCOME_UNKNOWN,
    UndeclaredIntentError,
    interrupted_turn_closers,
    repaired,
)
from .resume import (
    CutTurnOwner,
    Resumption,
    cut_short_note,
    cut_turn_owner,
    resume_session,
    resumption_of,
    resumptions,
)

__all__ = [
    "MAX_DEPTH",
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "ClaimingStore",
    "CutTurnOwner",
    "JsonlArchive",
    "JsonlSessionStore",
    "LineageError",
    "LogDamage",
    "LogDamaged",
    "NoStoredSession",
    "ReadOne",
    "ReadSome",
    "Resumption",
    "SessionBusy",
    "UndeclaredIntentError",
    "append_records",
    "claim_session",
    "cut_short_note",
    "cut_turn_owner",
    "damage_in",
    "damaged_logs",
    "interrupted_turn_closers",
    "lineage_faults",
    "materialize",
    "materialize_some",
    "open_session",
    "read_if_stored",
    "read_records",
    "read_session",
    "records_in",
    "repaired",
    "resume_session",
    "resumption_of",
    "resumptions",
    "session_path",
    "stored_session",
    "truncate_damaged",
]
