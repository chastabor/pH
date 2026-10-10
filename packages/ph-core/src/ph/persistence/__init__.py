"""`ph.persistence` — session storage backends, checkpoints, and crash repair."""

from __future__ import annotations

from .jsonl import (
    JsonlSessionStore,
    append_records,
    read_records,
    read_session,
    records_in,
    session_path,
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
from .resume import Resumption, resume_session, resumption_of, resumptions

__all__ = [
    "MAX_DEPTH",
    "TOOL_NOT_STARTED",
    "TOOL_OUTCOME_UNKNOWN",
    "ClaimingStore",
    "JsonlSessionStore",
    "LineageError",
    "NoStoredSession",
    "ReadOne",
    "ReadSome",
    "Resumption",
    "SessionBusy",
    "UndeclaredIntentError",
    "append_records",
    "claim_session",
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
]
