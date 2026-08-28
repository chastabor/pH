"""`ph.session` — the append-only log, the surface, and the derived history."""

from __future__ import annotations

from .derive import derive_event_message, derive_transcript
from .events import (
    SESSION_FORMAT_VERSION,
    SURFACE_EVENT_TYPES,
    SessionEvent,
    SurfaceIntent,
    SurfaceOp,
    SurfaceReplace,
    is_surface_eligible_type,
    now_ms,
)
from .folds import SessionFoldCache
from .json import (
    InvalidJsonValueError,
    JsonValue,
    dumps,
    freeze_json_value,
    is_json_value,
    snapshot_json_value,
    thaw_json,
)
from .known_event_types import IGNORABLE_SESSION_EVENT_TYPES, KNOWN_SESSION_EVENT_TYPES
from .request_header import (
    EpochHeader,
    RequestContext,
    canonical_header,
    fold_latest,
    fold_request_context,
    fold_request_header,
    header_equals,
)
from .session import Session, SessionHeader, SessionObserver
from .store import (
    SessionForkError,
    SessionStore,
    fork_boundaries,
    is_fork_boundary,
    new_session_id,
    open_turn_at,
)
from .surface import (
    SurfaceError,
    SurfaceFoldReplacement,
    SurfaceFoldResult,
    SurfaceManager,
    fold_surface,
    is_append_surface_event,
    is_replacement_surface_event,
    is_surface_event,
)

__all__ = [
    "IGNORABLE_SESSION_EVENT_TYPES",
    "KNOWN_SESSION_EVENT_TYPES",
    "SESSION_FORMAT_VERSION",
    "SURFACE_EVENT_TYPES",
    "EpochHeader",
    "InvalidJsonValueError",
    "JsonValue",
    "RequestContext",
    "Session",
    "SessionEvent",
    "SessionFoldCache",
    "SessionForkError",
    "SessionHeader",
    "SessionObserver",
    "SessionStore",
    "SurfaceError",
    "SurfaceFoldReplacement",
    "SurfaceFoldResult",
    "SurfaceIntent",
    "SurfaceManager",
    "SurfaceOp",
    "SurfaceReplace",
    "canonical_header",
    "derive_event_message",
    "derive_transcript",
    "dumps",
    "fold_latest",
    "fold_request_context",
    "fold_request_header",
    "fold_surface",
    "fork_boundaries",
    "freeze_json_value",
    "header_equals",
    "is_append_surface_event",
    "is_fork_boundary",
    "is_json_value",
    "is_replacement_surface_event",
    "is_surface_eligible_type",
    "is_surface_event",
    "new_session_id",
    "now_ms",
    "open_turn_at",
    "snapshot_json_value",
    "thaw_json",
]
