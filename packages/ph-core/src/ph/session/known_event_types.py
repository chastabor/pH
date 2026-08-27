"""The event vocabulary this build understands.

The persistence read path refuses to interpret a log containing a type outside
this set **unless** the event carries `ignorable`. Such a log was likely written
by a newer harness, and an unrecognized *required* event may change how the rest
of the log is read — so silently skipping it would reconstruct a wrong session
rather than an incomplete one.

Kept beside the code that appends these types, and checked by a test that walks
every `append(` call site in `ph-core`: a type that ships without an entry here
would be a log this build could write and then refuse to read.

@module ph.session.known_event_types
"""

from __future__ import annotations

__all__ = ["KNOWN_SESSION_EVENT_TYPES"]

KNOWN_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # agent lifecycle
        "agent/inbox/spliced",
        "assistant/chunk",
        "assistant/message",
        "request/context",
        "request/header",
        "session/end-seed",
        "step/end",
        "step/start",
        "tool/call",
        "tool/result",
        "turn/end",
        "turn/start",
        "user/message",
    }
)
