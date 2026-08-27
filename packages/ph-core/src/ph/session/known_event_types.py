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

__all__ = ["IGNORABLE_SESSION_EVENT_TYPES", "KNOWN_SESSION_EVENT_TYPES"]

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
        # policy and human-in-the-loop
        "approval/asked",
        "approval/decided",
        "approval/policy",
        "command/done",
        "command/run",
        "permission/preset",
        "sandbox/mode",
        # capability observations
        "fs/observed",
        # retry
        "llm/retry",
        # Code Mode dispatch records (log-only; see ph.tools.code_mode)
        "tool/code-dispatch",
        "tool/code-dispatch-start",
        # Persistent-kernel state (D17; emitted by ph-rlm's snapshot policy).
        # These are what make `persistence: "namespace"` admissible at all: the
        # seam takes the provider's promise at registration, and these events are
        # the promise being kept. Listed here rather than in ph-rlm because the
        # *reader* is ph-core — a log carrying a type this build does not know is
        # refused on the seed path, so the vocabulary has one home.
        "kernel/snapshot",
        "kernel/restored",
    }
)

IGNORABLE_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "kernel/snapshot",
        "kernel/restored",
    }
)
"""Types a *different* build may skip without misreading the rest of the log.

Ignorability is a property of the type, not of the call — "a reader that does
not recognize this `type` may skip it" is true of every record of the type or
none — so it is declared here, beside the type, and `Session.append` stamps it.
A per-call flag would let two call sites disagree about one type, and a
forgotten flag is an older build refusing a log it could have read.

The set is deliberately small: an unrecognized *required* event must still
refuse the seed, because skipping one can change how everything after it reads.
Only purely informational records — kernel state, and (later) subagent status —
belong here.
"""
