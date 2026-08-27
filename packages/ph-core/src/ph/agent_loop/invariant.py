"""Runtime invariant: model-visible means logged (P0-14, invariant I3).

Every message reaching a model request must be exactly `derive_messages()` at
that step. Not "equivalent", not "a superset" — *identical*, because that
equality is what makes the session log a complete trace: anything the model saw
can be audited, replayed and offloaded only if the log is the sole source it
came from.

The check runs as a `llm/stream` listener, so it sees the request the adapter
is about to receive rather than the one the loop believed it built. A request
is held to the invariant when it is session-bound and names no other purpose
(`GenerateOptions.is_loop_request`) — a conversation request cannot opt out by
leaving a flag at its default.

@module ph.agent_loop.invariant
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..cordis import Context, plugin
from ..llm.types import GenerateOptions

__all__ = ["ModelVisibleNotLoggedError", "apply"]


class ModelVisibleNotLoggedError(AssertionError):
    """A loop request carried messages that are not the session's derivation."""


@plugin("agent-loop-invariant", inject=["sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Assert `messages == derive_messages()` on every loop request."""

    async def check(request: GenerateOptions, next_: Callable[[], Any]) -> Any:
        session = ctx.sessions.get(request.session_id) if request.is_loop_request else None
        if session is not None:
            derived = session.derive_messages()
            sent = request.messages
            # A request built from the cache satisfies `is`; equality is the
            # fallback for one rebuilt from equal values, and is far cheaper
            # than re-serializing the whole conversation.
            if len(sent) != len(derived) or any(
                left is not right and left != right
                for left, right in zip(sent, derived, strict=True)
            ):
                raise ModelVisibleNotLoggedError(
                    "model-visible means logged: this request's messages are not "
                    f"session {request.session_id}'s derive_messages() "
                    f"({len(sent)} sent vs {len(derived)} derived)"
                )
        return await next_()

    # Prepended so the check runs before any listener that could rewrite the
    # request — the invariant is about what the LOOP built, not what middleware
    # made of it.
    ctx.on("llm/stream", check, prepend=True)
