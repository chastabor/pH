"""Runtime invariant: model-visible means logged (P0-14, invariant I3).

Every message that reaches a model request must be exactly `derive_messages()`
at that step. It must be *identical*, and not "equivalent" or "a superset".
That equality is what makes the session log a complete trace. Anything that the
model saw can be audited, replayed and offloaded only if the log is the sole
source that it came from.

The check runs as a `llm/stream` listener. It therefore sees the request that
the adapter is about to receive, and not the request that the loop believed it
built. A request is held to the invariant when it is session-bound and names no
other purpose (`GenerateOptions.is_loop_request`). A conversation request
cannot opt out by leaving a flag at its default.

@module ph.agent_loop.invariant
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from ..cordis import Context, Next, plugin
from ..keys import SESSIONS
from ..llm.types import GenerateOptions, StreamChunk
from ..seams.invariants import Invariant, contribute

__all__ = ["ModelVisibleNotLoggedError", "apply"]


class ModelVisibleNotLoggedError(AssertionError):
    """A loop request carried messages that are not the session's derivation."""


@plugin("agent-loop-invariant", affects="deployment", inject=[SESSIONS])
async def apply(ctx: Context, config: None) -> None:
    """Assert `messages == derive_messages()` on every loop request."""

    async def check(
        request: GenerateOptions, next_: Next[AsyncIterator[StreamChunk]]
    ) -> AsyncIterator[StreamChunk]:
        # `is_loop_request` implies a session id, but through a property that
        # the checker cannot see into. The explicit test lets `get` be typed.
        session_id = request.session_id
        session = (
            ctx.require(SESSIONS).get(session_id)
            if session_id is not None and request.is_loop_request
            else None
        )
        if session is not None:
            derived = session.derive_messages()
            sent = request.messages
            # A request built from the cache satisfies `is`. Equality is the
            # fallback for a request rebuilt from equal values, and it is far
            # cheaper than serializing the whole conversation again.
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

    # Prepended, so that the check runs before any listener that could rewrite
    # the request. The invariant is about what the LOOP built, and not about
    # what middleware made of it.
    ctx.on("llm/stream", check, prepend=True)

    # The invariant is declared as well as enforced (P6-01). Whether I3 holds
    # has always been a property of the *profile*, because this row is
    # optional, and nothing said so. `check=None` because there is no state
    # between requests to poll. What this invariant can correctly report is
    # that every request was held to it, and never that it holds right now.
    contribute(
        ctx,
        Invariant(
            id="model-visible-logged",
            statement="every loop request's messages are the session's own derivation",
            order=1,
        ),
    )
