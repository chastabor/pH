"""The Anthropic SSE envelope, for tests that simulate that provider.

Two of them now do, for two different reasons — P6-13 needs a wire that reports
prefix-cache usage, P7-03 needs one that hands out file ids and can be made to
forget one — and both had spelled out the same six events to say "the model
replied with this text". That is the part neither test is about, and the part
that goes wrong quietly: a `message_delta` missing its `usage` leaves the reply
looking free rather than failing.

**A dict builder, not an adapter.** It composes the frames `_StreamState.consume`
reads and nothing else, so it imports no `ph_app` and stays on the right side of
the package layering — the test that owns the simulation still owns every
decision in it, and gets the envelope for nothing.

@module ph.testing.anthropic_wire
"""

from __future__ import annotations

from typing import Any

__all__ = ["anthropic_reply"]


def anthropic_reply(
    text: str, *, usage: dict[str, int] | None = None, output_tokens: int = 8
) -> list[tuple[str, dict[str, Any]]]:
    """One text reply as `(event, payload)` pairs, ready to yield.

    `usage` rides `message_start` because that is where this provider reports
    input accounting — including the cache counters P6-13 asserts on — while
    output tokens arrive at the end, on `message_delta`. Splitting them is the
    provider's shape, not a convenience, and a test that merged them would agree
    with an adapter that read either one.
    """
    return [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {"usage": usage or {"input_tokens": 0, "output_tokens": 0}},
            },
        ),
        (
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
        ),
        (
            "content_block_delta",
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
        ),
        ("content_block_stop", {"type": "content_block_stop", "index": 0}),
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": output_tokens},
            },
        ),
        ("message_stop", {"type": "message_stop"}),
    ]
