"""The Anthropic SSE envelope, for tests that simulate that provider.

Two test modules simulate the provider, for two different reasons. P6-13 needs
a wire that reports prefix-cache usage. P7-03 needs a wire that issues file ids
and can be made to forget one. Both had written out the same six events to say
"the model replied with this text". Neither test is about that part, and that
part goes wrong silently: a `message_delta` without its `usage` makes the reply
look free and does not fail. A third module,
`packages/phern/tests/test_adapters.py`, takes the first three events to build
a stream that fails after the answer has begun.

**A dict builder, not an adapter.** It composes the frames that the
`_StreamState.consume` of the Anthropic adapter reads, and nothing else. So it
does not import `ph_app`, and the package layering holds. The test that owns
the simulation still owns every decision in it, and does not have to write the
envelope.

@module ph.testing.anthropic_wire
"""

from __future__ import annotations

from typing import Any

__all__ = ["anthropic_reply"]


def anthropic_reply(
    text: str, *, usage: dict[str, int] | None = None, output_tokens: int = 8
) -> list[tuple[str, dict[str, Any]]]:
    """One text reply as `(event, payload)` pairs, ready to yield.

    `usage` goes on `message_start`, because this provider reports input
    accounting there. That includes the cache counters that P6-13 asserts on.
    Output tokens arrive at the end, on `message_delta`. The split is the shape
    of the provider and not a convenience. A test that merged the two would
    agree with an adapter that read either one.
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
