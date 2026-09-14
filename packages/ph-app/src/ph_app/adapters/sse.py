"""Server-sent events, read incrementally.

Both provider APIs pH speaks stream SSE. Shared here because the framing is the
same and the bug is the same: a naive reader that splits on `\\n\\n` across
network reads loses events whose boundary lands mid-chunk.

@module ph_app.adapters.sse
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Protocol

from ph.json import JsonValue, loads

__all__ = ["TextStream", "iter_sse"]


class TextStream(Protocol):
    """What this reader needs of a response: decoded text, as it arrives.

    A Protocol rather than `httpx.Response` for the reason `AgentHandle` is one —
    it is the surface actually read, and the reader has no other business with
    the client library. `test_adapters._Response` is the second implementation,
    and a stub that could not stand in would make every SSE test a test of httpx.
    """

    def aiter_text(self) -> AsyncIterator[str]: ...


async def iter_sse(response: TextStream) -> AsyncIterator[tuple[str, JsonValue]]:
    """Yield `(event, data)` pairs from an SSE response.

    `data` is parsed JSON, or the raw string when it is not JSON (`[DONE]`).
    """
    buffer = ""
    async for chunk in response.aiter_text():
        buffer += chunk
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            event = "message"
            payloads: list[str] = []
            for line in raw.splitlines():
                if line.startswith("event:"):
                    event = line[len("event:") :].strip()
                elif line.startswith("data:"):
                    payloads.append(line[len("data:") :].strip())
            if not payloads:
                continue
            body = "\n".join(payloads)
            if body == "[DONE]":
                return
            try:
                yield event, loads(body)
            except json.JSONDecodeError:
                yield event, body
