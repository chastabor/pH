"""Server-sent events, read incrementally.

All three provider APIs that pH speaks stream SSE. The reader is shared here
because the framing is the same and the bug is the same: a naive reader that
splits on `\\n\\n` across network reads loses events whose boundary is in the
middle of a chunk.

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

    It is a Protocol and not `httpx.Response` for the same reason that
    `AgentHandle` is one. It is the surface that the reader actually reads, and
    the reader has no other use for the client library. `test_adapters._Response`
    is the second implementation. If a stub could not stand in, every SSE test
    would be a test of httpx.
    """

    def aiter_text(self) -> AsyncIterator[str]: ...


async def _blocks(response: TextStream) -> AsyncIterator[str]:
    """Each `\\n\\n`-terminated block, and then what the stream held when it ended.

    **Line endings are normalized, and the last block is not dropped** (G7). The
    spec allows `\\r\\n`, `\\n` and `\\r` as terminators, and pH read only
    `\\n\\n`. Some gateways and proxies emit a CRLF stream. Such a stream arrived
    as one buffer that never contained a boundary, and it yielded *nothing at
    all* for the whole of a turn. A server that closes without a trailing blank
    line left its final block behind. On a short reply that block is the entire
    answer, and on a long reply it is the `finish` that settles the turn.

    This function does framing only. What a block *means* belongs to `iter_sse`.
    That keeps the end-of-stream rule in one place and not one for each caller.
    """
    buffer = ""
    pending = ""
    async for chunk in response.aiter_text():
        # **A trailing `\r` is held back**, because the chunk boundary is the one
        # thing that this function owns. When each chunk was normalized on its
        # own, a `\r\n` split across two reads became two terminators. That cuts
        # one event in half and delivers its `data:` under the wrong event name.
        chunk = pending + chunk
        pending = chunk[-1] if chunk.endswith("\r") else ""
        if pending:
            chunk = chunk[:-1]
        buffer += chunk.replace("\r\n", "\n").replace("\r", "\n")
        while "\n\n" in buffer:
            raw, buffer = buffer.split("\n\n", 1)
            yield raw
    buffer += pending.replace("\r", "\n")
    if buffer:
        yield buffer


async def iter_sse(response: TextStream) -> AsyncIterator[tuple[str, JsonValue]]:
    """Yield `(event, data)` pairs from an SSE response.

    `data` is parsed JSON, or the raw string when the payload is not JSON. A
    `[DONE]` payload ends the iteration and is not yielded.
    """
    async for raw in _blocks(response):
        event = "message"
        payloads: list[str] = []
        for line in raw.splitlines():
            if line.startswith("event:"):
                event = line[len("event:") :].strip()
            elif line.startswith("data:"):
                payloads.append(line[len("data:") :].strip())
        if not payloads:
            # A comment, a `retry:`, or a trailing fragment of a stream that was
            # cut off part of the way through. None of them is an event.
            continue
        body = "\n".join(payloads)
        if body == "[DONE]":
            return
        try:
            data: JsonValue = loads(body)
        except json.JSONDecodeError:
            data = body
        yield event, data
