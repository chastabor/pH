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


async def _blocks(response: TextStream) -> AsyncIterator[str]:
    """Each `\n\n`-terminated block, and then whatever the stream ended holding.

    **Line endings are normalized, and the last block is not dropped** (G7). The
    spec allows `\r\n`, `\n` and `\r` as terminators and pH read only `\n\n`,
    so a CRLF stream — which is what some gateways and proxies emit — arrived as
    one buffer that never contained a boundary and yielded *nothing at all*, for
    the whole of a turn. And a server that closes without a trailing blank line
    left its final block behind: on a short reply that is the entire answer, and
    on a long one it is the `finish` that settles the turn.

    Framing only. What a block *means* is `iter_sse`'s, which is what keeps the
    end-of-stream rule in one place rather than one per caller.
    """
    buffer = ""
    pending = ""
    async for chunk in response.aiter_text():
        # **A trailing `\r` is held back**, because the chunk boundary is the one
        # thing this function owns: normalizing per chunk turned a `\r\n` split
        # across two reads into two terminators, which cuts one event in half and
        # delivers its `data:` under the wrong event name.
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

    `data` is parsed JSON, or the raw string when it is not JSON (`[DONE]`).
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
            # A comment, a `retry:`, or a trailing fragment the stream was cut
            # off mid-way through. None of them is an event.
            continue
        body = "\n".join(payloads)
        if body == "[DONE]":
            return
        try:
            data: JsonValue = loads(body)
        except json.JSONDecodeError:
            data = body
        yield event, data
