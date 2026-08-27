"""Server-sent events, read incrementally.

Both provider APIs pH speaks stream SSE. Shared here because the framing is the
same and the bug is the same: a naive reader that splits on `\\n\\n` across
network reads loses events whose boundary lands mid-chunk.

@module ph_app.adapters.sse
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

__all__ = ["iter_sse"]


async def iter_sse(response: Any) -> AsyncIterator[tuple[str, Any]]:
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
                yield event, json.loads(body)
            except json.JSONDecodeError:
                yield event, body
