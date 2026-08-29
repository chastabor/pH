"""JSONL framing over a byte stream — the daemon's wire (P5-01).

One JSON object per line, `\\n`-terminated, UTF-8. The same framing the session
log uses and the same framing `--mode rpc` speaks over stdio, so a client, a log
reader and a daemon peer all parse one format (I-7) and the daemon is a
*transport* change rather than a second protocol.

**Bounded on purpose.** A stream socket hands you bytes, not messages, so a peer
that never sends a newline is a peer that grows your buffer until the process
dies — and the daemon is the one process in pH that outlives the thing that
talked to it. `MAX_LINE` turns that into a refused connection with a reason.

@module ph_app.daemon.framing
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import anyio
from anyio.abc import ByteStream
from anyio.streams.buffered import BufferedByteReceiveStream

from ph.session import dumps

__all__ = ["MAX_LINE", "FramingError", "read_frames", "write_frame"]

MAX_LINE = 8 * 1024 * 1024
"""How long one frame may be — one *frame*, checked by `receive_until`.

Generous, because a frame can carry a session event and an event can carry a
tool result the spill store has not yet taken. Bounded, because the alternative
is not "large frames work" but "one peer can exhaust the daemon".
"""


class FramingError(Exception):
    """The peer sent something this transport cannot read.

    Distinct from a *protocol* error — a request whose method is unknown is the
    server's to answer, and a frame that is not a frame ends the connection,
    because after it there is no way to know where the next one starts.
    """


async def write_frame(stream: ByteStream, payload: dict[str, Any]) -> None:
    """Send one object.

    `ph.session.dumps` rather than `json.dumps`: it is the canonical encoder,
    and a daemon that framed events differently from the log they came out of
    would make the two formats one claim short of true.
    """
    await stream.send(f"{dumps(payload)}\n".encode())


async def read_frames(stream: ByteStream) -> AsyncIterator[dict[str, Any]]:
    """Yield each object the peer sends, until it closes.

    `BufferedByteReceiveStream.receive_until` rather than a hand-rolled buffer:
    anyio ships the delimiter scan with the cap built in, and the hand-rolled
    version was quadratic twice over — it re-scanned the whole buffer per chunk
    and recopied the tail per frame, which measured 57 ms for 4 096 frames
    arriving in one read. It also bounded the *accumulated buffer* rather than
    one frame, so many small frames in one chunk tripped a limit documented as
    per-frame.

    A peer that closes mid-frame ends the iteration rather than raising: it did
    not send that frame, and acting on half of one is how a supervisor executes
    a command nobody completed.
    """
    buffered = BufferedByteReceiveStream(stream)
    while True:
        try:
            line = await buffered.receive_until(b"\n", MAX_LINE)
        except (anyio.EndOfStream, anyio.IncompleteRead):
            return
        except anyio.DelimiterNotFound as error:
            raise FramingError(f"frame exceeds {MAX_LINE} bytes") from error
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise FramingError(f"not JSON: {error}") from error
        if not isinstance(payload, dict):
            raise FramingError("a frame must be a JSON object")
        yield payload
