"""JSONL framing over a byte stream: one JSON object a line, `\\n`-terminated, UTF-8.

This is the wire of the daemon, the format of the session log, and what
`--mode rpc` speaks over stdio. There is one format, parsed one way, so a
client, a log reader and a peer agree about where a message ends (I-7).

**Bounded on purpose.** A stream socket delivers bytes, not messages. A peer
that never sends a newline therefore grows the buffer until the process dies.
`MAX_LINE` turns that into a refused connection with a reason.

@module ph.framing
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream
from anyio.streams.buffered import BufferedByteReceiveStream

from .json import dumps

__all__ = [
    "MAX_LINE",
    "FramingError",
    "listening",
    "read_frames",
    "write_frame",
]

MAX_LINE = 8 * 1024 * 1024
"""How long one frame may be. The limit is on one *frame*, and `receive_until`
checks it.

The limit is generous, because a frame can carry a session event, and an event
can carry a tool result that the spill store has not yet taken. It is bounded,
because the alternative is not "large frames work" but "one peer can exhaust the
process".
"""


class FramingError(Exception):
    """The peer sent something that this transport cannot read.

    This is distinct from a *protocol* error. A request whose method is unknown
    is for the server to answer. A frame that is not a frame ends the
    connection, because after it there is no way to know where the next frame
    starts.
    """


async def write_frame(stream: ByteStream, payload: Mapping[str, object]) -> None:
    """Send one object.

    The type is `Mapping[str, object]` and not a protocol type, so that this
    module stays the transport and not the vocabulary. Every `TypedDict` is a
    `Mapping`, and a module that framed only the shapes that it knew would have
    to be told about the next one.

    The encoder is `ph.json.dumps` and not `json.dumps`, because it is the
    canonical encoder. A peer that framed events differently from the log that
    they came from would make the claim of one format false.
    """
    await stream.send(f"{dumps(payload)}\n".encode())


async def read_frames(stream: ByteStream) -> AsyncIterator[dict[str, Any]]:
    """Yield each object that the peer sends, until it closes.

    This uses `BufferedByteReceiveStream.receive_until` and not a hand-written
    buffer, because anyio ships the delimiter scan with the cap built in. The
    hand-written version was quadratic in two ways: it scanned the whole buffer
    again for each chunk, and it copied the tail again for each frame. It also
    bounded the *accumulated buffer* and not one frame, so many small frames in
    one chunk hit a limit that was documented as per-frame.

    A peer that closes mid-frame **ends the iteration and does not raise**. The
    peer did not send that frame. A supervisor that acts on half of a frame
    executes a command that nobody completed.

    **The cap is checked twice, and the second check is the one that makes it a
    cap.** `receive_until` looks for the delimiter *before* it tests the buffer
    against `max_bytes`. So an over-length frame is refused only when its
    newline has not yet arrived, and that depends on how much the socket
    delivers per read. `net.local.stream.recvspace` is 64 KiB on Linux and 8 KiB
    on macOS, so the same oversized frame was refused on one and accepted on the
    other (measured 2026-09-07, on a 6 MiB attachment that Linux let through and
    macOS closed the connection over). A wire limit that holds on one kernel is
    not a limit.
    """
    buffered = BufferedByteReceiveStream(stream)
    while True:
        try:
            line = await buffered.receive_until(b"\n", MAX_LINE)
        except (anyio.EndOfStream, anyio.IncompleteRead):
            return
        except anyio.DelimiterNotFound as error:
            raise FramingError(f"frame exceeds {MAX_LINE} bytes") from error
        if len(line) >= MAX_LINE:
            # The delimiter arrived inside a chunk that took the buffer past the
            # cap. Same refusal, same sentence, on every platform.
            raise FramingError(f"frame exceeds {MAX_LINE} bytes")
        if not line.strip():
            continue
        yield _parsed(line)


async def listening(path: str | Path, *, timeout: float | None = None) -> bool:
    """Whether something answers at `path` right now.

    The test is a connect and an immediate close. It is the only test that
    distinguishes "a socket file exists" from "a server is behind it". The
    launcher of the daemon and its server both decide on it. There is one copy,
    so a change here reaches both. `timeout` bounds the connect, for a listener
    that is hung."""
    try:
        with anyio.fail_after(timeout):
            stream = await anyio.connect_unix(str(path))
    except (OSError, TimeoutError):
        return False
    await stream.aclose()
    return True


def _parsed(line: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except (ValueError, RecursionError) as error:
        # `ValueError` covers both ways in which a line is not JSON: bytes that
        # are not UTF-8, and text that does not parse. Nesting deep enough to
        # exhaust the parser is a third way. The peer is responsible for all
        # three.
        raise FramingError(f"not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FramingError("a frame must be a JSON object")
    return payload
