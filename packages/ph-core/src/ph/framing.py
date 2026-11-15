"""JSONL framing over a byte stream: one JSON object a line, `\\n`-terminated, UTF-8.

The daemon's wire, the session log's format, and what `--mode rpc` speaks over
stdio. One format, parsed one way, so a client, a log reader and a peer agree
about where a message ends (I-7).

**Bounded on purpose.** A stream socket hands you bytes, not messages, so a peer
that never sends a newline is a peer that grows your buffer until the process
dies. `MAX_LINE` turns that into a refused connection with a reason.

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
"""How long one frame may be — one *frame*, checked by `receive_until`.

Generous, because a frame can carry a session event and an event can carry a
tool result the spill store has not yet taken. Bounded, because the alternative
is not "large frames work" but "one peer can exhaust the process".
"""


class FramingError(Exception):
    """The peer sent something this transport cannot read.

    Distinct from a *protocol* error — a request whose method is unknown is the
    server's to answer, and a frame that is not a frame ends the connection,
    because after it there is no way to know where the next one starts.
    """


async def write_frame(stream: ByteStream, payload: Mapping[str, object]) -> None:
    """Send one object.

    `Mapping[str, object]` rather than a protocol type, so this stays the
    transport and not the vocabulary: every `TypedDict` is a `Mapping`, and a
    module that framed only the shapes it knew about would have to be told about
    the next one.

    `ph.json.dumps` rather than `json.dumps`: it is the canonical encoder, and a
    peer that framed events differently from the log they came out of would make
    the two formats one claim short of true.
    """
    await stream.send(f"{dumps(payload)}\n".encode())


async def read_frames(stream: ByteStream) -> AsyncIterator[dict[str, Any]]:
    """Yield each object the peer sends, until it closes.

    `BufferedByteReceiveStream.receive_until` rather than a hand-rolled buffer: anyio
    ships the delimiter scan with the cap built in. The hand-rolled version was
    quadratic twice over — it re-scanned the whole buffer per chunk and recopied the
    tail per frame — and it bounded the *accumulated buffer* rather than one frame, so
    many small frames in one chunk tripped a limit documented as per-frame.

    A peer that closes mid-frame **ends the iteration rather than raising**: it did not
    send that frame, and acting on half of one is how a supervisor executes a command
    nobody completed.

    **The cap is checked twice, and the second check is the one that makes it a cap.**
    `receive_until` looks for the delimiter *before* it tests the buffer against
    `max_bytes`, so an over-length frame is refused only when its newline has not yet
    arrived — which depends on how much the socket hands over per read.
    `net.local.stream.recvspace` is 64 KiB on Linux and 8 KiB on macOS, so the same
    oversized frame was refused on one and accepted on the other (measured 2026-09-07,
    on a 6 MiB attachment that Linux let through and macOS closed the connection over).
    A wire limit that holds on one kernel is not a limit.
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

    A connect and an immediate close — the only test that tells "a socket file
    exists" from "a server is behind it", which the daemon's launcher and its
    server both decide on. One copy, so a change here reaches both. `timeout`
    bounds the connect, for a listener that is hung."""
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
        # `ValueError` is both of a line's ways not to be JSON: bytes that are not
        # UTF-8, and text that does not parse. Nesting deep enough to exhaust the
        # parser is a third, and all three are the peer's to answer for.
        raise FramingError(f"not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FramingError("a frame must be a JSON object")
    return payload
