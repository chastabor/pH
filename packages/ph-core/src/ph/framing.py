"""JSONL framing over a byte stream: one JSON object a line, `\\n`-terminated, UTF-8.

The daemon's wire, the session log's format, what `--mode rpc` speaks over stdio
— and the wire between a pH session and the tool commands it runs: a `call`
asking a running host (`ph.toolcli`), a confined command asking its session for
vectors (`ph_text_index`). One format, parsed one way, so a client, a log reader
and a peer agree about where a message ends (I-7).

**Bounded on purpose.** A stream socket hands you bytes, not messages, so a peer
that never sends a newline is a peer that grows your buffer until the process
dies. `MAX_LINE` turns that into a refused connection with a reason.

@module ph.framing
"""

from __future__ import annotations

import json
import socket
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream
from anyio.streams.buffered import BufferedByteReceiveStream

from .json import dumps

__all__ = [
    "MAX_LINE",
    "FramingError",
    "answer_one",
    "ask",
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


async def answer_one(
    stream: ByteStream, respond: Callable[[dict[str, Any]], Awaitable[Mapping[str, object]]]
) -> None:
    """Read one request from `stream`, send what `respond` makes of it, and close.

    For a server that answers one request a connection — a tool host, a session
    embedding for a command. A peer that goes away, or sends something that is
    not a frame, ends its own connection and nothing else: there is nobody left
    to answer, and the server goes on serving the next."""
    try:
        async with stream:
            async for frame in read_frames(stream):
                await write_frame(stream, await respond(frame))
                return
    except (FramingError, anyio.BrokenResourceError, anyio.ClosedResourceError):
        pass


async def listening(path: str | Path, *, timeout: float | None = None) -> bool:
    """Whether something answers at `path` right now.

    A connect and an immediate close — the only test that tells "a socket file
    exists" from "a server is behind it", which the daemon's launcher and a tool
    host both decide on. One copy, so a change here reaches both. `timeout`
    bounds the connect, for a listener that is hung."""
    try:
        with anyio.fail_after(timeout):
            stream = await anyio.connect_unix(str(path))
    except (OSError, TimeoutError):
        return False
    await stream.aclose()
    return True


def ask(
    path: str | Path, payload: Mapping[str, object], *, connect_timeout: float | None = None
) -> dict[str, Any]:
    """Send one request over the unix socket at `path` and return its one answer.

    Blocking, for a caller with no event loop of its own — a command deciding
    whether a host can answer it before it mounts anything — or on a worker
    thread. `connect_timeout` bounds reaching the server and nothing after: how
    long the answer may take is the caller's to say, by ending the process.

    :raises OSError: when nothing answers at `path`, or the connection drops.
    :raises FramingError: when the answer is not one frame.
    """
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as link:
        link.settimeout(connect_timeout)
        link.connect(str(path))
        link.settimeout(None)
        link.sendall(f"{dumps(payload)}\n".encode())
        with link.makefile("rb") as answers:
            line = answers.readline(MAX_LINE + 1)
    if len(line) > MAX_LINE:
        raise FramingError(f"frame exceeds {MAX_LINE} bytes")
    if not line.strip():
        raise FramingError("the peer closed without answering")
    return _parsed(line)


def _parsed(line: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(line)
    except json.JSONDecodeError as error:
        raise FramingError(f"not JSON: {error}") from error
    if not isinstance(payload, dict):
        raise FramingError("a frame must be a JSON object")
    return payload
