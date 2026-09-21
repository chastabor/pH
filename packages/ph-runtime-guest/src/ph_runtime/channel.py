"""Newline-delimited JSON over one duplex descriptor.

`json.dumps` never emits a literal newline (it escapes them inside strings), so
a line is exactly a frame and no length prefix or escaping layer is needed.

The read limit is generous because two frames are legitimately large: a program
the model wrote, and a `snapshot` carrying `dill` payloads. It is a cap on one
line, not an allocation.

@module ph_runtime.channel
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
from datetime import date, datetime
from datetime import time as time_of_day
from decimal import Decimal
from pathlib import PurePath
from typing import Any

from .protocol import FD_ENV, PROTOCOL_FD

__all__ = ["MAX_FRAME_BYTES", "Channel", "jsonable"]

MAX_FRAME_BYTES = 64 * 1024 * 1024


def jsonable(value: object) -> object:
    """One non-JSON value as something the host can use (F7).

    `default=repr` was the whole rule, and it is silently lossy for the two
    types a cell passes most: `tools.read(Path("notes.md"))` arrived as the
    string `"PosixPath('notes.md')"`, and a `datetime` as
    `"datetime.datetime(2026, 9, 20, 0, 0)"`. Neither is an error anywhere — the
    tool receives a string where the model wrote a value, and either refuses it
    with a message about a path that does not exist or, worse, uses it.

    The obvious spellings for the types that have one, and `repr` for the rest:
    a value with no JSON form is a value the tool's own schema is going to
    reject, and `repr` is what makes *that* refusal readable. Deliberately not a
    guest-side raise — the guest does not know the tool's schema, and refusing
    an argument a tool would have accepted is the worse error.

    It lives here, on the channel, because the lossiness is a property of *the
    wire* and not of one frame: an argument, a cell's value and every other
    frame leave through `send`, and a `Path` that survives as one and not the
    others is the same bug reported twice.
    """
    if isinstance(value, PurePath):
        return str(value)
    if isinstance(value, datetime | date | time_of_day):
        return value.isoformat()
    if isinstance(value, set | frozenset | tuple):
        return list(value)
    if isinstance(value, Decimal):
        return str(value)
    return repr(value)


class Channel:
    """The framed channel, and the only way out of this process to the host."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    @classmethod
    async def open(cls, fd: int | None = None) -> Channel:
        """Attach to the inherited descriptor.

        Wrapped in a `socket` object rather than opened as a file because the
        host hands over one end of a `socketpair`: a pipe would be one-way, and
        the guest has to both answer the host and call it.
        """
        if fd is None:
            fd = int(os.environ.get(FD_ENV, PROTOCOL_FD))
        sock = socket.socket(fileno=fd)
        sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=sock, limit=MAX_FRAME_BYTES)
        return cls(reader, writer)

    async def receive(self) -> dict[str, Any] | None:
        """The next frame, or `None` when the host has gone.

        A line that will not parse is skipped rather than fatal: the host is
        trusted for *content*, but a truncated write at shutdown should end the
        session quietly, not with a traceback into the log.
        """
        while True:
            try:
                line = await self._reader.readline()
            except (asyncio.IncompleteReadError, ConnectionResetError, ValueError, OSError):
                return None
            if not line:
                return None
            text = line.strip()
            if not text:
                continue
            try:
                frame = json.loads(text)
            except ValueError:
                continue
            if isinstance(frame, dict):
                return frame

    def send(self, frame: dict[str, Any]) -> None:
        """Queue one frame. Synchronous, so `print` inside a cell can call it."""
        # A dead host is not this process's problem to report: the
        # die-with-parent mechanism is what ends the guest (F3).
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            self._writer.write(json.dumps(frame, default=jsonable).encode("utf-8") + b"\n")

    async def drain(self) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            await self._writer.drain()

    async def aclose(self) -> None:
        await self.drain()
        with contextlib.suppress(RuntimeError):
            self._writer.close()
