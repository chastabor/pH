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
from typing import Any

from .protocol import FD_ENV, PROTOCOL_FD

__all__ = ["MAX_FRAME_BYTES", "Channel"]

MAX_FRAME_BYTES = 64 * 1024 * 1024


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
            self._writer.write(json.dumps(frame, default=repr).encode("utf-8") + b"\n")

    async def drain(self) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            await self._writer.drain()

    async def aclose(self) -> None:
        await self.drain()
        with contextlib.suppress(RuntimeError):
            self._writer.close()
