"""Newline-delimited JSON over one duplex descriptor.

`json.dumps` never emits a literal newline, because it escapes newlines inside
strings. A line is therefore exactly one frame, and no length prefix or escaping
layer is needed.

The read limit is large because two kinds of frame can be large for good reason:
a program that the model wrote, and a `snapshot` that carries `dill` payloads.
The limit is a cap on one line, not an allocation.

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
from typing import Any, Protocol, TypeGuard

from .protocol import FD_ENV, FRAME_BYTES_ENV, PROTOCOL_FD

__all__ = ["MAX_FRAME_BYTES", "Channel", "Model", "frame_limit", "is_model", "jsonable"]

MAX_FRAME_BYTES = 64 * 1024 * 1024
"""The read limit when the host gives none. See `FRAME_BYTES_ENV` (F8)."""


def frame_limit() -> int:
    """The largest frame that this guest reads: the number from the host, or the
    default.

    The host sizes the limit from `maxSnapshotBytes`, because one snapshotted
    variable must fit in one frame in *both* directions. With a fixed 64 MiB
    here, a restore lost the namespace as soon as a deployment raised the limit
    past about 48 MiB, because base64 uses four bytes for every three (F8). A
    value that is not a positive integer is ignored, because a limit of zero
    would refuse every frame.
    """
    try:
        named = int(os.environ.get(FRAME_BYTES_ENV, ""))
    except ValueError:
        return MAX_FRAME_BYTES
    return named if named > 0 else MAX_FRAME_BYTES


class Model(Protocol):
    """A pydantic model class, recognized by its shape. This venv does not need to
    hold pydantic. Model code that defines such a class brought pydantic itself."""

    @classmethod
    def model_json_schema(cls) -> dict[str, Any]: ...

    @classmethod
    def model_validate(cls, obj: Any) -> Any: ...  # noqa: ANN401


def is_model(value: object) -> TypeGuard[type[Model]]:
    """Whether `value` is a pydantic model class."""
    return (
        isinstance(value, type)
        and callable(getattr(value, "model_json_schema", None))
        and callable(getattr(value, "model_validate", None))
    )


def jsonable(value: object) -> object:
    """Convert one non-JSON value into something that the host can use (F7).

    The whole rule was once `default=repr`. That rule loses information silently
    for the two types that a cell passes most often. `tools.read(Path("notes.md"))`
    arrived as the string `"PosixPath('notes.md')"`, and a `datetime` arrived as
    `"datetime.datetime(2026, 9, 20, 0, 0)"`. Neither is an error anywhere. The
    tool receives a string where the model wrote a value. The tool then refuses
    it with a message about a path that does not exist, or, worse, uses it.

    This function gives the obvious spelling for each type that has one, and
    `repr` for the rest. A value with no JSON form is a value that the schema of
    the tool will reject, and `repr` makes *that* refusal readable. The guest
    deliberately does not raise. It does not know the schema of the tool, and
    refusing an argument that a tool would accept is the worse error.

    The function is here, on the channel, because the loss is a property of *the
    wire* and not of one frame. An argument, the value of a cell, and every other
    frame leave through `send`. A `Path` that survives in one frame and not in
    the others would be the same bug reported twice.

    A pydantic model class becomes the shape that it declares, its JSON Schema. A
    tool that takes a model class holds a value to that schema (the `output` of
    `rlm.run`). A model instance becomes its JSON. A cell gets such an instance
    from `rlm.wait`.
    """
    if is_model(value):
        return value.model_json_schema()
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
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
    """The framed channel. It is the only way from this process to the host."""

    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self._reader = reader
        self._writer = writer

    @classmethod
    async def open(cls, fd: int | None = None) -> Channel:
        """Attach to the inherited descriptor.

        The descriptor is wrapped in a `socket` object and not opened as a file,
        because the host passes one end of a `socketpair`. A pipe is one-way, and
        the guest must both answer the host and call it.
        """
        if fd is None:
            fd = int(os.environ.get(FD_ENV, PROTOCOL_FD))
        sock = socket.socket(fileno=fd)
        sock.setblocking(False)
        reader, writer = await asyncio.open_connection(sock=sock, limit=frame_limit())
        return cls(reader, writer)

    async def receive(self) -> dict[str, Any] | None:
        """Return the next frame, or `None` when the host is gone.

        A line that does not parse is skipped and is not fatal. The host is
        trusted for *content*. But a truncated write at shutdown must end the
        session quietly, without a traceback in the log.
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
        """Queue one frame. This method is synchronous, so that `print` inside a
        cell can call it."""
        # This process does not report a dead host. The die-with-parent mechanism
        # is what ends the guest (F3).
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            self._writer.write(json.dumps(frame, default=jsonable).encode("utf-8") + b"\n")

    async def drain(self) -> None:
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, RuntimeError):
            await self._writer.drain()

    async def aclose(self) -> None:
        await self.drain()
        with contextlib.suppress(RuntimeError):
            self._writer.close()
