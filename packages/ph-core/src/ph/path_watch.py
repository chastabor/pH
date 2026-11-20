"""Watch one directory entry for change, on notice from the kernel (P12-05).

The daemon must notice when the socket at its path stops being its own. That
happens when logout removes `$XDG_RUNTIME_DIR`, or when a second daemon binds a
new socket where the old one was (`ph.lingering.socket_identity` tells those
cases apart). The daemon used to `lstat` that path every thirty seconds for the
life of the process. With `EntryWatch`, the kernel reports when the entry may
have moved, and the caller looks only then. Nothing in this module is about
sockets. The watch reports "may have changed", and the check of the caller
decides what changed.

* **Linux**: `inotify`, through ctypes on `ph.libc`. The directory of the entry
  is watched for entries that are made, removed, or renamed in or out, and for
  the removal or unmount of the directory itself. Every ancestor up to `/` is
  watched for removal and for rename, because a move of any ancestor moves the
  path too. Events that name another entry in the directory are ignored.
* **macOS**: one kqueue `EVFILT_VNODE` for each directory, on a descriptor that
  is opened `O_EVTONLY`. `NOTE_WRITE` on a directory says only that *some* entry
  changed, so every change in the directory wakes the reader.
* **Any other platform, or when arming fails** (`ENOSPC` and `EMFILE` are the
  limits of inotify): the constructor raises `WatchUnavailable`.

**The watch is armed in the constructor**, synchronously. A caller can therefore
arm it immediately after it captures the value that it will compare against, and
nothing can change unseen between the two steps.

**What the watch cannot see**: a filesystem that is mounted over the directory
produces no event on either kernel. `docs/dev-notes/linux-macos-differences.md`
§10 records this.

@module ph.path_watch
"""

from __future__ import annotations

import ctypes
import logging
import os
import select
import struct
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Literal, TypeAlias

import anyio

from .libc import LIBC, failed

__all__ = ["EntryWatch", "Mechanism", "WatchUnavailable"]

log = logging.getLogger("ph.path_watch")

Mechanism: TypeAlias = Literal["inotify", "kqueue"]
"""The kernel interface that notifies the `EntryWatch` of a platform.
`phern agents doctor` reports it."""


class WatchUnavailable(Exception):
    """The entry cannot be watched here. The message says why."""


class _Watch:
    """The arming that every platform shares: the directory of the entry, then
    its ancestors.

    A platform supplies `_open` (its kernel handle), `_add` (one directory, with
    either the entry flags or the ancestor flags), `changes` and `close`.
    """

    mechanism: Mechanism

    def __init__(self, path: Path) -> None:
        try:
            self._open()
            self._directory = self._add(path.parent, entries=True)
        except OSError as error:
            self.close()
            raise WatchUnavailable(str(error)) from error
        self._name = os.fsencode(path.name)
        for ancestor in path.parent.parents:
            # This is best effort. An ancestor that this user cannot read is also
            # one that this user cannot rename, and a rename is the case that the
            # ancestor watches exist for.
            try:
                self._add(ancestor, entries=False)
            except OSError as error:
                log.debug("ph.path_watch: not watching %s: %s", ancestor, error)

    def _open(self) -> None:
        raise NotImplementedError

    def _add(self, directory: Path, *, entries: bool) -> int:
        raise NotImplementedError

    async def changes(self) -> AsyncIterator[None]:
        """Yield each time the entry may have changed. End when its directory is
        no longer watched."""
        raise NotImplementedError
        yield  # pragma: no cover — makes this an async generator for the checker

    def close(self) -> None:
        raise NotImplementedError


if sys.platform == "linux":
    _IN_MOVED_FROM = 0x00000040
    _IN_MOVED_TO = 0x00000080
    _IN_CREATE = 0x00000100
    _IN_DELETE = 0x00000200
    _IN_DELETE_SELF = 0x00000400
    _IN_MOVE_SELF = 0x00000800
    _IN_UNMOUNT = 0x00002000
    _IN_Q_OVERFLOW = 0x00004000
    _IN_IGNORED = 0x00008000

    _ENTRY = _IN_CREATE | _IN_DELETE | _IN_MOVED_FROM | _IN_MOVED_TO
    _SELF = _IN_DELETE_SELF | _IN_MOVE_SELF
    _ANYTHING = _IN_Q_OVERFLOW | _SELF | _IN_UNMOUNT | _IN_IGNORED
    """Events that may have moved the entry, whatever they name: a directory or
    an ancestor that was removed, renamed or unmounted, or dropped events (an
    overflowed queue comes with `wd` -1)."""
    _GONE = _IN_DELETE_SELF | _IN_IGNORED | _IN_UNMOUNT
    """Events after which the directory of the entry is no longer watched."""
    _HEADER = struct.Struct("iIII")
    """`struct inotify_event` without its name: `wd`, `mask`, `cookie`, `len`."""

    LIBC.inotify_init1.argtypes = [ctypes.c_int]
    LIBC.inotify_init1.restype = ctypes.c_int
    LIBC.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
    LIBC.inotify_add_watch.restype = ctypes.c_int

    class EntryWatch(_Watch):
        """Reports when the directory entry of a path may have changed (inotify)."""

        mechanism: Mechanism = "inotify"
        _fd: int = -1

        def _open(self) -> None:
            # `IN_NONBLOCK` and `IN_CLOEXEC` are defined as the `O_` flags.
            self._fd = LIBC.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
            if self._fd < 0:
                raise failed("inotify_init1")

        def _add(self, directory: Path, *, entries: bool) -> int:
            mask = _ENTRY | _SELF if entries else _SELF
            wd: int = LIBC.inotify_add_watch(self._fd, os.fsencode(directory), mask)
            if wd < 0:
                raise failed(f"inotify_add_watch({directory})")
            return wd

        async def changes(self) -> AsyncIterator[None]:
            while True:
                await anyio.wait_readable(self._fd)
                try:
                    data = os.read(self._fd, 64 * 1024)
                except BlockingIOError:
                    continue
                moved, gone = self._read(data)
                if moved:
                    yield
                if gone:
                    return

        def _read(self, data: bytes) -> tuple[bool, bool]:
            """Whether these events may have moved the entry, and whether its
            directory is gone."""
            moved = gone = False
            offset = 0
            while offset < len(data):
                wd, mask, _cookie, length = _HEADER.unpack_from(data, offset)
                name = data[offset + _HEADER.size : offset + _HEADER.size + length].rstrip(b"\0")
                offset += _HEADER.size + length
                if wd == self._directory and mask & _ENTRY:
                    moved = moved or name == self._name
                elif mask & _ANYTHING:
                    moved = True
                if wd == self._directory and mask & _GONE:
                    gone = True
            return moved, gone

        def close(self) -> None:
            if self._fd >= 0:
                os.close(self._fd)
                self._fd = -1

elif sys.platform == "darwin":
    _GONE_FFLAGS = select.KQ_NOTE_DELETE | select.KQ_NOTE_REVOKE
    _ANCESTOR_FFLAGS = _GONE_FFLAGS | select.KQ_NOTE_RENAME

    class EntryWatch(_Watch):
        """Reports when the directory entry of a path may have changed (kqueue)."""

        mechanism: Mechanism = "kqueue"
        _queue: select.kqueue | None = None

        def _open(self) -> None:
            self._fds: list[int] = []
            self._queue = select.kqueue()

        def _add(self, directory: Path, *, entries: bool) -> int:
            assert self._queue is not None
            fd = os.open(directory, os.O_EVTONLY)
            self._fds.append(fd)
            fflags = _ANCESTOR_FFLAGS | select.KQ_NOTE_WRITE if entries else _ANCESTOR_FFLAGS
            event = select.kevent(
                fd,
                filter=select.KQ_FILTER_VNODE,
                flags=select.KQ_EV_ADD | select.KQ_EV_CLEAR,
                fflags=fflags,
            )
            self._queue.control([event], 0, 0)
            return fd

        async def changes(self) -> AsyncIterator[None]:
            assert self._queue is not None
            while True:
                await anyio.wait_readable(self._queue.fileno())
                events = self._queue.control(None, 64, 0)
                if not events:
                    continue
                yield
                if any(e.ident == self._directory and e.fflags & _GONE_FFLAGS for e in events):
                    return

        def close(self) -> None:
            for fd in getattr(self, "_fds", ()):
                os.close(fd)
            self._fds = []
            if self._queue is not None:
                self._queue.close()
                self._queue = None

else:

    class EntryWatch(_Watch):
        """There is no watch on this platform, and arming reports that."""

        def _open(self) -> None:
            raise OSError(f"no filesystem watch on {sys.platform}")

        def close(self) -> None:
            return
