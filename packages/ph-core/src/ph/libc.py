"""One handle on the C library, for the ph-core modules that reach it through ctypes.

`ph.wall_clock` (`timerfd`) and `ph.path_watch` (`inotify`) call Linux interfaces
the standard library does not wrap at the 3.12 floor. Each used to open its own
`CDLL` and spell its own errno-to-`OSError` translation; the second caller is the
moment to stop. `ph_runtime.lifecycle` keeps its own handle: the guest is
dependency-free and does not import ph-core.

`CDLL(None)` is the running process's own symbol table, which on glibc and musl
alike includes libc, and on macOS resolves too (nothing there calls it today).
`use_errno=True` is what makes `failed` able to say why.

@module ph.libc
"""

from __future__ import annotations

import ctypes
import os

__all__ = ["LIBC", "failed"]

LIBC = ctypes.CDLL(None, use_errno=True)


def failed(call: str) -> OSError:
    """The `OSError` for the libc `call` that just returned failure, by its errno."""
    code = ctypes.get_errno()
    return OSError(code, f"{call}: {os.strerror(code)}")
