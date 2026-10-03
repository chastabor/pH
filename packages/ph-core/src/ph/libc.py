"""One handle on the C library, for the ph-core modules that reach it through ctypes.

`ph.wall_clock` (`timerfd`) and `ph.path_watch` (`inotify`) call Linux interfaces
the standard library does not wrap at the 3.12 floor. Each used to open its own
`CDLL` and spell its own errno-to-`OSError` translation; the second caller is the
moment to stop. `ph_runtime.lifecycle` keeps its own handle: the guest is
dependency-free and does not import ph-core.

`CDLL(None)` is the running process's own symbol table, which on glibc and musl
alike includes libc, and on macOS resolves too — where `ph.orphans` reads a pid's
start time through `sysctl`. `use_errno=True` is what makes `failed` able to say why.

@module ph.libc
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence

__all__ = ["LIBC", "failed", "sysctl"]

LIBC = ctypes.CDLL(None, use_errno=True)


def failed(call: str) -> OSError:
    """The `OSError` for the libc `call` that just returned failure, by its errno."""
    code = ctypes.get_errno()
    return OSError(code, f"{call}: {os.strerror(code)}")


def sysctl(name: Sequence[int], size: int) -> bytes:
    """The value of the MIB `name`, read into at most `size` bytes: what the kernel
    wrote, which is nothing when the name has no value (a pid that has gone).

    macOS and the BSDs. glibc dropped `sysctl(2)` in 2.32, so on Linux the symbol is
    missing and this raises `AttributeError`.

    :raises OSError: when the call fails, `ENOMEM` for a value larger than `size`.
    """
    mib = (ctypes.c_int * len(name))(*name)
    buffer = ctypes.create_string_buffer(size)
    written = ctypes.c_size_t(size)
    status = LIBC.sysctl(
        mib, ctypes.c_uint(len(name)), buffer, ctypes.byref(written), None, ctypes.c_size_t(0)
    )
    if status != 0:
        raise failed("sysctl")
    return buffer.raw[: written.value]
