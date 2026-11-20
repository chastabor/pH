"""One handle on the C library, for the ph-core modules that reach it through ctypes.

`ph.wall_clock` (`timerfd`) and `ph.path_watch` (`inotify`) call Linux interfaces
that the standard library does not wrap at the 3.12 floor. Each module used to
open its own `CDLL` and write its own translation from errno to `OSError`. The
second caller is the time to stop that. `ph_runtime.lifecycle` keeps its own
handle, because the guest has no dependencies and does not import ph-core.

`CDLL(None)` is the symbol table of the running process itself. On glibc and on
musl, that table includes libc. It also resolves on macOS, where `ph.orphans`
reads the start time of a pid through `sysctl`. `use_errno=True` is what lets
`failed` say why a call failed.

@module ph.libc
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Sequence

__all__ = ["LIBC", "failed", "sysctl"]

LIBC = ctypes.CDLL(None, use_errno=True)


def failed(call: str) -> OSError:
    """Build the `OSError` for the libc `call` that just failed, from its errno."""
    code = ctypes.get_errno()
    return OSError(code, f"{call}: {os.strerror(code)}")


def sysctl(name: Sequence[int], size: int) -> bytes:
    """Read the value of the MIB `name` into at most `size` bytes. Return what the
    kernel wrote, which is nothing when the name has no value (a pid that has
    gone).

    This works on macOS and the BSDs. glibc dropped `sysctl(2)` in 2.32, so on
    Linux the symbol is missing and this function raises `AttributeError`.

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
