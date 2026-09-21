"""One cross-process file lock, spelled once.

`filelock` is used at five places in this repo, and each one had written out the
same three things: `thread_local=False`, the parent `mkdir`, and a `Timeout`
translated into whatever error that caller's domain uses. Four got it right and
`ph_rlm.harness.service` did not — it omits `thread_local=False`, and is correct
only by accident, because the one method that takes it never crosses a thread.

**`thread_local=False` is the property worth centralizing**, for the reason
`ph.persistence.lease` states: filelock keeps its re-entrancy counter in a
thread-local by default, so a lock acquired on a worker thread and released from
the event loop finds a counter of zero and returns *having released nothing* —
no error, no warning, and a lock file held until the process dies. A lock here
belongs to the process, not to whichever thread took it. That is the whole
reason this module exists: an invariant whose absence is silent cannot be left
to five call sites to remember.

**What is deliberately not here: the timeout, and the error.** They are the two
things that genuinely differ. A best-effort cache write bounds an event-loop
stall at 0.25 s and gives up; a `uv pip install` waits fifteen minutes because
giving up means building over another process's tree. And each caller raises the
error its own callers already catch. So this owns the mechanism and `LockBusy`
is the one thing it throws, for the caller to translate.

@module ph.locks
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

__all__ = ["LockBusy", "acquire_file_lock", "file_lock"]


class LockBusy(RuntimeError):
    """Somebody else holds the lock, and waiting for it ran out.

    A message and nothing else. The first cut carried `path`, `what` and
    `timeout` as attributes "so a caller can write its own message", and no
    caller does: all four already hold those values in scope and raise their own
    domain error from this one. Fields nobody reads are a claim about how this
    is used that is not true.
    """


def acquire_file_lock(path: Path | str, *, timeout: float, what: str) -> Callable[[], None]:
    """Take the lock and hand back its release, for a caller that holds it open.

    The shape `ph.persistence.lease` needs: a lease is released by the scope
    unwinding rather than by leaving a `with`, so the release has to be a value.

    **No `mkdir`**, which this got wrong once: filelock calls
    `ensure_directory_exists(self.lock_file)` on every acquire, and its body is
    `Path(filename).parent.mkdir(parents=True, exist_ok=True)` on the same
    directory. `lease.py` carried a comment saying exactly that, and centralizing
    the idiom deleted the comment and re-added the call it was warning about.
    """
    resolved = Path(path)
    lock = FileLock(str(resolved), timeout=timeout, thread_local=False)
    try:
        lock.acquire()
    except Timeout as error:
        raise LockBusy(
            f"{what} is locked by another process at {resolved} (waited {timeout:.0f}s)"
        ) from error
    return lock.release


@contextmanager
def file_lock(path: Path | str, *, timeout: float, what: str) -> Iterator[None]:
    """Hold the lock for the body. Raises `LockBusy` rather than `Timeout`."""
    release = acquire_file_lock(path, timeout=timeout, what=what)
    try:
        yield
    finally:
        release()
