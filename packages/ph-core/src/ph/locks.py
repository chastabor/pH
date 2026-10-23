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

**The lock is `flock(2)`, and nothing else is accepted** (P13-11). The library's
`FileLock` is whichever lock the host can offer: its unix lock where `fcntl` imports,
and otherwise an *existence* lock — a file whose presence is the claim — with a
warning nobody reads at import. The unix lock itself switches to that existence
lock when a filesystem answers `flock` with "not supported", again with a warning.
An existence lock is not a lease a crash gives back: `ph.persistence.lease`'s whole
argument is that a killed process releases its sessions by dying, since the kernel
holds the lock on the open descriptor, and a lock file left behind by a kill would
refuse the next daemon every session it had. So this asks for the unix lock by name,
pins its fallback off (`fallback_to_soft=False`), and turns the refusal into
`LockUnsupported`, which names the path: a daemon on such a filesystem must not
start, and `phern doctor` says which filesystem (`describe_lock`).

**What is deliberately not here: the timeout, and the error.** They are the two
things that genuinely differ. A best-effort cache write bounds an event-loop
stall at 0.25 s and gives up; a `uv pip install` waits fifteen minutes because
giving up means building over another process's tree. And each caller raises the
error its own callers already catch. So this owns the mechanism and `LockBusy`
is the one thing it throws for contention, for the caller to translate.

**Two locks are spelled elsewhere.**
- **The orphan journal's** (`OrphanJournal._locked`). Its appends share a lock that
  its compaction takes whole, which `filelock` does not offer, and it is taken on
  every spawn, where this measured 80 µs.
- **The schedule index's claims** (`ScheduleIndex.claim`). A claim is held for as
  long as its change is on the way and probed without waiting, and `filelock`
  opens with `O_CREAT`, so a probe would make again the claim it was asking after.

@module ph.locks
"""

from __future__ import annotations

import errno
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import Timeout, UnixFileLock

from .paths import filesystem_of

__all__ = ["LockBusy", "LockUnsupported", "acquire_file_lock", "describe_lock", "file_lock"]


class LockBusy(RuntimeError):
    """Somebody else holds the lock, and waiting for it ran out.

    A message and nothing else. The first cut carried `path`, `what` and
    `timeout` as attributes "so a caller can write its own message", and no
    caller does: all four already hold those values in scope and raise their own
    domain error from this one. Fields nobody reads are a claim about how this
    is used that is not true.
    """


class LockUnsupported(RuntimeError):
    """This filesystem does not offer `flock`, and an existence lock is refused.

    Not `LockBusy`: nobody holds the lock, and waiting would not help. A caller
    that catches `LockBusy` to say "already active" must not say that here, so the
    two are different types, and this one is left to reach the top: a daemon
    whose leases cannot be kernel locks must not start.
    """


_UNSUPPORTED = frozenset({errno.ENOSYS, errno.EOPNOTSUPP, errno.ENOTSUP, errno.ENOLCK})
"""The errnos `flock(2)` answers with on a filesystem that does not implement it —
the library's own test for switching to its existence lock (`ENOSYS`), and the ones
a networked filesystem answers directly."""

PROBE = ".doctor.lock"
"""The lock file `describe_lock` takes and lets go, beside the leases."""


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
    lock = UnixFileLock(str(resolved), timeout=timeout, thread_local=False, fallback_to_soft=False)
    try:
        lock.acquire()
    except Timeout as error:
        raise LockBusy(
            f"{what} is locked by another process at {resolved} (waited {timeout:.0f}s)"
        ) from error
    except OSError as error:
        if error.errno in _UNSUPPORTED:
            raise LockUnsupported(
                f"{what} needs flock(2), which the filesystem holding {resolved} does not "
                f"offer ({error.strerror}); an existence lock would outlive a crash, so "
                "none is taken"
            ) from error
        raise
    return lock.release


@contextmanager
def file_lock(path: Path | str, *, timeout: float, what: str) -> Iterator[None]:
    """Hold the lock for the body. Raises `LockBusy` rather than `Timeout`."""
    release = acquire_file_lock(path, timeout=timeout, what=what)
    try:
        yield
    finally:
        release()


def describe_lock(directory: Path) -> list[tuple[str, str]]:
    """The lease mechanism as `phern doctor` prints it: what the lock is, which
    filesystem `directory` is on, and whether one was just taken and let go there.

    The probe is the real thing — a lock in the lease directory, held for no time —
    because the question is about the filesystem, and only the filesystem answers
    it. The directory is made if it is missing, as the first lease would make it.
    """
    rows = [
        ("mechanism", "flock(2) on a lock file, held by the open descriptor until it closes"),
        ("directory", str(directory)),
        ("filesystem", filesystem_of(directory)),
    ]
    try:
        with file_lock(directory / PROBE, timeout=0, what="the lease probe"):
            pass
    except (LockBusy, LockUnsupported, OSError) as error:
        rows.append(("probe", f"refused: {error}"))
    else:
        rows.append(("probe", "a lock was taken and released"))
    return rows
