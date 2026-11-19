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

**And one that is not a file at all** (`KeyedLocks`): an `anyio.Lock` per key, for
work in this process that one key's callers must take turns at — a parent's resume
sweeps, an index's copy. Here because the table that holds them is the part that
was got wrong twice: kept per key for good, it grew one lock per key ever asked.

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
import fcntl
import os
from collections.abc import Callable, Hashable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType

import anyio
from filelock import Timeout, UnixFileLock

from .paths import filesystem_of

__all__ = [
    "KeyedLocks",
    "LockBusy",
    "LockUnsupported",
    "acquire_file_lock",
    "describe_lock",
    "file_lock",
    "file_lock_held",
    "try_file_lock",
    "wait_for_file_lock",
]


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


def try_file_lock(path: Path | str, *, what: str) -> Callable[[], None] | None:
    """The lock if it is free this instant, else `None`. `LockUnsupported` still
    raises: nobody holds that lock, and asking again would not help."""
    try:
        return acquire_file_lock(path, timeout=0, what=what)
    except LockBusy:
        return None


def file_lock_held(path: Path | str) -> bool:
    """Whether some process holds the lock at `path` this instant.

    For a reader that wants to say "a run is building this" and must not take the
    lock to find out. Asked with a *shared*, non-blocking `flock`, which only an
    exclusive holder refuses — so two processes asking at one moment never see
    each other, as two that each tried the lock itself would. Opened without
    `O_CREAT`: a lock nobody has made is a lock nobody holds, and asking must not
    make it. Any other failure to ask — a filesystem with no `flock` — answers
    no: nobody is known to hold it.
    """
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return False
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        except OSError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return False
    finally:
        os.close(descriptor)


async def wait_for_file_lock(
    path: Path | str, *, what: str, poll: float = 0.1
) -> Callable[[], None]:
    """The lock, waited for as long as another holder has it.

    For a writer that is expected to queue behind another — an index being built
    by a second process — rather than be refused. **Polled rather than blocked in
    a worker thread**, so a canceled call stops waiting, and no abandoned thread
    takes the lock after its caller has gone and holds it for nobody.
    """
    while (release := try_file_lock(path, what=what)) is None:
        await anyio.sleep(poll)
    return release


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


@dataclass(slots=True, eq=False)
class _Turn:
    """One key's lock, and how many hold or wait for it."""

    lock: anyio.Lock = field(default_factory=anyio.Lock)
    users: int = 0


class KeyedLocks[K: Hashable]:
    """One `anyio.Lock` per key, made when first wanted and gone with the last caller
    holding or waiting for it — so the table is only as large as what is in use now.

    `hold(key)` is an async context manager that can be entered again and again, as
    an `anyio.Lock` can: a run that takes its turn many times keeps one handle.
    """

    __slots__ = ("_turns",)

    def __init__(self) -> None:
        self._turns: dict[K, _Turn] = {}

    def hold(self, key: K) -> _Hold[K]:
        """A handle that takes `key`'s turn on entry and gives it back on exit."""
        return _Hold(self, key)

    async def _acquire(self, key: K) -> None:
        turn = self._turns.get(key)
        if turn is None:
            turn = self._turns[key] = _Turn()
        turn.users += 1
        try:
            await turn.lock.acquire()
        except BaseException:
            self._leave(key, turn)
            raise

    def _release(self, key: K) -> None:
        turn = self._turns[key]
        turn.lock.release()
        self._leave(key, turn)

    def _leave(self, key: K, turn: _Turn) -> None:
        turn.users -= 1
        if not turn.users and self._turns.get(key) is turn:
            del self._turns[key]

    def __len__(self) -> int:
        return len(self._turns)


@dataclass(frozen=True, slots=True)
class _Hold[K: Hashable]:
    """`KeyedLocks.hold`'s handle."""

    locks: KeyedLocks[K]
    key: K

    async def __aenter__(self) -> None:
        await self.locks._acquire(self.key)

    async def __aexit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.locks._release(self.key)
