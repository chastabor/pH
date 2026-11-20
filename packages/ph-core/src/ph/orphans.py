"""The orphan journal (F5): child processes from a run that nothing cleaned up.

Every other cleanup path in pH is structural: an effect disposer, a scope that
unwinds, `await proc.wait()` in a `finally`. None of them runs under `SIGKILL`.
POSIX re-parents a child to PID 1 and does not kill it. So a hard-killed host
leaves a live Python process that holds the namespace of a model. Nothing in
that session will ever reconcile it, because nobody reopens a session that
died.

So pH journals each spawn and sweeps the journal at **every** pH start. The
journal is not `fsync`ed: it is in `$PH_RUNTIME`, which a reboot wipes
(`OrphanJournal._append`).

The pid is not enough to sweep by. Pids are reused, and killing the wrong
process is far worse than leaving a stray. Each record therefore carries a
**start token**: the record that the kernel keeps of when that pid started,
from `/proc` on Linux and from `sysctl` on macOS. A stray is killed only when
the token still matches. Where the token cannot be read at all, the record is
reported and **not** killed. A report that says "there may be a stray" is
better than a confident kill of something else.

**The journal is here and not in `ph_rlm.kernel`, where it was built.** The
reasoning above is about POSIX and `SIGKILL`, not about the RLM guest. The
guest was only the first child that pH hard-killed and then noticed.
`ph.seams.subprocess` spawns every other child (git, jj, agentfs, the sandbox
backend, `bash`, `!`). All of them have the same disposer-based cleanup and the
same gap under it. A second journal for them would be a second set of rules
about when it is safe to kill a pid.

@module ph.orphans
"""

from __future__ import annotations

import fcntl
import hashlib
import logging
import os
import signal
import struct
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .json import dumps
from .paths import RuntimeDirError, resolve_roots, write_atomic
from .persistence import records_in

__all__ = [
    "JOURNAL_NAME",
    "OrphanJournal",
    "SweepReport",
    "argv_digest",
    "host_journal",
    "process_alive",
    "process_start_token",
    "signal_child",
    "signal_group",
]

log = logging.getLogger("ph.orphans")

JOURNAL_NAME = "processes.jsonl"


def host_journal() -> OrphanJournal | None:
    """The journal of this host, or `None` where there is nowhere to put it.

    There is one constructor because two callers each wrote the same two lines
    themselves, and a third would have. `$PH_RUNTIME` is the tier, and
    `ph.paths` says why: a reboot wipes it, and a journal of pids that outlived
    a reboot would be dangerous as soon as those pids are reused.

    **`$PH_RUNTIME` is created here, owner-only** (`PathRoots.ensure_runtime`),
    before anything is journaled. A journal that created the directory itself
    did so at the default mode. On the `/tmp` fallback the next start then
    failed the check of that directory and refused to run (S22).

    This returns `None` and does not raise. A read-only `$PH_RUNTIME` is a fact
    of the deployment, and refusing to spawn `git` because of a diagnostic
    would trade a working harness for a tidier one. In that deployment the gap
    is open again, and the `subprocess` diagnostic reports it.
    """
    try:
        runtime = resolve_roots().ensure_runtime().runtime
    except (RuntimeDirError, OSError):
        runtime = None
    if runtime is None:
        log.warning("ph.orphans: no usable $PH_RUNTIME; spawns will not be journaled")
        return None
    # The access check is needed because a directory that exists and cannot be
    # written passes `ensure_runtime`. A command that runs inside the sandbox
    # of an agent sees the `$PH_RUNTIME` of the host bound read-only. A journal
    # on it then failed at every append, with a traceback for each spawn. This
    # is not a warning: inside a sandbox it is the expected state, and a command
    # that is started one time for each call would log it one time for each
    # call. Outside a sandbox the `subprocess` diagnostic reports it.
    if not os.access(runtime, os.W_OK):
        log.debug("ph.orphans: $PH_RUNTIME %s is read-only; spawns will not be journaled", runtime)
        return None
    return OrphanJournal(path=runtime / JOURNAL_NAME)


def argv_digest(argv: Sequence[str]) -> str:
    """A short digest of the command line, so a record names what it spawned."""
    return hashlib.sha256("\x00".join(argv).encode("utf-8")).hexdigest()[:16]


def process_start_token(pid: int) -> str | None:
    """A value that changes when a pid is reused, or `None` if it cannot be known here.

    The value is when the kernel says the process started. On Linux it is
    `starttime` in `/proc/<pid>/stat`. macOS has no `/proc`, so there it is the
    start time in the `kinfo_proc` that `sysctl` returns
    (`_darwin_start_token`).

    `None` is the correct answer on a platform that pH cannot ask, and the
    sweep depends on it: the sweep refuses to kill what it cannot identify.
    """
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        # The `comm` field is in parentheses and may itself contain spaces and
        # parentheses, so the fields after it are found from the *last* ')'.
        tail = raw.rpartition(")")[2].split()
        # /proc(5): field 22 overall is `starttime`, which is index 19 after comm.
        return tail[19] if len(tail) > 19 else None
    if sys.platform == "darwin":
        return _darwin_start_token(pid)
    return None


_KERN_PROC_PID = (1, 14, 1)
"""`CTL_KERN`, `KERN_PROC`, `KERN_PROC_PID`: the MIB that, with a pid after it,
names the `struct kinfo_proc` of that process (`<sys/sysctl.h>`)."""

_KINFO_PROC_SIZE = 648
"""`sizeof(struct kinfo_proc)` on 64-bit macOS, on both x86_64 and arm64. A
kernel whose struct grew answers `ENOMEM` to a buffer of this size, and that
reads as no token."""

_KINFO_PID_OFFSET = 40
"""Where `kp_proc.p_pid` sits: after `p_starttime` (a 16-byte `timeval`), two
pointers, `p_flag`, and `p_stat` padded to four."""


def _darwin_start_token(pid: int) -> str | None:
    """The start time of `pid` on macOS, from `sysctl`, as `ps` and psutil read it.
    It needs no privilege and answers for any pid."""
    # macOS only, and the one thing in this module that needs ctypes. A start on
    # Linux reads `/proc` and should not pay 1.4 ms to load a handle that it
    # never calls.
    from .libc import sysctl  # noqa: PLC0415

    try:
        raw = sysctl((*_KERN_PROC_PID, pid), _KINFO_PROC_SIZE)
    except (OSError, AttributeError):
        return None
    return _kinfo_start_token(raw, pid)


def _kinfo_start_token(raw: bytes, pid: int) -> str | None:
    """The start token in a `struct kinfo_proc`, or `None` unless `raw` is one for
    `pid`.

    `kp_proc.p_starttime` is the first field of the struct. It is a `timeval`,
    so the token is in microseconds where the Linux token is in clock ticks.
    **The pid is checked before the time is trusted**, because these offsets
    are constants and do not come from a compiler. They match `sizeof` and
    `offsetof` in the macOS 26 SDK (arm64, checked 2026-10-02). A layout that
    moves in a later kernel has to read as no token, which the sweep spares,
    and never as a steady wrong token. A pid that has gone comes back empty.
    """
    if len(raw) < _KINFO_PID_OFFSET + 4:
        return None
    seconds, micros = struct.unpack_from("=qi", raw, 0)
    (recorded,) = struct.unpack_from("=i", raw, _KINFO_PID_OFFSET)
    if recorded != pid or seconds <= 0:
        return None
    return f"{seconds}.{micros:06d}"


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What one sweep did, so that `phern doctor` can report it."""

    killed: tuple[int, ...] = ()
    stale: tuple[int, ...] = ()
    """Recorded, gone, and reaped from the journal."""
    unverifiable: tuple[int, ...] = ()
    """Still alive but not provably ours. Reported, never killed."""
    held: tuple[int, ...] = ()
    """Alive, and owned by a pH process that is also alive. Left to its owner."""


@dataclass(slots=True)
class OrphanJournal:
    """An append-only record of live runtime children, outside any session."""

    path: Path
    _owner: tuple[int, str | None] | None = None
    """The `(pid, start token)` of this process, read one time. Both are constant
    for the life of the process. `process_start_token` reads and parses `/proc`,
    which was 56% of the cost of a record when it ran for each spawn."""

    def _owned_by(self) -> tuple[int, str | None]:
        if self._owner is None:
            pid = os.getpid()
            self._owner = (pid, process_start_token(pid))
        return self._owner

    def record(self, *, pid: int, argv: Sequence[str], label: str | None = None) -> None:
        """Note a spawn before the child can do anything. The line is written and
        closed, which outlives a `SIGKILL` of pH (`_append` says why that is
        enough).

        **The owner is recorded beside the child**, and that makes this journal
        safe to share. There is one file per user per boot, so the live children
        of a daemon and the strays of a dead run are in it together. A sweep
        that told them apart only by "is the pid alive" would kill the children
        of the daemon. That is the exact failure that the start token exists to
        prevent, one level up. A record whose owner is still running belongs to
        a process that will clean it up itself, so `sweep` leaves it alone. The
        owner gets a token too, because an owner pid can be reused in exactly
        the way that a child pid can.
        """
        owner, owner_token = self._owned_by()
        self._append(
            {
                "op": "spawn",
                "pid": pid,
                "startToken": process_start_token(pid),
                "argv": argv_digest(argv),
                "label": label,
                "owner": owner,
                "ownerToken": owner_token,
            }
        )

    def forget(self, pid: int) -> None:
        """Note that a child was reaped on the normal path."""
        self._append({"op": "reap", "pid": pid})

    def sweep(self) -> SweepReport:
        """Kill the strays that are provably ours, forget the rest, and compact the journal.

        **Judged without the lock, compacted under the lock** (S22). A
        compaction renames a new file over the journal. So an append between the
        read and the rename landed on the file that was renamed away, and a
        child that was spawned then was never swept. Holding the lock from the
        read to the rename closed that gap. It also blocked every appender (the
        loop of a daemon, at its next spawn) for as long as the read, the
        `/proc` checks and the kills took: 240 ms at 100 000 records. So the
        lock is held only to carry over what was appended since the read, and to
        rename. A journal that another sweep compacted in the meantime is left
        as that sweep wrote it.

        A journal that the sweep cannot read or lock gets a warning and is left
        alone, for the reason that `host_journal` gives: a mount is not refused
        because of a diagnostic.
        """
        try:
            with self.path.open("rb") as handle:
                read = handle.read()
                identity = _identity(os.fstat(handle.fileno()))
        except FileNotFoundError:
            return SweepReport()
        except OSError:
            log.warning("ph.orphans: could not read the orphan journal", exc_info=True)
            return SweepReport()
        # Whole lines only. A line that is still being written is carried over
        # with the rest.
        judged, live = _outstanding(read)
        report, keep = _judge(live)
        # Nothing was read, so nothing is owed. Without this check every `phern`
        # invocation wrote a temp file and renamed it over an empty journal.
        if read:
            try:
                with self._locked(exclusive=True):
                    self._compact(keep, identity, judged)
            except OSError:
                log.warning("ph.orphans: could not compact the orphan journal", exc_info=True)
        if report.killed or report.unverifiable:
            log.info(
                "ph.orphans: swept %d stray runtime child(ren); %d unverifiable",
                len(report.killed),
                len(report.unverifiable),
            )
        return report

    def kill_owned(self) -> tuple[int, ...]:
        """Kill every child this process journaled and has not reaped, and whatever
        each one started. Returns the pids killed.

        **For a host leaving without its unwind**: the hard stop past a signal's grace
        period, or a second signal (`ph.resources.stop_on_signals`). The unwind would
        have stopped each child through its own disposer, and past the grace nothing
        is waiting for that. Leaving them would be the hard-killed host this journal
        exists for, made on purpose. The kernel guest and a `bwrap`-confined command
        die with their host anyway. An unconfined shell command does not, and on
        macOS no shell command does.

        Synchronous, and it takes no lock, because its caller may be a thread beside
        a loop that has stopped answering. Only this process's records, matched by
        owner pid and owner start token, and only a child whose start token still
        matches. That is the sweep's restraint (`_judge`): a child reaped just before
        its `reap` line was written left a pid that may now belong to anybody.
        Nothing is written. The next sweep finds these pids dead and compacts them
        away.
        """
        try:
            read = self.path.read_bytes()
        except OSError:
            return ()
        owner = self._owned_by()
        # Whole lines only, as `sweep` reads them. A line that is still being
        # appended is a spawn too new to have been recorded completely.
        _, live = _outstanding(read)
        killed: list[int] = []
        for pid, record in sorted(live.items()):
            if (record.get("owner"), record.get("ownerToken")) != owner:
                continue
            recorded = record.get("startToken")
            if recorded is None or recorded != process_start_token(pid):
                continue
            if _kill(pid):
                killed.append(pid)
        return tuple(killed)

    # ------------------------------------------------------------ internals --

    def _append(self, record: dict[str, Any]) -> None:
        """One line, written and closed, and deliberately **not** `fsync`ed.

        It was `fsync`ed before, on the reasoning that "the failure this guards
        against is the host dying". That reasoning fails because of where the
        file is. This journal is in `$PH_RUNTIME` precisely because a reboot
        wipes that directory, so an unclean host crash destroys the very records
        that an `fsync` would have saved. What the line has to survive is a
        `SIGKILL` of *pH*. A closed write is already in the page cache, which
        outlives the process.

        The cost was measured, because it was not small: 1023 µs per record
        against 22.6 µs on a disk-backed `$PH_RUNTIME`. It was paid on every
        `git`, `jj`, `bash` and `!` that the subprocess seam spawns. For the
        same reason appends share their lock (`_locked`) and do not take
        `ph.locks.file_lock`, which measured 80 µs a record.
        """
        try:
            with self._locked(), self.path.open("a", encoding="utf-8") as handle:
                handle.write(dumps(record) + "\n")
        except OSError:
            log.warning("ph.orphans: could not journal %r", record, exc_info=True)

    @contextmanager
    def _locked(self, *, exclusive: bool = False) -> Iterator[None]:
        """The lock of the journal. Appends share it, so they cost each other
        nothing. The compaction of a sweep takes it exclusively (`sweep`).

        The lock is on a file beside the journal, because a compaction gives the
        journal a new inode. The file is opened for each use, because `flock`
        belongs to an open file. A descriptor that the threads of this process
        shared would let a sweep on a worker thread and an append on the loop
        both hold the lock.
        """
        handle = os.open(
            self.path.with_name(f"{self.path.name}.lock"), os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            fcntl.flock(handle, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            yield
        finally:
            os.close(handle)

    def _compact(self, keep: list[dict[str, Any]], identity: tuple[int, int], judged: int) -> None:
        """Rewrite the journal to `keep` plus whatever was appended past `judged`
        since the journal was read. This runs under the exclusive lock, so no
        append is part-way. The journal is left alone when it is no longer the
        file that was read: another sweep compacted it in the meantime.

        The journal is rewritten and not appended to because this file is swept
        at every start. If it were left to grow, it would collect one pair of
        lines for each child for the lifetime of the installation.
        """
        try:
            current = os.stat(self.path)
        except FileNotFoundError:
            return
        if _identity(current) != identity or current.st_size < judged:
            return
        with self.path.open("rb") as handle:
            handle.seek(judged)
            since = handle.read()
        kept = "".join(dumps(record) + "\n" for record in keep).encode()
        # Not durable. `$PH_RUNTIME` does not outlive the reboot that a sync
        # would guard against. `_append` gives the same reason for not syncing.
        write_atomic(self.path, kept + since, durable=False)


def _identity(found: os.stat_result) -> tuple[int, int]:
    """Which file a path named when it was read: a compaction gives it another."""
    return found.st_dev, found.st_ino


def _outstanding(read: bytes) -> tuple[int, dict[int, dict[str, Any]]]:
    """The spawn records that the bytes of a journal leave outstanding, read up to
    the last whole line, and where that line ends. `sweep` and `kill_owned`
    share this one rule for which records are judged at all."""
    judged = read.rfind(b"\n") + 1
    return judged, _live(records_in(read[:judged].decode(errors="replace").splitlines()))


def _live(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """The spawn records that the lines of a journal leave outstanding, by pid."""
    live: dict[int, dict[str, Any]] = {}
    for record in records:
        pid = record.get("pid")
        if not isinstance(pid, int):
            continue
        if record.get("op") == "spawn":
            live[pid] = record
        else:
            live.pop(pid, None)
    return live


def _judge(live: dict[int, dict[str, Any]]) -> tuple[SweepReport, list[dict[str, Any]]]:
    """What becomes of each outstanding record (killed, stale, unverifiable or
    held), and the records that a compacted journal keeps."""
    killed: list[int] = []
    stale: list[int] = []
    unverifiable: list[int] = []
    held: list[int] = []
    for pid, record in sorted(live.items()):
        if _owner_alive(record):
            # The live children of another process. They are not strays, and
            # this process must not kill them. See `record`.
            held.append(pid)
            continue
        if not process_alive(pid):
            stale.append(pid)
            continue
        # The token is read after the liveness check, not before. For a dead
        # pid the token is a `/proc` read whose answer is discarded, and most
        # records in a long-lived journal are dead by the time anyone sweeps.
        token = process_start_token(pid)
        recorded = record.get("startToken")
        if token is None or recorded is None:
            # No token now and no recorded token are one fact: there is no
            # identity to match. The mismatch test below needs both sides, so
            # this test has to come first. In the other order, a record that
            # was written without a token matched nothing and reached the kill
            # unchecked. That inverts the rule of this module at the point the
            # rule exists for. `record` writes `startToken: null` whenever
            # `/proc` is unreadable at spawn, and a journal outlives a `SIGKILL`.
            unverifiable.append(pid)
            continue
        if recorded != token:
            # The pid came back as something else. The token is recorded so
            # that such a pid is left alone.
            stale.append(pid)
            continue
        if _kill(pid):
            killed.append(pid)
        else:
            stale.append(pid)
    # A held record stays. Its owner is still responsible for it, and
    # compacting it away would hide the child from the sweep that runs after
    # that owner finally dies.
    keep = [live[pid] for pid in sorted(unverifiable + held)]
    return SweepReport(tuple(killed), tuple(stale), tuple(unverifiable), tuple(held)), keep


def _owner_alive(record: dict[str, Any]) -> bool:
    """Whether the pH process that spawned this child is still running.

    A record with no owner is from a build that did not write one. Treating it
    as ownerless is the safe reading. The alternative is to assume that some
    live process owns it, and then nothing would ever be swept.
    """
    owner = record.get("owner")
    if not isinstance(owner, int) or not process_alive(owner):
        return False
    recorded = record.get("ownerToken")
    token = process_start_token(owner)
    # An owner pid that came back as something else is not the owner. An owner
    # that neither side can identify is counted as alive, which is the same
    # restraint that the token of the child gets.
    return recorded is None or token is None or recorded == token


def signal_group(pid: int | None, *, kill: bool, reaped: bool = False) -> bool:
    """Signal the whole process group of a child. `False` if there was none to signal.

    **The group, because making a child a session leader creates the need.**
    `start_new_session=True` is what lets `killpg` mean "this command". A shell
    command is usually more than one process, and signaling the shell leaves
    the pipeline that it started running. The two belong together, and taking
    one without the other is worse than taking neither. Before the child had a
    session of its own it shared the session of the caller, so a grandchild at
    least died with the terminal.

    **Here and not in `ph.seams.subprocess`**, where it started. The sweep in
    this module is the third caller, and it cannot import the seam, because
    the seam imports *this* module. That the sweep needed it is the argument
    for the move. Every pid that the journal records was spawned as a session
    leader, so killing the leader alone left the group on the one path that the
    journal exists to cover.

    `reaped` is for a caller whose child was spawned as a session leader and
    has been waited on. The group is then named by `pid` itself, because
    `getpgid` stops answering as soon as the child is reaped. That is exactly
    when a stop or a sweep still matters: the leader is gone and whatever it
    started is not. A pid is not reused while it is still the id of a live
    group, so the kill reaches that group or fails with `ESRCH`. Use `reaped`
    only then. While the child is alive `getpgid` is the better answer, because
    a child that called `setpgid` on itself is no longer in the group that its
    pid names.

    This returns `False` and does not raise on the paths where there is nothing
    to do. `killpg` is POSIX-only, and a group with no members left is not an
    error to report. In both cases the caller is left with the direct child as
    its best effort.
    """
    if not hasattr(os, "killpg") or pid is None or pid <= 0:
        # Not a pid of a child: `killpg(0)` is this process's own group.
        return False
    if reaped:
        group = pid
    else:
        try:
            group = os.getpgid(pid)
        except OSError:
            return False
    try:
        os.killpg(group, signal.SIGKILL if kill else signal.SIGTERM)
    except OSError:
        return False  # No group: already reaped, or it changed its own.
    return True


def process_alive(pid: int) -> bool:
    """Whether `pid` still exists.

    Public because a test that watches a process die needs the same answer that
    the sweep in this module needs. Three suites had written it out, and two of
    them read `PermissionError` differently. That is the one case where the
    correct answer is "yes, and not yours to signal".
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


class _Child(Protocol):
    """What `signal_child` needs of a process: an `anyio` one, or a test's."""

    @property
    def pid(self) -> int | None: ...
    @property
    def returncode(self) -> int | None: ...
    def kill(self) -> None: ...
    def terminate(self) -> None: ...


def signal_child(child: _Child, *, kill: bool) -> None:
    """Signal a child that pH spawned as a session leader, and everything that it
    started. The signal goes to its group, which is named by its pid after the
    child has been reaped (`signal_group`). Where there is no group to signal,
    the signal goes to the child alone. After the child is reaped there is
    nothing left to signal alone."""
    reaped = child.returncode is not None
    if signal_group(child.pid, kill=kill, reaped=reaped) or reaped:
        return
    with suppress(ProcessLookupError):
        if kill:
            child.kill()
        else:
            child.terminate()


def _kill(pid: int) -> bool:
    """Kill one stray **and whatever it started**.

    The group is killed, because every pid recorded here was spawned with
    `start_new_session=True`. See `signal_group`. Signaling the leader alone
    left its children running on exactly the path that this journal exists for:
    the restart after a host died without unwinding.
    """
    # **Only when the stray leads its own group**, and that is not a formality.
    # These pids come out of a *file*, and a pid that shares the group of this
    # process would make the sweep kill the sweeper. Every pid that pH records
    # is a session leader, so the check costs nothing and turns an assumption
    # into a verified fact. The start token applies the same restraint to
    # identity.
    with suppress(OSError):
        if os.getpgid(pid) == pid and signal_group(pid, kill=True):
            return True
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True
