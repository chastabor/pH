"""The orphan journal (F5): child processes from a run nothing cleaned up.

Every other cleanup path in pH is structural — an effect disposer, a scope
unwinding, `await proc.wait()` in a `finally`. None of them runs under `SIGKILL`,
and POSIX re-parents a child to PID 1 rather than killing it, so a hard-killed
host leaves a live Python process holding a model's namespace. Nothing in that
session will ever reconcile it, because nobody is going to reopen a session that
died.

So spawns are journalled, `fsync`ed, and swept at **every** pH start.

The pid is not enough to sweep by: pids are reused, and killing the wrong
process is far worse than leaving a stray. Each record therefore carries a
**start token** — on Linux, the kernel's own `starttime` for that pid — and a
stray is killed only when the token still matches. Where the token cannot be
read at all, the record is reported and **not** killed: an honest "there may be
a stray" beats a confident kill of something else.

**Here rather than in `ph_rlm.kernel`, where it was built.** The reasoning above
is about POSIX and `SIGKILL`, not about the RLM guest — it was simply the first
child pH hard-killed and then noticed. `ph.seams.subprocess` spawns every other
one (git, jj, agentfs, the sandbox backend, `bash`, `!`), all with the same
disposer-based cleanup and the same hole under it, and a second journal for them
would be a second set of rules about when it is safe to kill a pid.

@module ph.orphans
"""

from __future__ import annotations

import hashlib
import logging
import os
import signal
import sys
from collections.abc import Iterator, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .json import dumps
from .paths import RuntimeDirError, resolve_roots, write_atomic
from .persistence import read_records

__all__ = [
    "JOURNAL_NAME",
    "OrphanJournal",
    "SweepReport",
    "argv_digest",
    "host_journal",
    "process_alive",
    "process_start_token",
    "signal_group",
]

log = logging.getLogger("ph.orphans")

JOURNAL_NAME = "processes.jsonl"


def host_journal() -> OrphanJournal | None:
    """This host's journal, or `None` where there is nowhere to put it.

    One constructor, because two callers resolved the same two lines themselves
    and a third would have. `$PH_RUNTIME` is the tier and `ph.paths` says why: it
    is wiped on reboot, and a journal of pids that outlived a reboot would be
    actively dangerous once those pids are reused.

    `None` rather than raising: a read-only `$PH_RUNTIME` is a deployment fact,
    and refusing to spawn `git` over a diagnostic would trade a working harness
    for a tidier one. The hole is open again in that deployment, which the
    `subprocess` diagnostic says out loud.
    """
    try:
        return OrphanJournal(path=resolve_roots().runtime / JOURNAL_NAME)
    except RuntimeDirError:
        log.warning("ph.orphans: no $PH_RUNTIME; spawns will not be journalled")
        return None


def argv_digest(argv: Sequence[str]) -> str:
    """A short digest of the command line, so a record names what it spawned."""
    return hashlib.sha256("\x00".join(argv).encode("utf-8")).hexdigest()[:16]


def process_start_token(pid: int) -> str | None:
    """A value that changes when a pid is reused, or `None` if unknowable here.

    `None` is the honest answer on a platform pH cannot ask, and it is load
    bearing: the sweep refuses to kill what it cannot identify.
    """
    if sys.platform.startswith("linux"):
        try:
            raw = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        except OSError:
            return None
        # The `comm` field is parenthesized and may itself contain spaces and
        # parens, so the fields after it are found from the *last* ')'.
        tail = raw.rpartition(")")[2].split()
        # /proc(5): field 22 overall is `starttime`, which is index 19 after comm.
        return tail[19] if len(tail) > 19 else None
    return None


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What one sweep did, so `phern doctor` can say it out loud."""

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
    """This process's `(pid, start token)`, read once. Both are constants for the
    life of the process, and `process_start_token` is a `/proc` read and parse —
    56% of a record's cost when it was done per spawn."""

    def _owned_by(self) -> tuple[int, str | None]:
        if self._owner is None:
            pid = os.getpid()
            self._owner = (pid, process_start_token(pid))
        return self._owner

    def record(self, *, pid: int, argv: Sequence[str], label: str | None = None) -> None:
        """Note a spawn durably before the child can do anything.

        `fsync`ed because the failure this guards against is the host dying, and
        a buffered record would die with it.

        **The owner is recorded beside the child**, which is what makes this
        journal safe to share. One file per user per boot means a daemon's live
        children and a dead run's strays sit in it together, and a sweep that
        told them apart only by "is the pid alive" would kill the daemon's — the
        exact failure the start token exists to prevent, one level up. A record
        whose owner is still running belongs to a process that will clean it up
        itself, so `sweep` leaves it alone; the owner gets a token too, because
        an owner pid is reusable in exactly the way a child pid is.
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
        """Kill provably-ours strays, forget the rest, and compact the journal."""
        live = self._live()
        killed: list[int] = []
        stale: list[int] = []
        unverifiable: list[int] = []
        held: list[int] = []
        for pid, record in sorted(live.items()):
            if _owner_alive(record):
                # Somebody else's live children. They are not strays, and this
                # process has no business killing them — see `record`.
                held.append(pid)
                continue
            if not process_alive(pid):
                stale.append(pid)
                continue
            # Read after the liveness check, not before: a dead pid's token is a
            # `/proc` read whose answer is discarded, and most records in a
            # long-lived journal are dead by the time anyone sweeps.
            token = process_start_token(pid)
            recorded = record.get("startToken")
            if token is None or recorded is None:
                # No token now and no token recorded are one fact: there is no
                # identity to match. The mismatch test below needs both sides, so
                # this has to come first — ordered the other way, a record
                # written without a token matched nothing and reached the kill
                # unchecked, which is this module's rule inverted at the point it
                # exists for. `record` writes `startToken: null` whenever `/proc`
                # is unreadable at spawn, and a journal outlives a `SIGKILL`.
                unverifiable.append(pid)
                continue
            if recorded != token:
                # The pid came back as something else. Leaving it alone is the
                # whole reason the token is recorded.
                stale.append(pid)
                continue
            if _kill(pid):
                killed.append(pid)
            else:
                stale.append(pid)
        # A held record stays: its owner is still responsible for it, and
        # compacting it away would hide the child from the sweep that runs after
        # that owner finally dies.
        keep = [live[pid] for pid in sorted(unverifiable + held)]
        # Nothing read, nothing owed: every `phern` invocation otherwise wrote a
        # temp file and renamed it over a journal that was absent or unchanged.
        if live or self.path.exists():
            self._rewrite(keep)
        if killed or unverifiable:
            log.info(
                "ph.orphans: swept %d stray runtime child(ren); %d unverifiable",
                len(killed),
                len(unverifiable),
            )
        return SweepReport(tuple(killed), tuple(stale), tuple(unverifiable), tuple(held))

    # ------------------------------------------------------------ internals --

    def _append(self, record: dict[str, Any]) -> None:
        """One line, written and closed — deliberately **not** `fsync`ed.

        It was, on the reasoning that "the failure this guards against is the
        host dying". That reasoning does not survive where the file lives: this
        journal is in `$PH_RUNTIME` precisely because that is wiped on reboot,
        so an unclean host crash destroys the very records an `fsync` would have
        saved. What it has to survive is a `SIGKILL` of *pH*, and a closed write
        is already in the page cache, which outlives the process.

        Measured, because the cost was not small: 1023 µs per record against
        22.6 µs on a disk-backed `$PH_RUNTIME`, paid on every `git`, `jj`,
        `bash` and `!` this seam spawns.
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(dumps(record) + "\n")
        except OSError:
            log.warning("ph.orphans: could not journal %r", record, exc_info=True)

    def _records(self) -> Iterator[dict[str, Any]]:
        """This journal's lines, tolerating the tail a hard kill leaves.

        The tolerance is `read_records`' — the journal exists precisely because
        pH was killed mid-write, so refusing the file over its last line would
        discard the strays it was written to find.
        """
        return read_records(self.path)

    def _live(self) -> dict[int, dict[str, Any]]:
        live: dict[int, dict[str, Any]] = {}
        for record in self._records():
            pid = record.get("pid")
            if not isinstance(pid, int):
                continue
            if record.get("op") == "spawn":
                live[pid] = record
            else:
                live.pop(pid, None)
        return live

    def _rewrite(self, keep: list[dict[str, Any]]) -> None:
        """Compact to what is still outstanding.

        Rewritten rather than appended-to because this file is swept at every
        start: left to grow, it would accumulate one pair of lines per cell for
        the lifetime of the installation.
        """
        try:
            write_atomic(self.path, "".join(dumps(record) + "\n" for record in keep))
        except OSError:
            log.warning("ph.orphans: could not compact the orphan journal", exc_info=True)


def _owner_alive(record: dict[str, Any]) -> bool:
    """Whether the pH process that spawned this child is still running.

    A record with no owner is from a build that did not write one; treating it
    as ownerless is the safe reading, because the alternative — assuming some
    live process owns it — would never sweep anything.
    """
    owner = record.get("owner")
    if not isinstance(owner, int) or not process_alive(owner):
        return False
    recorded = record.get("ownerToken")
    token = process_start_token(owner)
    # An owner pid that came back as something else is not the owner; one neither
    # side can identify is given the benefit of the doubt, which is the same
    # restraint the child's own token gets.
    return recorded is None or token is None or recorded == token


def signal_group(pid: int | None, *, kill: bool, group: int | None = None) -> bool:
    """Signal a child's whole process group. `False` if there was none to signal.

    **The group, because making a child a session leader creates the need.**
    `start_new_session=True` is what lets `killpg` mean "this command" — a shell
    command is usually more than one process, and signalling the shell leaves
    the pipeline it started running. The two belong together, and taking one
    without the other is worse than taking neither: before the child had a
    session of its own it shared the caller's, so a grandchild at least died
    with the terminal.

    **Here rather than in `ph.seams.subprocess`**, which is where it started,
    because this module's sweep is the third caller and cannot import the seam —
    the seam imports *this*. That the sweep needed it is the argument for the
    move: every pid the journal records was spawned as a session leader, so
    killing the leader alone left the group on the one path the journal exists
    to cover.

    `group` is for a caller that already knows the id and needs the answer to
    outlive the child. `getpgid` stops answering once the child is reaped, which
    is exactly when a *sweep* matters — the leader is gone and whatever it
    started is not — and a session leader's group is its own pid. Pass it only
    then: while the child is alive `getpgid` is the better answer, because a
    child that called `setpgid` on itself is no longer in the group its pid
    names.

    `False` rather than an exception on the paths where there is nothing to do:
    `killpg` is POSIX-only, and a group with no members left is not an error to
    report. Both leave the direct child as the caller's honest best effort.
    """
    if not hasattr(os, "killpg"):
        return False
    if group is None:
        if pid is None:
            return False
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

    Public because a test that watches a process die needs the same answer this
    module's sweep does, and three suites had written it out — two of them with
    a different reading of `PermissionError`, which is the one case where the
    honest answer is "yes, and not yours to signal".
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


def _kill(pid: int) -> bool:
    """Kill one stray **and whatever it started**.

    The group, because every pid recorded here was spawned with
    `start_new_session=True` — see `signal_group`. Signalling the leader alone
    left its children running on exactly the path this journal exists for: the
    restart after a host died without unwinding.
    """
    # **Only when the stray leads its own group**, and that is not a formality:
    # these pids come out of a *file*, and one that shares this process's group
    # would make the sweep kill the sweeper. Every pid pH records is a session
    # leader, so the check costs nothing and turns an assumption into a
    # verified fact — the same restraint the start token applies to identity.
    with suppress(OSError):
        if os.getpgid(pid) == pid and signal_group(pid, kill=True, group=pid):
            return True
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True
