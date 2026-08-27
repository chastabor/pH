"""The orphan journal (F5): strays from a run nothing cleaned up.

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

@module ph_rlm.kernel.journal
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import signal
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["JOURNAL_NAME", "OrphanJournal", "SweepReport", "argv_digest", "process_start_token"]

log = logging.getLogger("ph_rlm.kernel.journal")

JOURNAL_NAME = "processes.jsonl"


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
        # The `comm` field is parenthesised and may itself contain spaces and
        # parens, so the fields after it are found from the *last* ')'.
        tail = raw.rpartition(")")[2].split()
        # /proc(5): field 22 overall is `starttime`, which is index 19 after comm.
        return tail[19] if len(tail) > 19 else None
    return None


@dataclass(frozen=True, slots=True)
class SweepReport:
    """What one sweep did, so `ph doctor` can say it out loud."""

    killed: tuple[int, ...] = ()
    stale: tuple[int, ...] = ()
    """Recorded, gone, and reaped from the journal."""
    unverifiable: tuple[int, ...] = ()
    """Still alive but not provably ours. Reported, never killed."""


@dataclass(slots=True)
class OrphanJournal:
    """An append-only record of live runtime children, outside any session."""

    path: Path

    def record(self, *, pid: int, argv: Sequence[str], namespace: str | None) -> None:
        """Note a spawn durably before the child can do anything.

        `fsync`ed because the failure this guards against is the host dying, and
        a buffered record would die with it.
        """
        self._append(
            {
                "op": "spawn",
                "pid": pid,
                "startToken": process_start_token(pid),
                "argv": argv_digest(argv),
                "namespace": namespace,
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
        for pid, record in sorted(live.items()):
            token = process_start_token(pid)
            if not _alive(pid):
                stale.append(pid)
                continue
            recorded = record.get("startToken")
            if recorded is not None and token is not None and recorded != token:
                # The pid came back as something else. Leaving it alone is the
                # whole reason the token is recorded.
                stale.append(pid)
                continue
            if token is None:
                unverifiable.append(pid)
                continue
            if _kill(pid):
                killed.append(pid)
            else:
                stale.append(pid)
        self._rewrite([live[pid] for pid in unverifiable])
        if killed or unverifiable:
            log.info(
                "ph_rlm.kernel: swept %d stray runtime child(ren); %d unverifiable",
                len(killed),
                len(unverifiable),
            )
        return SweepReport(tuple(killed), tuple(stale), tuple(unverifiable))

    # ------------------------------------------------------------ internals --

    def _append(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, separators=(",", ":")) + "\n"
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            log.warning("ph_rlm.kernel: could not journal %r", record, exc_info=True)

    def _records(self) -> Iterator[dict[str, Any]]:
        try:
            text = self.path.read_text(encoding="utf-8")
        except OSError:
            return
        for line in text.splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                yield record

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
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f"{self.path.name}.tmp")
            body = "".join(json.dumps(record, separators=(",", ":")) + "\n" for record in keep)
            temporary.write_text(body, encoding="utf-8")
            temporary.replace(self.path)
        except OSError:
            log.warning("ph_rlm.kernel: could not compact the orphan journal", exc_info=True)


def _alive(pid: int) -> bool:
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
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    return True
