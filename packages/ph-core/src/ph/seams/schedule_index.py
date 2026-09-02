"""Which sessions hold a live appointment, and when each is next due (P6-23).

**A schedule outlives its process; the thing that reads it does not.** P5-06 is
right about the log — `schedule/created` is still there on Wednesday — and a
fresh daemon still fires nothing, because `Supervisor.tick` iterates the roots it
has mounted and a boot has none. The appointment survives and is never kept,
which fails in the worst available shape: **silence**. Nothing errors, nothing
logs, and it is found by somebody noticing a run that did not happen.

**The obvious fix is the wrong one, and this file is the shape of the right one.**
"Mount every stored session at boot" fails three ways. `StoredSession` carries no
schedule, so "does this log hold an appointment" is answerable only by reading and
folding the whole log — 500 reads before the daemon answers a connection, to find
the three that matter. Mounting is not cheap: a root is a whole profile, a
workspace, possibly a kernel, which is the cost P5-05 exists to *release*. And
`Supervisor.start` takes P5-03's **lease**, so auto-mounting everything claims
every session on the machine and the next `ph -p` over any of them is refused with
`session_already_active` — a strictly worse failure, because it is loud,
immediate, and hits sessions with no schedule at all.

So this is an **index**: the seam knows the moment an appointment is created,
cancelled or fired, so it records which sessions have one and when each is next
due. A daemon reads one small file and mounts only what is actually due.

**It is a cache, and the logs stay authoritative (I-6).** Every value here is
derived from a log that still holds it, so a missing, stale or corrupt index costs
a late run and never a wrong one: a reader that finds nothing falls back to the
behaviour that shipped before this existed. The projection is never the source of
truth, which is what keeps it from becoming a second answer to "what is
scheduled" (A11).

**Reconciliation needs no scan, because opening a session is the rebuild.** The
seam re-derives an entry from the log on every `session/created`, so an index that
is missing, stale or written by a build that had none corrects itself the moment
anything touches that session — which is exactly the condition the old behaviour
required to fire a schedule at all. A wholesale rebuild would have to read every
stored log, which is the scan this file exists to avoid.

@module ph.seams.schedule_index
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from filelock import FileLock, Timeout

__all__ = ["INDEX_NAME", "Appointment", "ScheduleIndex"]

log = logging.getLogger("ph.seams.schedule_index")

INDEX_NAME = "schedules.json"
"""One file per `$PH_HOME`, beside the sessions it indexes.

A single document rather than a file per session: it is read whole on every
daemon tick and written only when an appointment changes, which is the opposite
of the access pattern a directory of files is good at.
"""

_VERSION = 1
_LOCK_TIMEOUT = 5.0


@dataclass(frozen=True, slots=True)
class Appointment:
    """One session's next due moment, and when that was last established.

    `updated` is what makes waking bounded. A daily cron refreshes it on every
    firing, so an index a daemon has been serving stays fresh; one nobody has
    served since March goes stale, and a daemon started today can decline to
    resurrect a root its owner abandoned. That question is the caller's — the
    index reports the age and does not enforce a policy with it.
    """

    session_id: str
    next_at: int
    updated: int


@dataclass(slots=True)
class ScheduleIndex:
    """The index as a value, so a caller states where it lives and nothing guesses."""

    root: Path

    @property
    def path(self) -> Path:
        return self.root / INDEX_NAME

    def read(self) -> dict[str, Appointment]:
        """Every appointment on record. Empty when there is nothing to say.

        **Every failure reads as empty**, deliberately: a missing file is the
        ordinary state of a `$PH_HOME` nobody has scheduled in, and a corrupt one
        is a cache that lost its contents. Both mean "this file cannot tell you
        what is due", and the honest response is the behaviour that shipped
        before the index existed — a late run — rather than an exception on the
        daemon's boot path.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError):
            log.warning("ph.seams.schedule_index: could not read %s", self.path, exc_info=True)
            return {}
        if not isinstance(raw, dict) or raw.get("version") != _VERSION:
            return {}
        found: dict[str, Appointment] = {}
        for session_id, entry in (raw.get("sessions") or {}).items():
            if not isinstance(entry, dict):
                continue
            next_at, updated = entry.get("nextAt"), entry.get("updated")
            if (
                isinstance(session_id, str)
                and isinstance(next_at, int)
                and isinstance(updated, int)
            ):
                found[session_id] = Appointment(session_id, next_at, updated)
        return found

    def record(self, session_id: str, *, next_at: int | None, now: int) -> None:
        """Set or clear one session's appointment.

        `next_at=None` removes the entry, which is what a cancellation and a
        `once` that has fired both mean: nothing further is owed. Removing rather
        than tombstoning keeps the file the size of the work outstanding instead
        of the size of every schedule ever made.

        Read-modify-write under a lock, because two sessions in two processes
        legitimately schedule at the same moment and each knows only its own
        entry. Best-effort throughout — an index that cannot be written costs a
        late run, and taking a session's own `create` down to protect a cache
        would be the projection outranking the log.

        **A no-op when nothing moved**, which is what makes this callable on every
        session open: reconciling an entry that is already right must not cost a
        lock and a rewrite, and every fork and every subagent opens a session.
        """
        try:
            with FileLock(f"{self.path}.lock", timeout=_LOCK_TIMEOUT, thread_local=False):
                found = self.read()
                current = found.get(session_id)
                if next_at is None:
                    if current is None:
                        return
                    found.pop(session_id)
                else:
                    if current is not None and current.next_at == next_at:
                        return
                    found[session_id] = Appointment(session_id, next_at, now)
                self._write(found)
        except Timeout:
            log.warning("ph.seams.schedule_index: %s is locked; not recording", self.path)
        except OSError:
            log.warning("ph.seams.schedule_index: could not write %s", self.path, exc_info=True)

    def _write(self, found: dict[str, Appointment]) -> None:
        """Serialize and rename, so a reader never sees a half-written index."""
        document: dict[str, Any] = {
            "version": _VERSION,
            "sessions": {
                one.session_id: {"nextAt": one.next_at, "updated": one.updated}
                for one in found.values()
            },
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(document, indent=2), encoding="utf-8")
        temporary.replace(self.path)
