"""One small JSON file beside the sessions it indexes: the machinery the schedule index
and the live-roots index share.

Both are caches over facts the logs still hold (I-6), so both answer five questions
the same way, and the answers are written once here:

- **Reading.** A file that is missing, unreadable, corrupt or of another version reads
  as empty and incomplete: the safe answer on a daemon's boot path. It is also how a
  format change migrates. An old file is rebuilt from the logs once, never converted.
- **Completeness.** Only a rebuild marks the file complete (`vouch`). Every other
  writer keeps the flag it found, so a reader can tell a file a rebuild vouched for
  from one that writers have only ever added to.
- **Writing.** Read, change and atomic rename under a file lock, so a reader never
  sees half a file and two hosts writing at once do not lose each other's entries.
- **No-op first.** A change that would not move the file is decided before the lock
  (K10) and again under it, so re-stating what the file says costs a read.
- **The rebuild merge.** What a rebuild did not read, it keeps, since a log it could
  not read is no evidence. What moved while it read, it keeps as moved.

What differs is each index's own: its file name, its version, its entry and how one
is spelled on disk, and anything beyond the file, such as the schedule index's claims.

@module ph.seams._json_index
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from ..locks import LockBusy, file_lock
from ..paths import write_atomic

__all__ = ["LOCK_TIMEOUT", "JsonIndex"]

log = logging.getLogger("ph.seams")

LOCK_TIMEOUT = 5.0
"""How long a write waits for the file lock before giving up (K10).

**Patient, because the wait is a worker thread's.** It was cut to a quarter second
while the schedule index's `record` ran on the event loop, where every caller reached
it and the wait was time the daemon served nothing. Both indexes write through a
writer that moves the write off the loop, and a wait there costs a thread and delays
only the writes queued behind it, so waiting out another host's write is worth more
than dropping this one. Giving up is still the answer past it: an index that cannot be
written costs a late answer, never a wrong one.
"""


@dataclass(slots=True)
class JsonIndex[E]:
    """One index file under `root`: `{version, complete, sessions: {id: entry}}`.

    A subclass names its file (`NAME`), its format (`VERSION`) and itself in messages
    (`WHAT`), and says how one entry reads and writes (`_decode`, `_encode`). The
    reading, locking, rewriting and merging are here.
    """

    root: Path
    NAME: ClassVar[str] = ""
    VERSION: ClassVar[int] = 0
    WHAT: ClassVar[str] = "an index"

    @property
    def path(self) -> Path:
        return self.root / self.NAME

    def _decode(self, key: str, raw: Mapping[str, Any]) -> E | None:
        """One entry as the file spells it, or `None` to drop one this build cannot read."""
        raise NotImplementedError

    def _encode(self, entry: E) -> dict[str, Any]:
        """One entry as the file spells it."""
        raise NotImplementedError

    def read(self) -> dict[str, E]:
        """Every entry on record. Empty when there is nothing to say.

        **Every failure reads as empty**, deliberately: a missing file is the ordinary
        state of a `$PH_HOME` nothing has been recorded in, and a corrupt one is a
        cache that lost its contents. Both mean "this file cannot tell you", and the
        answer is an empty one rather than an exception on a daemon's boot path. Both
        also read as incomplete (`_load`)."""
        return self._load()[0]

    def _load(self) -> tuple[dict[str, E], bool]:
        """The entries on record, and whether the file says it is complete."""
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {}, False
        except (OSError, ValueError):
            log.warning("ph.seams: could not read %s (%s)", self.path, self.WHAT, exc_info=True)
            return {}, False
        if not isinstance(raw, dict) or raw.get("version") != self.VERSION:
            return {}, False
        sessions = raw.get("sessions")
        found: dict[str, E] = {}
        for key, entry in (sessions if isinstance(sessions, dict) else {}).items():
            if not isinstance(key, str) or not isinstance(entry, dict):
                continue
            decoded = self._decode(key, entry)
            if decoded is not None:
                found[key] = decoded
        return found, raw.get("complete") is True

    def _change(
        self,
        moves: Callable[[Mapping[str, E]], bool],
        apply: Callable[[dict[str, E]], None],
    ) -> bool:
        """`apply` a writer's change, unless `moves` says it would not move the file.
        Whether the file now holds the change.

        **The no-op is decided before the lock, not inside it** (K10). Both indexes
        are re-stated far more often than they move, and paying a lock to discover
        nothing moved was what made a reconcile on every open cost something. Reading
        first is safe: a read that fails is empty, and a write is an atomic rename, so
        there is no torn state to see, and a racing writer only means the second ask
        under the lock finds nothing to do.
        """
        if not moves(self._load()[0]):
            return True

        def under_lock(found: dict[str, E]) -> bool:
            if not moves(found):
                return False
            apply(found)
            return True

        return self._rewrite(under_lock, vouch=False)

    def _merge(self, read: Mapping[str, E | None], *, before: Mapping[str, E], vouch: bool) -> bool:
        """Write a rebuild's answer: each entry `read` holds, or its removal for `None`.
        Whether it was written.

        **What the rebuild did not read, it keeps**, since a log it could not read or
        never found is no evidence the entry is gone. **What moved while it read, it
        keeps as moved**: an entry that differs from `before`, the file as the rebuild
        began, was written meanwhile by a writer who knew better. `vouch` marks the
        file complete; a rebuild that read a guessed store does not.
        """

        def apply(found: dict[str, E]) -> bool:
            for key, entry in read.items():
                if found.get(key) != before.get(key):
                    continue
                if entry is None:
                    found.pop(key, None)
                else:
                    found[key] = entry
            return True

        return self._rewrite(apply, vouch=vouch)

    def _rewrite(self, apply: Callable[[dict[str, E]], bool], *, vouch: bool) -> bool:
        """Read, `apply`, and write back, under the lock. Whether the file holds the
        change; `apply` returns `False` when there is nothing to write.

        `vouch` marks the file complete, which only a rebuild may. Otherwise the file
        keeps the `complete` it had.
        """
        try:
            with file_lock(f"{self.path}.lock", timeout=LOCK_TIMEOUT, what=self.WHAT):
                found, complete = self._load()
                if apply(found):
                    self._write(found, complete=vouch or complete)
        except LockBusy:
            log.warning("ph.seams: %s is locked; %s not written", self.path, self.WHAT)
            return False
        except OSError:
            log.warning("ph.seams: could not write %s (%s)", self.path, self.WHAT, exc_info=True)
            return False
        return True

    def _write(self, found: Mapping[str, E], *, complete: bool) -> None:
        """Serialize and rename, so a reader never sees a half-written index. Keys
        sorted, so two writers writing the same facts write the same bytes."""
        document: dict[str, Any] = {
            "version": self.VERSION,
            "complete": complete,
            "sessions": {key: self._encode(entry) for key, entry in found.items()},
        }
        write_atomic(self.path, json.dumps(document, indent=2, sort_keys=True))
