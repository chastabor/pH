"""One small JSON file beside the sessions that it indexes. This module holds the
code that the schedule index and the live-roots index share.

Both indexes are caches over facts that the logs still hold (I-6). So both
answer five questions in the same way, and the answers are written one time,
here:

- **Reading.** A file that is missing, unreadable, corrupt or of another version
  reads as empty and incomplete. That is the safe answer on the boot path of a
  daemon. A format change migrates in the same way: an old file is rebuilt from
  the logs one time and is never converted.
- **Completeness.** Only a rebuild marks the file complete (`vouch`). Every
  other writer keeps the flag that it found. A reader can therefore tell a file
  that a rebuild vouched for from a file that writers have only ever added to.
- **Writing.** A write is a read, a change and an atomic rename under a file
  lock. A reader therefore never sees half a file, and two hosts that write at
  the same time do not lose the entries of each other.
- **No-op first.** Whether a change would move the file is decided before the
  lock (K10) and again under it. To state again what the file already says
  therefore costs one read.
- **The rebuild merge.** A rebuild keeps what it did not read, because a log
  that it could not read is no evidence. It keeps what moved while it read, as
  moved.

Each index supplies what differs: its file name, its version, its entry and the
form of an entry on disk, and anything beyond the file, such as the claims of
the schedule index.

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
"""How long a write waits for the file lock before it gives up (K10).

**The wait is patient, because a worker thread does the waiting.** The value was
cut to a quarter of a second while the `record` of the schedule index ran on the
event loop. Every caller reached it there, and during the wait the daemon served
nothing. Both indexes now write through a writer that moves the write off the
event loop. A wait there costs a thread and delays only the writes that are
queued behind it. So to wait for the write of another host is worth more than to
drop this write. Past the timeout, the write still gives up: an index that
cannot be written costs a late answer and never a wrong one.
"""


@dataclass(slots=True)
class JsonIndex[E]:
    """One index file under `root`: `{version, complete, sessions: {id: entry}}`.

    A subclass names its file (`NAME`), its format (`VERSION`) and itself in
    messages (`WHAT`). It also says how one entry is read and written
    (`_decode`, `_encode`). The reading, the locking, the rewriting and the
    merging are here.
    """

    root: Path
    NAME: ClassVar[str] = ""
    VERSION: ClassVar[int] = 0
    WHAT: ClassVar[str] = "an index"

    @property
    def path(self) -> Path:
        return self.root / self.NAME

    def _decode(self, key: str, raw: Mapping[str, Any]) -> E | None:
        """Read one entry from its form in the file. Return `None` to drop an
        entry that this build cannot read."""
        raise NotImplementedError

    def _encode(self, entry: E) -> dict[str, Any]:
        """One entry in the form that the file holds."""
        raise NotImplementedError

    def read(self) -> dict[str, E]:
        """Every entry on record. The result is empty when there is nothing to say.

        **Every failure reads as empty, on purpose.** A missing file is the
        ordinary state of a `$PH_HOME` in which nothing has been recorded. A
        corrupt file is a cache that lost its contents. Both mean "this file
        cannot tell you", so the answer is an empty one and not an exception on
        the boot path of a daemon. Both also read as incomplete (`_load`)."""
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
        """`apply` the change of a writer, unless `moves` says that it would not
        move the file. Returns whether the file now holds the change.

        **The no-op is decided before the lock, and not inside it** (K10). Both
        indexes are stated again far more often than they move. To take a lock
        only to find that nothing moved is what gave a reconcile on every open
        its cost. To read first is safe. A read that fails is empty, and a write
        is an atomic rename, so there is no torn state to see. A racing writer
        only means that the second check, under the lock, finds nothing to do.
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
        """Write the answer of a rebuild: each entry that `read` holds, or its
        removal for `None`. Returns whether it was written.

        **The rebuild keeps what it did not read.** A log that it could not read
        or never found is no evidence that the entry is gone. **The rebuild
        keeps what moved while it read, as moved.** `before` is the file as the
        rebuild began. An entry that differs from `before` was written in the
        meantime by a writer that knew better. `vouch` marks the file complete.
        A rebuild that read a guessed store does not vouch.
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
        """Read, `apply`, and write back, under the lock. Returns whether the file
        holds the change. `apply` returns `False` when there is nothing to
        write.

        `vouch` marks the file complete, which only a rebuild may do. Otherwise
        the file keeps the `complete` that it had.
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
        """Serialize and rename, so that a reader never sees a half-written index.
        The keys are sorted, so two writers that write the same facts write the
        same bytes."""
        document: dict[str, Any] = {
            "version": self.VERSION,
            "complete": complete,
            "sessions": {key: self._encode(entry) for key, entry in found.items()},
        }
        write_atomic(self.path, json.dumps(document, indent=2, sort_keys=True))
