"""Which roots a host holds busy, and why — the live-roots index (P13-01).

**A root outlives its process; the thing that would resume it does not.** The log
keeps everything a restart needs — an open turn, a goal nobody settled, a child still
working, a prompt nobody claimed — and a fresh daemon mounts none of it, because
`Supervisor.rehydrate` reads only the schedule index and `NON_GUARANTEES` says so.
The work survives and is never picked up, which fails in the shape the schedule
index was built against: **silence**.

**The shape is the schedule index's, for its reasons.** "Mount every stored session
at boot" reads hundreds of logs to find the few that matter, mounts a profile, a
workspace and possibly a kernel for each, and claims every lease on the machine. So
this is an index: the host knows the moment a root is mounted, a turn starts or
ends, a goal opens or settles, a child is admitted or ends, a prompt lands in an
idle inbox, and a root is released — and it records, per root, which of those
*holds* still stand. A daemon, or the thin client that manages one, reads one small
file and resumes only what is owed.

**It is a cache, and the logs stay authoritative (I-6).** Every hold here is derived
from records a log still holds, so a missing, stale or corrupt index costs a root
left for the next pass and never a wrong one: `holds_of` is the fold a rebuild runs
over a stored log, and `rebuild_live_index` writes what every stored root's log
says. A file that cannot vouch for itself — missing, of another version, or
rewritten by a writer that found it so — reads as incomplete, and a reader that
needs certainty rebuilds first (`LiveSurvey.trusted`).

**What a hold means.** `turn`: the root's agent is mid-turn, or its log ends inside
one. `inbox`: a prompt is pending that no turn has claimed. `goal`: an `/autonomous`
goal is open. `children`: a sub-agent of this root is still live in its own log. A
root with no hold is not in the file. `failed` rides beside the holds: a root whose
ladder is spent still owes its work, and a resumer must know not to loop on it.

@module ph.seams.live_index
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypeAlias, get_args

import anyio
from anyio.abc import TaskGroup

from ..agent.inbox import replay_pending
from ..json import as_bool, as_int, as_obj, as_seq, as_str
from ..locks import LockBusy, file_lock
from ..paths import write_atomic
from ..persistence.protocol import SURVEY_LIMIT, SessionArchive
from ..session import SessionEvent, SessionHeader, open_turn_at
from ..wire import literal_lookup
from .goals import SET as GOAL_SET
from .goals import SETTLED as GOAL_SETTLED
from .goals import GoalState, fold_goal_event, open_goal
from .subagents import child_is_live, stored_tree

__all__ = [
    "HOLDS",
    "HOLD_TYPES",
    "INDEX_NAME",
    "Hold",
    "LiveEntry",
    "LiveIndex",
    "LiveIndexWriter",
    "LiveSurvey",
    "holds_of",
    "ordered_holds",
    "rebuild_live_index",
    "stored_children_live",
]

log = logging.getLogger("ph.seams.live_index")

Hold: TypeAlias = Literal["turn", "goal", "children", "inbox"]
"""Why a root still owes work. Closed, so a fifth reason is a type error here rather
than a word a resumer does not know."""

HOLDS: tuple[Hold, ...] = get_args(Hold)
"""Every hold, in the one order an entry lists them (`ordered_holds`)."""

_HOLD_NAMES: Mapping[str, Hold] = literal_lookup(Hold)
"""The read side: a hold named in the file that this build does not know is dropped
on read rather than carried as a string nothing branches on."""

HOLD_TYPES: frozenset[str] = frozenset(
    {"turn/start", "turn/end", GOAL_SET, GOAL_SETTLED, "agent/inbox/spliced"}
)
"""The records a root's holds are folded from — all a rebuild reads of a root's log,
so it validates five types and not every streamed chunk."""

INDEX_NAME = "live.json"
"""One file per `$PH_HOME`, beside `schedules.json`: read whole by whoever resumes,
written only when a root's holds move."""

_VERSION = 1

_LOCK_TIMEOUT = 5.0
"""How long a write waits for the file lock — a worker thread's wait, so patient
(`schedule_index._LOCK_TIMEOUT` says why)."""


def ordered_holds(found: Iterable[Hold]) -> tuple[Hold, ...]:
    """`found` in `HOLDS` order, so the host and a rebuild that agree write one entry."""
    held = set(found)
    return tuple(hold for hold in HOLDS if hold in held)


@dataclass(frozen=True, slots=True)
class LiveEntry:
    """One root that owes work, and why."""

    session_id: str
    holds: tuple[Hold, ...]
    updated: int = field(compare=False)
    """When this entry was written, epoch ms — so, since holds that have not moved
    are not written again, when the root has owed what it owes now. **Not compared**:
    two entries that differ only in when they were said are the same claim, which is
    what lets a host re-state one without a write."""
    family: str = ""
    """Where the log is filed, so a resumer opens it by path rather than by search."""
    cwd: str = ""
    profile: str = ""
    """The named profile the root ran on, or `""` when the host did not say — a
    rebuild from a log does not."""
    failed: bool = False
    """The root's ladder was spent and nothing has succeeded since. It still owes
    its work; a resumer that retried it would be one more attempt per restart."""

    def to_json(self) -> dict[str, Any]:
        return {
            "holds": list(self.holds),
            "updated": self.updated,
            "family": self.family,
            "cwd": self.cwd,
            "profile": self.profile,
            "failed": self.failed,
        }

    @classmethod
    def from_json(cls, session_id: str, raw: Mapping[str, Any]) -> LiveEntry:
        named = (_HOLD_NAMES.get(as_str(one)) for one in as_seq(raw.get("holds")))
        return cls(
            session_id=session_id,
            holds=ordered_holds(hold for hold in named if hold is not None),
            updated=as_int(raw.get("updated")),
            family=as_str(raw.get("family")),
            cwd=as_str(raw.get("cwd")),
            profile=as_str(raw.get("profile")),
            failed=as_bool(raw.get("failed")),
        )


@dataclass(frozen=True, slots=True)
class LiveSurvey:
    """What the index holds, and whether it can vouch for it."""

    entries: dict[str, LiveEntry]
    complete: bool
    """Whether a rebuild wrote this file and every writer since kept it whole. A
    reader that needs certainty rebuilds when this is false."""


@dataclass(slots=True)
class LiveIndex:
    """The index as a value, so a caller states where it lives and nothing guesses."""

    root: Path

    @property
    def path(self) -> Path:
        return self.root / INDEX_NAME

    def read(self) -> dict[str, LiveEntry]:
        """Every root on record. Empty when there is nothing to say."""
        return self.survey().entries

    def survey(self) -> LiveSurvey:
        """The entries, and whether the file can vouch for them.

        **Every failure reads as empty and incomplete**, as the schedule index reads:
        a missing file is the ordinary state of a `$PH_HOME` where nothing has run,
        and a corrupt one is a cache that lost its contents.
        """
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return LiveSurvey({}, complete=False)
        except (OSError, ValueError):
            log.warning("ph.seams.live_index: could not read %s", self.path, exc_info=True)
            return LiveSurvey({}, complete=False)
        document = as_obj(raw)
        if as_int(document.get("version")) != _VERSION:
            return LiveSurvey({}, complete=False)
        entries = {
            session_id: LiveEntry.from_json(session_id, as_obj(entry))
            for session_id, entry in as_obj(document.get("sessions")).items()
        }
        return LiveSurvey(entries, complete=as_bool(document.get("complete")))

    def record(self, session_id: str, entry: LiveEntry | None) -> bool:
        """Set or clear one root's entry. Blocking — see `LiveIndexWriter`."""
        return self.record_all({session_id: entry})

    def record_all(self, changes: Mapping[str, LiveEntry | None]) -> bool:
        """Set or clear several roots' entries — `None` clears: one read, and at most
        one rewrite. Whether the file now holds `changes`.

        The no-op is decided before the lock (the schedule index's K10), and an entry
        that differs only in `updated` is no change.
        """
        if not _moves(self.read(), changes):
            return True

        def apply(found: dict[str, LiveEntry]) -> bool:
            if not _moves(found, changes):
                return False
            for session_id, entry in changes.items():
                if entry is None:
                    found.pop(session_id, None)
                else:
                    found[session_id] = entry
            return True

        return self._rewrite(apply, vouch=False)

    def replace(
        self, read: Mapping[str, LiveEntry | None], *, since: LiveSurvey, vouch: bool = True
    ) -> bool:
        """Write a rebuild's answer and mark the file complete. Whether it was written.

        `read` holds each root whose log the rebuild read: its entry, or `None` for a
        root that owes nothing. **What the rebuild did not read, it keeps** — a log
        it could not read is no evidence the work is done. **What moved while it
        read, it keeps as moved**: an entry that differs from `since` was written
        meanwhile by a host that knew better.

        `vouch` is whether the rebuild read the store the deployment keeps its logs
        in; one that read a guessed directory writes what it found and leaves the
        file as complete as it was.
        """
        before = since.entries

        def apply(found: dict[str, LiveEntry]) -> bool:
            for session_id, entry in read.items():
                if found.get(session_id) != before.get(session_id):
                    continue
                if entry is None:
                    found.pop(session_id, None)
                else:
                    found[session_id] = entry
            return True

        return self._rewrite(apply, vouch=vouch)

    def _rewrite(self, apply: Callable[[dict[str, LiveEntry]], bool], *, vouch: bool) -> bool:
        try:
            with file_lock(f"{self.path}.lock", timeout=_LOCK_TIMEOUT, what="the live-roots index"):
                found = self.survey()
                if apply(found.entries):
                    self._write(found.entries, complete=vouch or found.complete)
        except LockBusy:
            log.warning("ph.seams.live_index: %s is locked; not written", self.path)
            return False
        except OSError:
            log.warning("ph.seams.live_index: could not write %s", self.path, exc_info=True)
            return False
        return True

    def _write(self, found: Mapping[str, LiveEntry], *, complete: bool) -> None:
        """Serialize and rename, so a reader never sees a half-written index. Keys
        sorted, so two writers writing the same facts write the same bytes."""
        document: dict[str, Any] = {
            "version": _VERSION,
            "complete": complete,
            "sessions": {one.session_id: one.to_json() for one in found.values()},
        }
        write_atomic(self.path, json.dumps(document, indent=2, sort_keys=True))


def _moves(found: Mapping[str, LiveEntry], changes: Mapping[str, LiveEntry | None]) -> bool:
    """Whether writing `changes` over `found` would change anything."""
    return any(found.get(session_id) != entry for session_id, entry in changes.items())


_UNSEEN = object()
"""What `_written` answers for a root this writer has never written — equal to no
entry and to no `None`."""


@dataclass(slots=True)
class LiveIndexWriter:
    """`LiveIndex.record` without the wait on the event loop (K10).

    Every host-side caller is on the loop at a transition, and `record` is a file
    lock, a read and an atomic rewrite. So `note` only notes the change, and a writer
    hands everything noted so far to `record_all` in one worker-thread hop.

    **Latest wins, per root, and a repeat costs nothing.** The writer remembers what
    it last wrote for each root, so a host re-stating an unchanged entry — which it
    does at every flush — is dropped in `note` without a thread hop or a read.
    `settle` is for a host shutting down: it waits for the batch in flight and
    whatever was noted behind it.
    """

    index: LiveIndex
    tasks: TaskGroup
    _pending: dict[str, LiveEntry | None] = field(default_factory=dict)
    _written: dict[str, LiveEntry | None] = field(default_factory=dict)
    """What the file was last made to hold for each root by this writer."""
    _writing: bool = False
    _idle: anyio.Event = field(default_factory=anyio.Event)

    def note(self, session_id: str, entry: LiveEntry | None) -> None:
        """Note one root's holds; start the writer if none is running."""
        if session_id not in self._pending and self._written.get(session_id, _UNSEEN) == entry:
            return
        self._pending[session_id] = entry
        if not self._writing:
            self._writing = True
            self._idle = anyio.Event()
            self.tasks.start_soon(self._drain)

    async def settle(self) -> None:
        """Wait until everything noted so far is in the file, or was refused."""
        if self._writing:
            await self._idle.wait()

    async def _drain(self) -> None:
        try:
            while self._pending:
                batch, self._pending = self._pending, {}
                try:
                    written = await anyio.to_thread.run_sync(self.index.record_all, batch)
                except Exception:
                    log.warning("ph.seams.live_index: a batch was not written", exc_info=True)
                    written = False
                if written:
                    self._written.update(batch)
        finally:
            self._writing = False
            self._idle.set()


def holds_of(
    header: SessionHeader, events: Iterable[SessionEvent], *, children_live: bool
) -> tuple[Hold, ...]:
    """What a stored root's log says it still owes — the fold a rebuild runs.

    `events` needs only `HOLD_TYPES`. Over the root's **own** events
    (`first_own_seq`): a fork inherits its source's transcript, not its source's
    open turn, goal or queue. `children_live` is the caller's, because it is a
    question about other logs (`stored_children_live`).
    """
    own = [event for event in events if event.seq >= header.first_own_seq]
    goals: dict[str, GoalState] = {}
    for event in own:
        fold_goal_event(goals, event)
    found: list[Hold] = []
    if own and open_turn_at(own, len(own) - 1) is not None:
        found.append("turn")
    if open_goal(goals) is not None:
        found.append("goal")
    if children_live:
        found.append("children")
    if any(replay_pending(own).values()):
        found.append("inbox")
    return ordered_holds(found)


def stored_children_live(store: SessionArchive, session_id: str, family: str) -> bool:
    """Whether any sub-agent beneath `session_id` is still live in its own log — the
    family as `load_children` reads it (`stored_tree`)."""
    tree = stored_tree(store, session_id, family, set())
    return any(child_is_live(state) for level in tree.values() for state in level.values())


def rebuild_live_index(
    index: LiveIndex,
    store: SessionArchive,
    *,
    now: int,
    since: LiveSurvey,
    live: Mapping[str, LiveEntry | None] | None = None,
    limit: int = SURVEY_LIMIT,
    vouch: bool = True,
) -> bool:
    """Rebuild `index` from every stored root, for an index that cannot vouch for
    itself. Blocking: it reads every root's log. Whether it was written.

    `since` is the survey the caller decided on, so the file is not read again to
    learn what it already knows. `live` is what a host knows of the roots it holds
    mounted: their logs lag the host until the next barrier, so their entries are
    taken from it rather than from disk. A log that will not read is logged and
    kept as the file had it (`replace`), so a damaged family costs its own entry's
    freshness and not the rebuild.
    """
    read: dict[str, LiveEntry | None] = dict(live or {})
    for row in store.stored(limit=limit):
        if row.delegating_parent is not None or row.session_id in read:
            continue
        try:
            header, events = store.read_own(
                row.session_id, family=row.family or None, types=HOLD_TYPES
            )
            holds = holds_of(
                header,
                events,
                children_live=stored_children_live(store, row.session_id, header.family),
            )
        except Exception:
            log.warning(
                "ph.seams.live_index: could not read %s; its entry stands",
                row.session_id,
                exc_info=True,
            )
            continue
        read[row.session_id] = (
            LiveEntry(
                session_id=row.session_id,
                holds=holds,
                updated=now,
                family=header.family,
                cwd=header.cwd or "",
            )
            if holds
            else None
        )
    return index.replace(read, since=since, vouch=vouch)
