"""Which roots a host holds busy, and why: the live-roots index (P13-01).

**A root outlives its process, and the thing that would resume it does not.**
The log keeps everything that a restart needs: an open turn, a goal that nobody
settled, a child that is still working, a prompt that nobody claimed. A fresh
daemon mounts none of it on its own. `Supervisor.rehydrate` reads only the
schedule index, and `NON_GUARANTEES` says that a daemon resumes owed roots only
when it is asked. Without a record of what is owed, the work survives and
nothing picks it up. That failure has the form that the schedule index was built
against: **silence**.

**The index has the form of the schedule index, for the same reasons.** "Mount
every stored session at boot" reads hundreds of logs to find the few that
matter. It mounts a profile, a workspace and possibly a kernel for each, and it
claims every lease on the machine. So this is an index. The host knows the
moment at which a root is mounted, a turn starts or ends, a goal opens or
settles, a child is admitted or ends, a prompt lands in an idle inbox, and a
root is released. It records, for each root, which of those *holds* still stand.
A daemon reads one small file and resumes only what is owed. The thin client
that manages a daemon asks the daemon for the same list (`sessions/owed`).

**The index is a cache, and the logs stay authoritative (I-6).** Every hold here
is derived from records that a log still holds. A missing, stale or corrupt
index therefore costs a root that is left for the next pass, and never a wrong
one. `holds_of` is the fold that a rebuild runs over a stored log, and
`rebuild_live_index` writes what the log of every stored root says. A file that
cannot vouch for itself reads as incomplete. That is a file that is missing or
of another version, or a file that a writer wrote again after it found the file
so. A reader that needs certainty rebuilds first (`LiveSurvey.complete`).

**What a hold means.**

- `turn`: the agent of the root is in the middle of a turn, or its log ends
  inside one.
- `inbox`: a prompt is pending that no turn has claimed.
- `goal`: an `/autonomous` goal is open.
- `children`: a sub-agent of this root is still live in its own log.

Two more facts are recorded beside the holds. `failed`: a root whose ladder is
spent still owes its work, and a resumer must know not to loop on it. `damaged`:
a log beneath the root cannot be read, so what that log owes is unknown
(P13-09). A root with no hold and nothing damaged is not in the file
(`LiveEntry.listed`).

@module ph.seams.live_index
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, TypeAlias, get_args

import anyio
from anyio.abc import TaskGroup

from ..agent.inbox import replay_pending
from ..json import as_bool, as_int, as_seq, as_str
from ..persistence.protocol import SessionArchive
from ..session import SessionEvent, open_turn_at
from ..wire import literal_lookup
from ._json_index import JsonIndex
from .goals import SET as GOAL_SET
from .goals import SETTLED as GOAL_SETTLED
from .goals import GoalState, fold_goal_event, open_goal
from .subagents import child_is_live, stored_tree

__all__ = [
    "HOLDS",
    "HOLD_TYPES",
    "INDEX_NAME",
    "Hold",
    "Ladder",
    "LiveEntry",
    "LiveIndex",
    "LiveIndexWriter",
    "LiveSurvey",
    "holds_of",
    "ordered_holds",
    "rebuild_live_index",
    "stored_children",
]

log = logging.getLogger("ph.seams.live_index")

Hold: TypeAlias = Literal["turn", "goal", "children", "inbox"]
"""Why a root still owes work. The set is closed, so a fifth reason is a type
error here, and not a word that a resumer does not know."""

HOLDS: tuple[Hold, ...] = get_args(Hold)
"""Every hold, in the one order in which an entry lists them (`ordered_holds`)."""

_HOLD_NAMES: Mapping[str, Hold] = literal_lookup(Hold)
"""The lookup for the read side. A hold that the file names and that this build
does not know is dropped on read. It is not carried as a string that nothing
branches on."""

HOLD_TYPES: frozenset[str] = frozenset(
    {"turn/start", "turn/end", GOAL_SET, GOAL_SETTLED, "agent/inbox/spliced"}
)
"""The records that the holds of a root are folded from. From the log of a root,
a rebuild reads only these five types and the types of the ladder
(`Ladder.types`). So it validates those types and not every streamed chunk."""


@dataclass(frozen=True, slots=True)
class Ladder:
    """The retry ladder of a host, as a rebuild reads it. It holds the record
    types that the ladder is folded from, and the fold over the own events of a
    root that says whether the ladder is spent (`LiveEntry.failed`).

    The ladder belongs to the host and not to this seam. Which records say that
    a root was retried and gave up is for `ph_app.daemon.recovery` to declare. A
    rebuild must be given a ladder. A rebuild that did not ask would write every
    root back as not failed. A resumer that trusts the file would then make one
    more attempt at each restart."""

    types: frozenset[str]
    spent: Callable[[Sequence[SessionEvent]], bool]


INDEX_NAME = "live.json"
"""One file for each `$PH_HOME`, beside `schedules.json`. Whoever resumes reads
it whole, and it is written only when the holds of a root move."""


def ordered_holds(found: Iterable[Hold]) -> tuple[Hold, ...]:
    """`found` in the order of `HOLDS`. The host and a rebuild that agree
    therefore write the same entry."""
    held = set(found)
    return tuple(hold for hold in HOLDS if hold in held)


@dataclass(frozen=True, slots=True)
class LiveEntry:
    """One root that owes work, and why."""

    session_id: str
    holds: tuple[Hold, ...]
    updated: int = field(compare=False)
    """When this entry was written, in epoch ms. Holds that have not moved are
    not written again, so this is also the time since which the root has owed
    what it owes now. **The field is not compared.** Two entries that differ
    only in when they were written are the same claim. A host can therefore
    state an entry again without a write."""
    family: str = ""
    """Where the log is filed, so that a resumer opens it by path and not by
    search."""
    cwd: str = ""
    profile: str = ""
    """The named profile that the root ran on, or `""` when the host did not say.
    A rebuild from a log does not say."""
    failed: bool = False
    """The ladder of the root was spent, and nothing has succeeded since. The
    root still owes its work. A resumer that retried it would make one more
    attempt at each restart."""
    damaged: tuple[str, ...] = ()
    """The logs beneath this root that cannot be read, by session id (P13-09).
    What they owe is unknown: a child in the middle of a turn, in a file that
    nobody can open, is counted as not live. So a root is listed for its damage
    alone, with no holds. Nothing that a resume does changes that entry.
    `phern doctor sessions --repair` does."""

    @property
    def listed(self) -> bool:
        """Whether this root belongs in the file: it holds work, or a log beneath
        it cannot be read. Both writers ask this, so they cannot disagree."""
        return bool(self.holds or self.damaged)

    def to_json(self) -> dict[str, Any]:
        return {
            "holds": list(self.holds),
            "updated": self.updated,
            "family": self.family,
            "cwd": self.cwd,
            "profile": self.profile,
            "failed": self.failed,
            "damaged": list(self.damaged),
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
            damaged=tuple(as_str(one) for one in as_seq(raw.get("damaged"))),
        )


@dataclass(frozen=True, slots=True)
class LiveSurvey:
    """What the index holds, and whether it can vouch for it."""

    entries: dict[str, LiveEntry]
    complete: bool
    """Whether a rebuild wrote this file and every writer since kept it whole. A
    reader that needs certainty rebuilds when this is false."""


@dataclass(slots=True)
class LiveIndex(JsonIndex[LiveEntry]):
    """The index as a value, so that a caller states where it lives and nothing
    guesses.

    `JsonIndex` does the reading, the locking, the rewriting and the rebuild
    merge of the file, and the schedule index shares it. A missing, corrupt or
    older file therefore reads as empty and incomplete here exactly as it does
    there.
    """

    NAME: ClassVar[str] = INDEX_NAME
    VERSION: ClassVar[int] = 1
    WHAT: ClassVar[str] = "the live-roots index"

    def survey(self) -> LiveSurvey:
        """The entries, and whether the file can vouch for them."""
        entries, complete = self._load()
        return LiveSurvey(entries, complete)

    def _decode(self, key: str, raw: Mapping[str, Any]) -> LiveEntry:
        return LiveEntry.from_json(key, raw)

    def _encode(self, entry: LiveEntry) -> dict[str, Any]:
        return entry.to_json()

    def record_all(self, changes: Mapping[str, LiveEntry | None]) -> bool:
        """Set or clear the entries of several roots, where `None` clears: one
        read, and at most one rewrite. Returns whether the file now holds
        `changes`. An entry that differs only in `updated` is no change."""

        def apply(found: dict[str, LiveEntry]) -> None:
            for session_id, entry in changes.items():
                if entry is None:
                    found.pop(session_id, None)
                else:
                    found[session_id] = entry

        return self._change(lambda found: _moves(found, changes), apply)

    def replace(
        self, read: Mapping[str, LiveEntry | None], *, since: LiveSurvey, vouch: bool = True
    ) -> bool:
        """Write the answer of a rebuild by the rule of `JsonIndex._merge`, and
        mark the file complete when `vouch` is set. Returns whether it was
        written."""
        return self._merge(read, before=since.entries, vouch=vouch)


def _moves(found: Mapping[str, LiveEntry], changes: Mapping[str, LiveEntry | None]) -> bool:
    """Whether writing `changes` over `found` would change anything."""
    return any(found.get(session_id) != entry for session_id, entry in changes.items())


_UNSEEN = object()
"""The default that the lookup in `_written` returns for a root that this writer
has never written. It equals no entry, and it does not equal `None`."""


@dataclass(slots=True)
class LiveIndexWriter:
    """`LiveIndex.record_all` without the wait on the event loop (K10).

    Every caller in the host is on the event loop at a transition, and
    `record_all` is a file lock, a read and an atomic rewrite. So `note` only notes the
    change, and a writer hands everything noted so far to `record_all` in one
    hop to a worker thread.

    **The latest change wins, for each root, and a repeat costs nothing.** The
    writer remembers what it last wrote for each root. A host states an
    unchanged entry again at every flush that wrote something. `note` drops such
    an entry without a thread hop or a read.

    **`settle` writes, and does not wait for the background drain to write.** It
    is for a host that shuts down. Such a host can be unwinding with `tasks`
    already cancelled: one of its tasks failed, or it was cancelled with nothing
    to turn that into a stop. The drain that its last `note` starts is spawned
    into that group. There the drain either never runs or is cut off before its
    write. So `settle` drains in the task of the caller, which a shutdown
    shields, after any batch in flight has landed. A batch that a cancel cut off
    goes back, and whoever writes next writes it.
    """

    index: LiveIndex
    tasks: TaskGroup
    _pending: dict[str, LiveEntry | None] = field(default_factory=dict)
    _written: dict[str, LiveEntry | None] = field(default_factory=dict)
    """What this writer last made the file hold for each root."""
    _scheduled: bool = False
    """Whether a background drain was started and has not ended. A drain that is
    cancelled before its first step never clears the flag. Only a cancelled
    group cancels a drain in that way. That is a shutdown, and there `settle`
    drains for itself."""
    _batch: anyio.Lock = field(default_factory=anyio.Lock)
    """Held while a batch is written, by whichever task writes it. Only one batch
    is written at a time, so a later batch never lands under an earlier one."""

    def note(self, session_id: str, entry: LiveEntry | None) -> None:
        """Note the holds of one root, and start the writer if none is running."""
        if session_id not in self._pending and self._written.get(session_id, _UNSEEN) == entry:
            return
        self._pending[session_id] = entry
        if not self._scheduled:
            self._scheduled = True
            self.tasks.start_soon(self._drain)

    async def settle(self) -> None:
        """Write everything noted so far, or let the write be refused, in the task
        of the caller."""
        async with self._batch:
            while self._pending:
                batch, self._pending = self._pending, {}
                try:
                    written = await anyio.to_thread.run_sync(self.index.record_all, batch)
                except Exception:
                    log.warning("ph.seams.live_index: a batch was not written", exc_info=True)
                    written = False
                except BaseException:
                    # Cancelled. Either the thread had not run, or it had run, and
                    # then to write the same entries again is no change. The batch
                    # goes back under anything that was noted since.
                    self._pending = {**batch, **self._pending}
                    raise
                if written:
                    self._written.update(batch)

    async def _drain(self) -> None:
        try:
            await self.settle()
        finally:
            self._scheduled = False


def holds_of(own: Sequence[SessionEvent], *, children_live: bool) -> tuple[Hold, ...]:
    """What the log of a stored root says that the root still owes. This is the
    fold that a rebuild runs.

    `own` is the **own** events of the root (`first_own_seq`), and only the
    `HOLD_TYPES` among them are needed. A fork inherits the transcript of its
    source, and not the open turn, the goal or the queue of its source.
    `children_live` comes from the caller, because it is a question about other
    logs (`stored_children`).
    """
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


def stored_children(
    store: SessionArchive, session_id: str, family: str
) -> tuple[bool, tuple[str, ...]]:
    """Whether any sub-agent beneath `session_id` is still live in its own log,
    and which logs beneath it cannot be read. The family is read as
    `load_children` reads it (`stored_tree`), so a rebuild and a mount cannot
    disagree about either answer."""
    tree = stored_tree(store, session_id, family, set())
    live = any(child_is_live(state) for level in tree.states.values() for state in level.values())
    return live, tuple(sorted(tree.unreadable))


def rebuild_live_index(
    index: LiveIndex,
    store: SessionArchive,
    *,
    now: int,
    since: LiveSurvey,
    ladder: Ladder,
    live: Mapping[str, LiveEntry | None] | None = None,
    vouch: bool = True,
) -> bool:
    """Rebuild `index` from every stored root, for an index that cannot vouch for
    itself. The function blocks: it reads the log of every root. Returns whether
    the index was written.

    `since` is the survey on which the caller decided, so the file is not read
    again to learn what the caller already knows. `live` is what a host knows of
    the roots that it holds mounted. Their logs lag the host until the next
    barrier, so their entries are taken from `live` and not from disk. `ladder`
    is the fold of the host for `failed` (`Ladder`).

    A root whose own log cannot be read is logged and kept as the file had it
    (`replace`). It costs the freshness of its own entry and not the rebuild. A
    child log that cannot be read goes into the `damaged` of the root, and the
    root is listed for it whether or not it holds anything else.

    **The rebuild reads every stored log, with no cap**, as the rebuild of the
    schedule index does. The listing is flat across families, so sub-agent logs
    are in it beside the roots, and each is skipped at the cost of one header.
    Under a cap they would crowd out the roots that the rebuild exists to find.
    A root that died busy would then be missed after the crash that this rebuild
    is for.
    """
    read: dict[str, LiveEntry | None] = dict(live or {})
    types = HOLD_TYPES | ladder.types
    for row in store.stored(limit=None):
        if row.delegating_parent is not None or row.session_id in read:
            continue
        try:
            header, events = store.read_own(row.session_id, family=row.family or None, types=types)
            own = [event for event in events if event.seq >= header.first_own_seq]
            children_live, damaged = stored_children(store, row.session_id, header.family)
            holds = holds_of(own, children_live=children_live)
            failed = ladder.spent(own)
        except Exception:
            log.warning(
                "ph.seams.live_index: could not read %s; its entry stands",
                row.session_id,
                exc_info=True,
            )
            continue
        entry = LiveEntry(
            session_id=row.session_id,
            holds=holds,
            updated=now,
            family=header.family,
            cwd=header.cwd or "",
            failed=failed,
            damaged=damaged,
        )
        read[row.session_id] = entry if entry.listed else None
    return index.replace(read, since=since, vouch=vouch)
