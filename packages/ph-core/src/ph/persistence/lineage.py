"""Reading a session that stores only its own events (Session_Tree_Feasibility §5).

A fork used to **copy**. `SessionStore.fork` sliced `log[:boundary + 1]` and
seeded a new session with it, so the file of the child held its inherited
history verbatim. Ten forks of a 10 000-event session copied 100 000 events, and
the same logical event carried a different identity in the parent and in the
child.

Reference-forking stores the reference instead. A file holds only its own
contiguous run. Its header says where the rest comes from: `parent_session`
(which log) and `seed_length` (how many of the leading events of that log).
Both fields existed before reference-forking, where they *described* a copy
that had already happened. Now they *define* the inherited prefix.

**A file is complete exactly when its first event has `seq == 0`.**

That is the whole test, and it needs no new field. A root starts at 0. A child
that holds a copy of its prefix starts at 0, because the copy is in the file. A
reference-forked child starts at `seed_length`, which is exactly the number of
leading events that it is owed. So the first seq both *signals* the reference
and *measures* it. Every log written before this module existed starts at 0. So
chain-aware reading changes nothing for a log that holds no reference.

**The prefix that a child depends on is immutable**, because the log is
append-only. A parent may grow forever without invalidating a descendant. So
this needs no lock, no coordination and no invalidation. One writer per file is
a consequence of the design and is not enforced.

**A broken chain fails loudly.** A read cannot continue past a missing
ancestor. A silent partial read would reconstruct a *wrong* session and not an
incomplete one. The refusal of unknown types in `_readmit` exists to prevent
the same failure.

@module ph.persistence.lineage
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeAlias

from ..session import SessionEvent, SessionHeader

__all__ = [
    "MAX_DEPTH",
    "LineageError",
    "ReadOne",
    "ReadSome",
    "lineage_faults",
    "materialize",
    "materialize_some",
]

ReadOne: TypeAlias = Callable[
    [str, int | None, str | None], tuple[SessionHeader, list[SessionEvent]]
]
"""How one backend reads one stored log, unchained, up to a boundary.

This is a callable and not a Protocol method. So the walk lives in one place,
and each backend keeps its own `read` as the code that knows about files or
rows. JSONL and Turso disagree about everything below this line and about
nothing above it.

`family` is **not** a hint. Every member of a lineage shares one directory, so
the walk knows where each ancestor lives. The walk hands the directory over, and
the backend does not search. Without it, a chained read paid a directory scan
for each generation. That cost scales with the size of the store and not with
the length of the log.

`upto` is **a hint, not a contract**. Events at or above it are not wanted. A
backend that returns them anyway is slower but still correct, because the walk
filters what it takes in every case. The choice is deliberate. If correctness
depended on the bound, a backend that quietly ignored it would produce a *wrong*
log. A quiet error of that kind is how reference-forking became a silent no-op
on Turso one time already.
"""

ReadSome: TypeAlias = Callable[
    [str, int | None, str | None], tuple[SessionHeader, int | None, list[SessionEvent]]
]
"""`ReadOne` for a reader that keeps only some of the events of a file
(`materialize_some`).

It also returns the seq of the first event of the file. The first kept event
cannot say whether the file holds its own history, or where its inheritance
ends. `upto` binds here: every event that the reader keeps lies below it."""

MAX_DEPTH = 64
"""How many ancestors one materialization may walk.

This is not a correctness bound. A legal chain of any depth reconstructs
correctly. The bytes read are the same at any depth, because each file
contributes a disjoint slice. It is a bound on syscalls. It is also a second
backstop against a corrupted header that names its own descendant, behind the
cycle check of the walk. In both cases a report of the chain is more useful
than a walk that does not end.
"""


class LineageError(Exception):
    """The inherited prefix of a session cannot be assembled.

    It has its own type because the options of the caller differ from those for
    a missing file. A resume that meets this error has found a log whose
    ancestor was deleted or moved. The correct response is to name the ancestor,
    and not to start a fresh session on top of a half-read one.

    **`code` and `session_id` are the parts that a caller may branch on**, as in
    `SessionForkError`. A daemon that meets `MISSING_ANCESTOR` can offer to
    re-point or materialize. A daemon that meets `CYCLE` or `GAP` has corruption
    and should say so. Without a code, the only discriminator is the message. A
    test that matches prose is a contract that nobody wrote down and that every
    rewording breaks. `DaemonError.reason` in `ph_app` makes the same argument.
    """

    def __init__(self, message: str, *, code: str, session_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.session_id = session_id


def materialize(
    read_one: ReadOne, session_id: str, *, family: str | None = None
) -> tuple[SessionHeader, list[SessionEvent]]:
    """The full log of one session, following its lineage to a file that starts at 0.

    `family` is the family of the first file, when the caller holds it
    (`SessionArchive.read`). Each ancestor is found by the family that the
    header of its child names.

    Returns the header of the *child*. The lineage supplies events, never
    identity. A materialized log is what every reader already expects: dense
    from `seq == 0` and contiguous. So reference-forking changes nothing for
    `_readmit`, the surface fold, the cursors of the daemon and both
    derivations.

    **The slices tile exactly.** Each ancestor contributes only what the
    generation below it still lacks. So the assembled log holds every event
    exactly one time.

    **What is read is bounded too.** `upto` carries the fork boundary. So Turso
    adds `WHERE seq < ?` and JSONL stops reading lines. Neither parses and
    validates a whole ancestor to keep fifty of its events.

    The walk is iterative and not recursive. The chain is data from disk, and a
    hand-edited header that names a cycle should meet a bound and not the stack
    of the interpreter.
    """
    header, events = read_one(session_id, None, family)
    # A file that starts at 0 says that it holds its own history. Check that
    # against the header before believing it. `Session._append` mints
    # `seq = len(self._log)`. So a reference-forked child built with an empty
    # seed writes its *first* event at 0. It would be read as a whole session,
    # and its inherited prefix would be dropped silently. `fork` does not copy,
    # so the counter of the child has to start at `seed_length`. A wrong log
    # read as a right one is the exact failure that this module exists to make
    # loud.
    inherited = header.first_own_seq
    if events and events[0].seq == 0 and inherited > len(events):
        raise LineageError(
            f"session {session_id!r} starts at seq 0, so it claims to hold its own "
            f"history, but its header inherits {inherited} event(s) from "
            f"{header.parent_session!r} and the file holds only {len(events)}",
            code="TRUNCATED",
            session_id=session_id,
        )
    assembled, chain = _inherited(
        _with_first(read_one), session_id, header, _first_of(events), events
    )
    if len(chain) > 1:
        _assert_contiguous(assembled, session_id, chain)
    return header, assembled


def materialize_some(
    read: ReadSome, session_id: str, *, family: str | None = None
) -> tuple[SessionHeader, list[SessionEvent]]:
    """`materialize` for a reader that keeps only some of the events of each file.

    A host uses it for what it folds out of a lineage before anything is mounted:
    the profile records of the lineage. The rest is not parsed.

    The walk is the same (`_inherited`), so a broken chain raises the same
    `LineageError`. This function cannot check that the assembled log is dense,
    because that check takes every event. The strict read that follows (the
    `materialize` of the open) refuses a gap.
    """
    header, first, events = read(session_id, None, family)
    assembled, _chain = _inherited(read, session_id, header, first, events)
    return header, assembled


def _first_of(events: list[SessionEvent]) -> int | None:
    """The seq of the first event of an unfiltered read. It is the first seq of the
    file, and that seq says what the file inherits."""
    return events[0].seq if events else None


def _with_first(read_one: ReadOne) -> ReadSome:
    """A `ReadOne` as a `ReadSome`. The first event of an unfiltered read is the
    first event of its file."""

    def read(
        session_id: str, upto: int | None, family: str | None
    ) -> tuple[SessionHeader, int | None, list[SessionEvent]]:
        header, events = read_one(session_id, upto, family)
        return header, _first_of(events), events

    return read


def _owed(header: SessionHeader, first: int | None) -> int:
    """How many leading events a file is owed by its lineage: its first seq. When it
    has no event to read a seq from, the answer is its `seed_length` if it names a
    parent.

    **A child that owns nothing still inherits everything.** A fork lost its
    history when such a file was read as a complete empty log. Such files exist.
    At a time when the constructor laid the marker and skipped it for a seed
    that already ended in one, a fork taken at an end-seed was stored with a
    header and no events. With no first event, `seed_length` is the boundary,
    which is what `seed_length` means. A root with nothing in it names no
    parent, and is owed nothing.
    """
    if first is not None:
        return first
    return header.first_own_seq if header.parent_session else 0


def _inherited(
    read: ReadSome,
    session_id: str,
    header: SessionHeader,
    first: int | None,
    own: list[SessionEvent],
) -> tuple[list[SessionEvent], list[str]]:
    """`own`, with everything that its lineage supplies below it, oldest first, and
    the chain that it came from.

    `materialize` and `materialize_some` share this walk, so its rules are kept
    in one place. A file that starts at 0 holds its own history and is returned
    as it is.
    """
    # Collect the ancestors newest-first. Take from each only what the generation
    # below it still lacks. `owed` is always the first seq of the generation
    # below. It is both the count to take and the boundary at which the fork was
    # made.
    owed = _owed(header, first)
    chain = [session_id]
    pieces = [own]
    parent = header.parent_session
    while owed > 0:
        if parent is None:
            raise LineageError(
                f"session {session_id!r} begins at seq {owed} and names no parent; "
                "a log that does not start at 0 must say where its history came from",
                code="NO_PARENT",
                session_id=session_id,
            )
        if parent in chain:
            raise LineageError(
                f"session {session_id!r} has a cycle in its lineage: "
                f"{' -> '.join([*chain, parent])}",
                code="CYCLE",
                session_id=parent,
            )
        if len(chain) >= MAX_DEPTH:
            raise LineageError(
                f"session {session_id!r} is more than {MAX_DEPTH} deep; "
                f"the chain so far is {' -> '.join(chain)}",
                code="TOO_DEEP",
                session_id=session_id,
            )
        try:
            ancestor, starts, events = read(parent, owed, header.family)
        except Exception as error:
            raise LineageError(
                f"session {session_id!r} inherits {owed} event(s) from {parent!r}, "
                f"which could not be read: {error}",
                code="MISSING_ANCESTOR",
                session_id=parent,
            ) from error
        chain.append(parent)
        # Keep only the part below `owed`. An ancestor that kept working past the
        # boundary contributes nothing above it. The prefix is immutable, so the
        # result is stable however far the ancestor has grown since.
        pieces.append([event for event in events if event.seq < owed])
        if starts is not None and starts < owed:
            owed = starts
        # An ancestor that supplied **nothing** does not end the search. It is
        # itself a reference-fork at or above this boundary, so what is owed is
        # unchanged and the prefix lies further up. An earlier version set
        # `owed` to zero here, which stopped the walk at the first such link.
        # Every cycle and every chain that was too deep then became a "gap"
        # report. The cycle guard and the depth guard above were unreachable,
        # and the tests for them found the fault.
        parent = ancestor.parent_session
    return [event for piece in reversed(pieces) for event in piece], chain


def _assert_contiguous(events: list[SessionEvent], session_id: str, chain: list[str]) -> None:
    """The assembled log must be dense from 0, or the error must say exactly where
    it is not.

    `_readmit` would refuse a gap anyway, one layer up, and it would not know
    which ancestor was short. The check here turns "seed must be contiguous from
    0" into a sentence that names the lineage that failed to supply the events.
    """
    for index, event in enumerate(events):
        if event.seq != index:
            raise LineageError(
                f"session {session_id!r} assembled a log with a gap at index {index} "
                f"(found seq {event.seq}); its lineage is {' -> '.join(reversed(chain))}",
                code="GAP",
                session_id=session_id,
            )


def lineage_faults(
    listed: Iterable[tuple[str, str | None]], exists: Callable[[str, str], bool]
) -> list[tuple[str, str]]:
    """Stored sessions whose lineage will not materialize, and why (§5.4a).

    **Reference-forking has this cost, and copying did not.** A copied child was
    self-contained. A referencing child is readable only while the log that it
    names is still there. The plan was to refuse to remove a session that has
    descendants. But nothing in pH removes a session log, so there is no removal
    to refuse. The risk is a person with `rm`. The failure that reaches that
    person today is a `LineageError` at resume, long after the removal, for one
    session at a time.

    So this function asks the question the other way round: given everything on
    record, which logs are *already* unreadable? It answers from the
    **listing**. Both backends fill `StoredSession.parent` from a one-line
    header peek that they already paid for. So a whole store is surveyed and no
    log is opened.

    Rows are `(session_id, what is wrong)`, the shape that a `phern doctor`
    section takes. An empty list keeps the section off the page entirely.

    Three faults can be decided from the listing, and a fourth cannot:

    * **A missing ancestor.** It is settled with `exists`, not with membership
      of the listing. `stored()` takes a limit, and a parent below the cut is
      present, not gone. A report of those parents would be a false alarm on
      every large store. `exists(ancestor, descendant)` is asked with the listed
      session whose chain reached the ancestor, so the function that answers
      can look where that lineage is filed (`lineage_faults_of`).
    * **A cycle**, when it closes inside the listing.
    * **A chain past `MAX_DEPTH`**, which `materialize` refuses outright. It is
      easy to miss, because forks are shallow. But `roll` adds a generation for
      each segment, so a long-running segmented session reaches the bound by
      ordinary use. A survey that did not report it would be silent on exactly
      the failure that segmentation introduces.
    * **A gap** in the assembled seqs is *not* checked. That check needs every
      event of every ancestor, which is the read that this function exists to
      avoid. `materialize` catches a gap at the point of use, where the cost is
      already being paid.
    """
    parents = dict(listed)
    return [
        (session_id, fault)
        for session_id in parents
        if (fault := _chain_fault(session_id, parents, exists)) is not None
    ]


def _chain_fault(
    session_id: str, parents: dict[str, str | None], exists: Callable[[str, str], bool]
) -> str | None:
    """Why the chain of one session will not materialize, or `None` if it will.

    One chain has one answer. So this is a function, and not rows appended to a
    shared list. A session contributes at most one fault. The signature says so,
    and the three exits need not be read as side effects on an accumulator.
    """
    seen = {session_id}
    current = parents[session_id]
    while current is not None:
        if current in seen:
            return f"lineage cycles back through {current}"
        if len(seen) >= MAX_DEPTH:
            return f"lineage is more than {MAX_DEPTH} deep; no read will follow it"
        if current not in parents:
            # The ancestor is outside the listing. If it is present on disk, the
            # chain continues somewhere that this survey cannot see, which is
            # not a fault. If it is absent, every descendant of it is already
            # unreadable.
            return None if exists(current, session_id) else f"ancestor {current} is missing"
        seen.add(current)
        current = parents[current]
    return None
