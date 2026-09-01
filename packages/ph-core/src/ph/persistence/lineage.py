"""Reading a session that stores only its own events (Session_Tree_Feasibility §5).

Today a fork **copies**: `SessionStore.fork` slices `log[:boundary + 1]` and seeds
a new session with it, so the child's file holds its inherited history verbatim.
Ten forks of a 10 000-event session copy 100 000 events, and the same logical
event carries a different identity in parent and child.

Reference-forking stores the reference instead: a file holds only its own
contiguous run, and its header already says where the rest comes from —
`parent_session` (which log) and `seed_length` (how many of its leading events).
Both fields exist today, where they *describe* a copy that already happened;
here they *define* it.

**A file is complete exactly when its first event has `seq == 0`.**

That is the whole test, and it needs no new field. A root starts at 0. A
copy-forked child starts at 0, because the copy is right there. A
reference-forked child starts at `seed_length`, which is precisely the number of
leading events it is owed — so the first seq both *signals* the reference and
*measures* it. Every log written before this module existed starts at 0, which
is what makes chain-aware reading a no-op until something writes a reference.

**The prefix a child depends on is immutable**, because the log is append-only.
A parent may grow forever without invalidating a descendant, which is why this
needs no lock, no coordination and no invalidation — and why one writer per file
falls out rather than being enforced.

**A broken chain fails loudly.** A missing ancestor cannot be read past: a
silent partial read would reconstruct a *wrong* session rather than an incomplete
one, which is the failure `_readmit`'s unknown-type refusal exists to prevent.

@module ph.persistence.lineage
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from ..session import SessionEvent, SessionHeader

__all__ = ["MAX_DEPTH", "LineageError", "ReadOne", "materialise"]

ReadOne: TypeAlias = Callable[[str], tuple[SessionHeader, list[SessionEvent]]]
"""How one backend reads one stored log, unchained.

A callable rather than a Protocol method so the walk lives in one place and each
backend keeps its own `read` as the thing that knows about files or rows. JSONL
and Turso disagree about everything below this line and about nothing above it.
"""

MAX_DEPTH = 64
"""How many ancestors one materialisation may walk.

Not a correctness bound — a legal chain of any depth reconstructs correctly, and
the bytes read are the same either way, since each file contributes a disjoint
slice. It is a syscall bound and a cycle backstop: a corrupted header naming its
own descendant would otherwise recurse until the stack gave out, and reporting
the chain is more useful than a `RecursionError`.
"""


class LineageError(Exception):
    """A session's inherited prefix cannot be assembled.

    Its own type because the caller's options differ from a missing file's: a
    resume that meets this has found a log whose ancestor was deleted or moved,
    and the answer is to name the ancestor rather than to start a fresh session
    on top of a half-read one.

    **`code` and `session_id` are the parts a caller may branch on**, mirroring
    `SessionForkError`. A daemon meeting `MISSING_ANCESTOR` can offer to re-point
    or materialise; one meeting `CYCLE` or `GAP` has corruption and should say so.
    Without a code the only discriminator is the message, and a test matching
    prose is a contract nobody wrote down that every rewording breaks — which is
    the argument `DaemonError.reason` already makes one package over.
    """

    def __init__(self, message: str, *, code: str, session_id: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.session_id = session_id


def materialise(read_one: ReadOne, session_id: str) -> tuple[SessionHeader, list[SessionEvent]]:
    """One session's full log, following its lineage to a file that starts at 0.

    Returns the *child's* header — the lineage supplies events, never identity.
    A materialised log is what every reader already expects: dense from `seq == 0`
    and contiguous, so `_readmit`, the surface fold, the daemon's cursors and
    both derivations are untouched by any of this.

    **The slices tile exactly**: each ancestor contributes only what the
    generation below it still lacks, so the assembled log contains every event
    once. What is *kept* is 1.00x the result.

    What is **read** is not, and an earlier version of this docstring claimed
    otherwise. `read_one` is a backend's whole-log read, so taking 50 events from
    a 10 000-event ancestor still parses and validates all 10 000 — measured at
    ~170x the useful work. That is invisible today, because nothing writes a
    reference-fork yet; it becomes real the moment `fork` stops copying, and the
    fix is a bounded read (`WHERE seq < ?` in Turso, an early break in JSONL)
    widening `ReadOne` to take the boundary. Recorded here rather than in a plan,
    because this is the docstring that was wrong.

    Walks iteratively rather than recursively: the chain is data from disk, and a
    hand-edited header naming a cycle should meet a bound rather than the
    interpreter's stack.
    """
    header, events = read_one(session_id)
    # A file that starts at 0 says it holds its own history. Cross-check that
    # against the header before believing it: `Session.append` mints
    # `seq = len(self._log)`, so a reference-forked child built with an empty
    # seed writes its *first* event at 0 and would be read as a whole session
    # with its inherited prefix silently dropped. That is the trap step 4 walks
    # into the moment `fork` stops copying — the child's counter has to start at
    # `seed_length` — and a wrong log read as a right one is the exact failure
    # this module exists to make loud.
    inherited = header.seed_length or 0
    if events and events[0].seq == 0 and inherited > len(events):
        raise LineageError(
            f"session {session_id!r} starts at seq 0, so it claims to hold its own "
            f"history, but its header inherits {inherited} event(s) from "
            f"{header.parent_session!r} and the file holds only {len(events)}",
            code="TRUNCATED",
            session_id=session_id,
        )
    if not events or events[0].seq == 0:
        return header, events

    # Collect ancestors newest-first, taking from each only what the generation
    # below it still lacks. `owed` is always the next file's first seq, which is
    # both the count to take and the boundary the fork was made at.
    chain = [session_id]
    pieces = [events]
    owed = events[0].seq
    parent = header.parent_session
    while owed > 0:
        if parent is None:
            raise LineageError(
                f"session {session_id!r} begins at seq {events[0].seq} and names no parent; "
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
            ancestor_header, ancestor_events = read_one(parent)
        except Exception as error:
            raise LineageError(
                f"session {session_id!r} inherits {owed} event(s) from {parent!r}, "
                f"which could not be read: {error}",
                code="MISSING_ANCESTOR",
                session_id=parent,
            ) from error
        chain.append(parent)
        # Only the part below `owed`; an ancestor that kept working past the
        # boundary contributes nothing above it. The prefix is immutable, so
        # this is stable however far the ancestor has since grown.
        taken = [event for event in ancestor_events if event.seq < owed]
        pieces.append(taken)
        if taken:
            owed = taken[0].seq
        # An ancestor that supplied **nothing** does not end the search: it is
        # itself a reference-fork at or above this boundary, so what is owed is
        # unchanged and the prefix lies further up. Zeroing `owed` here instead
        # stopped the walk at the first such link, which turned every cycle and
        # every over-deep chain into a "gap" report — the guards below were
        # unreachable, and the tests for them are what found it.
        parent = ancestor_header.parent_session

    assembled = [event for piece in reversed(pieces) for event in piece]
    _assert_contiguous(assembled, session_id, chain)
    return header, assembled


def _assert_contiguous(events: list[SessionEvent], session_id: str, chain: list[str]) -> None:
    """The assembled log must be dense from 0, or say exactly where it is not.

    `_readmit` would refuse a gap anyway, one layer up and with no idea which
    ancestor was short. Checking here is what turns "seed must be contiguous from
    0" into a sentence naming the lineage that failed to supply the events.
    """
    for index, event in enumerate(events):
        if event.seq != index:
            raise LineageError(
                f"session {session_id!r} assembled a log with a gap at index {index} "
                f"(found seq {event.seq}); its lineage is {' -> '.join(reversed(chain))}",
                code="GAP",
                session_id=session_id,
            )
