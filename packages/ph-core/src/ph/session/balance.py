"""Where the surface may be cut without separating a tool call from its result.

A `replace` may take out any set of current surface nodes (`ph.session.surface`), and
the fold does not ask whether what is left still pairs every `tool-call` with its
`tool-result`. A provider does: a result whose call is gone, or a call whose result
is, is an orphan several of them reject outright. So every producer that shadows a
range — compaction's summary, a model's own context edit — has to cut where no call
is outstanding, and this module is the one statement of where that is.

Folded over the surface in *current* order and from content (dsh's `tool-pairing`),
never from step boundaries: a landed replacement moves positions, so a rule written
in terms of steps would be right only until the first one.

@module ph.session.balance
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from ..llm.types import Message, ToolCallBlock, ToolResultBlock
from .derive import derive_event_message

if TYPE_CHECKING:
    from .session import Session

__all__ = ["balanced_cuts", "cuts_of", "cuts_over", "open_call_delta", "safe_cutoff"]


def open_call_delta(message: Message | None) -> int:
    """How one surface node changes the count of calls still awaiting a result.

    Counted over `derive_event_message`'s output — THE projection — rather than
    over the payload, so this cannot disagree with what the model was sent about
    how many calls a message made.
    """
    if message is None:
        return 0
    opened = sum(1 for block in message.content if isinstance(block, ToolCallBlock))
    closed = sum(1 for block in message.content if isinstance(block, ToolResultBlock))
    return opened - closed


def cuts_over(projected: Sequence[Message | None]) -> tuple[bool, ...]:
    """Whether each cut in an already-projected surface is tool-pairing balanced.

    A surface of *n* nodes has *n + 1* cuts; entry `i` is the cut before node
    `i`, so `balanced[i]` answers "may the first `i` nodes be replaced on their
    own". Cut `0` is trivially balanced and cut `n` is balanced exactly when the
    conversation has no call outstanding.

    Takes the projection rather than the session because a caller that has one
    already should not pay for a second: `derive_event_message` is a pydantic
    validation per node, and compaction's planner was deriving the whole surface,
    then deriving it again inside the balance fold.
    """
    return cuts_of(open_call_delta(message) for message in projected)


def cuts_of(deltas: Iterable[int]) -> tuple[bool, ...]:
    """`cuts_over`, from each node's `open_call_delta` — for a caller that keeps the
    deltas rather than the messages, as a memoizing one does."""
    cuts = [True]
    open_calls = 0
    for delta in deltas:
        open_calls += delta
        cuts.append(open_calls == 0)
    return tuple(cuts)


def balanced_cuts(session: Session) -> tuple[bool, ...]:
    """`cuts_over`, folded across a session's current surface."""
    events = session.events
    return cuts_over([derive_event_message(events[seq]) for seq in session.surface.nodes])


def safe_cutoff(projected: Sequence[Message | None], target: int) -> int:
    """The greatest balanced cut at or before `target`; `0` when there is none.

    *Backward*, so a pair that straddles the retention boundary is kept whole on
    the retained side — the model keeps a call it can still see the result of,
    and the summary is one exchange shorter. Advancing forward instead would
    summarize the call and hand the model an orphaned result, which several
    providers reject outright.

    `0` means "no safe range", not "cut nothing": the caller reports it rather
    than compacting an empty prefix.
    """
    cuts = cuts_over(projected)
    for index in range(min(target, len(cuts) - 1), -1, -1):
        if cuts[index]:
            return index
    return 0
