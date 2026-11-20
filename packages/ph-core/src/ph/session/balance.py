"""Where the surface may be cut without separating a tool call from its result.

A `replace` may take out any set of current surface nodes (`ph.session.surface`).
The fold does not ask whether what is left still pairs every `tool-call` with
its `tool-result`. A provider does ask. A result whose call is gone, or a call
whose result is gone, is an orphan, and several providers reject it outright.
So every producer that shadows a range (the summary of compaction, a context
edit by a model) must cut where no call is outstanding. This module is the one
statement of where that is.

The rule is folded over the surface in *current* order and from content (the
`tool-pairing` of dsh), never from step boundaries. A landed replacement moves
positions, so a rule in terms of steps would be right only until the first
replacement.

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
    """How one surface node changes the count of calls that still await a result.

    The count is taken over the output of `derive_event_message`, THE
    projection, and not over the payload. It therefore cannot disagree with
    what the model was sent about how many calls a message made.
    """
    if message is None:
        return 0
    opened = sum(1 for block in message.content if isinstance(block, ToolCallBlock))
    closed = sum(1 for block in message.content if isinstance(block, ToolResultBlock))
    return opened - closed


def cuts_over(projected: Sequence[Message | None]) -> tuple[bool, ...]:
    """Whether each cut in an already-projected surface is tool-pairing balanced.

    A surface of *n* nodes has *n + 1* cuts. Entry `i` is the cut before node
    `i`, so `balanced[i]` answers "may the first `i` nodes be replaced on their
    own". Cut `0` is trivially balanced. Cut `n` is balanced exactly when the
    conversation has no call outstanding.

    The function takes the projection and not the session, because a caller
    that already has a projection should not pay for a second one.
    `derive_event_message` is a pydantic validation per node, and the planner
    of compaction derived the whole surface and then derived it again inside
    the balance fold.
    """
    return cuts_of(open_call_delta(message) for message in projected)


def cuts_of(deltas: Iterable[int]) -> tuple[bool, ...]:
    """`cuts_over`, from the `open_call_delta` of each node. It is for a caller that
    keeps the deltas and not the messages, as a memoizing caller does."""
    cuts = [True]
    open_calls = 0
    for delta in deltas:
        open_calls += delta
        cuts.append(open_calls == 0)
    return tuple(cuts)


def balanced_cuts(session: Session) -> tuple[bool, ...]:
    """`cuts_over`, folded across the current surface of a session."""
    events = session.events
    return cuts_over([derive_event_message(events[seq]) for seq in session.surface.nodes])


def safe_cutoff(projected: Sequence[Message | None], target: int) -> int:
    """The greatest balanced cut at or before `target`. `0` when there is none.

    The search goes *backward*, so a pair that straddles the retention boundary
    is kept whole on the retained side. The model keeps a call whose result it
    can still see, and the summary is one exchange shorter. A forward search
    would summarize the call and give the model an orphaned result, which
    several providers reject outright.

    `0` means "no safe range" and not "cut nothing". The caller reports it and
    does not compact an empty prefix.
    """
    cuts = cuts_over(projected)
    for index in range(min(target, len(cuts) - 1), -1, -1):
        if cuts[index]:
            return index
    return 0
