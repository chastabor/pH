"""Cooperative cancellation, fused across nested owners.

dsh threads an `AbortSignal` through the tool pipeline: the caller owns one, a
`tools/execute` wrapper may *replace* it for its delegated lifetime (that is how
a timeout policy works), and the registry fuses every replacement with the
captured caller signal so a wrapper can narrow the lifetime but never widen it.

`CancelToken` is that contract. A child is canceled when it is canceled *or
when any ancestor is*, which makes the fusion structural rather than something
each wrapper has to remember.

Why not a bare `anyio.CancelScope`: a scope cancels the task that awaits inside
it, and the pipeline needs to *ask* whether cancellation happened at points
where no await is pending — to decide between "aborted before dispatch" (the
call had no effect) and "aborted" (the body ran). A token answers that question;
a scope only acts on it.

@module ph.cancel
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

__all__ = ["CancelToken", "Canceled", "Cancellation", "is_canceled"]


class Canceled(Exception):
    """Raised by `raise_if_canceled()`; carries the reason for the record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class Cancellation(Protocol):
    """What a *reader* needs of a cancellation: whether, and why.

    `CancelToken` satisfies it. Declared because most holders only ask — and
    `AgentDriver`'s docstring makes the argument about the verb it withholds:
    "a seam that could reach `cancel` through a parameter typed for reading is a
    seam that will, eventually." `cancel()` and `child()` belong to whoever owns
    the lifetime; `AgentHandle.signal` hands out this instead, so a row that
    prompts can ask whether the work is still wanted and cannot end it.
    """

    @property
    def canceled(self) -> bool: ...
    @property
    def cancel_reason(self) -> str | None: ...
    def raise_if_canceled(self) -> None: ...


@dataclass(slots=True)
class CancelToken:
    """A cooperative cancellation flag that inherits from its parent."""

    reason: str | None = None
    parent: CancelToken | None = None

    @property
    def canceled(self) -> bool:
        node: CancelToken | None = self
        while node is not None:
            if node.reason is not None:
                return True
            node = node.parent
        return False

    @property
    def cancel_reason(self) -> str | None:
        node: CancelToken | None = self
        while node is not None:
            if node.reason is not None:
                return node.reason
            node = node.parent
        return None

    def cancel(self, reason: str = "canceled") -> None:
        if self.reason is None:
            self.reason = reason

    def child(self, reason: str | None = None) -> CancelToken:
        """A narrower token: canceled by itself or by anything above it."""
        return CancelToken(reason=reason, parent=self)

    def raise_if_canceled(self) -> None:
        reason = self.cancel_reason
        if reason is not None:
            raise Canceled(reason)


def is_canceled(token: Cancellation | None) -> bool:
    """`False` for no token — the one place that rule is spelled out."""
    return token is not None and token.canceled
