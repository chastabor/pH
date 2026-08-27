"""Cooperative cancellation, fused across nested owners.

dsh threads an `AbortSignal` through the tool pipeline: the caller owns one, a
`tools/execute` wrapper may *replace* it for its delegated lifetime (that is how
a timeout policy works), and the registry fuses every replacement with the
captured caller signal so a wrapper can narrow the lifetime but never widen it.

`CancelToken` is that contract. A child is cancelled when it is cancelled *or
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

__all__ = ["CancelToken", "Cancelled", "is_cancelled"]


class Cancelled(Exception):
    """Raised by `raise_if_cancelled()`; carries the reason for the record."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(slots=True)
class CancelToken:
    """A cooperative cancellation flag that inherits from its parent."""

    reason: str | None = None
    parent: CancelToken | None = None

    @property
    def cancelled(self) -> bool:
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

    def cancel(self, reason: str = "cancelled") -> None:
        if self.reason is None:
            self.reason = reason

    def child(self, reason: str | None = None) -> CancelToken:
        """A narrower token: cancelled by itself or by anything above it."""
        return CancelToken(reason=reason, parent=self)

    def raise_if_cancelled(self) -> None:
        reason = self.cancel_reason
        if reason is not None:
            raise Cancelled(reason)


def is_cancelled(token: CancelToken | None) -> bool:
    """`False` for no token — the one place that rule is spelled out."""
    return token is not None and token.cancelled
