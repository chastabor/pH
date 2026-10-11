"""Context readouts: a line on a tool result when the context crosses a share of its
window, so the model knows to edit before the provider's limit, or compaction, decides
for it.

**On the result, not beside it.** A readout rides on the tool result whose output
crossed the threshold, appended in `tools/post-execute` as the context file's receipt
is (pi-clm's `sizeTrailer` does the same). Everything the model is shown is logged
(I3), and a notice message per threshold would be one more surface node to edit away.

**Measured as the receipts measure:** the section map's tokens — the meter's estimate
of every surface node — plus the result about to be logged, against the window the
last request was routed with. No window, no readouts: there is no share to report.

**Once per crossing, and again after an edit brings the context back down.** Per
session, `Readouts` keeps how many thresholds the context had reached; a readout fires
when a result raises that, and each reading sets it, so an edit that frees space lowers
it and the next crossing is reported again. In memory: after a restart the first
reading sets it without a readout, so a resumed session is not told again what it was
told before the restart.

@module ph_clm.budget
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ph.text import thousands

__all__ = ["Readouts"]


@dataclass(slots=True)
class Readouts:
    """Which thresholds each session's context has reached, and when that rises."""

    thresholds: tuple[float, ...]
    """Shares of the window, ascending — sorted here, since `line` counts on it."""
    _reached: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.thresholds = tuple(sorted(self.thresholds))

    def crossed(
        self, session_id: str, before: int, most: int, window: int, measure: Callable[[], int]
    ) -> tuple[int, int] | None:
        """When a result takes the context from `before` tokens to at most `most`: the
        level reached and the tokens after, if this session had not reached that level;
        `None` otherwise.

        `measure` counts the tokens after exactly, and is asked only when the answer
        needs it — when `most` could reach a threshold `before` had not, or when a
        readout fires. Tokenizing a long result is the costly part, and most results
        cross nothing.
        """
        floor = _reached(before, window, self.thresholds)
        known = self._reached.get(session_id, floor)
        after = measure() if _reached(most, window, self.thresholds) > floor else None
        level = floor if after is None else _reached(after, window, self.thresholds)
        self._reached[session_id] = level
        if level <= known:
            return None
        return level, measure() if after is None else after

    def line(self, tokens: int, window: int, level: int, how: str) -> str:
        """What a result carries when the context reached `level` thresholds: its size,
        what to do, and when the next reminder comes — or that this is the last."""
        share = tokens * 100 // window
        head = f"[context: ~{thousands(tokens)} of {thousands(window)} tokens ({share}%)."
        if level >= len(self.thresholds):
            return f"{head} This is the last reminder: free space now — {how}.]"
        following = round(self.thresholds[level] * 100)
        return (
            f"{head} Free space when it suits you — {how}. "
            f"The next reminder comes at {following}%.]"
        )

    def forget(self, session_id: str) -> None:
        self._reached.pop(session_id, None)


def _reached(tokens: int, window: int, thresholds: Sequence[float]) -> int:
    """How many of `thresholds`, as shares of `window`, `tokens` is at or past."""
    return sum(1 for share in thresholds if tokens >= share * window)
