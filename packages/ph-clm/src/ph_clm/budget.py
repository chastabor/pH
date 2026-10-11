"""Context readouts: a line on a tool result when the context crosses a share of its
window, so the model knows to edit before the provider's limit, or compaction, decides
for it.

**On the result, not beside it.** A readout rides on the tool result whose output
crossed the threshold, appended in `tools/post-execute` as the context file's receipt
is (pi-clm's `sizeTrailer` does the same). Everything the model is shown is logged
(I3), and a notice message per threshold would be one more surface node to edit away.

**Measured as compaction measures** (`TokenMeter.baseline`): the provider's count of
the last request — system prompt and tool schemas included — moved by what the
surface did since, plus the result about to be logged, against the window that request
was routed with. A receipt's context figures are the same number. No window, no
readouts: there is no share to report.

**The last reminder comes a margin below compaction's trigger**, read from the engine
(`CompactionSeam.trigger_fraction`), so the model always has room to act before a
summary is written for it. A fixed last share did not: on a 16k slot, 75% left one
file read before compaction's 85%, and a live run spent it on a malformed call and
lost the sections it meant to edit to the summary (`Shares.of`).

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

__all__ = ["Readouts", "Shares"]


@dataclass(frozen=True, slots=True)
class Shares:
    """The thresholds one reading is judged against, ascending, as shares of its
    window — and the share at which compaction summarizes on its own, when an engine
    says."""

    thresholds: tuple[float, ...]
    window: int
    compacts_at: float | None

    @classmethod
    def of(
        cls, remind_at: Sequence[float], margin: float, compacts_at: float | None, window: int
    ) -> Shares:
        """`remind_at`, with the last reminder moved to `margin` below compaction's
        trigger: any configured share at or past that point gives way to it. Unchanged
        with no engine, or a margin that would put it at or below zero; empty stays
        empty, which is readouts off."""
        shares = sorted(remind_at)
        # Rounded, so 0.85 - 0.2 is the 0.65 it means rather than 0.6499999999999999.
        last = None if compacts_at is None else round(compacts_at - margin, 6)
        if shares and last is not None and last > 0:
            shares = [*(share for share in shares if share < last), last]
        return cls(tuple(shares), window, compacts_at)

    def reached(self, tokens: int) -> int:
        """How many thresholds `tokens` is at or past."""
        return sum(1 for share in self.thresholds if tokens >= share * self.window)

    def line(self, tokens: int, how: str) -> str:
        """What a result carries when the context reached `tokens`: its size, what to
        do, and when the next reminder comes — or that this is the last, and what
        happens after it."""
        level = self.reached(tokens)
        share = tokens * 100 // self.window
        head = f"[context: ~{thousands(tokens)} of {thousands(self.window)} tokens ({share}%)."
        if level >= len(self.thresholds):
            after = (
                ""
                if self.compacts_at is None
                else f" At {round(self.compacts_at * 100)}% older context is summarized for you."
            )
            return f"{head} This is the last reminder: free space now — {how}.{after}]"
        following = round(self.thresholds[level] * 100)
        return (
            f"{head} Free space when it suits you — {how}. "
            f"The next reminder comes at {following}%.]"
        )


@dataclass(slots=True)
class Readouts:
    """Which thresholds each session's context has reached, and when that rises."""

    _reached: dict[str, int] = field(default_factory=dict)

    def crossed(
        self, session_id: str, shares: Shares, before: int, most: int, measure: Callable[[], int]
    ) -> int | None:
        """When a result takes the context from `before` tokens to at most `most`: the
        tokens after, if that reached a threshold this session had not; `None`
        otherwise.

        `measure` counts the tokens after exactly, and is asked only when the answer
        needs it — when `most` could reach a threshold `before` had not, or when a
        readout fires. Tokenizing a long result is the costly part, and most results
        cross nothing.
        """
        floor = shares.reached(before)
        known = self._reached.get(session_id, floor)
        after = measure() if shares.reached(most) > floor else None
        level = floor if after is None else shares.reached(after)
        self._reached[session_id] = level
        if level <= known:
            return None
        return measure() if after is None else after

    def forget(self, session_id: str) -> None:
        self._reached.pop(session_id, None)
