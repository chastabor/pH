"""Context readouts: one line on a tool result when the context passes a threshold.

A threshold is a share of the context window. The readout tells the model to edit
its context before the limit of the provider, or compaction, decides for it.

**The readout is on the result, not beside it.** A readout is appended to the
tool result whose output passed the threshold. This happens in
`tools/post-execute`, as it does for the receipt of the context file.
(`sizeTrailer` in pi-clm does the same.) Everything that the model is shown is
logged (I3). A separate notice message for each threshold would be one more
surface node for the model to edit away.

**The context is measured as compaction measures it** (`TokenMeter.baseline`).
The start is the token count that the provider gave for the last request, which
includes the system prompt and the tool schemas. That count is adjusted by what
the surface did since then, and the result that is about to be logged is added.
The total is compared with the window that the request was routed with. The
context figures in a receipt are the same number. With no window there are no
readouts, because there is no share to report.

**The last reminder comes a margin below the compaction trigger.** The trigger is
read from the engine (`CompactionSeam.trigger_fraction`). The model then always
has room to act before compaction writes a summary for it. A fixed last threshold
did not give that room. On a 16k slot, 75% left room for one file read before the
85% of compaction. A live run used that room on a malformed call, and the summary
took the sections that the run meant to edit (`Shares.of`).

**A readout fires one time for each crossing, and again after an edit brings the
context back down.** For each session, `Readouts` keeps the number of thresholds
that the context had reached. A readout fires when a result raises that number.
Each reading sets the number, so an edit that frees space lowers it, and the next
crossing is reported again. The number is kept in memory only. After a restart,
the first reading sets it without a readout. A resumed session is therefore not
told again what it was told before the restart.

@module ph_clm.budget
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ph.session import Session, SessionTable
from ph.text import thousands

__all__ = ["Readouts", "Shares"]


@dataclass(frozen=True, slots=True)
class Shares:
    """The thresholds that one reading is compared with, in ascending order, as
    shares of the window. Also the share at which compaction summarizes without a
    request, when an engine gives one."""

    thresholds: tuple[float, ...]
    window: int
    compacts_at: float | None

    @classmethod
    def of(
        cls, remind_at: Sequence[float], margin: float, compacts_at: float | None, window: int
    ) -> Shares:
        """Build the thresholds from `remind_at`, with the last reminder moved to
        `margin` below the compaction trigger. That last reminder replaces every
        configured share at or past that point. The shares are unchanged when
        there is no engine, or when the margin would put the last reminder at or
        below zero. An empty list stays empty, which means readouts are off."""
        shares = sorted(remind_at)
        # Round the value, so that 0.85 - 0.2 gives 0.65 and not 0.6499999999999999.
        last = None if compacts_at is None else round(compacts_at - margin, 6)
        if shares and last is not None and last > 0:
            shares = [*(share for share in shares if share < last), last]
        return cls(tuple(shares), window, compacts_at)

    def reached(self, tokens: int) -> int:
        """The number of thresholds that `tokens` is at or past."""
        return sum(1 for share in self.thresholds if tokens >= share * self.window)

    def line(self, tokens: int, how: str) -> str:
        """The line that a result carries when the context reached `tokens`. It
        gives the size of the context, what to do, and when the next reminder
        comes. For the last reminder, it says so and says what happens next."""
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
    """The number of thresholds that the context of each session has reached. It
    also detects when that number rises."""

    _reached: SessionTable[int] = field(default_factory=SessionTable)
    """Kept for each session, and removed with the session."""

    def crossed(
        self, session: Session, shares: Shares, before: int, most: int, measure: Callable[[], int]
    ) -> int | None:
        """Check a result that takes the context from `before` tokens to at most
        `most` tokens. Returns the token count after the result if the context
        reached a threshold that this session had not reached. Returns `None`
        otherwise.

        `measure` counts the tokens after the result exactly. It is called only
        when the answer needs it: when `most` could reach a threshold that
        `before` had not reached, or when a readout fires. Tokenizing a long
        result is the expensive part, and most results cross no threshold.
        """
        floor = shares.reached(before)
        known = self._reached.get(session, floor)
        after = measure() if shares.reached(most) > floor else None
        level = floor if after is None else shares.reached(after)
        self._reached.keep(session, level)
        if level <= known:
            return None
        return measure() if after is None else after
