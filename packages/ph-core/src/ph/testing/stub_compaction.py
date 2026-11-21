"""A compaction engine that compacts nothing and reports where it would compact.

Three suites had each written out this class: two inline in one file, and a
third in another package. Each copy gained `trigger_fraction` by hand when the
protocol gained it. A row that only *reads* the engine (the readouts of ph-clm,
the registration rule of the seam itself) needs the share and not a summary.

@module ph.testing.stub_compaction
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.types import AgentHandle
from ..seams.compaction import CompactionResult

__all__ = ["StubCompactionEngine"]


@dataclass(slots=True)
class StubCompactionEngine:
    """Reports `share` as its trigger and never compacts."""

    share: float | None = None
    """`None` is an engine that does not compact on pressure."""

    async def compact_now(
        self, agent: AgentHandle, *, instructions: str = ""
    ) -> CompactionResult | None:
        return None

    def trigger_fraction(self) -> float | None:
        return self.share
