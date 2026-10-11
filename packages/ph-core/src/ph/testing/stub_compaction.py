"""A compaction engine that compacts nothing and says where it would.

Three suites had each written this class out — two inline in one file, a third one
package over — and every copy grew `trigger_fraction` by hand when the protocol
did. A row that only *reads* the engine (ph-clm's readouts, the seam's own
registration rule) needs the share, not a summary.

@module ph.testing.stub_compaction
"""

from __future__ import annotations

from dataclasses import dataclass

from ..agent.types import AgentHandle
from ..seams.compaction import CompactionResult, CompactionTrigger

__all__ = ["StubCompactionEngine"]


@dataclass(slots=True)
class StubCompactionEngine:
    """Reports `share` as its trigger and never compacts."""

    share: float | None = None
    """`None` is an engine that does not compact on pressure."""

    async def compact_if_needed(
        self, agent: AgentHandle, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        return None

    async def compact_now(
        self, agent: AgentHandle, *, instructions: str = ""
    ) -> CompactionResult | None:
        return None

    def trigger_fraction(self) -> float | None:
        return self.share
