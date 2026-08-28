"""Reading prime-agent's own session fixtures, and shaping them for comparison.

P3-23 asks what a prime-agent trajectory looks like *under pH's surface*. It
cannot be answered by re-running one — there is no model, no key and no network
in a test — so it is answered structurally: read the recorded trajectory, reduce
it to the facts the port could change (turn counts, tool-call shapes), and state
which differences are expected and which are not.

The fixtures live in `sources/`, a vendored reference checkout that is
deliberately **not part of this repo**. Everything here therefore degrades to
"absent" rather than failing, and the durable artifact is the checked-in report
(`docs/dev-notes/prime-agent-replay.md`) rather than the fixtures.

@module fixture_replay
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["FIXTURE_DIR", "TrajectoryShape", "available_fixtures", "read_shape"]

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "sources/prime-agent/packages/coding-agent/test/fixtures"
)
"""Where the vendored checkout puts them. Absent in a clean clone."""

FIXTURES = ("before-compaction.jsonl", "large-session.jsonl")


MISSING = "?"
"""What an absent type, role or tool name counts as. One sentinel, so a
malformed record is visible in the tally rather than three different blanks."""


@dataclass(slots=True)
class TrajectoryShape:
    """What one recorded trajectory is made of, at the grain the port changes."""

    name: str
    record_types: Counter[str] = field(default_factory=Counter)
    roles: Counter[str] = field(default_factory=Counter)
    tool_calls: Counter[str] = field(default_factory=Counter)

    @property
    def records(self) -> int:
        return sum(self.record_types.values())

    @property
    def total_tool_calls(self) -> int:
        return sum(self.tool_calls.values())


def available_fixtures() -> list[Path]:
    """The fixtures present on this machine, in a stable order."""
    return [path for name in FIXTURES if (path := FIXTURE_DIR / name).is_file()]


def read_shape(path: Path) -> TrajectoryShape:
    """Reduce one prime-agent JSONL session to its shape.

    Tolerant of records this reader does not know: the fixtures are another
    project's format at a version this port does not pin, so an unreadable line
    is skipped rather than fatal — the same rule the harness fold uses.
    """
    shape = TrajectoryShape(name=path.name)
    # Streamed, not `read_text().splitlines()`: the larger fixture is 2.3 MB and
    # the eager form holds the decoded string *and* the line list at once —
    # 16.7 MB peak against 0.6 MB, for no gain in speed.
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(record, dict):
                continue
            shape.record_types[str(record.get("type") or MISSING)] += 1

            message = record.get("message")
            if not isinstance(message, dict):
                continue
            shape.roles[str(message.get("role") or MISSING)] += 1
            for block in message.get("content") or []:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    shape.tool_calls[str(block.get("name") or MISSING)] += 1
    return shape
