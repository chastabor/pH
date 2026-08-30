"""Reading prime-agent's own session fixtures, and shaping them for comparison.

P3-23 asks what a prime-agent trajectory looks like *under pH's surface*. It
cannot be answered by re-running one — there is no model, no key and no network
in a test — so it is answered structurally: read the recorded trajectory, reduce
it to the facts the port could change (turn counts, tool-call shapes), and state
which differences are expected and which are not.

The fixtures live in `sources/`, a vendored reference checkout that is
deliberately **not part of this repo** — it is another project's code, and
copying 1.5 MB of it here to make a test run is a redistribution decision this
repository has no reason to take.

**So the reduction is checked in and the corpus is not.** `shapes.json` holds
what these tests actually assert on — the record-type, role and tool-name
tallies — in 779 bytes of our own derivation rather than anyone else's source.
The tests read it always, which is what makes them run on a clean clone and in
CI; where the vendored checkout *does* exist, `test_fixture_replay` additionally
re-derives from the raw JSONL and requires the two to agree, so the committed
reduction cannot drift from the corpus it claims to summarise.

That split is the whole point. Before it, every test in this module skipped on
every runner — `sources/` has no tracked files — so P3-23's claim that "nothing
in a recorded trajectory is unrepresentable here" was guarded only on a machine
that happened to have the checkout. A guard that reads as enforced and is not is
worse than none.

@module fixture_replay
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "FIXTURE_DIR",
    "SHAPES_FILE",
    "TrajectoryShape",
    "available_fixtures",
    "read_shape",
    "recorded_shapes",
    "to_wire",
]

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "sources/prime-agent/packages/coding-agent/test/fixtures"
)
"""Where the vendored checkout puts them. Absent in a clean clone."""

FIXTURES = ("before-compaction.jsonl", "large-session.jsonl")

SHAPES_FILE = Path(__file__).with_name("shapes.json")
"""The committed reduction of those fixtures — what the tests assert on.

Regenerate with `python -m fixture_replay` from this directory, which requires
the vendored checkout; the test suite then checks the result still matches."""


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


def to_wire(shapes: list[TrajectoryShape]) -> str:
    """The reduction as the committed JSON, sorted so a regeneration diffs cleanly.

    `sort_keys` and a trailing newline because this file is reviewed as a diff:
    a tally whose keys moved would otherwise look like a change in the corpus
    when it is a change in dictionary ordering.
    """
    return (
        json.dumps(
            [
                {
                    "name": shape.name,
                    "record_types": dict(shape.record_types),
                    "roles": dict(shape.roles),
                    "tool_calls": dict(shape.tool_calls),
                }
                for shape in shapes
            ],
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def recorded_shapes() -> list[TrajectoryShape]:
    """The committed reduction, read back. Available on a clean clone.

    This is what every assertion in `test_fixture_replay` runs against, so those
    assertions run everywhere. The raw corpus is consulted by exactly one test —
    the one that checks this file still describes it.
    """
    payload = json.loads(SHAPES_FILE.read_text(encoding="utf-8"))
    return [
        TrajectoryShape(
            name=str(entry["name"]),
            record_types=Counter(entry["record_types"]),
            roles=Counter(entry["roles"]),
            tool_calls=Counter(entry["tool_calls"]),
        )
        for entry in payload
    ]


if __name__ == "__main__":  # pragma: no cover - a maintenance command
    found = available_fixtures()
    if not found:
        raise SystemExit(f"no fixtures under {FIXTURE_DIR}; the vendored checkout is absent")
    SHAPES_FILE.write_text(to_wire([read_shape(path) for path in found]), encoding="utf-8")
    print(f"wrote {SHAPES_FILE} from {len(found)} fixtures")
