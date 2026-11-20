"""Read the session fixtures of prime-agent, and shape them for comparison.

P3-23 asks what a prime-agent trajectory looks like *under the surface of pH*.
Re-running a trajectory cannot answer that, because a test has no model, no key
and no network. The question is therefore answered structurally: read the
recorded trajectory, reduce it to the facts that the port could change (turn
counts, tool-call shapes), and state which differences are expected and which
are not.

The fixtures are in `sources/`, a vendored reference checkout that is
deliberately **not part of this repo**. It is the code of another project.
Copying 1.5 MB of it here to make a test run would be a redistribution decision
that this repository has no reason to take.

**The reduction is therefore checked in, and the corpus is not.** `shapes.json`
holds what these tests actually assert on: the tallies of record types, roles
and tool names. It is 780 bytes of our own derivation, and not the source of
anyone else. The tests always read it, so they run on a clean clone and in CI.
Where the vendored checkout *does* exist, `test_fixture_replay` also derives the
reduction again from the raw JSONL and requires the two to agree. The committed
reduction therefore cannot drift from the corpus that it claims to summarize.

That split is the purpose. Before it, every test in `test_fixture_replay`
skipped on every runner, because `sources/` has no tracked files. The claim of
P3-23 that "nothing in a recorded trajectory is unrepresentable here" was
guarded only on a machine that happened to have the checkout. A guard that reads
as enforced and is not enforced is worse than no guard.

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
"""The committed reduction of those fixtures. The tests assert on it.

Regenerate it with `python -m fixture_replay` from this directory. That needs
the vendored checkout. The test suite then checks that the result still
matches."""


MISSING = "?"
"""What an absent type, role or tool name is counted as. There is one sentinel,
so a malformed record is visible in the tally and does not appear as three
different blanks."""


@dataclass(slots=True)
class TrajectoryShape:
    """The contents of one recorded trajectory, at the level of detail that the
    port changes."""

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

    The reader tolerates records that it does not know. The fixtures are in the
    format of another project, at a version that this port does not pin. An
    unreadable line is therefore skipped and is not fatal, which is the same
    rule that the harness fold uses.
    """
    shape = TrajectoryShape(name=path.name)
    # The file is streamed, not read with `read_text().splitlines()`. The larger
    # fixture is 2.3 MB, and the eager form holds the decoded string *and* the
    # line list at the same time: a peak of 16.7 MB against 0.6 MB, for no gain
    # in speed.
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
    """The reduction as the committed JSON, sorted so that a regeneration gives a
    clean diff.

    `sort_keys` and a trailing newline are used because this file is reviewed as
    a diff. Without them, a tally whose keys moved would look like a change in
    the corpus when it is a change in dictionary order.
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
    """The committed reduction, read back. It is available on a clean clone.

    Every assertion in `test_fixture_replay` runs against this, so those
    assertions run everywhere. Exactly one test reads the raw corpus: the test
    that checks that this file still describes it.
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
