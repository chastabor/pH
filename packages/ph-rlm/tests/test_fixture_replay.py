"""The recorded trajectories of prime-agent, compared with the surface of pH (P3-23).

The row asks for a *report* with its diffs triaged. The report is checked in at
`docs/dev-notes/prime-agent-replay.md`. What can be tested, and is tested here,
is the claim that the report depends on: **nothing in a recorded trajectory is
unrepresentable here**.

The claim is asserted in the way that `test_conformance.py` asserts the claim of
the protocol: the tests enumerate the vocabulary of the fixtures and require
every member to be accounted for. The first draft of this module instead
compared a set of diff names with a set of diff names, and *both were written in
the same function*. That comparison could not fail. A fifth diff could appear
only if someone edited the test, and that edit would have added the diff to the
expected set too. A coverage table over the record types and roles of the
fixtures is the version that breaks when a new fixture brings something new.

The fixtures are vendored under `sources/`, which is not part of this repo. The
tests therefore read the committed reduction in `shapes.json`. Only the one test
that reads the raw fixtures is skipped when they are absent.
"""

from __future__ import annotations

import pytest
from fixture_replay import (
    SHAPES_FILE,
    TrajectoryShape,
    available_fixtures,
    read_shape,
    recorded_shapes,
    to_wire,
)
from rlm_fixtures import ShippedProfile

from ph.keys import TOOLS
from ph_app.profiles import resolve_profile

pytestmark = pytest.mark.anyio

FIXTURE_PATHS = available_fixtures()
needs_fixtures = pytest.mark.skipif(
    not FIXTURE_PATHS, reason="sources/prime-agent is a vendored checkout, not part of this repo"
)
"""For the **one** test that needs the raw corpus: the test that asserts that the
committed reduction still describes it. Everything else reads `shapes.json` and
therefore runs on a clean clone."""


@pytest.fixture(scope="module")
def shapes() -> dict[str, TrajectoryShape]:
    """Both trajectories, from the committed reduction.

    The shapes are read from `shapes.json` and not derived again. That makes
    this module a guard, and not an aid that works only on the machine of a
    developer. `sources/` has no tracked files, so every assertion here used to
    skip on every runner, and the claim of P3-23 was enforced nowhere. 780 bytes
    of our own derivation restore the guard without redistributing 1.5 MB of the
    fixtures of another project.
    """
    return {shape.name: shape for shape in recorded_shapes()}


# ------------------------------------------------------- the vocabulary --

RECORD_TYPES: dict[str, str] = {
    "session": "session header — `Session.header` (id, cwd, provider, model)",
    "message": "`user/message`, `assistant/message`, `tool/call`, `tool/result`",
    "compaction": "a surface `replace` (I4): the summary shadows the range it stands for",
    "model_change": "`request/header` — the call config is snapshotted per request",
    "thinking_level_change": "`request/header` — `reasoning_effort` travels in the same config",
}
"""Every record type that the fixtures carry, and where the same fact is kept in pH.

The values are prose, and the test checks only that an entry is *present*. The
mapping is the argument of the report, and this table is where the argument is
held to the fixtures. A type with no entry fails, which is the purpose. The
report claims "no record type in either fixture is unrepresentable", and that
claim needs a place where it can break."""

ROLES: dict[str, str] = {
    "user": "`user/message`",
    "assistant": "`assistant/message`",
    "toolResult": "`tool/result`",
    # The one role with no counterpart, and the fourth diff of the report.
    "bashExecution": "",
}

UNREPRESENTED_ROLES: frozenset[str] = frozenset({"bashExecution"})
"""The roles that pH has no place for.

`bashExecution` is prime-agent logging the interactive shell of a *user* into
the session, beside the turns of the model. `ctx.commands` is the mechanism that
would carry it, because it dispatches without a model turn. Nothing that ships
uses it to run a shell command. A user who runs `ls` today therefore does it
outside the session, and the log does not know. This is a missing feature and
not a porting error."""


def test_every_record_type_has_a_home_here(shapes: dict[str, TrajectoryShape]) -> None:
    """The central claim of the report, checked against the fixtures that support it."""
    seen = {kind for shape in shapes.values() for kind in shape.record_types}
    assert seen, "the fixtures parsed to nothing"
    unmapped = seen - set(RECORD_TYPES)
    assert unmapped == set(), f"record types with no pH counterpart recorded: {sorted(unmapped)}"
    # The table also describes these fixtures, and not an older set of them.
    assert set(RECORD_TYPES) - seen == set(), "the table maps types no fixture carries"


def test_every_role_is_mapped_or_named_as_a_gap(shapes: dict[str, TrajectoryShape]) -> None:
    """The one real capability gap, isolated and not only described.

    A role with an empty mapping is a gap. A role with no entry at all is a
    difference that nobody examined, and this module exists to prevent that.
    """
    seen = {role for shape in shapes.values() for role in shape.roles}
    assert seen - set(ROLES) == set(), f"roles with no entry: {sorted(seen - set(ROLES))}"
    gaps = {role for role, home in ROLES.items() if not home}
    assert gaps == UNREPRESENTED_ROLES


# --------------------------------------------------------- the surface --


def test_the_fixtures_are_the_coding_agent_not_the_rlm(shapes: dict[str, TrajectoryShape]) -> None:
    """The first finding, which frames the rest.

    Both vendored trajectories come from the *coding* agent of prime-agent:
    `bash`, `edit`, `read` and `write` as native tool calls. Neither uses
    `ipython`, and neither spawns a child. The fixtures therefore exercise the
    surface that C1-C3 replaced, and not the RLM loop that this bundle ports.
    The `access` default, which the row predicted as a diff, cannot be observed
    from them. This is asserted and not assumed, so that a fixture that is added
    later and *does* delegate fails here, and the report is rewritten.
    """
    for shape in shapes.values():
        assert shape.total_tool_calls > 0
        # The name that prime-agent gives its RLM tool, which pH presents as `python`.
        assert "ipython" not in shape.tool_calls, "a fixture used the RLM surface after all"
        assert not any(name.startswith("rlm") for name in shape.tool_calls)


async def test_every_tool_the_fixtures_called_exists_under_the_rlm_profile(
    shipped_profile: ShippedProfile, shapes: dict[str, TrajectoryShape]
) -> None:
    """The claim that makes the trajectory expressible at all.

    Each tool name of prime-agent is a registered tool in the shipped profile,
    so the same call is available. It is available as a binding inside a cell
    and not as a native call, which is the translation that the report
    describes.
    """
    ctx, _session, agent = await shipped_profile(profile=resolve_profile("rlm"))
    visible = set(ctx.require(TOOLS).view(agent.ctx).visible)

    called = {name for shape in shapes.values() for name in shape.tool_calls}
    assert called, "the fixtures recorded no tool calls"
    missing = called - visible
    assert missing == set(), f"prime-agent called tools pH does not offer: {sorted(missing)}"


def test_the_fixtures_carry_the_compaction_case(shapes: dict[str, TrajectoryShape]) -> None:
    """One fixture is named for compaction, and compaction is the surface
    operation that the port models differently (`surface_op: replace`, I4). It
    is therefore worth knowing that the reference trajectory actually contains
    one."""
    before = shapes.get("before-compaction.jsonl")
    if before is None:
        pytest.skip("only some fixtures are vendored here")
    assert before.record_types.get("compaction", 0) >= 1


# ------------------------------------------- the corpus, where it exists --


@needs_fixtures
def test_the_committed_reduction_still_describes_the_corpus() -> None:
    """The one test that reads the raw JSONL. The other tests are valid only if
    this one passes.

    Everything above asserts against `shapes.json`, so everything above is worth
    having only if that file still says what the fixtures say. The reduction is
    derived again here and compared **byte for byte with the output of the
    serializer**, not field by field. A hand-written comparison would be a
    second encoding of the same reduction, and it would drift from `to_wire`
    the first time that either changed.

    The test is skipped on a clean clone, and this split makes that trade on
    purpose. The guard on the surface of pH runs everywhere, and the guard on
    the *input to* that guard runs wherever the input exists. Regenerate with
    `python -m fixture_replay` from this directory when a fixture changes.
    """
    derived = to_wire([read_shape(path) for path in FIXTURE_PATHS])
    committed = SHAPES_FILE.read_text(encoding="utf-8")
    assert derived == committed, (
        f"{SHAPES_FILE.name} no longer matches the fixtures under "
        f"{FIXTURE_PATHS[0].parent} — regenerate with `python -m fixture_replay`"
    )


def test_the_reduction_covers_every_fixture_the_reader_knows() -> None:
    """The committed file names every fixture, so a fixture cannot be left
    unreduced.

    Without this test, dropping an entry from `shapes.json` would quietly narrow
    every assertion above to the entries that remained. The corpus test would
    still pass on a machine that had also lost the fixture. This test runs
    everywhere, because it is a claim about the committed file and not about the
    corpus.
    """
    from fixture_replay import FIXTURES

    assert [shape.name for shape in recorded_shapes()] == list(FIXTURES)
