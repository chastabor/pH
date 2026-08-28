"""Prime Agent's recorded trajectories, against pH's surface (P3-23).

The row asks for a *report* with its diffs triaged; the report is checked in at
`docs/dev-notes/prime-agent-replay.md`. What is testable — and what is here — is
the claim the report rests on: **nothing in a recorded trajectory is
unrepresentable here**.

That is asserted the way `test_conformance.py` asserts the protocol's: by
enumerating the fixtures' own vocabulary and requiring every member to be
accounted for. The first draft of this module instead compared a set of diff
names against a set of diff names *both written in the same function*, which
could not fail — a fifth diff could only appear by someone editing the test, and
that edit would have added it to the expected set too. A coverage table over the
fixtures' record types and roles is the version that breaks when a new fixture
brings something new.

The fixtures are vendored under `sources/`, which is not part of this repo, so
everything here skips when they are absent.
"""

from __future__ import annotations

from typing import Any

import pytest
from fixture_replay import TrajectoryShape, available_fixtures, read_shape

from ph_app.profiles import resolve_profile
from ph_rlm.presentation import IPYTHON

pytestmark = pytest.mark.anyio

FIXTURE_PATHS = available_fixtures()
needs_fixtures = pytest.mark.skipif(
    not FIXTURE_PATHS, reason="sources/prime-agent is a vendored checkout, not part of this repo"
)


@pytest.fixture(scope="module")
def shapes() -> dict[str, TrajectoryShape]:
    """Both trajectories, reduced once. 3.3 MB of JSONL is not worth re-parsing
    per test."""
    return {shape.name: shape for shape in map(read_shape, FIXTURE_PATHS)}


# ------------------------------------------------------- the vocabulary --

RECORD_TYPES: dict[str, str] = {
    "session": "session header — `Session.header` (id, cwd, provider, model)",
    "message": "`user/message`, `assistant/message`, `tool/call`, `tool/result`",
    "compaction": "a surface `replace` (I4): the summary shadows the range it stands for",
    "model_change": "`request/header` — the call config is snapshotted per request",
    "thinking_level_change": "`request/header` — `reasoning_effort` travels in the same config",
}
"""Every record type the fixtures carry, and where the same fact lives in pH.

Prose, checked for *presence*: the mapping is the report's argument and this is
where it is held to the fixtures. A type with no entry fails, which is the whole
point — the report claims "no record type in either fixture is unrepresentable",
and that claim needs somewhere to break."""

ROLES: dict[str, str] = {
    "user": "`user/message`",
    "assistant": "`assistant/message`",
    "toolResult": "`tool/result`",
    # The one with no counterpart, and the report's fourth diff.
    "bashExecution": "",
}

UNREPRESENTED_ROLES: frozenset[str] = frozenset({"bashExecution"})
"""Roles pH has no home for.

`bashExecution` is prime-agent logging a *user's* interactive shell into the
session beside the model's turns. `ctx.commands` is the mechanism that would
carry it — dispatch without a model turn — and nothing shipped uses it to shell
out, so a person running `ls` today does it outside the session and the log does
not know. A missing feature, not a porting error."""


@needs_fixtures
def test_every_record_type_has_a_home_here(shapes: dict[str, TrajectoryShape]) -> None:
    """The report's central claim, held against the fixtures that back it."""
    seen = {kind for shape in shapes.values() for kind in shape.record_types}
    assert seen, "the fixtures parsed to nothing"
    unmapped = seen - set(RECORD_TYPES)
    assert unmapped == set(), f"record types with no pH counterpart recorded: {sorted(unmapped)}"
    # And the table describes these fixtures rather than an older set of them.
    assert set(RECORD_TYPES) - seen == set(), "the table maps types no fixture carries"


@needs_fixtures
def test_every_role_is_mapped_or_named_as_a_gap(shapes: dict[str, TrajectoryShape]) -> None:
    """The one genuine capability gap, isolated rather than described.

    A role with an empty mapping is a gap; a role with no entry at all is an
    unexamined difference, which is what this module exists to prevent.
    """
    seen = {role for shape in shapes.values() for role in shape.roles}
    assert seen - set(ROLES) == set(), f"roles with no entry: {sorted(seen - set(ROLES))}"
    gaps = {role for role, home in ROLES.items() if not home}
    assert gaps == UNREPRESENTED_ROLES


# --------------------------------------------------------- the surface --


@needs_fixtures
def test_the_fixtures_are_the_coding_agent_not_the_rlm(shapes: dict[str, TrajectoryShape]) -> None:
    """The first finding, and the one that frames the rest.

    Both vendored trajectories are prime-agent's *coding* agent: `bash`, `edit`,
    `read`, `write` as native tool calls. Neither uses `ipython`, and neither
    spawns a child — so the fixtures exercise the surface C1-C3 replaced, not the
    RLM loop this bundle ports, and the `access` default the row predicted as a
    diff cannot be observed from them. Asserted rather than assumed, so a fixture
    added later that *does* delegate fails here and the report is rewritten.
    """
    for shape in shapes.values():
        assert shape.total_tool_calls > 0
        assert IPYTHON not in shape.tool_calls, "a fixture used the RLM surface after all"
        assert not any(name.startswith("rlm") for name in shape.tool_calls)


@needs_fixtures
async def test_every_tool_the_fixtures_called_exists_under_the_rlm_profile(
    shipped_profile: Any, shapes: dict[str, TrajectoryShape]
) -> None:
    """The claim that makes the trajectory expressible at all.

    Each of prime-agent's tool names is a registered tool in the shipped profile,
    so the same call is available — as a binding inside a cell rather than a
    native call, which is the translation the report describes.
    """
    ctx, _session, agent = await shipped_profile(profile=resolve_profile("rlm"))
    visible = set(ctx.tools.view(agent.ctx).visible)

    called = {name for shape in shapes.values() for name in shape.tool_calls}
    assert called, "the fixtures recorded no tool calls"
    missing = called - visible
    assert missing == set(), f"prime-agent called tools pH does not offer: {sorted(missing)}"


@needs_fixtures
def test_the_fixtures_carry_the_compaction_case(shapes: dict[str, TrajectoryShape]) -> None:
    """One fixture is named for it, and compaction is the surface operation the
    port models differently (`surface_op: replace`, I4) — so it is worth knowing
    the reference trajectory actually contains one."""
    before = shapes.get("before-compaction.jsonl")
    if before is None:
        pytest.skip("only some fixtures are vendored here")
    assert before.record_types.get("compaction", 0) >= 1
