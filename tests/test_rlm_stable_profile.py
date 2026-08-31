"""P4-15 — `rlm-stable`: the whole harness with the gates on.

The gate is one word — *boots* — and it is worth saying why that is not trivial.
This profile is the first that composes **both** capability bundles at once, so
it is the first place their rows meet: `rlm`'s Code Mode transport beside
`stabilize`'s tool gates, one `permissions-fs` over both, one containment tier
under both. A row that only ever ran beside its own siblings gets its first
integration here.

**A repo-level scenario, not a package test.** It composes `ph-app`'s profile
over `ph-rlm` and `ph-stabilize`, and no one of those three depends on the other
two — the entry-point indirection exists to keep that boundary honest. Living in
one package's suite would make a stabilize regression report as an rlm failure,
and would break that suite in any environment holding only one of them.

What it asserts is what a *composition* can get wrong: that the rows meant to be
on are on, and — the one that matters — that the human gate actually fires. The
first draft of this file read the YAML back through the loader and asserted the
values it had just written, which is a change-detector: it passed while the Code
Mode half of the gate was inert.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.cordis import DEPLOYMENT, Loader
from ph.testing import FAKE_OPTIONS, run_tool
from ph_app.profiles import available_profiles, resolve_profile

pytestmark = pytest.mark.anyio

PROFILE = "rlm-stable"


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`resolve_profile` appends `$PH_HOME/profiles/<name>.yaml` when it exists.

    Autouse because the tests that read the composition never touch the `mount`
    fixture, which is the only other thing that sets `PH_HOME` — so a developer
    with a personal `rlm-stable.yaml` overlay would otherwise be testing theirs.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))


def _rows() -> dict[str, Any]:
    """The composed rows by id — `test_rlm_profile.py`'s spelling."""
    return {row.id: row for row in Loader.from_paths(resolve_profile(PROFILE)).rows}


def test_the_profile_is_offered_by_name() -> None:
    """`available_profiles` answers with the same resolution `--profile` runs,
    so a profile is never offered and then refused."""
    assert PROFILE in available_profiles()


def test_it_composes_both_bundles() -> None:
    """The first profile that does. Rows from `rlm` and from `stabilize` have
    only ever run beside their own siblings until here."""
    rows = _rows()

    assert "code-runtime-python" in rows, "the rlm bundle did not compose"
    assert "tool-result-offload" in rows, "the stabilize bundle did not compose"


def test_the_rows_their_bundles_ship_disabled_are_turned_on() -> None:
    """The point of the profile. Both rows are `disabled: true` where they are
    defined, and both name this profile in the comment that says so."""
    rows = _rows()

    assert rows["tool-todo"].disabled is False
    assert rows["rlm-context-loader"].disabled is False
    # Below 200 000 characters the loader stands down entirely — Q4's threshold,
    # so a small `context/` directory costs nothing.
    assert rows["rlm-context-loader"].config["minChars"] == 200_000


def test_the_human_gate_names_calls_rather_than_everything() -> None:
    """A harness that asks about everything teaches its user to approve without
    reading, so the set is short — and it names `ph-stabilize`'s own pattern set
    rather than retyping it, which is how the two came to disagree about whether
    an ordinary `git push` is destructive."""
    gated = _rows()["hitl"].config["interruptOn"]

    assert set(gated) == {"bash", "run_code"}
    assert all(rule["preset"] == "destructive" for rule in gated.values())
    # A program is not something to hand-patch in a modal, and a half-edited one
    # is worse than either answer.
    assert "edit" not in gated["run_code"]["allowedDecisions"]


async def test_the_profile_boots_and_runs_a_turn(mount: Any) -> None:
    """The gate. Mounted through the real resolution, so a row that cannot
    activate beside the other bundle's rows fails here rather than at a user's
    first prompt."""
    ctx = await mount(profile=resolve_profile(PROFILE))
    session = ctx.sessions.create("stable")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await agent.prompt("hello")

    assert [event.type for event in session.events].count("turn/start") == 1
    # A live tool, not just composed config: `tool-todo` is the row that ships
    # disabled, so its tool existing is the flip having actually taken effect.
    assert ctx.tools.get("write_todos", scope=DEPLOYMENT) is not None
    # And both bundles' postures survived meeting each other.
    assert ctx.fs_permissions.rules, "the stabilize write scope is not in force"
    assert ctx.containment.for_role(child=True) == "worktree"


async def test_the_human_gate_fires_on_the_renamed_transport(mount: Any) -> None:
    """The claim the first draft of this file could not make.

    A profile writes `run_code` because that is the *reserved* transport name,
    but the registry renames the transport in place to whatever the presentation
    calls it — `ipython`, here. Keyed literally, the whole Code Mode half of a
    deployment's human gate is silently inert: the config parses, a test that
    reads the YAML back passes, and nothing ever asks.
    """
    ctx = await mount(profile=resolve_profile(PROFILE))
    session = ctx.sessions.create("gated")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    asked: list[str] = []
    ctx.approval.register_answerer(
        lambda request, next_: asked.append(request.tool_name) or "rejected"  # type: ignore[func-returns-value]
    )

    assert ctx.tools.view(DEPLOYMENT).transport_name == "ipython", (
        "the premise of this test changed"
    )
    await run_tool(ctx, "ipython", {"code": "import subprocess"}, agent=agent, session=session)

    assert asked, "the gate never fired: the rule is keyed to a name nothing presents"


async def test_an_ordinary_program_is_not_gated(mount: Any) -> None:
    """The other half, and the reason the rule carries a condition: a gate that
    asks about every cell is one a person learns to approve without reading."""
    ctx = await mount(profile=resolve_profile(PROFILE))
    session = ctx.sessions.create("ungated")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    asked: list[str] = []
    ctx.approval.register_answerer(
        lambda request, next_: asked.append(request.tool_name) or "rejected"  # type: ignore[func-returns-value]
    )

    await run_tool(ctx, "ipython", {"code": "print(1 + 1)"}, agent=agent, session=session)

    assert asked == []
