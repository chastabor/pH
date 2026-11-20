"""The `rlm` bundle mounts, and its rows agree with each other.

No other test covers a bundle. Every row here has its own unit tests, but "these
rows, in one profile, on top of base" is a separate claim. The failures that it
catches are in ordering and naming, not in logic.

The behavioral half of that claim is in `test_governance_gate.py`, which mounts
the same profile through the same fixture. "The gate runs against what ships"
is therefore true by construction, and not because two copies of one recipe
agree.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from rlm_fixtures import HOST_INTERPRETER, ShippedProfile
from runtime_helpers import dispatch_names, run_python_cell

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Profile, ProfileDocument, load_profile_documents
from ph.keys import AGENTS, CODE_RUNTIME, SUBAGENTS, SYSTEM_PROMPT, TOOLS
from ph.system_prompt.assembly import render_prompt
from ph.testing import report_section
from ph.tools.registry import RUN_CODE
from ph_rlm import BUNDLE
from ph_rlm.keys import PYTHON_RUNTIME
from ph_rlm.presentation import PYTHON

pytestmark = pytest.mark.anyio


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    """A row whose `name:` does not resolve fails at mount, in someone's session."""
    from ph.cordis.loader import resolve_plugin

    rows = yaml.safe_load(BUNDLE.read_text(encoding="utf-8"))
    named = [row for row in rows if isinstance(row, dict) and "name" in row]
    assert named, "the bundle declares no rows"
    for row in named:
        assert resolve_plugin(row["name"]) is not None, row["name"]


def test_a_patch_in_the_bundle_addresses_a_row_that_exists() -> None:
    """A patch naming an unknown id is a `LoaderError`, so composing proves it."""
    documents = load_profile_documents([BASE, HEADLESS, BUNDLE])
    documents.append(
        ProfileDocument("test-overlay", [{"id": "code-runtime-python", "config": HOST_INTERPRETER}])
    )
    Profile.from_documents(documents)


async def test_every_row_in_the_profile_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that mounts nothing is worse than a row that fails, because it looks
    fine.

    `Mount.inactive()` is the answer of the mount itself to "which rows are not
    active": a row that never met its `inject` keys, or a row whose `apply`
    failed. A profile with any such row is one where a capability is silently
    missing.

    **The profile is mounted first, and that is what makes this a test.**
    `inactive()` reads the forks that `mount()` creates. The version that this
    test replaced asked the question of an empty table, and it returned `[]` for
    every profile, broken or not. That was found when P4-01 copied the test into
    `ph-stabilize` and the copy was checked for falsifiability.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    documents = load_profile_documents([BASE, HEADLESS, BUNDLE])
    documents.append(
        ProfileDocument("test-overlay", [{"id": "code-runtime-python", "config": HOST_INTERPRETER}])
    )
    profile = Profile.from_documents(documents)
    ctx = Context()
    try:
        mount = await profile.mount(ctx)
        assert mount.inactive() == []
    finally:
        await ctx.drain()
        await ctx.dispose()


async def test_the_bundle_mounts_over_base(shipped_profile: ShippedProfile) -> None:
    ctx, _session, _agent = await shipped_profile()
    provider = ctx.require(CODE_RUNTIME).require()
    assert provider.language == "python"
    assert provider.persistence == "namespace"
    # These two rows must arrive together. The provider promises to snapshot at
    # registration, and this is the row that keeps the promise.
    assert ctx.require(PYTHON_RUNTIME).snapshots is not None


async def test_a_cell_runs_and_its_state_reaches_the_log(shipped_profile: ShippedProfile) -> None:
    """The end-to-end claim of everything that has landed so far, in one test.

    The tool is called as `python`, because that is the only name that the model
    of this profile is ever offered. `rlm-presentation` renames the transport. A
    bundle test that bypassed the rename would test a surface that nobody ships.
    """
    ctx, session, agent = await shipped_profile()
    result = await run_python_cell(
        ctx, "remembered = 'yes'\nlen(remembered)", agent=agent, session=session
    )
    assert result.is_error is False
    assert result.value["value"] == 3
    assert any(event.type == "kernel/snapshot" for event in session.events)


async def test_the_shipped_profile_offers_exactly_one_callable(
    shipped_profile: ShippedProfile,
) -> None:
    """The surface of prime-agent is kept: one entry, and that entry is not the
    reserved name."""
    ctx, _session, agent = await shipped_profile()
    view = ctx.require(TOOLS).view(agent.ctx)

    assert view.mode == "code"
    assert view.transport_name == PYTHON
    # Under Code Mode the model is handed one schema, the schema of the
    # transport, under the name that the profile presents. Everything else is
    # reached as a binding (P1-04, C6). If the model is handed no schema, a real
    # provider has no tool for the model to call.
    assert [one.name for one in ctx.require(TOOLS).schemas(scope=agent.ctx)] == [PYTHON]
    assert ctx.require(TOOLS).get(RUN_CODE, scope=agent.ctx) is None
    # The base tool rows are still mounted. They are not directly callable, but
    # they are present, which makes them bindings and not absences.
    assert len(view.visible) > 1


async def test_the_shipped_sdk_block_lists_the_four_namespaces(
    shipped_profile: ShippedProfile,
) -> None:
    """The gate of P3-09, which can be met only now that the four namespaces of
    P3-10 exist.

    `tools` belongs to Code Mode itself. Two other rows claim the other three:
    `rlm-bindings` claims `rlm`, and `rlm-messaging` claims `agent_message` and
    `agent_observe`. The block is generated from the same factories that the run
    asks, so this test also asserts that a cell can reach every namespace.
    """
    ctx, _session, agent = await shipped_profile()
    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    text = render_prompt(assembly)

    assert "tools.read" in text or "tools.write" in text
    assert "rlm.run" in text
    assert "agent_message.send" in text
    assert "agent_observe.get" in text
    # And the governed tool names are not offered a second time.
    for tool in ("tools.rlm_run", "tools.agent_message_send", "tools.agent_observe_get"):
        assert tool not in text, f"{tool} is offered twice"


async def test_a_cell_can_delegate_through_the_shipped_profile(
    shipped_profile: ShippedProfile,
) -> None:
    """The rows that must arrive together: transport, bindings and provider.

    Each of them has unit tests. This test claims that the *shipped* composition
    wires them to each other: a spawn that is written the way the model writes
    it reaches the provider that the bundle registers.
    """
    ctx, session, agent = await shipped_profile()
    result = await run_python_cell(
        ctx,
        "h = await rlm.run(prompt='look into it', name='scout')\nh['name']",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == "scout"
    # One governed dispatch is in the log of the parent, and the admission that
    # it caused is in the log of the child.
    assert dispatch_names(session) == ["rlm_run"]
    (child,) = ctx.require(SUBAGENTS).children(session.id).values()
    # `write` is the correct answer. The tier of the shipped profile is
    # `advisory`, where the child shares the checkout of its parent and can write
    # it. `read` here would be a guarantee that nothing in this profile enforces
    # (§4.8).
    assert child.granted_access == "write"
    # The child can also run code. Its grant lists what its parent sees, and
    # there the transport has its presented name. A restriction that was applied
    # under the registered name used to leave every child with no tool to call.
    child_agent = ctx.require(AGENTS).get(child.session_id)
    assert child_agent is not None
    offered = ctx.require(TOOLS).schemas(scope=child_agent.ctx)
    assert [one.name for one in offered] == [PYTHON]


async def test_the_shipped_runtime_gets_the_configured_graces(
    shipped_profile: ShippedProfile,
) -> None:
    """The `Config → PythonCodeRuntime → Kernel` handoff, which no other test
    covered.

    Three timing settings travel that path field by field. A setting that is
    added to `Config` but never threaded through is one that a deployment can set
    with no effect. `cancel_grace` was exactly that before P3-21.
    """
    ctx, _session, _agent = await shipped_profile(
        {"code-runtime-python": {"cancelGraceSeconds": 0.25, "shutdownGraceSeconds": 1.5}}
    )
    runtime = ctx.require(PYTHON_RUNTIME)
    assert runtime.cancel_grace == 0.25
    assert runtime.shutdown_grace == 1.5
    assert runtime.boot_timeout == 30.0, "an untouched knob keeps its default"


async def test_the_runtime_tells_doctor_what_runs_model_code(
    shipped_profile: ShippedProfile,
) -> None:
    """The reading of I-2, contributed and not imported (P4-12).

    `phern` cannot import this package, so the worker model reaches
    `phern doctor` through `ctx.diagnostics`. The assertion that matters is the
    last one: **reading the section must not resolve the interpreter**.
    `environment()` shells out to `uv`, and a diagnostic that builds a venv the
    first time that somebody runs it is a diagnostic that people stop running.
    """
    ctx, _session, _agent = await shipped_profile()

    rows = report_section(ctx, "Code runtime")

    assert rows["workers"].startswith("one CPython child per agent")
    assert rows["live kernels"] == "0"
    assert "not resolved yet" in rows["interpreter"]
