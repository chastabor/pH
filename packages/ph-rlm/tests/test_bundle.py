"""The `rlm` bundle mounts, and its rows agree with each other.

A bundle is the one artifact nothing else tests: every row here has its own unit
tests, but "these rows, in one profile, on top of base" is a separate claim, and
the failures it catches are ordering and naming rather than logic.

The behavioural half of that claim lives in `test_governance_gate.py`, which
mounts the same profile through the same fixture — so "the gate runs against what
ships" is true by construction rather than by two copies of one recipe.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml
from conftest import HOST_INTERPRETER
from runtime_helpers import dispatch_names, run_ipython_cell

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Loader
from ph.system_prompt.assembly import AssembleContext, render_prompt
from ph.tools.registry import RUN_CODE
from ph_rlm import BUNDLE
from ph_rlm.presentation import IPYTHON

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
    documents = Loader.from_paths([BASE, HEADLESS, BUNDLE]).documents
    documents.append(("test-overlay", [{"id": "code-runtime-python", "config": HOST_INTERPRETER}]))
    Loader.from_documents(documents)


async def test_every_row_in_the_profile_activates(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that mounts nothing is worse than one that fails: it looks fine.

    `Loader.inactive()` is the loader's own answer to "which rows never met their
    `inject` keys", and a profile with any is one where a capability silently
    is not there.

    **Mounted first**, and that is the whole test: `inactive()` reads the forks
    `mount()` creates, so the version this replaced asked the question of an
    empty table and returned `[]` for every profile, broken or not. Found when
    P4-01 copied it into `ph-stabilize` and the copy was checked for
    falsifiability.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    documents = Loader.from_paths([BASE, HEADLESS, BUNDLE]).documents
    documents.append(("test-overlay", [{"id": "code-runtime-python", "config": HOST_INTERPRETER}]))
    loader = Loader.from_documents(documents)
    ctx = Context()
    try:
        await loader.mount(ctx)
        assert loader.inactive() == []
    finally:
        await ctx.drain()
        await ctx.dispose()


async def test_the_bundle_mounts_over_base(shipped_profile: Any) -> None:
    ctx, _session, _agent = await shipped_profile()
    provider = ctx.code_runtime.require()
    assert provider.language == "python"
    assert provider.persistence == "namespace"
    # The two rows that have to arrive together: the provider promises to
    # snapshot at registration, and this is the row that keeps the promise.
    assert ctx.python_runtime.snapshots is not None


async def test_a_cell_runs_and_its_state_reaches_the_log(shipped_profile: Any) -> None:
    """The end-to-end claim of everything landed so far, in one test.

    Called as `ipython`, because that is the only name this profile's model is
    ever offered — `rlm-presentation` renames the transport, and a bundle test
    that reached past the rename would be testing a surface nobody ships.
    """
    ctx, session, agent = await shipped_profile()
    result = await run_ipython_cell(
        ctx, "remembered = 'yes'\nlen(remembered)", agent=agent, session=session
    )
    assert result.is_error is False
    assert result.value["value"] == 3
    assert any(event.type == "kernel/snapshot" for event in session.events)


async def test_the_shipped_profile_offers_exactly_one_callable(shipped_profile: Any) -> None:
    """Prime Agent's surface, kept: one entry, and the reserved name is not it."""
    ctx, _session, agent = await shipped_profile()
    view = ctx.tools.view(agent.ctx)

    assert view.mode == "code"
    assert view.transport_name == IPYTHON
    # Under Code Mode the model is handed no schema list at all; everything other
    # than the transport is reached as a binding (P1-04, C6).
    assert ctx.tools.schemas(scope=agent.ctx) == []
    assert ctx.tools.get(RUN_CODE, scope=agent.ctx) is None
    # And the base tool rows are still mounted — not directly callable, but
    # present, which is what makes them bindings rather than absences.
    assert len(view.visible) > 1


async def test_the_shipped_sdk_block_lists_the_four_namespaces(shipped_profile: Any) -> None:
    """P3-09's gate, satisfiable only now that P3-10's four namespaces exist.

    `tools` is Code Mode's own; `rlm`, `agent_message` and `agent_observe` are
    claims by three separate rows. The block is generated from the same factories
    the run asks, so this is also the assertion that a cell can reach every one.
    """
    ctx, _session, agent = await shipped_profile()
    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    text = render_prompt(assembly)

    assert "tools.read" in text or "tools.write" in text
    assert "rlm.run" in text
    assert "agent_message.send" in text
    assert "agent_observe.get" in text
    # And the governed tool names are not offered a second time.
    for tool in ("tools.rlm_run", "tools.agent_message_send", "tools.agent_observe_get"):
        assert tool not in text, f"{tool} is offered twice"


async def test_a_cell_can_delegate_through_the_shipped_profile(shipped_profile: Any) -> None:
    """The rows that must arrive together: transport, bindings, provider.

    Every one has unit tests; this is the claim that the *shipped* composition
    wires them to each other — a spawn written the way the model writes it,
    reaching the provider the bundle registers.
    """
    ctx, session, agent = await shipped_profile()
    result = await run_ipython_cell(
        ctx,
        "h = await rlm.run(prompt='look into it', name='scout')\nh['name']",
        agent=agent,
        session=session,
    )
    assert result.is_error is False
    assert result.value["value"] == "scout"
    # One governed dispatch, and the admission it caused, in one log.
    assert dispatch_names(session) == ["rlm_run"]
    admitted = [event for event in session.events if event.type == "subagent/admitted"]
    assert len(admitted) == 1
    assert admitted[0].data["grantedAccess"] == "read"


async def test_the_shipped_runtime_gets_the_configured_graces(shipped_profile: Any) -> None:
    """The `Config → PythonCodeRuntime → Kernel` handoff, which nothing covered.

    Three timing knobs travel that path field by field, and a knob added to
    `Config` but never threaded is one a deployment can set to no effect — which
    is exactly what `cancel_grace` was before P3-21.
    """
    ctx, _session, _agent = await shipped_profile(
        {"code-runtime-python": {"cancelGraceSeconds": 0.25, "shutdownGraceSeconds": 1.5}}
    )
    runtime = ctx.python_runtime
    assert runtime.cancel_grace == 0.25
    assert runtime.shutdown_grace == 1.5
    assert runtime.boot_timeout == 30.0, "an untouched knob keeps its default"
