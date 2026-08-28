"""The `rlm` profile: it composes, it boots, and a turn runs through it (P3-20).

`test_bundle.py` asserts things about the bundle *document*. This is about the
**profile** — `ph --profile rlm` — which is a different claim with a different
failure mode: a bundle can be perfectly well-formed and still not be reachable,
because nothing wired it to a name a person can type.

The wiring is deliberately indirect. `ph-app` does not depend on `ph-rlm` — the
same rule that lets the app read `subagent/*` events without importing the row
that emits them — so the profile discovers the bundle through the `ph.bundles`
entry-point group rather than importing it. The tests below are what make that
indirection safe: a typo in the entry point would leave `--profile rlm` failing
for a user, and nothing else in the suite would notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from conftest import HOST_INTERPRETER
from runtime_helpers import dispatch_names, run_ipython_cell

from ph.bundles import installed_bundles, resolve_bundle
from ph.cordis import Loader
from ph.session.json import freeze_json_value
from ph.testing import FAKE_OPTIONS
from ph.tools import ToolResult
from ph_app.profiles import available_profiles, resolve_profile
from ph_rlm import BUNDLE
from ph_rlm.presentation import IPYTHON

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """`resolve_profile` appends `$PH_HOME/profiles/<name>.yaml` when it exists,
    so a developer with a real `rlm.yaml` overlay would fail these otherwise."""
    monkeypatch.setenv("PH_HOME", str(tmp_path))


# ------------------------------------------------------------- discovery --


def test_the_bundle_is_discoverable_without_importing_it() -> None:
    """The entry point is the whole coupling between the app and this bundle."""
    assert "rlm" in installed_bundles()
    assert resolve_bundle("rlm") == BUNDLE


def test_an_unregistered_bundle_resolves_to_nothing() -> None:
    """`None`, not an exception: "that profile needs ph-rlm installed" is a
    configuration answer, and the caller has the context to word it."""
    assert resolve_bundle("no-such-bundle") is None


def test_the_profile_is_tui_plus_the_bundle() -> None:
    """The interactive posture, because a person is present for the approvals
    Code Mode's dispatches raise."""
    assert [path.name for path in resolve_profile("rlm")] == [
        "base.yaml",
        "headless.yaml",
        "tui.yaml",
        "bundle.yaml",
    ]


def test_the_profile_is_offered_by_name() -> None:
    assert "rlm" in available_profiles()


# ------------------------------------------------------------- it composes --


def test_the_composed_profile_is_code_mode_on_the_shipped_runtime() -> None:
    """What `--dump-config --profile rlm` would print: the facts a person reads
    to know which pH they are running."""
    rows = {row.id: row for row in Loader.from_paths(resolve_profile("rlm")).rows}

    assert rows["tools"].config["mode"] == "code"
    assert rows["code-runtime-python"].name == "code-runtime-python"
    assert rows["rlm-presentation"].name == "rlm-presentation"
    # The interactive posture survived the bundle layering on top of it.
    assert rows["sandbox"].config["defaultMode"] == "workspace-write"
    # Off by default, and still in the profile so a deployment can turn it on
    # by id rather than by forking the bundle.
    assert rows["rlm-context-loader"].disabled is True


# ------------------------------------------------------------ the smoke run --


async def test_a_turn_runs_end_to_end_on_the_composed_profile(mount: Any) -> None:
    """P3-20's gate: the profile boots and a turn goes through it.

    Mounted from `resolve_profile("rlm")` through the shared `mount` fixture, so
    what boots here is what `ph --profile rlm` composes — including the `tui`
    layer, which the bundle-level fixtures do not carry — and the session root
    is redirected rather than the developer's own. The interpreter is pinned to
    the host's, because building the managed venv shells out to `uv` and reaches
    the network.
    """
    ctx = await mount(
        {"id": "code-runtime-python", "config": HOST_INTERPRETER},
        profile=resolve_profile("rlm"),
    )
    session = ctx.sessions.create("smoke")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    # The whole stack in one call: a cell into the kernel, a governed binding
    # call back out through the tool pipeline, and a value returned.
    result = await run_ipython_cell(
        ctx,
        "found = await tools.glob(pattern='*.nothing', path='.')\nlen(found['paths'])",
        agent=agent,
        session=session,
    )
    assert result.is_error is False, result.error
    assert result.value["value"] == 0
    assert dispatch_names(session) == ["glob"]

    # And the model's surface is the one the profile promises.
    assert ctx.tools.get(IPYTHON, scope=agent.ctx) is not None
    assert ctx.harness is not None
    assert ctx.commands.get("refine") is not None

    # What the TUI's card is built from, read off the *mounted* definition. Both
    # halves of P3-19's cell hid here: `present_call` computed the program and
    # dropped it, and `present_result` tested its meta with `isinstance(..., dict)`
    # — which a frozen `MappingProxyType` off the log is not. Every widget test
    # constructs a `ToolCard` by hand, so nothing else crosses this seam.
    definition = ctx.tools.get(IPYTHON, scope=agent.ctx)
    assert definition is not None
    call_view = definition.present_call({"program": "rows = 1\nrows"})
    assert call_view.card == "terminal"
    assert call_view.input == "rows = 1", "the header line"
    assert call_view.body == "rows = 1\nrows", "the program the code cell renders"

    # The card's meta, through the chain the log puts it through: the tool
    # computes it, the log freezes it, the view reads it back. Frozen, because
    # `MappingProxyType` is what a live payload is and is *not* a `dict`.
    computed = definition.output.presentation_meta({}, {"value": 1, "dispatches": 1})
    result_view = definition.present_result(
        {"program": "rows = 1"},
        ToolResult(content=(), is_error=False, meta=freeze_json_value(computed)),
    )
    assert result_view is not None
    assert result_view.meta, "a frozen meta off the log was dropped"
    assert result_view.meta["dispatches"] == 1
