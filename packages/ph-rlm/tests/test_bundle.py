"""The `rlm` bundle mounts, and its rows agree with each other.

A bundle is the one artifact nothing else tests: every row here has its own unit
tests, but "these rows, in one profile, on top of base" is a separate claim, and
the failures it catches are ordering and naming rather than logic.
"""

from __future__ import annotations

from typing import Any

import pytest
import yaml

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Loader
from ph.testing import FAKE_OPTIONS
from ph.tools.definition import ToolExecutionInput
from ph.tools.registry import RUN_CODE
from ph_rlm import BUNDLE

pytestmark = pytest.mark.anyio

# The interpreter is pinned to the host's so this needs no `uv` and no network;
# everything else is the shipped configuration.
HOST_INTERPRETER: dict[str, Any] = {
    "id": "code-runtime-python",
    "config": {"python": "host", "sweepOrphans": False},
}


def _documents() -> list[tuple[str, Any]]:
    documents = Loader.from_paths([BASE, HEADLESS, BUNDLE]).documents
    documents.append(("test-overlay", [HOST_INTERPRETER]))
    return documents


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
    Loader.from_documents(_documents())


async def test_the_bundle_mounts_over_base() -> None:
    ctx = Context()
    try:
        await Loader.from_documents(_documents()).mount(ctx)
        provider = ctx.code_runtime.require()
        assert provider.language == "python"
        assert provider.persistence == "namespace"
        # The two rows that have to arrive together: the provider promises to
        # snapshot at registration, and this is the row that keeps the promise.
        assert ctx.python_runtime.snapshots is not None
    finally:
        await ctx.drain()
        await ctx.dispose()


async def test_a_cell_runs_and_its_state_reaches_the_log() -> None:
    """The end-to-end claim of everything landed so far, in one test."""
    ctx = Context()
    try:
        await Loader.from_documents(_documents()).mount(ctx)
        session = ctx.sessions.create("bundle-smoke")
        agent = ctx.agents.create(session, FAKE_OPTIONS)
        result = await ctx.tools.execute(
            ToolExecutionInput(
                call_id="c1",
                name=RUN_CODE,
                arguments={"program": "remembered = 'yes'\nlen(remembered)"},
                scope=agent.ctx,
                session=session,
                agent=agent,
            )
        )
        assert result.is_error is False
        assert result.value["value"] == 3
        assert any(event.type == "kernel/snapshot" for event in session.events)
    finally:
        await ctx.drain()
        await ctx.dispose()
