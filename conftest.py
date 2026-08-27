"""Shared pytest configuration.

Every async test runs on asyncio, because Textual requires it (D3) and a
harness that passed its tests on a backend it never ships on would be proving
the wrong thing.

`mount` is the one way a test stands up a profile: base + headless plus any
overlay rows, on a fresh root context that is disposed on teardown. Tests that
took a root of their own and remembered to dispose it were each re-deriving
`ph_app.runtime.mounted`.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Loader

MountProfile = Callable[..., Awaitable[Context]]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
async def mount(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[MountProfile]:
    """`await mount(*overlay_rows)` → a mounted root; disposed after the test.

    Sessions persist under `tmp_path` unless an overlay says otherwise, so no
    test writes into the developer's real `$PH_HOME`.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    roots: list[Context] = []

    async def _mount(*overlay_rows: dict[str, Any], profile: Any = None) -> Context:
        """`profile` layers a bundle between the base and the overlay.

        One keyword rather than a second fixture, because "mount the shipped
        profile" and "mount base plus these rows" differ by one document and
        should not differ by a lifecycle.
        """
        paths = [BASE, HEADLESS] if profile is None else [BASE, HEADLESS, profile]
        documents = Loader.from_paths(paths).documents
        if overlay_rows:
            documents.append(("test-overlay", list(overlay_rows)))
        ctx = Context()
        await Loader.from_documents(documents).mount(ctx)
        roots.append(ctx)
        return ctx

    yield _mount
    for ctx in reversed(roots):
        await ctx.drain()
        await ctx.dispose()
