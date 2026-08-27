"""Fixtures for the runtime tests.

Every kernel here runs on `python: "host"` — pH's own interpreter. The managed
venv is what ships, but building it shells out to `uv` and reaches the network,
which would make the conformance suite slow and offline-hostile. `test_venv.py`
covers the managed path's decisions separately, without building one.

Two fixtures, at two levels. `make_kernel` drives the runtime alone, with plain
functions standing in for bindings; `mounted_runtime` mounts the real profile so
a binding call goes out over fd 3 and comes back through the whole tool
pipeline. The rows and the execute incantation live here because all three test
modules need the same ones, and a row set that drifts between them tests
different things while appearing to test one.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from ph.testing import FAKE_OPTIONS
from ph_rlm.kernel.journal import OrphanJournal
from ph_rlm.kernel.manager import Kernel, KernelLimits, _declare
from ph_rlm.kernel.venv import resolve_interpreter

MakeKernel = Callable[..., "Kernel"]

HOST_RUNTIME_ROW: dict[str, Any] = {
    "id": "code-runtime-python",
    "name": "code-runtime-python",
    "config": {"python": "host", "sweepOrphans": False},
}
"""The shipped runtime row with the interpreter pinned to the host's."""

CODE_MODE_ROWS: list[dict[str, Any]] = [
    # A *patch*: `id` alone addresses the base row. With a `name:` it would
    # insert a second `tools` row, and two providers of one service is a
    # conflict the context refuses.
    {"id": "tools", "config": {"mode": "code"}},
    # Code Mode is opt-in in `ph-base` — the transport is not registered until a
    # profile asks for it. The `rlm` bundle asks; so does every test here.
    {"id": "tools-code-mode", "name": "tools-code-mode"},
]

SNAPSHOT_ROW: dict[str, Any] = {"id": "rlm-kernel-snapshot", "name": "rlm-kernel-snapshot"}

PRESENTATION_ROW: dict[str, Any] = {"id": "rlm-presentation", "name": "rlm-presentation"}
"""The model-facing rename. Off unless a test asks for it, because most tests
here address the transport by its reserved name."""


@pytest.fixture
async def make_kernel(tmp_path: Path) -> AsyncIterator[MakeKernel]:
    """`await make_kernel(**limits)` → a started kernel, closed after the test.

    Namespaces are passed as `CodeBindingNamespace`s and declared here, so a
    caller does not hold the same list in two shapes.
    """
    started: list[Kernel] = []
    environment = resolve_interpreter(cache=tmp_path, mode="host")

    async def make(
        *,
        namespaces: tuple[Any, ...] = (),
        cancel_grace: float = 2.0,
        **limits: object,
    ) -> Kernel:
        kernel = Kernel(
            namespace="agent-test",
            environment=environment,
            limits=KernelLimits(**limits),  # type: ignore[arg-type]
            journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
            boot_timeout=60.0,
            cancel_grace=cancel_grace,
        )
        await kernel.start([_declare(namespace) for namespace in namespaces])
        started.append(kernel)
        return kernel

    yield make
    for kernel in started:
        await kernel.aclose()


@pytest.fixture
def mounted_runtime(mount: Any) -> Callable[..., Any]:
    """`await mounted_runtime(...)` → `(ctx, session, agent)` on the real profile.

    `snapshots=False` mounts the runtime *without* the snapshot policy, which is
    how `test_runtime_integration.py` checks the runtime on its own;
    `presentation=True` adds the row that renames the transport to `ipython`;
    `extra_rows` appends whatever else a test needs — profile rows, or a config
    *patch* (an `id` with no `name`), which is how a test reaches the budgets on
    `tools-code-mode` without this fixture growing a parameter per knob.
    """

    async def build(
        *,
        session_id: str = "runtime-test",
        snapshots: bool = True,
        presentation: bool = False,
        snapshot_config: dict[str, Any] | None = None,
        extra_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[Any, Any, Any]:
        rows = [*CODE_MODE_ROWS, HOST_RUNTIME_ROW]
        if snapshots:
            rows.append(
                {**SNAPSHOT_ROW, "config": snapshot_config} if snapshot_config else SNAPSHOT_ROW
            )
        if presentation:
            rows.append(PRESENTATION_ROW)
        rows.extend(extra_rows or [])
        ctx = await mount(*rows)
        session = ctx.sessions.create(session_id)
        return ctx, session, ctx.agents.create(session, FAKE_OPTIONS)

    return build
