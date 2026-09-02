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
from ph_rlm import BUNDLE
from ph_rlm.harness import HarnessEdit
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

PROVIDER_ROW: dict[str, Any] = {"id": "rlm-subagent-provider", "name": "rlm-subagent-provider"}
BINDINGS_ROW: dict[str, Any] = {"id": "rlm-bindings", "name": "rlm-bindings"}
MESSAGING_ROW: dict[str, Any] = {"id": "rlm-messaging", "name": "rlm-messaging"}
DOCTRINE_ROW: dict[str, Any] = {"id": "rlm-prompt", "name": "rlm-prompt"}
"""The delegation rows, here rather than re-spelled in four test modules — a row
set that drifts between them tests different things while appearing to test one."""

HARNESS_ROW: dict[str, Any] = {"id": "rlm-harness", "name": "rlm-harness"}
"""The Continual Harness row (P3-16), shared by both harness test modules."""

INVARIANT_ROW: dict[str, Any] = {"id": "rlm-harness-invariant", "name": "rlm-harness-invariant"}
"""I6's harness half (P6-01). Mounted only by the module that asserts on it."""

Harnessed = Callable[..., Any]


@pytest.fixture
def harnessed(mounted_runtime: Any) -> Harnessed:
    """`await harnessed(*rows)` -> `(ctx, session, agent)` with the harness mounted.

    On the real runtime, so H1 probes a live kernel, and with `$PH_HOME` under
    `tmp_path` (the root `mount` fixture), so the global log is this test's.
    Extra rows ride along so the invariant module can add its own without a
    second fixture of the same name mounting a different set.
    """

    async def build(*rows: dict[str, Any]) -> tuple[Any, Any, Any]:
        return await mounted_runtime(session_id="harness", extra_rows=[HARNESS_ROW, *rows])

    return build


def note_edit(entry_id: str, title: str = "a thing learned") -> HarnessEdit:
    """The smallest refinement: one created note."""
    return HarnessEdit(action="create", kind="note", id=entry_id, title=title, content="body")


HOST_INTERPRETER: dict[str, Any] = {"python": "host", "sweepOrphans": False}
"""The offline pin for `code-runtime-python`, as a *config fragment*.

Not a row: a loader patch replaces a row's whole config, so a test that also
tuned `cancelGraceSeconds` by appending a second patch would silently drop the
pin and start building a `uv` venv. `shipped_profile` merges fragments per row so
that cannot happen — which is exactly what it did before this was merged."""
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
def shipped_profile(mount: Any) -> Callable[..., Any]:
    """`await shipped_profile()` → `(ctx, session, agent)` on the real `rlm` bundle.

    `ph-base` + `headless` + `rlm/bundle.yaml` through the loader, so a row
    deleted from the shipped profile breaks a test rather than a hand-assembled
    row set quietly standing in for it.

    `config` is `{row_id: {key: value}}` and is **merged** per row before it
    becomes one patch. Appending raw patches instead is how the interpreter pin
    got dropped: two patches for one row means the second replaces the first
    whole, and nothing says so.

    `profile` is what to layer. The default is the bundle document; a caller that
    wants the *composed* profile — what `ph --profile rlm` resolves, `tui` layer
    and all — passes `resolve_profile("rlm")`. It goes through here rather than
    through `mount` directly so that merging rule keeps applying: three call
    sites had re-spelled the pin as a raw patch, which is the bypass this
    docstring is about.
    """

    async def build(
        config: dict[str, dict[str, Any]] | None = None,
        *,
        session_id: str = "profile",
        profile: Any = BUNDLE,
    ) -> tuple[Any, Any, Any]:
        merged: dict[str, dict[str, Any]] = {"code-runtime-python": dict(HOST_INTERPRETER)}
        for row_id, overrides in (config or {}).items():
            merged.setdefault(row_id, {}).update(overrides)
        ctx = await mount(
            *({"id": row_id, "config": values} for row_id, values in merged.items()),
            profile=profile,
        )
        session = ctx.sessions.create(session_id)
        return ctx, session, ctx.agents.create(session, FAKE_OPTIONS)

    return build


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
