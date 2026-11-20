"""Fixtures for the runtime tests.

Every kernel here runs on `python: "host"`, the interpreter of pH itself. The
managed venv is what ships. But building it shells out to `uv` and reaches the
network, which would make the conformance suite slow and unusable offline.
`test_venv.py` covers the decisions of the managed path separately, without
building a venv.

`make_kernel` and `mounted_runtime` work at two levels. `make_kernel` drives the
runtime alone, with plain functions that stand in for bindings.
`mounted_runtime` mounts the real profile, so a binding call goes out over fd 3
and comes back through the whole tool pipeline. The rows are declared here, and
the helper that runs a cell is in `runtime_helpers.py`, because many test
modules need the same ones. A row set that drifts between modules tests
different things and appears to test one thing.
"""

from __future__ import annotations

import shutil
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from pathlib import Path
from typing import Any

import anyio
import pytest
from runtime_helpers import DISPATCH_SETTLED, DISPATCH_START

from ph.agent.types import AgentDriver
from ph.cordis import Context
from ph.keys import AGENTS, SESSIONS, WORKSPACE
from ph.llm.fake import FakeAdapter
from ph.orphans import OrphanJournal
from ph.paths import resolve_roots
from ph.persistence.lease import LEASES
from ph.seams.subagents import FamilyRole
from ph.seams.workspace import Workspace
from ph.session import Session
from ph.testing import FAKE_OPTIONS, MountProfile, log_event, models_overlay
from ph_rlm import BUNDLE
from ph_rlm.harness import HarnessEdit, HarnessReference
from ph_rlm.kernel.manager import Kernel, KernelLimits, _declare
from ph_rlm.kernel.venv import resolve_interpreter
from ph_rlm.messaging import SEND_TOOL

type MakeKernel = Callable[..., Awaitable["Kernel"]]
"""`await make_kernel(...)` → a live kernel. The alias is awaitable because
`make` is `async`. The alias said otherwise before, and nothing checked it while
the trees were outside mypy (issue 32)."""

type ShippedProfile = Callable[..., Awaitable[tuple[Context, Session, AgentDriver]]]
"""`await shipped_profile(...)` → `(ctx, session, agent)` on the shipped bundle.

This is the same triple that `MountedRuntime` returns, under a separate name on
purpose. The difference is *which* profile was mounted, and that is the purpose
of the fixture below. A test that was annotated with the other name would read
as the wrong mount."""

type MountedRuntime = Callable[..., Awaitable[tuple[Context, Session, AgentDriver]]]
"""`await mounted_runtime(...)` → `(ctx, session, agent)` on the real profile.

The alias has a name because seventeen tests took the fixture as `Any` and then
read services from the `ctx` that it returns. That is how 1,581 untyped reads
hid behind one fixture (issue 32).

**The agent is an `AgentDriver`, not an `AgentHandle`.** The handle is the
read-only half, and `AgentRegistry.create` returns the driver. The narrower name
was used here while only three suites used the alias, and none of them steered
the agent. The seven suites that wrote `Any` did steer it, so
`await agent.prompt(...)` failed as soon as they used the alias."""

HOST_RUNTIME_ROW: dict[str, Any] = {
    "id": "code-runtime-python",
    "name": "code-runtime-python",
    "config": {"python": "host"},
}
"""The shipped runtime row, with the interpreter pinned to the host interpreter."""

CODE_MODE_ROWS: list[dict[str, Any]] = [
    # This is a *patch*: `id` alone addresses the base row. With a `name:` it
    # would insert a second `tools` row, and two providers of one service are a
    # conflict that the context refuses.
    {"id": "tools", "config": {"mode": "code"}},
    # Code Mode is opt-in in `ph-base`: the transport is not registered until a
    # profile asks for it. The `rlm` bundle asks, and so does every test here.
    {"id": "tools-code-mode", "name": "tools-code-mode"},
]

SNAPSHOT_ROW: dict[str, Any] = {"id": "rlm-kernel-snapshot", "name": "rlm-kernel-snapshot"}

PRESENTATION_ROW: dict[str, Any] = {"id": "rlm-presentation", "name": "rlm-presentation"}

PROVIDER_ROW: dict[str, Any] = {"id": "rlm-subagent-provider", "name": "rlm-subagent-provider"}
BINDINGS_ROW: dict[str, Any] = {"id": "rlm-bindings", "name": "rlm-bindings"}
MESSAGING_ROW: dict[str, Any] = {"id": "rlm-messaging", "name": "rlm-messaging"}
DOCTRINE_ROW: dict[str, Any] = {"id": "rlm-prompt", "name": "rlm-prompt"}
"""The delegation rows. They are declared here and are not repeated in the six
test modules that use them. A row set that drifts between modules tests
different things and appears to test one thing."""

MODELS_ROW = models_overlay(classify={"provider": "fake", "model": "fake-9"})
"""The models list of a parent: `main` on its own route, and `classify`, the
model that a sorting skill asks for."""

HARNESS_ROW: dict[str, Any] = {"id": "rlm-harness", "name": "rlm-harness"}
"""The Continual Harness row (P3-16), shared by the three harness test modules."""

INVARIANT_ROW: dict[str, Any] = {"id": "rlm-harness-invariant", "name": "rlm-harness-invariant"}
"""The harness half of I6 (P6-01). Only the module that asserts on it mounts it."""

Harnessed = Callable[..., Any]


def reply_to_parent(
    child: Session,
    sub_call_id: str = "reply-1",
    *,
    role: FamilyRole = "parent",
    failed: bool = False,
) -> None:
    """Write a send into the log of `child`, as a Code Mode dispatch records one:
    its start, which names the receiver, and then its settle. The records have
    the shape of a dispatch and not of a native call, because a dispatch derives
    no message. A log that a test writes the send into therefore stays a log
    that the model could have produced, mid-turn or across a restart."""
    ref = {
        "rootCallId": "cell",
        "parentCallId": "cell",
        "subCallId": sub_call_id,
        "name": SEND_TOOL,
    }
    log_event(
        child,
        DISPATCH_START,
        {**ref, "arguments": {"message": "here it is", "receiver_role": role}},
    )
    log_event(child, DISPATCH_SETTLED, {**ref, "isError": failed, "content": []})


def logs_after_a_crash() -> Path:
    """The sessions directory as a process that died would leave it, under a root
    of its own, for a second harness to resume from.

    This is a snapshot and not the live files. The first harness is still alive
    and holds the leases of its sessions (I-5, L2), and a second harness that
    opened the same files would correctly be refused them. A process that died
    holds nothing and leaves only what it wrote. The leases are therefore not
    copied, and what the first harness writes while it unwinds at teardown never
    reaches the snapshot.
    """
    snapshot = resolve_roots().sessions_dir().parent / "sessions-after-restart"
    shutil.copytree(resolve_roots().sessions_dir(), snapshot, ignore=shutil.ignore_patterns(LEASES))
    return snapshot


@pytest.fixture
def harnessed(mounted_runtime: MountedRuntime) -> Harnessed:
    """`await harnessed(*rows)` -> `(ctx, session, agent)` with the harness mounted.

    The fixture uses the real runtime, so H1 probes a live kernel. `$PH_HOME` is
    under `tmp_path` (the root `mount` fixture), so the global log belongs to
    this test. Extra rows are appended, so that the invariant module can add its
    own without a second fixture of the same name that mounts a different set.
    """

    async def build(*rows: dict[str, Any]) -> tuple[Any, Any, Any]:
        return await mounted_runtime(session_id="harness", extra_rows=[HARNESS_ROW, *rows])

    return build


async def agent_with_workspace(
    ctx: Context, session: Session, agent: AgentDriver, base: Path
) -> Workspace:
    """Acquire the workspace of an agent. The lifecycle row acquires it at the
    first step of an agent, and these tests never take a step, so the workspace
    is acquired in the way that the ladder tests do.

    The base is created because the kernel will `chdir` to it. The shared
    provider returns the directory that it was given and does not create it, and
    a missing cwd fails the spawn and not the confinement.
    """
    base.mkdir(parents=True, exist_ok=True)
    return await ctx.require(WORKSPACE).acquire(session_id=session.id, agent_id=agent.id, base=base)


def skill_edit(entry_id: str, module: str, name: str) -> HarnessEdit:
    """One created skill entry that names `module.name` for H1 to resolve."""
    return HarnessEdit(
        action="create",
        kind="skill",
        id=entry_id,
        title=f"{module}.{name}",
        content="call it",
        reference=HarnessReference(module=module, callable=name),
    )


def note_edit(entry_id: str, title: str = "a thing learned") -> HarnessEdit:
    """The smallest refinement: one created note."""
    return HarnessEdit(action="create", kind="note", id=entry_id, title=title, content="body")


HOST_INTERPRETER: dict[str, Any] = {"python": "host"}
"""The offline pin for `code-runtime-python`, as a *config fragment*.

It is not a row. A loader patch replaces the whole config of a row. A test that
also tuned `cancelGraceSeconds` by appending a second patch would therefore
silently drop the pin and start building a `uv` venv. `shipped_profile` merges
fragments for each row so that this cannot happen. It is exactly what happened
before the fragments were merged."""


@pytest.fixture
async def make_kernel(tmp_path: Path, guest_coverage: None) -> AsyncIterator[MakeKernel]:
    """`await make_kernel(**limits)` → a started kernel, closed after the test.

    `guest_coverage` is requested and not used. A kernel is a guest subprocess,
    and that fixture makes the coverage of the guest collectible. Because it is
    declared here, the instrument finds the tests that start guests, and nobody
    has to maintain a list of them.

    Namespaces are passed as `CodeBindingNamespace`s and declared here, so a
    caller does not hold the same list in two shapes.
    """
    started: list[Kernel] = []
    environment = resolve_interpreter(cache=tmp_path, mode="host")

    async def make(
        *,
        namespaces: tuple[Any, ...] = (),
        cancel_grace: float = 2.0,
        probe_seconds: float | None = 1.0,
        **limits: object,
    ) -> Kernel:
        kernel = Kernel(
            namespace="agent-test",
            environment=environment,
            limits=KernelLimits(**limits),  # type: ignore[arg-type]
            journal=OrphanJournal(path=tmp_path / "processes.jsonl"),
            boot_timeout=60.0,
            cancel_grace=cancel_grace,
            probe_seconds=probe_seconds,
        )
        await kernel.start([_declare(namespace) for namespace in namespaces])
        started.append(kernel)
        return kernel

    yield make
    for kernel in started:
        await kernel.aclose()


@pytest.fixture
def shipped_profile(mount: MountProfile, guest_coverage: None) -> ShippedProfile:
    """`await shipped_profile()` → `(ctx, session, agent)` on the real `rlm` bundle.

    The fixture mounts `ph-base` + `headless` + `rlm/bundle.yaml` through the
    loader. A row that is deleted from the shipped profile therefore breaks a
    test, and a hand-assembled row set does not quietly stand in for it.

    `config` is `{row_id: {key: value}}`. It is **merged** for each row before
    it becomes one patch. Appending raw patches is how the interpreter pin was
    dropped: with two patches for one row, the second replaces the whole of the
    first, and nothing reports it.

    `profile` is what to layer. The default is the bundle document. A caller
    that wants the *composed* profile (what `phern --profile rlm` resolves,
    including the `tui` layer) passes `resolve_profile("rlm")`. The profile goes
    through this fixture and not through `mount` directly, so that the merging
    rule still applies. Three call sites had written the pin again as a raw
    patch, which is the bypass that this docstring is about.
    """

    async def build(
        config: dict[str, dict[str, Any]] | None = None,
        *,
        session_id: str = "profile",
        profile: object = BUNDLE,
    ) -> tuple[Context, Session, AgentDriver]:
        merged: dict[str, dict[str, Any]] = {"code-runtime-python": dict(HOST_INTERPRETER)}
        for row_id, overrides in (config or {}).items():
            merged.setdefault(row_id, {}).update(overrides)
        ctx = await mount(
            *({"id": row_id, "config": values} for row_id, values in merged.items()),
            profile=profile,
        )
        session = ctx.require(SESSIONS).create(session_id)
        return ctx, session, ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    return build


@pytest.fixture
def mounted_runtime(mount: MountProfile, guest_coverage: None) -> MountedRuntime:
    """`await mounted_runtime(...)` → `(ctx, session, agent)` on the real profile.

    * `snapshots=False` mounts the runtime *without* the snapshot policy.
      `test_runtime_integration.py` uses it to check the runtime on its own.
    * `presentation=True` adds the row that renames the transport to `python`,
      the name that the model sees. It is off unless a test asks for it, because
      most tests here address the transport by its reserved name.
    * `extra_rows` appends whatever else a test needs: profile rows, or a config
      *patch* (an `id` with no `name`). A patch is how a test reaches the
      budgets on `tools-code-mode` without a new parameter on this fixture for
      each setting.
    """

    async def build(
        *,
        session_id: str = "runtime-test",
        snapshots: bool = True,
        presentation: bool = False,
        snapshot_config: dict[str, Any] | None = None,
        extra_rows: list[dict[str, Any]] | None = None,
    ) -> tuple[Context, Session, AgentDriver]:
        rows = [*CODE_MODE_ROWS, HOST_RUNTIME_ROW]
        if snapshots:
            rows.append(
                {**SNAPSHOT_ROW, "config": snapshot_config} if snapshot_config else SNAPSHOT_ROW
            )
        if presentation:
            rows.append(PRESENTATION_ROW)
        rows.extend(extra_rows or [])
        ctx = await mount(*rows)
        session = ctx.require(SESSIONS).create(session_id)
        return ctx, session, ctx.require(AGENTS).create(session, FAKE_OPTIONS)

    return build


# ------------------------------------------------------------ a held model --


class ModelGate:
    """Holds the model call of every child until the test lets one through.

    The gate is patched over the `stream` of the fake adapter. A child is
    therefore "running" for as long as the test says, and no timing is guessed.
    `arrived` counts calls, because the held list shrinks as calls are released.

    The `gate` fixture below opens the gate again, and that is not only
    tidiness. A call that is left parked here holds open every wait on its job,
    so one failed assertion becomes a hung suite. If each test wrote the release
    as a `finally`, that would be a rule that the fourth test must remember. The
    release is therefore here, beside the fixtures that every rlm suite shares.
    """

    def __init__(self) -> None:
        self.held: list[anyio.Event] = []
        self.arrived = 0
        self.open = False

    def release_one(self) -> None:
        self.held.pop(0).set()

    def twice(self) -> bool:
        """Whether a second child has reached the model. This is the proof of a
        readmit."""
        return self.arrived >= 2

    def release_all(self) -> None:
        self.open = True
        for event in self.held:
            event.set()
        self.held.clear()

    def patch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        original = FakeAdapter.stream
        gate = self

        async def gated(self: Any, options: Any) -> Any:  # noqa: ANN401
            gate.arrived += 1
            if not gate.open:
                event = anyio.Event()
                gate.held.append(event)
                await event.wait()
            async for chunk in original(self, options):
                yield chunk

        monkeypatch.setattr(FakeAdapter, "stream", gated)


@pytest.fixture
def gate(monkeypatch: pytest.MonkeyPatch) -> Iterator[ModelGate]:
    """A held model, released whatever the test does. See `ModelGate`."""
    held = ModelGate()
    held.patch(monkeypatch)
    yield held
    held.release_all()
