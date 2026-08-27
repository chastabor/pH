"""Phase 1 capability seams: the guarantees each one is supposed to make.

Grouped because each seam's contract is small and its *failure mode* is the
interesting part: approval fails closed, sandbox refuses rather than passing
through, credentials never surrender a value above the adapter edge, and a
persistent code runtime cannot register without promising to snapshot.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph.cordis import Context
from ph.seams.approval import ApprovalRequest, ApprovalService, pending_approvals
from ph.seams.code_runtime import (
    CodeBindingNamespace,
    CodeRuntimeSeam,
    PersistenceObligationError,
    validate_binding_name,
)
from ph.seams.commands import CommandDefinition, CommandRegistry
from ph.seams.credentials import CredentialService
from ph.seams.jobs import JobService
from ph.seams.permission_presets import PermissionPresetService
from ph.seams.sandbox import SandboxError, SandboxPolicy, SandboxSeam
from ph.seams.settings import SettingsService
from ph.seams.skills import Skill, SkillService
from ph.seams.spill import SpillStore
from ph.seams.subprocess import SubprocessService, SubprocessSpawnSpec, scrub_env
from ph.session import Session
from ph.testing import StubAgent

pytestmark = pytest.mark.anyio


def _agent(session: Session) -> StubAgent:
    return StubAgent(session=session)


# ------------------------------------------------------------------ approval --


async def test_approval_records_both_halves_and_returns_the_outcome() -> None:
    root = Context()
    service = ApprovalService(ctx=root)
    session = Session("s")

    async def answerer(request: ApprovalRequest, next_: Any) -> str:
        return "allowed-once"

    root.on("approval/request", answerer)
    outcome = await service.request(agent=_agent(session), tool_name="edit", call_id="c1")
    assert outcome == "allowed-once"
    assert [event.type for event in session.events] == ["approval/asked", "approval/decided"]


async def test_register_answerer_is_the_waterfall_by_another_name() -> None:
    root = Context()
    service = ApprovalService(ctx=root)
    session = Session("s")

    async def answerer(request: ApprovalRequest, next_: Any) -> str:
        return "rejected"

    # One routing mechanism. A front-end reaching for the discoverable method
    # must land on the same path the pipeline reads, or every ask would report
    # "no approval channel" for a UI that is mounted.
    service.register_answerer(answerer)
    assert await service.request(agent=_agent(session), tool_name="edit") == "rejected"


async def test_approval_with_no_answerer_is_unavailable() -> None:
    root = Context()
    session = Session("s")
    outcome = await ApprovalService(ctx=root).request(agent=_agent(session), tool_name="edit")
    # Fail closed: a permission system whose failure mode is "allow" is not one.
    assert outcome == "unavailable"


async def test_an_answerer_that_raises_denies() -> None:
    root = Context()
    session = Session("s")
    service = ApprovalService(ctx=root)

    async def broken(request: ApprovalRequest, next_: Any) -> str:
        raise RuntimeError("the UI fell over")

    root.on("approval/request", broken)
    assert await service.request(agent=_agent(session), tool_name="edit") == "unavailable"


async def test_an_asked_approval_with_no_decision_is_pending_on_resume() -> None:
    session = Session("s")
    session.append("approval/asked", {"toolName": "edit", "callId": "c1"})
    (pending,) = pending_approvals(session)
    assert pending.tool_name == "edit"

    session.append("approval/decided", {"toolName": "edit", "callId": "c1", "outcome": "rejected"})
    # Derived from the log, so a crash between the two cannot lose the question.
    assert pending_approvals(session) == []


async def test_a_never_policy_answers_without_asking_anyone() -> None:
    root = Context()
    session = Session("s")
    service = ApprovalService(ctx=root)
    service.set_policy(session, "never")
    asked: list[str] = []
    root.on("approval/request", lambda request, next_: asked.append("asked") or "allowed-once")

    assert await service.request(agent=_agent(session), tool_name="edit") == "rejected"
    assert asked == []
    decided = [event for event in session.events if event.type == "approval/decided"]
    # Still recorded, so the log says why it was refused.
    assert decided[-1].data["automatic"] is True


# ------------------------------------------------------------------- sandbox --


async def test_confining_without_a_backend_raises_rather_than_passing_through() -> None:
    seam = SandboxSeam(ctx=Context())
    assert not seam.available
    with pytest.raises(SandboxError) as caught:
        seam.confine(("rm", "-rf", "/"), SandboxPolicy(mode="read-only"))
    assert caught.value.code == "SANDBOX_UNAVAILABLE"
    # The message has to name what is missing, because the caller's next move
    # depends on whether a backend exists at all.
    assert "sandbox-local" in str(caught.value)


async def test_sandbox_mode_resolution_order() -> None:
    seam = SandboxSeam(ctx=Context(), default_mode="read-only")
    session = Session("s")
    assert seam.resolve_mode(session) == "read-only"
    seam.set_mode(session, "workspace-write")
    assert seam.resolve_mode(session) == "workspace-write"
    # Explicit beats the log, which beats the deployment default.
    assert seam.resolve_mode(session, explicit="danger-full-access") == "danger-full-access"


async def test_a_permission_preset_sets_both_knobs() -> None:
    root = Context()
    session = Session("s")
    root.provide("sandbox", SandboxSeam(ctx=root))
    root.provide("approval", ApprovalService(ctx=root))
    presets = PermissionPresetService(ctx=root)

    presets.apply_preset("workspace-write", session)
    types = [event.type for event in session.events]
    assert types == ["permission/preset", "sandbox/mode", "approval/policy"]
    assert presets.resolve(session).name == "workspace-write"


# ---------------------------------------------------------------- subprocess --


def test_the_environment_is_scrubbed_of_anything_credential_shaped() -> None:
    base = {
        "PATH": "/usr/bin",
        "FOO_API_KEY": "sk-secret",
        "GH_TOKEN": "ghp_x",
        "DB_PASSWORD": "hunter2",
        "MY_SECRET_THING": "s",
        "HOME": "/home/x",
    }
    scrubbed = scrub_env(base)
    assert scrubbed == {"PATH": "/usr/bin", "HOME": "/home/x"}


async def test_a_child_does_not_inherit_a_planted_secret(tmp_path: Path) -> None:
    root = Context()
    service = SubprocessService(ctx=root)
    os.environ["PH_TEST_PLANTED_API_KEY"] = "sk-do-not-leak"
    try:
        code, out, _err = await service.run(
            SubprocessSpawnSpec(
                argv=(
                    "python3",
                    "-c",
                    "import os; print(len([k for k in os.environ if 'PLANTED' in k]))",
                ),
                cwd=tmp_path,
            )
        )
    finally:
        del os.environ["PH_TEST_PLANTED_API_KEY"]
    assert code == 0
    assert out.strip() == "0"


async def test_a_spawned_child_is_terminated_and_reaped_on_disposal(tmp_path: Path) -> None:
    root = Context()
    scope = root.scope("agent")
    service = SubprocessService(ctx=root)
    child = await service.spawn(
        SubprocessSpawnSpec(
            argv=("python3", "-c", "import time; time.sleep(30)"), cwd=tmp_path, grace_ms=200
        ),
        scope=scope,
    )
    assert child.returncode is None
    await scope.dispose()
    # Terminated by the effect and reaped, so no zombie survives the scope (F4).
    assert child.returncode is not None


# --------------------------------------------------------------------- spill --


async def test_spill_round_trips_and_names_the_way_back(tmp_path: Path) -> None:
    store = SpillStore(ctx=Context(), root=tmp_path)
    ref = await store.save_text(
        owner="agent-a", source="tool result", suggested_name="bash.txt", content="x" * 100
    )
    assert ref.bytes == 100
    assert await store.load_text(ref.locator) == "x" * 100
    # The preview has to tell the model how to get the rest in its own terms.
    assert ref.locator in ref.retrieval_hint


async def test_identical_content_spills_once(tmp_path: Path) -> None:
    store = SpillStore(ctx=Context(), root=tmp_path)
    first = await store.save_text(owner="a", source="s", suggested_name="n", content="same")
    second = await store.save_text(owner="a", source="s", suggested_name="n", content="same")
    assert first.locator == second.locator


# ---------------------------------------------------------------- credentials --


async def test_a_credential_reference_carries_no_value() -> None:
    service = CredentialService(ctx=Context())
    ref = service.reference("PH_TEST_KEY")
    assert "PH_TEST_KEY" in ref.to_wire()["name"]
    assert "value" not in ref.to_wire()

    service.provide_value("PH_TEST_KEY", "sk-secret")
    secret = service.require(ref)
    # Even a resolved secret refuses to print itself: a value that reaches a log
    # through a traceback has still leaked.
    assert "sk-secret" not in repr(secret)
    assert "sk-secret" not in str(secret)
    assert secret.reveal() == "sk-secret"


async def test_a_missing_credential_raises_rather_than_returning_empty() -> None:
    service = CredentialService(ctx=Context())
    with pytest.raises(KeyError, match="not available"):
        service.require(service.reference("PH_TEST_ABSENT_KEY"))


# --------------------------------------------------------------- code runtime --


async def test_a_persistent_runtime_must_promise_to_snapshot() -> None:
    seam = CodeRuntimeSeam(ctx=Context())

    class Forgetful:
        language = "python"
        isolation = "process"
        persistence = "namespace"

        async def run(self, request: Any) -> Any: ...

    with pytest.raises(PersistenceObligationError) as caught:
        seam.register(Forgetful())
    # Checked at registration, not the first time someone forks and discovers
    # the state was never durable (D6/D17).
    assert "kernel/snapshot" in str(caught.value)

    class Honest(Forgetful):
        declares_kernel_snapshots = True

    seam.register(Honest())
    assert seam.require().persistence == "namespace"


async def test_a_stateless_runtime_registers_freely() -> None:
    seam = CodeRuntimeSeam(ctx=Context())

    class Fresh:
        language = "python"
        isolation = "process"
        persistence = "none"

        async def run(self, request: Any) -> Any: ...

    seam.register(Fresh())
    assert seam.provider is not None


@pytest.mark.parametrize("name", ["class", "await", "interface", "2fast", "has-dash", ""])
def test_binding_names_must_be_portable(name: str) -> None:
    # One bindings list must be valid against every backend, so a name reserved
    # in any language pH renders an SDK for is reserved everywhere.
    with pytest.raises(ValueError):
        validate_binding_name(name)


def test_a_namespace_validates_its_own_bindings() -> None:
    CodeBindingNamespace(name="tools", bindings=(), description="ok")
    with pytest.raises(ValueError, match="reserved"):
        CodeBindingNamespace(name="class", bindings=())


async def test_a_disposed_renderer_leaves_an_absence_not_a_fallback() -> None:
    seam = CodeRuntimeSeam(ctx=Context())
    release = seam.register_sdk_renderer("python", lambda namespaces: "custom")
    assert seam.sdk_renderer("python") is not None
    release()
    # No silent revert to a default: a different listing would change the
    # prompt and invalidate the cached prefix with no error (A12).
    assert seam.sdk_renderer("python") is None


# ---------------------------------------------------------- commands and jobs --


async def test_a_command_dispatches_without_opening_a_turn() -> None:
    root = Context()
    registry = CommandRegistry(ctx=root)
    session = Session("s")
    registry.register(
        CommandDefinition(name="ping", summary="say pong", run=lambda arg, _ctx: "pong")
    )
    assert await registry.dispatch("/ping", session=session) == "pong"
    types = [event.type for event in session.events]
    assert types == ["command/run", "command/done"]
    # A command is something the human did; routing it through a model turn
    # would make the log say the model decided it.
    assert not any(event_type.startswith("turn/") for event_type in types)


async def test_a_failing_command_still_records_its_outcome() -> None:
    root = Context()
    registry = CommandRegistry(ctx=root)
    session = Session("s")

    def broken(_arg: str, _ctx: Any) -> str:
        raise ValueError("bad argument")

    registry.register(CommandDefinition(name="oops", summary="fails", run=broken))
    with pytest.raises(ValueError):
        await registry.dispatch("/oops", session=session)
    done = session.events[-1]
    assert done.data["outcome"] == "error"
    assert done.data["detail"] == "bad argument"


async def test_a_job_runs_and_reports() -> None:
    root = Context()
    service = JobService(ctx=root)
    job = await service.start(kind="test", label="a job", run=lambda _job: 42)
    assert service.get(job.id) is job

    # `start` hands back a handle, not an outcome — a job outliving the step that
    # started it is the whole point of the seam. `drain()` is what settles it.
    await root.drain()
    assert job.state == "done"
    assert job.result == 42


async def test_start_does_not_wait_for_the_body() -> None:
    """With no task group bound, a job detaches rather than running inline.

    The inline fallback was a placeholder — `bind()` never had a caller, so it was
    the only branch that ever ran — and it made `start()` block for exactly as
    long as the work, which is what a caller uses a job to avoid.
    """
    root = Context()
    service = JobService(ctx=root)
    entered = anyio.Event()
    release = anyio.Event()

    async def body(_job: Any) -> str:
        entered.set()
        await release.wait()
        return "eventually"

    job = await service.start(kind="test", label="slow", run=body)
    assert job.state == "running", "start waited for the body"
    release.set()
    await root.drain()
    assert job.state == "done"
    assert job.result == "eventually"
    assert entered.is_set()


async def test_a_cancelled_job_reports_cancelled() -> None:
    root = Context()
    service = JobService(ctx=root)

    def body(job: Any) -> str:
        job.cancel()
        return "partial"

    job = await service.start(kind="test", label="a job", run=body)
    await root.drain()
    assert job.state == "cancelled"


# ------------------------------------------------------- settings and skills --


async def test_settings_round_trip_a_dotted_key(tmp_path: Path) -> None:
    service = SettingsService(ctx=Context(), path=tmp_path / "settings.json")
    await service.set("model.provider", "deepseek")
    assert service.get("model.provider") == "deepseek"
    assert service.get("model.absent", "fallback") == "fallback"

    reloaded = SettingsService(ctx=Context(), path=tmp_path / "settings.json")
    assert reloaded.get("model.provider") == "deepseek"


async def test_a_corrupt_settings_file_does_not_stop_startup(tmp_path: Path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("{not json", encoding="utf-8")
    # Defaults are always a valid answer for a preference.
    assert SettingsService(ctx=Context(), path=path).get("anything", "default") == "default"


async def test_skill_bounds_are_enforced() -> None:
    service = SkillService(ctx=Context())
    service.register(Skill(name="review", description="Review a diff."))
    with pytest.raises(ValueError, match="already registered"):
        service.register(Skill(name="review", description="again"))
    with pytest.raises(ValueError, match=r"1\.\.64"):
        service.register(Skill(name="x" * 65, description="too long a name"))
    with pytest.raises(ValueError, match="at most 1024"):
        service.register(Skill(name="ok", description="y" * 1025))
