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
from ph.seams._names import SLUG_CHARACTERS
from ph.seams.approval import ApprovalRequest, ApprovalService, Edited, pending_approvals
from ph.seams.code_runtime import (
    CodeBindingNamespace,
    CodeRuntimeSeam,
    PersistenceObligationError,
    validate_binding_name,
)
from ph.seams.commands import CommandDefinition, CommandRegistry
from ph.seams.compaction import (
    CompactionError,
    CompactionNote,
    CompactionSeam,
)
from ph.seams.credentials import CredentialService
from ph.seams.jobs import JobService
from ph.seams.permission_presets import PermissionPresetService
from ph.seams.sandbox import SandboxError, SandboxPolicy, SandboxSeam
from ph.seams.settings import SettingsService
from ph.seams.skills import NAME_MAX, NAME_PATTERN, Skill, SkillService
from ph.seams.spill import SpillStore
from ph.seams.subprocess import SubprocessService, SubprocessSpawnSpec, scrub_env
from ph.seams.tui_screens import ID_MAX, ScreenDefinition, TuiScreenRegistry
from ph.seams.tui_status import StatusField, StatusReading, TuiStatusRegistry
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


async def test_an_answerer_cannot_decide_what_the_asking_row_withheld() -> None:
    """`allowed_decisions` is policy, so the seam that fails closed enforces it.

    A row withholds `edit` for a tool whose arguments must not be hand-written.
    Leaving that to the front end would make it a *rendering* rule — obeyed by
    the modal that hides the button and ignored by the next answerer, an RPC one
    or a test's, that returns an `Edited` anyway.

    A refusal is never withheld: `rejected`, `cancelled` and `unavailable` are
    how this seam says no, and a row that could suppress them would be a row
    that could force a call through.
    """
    root = Context()
    service = ApprovalService(ctx=root)
    session = Session("s")

    async def answerer(request: ApprovalRequest, next_: Any) -> Any:
        return Edited(arguments={"path": "elsewhere"})

    root.on("approval/request", answerer)
    outcome = await service.request(
        agent=_agent(session),
        tool_name="write",
        allowed_decisions=("approve", "reject"),
    )
    assert outcome == "unavailable"

    decided = next(event for event in session.events if event.type == "approval/decided")
    assert decided.data["outcome"] == "unavailable"
    assert "arguments" not in decided.data, "a withheld decision was recorded as if it took"


async def test_the_ask_does_not_carry_the_arguments_it_shows() -> None:
    """The field exists so a front end can put the call in front of a human;
    `tool/call` already recorded it (B4), and two copies of one fact in an
    append-only log are two that can disagree."""
    root = Context()
    session = Session("s")
    root.on("approval/request", lambda request, next_: "rejected")
    await ApprovalService(ctx=root).request(
        agent=_agent(session), tool_name="write", arguments={"path": "/etc/hosts"}
    )
    (asked,) = [event for event in session.events if event.type == "approval/asked"]
    assert "arguments" not in asked.data
    # Nor an empty one for a field nobody set: the record is built from what the
    # ask actually says, not by subtracting one key from the answerer's view.
    assert "allowedDecisions" not in asked.data


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


# ----------------------------------------------------------------- compaction --


async def test_an_absent_engine_is_a_no_op_automatically_and_an_error_on_request() -> None:
    """The seam's one asymmetry, and it is deliberate.

    A profile that layered no compaction row has *chosen* not to compact, so the
    policy hooks must not raise at it every step. But a person who typed
    `/compact` asked for something this deployment cannot do, and answering them
    with silence would read as "there was nothing to compact".
    """
    seam = CompactionSeam(ctx=Context())
    session = Session("no-engine")

    assert await seam.compact_if_needed(_agent(session), "pressure") is None
    with pytest.raises(CompactionError) as caught:
        await seam.compact_now(_agent(session))
    assert caught.value.code == "unavailable"


async def test_only_one_engine_may_hold_the_seam() -> None:
    """Two answers to "when and how is history replaced" is a contradiction."""
    seam = CompactionSeam(ctx=Context())

    class Engine:
        async def compact_if_needed(self, agent: Any, trigger: Any) -> Any:
            return None

        async def compact_now(self, agent: Any) -> Any:
            return None

    release = seam.register(Engine())
    with pytest.raises(RuntimeError, match="already registered"):
        seam.register(Engine())
    release()
    seam.register(Engine())


async def test_notes_render_in_order_and_an_empty_one_contributes_nothing() -> None:
    """`order` decides the sequence; `""` means absent, as `PromptContext` does."""
    seam = CompactionSeam(ctx=Context())
    session = Session("notes")
    seam.note(CompactionNote(name="second", text=lambda _s: "B", order=10))
    seam.note(CompactionNote(name="first", text=lambda _s: "A", order=1))
    seam.note(CompactionNote(name="quiet", text=lambda _s: ""))

    assert seam.notes(session) == ["A", "B"]


async def test_a_note_that_raises_is_dropped_rather_than_taking_the_compaction_down() -> None:
    """A summary missing one block is worth more than an uncompacted session."""
    seam = CompactionSeam(ctx=Context())
    session = Session("broken-note")

    def explode(_session: Session) -> str:
        raise RuntimeError("no runtime")

    seam.note(CompactionNote(name="broken", text=explode))
    seam.note(CompactionNote(name="fine", text=lambda _s: "still here"))

    assert seam.notes(session) == ["still here"]


async def test_a_note_registered_for_one_agent_does_not_reach_another() -> None:
    """The one visibility rule, shared with event dispatch and every scoped
    registry (B7): a global registration reaches everything, an agent-scoped one
    reaches that agent alone."""
    root = Context()
    seam = CompactionSeam(ctx=root)
    session = Session("scoped-notes")
    mine, theirs = root.scope("agent:mine"), root.scope("agent:theirs")
    seam.note(CompactionNote(name="mine", text=lambda _s: "only mine"), scope=mine)

    assert seam.notes(session, scope=mine) == ["only mine"]
    assert seam.notes(session, scope=theirs) == []


# ----------------------------------------------------------------- tui status --


async def test_a_status_field_reads_and_orders() -> None:
    """The footer's registration seam: `order`, then id, and nothing else."""
    registry = TuiStatusRegistry(ctx=Context())
    session = Session("footer")
    registry.register(StatusField(id="second", read=lambda _s: StatusReading("b"), order=10))
    registry.register(StatusField(id="first", read=lambda _s: StatusReading("a"), order=1))

    assert [one.text for one in registry.readings(session)] == ["a", "b"]


async def test_a_field_with_nothing_to_say_shows_nothing() -> None:
    """`None` rather than a placeholder: a line that always carries every field
    is a line where the one that matters cannot be seen."""
    registry = TuiStatusRegistry(ctx=Context())
    registry.register(StatusField(id="quiet", read=lambda _s: None))

    assert registry.readings(Session("footer")) == []


async def test_a_field_that_raises_does_not_take_the_footer_down() -> None:
    """It is read on every spinner frame; one bad row must not stop the rest."""
    registry = TuiStatusRegistry(ctx=Context())

    def explode(_session: Session) -> StatusReading:
        raise RuntimeError("no")

    registry.register(StatusField(id="broken", read=explode))
    registry.register(StatusField(id="fine", read=lambda _s: StatusReading("still here")))

    assert [one.text for one in registry.readings(Session("footer"))] == ["still here"]


async def test_a_field_unwinds_with_the_row_that_registered_it() -> None:
    """I2 in the footer: a row that unloads leaves no reading behind."""
    root = Context()
    registry = TuiStatusRegistry(ctx=root)
    row = root.scope("row")
    registry.register(StatusField(id="owned", read=lambda _s: StatusReading("x")), scope=row)
    assert registry.readings(Session("footer"))

    await row.dispose()

    assert registry.readings(Session("footer")) == []


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


async def test_a_job_is_an_effect_of_the_scope_that_owns_it() -> None:
    """I2: disposing the owner abandons the work rather than leaving it running.

    Without an owner the table only ever grew, and the bound would have had to be
    a number somebody picked.
    """
    root = Context()
    service = JobService(ctx=root)
    owner = root.scope("owner")
    release = anyio.Event()

    async def body(job: Any) -> str:
        await release.wait()
        return "unreached" if not job.token.cancelled else "noticed"

    job = await service.start(kind="test", label="owned", run=body, scope=owner)
    assert service.get(job.id) is job

    await owner.dispose()
    assert job.token.cancelled, "the owner went away and the work was not cancelled"
    assert service.get(job.id) is None, "the entry outlived its owner"
    release.set()
    await root.drain()


async def test_releasing_a_finished_job_does_not_report_it_cancelled() -> None:
    """The distinction the two halves exist for.

    A subagent's drive job disposes the child as its own last act, so a job that
    abandoned itself on that teardown would report `cancelled` for work that
    completed. `forget` drops the entry and cancels nothing.
    """
    root = Context()
    service = JobService(ctx=root)
    owner = root.scope("owner")

    job = await service.start(kind="test", label="brief", run=lambda _job: 7, scope=owner)
    await root.drain()
    assert job.state == "done"

    assert service.forget(job.id) is True
    assert service.get(job.id) is None
    assert service.forget(job.id) is False, "forgetting twice is not an error"

    # And the owner no longer holds a teardown for work that is over: disposing
    # it must not revive or re-report the job.
    await owner.dispose()
    assert job.state == "done"
    assert not job.token.cancelled


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


# ------------------------------------------------------------- tui screens --
# The front end's registration seam (P4-17). What is worth testing here is not
# that a dict holds a value, but the lifetime: a screen, and everything a front
# end derived from it, has to be one effect of the row that registered it.


def _screen(screen_id: str, **overrides: Any) -> ScreenDefinition:
    return ScreenDefinition(
        id=screen_id,
        label=screen_id.title(),
        build=lambda session: (screen_id, session),
        **overrides,
    )


async def test_screens_list_in_slot_order_then_id() -> None:
    """dsh orders slot entries by `order`; the id breaks ties so the list is
    stable rather than dependent on which row mounted first."""
    registry = TuiScreenRegistry(ctx=Context())
    registry.register(_screen("zebra", order=10))
    registry.register(_screen("apple"))
    registry.register(_screen("acorn"))

    assert [screen.id for screen in registry.list()] == ["zebra", "acorn", "apple"]


async def test_a_screen_id_must_be_addressable_as_a_command() -> None:
    """An id becomes `/<id>`, so what bounds it is the command grammar: a name
    with a space in it would be a command whose argument is part of its name."""
    registry = TuiScreenRegistry(ctx=Context())
    registry.register(_screen("audit"))
    with pytest.raises(ValueError, match="already registered"):
        registry.register(_screen("audit"))
    with pytest.raises(ValueError, match=r"1\.\.32"):
        registry.register(_screen("two words"))


def test_a_skill_name_and_a_screen_id_are_one_rule_at_two_bounds() -> None:
    """Both are a token a person types, so both are the same format.

    Pinned because the two seams used to hold a copy of the regex each, and two
    copies of a rule are a rule that will eventually be two. What may differ is
    the *bound*.

    Asked through the seams' own `register`, never of the shared helper: a test
    that called `require_slug` twice would be comparing it with itself and would
    pass for a seam that had quietly gone back to validating inline.
    """
    root = Context()

    def refuse_skill(name: str) -> str | None:
        try:
            SkillService(ctx=root).register(Skill(name=name, description="d"))
        except ValueError as error:
            return str(error)
        return None

    def refuse_screen(screen_id: str) -> str | None:
        try:
            TuiScreenRegistry(ctx=root).register(_screen(screen_id))
        except ValueError as error:
            return str(error)
        return None

    # The format is one thing: whichever characters one seam admits at a length
    # both allow, so does the other. Enumerated over the printable range rather
    # than a hand-picked pair, so a widened character class cannot slip past.
    printable = [chr(code) for code in range(32, 127)]
    assert {char for char in printable if refuse_skill(char) is None} == {
        char for char in printable if refuse_screen(char) is None
    }

    # `skills.NAME_PATTERN` is exported for a reader that *tests* rather than
    # raises (`rlm-skills-python` warns past a bad frontmatter name instead of
    # refusing the scan), so it is a second holder of the rule and is pinned to
    # what `register` enforces. `tui_screens` publishes no such constant — an
    # exported pattern with no consumer is a copy nothing can be held to.
    assert {char for char in printable if NAME_PATTERN.match(char)} == {
        char for char in printable if refuse_skill(char) is None
    }

    # The bound is not one thing, and it is the only thing that differs.
    assert ID_MAX < NAME_MAX
    assert refuse_screen("x" * ID_MAX) is None and refuse_screen("x" * (ID_MAX + 1))
    assert refuse_skill("x" * NAME_MAX) is None and refuse_skill("x" * (NAME_MAX + 1))
    assert refuse_screen("x" * NAME_MAX), "a screen id is held to its own bound"

    # And one sentence refuses both, naming the vocabulary rather than restating
    # the rule. Compared with the two things allowed to differ substituted out,
    # which is the claim: nothing *else* about the refusal may diverge.
    skill, screen = refuse_skill("two words"), refuse_screen("two words")
    assert skill is not None and screen is not None
    normalize = lambda text, kind, bound: text.replace(kind, "<kind>").replace(  # noqa: E731
        f"1..{bound}", "<bound>"
    )
    assert normalize(skill, "skill name", NAME_MAX) == normalize(screen, "screen id", ID_MAX)
    assert SLUG_CHARACTERS in screen


async def test_a_front_end_presents_screens_registered_before_and_after_it() -> None:
    """Attachment order must not decide what gets drawn: rows mount before a
    terminal exists in a resume, and after it when one is loaded late."""
    root = Context()
    registry = TuiScreenRegistry(ctx=root)
    drawn: list[str] = []
    registry.register(_screen("early"))

    registry.present_with(lambda screen: drawn.append(screen.id))
    registry.register(_screen("late"))

    assert drawn == ["early", "late"]


async def test_unloading_a_row_undoes_what_the_front_end_drew_for_it() -> None:
    """The gate, and the property ported from dsh: *"the registration rides the
    slot service's effect wrapper, so plugin unload removes the tab."*"""
    root = Context()
    registry = TuiScreenRegistry(ctx=root)
    undrawn: list[str] = []
    registry.present_with(lambda screen: lambda: undrawn.append(screen.id))

    row = root.scope("a-row")
    registry.register(_screen("audit"), scope=row)
    assert registry.get("audit") is not None

    await row.dispose()

    assert registry.get("audit") is None, "the screen outlived the row that registered it"
    assert undrawn == ["audit"], "the verb and the key outlived the row that registered them"


async def test_a_detaching_front_end_undoes_its_own_presentations() -> None:
    """The other lifetime. A terminal can close while the harness runs on, and
    a command left pointing at a dead app is worse than no command."""
    root = Context()
    registry = TuiScreenRegistry(ctx=root)
    undrawn: list[str] = []
    detach = registry.present_with(lambda screen: lambda: undrawn.append(screen.id))
    row = root.scope("a-row")
    registry.register(_screen("audit"), scope=row)

    detach()
    assert undrawn == ["audit"]
    assert registry.get("audit") is not None, "detaching a front end must not unregister rows"

    # The same teardown belongs to both lifetimes; whichever runs second does
    # nothing, which is why `add_disposer` hands back an idempotent release.
    await row.dispose()
    assert undrawn == ["audit"]


def test_settled_statuses_are_drawn_from_the_declared_vocabulary() -> None:
    """`SETTLED_STATUSES` is a subset of `SubagentStatus`, checked at runtime.

    The declaration constrains *writers* — `_status(status: SubagentStatus)` is
    type-checked, so no producer can emit a name that is not here. It cannot
    constrain readers: the value goes through `SessionEvent.data`, which is
    `Any` so a log written by another build can round-trip, and by the time a
    consumer folds it back out there is nothing left for mypy to check. A
    reader's `row.get("status") in {...}` type-checks against any strings at all,
    which is how P5-05 shipped a settled-set of `{completed, failed, cancelled,
    deleted}` — three names no producer writes — and pinned every parent that
    had ever run a child.

    `get_args` is the repo's answer to that asymmetry where it has been asked
    before (`sandbox.py` checks `event.data.get("mode")` against
    `get_args(SandboxMode)`); this is the same check one level up, on the subset
    rather than the value.
    """
    from typing import get_args

    from ph.seams.subagents import SETTLED_STATUSES, SubagentStatus

    declared = set(get_args(SubagentStatus))
    assert declared >= SETTLED_STATUSES, (
        f"settled statuses no producer can write: {SETTLED_STATUSES - declared}"
    )
    # And the live side is non-empty, or the predicate would call everything settled.
    assert declared - SETTLED_STATUSES
