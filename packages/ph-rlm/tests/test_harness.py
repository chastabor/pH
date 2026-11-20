"""The Continual Harness: state as a fold, and the checks on writing it (P3-16).

The gates of the plan for this row, with one test for each:

* delete-and-re-derive is byte-identical
* concurrent global writes are safe
* a missing import is rejected, with the failure on the event
* an `edit` entry renders the binding form
* a global edit prompts
* rollback restores the fold

One claim needs a statement here: why the state is a fold at all (D14). Prime
Agent keeps this state in a JSON file. A file cannot express "the harness as of
a fork boundary". It also needs an mtime guard and a conflict rule, and an
append-only log needs neither.

The tests mount the real kernel, because the probe of H1 resolves references in
the runtime that the model uses. A check against this process would prove
something about the imports of the harness and not about the imports of the
model.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import anyio
import pytest
from rlm_fixtures import Harnessed, agent_with_workspace, note_edit, skill_edit
from runtime_helpers import run_in_kernel

from ph.cordis import Context
from ph.keys import APPROVAL, COMMANDS, SESSIONS, SYSTEM_PROMPT
from ph.paths import write_atomic
from ph.persistence import append_records
from ph.seams.code_runtime import CodeRunRequest, CodeRunResult, CodeRuntimeSeam
from ph.session import SessionStore
from ph.system_prompt import (
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from ph.testing import assert_fold_laws, not_none, raising, stored_events, stored_types
from ph_rlm.harness import (
    GLOBAL_LOG_NAME,
    PROJECTION_NAME,
    REFINED,
    HarnessEdit,
    HarnessEntry,
    HarnessState,
    RefinementProposal,
    RefinementRefused,
    fold_events,
    fold_session,
    read_global_events,
    render_state,
)
from ph_rlm.harness.state import extend_session
from ph_rlm.keys import HARNESS

pytestmark = pytest.mark.anyio


def _allow(ctx: Context) -> list[str]:
    """Answer approval prompts, and record what was asked."""
    asked: list[str] = []

    async def answerer(request: Any, _next: Any) -> str:  # noqa: ANN401
        asked.append(request.tool_name)
        return "allowed-once"

    ctx.require(APPROVAL).register_answerer(answerer)
    return asked


async def _refined(harnessed: Harnessed) -> tuple[Any, Any, Any]:
    """A harness that has taken two refinements: a create, then an update and a create.

    There is one definition because two tests need the same shape of log. A
    second copy would keep passing against a log that the service has stopped
    writing.
    """
    ctx, session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="one", edits=[note_edit("first")]), agent=agent
    )
    await ctx.require(HARNESS).apply(
        RefinementProposal(
            summary="two",
            edits=[
                HarnessEdit(
                    action="update", kind="note", id="first", title="revised", content="v2"
                ),
                note_edit("second"),
            ],
        ),
        agent=agent,
    )
    return ctx, session, agent


# --------------------------------------------------------------- the fold --


async def test_a_sessions_merged_state_goes_with_the_session(harnessed: Harnessed) -> None:
    """The per-session memo that `state()` reads through is released with the session.

    `_merged` used to keep one `(local, global, merged)` triple for each session
    that the mount ever asked about, and the fold cache of the service used to
    keep one fold, for the life of the mount.

    Sabotage: make `_merged` a plain dict, and the memo stays.
    """
    ctx, session, _agent = await harnessed()
    harness = ctx.require(HARNESS)
    harness.state(session)
    assert session in harness._merged

    ctx.require(SESSIONS).dispose(session.id)

    assert session not in harness._merged


async def test_an_applied_refinement_is_the_state(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    record = await ctx.require(HARNESS).apply(
        RefinementProposal(summary="learned something", edits=[note_edit("prefer-uv")]),
        agent=agent,
    )

    assert [event.type for event in session.events if event.type.startswith("harness/")] == [
        REFINED
    ]
    entry = ctx.require(HARNESS).state(session).entry("note", "prefer-uv")
    assert entry is not None
    assert (entry.title, entry.version, entry.scope) == ("a thing learned", 1, "local")
    assert [one.refine_id for one in ctx.require(HARNESS).state(session).refinements] == [
        record.refine_id
    ]


async def test_deleting_the_state_and_re_deriving_is_byte_identical(harnessed: Harnessed) -> None:
    """The gate of the plan, and the whole D14 claim: the log *is* the state.

    The assertion compares against a cold fold and not the cached one, so the
    cache cannot be what makes them agree.
    """
    ctx, session, _ = await _refined(harnessed)

    assert fold_session(session).to_wire() == ctx.require(HARNESS).local(session).to_wire()
    # An update also bumped the version and did not replace the history.
    revised = fold_session(session).entry("note", "first")
    assert revised is not None
    assert (revised.version, revised.title) == (2, "revised")


async def test_the_harness_fold_obeys_the_fold_laws(harnessed: Harnessed) -> None:
    """`extend_session` is `fold_session` resumed from a prefix, and the cache
    relies on the two being the same fold. The test checks every prefix of a log
    that the real service wrote: a create, then an update and a second create in
    one refinement. It does not assume the law from the fact that the two
    functions share a step."""
    _, session, _ = await _refined(harnessed)

    assert_fold_laws(session, fold_session, extend_session)


async def test_a_fork_inherits_the_harness_as_of_its_boundary(harnessed: Harnessed) -> None:
    """What a side file could not express, and the reason that this is a fold.

    A file gives every fork the latest state of the parent. A fold of only the
    events at or before the boundary gives the fork the harness that existed
    then.
    """
    ctx, session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="before", edits=[note_edit("early")]),
        agent=agent,
    )
    boundary = session.events[-1].seq
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="after", edits=[note_edit("late")]), agent=agent
    )

    assert set(fold_session(session).entries["note"]) == {"early", "late"}
    forked = ctx.require(SESSIONS).fork(session, boundary, child_session_id="forked")
    assert set(fold_session(forked).entries["note"]) == {"early"}
    # Nothing was copied and no state was handed over: the seeded prefix folds.
    assert ctx.require(HARNESS).local(forked).entry("note", "late") is None


def test_a_record_from_a_newer_build_does_not_break_the_fold() -> None:
    """One unreadable refinement must not cost the rest of the harness."""
    state = fold_events(
        [
            {"invented": True},
            {
                "refineId": "r1",
                "scope": "local",
                "summary": "fine",
                "appliedEdits": [
                    {
                        "action": "create",
                        "kind": "note",
                        "id": "kept",
                        "after": {"kind": "note", "id": "kept", "title": "t", "content": "c"},
                    }
                ],
            },
        ]
    )
    assert set(state.entries["note"]) == {"kept"}
    assert [one.refine_id for one in state.refinements] == ["r1"]


# ------------------------------------------------------------ validation --


async def test_h5_the_doctrine_is_not_editable(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="not editable"):
        await ctx.require(HARNESS).apply(
            RefinementProposal(
                summary="rewrite everything",
                edits=[note_edit("base_system_prompt", "a new doctrine")],
            ),
            agent=agent,
        )
    assert [event for event in session.events if event.type == REFINED] == []


async def test_h1_an_unresolvable_reference_is_rejected_on_the_event(harnessed: Harnessed) -> None:
    """`/refine` cannot create capability (I7), and the refusal can be audited.

    The reference is resolved in the runtime that the model uses, so an entry
    is true only if the kernel can reach it. The rejection is recorded beside
    the edits that did apply, and is not dropped.
    """
    ctx, session, agent = await harnessed()
    record = await ctx.require(HARNESS).apply(
        RefinementProposal(
            summary="mixed",
            edits=[
                note_edit("this-one-is-fine"),
                skill_edit("imaginary", "ph_nonexistent_module", "nope"),
            ],
        ),
        agent=agent,
    )

    assert [edit.id for edit in record.applied_edits] == ["this-one-is-fine"]
    assert any("imaginary" in reason and "does not resolve" in reason for reason in record.rejected)
    # On the event, so a reader of the log can see what was refused and why.
    logged = next(event for event in session.events if event.type == REFINED)
    assert list(logged.data["rejected"]) == record.rejected
    assert ctx.require(HARNESS).state(session).entry("skill", "imaginary") is None


def _skill(entry_id: str, module: str, name: str) -> RefinementProposal:
    """A proposal of one skill entry naming `module.name`."""
    return RefinementProposal(summary=f"use {module}", edits=[skill_edit(entry_id, module, name)])


def _on_each_run(monkeypatch: pytest.MonkeyPatch, note: Callable[[CodeRunRequest], None]) -> None:
    """Call `note` with every request to the code runtime, before the runtime runs it."""
    real = CodeRuntimeSeam.run

    async def spy(self: CodeRuntimeSeam, request: CodeRunRequest) -> CodeRunResult:
        note(request)
        return await real(self, request)

    monkeypatch.setattr(CodeRuntimeSeam, "run", spy)


async def test_h1_probes_within_the_agents_boundary_beside_its_kernel(
    harnessed: Harnessed, tmp_path: Path
) -> None:
    """S21. The probe imports a module that the model named, and an import runs
    the top level of the module. So the probe runs within the boundary of the
    agent that proposed it: the workspace of the agent and its confinement. From
    a namespace that no agent owned, the probe had neither of those, and it ran
    in the directory of the daemon. The probe also runs beside the working
    kernel of the agent, not in it. A module that hangs or crashes on import
    costs the kernel of the probe, and nothing that it imports reaches the
    namespace or the snapshots of the model.

    The test shows this with a module that only the workspace of the agent
    holds. The probe resolves it, and the kernel of the model has not imported
    it.

    Sabotage: build the probe's kernel from its namespace rather than its agent, and
    it does not resolve; probe in the agent's own namespace, and the model's kernel
    has imported it.
    """
    ctx, session, agent = await harnessed()
    workspace = await agent_with_workspace(ctx, session, agent, tmp_path / "project")
    (workspace.root / "ph_only_here.py").write_text("def here():\n    return 1\n", encoding="utf-8")

    record = await ctx.require(HARNESS).apply(
        _skill("local-helper", "ph_only_here", "here"), agent=agent
    )

    assert [edit.id for edit in record.applied_edits] == ["local-helper"], record.rejected
    looked = await run_in_kernel(ctx, agent.id, "import sys; print('ph_only_here' in sys.modules)")
    assert looked.logs.strip() == "False", "the probe ran in the model's own kernel"


async def test_h1_a_probe_is_on_disk_before_it_runs(
    harnessed: Harnessed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S21. Auto-refine probes with nobody asking. So the log says what was about
    to run before the kernel runs it, and then what the probe found. Both records
    are in the log of the agent.

    Sabotage: run the probe outside `intents.claim`, and nothing is on disk.
    """
    ctx, session, agent = await harnessed()
    stored_at_run: list[list[str]] = []
    _on_each_run(monkeypatch, lambda _request: stored_at_run.append(stored_types(ctx, session.id)))

    record = await ctx.require(HARNESS).apply(_skill("finding-files", "glob", "glob"), agent=agent)

    assert stored_at_run and "harness/probe" in stored_at_run[0], "the kernel ran first"
    probe = next(event for event in session.events if event.type == "harness/probe")
    probed = next(event for event in session.events if event.type == "harness/probed")
    assert probe.data["refineId"] == record.refine_id
    assert (probe.data["module"], probe.data["callable"]) == ("glob", "glob")
    assert (probed.data["probeSeq"], probed.data["unresolved"]) == (probe.seq, None)


async def test_h1_a_probe_the_log_cannot_record_does_not_run(
    harnessed: Harnessed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """S21. The probe fails closed, as a durable intent does: with no record on
    disk, there is no import. The entry that the probe was checking is refused,
    because nothing saw it resolve.

    Sabotage: make `HARNESS_PROBE`'s barrier `buffered`, and the kernel runs.
    """
    ctx, _session, agent = await harnessed()
    ran: list[str] = []
    _on_each_run(monkeypatch, lambda request: ran.append(request.program))
    monkeypatch.setattr(SessionStore, "flush", raising(OSError("read-only file system")))

    with pytest.raises(RefinementRefused, match="could not be recorded, so it did not run"):
        await ctx.require(HARNESS).apply(_skill("finding-files", "glob", "glob"), agent=agent)
    assert ran == [], "the probe ran with no record of it on disk"


async def test_h1_a_skill_without_a_reference_teaches_nothing(harnessed: Harnessed) -> None:
    ctx, _session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="no reference"):
        await ctx.require(HARNESS).apply(
            RefinementProposal(
                summary="vague",
                edits=[
                    HarnessEdit(
                        action="create", kind="skill", id="hand-waving", title="do it", content="?"
                    )
                ],
            ),
            agent=agent,
        )


async def test_h2_a_skill_for_a_bound_tool_renders_the_binding_form(harnessed: Harnessed) -> None:
    """The gate of the plan: an entry that names a bound tool renders `await tools.…`.

    The pattern is rendered and is not accepted from the proposal, so a
    refinement cannot write prompt text that steers the model onto the
    ungoverned raw-namespace path.
    """
    ctx, session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(
            summary="how to search", edits=[skill_edit("finding-files", "glob", "glob")]
        ),
        agent=agent,
    )
    entry = ctx.require(HARNESS).state(session).entry("skill", "finding-files")
    assert entry is not None
    assert entry.call_pattern == "await tools.glob(...)"


async def test_h2_an_unbound_reference_does_not_claim_to_be_a_binding(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="plain python", edits=[skill_edit("dumping", "json", "dumps")]),
        agent=agent,
    )
    entry = ctx.require(HARNESS).state(session).entry("skill", "dumping")
    assert entry is not None
    assert entry.call_pattern == "json.dumps(...)"
    assert "tools." not in entry.call_pattern


async def test_deleting_an_entry_that_is_not_there_is_refused(harnessed: Harnessed) -> None:
    ctx, _session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="no such note"):
        await ctx.require(HARNESS).apply(
            RefinementProposal(
                summary="tidying", edits=[HarnessEdit(action="delete", kind="note", id="ghost")]
            ),
            agent=agent,
        )


# ------------------------------------------------------------------- H3 --


async def test_h3_a_canceled_turn_is_not_prompted_for_a_global_edit(
    harnessed: Harnessed,
) -> None:
    """H3 asks the user, so a turn that the user already stopped is not asked.

    `ApprovalService.request` uses the token of the agent as its default
    cancellation. Every prompt therefore inherits the rule, and no caller has
    to restate it.
    """
    ctx, _session, agent = await harnessed()
    asked = _allow(ctx)
    agent.signal.cancel("the user stopped the turn")

    with pytest.raises(RefinementRefused):
        await ctx.require(HARNESS).apply(
            RefinementProposal(summary="everyone", edits=[note_edit("shared")]),
            scope="global",
            agent=agent,
        )
    assert asked == [], "a canceled turn must not reach the answerer"


async def test_h3_a_global_edit_prompts_and_a_local_one_does_not(harnessed: Harnessed) -> None:
    """A global entry is injected into every future session, including other
    projects, so the user is asked. A local entry concerns only this session."""
    ctx, session, agent = await harnessed()
    asked = _allow(ctx)

    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="local", edits=[note_edit("mine")]), agent=agent
    )
    assert asked == [], "a local refinement asked for approval"

    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="everyone", edits=[note_edit("shared")]),
        scope="global",
        agent=agent,
    )
    assert asked == ["refine"]
    # Both halves are in the log, so the answer is as durable as the refinement.
    assert [event.type for event in session.events if event.type.startswith("approval/")] == [
        "approval/asked",
        "approval/decided",
    ]


async def test_h3_a_declined_global_edit_writes_nothing(harnessed: Harnessed) -> None:
    ctx, _session, agent = await harnessed()

    async def refuse(_request: object, _next: object) -> str:
        return "rejected"

    ctx.require(APPROVAL).register_answerer(refuse)

    with pytest.raises(RefinementRefused, match="not approved"):
        await ctx.require(HARNESS).apply(
            RefinementProposal(summary="everyone", edits=[note_edit("shared")]),
            scope="global",
            agent=agent,
        )
    assert read_global_events(ctx.require(HARNESS).directory) == []


async def test_h3_fails_closed_with_nowhere_to_ask(harnessed: Harnessed) -> None:
    """B3: no answerer is not consent."""
    ctx, _session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="not approved"):
        await ctx.require(HARNESS).apply(
            RefinementProposal(summary="everyone", edits=[note_edit("shared")]),
            scope="global",
            agent=agent,
        )
    assert read_global_events(ctx.require(HARNESS).directory) == []


# --------------------------------------------------------------- global --


async def test_a_global_refinement_folds_from_its_own_log(harnessed: Harnessed) -> None:
    """The global scope is a log too, so "state is a fold" holds at both."""
    ctx, session, agent = await harnessed()
    _allow(ctx)
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="deployment-wide", edits=[note_edit("house-style")]),
        scope="global",
        agent=agent,
    )

    assert (ctx.require(HARNESS).directory / GLOBAL_LOG_NAME).exists()
    assert len(read_global_events(ctx.require(HARNESS).directory)) == 1
    # The record is not in the log of the session. That is why it outlives this
    # session.
    assert [event for event in session.events if event.type == REFINED] == []
    world = ctx.require(HARNESS).globals().entry("note", "house-style")
    assert world is not None and world.scope == "global"
    # And the model is shown local layered over global.
    assert ctx.require(HARNESS).state(session).entry("note", "house-style") is not None


async def test_a_global_edit_is_written_only_after_the_approval_that_allowed_it(
    harnessed: Harnessed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L4. The log of the approving session is on disk before the global log is
    appended, and the global record names the approval.

    Every future session reads the global log. Its record used to be appended
    while the `approval/decided` of the session was still in memory. A crash in
    between left a deployment-wide edit in force, and repair settled its
    approval as outcome unknown. The result was an edit with no record that
    anyone allowed it.

    The test asks the store at the moment when the global log is appended,
    through the durable writer that the append goes through.

    Sabotage: drop the session write in `_commit` and the stored session has no
    decision when the edit is written; drop `call_id` from the ask and the decision
    no longer names the record.
    """
    ctx, _session, agent = await harnessed()
    _allow(ctx)
    approving = not_none(agent.session)
    decided_then: list[Any] = []

    def probe(path: Path, records: list[Any]) -> None:
        decided_then.extend(
            event.data.get("callId")
            for event in stored_events(ctx, approving.id)
            if event.type == "approval/decided"
        )
        append_records(path, records)

    monkeypatch.setattr("ph_rlm.harness.service.append_records", probe)
    record = await ctx.require(HARNESS).apply(
        RefinementProposal(summary="deployment-wide", edits=[note_edit("house-style")]),
        scope="global",
        agent=agent,
    )

    assert decided_then == [record.refine_id], (
        "the approval that allowed this edit was not on disk when the edit was"
    )
    assert read_global_events(ctx.require(HARNESS).directory)[-1]["approvedIn"] == approving.id


async def test_a_global_edit_whose_approval_cannot_be_written_is_refused(
    harnessed: Harnessed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """L4, failing closed. When the approval is not written, the edit is not made.

    If the edit were written anyway, it would be the same edit in force with no
    record of who allowed it. The ordering closes that gap.
    """
    ctx, _session, agent = await harnessed()
    _allow(ctx)

    async def unwritten(_ctx: Context, _session: Any) -> bool:  # noqa: ANN401
        return False

    monkeypatch.setattr("ph_rlm.harness.service.session_written", unwritten)
    with pytest.raises(RefinementRefused, match="could not be written"):
        await ctx.require(HARNESS).apply(
            RefinementProposal(summary="deployment-wide", edits=[note_edit("house-style")]),
            scope="global",
            agent=agent,
        )
    assert read_global_events(ctx.require(HARNESS).directory) == []


async def test_a_local_refinement_is_on_disk_before_its_projection(
    harnessed: Harnessed, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local half of L4. The `harness/refined` of the session is written
    before the projection that restates it, so the file on disk is never ahead
    of the log.

    Sabotage: drop the session write in `_commit` and the projection is written
    while the stored log has no record of the refinement.
    """
    ctx, session, agent = await harnessed()
    refined_then: list[bool] = []

    def probe(path: Path, payload: str, *, durable: bool = True) -> None:
        refined_then.append(any(event.type == REFINED for event in stored_events(ctx, session.id)))
        write_atomic(path, payload, durable=durable)

    monkeypatch.setattr("ph_rlm.harness.service.write_atomic", probe)
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="learned", edits=[note_edit("local")]),
        agent=agent,
    )

    assert refined_then == [True], "the projection was written ahead of the log"


async def test_a_local_entry_shadows_a_global_one(harnessed: Harnessed) -> None:
    """This session learned something specific. A deployment-wide note does not
    overrule it."""
    ctx, session, agent = await harnessed()
    _allow(ctx)
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="everywhere", edits=[note_edit("style", "the house rule")]),
        scope="global",
        agent=agent,
    )
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="here", edits=[note_edit("style", "what this repo does")]),
        agent=agent,
    )
    entry = ctx.require(HARNESS).state(session).entry("note", "style")
    assert entry is not None
    assert entry.title == "what this repo does"


async def test_concurrent_global_writes_are_both_recorded(harnessed: Harnessed) -> None:
    """The gate of the plan. Concurrent sessions share this log, so the lock
    makes eight refinements eight records and not one torn line."""
    ctx, _session, agent = await harnessed()
    _allow(ctx)

    async def write(index: int) -> None:
        await ctx.require(HARNESS).apply(
            RefinementProposal(summary=f"n{index}", edits=[note_edit(f"entry-{index}")]),
            scope="global",
            agent=agent,
        )

    async with anyio.create_task_group() as tasks:
        for index in range(8):
            tasks.start_soon(write, index)

    assert len(read_global_events(ctx.require(HARNESS).directory)) == 8, (
        "an append was lost or torn"
    )
    assert set(ctx.require(HARNESS).globals().entries["note"]) == {
        f"entry-{index}" for index in range(8)
    }


# ------------------------------------------------------------- rollback --


async def test_h6_rollback_restores_the_fold(harnessed: Harnessed) -> None:
    """The gate of the plan. A rollback is derivable because each apply recorded
    what it replaced."""
    ctx, session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="original", edits=[note_edit("kept", "the first title")]),
        agent=agent,
    )
    before = ctx.require(HARNESS).state(session).to_wire()

    bad = await ctx.require(HARNESS).apply(
        RefinementProposal(
            summary="a regrettable change",
            edits=[
                HarnessEdit(action="update", kind="note", id="kept", title="worse", content="x"),
                note_edit("also-added"),
            ],
        ),
        agent=agent,
    )
    changed = ctx.require(HARNESS).state(session).entry("note", "kept")
    assert changed is not None and changed.title == "worse"

    await ctx.require(HARNESS).rollback(bad.refine_id, session=session, agent=agent)
    after = ctx.require(HARNESS).state(session)
    restored = after.entry("note", "kept")
    assert restored is not None and restored.title == "the first title"
    assert after.entry("note", "also-added") is None
    # The entries are the same byte for byte, version included. An inverse that
    # left a v3 of a v1 entry would not be an inverse.
    assert after.entries == HarnessState.model_validate(before).entries
    # The entries are back. The *history* is longer, because a rollback is
    # itself a refinement and an append-only log does not forget.
    assert len(after.refinements) == 3
    assert after.refinements[-1].rollback_of == bad.refine_id


async def test_h6_a_refinement_is_not_rolled_back_twice(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    record = await ctx.require(HARNESS).apply(
        RefinementProposal(summary="one", edits=[note_edit("thing")]), agent=agent
    )
    await ctx.require(HARNESS).rollback(record.refine_id, session=session, agent=agent)
    with pytest.raises(RefinementRefused, match="already been rolled back"):
        await ctx.require(HARNESS).rollback(record.refine_id, session=session, agent=agent)


async def test_h6_rolling_back_a_global_refinement_asks_too(harnessed: Harnessed) -> None:
    """Undoing a deployment-wide entry changes every future session exactly as
    writing one does, so it goes through the same gate."""
    ctx, session, agent = await harnessed()
    asked = _allow(ctx)
    record = await ctx.require(HARNESS).apply(
        RefinementProposal(summary="everyone", edits=[note_edit("shared")]),
        scope="global",
        agent=agent,
    )

    await ctx.require(HARNESS).rollback(record.refine_id, session=session, agent=agent)
    assert asked == ["refine", "refine"]
    assert ctx.require(HARNESS).globals().entry("note", "shared") is None
    # The rollback is in the global log, not in the log of this session. A
    # rollback belongs to the scope that it undoes. Otherwise, reopening the
    # session would bring the entry back.
    assert len(read_global_events(ctx.require(HARNESS).directory)) == 2
    assert [event for event in session.events if event.type == REFINED] == []


async def test_the_command_reports_and_rolls_back(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    record = await ctx.require(HARNESS).apply(
        RefinementProposal(summary="one", edits=[note_edit("thing")]), agent=agent
    )

    shown = await ctx.require(COMMANDS).dispatch("/refine --show", session=session, agent=agent)
    assert shown is not None and "[local:thing] a thing learned" in shown
    assert record.summary in shown

    rolled = await ctx.require(COMMANDS).dispatch(
        f"/refine --rollback {record.refine_id}", session=session, agent=agent
    )
    assert rolled is not None and record.refine_id in rolled
    assert ctx.require(HARNESS).state(session).entry("note", "thing") is None
    # Recorded as something the *user* did, not as a model turn.
    assert [event.type for event in session.events if event.type.startswith("command/")] == [
        "command/run",
        "command/done",
        "command/run",
        "command/done",
    ]
    assert [event.type for event in session.events if event.type.startswith("turn/")] == []


async def test_the_command_refuses_an_unknown_id(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    shown = await ctx.require(COMMANDS).dispatch(
        "/refine --rollback nope", session=session, agent=agent
    )
    assert shown is not None and "refusing" in shown and "nope" in shown


# ------------------------------------------------------- prompt and file --


async def test_the_harness_reaches_the_model_as_a_snapshot(harnessed: Harnessed) -> None:
    """A12: a refinement changes this text in the middle of a session, so the text
    must not be in the cached prefix. There, every apply would bill the whole
    prompt again."""
    ctx, _session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(
            summary="learned how to run them",
            edits=[
                HarnessEdit(
                    action="create",
                    kind="procedure",
                    id="run-the-tests",
                    title="running the tests",
                    content="uv run pytest",
                    path="docs/testing.md",
                )
            ],
        ),
        agent=agent,
    )
    assembly = await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)
    snapshot = join_context_sections(render_context_sections(assembly))

    assert "# Continual Harness State" in snapshot
    assert "[local:run-the-tests] running the tests (v1, docs/testing.md)" in snapshot
    assert "## Recent refinements" in snapshot
    assert "Continual Harness" not in render_prompt(assembly)


def test_an_empty_harness_renders_nothing() -> None:
    """A heading with nothing under it costs tokens to say nothing."""
    assert render_state(HarnessState(), per_kind=5, refinements=5) == ""


def test_the_prompt_section_is_bounded_and_stable() -> None:
    """The section is bounded, because a harness that grew without a bound would
    be the whole prompt. It is ordered by id, so the same state renders the same
    bytes (A12)."""
    entries = {
        f"n{index:02d}": HarnessEntry(kind="note", id=f"n{index:02d}", title="t", content="c")
        for index in range(20)
    }
    text = render_state(HarnessState(entries={"note": entries}), per_kind=3, refinements=5)

    assert text.count("- [local:") == 3
    assert "and 17 more" in text
    reordered = HarnessState(entries={"note": dict(reversed(list(entries.items())))})
    assert render_state(reordered, per_kind=3, refinements=5) == text


async def test_the_projection_is_written_and_equals_the_fold(harnessed: Harnessed) -> None:
    """The invariant that keeps a projection a projection.

    `harness_state.json` is for humans, the trajectory viewer and export. Nothing
    reads it back. If it can drift, something has started to treat it as state.
    """
    ctx, session, agent = await harnessed()
    await ctx.require(HARNESS).apply(
        RefinementProposal(summary="one", edits=[note_edit("thing")]), agent=agent
    )

    path = ctx.require(HARNESS).projection_path(session)
    assert path.exists()
    # The path is per session, because the projection is of local layered over
    # global. With one shared path, two sessions would each overwrite the file
    # of the other.
    assert path == ctx.require(HARNESS).directory / session.id / PROJECTION_NAME
    assert ctx.require(HARNESS).verify_projection(session) is True

    # Deleting it loses nothing: the state is the log.
    path.unlink()
    assert ctx.require(HARNESS).state(session).entry("note", "thing") is not None
    assert ctx.require(HARNESS).verify_projection(session) is False
    await ctx.require(HARNESS).write_projection(session)
    assert ctx.require(HARNESS).verify_projection(session) is True


async def test_the_row_mounts_without_the_delegation_rows(harnessed: Harnessed) -> None:
    """The harness is not a delegation feature: `harnessed` mounts neither
    `rlm-subagent-provider` nor `rlm-prompt`, and `/refine` still works."""
    ctx, session, _agent = await harnessed()
    assert ctx.require(HARNESS).state(session).entries == {}
    assert ctx.require(COMMANDS).get("refine") is not None
