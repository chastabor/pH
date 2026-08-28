"""The Continual Harness: state as a fold, and the checks on writing it (P3-16).

The plan's gates for this row, one test each: *delete-and-re-derive
byte-identical; concurrent global writes safe; a missing import rejected with the
failure on the event; an `edit` entry renders the binding form; a global edit
prompts; rollback restores the fold.*

The claim worth stating once is why it is a fold at all (D14). Prime Agent keeps
this in a JSON file; a file cannot express "the harness as of a fork boundary",
and it needs an mtime guard and a conflict rule that an append-only log does not.

The real kernel is mounted throughout, because H1's probe resolves references in
the runtime the model actually uses — checking them against this process would
prove something about the harness's imports rather than the model's.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import anyio
import pytest
from conftest import HARNESS_ROW

from ph.system_prompt import (
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from ph.system_prompt.assembly import AssembleContext
from ph_rlm.harness import (
    GLOBAL_LOG_NAME,
    PROJECTION_NAME,
    REFINED,
    HarnessEdit,
    HarnessEntry,
    HarnessReference,
    HarnessState,
    RefinementProposal,
    RefinementRefused,
    fold_events,
    fold_session,
    read_global_events,
    render_state,
)

pytestmark = pytest.mark.anyio

Harnessed = Callable[[], Any]


@pytest.fixture
def harnessed(mounted_runtime: Any) -> Harnessed:
    """`await harnessed()` -> `(ctx, session, agent)` with the harness mounted.

    On the real runtime, so H1 probes a live kernel, and with `$PH_HOME` under
    `tmp_path` (the root `mount` fixture), so the global log is this test's.
    """

    async def build() -> tuple[Any, Any, Any]:
        return await mounted_runtime(session_id="harness", extra_rows=[HARNESS_ROW])

    return build


def _note(entry_id: str, title: str = "a thing learned") -> HarnessEdit:
    return HarnessEdit(action="create", kind="note", id=entry_id, title=title, content="body")


def _allow(ctx: Any) -> list[str]:
    """Answer approval prompts, and record what was asked."""
    asked: list[str] = []

    async def answerer(request: Any, _next: Any) -> str:
        asked.append(request.tool_name)
        return "allowed-once"

    ctx.approval.register_answerer(answerer)
    return asked


# --------------------------------------------------------------- the fold --


async def test_an_applied_refinement_is_the_state(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    record = await ctx.harness.apply(
        RefinementProposal(summary="learned something", edits=[_note("prefer-uv")]),
        session=session,
        agent=agent,
    )

    assert [event.type for event in session.events if event.type.startswith("harness/")] == [
        REFINED
    ]
    entry = ctx.harness.state(session).entry("note", "prefer-uv")
    assert entry is not None
    assert (entry.title, entry.version, entry.scope) == ("a thing learned", 1, "local")
    assert [one.refine_id for one in ctx.harness.state(session).refinements] == [record.refine_id]


async def test_deleting_the_state_and_re_deriving_is_byte_identical(harnessed: Harnessed) -> None:
    """The plan's gate, and the whole D14 claim: the log *is* the state.

    Asserted against a cold fold rather than the cached one, so the cache cannot
    be what makes them agree.
    """
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
        RefinementProposal(summary="one", edits=[_note("first")]), session=session, agent=agent
    )
    await ctx.harness.apply(
        RefinementProposal(
            summary="two",
            edits=[
                HarnessEdit(
                    action="update", kind="note", id="first", title="revised", content="v2"
                ),
                _note("second"),
            ],
        ),
        session=session,
        agent=agent,
    )

    assert fold_session(session).to_wire() == ctx.harness.local(session).to_wire()
    # And an update bumped the version rather than replacing history.
    revised = fold_session(session).entry("note", "first")
    assert revised is not None
    assert (revised.version, revised.title) == (2, "revised")


async def test_a_fork_inherits_the_harness_as_of_its_boundary(harnessed: Harnessed) -> None:
    """What a side file could not express, and the reason this is a fold.

    A file hands every fork the parent's latest; folding only the events at or
    before the boundary hands it the harness that existed then.
    """
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
        RefinementProposal(summary="before", edits=[_note("early")]), session=session, agent=agent
    )
    boundary = session.events[-1].seq
    await ctx.harness.apply(
        RefinementProposal(summary="after", edits=[_note("late")]), session=session, agent=agent
    )

    assert set(fold_session(session).entries["note"]) == {"early", "late"}
    forked = ctx.sessions.fork(session, boundary, child_session_id="forked")
    assert set(fold_session(forked).entries["note"]) == {"early"}
    # Nothing was copied and no state was handed over: the seeded prefix folds.
    assert ctx.harness.local(forked).entry("note", "late") is None


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
        await ctx.harness.apply(
            RefinementProposal(
                summary="rewrite everything",
                edits=[_note("base_system_prompt", "a new doctrine")],
            ),
            session=session,
            agent=agent,
        )
    assert [event for event in session.events if event.type == REFINED] == []


async def test_h1_an_unresolvable_reference_is_rejected_on_the_event(harnessed: Harnessed) -> None:
    """`/refine` cannot conjure capability (I7), and the refusal is auditable.

    Resolved in the runtime the model uses, so an entry is only true if the
    kernel can reach it — and the rejection is recorded beside the edits that
    did apply rather than dropped.
    """
    ctx, session, agent = await harnessed()
    record = await ctx.harness.apply(
        RefinementProposal(
            summary="mixed",
            edits=[
                _note("this-one-is-fine"),
                HarnessEdit(
                    action="create",
                    kind="skill",
                    id="imaginary",
                    title="a skill that does not exist",
                    content="call it",
                    reference=HarnessReference(module="ph_nonexistent_module", callable="nope"),
                ),
            ],
        ),
        session=session,
        agent=agent,
    )

    assert [edit.id for edit in record.applied_edits] == ["this-one-is-fine"]
    assert any("imaginary" in reason and "does not resolve" in reason for reason in record.rejected)
    # On the event, so a reader of the log can see what was refused and why.
    logged = next(event for event in session.events if event.type == REFINED)
    assert list(logged.data["rejected"]) == record.rejected
    assert ctx.harness.state(session).entry("skill", "imaginary") is None


async def test_h1_a_skill_without_a_reference_teaches_nothing(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="no reference"):
        await ctx.harness.apply(
            RefinementProposal(
                summary="vague",
                edits=[
                    HarnessEdit(
                        action="create", kind="skill", id="hand-waving", title="do it", content="?"
                    )
                ],
            ),
            session=session,
            agent=agent,
        )


async def test_h2_a_skill_for_a_bound_tool_renders_the_binding_form(harnessed: Harnessed) -> None:
    """The plan's gate: an entry naming a bound tool renders `await tools.…`.

    Rendered rather than accepted from the proposal, so a refinement cannot write
    prompt text steering the model onto the ungoverned raw-namespace path.
    """
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
        RefinementProposal(
            summary="how to search",
            edits=[
                HarnessEdit(
                    action="create",
                    kind="skill",
                    id="finding-files",
                    title="finding files by name",
                    content="match a pattern",
                    reference=HarnessReference(module="glob", callable="glob"),
                )
            ],
        ),
        session=session,
        agent=agent,
    )
    entry = ctx.harness.state(session).entry("skill", "finding-files")
    assert entry is not None
    assert entry.call_pattern == "await tools.glob(...)"


async def test_h2_an_unbound_reference_does_not_claim_to_be_a_binding(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
        RefinementProposal(
            summary="plain python",
            edits=[
                HarnessEdit(
                    action="create",
                    kind="skill",
                    id="dumping",
                    title="dumping json",
                    content="use it",
                    reference=HarnessReference(module="json", callable="dumps"),
                )
            ],
        ),
        session=session,
        agent=agent,
    )
    entry = ctx.harness.state(session).entry("skill", "dumping")
    assert entry is not None
    assert entry.call_pattern == "json.dumps(...)"
    assert "tools." not in entry.call_pattern


async def test_deleting_an_entry_that_is_not_there_is_refused(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="no such note"):
        await ctx.harness.apply(
            RefinementProposal(
                summary="tidying", edits=[HarnessEdit(action="delete", kind="note", id="ghost")]
            ),
            session=session,
            agent=agent,
        )


# ------------------------------------------------------------------- H3 --


async def test_h3_a_global_edit_prompts_and_a_local_one_does_not(harnessed: Harnessed) -> None:
    """A global entry is injected into every future session, including other
    projects, so the human is asked. A local one is this session's business."""
    ctx, session, agent = await harnessed()
    asked = _allow(ctx)

    await ctx.harness.apply(
        RefinementProposal(summary="local", edits=[_note("mine")]), session=session, agent=agent
    )
    assert asked == [], "a local refinement asked for approval"

    await ctx.harness.apply(
        RefinementProposal(summary="everyone", edits=[_note("shared")]),
        scope="global",
        session=session,
        agent=agent,
    )
    assert asked == ["refine"]
    # Both halves in the log, so the answer is as durable as the refinement.
    assert [event.type for event in session.events if event.type.startswith("approval/")] == [
        "approval/asked",
        "approval/decided",
    ]


async def test_h3_a_declined_global_edit_writes_nothing(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()

    async def refuse(_request: Any, _next: Any) -> str:
        return "rejected"

    ctx.approval.register_answerer(refuse)

    with pytest.raises(RefinementRefused, match="not approved"):
        await ctx.harness.apply(
            RefinementProposal(summary="everyone", edits=[_note("shared")]),
            scope="global",
            session=session,
            agent=agent,
        )
    assert read_global_events(ctx.harness.directory) == []


async def test_h3_fails_closed_with_nowhere_to_ask(harnessed: Harnessed) -> None:
    """B3: no answerer is not consent."""
    ctx, session, agent = await harnessed()
    with pytest.raises(RefinementRefused, match="not approved"):
        await ctx.harness.apply(
            RefinementProposal(summary="everyone", edits=[_note("shared")]),
            scope="global",
            session=session,
            agent=agent,
        )
    assert read_global_events(ctx.harness.directory) == []


# --------------------------------------------------------------- global --


async def test_a_global_refinement_folds_from_its_own_log(harnessed: Harnessed) -> None:
    """The global scope is a log too, so "state is a fold" holds at both."""
    ctx, session, agent = await harnessed()
    _allow(ctx)
    await ctx.harness.apply(
        RefinementProposal(summary="deployment-wide", edits=[_note("house-style")]),
        scope="global",
        session=session,
        agent=agent,
    )

    assert (ctx.harness.directory / GLOBAL_LOG_NAME).exists()
    assert len(read_global_events(ctx.harness.directory)) == 1
    # Not in the session's log — that is what makes it outlive this session.
    assert [event for event in session.events if event.type == REFINED] == []
    world = ctx.harness.globals().entry("note", "house-style")
    assert world is not None and world.scope == "global"
    # And the model is shown local layered over global.
    assert ctx.harness.state(session).entry("note", "house-style") is not None


async def test_a_local_entry_shadows_a_global_one(harnessed: Harnessed) -> None:
    """This session learned something specific; a deployment-wide note does not
    get to overrule it."""
    ctx, session, agent = await harnessed()
    _allow(ctx)
    await ctx.harness.apply(
        RefinementProposal(summary="everywhere", edits=[_note("style", "the house rule")]),
        scope="global",
        session=session,
        agent=agent,
    )
    await ctx.harness.apply(
        RefinementProposal(summary="here", edits=[_note("style", "what this repo does")]),
        session=session,
        agent=agent,
    )
    entry = ctx.harness.state(session).entry("note", "style")
    assert entry is not None
    assert entry.title == "what this repo does"


async def test_concurrent_global_writes_are_both_recorded(harnessed: Harnessed) -> None:
    """The plan's gate. Concurrent sessions share this log, so the lock is what
    makes eight refinements eight records rather than one torn line."""
    ctx, session, agent = await harnessed()
    _allow(ctx)

    async def write(index: int) -> None:
        await ctx.harness.apply(
            RefinementProposal(summary=f"n{index}", edits=[_note(f"entry-{index}")]),
            scope="global",
            session=session,
            agent=agent,
        )

    async with anyio.create_task_group() as tasks:
        for index in range(8):
            tasks.start_soon(write, index)

    assert len(read_global_events(ctx.harness.directory)) == 8, "an append was lost or torn"
    assert set(ctx.harness.globals().entries["note"]) == {f"entry-{index}" for index in range(8)}


# ------------------------------------------------------------- rollback --


async def test_h6_rollback_restores_the_fold(harnessed: Harnessed) -> None:
    """The plan's gate. Derivable because each apply recorded what it replaced."""
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
        RefinementProposal(summary="original", edits=[_note("kept", "the first title")]),
        session=session,
        agent=agent,
    )
    before = ctx.harness.state(session).to_wire()

    bad = await ctx.harness.apply(
        RefinementProposal(
            summary="a regrettable change",
            edits=[
                HarnessEdit(action="update", kind="note", id="kept", title="worse", content="x"),
                _note("also-added"),
            ],
        ),
        session=session,
        agent=agent,
    )
    changed = ctx.harness.state(session).entry("note", "kept")
    assert changed is not None and changed.title == "worse"

    await ctx.harness.rollback(bad.refine_id, session=session, agent=agent)
    after = ctx.harness.state(session)
    restored = after.entry("note", "kept")
    assert restored is not None and restored.title == "the first title"
    assert after.entry("note", "also-added") is None
    # Byte-for-byte the entries that were there, version included: an inverse that
    # left a v3 of a v1 entry would not be an inverse.
    assert after.entries == HarnessState.model_validate(before).entries
    # The entries are back; the *history* is longer, because a rollback is itself
    # a refinement and an append-only log does not forget.
    assert len(after.refinements) == 3
    assert after.refinements[-1].rollback_of == bad.refine_id


async def test_h6_a_refinement_is_not_rolled_back_twice(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    record = await ctx.harness.apply(
        RefinementProposal(summary="one", edits=[_note("thing")]), session=session, agent=agent
    )
    await ctx.harness.rollback(record.refine_id, session=session, agent=agent)
    with pytest.raises(RefinementRefused, match="already been rolled back"):
        await ctx.harness.rollback(record.refine_id, session=session, agent=agent)


async def test_h6_rolling_back_a_global_refinement_asks_too(harnessed: Harnessed) -> None:
    """Undoing a deployment-wide entry changes every future session exactly as
    writing one does, so it goes through the same gate."""
    ctx, session, agent = await harnessed()
    asked = _allow(ctx)
    record = await ctx.harness.apply(
        RefinementProposal(summary="everyone", edits=[_note("shared")]),
        scope="global",
        session=session,
        agent=agent,
    )

    await ctx.harness.rollback(record.refine_id, session=session, agent=agent)
    assert asked == ["refine", "refine"]
    assert ctx.harness.globals().entry("note", "shared") is None
    # In the global log, not this session's — a rollback belongs to the scope it
    # undoes, or reopening the session would resurrect the entry.
    assert len(read_global_events(ctx.harness.directory)) == 2
    assert [event for event in session.events if event.type == REFINED] == []


async def test_the_command_reports_and_rolls_back(harnessed: Harnessed) -> None:
    ctx, session, agent = await harnessed()
    record = await ctx.harness.apply(
        RefinementProposal(summary="one", edits=[_note("thing")]), session=session, agent=agent
    )

    shown = await ctx.commands.dispatch("/refine --show", session=session, agent=agent)
    assert shown is not None and "[local:thing] a thing learned" in shown
    assert record.summary in shown

    rolled = await ctx.commands.dispatch(
        f"/refine --rollback {record.refine_id}", session=session, agent=agent
    )
    assert rolled is not None and record.refine_id in rolled
    assert ctx.harness.state(session).entry("note", "thing") is None
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
    shown = await ctx.commands.dispatch("/refine --rollback nope", session=session, agent=agent)
    assert shown is not None and "refusing" in shown and "nope" in shown


# ------------------------------------------------------- prompt and file --


async def test_the_harness_reaches_the_model_as_a_snapshot(harnessed: Harnessed) -> None:
    """A12: a refinement changes this text mid-session, so it must not sit in the
    cached prefix — every apply would re-bill the whole prompt."""
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
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
        session=session,
        agent=agent,
    )
    assembly = await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))
    snapshot = join_context_sections(render_context_sections(assembly))

    assert "# Continual Harness State" in snapshot
    assert "[local:run-the-tests] running the tests (v1, docs/testing.md)" in snapshot
    assert "## Recent refinements" in snapshot
    assert "Continual Harness" not in render_prompt(assembly)


def test_an_empty_harness_renders_nothing() -> None:
    """A heading with nothing under it costs tokens to say nothing."""
    assert render_state(HarnessState(), per_kind=5, refinements=5) == ""


def test_the_prompt_section_is_bounded_and_stable() -> None:
    """Bounded, because a harness that grew unchecked would be the whole prompt;
    id-ordered, so the same state renders the same bytes (A12)."""
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

    `harness_state.json` is for humans, `ph trace` and export. Nothing reads it
    back — and if it can drift, something has started treating it as state.
    """
    ctx, session, agent = await harnessed()
    await ctx.harness.apply(
        RefinementProposal(summary="one", edits=[_note("thing")]), session=session, agent=agent
    )

    path = ctx.harness.projection_path(session)
    assert path.exists()
    # Per session, because the projection is of local layered over global: one
    # shared path would have two sessions overwriting each other's.
    assert path == ctx.harness.directory / session.id / PROJECTION_NAME
    assert ctx.harness.verify_projection(session) is True

    # Deleting it loses nothing: the state is the log.
    path.unlink()
    assert ctx.harness.state(session).entry("note", "thing") is not None
    assert ctx.harness.verify_projection(session) is False
    await ctx.harness.write_projection(session)
    assert ctx.harness.verify_projection(session) is True


async def test_the_row_mounts_without_the_delegation_rows(harnessed: Harnessed) -> None:
    """The harness is not a delegation feature: `harnessed` mounts neither
    `rlm-subagent-provider` nor `rlm-prompt`, and `/refine` still works."""
    ctx, session, _agent = await harnessed()
    assert ctx.harness.state(session).entries == {}
    assert ctx.commands.get("refine") is not None
