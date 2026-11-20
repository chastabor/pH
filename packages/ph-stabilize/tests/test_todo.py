"""P4-01: `tool-todo`, planning as a cognitive anchor (G1).

Gates, with one test for each:

* The `write_todos` parallel-call error.
* The reminder text matches.

The parallel-call test is the one to read. Its important assertion is not that
the calls failed. It is that **no todos were written**. The upstream test
asserts the same. The reason is the storage design: the tool replaces the whole
list, so a rule that let the first call through and failed the second would
leave the session with a list written by a call that the model was told had
failed. The test runs through the real batch scheduler, so it tests the answer
of the pipeline and not the opinion that this module has of it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from stabilize_helpers import PROFILE, bash_call, run_tool_calls, todo_call

from ph.cordis import DEPLOYMENT, Context, Profile, ProfileDocument, load_profile_documents
from ph.json import as_obj, as_seq, as_str
from ph.keys import SESSIONS, SYSTEM_PROMPT, TOOLS
from ph.llm.types import ToolCallBlock
from ph.session import Session, SurfaceIntent, SurfaceReplace
from ph.session.known_event_types import IGNORABLE_SESSION_EVENT_TYPES
from ph.system_prompt.assembly import (
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from ph.testing import MountProfile, StubAgent, assistant_payload, log_event, result_text
from ph.tools.code_mode import CodeDispatchRef
from ph_stabilize import BUNDLE
from ph_stabilize.todo import (
    MAX_TODO_CONTENT,
    MAX_TODOS,
    PARALLEL_CALL_ERROR,
    TOOL_NAME,
    WRITE_TODOS_SYSTEM_PROMPT,
    WriteTodosArgs,
    _parallel_write_todos,
    blocked_by,
    render_todo_list,
    todos_of,
    unevidenced,
)

pytestmark = pytest.mark.anyio


ENABLED: dict[str, Any] = {"id": "tool-todo", "disabled": False}
"""The opt-in, written as a profile writes it: a patch that enables the row."""


def _todos(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    """The *arguments* shape: what the model sends."""
    return [{"content": content, "status": status} for content, status in pairs]


def _plan(session: Session) -> list[tuple[str, str]]:
    """The logged list as `(content, status)`.

    The comparison uses the two fields that these tests are about. The log also
    holds `requires` and the harness-issued `worked`. Those fields have their
    own gates, and an equality check on the full dict here would fail every time
    either of them grows.
    """
    return [(str(one["content"]), str(one["status"])) for one in todos_of(session)]


async def _run(
    ctx: Context,
    session: Session,
    *calls: ToolCallBlock,
    step: int = 1,
) -> None:
    """One batch, through the shared driver.

    The driver first commits the assistant message, and that commit is
    necessary. The loop commits the message before it executes any of the calls
    in it, and the parallel rule is read from that message. A test that skipped
    the commit would exercise a path that the harness never takes. The driver is
    shared with `test_limits`, so this is stated in one place.
    """
    await run_tool_calls(ctx, session, *calls, step=step)


# ----------------------------------------------------------------- the bundle --


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    """A row whose `name:` does not resolve fails at mount, in someone's session.

    The bundle is composed through the loader itself. The rows of this bundle
    are plain, so the bundle stands alone. A second, hand-written parser of the
    bundle grammar could drift from the real parser in the loader.
    """
    from ph.cordis.loader import resolve_plugin

    rows = Profile.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        name = as_str(row["name"])
        assert resolve_plugin(name) is not None, name


async def test_every_enabled_row_in_the_profile_activates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that mounts nothing is worse than a row that fails: it looks fine.

    The profile is really mounted, because `Mount.inactive()` reads the forks
    that `mount()` creates. The first version of this test asked before the
    mount. `Mount.inactive()` then answered over an empty table and could not
    fail.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    documents = load_profile_documents(PROFILE)
    documents.append(ProfileDocument("test-overlay", [dict(ENABLED)]))
    profile = Profile.from_documents(documents)
    ctx = Context()
    try:
        mount = await profile.mount(ctx)
        assert mount.inactive() == []
    finally:
        await ctx.drain()
        await ctx.dispose()


async def test_the_row_is_opt_in(mount: MountProfile) -> None:
    """Layering the bundle does not, by itself, give the model a tool.

    The row is `disabled: true` in the bundle, and the profile that wants it
    enables it. That is the `rlm-context-loader` idiom, and it is what the plan
    means by "opt-in row; on in rlm-stable". A bundle that forced the tool on
    every profile that wants any *other* stabilization row would make the
    comment beside the row false.
    """
    without = await mount(profile=PROFILE)
    assert without.require(TOOLS).get(TOOL_NAME, scope=DEPLOYMENT) is None

    enabled = await mount(ENABLED, profile=PROFILE)
    assert enabled.require(TOOLS).get(TOOL_NAME, scope=DEPLOYMENT) is not None


# ------------------------------------------------------------------ the list --


async def test_the_list_is_written_to_the_log_and_nowhere_else(mount: MountProfile) -> None:
    """The storage design: a fold, not a table.

    For that reason the list survives a resume and a fork at no extra cost, and
    the sidebar and the view of the model stay one projection and not two.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("planning")

    await _run(
        ctx, session, todo_call("c1", _todos(("survey", "in_progress"), ("port", "pending")))
    )

    written = [event for event in session.events if event.type == "todo/write"]
    assert len(written) == 1
    assert _plan(session) == [("survey", "in_progress"), ("port", "pending")]


async def test_a_second_call_replaces_the_whole_list(mount: MountProfile) -> None:
    """Whole-list replacement, which is why two calls in one turn are ambiguous."""
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("planning")

    await _run(ctx, session, todo_call("c1", _todos(("survey", "in_progress"))))
    await _run(
        ctx, session, todo_call("c2", _todos(("survey", "completed"), ("port", "in_progress")))
    )

    assert _plan(session) == [("survey", "completed"), ("port", "in_progress")]
    assert todos_of(Session("empty")) == [], "a session that never planned has no list"


# ------------------------------------------------------------------- bounded --


async def test_a_runaway_entry_is_refused_and_writes_nothing(mount: MountProfile) -> None:
    """The list is the one model-written string that is in *every* later prompt.

    It is a `context`, rebuilt on each turn. After compaction has shadowed the
    turns that wrote it, it is the only remaining statement of the plan. An
    entry is therefore not a place to paste a diff. Text that is unbounded here
    is unbounded in every request for the rest of the session. That is the
    lesson of P7-13 one layer up: the bound belongs where the unbounded thing is
    *written*, and not at each reader.

    The entry is refused and not clipped, and the log is left unchanged. A model
    that was told that its text was accepted, when the text had been truncated,
    would continue to plan against a list that says something else.

    Sabotage: drop `max_length` from `TodoItem.content` and a 10 KB paste enters
    the prompt of every turn that follows.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("runaway")

    await _run(ctx, session, todo_call("c1", _todos(("x" * (MAX_TODO_CONTENT + 1), "pending"))))

    assert todos_of(session) == [], "nothing was written"
    result = session.latest("tool/result")
    assert result is not None
    assert as_obj(as_seq(as_obj(result.data["message"])["content"])[0])["isError"] is True


async def test_a_list_longer_than_the_cap_is_refused(mount: MountProfile) -> None:
    """The count bounds the product: entries inside the length limit still add up.

    The cap is well past any real plan. The upstream guidance is to skip the
    tool entirely when the work is "a few tool calls".
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("too-many")

    over = tuple((f"step {n}", "pending") for n in range(MAX_TODOS + 1))
    await _run(ctx, session, todo_call("c1", _todos(*over)))

    assert todos_of(session) == []


def test_the_model_is_told_the_bound_rather_than_discovering_it() -> None:
    """A refusal that a model can avoid: the cap is in the schema of the tool.

    `maxLength` reaches the model with the parameter description. The model can
    therefore see that a plan would not fit before it calls the tool, and does
    not have to interpret a rejection.
    """
    schema = WriteTodosArgs.model_json_schema()

    assert schema["$defs"]["TodoItem"]["properties"]["content"]["maxLength"] == MAX_TODO_CONTENT
    assert schema["properties"]["todos"]["maxItems"] == MAX_TODOS


# -------------------------------------------------------- the fork: requires --


def _with(content: str, status: str, requires: list[str]) -> dict[str, Any]:
    return {"content": content, "status": status, "requires": requires}


async def test_a_plan_states_its_own_dependencies(mount: MountProfile) -> None:
    """**The first half of P7-16.** The fork from langchain, and its benefit.

    The upstream schema is `{content, status}`. Its *prompt* works around the
    absence of dependencies: "When blocked, create a new task describing what
    needs to be resolved". That is a graph written as a naming convention, and
    nothing can read it, order it or check it. With a declared dependency,
    `blocked_by` answers the question that the convention could only hint at:
    what can really be started now.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("ordered")

    await _run(
        ctx,
        session,
        todo_call(
            "c1",
            [
                _with("write the seam", "in_progress", []),
                _with("gate it", "pending", ["write the seam"]),
                _with("document it", "pending", ["gate it"]),
            ],
        ),
    )

    assert blocked_by(todos_of(session)) == {
        "gate it": ["write the seam"],
        "document it": ["gate it"],
    }
    # And the model is reminded at the moment it chooses what to do next.
    assert "(waiting on: write the seam)" in render_todo_list(todos_of(session))


async def test_a_dependency_on_nothing_is_refused(mount: MountProfile) -> None:
    """A reference the list cannot satisfy is a plan error, not a silent no-op.

    Sabotage: drop the membership check and the entry waits forever on a name
    that will never be completed, with `blocked_by` reporting it as blocked and
    nothing saying why.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("dangling")

    await _run(
        ctx, session, todo_call("c1", [_with("gate it", "pending", ["a step I never wrote"])])
    )

    assert todos_of(session) == [], "nothing was written"
    assert "not in the list" in result_text(session, "c1")


async def test_a_cycle_is_refused(mount: MountProfile) -> None:
    """A plan that cannot be satisfied, caught where the model can still fix it.

    `ResolveStepOrder` in Playbooks only guards its recursion with a visited
    set, so a cycle there is accepted and silently ordered. Refusal is the
    better answer, because a plan in which nothing can start is a plan that the
    model wrote by mistake.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("cyclic")

    await _run(
        ctx,
        session,
        todo_call("c1", [_with("a", "pending", ["b"]), _with("b", "pending", ["a"])]),
    )

    assert todos_of(session) == []
    assert "cycle" in result_text(session, "c1")


async def test_an_entry_that_waits_on_itself_is_the_one_node_cycle(mount: MountProfile) -> None:
    """The cycle walk catches this. It has no rule of its own.

    A separate self-reference check was one more branch that said what the DFS
    already says: `a -> a` is a cycle with one node. Two mechanisms for one
    contradiction can come to disagree.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("ouroboros")

    await _run(ctx, session, todo_call("c1", [_with("a", "pending", ["a"])]))

    assert todos_of(session) == []
    assert "cycle: a -> a" in result_text(session, "c1")


async def test_claiming_to_have_started_something_still_blocked_is_refused(
    mount: MountProfile,
) -> None:
    """The model is checked against its own statement, and not against a policy.

    An entry that is `in_progress` while something that it *said* it waits on is
    unfinished is a contradiction that the model can only have meant by mistake.
    This stays on the right side of P5-16 because `requires` is optional. A
    model that finds the rule inconvenient does not declare the dependency, and
    then nothing here has an opinion about its plan.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("jumped")

    await _run(
        ctx,
        session,
        todo_call("c1", [_with("build", "pending", []), _with("ship", "in_progress", ["build"])]),
    )

    assert todos_of(session) == []
    assert "waits on" in result_text(session, "c1")


# ------------------------------------------------------- the fork: evidence --


async def test_a_completion_carries_what_the_harness_saw(mount: MountProfile) -> None:
    """**The second half of P7-16.** The one field in the list that the model does not write.

    `worked` is counted from `tool/call`. Since P7-15 that record exists only
    for a call that the pipeline let through and was about to run. `worked` is
    therefore work that the harness *saw*, and not work that the model claims. A
    receipt that the claimant issues is not a receipt.

    Sabotage: accept `worked` as an argument instead, and a model that ticked a
    box without doing anything can say otherwise.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("witnessed")

    await _run(ctx, session, todo_call("c1", _todos(("survey", "in_progress"))))
    await _run(ctx, session, bash_call("b1", "true"), step=2)
    await _run(ctx, session, todo_call("c2", _todos(("survey", "completed"))), step=3)

    (done,) = [one for one in todos_of(session) if one["status"] == "completed"]
    assert done["worked"] == 1, "one bash call happened in the window"
    assert unevidenced(todos_of(session)) == []


async def test_a_box_ticked_with_no_work_behind_it_is_visible(mount: MountProfile) -> None:
    """The failure that a checklist marked by the model otherwise hides (P5-16, G1).

    Nothing ran between the two writes, so the completion is a bare claim. It is
    not *wrong*: "decide the approach" is a real step with no tool calls. But
    the difference between a tick with work behind it and a tick without is
    exactly what a person who watches a plan wants to see, and the list could
    not show it before.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("bare")

    await _run(ctx, session, todo_call("c1", _todos(("port the row", "in_progress"))))
    await _run(ctx, session, todo_call("c2", _todos(("port the row", "completed"))), step=2)

    assert unevidenced(todos_of(session)) == ["port the row"]


async def test_a_receipt_travels_with_its_entry(mount: MountProfile) -> None:
    """A write replaces the whole list, so evidence must be carried forward.

    The receipt of an entry is about the window in which the entry was
    *finished*, and not the window of whichever write mentions it next. A later
    write that runs no tools must therefore not erase what an earlier write
    witnessed.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("carried")

    await _run(ctx, session, todo_call("c1", _todos(("first", "in_progress"))))
    await _run(ctx, session, bash_call("b1", "true"), step=2)
    await _run(
        ctx,
        session,
        todo_call("c2", _todos(("first", "completed"), ("second", "in_progress"))),
        step=3,
    )
    # A later write with no work in its window at all.
    await _run(
        ctx,
        session,
        todo_call("c3", _todos(("first", "completed"), ("second", "completed"))),
        step=4,
    )

    kept = {one["content"]: one.get("worked") for one in todos_of(session)}
    assert kept["first"] == 1, "the earlier window survived a later write"
    assert kept["second"] == 0
    assert unevidenced(todos_of(session)) == ["second"]


# --------------------------------------------------------- the parallel rule --


async def test_two_calls_in_one_message_both_fail_and_write_nothing(mount: MountProfile) -> None:
    """The gate of the row, with the second assertion of the upstream test.

    Both calls are refused *before* either body runs. The session is therefore
    left with no list at all, and not with the list of the first call.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("parallel")

    await _run(
        ctx,
        session,
        todo_call("c1", _todos(("first", "in_progress"))),
        todo_call("c2", _todos(("first", "completed"), ("second", "pending"))),
    )

    results = [event for event in session.events if event.type == "tool/result"]
    assert len(results) == 2, "every call in the message gets a result"
    for event in results:
        # `failureKind` is what the pipeline records. `isError` is the field of
        # the *presentation* view and is absent here. A test that asserted the
        # wrong key would have passed this whole file while the rule did nothing.
        assert event.data.get("failureKind") == "denied", str(event.data)
        assert PARALLEL_CALL_ERROR in str(event.data)

    assert not [event for event in session.events if event.type == "todo/write"]
    assert todos_of(session) == [], "a refused call must not leave a list behind"


async def test_one_call_in_a_message_is_allowed(mount: MountProfile) -> None:
    """The gate can be falsified: the rule is about *parallel* calls.

    Without this test, a listener that denied every `write_todos` would pass the
    test above, and nothing else here would notice.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("single")

    await _run(ctx, session, todo_call("c1", _todos(("survey", "in_progress"))))

    (result,) = [event for event in session.events if event.type == "tool/result"]
    assert result.data.get("failureKind") is None, str(result.data)
    assert _plan(session) == [("survey", "in_progress")]


async def test_two_calls_across_two_messages_are_both_allowed(mount: MountProfile) -> None:
    """ "Parallel" means one assistant message, not one session.

    A rule that counted every `write_todos` in the log would refuse the second
    turn of every planned task.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("sequential")

    await _run(ctx, session, todo_call("c1", _todos(("survey", "in_progress"))))
    await _run(ctx, session, todo_call("c2", _todos(("survey", "completed"))))

    assert not [
        event
        for event in session.events
        if event.type == "tool/result" and event.data.get("isError")
    ]


# ------------------------------------------------------------ the prompt text --


async def test_the_reminder_text_reaches_the_prompt_verbatim(mount: MountProfile) -> None:
    """The other gate of the row.

    The assertion is against the *assembled* prompt and not against the
    registered section. The plan promises that the model reads this text, and a
    section that is registered at an order that nothing renders would satisfy
    the weaker claim.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    text = render_prompt(await ctx.require(SYSTEM_PROMPT).assemble(DEPLOYMENT))

    assert WRITE_TODOS_SYSTEM_PROMPT in text
    # The four sentences that the port plan names. They tie the gate to the plan
    # and not only to the constant beside it.
    for sentence in (
        "Use this tool for complex objectives",
        "mark todos as completed as soon as you are done with a step",
        "it is better to just complete the objective directly and NOT use this tool",
        "write your final answer in the message AFTER your last `write_todos` call",
    ):
        assert sentence in text, sentence


async def test_the_list_rides_the_context_and_not_the_cached_prefix(mount: MountProfile) -> None:
    """A12. The advice is static, so it caches. The list changes, so it must not.

    A plan update that moved the prefix would invalidate the cache on every
    `write_todos` call, and that is the turn that a long task makes most often.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("cache")
    agent = StubAgent(ctx, session)

    async def assembled() -> Any:  # noqa: ANN401
        return await ctx.require(SYSTEM_PROMPT).assemble(agent.ctx, agent=agent)

    before = await assembled()
    await _run(ctx, session, todo_call("c1", _todos(("survey", "in_progress"))))
    after = await assembled()

    assert render_prompt(before) == render_prompt(after), "writing a todo moved the cached prefix"
    assert "survey" not in render_prompt(after)

    # And it does reach the model, on the side that is allowed to change.
    contexts = join_context_sections(render_context_sections(after))
    assert render_todo_list(todos_of(session)) in contexts
    assert "survey" not in join_context_sections(render_context_sections(before))


def test_the_event_type_is_required_not_ignorable() -> None:
    """The event type is required and not ignorable. The list reaches the model
    through a prompt context, so a reader that skipped the event would assemble a
    different prompt than the session had. (`test_log_writers.py` checks that the
    type is *known* at all, for every write in every package.)
    """
    assert "todo/write" not in IGNORABLE_SESSION_EVENT_TYPES


async def test_work_inside_a_code_cell_counts_as_work(mount: MountProfile) -> None:
    """D12: `TOOL_DISPATCH_EVENT_TYPES` explains why both records are one fact.

    A count of `tool/call` alone scored a Code Mode cell that read nine files
    and edited three as *one* piece of work.

    The records are appended directly and not driven through a code runtime. The
    test is about the arithmetic of the fold, and a real cell would put a kernel
    between this assertion and the thing that it is about.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.require(SESSIONS).create("code-mode")

    await _run(ctx, session, todo_call("c1", _todos(("survey", "in_progress"))))
    for index in range(3):
        # Through the real wire type, as `test_limits.py` does for the same
        # event. A renamed field then fails here and is not silently uncounted.
        log_event(
            session,
            "tool/code-dispatch-start",
            CodeDispatchRef(
                root_call_id="r1", parent_call_id="r1", sub_call_id=f"s{index}", name="read"
            ).to_wire(),
        )
    await _run(ctx, session, todo_call("c2", _todos(("survey", "completed"))), step=3)

    (done,) = [one for one in todos_of(session) if one["status"] == "completed"]
    assert done["worked"] == 3, "the dispatches inside the cell were not counted"
    assert unevidenced(todos_of(session)) == []


def _assistant(calls: list[ToolCallBlock], message_id: str) -> dict[str, Any]:
    """An `assistant/message` payload carrying these calls, and nothing else."""
    return assistant_payload(
        "", message_id, content=[call.model_dump(mode="json", by_alias=True) for call in calls]
    )


def test_the_parallel_rule_reads_the_message_being_executed_not_a_rewrite() -> None:
    """L12: `SurfaceReplace` appends, so the newest event is not the newest work.

    The rule asked `Session.latest("assistant/message")`, which answers with the
    most recently *appended* message. `SummarizeEngine.truncate_arguments` in
    `compaction` rewrites messages at the **front** of the conversation
    (`nodes[:cutoff]`) and appends each rewrite at the end. After that pass, the
    newest `assistant/message` in the log is a near-copy of something that was
    answered many steps ago.

    The fault is latent in the harness as it is now. Nothing appends a rewrite
    between a message and its own tool dispatch, and argument elision preserves
    the `write_todos` blocks that the rule counts. The fix is at the reader and
    does not wait for the first producer that drops a block or interleaves. The
    failure would be silent in the permissive direction: the gate stops denying.

    The assertion is on the reader directly. A test that drove the loop would
    assert that the harness does not currently interleave, and that is the thing
    that could change.

    Sabotage: `session.latest("assistant/message")` in place of the projection
    and the two-call message is read as the one-call rewrite.
    """
    session = Session("rewritten")
    old = log_event(
        session,
        "assistant/message",
        _assistant([todo_call("c0", _todos(("early", "completed")))], "m1"),
        SurfaceIntent("append"),
    )
    # The message actually in flight: two calls, which the rule exists to refuse.
    log_event(
        session,
        "assistant/message",
        _assistant(
            [
                todo_call("c1", _todos(("first", "in_progress"))),
                todo_call("c2", _todos(("second", "pending"))),
            ],
            "m2",
        ),
        SurfaceIntent("append"),
    )
    assert _parallel_write_todos(session), "the baseline: two calls in one message"

    # An argument truncation happens and rewrites the *first* message in place.
    # It is the same message, so it has the same id. It is now the newest
    # `assistant/message` event in the log, and it is not new work.
    log_event(
        session,
        "assistant/message",
        _assistant([todo_call("c0", _todos(("early", "completed")))], "m1"),
        SurfaceIntent(SurfaceReplace(replaces=(old.seq,)), (old.seq,)),
    )

    assert _parallel_write_todos(session), "a rewrite of an older message answered for this one"
