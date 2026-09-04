"""P4-01 — `tool-todo`: planning as a cognitive anchor (G1).

The row's gates, one test each: *`write_todos` parallel-call error; reminder
text matches.*

The parallel-call test is the one to read, and the assertion that matters is
not that the calls failed — it is that **no todos were written**. Upstream's own
test says so, and the reason is the storage design: the tool replaces the whole
list, so a rule that let the first call through and failed the second would
leave the session holding a list written by a call the model was told had
failed. Driven through the real batch scheduler, so what is tested is the
pipeline's answer rather than this module's opinion of it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from stabilize_helpers import PROFILE, bash_call, result_text, run_tool_calls

from ph.cordis import DEPLOYMENT, Context, Profile, load_profile_documents
from ph.llm.types import ToolCallBlock
from ph.session import Session
from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES
from ph.system_prompt.assembly import (
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from ph.testing import StubAgent
from ph_stabilize import BUNDLE
from ph_stabilize.todo import (
    MAX_TODO_CONTENT,
    MAX_TODOS,
    PARALLEL_CALL_ERROR,
    TOOL_NAME,
    WRITE_TODOS_SYSTEM_PROMPT,
    WriteTodosArgs,
    blocked_by,
    render_todo_list,
    todos_of,
    unevidenced,
)

pytestmark = pytest.mark.anyio


ENABLED: dict[str, Any] = {"id": "tool-todo", "disabled": False}
"""The opt-in, spelled as a profile spells it: a patch flipping the row."""


def _todos(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    """The *arguments* shape — what the model sends."""
    return [{"content": content, "status": status} for content, status in pairs]


def _plan(session: Session) -> list[tuple[str, str]]:
    """The logged list as `(content, status)`.

    Compared on the two fields these tests are about, because the log also holds
    `requires` and the harness-issued `worked` — fields with their own gates, and
    a full-dict equality here would fail every time either of them grows.
    """
    return [(str(one["content"]), str(one["status"])) for one in todos_of(session)]


def _call(call_id: str, todos: list[dict[str, str]]) -> ToolCallBlock:
    return ToolCallBlock(id=call_id, name=TOOL_NAME, arguments=json.dumps({"todos": todos}))


async def _run(ctx: Any, session: Session, *calls: ToolCallBlock, step: int = 1) -> None:
    """One batch, through the shared driver.

    The assistant message it commits first is load-bearing: the loop does that
    before executing any of a message's calls, and the parallel rule is read
    from it — a test that skipped it would exercise a path the harness never
    takes. Shared with `test_limits` so there is one statement of that.
    """
    await run_tool_calls(ctx, session, *calls, step=step)


# ----------------------------------------------------------------- the bundle --


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    """A row whose `name:` does not resolve fails at mount, in someone's session.

    Composed through the loader itself — this bundle's rows are plain, so it
    stands alone — rather than through a second hand parser of the bundle
    grammar that could drift from the loader's real one.
    """
    from ph.cordis.loader import resolve_plugin

    rows = Profile.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        assert resolve_plugin(row["name"]) is not None, row["name"]


async def test_every_enabled_row_in_the_profile_activates(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that mounts nothing is worse than one that fails: it looks fine.

    Mounted for real, because `Mount.inactive()` reads the forks `mount()`
    creates — asked before mounting, as this test's first version asked, it
    answers over an empty table and cannot fail.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    documents = load_profile_documents(PROFILE)
    documents.append(("test-overlay", [dict(ENABLED)]))
    profile = Profile.from_documents(documents)
    ctx = Context()
    try:
        mount = await profile.mount(ctx)
        assert mount.inactive() == []
    finally:
        await ctx.drain()
        await ctx.dispose()


async def test_the_row_is_opt_in(mount: Any) -> None:
    """Layering the bundle does not, by itself, hand the model a tool.

    `disabled: true` in the bundle, flipped by the profile that wants it — the
    `rlm-context-loader` idiom, and what the plan means by "opt-in row; on in
    rlm-stable". A bundle that forced the tool on every profile wanting any
    *other* stabilization row would make the comment beside it a lie.
    """
    without = await mount(profile=PROFILE)
    assert without.tools.get(TOOL_NAME, scope=DEPLOYMENT) is None

    enabled = await mount(ENABLED, profile=PROFILE)
    assert enabled.tools.get(TOOL_NAME, scope=DEPLOYMENT) is not None


# ------------------------------------------------------------------ the list --


async def test_the_list_is_written_to_the_log_and_nowhere_else(mount: Any) -> None:
    """The storage design: a fold, not a table.

    That is what makes the list survive a resume and a fork for free, and what
    keeps the sidebar and the model's view one projection rather than two.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("planning")

    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"), ("port", "pending"))))

    written = [event for event in session.events if event.type == "todo/write"]
    assert len(written) == 1
    assert _plan(session) == [("survey", "in_progress"), ("port", "pending")]


async def test_a_second_call_replaces_the_whole_list(mount: Any) -> None:
    """Whole-list replacement, which is why two calls in one turn are ambiguous."""
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("planning")

    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"))))
    await _run(ctx, session, _call("c2", _todos(("survey", "completed"), ("port", "in_progress"))))

    assert _plan(session) == [("survey", "completed"), ("port", "in_progress")]
    assert todos_of(Session("empty")) == [], "a session that never planned has no list"


# ------------------------------------------------------------------- bounded --


async def test_a_runaway_entry_is_refused_and_writes_nothing(mount: Any) -> None:
    """The list is the one model-written string that rides *every* later prompt.

    It is a `context`, rebuilt each turn, and once compaction has shadowed the
    turns that wrote it, the only surviving statement of the plan — so an entry
    is not a place to paste a diff. Unbounded here is unbounded in every request
    for the rest of the session, which is P7-13's lesson one layer up: the bound
    belongs where the unbounded thing is *written*, not at each reader.

    Refused rather than clipped, and the log is left alone: a model told its text
    was accepted when it had been truncated would go on planning against a list
    that says something else.

    Sabotage: drop `max_length` from `TodoItem.content` and a 10 KB paste enters
    the prompt of every turn that follows.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("runaway")

    await _run(ctx, session, _call("c1", _todos(("x" * (MAX_TODO_CONTENT + 1), "pending"))))

    assert todos_of(session) == [], "nothing was written"
    result = session.latest("tool/result")
    assert result is not None
    assert result.data["message"]["content"][0]["isError"] is True


async def test_a_list_longer_than_the_cap_is_refused(mount: Any) -> None:
    """The count bounds the product: entries inside the length limit still add up.

    Well past any real plan — upstream's own guidance is to skip the tool
    entirely when the work is "a few tool calls".
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("too-many")

    over = tuple((f"step {n}", "pending") for n in range(MAX_TODOS + 1))
    await _run(ctx, session, _call("c1", _todos(*over)))

    assert todos_of(session) == []


def test_the_model_is_told_the_bound_rather_than_discovering_it() -> None:
    """A refusal a model can avoid: the cap is in the tool's own schema.

    `maxLength` reaches the model with the parameter description, so a plan that
    would not fit is a thing it can see before calling rather than a rejection it
    has to interpret.
    """
    schema = WriteTodosArgs.model_json_schema()

    assert schema["$defs"]["TodoItem"]["properties"]["content"]["maxLength"] == MAX_TODO_CONTENT
    assert schema["properties"]["todos"]["maxItems"] == MAX_TODOS


# -------------------------------------------------------- the fork: requires --


def _with(content: str, status: str, requires: list[str]) -> dict[str, Any]:
    return {"content": content, "status": status, "requires": requires}


async def test_a_plan_states_its_own_dependencies(mount: Any) -> None:
    """**P7-16's first half.** The fork from langchain, and what it buys.

    Upstream's schema is `{content, status}`, and the absence of dependencies is
    what its *prompt* works around: "When blocked, create a new task describing
    what needs to be resolved" — a graph written as a naming convention, which
    nothing can read, order or check. Declared, `blocked_by` answers the question
    the convention could only gesture at: what is actually startable now.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("ordered")

    await _run(
        ctx,
        session,
        _call(
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


async def test_a_dependency_on_nothing_is_refused(mount: Any) -> None:
    """A reference the list cannot satisfy is a plan error, not a silent no-op.

    Sabotage: drop the membership check and the entry waits forever on a name
    that will never be completed, with `blocked_by` reporting it as blocked and
    nothing saying why.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("dangling")

    await _run(ctx, session, _call("c1", [_with("gate it", "pending", ["a step I never wrote"])]))

    assert todos_of(session) == [], "nothing was written"
    assert "not in the list" in result_text(session, "c1")


async def test_a_cycle_is_refused(mount: Any) -> None:
    """An unsatisfiable plan, caught where the model can still fix it.

    Playbooks' own `ResolveStepOrder` only guards its recursion with a visited
    set, so a cycle there is accepted and silently ordered; refusing is the
    better answer, because a plan where nothing can start is a plan the model
    wrote by mistake.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("cyclic")

    await _run(
        ctx,
        session,
        _call("c1", [_with("a", "pending", ["b"]), _with("b", "pending", ["a"])]),
    )

    assert todos_of(session) == []
    assert "cycle" in result_text(session, "c1")


async def test_an_entry_that_waits_on_itself_is_the_one_node_cycle(mount: Any) -> None:
    """Caught by the cycle walk rather than by a rule of its own.

    A separate self-reference check was one more branch saying what the DFS
    already says — `a -> a` is a cycle with one node — and two mechanisms for one
    contradiction is how they come to disagree.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("ouroboros")

    await _run(ctx, session, _call("c1", [_with("a", "pending", ["a"])]))

    assert todos_of(session) == []
    assert "cycle: 'a' -> 'a'" in result_text(session, "c1") or "cycle" in result_text(
        session, "c1"
    )


async def test_claiming_to_have_started_something_still_blocked_is_refused(mount: Any) -> None:
    """The model checked against its own statement — not against a policy.

    An entry `in_progress` while something it *said* it waits on is unfinished is
    a contradiction the model can only have meant by mistake. This stays on the
    right side of P5-16 because `requires` is optional: a model that finds the
    rule inconvenient simply does not declare the dependency, and then nothing
    here has an opinion about its plan.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("jumped")

    await _run(
        ctx,
        session,
        _call("c1", [_with("build", "pending", []), _with("ship", "in_progress", ["build"])]),
    )

    assert todos_of(session) == []
    assert "waits on" in result_text(session, "c1")


# ------------------------------------------------------- the fork: evidence --


async def test_a_completion_carries_what_the_harness_saw(mount: Any) -> None:
    """**P7-16's second half.** The one field in the list the model does not write.

    `worked` is counted from `tool/call` — which since P7-15 exists only for a
    call the pipeline let through and was about to run — so it is work the
    harness *saw*, not work the model claims. A receipt the claimant issues is
    not a receipt.

    Sabotage: accept `worked` as an argument instead, and a model that ticked a
    box without doing anything can say otherwise.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("witnessed")

    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"))))
    await _run(ctx, session, bash_call("b1", "true"), step=2)
    await _run(ctx, session, _call("c2", _todos(("survey", "completed"))), step=3)

    (done,) = [one for one in todos_of(session) if one["status"] == "completed"]
    assert done["worked"] == 1, "one bash call happened in the window"
    assert unevidenced(todos_of(session)) == []


async def test_a_box_ticked_with_no_work_behind_it_is_visible(mount: Any) -> None:
    """The failure a model-marked checklist otherwise hides (P5-16, G1).

    Nothing ran between the two writes, so the completion is a bare claim. It is
    not *wrong* — "decide the approach" is a real step with no tool calls — but
    the difference between a tick with work behind it and one without is exactly
    what a person watching a plan wants, and what the list could not say before.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("bare")

    await _run(ctx, session, _call("c1", _todos(("port the row", "in_progress"))))
    await _run(ctx, session, _call("c2", _todos(("port the row", "completed"))), step=2)

    assert unevidenced(todos_of(session)) == ["port the row"]


async def test_a_receipt_travels_with_its_entry(mount: Any) -> None:
    """A write replaces the whole list, so evidence has to be carried forward.

    An entry's receipt is about the window it was *finished* in, not the window
    of whatever write happens to mention it next — so a later write that runs no
    tools must not erase what an earlier one witnessed.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("carried")

    await _run(ctx, session, _call("c1", _todos(("first", "in_progress"))))
    await _run(ctx, session, bash_call("b1", "true"), step=2)
    await _run(
        ctx, session, _call("c2", _todos(("first", "completed"), ("second", "in_progress"))), step=3
    )
    # A later write with no work in its window at all.
    await _run(
        ctx, session, _call("c3", _todos(("first", "completed"), ("second", "completed"))), step=4
    )

    kept = {one["content"]: one.get("worked") for one in todos_of(session)}
    assert kept["first"] == 1, "the earlier window survived a later write"
    assert kept["second"] == 0
    assert unevidenced(todos_of(session)) == ["second"]


# --------------------------------------------------------- the parallel rule --


async def test_two_calls_in_one_message_both_fail_and_write_nothing(mount: Any) -> None:
    """The row's gate, with upstream's own second assertion.

    Both calls are refused *before* either body runs, so the session is left
    with no list at all — not with the first call's.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("parallel")

    await _run(
        ctx,
        session,
        _call("c1", _todos(("first", "in_progress"))),
        _call("c2", _todos(("first", "completed"), ("second", "pending"))),
    )

    results = [event for event in session.events if event.type == "tool/result"]
    assert len(results) == 2, "every call in the message gets a result"
    for event in results:
        # `failureKind`, which is what the pipeline records; `isError` is the
        # *presentation* view's field and is absent here. A test asserting the
        # wrong key would have passed this whole file while the rule did nothing.
        assert event.data.get("failureKind") == "denied", str(event.data)
        assert PARALLEL_CALL_ERROR in str(event.data)

    assert not [event for event in session.events if event.type == "todo/write"]
    assert todos_of(session) == [], "a refused call must not leave a list behind"


async def test_one_call_in_a_message_is_allowed(mount: Any) -> None:
    """The gate's own falsifiability: the rule is about *parallel* calls.

    Without this, a listener that denied every `write_todos` would pass the test
    above and nothing else here would notice.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("single")

    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"))))

    (result,) = [event for event in session.events if event.type == "tool/result"]
    assert result.data.get("failureKind") is None, str(result.data)
    assert _plan(session) == [("survey", "in_progress")]


async def test_two_calls_across_two_messages_are_both_allowed(mount: Any) -> None:
    """ "Parallel" means one assistant message, not one session.

    A rule that counted every `write_todos` in the log would refuse the second
    turn of every planned task.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("sequential")

    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"))))
    await _run(ctx, session, _call("c2", _todos(("survey", "completed"))))

    assert not [
        event
        for event in session.events
        if event.type == "tool/result" and event.data.get("isError")
    ]


# ------------------------------------------------------------ the prompt text --


async def test_the_reminder_text_reaches_the_prompt_verbatim(mount: Any) -> None:
    """The row's other gate.

    Asserted against the *assembled* prompt rather than against the registered
    section, because what the plan promises is that the model reads this — a
    section registered at an order nothing renders would satisfy the weaker
    claim.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    text = render_prompt(await ctx.system_prompt.assemble(DEPLOYMENT))

    assert WRITE_TODOS_SYSTEM_PROMPT in text
    # The four sentences the port plan names, so the gate is tied to the plan
    # and not only to the constant beside it.
    for sentence in (
        "Use this tool for complex objectives",
        "mark todos as completed as soon as you are done with a step",
        "it is better to just complete the objective directly and NOT use this tool",
        "write your final answer in the message AFTER your last `write_todos` call",
    ):
        assert sentence in text, sentence


async def test_the_list_rides_the_context_and_not_the_cached_prefix(mount: Any) -> None:
    """A12. The advice is static, so it caches; the list changes, so it must not.

    A plan update that moved the prefix would invalidate the cache on every
    `write_todos` call — which is exactly the turn a long task makes most often.
    """
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("cache")
    agent = StubAgent(ctx, session)

    async def assembled() -> Any:
        return await ctx.system_prompt.assemble(agent.ctx, agent=agent)

    before = await assembled()
    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"))))
    after = await assembled()

    assert render_prompt(before) == render_prompt(after), "writing a todo moved the cached prefix"
    assert "survey" not in render_prompt(after)

    # And it does reach the model, on the side that is allowed to change.
    contexts = join_context_sections(render_context_sections(after))
    assert render_todo_list(todos_of(session)) in contexts
    assert "survey" not in join_context_sections(render_context_sections(before))


def test_the_event_type_is_in_the_vocabulary() -> None:
    """`todo/write` was a declared forward reference until this row.

    Required rather than ignorable: the list reaches the model through a prompt
    context, so a reader that skipped the event would assemble a different
    prompt than the session had.
    """
    assert "todo/write" in KNOWN_SESSION_EVENT_TYPES

    from ph.session.known_event_types import IGNORABLE_SESSION_EVENT_TYPES

    assert "todo/write" not in IGNORABLE_SESSION_EVENT_TYPES
