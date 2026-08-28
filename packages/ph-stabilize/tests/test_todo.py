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
from stabilize_helpers import PROFILE

from ph.cancel import CancelToken
from ph.cordis import Context, Loader
from ph.llm.types import ToolCallBlock
from ph.session import Session, SurfaceIntent
from ph.session.known_event_types import KNOWN_SESSION_EVENT_TYPES
from ph.system_prompt.assembly import (
    AssembleContext,
    join_context_sections,
    render_context_sections,
    render_prompt,
)
from ph.testing import StubAgent, assistant_payload
from ph.tools.batch import execute_tool_calls
from ph_stabilize import BUNDLE
from ph_stabilize.todo import (
    PARALLEL_CALL_ERROR,
    TOOL_NAME,
    WRITE_TODOS_SYSTEM_PROMPT,
    render_todo_list,
    todos_of,
)

pytestmark = pytest.mark.anyio


ENABLED: dict[str, Any] = {"id": "tool-todo", "disabled": False}
"""The opt-in, spelled as a profile spells it: a patch flipping the row."""


def _todos(*pairs: tuple[str, str]) -> list[dict[str, str]]:
    return [{"content": content, "status": status} for content, status in pairs]


def _call(call_id: str, todos: list[dict[str, str]]) -> ToolCallBlock:
    return ToolCallBlock(id=call_id, name=TOOL_NAME, arguments=json.dumps({"todos": todos}))


def _asked_for(session: Session, *calls: ToolCallBlock) -> None:
    """Commit the assistant message that requested these calls.

    The loop does this before executing any of them, and the parallel rule is
    read from it — so a test that skipped it would be testing a code path the
    harness never takes.
    """
    blocks = [call.model_dump(mode="json", by_alias=True) for call in calls]
    session.append(
        "assistant/message", assistant_payload("", "a1", content=blocks), SurfaceIntent("append")
    )


async def _run(ctx: Any, session: Session, *calls: ToolCallBlock) -> None:
    agent = StubAgent(ctx, session)
    _asked_for(session, *calls)
    await execute_tool_calls(ctx, agent, 1, 1, list(calls), CancelToken(), lambda _c: None)


# ----------------------------------------------------------------- the bundle --


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    """A row whose `name:` does not resolve fails at mount, in someone's session.

    Composed through the loader itself — this bundle's rows are plain, so it
    stands alone — rather than through a second hand parser of the bundle
    grammar that could drift from the loader's real one.
    """
    from ph.cordis.loader import resolve_plugin

    rows = Loader.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        assert resolve_plugin(row["name"]) is not None, row["name"]


async def test_every_enabled_row_in_the_profile_activates(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A row that mounts nothing is worse than one that fails: it looks fine.

    Mounted for real, because `Loader.inactive()` reads the forks `mount()`
    creates — asked before mounting, as this test's first version asked, it
    answers over an empty table and cannot fail.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))
    documents = Loader.from_paths(PROFILE).documents
    documents.append(("test-overlay", [dict(ENABLED)]))
    loader = Loader.from_documents(documents)
    ctx = Context()
    try:
        await loader.mount(ctx)
        assert loader.inactive() == []
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
    assert without.tools.get(TOOL_NAME) is None

    enabled = await mount(ENABLED, profile=PROFILE)
    assert enabled.tools.get(TOOL_NAME) is not None


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
    assert todos_of(session) == _todos(("survey", "in_progress"), ("port", "pending"))


async def test_a_second_call_replaces_the_whole_list(mount: Any) -> None:
    """Whole-list replacement, which is why two calls in one turn are ambiguous."""
    ctx = await mount(ENABLED, profile=PROFILE)
    session = ctx.sessions.create("planning")

    await _run(ctx, session, _call("c1", _todos(("survey", "in_progress"))))
    await _run(ctx, session, _call("c2", _todos(("survey", "completed"), ("port", "in_progress"))))

    assert todos_of(session) == _todos(("survey", "completed"), ("port", "in_progress"))
    assert todos_of(Session("empty")) == [], "a session that never planned has no list"


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
    assert todos_of(session) == _todos(("survey", "in_progress"))


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
    text = render_prompt(await ctx.system_prompt.assemble(AssembleContext()))

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
        return await ctx.system_prompt.assemble(AssembleContext(scope=agent.ctx, agent=agent))

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
