"""The planner and auto-refine: P3-16's second increment.

The first increment could apply a `RefinementProposal`; this is where one comes
from. Two claims carry the module:

* **A refinement is not a turn.** The planner's prompt is *about* the
  conversation rather than part of it, so its call names `purpose="refine"`,
  logs no message, and is exempt from the "model-visible means logged" invariant
  by declaration rather than by omission.
* **When it runs unasked is a fold.** `due()` reads turns, elapsed time and
  compaction off the log, using the triggering event's own timestamp as the
  clock — so the decision is the same on a resumed session, on a fork, and here,
  with nothing to freeze.

No kernel is mounted: nothing here runs a cell, and H1's probe has its own tests
in `test_harness.py`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest
from conftest import HARNESS_ROW

from ph.session import SurfaceIntent
from ph.session.events import SessionEvent, SurfaceReplace
from ph.testing import FAKE_OPTIONS, user_payload
from ph_rlm.harness import (
    CONSIDERED,
    REFINED,
    HarnessEdit,
    PlannerError,
    RefinementProposal,
    due,
)
from ph_rlm.harness.planner import (
    REFINEMENT_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    parse_json_object,
)

pytestmark = pytest.mark.anyio

Refining = Callable[..., Any]

PROPOSAL = {
    "summary": "learned how this project runs its tests",
    "rationale": "the user corrected the command twice",
    "expectedOutcome": "the next session runs them the right way first",
    "edits": [
        {
            "action": "create",
            "kind": "procedure",
            "id": "run-the-tests",
            "title": "running the tests",
            "content": "uv run pytest",
            "reason": "stated twice",
        }
    ],
}
YES = json.dumps({"shouldRefine": True, "rationale": "a procedure was established"})
NO = json.dumps({"shouldRefine": False, "rationale": "routine work"})


@pytest.fixture
def refining(mount: Any) -> Refining:
    """`await refining(**config)` → `(ctx, session, agent)` with a scripted model."""

    async def build(**config: Any) -> tuple[Any, Any, Any]:
        row = {**HARNESS_ROW, "config": config} if config else HARNESS_ROW
        ctx = await mount(row)
        session = ctx.sessions.create("planning")
        return ctx, session, ctx.agents.create(session, FAKE_OPTIONS)

    return build


def script(ctx: Any, *, review: str = NO, planner: str = "{}") -> list[Any]:
    """Answer the review gate and the planner differently, and keep the requests."""

    def respond(request: Any) -> str:
        return review if request.system == REVIEW_SYSTEM_PROMPT else planner

    ctx.llm_fake.respond = respond
    return ctx.llm_fake.requests


def turn(session: Any, index: int = 1) -> None:
    session.append("turn/end", {"turn": index, "reason": {"kind": "completed"}})


def say(session: Any, text: str, message_id: str = "m1") -> Any:
    return session.append("user/message", user_payload(text, message_id), SurfaceIntent("append"))


# ------------------------------------------------------------ the JSON --


def test_a_fenced_reply_still_parses() -> None:
    """ "Return only JSON" is an instruction, not a guarantee."""
    parsed = parse_json_object('Sure!\n```json\n{"summary": "x", "edits": []}\n```\nDone.')
    assert parsed == {"summary": "x", "edits": []}


def test_a_reply_that_is_not_an_object_is_refused() -> None:
    """A list would otherwise validate as an empty proposal and half-apply."""
    with pytest.raises(PlannerError, match="JSON object"):
        parse_json_object("[1, 2, 3]")


# ---------------------------------------------------------- the request --


async def test_the_planner_call_is_not_a_turn(refining: Refining) -> None:
    """`purpose="refine"`: session-bound so usage is attributed, outside the loop
    so the invariant does not hold its messages to `derive_messages()`."""
    ctx, session, agent = await refining()
    requests = script(ctx, planner=json.dumps(PROPOSAL))
    say(session, "hello")
    before = len(session.events)

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    planner_request = next(one for one in requests if one.system == REFINEMENT_SYSTEM_PROMPT)
    assert planner_request.purpose == "refine"
    assert planner_request.session_id == session.id
    assert planner_request.is_loop_request is False
    # `min(model max, 32_000)`; the fake adapter declares no default, so the cap.
    assert planner_request.max_tokens == 32_000

    # Nothing of the conversation's own vocabulary was written by the pass.
    added = {event.type for event in session.events[before:]}
    assert added & {"assistant/message", "assistant/chunk", "turn/start", "turn/end"} == set()
    assert REFINED in added


async def test_the_planner_prompt_carries_the_state_the_history_and_the_tail(
    refining: Refining,
) -> None:
    ctx, session, agent = await refining()
    requests = script(ctx, planner=json.dumps(PROPOSAL))
    await ctx.harness.apply(
        RefinementProposal(
            summary="an earlier lesson",
            edits=[
                HarnessEdit(
                    action="create",
                    kind="note",
                    id="prefer-uv",
                    title="prefer uv",
                    content="the project uses uv, not pip",
                )
            ],
        ),
        session=session,
        agent=agent,
    )
    say(session, "run the tests")

    await ctx.commands.dispatch("/refine focus on the test commands", session=session, agent=agent)
    await ctx.drain()

    prompt = (
        next(one for one in requests if one.system == REFINEMENT_SYSTEM_PROMPT)
        .messages[0]
        .content[0]
        .text
    )
    # What it already knows, so the model updates rather than duplicates.
    assert "[local:prefer-uv] prefer uv" in prompt
    assert "the project uses uv, not pip" in prompt
    assert "an earlier lesson" in prompt
    # What the user asked for, named as theirs rather than mixed into the tail.
    assert "# What the user asked for\n\nfocus on the test commands" in prompt
    assert "You are editing the **local** harness." in prompt
    assert "user: run the tests" in prompt


async def test_only_the_tail_of_a_long_conversation_is_sent(refining: Refining) -> None:
    """The planner reads the end of the conversation, not a truncated beginning:
    what was just learned is what is worth keeping."""
    ctx, session, agent = await refining(conversationChars=200)
    requests = script(ctx, planner=json.dumps(PROPOSAL))
    say(session, "x" * 5_000, "m1")
    say(session, "the last thing said", "m2")

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    prompt = (
        next(one for one in requests if one.system == REFINEMENT_SYSTEM_PROMPT)
        .messages[0]
        .content[0]
        .text
    )
    assert "the last thing said" in prompt
    assert "x" * 500 not in prompt


async def test_the_conversation_is_what_the_model_saw(refining: Refining) -> None:
    """`derive_messages()`, not the raw log: a compacted range reaches the planner
    as its summary, so it cannot learn from history the agent was told to forget.
    """
    ctx, session, agent = await refining()
    requests = script(ctx, planner=json.dumps(PROPOSAL))
    first = say(session, "the forgotten original", "m1")
    session.append(
        "user/message",
        user_payload("(summary of earlier conversation)", "m2"),
        SurfaceIntent(SurfaceReplace(replaces=(first.seq,)), (first.seq,)),
    )

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    prompt = (
        next(one for one in requests if one.system == REFINEMENT_SYSTEM_PROMPT)
        .messages[0]
        .content[0]
        .text
    )
    assert "(summary of earlier conversation)" in prompt
    assert "the forgotten original" not in prompt


# ------------------------------------------------------------ the command --


async def test_the_command_schedules_a_job_and_the_refinement_lands(refining: Refining) -> None:
    """A planner pass outlives the keystroke that asked for it — `ctx.jobs`' own
    example — so the command answers with the job and the outcome arrives in the
    log."""
    ctx, session, agent = await refining()
    script(ctx, planner=json.dumps(PROPOSAL))

    shown = await ctx.commands.dispatch("/refine", session=session, agent=agent)
    assert shown is not None and "background" in shown and "refine-" in shown
    # Nothing has been applied yet: the command returned before the model did.
    assert ctx.harness.state(session).entry("procedure", "run-the-tests") is None

    await ctx.drain()
    entry = ctx.harness.state(session).entry("procedure", "run-the-tests")
    assert entry is not None and entry.content == "uv run pytest"


async def test_show_prints_what_the_bound_hides(refining: Refining) -> None:
    """`--show` is unbounded on purpose: the entries the prompt section elides are
    exactly the ones a human cannot see any other way."""
    ctx, session, agent = await refining(maxPerKind=1)
    await ctx.harness.apply(
        RefinementProposal(
            summary="two notes",
            edits=[
                HarnessEdit(action="create", kind="note", id="aaa", title="first", content="x"),
                HarnessEdit(action="create", kind="note", id="zzz", title="second", content="y"),
            ],
        ),
        session=session,
        agent=agent,
    )

    shown = await ctx.commands.dispatch("/refine --show", session=session, agent=agent)
    assert shown is not None
    assert "[local:aaa] first" in shown and "[local:zzz] second" in shown
    assert "more" not in shown
    assert ctx.llm_fake.requests == [], "--show asked a model anything"


async def test_show_on_an_empty_harness_says_so(refining: Refining) -> None:
    ctx, session, agent = await refining()
    assert await ctx.commands.dispatch("/refine --show", session=session, agent=agent) == (
        "the harness is empty"
    )


async def test_a_user_pass_skips_the_review_gate(refining: Refining) -> None:
    """The human already decided. Asking a model whether to do what they asked
    would spend a call to second-guess them."""
    ctx, session, agent = await refining()
    requests = script(ctx, review=NO, planner=json.dumps(PROPOSAL))

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    assert [one.system for one in requests] == [REFINEMENT_SYSTEM_PROMPT]
    assert ctx.harness.state(session).entry("procedure", "run-the-tests") is not None


async def test_a_pass_that_proposes_nothing_is_recorded_and_visible(refining: Refining) -> None:
    """ "Nothing worth keeping" is a correct answer, and the user who typed
    `/refine` has to be able to see it."""
    ctx, session, agent = await refining()
    script(ctx, planner=json.dumps({"summary": "nothing durable here", "edits": []}))

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    considered = [event for event in session.events if event.type == CONSIDERED]
    assert len(considered) == 1
    assert considered[0].data["trigger"] == "user"
    assert considered[0].data["reason"] == "nothing durable here"
    assert [event for event in session.events if event.type == REFINED] == []


async def test_a_planner_that_returns_junk_records_it_instead_of_raising(
    refining: Refining,
) -> None:
    """The consideration is what advances the cooldown, so a broken planner costs
    one call rather than one per turn."""
    ctx, session, agent = await refining()
    script(ctx, planner="I'm afraid I can't do that.")

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    (considered,) = [event for event in session.events if event.type == CONSIDERED]
    assert "JSON object" in considered.data["reason"]


async def test_a_second_pass_is_refused_while_one_is_running(refining: Refining) -> None:
    ctx, session, agent = await refining()
    script(ctx, planner=json.dumps(PROPOSAL))

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    second = await ctx.commands.dispatch("/refine", session=session, agent=agent)
    assert second is not None and "already running" in second
    await ctx.drain()
    assert len([event for event in session.events if event.type == REFINED]) == 1


async def test_a_finished_pass_leaves_no_job_behind(refining: Refining) -> None:
    """Released, not abandoned (`jobs.forget`): the work is over, so the entry
    goes. Otherwise an auto-refining session accretes one job per pass."""
    ctx, session, agent = await refining()
    script(ctx, planner=json.dumps(PROPOSAL))

    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    assert [job.kind for job in ctx.jobs.list()] == ["refine"]

    await ctx.drain()
    assert ctx.jobs.list() == []
    # Forgotten because it finished, not cancelled: the refinement still landed.
    assert [event for event in session.events if event.type == REFINED] != []


# ------------------------------------------------------------ the trigger --


class _Log:
    """A log prefix, for the parts of `due()` that are pure fold."""

    def __init__(self, events: list[SessionEvent]) -> None:
        self.events = events


def _turns(count: int, *, start_ms: int = 0, step_ms: int = 1_000) -> list[SessionEvent]:
    return [
        SessionEvent(type="turn/end", seq=index, time=start_ms + index * step_ms, data={})
        for index in range(count)
    ]


def test_the_turn_threshold_is_what_triggers_a_pass() -> None:
    assert due(_Log(_turns(24)), turns_between=25) is None
    assert due(_Log(_turns(25)), turns_between=25) == "turns"


def test_a_recent_consideration_holds_the_next_one_off() -> None:
    """The cooldown runs from the triggering event's own timestamp, so this is a
    fold rather than a race with the wall clock."""
    considered = SessionEvent(type=CONSIDERED, seq=0, time=0, data={})
    minute = 60_000
    within = [considered, *_turns(30, start_ms=minute, step_ms=minute // 2)]
    assert due(_Log(within), cooldown_minutes=20) is None

    past = [considered, *_turns(30, start_ms=minute, step_ms=minute)]
    assert due(_Log(past), cooldown_minutes=20) == "turns"


async def test_a_refinement_itself_starts_the_cooldown(refining: Refining) -> None:
    """An explicit `/refine` should quiet the automatic pass exactly as a
    declined review does — both are "we just looked at this"."""
    ctx, session, agent = await refining()
    script(ctx, planner=json.dumps(PROPOSAL))
    for index in range(30):
        turn(session, index)
    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()

    assert [event for event in session.events if event.type == REFINED] != []
    assert due(session) is None


def test_a_compaction_triggers_a_pass_on_its_own() -> None:
    """The one moment the conversation gets *shorter*: what the summary dropped is
    what the harness should have kept, and after it nobody can read it."""
    # The compaction's own record, not a bare surface `replace`: `input-offload`
    # (P4-02) also replaces on the surface, and a paste being relocated teaches
    # the harness nothing. Written as this event because that is what
    # `compaction-summarize` appends beside its replacement (P4-03).
    compaction = SessionEvent(type="compaction/summarized", seq=0, time=0, data={})
    assert due(_Log([compaction, *_turns(2, start_ms=1_000)]), turns_between=25) == "compaction"


# --------------------------------------------------------- auto-refine --


async def test_auto_refine_fires_at_the_threshold_and_stays_local(refining: Refining) -> None:
    """H7. Local, always: an automatic global edit would put an approval prompt in
    front of a user who asked for nothing, about a change reaching every project.
    """
    ctx, session, _agent = await refining(turnsBetweenRefinements=3, cooldownMinutes=0)
    script(ctx, review=YES, planner=json.dumps(PROPOSAL))

    for index in range(3):
        turn(session, index)
        await ctx.drain()

    entry = ctx.harness.state(session).entry("procedure", "run-the-tests")
    assert entry is not None and entry.scope == "local"
    # The gate ran first, and only then the planner.
    assert [one.system for one in ctx.llm_fake.requests] == [
        REVIEW_SYSTEM_PROMPT,
        REFINEMENT_SYSTEM_PROMPT,
    ]


async def test_a_declined_review_costs_one_cheap_call(refining: Refining) -> None:
    ctx, session, _agent = await refining(turnsBetweenRefinements=3, cooldownMinutes=0)
    script(ctx, review=NO)

    for index in range(6):
        turn(session, index)
        await ctx.drain()

    assert [one.system for one in ctx.llm_fake.requests].count(REFINEMENT_SYSTEM_PROMPT) == 0
    considered = [event for event in session.events if event.type == CONSIDERED]
    assert [one.data["trigger"] for one in considered] == ["turns", "turns"]
    assert considered[0].data["reason"] == "routine work"


async def test_a_veto_stops_it_before_any_model_call(refining: Refining) -> None:
    """`harness/before-refine` runs first, so a refusal spends no tokens — and is
    recorded, because a refusal nobody can see is indistinguishable from a pass
    that never fired."""
    ctx, session, agent = await refining(turnsBetweenRefinements=1, cooldownMinutes=0)
    script(ctx, review=YES, planner=json.dumps(PROPOSAL))

    async def refuse(request: Any, _next: Any) -> str:
        return f"this deployment does not refine (trigger: {request.trigger})"

    ctx.on("harness/before-refine", refuse)

    turn(session)
    await ctx.drain()
    assert ctx.llm_fake.requests == []
    (auto,) = [event for event in session.events if event.type == CONSIDERED]
    assert auto.data["reason"] == "vetoed: this deployment does not refine (trigger: turns)"

    # And the trigger is on the payload, so a listener can tell the automatic
    # passes from a human's `/refine`.
    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()
    assert ctx.llm_fake.requests == []
    asked = [event for event in session.events if event.type == CONSIDERED][-1]
    assert asked.data["reason"] == "vetoed: this deployment does not refine (trigger: user)"


async def test_auto_refine_can_be_turned_off(refining: Refining) -> None:
    ctx, session, agent = await refining(autoRefine=False, turnsBetweenRefinements=1)
    script(ctx, review=YES, planner=json.dumps(PROPOSAL))

    for index in range(3):
        turn(session, index)
        await ctx.drain()

    assert ctx.llm_fake.requests == []
    # The command still works: turning the automatic pass off is not turning the
    # harness off.
    await ctx.commands.dispatch("/refine", session=session, agent=agent)
    await ctx.drain()
    assert ctx.harness.state(session).entry("procedure", "run-the-tests") is not None
