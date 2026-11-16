"""The live decision-model gate: a real server answering `/systemone`.

Opt-in, like the other live suites, and by its own variable: a decision model is
another server than the one that chats, on another port, so `LLAMA_BASE_URL`
says nothing about it.

    DECISION_BASE_URL=http://127.0.0.1:9932/v1 uv run pytest tests/test_local_decision.py

Two questions, the two the unit suites answer against a double. Does the route
decide — `ctx.llm.decide` through phern's OpenAI-compatible adapter, each kind
of question answered as the kind it is owed? And does `skill-steps` hold its
nudge for a turn that is waiting on a person, and send it for one that is not —
which is a claim about a model as much as about the row, and why it is asked of
a real one: the line a nudge is held at (`HOLD_AT`) was drawn through these
answers.

The stops are not close calls on purpose. This gate says the wiring and the
wording work; where the unsure middle falls is a model's to move, and a test
that pinned it would fail on the next quantization.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from local_server import local_model

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context
from ph.keys import AGENTS, LLM, SESSIONS
from ph.llm.adapter import (
    Choice,
    ChoiceAnswer,
    DecisionRequest,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
)
from ph.session import SurfaceIntent
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    StoppingAgent,
    assistant_payload,
    log_event,
    not_none,
    run_tool,
    write_skill,
)
from ph_stabilize import BUNDLE, skill_steps

pytestmark = [
    pytest.mark.anyio,
    pytest.mark.skipif(
        not os.environ.get("DECISION_BASE_URL"),
        reason="DECISION_BASE_URL is not set; the live decision-model gate is opt-in",
    ),
]

BASE_URL = os.environ.get("DECISION_BASE_URL", "")

STEPS = ["Run the migration against staging", "Apply the migration to production"]


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> str:
    """The model to ask for, discovered, with the route's credential defaulted."""
    return local_model(monkeypatch, base_url=BASE_URL, prefix="DECISION")


async def _deployment(mount: MountProfile, tmp_path: Path, model: str) -> Context:
    """Planning, the steering row asking the live route, and a two-step skill."""
    write_skill(
        tmp_path,
        "migrate",
        description="migrate a database",
        extra="steps:\n" + "".join(f"  - {one}\n" for one in STEPS),
        body="Do it.",
    )
    return await mount(
        {"id": "tool-todo", "disabled": False},
        {
            "id": "skill-steps",
            "disabled": False,
            "config": {"decision": {"provider": "decision", "model": model}},
        },
        {"id": "skills-progressive", "config": {"paths": [str(tmp_path)]}},
        {
            "id": "llm-decision",
            "name": "llm-openai-compatible",
            "config": {
                "profiles": [
                    {"provider": "decision", "baseUrl": BASE_URL, "apiKeyEnv": "DECISION_API_KEY"}
                ]
            },
        },
        profile=[BASE, HEADLESS, BUNDLE],
    )


async def test_the_route_answers_each_kind_of_question(
    mount: MountProfile, tmp_path: Path, model: str
) -> None:
    """The hookup: the adapter's request is one this server takes, and its reply
    one the adapter reads — a noul, a choice and a score in one request."""
    ctx = await _deployment(mount, tmp_path, model)
    llm = ctx.require(LLM)
    assert llm.decides("decision"), "the OpenAI-compatible adapter is one that decides"

    decided = await llm.decide(
        DecisionRequest(
            provider="decision",
            model=model,
            state="Customer message: I was charged twice last week and nobody has replied.",
            questions={
                "angry": Noul("Is the customer unhappy?"),
                "team": Choice(
                    "Which team should handle this?",
                    {"billing": "Charges, invoices and refunds", "shipping": "", "technical": ""},
                ),
                "urgency": Score("How urgent is it?", ("can wait", "soon", "right now")),
            },
        )
    )

    assert decided.answer("angry", NoulAnswer).noul > 0.5
    team = decided.answer("team", ChoiceAnswer)
    assert team.choice == "billing"
    assert sum(team.probabilities.values()) == pytest.approx(1.0, abs=1e-6)
    urgency = decided.answer("urgency", ScoreAnswer)
    assert len(urgency.probabilities) == 3 and 0.0 <= urgency.score <= 2.0
    assert decided.input_tokens, "and the server said what the request cost"


@pytest.mark.parametrize(
    ("said", "held"),
    [
        (
            "The dry run passed on staging with no errors. Before I apply this to production, "
            "do you approve? Reply yes and I will go on.",
            True,
        ),
        (
            "I found two ways to do this. Which do you prefer: (a) migrate in place, or "
            "(b) copy the table first?",
            True,
        ),
        ("The staging run finished and all 14 tables migrated. That covers the first step.", False),
        ("Done with staging. Next I will apply the migration to production.", False),
    ],
)
async def test_a_nudge_is_held_for_a_person_and_sent_otherwise(
    mount: MountProfile, tmp_path: Path, model: str, said: str, held: bool
) -> None:
    """The feature, end to end against a real model: a turn that stopped to ask
    is let end, and one that merely stopped is nudged on."""
    ctx = await _deployment(mount, tmp_path, model)
    session = ctx.require(SESSIONS).create("live-decision")
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    await run_tool(ctx, "skill", {"name": "migrate"}, agent=agent, session=session)
    log_event(session, "assistant/message", assistant_payload(said, "a9"), SurfaceIntent("append"))
    stopping = StoppingAgent(session)

    await ctx.serial("agent/turn-stopping", stopping, 1)

    waiting = not_none(session.latest(skill_steps.DECIDED)).data["waiting"]
    assert bool(stopping.steers) is not held, f"the model read waiting as {waiting}"
