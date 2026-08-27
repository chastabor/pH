"""Repo-level scenarios: the whole stack, exercised the way a user would.

Phase 0's `Result` line is *"one-shot Q&A from the terminal; an inspectable
JSONL that dsh tooling can already read"*. These tests check the claims behind
that sentence end to end rather than per module.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.persistence.jsonl import read_session
from ph.session import Session

pytestmark = pytest.mark.anyio

FAKE = AgentOptions(provider="fake", model="fake-1")


def _replies(*replies: str) -> dict[str, Any]:
    return {"id": "llm-fake", "config": {"providers": ["fake"], "replies": list(replies)}}


async def test_a_session_survives_a_restart_and_continues(mount: Any, tmp_path: Path) -> None:
    ctx = await mount(_replies("first answer", "second answer"))
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, FAKE).prompt("first question")
    await ctx.sessions.flush(session)
    await ctx.dispose()

    # A second process opens the stored log and keeps going.
    header, events = read_session(tmp_path / "sessions" / "s.jsonl")
    resumed_ctx = await mount(_replies("second answer"))
    resumed = resumed_ctx.sessions.adopt(Session("s", seed=events, header=header))
    await resumed_ctx.agents.create(resumed, FAKE).prompt("second question")

    texts = [
        block.text
        for message in resumed.derive_messages()
        for block in message.content
        if getattr(block, "type", "") == "text"
    ]
    assert texts == ["first question", "first answer", "second question", "second answer"]
    # Turn numbering continues rather than restarting, so the trace reads as one
    # conversation.
    assert [e.data["turn"] for e in resumed.events if e.type == "turn/start"] == [1, 2]


async def test_a_fork_branches_without_touching_the_parent(mount: Any) -> None:
    ctx = await mount(_replies("answer"))
    parent = ctx.sessions.create("parent")
    await ctx.agents.create(parent, FAKE).prompt("shared question")

    child = ctx.sessions.fork(parent, parent.events[-1].seq, "child")
    await ctx.agents.create(child, FAKE).prompt("branch question")

    parent_texts = [m.content[0].text for m in parent.derive_messages()]
    child_texts = [m.content[0].text for m in child.derive_messages()]
    assert parent_texts == ["shared question", "answer"]
    assert child_texts[:2] == parent_texts
    assert child_texts[2] == "branch question"
    assert child.header.parent_session == "parent"


async def test_disposal_unwinds_the_whole_tree(mount: Any) -> None:
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE)
    await agent.prompt("hello")
    assert agent.ctx.active
    await ctx.dispose()
    # Nothing is left holding a service or a listener: cleanup is structural,
    # not remembered (invariant I2).
    assert not agent.ctx.active
    assert not ctx.active
    assert not ctx.has("sessions")


async def test_every_message_the_model_saw_is_in_the_log(mount: Any) -> None:
    ctx = await mount(_replies("answer"))
    session = ctx.sessions.create("s")
    await ctx.agents.create(session, FAKE).prompt("question")

    sent = {message.id for request in ctx.llm_fake.requests for message in request.messages}
    logged = {message.id for message in session.derive_messages()}
    # Invariant I3, checked over the whole run rather than per request.
    assert sent <= logged
