"""What every ph-clm suite builds on: a mounted bundle, an agent, a conversation."""

from __future__ import annotations

import itertools
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from typing import Any

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.keys import AGENTS, LLM_FAKE, SESSIONS, TOKEN_METER
from ph.llm.types import GenerateOptions, StreamChunk
from ph.session import Session, SurfaceIntent
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    assistant_payload,
    log_event,
    run_tool,
    tool_call_chunks,
    tool_result_payload,
    user_payload,
)
from ph.tools.definition import ToolExecutionResult
from ph_clm.sections import Section, SectionMap, render_messages

__all__ = [
    "CALL_ID",
    "MIRROR_ROW",
    "OUTPUT",
    "ROW",
    "agent_in",
    "answer_first_request",
    "converse",
    "edit",
    "human_text",
    "model_text",
    "mounted",
    "sections_of",
]

ROW: dict[str, Any] = {"id": "clm-context", "name": "clm-context"}
MIRROR_ROW: dict[str, Any] = {"id": "clm-mirror", "name": "clm-mirror"}

OUTPUT = "parser.py line 1\n" + "noise from a long listing\n" * 60
"""A tool output long enough that removing it is worth something."""

CALL_ID = "t1"
"""The id of the one call `answer_first_request` scripts."""

_SEQ = itertools.count()


async def mounted(mount: MountProfile, *extra: dict[str, Any], **config: object) -> Context:
    """The `clm-context` row, with `config`, and any rows after it."""
    return await mount({**ROW, "config": config} if config else ROW, *extra)


def agent_in(ctx: Context) -> ReactLoopAgent:
    """A fresh agent, on a session id nothing else has taken."""
    agent = ctx.require(AGENTS).create(
        ctx.require(SESSIONS).create(f"clm-{next(_SEQ)}"), FAKE_OPTIONS
    )
    assert isinstance(agent, ReactLoopAgent)
    return agent


def converse(session: Session) -> None:
    """task · step(read) · reply · user · step(grep) · reply."""
    log_event(
        session, "user/message", user_payload("fix the parser", "m1"), SurfaceIntent("append")
    )
    call = {"type": "tool-call", "id": "c1", "name": "read", "arguments": '{"path": "parser.py"}'}
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m2", content=[call]),
        SurfaceIntent("append"),
    )
    log_event(
        session, "tool/result", tool_result_payload(OUTPUT, "m3", "c1"), SurfaceIntent("append")
    )
    log_event(
        session,
        "assistant/message",
        assistant_payload("the bug is on line 1 of parser.py", "m4"),
        SurfaceIntent("append"),
    )
    log_event(
        session, "user/message", user_payload("now run the tests", "m5"), SurfaceIntent("append")
    )
    grep = {"type": "tool-call", "id": "c2", "name": "grep", "arguments": '{"pattern": "def"}'}
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m6", content=[grep]),
        SurfaceIntent("append"),
    )
    log_event(
        session,
        "tool/result",
        tool_result_payload("3 matches", "m7", "c2"),
        SurfaceIntent("append"),
    )
    log_event(
        session, "assistant/message", assistant_payload("all green", "m8"), SurfaceIntent("append")
    )


def sections_of(ctx: Context, session: Session) -> tuple[Section, ...]:
    """The session's section map, built fresh."""
    return SectionMap(ctx.require(TOKEN_METER), protect_task=True)(session)


def model_text(session: Session) -> str:
    """What the model is sent, tool output included."""
    return render_messages(session.derive_messages())


def human_text(session: Session) -> str:
    """What the person's transcript shows."""
    return render_messages(session.transcript())


async def edit(
    ctx: Context, agent: ReactLoopAgent, name: str, arguments: dict[str, object]
) -> ToolExecutionResult:
    """One context tool call, the way the loop makes it."""
    return await run_tool(ctx, name, arguments, agent=agent, session=agent.session)


def answer_first_request(
    ctx: Context, name: str, arguments: Callable[[], dict[str, object]]
) -> None:
    """Script the model: the loop's first request is answered with one call to `name`,
    its arguments built at that moment; every later request goes to the fake."""
    asked = False

    async def first(options: GenerateOptions, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        nonlocal asked
        if options.is_loop_request and not asked:
            asked = True
            ctx.require(LLM_FAKE).requests.append(options)
            return _stream(tool_call_chunks(CALL_ID, name, json.dumps(arguments())))
        return await next_(options)

    ctx.on("llm/stream", first)


async def _stream(chunks: Iterable[StreamChunk]) -> AsyncIterator[StreamChunk]:
    for chunk in chunks:
        yield chunk
