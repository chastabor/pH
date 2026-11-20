"""The base for every ph-clm test suite: a mounted bundle, an agent, and a conversation."""

from __future__ import annotations

import itertools
import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.keys import AGENTS, LLM_FAKE, SESSIONS, TOKEN_METER
from ph.llm.types import GenerateOptions
from ph.seams.token_meter import TokenMeter
from ph.session import Session, SurfaceIntent, render_messages
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    assistant_payload,
    log_event,
    run_tool,
    streamed,
    tool_call_chunks,
    tool_result_payload,
    user_payload,
)
from ph.tools.definition import ToolExecutionResult
from ph_clm.sections import Section, SectionMap

__all__ = [
    "CALL_ID",
    "MIRROR_ROW",
    "OUTPUT",
    "ROW",
    "agent_in",
    "answer_first_request",
    "context_tokens",
    "converse",
    "edit",
    "filler",
    "human_text",
    "model_text",
    "mounted",
    "readout_share",
    "sections_of",
    "without_section",
]

ROW: dict[str, Any] = {"id": "clm-context", "name": "clm-context"}
MIRROR_ROW: dict[str, Any] = {"id": "clm-mirror", "name": "clm-mirror"}

OUTPUT = "parser.py line 1\n" + "noise from a long listing\n" * 60
"""A tool output that is long enough to make its removal worthwhile."""

CALL_ID = "t1"
"""The id of the one call that `answer_first_request` scripts."""

_SEQ = itertools.count()


async def mounted(mount: MountProfile, *extra: dict[str, Any], **config: object) -> Context:
    """Mount the `clm-context` row with `config`, and any rows after it."""
    return await mount({**ROW, "config": config} if config else ROW, *extra)


def agent_in(ctx: Context) -> ReactLoopAgent:
    """Create a fresh agent, on a session id that nothing else has taken."""
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


def without_section(text: str, name: str) -> str:
    """A context file with the block of one section cut out: an edit that removes it."""
    return re.sub(rf"\n\[\[SECTION {name} .*?(?=\n\[\[SECTION |\Z)", "", text, flags=re.S)


def sections_of(ctx: Context, session: Session) -> tuple[Section, ...]:
    """The section map of the session, built fresh."""
    return SectionMap(ctx.require(TOKEN_METER), protect_task=True)(session)


def model_text(session: Session) -> str:
    """The text that the model is sent, including tool output."""
    return render_messages(session.derive_messages())


def human_text(session: Session) -> str:
    """The text that the transcript shows to the user."""
    return render_messages(session.transcript())


async def edit(
    ctx: Context, agent: ReactLoopAgent, name: str, arguments: dict[str, object]
) -> ToolExecutionResult:
    """Make one context tool call, in the way that the loop makes it."""
    return await run_tool(ctx, name, arguments, agent=agent, session=agent.session)


def answer_first_request(
    ctx: Context, name: str, arguments: Callable[[], dict[str, object]]
) -> None:
    """Script the model. The first request of the loop is answered with one call to
    `name`, and its arguments are built at that moment. Every later request goes
    to the fake."""
    asked = False

    async def first(options: GenerateOptions, next_: Callable[..., Awaitable[Any]]) -> Any:  # noqa: ANN401
        nonlocal asked
        if options.is_loop_request and not asked:
            asked = True
            ctx.require(LLM_FAKE).requests.append(options)
            return streamed(tool_call_chunks(CALL_ID, name, json.dumps(arguments())))
        return await next_(options)

    ctx.on("llm/stream", first)


def context_tokens(ctx: Context, session: Session) -> int:
    """The size of the context, as compaction, the `fit` gate, and a readout count it."""
    return ctx.require(TOKEN_METER).baseline(session).tokens


def filler(meter: TokenMeter, tokens: int, unit: str = "word ") -> str:
    """Text that the meter counts as at least `tokens`, in whole `unit`s. It starts
    with as many units as the estimate for one unit suggests, then adds one unit
    at a time until the meter agrees."""
    text = unit * (tokens // max(1, meter.measure_text(unit)))
    while meter.measure_text(text) < tokens:
        text += unit
    return text


def readout_share(line: str) -> int:
    """The share of the window that a readout line reports: `(57%)` → 57."""
    found = re.search(r"tokens \((\d+)%\)", line)
    assert found is not None, line
    return int(found.group(1))
