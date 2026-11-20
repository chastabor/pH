"""Shared scaffolding for the tests of the `stabilize` bundle.

This is a module of its own and not a `conftest.py`, like the `tui_helpers` of
`phern`. This package deliberately has **no** conftest. `sys.modules` holds one
slot named `conftest`. When a second conftest was added here, pytest loaded it
after the conftest that ph-rlm had at the time, and it took the slot. Every
`from conftest import ...` in that suite then failed to collect. The shared
setup here is plain functions and not fixtures. A test calls a function by name,
and does not import a decorated symbol that it then shadows with a parameter of
the same name.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from ph.bundles import BASE, HEADLESS
from ph.cordis import Context
from ph.keys import AGENTS, APPROVAL, SESSIONS, WORKSPACE
from ph.llm.types import ToolCallBlock, ToolResultBlock
from ph.seams import spill
from ph.session import Session, derive_event_message
from ph.testing import FAKE_OPTIONS, StubWorkspaceProvider, full_disk_batch, log_event
from ph_stabilize import BUNDLE

__all__ = [
    "PROFILE",
    "answer_approvals",
    "bash_call",
    "blob",
    "break_spill",
    "events_of",
    "result_block",
    "row",
    "run_tool_calls",
    "scoped_agent",
    "todo_call",
]

PROFILE = [BASE, HEADLESS, BUNDLE]
"""Base, the fake adapter, and this bundle: what a profile that layers the bundle
gets."""


def blob(size: int, *, lines: int = 40) -> str:
    """Exactly `size` characters over about `lines` lines.

    The size is exact because the boundary tests depend on one character on
    either side of a threshold. A generator that overshot would test a number
    that nobody chose. The line breaks give `content_preview` a middle to omit.
    """
    chunk = max(1, size // lines)
    return "\n".join("x" * chunk for _ in range(lines)).ljust(size, "y")[:size]


def break_spill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every spill write fail, for the fail-open gates.

    The store writes through `write_atomic_all`, and every `try_save*` reaches
    it. A full disk therefore fails there, with one error for each blob. It is
    deliberately the call that every producer makes *before* it has appended
    anything. That ordering is what keeps the fallback available. See
    `SpillStore.try_save`.
    """
    monkeypatch.setattr(spill, "write_atomic_all", full_disk_batch)


def events_of(session: Session, event_type: str) -> list[Any]:
    """Every event of one type. Three test modules each wrote this out before it
    was shared."""
    return [event for event in session.events if event.type == event_type]


async def run_tool_calls(ctx: Context, session: Session, *calls: Any, step: int = 1) -> Any:  # noqa: ANN401
    """Commit the assistant message that asked, then run the batch for real.

    The commit is necessary, and it is the reason why this helper is shared.
    The loop appends the assistant message *before* it dispatches any of the
    calls in it, and rows read the message back. For example, the parallel rule
    of `tool-todo` counts the `write_todos` calls in it. A test that skipped the
    commit would exercise a path that the harness never takes. With two copies
    of that invariant, only one would be fixed when the ordering of the loop
    changes.

    Returns the `BatchOutcome`. Its `concluded` field tells the loop that a
    turn is over.
    """
    from ph.cancel import CancelToken
    from ph.session import SurfaceIntent
    from ph.testing import StubAgent, assistant_payload
    from ph.tools.batch import execute_tool_calls

    blocks = [call.model_dump(mode="json", by_alias=True) for call in calls]
    log_event(
        session,
        "assistant/message",
        assistant_payload("", f"a{step}", content=blocks),
        SurfaceIntent("append"),
    )
    return await execute_tool_calls(
        ctx, StubAgent(ctx, session), 1, step, list(calls), CancelToken(), lambda _c: None
    )


def row(plugin_id: str, **config: object) -> dict[str, Any]:
    """One row, with its ceilings or its rules spelled out in the test."""
    return {"id": plugin_id, "config": config}


def bash_call(call_id: str, command: str = "true") -> ToolCallBlock:
    """The one tool every row in this bundle is exercised against."""
    return ToolCallBlock(id=call_id, name="bash", arguments=json.dumps({"command": command}))


def todo_call(call_id: str, todos: list[dict[str, Any]]) -> ToolCallBlock:
    """One `write_todos` call. It is beside `bash_call` for the same reason.

    Two test modules each wrote this out identically before it was shared. The
    tool takes the ENTIRE list in one argument, so a change to the shape of that
    argument must reach both suites at the same time. With two copies, one of
    them would keep passing against a call that the harness no longer makes.
    """
    from ph_stabilize.todo import TOOL_NAME

    return ToolCallBlock(id=call_id, name=TOOL_NAME, arguments=json.dumps({"todos": todos}))


def answer_approvals(ctx: Context, answer: Any) -> list[Any]:  # noqa: ANN401
    """Register the answerer that a front end would register, and return what it
    was asked.

    `answer` may be a value or a zero-argument callable, so a test that changes
    its answer during a run does not need a second helper.

    The returned list fills as prompts arrive. Half the questions that a
    permission test asks are about *how many* prompts there were and *what they
    said*: "prompts zero times", "prompts once", "the prompt names the
    boundary". Without this helper, tests answered those questions in three
    ways:

    - An ad-hoc answerer.
    - A `lambda: asked.append(...) or "allowed-once"` passed in as the callable
      `answer`.
    - A read of `approval/asked` back out of the log.
    """
    asked: list[Any] = []

    async def respond(request: object, _next: object = None) -> Any:  # noqa: ANN401
        asked.append(request)
        chosen = answer() if callable(answer) else answer
        # An awaitable answer is an answer that waits. For example, it waits
        # for an event that a test sets after it has looked at the log during
        # the run.
        return await chosen if inspect.isawaitable(chosen) else chosen

    ctx.require(APPROVAL).register_answerer(respond)
    return asked


async def scoped_agent(
    ctx: Context,
    tmp_path: Path,
    *,
    session_id: str = "s1",
) -> tuple[Any, Any]:
    """`(agent, workspace)`: an agent that holds a workspace of its own.

    This uses a stub tier and not real git. A permission rule asks "is this
    path inside the workspace the seam handed out", and that holds for every
    tier that hands out a fresh root. `test_workspace_git.py` tests the
    checkout.
    """
    session = ctx.require(SESSIONS).create(session_id)
    agent = ctx.require(AGENTS).create(session, FAKE_OPTIONS)
    ctx.require(WORKSPACE).register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    workspace = await ctx.require(WORKSPACE).acquire(
        session_id=session_id,
        agent_id=agent.id,
        base=tmp_path / "project",
        session=session,
    )
    return agent, workspace


def result_block(session: Session, call_id: str) -> Any:  # noqa: ANN401
    """The `ToolResultBlock` that one call produced, or `None`.

    This was split out of `result_text`, so that a test that asserts `is_error`
    also reaches the block through `derive_event_message`. The alternative is to
    index `event.data["message"]["content"][0]` by hand, and a second route to a
    shape keeps passing after the shape moves.
    """
    for event in events_of(session, "tool/result"):
        message = derive_event_message(event)
        for block in message.content if message else ():
            if isinstance(block, ToolResultBlock) and block.tool_call_id == call_id:
                return block
    return None
