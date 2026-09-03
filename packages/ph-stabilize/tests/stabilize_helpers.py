"""Shared scaffolding for the `stabilize` bundle's tests.

A module of its own rather than `conftest.py`, mirroring `ph-app`'s
`tui_helpers`. This package has **no** conftest, and deliberately: `sys.modules`
holds one slot named `conftest`, so adding a second one that pytest loads after
ph-rlm's took the slot from it and every `from conftest import ...` in that
suite failed to collect. Its one piece of shared setup is a plain function rather than a
fixture, so a test calls it by name instead of importing a decorated symbol it
then shadows with a parameter of the same name.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from ph.bundles import BASE, HEADLESS
from ph.llm.types import ToolCallBlock, ToolResultBlock, text_of
from ph.seams.spill import SpillStore
from ph.session import derive_event_message
from ph.testing import FAKE_OPTIONS, StubWorkspaceProvider
from ph_stabilize import BUNDLE

__all__ = [
    "PROFILE",
    "answer_approvals",
    "bash_call",
    "blob",
    "break_spill",
    "events_of",
    "result_block",
    "result_text",
    "row",
    "run_tool_calls",
    "scoped_agent",
]

PROFILE = [BASE, HEADLESS, BUNDLE]
"""Base, the fake adapter, and this bundle — what a profile layering it gets."""


def blob(size: int, *, lines: int = 40) -> str:
    """Exactly `size` characters over about `lines` lines.

    Exact, because the boundary tests turn on one character either side of a
    threshold, and a generator that overshot would be testing a number nobody
    chose. The line breaks are what give `content_preview` a middle to omit.
    """
    chunk = max(1, size // lines)
    return "\n".join("x" * chunk for _ in range(lines)).ljust(size, "y")[:size]


def break_spill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every spill write fail, for the fail-open gates.

    Patched on the class: `SpillStore` is a slots dataclass, so the instance has
    no room for an override.
    """

    async def refuse(_self: Any, **_kwargs: Any) -> Any:
        raise OSError("no space left on device")

    monkeypatch.setattr(SpillStore, "save_text", refuse)


def events_of(session: Any, event_type: str) -> list[Any]:
    """Every event of one type. Written out in three test modules before this."""
    return [event for event in session.events if event.type == event_type]


async def run_tool_calls(ctx: Any, session: Any, *calls: Any, step: int = 1) -> Any:
    """Commit the assistant message that asked, then run the batch for real.

    The commit is load-bearing and is why this is shared: the loop appends the
    assistant message *before* dispatching any of its calls, and rows read it
    back — `tool-todo`'s parallel rule counts `write_todos` in it. A test that
    skipped it would exercise a path the harness never takes, and two copies of
    that invariant are one that gets fixed when the loop's ordering changes.

    Returns the `BatchOutcome`, whose `concluded` is how the loop learns a turn
    is over.
    """
    from ph.cancel import CancelToken
    from ph.session import SurfaceIntent
    from ph.testing import StubAgent, assistant_payload
    from ph.tools.batch import execute_tool_calls

    blocks = [call.model_dump(mode="json", by_alias=True) for call in calls]
    session.append(
        "assistant/message",
        assistant_payload("", f"a{step}", content=blocks),
        SurfaceIntent("append"),
    )
    return await execute_tool_calls(
        ctx, StubAgent(ctx, session), 1, step, list(calls), CancelToken(), lambda _c: None
    )


def row(plugin_id: str, **config: Any) -> dict[str, Any]:
    """One row, with its ceilings or its rules spelled out in the test."""
    return {"id": plugin_id, "config": config}


def bash_call(call_id: str, command: str = "true") -> ToolCallBlock:
    """The one tool every row in this bundle is exercised against."""
    return ToolCallBlock(id=call_id, name="bash", arguments=json.dumps({"command": command}))


def result_text(session: Any, call_id: str) -> str:
    """What the model reads back from one call.

    Through `derive_event_message` — THE projection — rather than by indexing
    `event.data["message"]["content"][0]`, for the reason `limits._result_facts`
    gives: a second route to that shape is one that keeps passing after the
    shape moves, and a test that reads the log by hand stops testing what the
    model sees.
    """
    block = result_block(session, call_id)
    return "" if block is None else text_of(block.content)


def answer_approvals(ctx: Any, answer: Any) -> list[Any]:
    """Register the answerer a front end would, and return what it was asked.

    `answer` may be a value or a zero-argument callable, so a test that changes
    its mind mid-run does not need a second helper.

    The returned list fills as prompts arrive. Half the questions a permission
    test asks are about *how many* prompts and *what they said* — "prompts zero
    times", "prompts once", "the prompt names the boundary" — and without this
    they were being answered three ways: an ad-hoc answerer, a
    `lambda: asked.append(...) or "allowed-once"` smuggled through the callable
    escape hatch, and reading `approval/asked` back out of the log.
    """
    asked: list[Any] = []

    async def respond(request: Any, _next: Any = None) -> Any:
        asked.append(request)
        chosen = answer() if callable(answer) else answer
        # An awaitable answer is one that waits — for an event a test sets once
        # it has looked at the log mid-flight, say.
        return await chosen if inspect.isawaitable(chosen) else chosen

    ctx.approval.register_answerer(respond)
    return asked


async def scoped_agent(ctx: Any, tmp_path: Path, *, session_id: str = "s1") -> tuple[Any, Any]:
    """`(agent, workspace)` — an agent holding a workspace of its own.

    A stub tier rather than real git: what a permission rule asks is "is this
    path inside the workspace the seam handed out", which is true of every tier
    that hands out a fresh root. `test_workspace_git.py` owns the checkout.
    """
    session = ctx.sessions.create(session_id)
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    ctx.workspace.register_provider(StubWorkspaceProvider(root=tmp_path / "trees"))
    workspace = await ctx.workspace.acquire(
        session_id=session_id,
        agent_id=agent.id,
        base=tmp_path / "project",
        session=session,
    )
    return agent, workspace


def result_block(session: Any, call_id: str) -> Any:
    """The `ToolResultBlock` one call produced, or `None`.

    Split out of `result_text` so a test asserting `is_error` reaches it through
    `derive_event_message` too, rather than indexing
    `event.data["message"]["content"][0]` by hand — the second route to a shape
    is the one that keeps passing after the shape moves.
    """
    for event in events_of(session, "tool/result"):
        message = derive_event_message(event)
        for block in message.content if message else ():
            if isinstance(block, ToolResultBlock) and block.tool_call_id == call_id:
                return block
    return None
