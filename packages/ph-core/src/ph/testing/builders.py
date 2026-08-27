"""Builders for tests and fixtures.

Two families. **Payloads**: the three surface types have a shape every test
needs and none should retype — a user message, an assistant message inside its
`assistant/message` payload, a tool result inside its `tool/result` payload.
**Tool scaffolding**: a string-output tool, a bare registry, an agent stub, and
the fake-provider options. Each was being re-declared per test module.

@module ph.testing.builders
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..agent.types import AgentOptions
from ..cordis import Context
from ..session import Session
from ..tools.definition import ToolDefinition, ToolOutput, define_tool, text_content
from ..tools.registry import ToolRuntime

__all__ = [
    "FAKE_OPTIONS",
    "StubAgent",
    "assistant_payload",
    "raising",
    "run_tool",
    "simple_tool",
    "tool_result_payload",
    "tool_runtime",
    "user_payload",
]

FAKE_OPTIONS = AgentOptions(provider="fake", model="fake-1")
"""The options every test that drives the fake adapter uses."""


def simple_tool(
    name: str,
    execute: Callable[..., Any] | None = None,
    *,
    description: str | None = None,
    safe: bool | Callable[[Any], bool] = False,
    **kwargs: Any,
) -> ToolDefinition:
    """A tool taking a free-form object and returning a string.

    The shape every pipeline and batch test wants: no schema to satisfy, a
    string the assertions can read back. `execute` defaults to returning `name`;
    `safe` is the concurrency classification, a bool or a classifier.
    """
    return define_tool(
        name,
        description or f"the {name} tool",
        parameters={"type": "object", "properties": {}},
        output=ToolOutput(
            schema={"type": "string"}, render=lambda _args, value: text_content(value)
        ),
        execute=execute or (lambda _args, _run: name),
        is_concurrency_safe=safe,
        **kwargs,
    )


async def run_tool(
    ctx: Any,
    name: str,
    arguments: Any = None,
    *,
    agent: Any,
    session: Any = None,
    call_id: str = "call-1",
) -> Any:
    """Execute one tool the way the loop does, for a test that is not the loop.

    The `ToolExecutionInput(...)` incantation — `scope=agent.ctx`, `session=`,
    `agent=` — was written out at nine call sites across six files, which is
    nine places for a test to accidentally pass a different scope than the one
    whose policy it meant to exercise.
    """
    from ..tools.definition import ToolExecutionInput

    return await ctx.tools.execute(
        ToolExecutionInput(
            call_id=call_id,
            name=name,
            arguments={} if arguments is None else arguments,
            scope=getattr(agent, "ctx", None),
            session=session,
            agent=agent,
        )
    )


def raising(error: BaseException) -> Callable[..., Any]:
    """A body that raises `error` — readable where a generator trick was not."""

    def body(*_args: Any, **_kwargs: Any) -> Any:
        raise error

    return body


def tool_runtime() -> tuple[Context, ToolRuntime]:
    """A root context with a bare registry provided as `tools`."""
    root = Context()
    runtime = ToolRuntime(ctx=root)
    root.provide("tools", runtime)
    return root, runtime


class StubAgent:
    """The minimum an approval prompt or a tool call needs of an agent."""

    def __init__(
        self, ctx: Context | None = None, session: Session | None = None, agent_id: str = "agent-a"
    ) -> None:
        self.ctx = ctx if ctx is not None else Context()
        self.session = session
        self.id = agent_id
        self.options = FAKE_OPTIONS


def user_payload(text: str, message_id: str = "m1") -> dict[str, Any]:
    """A `user/message` payload for typed human text."""
    return {
        "id": message_id,
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "source": {"kind": "user"},
    }


def assistant_payload(
    text: str, message_id: str, *, turn: int = 1, step: int = 1, provider: str = "fake"
) -> dict[str, Any]:
    """An `assistant/message` payload; empty `text` gives an empty-content message."""
    return {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "assistant",
            "content": [{"type": "text", "text": text}] if text else [],
            "source": {"kind": "model", "provider": provider, "model": "m"},
        },
    }


def tool_result_payload(
    text: str, message_id: str, call_id: str = "c1", *, turn: int = 1, step: int = 1
) -> dict[str, Any]:
    """A `tool/result` payload carrying one text block."""
    return {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "user",
            "content": [
                {
                    "type": "tool-result",
                    "toolCallId": call_id,
                    "content": [{"type": "text", "text": text}],
                    "isError": False,
                }
            ],
            "source": {"kind": "tool", "callId": call_id},
        },
    }
