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
from ..cordis import DEPLOYMENT, Boundary, Context
from ..llm.types import ContextForm, PluginSource
from ..session import Session
from ..tools.definition import ToolDefinition, ToolOutput, define_tool, text_content
from ..tools.registry import ToolRuntime

__all__ = [
    "FAKE_OPTIONS",
    "StubAgent",
    "assistant_payload",
    "plugin_payload",
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


def boundary_for(scope: Boundary | None, agent: Any) -> Boundary:
    """What a test meant, when it did not say (P6-32).

    The agent's own scope, which is what a test almost always means; `DEPLOYMENT`
    for no agent at all, because a call with no agent is not narrowed by
    anything. Stated here rather than defaulted in the seam, which is the whole
    of that row: the helper knows what its caller meant, the registry does not.

    **An agent whose `ctx` cannot be read refuses, exactly as production does.**
    The first version resolved it to `DEPLOYMENT`, which reintroduced the deleted
    defect one layer up: a policy test whose stub forgot its `ctx` would silently
    exercise the unrestricted view, and a visibility assertion would pass
    vacuously — silent-wide, scoped to precisely the population most likely to
    write it. A test that means the wide view spells `DEPLOYMENT`, which is the
    row's own principle: the answer that widens is the one you type.
    """
    if scope is not None:
        return scope
    if agent is None:
        return DEPLOYMENT
    own = getattr(agent, "ctx", None)
    if not isinstance(own, Context):
        raise TypeError(
            f"{type(agent).__name__} was passed to run_tool as `agent` but exposes no "
            "`ctx: Context`; pass `scope=` beside it (or `scope=DEPLOYMENT` for the "
            "deployment-wide view, on purpose)"
        )
    return own


async def run_tool(
    ctx: Any,
    name: str,
    arguments: Any = None,
    *,
    agent: Any,
    scope: Boundary | None = None,
    session: Any = None,
    call_id: str = "call-1",
) -> Any:
    """Execute one tool the way the loop does, for a test that is not the loop.

    The `ToolExecutionInput(...)` incantation — `scope=agent.ctx`, `session=`,
    `agent=` — was written out at nine call sites across six files, which is
    nine places for a test to accidentally pass a different scope than the one
    whose policy it meant to exercise.

    `scope=` is spellable **separately** from `agent=` because they are two
    values, and P6-24 is about the case where they differ — a Code Mode
    sub-dispatch, a subagent whose driver holds a child ctx. Without it this
    helper could not construct the divergence it exists to let a test assert.

    Its `None` is the *helper's* "you did not say", resolved by `boundary_for`
    below, and not the seam's — that one P6-32 deleted, because a seam given no
    boundary used to answer with the widest one it had.
    """
    from ..tools.definition import ToolExecutionInput

    return await ctx.tools.execute(
        ToolExecutionInput(
            call_id=call_id,
            name=name,
            arguments={} if arguments is None else arguments,
            scope=boundary_for(scope, agent),
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


def plugin_payload(
    text: str,
    message_id: str = "m1",
    *,
    plugin: str,
    form: ContextForm | None = None,
    summary: str | None = None,
) -> dict[str, Any]:
    """A `user/message` payload for text a *plugin* injected, not a person.

    The second shape a `user/message` takes, and the one a hand-built fixture
    keeps getting wrong: injected context, an offload preview and a compaction
    summary all ride the user role, and what distinguishes them is `source`.
    A test that reached for `user_payload` for one of those was asserting
    against a message no producer writes.
    """
    return {
        "id": message_id,
        "role": "user",
        "content": [{"type": "text", "text": text}],
        "source": PluginSource(plugin=plugin, form=form, summary=summary).to_wire(),
    }


def assistant_payload(
    text: str,
    message_id: str,
    *,
    turn: int = 1,
    step: int = 1,
    provider: str = "fake",
    content: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """An `assistant/message` payload; empty `text` gives an empty-content message.

    `content` supplies the blocks outright — a message carrying tool calls —
    so a test does not reach into this dict's shape to overwrite them.
    """
    if content is None:
        content = [{"type": "text", "text": text}] if text else []
    return {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "assistant",
            "content": content,
            "source": {"kind": "model", "provider": provider, "model": "m"},
        },
    }


def tool_result_payload(
    text: str,
    message_id: str,
    call_id: str = "c1",
    *,
    turn: int = 1,
    step: int = 1,
    is_error: bool = False,
) -> dict[str, Any]:
    """A `tool/result` payload carrying one text block.

    `is_error` is a keyword for the same reason `assistant_payload` takes
    `content`: so a test states the outcome it wants rather than reaching into
    this dict's shape to overwrite it afterwards.
    """
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
                    "isError": is_error,
                }
            ],
            "source": {"kind": "tool", "callId": call_id},
        },
    }
