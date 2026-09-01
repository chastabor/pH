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
from pathlib import Path
from typing import Any

from ..agent.types import AgentOptions
from ..cordis import DEPLOYMENT, Boundary, Context
from ..llm.types import ContextForm, PluginSource
from ..seams.workspace import (
    ACQUIRED,
    DISPOSED,
    RETAINED,
    SharedWorkspaceProvider,
    WorkspaceSeam,
)
from ..session import Session, SessionEvent, SessionHeader
from ..tools.definition import ToolDefinition, ToolOutput, define_tool, text_content
from ..tools.registry import ToolRuntime

__all__ = [
    "FAKE_OPTIONS",
    "StubAgent",
    "assistant_payload",
    "plugin_payload",
    "raising",
    "reference_fork",
    "run_tool",
    "simple_tool",
    "tool_result_payload",
    "tool_runtime",
    "user_payload",
    "workspace_acquired",
    "workspace_disposed",
    "workspace_log",
    "workspace_retained",
    "workspace_seam",
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


def workspace_seam(scratch_root: Path) -> WorkspaceSeam:
    """A bare `ctx.workspace` on its own root context, with no tier registered.

    `tool_runtime()` above is the same shape for the same reason: two test
    modules had written this four-line constructor byte-identically, across
    twenty call sites, and a seam whose construction drifts is one where two
    suites disagree about what a default workspace is.

    No provider, deliberately — a test that wants a tier registers one, and
    `SharedWorkspaceProvider` is what the seam falls back to either way.
    """
    return WorkspaceSeam(ctx=Context(), shared=SharedWorkspaceProvider(), scratch_root=scratch_root)


def workspace_acquired(
    agent_id: str, root: str, *, kind: str = "worktree", ref: str = "ph/s/a"
) -> tuple[str, dict[str, Any]]:
    """The opening half of the durable workspace pair (P4-14, P6-28)."""
    return (ACQUIRED, {"agentId": agent_id, "kind": kind, "root": root, "ref": ref})


def workspace_retained(agent_id: str, reason: str) -> tuple[str, dict[str, Any]]:
    """A tree marked as evidence; an empty `reason` withdraws the mark."""
    return (RETAINED, {"agentId": agent_id, "retained": reason})


def workspace_disposed(agent_id: str, **extra: Any) -> tuple[str, dict[str, Any]]:
    """The closing half — `kept=`, `retained=`, `reconciled=` as the test needs."""
    return (DISPOSED, {"agentId": agent_id, **extra})


def workspace_log(*events: tuple[str, dict[str, Any]], session_id: str = "s") -> Session:
    """A session built from hand-written events, for testing a fold.

    **Here rather than in each test module**, and it took two modules writing
    these four builders before the reason showed: `test_workspace_reconcile` and
    `test_workspace_retention` fold the *same* events through the *same*
    function, so a fixture that drifts between them is how two suites come to
    disagree about the payload shape one producer writes.

    Through `ACQUIRED`/`DISPOSED`/`RETAINED` rather than the literals, for the
    reason those constants already give for themselves: "a literal spelled at
    five sites is five chances not to", and a test file is a site.
    """
    session = Session(session_id)
    for kind, data in events:
        session.append(kind, data)
    return session


def reference_fork(
    child: str, parent: str, *, boundary: int
) -> tuple[SessionHeader, list[SessionEvent]]:
    """A child that stores **only its own events**, beginning at `boundary`.

    Hand-built because `fork` still copies: the reader lands before anything
    writes a reference, which is the whole point of that sequencing — the walk
    has to be exercised before `fork` depends on it, not after.

    Here rather than in either suite because both need it and they were building
    it differently: the core one through a store, the view one by writing the
    wire format out longhand (`"seedLength"`, `"version": 0`, the header
    envelope). A test that spells the format itself keeps passing when the format
    changes, which makes it evidence for nothing. Returning the header and the
    events lets each caller persist them its own way while the *shape* of a
    reference-fork has one definition.

    The first event sits at `boundary`, which both marks the file as owing a
    prefix and says how long that prefix is.
    """
    header = SessionHeader(id=child, created_at=1, parent_session=parent, seed_length=boundary)
    own = [
        SessionEvent(type="turn/start", seq=boundary, time=1, data={"turn": boundary}),
        SessionEvent(
            type="turn/end",
            seq=boundary + 1,
            time=1,
            data={"turn": boundary, "reason": {"kind": "completed"}},
        ),
    ]
    return header, own
