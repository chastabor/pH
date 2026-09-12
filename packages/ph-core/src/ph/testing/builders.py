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
from typing import TYPE_CHECKING, Any

import anyio

from ..agent.types import AgentHandle, AgentOptions
from ..cordis import DEPLOYMENT, Boundary, Context
from ..json import dumps
from ..keys import SESSION_PERSISTENCE, SKILLS, TOOLS
from ..llm.types import ContextForm, PluginSource, ReasoningBlock, TextBlock
from ..persistence.jsonl import HEADER_LINE_TYPE, locate_session, session_path
from ..seams.skills import SkillService
from ..seams.workspace import (
    ACQUIRED,
    DISPOSED,
    RETAINED,
    SharedWorkspaceProvider,
    WorkspaceSeam,
)
from ..session import Session, SessionEvent, SessionHeader, SessionKind
from ..tools.definition import ToolDefinition, ToolOutput, define_tool, text_content
from ..tools.registry import ToolRuntime

__all__ = [
    "FAKE_OPTIONS",
    "StubAgent",
    "assistant_payload",
    "parked_gate",
    "plugin_payload",
    "raising",
    "reference_fork",
    "run_tool",
    "simple_tool",
    "stored_log",
    "tool_result_payload",
    "tool_runtime",
    "user_payload",
    "workspace_acquired",
    "workspace_disposed",
    "workspace_log",
    "workspace_retained",
    "workspace_seam",
    "write_reference_fork",
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


def boundary_for(scope: Boundary | None, agent: AgentHandle) -> Boundary:
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


def parked_gate(ctx: Any, *, only: str | None = None) -> tuple[anyio.Event, anyio.Event]:
    """A `tools/pre-execute` gate that stops and waits to be let go.

    Returns `(reached, release)`: the listener sets the first when a call arrives
    at the gate and awaits the second before letting it through. `only` names
    the one tool it parks; anything else passes untouched, which is what a test
    of one call among several wants.

    For the tests that observe the log *mid-flight* — what has been written while
    a call is still parked on its gate (P7-15). Written out in three test modules
    before this.
    """
    reached, release = anyio.Event(), anyio.Event()

    async def parked(execution: Any, next_: Any) -> Any:
        if only is None or execution.name == only:
            reached.set()
            await release.wait()
        return await next_(execution)

    ctx.on("tools/pre-execute", parked)
    return reached, release


async def run_tool(
    ctx: Any,
    name: str,
    arguments: Any = None,
    *,
    agent: AgentHandle,
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

    return await ctx.require(TOOLS).execute(
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


def skill_service() -> tuple[Context, SkillService]:
    """A root context with a bare skill registry provided as `skills`.

    `tool_runtime`'s shape for the sibling registry, and here for its stated
    reason: the three-line constructor was written out per test module, and a
    seam whose construction drifts is one where two suites disagree about what a
    default looks like.
    """
    root = Context()
    service = SkillService(ctx=root)
    root.provide(SKILLS, service)
    return root, service


def tool_runtime() -> tuple[Context, ToolRuntime]:
    """A root context with a bare registry provided as `tools`."""
    root = Context()
    runtime = ToolRuntime(ctx=root)
    root.provide(TOOLS, runtime)
    return root, runtime


class StubAgent:
    """The minimum an approval prompt or a tool call needs of an agent.

    Exactly `AgentHandle`, and held to it below: a stub that drifts from the
    surface the seams read would let every test that uses it pass against a
    shape no real agent has.
    """

    def __init__(
        self, ctx: Context | None = None, session: Session | None = None, agent_id: str = "agent-a"
    ) -> None:
        self.ctx = ctx if ctx is not None else Context()
        self.session = session
        self.id = agent_id
        self.options = FAKE_OPTIONS


if TYPE_CHECKING:
    _STUB_IS_A_HANDLE: AgentHandle = StubAgent()


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
    child: str, parent: str, *, boundary: int, kind: SessionKind = "fork", family: str | None = None
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
    header = SessionHeader(
        id=child,
        created_at=1,
        parent_session=parent,
        seed_length=boundary,
        kind=kind,
        # A child inherits its parent's lineage. Defaulting to `parent` is right
        # for the common case — forking a root, whose family is its own id — and
        # a test chaining segments off a deeper session says which.
        family=family or parent,
    )
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


def write_reference_fork(
    root: Path,
    child: str,
    parent: str,
    *,
    boundary: int,
    kind: SessionKind = "fork",
    family: str | None = None,
) -> Path:
    """`reference_fork` written under `root` in the format `read_session` reads.

    Through `to_wire` and the real header line type rather than a dict spelled
    out longhand: a test that writes the wire format itself keeps passing after
    the format changes, which makes it evidence for nothing.

    Takes the **root**, not a path. The first version took a path to serve a
    caller that stores a log under a name that is not its id — no such caller
    exists, and three of the five sites were spelling `f"{child}.jsonl"` at the
    call instead, which is the same "the test states the format itself" failure
    one layer out. `session_path` is the one naming rule.
    """
    header, own = reference_fork(child, parent, boundary=boundary, kind=kind, family=family)
    path = session_path(root, child, header.family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{dumps(record)}\n"
            for record in [
                {"type": HEADER_LINE_TYPE, "header": header.to_wire()},
                *(event.to_wire(thaw=False) for event in own),
            ]
        ),
        encoding="utf-8",
    )
    return path


def stored_log(root: Path, session_id: str, *, family: str | None = None) -> Path:
    """Where a session's log is, whatever layout the store wrote it in.

    A test that spells `root / f"{session_id}.jsonl"` pins the layout, and the
    family directories moved it — twenty-two of them broke at once, which is the
    argument for this existing rather than for a second round of hand-editing.

    Falls back to the path the log *would* take, so a test can assert a file is
    absent before the first flush and poll for it afterwards. `family` defaults
    to the session's own id, which is right for a root; a fork or a segment
    inherits its parent's and should say so.
    """
    return locate_session(root, session_id) or session_path(root, session_id, family or session_id)


def as_kind[B](block: object, kind: type[B]) -> B:
    """This value, narrowed to `kind` — the assertion a test was already making.

    A `ContentBlock` is a five-way union, so `blocks[0].name` is an error on four
    of them: a test that indexes a message's content and reads a field is
    *claiming* which kind it got, and until the test trees came under mypy
    (issue 32) nothing checked the claim. 103 such reads existed, and the check
    they wanted is one `isinstance`.

    Named for the `as_int`/`as_obj`/`as_seq` family, which does the same job for
    a payload field: state the shape you are reading and fail on the spot if it
    is not that. `assert` rather than a raise because the caller is a test and
    the failure is a wrong expectation, not a refusal to handle.

    Not block-specific despite where it started: `Message.source` is a four-way
    union read the same way, and one helper is better than a second copy under
    another name.
    """
    assert isinstance(block, kind), f"expected a {kind.__name__}, got {type(block).__name__}"
    return block


def block_text(block: object) -> str:
    """The text of a block that carries text — `TextBlock` or `ReasoningBlock`.

    Both spell it `text`, and 75 of the reads `as_kind` exists for wanted only
    the string. Naming both kinds here keeps the call site from having to decide
    which of the two it is holding when it does not care.

    Not `text_of`: that one joins a *sequence* and silently skips anything that
    is not a `TextBlock`, so a test asserting on reasoning text would get `""`
    and pass for the wrong reason.
    """
    assert isinstance(block, TextBlock | ReasoningBlock), (
        f"expected a block carrying text, got {type(block).__name__}"
    )
    return block.text


def session_of(agent: AgentHandle) -> Session:
    """The agent's session, which a test that has an agent always has.

    `AgentHandle.session` is `Session | None` because a handle exists before the
    session is attached, and every read of it in production narrows first. A
    test that has just created an agent through `ctx.require(AGENTS).create(...)`
    knows better, and said so by reading `agent.session.events` — a claim
    nothing checked until the test trees came under mypy (issue 32).
    """
    session = agent.session
    assert session is not None, "this agent has no session"
    return session


def noted[I, T](bucket: list[I], item: I, answer: T) -> T:
    """Record that a listener ran, then answer with a value already in hand.

    Twelve listeners in the test trees were written `bucket.append(x) or answer`
    — a statement smuggled into a lambda, relying on `append` returning `None`.
    That reads the value of a call that has none, which mypy refuses once the
    trees are checked (issue 32), and `(append(x), answer)[1]` does not help:
    the value is still *used*, just as a tuple element. A function is the only
    place a statement can go.
    """
    bucket.append(item)
    return answer


def noting[I, T](bucket: list[I], item: I, answer: Callable[[], T]) -> T:
    """Record that a listener ran, **then** produce the answer.

    `noted`'s sibling for the case where the answer is a call — `next_()` in a
    middleware chain. The callable matters: passing `next_()` as a value would
    run the rest of the chain *before* this listener recorded itself, and these
    tests assert the order (`["pre", "body", "post"]`).
    """
    bucket.append(item)
    return answer()


def store_root(ctx: Context) -> Path:
    """Where the mounted session store keeps its logs, for a test that needs it.

    **Narrowed rather than promised.** `SessionPersistence` says nothing about
    where a backend writes — its module is titled "what a backend owes, without
    saying where it writes" — so `root` is a field of the two path-backed
    implementations and not of the Protocol. Seven tests read
    `ctx.require(SESSION_PERSISTENCE).root` anyway, which compiled only because
    the test trees were outside mypy (issue 32).

    Widening the Protocol to suit them would contradict a stated decision, so
    the narrowing happens here and the assertion names what the test is
    assuming: a store that has a path on disk at all.
    """
    from ..persistence.jsonl import JsonlSessionStore
    from ..persistence.turso import TursoSessionStore

    store = ctx.require(SESSION_PERSISTENCE)
    assert isinstance(store, JsonlSessionStore | TursoSessionStore), (
        f"{type(store).__name__} keeps no logs on disk, so it has no root"
    )
    return store.root


def not_none[T](value: T | None, what: str = "") -> T:
    """This value, which a test that reached this line knows is there.

    A seam's `get(...)` answers `X | None` because absence is normal in
    production, and every caller there narrows. A test that has just registered
    the thing it is asking for knows better — and said so by reading the
    attribute, which compiled only while the test trees were outside mypy
    (issue 32).

    Prefer a plain `assert x is not None` where the value is used more than
    once: the local it binds narrows for the rest of the function, and reads
    better than repeating this call. This is for the single-use case, where
    binding a name costs a line and buys nothing.
    """
    assert value is not None, what or "expected a value, got None"
    return value
