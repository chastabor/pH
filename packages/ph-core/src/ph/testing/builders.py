"""Builders for tests and fixtures.

The module began with two families:

- **Payloads.** The three surface types have a shape that every test needs and
  that no test should type again: a user message, an assistant message inside
  its `assistant/message` payload, and a tool result inside its `tool/result`
  payload.
- **Tool scaffolding.** A string-output tool, a bare registry, an agent stub,
  and the fake-provider options. Each test module declared these again.

The module now also holds bounded waits (`settled`), builders for hand-written
and stored logs (`log_event`, `write_stored_log`, `StoredLogs`), and helpers
that narrow a type for a test (`as_kind`, `not_none`).

@module ph.testing.builders
"""

from __future__ import annotations

import errno
import itertools
import os
from collections.abc import AsyncIterator, Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NoReturn

import anyio

from ..agent.types import AgentDriver, AgentHandle, AgentOptions, AgentStatus
from ..cancel import CancelToken
from ..cordis import DEPLOYMENT, Boundary, Context, Next
from ..host import host_config_path, load_host_config
from ..json import JsonValue, as_obj, as_seq, as_str, dumps
from ..keys import AGENTS, JOBS, SESSION_PERSISTENCE, SESSIONS, SKILLS, TOOLS
from ..llm.types import (
    ContentBlock,
    ContextForm,
    PluginSource,
    ReasoningBlock,
    StreamChunk,
    TextBlock,
    TokenUsage,
    ToolResultBlock,
    text_of,
)
from ..paths import resolve_roots
from ..persistence.families import locate_under
from ..persistence.jsonl import (
    HEADER_LINE_TYPE,
    JsonlSessionStore,
    locate_session,
    read_session,
    session_path,
)
from ..persistence.lease import leased
from ..persistence.protocol import (
    NoStoredSession,
    StoredSession,
    descendants_among,
    gated,
    read_if_stored,
    stored_row,
)
from ..persistence.resume import resume_session
from ..persistence.turso import TursoSessionStore
from ..seams.skills import SkillService
from ..seams.subagents import ADMITTED
from ..seams.workspace import (
    ACQUIRED,
    ACQUIRING,
    DISPOSED,
    RETAINED,
    SharedWorkspaceProvider,
    WorkspaceSeam,
)
from ..session import (
    RevisionRecord,
    RevisionSize,
    Session,
    SessionBatch,
    SessionEvent,
    SessionHeader,
    SessionKind,
    SurfaceIntent,
    child_session_id,
    derive_event_message,
    freeze_json_value,
    intents,
    known_event_types,
    unsettled_why,
)
from ..session import kinds as core_kinds
from ..session.writers import LogWriter, scaffolding_writer
from ..tools import PreToolDecision, ToolExecution, ToolExecutionResult
from ..tools.definition import (
    Done,
    NotDone,
    Reconciled,
    ToolDefinition,
    ToolExecutionInput,
    ToolOutput,
    define_tool,
    text_content,
)
from ..tools.registry import ToolRuntime

__all__ = [
    "EMOJI_FAMILY",
    "FAKE_OPTIONS",
    "SCAFFOLDING",
    "FarSide",
    "StoppingAgent",
    "StoredLogs",
    "StubAgent",
    "admitted_child",
    "as_kind",
    "assistant_payload",
    "block_text",
    "boundary_for",
    "code_mode_stub",
    "delivered_ids",
    "disk_fills_midway",
    "external_tool",
    "full_disk_batch",
    "hold_session",
    "isolated_intent_kinds",
    "isolated_log_types",
    "live_settles",
    "llama_refusal",
    "log_event",
    "log_interrupted_call",
    "logged_events",
    "models_overlay",
    "new_agent",
    "not_none",
    "noted",
    "noting",
    "open_fds",
    "parked_gate",
    "plugin_payload",
    "raising",
    "reconciled_call",
    "reference_fork",
    "result_text",
    "resume_stored",
    "revision_record",
    "run_tool",
    "searches_into",
    "session_of",
    "settle_background",
    "settled",
    "simple_tool",
    "skill_service",
    "splice_line",
    "store_root",
    "stored_events",
    "stored_log",
    "stored_types",
    "streamed",
    "tool_result_payload",
    "tool_runtime",
    "unwritten",
    "user_payload",
    "workspace_acquired",
    "workspace_acquiring",
    "workspace_disposed",
    "workspace_log",
    "workspace_retained",
    "workspace_seam",
    "write_host_config",
    "write_profile",
    "write_reference_fork",
    "write_stored_log",
]

FAKE_OPTIONS = AgentOptions(provider="fake", model="fake-1")
"""The options every test that drives the fake adapter uses."""


def models_overlay(**routes: Mapping[str, object]) -> dict[str, Any]:
    """An overlay `models` row that lists `routes` by key. Its default is `main`,
    which runs on the route of `FAKE_OPTIONS` unless `routes` lists another
    route under that key. The row has the form that `mount` takes."""
    listed = {"main": {"provider": "fake", "model": "fake-1"}, **routes}
    return {"id": "models", "config": {"default": "main", "models": listed}}


@contextmanager
def _failing_after(seconds: float, what: str) -> Iterator[None]:
    """`anyio.fail_after`, which fails the test with what was being waited for.

    `fail_after` alone raises a bare `TimeoutError` that names neither the wait
    nor the reason. A stuck wait is exactly the failure that arrives with no
    other evidence. Every bounded wait here goes through this function, so none
    can drop the name.
    """
    # `ph.testing` is shipped, and its helpers are imported where pytest is not
    # installed.
    import pytest  # noqa: PLC0415

    try:
        with anyio.fail_after(seconds):
            yield
    except TimeoutError:
        pytest.fail(f"timed out waiting for {what}")


async def settled(check: Callable[[], object], what: str, *, seconds: float = 5.0) -> None:
    """Poll until `check()` is true, or fail and say what was waited for.

    **This is the one copy of this loop.** Suites had written it out under
    several names, and a copy that is missed fails as a *hang*, the least
    legible failure that a suite has. `ph.testing` is the declared home for
    scaffolding that no shipped module imports, and this is scaffolding.
    `seconds` is for a suite whose waits are longer, as the waits of a daemon
    are.
    """
    with _failing_after(seconds, what):
        while not check():
            await anyio.sleep(0.005)


async def settle_background(ctx: Context) -> None:
    """Wait for the work that a test left running: every detached task, then every job.

    `ctx.drain()` did this until jobs left its pool, and it was never quite a
    wait. It is shielded, so the deadline of a test could not reach it. It is
    bounded, so work that still ran after `DRAIN_SECONDS` was left with no
    message, and the assertions after it failed on that work. This function
    loops, because each kind of work can start the other: the drive of a child
    ends by waking listeners, and a listener can start a job.
    """
    jobs = ctx.get(JOBS)
    with _failing_after(10, "background jobs and detached tasks to end"):
        await ctx.settled()
        while jobs is not None and jobs.in_flight:
            await jobs.settled()
            await ctx.settled()


def simple_tool(
    name: str,
    execute: Callable[..., Any] | None = None,
    *,
    description: str | None = None,
    safe: bool | Callable[[Any], bool] = False,
    **kwargs: Any,  # noqa: ANN401
) -> ToolDefinition:
    """A tool that takes a free-form object and returns a string.

    This is the shape that every pipeline and batch test wants: no schema to
    satisfy, and a string that the assertions can read back. `execute` defaults
    to a function that returns `name`. `safe` is the concurrency
    classification: a bool or a classifier.
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


@dataclass(slots=True)
class FarSide:
    """What the effect of an external tool reached, in order. A test reads it as a counter."""

    deliveries: list[str] = field(default_factory=list)
    ids: set[str] = field(default_factory=set)
    """The ids that it has seen. A tool that can ask its far side would ask for these."""


def external_tool(
    name: str = "send", *, keyed: bool = True, fails: bool = False, reconciles: bool = False
) -> tuple[ToolDefinition, FarSide]:
    """A tool whose effect is outside the harness (a delivered message), and its far side.

    The tool is called with `{"id": ..., "message": ...}`.

    - `keyed` declares `id` as the effect key of the call
      (`ToolDefinition.idempotency_key`).
    - `fails` makes the delivery land and the call report an error. This is the
      ambiguous case that a retry must handle.
    - `reconciles` lets the tool answer `reconcile` by asking the far side about
      the id.
    """
    far = FarSide()

    def deliver(args: Any, _run: Any) -> str:  # noqa: ANN401
        far.deliveries.append(as_str(args.get("message")))
        far.ids.add(as_str(args.get("id")))
        if fails:
            raise RuntimeError("the far side accepted it and then the connection dropped")
        return f"delivered {as_str(args.get('id'))}"

    def effect(args: Any) -> str | None:  # noqa: ANN401
        return as_str(args.get("id")) or None

    async def asked(args: Any, _opened: SessionEvent, _session: Session) -> Reconciled:  # noqa: ANN401
        sent = as_str(args.get("id"))
        return Done(f"delivered {sent}") if sent in far.ids else NotDone()

    tool = simple_tool(
        name,
        deliver,
        idempotency_key=effect if keyed else None,
        reconcile=asked if reconciles else None,
    )
    return tool, far


SCAFFOLDING: LogWriter = scaffolding_writer()
"""The writer that tests build logs with. It writes every type that this build reads (T6).

A test is not a writer of record. It writes a crashed turn, an ask that nobody
answered, or a log that a newer build left. So it holds the one writer that may
write any known type, and a test kind carries that writer
(`IntentKind(..., writer=SCAFFOLDING)`). The writer is minted here, because
`scaffolding_writer` refuses a caller outside `ph.testing`."""


def log_event(
    log: Session | SessionBatch,
    event_type: str,
    data: Mapping[str, JsonValue],
    surface: SurfaceIntent | None = None,
) -> SessionEvent:
    """Write one event into a hand-built log, through `SCAFFOLDING`. A test calls
    this where it used to call `session.append` (T6)."""
    return SCAFFOLDING.append(log, event_type, data, surface)


async def streamed(chunks: Iterable[StreamChunk]) -> AsyncIterator[StreamChunk]:
    """Scripted chunks (`text_chunks`, `tool_call_chunks`) as the stream that a
    provider sends, for an `llm/stream` listener that answers a request itself."""
    for chunk in chunks:
        yield chunk


def revision_record(
    event_type: str, data: Mapping[str, JsonValue] | None = None, size: RevisionSize | None = None
) -> RevisionRecord:
    """The record of a producer, for a test that revises the surface through
    `ph.session.revise`. `SCAFFOLDING` writes it, so `event_type` may be any
    type that this build reads, including one that the test declared."""
    return RevisionRecord(SCAFFOLDING, event_type, data or {}, size)


def admitted_child(
    ctx: Context,
    parent: Session,
    run_id: str,
    admission: Mapping[str, JsonValue] | None = None,
    *,
    admitted: bool = True,
) -> Session:
    """The own log of a sub-agent, published in the store of `ctx` under `parent`
    and admitted. It is named and filed as `open_child_log` opens one (Phase 11).

    `admission` adds to the record, in the spelling of the log (`goalId`,
    `prompt`). The run id, a name and an owner are already there.
    `admitted=False` leaves a log with no admission, which is not a child. The
    record is written through `log_event`, so a test says what the log of the
    child holds and does not run a provider to get there.
    """
    child = ctx.require(SESSIONS).create(
        child_session_id(parent.id, run_id),
        meta={
            "parentSession": parent.id,
            "family": parent.header.family,
            "origin": "subagent",
            "delegationDepth": (parent.header.delegation_depth or 0) + 1,
        },
    )
    if admitted:
        log_event(
            child, ADMITTED, {"runId": run_id, "name": run_id, "owner": "stub", **(admission or {})}
        )
    return child


@contextmanager
def isolated_intent_kinds(*, core: bool) -> Iterator[None]:
    """Declare intent kinds into a table of this block only, so the kinds of a test do not leak.

    With `core`, the table starts with the kinds of ph-core itself
    (`ph.session.kinds.KINDS`, and only those). That is for a test of repair
    that must still see them settled. Without `core`, the table starts empty,
    for a test of `declare_intent` itself. In both cases the table is what a
    process that declared nothing else would hold, whatever this interpreter
    has imported. The kinds of a package (`ph_app.kinds`) are left out, so the
    block sees repair refuse a log that holds them (T4).

    One case is not enforced. A package leaf that is *first* imported inside
    the block declares its kind into the temporary table, and the real table
    never sees it.
    """
    real = intents._KINDS
    intents._KINDS = {kind.opened: kind for kind in core_kinds.KINDS} if core else {}
    try:
        yield
    finally:
        intents._KINDS = real


@contextmanager
def isolated_log_types() -> Iterator[None]:
    """Declare log types into a table of this block only, which starts from what
    the process has declared so far. The declarations of a test then do not
    outlive it. The completeness check of a front end, in a suite that runs
    beside this one, never meets them.
    """
    real = known_event_types._DECLARED
    known_event_types._DECLARED = dict(real)
    try:
        yield
    finally:
        known_event_types._DECLARED = real


def boundary_for(scope: Boundary | None, agent: AgentHandle | None) -> Boundary:
    """What a test meant when it did not say (P6-32).

    With an agent, the answer is the scope of the agent itself, which is what a
    test almost always means. With no agent, the answer is `DEPLOYMENT`,
    because nothing narrows a call with no agent. The default is stated here
    and not in the seam, and that is the whole of P6-32: the helper knows what
    its caller meant, and the registry does not.

    **A stub with no `ctx` no longer reaches this function to be refused.**
    That case was a `TypeError` raised on an `isinstance` behind a `getattr`.
    `AgentHandle` declares `ctx: Context`, so a stub without it now fails the
    Protocol at `_STUB_IS_A_HANDLE` below and not at a call site. What the
    refusal protected still holds, and the type now enforces it: a test that
    means the wide view writes `DEPLOYMENT`, because the answer that widens is
    the one that must be typed.
    """
    if scope is not None:
        return scope
    if agent is None:
        return DEPLOYMENT
    return agent.ctx


def parked_gate(ctx: Context, *, only: str | None = None) -> tuple[anyio.Event, anyio.Event]:
    """A `tools/pre-execute` gate that stops and waits to be released.

    Returns `(reached, release)`. The listener sets `reached` when a call
    arrives at the gate, and awaits `release` before it lets the call through.
    `only` names the one tool that the gate parks. Every other tool passes
    untouched, which is what a test of one call among several wants.

    This is for the tests that observe the log *mid-flight*: what has been
    written while a call is still parked on its gate (P7-15). Three test
    modules had written it out before this.
    """
    reached, release = anyio.Event(), anyio.Event()

    async def parked(execution: ToolExecution, next_: Next[PreToolDecision]) -> PreToolDecision:
        if only is None or execution.name == only:
            reached.set()
            await release.wait()
        return await next_(execution)

    ctx.on("tools/pre-execute", parked)
    return reached, release


async def run_tool(
    ctx: Context,
    name: str,
    arguments: Any = None,  # noqa: ANN401
    *,
    agent: AgentHandle,
    scope: Boundary | None = None,
    session: Session | None = None,
    call_id: str = "call-1",
    parent: object | None = None,
) -> ToolExecutionResult:
    """Execute one tool the way the loop does, for a test that is not the loop.

    The `ToolExecutionInput(...)` call (`scope=agent.ctx`, `session=`,
    `agent=`) was written out at nine call sites across six files. That gave a
    test nine places to pass, by accident, a different scope from the one whose
    policy it meant to exercise.

    `scope=` can be given **separately** from `agent=` because they are two
    values, and P6-24 is about the case where they differ: a Code Mode
    sub-dispatch, or a subagent whose driver holds a child ctx. Without it,
    this helper could not construct the divergence that it exists to let a
    test assert.

    The `None` of `scope` is the "you did not say" of the *helper*, which
    `boundary_for` above resolves. It is not the `None` of the seam. P6-32
    deleted that one, because a seam that was given no boundary answered with
    the widest one it had.

    `parent` makes the call a dispatch inside a Code Mode cell. A row that
    reads back or measures the own call of the cell must leave such a dispatch
    alone.
    """
    return await ctx.require(TOOLS).execute(
        ToolExecutionInput(
            call_id=call_id,
            name=name,
            arguments={} if arguments is None else arguments,
            scope=boundary_for(scope, agent),
            session=session,
            agent=agent,
            parent=parent,
        )
    )


def disk_fills_midway(fd: int, payload: bytes) -> None:
    """A `write_all` for a disk that fills in the middle of a payload: half the
    bytes land, then `ENOSPC`. Every atomic write, and every write that is taken
    back, must survive the half-written file that this leaves."""
    os.write(fd, payload[: len(payload) // 2])
    raise OSError(errno.ENOSPC, "No space left on device")


def full_disk_batch(items: Iterable[object], **_kwargs: object) -> list[OSError]:
    """A `write_atomic_all` for a disk that is full. Every file is refused with
    `ENOSPC`, each in its place, as `write_atomic_all` answers a write that it
    could not make. It is for the fail-open gates of the spill store, patched in
    where `ph.seams.spill` holds `write_atomic_all`."""
    return [OSError(errno.ENOSPC, "No space left on device") for _ in items]


def raising(error: BaseException) -> Callable[..., NoReturn]:
    """A body that raises `error`. It is readable where a generator trick was not."""

    def body(*_args: object, **_kwargs: object) -> NoReturn:
        raise error

    return body


def searches_into(bucket: list[str]) -> Callable[[Path, str, str], Path | None]:
    """`locate_under`, which also notes into `bucket` each id that the store is
    searched for. It is patched in where a backend holds `locate_under`
    (`ph.persistence.jsonl`), for a test that counts searches."""

    def search(root: Path, name: str, suffix: str) -> Path | None:
        return noted(bucket, name, locate_under(root, name, suffix))

    return search


def skill_service() -> tuple[Context, SkillService]:
    """A root context with a bare skill registry provided as `skills`.

    This is the shape of `tool_runtime` for the sibling registry, and it is here
    for the same reason. Each test module had written out the three-line
    constructor. When the construction of a seam drifts, two suites disagree
    about what a default looks like.
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


class StoppingAgent:
    """An agent, as far as `agent/turn-stopping` is concerned.

    It drives the listeners of that boundary through `ctx.serial` and not
    through a whole turn. The loop has no step cap of its own, so a fake
    provider that a listener keeps steering would be steered forever. The nudge
    budget of `skill-steps` and the one reminder of the subagent seam exist to
    bound that, and an endless turn can test neither.

    `steer` **appends what the driver would append**, and the tests depend on
    that. Both listeners count their own past steers by folding those same
    messages out of the log. A stub that only collected them in a list would
    leave the stand-down and the one-time rule untested.
    """

    def __init__(self, session: Session) -> None:
        self.session = session
        self.steers: list[Any] = []

    def steer(self, message: Any) -> None:  # noqa: ANN401
        self.steers.append(message)
        log_event(self.session, "user/message", message.to_wire(), SurfaceIntent("append"))


EMOJI_FAMILY = "\U0001f468\u200d\U0001f469\u200d\U0001f467"
"""One character of five code points: a man, a woman and a girl joined by ZWJs.
A test of a cut that must fall between characters cuts this string: all four of
its inner code-point cuts land inside it."""


def llama_refusal(sent: int, reads: int) -> str:
    """What llama.cpp says when an input (of an embedding or of a decision) is
    longer than its batch. The text holds both numbers, in the form that
    `ph.llm.adapter.too_long_facts` reads."""
    return (
        f"input ({sent} tokens) is too large to process. increase the "
        f"physical batch size (current batch size: {reads})"
    )


class StubAgent:
    """The minimum that an approval prompt or a tool call needs of an agent.

    It is exactly `AgentHandle`, and the check below holds it to that. A stub
    that drifts from the surface that the seams read would let every test that
    uses it pass against a shape that no real agent has.
    """

    def __init__(
        self,
        ctx: Context | None = None,
        session: Session | None = None,
        agent_id: str = "agent-a",
        status: AgentStatus = "idle",
    ) -> None:
        self.ctx = ctx if ctx is not None else Context()
        self.session = session
        self.id = agent_id
        self.options = FAKE_OPTIONS
        self.status = status
        self.signal = CancelToken()


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
    """A `user/message` payload for text that a *plugin* injected, not a person.

    This is the second shape of a `user/message`, and the one that a hand-built
    fixture keeps getting wrong. Injected context, an offload preview and a
    compaction summary all use the user role, and `source` distinguishes them.
    A test that used `user_payload` for one of those asserted against a message
    that no producer writes.
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
    usage: TokenUsage | None = None,
) -> dict[str, Any]:
    """An `assistant/message` payload. An empty `text` gives a message with empty content.

    `content` supplies the blocks outright, for a message that carries tool
    calls. `usage` supplies the count that the provider gave for the request
    that it answered. So a test does not reach into the shape of this dict to
    set either.
    """
    if content is None:
        content = [{"type": "text", "text": text}] if text else []
    payload: dict[str, Any] = {
        "turn": turn,
        "step": step,
        "message": {
            "id": message_id,
            "role": "assistant",
            "content": content,
            "source": {"kind": "model", "provider": provider, "model": "m"},
        },
    }
    if usage is not None:
        payload["usage"] = usage.to_wire()
    return payload


def tool_result_payload(
    text: str,
    message_id: str,
    call_id: str = "c1",
    *,
    turn: int = 1,
    step: int = 1,
    is_error: bool = False,
) -> dict[str, Any]:
    """A `tool/result` payload that carries one text block.

    `is_error` is a keyword for the same reason that `assistant_payload` takes
    `content`: a test states the outcome that it wants, and does not reach into
    the shape of this dict to overwrite it afterwards.
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

    `tool_runtime()` above has the same shape for the same reason. Two test
    modules had written this four-line constructor byte for byte the same,
    across twenty call sites. When the construction of a seam drifts, two
    suites disagree about what a default workspace is.

    There is deliberately no provider. A test that wants a tier registers one,
    and the seam falls back to `SharedWorkspaceProvider` either way.
    """
    return WorkspaceSeam(ctx=Context(), shared=SharedWorkspaceProvider(), scratch_root=scratch_root)


def workspace_acquired(
    agent_id: str, root: str, *, kind: str = "worktree", ref: str = "ph/s/a"
) -> tuple[str, dict[str, Any]]:
    """The opening half of the durable workspace pair (P4-14, P6-28)."""
    return (ACQUIRED, {"agentId": agent_id, "kind": kind, "root": root, "ref": ref})


def workspace_acquiring(
    agent_id: str, root: str, *, kind: str = "worktree", ref: str = "ph/s/a"
) -> tuple[str, dict[str, Any]]:
    """Where a tier is about to put a tree, recorded before it makes one (S12)."""
    return (ACQUIRING, {"agentId": agent_id, "kind": kind, "root": root, "ref": ref})


def workspace_retained(agent_id: str, reason: str) -> tuple[str, dict[str, Any]]:
    """A tree marked as evidence. An empty `reason` withdraws the mark."""
    return (RETAINED, {"agentId": agent_id, "retained": reason})


def workspace_disposed(agent_id: str, **extra: object) -> tuple[str, dict[str, Any]]:
    """The closing half, with `kept=`, `retained=` and `reconciled=` as the test needs."""
    return (DISPOSED, {"agentId": agent_id, **extra})


def workspace_log(*events: tuple[str, dict[str, Any]], session_id: str = "s") -> Session:
    """A session built from hand-written events, for a test of a fold.

    **It is here and not in each test module.** The reason showed only after
    two modules had written these four builders. `test_workspace_reconcile`
    and `test_workspace_retention` fold the *same* events through the *same*
    function. A fixture that drifts between them makes two suites disagree
    about the payload shape that one producer writes.

    The builders use `ACQUIRED`, `ACQUIRING`, `DISPOSED` and `RETAINED` and not
    the literals. Those constants exist so that the fold and the producers
    agree on the names exactly, and a test file must agree with them too.
    """
    session = Session(session_id)
    for kind, data in events:
        log_event(session, kind, data)
    return session


def reference_fork(
    child: str, parent: str, *, boundary: int, kind: SessionKind = "fork", family: str | None = None
) -> tuple[SessionHeader, list[SessionEvent]]:
    """A child that stores **only its own events**, which begin at `boundary`.

    It was built by hand while `fork` still copied the prefix. The reader
    landed before anything wrote a reference, so that the walk was exercised
    before `fork` depended on it and not after. `fork` now stores a reference
    itself, and this builder gives a test a stored reference with no live
    source session.

    It is here and not in either suite because both need it, and they built it
    differently. The core suite built it through a store. The view suite wrote
    out the wire format by hand (`"seedLength"`, `"version": 0`, the header
    envelope). A test that writes the format itself keeps passing when the
    format changes, so it is evidence for nothing. The function returns the
    header and the events, so each caller persists them in its own way while
    the *shape* of a reference-fork has one definition.

    The first event is at `boundary`. That marks the file as one that depends
    on a prefix, and it says how long the prefix is.
    """
    header = SessionHeader(
        id=child,
        created_at=1,
        parent_session=parent,
        seed_length=boundary,
        kind=kind,
        # A child inherits the lineage of its parent. This helper is given only
        # the *id* of the parent, so it cannot derive the lineage. The default is
        # therefore right only for a root created with **no cwd**, whose family
        # is its own id. Since format 1, a root that works in a directory is
        # `<cwd-tag>-<id>`, and a caller that forks one must say so. A wrong
        # family files the child in a directory of its own, which
        # `test_a_listing_row_says_the_same_thing_from_either_backend` asserts
        # against.
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
    """`reference_fork` written under `root` in the format that `read_session` reads.

    It goes through `to_wire` and the real header line type, and not through a
    dict written out by hand. A test that writes the wire format itself keeps
    passing after the format changes, so it is evidence for nothing.

    It takes the **root**, not a path. The first version took a path, to serve
    a caller that stores a log under a name that is not its id. No such caller
    exists. Three of the five sites wrote `f"{child}.jsonl"` at the call, which
    is the same failure one layer out: the test states the format itself.
    `session_path` is the one naming rule.
    """
    header, own = reference_fork(child, parent, boundary=boundary, kind=kind, family=family)
    return write_stored_log(root, header, own)


def write_stored_log(root: Path, header: SessionHeader, events: Sequence[SessionEvent]) -> Path:
    """A log as the JSONL store leaves it under `root`: the line of `header`, then `events`.

    This has the reason of `write_reference_fork`, one level down. The header
    line type, the encoders and the path belong to the store, so a test states
    only what the log holds.
    """
    path = session_path(root, header.id, header.family)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            f"{dumps(record)}\n"
            for record in [
                {"type": HEADER_LINE_TYPE, "header": header.to_wire()},
                *(event.to_wire(thaw=False) for event in events),
            ]
        ),
        encoding="utf-8",
    )
    return path


@contextmanager
def hold_session(root: Path, session_id: str) -> Iterator[Path]:
    """Hold the lease of one session as another process would. Yields the log path.

    It goes through `leased`, the function of the lease itself, and not through
    a lock file written by hand. The reason is that of `stored_log`, one layer
    over. Two suites each wrote `FileLock(f"{log}.lock")`, which was where the
    lease lived until it stopped being derived from the log at all. A test that
    writes the layout locks the wrong file as soon as the layout moves, and
    then tests nothing while it still passes.
    """
    with leased(root, session_id):
        yield stored_log(root, session_id)


def stored_log(root: Path, session_id: str, *, family: str | None = None) -> Path:
    """The path of the log of a session, in whatever layout the store wrote it.

    A test that writes `root / f"{session_id}.jsonl"` depends on the layout,
    and the family directories moved it. Twenty-two such tests broke at once.
    That is the argument for this function and against a second round of
    hand-editing.

    It falls back to the path that the log *would* take. So a test can assert
    that a file is absent before the first flush, and poll for it afterwards.
    `family` defaults to the id of the session itself, which is right for a
    root. A fork or a segment inherits the family of its parent and should say
    so.
    """
    return locate_session(root, session_id) or session_path(root, session_id, family or session_id)


def splice_line(path: Path, line: bytes = b"{not a record\n", *, at: int = 2) -> bytes:
    """Put `line` into a stored log at index `at` (the header is 0) and return
    what the file now holds. No crash makes this damage, because a crash only
    cuts the end of a log. It is the kind of damage that a reader must refuse
    and that `phern doctor sessions` must find (P13-09)."""
    lines = path.read_bytes().splitlines(keepends=True)
    lines.insert(at, line)
    damaged = b"".join(lines)
    path.write_bytes(damaged)
    return damaged


def as_kind[B](block: object, kind: type[B]) -> B:
    """This value, narrowed to `kind`: the assertion that a test was already making.

    A `ContentBlock` is a five-way union, so `blocks[0].name` is an error on
    four of them. A test that indexes the content of a message and reads a
    field *claims* which kind it got. Until the test trees came under mypy
    (issue 32), nothing checked the claim. There were 103 such reads, and the
    check that they wanted is one `isinstance`.

    The name follows the `as_int`/`as_obj`/`as_seq` family, which does the same
    job for a payload field: state the shape that you read, and fail at once if
    it is not that shape. It uses `assert` and not a raise because the caller
    is a test, and the failure is a wrong expectation and not a refusal to
    handle.

    It is not specific to blocks, although it started there. `Message.source`
    is a four-way union that is read the same way, and one helper is better
    than a second copy under another name.
    """
    assert isinstance(block, kind), f"expected a {kind.__name__}, got {type(block).__name__}"
    return block


def block_text(block: object) -> str:
    """The text of a block that carries text: a `TextBlock` or a `ReasoningBlock`.

    Both name it `text`, and 75 of the reads that `as_kind` exists for wanted
    only the string. Both kinds are named here, so a call site that does not
    care which of the two it holds does not have to decide.

    This is not `text_of`. That function joins a *sequence* and silently skips
    anything that is not a `TextBlock`. A test that asserted on reasoning text
    would get `""` and pass for the wrong reason.
    """
    assert isinstance(block, TextBlock | ReasoningBlock), (
        f"expected a block carrying text, got {type(block).__name__}"
    )
    return block.text


_AGENT_IDS = itertools.count()


def new_agent(ctx: Context, prefix: str = "agent") -> AgentDriver:
    """A fresh agent on the fake adapter, on a session id that nothing else has taken.

    The id is counted and not fixed. A test that needs two agents (an index
    call and a query call) would otherwise collide on the session id. That
    fails as `SESSION_ALREADY_EXISTS`, several frames from the cause."""
    session = ctx.require(SESSIONS).create(f"{prefix}-{next(_AGENT_IDS)}")
    return ctx.require(AGENTS).create(session, FAKE_OPTIONS)


def session_of(agent: AgentHandle) -> Session:
    """The session of the agent, which a test that has an agent always has.

    `AgentHandle.session` is `Session | None` because a handle exists before
    the session is attached, and every production read of it narrows first. A
    test that has just created an agent through
    `ctx.require(AGENTS).create(...)` knows that the session is there. It said
    so by reading `agent.session.events`, and nothing checked that claim until
    the test trees came under mypy (issue 32).
    """
    session = agent.session
    assert session is not None, "this agent has no session"
    return session


def open_fds() -> set[str]:
    """The open descriptors of this process, by number. `/dev/fd` lists them on
    both Linux and macOS. So a test of something that holds a descriptor (a
    timer, a watch) can say that it gave the descriptor back."""
    return set(os.listdir("/dev/fd"))


def noted[I, T](bucket: list[I], item: I, answer: T) -> T:
    """Record that a listener ran, then answer with a value that is already in hand.

    Twelve listeners in the test trees were written as
    `bucket.append(x) or answer`: a statement hidden in a lambda, which relies
    on `append` returning `None`. That reads the value of a call that has none,
    and mypy refuses it now that the trees are checked (issue 32).
    `(append(x), answer)[1]` does not help, because the value is still *used*,
    as a tuple element. A statement can go only in a function.
    """
    bucket.append(item)
    return answer


def noting[I, T](bucket: list[I], item: I, answer: Callable[[], T]) -> T:
    """Record that a listener ran, **then** produce the answer.

    This is the sibling of `noted` for the case where the answer is a call,
    such as `next_()` in a middleware chain. The callable matters. To pass
    `next_()` as a value would run the rest of the chain *before* this listener
    recorded itself, and these tests assert the order
    (`["pre", "body", "post"]`).
    """
    bucket.append(item)
    return answer()


class StoredLogs:
    """A `SessionArchive` in memory. It holds whole logs, of which one can be
    unreadable, and it records the `family` of every read. That record is the
    fold's half of "an id plus a family is a path", and nothing else would
    notice if it went missing.

    It is the one stand-in for the folds over stored sessions (the attachments
    collector, the workspace collector, and the rebuild of the live-roots
    index). A change to the surface of the archive is then made one time here,
    and not in a copy per test module. Rows are built as both backends build
    them (`stored_row`), and `read_own` keeps only the `types` that were asked
    for, as the backends do.
    """

    def __init__(self, *sessions: Session, broken: str = "", truncate: bool = False) -> None:
        self.sessions = {one.id: one for one in sessions}
        self.broken = broken
        """A session id whose every read raises, as a torn log does."""
        self.truncate = truncate
        """Answer every listing as though it filled its `limit`."""
        self.asked: dict[str, str | None] = {}
        """The `family` that each read of each session was given. The latest read wins."""
        self.whole: list[str] = []
        """The sessions read whole (`read`), in order."""
        self.limits: list[int | None] = []
        """The `limit` each listing was asked with, in order."""

    def stored(self, *, limit: int | None = 50) -> list[StoredSession]:
        self.limits.append(limit)
        rows = [
            stored_row(one.id, one.header, float(index))
            for index, one in enumerate(self.sessions.values())
        ]
        if limit is None:
            return rows
        return rows * limit if self.truncate else rows[:limit]

    def holding(
        self, types: frozenset[str], *, gate: str
    ) -> Iterator[tuple[str, list[SessionEvent]]]:
        for session_id in self.sessions:
            if session_id == self.broken:
                continue  # left out, as a store leaves out a log it cannot read
            yield session_id, gated(self.read_own(session_id, types=types)[1], gate)

    def descendants_of(self, parent_id: str, family: str) -> tuple[StoredSession, ...]:
        return descendants_among(
            parent_id,
            (
                stored_row(one.id, one.header, 0.0)
                for one in self.sessions.values()
                if one.header.family == family
            ),
        )

    def read_own(
        self,
        session_id: str,
        upto: int | None = None,
        family: str | None = None,
        *,
        types: frozenset[str] | None = None,
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        header, events = self._open(session_id, family)
        return header, [one for one in events if types is None or one.type in types]

    def read(
        self, session_id: str, *, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        """Whole logs in memory, so the materialized read *is* the unchained one."""
        self.whole.append(session_id)
        return self._open(session_id, family)

    def _open(
        self, session_id: str, family: str | None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        self.asked[session_id] = family
        if session_id == self.broken:
            raise ValueError("a torn log")
        session = self.sessions.get(session_id)
        if session is None:
            raise NoStoredSession(f"no stored session {session_id!r}")
        return session.header, list(session.events)


def stored_events(ctx: Context, session_id: str) -> list[SessionEvent]:
    """The events that the store of a session holds right now, read through the
    Protocol. The list is empty when the store holds no such session.

    A durability test asks this: not what is in memory, but what a resume would
    be given. Seven tests wrote out the read before `stored_types` did, and two
    more wrote its tolerant form before this function.
    """
    stored = read_if_stored(ctx.require(SESSION_PERSISTENCE), session_id)
    return [] if stored is None else stored[1]


async def resume_stored(ctx: Context, session_id: str) -> Session:
    """`resume_session` over the log that the store of `ctx` holds. This is
    `open_session` without its claim or its start, for a test of the resume
    itself."""
    return await resume_session(ctx, session_id, *ctx.require(SESSION_PERSISTENCE).read(session_id))


def stored_types(ctx: Context, session_id: str) -> list[str]:
    """The event types `stored_events` reads."""
    return [event.type for event in stored_events(ctx, session_id)]


def delivered_ids(log: Session) -> list[str]:
    """The id of each message delivered into the inbox of `log`, in order and
    with repeats kept, as its `agent/inbox/spliced` records say. A test counts a
    notice this way."""
    return [
        as_str(as_obj(message).get("id"))
        for event in log.events
        if event.type == "agent/inbox/spliced"
        for message in as_seq(event.data.get("inserted"))
    ]


def live_settles(session: Session, event_type: str) -> list[SessionEvent]:
    """The `event_type` records that a live settle wrote. This leaves out the
    records that repair, or a claim whose body raised, closed with an
    `unsettled` marker."""
    return [
        event
        for event in session.events
        if event.type == event_type and unsettled_why(event.data) is None
    ]


def log_interrupted_call(
    session: Session,
    name: str,
    arguments: str,
    *,
    call_id: str = "c1",
    dispatches: Sequence[tuple[str, Mapping[str, JsonValue]]] = (),
) -> None:
    """A turn that recorded a tool call as started and died before its result.

    This is what a crash in the middle of a call leaves: the request of the
    assistant and its `tool/call`, with nothing after them. `dispatches` are
    the Code Mode dispatches that the call had started, each as
    `(tool, arguments)`. They get the sub-call ids that a cell gives them
    (`{call}:code:{n}`), and they are never settled.
    """
    log_event(session, "turn/start", {"turn": 1})
    log_event(session, "step/start", {"turn": 1, "step": 1})
    call = {"type": "tool-call", "id": call_id, "name": name, "arguments": arguments}
    log_event(
        session,
        "assistant/message",
        assistant_payload("", "m1", content=[call]),
        SurfaceIntent("append", ()),
    )
    log_event(
        session,
        "tool/call",
        {"turn": 1, "step": 1, "callId": call_id, "name": name, "arguments": arguments},
    )
    for index, (tool, tool_arguments) in enumerate(dispatches):
        log_event(
            session,
            "tool/code-dispatch-start",
            {
                "rootCallId": call_id,
                "parentCallId": call_id,
                "subCallId": f"{call_id}:code:{index}",
                "name": tool,
                "arguments": tool_arguments,
            },
        )


async def reconciled_call(
    ctx: Context,
    session: Session,
    name: str,
    arguments: Mapping[str, JsonValue],
    *,
    call_id: str = "call-1",
) -> tuple[ContentBlock, ...] | NotDone | None:
    """What tool `name` says about one call that a crash left open, asked as a resume asks.

    The `tool/call` record that `log_interrupted_call` writes is passed to
    `ToolRuntime.reconciled` as `resume_session` passes it. The answer is the
    rendered result for `Done`, `NotDone`, or `None` for `Unknown`.
    """
    record = SessionEvent(
        type="tool/call",
        seq=0,
        time=0,
        data=freeze_json_value({"callId": call_id, "name": name, "arguments": dumps(arguments)}),
    )
    return await ctx.require(TOOLS).reconciled(record, session, scope=DEPLOYMENT)


def result_text(session: Session, call_id: str) -> str:
    """What the model reads back from one call.

    It goes through `derive_event_message`, THE projection, and not through an
    index into `event.data["message"]["content"][0]`. A second route to that
    shape keeps passing after the shape moves, and a test that reads the log by
    hand stops testing what the model sees.
    """
    for event in session.events:
        if event.type != "tool/result":
            continue
        message = derive_event_message(event)
        for block in message.content if message else ():
            if isinstance(block, ToolResultBlock) and block.tool_call_id == call_id:
                return text_of(block.content)
    return ""


def code_mode_stub() -> dict[str, Any]:
    """The patch that mounts Code Mode over the stub runtime, for a test to pass to `mount`.

    Each call returns a fresh document, because a mount may keep what it is
    given.
    """
    return {
        "insert": [
            {"id": "code-runtime-stub", "name": "code-runtime-stub"},
            {"id": "tools-code-mode", "name": "tools-code-mode"},
        ]
    }


def store_root(ctx: Context) -> Path:
    """Where the mounted session store keeps its logs, for a test that needs it.

    **Narrowed, not promised.** `SessionPersistence` says nothing about where a
    backend writes. The title of its module is "what a backend must provide,
    with no statement of where it writes". So `root` is a field of the two
    path-backed implementations and not of the Protocol. Seven tests read
    `ctx.require(SESSION_PERSISTENCE).root` anyway, which compiled only because
    the test trees were outside mypy (issue 32).

    To widen the Protocol for them would contradict a stated decision. So the
    narrowing happens here, and the assertion names what the test assumes: a
    store that has a path on disk at all.
    """
    store = ctx.require(SESSION_PERSISTENCE)
    assert isinstance(store, JsonlSessionStore | TursoSessionStore), (
        f"{type(store).__name__} keeps no logs on disk, so it has no root"
    )
    return store.root


def not_none[T](value: T | None, what: str = "") -> T:
    """This value, which a test that reached this line knows is there.

    The `get(...)` of a seam answers `X | None` because absence is normal in
    production, and every caller there narrows. A test that has just registered
    the thing that it asks for knows that it is there. It said so by reading
    the attribute, which compiled only while the test trees were outside mypy
    (issue 32).

    Prefer a plain `assert x is not None` where the value is used more than one
    time. The local that it binds narrows for the rest of the function, and
    reads better than a repeat of this call. This function is for the
    single-use case, where a bound name costs a line and gives no benefit.
    """
    assert value is not None, what or "expected a value, got None"
    return value


def write_host_config(text: str, *, home: Path | None = None) -> Path:
    """Write `$PH_HOME/daemon.yaml` as a user would before a start, and return its path.

    `load_host_config` is read one time per process (`ph.host`). So a test that
    writes the file after something resolved the roots is a process that has
    not restarted. This function forgets the earlier read, and that *is* the
    restart. `home` defaults to the `$PH_HOME` that this test runs under.
    """
    root = home if home is not None else resolve_roots().home
    path = host_config_path(root)
    path.write_text(text, encoding="utf-8")
    load_host_config.cache_clear()
    return path


def write_profile(name: str, text: str) -> Path:
    """Write the user's named profile `name` under the `$PH_HOME` of this test. Returns the path."""
    path = resolve_roots().profile_overlay(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def logged_events(session_id: str) -> list[SessionEvent]:
    """The events that the log of a session holds on disk under the `$PH_HOME` of this test.

    A durability assertion reads this when no `Context` is at hand, such as
    after a command run through `CliRunner`. `stored_events` reads through a
    mounted store.
    """
    _header, events = read_session(stored_log(resolve_roots().sessions_dir(), session_id))
    return events


async def unwritten(*_args: object) -> bool:
    """`session_written` for a log that cannot be written. Patch it in where a
    caller of `session_written` must then refuse, or change nothing."""
    return False
