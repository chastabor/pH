"""`rlm-bindings`: the `rlm` namespace, as governed tool calls (P3-10, C2/C3).

In prime-agent, `await rlm("task", name=..., model=...)` traveled over a Jupyter
comm channel as a `host.request`. The call was therefore invisible to
`tools/pre-execute`, to `ctx.approval`, to the call limits and to the offload
policy. Here the same call is a **binding**. It leaves the kernel as one `call`
frame, goes through the whole tool pipeline, and is recorded as a durable
`tool/code-dispatch-start`/`tool/code-dispatch` pair. A deployment can therefore
deny or `ask` on subagent spawning by policy, and `max_subagent_spawns_per_run`
counts the call. Neither needs any code in this module.

Two decisions need a statement:

**The name that a program uses is not the tool name.** A cell writes
`rlm.run(...)`, and the governed tool is `rlm_run`. A namespace cannot claim a
bare global name such as `run`. It does not need to, because the SDK renders the
namespaced form and the log records the tool. Only the tools exist as a registry
surface, so a policy row that addresses `rlm_run` means exactly one thing.

**Spawning is `counts_as_spawn`.** The bridge counts a spawn against the spawn
budget of C4 as well as against the general dispatch budget. One approved cell
therefore cannot fan out past `max_subagent_spawns_per_run` on that one approval.

`find_models` is deliberately absent. It would need a model *catalog* on
`ctx.llm`, which does not exist yet. A discovery call that could only answer "I
don't know" would be worse than no call. A child that names no model already
inherits the model of its parent.

@module ph_rlm.bindings
"""

from __future__ import annotations

from typing import Any

from ph.cancel import Canceled, raced
from ph.cordis import Context, plugin
from ph.json import JsonObject, as_str, dumps, thaw_json
from ph.keys import SUBAGENTS, TOOLS
from ph.llm.types import ContentBlock
from ph.seams.code_runtime import CodeBindingNamespace
from ph.seams.models import ModelChoice
from ph.seams.subagents import (
    Access,
    SubagentRequest,
    SubagentRun,
    admission_notes,
    admitted_by,
    ended_text,
)
from ph.session import Session, SessionEvent
from ph.tools import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ph.tools.code_mode import CodeBindingsRequest, ToolCallError, governed_binding
from ph.tools.definition import Done, NotDone, Reconciled
from ph.wire import WireModel

from .subagents import PROVIDER_NAME

__all__ = ["NAMESPACE", "Config", "apply"]

NAMESPACE = "rlm"

RUN_TOOL = "rlm_run"
LIST_TOOL = "rlm_list_subagents"
DELETE_TOOL = "rlm_delete_subagent"
WAIT_TOOL = "rlm_wait"

_RUN_DESCRIPTION = (
    "Delegate a task to a child agent. Returns immediately with an admission "
    "handle — never the answer. The child replies by sending you an agent "
    "message, which arrives on a later turn; do not wait or sleep for it."
)


class RunArgs(ToolModel):
    """`rlm.run(...)`. Snake_case because the model types these names."""

    prompt: str
    name: str | None = None
    model: str | None = None
    """The key of a model that the profile of the caller lists. `None` runs the
    child on the model of the caller (S7b)."""
    output: dict[str, Any] | None = None
    """A JSON Schema object for the result that the caller needs back. The child
    returns a value that matches it, or the child fails. The value reaches the
    caller in the ending message of the child and in its roster row. A skill that
    the child is given may declare a schema, and this argument overrides it."""
    thinking: str | None = None
    """A reasoning effort for the child. The value is a name that the profile
    defines for the model of the child, or `default` for the effort of the
    provider itself. `None` runs the child at the effort that its skills or its
    model name."""
    access: Access = "read"
    """`read` | `write`. The default is `read` (E4). A child that only needs to
    read must not get a writable repo because nobody said otherwise. The field is
    typed, so pydantic validates it and the model sees the two values in the
    schema."""
    preset: str | None = None
    """A named kind of child that this deployment configured. It fills in what the
    caller did not name. It cannot give the child more than the caller has
    (P4-13b)."""
    skills: tuple[str, ...] | None = None
    """The skills that the child gets, by name from the catalog of the caller. To
    name a skill is also an instruction: the full text of the skill goes in the
    prompt of the child. The child therefore starts by following that procedure
    and does not have to fetch it. `None` gives the child every skill that the
    caller has. The caller cannot name a skill that it does not have (P4-13b)."""
    tools: tuple[str, ...] | None = None
    """The tools that the child may call. `None` gives the child every tool that
    the caller has. The caller cannot name a tool that it does not have."""
    profile: str | None = None
    """A named profile for the child. The child keeps only what that profile runs
    of what the caller holds. A profile that would give the child more than the
    caller has is refused (S7b)."""


class DeleteArgs(ToolModel):
    child_id: str
    reason: str = "user"


class WaitArgs(ToolModel):
    child_id: str
    """The child to wait for: the `childId` that its spawn returned."""


class Settled(WireModel):
    """What a wait returns for a child that ended `done`: the answer of the child,
    and its value."""

    answer: str = ""
    value: dict[str, Any] | None = None
    """The value of the child, checked against the `output` schema. It is set for
    a child that was asked for an `output`."""


class SpawnHandle(WireModel):
    """What a spawn returns. It is deliberately not the answer."""

    child_id: str
    name: str
    session_id: str
    model: str
    requested_access: Access
    granted_access: Access
    note: str | None = None
    """How the child was admitted, in the sentences of the seam
    (`admission_notes`): a narrower access than was asked for, a model that its
    skill asked for and that it runs without, media that it cannot see, or a
    reasoning effort that it runs without.

    The model needs prose, and the *log* keeps the codes. One generator writes the
    sentences, so the two cannot disagree, and neither goes stale when the
    workspace tier lands."""


class Config(WireModel):
    """The row config for `rlm-bindings`."""

    provider: str = PROVIDER_NAME
    """The `ctx.subagents` provider that `rlm.run` delegates to."""


def _spawn_handle(run: SubagentRun) -> dict[str, Any]:
    """What a spawn returns for the child that `run` names. Either the spawn just
    admitted the child, or, on a resume, the log of the child says that the spawn
    admitted it."""
    return SpawnHandle(
        child_id=run.id,
        name=run.name,
        session_id=run.session_id,
        model=run.model,
        requested_access=run.requested_access,
        granted_access=run.granted_access,
        note=" ".join(admission_notes(run)) or None,
    ).to_wire()


def _render_handle(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    lines = [
        f"admitted {value['name']} ({value['childId']}) on {value['model']}",
        f"session: {value['sessionId']}",
        f"workspace access: {value['grantedAccess']} (requested {value['requestedAccess']})",
    ]
    if value.get("note"):
        lines.append(as_str(value["note"]))
    lines.append("It will reply by agent message; keep working and check later.")
    return text_content("\n".join(lines))


@plugin("rlm-bindings", affects="environment", config=Config, inject=[TOOLS, SUBAGENTS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the `rlm_*` tools and group them as the `rlm` code namespace."""

    tools = ctx.require(TOOLS)

    async def run_child(args: RunArgs, run: ToolRunContext) -> dict[str, Any]:
        if run.agent is None:
            raise ToolCallError(RUN_TOOL, "this tool has to be called by an agent")
        # The error is **not caught and re-raised** here (D15).
        # `SubagentSpawnError` is a `HarnessError`, so the code, the
        # `failure_kind` and whether the turn should end all reach
        # `registry._failure`. A catch here would flatten them.
        handle = await ctx.require(SUBAGENTS).start(
            config.provider,
            SubagentRequest(
                prompt=args.prompt,
                parent=run.agent,
                # The boundary in which the ceiling is computed. It is stated here
                # and not derived from the routing target (P6-31, P6-24).
                scope=run.scope,
                name=args.name,
                reasoning_effort=args.thinking,
                choice=ModelChoice(key=args.model or ""),
                output=args.output,
                profile=args.profile,
                access=args.access,
                preset=args.preset,
                skills=args.skills,
                tools=args.tools,
                # Inside a cell, this is the id of the dispatch itself. Outside a
                # cell, it is the call id of the model. In both cases it is the
                # record that a crash leaves for `reconciled_spawn`.
                call_id=run.call_id,
            ),
        )
        return _spawn_handle(handle)

    async def reconciled_spawn(_args: Any, call: SessionEvent, session: Session) -> Reconciled:  # noqa: ANN401
        """Whether an `rlm.run` that a crash cut short admitted a child. The logs
        of the children give the answer (S2).

        The whole value of the call is the admission handle, and the admission
        reaches the disk of the child before the child runs. The child is
        therefore the answer:

        * `Done`, with the handle that the call would have returned, when an
          admission names this call. What the child has done since then does not
          matter, because the resume sweep puts the child back to work.
        * `NotDone` when no admission names this call. No child was started, and
          a spawn is what the program asked for.

        The children are read from the store (`load_children`), because this
        question is asked while the parent resumes, before any child is open.
        """
        children = await ctx.require(SUBAGENTS).load_children(session.id, session.header.family)
        child = admitted_by(children, call)
        return NotDone() if child is None else Done(_spawn_handle(child.run()))

    def list_children(_args: object, run: ToolRunContext) -> dict[str, Any]:
        """The children, each as its own log describes it. There is no side table."""
        session = run.session
        children = ctx.require(SUBAGENTS).children(session.id) if session is not None else {}
        return {"children": [state.to_wire() for state in children.values()]}

    async def wait_for_child(args: WaitArgs, run: ToolRunContext) -> dict[str, Any]:
        """A failed child fails the call, and the error keeps the last text of the
        child. A canceled wait stops waiting and leaves the child working. The
        `task` tool, in contrast, releases the child."""
        session = run.session
        try:
            awaited = ctx.require(SUBAGENTS).waiter(session.id if session else "", args.child_id)
        except LookupError as refused:
            raise ToolCallError(WAIT_TOOL, str(refused)) from refused
        result = await raced(run.signal, awaited)
        if result is None:
            raise Canceled(f"the wait for {args.child_id} was canceled")
        if result.status != "done":
            raise ToolCallError(WAIT_TOOL, ended_text(args.child_id, result))
        return Settled(
            answer=result.answer,
            value=thaw_json(result.value) if result.value is not None else None,
        ).to_wire()

    async def delete_child(args: DeleteArgs, run: ToolRunContext) -> dict[str, Any]:
        session = run.session
        removed = (
            await ctx.require(SUBAGENTS).delete(session, args.child_id, reason=args.reason)
            if session is not None
            else False
        )
        return {"deleted": removed, "childId": args.child_id}

    tools.register(
        define_tool(
            RUN_TOOL,
            _RUN_DESCRIPTION,
            parameters=RunArgs,
            output=ToolOutput(schema=SpawnHandle, render=_render_handle),
            execute=run_child,
            reconcile=reconciled_spawn,
        )
    )
    tools.register(
        define_tool(
            LIST_TOOL,
            "Your children: name, id, status, and whether each was deleted.",
            parameters={"type": "object", "properties": {}},
            output={"type": "object"},
            render=_render_roster,
            execute=list_children,
            is_concurrency_safe=True,
        )
    )
    tools.register(
        define_tool(
            WAIT_TOOL,
            "Wait for one of your children to end, and get its value: what it returned "
            "in the shape you asked for with `output`, else its answer. The one wait "
            "there is, for when your next step needs it; a child that failed is an error.",
            parameters=WaitArgs,
            output=ToolOutput(schema=Settled, render=_render_settled),
            execute=wait_for_child,
        )
    )
    tools.register(
        define_tool(
            DELETE_TOOL,
            "Revoke a child. Its transcript stays on disk, ending in a tombstone.",
            parameters=DeleteArgs,
            output={"type": "object"},
            render=_render_deleted,
            execute=delete_child,
        )
    )

    def namespace(request: CodeBindingsRequest) -> CodeBindingNamespace:
        """`rlm.run` / `.list_subagents` / `.wait` / `.delete_subagent`, bound to
        the run."""
        view = ctx.require(TOOLS).view(request.scope)
        specs = (
            ("run", RUN_TOOL, True),
            ("list_subagents", LIST_TOOL, False),
            ("wait", WAIT_TOOL, False),
            ("delete_subagent", DELETE_TOOL, False),
        )
        bindings = [
            governed_binding(request, public, definition.schema(), counts_as_spawn=spawns)
            for public, tool_name, spawns in specs
            # A tool that is restricted away for this agent is also absent from
            # the SDK block. The prompt therefore cannot offer what a cell could
            # not call.
            if (definition := view.visible.get(tool_name)) is not None
        ]
        return CodeBindingNamespace(
            name=NAMESPACE,
            description="delegate to child agents; every spawn is governed and recorded",
            bindings=tuple(bindings),
        )

    tools.register_code_namespace(NAMESPACE, namespace)


def _render_roster(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    rows = value.get("children") or []
    if not rows:
        return text_content("no children")
    lines = [
        f"- {row.get('name')} ({row.get('runId')}) "
        f"{'deleted' if row.get('deleted') else row.get('status', 'queued')}"
        for row in rows
    ]
    return text_content("\n".join(lines))


def _render_settled(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    returned = value.get("value")
    return text_content(dumps(returned) if returned is not None else as_str(value.get("answer")))


def _render_deleted(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    return text_content(f"deleted {value['childId']}" if value["deleted"] else "no such child")
