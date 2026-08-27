"""`rlm-bindings` — the `rlm` namespace, as governed tool calls (P3-10, C2/C3).

Prime Agent's `await rlm("task", name=..., model=...)` travelled over a Jupyter
comm channel as a `host.request`, which meant it was invisible to
`tools/pre-execute`, to `ctx.approval`, to the call limits and to the offload
policy. Here the same call is a **binding**: one `call` frame out of the kernel,
through the whole tool pipeline, landing as a durable
`tool/code-dispatch-start`/`tool/code-dispatch` pair. A deployment can therefore
deny or `ask` on subagent spawning by policy, and `max_subagent_spawns_per_run`
counts it, without a line of code here.

Two decisions worth stating:

**The program-facing name is not the tool name.** A cell writes `rlm.run(...)`;
the governed tool is `rlm_run`. A namespace cannot claim a bare global name like
`run` — and it does not need to, because the SDK renders the namespaced form
while the log records the tool. Only the tools exist as a registry surface, so a
policy row addresses `rlm_run` and means exactly one thing.

**Spawning is `counts_as_spawn`.** The bridge holds it to C4's spawn budget
rather than the general dispatch budget, so one approved cell cannot fan out
past `max_subagent_spawns_per_run` on the strength of that one approval.

`find_models` is deliberately absent: it would need a model *catalogue* on
`ctx.llm`, which does not exist yet. Advertising a discovery call that could only
answer "I don't know" would be worse than not offering one, and the model already
inherits the parent's model when it names none.

@module ph_rlm.bindings
"""

from __future__ import annotations

from typing import Any

from ph.cordis import Context, plugin
from ph.seams.code_runtime import CodeBindingNamespace
from ph.seams.subagents import (
    Access,
    DowngradeReason,
    SubagentRequest,
    SubagentSpawnError,
    downgrade_text,
    subagent_roster,
)
from ph.tools import ToolModel, ToolOutput, define_tool, text_content
from ph.tools.code_mode import CodeBindingsRequest, ToolCallError, governed_binding
from ph.wire import WireModel

from .subagents import PROVIDER_NAME

__all__ = ["NAMESPACE", "Config", "apply"]

NAMESPACE = "rlm"

RUN_TOOL = "rlm_run"
LIST_TOOL = "rlm_list_subagents"
DELETE_TOOL = "rlm_delete_subagent"

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
    thinking: str | None = None
    access: Access = "read"
    """`read` | `write`. Defaults to `read` (E4): a child that only needs to read
    must not be handed a writable repo because nobody said otherwise. Typed, so
    pydantic validates it and the model sees the two values in the schema."""


class DeleteArgs(ToolModel):
    child_id: str
    reason: str = "user"


class SpawnHandle(WireModel):
    """What a spawn hands back. Deliberately not the answer."""

    child_id: str
    name: str
    session_id: str
    model: str
    requested_access: Access
    granted_access: Access
    note: str | None = None
    """The sentence for `SubagentRun.downgrade_reason`, rendered by the seam.

    The model needs prose; the *log* keeps the code. One generator, so the two
    cannot disagree and neither goes stale when the workspace tier lands."""


class Config(WireModel):
    """Row config for `rlm-bindings`."""

    provider: str = PROVIDER_NAME
    """Which `ctx.subagents` provider `rlm.run` delegates to."""


def _render_handle(_args: Any, value: Any) -> Any:
    lines = [
        f"admitted {value['name']} ({value['childId']}) on {value['model']}",
        f"session: {value['sessionId']}",
        f"workspace access: {value['grantedAccess']} (requested {value['requestedAccess']})",
    ]
    if value.get("note"):
        lines.append(str(value["note"]))
    lines.append("It will reply by agent message; keep working and check later.")
    return text_content("\n".join(lines))


@plugin("rlm-bindings", config=Config, inject=["tools", "subagents", "rlm_children"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the `rlm_*` tools and group them as the `rlm` code namespace."""

    async def run_child(args: RunArgs, run: Any) -> Any:
        try:
            handle = await ctx.subagents.start(
                config.provider,
                SubagentRequest(
                    prompt=args.prompt,
                    parent=run.agent,
                    name=args.name,
                    reasoning_effort=args.thinking,
                    model=args.model,
                    access=args.access,
                ),
            )
        except SubagentSpawnError as error:
            # The model's to handle: it can retry with a different name, a
            # shallower plan, or by doing the work itself.
            raise ToolCallError(RUN_TOOL, str(error)) from error
        reason: DowngradeReason | None = handle.downgrade_reason
        return SpawnHandle(
            child_id=handle.id,
            name=handle.name,
            session_id=handle.session_id,
            model=handle.model,
            requested_access=handle.requested_access,
            granted_access=handle.granted_access,
            note=downgrade_text(reason) if reason is not None else None,
        ).to_wire()

    def list_children(_args: Any, run: Any) -> Any:
        """The roster, folded from the parent's own log — never a side table."""
        session = run.session
        rows = list(subagent_roster(session).values()) if session is not None else []
        return {"children": rows}

    async def delete_child(args: DeleteArgs, run: Any) -> Any:
        session = run.session
        removed = (
            await ctx.rlm_children.delete(session, args.child_id, reason=args.reason)
            if session is not None
            else False
        )
        return {"deleted": removed, "childId": args.child_id}

    ctx.tools.register(
        define_tool(
            RUN_TOOL,
            _RUN_DESCRIPTION,
            parameters=RunArgs,
            output=ToolOutput(schema=SpawnHandle, render=_render_handle),
            execute=run_child,
        )
    )
    ctx.tools.register(
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
    ctx.tools.register(
        define_tool(
            DELETE_TOOL,
            "Revoke a child. Its transcript stays on disk; the roster keeps a tombstone.",
            parameters=DeleteArgs,
            output={"type": "object"},
            render=_render_deleted,
            execute=delete_child,
        )
    )

    def namespace(request: CodeBindingsRequest) -> CodeBindingNamespace:
        """`rlm.run` / `.list_subagents` / `.delete_subagent`, bound to the run."""
        view = ctx.tools.view(request.scope)
        specs = (
            ("run", RUN_TOOL, True),
            ("list_subagents", LIST_TOOL, False),
            ("delete_subagent", DELETE_TOOL, False),
        )
        bindings = [
            governed_binding(request, public, definition, counts_as_spawn=spawns)
            for public, tool_name, spawns in specs
            # A tool restricted away for this agent is absent from the SDK block
            # too, so the prompt cannot offer what a cell could not call.
            if (definition := view.visible.get(tool_name)) is not None
        ]
        return CodeBindingNamespace(
            name=NAMESPACE,
            description="delegate to child agents; every spawn is governed and recorded",
            bindings=tuple(bindings),
        )

    ctx.tools.register_code_namespace(NAMESPACE, namespace)


def _render_roster(_args: Any, value: Any) -> Any:
    rows = value.get("children") or []
    if not rows:
        return text_content("no children")
    lines = [
        f"- {row.get('name')} ({row.get('runId')}) "
        f"{'deleted' if row.get('deleted') else row.get('status', 'queued')}"
        for row in rows
    ]
    return text_content("\n".join(lines))


def _render_deleted(_args: Any, value: Any) -> Any:
    return text_content(f"deleted {value['childId']}" if value["deleted"] else "no such child")
