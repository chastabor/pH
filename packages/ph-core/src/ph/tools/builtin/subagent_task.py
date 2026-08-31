"""`subagent-task` — delegate a piece of work and wait for the answer.

The other spelling of delegation already exists: `rlm_run` admits a child and
hands back a handle, and the child replies later by agent message. That shape
needs an inbox, a roster and a model that knows to keep working and check back
— all of which Code Mode has and a plain tool-calling deployment does not. So
this row is the **blocking** one: one call, one answer, no protocol for the
model to learn.

Both are worth having, and the difference is not a preference. A parent that
fans out eight children wants `rlm_run`, because waiting on the first would
serialize the other seven. A parent that needs one sub-answer to continue wants
this, because a handle it cannot await is a handle it will forget to collect.

**It registers only when a provider is actually mounted**, at `profile/mounted`
rather than at this row's own `apply`. Two reasons, and the second is the one
that matters: a tool advertised in every prompt and refused on every call spends
the context window teaching the model a capability the deployment does not have
— and reading the provider at `apply` would have made this row's position in the
profile decide whether it works, which is the ordering trap `base.yaml` promises
its rows do not have.

@module ph.tools.builtin.subagent_task
"""

from __future__ import annotations

from functools import partial
from typing import Any

from pydantic import Field

from ...cordis import Context, plugin
from ...seams.subagents import (
    Access,
    SubagentRequest,
    SubagentSpawnError,
    SubagentStatus,
    downgrade_text,
)
from ...wire import WireModel
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..presentation import ToolCallView, ToolResultView
from ..registry import register_when_composed

__all__ = ["Config", "TaskArgs", "TaskValue", "apply"]

TOOL = "task"

DESCRIPTION = """Delegate one self-contained piece of work to a subagent and wait for its answer.

The child starts from your prompt alone: it does not see this conversation, so
say what to do, what to look at, and what to report back. Use it for work that
is separable and has a summarizable answer — a search across many files, a
review, a question you would otherwise read twenty files to settle. Do the work
yourself when it is short, or when you need the intermediate steps rather than
the conclusion."""


class TaskArgs(ToolModel):
    prompt: str = Field(description="The child's whole instruction. It sees nothing else.")
    name: str | None = Field(None, description="A short label for this child, shown in the roster.")
    access: Access = Field(
        "read",
        description=(
            "Whether the child may write the workspace. Ask for `write` only when "
            "the task is to change files; the deployment may grant less."
        ),
    )
    preset: str | None = Field(
        None,
        description=(
            "A named kind of child this deployment configured. Fills in skills, tools "
            "and access you did not name; it cannot give the child more than you have."
        ),
    )
    skills: tuple[str, ...] | None = Field(
        None,
        description=(
            "Skills the child gets, by name from your own catalog. Naming one is also "
            "an instruction: its full text goes in the child's prompt, so it starts by "
            "following that procedure. Omit to give it everything you have. You cannot "
            "name a skill you do not have yourself."
        ),
    )
    tools: tuple[str, ...] | None = Field(
        None,
        description=(
            "Tools the child may call. Omit to give it everything you have. Narrowing "
            "is a way to keep a child on its task; you cannot name a tool you do not "
            "have yourself."
        ),
    )


class TaskValue(ToolModel):
    child_id: str
    name: str
    session_id: str
    status: SubagentStatus
    answer: str = ""
    granted_access: Access = "read"
    note: str | None = None
    """Why `granted_access` is narrower than asked, rendered from the seam's code
    so the prose the model reads and the code the log keeps cannot disagree."""


class Config(WireModel):
    """Row config."""

    provider: str = ""
    """Which `ctx.subagents` provider runs the child.

    Empty means "the one that is mounted", which is the ordinary case and saves
    every profile from naming it. A deployment that mounts two must choose: the
    row stands down and says so rather than picking one, because "run a child
    agent" having two answers is exactly why the seam names providers at all.
    """


def _render(_args: Any, value: Any) -> Any:
    parts = [str(value.get("answer") or "(the child produced no answer)")]
    if value.get("note"):
        parts.append(str(value["note"]))
    return text_content("\n\n".join(parts))


@plugin("subagent-task", config=Config, inject=["tools", "subagents"])
async def apply(ctx: Context, config: Config) -> None:
    """Register the blocking delegation tool, once a provider exists to run it."""

    async def delegate(provider: str, args: TaskArgs, run: ToolRunContext) -> Any:
        try:
            handle = await ctx.subagents.start(
                provider,
                SubagentRequest(
                    prompt=args.prompt,
                    parent=run.agent,
                    # The boundary the ceiling is computed in, stated rather than
                    # derived from the routing target (P6-31, P6-24).
                    scope=run.scope,
                    name=args.name,
                    access=args.access,
                    preset=args.preset,
                    skills=args.skills,
                    tools=args.tools,
                ),
            )
        except SubagentSpawnError as error:
            # The model's to handle: it can retry with a different name, a
            # shallower plan, or by doing the work itself.
            raise ValueError(str(error)) from error
        if handle.result is None:
            raise ValueError(
                f"the {provider!r} subagent provider cannot be waited on; "
                "this deployment needs the handle-and-collect tools instead"
            )
        outcome = await handle.result()
        if outcome.status != "done":
            # A failure, not a value with a sad field: a child that was cancelled
            # or fell over did not answer the question, and a parent reading
            # `answer: ""` as an answer is the misreading this prevents.
            raise ValueError(
                f"subagent {handle.name} ({handle.session_id}) ended as {outcome.status}"
                + (f": {outcome.error}" if outcome.error else "")
            )
        reason = handle.downgrade_reason
        return TaskValue(
            child_id=handle.id,
            name=handle.name,
            session_id=handle.session_id,
            status=outcome.status,
            answer=outcome.answer,
            granted_access=handle.granted_access,
            note=downgrade_text(reason) if reason is not None else None,
        ).model_dump()

    def build_tool() -> Any:
        """The tool, bound to the provider that will run it.

        Resolved here rather than per call, so the deployment's answer to "which
        provider" is taken once, at the moment the profile is whole — and the
        refusal a caller reads names the provider the tool was actually built
        for rather than whatever is registered when it happens to fail.
        """
        provider = ctx.subagents.resolve(config.provider or None)
        if provider is None:
            return None
        return define_tool(
            TOOL,
            DESCRIPTION,
            parameters=TaskArgs,
            output=ToolOutput(schema=TaskValue, render=_render),
            execute=partial(delegate, provider),
            # The fan-out §4.8 opens with: several children at once is the point
            # of delegating, and each one has its own workspace.
            is_concurrency_safe=True,
            present_call=lambda args: ToolCallView(
                card="generic",
                title=f"Delegate to {args.get('name') or 'a subagent'}",
                input=str(args.get("prompt", ""))[:200],
            ),
            present_result=lambda args, result: ToolResultView(
                card="generic",
                title=f"Subagent {args.get('name') or 'result'}",
                is_error=result.is_error,
            ),
        )

    register_when_composed(ctx, build_tool)
