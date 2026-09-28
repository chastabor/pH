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

from ...cancel import Canceled, raced
from ...cordis import Context, maybe_await, plugin
from ...json import JsonObject, as_str
from ...keys import SUBAGENTS, TOOLS
from ...llm.types import ContentBlock
from ...seams.subagents import (
    Access,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentStatus,
    admitted_by,
    child_is_live,
    downgrade_text,
)
from ...session import Session, SessionEvent
from ...wire import WireModel
from ..definition import (
    Done,
    NotDone,
    Reconciled,
    ToolDefinition,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    Unknown,
    define_tool,
    text_content,
)
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
    model: str | None = Field(
        None,
        description=(
            "A model your profile lists, by key. Omit to run the child on your own; a "
            "skill you give it may name the one it needs."
        ),
    )
    profile: str | None = Field(
        None,
        description=(
            "A named profile for the child: it keeps only what that profile runs of "
            "what you hold. One that would give it more than you have is refused."
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
    """What the model should know beside the answer: for a child that outlived its
    wait, where it is (`STILL_WORKING`); and why `granted_access` is narrower than
    asked, rendered from the seam's code so the prose the model reads and the code
    the log keeps cannot disagree."""


class Config(WireModel):
    """Row config."""

    provider: str = ""
    """Which `ctx.subagents` provider runs the child.

    Empty means "the one that is mounted", which is the ordinary case and saves
    every profile from naming it. A deployment that mounts two must choose: the
    row stands down and says so rather than picking one, because "run a child
    agent" having two answers is exactly why the seam names providers at all.
    """


STILL_WORKING = (
    "This child was still working when the harness stopped, so the wait for it ended "
    "here. It is started again with its parent and reports back by message when it "
    "finishes; do not delegate the same task again."
)
"""What a `task` call says about a child that outlived its wait (S1, S2): suspended by a
harness that is stopping, or found in the roster on resume, still live."""


def _task_value(run: SubagentRun, status: SubagentStatus, answer: str = "") -> dict[str, Any]:
    """The call's value for the child `run` names, however the call learned of it — its
    own wait, or the roster on resume. One builder, so the two cannot disagree."""
    notes = [] if status == "done" else [STILL_WORKING]
    if run.downgrade_reason is not None:
        notes.append(downgrade_text(run.downgrade_reason))
    return TaskValue(
        child_id=run.id,
        name=run.name,
        session_id=run.session_id,
        status=status,
        answer=answer,
        granted_access=run.granted_access,
        note=" ".join(notes) or None,
    ).model_dump()


def _render(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("status") != "done":
        # A child that outlived its wait has no answer yet; the note says where it is.
        return text_content(as_str(value.get("note")))
    parts = [as_str(value.get("answer"), "(the child produced no answer)")]
    if value.get("note"):
        parts.append(as_str(value["note"]))
    return text_content("\n\n".join(parts))


@plugin("subagent-task", affects="environment", config=Config, inject=[TOOLS, SUBAGENTS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the blocking delegation tool, once a provider exists to run it."""

    async def delegate(provider: str, args: TaskArgs, run: ToolRunContext) -> dict[str, Any]:
        if run.agent is None:
            # A spawn is made *by* somebody: the provider reads the parent's
            # session to log the admission and its options to build the child.
            raise ValueError(f"the {TOOL!r} tool has to be called by an agent")
        # **Not caught and re-raised** (D15): `SubagentSpawnError` is a
        # `HarnessError`, so the code, the `failure_kind` and whether the turn
        # should end reach `registry._failure` instead of being flattened here.
        handle = await ctx.require(SUBAGENTS).start(
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
                profile=args.profile,
                model_key=args.model,
                skills=args.skills,
                tools=args.tools,
                call_id=run.call_id,
            ),
        )
        if handle.result is None:
            raise ValueError(
                f"the {provider!r} subagent provider cannot be waited on; "
                "this deployment needs the handle-and-collect tools instead"
            )
        outcome = await _collected(handle, run)
        # `queued` is a child suspended by a harness that is stopping (S1): coming
        # back, not failed — and a parent told it failed delegates it again.
        if outcome.status not in ("done", "queued"):
            # A failure, not a value with a sad field: a child that was canceled
            # or fell over did not answer the question, and a parent reading
            # `answer: ""` as an answer is the misreading this prevents.
            raise ValueError(
                f"subagent {handle.name} ({handle.session_id}) ended as {outcome.status}"
                + (f": {outcome.error}" if outcome.error else "")
            )
        return _task_value(handle, outcome.status, outcome.answer)

    async def reconciled(_args: Any, call: SessionEvent, session: Session) -> Reconciled:  # noqa: ANN401
        """Did a `task` call a crash cut short start a child? The parent's roster says (S2).

        No admission under this call's id is a call that started nothing — the
        admission reaches disk before the child runs — so it is `NotDone`, and asking
        again is what the call wanted. A child still live is `Done` as the wait's
        honest ending: it was delegated, the resume puts it back to work, and its
        answer comes by message. One that ended without an answer is `Unknown`, as a
        failed wait would have read.
        """
        row = admitted_by(ctx.require(SUBAGENTS).roster(session), call)
        if row is None:
            return NotDone()
        if not child_is_live(row):
            return Unknown()
        status: SubagentStatus = "queued" if row.get("status") == "queued" else "running"
        return Done(_task_value(SubagentRun.of(row), status))

    async def _collected(handle: SubagentRun, run: ToolRunContext) -> SubagentResult:
        """Wait for the child, releasing it if this call is canceled (C7).

        `delegate` observed nothing, so a parent interrupted mid-delegation went
        on waiting for a child nobody had told to stop — and the child went on
        spending model calls against the same budget, for a turn whose result
        was already discarded.

        `dispose` is the release the seam already provides for it ("releases the
        child early"), so the cancel reaches the child through the path a
        disposed parent would have used rather than through a second mechanism.
        """
        awaiter = handle.result
        assert awaiter is not None  # the caller checked; this narrows it
        collected = await raced(run.signal, awaiter)
        if collected is not None:
            return collected
        # Cancelled. Released first, then reported — a parent that raises while
        # its child is still running is the state this exists to prevent.
        await maybe_await(handle.dispose() if handle.dispose is not None else None)
        # **`Canceled`, not `ValueError`** (N3). `registry._failure` already maps
        # this class to `aborted_result`; a `ValueError` took the other branch
        # and told the model the `task` tool had *failed*, so a person's own
        # interrupt read as the harness breaking — and a model that reads
        # breakage retries.
        raise Canceled(f"the wait for subagent {handle.name} was canceled")

    def build_tool() -> ToolDefinition | None:
        """The tool, bound to the provider that will run it.

        Resolved here rather than per call, so the deployment's answer to "which
        provider" is taken once, at the moment the profile is whole — and the
        refusal a caller reads names the provider the tool was actually built
        for rather than whatever is registered when it happens to fail.
        """
        provider = ctx.require(SUBAGENTS).resolve(config.provider or None)
        if provider is None:
            return None
        return define_tool(
            TOOL,
            DESCRIPTION,
            parameters=TaskArgs,
            output=ToolOutput(schema=TaskValue, render=_render),
            execute=partial(delegate, provider),
            reconcile=reconciled,
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
