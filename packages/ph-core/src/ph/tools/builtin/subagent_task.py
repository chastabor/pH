"""`subagent-task`: delegate a piece of work and wait for the answer.

The other form of delegation already exists. `rlm_run` admits a child and
returns a handle, and the child replies later by agent message. That form needs
an inbox, a roster, and a model that knows to keep working and check back. Code
Mode has all of these, and a plain tool-calling deployment does not. So this row
is the **blocking** form: one call, one answer, and no protocol for the model to
learn.

Both forms are worth having, and the difference is not a preference. A parent
that fans out eight children needs `rlm_run`, because a wait for the first child
would serialize the other seven. A parent that needs one sub-answer to continue
needs this tool, because it will forget to collect a handle that it cannot
await.

**The tool registers only when a provider is mounted.** It registers at
`profile/mounted` and not in the `apply` of this row, for two reasons. The
second reason matters more. First, a tool that is advertised in every prompt and
refused on every call uses the context window to teach the model a capability
that the deployment does not have. Second, if the row read the provider at
`apply`, the position of this row in the profile would decide whether the row
works. `base.yaml` promises that its rows do not have that ordering trap.

@module ph.tools.builtin.subagent_task
"""

from __future__ import annotations

from functools import partial
from typing import Any

from pydantic import Field

from ...cancel import Canceled, raced
from ...characters import head
from ...cordis import Context, maybe_await, plugin, releasing
from ...json import JsonObject, as_str, dumps
from ...keys import SUBAGENTS, TOOLS
from ...llm.types import ContentBlock
from ...seams.models import ModelChoice
from ...seams.subagents import (
    Access,
    SubagentRequest,
    SubagentResult,
    SubagentRun,
    SubagentStatus,
    admission_notes,
    admitted_by,
    child_is_live,
    ended_text,
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
    output: dict[str, Any] | None = Field(
        None,
        description=(
            "A JSON Schema object for the result you need back, when you need a "
            "structure rather than prose. The child returns a value matching it, which "
            "comes back as `value`, or the call fails. A skill you give it may declare "
            "one; this overrides it."
        ),
    )


class TaskValue(ToolModel):
    child_id: str
    name: str
    session_id: str
    status: SubagentStatus
    answer: str = ""
    value: dict[str, Any] | None = None
    """The result of the child in the shape that was asked for (`output`),
    checked."""
    granted_access: Access = "read"
    note: str | None = None
    """What the model should know in addition to the answer. For a child that
    outlived its wait, the note says where the child is (`STILL_WORKING`). The
    note also says how the child was admitted (`admission_notes`): why
    `granted_access` is narrower than the request, and why the child runs without
    the model that its skill asked for. The text is rendered from the codes of
    the seam, so the prose that the model reads and the codes that the log keeps
    cannot disagree."""


class Config(WireModel):
    """Row config."""

    provider: str = ""
    """The `ctx.subagents` provider that runs the child.

    An empty value means "the provider that is mounted". That is the ordinary
    case, and it saves every profile from naming the provider. A deployment that
    mounts two providers must choose. Without a choice, the row stands down and
    says so, and does not pick one. "Run a child agent" with two answers is the
    reason that the seam names providers.
    """


WAIT_CANCELED = "the `task` call that waited for this child was canceled"
"""The reason on the `canceled` record and the tombstone of a child whose wait
was canceled (`_collected`)."""

STILL_WORKING = (
    "This child was still working when the harness stopped, so the wait for it ended "
    "here. It is started again with its parent and reports back by message when it "
    "finishes; do not delegate the same task again."
)
"""What a `task` call says about a child that outlived its wait (S1, S2). Either
a harness that was stopping suspended the child, or the resume found the child
in the roster, still live."""


def _task_value(
    run: SubagentRun,
    status: SubagentStatus,
    answer: str = "",
    value: JsonObject | None = None,
) -> dict[str, Any]:
    """The value of the call for the child that `run` names. The call learned of
    the child from its own wait, or from the roster on resume. One builder
    serves both, so the two cannot disagree."""
    notes = [] if status == "done" else [STILL_WORKING]
    notes.extend(admission_notes(run))
    return TaskValue(
        child_id=run.id,
        name=run.name,
        session_id=run.session_id,
        status=status,
        answer=answer,
        value=dict(value) if value is not None else None,
        granted_access=run.granted_access,
        note=" ".join(notes) or None,
    ).model_dump()


def _render(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    if value.get("status") != "done":
        # A child that outlived its wait has no answer yet. The note says where
        # the child is.
        return text_content(as_str(value.get("note")))
    returned = value.get("value")
    parts = [
        dumps(returned)
        if returned is not None
        else as_str(value.get("answer"), "(the child produced no answer)")
    ]
    if value.get("note"):
        parts.append(as_str(value["note"]))
    return text_content("\n\n".join(parts))


@plugin("subagent-task", affects="environment", config=Config, inject=[TOOLS, SUBAGENTS])
async def apply(ctx: Context, config: Config) -> None:
    """Register the blocking delegation tool when a provider exists to run it."""

    async def delegate(provider: str, args: TaskArgs, run: ToolRunContext) -> dict[str, Any]:
        if run.agent is None:
            # A spawn is made *by* somebody. The provider reads the session of
            # the parent to log the admission, and the options of the parent to
            # build the child.
            raise ValueError(f"the {TOOL!r} tool has to be called by an agent")
        # **The error is not caught and raised again** (D15).
        # `SubagentSpawnError` is a `HarnessError`, so the code, the
        # `failure_kind`, and whether the turn should end all reach
        # `registry._failure`. A catch here would flatten them.
        handle = await ctx.require(SUBAGENTS).start(
            provider,
            SubagentRequest(
                prompt=args.prompt,
                parent=run.agent,
                # The boundary that the ceiling is computed in. It is stated
                # here and not derived from the routing target (P6-31, P6-24).
                scope=run.scope,
                name=args.name,
                access=args.access,
                preset=args.preset,
                profile=args.profile,
                choice=ModelChoice(key=args.model or ""),
                output=args.output,
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
        outcome = await _collected(handle, run, run.agent.session)
        # `queued` is a child that a stopping harness suspended (S1). The child
        # will come back, and it has not failed. A parent that is told that the
        # child failed delegates the task again.
        if outcome.status not in ("done", "queued"):
            # This is a failure, and not a value with an error field. A child
            # that was canceled or that crashed did not answer the question. A
            # parent that reads `answer: ""` as an answer is the misreading that
            # this raise prevents.
            raise ValueError(
                f"subagent {ended_text(f'{handle.name} ({handle.session_id})', outcome)}"
            )
        return _task_value(handle, outcome.status, outcome.answer, outcome.value)

    async def reconciled(_args: Any, call: SessionEvent, session: Session) -> Reconciled:  # noqa: ANN401
        """Did a `task` call that a crash cut short start a child? The logs of
        the children say (S2).

        The question is asked while the parent resumes, before any child is
        open. So the children are read from the store (`load_children`): each
        admission names the call that asked for it. When no admission carries
        the id of this call, the call started nothing, because the admission
        reaches the disk of the child before the child runs. The answer is then
        `NotDone`, and asking again is what the call wanted. A child that is
        still live is `Done`, which is the correct ending of the wait: the child
        was delegated, the resume puts it back to work, and its answer comes by
        message. A child that has ended is `Unknown`. The answer that the
        cut-short wait would have returned reached nobody, and a child that
        ended cannot be asked for it again.
        """
        children = await ctx.require(SUBAGENTS).load_children(session.id, session.header.family)
        child = admitted_by(children, call)
        if child is None:
            return NotDone()
        if not child_is_live(child):
            return Unknown()
        status: SubagentStatus = "queued" if child.status == "queued" else "running"
        return Done(_task_value(child.run(), status))

    async def _collected(
        handle: SubagentRun, run: ToolRunContext, parent: Session | None
    ) -> SubagentResult:
        """Wait for the child, and revoke it if this call is canceled (C7).

        `delegate` used to observe nothing. A parent that was interrupted during
        a delegation went on waiting for a child that nobody had told to stop.
        The child went on spending model calls against the same budget, for a
        turn whose result was already discarded.

        **The cancel is a revocation** (`SubagentService.delete`). The provider
        stops the child, and the log of the child gets `canceled` and a
        tombstone. `SubagentRun.dispose` alone is not enough. It stops the child
        and writes nothing, because the seam uses it for a child that the seam
        has already ended. The log of a child that was only disposed still read
        `running`, and the next start of the harness put that child back to work
        for a wait that nobody held.

        `dispose` still follows the revocation. A provider that cannot revoke
        (`RevokingProvider`) gets only the tombstone from the seam, and
        `dispose` is what stops its child. For a child that the provider revoked
        it does nothing. A parent with no session (a stub agent) has no roster
        to revoke from, and its child is only disposed.
        """
        awaiter = handle.result
        assert awaiter is not None  # the caller checked; this narrows it
        collected = await raced(run.signal, awaiter)
        if collected is not None:
            return collected
        # Canceled. The child is revoked first and the cancel is reported after.
        # A parent that raises while its child is still running is the state
        # that this function exists to prevent. The shield is for a cancel that
        # reaches this task as well as the token: the ending of the child must
        # still be written (`releasing`).
        with releasing():
            if parent is not None:
                await ctx.require(SUBAGENTS).delete(parent, handle.id, reason=WAIT_CANCELED)
            if handle.dispose is not None:
                await maybe_await(handle.dispose())
        # **`Canceled`, not `ValueError`** (N3). `registry._failure` already
        # maps this class to `aborted_result`. A `ValueError` took the other
        # branch and told the model that the `task` tool had *failed*. The
        # interrupt of the user then read as a fault of the harness, and a model
        # that reads a fault tries again.
        raise Canceled(f"the wait for subagent {handle.name} was canceled")

    def build_tool() -> ToolDefinition | None:
        """The tool, bound to the provider that will run it.

        The provider is resolved here and not for each call. The deployment
        answers "which provider" one time, at the moment the profile is whole.
        The refusal that a caller reads therefore names the provider that the
        tool was built for, and not whatever is registered when a call fails.
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
            # This is the fan-out that §4.8 opens with. Several children at one
            # time is the purpose of delegation, and each child has its own
            # workspace.
            is_concurrency_safe=True,
            present_call=lambda args: ToolCallView(
                card="generic",
                title=f"Delegate to {args.get('name') or 'a subagent'}",
                input=head(str(args.get("prompt", "")), 200),
            ),
            present_result=lambda args, result: ToolResultView(
                card="generic",
                title=f"Subagent {args.get('name') or 'result'}",
                is_error=result.is_error,
            ),
        )

    register_when_composed(ctx, build_tool)
