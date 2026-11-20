"""`rlm-harness`: the Continual Harness. Its state is a fold, and `/refine` is a job.

The fold and the checks are in `state.py` and `service.py`. The model that writes
a proposal is in `planner.py`. The rule for when a refinement runs without a
request is in `auto.py`. This module is the row: what it provides, what it puts
in the prompt, and the one command.

Four design decisions need an explanation:

* **The command is not a tool.** The *user* asks for `/refine`. A route through
  a model turn would be slower. It would also make the log misleading: the log
  would show the model deciding something that the user decided. The command
  dispatches directly and records `command/run` and `command/done`
  (`ctx.commands`).
* **The gate of H3 is an approval request, not a tool call.** The plan describes
  a global edit as a `tools/pre-execute` `ask` "on the `refine` tool". There is
  no such tool, because `/refine` is a command, so no `pre-execute` waterfall
  runs for it. `ctx.approval.request(tool_name="refine")` asks the same question
  through the same seam. The answerer of the deployment sees the request, the
  log records `approval/asked` and `approval/decided`, and the request fails
  closed when there is nowhere to ask (B3). A `refine` tool that the model can
  call would follow the wording of the plan. It would also give the model the
  one operation that the design reserves for the user.
* **Refining is a job, not a command body.** A planner pass makes a model call
  that outlives the keystroke that asked for it. That is the example that
  `ctx.jobs` itself gives. The scope of the *agent* owns the job. A refinement
  is about one conversation, and when that conversation goes, the pass has
  nothing left to learn from.
* **An automatic pass is local only.** `refine --global` is a decision for a
  human. An automatic global pass would put an approval prompt in front of a
  user who asked for nothing, about a change that reaches every future project.
  Nothing here passes `scope="global"` unless a person typed it.

@module ph_rlm.harness
"""

from __future__ import annotations

import logging

from ph.cancel import raced
from ph.cordis import Context, MaybeAwaitable, plugin
from ph.keys import AGENTS, COMMANDS, JOBS, LLM, SESSIONS, SYSTEM_PROMPT, TOOLS
from ph.paths import resolve_roots
from ph.seams.commands import CommandContext, CommandDefinition, CommandVerb, Verbs
from ph.seams.jobs import Job
from ph.session import Session, SessionEvent
from ph.session.writers import log_writer
from ph.system_prompt.assembly import AssembleContext, PromptContext
from ph.wire import WireModel

from ..keys import HARNESS
from .auto import (
    CONSIDERED,
    COOLDOWN_MINUTES,
    TURNS_BETWEEN_REFINEMENTS,
    RefineRequest,
    due,
    veto_reason,
)
from .planner import CONVERSATION_CHARS, PLANNER_MAX_TOKENS, PlannerError, RefinementPlanner
from .service import HarnessService, RefinementRefused
from .state import (
    GLOBAL_LOG_NAME,
    KINDS,
    PROJECTION_NAME,
    REFINED,
    HarnessEdit,
    HarnessEntry,
    HarnessReference,
    HarnessScope,
    HarnessState,
    RefinementProposal,
    entry_label,
    fold_events,
    fold_session,
    read_global_events,
    refinement_line,
)

_LOG = log_writer(__name__)

__all__ = [
    # What a consumer of the harness needs: the row, the state vocabulary that
    # a caller of `ctx.harness` handles, and the folds. Everything else stays in
    # its submodule. The sibling `kernel` package follows the same rule. A
    # re-export list that is longer than what consumers use only adds churn.
    "CONSIDERED",
    "GLOBAL_LOG_NAME",
    "PROJECTION_NAME",
    "REFINED",
    "Config",
    "HarnessEdit",
    "HarnessEntry",
    "HarnessReference",
    "HarnessState",
    "PlannerError",
    "RefinementProposal",
    "RefinementRefused",
    "apply",
    "due",
    "fold_events",
    "fold_session",
    "read_global_events",
    "render_state",
]

log = logging.getLogger("ph_rlm.harness")

MAX_PER_KIND = 12
MAX_REFINEMENTS = 5
"""The bounds that prime-agent puts on the prompt section. Without them, a
harness that kept growing would in the end be the whole prompt."""


class Config(WireModel):
    """The row config for `rlm-harness`."""

    max_per_kind: int = MAX_PER_KIND
    max_refinements: int = MAX_REFINEMENTS
    auto_refine: bool = True
    """Whether the harness refines itself at the end of a turn (H7). The port
    keeps this on by default. A deployment that wants a refinement only on
    request sets this to false. The `/refine` command works with either value."""
    turns_between_refinements: int = TURNS_BETWEEN_REFINEMENTS
    cooldown_minutes: int = COOLDOWN_MINUTES
    conversation_chars: int = CONVERSATION_CHARS
    max_tokens: int = PLANNER_MAX_TOKENS


def render_state(state: HarnessState, *, per_kind: int, refinements: int) -> str:
    """`formatHarnessStateForPrompt`, ported: bounded lists, newest last.

    The text is rendered from the fold, so every entry that the model is told
    about is an entry that the log carries. The result is empty when the harness
    is empty, because a heading with nothing under it costs tokens and says
    nothing.
    """
    lines: list[str] = []
    for kind in KINDS:
        entries = state.of_kind(kind)
        if not entries:
            continue
        lines.append(f"## {kind.title()}s")
        for entry in entries[:per_kind]:
            where = f", {entry.path}" if entry.path else ""
            call = f" — {entry.call_pattern}" if entry.call_pattern else ""
            lines.append(f"- {entry_label(entry, detail=where)}{call}")
        if len(entries) > per_kind:
            lines.append(f"- … and {len(entries) - per_kind} more")
    recent = state.refinements[-refinements:]
    if recent:
        lines.append("## Recent refinements")
        lines.extend(refinement_line(one) for one in recent)
    if not lines:
        return ""
    return "# Continual Harness State\n" + "\n".join(lines)


@plugin(
    "rlm-harness",
    affects="environment",
    config=Config,
    inject=[SYSTEM_PROMPT, COMMANDS, TOOLS, SESSIONS, AGENTS, LLM, JOBS],
)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the harness: the state, the prompt section, `/refine`, auto-refine."""
    service = HarnessService(ctx=ctx, directory=resolve_roots().harness_dir())
    ctx.provide(HARNESS, service)
    planner = RefinementPlanner(
        ctx=ctx,
        service=service,
        conversation_chars=config.conversation_chars,
        max_tokens=config.max_tokens,
    )
    running: set[str] = set()
    """The sessions that have a pass in progress. This is in-process state on
    purpose. It describes the concurrency of this runner, not the session, and a
    resumed session with no pass running is correct to start one."""

    def section(request: AssembleContext) -> str:
        """The harness as a cache-safe snapshot (A12).

        This is a `context()`, not a `section`. A refinement changes this text in
        the middle of a session. In the cached prefix, every apply would bill the
        whole prompt again.
        """
        session = request.session
        return render_state(
            service.state(session),
            per_kind=config.max_per_kind,
            refinements=config.max_refinements,
        )

    ctx.require(SYSTEM_PROMPT).context(PromptContext(name="rlm:harness", order=20, text=section))

    # ----------------------------------------------------------- the pass --

    async def refine(request: RefineRequest) -> str:
        """Plan one refinement and apply it. Return the line to show.

        Every outcome that is *not* a refinement records
        `harness/refine-considered`: a declined review, an empty proposal, a
        planner that failed. That event advances the cooldown, so a broken
        planner or a quiet conversation costs one cheap call and not one call
        for each turn.
        """
        when_idle = getattr(request.agent, "when_idle", None)
        if when_idle is not None:
            # This is the "waits for agent idle" of the plan. A pass that read a
            # half-finished turn would refine on a conversation that the user is
            # still having.
            await when_idle()

        vetoed = await veto_reason(ctx, request)
        if vetoed is not None:
            # The veto is recorded, not silent. A user who typed `/refine` must
            # get an answer, and after the command has returned this is the only
            # place where one can appear. The veto costs no tokens, because the
            # waterfall runs before the review call.
            return _considered(request, f"vetoed: {vetoed}")

        return await _plan_and_apply(request)

    async def _plan_and_apply(request: RefineRequest) -> str:
        session, agent = request.session, request.agent
        instructions = request.instructions
        try:
            if request.trigger != "user":
                verdict = await planner.review(session, agent)
                if not verdict.should_refine:
                    return _considered(request, verdict.rationale or "nothing to record")
                instructions = instructions or verdict.instructions

            proposal = await planner.plan(
                session, agent, scope=request.scope, instructions=instructions
            )
            if not proposal.edits:
                return _considered(request, proposal.summary or "no edits proposed")
            record = await service.apply(proposal, scope=request.scope, agent=agent)
        except (PlannerError, RefinementRefused) as error:
            log.debug("ph_rlm.harness: refinement did not apply", exc_info=True)
            return _considered(request, str(error))

        applied = len(record.applied_edits)
        refused = f", {len(record.rejected)} refused" if record.rejected else ""
        return f"[{record.refine_id}] {record.summary} — {applied} edit(s) applied{refused}"

    def _considered(request: RefineRequest, reason: str) -> str:
        _LOG.append(
            request.session,
            CONSIDERED,
            {"trigger": request.trigger, "scope": request.scope, "reason": reason},
        )
        return f"no refinement: {reason}"

    async def start(request: RefineRequest) -> Job:
        """Run one pass as a job owned by the agent's scope."""

        async def body(job: Job) -> str:
            # The pass is raced against the token of the job. The cancellation
            # of a job is cooperative, and a pass is one or two model calls.
            # Without the race, those calls would continue after the scope of
            # the pass was disposed, past the bounded teardown of a mount or the
            # shutdown of a daemon.
            refined = await raced(job.token, lambda: refine(request))
            return refined if refined is not None else "refinement canceled"

        def settled(job: Job) -> None:
            # This runs however the job ended, including a job that was canceled
            # before `body` ran (S6). When `body` removed the mark, the mark
            # outlived such a job, and the session answered "already running"
            # for the life of the mount.
            running.discard(request.session.id)
            # The job is released, not abandoned. The work is over, so the entry
            # goes now and does not stay in the table until the agent goes.
            # Without this, a session that refines automatically gains one job
            # for each pass. This uses `get`, as `_quiesce` (`subagents.py`)
            # does: a pass that its owner stopped when a mount unwound ends
            # after the row that holds the table has gone.
            jobs = ctx.get(JOBS)
            if jobs is not None:
                jobs.forget(job.id)

        running.add(request.session.id)
        try:
            return await ctx.require(JOBS).start(
                kind="refine",
                label=f"refine {request.session.id} ({request.trigger})",
                run=body,
                scope=request.agent.ctx,
                on_settled=settled,
            )
        except BaseException:
            # A start that failed never ran `body`, so nothing else would remove
            # the mark. Without this, the session answers "already running"
            # forever because of a job that does not exist.
            running.discard(request.session.id)
            raise

    # -------------------------------------------------------- the command --

    def show(_rest: str, invocation: CommandContext) -> str:
        """`/refine --show`: the whole harness, whichever scope was named."""
        # The output is unbounded, unlike the prompt section. A user who audits
        # the harness wants the entries that the bound hides, because there is
        # no other way to see them.
        state = service.state(invocation.session)
        entries = sum(len(rows) for rows in state.entries.values())
        shown = render_state(state, per_kind=entries or 1, refinements=len(state.refinements))
        return shown or "the harness is empty"

    async def command(argument: str, invocation: CommandContext) -> str:
        """`/refine [--global] [--show] [--rollback <id>] [instructions]`."""
        words = argument.split()
        scope: HarnessScope = "global" if "--global" in words else "local"
        rest = [word for word in words if word != "--global"]
        session, agent = invocation.session, invocation.agent
        if "--show" in rest:
            return show("", invocation)
        if rest and rest[0] == "--rollback":
            if len(rest) < 2:
                return "usage: /refine --rollback <refine-id>"
            try:
                record = await service.rollback(rest[1], session=session, agent=agent)
            except RefinementRefused as refused:
                return f"refusing: {refused}"
            return f"rolled back {rest[1]} as {record.refine_id}"
        if session is None or agent is None:
            return "refusing: /refine needs a session and an agent to refine from"
        if session.id in running:
            return "a refinement is already running for this session"

        job = await start(
            RefineRequest(
                session=session,
                agent=agent,
                scope=scope,
                trigger="user",
                instructions=" ".join(rest),
            )
        )
        # The outcome reaches the transcript without help from here. A
        # refinement appends `harness/refined`. A pass that declined appends
        # `harness/refine-considered`. That event is rendered for a pass that
        # the user started, so that an answer which nobody would see otherwise
        # is visible.
        return f"refining the {scope} harness in the background ({job.id})"

    ctx.require(COMMANDS).register(
        CommandDefinition(
            name="refine",
            summary="Refine the Continual Harness, or roll a refinement back.",
            # A line that starts with `--show` only asks. That includes
            # `--show --global`, because the show is the whole harness for either
            # scope. Everything else acts: instructions and a rollback. A `--show`
            # behind `--global` is still shown, but through the durable path that
            # an action takes. When unsure, the durable path is the safe choice.
            run=Verbs(
                {"--show": CommandVerb(show, record="after")}, otherwise=CommandVerb(command)
            ),
            argument_hint="[--global] [--show] [--rollback <id>] [instructions]",
        )
    )

    # ------------------------------------------------------- auto-refine --

    def on_session_event(session: Session, event: SessionEvent) -> MaybeAwaitable[None]:
        """H7: consider a refinement at the end of a turn.

        The check runs on `turn/end` and not on a timer, because that is the one
        moment when the conversation is both complete and quiet.

        **The listener is sync until there is something to consider.**
        `session/event` fires for every event of every session, including each
        streamed chunk. An async listener is a task each time: spawned, held and
        ended for each chunk to read one field. Now only the end of a turn gets
        a task.
        """
        return consider(session) if event.type == "turn/end" else None

    async def consider(session: Session) -> None:
        # This check is here and not in the sync half. It runs in the same step
        # of the task as the `start` that claims the session, so two ends that
        # arrive in one turn of the loop cannot both start a pass.
        if session.id in running:
            return
        trigger = due(
            session,
            turns_between=config.turns_between_refinements,
            cooldown_minutes=config.cooldown_minutes,
        )
        if trigger is None:
            return
        # The id of the loop agent is the id of its session, so the registry
        # answers the hop from session to agent directly. The docstring of
        # `get` warns against a side table where `get` already is one.
        agent = ctx.require(AGENTS).get(session.id)
        if agent is None:
            return
        await start(RefineRequest(session=session, agent=agent, scope="local", trigger=trigger))

    if config.auto_refine:
        ctx.on("session/event", on_session_event)
