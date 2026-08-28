"""`rlm-harness` — the Continual Harness: state as a fold, `/refine` as a job.

The fold and the checks are in `state.py` and `service.py`; the model that writes
a proposal is in `planner.py`; when it runs unasked is in `auto.py`. This module
is the row: what it provides, what it puts in the prompt, and the one command.

Four placements worth stating:

* **The command is not a tool.** `/refine` is something the *human* asks for, so
  routing it through a model turn would be both slower and dishonest — the log
  would show the model deciding something the user decided. It dispatches
  directly and records `command/run`/`command/done` (`ctx.commands`).
* **H3's gate is an approval request, not a tool call.** The plan describes a
  global edit as a `tools/pre-execute` `ask` "on the `refine` tool" — but there is
  no such tool, precisely because `/refine` is a command, so no `pre-execute`
  waterfall ever fires for it. `ctx.approval.request(tool_name="refine")` asks
  the same question through the same seam: the deployment's answerer sees it, the
  log records `approval/asked` and `approval/decided`, and it fails closed with
  nowhere to ask (B3). Registering a model-callable `refine` tool to satisfy the
  letter would hand the model the one operation the design says is the human's.
* **Refining is a job, not a command body.** A planner pass makes a model call
  that outlives the keystroke that asked for it, which is `ctx.jobs`' own example.
  It is owned by the *agent's* scope: a refinement is about one conversation, and
  when that conversation goes there is nothing left for the pass to learn from.
* **An automatic pass is local-only.** `refine --global` is a human's decision;
  an automatic one would put an approval prompt in front of a user who did not
  ask for anything, about a change reaching every future project. Nothing here
  passes `scope="global"` unless a person typed it.

@module ph_rlm.harness
"""

from __future__ import annotations

import logging
from typing import Any

from ph.cordis import Context, plugin
from ph.paths import resolve_roots
from ph.seams.commands import CommandDefinition
from ph.system_prompt.assembly import PromptContext
from ph.wire import WireModel

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

__all__ = [
    # What a consumer of the harness actually needs: the row, the state
    # vocabulary a caller of `ctx.harness` handles, and the folds. Everything
    # else lives in its submodule — the sibling `kernel` package keeps the same
    # line, and a re-export list longer than the used surface is only churn.
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
"""Prime Agent's bounds on the prompt section. A harness that grew without them
would eventually be the whole prompt."""


class Config(WireModel):
    """Row config for `rlm-harness`."""

    max_per_kind: int = MAX_PER_KIND
    max_refinements: int = MAX_REFINEMENTS
    auto_refine: bool = True
    """H7, ported on. A deployment that would rather refine only when asked sets
    this false; the command works either way."""
    turns_between_refinements: int = TURNS_BETWEEN_REFINEMENTS
    cooldown_minutes: int = COOLDOWN_MINUTES
    conversation_chars: int = CONVERSATION_CHARS
    max_tokens: int = PLANNER_MAX_TOKENS


def render_state(state: HarnessState, *, per_kind: int, refinements: int) -> str:
    """`formatHarnessStateForPrompt`, ported: bounded lists, newest last.

    Rendered from the fold, so an entry the model is told about is one the log
    actually carries. Empty when the harness is empty — a heading with nothing
    under it costs tokens to say nothing.
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
    config=Config,
    inject=["system_prompt", "commands", "tools", "sessions", "agents", "llm", "jobs"],
)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the harness: the state, the prompt section, `/refine`, auto-refine."""
    service = HarnessService(ctx=ctx, directory=resolve_roots().harness_dir())
    ctx.provide("harness", service)
    planner = RefinementPlanner(
        ctx=ctx,
        service=service,
        conversation_chars=config.conversation_chars,
        max_tokens=config.max_tokens,
    )
    running: set[str] = set()
    """Sessions with a pass in flight. In-process state on purpose: it is about
    this runner's concurrency, not about the session, and a resumed session with
    no pass running is correct to start one."""

    def section(request: Any) -> str:
        """The harness as a cache-safe snapshot (A12).

        A `context()`, not a `section`: a refinement changes this text mid-session,
        and in the cached prefix every apply would re-bill the whole prompt.
        """
        session = getattr(request.agent, "session", None)
        return render_state(
            service.state(session),
            per_kind=config.max_per_kind,
            refinements=config.max_refinements,
        )

    ctx.system_prompt.context(PromptContext(name="rlm:harness", order=20, text=section))

    # ----------------------------------------------------------- the pass --

    async def refine(request: RefineRequest) -> str:
        """Plan one refinement and apply it. Returns the line worth showing.

        Every outcome that is *not* a refinement records `harness/refine-considered`
        — a declined review, an empty proposal, a planner that failed. That event
        is what advances the cooldown, so a broken planner or a quiet conversation
        costs one cheap call rather than one per turn.
        """
        when_idle = getattr(request.agent, "when_idle", None)
        if when_idle is not None:
            # The plan's "waits for agent idle": a pass that read a half-finished
            # turn would refine on a conversation the user is still having.
            await when_idle()

        vetoed = await veto_reason(ctx, request)
        if vetoed is not None:
            # Recorded, not silent: a user who typed `/refine` is owed an answer,
            # and this is the only place one can appear once the command has
            # returned. It costs no tokens, which is what the waterfall running
            # before the review call is for.
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
            record = await service.apply(
                proposal, scope=request.scope, session=session, agent=agent
            )
        except (PlannerError, RefinementRefused) as error:
            log.debug("ph_rlm.harness: refinement did not apply", exc_info=True)
            return _considered(request, str(error))

        applied = len(record.applied_edits)
        refused = f", {len(record.rejected)} refused" if record.rejected else ""
        return f"[{record.refine_id}] {record.summary} — {applied} edit(s) applied{refused}"

    def _considered(request: RefineRequest, reason: str) -> str:
        request.session.append(
            CONSIDERED,
            {"trigger": request.trigger, "scope": request.scope, "reason": reason},
        )
        return f"no refinement: {reason}"

    async def start(request: RefineRequest) -> Any:
        """Run one pass as a job owned by the agent's scope."""

        async def body(job: Any) -> str:
            try:
                return await refine(request)
            finally:
                running.discard(request.session.id)
                # Released, not abandoned: the work is over, so the entry goes
                # rather than sitting in the table until the agent does. Without
                # this an auto-refining session accretes one job per pass.
                ctx.jobs.forget(job.id)

        running.add(request.session.id)
        try:
            return await ctx.jobs.start(
                kind="refine",
                label=f"refine {request.session.id} ({request.trigger})",
                run=body,
                scope=getattr(request.agent, "ctx", None) or ctx,
            )
        except BaseException:
            # A start that failed never ran `body`, so nothing else would take
            # the flag down — without this, the session answers "already
            # running" forever on the strength of a job that does not exist.
            running.discard(request.session.id)
            raise

    # -------------------------------------------------------- the command --

    async def command(argument: str, invocation: Any) -> str:
        """`/refine [--global] [--show] [--rollback <id>] [instructions]`."""
        words = argument.split()
        scope: HarnessScope = "global" if "--global" in words else "local"
        rest = [word for word in words if word != "--global"]
        session, agent = invocation.session, invocation.agent
        if "--show" in rest:
            # Unbounded, unlike the prompt section: a human auditing the harness
            # wants the entries the bound hides, which are exactly the ones they
            # cannot see any other way.
            state = service.state(session)
            entries = sum(len(rows) for rows in state.entries.values())
            shown = render_state(state, per_kind=entries or 1, refinements=len(state.refinements))
            return shown or "the harness is empty"
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
        # The outcome reaches the transcript on its own: a refinement appends
        # `harness/refined`, and a pass that declined appends
        # `harness/refine-considered`, which is rendered for a user-triggered
        # pass precisely so an answer nobody would otherwise see is visible.
        return f"refining the {scope} harness in the background ({job.id})"

    ctx.commands.register(
        CommandDefinition(
            name="refine",
            summary="Refine the Continual Harness, or roll a refinement back.",
            run=command,
            argument_hint="[--global] [--show] [--rollback <id>] [instructions]",
        )
    )

    # ------------------------------------------------------- auto-refine --

    async def on_session_event(session: Any, event: Any) -> None:
        """H7: consider refining at the end of a turn.

        On `turn/end` rather than on a timer, because that is the one moment the
        conversation is both complete and quiet.
        """
        if event.type != "turn/end" or session.id in running:
            return
        trigger = due(
            session,
            turns_between=config.turns_between_refinements,
            cooldown_minutes=config.cooldown_minutes,
        )
        if trigger is None:
            return
        # The loop agent's id is its session's id, so the registry answers the
        # session→agent hop directly — its own docstring warns against keeping
        # a side table where `get` already is one.
        agent = ctx.agents.get(session.id)
        if agent is None:
            return
        await start(RefineRequest(session=session, agent=agent, scope="local", trigger=trigger))

    if config.auto_refine:
        ctx.on("session/event", on_session_event)
