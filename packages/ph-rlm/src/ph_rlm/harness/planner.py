"""The planner: the model that writes a refinement, and the gate that decides to.

A `RefinementProposal` comes from one non-reasoning model call whose whole output
is JSON.

There are two calls, and their sizes differ on purpose. **The review gate** is
cheap and answers one question: is there anything here that is worth recording?
Auto-refine fires on a turn count, and most turns teach nothing. **The
planner** is the expensive call. It runs only when the gate says yes, or when a
user typed `/refine`.

**Neither call is a turn.** Both are `purpose="refine"` requests. They are bound
to the session, so that usage is attributed, but they are outside
`is_loop_request`. The "model-visible means logged" invariant therefore does not
hold them to `derive_messages()`. That is correct, because their prompt is
*about* the conversation and is not part of it. For the same reason, nothing
here appends `assistant/message`. The only durable record that a refinement
leaves is `harness/refined`, and the proposal that produced it is on that event.

**The model never writes a `call_pattern`.** It names a `reference`, and the
harness renders the call form (H2). A planner that could write its own call
pattern could write prompt text that steers the next session onto the ungoverned
raw-namespace path. See `service.render_call_pattern`.

@module ph_rlm.harness.planner
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from ph.agent.types import AgentHandle
from ph.characters import tail
from ph.cordis import Context
from ph.keys import LLM
from ph.llm.structured import SchemaViolation, ask_for_shape
from ph.llm.types import GenerateOptions, create_message, text_of
from ph.session import Session
from ph.text import block_marker
from ph.wire import WireModel

from .service import HarnessService
from .state import (
    KINDS,
    HarnessScope,
    HarnessState,
    RefinementProposal,
    entry_label,
    refinement_line,
)

__all__ = [
    "PLANNER_MAX_TOKENS",
    "REFINEMENT_SYSTEM_PROMPT",
    "REVIEW_SYSTEM_PROMPT",
    "PlannerError",
    "RefinementPlanner",
    "ReviewVerdict",
]

log = logging.getLogger("ph_rlm.harness")

PLANNER_MAX_TOKENS = 32_000
"""The ceiling from the plan. A refinement is a few small edits. A model that
could use its full window here would spend the budget of a session on describing
them."""

CONVERSATION_CHARS = 80_000
"""How much of the tail of the conversation the planner is shown."""

REFINEMENT_SYSTEM_PROMPT = """\
You maintain the Continual Harness: a small, durable set of notes, procedures and
skills that is injected into this agent's system prompt in every future session.

You are reading a conversation that has just happened. Your job is to decide what,
if anything, it taught that is worth keeping, and to express that as a few precise
edits.

# What the harness holds

- `note` — something learned that does not fit anywhere else: a fact about this
  project, a preference the user stated, a constraint that keeps coming back.
- `procedure` — how to do something with what already exists: the steps, in order,
  with the actual commands.
- `skill` — a pointer to installed capability. A skill entry MUST carry a
  `reference` naming a Python module and a callable that already exist. If the
  import does not resolve, the edit is rejected: you cannot create capability
  here, only point at it. Ask for a plugin instead.

# Rules

- Write for a reader who has none of this conversation's context. "The tests are
  run with `uv run pytest`" is useful; "the tests are run the way we discussed" is
  not.
- Prefer few, specific edits. Two good entries beat nine vague ones, and the
  prompt section is bounded — an entry that displaces a better one is a cost.
- Update an existing entry rather than adding a near-duplicate. Use its exact id.
- Delete an entry the conversation showed to be wrong.
- Do not restate the agent's base instructions. The id `base_system_prompt` is not
  editable and any edit naming it is rejected.
- Do not write call syntax into `content`. Name the `reference`; the harness
  renders the call form itself.
- If the conversation taught nothing durable, return an empty `edits` list. That
  is a correct answer and a common one.

# Output

Return ONE JSON object and nothing else — no prose, no code fence:

{
  "summary": "one line, what this refinement changes",
  "rationale": "why the conversation supports it",
  "expectedOutcome": "what a future session does differently",
  "edits": [
    {
      "action": "create" | "update" | "delete",
      "kind": "note" | "procedure" | "skill",
      "id": "kebab-case-id",
      "title": "short title",
      "content": "the body",
      "path": "optional/file/path.md",
      "reference": {"type": "python", "module": "pkg.mod", "callable": "name"},
      "reason": "why this edit"
    }
  ]
}
"""

REVIEW_SYSTEM_PROMPT = """\
You decide whether a conversation is worth refining the Continual Harness for.

The harness is a small durable set of notes, procedures and skills injected into
every future session. Refining costs a model call and, done badly, costs prompt
space forever. Most conversations teach nothing durable.

Say yes only if the conversation contains something a *future* session would want
and does not already have: a project fact, a stated preference, a procedure that
worked, a correction of something the harness currently says.

Say no for routine work, for anything already in the state you were shown, and
whenever you are unsure.

Return ONE JSON object and nothing else:

{"shouldRefine": true | false, "rationale": "one line",
 "instructions": "optional: what to focus on"}
"""


class PlannerError(Exception):
    """The planner call failed or did not return usable JSON."""


class ReviewVerdict(WireModel):
    """The cheap gate's answer (H7)."""

    should_refine: bool = False
    rationale: str = ""
    instructions: str = ""


@dataclass(slots=True)
class RefinementPlanner:
    """Turns a conversation into a `RefinementProposal`."""

    ctx: Context
    service: HarnessService
    conversation_chars: int = CONVERSATION_CHARS
    max_tokens: int = PLANNER_MAX_TOKENS

    # ------------------------------------------------------------- the gate --

    async def review(self, session: Session, agent: AgentHandle) -> ReviewVerdict:
        """The cheap pass of H7: is this conversation worth planning over?

        The pass fails *closed* in the cheap direction. An answer that cannot be
        parsed means no refinement, and not an expensive call on a broken reply.
        """
        state = self.service.state(session)
        prompt = (
            f"{self._state_overview(state)}\n\n# The conversation\n\n{self._conversation(session)}"
        )
        try:
            return await self._shaped(
                agent, session, system=REVIEW_SYSTEM_PROMPT, user=prompt, shape=ReviewVerdict
            )
        except PlannerError as error:
            log.debug("ph_rlm.harness: review gate declined: %s", error)
            return ReviewVerdict(rationale=str(error))

    # ---------------------------------------------------------- the planner --

    async def plan(
        self,
        session: Session,
        agent: AgentHandle,
        *,
        scope: HarnessScope = "local",
        instructions: str = "",
    ) -> RefinementProposal:
        """One call, one proposal. `HarnessService.apply` applies it.

        Planning is kept apart from applying so that the planner cannot become a
        second path into the state. Whatever the planner returns still goes
        through the same validation as a proposal that is typed by hand.
        """
        return await self._shaped(
            agent,
            session,
            system=REFINEMENT_SYSTEM_PROMPT,
            user=self._planner_prompt(session, scope=scope, instructions=instructions),
            shape=RefinementProposal,
        )

    def _planner_prompt(self, session: Session, *, scope: HarnessScope, instructions: str) -> str:
        state = self.service.state(session)
        parts = [
            self._state_overview(state),
            self._history(state),
            f"# Scope\n\nYou are editing the **{scope}** harness."
            + (
                "\nGlobal entries reach every future session in every project, so only"
                " deployment-wide truths belong here."
                if scope == "global"
                else "\nLocal entries belong to this session and the forks taken from it."
            ),
        ]
        if instructions:
            # The instructions come after the harness sections and before the
            # conversation, under a heading that names them as the request of
            # the user. The model then does not read them as one more piece of
            # the conversation that it is summarizing.
            parts.append(f"# What the user asked for\n\n{instructions}")
        parts.append(f"# The conversation\n\n{self._conversation(session)}")
        return "\n\n".join(parts)

    def _state_overview(self, state: HarnessState) -> str:
        """Everything that the harness already holds, so that the model can update
        an entry and not duplicate it. Each entry is labeled by `entry_label`,
        the same identity that the prompt section shows, so "use its exact id" is
        an instruction that both texts obey."""
        lines: list[str] = ["# The harness as it stands"]
        for kind in KINDS:
            entries = state.of_kind(kind)
            if not entries:
                continue
            lines.append(f"\n## {kind.title()}s")
            lines.extend(f"- {entry_label(entry)}\n  {entry.content}" for entry in entries)
        if len(lines) == 1:
            lines.append("\nThe harness is empty.")
        return "\n".join(lines)

    def _history(self, state: HarnessState) -> str:
        recent = state.refinements[-10:]
        if not recent:
            return "# Recent refinements\n\nNone."
        return "# Recent refinements\n\n" + "\n".join(refinement_line(one) for one in recent)

    def _conversation(self, session: Session) -> str:
        """The tail of the derived conversation, as text.

        The source is `derive_messages()` and not the raw log, so the planner
        reads what the model saw. A compacted range is summarized here exactly
        as it is for the agent. The planner therefore does not learn from
        history that the agent has already been told to forget.
        """
        budget = self.conversation_chars
        parts: list[str] = []
        used = 0
        for message in reversed(session.derive_messages()):
            text = text_of(message.content, placeholder=block_marker)
            parts.append(f"{message.role}: {text}")
            used += len(parts[-1]) + 2
            if used > budget:
                # The messages are rendered newest first, and the loop stops at
                # the budget. A long session therefore pays for the tail that it
                # sends, and not for the history that it would drop in any case.
                break
        rendered = "\n\n".join(reversed(parts))
        if len(rendered) <= budget:
            return rendered
        return "…\n\n" + tail(rendered, budget)

    # ------------------------------------------------------------ the call --

    async def _shaped[Shape: WireModel](
        self, agent: AgentHandle, session: Session, *, system: str, user: str, shape: type[Shape]
    ) -> Shape:
        """One non-loop model call that must come back as `shape`.

        The call goes through `ph.llm.structured` (P7-17), and not "ask for JSON
        and parse". The `shape` class is the wire schema *and* the validator, so
        what comes back is already the typed thing that this planner wanted. The
        visible difference: a confirmation sentence in front of the document now
        leads to a corrected call and not to a declined refinement.

        `purpose="refine"` keeps the call out of the conversation. The call is
        bound to the session, so that usage is attributed and a telemetry sink
        sees it. It is not a loop request, so the "model-visible means logged"
        invariant does not hold its messages to `derive_messages()`. Nothing is
        appended. The record of the refinement is `harness/refined`, and the
        output of this call is on it.
        """
        options = agent.options
        provider = str(options.provider or "")
        model = str(options.model or "")
        if not provider or not model:
            raise PlannerError("the agent has no model route to plan a refinement with")

        resolved = self.ctx.require(LLM).resolve_model(provider, model)
        request = GenerateOptions(
            provider=provider,
            model=model,
            messages=(
                create_message(
                    role="user",
                    content=[{"type": "text", "text": user}],
                    source={"kind": "plugin", "plugin": "rlm-harness", "form": "instructions"},
                ),
            ),
            system=system,
            # `min(model max, max_tokens)`, where `max_tokens` is 32_000 by
            # default. A refinement is a few small edits, and no reasoning
            # budget is requested, because this is a summarization.
            max_tokens=min(resolved.default_max_tokens or self.max_tokens, self.max_tokens),
            session_id=session.id,
            purpose="refine",
        )
        try:
            return await ask_for_shape(
                self.ctx.require(LLM).stream, request, shape, enforced=resolved.structured_output
            )
        except SchemaViolation as violation:
            raise PlannerError(str(violation)) from violation
