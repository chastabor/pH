"""The planner: the model that writes a refinement, and the gate that decides to.

A `RefinementProposal` comes from one non-reasoning model call whose whole output
is JSON.

Two calls, deliberately different sizes. **The review gate** is cheap and answers
one question — is there anything here worth recording — because auto-refine fires
on a turn count and most turns teach nothing. **The planner** is the expensive one
and only runs when something says yes, or when a human typed `/refine`.

**Neither call is a turn.** They are `purpose="refine"` requests: session-bound so
usage is attributed, but outside `is_loop_request`, so the "model-visible means
logged" invariant does not hold them to `derive_messages()` — which is right,
because their prompt is *about* the conversation rather than part of it. That is
also why nothing here appends `assistant/message`: the only durable record a
refinement leaves is `harness/refined`, and the proposal that produced it is on
that event.

**The model never writes a `call_pattern`.** It names a `reference`, and the
harness renders the call form (H2). A planner that could author its own would be
able to write prompt text steering the next session onto the ungoverned
raw-namespace path — see `service.render_call_pattern`.

@module ph_rlm.harness.planner
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ph.cordis import Context
from ph.llm.structured import SchemaViolation, ask_for_shape
from ph.llm.types import GenerateOptions, create_message, text_of
from ph.session import Session
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
"""The ceiling from the plan. A refinement is a handful of small edits; a model
allowed its full window here would spend a session's budget describing them."""

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

    async def review(self, session: Session, agent: Any) -> ReviewVerdict:
        """H7's cheap pass: is this conversation worth planning over?

        Fails *closed* in the cheap direction — an unparseable answer means no
        refinement rather than an expensive call on a broken reply.
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
        agent: Any,
        *,
        scope: HarnessScope = "local",
        instructions: str = "",
    ) -> RefinementProposal:
        """One call, one proposal. Applying it is `HarnessService.apply`'s job.

        Kept apart from applying so the planner cannot become a second path into
        the state: whatever it returns still goes through the same validation as
        a proposal typed by hand.
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
            # Last, and named as the human's, so the model does not read it as
            # one more piece of the conversation it is summarizing.
            parts.append(f"# What the user asked for\n\n{instructions}")
        parts.append(f"# The conversation\n\n{self._conversation(session)}")
        return "\n\n".join(parts)

    def _state_overview(self, state: HarnessState) -> str:
        """Everything the harness already holds, so the model can update rather
        than duplicate — labelled by `entry_label`, the same identity the prompt
        section shows, so "use its exact id" is an instruction both texts obey."""
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

        `derive_messages()` rather than the raw log, so the planner reads what the
        model actually saw — a compacted range is summarized here exactly as it is
        for the agent, instead of the planner learning from history the agent has
        already been told to forget.
        """
        budget = self.conversation_chars
        parts: list[str] = []
        used = 0
        for message in reversed(session.derive_messages()):
            text = text_of(message.content, placeholder=lambda kind: f"[{kind}]")
            parts.append(f"{message.role}: {text}")
            used += len(parts[-1]) + 2
            if used > budget:
                # Rendered newest-first and stopped at the budget, so a long
                # session pays for the tail it sends, not for the history it
                # was always going to drop.
                break
        rendered = "\n\n".join(reversed(parts))
        if len(rendered) <= budget:
            return rendered
        return "…\n\n" + rendered[-budget:]

    # ------------------------------------------------------------ the call --

    async def _shaped[Shape: WireModel](
        self, agent: Any, session: Session, *, system: str, user: str, shape: type[Shape]
    ) -> Shape:
        """One non-loop model call that must come back as `shape`.

        Through `ph.llm.structured` (P7-17) rather than "ask for JSON and parse":
        the model is the wire schema *and* the validator, so what comes back is
        already the typed thing this planner wanted. What changed for a reader is
        that a confirmation sentence in front of the document is now a corrected
        call rather than a declined refinement.

        `purpose="refine"` is what keeps it out of the conversation: session-bound
        so usage is attributed and a telemetry sink sees it, but not a loop
        request, so the "model-visible means logged" invariant does not hold its
        messages to `derive_messages()`. Nothing is appended — the refinement's
        record is `harness/refined`, and this call's output is on it.
        """
        options = getattr(agent, "options", None)
        provider = str(getattr(options, "provider", "") or "")
        model = str(getattr(options, "model", "") or "")
        if not provider or not model:
            raise PlannerError("the agent has no model route to plan a refinement with")

        resolved = self.ctx.llm.resolve_model(provider, model)
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
            # `min(model max, 32_000)`: a refinement is a handful of small edits,
            # and no reasoning budget is asked for — this is a summarization.
            max_tokens=min(resolved.default_max_tokens or self.max_tokens, self.max_tokens),
            session_id=session.id,
            purpose="refine",
        )
        try:
            return await ask_for_shape(
                self.ctx.llm.stream, request, shape, enforced=resolved.structured_output
            )
        except SchemaViolation as violation:
            raise PlannerError(str(violation)) from violation
