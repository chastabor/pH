"""The planner: the model that writes a refinement, and the gate that decides to.

P3-16's second increment. The first shipped everything a refinement *is* — the
fold, validation, apply, rollback — and left one question open: where does a
`RefinementProposal` come from. Here, from one non-reasoning model call whose
whole output is JSON.

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

import json
import logging
from dataclasses import dataclass
from typing import Any

from ph.cordis import Context
from ph.llm import BlockAssembler
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
    "parse_json_object",
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


def parse_json_object(text: str) -> dict[str, Any]:
    """The one JSON object in a model reply.

    Tolerant of a fence and of prose around it, because "return only JSON" is an
    instruction and not a guarantee — and strict about the result being an
    object, because a planner that returned a list would otherwise half-apply.
    """
    candidates = [text.strip()]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise PlannerError("the model did not return a JSON object")


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
            reply = await self._complete(agent, session, system=REVIEW_SYSTEM_PROMPT, user=prompt)
            parsed = parse_json_object(reply)
        except PlannerError as error:
            log.debug("ph_rlm.harness: review gate declined: %s", error)
            return ReviewVerdict(rationale=str(error))
        return ReviewVerdict(
            should_refine=bool(parsed.get("shouldRefine")),
            rationale=str(parsed.get("rationale") or ""),
            instructions=str(parsed.get("instructions") or ""),
        )

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
        reply = await self._complete(
            agent,
            session,
            system=REFINEMENT_SYSTEM_PROMPT,
            user=self._planner_prompt(session, scope=scope, instructions=instructions),
        )
        parsed = parse_json_object(reply)
        try:
            return RefinementProposal.model_validate(parsed)
        except Exception as error:
            raise PlannerError(f"the proposal did not validate: {error}") from error

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

    async def _complete(self, agent: Any, session: Session, *, system: str, user: str) -> str:
        """One non-loop model call, on the agent's own route; the reply as text.

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
        # `BlockAssembler` is the loop's own assembly — the one algorithm that
        # decides what a chunk stream said, so the planner cannot disagree with
        # the transcript about a reply. `text_of` then drops reasoning blocks
        # rather than handing them to the JSON parser.
        assembler = BlockAssembler()
        async for chunk in await self.ctx.llm.stream(request):
            assembler.push(chunk)
        if assembler.finish.kind == "error":
            failure = assembler.finish.failure
            raise PlannerError(failure.message if failure is not None else "the model call failed")
        return text_of(assembler.blocks()).strip()
