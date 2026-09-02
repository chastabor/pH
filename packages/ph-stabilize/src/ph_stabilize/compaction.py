"""`compaction-summarize` — the conversation replaced by a summary (P4-03, G4).

Deep Agents' `SummarizationMiddleware` and dsh's `compaction-basic`, landed on
pH's seams as one `CompactionEngine`. The numbers are upstream's, from the pinned
releases: trigger at `0.85` of the window and keep `0.10` of it when the window is
known, `170 000` tokens / `6 messages` when it is not
(`compute_summarization_defaults` in `deepagents`).

Invariants this row holds:

* **It is a surface `replace` (A3)** — the whole safety argument. The summary is
  appended as a `user/message` whose `surfaceOp` shadows the range it stands for,
  citing every shadowed seq. `derive_messages()` yields the summary; the log keeps
  every event; `transcript()` still shows the person the conversation they had.
  The same sentence covers the two cheaper remedies below, so **the log never
  changes**.
* **The cut never splits a call from its result.** A cutoff from the retention
  budget is moved *back* to the nearest balanced boundary. Balance is folded over
  the surface in current order (dsh's `tool-pairing`), not over step markers,
  because a previous compaction has already moved what "position" means. If the
  only balanced cut is the start of the conversation, nothing is compacted and
  the attempt **says so** — shipping an orphaned `tool-result` to a provider that
  rejects it is not a repair.
* **Two cheaper remedies run before the expensive one.** Over-long call arguments
  in *retained* history are elided on every pressure check (§7.4 item 2); on
  overflow the trailing tool-result batch is spilled and pointed at (§7.4 item 7)
  before summarization is attempted at all — that batch is precisely the shape
  that leaves no balanced cut.
* **The request replays the conversation's own envelope**: same `system`, same
  `tools`, the shadowed messages as themselves, and the extraction prompt
  appended as the last *user* message. So the request is a strict prefix of the
  one the loop just made, which is what makes the call nearly free on a provider
  that caches prefixes (A12) — and why the prompt cannot go in `system`, since
  anything put there changes the prefix. It also means the summarizer sees the
  whole range as real `tool-call`/`tool-result` blocks rather than a rendered
  tail, so nothing is silently withheld from it.
* **Carrying the tools is the cost of carrying the prefix**, and the two are not
  separable. A model primed to act may answer with a tool call and no prose, so a
  reply with no text is retried once in the self-contained shape — without tools,
  and with the whole range. Cheap first, correct second.
* **Overflow never replays**, because the envelope in question is the request a
  provider just refused for being too large. It sends upstream's own 4 000-token
  tail (`_DEFAULT_TRIM_TOKEN_LIMIT`, `strategy="last"`) rendered as text, which is
  also the only shape that can carry it — a *suffix* of a balanced range may begin
  with an orphaned tool result. When that tail is trimmed the rendered text
  **says so**, because a model told to write "SESSION INTENT" from the middle of a
  conversation, and not told that is what it is reading, will state the middle as
  the intent.

Two triggers, and they ask genuinely different questions: `agent/pre-step` asks
"will the next request be too big" from an *estimate*, because no usage number
exists for a request nobody has made; `agent/request-error` asks nothing — the
provider has already answered `CONTEXT_WINDOW_EXCEEDED`, which `llm-retry`
declines to retry so this row can see it.

@module ph_stabilize.compaction
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ph.agent.types import PreStepDecision, RequestErrorAction, RequestFailure
from ph.agent_loop import AgentCancelled
from ph.cancel import Cancelled
from ph.cordis import DEPLOYMENT, Context, plugin
from ph.llm import BlockAssembler
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    GenerateOptions,
    Message,
    PluginSource,
    ReasoningBlock,
    TextBlock,
    TokenUsage,
    ToolCallBlock,
    ToolResultBlock,
    create_message,
    create_user_message,
    text_of,
)
from ph.seams.compaction import CompactionError, CompactionResult, CompactionTrigger
from ph.seams.spill import SpillClaim
from ph.seams.token_meter import TokenBaseline
from ph.session import (
    EpochHeader,
    Session,
    SessionEvent,
    SurfaceIntent,
    derive_event_message,
    thaw_json,
)
from ph.session.events import SurfaceReplace
from ph.session.json import dumps
from ph.text import count_of
from ph.wire import WireModel

from .offload import HISTORY_PREFIX, spill_tool_result

__all__ = [
    "KEEP_FRACTION",
    "MAX_ARG_LENGTH",
    "REPLACEMENT_WITHOUT_PATH",
    "SUMMARY_MAX_TOKENS",
    "TRIGGER_FRACTION",
    "TRIGGER_TOKENS",
    "TRUNCATION_TEXT",
    "Config",
    "SummarizeEngine",
    "TruncateArgsConfig",
    "apply",
    "balanced_cuts",
    "cuts_over",
    "render_for_summary",
    "safe_cutoff",
    "truncated_arguments",
    "truncated_assistant_payload",
]

log = logging.getLogger("ph_stabilize.compaction")

# --------------------------------------------------------------------------
# `compute_summarization_defaults` in
# `deepagents/middleware/summarization.py`: the fractions when the model
# profile knows its window, the fixed pair when it does not. Both branches ship
# here because pH has the same split — `request/context` carries a
# `context_window` only when the adapter resolved one.
# --------------------------------------------------------------------------

TRIGGER_FRACTION = 0.85
"""Compact when the estimate reaches this fraction of a known window."""

KEEP_FRACTION = 0.10
"""Leave this fraction of a known window untouched at the tail."""

TRIGGER_TOKENS = 170_000
"""The window-unknown trigger. Deliberately conservative, as upstream says:
without a window there is nothing to be a fraction of, and overshooting an
unknown limit costs the turn."""

KEEP_MESSAGES = 6
"""The window-unknown retention. Messages, not tokens, for the same reason."""

SUMMARY_MAX_TOKENS = 8_192
"""Generation cap for the summarize call — dsh's `compaction-basic` default."""

SUMMARY_INPUT_TOKENS = 4_000
"""How much of the range being replaced the summarizer is shown, tail first.
langchain's `_DEFAULT_TRIM_TOKEN_LIMIT`."""

# --- argument truncation (§7.4 item 2; upstream `truncate_args_settings`) ----

EXTRA_TRUNCATE_TOOLS: tuple[str, ...] = ()
"""Names to elide regardless of what they declare — the escape hatch.

Empty by default, because `ToolDefinition.arguments_disposable` is where the
answer belongs. Upstream matches a hardcoded `{"write_file", "edit_file"}`, and
that is exactly the pattern this bundle rejected one row over: pH's fs tools are
named `read`/`write`/`edit`, so upstream's list matches *nothing* here, and a
deployment that renames them or an MCP server that adds its own would silently
get no truncation with nothing failing. So the tool declares it, the same way
`self_limits` is declared for the offload row.

The restriction itself is kept, and is doing real work: a long `run_code` cell
in old history is exactly the argument a model re-reads, while a file body it
already wrote — and can `read` back — is not."""

MAX_ARG_LENGTH = 2_000
"""Above this, one string argument is elided. Upstream's `max_length`."""

ARG_HEAD_CHARS = 20
"""How much of an elided argument is kept — upstream's `value[:20]`."""

TRUNCATION_TEXT = "...(argument truncated)"
"""Upstream's `truncation_text`, verbatim."""

TRUNCATE_TRIGGER_MESSAGES = 20
"""The window-unknown trigger for truncation — upstream's `("messages", 20)`.
Different from summarization's 170 000 tokens, and deliberately: eliding an
argument costs no model call, so it can fire far earlier."""

TRUNCATE_KEEP_MESSAGES = 20
"""The window-unknown retention for truncation — upstream's `("messages", 20)`,
and not summarization's 6: what is *kept whole* here is a much longer tail,
because truncation is a cheap trim rather than a replacement of the history."""

# --- the overflow clip (§7.4 item 7; upstream `_clip_overflow_tail`) ---------

OVERFLOW_CLIP_TOKENS = 5_000
"""The clip threshold when no window is known — upstream's own fallback in
`_derive_overflow_clip_threshold_tokens`, which it describes as equivalent to a
20 000-character floor under a `chars / 4` approximation."""

TRIMMED_NOTICE = "… earlier messages omitted; the full history is in the file named below …"
"""pH's own addition to upstream's trim, argued in the module docstring."""

# --------------------------------------------------------------------------
# Verbatim from `langchain.agents.middleware.summarization.DEFAULT_SUMMARY_PROMPT`
# (checked against 1.3.18, the release P4-01 pinned). Copied rather than
# imported for the same reason `tool-todo` copies its prompts: pH does not
# depend on langchain, and prompt text is what an upgrade reworders silently.
#
# The `<messages>` marker on its own line is documented upstream as part of the
# constant's public contract — deepagents splices its media-reference block in
# immediately before it — and `_with_notes` below uses that same seam for the
# state notes (G10). Keep the marker byte-identical.
# --------------------------------------------------------------------------

SUMMARY_PROMPT = """<role>
Context Extraction Assistant
</role>

<primary_objective>
Your sole objective in this task is to extract the highest quality/most relevant context from the conversation history below.
</primary_objective>

<objective_information>
You're nearing the total number of input tokens you can accept, so you must extract the highest quality/most relevant pieces of information from your conversation history.
This context will then overwrite the conversation history presented below. Because of this, ensure the context you extract is only the most important information to continue working toward your overall goal.
</objective_information>

<instructions>
The conversation history below will be replaced with the context you extract in this step.
You want to ensure that you don't repeat any actions you've already completed, so the context you extract from the conversation history should be focused on the most important information to your overall goal.

You should structure your summary using the following sections. Each section acts as a checklist - you must populate it with relevant information or explicitly state "None" if there is nothing to report for that section:

## SESSION INTENT

What is the user's primary goal or request? What overall task are you trying to accomplish? This should be concise but complete enough to understand the purpose of the entire session.

## SUMMARY

Extract and record all of the most important context from the conversation history. Include important choices, conclusions, or strategies determined during this conversation. Include the reasoning behind key decisions. Document any rejected options and why they were not pursued.

## ARTIFACTS

What artifacts, files, or resources were created, modified, or accessed during this conversation? For file modifications, list specific file paths and briefly describe the changes made to each. This section prevents silent loss of artifact information.

## NEXT STEPS

What specific tasks remain to be completed to achieve the session intent? What should you do next?

</instructions>

The user will message you with the full message history from which you'll extract context to create a replacement. Carefully read through it all and think deeply about what information is most important to your overall goal and should be saved:

With all of this in mind, please carefully read over the entire conversation history, and extract the most important and relevant context to replace it so that you can free up space in the conversation history.
Respond ONLY with the extracted context. Do not include any additional information, or text before or after the extracted context.

<messages>
Messages to summarize:
{messages}
</messages>"""  # noqa: E501

MESSAGES_MARKER = "\n<messages>\n"
"""Upstream's documented splice point. `str.replace` against it is how
deepagents extends the prompt, and how the state notes get in."""

SUMMARY_INSTRUCTION = SUMMARY_PROMPT.split(MESSAGES_MARKER)[0].rstrip()
"""The same prompt with its `<messages>` block removed, for the replay shape.

Derived by splitting rather than retyped: the text before the marker stays
byte-identical to the pinned release and moves with it if the constant is ever
updated, which is the whole reason the marker is part of upstream's contract.

The block has to go because under a replay the conversation *is* the message
list — inlining it again would send it twice. One sentence reads slightly
differently as a result ("The user will message you with the full message
history"), and it happens to be closer to true here than in the inlined shape:
the history really does precede this instruction as messages."""

FOCUS_PROMPT = """<what_the_user_asked_you_to_focus_on>
The person running this session asked for the summary while telling you what
they are about to work on. Weight the extracted context towards it: keep the
detail that work will need, and be terser about the rest. This is a request
about the summary, not a message in the conversation you are summarizing.

{instructions}
</what_the_user_asked_you_to_focus_on>"""
"""pH's own block. Upstream has no equivalent — deepagents' `compact_conversation`
takes no arguments and dsh's `/compact` refuses them — but a person who compacts
*because* they are switching subject knows something the summarizer cannot infer
from the conversation, and that is exactly the moment they type the command."""

NOTES_PROMPT = """<state_that_survives_this_summary>
The following is NOT conversation and is NOT being replaced. It is live state
that this session still holds and will still hold after the summary lands. Do
not describe it as lost, and preserve anything about it that the next steps
depend on.

{notes}
</state_that_survives_this_summary>"""

# --------------------------------------------------------------------------
# Verbatim from `deepagents`' `_build_new_messages_with_path`. The
# with-a-path wording is the one that matters: it is what turns compaction into
# a relocation the model can undo by reading a file, which is the same promise
# `ctx.spill_store` makes for an offloaded tool result.
# --------------------------------------------------------------------------

REPLACEMENT_WITH_PATH = """\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""  # noqa: E501

REPLACEMENT_WITHOUT_PATH = "Here is a summary of the conversation to date:\n\n{summary}"
"""Upstream's fallback, and it is load-bearing rather than cosmetic: a spill
that failed must not leave the model reading a path that holds nothing."""


class TruncateArgsConfig(WireModel):
    """When and how far back tool-call arguments are elided (§7.4 item 2).

    Its own block rather than five loose keys on `Config`, because this is a
    *separate pass* with its own trigger and its own retention: upstream reuses
    the summarization fractions when the window is known and switches to a
    message count when it is not, and both halves are policy a deployment may
    want to move without touching summarization.
    """

    enabled: bool = True
    tools: tuple[str, ...] = EXTRA_TRUNCATE_TOOLS
    """Extra names, beyond what tools declare — see `EXTRA_TRUNCATE_TOOLS`."""
    max_length: int = MAX_ARG_LENGTH
    trigger_messages: int = TRUNCATE_TRIGGER_MESSAGES
    keep_messages: int = TRUNCATE_KEEP_MESSAGES


class Config(WireModel):
    """Row config."""

    auto: bool = True
    """Whether the two automatic triggers are armed. `/compact` works either
    way — dsh's `auto` knob, and the reason it is separate is that a deployment
    that wants a human to decide when history is replaced still wants the
    human to be *able* to."""
    trigger_fraction: float = TRIGGER_FRACTION
    keep_fraction: float = KEEP_FRACTION
    trigger_tokens: int = TRIGGER_TOKENS
    keep_messages: int = KEEP_MESSAGES
    max_tokens: int = SUMMARY_MAX_TOKENS
    summary_input_tokens: int = SUMMARY_INPUT_TOKENS
    truncate_args: TruncateArgsConfig = TruncateArgsConfig()
    overflow_clip_tokens: int = OVERFLOW_CLIP_TOKENS


# ------------------------------------------------------------- tool pairing --


def _open_call_delta(message: Message | None) -> int:
    """How one surface node changes the count of calls still awaiting a result.

    Counted over `derive_event_message`'s output — THE projection — rather than
    over the payload, so this cannot disagree with what the model was sent about
    how many calls a message made.
    """
    if message is None:
        return 0
    opened = sum(1 for block in message.content if isinstance(block, ToolCallBlock))
    closed = sum(1 for block in message.content if isinstance(block, ToolResultBlock))
    return opened - closed


def cuts_over(projected: Sequence[Message | None]) -> tuple[bool, ...]:
    """Whether each cut in an already-projected surface is tool-pairing balanced.

    A surface of *n* nodes has *n + 1* cuts; entry `i` is the cut before node
    `i`, so `balanced[i]` answers "may the first `i` nodes be replaced on their
    own". Cut `0` is trivially balanced and cut `n` is balanced exactly when the
    conversation has no call outstanding.

    Takes the projection rather than the session because a caller that has one
    already should not pay for a second: `derive_event_message` is a pydantic
    validation per node, and `_plan` was deriving the whole surface, then
    deriving it again inside the balance fold.
    """
    cuts = [True]
    open_calls = 0
    for message in projected:
        open_calls += _open_call_delta(message)
        cuts.append(open_calls == 0)
    return tuple(cuts)


def balanced_cuts(session: Session) -> tuple[bool, ...]:
    """`cuts_over`, folded across a session's current surface.

    Folded over the surface in *current* order, which is dsh's reason for
    deriving this from content rather than from step boundaries: a landed
    replacement moves positions, so a rule written in terms of steps would be
    right only until the first compaction.
    """
    events = session.events
    return cuts_over([derive_event_message(events[seq]) for seq in session.surface.nodes])


def safe_cutoff(projected: Sequence[Message | None], target: int) -> int:
    """The greatest balanced cut at or before `target`; `0` when there is none.

    *Backward*, so a pair that straddles the retention boundary is kept whole on
    the retained side — the model keeps a call it can still see the result of,
    and the summary is one exchange shorter. Advancing forward instead would
    summarize the call and hand the model an orphaned result, which several
    providers reject outright.

    `0` means "no safe range", not "cut nothing": the caller reports it rather
    than compacting an empty prefix.
    """
    cuts = cuts_over(projected)
    for index in range(min(target, len(cuts) - 1), -1, -1):
        if cuts[index]:
            return index
    return 0


def _trailing_results(session: Session) -> tuple[int, ...]:
    """The run of `tool/result` nodes the surface currently ends with.

    Upstream's `_find_tail_tool_message_batch`, over surface positions. Empty
    when the conversation does not end in tool results — which is the ordinary
    case, and the cheap check that keeps this pass free when it does not apply.
    """
    events = session.events
    nodes = session.surface.nodes
    end = len(nodes)
    while end > 0 and events[nodes[end - 1]].type == "tool/result":
        end -= 1
    return nodes[end:]


# ------------------------------------------------------- argument truncation --


def truncated_arguments(arguments: str, max_length: int) -> str | None:
    """One call's arguments with over-long string values elided; `None` if unchanged.

    Upstream truncates *per argument*, over an already-parsed `args` dict. pH's
    `ToolCallBlock.arguments` is the raw JSON string the model produced, so this
    has to parse and re-serialize — and that is the one place the port knowingly
    departs from the block's "never re-serialized" contract. It is admissible
    only because the result is a **replacement**: the original event, with the
    model's exact bytes, stays in the log and is one seq away. A call whose
    arguments are not a JSON object is left alone, because a malformed argument
    string is not something to rewrite behind the model's back.
    """
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    elided = {
        key: (value[:ARG_HEAD_CHARS] + TRUNCATION_TEXT)
        if isinstance(value, str) and len(value) > max_length
        else value
        for key, value in parsed.items()
    }
    return None if elided == parsed else dumps(elided)


def _elided_arguments(block: Any, elides: Callable[[str], bool], max_length: int) -> str | None:
    """One block's replacement arguments, read from the *frozen* payload.

    Whole-string length first, which is a sound pre-filter and a cheap one: if
    the entire arguments JSON fits in `max_length`, no single value inside it can
    exceed it. That matters because this runs over every retained assistant
    message on every step once pressure is up, and in the steady state — after
    the first pass has elided everything eligible — the honest answer is always
    "nothing", which should cost a length check rather than a parse.
    """
    if not isinstance(block, Mapping) or block.get("type") != "tool-call":
        return None
    arguments = block.get("arguments")
    if not isinstance(arguments, str) or len(arguments) <= max_length:
        return None
    return truncated_arguments(arguments, max_length) if elides(str(block.get("name"))) else None


def truncated_assistant_payload(
    event: SessionEvent, *, elides: Callable[[str], bool], max_length: int
) -> tuple[dict[str, Any], int] | None:
    """One `assistant/message` payload with long call arguments elided.

    Returns the replacement payload and the characters it saves, or `None` when
    there is nothing to elide — and answers `None` **without deep-copying**,
    which is the common case: `thaw_json` is a full recursive copy of a payload
    that may carry a whole tool-call batch, and paying it per message per step to
    discover there is nothing to do was the pass's real cost.

    **`usage` is dropped, and that is load-bearing.** The replacement is
    appended at the end of the log, and `TokenMeter.last_usage` scans *backward*
    for the newest `assistant/message` carrying one — so a replacement that
    copied an old turn's usage would become the meter's baseline and tell the
    compaction trigger the session had shrunk. The TUI's own footer reads the
    last usage it sees and would have shown the same stale number. The usage
    belongs to the request that produced the original, which still has it.
    """
    message = event.data.get("message") if isinstance(event.data, Mapping) else None
    blocks = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(blocks, (list, tuple)):
        return None
    elisions = {
        index: elided
        for index, block in enumerate(blocks)
        if (elided := _elided_arguments(block, elides, max_length)) is not None
    }
    if not elisions:
        return None
    plain = thaw_json(event.data)
    rewritten = plain["message"]["content"]
    saved = 0
    for index, elided in elisions.items():
        saved += len(rewritten[index]["arguments"]) - len(elided)
        rewritten[index] = {**rewritten[index], "arguments": elided}
    plain.pop("usage", None)
    return plain, saved


# ------------------------------------------------------------------ reading --


def _block_text(block: Any) -> str:
    if isinstance(block, TextBlock):
        return block.text
    if isinstance(block, ToolCallBlock):
        return f"[tool-call {block.name} {block.arguments}]"
    if isinstance(block, ToolResultBlock):
        body = text_of(block.content, placeholder=lambda kind: f"[{kind}]")
        return f"[tool-result {block.tool_call_id}]\n{body}"
    if isinstance(block, ReasoningBlock):
        # The model's own scratch, not conversation. Several providers refuse to
        # accept reasoning back at all, and summarizing it would let a discarded
        # line of thought outlive the turn that discarded it.
        return ""
    return f"[{getattr(block, 'type', 'unknown')}]"


def render_for_summary(messages: tuple[Message, ...], *, trimmed: bool) -> str:
    """The range being replaced, as the summarizer reads it.

    Role-tagged blocks, which is the shape upstream's `get_buffer_string(
    format="xml")` produces: tool calls keep their name and arguments and tool
    results keep their text, because "what did I already do" is the question the
    ARTIFACTS and NEXT STEPS sections are answering.
    """
    rendered: list[str] = []
    for message in messages:
        body = "\n".join(part for part in (_block_text(b) for b in message.content) if part)
        rendered.append(f"<{message.role}>\n{body}\n</{message.role}>")
    joined = "\n\n".join(rendered)
    return f"{TRIMMED_NOTICE}\n\n{joined}" if trimmed else joined


def _with_extras(prompt: str, extras: str) -> str:
    """Splice pH's own blocks in at upstream's documented seam.

    Immediately before the `<messages>` marker, which is where deepagents
    splices its media-reference block — so the state notes and the person's
    focus read as part of the instructions rather than as one more piece of the
    history being compressed. The replay shape has no marker to splice at and
    appends them to the instruction instead; both end up in the same place
    relative to the conversation, which is what matters.
    """
    if not extras:
        return prompt
    return prompt.replace(MESSAGES_MARKER, f"\n{extras}\n{MESSAGES_MARKER}", 1)


def _instruction_message(text: str) -> Message:
    """The one message a summarize call adds, however it is shaped."""
    return create_message(
        role="user",
        content=[{"type": "text", "text": text}],
        source={"kind": "plugin", "plugin": "compaction-summarize", "form": "instructions"},
    )


# ------------------------------------------------------------------- the row --


@dataclass(frozen=True, slots=True)
class _Plan:
    """One compaction's selected range, decided before any model call."""

    shadowed_seqs: tuple[int, ...]
    messages: tuple[Message, ...]
    shadowed_tokens: int
    kept: int


@dataclass(slots=True)
class SummarizeEngine:
    """The `ctx.compaction` provider."""

    ctx: Context
    config: Config
    _running: set[str] = field(default_factory=set)
    """Sessions with a compaction in flight. In-process, deliberately: it is a
    statement about this runner's concurrency, not about the session, and a
    resumed session with nothing running is correct to start one. dsh's durable
    `compaction/start` marker exists to exclude *other processes*, which pH's
    single-writer session store already does."""

    # ------------------------------------------------------------- the seam --

    async def compact_if_needed(
        self, agent: Any, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        """Automatic policy. Never raises — and that is enforced, not intended.

        The caller is a loop hook, so anything escaping here ends the person's
        turn: `agent/pre-step` propagates to `_turn`, and the driver contains it
        with one debug line. A stabilization row must not be able to take a turn
        down because it has a bug, which is why the guard is `Exception` and not
        just `CompactionError` — and why every path through it leaves a record.

        Cancellation is re-raised. `Cancelled` and `AgentCancelled` both derive
        from `Exception`, so a bare guard would swallow a person pressing stop
        and let the turn carry on.
        """
        session: Session | None = getattr(agent, "session", None)
        if session is None or not self.config.auto or session.id in self._running:
            return None
        try:
            return await self._automatic(agent, session, trigger)
        except (Cancelled, AgentCancelled):
            raise
        except Exception as error:
            self._record_failure(session, trigger, error)
            return None

    async def _automatic(
        self, agent: Any, session: Session, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        """The policy itself, unguarded — the guard is its caller's."""
        meter = self.ctx.token_meter
        baseline: TokenBaseline = meter.baseline(session)
        if trigger == "pressure" and not self._under_pressure(baseline):
            return None
        # Before the expensive remedy, the free one: eliding a 40 KB file body
        # the model already wrote costs no model call, and often takes enough
        # off that the summary is not needed this step (§7.4 item 2).
        if self.truncate_arguments(agent, session, trigger, baseline):
            # Only re-measure when something actually moved. Nothing else can
            # change the baseline between these two lines, and asking again
            # unconditionally cost a reverse scan of the log per step.
            baseline = meter.baseline(session)
        if trigger == "pressure" and not self._under_pressure(baseline):
            return None
        if self._declined_since_the_surface_moved(session):
            # A decline stands until something is added. Without this the
            # overflow path would re-attempt inside the same `while True` step
            # loop, and each attempt costs a model call to reach the same answer.
            return None
        return await self._compact(agent, session, trigger)

    async def compact_now(self, agent: Any, *, instructions: str = "") -> CompactionResult | None:
        """`/compact [instructions]`: compact useful history whatever the pressure.

        Raises rather than declining quietly, because a person asked and is owed
        the reason. The idle check is the manual path's alone — the automatic
        triggers run *inside* a step, where the agent is running by definition.

        `instructions` is the person saying what the next stretch of work is
        about, so the summary keeps what that will need. It reaches the
        summarizer labelled as theirs and is recorded on the event, because a
        summary that emphasises one thing over another should say who asked.
        """
        session: Session | None = getattr(agent, "session", None)
        if session is None:
            raise CompactionError("busy", "this agent has no session to compact")
        if session.id in self._running:
            raise CompactionError("busy", "a compaction is already running for this session")
        if getattr(agent, "status", "idle") != "idle":
            raise CompactionError("busy", "the agent is working; compaction needs an idle session")
        try:
            return await self._compact(agent, session, "manual", instructions=instructions)
        except (Cancelled, AgentCancelled):
            raise
        except Exception as error:
            # Recorded *and* re-raised, unlike the automatic path: the person is
            # owed the reason as a sentence and the log is owed it as a record.
            # The three `busy` refusals above are deliberately not recorded —
            # nothing was attempted, so there is no attempt to account for.
            self._record_failure(session, "manual", error)
            raise

    # ---------------------------------------------------------- the triggers --

    def _under_pressure(self, baseline: TokenBaseline) -> bool:
        """Deep Agents' two-branch threshold, on pH's own baseline.

        The baseline is provider-reported usage plus an estimate of what has been
        appended since (D15), so this is as close to "what the next request will
        cost" as anything can be before the request exists.
        """
        pressure = baseline.pressure
        if pressure is not None:
            return pressure >= self.config.trigger_fraction
        return baseline.tokens >= self.config.trigger_tokens

    def _declined_since_the_surface_moved(self, session: Session) -> bool:
        declined = session.latest("compaction/declined")
        if declined is None:
            return False
        nodes = session.surface.nodes
        return not nodes or declined.seq > nodes[-1]

    # ----------------------------------------------------------- the compact --

    async def _compact(
        self,
        agent: Any,
        session: Session,
        trigger: CompactionTrigger,
        *,
        instructions: str = "",
    ) -> CompactionResult | None:
        plan = self._plan(session)
        if plan is None:
            return None
        self._running.add(session.id)
        try:
            summary, usage, shape = await self._summarize(
                agent, session, plan, instructions, trigger
            )
            if not summary:
                # Landing an empty summary would shadow the conversation with
                # nothing, which is the one outcome worse than not compacting.
                raise CompactionError("summary", "the summarize call returned no text")
            return await self._land(
                agent, session, trigger, plan, summary, usage, instructions, shape
            )
        finally:
            self._running.discard(session.id)

    def _plan(self, session: Session) -> _Plan | None:
        """Choose the range to replace. No model call, no appends — so a plan
        that finds nothing costs nothing, which is what `/compact` on a short
        session must cost.

        Indexed by *surface position*, not by position in `derive_messages()`.
        The two are usually the same list, but not always: an `assistant/message`
        logged only to host a max-tokens step's usage is a surface node that
        projects to no message. Counting messages would then cite a range one
        short of what it shadows, and `source_event_seqs` must name every
        shadowed node or the append is refused.
        """
        events = session.events
        nodes = session.surface.nodes
        if len(nodes) < 2:
            return None
        projected = [derive_event_message(events[seq]) for seq in nodes]
        cutoff = safe_cutoff(projected, self._retention_cutoff(session, projected))
        if cutoff == 0:
            return None
        shadowed = tuple(one for one in projected[:cutoff] if one is not None)
        if not shadowed:
            # A range of nodes that all project to nothing — the empty-content
            # assistant message again. Summarizing it would spend a model call
            # to shadow silence with a paragraph about silence.
            return None
        return _Plan(
            shadowed_seqs=nodes[:cutoff],
            messages=shadowed,
            shadowed_tokens=sum(self.ctx.token_meter.measure(one) for one in shadowed),
            kept=len(nodes) - cutoff,
        )

    def _retention_cutoff(self, session: Session, projected: list[Message | None]) -> int:
        """The surface position the retention budget asks for, before pairing
        moves it.

        Clamped to `len - 1` so at least one node always survives: a replacement
        that shadowed the entire surface would leave the model holding a summary
        and no live turn, which upstream clamps against for the same reason.
        """
        window = self._window(session)
        ceiling = len(projected) - 1
        if window is None:
            return max(0, min(ceiling, len(projected) - self.config.keep_messages))
        budget = max(1, int(window * self.config.keep_fraction))
        kept = 0
        for index in range(len(projected) - 1, -1, -1):
            message = projected[index]
            kept += 0 if message is None else self.ctx.token_meter.measure(message)
            if kept > budget:
                return min(ceiling, index + 1)
        return 0

    # ------------------------------------------------- argument truncation --

    def _elides_arguments(self, agent: Any) -> Callable[[str], bool]:
        """Whether a named tool's call arguments may be elided, asked of the tool.

        `ToolDefinition.arguments_disposable`, with the config list as an
        override — the same shape `tool-result-offload` uses for `self_limits`,
        and for the same reason: these are registered plugins, so a deployment
        renames them and an MCP server adds its own. The lookup is scope-aware,
        which is what makes the answer right for an agent-shadowed registration.
        """
        extra = self.config.truncate_args.tools
        scope = getattr(agent, "ctx", None) or self.ctx

        def elides(name: str) -> bool:
            if name in extra:
                return True
            definition = self.ctx.tools.get(name, scope=scope)
            return bool(definition is not None and definition.arguments_disposable)

        return elides

    def truncate_arguments(
        self, agent: Any, session: Session, trigger: CompactionTrigger, baseline: TokenBaseline
    ) -> tuple[int, ...]:
        """Elide long call arguments in retained history (§7.4 item 2).

        Model-free, and that is the point: a summary costs a model call and
        replaces conversation, while this costs nothing and removes only bytes
        the model itself sent and the log still holds. Upstream runs it as a
        separate pass with its own trigger and retention for the same reason.

        Each rewrite is an `assistant/message` surface `replace` citing the one
        node it stands for, so the log keeps the model's exact bytes and only
        the derivation shortens. Returns the seqs rewritten.
        """
        settings = self.config.truncate_args
        if not settings.enabled:
            return ()
        nodes = session.surface.nodes
        if not self._should_truncate(baseline, nodes):
            return ()
        # Captured before the first append: every rewrite replaces a node, which
        # moves the surface — iterating a live view would skip or revisit.
        events = session.events
        cutoff = self._truncate_cutoff(session, nodes)
        elides = self._elides_arguments(agent)
        rewritten: list[int] = []
        saved = 0
        for seq in nodes[:cutoff]:
            event = events[seq]
            if event.type != "assistant/message":
                continue
            replacement = truncated_assistant_payload(
                event, elides=elides, max_length=settings.max_length
            )
            if replacement is None:
                continue
            payload, savings = replacement
            session.append(
                "assistant/message",
                payload,
                SurfaceIntent(surface_op=SurfaceReplace(replaces=(seq,)), source_event_seqs=(seq,)),
            )
            rewritten.append(seq)
            saved += savings
        if not rewritten:
            return ()
        # The harness's own statement of what it elided. Without it the only
        # record is a diff between two events, and no record at all of *why*.
        session.append(
            "compaction/args-truncated",
            {"trigger": trigger, "seqs": rewritten, "savedChars": saved},
        )
        return tuple(rewritten)

    def _should_truncate(self, baseline: TokenBaseline, nodes: tuple[int, ...]) -> bool:
        """Upstream's two-branch truncation trigger.

        The same fraction as summarization when the window is known, and a plain
        message count when it is not — 20 rather than summarization's 170 000
        tokens, because eliding an argument is cheap enough to do early.

        Takes the baseline rather than asking for one: `TokenMeter.baseline`
        reverse-scans the log for the newest reported usage, and before any has
        been reported it estimates the whole conversation. Two callers asked for
        it on the same unchanged log within a few lines of each other.
        """
        if baseline.pressure is not None:
            return baseline.pressure >= self.config.trigger_fraction
        return len(nodes) >= self.config.truncate_args.trigger_messages

    def _truncate_cutoff(self, session: Session, nodes: tuple[int, ...]) -> int:
        """Surface positions below this may be elided; the rest stay verbatim."""
        window = self._window(session)
        if window is None:
            return max(0, len(nodes) - self.config.truncate_args.keep_messages)
        events = session.events
        projected = [derive_event_message(events[seq]) for seq in nodes]
        return self._retention_cutoff(session, projected)

    # -------------------------------------------------------- the overflow clip --

    async def clip_overflow_tail(self, agent: Any) -> tuple[int, ...]:
        """Shrink the trailing tool-result batch before summarizing (§7.4 item 7).

        The case that motivates it is the one summarization is least able to help: a step
        ends with a batch of tool results larger than the retention budget, so every
        balanced cut leaves it in place and `safe_cutoff` correctly declines. Clipping is
        what makes the *next* attempt possible — and often makes the request fit on its
        own, which is why the caller retries when this alone changed something.

        **One path where upstream has two.** deepagents head-slices a `read_file` result
        and points back at its `file_path`, and offloads everything else; pH offloads
        everything. The spill store is content-addressed, so re-spilling the same text
        costs one file rather than one per occurrence, and the model gets one shape of
        pointer to learn instead of two.

        A `tool/result` surface `replace` is the mechanism, and the surface validator
        constrains it to changing content alone — the invariant that makes this safe to
        do to history the model has already read.
        """
        session: Session | None = getattr(agent, "session", None)
        if session is None:
            return ()
        events = session.events
        tail = _trailing_results(session)
        if not tail:
            return ()
        budget = self._clip_budget(session)
        measured = sum(
            self.ctx.token_meter.measure(message)
            for message in (derive_event_message(events[seq]) for seq in tail)
            if message is not None
        )
        if measured < budget:
            return ()
        clipped: list[int] = []
        for seq in tail:
            if await self._clip_one(session, events[seq]):
                clipped.append(seq)
        return tuple(clipped)

    def _clip_budget(self, session: Session) -> int:
        """The keep budget, or upstream's 5 000-token floor when none is known."""
        window = self._window(session)
        if window is None:
            return self.config.overflow_clip_tokens
        return max(1, int(window * self.config.keep_fraction))

    async def _clip_one(self, session: Session, event: SessionEvent) -> bool:
        """Spill one tool result and replace its content with a pointer."""
        message = derive_event_message(event)
        if message is None:
            return False
        block = next((one for one in message.content if isinstance(one, ToolResultBlock)), None)
        if block is None:
            return False
        text = text_of(block.content, placeholder=lambda kind: f"[{kind}]")
        call_id = block.tool_call_id
        # The offload row's own operation, not a second copy of it: where the file
        # goes, the `offload/spilled` accounting and the sentence the model reads to
        # find it are one relocation however it was triggered. `source` becomes
        # `SpillRef.retrieval_hint` — what the model is actually told — so two
        # spellings disagree about that.
        replacement = await spill_tool_result(
            self.ctx, session, call_id=call_id, source=f"{call_id} result", text=text
        )
        if replacement is None:
            # Fail open, as everywhere else in this bundle: a clip that cannot
            # store the content must not be the reason the model loses it.
            return False
        if len(replacement) >= len(text):
            # Replacing a small result with a nine-hundred-character pointer
            # makes the request bigger. Upstream clips every message in an
            # over-budget batch; the batch is what must shrink, and a member
            # that would grow is not part of shrinking it.
            return False
        payload = thaw_json(event.data)
        blocks = payload.get("message", {}).get("content")
        if not isinstance(blocks, list) or not blocks:
            return False
        # Only the result block's content changes — everything else, the message
        # id included, must match: `Session.append` refuses a `tool/result`
        # replacement that touches anything but content.
        blocks[0] = {**blocks[0], "content": [{"type": "text", "text": replacement}]}
        session.append(
            "tool/result",
            payload,
            SurfaceIntent(
                surface_op=SurfaceReplace(replaces=(event.seq,)),
                source_event_seqs=(event.seq,),
            ),
        )
        return True

    def _window(self, session: Session) -> int | None:
        context = session.request_context()
        return None if context is None else context.context_window

    # --------------------------------------------------------- the model call --

    async def _summarize(
        self,
        agent: Any,
        session: Session,
        plan: _Plan,
        instructions: str,
        trigger: CompactionTrigger,
    ) -> tuple[str, TokenUsage | None, str]:
        """One `purpose="compaction"` call; the summary, its cost, and its shape.

        Session-bound so usage is attributed, but outside `is_loop_request`: the
        prompt is *about* the conversation rather than part of it, which is why
        the loop's "model-visible means logged" invariant does not hold it to
        `derive_messages()`.

        **Which shape, and why it is not one.** A `replay` reuses the session's
        own `system` and `tools` so the request is a strict *prefix* of the
        conversation's — the whole point, since the shadowed range is a prefix of
        the surface and a cached prefix is what makes this call nearly free
        (A12). `direct` sends the extraction instruction as its own system prompt
        with no tools. Overflow always takes `direct`, and has to: replaying the
        request a provider just refused for being too large, plus an
        instruction, refuses again.
        """
        header = session.request_header()
        extras = self._extras(session, agent, instructions)
        if trigger == "overflow" or header is None:
            # The trimmed tail, inlined as text rather than sent as messages: a
            # *suffix* of a balanced range can begin with an orphaned tool
            # result, and rendering sidesteps a structure the range no longer
            # has. The full range does not need this — `safe_cutoff` guarantees
            # it is balanced — which is why only this path trims.
            shown, trimmed = self._tail(plan.messages)
            rendered = render_for_summary(shown, trimmed=trimmed)
            summary, usage = await self._call(self._direct(session, agent, rendered, extras))
            return summary, usage, "direct"

        summary, usage = await self._call(self._replay(session, header, plan, extras))
        if summary:
            return summary, usage, "replay"
        # The replay carries the session's tools, so a model primed to *act* can
        # answer with a tool call and no prose. One retry without them, and with
        # the whole range rather than the tail — nothing is hidden by falling
        # back, only the cache is given up. Cheap first, correct second.
        log.debug("ph_stabilize.compaction: the replayed summarize call returned no text")
        rendered = render_for_summary(plan.messages, trimmed=False)
        summary, usage = await self._call(self._direct(session, agent, rendered, extras))
        return summary, usage, "direct-after-replay"

    def _extras(self, session: Session, agent: Any, instructions: str) -> str:
        """The blocks pH adds to upstream's instruction: state notes and focus."""
        # The agent's own boundary, or the deployment — spelled, because
        # `or self.ctx` is the widening and P6-32 buys nothing if the call
        # site keeps computing it under another name. Deriving the *scope*
        # from an agent is the shape that row blesses; defaulting the
        # *widest* one is the shape it deletes.
        own = getattr(agent, "ctx", None)
        notes = self.ctx.compaction.notes(session, scope=own if own is not None else DEPLOYMENT)
        blocks = [NOTES_PROMPT.format(notes="\n\n".join(notes))] if notes else []
        if instructions.strip():
            blocks.append(FOCUS_PROMPT.format(instructions=instructions.strip()))
        return "\n\n".join(blocks)

    def _replay(
        self, session: Session, header: EpochHeader, plan: _Plan, extras: str
    ) -> GenerateOptions:
        """The last routed request's envelope, its shadowed messages, one instruction.

        The instruction is the **last user message**, not the system prompt, and
        that placement is the whole mechanism: anything moved into `system`
        changes the prefix and forfeits the cache this shape exists to hit.

        Two deliberate departures from byte-identical. `max_tokens` is the
        summary cap rather than the conversation's, because this generates a
        summary and not a turn. And `reasoning_effort` is dropped — a reasoning
        model would otherwise spend a thinking budget on an extraction, which
        the harness planner declines for the same reason. Neither is part of the
        token prefix, so neither costs the cache hit.
        """
        instruction = SUMMARY_INSTRUCTION if not extras else f"{SUMMARY_INSTRUCTION}\n\n{extras}"
        return GenerateOptions(
            provider=header.config.provider,
            model=header.config.model,
            messages=(*plan.messages, _instruction_message(instruction)),
            system=header.system,
            tools=tuple(header.tools or ()),
            temperature=header.config.temperature,
            stop=tuple(header.config.stop or ()),
            max_tokens=self.config.max_tokens,
            session_id=session.id,
            purpose="compaction",
        )

    def _direct(self, session: Session, agent: Any, rendered: str, extras: str) -> GenerateOptions:
        """The self-contained shape: upstream's prompt, no tools, no cache hit.

        Routed from the agent's own options rather than the header, because this
        is the path a session with no logged request takes and there may be no
        header to read.
        """
        options = getattr(agent, "options", None)
        provider = str(getattr(options, "provider", "") or "")
        model = str(getattr(options, "model", "") or "")
        if not provider or not model:
            raise CompactionError("summary", "the agent has no model route to summarize with")
        system = _with_extras(SUMMARY_PROMPT, extras).format(messages=rendered)
        return GenerateOptions(
            provider=provider,
            model=model,
            messages=(_instruction_message("Extract the context now."),),
            system=system,
            max_tokens=self.config.max_tokens,
            session_id=session.id,
            purpose="compaction",
        )

    async def _call(self, request: GenerateOptions) -> tuple[str, TokenUsage | None]:
        """Run one request; its text and what it cost. Empty text is not an error here.

        `BlockAssembler` is the loop's own assembly, so this cannot disagree with
        the transcript about what a reply said; `text_of` then drops reasoning
        blocks rather than pasting them into the summary.
        """
        assembler = BlockAssembler()
        async for chunk in await self.ctx.llm.stream(request):
            assembler.push(chunk)
        if assembler.finish.kind == "error":
            failure = assembler.finish.failure
            raise CompactionError(
                "summary", failure.message if failure is not None else "the summarize call failed"
            )
        return text_of(assembler.blocks()).strip(), assembler.usage

    def _tail(self, messages: tuple[Message, ...]) -> tuple[tuple[Message, ...], bool]:
        """The last `summary_input_tokens` worth, and whether anything was cut."""
        budget = self.config.summary_input_tokens
        if budget <= 0:
            return messages, False
        used = 0
        for index in range(len(messages) - 1, -1, -1):
            used += self.ctx.token_meter.measure(messages[index])
            if used > budget:
                return messages[index + 1 :] or messages[-1:], True
        return messages, False

    # ------------------------------------------------------------ the landing --

    async def _land(
        self,
        agent: Any,
        session: Session,
        trigger: CompactionTrigger,
        plan: _Plan,
        summary: str,
        usage: TokenUsage | None,
        instructions: str,
        shape: str,
    ) -> CompactionResult:
        """Write the history, record the accounting, then replace the surface."""
        history = render_for_summary(plan.messages, trimmed=False)
        ref = await self.ctx.spill_store.try_save_text(
            owner=session.id,
            source="conversation history",
            suggested_name=f"{HISTORY_PREFIX}/{session.seq}.md",
            content=history,
        )
        options = getattr(agent, "options", None)
        # No `await` between here and the replacement: the two events are
        # adjacent by construction, which is what lets a consumer price a
        # shadowed range from the record immediately before it.
        session.append(
            "compaction/summarized",
            {
                "trigger": trigger,
                "shadowedSeqs": list(plan.shadowed_seqs),
                "shadowedTokens": plan.shadowed_tokens,
                "kept": plan.kept,
                "provider": str(getattr(options, "provider", "") or ""),
                "model": str(getattr(options, "model", "") or ""),
                "maxTokens": self.config.max_tokens,
                "locator": None if ref is None else ref.locator,
                "usage": None if usage is None else usage.to_wire(),
                "instructions": instructions or None,
                # Which request shape paid for this summary. The cache is the
                # whole reason `replay` exists, and `usage.cacheReadTokens`
                # beside it is what makes "did we actually get the hit" a
                # question the log can answer rather than an assumption.
                "shape": shape,
            },
        )
        text = (
            REPLACEMENT_WITHOUT_PATH.format(summary=summary)
            if ref is None
            else REPLACEMENT_WITH_PATH.format(file_path=ref.locator, summary=summary)
        )
        replacement = session.append(
            "user/message",
            create_user_message(
                content=[{"type": "text", "text": text}],
                # The harness speaking, and saying which kind of speech it is.
                # `form="compaction"` is the discriminator a reader needs: an
                # offloaded paste is also a plugin-authored replacement, and
                # calling either one by the other's name tells the person
                # something false about their own conversation.
                source=PluginSource(
                    plugin="compaction-summarize",
                    form="compaction",
                    summary=(
                        f"{count_of(len(plan.shadowed_seqs), 'message')} summarized"
                        f" (~{plan.shadowed_tokens} tokens)"
                    ),
                ),
            ).to_wire(),
            SurfaceIntent(
                # The set it already has, passed through — where the range
                # forced it to collapse the list to its two ends and let the fold
                # re-derive what sat between them.
                surface_op=SurfaceReplace(replaces=plan.shadowed_seqs),
                source_event_seqs=plan.shadowed_seqs,
            ),
        )
        return CompactionResult(
            trigger=trigger,
            summary=summary,
            shadowed_seqs=plan.shadowed_seqs,
            shadowed_tokens=plan.shadowed_tokens,
            replacement_seq=replacement.seq,
            locator=None if ref is None else ref.locator,
        )

    def _record_failure(
        self, session: Session, trigger: CompactionTrigger, error: BaseException
    ) -> None:
        """The one place a failed attempt becomes a record.

        One site, two entry points, because the alternative had them disagree:
        the automatic path recorded a decline and the manual path recorded
        nothing at all — while telling the person "the attempt is recorded in
        the session log", which was the closest thing to a lie this row had.

        An unexpected exception is a *bug*, not a policy outcome, so it is
        logged with its traceback as well as recorded — the event says a
        compaction did not happen, and the log says why in a form a developer
        can act on.
        """
        if isinstance(error, CompactionError):
            self._decline(session, trigger, error.code, str(error))
            return
        log.exception("ph_stabilize.compaction: the compaction failed unexpectedly")
        self._decline(session, trigger, "error", f"{type(error).__name__}: {error}")

    def _decline(
        self, session: Session, trigger: CompactionTrigger, code: str, reason: str
    ) -> None:
        session.append("compaction/declined", {"trigger": trigger, "code": code, "reason": reason})


@plugin(
    "compaction-summarize",
    inject=["compaction", "token_meter", "llm", "spill_store", "tools"],
    config=Config,
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the engine and arm the two automatic triggers."""
    ctx.spill_store.claim(SpillClaim.under_session("compaction-summarize", "compaction/summarized"))

    engine = SummarizeEngine(ctx=ctx, config=config)
    ctx.compaction.register(engine)

    async def on_pre_step(request: Any, next_: Any) -> Any:
        decision = await next_(request)
        # After the rest of the chain, so a step another row rejected is not
        # compacted for: the cheapest compaction is the one a limit made
        # unnecessary.
        if isinstance(decision, PreStepDecision) and decision.kind == "enter":
            await engine.compact_if_needed(request.agent, "pressure")
        return decision

    async def on_request_error(failure: RequestFailure, next_: Any) -> Any:
        if failure.failure.code != CONTEXT_WINDOW_EXCEEDED:
            return await next_(failure)
        # Clip first, summarize second — upstream's order, and the order that
        # matters: an over-budget trailing tool-result batch is precisely the
        # shape that leaves no balanced cut, so summarization asked first would
        # decline and the clip would never be reached.
        clipped = await engine.clip_overflow_tail(failure.agent)
        result = await engine.compact_if_needed(failure.agent, "overflow")
        if result is None and not clipped:
            # Nothing changed, so retrying would send the same request to the
            # same refusal. `llm-retry` has already declined this code for the
            # same reason; the turn ends with the provider's own error.
            return await next_(failure)
        # Retry whenever *either* pass moved the surface — dsh's rule, and the
        # reason the clip is worth having on its own: it often makes the request
        # fit without spending a summary. The loop re-derives on its next
        # attempt, so returning `retry` is the whole application of both.
        return RequestErrorAction(kind="retry")

    ctx.on("agent/pre-step", on_pre_step)
    ctx.on("agent/request-error", on_request_error)
