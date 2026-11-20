"""`compaction-summarize`: the conversation replaced by a summary (P4-03, G4).

This row is the `SummarizationMiddleware` of Deep Agents and the
`compaction-basic` of dsh, built on the seams of pH as one `CompactionEngine`.
The numbers come from upstream, from the pinned releases. When the window is
known, the row triggers at `0.85` of the window and keeps `0.10` of it. When the
window is not known, the row triggers at `170 000` tokens and keeps `6 messages`
(`compute_summarization_defaults` in `deepagents`).

Invariants that this row holds:

* **It is a surface `replace` (A3).** The safety of the row depends on this. The
  summary is appended as a `user/message`. Its `surfaceOp` shadows the range
  that the summary stands for, and cites every shadowed seq. `derive_messages()`
  yields the summary. The log keeps every event. `transcript()` still shows the
  user the conversation that they had. The same applies to the two cheaper
  remedies below, so **the log never changes**.
* **The cut never splits a call from its result.** A cutoff from the retention
  budget is moved *back* to the nearest balanced boundary. Balance is folded
  over the surface in current order (the `tool-pairing` of dsh), not over step
  markers, because a previous compaction has already changed what "position"
  means. If the only balanced cut is the start of the conversation, nothing is
  compacted and the attempt **reports that**. To send an orphaned `tool-result`
  to a provider that rejects it is not a repair.
* **Two cheaper remedies run before the expensive one.** On every pressure
  check, over-long call arguments in history older than the retained tail are
  elided (§7.4 item 2). On overflow, the trailing tool-result batch is spilled
  and replaced with a pointer (§7.4 item 7) before any summarization is
  attempted. That batch is exactly the shape that leaves no balanced cut.
* **A request past the whole window takes the overflow remedy before it is
  sent.** The results of one step can outgrow the window by themselves (a 64 KB
  cell on a 16 384-token slot), and a summary leaves the latest turn whole. Such
  a request was therefore sent anyway, was refused, and was clipped only then.
  The estimate shows the problem at `agent/request`, when the batch of the step
  is in the log. The clip therefore runs there (trigger `window`), and the
  pressure check of the step leaves the request to that check.
* **The request replays the envelope of the conversation**: the same `system`,
  the same `tools`, the shadowed messages as themselves, and the extraction
  prompt appended as the last *user* message. The request is therefore a strict
  prefix of the request that the loop just made. That makes the call nearly
  free on a provider that caches prefixes (A12). It is also why the prompt
  cannot go in `system`: anything put there changes the prefix. In addition, the
  summarizer sees the whole range as real `tool-call`/`tool-result` blocks and
  not as a rendered tail, so nothing is silently withheld from it.
* **Carrying the tools is the cost of carrying the prefix.** The two cannot be
  separated. A model that is primed to act may answer with a tool call and no
  prose. A reply with no text is therefore retried one time in the
  self-contained shape: without tools, and with the whole range. Cheap first,
  correct second.
* **Overflow never replays**, because the envelope in question is the request
  that a provider just refused for being too large. A conversation that the
  estimate still puts past the window after the clip does not replay either.
  Such a call sends the 4 000-token tail of upstream
  (`_DEFAULT_TRIM_TOKEN_LIMIT`, `strategy="last"`), rendered as text. Text is
  also the only shape that can carry the tail, because a *suffix* of a balanced
  range may begin with an orphaned tool result. When that tail is trimmed, the
  rendered text **says so**. A model that is told to write "SESSION INTENT" from
  the middle of a conversation, and is not told that it reads the middle, states
  the middle as the intent.

There are three triggers, and they ask different questions:

* `agent/pre-step` asks "will the next request be too big". It answers from an
  *estimate*, because no usage number exists for a request that nobody has made.
* `agent/request` asks "would a provider refuse this request at all". It answers
  from the same estimate, after the batch of the step is in the log.
* `agent/request-error` asks nothing. The provider has already answered
  `CONTEXT_WINDOW_EXCEEDED`, which `llm-retry` declines to retry so that this
  row can see it.

@module ph_stabilize.compaction
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, assert_never, cast

from ph.agent.types import (
    AgentHandle,
    PreStepDecision,
    PreStepRequest,
    RequestErrorAction,
    RequestFailure,
    RequestProposal,
)
from ph.cancel import Canceled
from ph.characters import head
from ph.cordis import Context, Next, plugin
from ph.json import as_obj, as_seq, as_str, dumps
from ph.keys import COMPACTION, LLM, SPILL_STORE, TOKEN_METER, TOOLS
from ph.llm import BlockAssembler
from ph.llm.types import (
    CONTEXT_WINDOW_EXCEEDED,
    ContentBlock,
    GenerateOptions,
    LlmCallConfig,
    MediaBlock,
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
    RevisionRecord,
    RevisionSize,
    Session,
    SessionEvent,
    derive_event_message,
    editable_message,
    is_stand_in,
    rewrite_many,
    safe_cutoff,
    substitute,
)
from ph.session.writers import log_writer
from ph.text import block_marker, count_of
from ph.wire import WireModel

from .offload import HISTORY_PREFIX, plan_tool_result_spill, spill_tool_result

_LOG = log_writer(__name__)

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
    "render_for_summary",
    "truncated_arguments",
    "truncated_assistant_message",
]

log = logging.getLogger("ph_stabilize.compaction")

ROW = "compaction-summarize"
"""The plugin id of this row. cordis counts the retries of the row under this
name (`RequestFailure.retries_by`)."""

# --------------------------------------------------------------------------
# From `compute_summarization_defaults` in
# `deepagents/middleware/summarization.py`: the fractions when the model
# profile knows its window, and the fixed pair when it does not. Both branches
# are here because pH has the same split: `request/context` carries a
# `context_window` only when the adapter resolved one.
# --------------------------------------------------------------------------

TRIGGER_FRACTION = 0.85
"""Compact when the estimate reaches this fraction of a known window."""

KEEP_FRACTION = 0.10
"""Leave this fraction of a known window untouched at the tail."""

TRIGGER_TOKENS = 170_000
"""The trigger when the window is unknown. It is deliberately conservative, as
upstream says. Without a window there is nothing to take a fraction of, and to
overshoot an unknown limit costs the turn."""

KEEP_MESSAGES = 6
"""The retention when the window is unknown. It counts messages and not tokens,
for the same reason."""

SUMMARY_MAX_TOKENS = 8_192
"""The generation cap for the summarize call. It is the default of
`compaction-basic` in dsh."""

SUMMARY_INPUT_TOKENS = 4_000
"""How much of the replaced range the summarizer is shown when the range is
trimmed to its tail. The value is `_DEFAULT_TRIM_TOKEN_LIMIT` of langchain."""

# --- argument truncation (§7.4 item 2; upstream `truncate_args_settings`) ----

EXTRA_TRUNCATE_TOOLS: tuple[str, ...] = ()
"""Names of tools whose arguments are elided whatever the tools declare. This is
the override.

The tuple is empty by default, because `ToolDefinition.arguments_disposable` is
where the answer belongs. Upstream matches a hardcoded
`{"write_file", "edit_file"}`, and this bundle rejected exactly that pattern in
the offload row. The fs tools of pH are named `read`/`write`/`edit`, so the list
of upstream matches *nothing* here. A deployment that renames the tools, or an
MCP server that adds its own, would silently get no truncation, and nothing
would fail. The tool therefore declares the answer, in the same way that
`self_limits` is declared for the offload row.

The restriction itself is kept, and it has a real effect. A long `run_code` cell
in old history is exactly the argument that a model reads again. A file body
that the model already wrote, and can `read` back, is not."""

MAX_ARG_LENGTH = 2_000
"""Above this, one string argument is elided. Upstream's `max_length`."""

ARG_HEAD_CHARS = 20
"""How much of an elided argument is kept. Upstream uses `value[:20]`."""

TRUNCATION_TEXT = "...(argument truncated)"
"""Upstream's `truncation_text`, verbatim."""

TRUNCATE_TRIGGER_MESSAGES = 20
"""The truncation trigger when the window is unknown: `("messages", 20)` from
upstream. It differs from the 170 000 tokens of summarization deliberately.
Eliding an argument costs no model call, so it can run far earlier."""

TRUNCATE_KEEP_MESSAGES = 20
"""The truncation retention when the window is unknown: `("messages", 20)` from
upstream, and not the 6 of summarization. The tail that is *kept whole* here is
much longer, because truncation is a cheap trim and not a replacement of the
history."""

# --- the overflow clip (§7.4 item 7; upstream `_clip_overflow_tail`) ---------

OVERFLOW_CLIP_TOKENS = 5_000
"""The clip threshold when no window is known. It is the fallback of upstream in
`_derive_overflow_clip_threshold_tokens`. Upstream describes it as equivalent to
a 20 000-character floor under a `chars / 4` approximation."""

TRIMMED_NOTICE = "… earlier messages omitted; the full history is in the file named below …"
"""An addition of pH to the trim of upstream. The module docstring gives the
reason."""

# --------------------------------------------------------------------------
# Verbatim from `langchain.agents.middleware.summarization.DEFAULT_SUMMARY_PROMPT`
# (checked against 1.3.18, the release that P4-01 pinned). The text is copied
# and not imported, for the same reason that `tool-todo` copies its prompts: pH
# does not depend on langchain, and an upgrade can reword prompt text silently.
#
# Upstream documents the `<messages>` marker on its own line as part of the
# public contract of the constant. deepagents splices its media-reference block
# in immediately before the marker, and `_with_extras` below uses the same
# splice point for the state notes (G10). Keep the marker byte-identical.
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
"""The documented splice point of upstream. deepagents extends the prompt with
`str.replace` against it, and the state notes enter the prompt the same way."""

SUMMARY_INSTRUCTION = SUMMARY_PROMPT.split(MESSAGES_MARKER)[0].rstrip()
"""The same prompt with its `<messages>` block removed, for the replay shape.

The text is derived by a split and is not retyped. The text before the marker
therefore stays byte-identical to the pinned release, and it moves with the
constant if the constant is ever updated. That is the reason why the marker is
part of the contract of upstream.

The block must go because under a replay the conversation *is* the message
list, and to inline it again would send it twice. As a result, one sentence
reads slightly differently ("The user will message you with the full message
history"). That sentence is closer to true here than in the inlined shape: the
history really does precede this instruction as messages."""

FOCUS_PROMPT = """<what_the_user_asked_you_to_focus_on>
The person running this session asked for the summary while telling you what
they are about to work on. Weight the extracted context towards it: keep the
detail that work will need, and be terser about the rest. This is a request
about the summary, not a message in the conversation you are summarizing.

{instructions}
</what_the_user_asked_you_to_focus_on>"""
"""A block that pH adds. Upstream has no equivalent: the `compact_conversation`
of deepagents takes no arguments, and the `/compact` of dsh refuses them. But a
user who compacts *because* they are switching subject knows something that the
summarizer cannot infer from the conversation, and that is exactly the moment
when they type the command."""

NOTES_PROMPT = """<state_that_survives_this_summary>
The following is NOT conversation and is NOT being replaced. It is live state
that this session still holds and will still hold after the summary lands. Do
not describe it as lost, and preserve anything about it that the next steps
depend on.

{notes}
</state_that_survives_this_summary>"""

# --------------------------------------------------------------------------
# Verbatim from `_build_new_messages_with_path` in `deepagents`. The wording
# with a path is the one that matters. It turns compaction into a relocation
# that the model can undo by reading a file. `ctx.spill_store` makes the same
# promise for an offloaded tool result.
# --------------------------------------------------------------------------

REPLACEMENT_WITH_PATH = """\
You are in the middle of a conversation that has been summarized.

The full conversation history has been saved to {file_path} should you need to refer back to it for details.

A condensed summary follows:

<summary>
{summary}
</summary>"""  # noqa: E501

REPLACEMENT_WITHOUT_PATH = "Here is a summary of the conversation to date:\n\n{summary}"
"""The fallback of upstream. It is necessary and not cosmetic: a spill that
failed must not leave the model reading a path that holds nothing."""


class TruncateArgsConfig(WireModel):
    """When tool-call arguments are elided, and how far back (§7.4 item 2).

    This is its own block and not five loose keys on `Config`, because it
    configures a *separate pass* with its own trigger and its own retention.
    Upstream reuses the summarization fractions when the window is known, and
    switches to a message count when it is not. Both halves are policy that a
    deployment may want to change without touching summarization.
    """

    enabled: bool = True
    tools: tuple[str, ...] = EXTRA_TRUNCATE_TOOLS
    """Extra names, beyond what the tools declare. See `EXTRA_TRUNCATE_TOOLS`."""
    max_length: int = MAX_ARG_LENGTH
    trigger_messages: int = TRUNCATE_TRIGGER_MESSAGES
    keep_messages: int = TRUNCATE_KEEP_MESSAGES


class Config(WireModel):
    """Row config."""

    auto: bool = True
    """Whether the three automatic triggers are armed. `/compact` works in both
    cases. This is the `auto` setting of dsh. It is separate because a
    deployment that wants a human to decide when history is replaced still wants
    the human to be *able* to do it."""
    trigger_fraction: float = TRIGGER_FRACTION
    keep_fraction: float = KEEP_FRACTION
    trigger_tokens: int = TRIGGER_TOKENS
    keep_messages: int = KEEP_MESSAGES
    max_tokens: int = SUMMARY_MAX_TOKENS
    summary_input_tokens: int = SUMMARY_INPUT_TOKENS
    truncate_args: TruncateArgsConfig = TruncateArgsConfig()
    overflow_clip_tokens: int = OVERFLOW_CLIP_TOKENS


# ------------------------------------------------------------- tool pairing --


def _past_the_window(baseline: TokenBaseline) -> bool:
    """Whether the next request, by the estimate, fills the whole window. A
    provider would only refuse such a request. `False` when the window is
    unknown, because then a refusal is the only information about it."""
    pressure = baseline.pressure
    return pressure is not None and pressure >= 1


def _trailing_results(session: Session) -> tuple[int, ...]:
    """The run of `tool/result` nodes at the end of the surface, before any user
    messages that the batch of a step put after them.

    This is `_find_tail_tool_message_batch` of upstream, over surface positions.
    The result is empty when the conversation does not end in tool results. That
    is the ordinary case, and this cheap check keeps the pass free when it does
    not apply.

    The search goes **past the batch**, because pH splices a batch between the
    results of a step and its next request: prompt context that changed (the
    session facts of rlm, added again after a summary has shadowed them), a
    steer, or the context of a result. A refusal comes after that batch. A
    search for results only at the very end found none, and the turn ended in a
    refusal that the clip could have recovered.
    """
    nodes = session.surface.nodes

    def kind(position: int) -> str:
        event = session.at(nodes[position])
        return "" if event is None else event.type

    end = len(nodes)
    while end > 0 and kind(end - 1) == "user/message":
        end -= 1
    start = end
    while start > 0 and kind(start - 1) == "tool/result":
        start -= 1
    return nodes[start:end]


# ------------------------------------------------------- argument truncation --


def truncated_arguments(arguments: str, max_length: int) -> str | None:
    """The arguments of one call with over-long string values elided, or `None`
    if nothing changed.

    Upstream truncates *per argument*, over an `args` dict that is already
    parsed. `ToolCallBlock.arguments` in pH is the raw JSON string that the
    model produced, so this function must parse it and serialize it again. This
    is the one place where the port knowingly departs from the "never
    re-serialized" contract of the block. The departure is acceptable only
    because the result is a **replacement**: the original event, with the exact
    bytes of the model, stays in the log and is one seq away. A call whose
    arguments are not a JSON object is left alone, because a malformed argument
    string must not be rewritten without telling the model.
    """
    try:
        parsed = json.loads(arguments)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, dict):
        return None
    elided = {
        key: (head(value, ARG_HEAD_CHARS) + TRUNCATION_TEXT)
        if isinstance(value, str) and len(value) > max_length
        else value
        for key, value in parsed.items()
    }
    return None if elided == parsed else dumps(elided)


def _elided_arguments(
    block: object,
    elides: Callable[[str], bool],
    max_length: int,
) -> tuple[str, int] | None:
    """The replacement arguments of one block and the length that they replace,
    read from the *frozen* payload.

    The original length is returned here and is not measured again later,
    because this is the only pass that has proved that `arguments` is a `str`. A
    caller that derived the length again from the thawed copy would need a
    second narrowing. No test can reach the failure branch of that narrowing. If
    the branch ever ran, its arithmetic would report a *negative* saving into
    the row that the TUI prints.

    The length of the whole string is checked first. That check is a sound
    filter and a cheap one: if the entire arguments JSON fits in `max_length`,
    no single value inside it can exceed `max_length`. The check matters because
    this function runs over every assistant message below the cutoff on every
    step once pressure is up. In the steady state, after the first pass has
    elided everything eligible, the correct answer is always "nothing". That
    answer should cost a length check and not a parse.
    """
    if not isinstance(block, Mapping) or block.get("type") != "tool-call":
        return None
    arguments = block.get("arguments")
    if not isinstance(arguments, str) or len(arguments) <= max_length:
        return None
    if not elides(as_str(block.get("name"))):
        return None
    shorter = truncated_arguments(arguments, max_length)
    return None if shorter is None else (shorter, len(arguments))


def truncated_assistant_message(
    event: SessionEvent, *, elides: Callable[[str], bool], max_length: int
) -> tuple[dict[str, Any], int] | None:
    """The message of one `assistant/message` with long call arguments elided.

    Returns the replacement message and the characters that it saves, or `None`
    when there is nothing to elide. It answers `None` **without a deep copy**,
    and `None` is the common case. `thaw_json` is a full recursive copy of a
    message that may carry a whole tool-call batch. To pay for that copy for
    each message on each step, only to find that there is nothing to do, was the
    real cost of the pass.

    Only the message is returned. The revision functions of `ph.session.revise`
    build the payload around it, and drops the `usage`, `turn` and `step` that belong to the
    original (D8: that module gives the reason one time, for every producer of a
    rewrite).
    """
    blocks = as_seq(as_obj(event.data.get("message")).get("content"))
    elisions = {
        index: elided
        for index, block in enumerate(blocks)
        if (elided := _elided_arguments(block, elides, max_length)) is not None
    }
    if not elisions:
        return None
    saved = sum(before - len(elided) for elided, before in elisions.values())
    # The frozen pass above proved the shape: a message whose `content` is a
    # block list. `thaw_json` preserves that shape exactly: object to `dict`,
    # array to `list`, at every level. The claim is therefore already checked,
    # and to check it again would cost two branches that no test can reach.
    # `cast` is the correct way to write that. The rule in `workspace.py` against
    # casting a `Literal` off JSON is about an *unchecked* claim, and this claim
    # is checked.
    message = editable_message(event)
    rewritten = cast("list[dict[str, Any]]", message["content"])
    for index, (shorter, _before) in elisions.items():
        rewritten[index] = {**rewritten[index], "arguments": shorter}
    return message, saved


# ------------------------------------------------------------------ reading --


def _block_text(block: ContentBlock) -> str:
    """One block as the line the summarizing model reads."""
    match block:
        case TextBlock():
            return block.text
        case ToolCallBlock():
            return f"[tool-call {block.name} {block.arguments}]"
        case ToolResultBlock():
            body = text_of(block.content, placeholder=block_marker)
            return f"[tool-result {block.tool_call_id}]\n{body}"
        case ReasoningBlock():
            # The scratch work of the model, not conversation. Several providers
            # refuse to accept reasoning back at all. To summarize it would let a
            # discarded line of thought outlive the turn that discarded it.
            return ""
        case MediaBlock():
            return block_marker(block.type)
        case _ as unhandled:
            assert_never(unhandled)


def render_for_summary(messages: tuple[Message, ...], *, trimmed: bool) -> str:
    """The range that is replaced, as the summarizer reads it.

    The output is role-tagged blocks, the shape that `get_buffer_string(
    format="xml")` of upstream produces. Tool calls keep their name and
    arguments, and tool results keep their text, because the ARTIFACTS and NEXT
    STEPS sections answer the question "what did I already do".
    """
    rendered: list[str] = []
    for message in messages:
        body = "\n".join(part for part in (_block_text(b) for b in message.content) if part)
        rendered.append(f"<{message.role}>\n{body}\n</{message.role}>")
    joined = "\n\n".join(rendered)
    return f"{TRIMMED_NOTICE}\n\n{joined}" if trimmed else joined


def _with_extras(prompt: str, extras: str) -> str:
    """Splice the blocks that pH adds in at the documented splice point of
    upstream.

    The blocks go immediately before the `<messages>` marker, where deepagents
    splices its media-reference block. The state notes and the focus of the user
    then read as part of the instructions, and not as one more piece of the
    history that is compressed. The replay shape has no marker to splice at, so
    it appends the blocks to the instruction. In both shapes the blocks are part
    of the instruction, which is what matters.
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
    """The sessions that have a compaction in flight. This state is in-process
    deliberately. It is a statement about the concurrency of this runner and not
    about the session, and a resumed session with nothing running is correct to
    start a compaction. The durable `compaction/start` marker of dsh exists to
    exclude *other processes*, which the single-writer session store of pH
    already does."""

    # ------------------------------------------------------------- the seam --

    def trigger_fraction(self) -> float | None:
        """The configured `trigger_fraction` while automatic compaction is on.
        `None` when it is off, because then nothing compacts on pressure."""
        return self.config.trigger_fraction if self.config.auto else None

    async def compact_if_needed(
        self, agent: AgentHandle, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        """The automatic policy. It never raises, and the code enforces that.

        The caller is a loop hook, so anything that escapes here ends the turn
        of the user: `agent/pre-step` propagates to `_turn`, and the driver
        contains it with one debug line. A bug in a stabilization row must not
        be able to end a turn. The guard is therefore `Exception` and not only
        `CompactionError`, and every path through the guard leaves a record.

        Cancellation is raised again. `Canceled` derives from `Exception`, so a
        bare guard would swallow a stop request from the user and let the turn
        continue.
        """
        session: Session | None = agent.session
        if session is None or not self.config.auto or session.id in self._running:
            return None
        try:
            return await self._automatic(agent, session, trigger)
        except Canceled:
            raise
        except Exception as error:
            self._record_failure(session, trigger, error)
            return None

    async def _automatic(
        self, agent: AgentHandle, session: Session, trigger: CompactionTrigger
    ) -> CompactionResult | None:
        """The policy itself, without a guard. The caller holds the guard."""
        meter = self.ctx.require(TOKEN_METER)
        baseline: TokenBaseline = meter.baseline(session)
        estimated = trigger in ("pressure", "window")
        if estimated and not self._under_pressure(baseline):
            return None
        past = _past_the_window(baseline)
        if trigger == "pressure" and past:
            # A request that no provider would take is left to the check of the
            # request itself (`window`, at `agent/request`). That check measures
            # with the batch of the step in the log, and clips there before a
            # summary is paid for.
            return None
        if trigger == "window":
            if not past:
                return None
            # The overflow remedy runs before the refusal and not after it. The
            # clip is first, because the results of the step itself are most
            # often what puts a request past the whole window.
            if await self.clip_overflow_tail(agent):
                baseline = meter.baseline(session)
        # The free remedy comes before the expensive one. Eliding a 40 KB file
        # body that the model already wrote costs no model call, and often
        # removes enough that the summary is not needed this step (§7.4 item 2).
        if self.truncate_arguments(agent, session, trigger, baseline):
            # Measure again, because the elision moved the baseline. The baseline
            # follows the surface since the last reported usage, so the removed
            # text comes off the count now and not at the next response.
            baseline = meter.baseline(session)
        if estimated and not self._under_pressure(baseline):
            return None
        if self._declined_since_the_surface_moved(session):
            # A decline stands until something is added. Without this check the
            # overflow path would attempt again inside the same `while True` step
            # loop, and each attempt costs a model call to reach the same answer.
            return None
        oversized = trigger == "overflow" or _past_the_window(baseline)
        return await self._compact(agent, session, trigger, oversized=oversized)

    async def compact_now(
        self, agent: AgentHandle, *, instructions: str = ""
    ) -> CompactionResult | None:
        """`/compact [instructions]`: compact useful history whatever the pressure.

        This method raises and does not decline quietly, because a user asked
        and must be given the reason. Only the manual path has the idle check.
        The automatic triggers run *inside* a step, where the agent is running
        by definition.

        In `instructions` the user says what the next part of the work is
        about, so that the summary keeps what that work needs. The text reaches
        the summarizer labeled as the text of the user, and it is recorded on
        the event, because a summary that emphasizes one thing over another
        should say who asked.
        """
        session: Session | None = agent.session
        if session is None:
            raise CompactionError("busy", "this agent has no session to compact")
        if session.id in self._running:
            raise CompactionError("busy", "a compaction is already running for this session")
        if agent.status != "idle":
            raise CompactionError("busy", "the agent is working; compaction needs an idle session")
        try:
            return await self._compact(agent, session, "manual", instructions=instructions)
        except Canceled:
            raise
        except Exception as error:
            # Recorded *and* raised again, unlike the automatic path. The user
            # must get the reason as a sentence, and the log must get it as a
            # record. The three `busy` refusals above are deliberately not
            # recorded: nothing was attempted, so there is no attempt to account
            # for.
            self._record_failure(session, "manual", error)
            raise

    # ---------------------------------------------------------- the triggers --

    def _under_pressure(self, baseline: TokenBaseline) -> bool:
        """The two-branch threshold of Deep Agents, on the baseline of pH.

        The baseline is provider-reported usage plus an estimate of what has
        been appended since (D15). This is as close to "what the next request
        will cost" as anything can be before the request exists.
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
        agent: AgentHandle,
        session: Session,
        trigger: CompactionTrigger,
        *,
        instructions: str = "",
        oversized: bool = False,
    ) -> CompactionResult | None:
        """`oversized`: the conversation is past the window, according to the
        provider or to the estimate. The summarizer is then shown the tail of
        the conversation and not a replay."""
        plan = self._plan(session)
        if plan is None:
            return None
        self._running.add(session.id)
        try:
            summary, usage, shape = await self._summarize(
                agent, session, plan, instructions, oversized=oversized
            )
            if not summary:
                # To land an empty summary would shadow the conversation with
                # nothing. That is the one outcome that is worse than no
                # compaction.
                raise CompactionError("summary", "the summarize call returned no text")
            return await self._land(
                agent, session, trigger, plan, summary, usage, instructions, shape
            )
        finally:
            self._running.discard(session.id)

    def _plan(self, session: Session) -> _Plan | None:
        """Choose the range to replace. There is no model call and no append, so
        a plan that finds nothing costs nothing. `/compact` on a short session
        must cost exactly that.

        The range is indexed by *surface position*, not by position in
        `derive_messages()`. The two are usually the same list, but not always:
        an `assistant/message` that is logged only to hold the usage of a
        max-tokens step is a surface node that projects to no message. A count
        of messages would then cite a range one short of what it shadows.
        `source_event_seqs` must name every shadowed node, or the append is
        refused.
        """
        events = session.events
        nodes = session.surface.nodes
        if len(nodes) < 2:
            return None
        projected = [derive_event_message(events[seq]) for seq in nodes]
        cutoff = safe_cutoff(projected, self._retention_cutoff(session, projected))
        if cutoff == 0:
            return None
        above = [
            (seq, message)
            for seq, message in zip(nodes[:cutoff], projected[:cutoff], strict=True)
            if message is not None
        ]
        if not above:
            # A range of nodes that all project to nothing: the empty-content
            # assistant message again. To summarize it would spend a model call
            # to shadow silence with a paragraph about silence.
            return None
        shadowed = tuple(message for _, message in above)
        if all(is_stand_in(events[seq]) for seq, _ in above):
            # Nothing is here but the last summary (D9). The retained tail can
            # be one pair over `keep_fraction` and stay there. Pressure then
            # never drops, so the pre-step trigger fires again at the next step,
            # and the only thing above the cutoff is the paragraph that the
            # previous call wrote. To summarize a summary shortens nothing and
            # costs a model call, one for each step, for as long as the turn
            # runs.
            #
            # To decline is the correct answer here, and a floor on the range is
            # not. The tail really is too big for the budget. The caller that
            # can act on that is `clip_overflow_tail`, which runs before this
            # and reports separately.
            return None
        meter = self.ctx.require(TOKEN_METER)
        return _Plan(
            shadowed_seqs=nodes[:cutoff],
            messages=shadowed,
            shadowed_tokens=sum(meter.measure(one) for one in shadowed),
            kept=len(nodes) - cutoff,
        )

    def _retention_cutoff(self, session: Session, projected: list[Message | None]) -> int:
        """The surface position that the retention budget asks for, before
        pairing moves it.

        The position is clamped to `len - 1` so that at least one node always
        survives. A replacement that shadowed the entire surface would leave the
        model with a summary and no live turn. Upstream clamps for the same
        reason.
        """
        window = self._window(session)
        ceiling = len(projected) - 1
        if window is None:
            return max(0, min(ceiling, len(projected) - self.config.keep_messages))
        budget = max(1, int(window * self.config.keep_fraction))
        kept = 0
        for index in range(len(projected) - 1, -1, -1):
            message = projected[index]
            kept += 0 if message is None else self.ctx.require(TOKEN_METER).measure(message)
            if kept > budget:
                return min(ceiling, index + 1)
        return 0

    # ------------------------------------------------- argument truncation --

    def _elides_arguments(self, agent: AgentHandle) -> Callable[[str], bool]:
        """Whether the call arguments of a named tool may be elided. The tool
        gives the answer.

        The answer is `ToolDefinition.arguments_disposable`, with the config
        list as an override. `tool-result-offload` uses the same shape for
        `self_limits`, and for the same reason: these are registered plugins, so
        a deployment renames them and an MCP server adds its own. The lookup is
        scope-aware, which makes the answer right for a registration that an
        agent shadows.
        """
        extra = self.config.truncate_args.tools
        scope = agent.ctx

        def elides(name: str) -> bool:
            if name in extra:
                return True
            definition = self.ctx.require(TOOLS).get(name, scope=scope)
            return bool(definition is not None and definition.arguments_disposable)

        return elides

    def truncate_arguments(
        self,
        agent: AgentHandle,
        session: Session,
        trigger: CompactionTrigger,
        baseline: TokenBaseline,
    ) -> tuple[int, ...]:
        """Elide long call arguments in history older than the retained tail
        (§7.4 item 2).

        The pass uses no model, and that is its purpose. A summary costs a model
        call and replaces conversation. This pass costs nothing, and removes
        only bytes that the model itself sent and that the log still holds.
        Upstream runs it as a separate pass with its own trigger and retention
        for the same reason.

        Each rewrite is an `assistant/message` surface `replace` that cites the
        one node that it stands for. The log therefore keeps the exact bytes of
        the model, and only the derivation gets shorter. Returns the seqs that
        were rewritten.
        """
        settings = self.config.truncate_args
        if not settings.enabled:
            return ()
        nodes = session.surface.nodes
        if not self._should_truncate(baseline, nodes):
            return ()
        # Captured before the first append. Every rewrite replaces a node, which
        # moves the surface, so a loop over a live view would skip or revisit.
        events = session.events
        cutoff = self._truncate_cutoff(session, nodes)
        elides = self._elides_arguments(agent)
        rewrites: list[tuple[SessionEvent, dict[str, Any]]] = []
        saved = 0
        for at in nodes[:cutoff]:
            event = events[at]
            if event.type != "assistant/message":
                continue
            replacement = truncated_assistant_message(
                event, elides=elides, max_length=settings.max_length
            )
            if replacement is None:
                continue
            message, savings = replacement
            rewrites.append((event, message))
            saved += savings
        rewritten = [event.seq for event, _ in rewrites]
        if rewrites:
            # One batch: the rewrites and the record of them land together or
            # not at all (P10-14). When they landed one at a time, a rewrite that
            # the surface refused partway left the earlier ones in the log with
            # no `compaction/args-truncated` to say why. The record, had it been
            # written, would have described rewrites that never landed. There is
            # one record for all of them, and it names each (`revises`). The
            # record is the statement of the harness itself about what it
            # elided. Without it, the only record is a diff between two events,
            # and there is no record at all of *why*.
            meter = self.ctx.require(TOKEN_METER)
            record = RevisionRecord(
                _LOG,
                "compaction/args-truncated",
                {"trigger": trigger, "seqs": rewritten, "savedChars": saved},
                RevisionSize(
                    "tokens",
                    sum(meter.node_tokens(session, seq) for seq in rewritten),
                    sum(meter.measure(Message.model_validate(new)) for _, new in rewrites),
                ),
            )
            rewrite_many(session, rewrites, record)
        return tuple(rewritten)

    def _should_truncate(self, baseline: TokenBaseline, nodes: tuple[int, ...]) -> bool:
        """The two-branch truncation trigger of upstream.

        The trigger is the same fraction as summarization when the window is
        known, and a plain message count when it is not. The count is 20, and
        not the 170 000 tokens of summarization, because eliding an argument is
        cheap enough to do early.

        The method takes the baseline that its caller already measured, on the
        same unchanged log, and does not ask for one again.
        """
        if baseline.pressure is not None:
            return baseline.pressure >= self.config.trigger_fraction
        return len(nodes) >= self.config.truncate_args.trigger_messages

    def _truncate_cutoff(self, session: Session, nodes: tuple[int, ...]) -> int:
        """Surface positions below this may be elided. The rest stay verbatim."""
        window = self._window(session)
        if window is None:
            return max(0, len(nodes) - self.config.truncate_args.keep_messages)
        events = session.events
        projected = [derive_event_message(events[seq]) for seq in nodes]
        return self._retention_cutoff(session, projected)

    # -------------------------------------------------------- the overflow clip --

    async def clip_overflow_tail(self, agent: AgentHandle) -> tuple[int, ...]:
        """Shrink the trailing tool-result batch before summarizing (§7.4 item 7).

        The motivating case is the one where summarization can help least. A
        step ends with a batch of tool results that is larger than the retention
        budget, so every balanced cut leaves the batch in place, and
        `safe_cutoff` correctly declines. The clip makes the *next* attempt
        possible. Often the clip alone makes the request fit, which is why the
        caller retries when only the clip changed something.

        **One path where upstream has two.** deepagents head-slices a
        `read_file` result and points back at its `file_path`, and offloads
        everything else. pH offloads everything. The spill store is
        content-addressed, so to spill the same text again costs one file and
        not one file for each occurrence. The model also has one shape of
        pointer to learn and not two.

        The mechanism is a `tool/result` surface `replace`. The surface
        validator limits it to a change of content alone. That invariant makes
        this safe to do to history that the model has already read.

        The clip is requested when a provider refuses a request. It is also
        requested before a refusal: at `agent/request`, when the estimate puts
        the request past the whole window (`_past_the_window`).
        """
        session: Session | None = agent.session
        if session is None:
            return ()
        tail = _trailing_results(session)
        if not tail:
            return ()
        # The per-node counts of the meter itself. The baseline that asked for
        # this clip has just filled them, so the largest messages in the session
        # are not tokenized twice.
        meter = self.ctx.require(TOKEN_METER)
        if sum(meter.node_tokens(session, seq) for seq in tail) < self._clip_budget(session):
            return ()
        clipped: list[int] = []
        for at in tail:
            event = session.at(at)
            if event is not None and await self._clip_one(session, event):
                clipped.append(at)
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
        text = text_of(block.content, placeholder=block_marker)
        call_id = block.tool_call_id
        # This is the operation of the offload row itself, not a second copy of
        # it. Where the file goes, the `offload/spilled` accounting, and the
        # sentence that the model reads to find the file are one relocation,
        # whatever triggered it. `source` becomes `SpillRef.retrieval_hint`,
        # which is what the model is actually told, so two spellings would
        # disagree about that.
        #
        # **The spill is planned, and every reason not to clip is checked, before
        # anything is written** (S14). When the spill came first, a clip that
        # then declined left a blob and an `offload/spilled` record ("full text
        # at X" in the transcript) for a result that stayed inline.
        planned = plan_tool_result_spill(
            self.ctx, session, call_id=call_id, source=f"{call_id} result", text=text
        )
        if planned is None or len(planned.replacement) >= len(text):
            # A text that the store cannot take stays as it is. And to replace a
            # small result with a nine-hundred-character pointer makes the
            # request bigger. Upstream clips every message in an over-budget
            # batch. But the batch is what must shrink, and a member that would
            # grow is not part of shrinking it.
            return False
        clipped = editable_message(event)
        blocks = clipped.get("content")
        if not isinstance(blocks, list) or not blocks or not isinstance(blocks[0], dict):
            return False
        # Only the content of the result block changes. The revision function
        # builds the rest of the payload from the original, and core refuses a
        # `tool/result` rewrite that touches anything but content.
        blocks[0] = {**blocks[0], "content": [{"type": "text", "text": planned.replacement}]}

        # The replacement goes in the batch of the spill itself: the accounting
        # and the replacement that it describes land together or not at all. The
        # clip also fails open, as everywhere else in this bundle: a clip that
        # cannot store the content must not be the reason that the model loses
        # it.
        return await spill_tool_result(self.ctx, session, planned, rewriting=(event, clipped))

    def _window(self, session: Session) -> int | None:
        context = session.request_context()
        return None if context is None else context.context_window

    # --------------------------------------------------------- the model call --

    async def _summarize(
        self,
        agent: AgentHandle,
        session: Session,
        plan: _Plan,
        instructions: str,
        *,
        oversized: bool,
    ) -> tuple[str, TokenUsage | None, str]:
        """One `purpose="compaction"` call. Returns the summary, its cost, and
        its shape.

        The call is session-bound so that usage is attributed, but it is outside
        `is_loop_request`. The prompt is *about* the conversation and not part
        of it, so the "model-visible means logged" invariant of the loop does
        not hold it to `derive_messages()`.

        **Which shape, and why there is more than one.** A `replay` reuses the
        `system` and `tools` of the session, so that the request is a strict
        *prefix* of the request of the conversation. That is the purpose: the
        shadowed range is a prefix of the surface, and a cached prefix makes
        this call nearly free (A12). `direct` sends the extraction instruction
        as its own system prompt with no tools. An `oversized` conversation
        always takes `direct`, and it must. A replay of the request that a
        provider just refused for being too large, or of one that the estimate
        says it would refuse, plus an instruction, is refused again.
        """
        header = session.request_header()
        extras = self._extras(session, agent, instructions)
        if oversized or header is None:
            # The trimmed tail is inlined as text and not sent as messages. A
            # *suffix* of a balanced range can begin with an orphaned tool
            # result, and rendering avoids a structure that the range no longer
            # has. The full range does not need this, because `safe_cutoff`
            # guarantees that it is balanced. That is why only this path trims.
            shown, trimmed = self._tail(plan.messages)
            rendered = render_for_summary(shown, trimmed=trimmed)
            summary, usage = await self._call(self._direct(session, agent, rendered, extras))
            return summary, usage, "direct"

        summary, usage = await self._call(self._replay(session, header, plan, extras))
        if summary:
            return summary, usage, "replay"
        # The replay carries the tools of the session, so a model that is primed
        # to *act* can answer with a tool call and no prose. Retry one time
        # without the tools, and with the whole range and not the tail. The
        # fallback hides nothing. It gives up only the cache. Cheap first,
        # correct second.
        log.debug("ph_stabilize.compaction: the replayed summarize call returned no text")
        rendered = render_for_summary(plan.messages, trimmed=False)
        summary, usage = await self._call(self._direct(session, agent, rendered, extras))
        return summary, usage, "direct-after-replay"

    def _extras(self, session: Session, agent: AgentHandle, instructions: str) -> str:
        """The blocks pH adds to upstream's instruction: state notes and focus."""
        # The scope is written out: the boundary of the agent itself, or the
        # deployment. `or self.ctx` is the widening, and P6-32 gains nothing if
        # the call site still computes it under another name. To derive the
        # *scope* from an agent is the shape that P6-32 approves. To default to
        # the *widest* scope is the shape that P6-32 deletes.
        notes = self.ctx.require(COMPACTION).notes(session, scope=agent.ctx)
        blocks = [NOTES_PROMPT.format(notes="\n\n".join(notes))] if notes else []
        if instructions.strip():
            blocks.append(FOCUS_PROMPT.format(instructions=instructions.strip()))
        return "\n\n".join(blocks)

    def _replay(
        self, session: Session, header: EpochHeader, plan: _Plan, extras: str
    ) -> GenerateOptions:
        """The envelope of the last routed request, the shadowed messages, and
        one instruction.

        The instruction is the **last user message**, not the system prompt.
        That placement is the whole mechanism: anything moved into `system`
        changes the prefix and loses the cache that this shape exists to hit.

        There are two deliberate departures from byte-identical. `max_tokens` is
        the summary cap and not the cap of the conversation, because this call
        generates a summary and not a turn. And the fields of the effort are not
        sent. A reasoning model would otherwise spend a thinking budget on an
        extraction, which the harness planner declines for the same reason.
        Neither departure is part of the token prefix, so neither costs the
        cache hit.
        """
        instruction = SUMMARY_INSTRUCTION if not extras else f"{SUMMARY_INSTRUCTION}\n\n{extras}"
        return GenerateOptions(
            provider=header.config.provider,
            model=header.config.model,
            messages=(*plan.messages, _instruction_message(instruction)),
            system=header.system,
            tools=tuple(header.tools or ()),
            sampling=header.config.sampling,
            stop=tuple(header.config.stop or ()),
            max_tokens=self.config.max_tokens,
            session_id=session.id,
            purpose="compaction",
        )

    def _direct(
        self, session: Session, agent: AgentHandle, rendered: str, extras: str
    ) -> GenerateOptions:
        """The self-contained shape: the prompt of upstream, no tools, no cache
        hit.

        The route comes from the options of the agent and not from the header.
        A session with no logged request takes this path, and it may have no
        header to read.
        """
        options = agent.options
        provider = str(options.provider or "")
        model = str(options.model or "")
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
        """Run one request. Returns its text and its cost. Empty text is not an
        error here.

        `BlockAssembler` is the assembly that the loop itself uses, so this
        cannot disagree with the transcript about what a reply said. `text_of`
        then drops reasoning blocks and does not paste them into the summary.
        """
        assembler = BlockAssembler()
        async for chunk in await self.ctx.require(LLM).stream(request):
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
            used += self.ctx.require(TOKEN_METER).measure(messages[index])
            if used > budget:
                return messages[index + 1 :] or messages[-1:], True
        return messages, False

    # ------------------------------------------------------------ the landing --

    async def _land(
        self,
        agent: AgentHandle,
        session: Session,
        trigger: CompactionTrigger,
        plan: _Plan,
        summary: str,
        usage: TokenUsage | None,
        instructions: str,
        shape: str,
    ) -> CompactionResult:
        """Write the history, then replace the surface, then write the accounting."""
        history = render_for_summary(plan.messages, trimmed=False)
        store = self.ctx.require(SPILL_STORE)
        ref = await store.try_save_text(
            owner=session.id,
            source="conversation history",
            suggested_name=f"{HISTORY_PREFIX}/{session.seq}.md",
            content=history,
        )
        options = agent.options
        text = (
            REPLACEMENT_WITHOUT_PATH.format(summary=summary)
            if ref is None
            else REPLACEMENT_WITH_PATH.format(file_path=ref.locator, summary=summary)
        )
        message = create_user_message(
            content=[{"type": "text", "text": text}],
            # The harness speaks here, and says which kind of speech this is.
            # `form="compaction"` is the discriminator that a reader needs. An
            # offloaded paste is also a plugin-authored replacement. To call
            # either one by the name of the other tells the user something
            # false about their own conversation.
            source=PluginSource(
                plugin="compaction-summarize",
                form="compaction",
                summary=(
                    f"{count_of(len(plan.shadowed_seqs), 'message')} summarized"
                    f" (~{plan.shadowed_tokens} tokens)"
                ),
            ),
        )
        # The record of the summary. The revision function writes it right after
        # the replacement that it explains, stamps it with that replacement, and
        # puts both in one batch (P10-14). A replacement that the surface refuses
        # leaves no record of a summary that replaced nothing.
        record = RevisionRecord(
            _LOG,
            "compaction/summarized",
            {
                "trigger": trigger,
                "shadowedSeqs": list(plan.shadowed_seqs),
                "shadowedTokens": plan.shadowed_tokens,
                "kept": plan.kept,
                "provider": str(options.provider or ""),
                "model": str(options.model or ""),
                "maxTokens": self.config.max_tokens,
                "locator": None if ref is None else ref.locator,
                "usage": None if usage is None else usage.to_wire(),
                "instructions": instructions or None,
                # The request shape that paid for this summary. The cache is the
                # only reason why `replay` exists. With `usage.cacheReadTokens`
                # beside it, the log can answer "did we actually get the hit",
                # and nobody has to assume the answer.
                "shape": shape,
            },
            RevisionSize(
                "tokens", plan.shadowed_tokens, self.ctx.require(TOKEN_METER).measure(message)
            ),
        )
        replacement = substitute(session, plan.shadowed_seqs, message, record)
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
        """The one place where a failed attempt becomes a record.

        There is one site for two entry points, because the two used to
        disagree. The automatic path recorded a decline. The manual path
        recorded nothing at all, while it told the user "the attempt is recorded
        in the session log". That was the closest thing to a false statement
        that this row had.

        An unexpected exception is a *bug*, not a policy outcome. It is
        therefore logged with its traceback and also recorded. The event says
        that a compaction did not happen, and the log says why in a form that a
        developer can act on.
        """
        if isinstance(error, CompactionError):
            self._decline(session, trigger, error.code, str(error))
            return
        log.exception("ph_stabilize.compaction: the compaction failed unexpectedly")
        self._decline(session, trigger, "error", f"{type(error).__name__}: {error}")

    def _decline(
        self, session: Session, trigger: CompactionTrigger, code: str, reason: str
    ) -> None:
        _LOG.append(
            session, "compaction/declined", {"trigger": trigger, "code": code, "reason": reason}
        )


@plugin(
    ROW,
    affects="environment",
    inject=[COMPACTION, TOKEN_METER, LLM, SPILL_STORE, TOOLS],
    config=Config,
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the engine and arm the three automatic triggers."""
    ctx.require(SPILL_STORE).claim(
        SpillClaim.under_session(
            # `REPLACEMENT_WITH_PATH` tells the model that the history "has been
            # saved to {file_path} should you need to refer back to it". That is
            # the definition of a handed path, so a `deny read outside-workspace`
            # rule must not refuse the file that this row just pointed at.
            "compaction-summarize",
            "compaction/summarized",
            hands_paths=True,
        )
    )

    engine = SummarizeEngine(ctx=ctx, config=config)
    ctx.require(COMPACTION).register(engine)

    async def on_pre_step(
        request: PreStepRequest,
        next_: Next[PreStepDecision],
    ) -> PreStepDecision:
        decision = await next_(request)
        # After the rest of the chain, so that a step that another row rejected
        # causes no compaction. The cheapest compaction is the one that a limit
        # made unnecessary.
        if decision.kind == "enter":
            await engine.compact_if_needed(request.agent, "pressure")
        return decision

    async def on_request(proposal: RequestProposal, next_: Next[LlmCallConfig]) -> LlmCallConfig:
        config = await next_(proposal)
        # At the request itself: after the batch of the step is in the log, and
        # before the loop derives the messages. input-offload revises at the
        # same point. A request that the estimate puts past the whole window is
        # clipped, or summarized, before it is sent, and is not sent to be
        # refused.
        await engine.compact_if_needed(proposal.agent, "window")
        return config

    async def on_request_error(
        failure: RequestFailure, next_: Next[RequestErrorAction | None]
    ) -> RequestErrorAction | None:
        if failure.failure.code != CONTEXT_WINDOW_EXCEEDED:
            return await next_(failure)
        # Clip first, summarize second. This is the order of upstream, and the
        # order matters. An over-budget trailing tool-result batch is exactly
        # the shape that leaves no balanced cut. If summarization ran first, it
        # would decline, and the clip would never be reached.
        clipped = await engine.clip_overflow_tail(failure.agent)
        result = await engine.compact_if_needed(failure.agent, "overflow")
        if result is None and not clipped:
            # Nothing changed, so a retry would send the same request to the
            # same refusal. `llm-retry` has already declined this code for the
            # same reason. The turn ends with the error of the provider.
            return await next_(failure)
        # Retry whenever *either* pass moved the surface. This is the rule of
        # dsh, and the reason why the clip is worth having by itself: it often
        # makes the request fit without the cost of a summary. The loop derives
        # the messages again on its next attempt, so to return `retry` is all
        # that is needed to apply both.
        return RequestErrorAction(kind="retry")

    ctx.on("agent/pre-step", on_pre_step)
    ctx.on("agent/request", on_request)
    ctx.on("agent/request-error", on_request_error)
