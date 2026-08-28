"""`tool-result-offload` — a large result relocated, not lost (P4-02, G2/C5).

Deep Agents' `FilesystemMiddleware._intercept_large_tool_result`, ported onto
`tools/post-execute` and `ctx.spill_store`. One tool result that runs to 200 000
characters can end a session's usefulness on its own, and the model rarely needs
all of it — so the result is written to the spill store and the model is handed
a head-and-tail preview plus the path to read the rest.

**It is a relocation, and the wording is load-bearing.** The spill seam's own
docstring makes the point: the harness must never tell the model something is
gone when it is on disk. The replacement names the file and says how to page
through it, which is why `SpillRef.retrieval_hint` exists.

**Individually, per result.** A Code Mode cell that makes forty dispatches gets
forty separate answers here, because every dispatch crosses this same waterfall
(C5) — one oversized `tools.read` is spilled while its siblings stay inline. The
alternative, melting a cell's whole output together, is what C5 forbids.

**Fail open.** If the spill write fails the original result is kept, exactly as
upstream: an offload that cannot store the content must not be the reason the
model loses it.

**Self-limiting tools are left alone, and they say so themselves.** A tool that
bounds its own output and offers the model a way to page — `read` with an offset
and a limit, `grep` with a match cap — sets `ToolDefinition.self_limits`, and a
model that hit that cap already knows how to ask for the next page. Upstream
matches a hardcoded list of *its* tool names; pH asks the tool, because tools
here are registered plugins: a deployment renames them and an MCP server adds
its own, and a name list in this package cannot know. `excluded_tools` remains
as the escape hatch for a third-party tool whose author has not declared.

A Code Mode cell's output is deliberately *not* self-limiting: its 65 536-char
caps are per stream, so one cell can still emit twice the threshold.

@module ph_stabilize.offload
"""

from __future__ import annotations

from typing import Any

from ph.cordis import Context, plugin
from ph.llm.types import text_of
from ph.session import Session
from ph.tools.definition import Accept, ToolExecution, ToolExecutionResult, text_content
from ph.wire import WireModel

__all__ = [
    "HISTORY_PREFIX",
    "NUM_CHARS_PER_TOKEN",
    "TOOL_TOKEN_LIMIT_BEFORE_EVICT",
    "TOO_LARGE_TOOL_MSG",
    "Config",
    "apply",
    "content_preview",
    "over_token_limit",
    "spill_tool_result",
]

NUM_CHARS_PER_TOKEN = 4
"""Upstream's estimator. Deliberately not `ctx.token_meter`: the threshold is a
*guard rail*, and a cheap arithmetic bound that never varies by provider is
worth more here than an accurate count that costs a tokenizer pass on every
result of every call."""

TOOL_TOKEN_LIMIT_BEFORE_EVICT = 20_000
"""`tool_token_limit_before_evict`, so the char threshold is 80 000."""

PREVIEW_HEAD_LINES = 5
PREVIEW_TAIL_LINES = 5
PREVIEW_LINE_CLIP = 1_000

TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""  # noqa: E501
"""Verbatim from `deepagents/middleware/_message_eviction.py`."""


class Config(WireModel):
    """Row config.

    Two thresholds because the port merges two prior arts: Deep Agents counts
    estimated *tokens*, dsh's `spill-policy` counts *bytes*. Either one tripping
    offloads, and the byte knob is off by default — a deployment that has met a
    provider limit in bytes can say so without restating it as tokens.
    """

    token_limit: int | None = TOOL_TOKEN_LIMIT_BEFORE_EVICT
    """`None` disables offloading, as upstream's `None` does."""
    max_inline_bytes: int | None = None
    excluded_tools: tuple[str, ...] = ()
    """Names to skip regardless of what they declare — the escape hatch for a
    third-party tool that self-limits without saying so. Empty by default,
    because `ToolDefinition.self_limits` is where the answer belongs."""


def _number_lines(lines: list[str], start_line: int) -> str:
    """Right-justified gutter, two spaces, source — upstream's format.

    Upstream also chunks lines past 5 000 characters with `5.1`-style
    continuation markers. Not ported: every caller here has already clipped to
    1 000, so the branch is unreachable, and unreachable machinery is a claim
    nobody can check.
    """
    width = max((len(str(start_line + index)) for index in range(len(lines))), default=0)
    return "\n".join(f"{start_line + index:>{width}}  {line}" for index, line in enumerate(lines))


def content_preview(text: str) -> str:
    """Head and tail with a truncation marker between — upstream's `_create_content_preview`."""
    head_lines, tail_lines = PREVIEW_HEAD_LINES, PREVIEW_TAIL_LINES
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return _number_lines([line[:PREVIEW_LINE_CLIP] for line in lines], 1)
    head = _number_lines([line[:PREVIEW_LINE_CLIP] for line in lines[:head_lines]], 1)
    notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    tail = _number_lines(
        [line[:PREVIEW_LINE_CLIP] for line in lines[-tail_lines:]], len(lines) - tail_lines + 1
    )
    return head + notice + tail


def over_token_limit(text: str, limit: int | None) -> bool:
    """The estimated-token threshold, shared with `input-offload`.

    `>` and not `>=`, which is upstream's own comparison and both rows' gate: at
    exactly the limit a result stays inline, and one character over it does not.
    A guard rail that fired *at* its limit would offload the thing the limit was
    chosen to still admit. Written once because a `>=` slip in a second copy
    would be caught only by that row's own boundary pair.
    """
    return limit is not None and len(text) > NUM_CHARS_PER_TOKEN * limit


def oversized(text: str, config: Config) -> bool:
    """Whether one result's text trips either threshold.

    The cheap character count first: the byte count is only asked when a
    deployment set `max_inline_bytes`, and encoding a 2 MB result to answer a
    question the char count already answered would be 70 µs per call.
    """
    if over_token_limit(text, config.token_limit):
        return True
    return config.max_inline_bytes is not None and (
        len(text.encode("utf-8")) > config.max_inline_bytes
    )


HISTORY_PREFIX = "conversation_history"
"""Where a *conversation* is relocated to, as opposed to a tool result.

Here rather than in either of the two rows that write to it — `input-offload`
for a pasted message, `compaction-summarize` for a shadowed range — because a
directory two rows share is a convention, and a convention stated twice is one a
rename silently splits.
"""


async def spill_tool_result(
    ctx: Context, session: Session, *, call_id: str, source: str, text: str
) -> str | None:
    """Relocate one tool result and return the text that stands in for it.

    The whole recipe in one place: where the file goes, the `offload/spilled`
    accounting, and the wording the model reads to find it. Two rows perform this
    — this one when a result is oversized on arrival (G2), and the overflow clip
    when a retained batch has to shrink (§7.4 item 7) — and the second had
    already drifted on `source`, which becomes `SpillRef.retrieval_hint` and is
    therefore the *sentence the model is given*. One relocation described two
    ways depending on which row did it is exactly the drift this prevents.

    `None` is the fail-open path both callers need: an offload that cannot store
    the content must not be the reason the model loses it. The seam logs why.
    """
    ref = await ctx.spill_store.try_save_text(
        owner=session.id,
        source=source,
        suggested_name=f"large_tool_results/{call_id}",
        content=text,
    )
    if ref is None:
        return None
    # Where the original went. Declared ignorable in the vocabulary (the property
    # is the type's, not this call site's) — a reader that skips it loses the
    # forwarding address, not the conversation, because the replacement the model
    # saw is what `tool/result` carries.
    session.append(
        "offload/spilled",
        {"callId": call_id, "locator": ref.locator, "bytes": ref.bytes},
    )
    return TOO_LARGE_TOOL_MSG.format(
        tool_call_id=call_id, file_path=ref.locator, content_sample=content_preview(text)
    )


@plugin("tool-result-offload", inject=["tools", "spill_store"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Replace an oversized result with a preview and a path to the rest."""

    async def offload(execution: ToolExecution, result: ToolExecutionResult, next_: Any) -> Any:
        decision = await next_(execution, result)
        session = execution.session
        if session is None or _self_limiting(execution):
            return decision
        if not isinstance(decision, Accept):
            return decision
        if decision.has_value:
            # A value replacement is re-rendered by the registry *after* this
            # waterfall, so the content that will reach the model does not exist
            # yet and there is nothing here to measure.
            return decision
        # The projection as it will actually be sent, which is the earlier
        # listener's if one rewrote it. Measuring the body's own content instead
        # would let any future post-execute row — a redactor, a truncator —
        # switch this guard rail off by touching the result at all.
        content = decision.content if decision.content is not None else result.content
        text = text_of(content)
        if not oversized(text, config):
            return decision
        replacement = await spill_tool_result(
            ctx,
            session,
            call_id=execution.call_id,
            source=f"{execution.name} result",
            text=text,
        )
        if replacement is None:
            # Fail open, as upstream: an offload that cannot store the content
            # must not be the reason the model loses it.
            return decision
        return Accept(
            content=text_content(replacement),
            additional_contexts=decision.additional_contexts,
        )

    def _self_limiting(execution: ToolExecution) -> bool:
        """Whether this tool bounds its own output and can page.

        Asked of the tool, with the config list as an override — the row injects
        `tools` for exactly this, and a scope-aware lookup is what makes the
        answer right for an agent-shadowed registration.
        """
        if execution.name in config.excluded_tools:
            return True
        definition = ctx.tools.get(execution.name, scope=execution.scope)
        return bool(definition is not None and definition.self_limits)

    ctx.on("tools/post-execute", offload)
