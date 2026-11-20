"""`tool-result-offload`: a large result is relocated, not lost (P4-02, G2/C5).

This row ports `FilesystemMiddleware._intercept_large_tool_result` of Deep
Agents onto `tools/post-execute` and `ctx.spill_store`. One tool result of
200 000 characters can end the usefulness of a session, and the model rarely
needs all of it. The row therefore writes the result to the spill store. The
model gets a head-and-tail preview and the path where it can read the rest.

**It is a relocation, and the wording matters.** The docstring of the spill
seam makes the point: the harness must never tell the model that something is
gone when it is on disk. The replacement names the file and says how to page
through it. That is the reason why `SpillRef.retrieval_hint` exists.

**A spilled result is a file, and a search of it is usually the better way to
use it.** `read` reaches the file by the absolute path that the replacement
names. `grep` and `glob` reach it when they get that path as their root (the
`path` argument of each tool). This works because the fs tools accept an
absolute root and the default rule set restricts only writes. `permissions-fs`
now states this reach, so it no longer holds only by accident.

`text_index` can also index a spilled result. A caller must ask for it by path,
with a `glob` that matches a blob that is named by its digest. A spilled result
is deliberately *not* in the default walk. To embed every spilled result of
every session without a request would cost minutes of model time for output that
nobody has searched. `SPILL_TOOLS_HINT` tells the model about `grep` and `glob`.

**Each result is handled on its own.** A Code Mode cell that makes forty
dispatches gets forty separate answers here, because every dispatch crosses this
same waterfall (C5). One oversized result is spilled, and the other results of
the cell stay inline. The alternative is to merge the whole output of a cell
into one result, and C5 forbids that.

**The row fails open.** If the spill write fails, the row keeps the original
result, exactly as upstream does. An offload that cannot store the content must
not be the reason that the model loses it.

**Self-limiting tools are left alone, and they declare it themselves.** Some
tools limit their own output and give the model a way to page: `read` with an
offset and a limit, `grep` with a match cap. Such a tool sets
`ToolDefinition.self_limits`. A model that reached that cap already knows how to
ask for the next page. Upstream matches a hardcoded list of *its* tool names. pH
asks the tool, because tools here are registered plugins: a deployment renames
them and an MCP server adds its own, and a name list in this package cannot know
them. `excluded_tools` remains as the escape hatch for a third-party tool whose
author has not declared.

The output of a Code Mode cell is deliberately *not* self-limiting. Its
65 536-byte caps are per stream, so one cell can still emit twice the threshold.

@module ph_stabilize.offload
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from ph.characters import head
from ph.cordis import DEPLOYMENT, Boundary, Context, Next, plugin
from ph.json import JsonObject, JsonValue
from ph.keys import SPILL_STORE, TOOLS
from ph.llm.types import text_of
from ph.seams.spill import PlannedBlob, SpillClaim
from ph.session import RevisionRecord, RevisionSize, Session, SessionEvent, rewrite
from ph.session.writers import log_writer
from ph.tools.definition import (
    Accept,
    PostToolDecision,
    ToolExecution,
    ToolExecutionResult,
    text_content,
)
from ph.wire import WireModel

_LOG = log_writer(__name__)

__all__ = [
    "GENERIC_READER",
    "HISTORY_PREFIX",
    "NUM_CHARS_PER_TOKEN",
    "SPILL_TOOLS_HINT",
    "TOOL_TOKEN_LIMIT_BEFORE_EVICT",
    "TOO_LARGE_TOOL_MSG",
    "UPSTREAM_READER",
    "UPSTREAM_TOO_LARGE_TOOL_MSG",
    "Config",
    "apply",
    "content_preview",
    "over_token_limit",
    "spill_tool_result",
    "spill_wording",
]

NUM_CHARS_PER_TOKEN = 4
"""The estimator of upstream. It is deliberately not `ctx.token_meter`. The
threshold is a *guard rail*. A cheap arithmetic bound that never varies by
provider is worth more here than an accurate count that costs a tokenizer pass
on every result of every call."""

TOOL_TOKEN_LIMIT_BEFORE_EVICT = 20_000
"""`tool_token_limit_before_evict`, so the char threshold is 80 000."""

PREVIEW_HEAD_LINES = 5
PREVIEW_TAIL_LINES = 5
PREVIEW_LINE_CLIP = 1_000

UPSTREAM_TOO_LARGE_TOOL_MSG = """Tool result too large, the result of this tool call {tool_call_id} was saved in the filesystem at this path: {file_path}

You can read the result from the filesystem by using the read_file tool, but make sure to only read part of the result at a time.

You can do this by specifying an offset and limit in the read_file tool call. For example, to read the first 100 lines, you can use the read_file tool with offset=0 and limit=100.

Here is a preview showing the head and tail of the result (lines of the form `... [N lines truncated] ...` indicate omitted lines in the middle of the content):

{content_sample}
"""  # noqa: E501
"""Verbatim from `deepagents/middleware/_message_eviction.py`, and never sent.

The text is kept byte-identical, because the value of tracking a port is that an
upstream change produces a visible diff. `TOO_LARGE_TOOL_MSG` below is the text
that is sent."""

UPSTREAM_READER = "read_file"
"""The one token in the block above that is wrong here.

deepagents names its own tool. The readers of pH are *registered plugins*: a
deployment renames them, and an MCP server adds its own. The name is therefore
localized. This row used to correct the name in a second paragraph instead. It
appended a sentence that said "pH's reader is `read`, not the `read_file` the
paragraph above names". That sentence spent model attention to retract text that
pH itself had emitted. `ToolDefinition.reads_paths` is how the row finds the
real name."""

GENERIC_READER = "file-reading"
"""The word that stands in when no visible tool declares `reads_paths`.

A deployment can disable the fs tools. "use the {reader} tool" with an empty
name is worse than no name. The path is still correct, and the path is what
matters."""

TOO_LARGE_TOOL_MSG = UPSTREAM_TOO_LARGE_TOOL_MSG.replace(UPSTREAM_READER, "{reader}")
"""Upstream's wording with the tool name left to the deployment."""

SPILL_TOOLS_HINT = """This path is an ordinary file on disk, so {searchers} reach it with the path above as their `path` argument — searching it is usually better than paging through it."""  # noqa: E501
"""A sentence that pH adds: the capability that upstream does not mention.

The block above describes paging as the only way through the file, so a model
that is given 40 MB reads it from the top. Search is the better way through a
large result, and nothing told the model that search was available. The sentence
is appended only when some visible tool declares `searches_paths`. This row
exists to stop the naming of a tool that this deployment does not have, and it
must not make the same mistake in the other direction.
"""


class Config(WireModel):
    """Row config.

    There are two thresholds because the port merges two earlier designs: Deep
    Agents counts estimated *tokens*, and the `spill-policy` of dsh counts
    *bytes*. A result is offloaded when it exceeds either threshold. The byte
    setting is off by default. A deployment that has met a provider limit in
    bytes can state the limit in bytes and does not have to restate it in
    tokens.
    """

    token_limit: int | None = TOOL_TOKEN_LIMIT_BEFORE_EVICT
    """`None` disables offloading, as upstream's `None` does."""
    max_inline_bytes: int | None = None
    excluded_tools: tuple[str, ...] = ()
    """Names to skip, whatever those tools declare. This is the escape hatch for
    a third-party tool that self-limits but does not declare it. The default is
    empty, because `ToolDefinition.self_limits` is where the answer belongs."""


def _number_lines(lines: list[str], start_line: int) -> str:
    """Number the lines in the format of upstream: a right-justified gutter, two
    spaces, then the source line.

    Upstream also splits a line of more than 5 000 characters into chunks, with
    continuation markers in the style of `5.1`. That branch is not ported. Every
    caller here has already clipped each line to 1 000 characters, so the branch
    could never run, and nobody can check code that never runs.
    """
    width = max((len(str(start_line + index)) for index in range(len(lines))), default=0)
    return "\n".join(f"{start_line + index:>{width}}  {line}" for index, line in enumerate(lines))


def content_preview(text: str) -> str:
    """The head and the tail of `text`, with a truncation marker between them.
    This is the `_create_content_preview` of upstream."""
    head_lines, tail_lines = PREVIEW_HEAD_LINES, PREVIEW_TAIL_LINES
    lines = text.splitlines()
    if len(lines) <= head_lines + tail_lines:
        return _number_lines([head(line, PREVIEW_LINE_CLIP) for line in lines], 1)
    top = _number_lines([head(line, PREVIEW_LINE_CLIP) for line in lines[:head_lines]], 1)
    notice = f"\n... [{len(lines) - head_lines - tail_lines} lines truncated] ...\n"
    bottom = _number_lines(
        [head(line, PREVIEW_LINE_CLIP) for line in lines[-tail_lines:]],
        len(lines) - tail_lines + 1,
    )
    return top + notice + bottom


def over_token_limit(text: str, limit: int | None) -> bool:
    """The threshold in estimated tokens, shared with `input-offload`.

    The comparison is `>` and not `>=`. It is the comparison of upstream and the
    gate of both rows: at exactly the limit a result stays inline, and one
    character over the limit it does not. A guard rail that fired *at* its limit
    would offload the content that the limit was chosen to still admit. The
    comparison is written one time, because a `>=` slip in a second copy would
    be caught only by the boundary pair of that row.
    """
    return limit is not None and len(text) > NUM_CHARS_PER_TOKEN * limit


def oversized(text: str, config: Config) -> bool:
    """Whether the text of one result exceeds either threshold.

    The cheap character count comes first. The byte count is computed only when
    a deployment set `max_inline_bytes`. A large result is therefore not encoded
    to answer a question that the character count already answered.
    """
    if over_token_limit(text, config.token_limit):
        return True
    return config.max_inline_bytes is not None and (
        len(text.encode("utf-8")) > config.max_inline_bytes
    )


HISTORY_PREFIX = "conversation_history"
"""The directory that a *conversation* is relocated to, as opposed to a tool
result.

Two rows write to it: `input-offload` for a pasted message, and
`compaction-summarize` for a shadowed range. The name is defined here and not in
either of those rows. A directory that two rows share is a convention, and when
a convention is stated twice, a rename splits it and nothing reports the split.
"""


@dataclass(frozen=True, slots=True)
class ResultSpill:
    """The relocation of one tool result, decided and not yet made.

    It holds where the result will go and the text that stands in for it
    (`plan_tool_result_spill`). `spill_tool_result` makes the relocation."""

    call_id: str
    source: str
    """Becomes `SpillRef.retrieval_hint`."""
    blob: PlannedBlob
    """The bytes of the result and their locator. They are hashed one time, for
    both the wording and the write."""
    replacement: str
    """The text names the locator that the store will derive. That is why the
    text can be built before the write."""


def plan_tool_result_spill(
    ctx: Context,
    session: Session,
    *,
    call_id: str,
    source: str,
    text: str,
    scope: Boundary | None = None,
) -> ResultSpill | None:
    """Where one tool result would go, and what the model would read in its
    place.

    The answer is `None` for a text that the store cannot take at all
    (`SpillStore.plan_text`). Such a text stays inline.

    **The relocation is decided before anything is written** (S14). The locator
    is derived and not stored: the naming rule of the store is a function of the
    content. A caller can therefore refuse the spill because of the replacement
    itself. The overflow clip does so when the pointer would be longer than the
    result. At that point the store holds no byte of the result, and the log
    holds no record of a relocation that did not happen.

    This is one recipe in two halves. This function decides where the file goes
    and the wording that the model reads to find it. `spill_tool_result` does
    the write and the `offload/spilled` accounting. Two rows use the recipe.
    This row uses it when a result is oversized on arrival (G2). The overflow
    clip uses it when a retained batch must shrink (§7.4 item 7). The overflow
    clip had already drifted on `source`. `source` becomes
    `SpillRef.retrieval_hint`, so it is the *sentence that the model is given*.
    One relocation that is described in two ways, depending on which row did it,
    is exactly the drift that the shared recipe prevents.
    """
    blob = ctx.require(SPILL_STORE).plan_text(
        owner=session.id, suggested_name=_result_name(call_id), text=text
    )
    if blob is None:
        return None
    replacement = spill_wording(
        ctx,
        scope,
        TOO_LARGE_TOOL_MSG,
        tool_call_id=call_id,
        file_path=str(blob.locator),
        content_sample=content_preview(text),
    )
    return ResultSpill(call_id, source, blob, replacement)


async def spill_tool_result(
    ctx: Context,
    session: Session,
    planned: ResultSpill,
    *,
    rewriting: tuple[SessionEvent, Mapping[str, JsonValue]] | None = None,
) -> bool:
    """Make a planned relocation: write the blob, then record it.

    `rewriting` is set when the relocation is one half of a replacement, as it
    is for the overflow clip. It holds the result to rewrite in place and the
    message to rewrite it to. The `offload/spilled` record then goes through the
    revision function together with the rewrite, in one batch (S14). When the
    two were separate, a refused replace or a torn tail left an accounting
    record that described a replacement that never landed.

    `False` is the fail-open path that both callers need: an offload that cannot
    store the content must not be the reason that the model loses it. The seam
    logs the reason.
    """
    store = ctx.require(SPILL_STORE)
    ref = await store.try_save(planned.blob, source=planned.source)
    if ref is None:
        return False
    # The record of where the original went. The vocabulary declares this event
    # type ignorable (the property belongs to the type, not to this call site).
    # A reader that skips the record loses the forwarding address and not the
    # conversation, because `tool/result` carries the replacement that the model
    # saw. The record is written after the blob is on disk
    # (`SpillStore.try_save`), so no record names bytes that are not there.
    data: JsonObject = {"callId": planned.call_id, "locator": ref.locator, "bytes": ref.bytes}
    if rewriting is None:
        _LOG.append(session, "offload/spilled", data)
        return True
    size = RevisionSize("bytes", ref.bytes, len(planned.replacement.encode("utf-8")))
    rewrite(session, *rewriting, RevisionRecord(_LOG, "offload/spilled", data, size))
    return True


def _result_name(call_id: str) -> str:
    """The name that a tool result is spilled under. This is the half of the
    locator that this module chooses."""
    return f"large_tool_results/{call_id}"


def spill_wording(ctx: Context, scope: Boundary | None, template: str, **fields: str) -> str:
    """The replacement text of one spill, with the names of the tools that this
    deployment has.

    Both rows share this function. A spilled paste and a spilled tool result are
    the same kind of file in the same store, and the model cannot tell which row
    wrote the path that it was given. A sentence that is true of one and absent
    from the other is therefore worse than either alternative. That was the
    state before: the tool-result side got the correction, and the
    pasted-message side still told the model to call `read_file`.
    """
    # The registry is asked at `scope` because a tool can be registered for one
    # agent: the names in a replacement must be the names that *that* model can
    # call. `None` means the deployment view, which is what the overflow clip
    # has.
    readers, searchers = ctx.require(TOOLS).path_tools(scope=scope or DEPLOYMENT)
    said = template.format(reader=readers[0] if readers else GENERIC_READER, **fields)
    if not searchers:
        return said
    # The hint goes after the preview, not before it. The model reads the
    # preview first, to decide whether it needs the rest at all.
    hint = SPILL_TOOLS_HINT.format(searchers=" and ".join(f"`{name}`" for name in searchers))
    return f"{said}\n{hint}\n"


@plugin("tool-result-offload", affects="environment", inject=[TOOLS, SPILL_STORE], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Replace an oversized result with a preview and a path to the rest."""
    ctx.require(SPILL_STORE).claim(
        SpillClaim.under_session("tool-result-offload", "offload/spilled", hands_paths=True)
    )

    async def offload(
        execution: ToolExecution,
        result: ToolExecutionResult,
        next_: Next[PostToolDecision],
    ) -> PostToolDecision:
        decision = await next_(execution, result)
        session = execution.session
        if session is None or _self_limiting(execution):
            return decision
        if not isinstance(decision, Accept):
            return decision
        # Measure the projection as it will be sent. If an earlier listener
        # rewrote it, that is the projection of the earlier listener. If this
        # row measured the content of the body itself, any future post-execute
        # row (a redactor, a truncator) would switch this guard rail off by
        # touching the result at all.
        content = ctx.require(TOOLS).projected_content(execution, decision, result)
        if content is None:
            return decision
        text = text_of(content)
        if not oversized(text, config):
            # Hand the render back and do not drop it. When a decision carries
            # no content, the registry renders a replaced value itself. To
            # return `decision` here would therefore cause a *second* render of
            # a value that this row already rendered to measure it. That is the
            # one case where the "not paid twice" promise of `projected` would
            # not hold.
            return replace(decision, content=content) if decision.has_value else decision
        planned = plan_tool_result_spill(
            ctx,
            session,
            call_id=execution.call_id,
            source=f"{execution.name} result",
            text=text,
            # The names in the replacement must be the names that *this* model
            # can call, and a tool can be registered for one agent.
            scope=execution.scope,
        )
        if planned is None or not await spill_tool_result(ctx, session, planned):
            # Fail open, as upstream does: an offload that cannot store the
            # content must not be the reason that the model loses it.
            return decision
        # **The value passes through unchanged** (D10). Under Code Mode, the
        # program receives the value and already holds it in a variable. The
        # context cost was never the value. The cost is the render that lands in
        # the transcript. Under native tool calling, nothing reads the value, so
        # passing it on costs nothing there either. One rule covers both
        # transports: spill the render, leave the value alone.
        return replace(decision, content=text_content(planned.replacement))

    def _self_limiting(execution: ToolExecution) -> bool:
        """Whether this tool limits its own output and can page.

        The tool is asked, and the config list is an override. The row injects
        `tools` for exactly this question. A scope-aware lookup gives the
        correct answer for a registration that an agent shadows.
        """
        if execution.name in config.excluded_tools:
            return True
        definition = ctx.require(TOOLS).get(execution.name, scope=execution.scope)
        return bool(definition is not None and definition.self_limits)

    ctx.on("tools/post-execute", offload)
