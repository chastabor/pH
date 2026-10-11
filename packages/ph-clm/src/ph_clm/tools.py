"""`clm-context` — the section map and the five context tools.

The tools are the explicit front end of `ph_clm.edits`: the model names sections
and says what to do with them, and the edit lands in the session log before the
call returns, so the call's own result is the receipt — no notice is added to the
context to say it happened. Under Code Mode they are `tools.context_*` like any
other tool, so a cell can compute an edit in Python and make it.

**Readouts ride on tool results** (`ph_clm.budget`): when a result takes the context
past a share of its window (`remindAt`), that result ends with one line saying how
large the context is and how to free space. The outermost listener on the result, so
it measures what is logged: after the context file's read-back has landed the call's
edit, and after any row that spills or trims the result.

An edit lands inside the step that asked for it, on sections that step cannot be
part of: the step in flight is protected (`SectionMap._protection`). A crash between
the edit and its result is answered by `reconcile`: the edit and its `clm/revised`
record land in one batch, so the record is in the log exactly when the edit is.

@module ph_clm.tools
"""

from __future__ import annotations

import difflib
from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any

from pydantic import Field

from ph.cordis import Context, Next, plugin
from ph.json import JsonObject
from ph.keys import TOKEN_METER, TOOLS
from ph.llm.types import ContentBlock, TextBlock, text_of
from ph.seams.invariants import contribute_fold_cache
from ph.seams.token_meter import TokenMeter
from ph.session import Session, SessionEvent, derive_event_message, originals
from ph.text import thousands
from ph.tools.definition import (
    Accept,
    Done,
    NotDone,
    PostToolDecision,
    Reconciled,
    ToolDefinition,
    ToolExecution,
    ToolExecutionResult,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    Unknown,
    call_id_of,
    define_tool,
    text_content,
)
from ph.tools.presentation import simple_views
from ph.wire import WireModel

from .budget import Readouts
from .edits import Editor, Gate, Revision, receipt, revision_of
from .keys import CLM, CLM_MIRROR
from .sections import (
    EditRefused,
    Section,
    SectionMap,
    label,
    render_messages,
    resolve_one,
    span_label,
)

__all__ = ["Config", "apply"]

DIFF_CHARS = 20_000
"""The most of a diff one result carries; the rest is cut with a note saying so."""


class Config(WireModel):
    """Row config."""

    protect_task: bool = True
    """Keep the first message the person typed out of every edit — the paper harness
    pins its task the same way. `false` makes it as editable as anything else."""
    gate: Gate = "none"
    """`none` takes any edit; `fit` refuses growth past the context window; `shrink`
    refuses any growth."""
    map_limit: int = Field(default=60, ge=1)
    """How many sections `context_sections` lists when the model does not say."""
    remind_at: list[Annotated[float, Field(gt=0, lt=1)]] = Field(
        default_factory=lambda: [0.5, 0.75]
    )
    """Shares of the context window at which a tool result carries a readout of the
    context's size, once per crossing. Below compaction's 0.85 by default, so the model
    is told before a summary is written for it. Empty turns readouts off."""


# ------------------------------------------------------------------ schemas --


class SectionsArgs(ToolModel):
    start: str | None = Field(
        None, description="List from this section on (an id like S412). Default: the first."
    )
    limit: int | None = Field(None, ge=1, le=500, description="How many sections to list.")


class TombstoneArgs(ToolModel):
    sections: list[str] = Field(
        min_length=1,
        description="The sections to remove: ids (S412) or spans (S412..S431), one unbroken run.",
    )
    reason: str = Field(description="Why, in a few words. It stays in context as the marker.")


class ReplaceArgs(ToolModel):
    sections: list[str] = Field(
        min_length=1,
        description="The sections to replace: ids (S412) or spans (S412..S431), one unbroken run.",
    )
    text: str = Field(description="What stands for them from now on: a summary, or a rewrite.")


class RewriteArgs(ToolModel):
    section: str = Field(description="The section holding the text, e.g. S412.")
    old: str = Field(description="The exact text to change. It must occur once in the section.")
    new: str = Field(description="What it becomes. Empty deletes it.")


class RecallArgs(ToolModel):
    section: str = Field(description="A revised section, e.g. S2210.")
    max_tokens: int = Field(2000, ge=128, le=8000, description="The most to return.")


class DiffArgs(ToolModel):
    section: str = Field(description="A revised section, e.g. S2210.")


class SectionValue(ToolModel):
    id: str
    kind: str
    tokens: int
    after: int
    protected: str | None
    calls: list[str]
    preview: str
    stands_for: str | None


class MapValue(ToolModel):
    sections: list[SectionValue]
    total_sections: int
    total_tokens: int
    next: str | None
    """The section to pass as `start` to list on, when the map was cut."""


class EditValue(ToolModel):
    verb: str
    sections: str
    replacement: str
    tokens_before: int
    tokens_after: int
    reread: int
    context_before: int
    context_after: int
    """`context_before - tokens_before + tokens_after`, given so a cell need not."""
    receipt: str
    """What the model reads: what changed and what it cost (`edits.receipt`)."""


class RecallValue(ToolModel):
    section: str
    text: str
    tokens: int
    truncated: bool


class DiffValue(ToolModel):
    section: str
    diff: str
    truncated: bool


# ---------------------------------------------------------------- the row --


@plugin("clm-context", affects="environment", inject=[TOOLS, TOKEN_METER], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Register the section map and the five context tools."""
    tools = ctx.require(TOOLS)
    meter = ctx.require(TOKEN_METER)
    mapper = SectionMap(meter, protect_task=config.protect_task)
    editor = Editor(mapper, config.gate)
    ctx.provide(CLM, editor)
    # The map's per-event facts are a fold over the log, and every such cache gets
    # its own invariant row (`docs/seams/invariants.md`, "Fold caches get one row each").
    contribute_fold_cache(ctx, id="clm-section-map", subject="section map", stale=mapper.stale)
    readouts = Readouts(tuple(config.remind_at))

    def forget(session: Session) -> None:
        mapper.forget(session.id)
        readouts.forget(session.id)

    ctx.on("session/disposed", forget)

    async def after_call(
        execution: ToolExecution, result: ToolExecutionResult, next_: Next[PostToolDecision]
    ) -> PostToolDecision:
        decision = await next_(execution, result)
        session = execution.session
        if execution.parent is not None or session is None or not isinstance(decision, Accept):
            return decision
        route = session.request_context()
        window = None if route is None else route.context_window
        if not window:
            return decision
        content = tools.projected_content(execution, decision, result)
        if content is None:
            return decision
        text = text_of(content)
        before = mapper.tokens(session)
        # A token is at least a byte, so the result's UTF-8 length bounds its count, and
        # it is tokenized only when that bound could cross a threshold.
        crossed = readouts.crossed(
            session.id,
            before,
            before + len(text.encode()),
            window,
            lambda: before + meter.measure_text(text),
        )
        if crossed is None:
            return decision
        level, after = crossed
        line = readouts.line(after, window, level, _how(ctx, execution))
        return replace(decision, content=[*content, TextBlock(text=line)])

    if readouts.thresholds:
        # Outermost, whatever order the rows mounted in (see the module docstring).
        ctx.on("tools/post-execute", after_call, prepend=True)

    def sections_tool(args: SectionsArgs, run: ToolRunContext) -> JsonObject:
        session = _session(run)
        sections = mapper(session)
        first = (
            sections.index(resolve_one(session, sections, args.start, editing=False))
            if args.start is not None
            else 0
        )
        limit = args.limit or config.map_limit
        rest = sections[first + limit :]
        return MapValue(
            sections=[_section_value(section) for section in sections[first : first + limit]],
            total_sections=len(sections),
            total_tokens=sum(section.tokens for section in sections),
            next=rest[0].name if rest else None,
        ).model_dump()

    def recall_tool(args: RecallArgs, run: ToolRunContext) -> JsonObject:
        session = _session(run)
        section = _revised(session, mapper, args.section)
        text, truncated = _bounded(meter, _original_text(session, section), args.max_tokens)
        return RecallValue(
            section=section.name,
            text=text,
            tokens=meter.measure_text(text),
            truncated=truncated,
        ).model_dump()

    def diff_tool(args: DiffArgs, run: ToolRunContext) -> JsonObject:
        session = _session(run)
        section = _revised(session, mapper, args.section)
        current = render_messages(
            message
            for event in map(session.at, section.nodes)
            if event is not None and (message := derive_event_message(event)) is not None
        )
        lines = difflib.unified_diff(
            _original_text(session, section).splitlines(),
            current.splitlines(),
            fromfile=f"{section.name} as first written",
            tofile=f"{section.name} now",
            lineterm="",
        )
        diff = "\n".join(lines)
        return DiffValue(
            section=section.name, diff=diff[:DIFF_CHARS], truncated=len(diff) > DIFF_CHARS
        ).model_dump()

    tools.register(
        define_tool(
            "context_sections",
            "List the sections of your own context — what you see of this conversation — "
            "with each one's id, kind, size, and how many tokens follow it. Ids look like "
            "S412 and never change; the other context tools take them.",
            parameters=SectionsArgs,
            output=ToolOutput(schema=MapValue, render=_render_map),
            execute=sections_tool,
            effect_free=True,
            is_concurrency_safe=True,
            **simple_views("generic", "Context sections", "start"),
        )
    )
    tools.register(
        _edit_tool(
            "context_tombstone",
            "Remove a run of sections from your context, leaving a one-line marker with "
            "your reason. The originals stay in the session log; context_recall reads "
            "them back. Everything after the edit is re-read once, so edit late "
            "sections, or many at once, rather than a small early one.",
            TombstoneArgs,
            lambda args, session, call: editor.tombstone(
                session, args.sections, args.reason, call_id=call
            ),
            "Remove context",
        )
    )
    tools.register(
        _edit_tool(
            "context_replace",
            "Replace a run of sections in your context with text you write — a summary "
            "that keeps what still matters, or a rewrite of the span. The originals stay "
            "in the session log. A detailed summary costs little: everything after the "
            "edit is re-read once regardless.",
            ReplaceArgs,
            lambda args, session, call: editor.replace(
                session, args.sections, args.text, call_id=call
            ),
            "Replace context",
        )
    )
    tools.register(
        _edit_tool(
            "context_rewrite",
            "Change one passage inside one section's tool output or reply, in place — "
            "shorten a stale output, correct a note. `old` must occur exactly once in "
            "the section. The original stays in the session log.",
            RewriteArgs,
            lambda args, session, call: editor.rewrite(
                session, args.section, args.old, args.new, call_id=call
            ),
            "Rewrite context",
        )
    )
    tools.register(
        define_tool(
            "context_recall",
            "Read back the original conversation a revised section stands for. The text "
            "arrives as this call's result; the revision stays as it is.",
            parameters=RecallArgs,
            output=ToolOutput(schema=RecallValue, render=_render_recall),
            execute=recall_tool,
            effect_free=True,
            is_concurrency_safe=True,
            **simple_views("generic", "Recall context", "section"),
        )
    )
    tools.register(
        define_tool(
            "context_diff",
            "Show a unified diff of a revised section against the originals it stands for.",
            parameters=DiffArgs,
            output=ToolOutput(schema=DiffValue, render=_render_diff),
            execute=diff_tool,
            effect_free=True,
            is_concurrency_safe=True,
            **simple_views("diff", "Context diff", "section"),
        )
    )


def _edit_tool[A: ToolModel](
    name: str,
    description: str,
    parameters: type[A],
    edit: Callable[[A, Session, str], Revision],
    title: str,
) -> ToolDefinition:
    """One editing tool: its edit lands in the call, and a crash is reconciled."""

    def execute(args: A, run: ToolRunContext) -> JsonObject:
        return _edit_value(edit(args, _session(run), run.call_id))

    return define_tool(
        name,
        description,
        parameters=parameters,
        output=ToolOutput(schema=EditValue, render=_render_edit),
        execute=execute,
        is_concurrency_safe=False,
        reconcile=_reconcile_edit,
        **simple_views(
            "generic", title, "section" if "section" in parameters.model_fields else "sections"
        ),
    )


async def _reconcile_edit(
    _arguments: Any,  # noqa: ANN401
    opened: SessionEvent,
    session: Session,
) -> Reconciled:
    """Did this edit land? Asked on resume, for a call a crash left open.

    The edit and its `clm/revised` record are one batch, so the record is in the log
    exactly when the edit is: found, the call is done and the model is shown its
    receipt; absent, nothing changed and running it again is safe.
    """
    call_id = call_id_of(opened)
    if not call_id:
        return Unknown()
    for event in session.events_from(opened.seq):
        revision = revision_of(event)
        if revision is not None and revision.call_id == call_id:
            return Done(_edit_value(revision))
    return NotDone()


# ------------------------------------------------------------------ helpers --


def _how(ctx: Context, execution: ToolExecution) -> str:
    """How a readout says to free space: the context file when a row keeps one, and
    the tools either way."""
    verbs = "context_tombstone, context_replace or context_rewrite"
    mirror = ctx.get(CLM_MIRROR)
    if mirror is None or execution.session is None or execution.agent is None:
        return f"use {verbs}"
    return f"edit {mirror.path(execution.session, execution.agent)}, or use {verbs}"


def _session(run: ToolRunContext) -> Session:
    if run.session is None:
        raise EditRefused("the context tools need a session to edit")
    return run.session


def _revised(session: Session, mapper: SectionMap, name: str) -> Section:
    section = resolve_one(session, mapper(session), name, editing=False)
    if not section.stands_for:
        raise EditRefused(
            f"{section.name} has not been revised; it is in your context as it was written"
        )
    return section


def _original_text(session: Session, section: Section) -> str:
    """The conversation a section stands for, as it was first written."""
    return render_messages(message for seq in section.nodes for message in originals(session, seq))


def _bounded(meter: TokenMeter, text: str, max_tokens: int) -> tuple[str, bool]:
    """`text` cut to at most `max_tokens`, and whether it was cut."""
    if meter.measure_text(text) <= max_tokens:
        return text, False
    marker = "\n[recall cut at the token bound; the session log holds the rest]"
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if meter.measure_text(text[:middle] + marker) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return text[:low] + marker, True


def _section_value(section: Section) -> SectionValue:
    return SectionValue(
        id=section.name,
        kind=section.kind,
        tokens=section.tokens,
        after=section.after,
        protected=section.protected,
        calls=list(section.calls),
        preview=section.preview,
        stands_for=(
            span_label(min(section.stands_for), max(section.stands_for))
            if section.stands_for
            else None
        ),
    )


def _edit_value(revision: Revision) -> JsonObject:
    return EditValue(
        verb=revision.verb,
        sections=span_label(revision.first, revision.last),
        replacement=label(revision.replacement),
        tokens_before=revision.tokens_before,
        tokens_after=revision.tokens_after,
        reread=revision.reread,
        context_before=revision.context_before,
        context_after=revision.context_after,
        receipt=receipt([revision]),
    ).model_dump()


def _render_map(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    lines = [
        f"{value['total_sections']} sections, ~{thousands(value['total_tokens'])} tokens. "
        "`after` is what an edit there makes the provider re-read."
    ]
    for one in value["sections"]:
        what = one["kind"]
        if one["calls"]:
            what += " " + ",".join(one["calls"])
        if one["stands_for"]:
            what += f" (stands for {one['stands_for']})"
        line = f"{one['id']}  {what}  ~{thousands(one['tokens'])}  after ~{thousands(one['after'])}"
        if one["protected"]:
            line += "  [protected]"
        if one["preview"]:
            line += f"  — {one['preview']}"
        lines.append(line)
    if value["next"]:
        lines.append(f"… more from {value['next']}: pass start={value['next']!r}.")
    return text_content("\n".join(lines))


def _render_edit(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    return text_content(value["receipt"])


def _render_recall(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    head = f"[{value['section']} as first written, ~{thousands(value['tokens'])} tokens]"
    return text_content(f"{head}\n{value['text']}")


def _render_diff(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    body = value["diff"] or "(no difference)"
    if value["truncated"]:
        body += "\n[diff cut; the session log holds both versions]"
    return text_content(body)
