"""`clm-context`: the section map and the seven context tools.

The tools are the explicit front end of `ph_clm.edits`. The model names sections
and says what to do with them. The edit lands in the session log before the call
returns, so the result of the call is the receipt. No notice is added to the
context to say that the edit happened. In Code Mode the tools are
`tools.context_*` like any other tool, so a cell can compute an edit in Python
and make it.

**Readouts are appended to tool results** (`ph_clm.budget`). When a result takes
the context past a share of its window (`remindAt`), that result gets one more
line that gives the size of the context and says how to free space. The readout
listener is outside every row that spills or trims the result, so it measures
what is logged. The row of the context file (`clm-mirror`) is outside the
readout listener. It lands the edit of the call before the readout measures, and
it appends its receipt after the readout.

An edit lands inside the step that asked for it, on sections that the step
cannot be part of. The step in flight is protected (`SectionMap._protection`).
`reconcile` handles a crash between the edit and its result. The edit and its
`clm/revised` record land in one batch, so the record is in the log exactly when
the edit is.

@module ph_clm.tools
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Annotated, Any

from pydantic import Field

from ph.characters import head
from ph.cordis import Context, Next, plugin
from ph.json import JsonObject
from ph.keys import COMPACTION, TOKEN_METER, TOOLS
from ph.llm.types import ContentBlock, TextBlock, text_of
from ph.seams.invariants import contribute_fold_cache
from ph.session import (
    Session,
    SessionEvent,
    is_in_place_rewrite,
    messages_of,
    originals,
    render_messages,
    shape_of,
    unified_diff,
)
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

from .budget import Readouts, Shares
from .edits import Editor, Gate, Revision, receipt, revision_of
from .keys import CLM, CLM_MIRROR
from .sections import (
    EditRefused,
    Section,
    SectionMap,
    label,
    resolve_one,
    span_label,
)

__all__ = ["Config", "apply"]

DIFF_CHARS = 20_000
"""The maximum length of a diff in one result. The rest is cut, with a note that says so."""


class Config(WireModel):
    """The row config for `clm-context`."""

    protect_task: bool = True
    """Keep the first message that the user typed out of every edit. The harness
    of the paper pins its task in the same way. `false` makes that message as
    editable as any other."""
    gate: Gate = "none"
    """`none` accepts any edit. `fit` refuses growth past the context window.
    `shrink` refuses any growth."""
    map_limit: int = Field(default=60, ge=1)
    """How many sections `context_sections` lists when the model gives no limit."""
    remind_at: list[Annotated[float, Field(gt=0, lt=1)]] = Field(
        default_factory=lambda: [0.5, 0.75]
    )
    """The shares of the context window at which a tool result carries a readout
    of the size of the context, one time for each crossing. An empty list turns
    readouts off. When a compaction engine is mounted, the last reminder comes
    `lastMargin` below the compaction trigger. It replaces every share here at or
    past that point."""
    last_margin: float = Field(default=0.2, gt=0, lt=1)
    """How far below the compaction trigger the last reminder comes, as a share of
    the window. This gives the model room to act before compaction writes a
    summary for it. With the default trigger of 0.85, a value of 0.2 puts the
    last reminder at 65%. On a 16k slot that is about 3,000 tokens, or two or
    three steps. A value of 0.1 left room for one file read, and a live run lost
    its edit to the summary."""


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


class RestoreArgs(ToolModel):
    section: str = Field(description="A section you revised, e.g. S2210.")


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
    """The section to pass as `start` to continue the list, when the map was cut."""


class EditValue(ToolModel):
    verb: str
    sections: str
    replacement: str
    tokens_before: int
    tokens_after: int
    reread: int
    context_before: int
    context_after: int
    """`context_before - tokens_before + tokens_after`, given so that a cell does
    not need to compute it."""
    receipt: str
    """The text that the model reads: what changed and what it cost
    (`edits.receipt`)."""


class RecallValue(ToolModel):
    section: str
    text: str
    tokens: int
    truncated: bool


class DiffValue(ToolModel):
    section: str
    edits: list[str]
    """The revisions behind the section, oldest first (`Session.history`)."""
    diff: str
    truncated: bool


# ---------------------------------------------------------------- the row --


@plugin(
    "clm-context", affects="environment", inject=[TOOLS, TOKEN_METER, COMPACTION], config=Config
)
async def apply(ctx: Context, config: Config) -> None:
    """Register the section map and the context tools."""
    tools = ctx.require(TOOLS)
    meter = ctx.require(TOKEN_METER)
    mapper = SectionMap(meter, protect_task=config.protect_task)
    editor = Editor(mapper, config.gate)
    ctx.provide(CLM, editor)
    # The per-event facts of the map are a fold over the log. Every such cache gets
    # its own invariant row (`docs/seams/invariants.md`, "Fold caches get one row
    # each").
    contribute_fold_cache(ctx, id="clm-section-map", subject="section map", stale=mapper.stale)
    readouts = Readouts()

    async def after_call(
        execution: ToolExecution, result: ToolExecutionResult, next_: Next[PostToolDecision]
    ) -> PostToolDecision:
        decision = await next_(execution, result)
        session = execution.session
        if execution.parent is not None or session is None or not isinstance(decision, Accept):
            return decision
        baseline = meter.baseline(session)
        window = baseline.context_window
        if not window:
            return decision
        content = tools.projected_content(execution, decision, result)
        if content is None:
            return decision
        text = text_of(content)
        before = baseline.tokens
        compacts_at = ctx.require(COMPACTION).trigger_fraction()
        shares = Shares.of(config.remind_at, config.last_margin, compacts_at, window)
        # A token is at least one byte, so the UTF-8 length of the result is an
        # upper bound on its token count. The result is tokenized only when that
        # bound could cross a threshold.
        after = readouts.crossed(
            session,
            shares,
            before,
            before + len(text.encode()),
            lambda: before + meter.measure_text(text),
        )
        if after is None:
            return decision
        line = shares.line(after, _how(ctx, execution))
        return replace(decision, content=[*content, TextBlock(text=line)])

    if config.remind_at:
        # Put this outside every row that spills or trims the result, whatever
        # the mount order of the rows (see the module docstring).
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
        text, truncated = _bounded(
            meter.measure_text, _original_text(session, section), args.max_tokens
        )
        return RecallValue(
            section=section.name,
            text=text,
            tokens=meter.measure_text(text),
            truncated=truncated,
        ).model_dump()

    def diff_tool(args: DiffArgs, run: ToolRunContext) -> JsonObject:
        session = _session(run)
        section = _revised(session, mapper, args.section)
        current = render_messages(messages_of(session, section.nodes))
        labels = (f"{section.name} as first written", f"{section.name} now")
        diff = "\n".join(unified_diff(_original_text(session, section), current, labels))
        return DiffValue(
            section=section.name,
            edits=[
                _edit_line(session, event)
                for seq in section.nodes
                for event in session.history(seq)
            ],
            diff=head(diff, DIFF_CHARS),
            truncated=len(diff) > DIFF_CHARS,
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
        _edit_tool(
            "context_restore",
            "Undo the latest edit of one section, one level. A rewrite goes back to the "
            "text before it, exactly; a removed or replaced run comes back as the text it "
            "held, in one message under a label, and restoring that undoes the restore. "
            "Everything after the edit is re-read once; to read the originals without "
            "changing your context, use context_recall.",
            RestoreArgs,
            lambda args, session, call: editor.restore(session, args.section, call_id=call),
            "Restore context",
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
            "Show a unified diff of a revised section against the originals it stands for, "
            "after the list of edits that led there, oldest first.",
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
    """Define one editing tool. Its edit lands during the call, and a crash is
    reconciled."""

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
    """Decide whether this edit landed. Called on resume, for a call that a crash
    left open.

    The edit and its `clm/revised` record are one batch, so the record is in the
    log exactly when the edit is. If the record is found, the call is done and
    the model is shown its receipt. If the record is absent, nothing changed and
    it is safe to run the call again.
    """
    call_id = call_id_of(opened)
    if not call_id:
        return Unknown()
    for event in session.events_from(opened.seq):
        revision = revision_of(event)
        if revision is not None and revision.call_id == call_id:
            landed = session.at(revision.replacement)
            in_place = landed is not None and is_in_place_rewrite(landed)
            return Done(_edit_value(replace(revision, in_place=in_place)))
    return NotDone()


# ------------------------------------------------------------------ helpers --


def _how(ctx: Context, execution: ToolExecution) -> str:
    """How a readout tells the model to free space. It names the context file
    when a row keeps one, and it names the tools in every case. In Code Mode it
    shows the full call that a cell writes, including the keywords. A live model
    passed the sections as positional arguments, and the error cost it its one
    chance to edit."""
    if ctx.require(TOOLS).mode_for(execution.scope) == "code":
        verbs = (
            "in a cell, await tools.context_tombstone(sections=['S12'], reason='...') "
            "or tools.context_replace(sections=['S12..S20'], text='...')"
        )
    else:
        verbs = "use context_tombstone, context_replace or context_rewrite"
    mirror = ctx.get(CLM_MIRROR)
    if mirror is None or execution.session is None or execution.agent is None:
        return verbs
    return f"edit {mirror.path(execution.session, execution.agent)}, or {verbs}"


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


def _edit_line(session: Session, event: SessionEvent) -> str:
    """One revision behind a section, as `context_diff` lists it: where it landed,
    what it did, and the records that its producer wrote for it."""
    records = ", ".join(record.type for record in session.records_of(event.seq))
    return f"seq {event.seq}: {shape_of(event)}" + (f" ({records})" if records else "")


def _original_text(session: Session, section: Section) -> str:
    """The conversation that a section stands for, as it was first written."""
    return render_messages(message for seq in section.nodes for message in originals(session, seq))


def _bounded(measure: Callable[[str], int], text: str, max_tokens: int) -> tuple[str, bool]:
    """Cut `text` to at most `max_tokens`, and report whether it was cut. The
    function finds the longest start that fits by halving. It then moves the cut
    back so that it falls between characters (`ph.characters.head`). A shorter
    start never measures more."""
    if measure(text) <= max_tokens:
        return text, False
    marker = "\n[recall cut at the token bound; the session log holds the rest]"
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if measure(text[:middle] + marker) <= max_tokens:
            low = middle
        else:
            high = middle - 1
    return head(text, low) + marker, True


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
    heading = f"[{value['section']} as first written, ~{thousands(value['tokens'])} tokens]"
    return text_content(f"{heading}\n{value['text']}")


def _render_diff(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    edits = "".join(f"- {line}\n" for line in value["edits"])
    body = value["diff"] or "(no difference)"
    if value["truncated"]:
        body += "\n[diff cut; the session log holds both versions]"
    return text_content(f"Edits, oldest first:\n{edits}\n{body}" if edits else body)
