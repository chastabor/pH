"""The transcript: rows that render from `TuiState` and nothing else.

This module enforces two rules, so that nobody has to remember them (P2-06).

**Never build markup with an f-string.** User text, tool output and provider
errors all contain `[`: a path like `foo[0]`, a Python list, a Rich tag that
somebody typed on purpose. `Content(text)` renders the text literally. An
f-string passed to a markup parser makes the output of the model a formatting
language, and at worst the parser swallows the output. Where styling *is*
wanted, `Content.from_markup` takes `$variables`, so the substituted value can
never be parsed as markup.

**Stream by appending, not by parsing again.** `MarkdownStream.write(fragment)`
appends to a live Markdown widget. A call to `update()` with the accumulated
text on every delta parses the whole message again for each token. That is
quadratic, and the display visibly stutters by the second paragraph. The view
tracks what it has already written to each row and sends only the tail.

The same rule applies one layer up. A settled row **remembers what it last
drew** and skips `update()` when nothing changed. `sync` visits every visible
row on every dirty frame, and `Static.update` always runs the layout again.
Without the memo, a 200-row transcript rendered 200 widgets again for each
frame while it streamed.

@module ph_app.tui.widgets.transcript
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any, ClassVar, TypeAlias

import anyio
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Collapsible, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from ph.json import JsonValue, as_int
from ph.text import count_of

from ...wire import index_at_or_before
from ..state import ChatItem, ToolCard
from .gutter import ROW_BAR, Barred, GutterScroll
from .selection import SOURCE_BLOCKS

__all__ = [
    "CodeCellWidget",
    "StreamingMessage",
    "ToolCardWidget",
    "TranscriptRow",
    "TranscriptView",
]

_ROLE_LABEL = {
    "user": "you",
    "assistant": "pH",
    "thinking": "thinking",
    "context": "context",
    "notice": "·",
    "error": "!",
    "compaction": "compacted",
}

MARKDOWN_ROLES: frozenset[str] = frozenset({"assistant", "thinking"})
"""Roles rendered as Markdown.

This is a consistency requirement, not a display preference. The model writes
Markdown, so its rows render it. If a *streamed* row rendered Markdown while a
*replayed* row showed raw asterisks, a resumed session would look different from
the session that the user left. That is the failure that the P2-01 gate guards
against, one layer up.

The text of the user is deliberately absent: what somebody typed is shown as
they typed it.
"""

_ROLE_STYLE = {
    "user": "$ph-user-text",
    "assistant": "$ph-assistant-text",
    "thinking": "$ph-thinking-text",
    "context": "$ph-muted",
    "notice": "$ph-muted",
    "error": "$ph-error",
    "compaction": "$ph-warning",
}


def _status_markup(card: ToolCard) -> tuple[str, str]:
    """Return the glyph and its style for the state of a call. Cards and dispatches
    share this one rule."""
    glyph = "…" if not card.settled else ("✗" if card.is_error else "✓")
    return glyph, ("$ph-tool-error" if card.is_error else "$ph-tool-success")


class TranscriptRow(Barred, Vertical):
    """A settled row: a styled label over its body.

    The gutter draws the bar of a row (see `widgets.gutter`). No role has a bar
    except a compaction marker. That is the one settled row that marks itself:
    it is the place where the conversation was folded.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {ROW_BAR}

    DEFAULT_CSS = """
    TranscriptRow { height: auto; margin: 0 1 1 0; }
    TranscriptRow > .row-label { color: $ph-muted; }
    TranscriptRow > .row-body { height: auto; }
    TranscriptRow.-error > .row-body { color: $ph-error; }
    TranscriptRow.-thinking > .row-body { color: $ph-thinking-text; }
    TranscriptRow.-context > .row-body { color: $ph-muted; }
    TranscriptRow.-compaction > .row--bar { color: $ph-warning; }
    TranscriptRow.-shadowed > .row-body { color: $ph-muted; }
    """

    def __init__(self, item: ChatItem) -> None:
        super().__init__(id=f"row-{_slug(item.key)}")
        self.item = item
        self._shown: tuple[str, bool] | None = None
        self.add_class(f"-{item.role}")

    def compose(self) -> ComposeResult:
        label = _ROLE_LABEL.get(self.item.role, self.item.role)
        style = _ROLE_STYLE.get(self.item.role, "$ph-foreground")
        # `$label` is substituted, never parsed. A role name could not introduce
        # markup here, but the rule is reliable because it is the same everywhere.
        yield Static(Content.from_markup(f"[{style}]$label[/]", label=label), classes="row-label")
        yield Static(Content(self.item.text), classes="row-body")

    def on_mount(self) -> None:
        self._shown = (self.item.text, self.item.shadowed)
        # Dimmed, not removed: the user read it, and the model no longer has it.
        self.set_class(self.item.shadowed, "-shadowed")

    def refresh_text(self) -> None:
        current = (self.item.text, self.item.shadowed)
        if current == self._shown:
            return
        self._shown = current
        self.set_class(self.item.shadowed, "-shadowed")
        # The row can be mounted and not yet composed: it was added and changed
        # inside one frame. A `/command` that logs `run` and `done` around an
        # await does that deterministically. `compose` reads `self.item` when it
        # runs, so it will draw the latest text, and there is nothing to update
        # yet. The widget owns the lifetime of its children, so the widget is
        # what handles this case.
        with suppress(NoMatches):
            self.query_one(".row-body", Static).update(Content(self.item.text))


class StreamingMessage(Barred, Markdown):
    """An assistant or thinking row that grows a fragment at a time.

    **A copy gives Markdown, not what was drawn.** `SOURCE_BLOCKS` is all that
    this side needs for that. Every block answers a selection with its own span
    of `item.text` and not with its rendering. `ph_app.tui.widgets.selection`
    explains this. The other transcript rows need nothing, because they are
    `Static`s that already hold the text.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {ROW_BAR}

    DEFAULT_CSS = """
    /* `padding: 0` undoes Textual's own `Markdown { padding: 0 2 1 2 }`: the
       answer starts in column 1 with every other row and the prompt, and the
       row's margin is what spaces it — two blank rows under an answer, one
       under everything else, was that padding's bottom row. */
    StreamingMessage { height: auto; margin: 0 1 1 0; padding: 0; }
    /* The model's answer carries no label — it is the default voice of the
       transcript. Reasoning does need marking, since it can be toggled off and
       a reader must be able to tell which they are looking at: its bar, drawn
       by the gutter, is the mark. */
    StreamingMessage.-thinking { color: $ph-thinking-text; }
    StreamingMessage.-thinking > .row--bar { color: $ph-thinking-text; }
    StreamingMessage.-streaming MarkdownFence { overflow-x: hidden; scrollbar-size-horizontal: 0; }
    StreamingMessage.-finalized MarkdownFence { overflow-x: auto; scrollbar-size-horizontal: 1; }
    """

    BLOCKS = SOURCE_BLOCKS

    def __init__(self, item: ChatItem) -> None:
        super().__init__("", id=f"row-{_slug(item.key)}")
        self.item = item
        self._stream: MarkdownStream | None = None
        self._written = ""
        """What has been handed to the stream: the string, not its length.

        A count is a safe cursor only while every update *extends* the last one.
        The text of the transcript is a derivation that can be replaced.
        Attaching in the middle of an answer builds the row from a snapshot, and
        the settled `assistant/message` is assembled from blocks and not from the
        concatenated chunks. When the new text was not an extension, slicing it
        at the old length appended the middle of the answer to the end of it. The
        row showed that until something forced a rebuild."""
        self.add_class(f"-{item.role}")
        self.add_class("-streaming")

    async def write_tail(self, text: str) -> None:
        """Append the part of `text` that has not been written yet.

        **The decision compares the strings.** "This is more of what we have"
        and "this is something else" are then told apart and not assumed. Only
        the first is an append. The second is a rewrite, which costs one more
        render of a row that was wrong anyway.
        """
        if text == self._written:
            return
        if not text.startswith(self._written):
            # The text of the row changed under the stream, so render it whole.
            # The stream is stopped first, which flushes whatever the stream
            # still had pending. An `update` before that would be overwritten by
            # the fragments that arrive after it. A later extension opens a
            # stream again and appends to `Markdown.source`, which `update` has
            # just set, so the two compose.
            await self._stop_stream()
            self._written = text
            await self.update(text)
            return
        fragment = text[len(self._written) :]
        self._written = text
        if self._stream is None:
            self._stream = self.get_stream(self)
        await self._stream.write(fragment)

    async def finalize(self, text: str) -> None:
        """Flush, then restore finalized Markdown chrome (scrollbars come back)."""
        await self.write_tail(text)
        await self._stop_stream()
        self.remove_class("-streaming")
        self.add_class("-finalized")

    async def on_unmount(self) -> None:
        await self._stop_stream()

    async def _stop_stream(self) -> None:
        if self._stream is not None:
            stream, self._stream = self._stream, None
            await stream.stop()


class ToolCardWidget(Barred, Vertical):
    """One tool call: a header line, its body, and any Code Mode sub-dispatches.

    Dispatch rows are mounted as they arrive, keyed by call id. They are not
    composed one time with the card. Live, the card mounts on the tick after
    `tool/call`, before any `tool/code-dispatch-start` exists. Content built at
    compose time would therefore show the "governed calls" section on replay and
    never live. The same fold must give the same picture: that is the promise of
    P2-01 at the level of the widget.
    """

    COMPONENT_CLASSES: ClassVar[set[str]] = {ROW_BAR}

    DEFAULT_CSS = """
    ToolCardWidget { height: auto; margin: 0 1 1 0; }
    /* The bar the gutter draws beside the card: pending, done, failed. `-error`
       after `-settled`, because a failed call is settled too and the two rules
       weigh the same — the later one is the one that shows. */
    ToolCardWidget > .row--bar { color: $ph-border; }
    ToolCardWidget.-settled > .row--bar { color: $ph-tool-success; }
    ToolCardWidget.-error > .row--bar { color: $ph-tool-error; }
    ToolCardWidget > .tool-body { color: $ph-muted; height: auto; }
    ToolCardWidget .dispatch { color: $ph-muted; }
    """

    def __init__(self, item: ChatItem) -> None:
        super().__init__(id=f"row-{_slug(item.key)}")
        self.item = item
        self._shown: tuple[object, ...] | None = None
        self._dispatch_rows: dict[str, tuple[Static, tuple[bool, bool]]] = {}
        self._dispatch_box = Vertical(classes="dispatches")

    def compose(self) -> ComposeResult:
        yield Static(self._header(), classes="tool-header")
        yield from self._rows_after_header()
        yield Static(Content(self._body()), classes="tool-body")
        yield from self._rows_after_body()
        with Collapsible(title="governed calls", collapsed=True, id="dispatches"):
            yield self._dispatch_box

    def _rows_after_header(self) -> ComposeResult:
        """What a card kind adds above its output. Empty for most cards."""
        return iter(())

    def _rows_after_body(self) -> ComposeResult:
        """What a card kind adds below its output."""
        return iter(())

    async def on_mount(self) -> None:
        self.query_one(Collapsible).display = False
        await self.refresh_card()

    def _snapshot(self) -> tuple[object, ...]:
        card = self.item.tool
        if card is None:
            return (self.item.text,)
        return (
            card.settled,
            card.is_error,
            card.failure_kind,
            card.title,
            card.subtitle,
            card.body,
        )

    def _header(self) -> Content:
        card = self.item.tool
        if card is None:
            return Content("tool")
        glyph, style = _status_markup(card)
        # Every dynamic part is a substituted variable, so a path containing
        # brackets cannot become markup.
        return Content.from_markup(
            f"[{style}]$glyph[/] [b]$title[/b] [$ph-muted]$subtitle[/] [{style}]$failure[/]",
            glyph=glyph,
            title=card.title or card.name,
            subtitle=card.subtitle,
            failure=card.failure_kind,
        )

    def _body(self) -> str:
        card = self.item.tool
        return self.item.text if card is None else card.body

    async def refresh_card(self) -> bool:
        """Redraw if anything changed, and return whether it did.

        The return value lets a subclass extend the card and keep the memo.
        `Static.update` always runs the layout again, so a row that redrew
        unconditionally would force a full arrange on every dirty frame. This
        memo exists to remove that cost, and `sync` visits every row.
        """
        current = self._snapshot()
        changed = current != self._shown
        if changed:
            self._shown = current
            card = self.item.tool
            self.set_class(bool(card and card.settled), "-settled")
            self.set_class(bool(card and card.is_error), "-error")
            try:
                self.query_one(".tool-header", Static).update(self._header())
                self.query_one(".tool-body", Static).update(Content(self._body()))
            except NoMatches:
                # The same gap as in `TranscriptRow.refresh_text`. `compose`
                # draws from `self.item` when it runs.
                return changed
        await self._sync_dispatches()
        return changed

    async def _sync_dispatches(self) -> None:
        card = self.item.tool
        if card is None or not card.dispatches:
            return
        try:
            collapsible = self.query_one(Collapsible)
        except NoMatches:
            return
        collapsible.display = True
        collapsible.title = count_of(len(card.dispatches), "governed call")
        for dispatch in card.dispatches:
            state = (dispatch.settled, dispatch.is_error)
            known = self._dispatch_rows.get(dispatch.call_id)
            if known is None:
                row = Static(self._dispatch_line(dispatch), classes="dispatch")
                self._dispatch_rows[dispatch.call_id] = (row, state)
                await self._dispatch_box.mount(row)
            elif known[1] != state:
                known[0].update(self._dispatch_line(dispatch))
                self._dispatch_rows[dispatch.call_id] = (known[0], state)

    @staticmethod
    def _dispatch_line(dispatch: ToolCard) -> Content:
        glyph, style = _status_markup(dispatch)
        return Content.from_markup(
            f"[{style}]$glyph[/] $name [$ph-muted]$args[/]",
            glyph=glyph,
            name=dispatch.name,
            args=dispatch.arguments,
        )


def _cell_facts(details: dict[str, JsonValue]) -> str:
    """Return the `CellDetails` line, with only the facts that are true.

    The details are read as a plain mapping, and the model is not imported. The
    payload belongs to ph-rlm, and `ph_app` does not import the bundle. A field
    that this build does not know is ignored. That lets a tool enrich its own
    card without the transcript learning its schema.

    The dispatch *count* is deliberately absent. The collapsible below already
    reports it, from the `tool/code-dispatch-start` fold that owns those rows.
    Two projections of one number in one widget can disagree, which A11 forbids.
    """
    facts: list[str] = []
    attachments = as_int(details.get("attachments"))
    if attachments:
        facts.append(count_of(attachments, "attachment"))
    if details.get("truncated"):
        # The count belongs to the body, which ends with the total of the run:
        # one number, one place (A11).
        facts.append("output truncated")
    if details.get("reset"):
        facts.append("kernel restarted — namespace empty")
    return " · ".join(facts)


class CodeCellWidget(ToolCardWidget):
    """A Code Mode cell: the program, what it printed, and every call it made.

    The card kind `terminal` has its own widget for one reason: the *program*
    is the interesting half. The card of every other tool is a header and a
    body, because its input is a path or a query that fits on the header line.
    The input of a cell is the code that the model wrote. A transcript that
    showed only its first line would hide what a reader is trying to follow.

    Everything drawn here comes from the settled record: `ToolCallView.body` for
    the program and `tool/result.meta` for the facts line. A replayed cell is
    therefore the cell that ran (P2-01).
    """

    DEFAULT_CSS = """
    CodeCellWidget > .cell-program {
        background: $ph-panel; color: $ph-foreground; padding: 0 1; height: auto;
    }
    CodeCellWidget > .cell-facts { color: $ph-muted; }
    """

    def _rows_after_header(self) -> ComposeResult:
        yield Static(Content(self._program()), classes="cell-program")

    def _rows_after_body(self) -> ComposeResult:
        yield Static(Content(""), classes="cell-facts")

    def _program(self) -> str:
        card = self.item.tool
        return "" if card is None else card.input_text

    def _facts(self) -> str:
        card = self.item.tool
        return "" if card is None else _cell_facts(card.details)

    def _snapshot(self) -> tuple[object, ...]:
        return (*super()._snapshot(), self._program(), self._facts())

    async def refresh_card(self) -> bool:
        changed = await super().refresh_card()
        if changed:
            # `self._shown` is the tuple that `super()` just stored, so the
            # program and the facts are read back and not computed again.
            program, facts = self._shown[-2:] if self._shown else ("", "")
            self.query_one(".cell-program", Static).update(Content(str(program)))
            widget = self.query_one(".cell-facts", Static)
            widget.display = bool(facts)
            widget.update(Content(str(facts)))
        return changed


RowWidget: TypeAlias = ToolCardWidget | StreamingMessage | TranscriptRow
"""One transcript row, in whatever shape the item asked for.

This is not Textual's `Widget`. The three classes have different bases
(`Vertical`, `Markdown`) and share exactly one thing that this view needs:
`item`, the `ChatItem` that they were built from. `_row_for_seq` reads
`row.item.seq` and `seq_in_view` reads it again. A base that only promises
"mountable" would not typecheck the navigation that this widget exists to
provide. `CodeCellWidget` is covered as a subclass of `ToolCardWidget`.
"""


class TranscriptView(GutterScroll, VerticalScroll):
    """Renders a list of rows: it mounts new rows and updates changed ones.

    The view receives rows that are already filtered. `TuiState.visible_items`
    owns which toggles hide what, so the view has no opinion about the meaning
    of the transcript.

    Column 0 is the gutter (`widgets.gutter`). It holds the scrollbar, which is
    always shown because the bars of the rows are drawn in it. The track has
    the color of the background, and the thumb has the color of the headings.
    That is the color of the status line and of the side panel, so the thumb
    follows the theme with them. Rows start in column 1.
    """

    DEFAULT_CSS = """
    TranscriptView {
        height: 1fr; background: $ph-background; padding: 1 0 0 1;
        overflow-y: scroll; scrollbar-size-vertical: 1;
        scrollbar-background: $ph-background; scrollbar-color: $ph-accent;
        scrollbar-background-hover: $ph-background; scrollbar-color-hover: $ph-accent;
        scrollbar-background-active: $ph-background; scrollbar-color-active: $ph-accent;
    }
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self._rows: dict[str, RowWidget] = {}
        self._followed = False
        self._mounting = anyio.Lock()
        """Serializes `sync`, because Textual refuses a mount while one is pending.

        Two draws can overlap. `_draw` clears its coalescing timer before it
        awaits, so a change that arrives during a draw schedules the next one.
        The second `sync` then calls `mount()` on a node whose previous child is
        still mounting, which raises `MountError: Can't mount widget(s) before
        … is mounted`. It appeared as an unrelated test that failed under
        coverage, where everything is slow enough for the windows to overlap.

        The lock belongs here and not at the caller. The ordering constraint
        belongs to this widget, and a caller that had to know about it would
        eventually forget."""

    async def sync(self, items: list[ChatItem]) -> None:
        """Bring the view in line with `items`. Idempotent and cheap when nothing changed."""
        async with self._mounting:
            # **A queued draw may arrive after this view is gone.** Because
            # `sync` is serialized, a second one can begin later than it used
            # to: after the app started unmounting. That is the same hazard that
            # `on_unmount` stops the timers for. Mounting into a detached node
            # raises, so a draw that arrives too late renders nothing.
            if self.is_mounted:
                await self._sync(items)

    async def _sync(self, items: list[ChatItem]) -> None:
        for item in items:
            widget = self._rows.get(item.key)
            if widget is None:
                widget = self._build(item)
                self._rows[item.key] = widget
                await self.mount(widget)
            if not widget.is_mounted:
                # The widget was mounted a moment ago and has been detached
                # since, so the view is coming apart around this loop. Its
                # content would go into a node that nothing renders, and
                # `Markdown.append` refuses it.
                continue
            await self._update(widget, item)
        # The gutter draws the bars of the rows. A row that settled or failed in
        # this pass changed the color of its bar and not its size. Nothing else
        # would repaint the gutter for that change. The refresh covers one
        # column.
        self.vertical_scrollbar.refresh()
        # Stick to the bottom while rows arrive, and release when the reader
        # scrolls up. Textual's `anchor()` does that from one call. It is
        # engaged only when there is something to scroll: anchoring an underfull
        # pane scrolls it to a *negative* offset, and every row renders pushed
        # to the bottom.
        #
        # The anchor is armed again only from the bottom. Textual releases the
        # anchor when the reader scrolls away. An unconditional re-arm here
        # undid that on the next dirty frame. It also meant that a jump to a
        # record (`scroll_to_seq`) lasted only until the next event.
        if (
            self.max_scroll_y > 0
            and not self.is_anchored
            and (not self._followed or self.scroll_offset.y >= self.max_scroll_y)
        ):
            self._followed = True
            self.anchor()

    def _build(self, item: ChatItem) -> RowWidget:
        if item.role == "tool":
            kind = item.tool.card if item.tool is not None else "generic"
            return CodeCellWidget(item) if kind == "terminal" else ToolCardWidget(item)
        if item.role in MARKDOWN_ROLES:
            return StreamingMessage(item)
        return TranscriptRow(item)

    async def _update(self, widget: RowWidget, item: ChatItem) -> None:
        if isinstance(widget, ToolCardWidget):
            await widget.refresh_card()
        elif isinstance(widget, StreamingMessage):
            # A settled row is finalized on its first sync. That makes a
            # replayed message look identical to one that streamed in.
            if item.streaming:
                await widget.write_tail(item.text)
            else:
                await widget.finalize(item.text)
        else:
            widget.refresh_text()

    async def rebuild(self, items: list[ChatItem]) -> None:
        """Drop every row and render from scratch. A toggle needs this."""
        await self.remove_children()
        self._rows.clear()
        await self.sync(items)

    # ------------------------------------------------------ cross-navigation --
    # The join with the auditor's view (P4-17): `ChatItem.seq` and
    # `TrajectoryRecord.source_seq` are the same log position, stored on both
    # sides since P3-24. These two methods are all that was needed to read it.

    def scroll_to_seq(self, seq: int) -> bool:
        """Bring the row for a log seq into view. Return `False` if none is shown.

        The anchor is released first. A view that is stuck to the bottom would
        scroll straight back, and the jump would appear to do nothing.
        """
        widget = self._row_for_seq(seq)
        if widget is None:
            return False
        self.release_anchor()
        self.scroll_to_widget(widget, top=True, animate=False, immediate=True)
        return True

    def seq_in_view(self) -> int | None:
        """Return the seq of the topmost row on screen. Return `None` when no row
        is on screen, or when that row has no position in the log.

        This gives "where the reader is" without inventing a focus for each row.
        Rows are widgets in a scroll, not a table with a cursor. So the correct
        answer to "which row am I looking at" is the first row that is still
        visible.
        """
        top = self.content_region.y
        for widget in self._rows.values():
            region = widget.region
            if region.height and region.bottom > top:
                # `ChatItem.seq` is `-1` for a row that the log has no position
                # for. That is not a position that a screen can open at.
                seq = int(widget.item.seq)
                return seq if seq >= 0 else None
        return None

    def _row_for_seq(self, seq: int) -> RowWidget | None:
        """Return the row for `seq`, or the nearest row before it.

        `index_at_or_before` owns the "nearest" rule and the reason for it, so
        this side and the trajectory side cannot drift apart. `_rows` is in
        transcript order, which is log order.
        """
        rows = list(self._rows.values())
        index = index_at_or_before((row.item.seq for row in rows), seq)
        return rows[index] if index >= 0 else None


def _slug(key: str) -> str:
    return "".join(char if char.isalnum() or char == "-" else "-" for char in key)
