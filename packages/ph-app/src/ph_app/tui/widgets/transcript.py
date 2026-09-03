"""The transcript: rows that render from `TuiState` and nothing else.

Two disciplines are enforced here rather than remembered (P2-06).

**Never build markup with an f-string.** User text, tool output and provider
errors all contain `[` — a path like `foo[0]`, a Python list, a Rich tag someone
typed on purpose. `Content(text)` renders it literally; an f-string into a
markup parser makes the model's output a formatting language, and at worst
swallows it. Where styling *is* wanted, `Content.from_markup` takes `$variables`
so the substituted value can never be parsed as markup.

**Stream by appending, not by reparsing.** `MarkdownStream.write(fragment)`
appends to a live Markdown widget; re-`update()`ing the accumulated text on every
delta re-parses the whole message per token, which is quadratic and visibly
janky by the second paragraph. The view tracks how much of each row it has
already written and sends only the tail.

The same discipline, one layer up: a settled row **remembers what it last drew**
and skips `update()` when nothing changed. `sync` visits every visible row on
every dirty frame, and `Static.update` always re-lays-out — so without the memo
a 200-row transcript re-rendered 200 widgets per frame while streaming.

@module ph_app.tui.widgets.transcript
"""

from __future__ import annotations

from contextlib import suppress
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Collapsible, Markdown, Static
from textual.widgets.markdown import MarkdownStream

from ph.text import count_of

from ...wire import index_at_or_before
from ..state import ChatItem, ToolCard

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

Not a display preference — a consistency requirement. The model writes Markdown,
so its rows render it; a *streamed* row that rendered Markdown while a
*replayed* one showed raw asterisks would make a resumed session look different
from the session the person left, which is the same failure the P2-01 gate
guards against, one layer up.

The user's own text is deliberately absent: what someone typed is shown as they
typed it.
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
    """The glyph and its style for a call's state — one rule for cards and dispatches."""
    glyph = "…" if not card.settled else ("✗" if card.is_error else "✓")
    return glyph, ("$ph-tool-error" if card.is_error else "$ph-tool-success")


class TranscriptRow(Vertical):
    """A settled row: a styled label over its body."""

    DEFAULT_CSS = """
    TranscriptRow { height: auto; margin: 0 1 1 1; }
    TranscriptRow > .row-label { color: $ph-muted; }
    TranscriptRow > .row-body { height: auto; }
    TranscriptRow.-error > .row-body { color: $ph-error; }
    TranscriptRow.-thinking > .row-body { color: $ph-thinking-text; }
    TranscriptRow.-context > .row-body { color: $ph-muted; }
    TranscriptRow.-compaction { border-left: outer $ph-warning; padding-left: 1; }
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
        # `$label` is substituted, never parsed: a role name could not introduce
        # markup here, but the same rule everywhere is what makes it reliable.
        yield Static(Content.from_markup(f"[{style}]$label[/]", label=label), classes="row-label")
        yield Static(Content(self.item.text), classes="row-body")

    def on_mount(self) -> None:
        self._shown = (self.item.text, self.item.shadowed)
        # Dimmed, not removed: the person read it, the model no longer has it.
        self.set_class(self.item.shadowed, "-shadowed")

    def refresh_text(self) -> None:
        current = (self.item.text, self.item.shadowed)
        if current == self._shown:
            return
        self._shown = current
        self.set_class(self.item.shadowed, "-shadowed")
        # Mounted, not yet composed — a row that was added and changed inside one
        # frame, which a `/command` that logs `run` and `done` around an await
        # does deterministically. `compose` reads `self.item` when it runs, so
        # the latest text is what it will draw; there is nothing to update yet.
        # The widget owns its children's lifetime, so the widget is what says so.
        with suppress(NoMatches):
            self.query_one(".row-body", Static).update(Content(self.item.text))


class StreamingMessage(Markdown):
    """An assistant or thinking row that grows a fragment at a time."""

    DEFAULT_CSS = """
    StreamingMessage { height: auto; margin: 0 1 1 1; }
    /* The model's answer carries no label — it is the default voice of the
       transcript. Reasoning does need marking, since it can be toggled off and
       a reader must be able to tell which they are looking at. */
    StreamingMessage.-thinking {
        color: $ph-thinking-text; border-left: outer $ph-thinking-text; padding-left: 1;
    }
    StreamingMessage.-streaming MarkdownFence { overflow-x: hidden; scrollbar-size-horizontal: 0; }
    StreamingMessage.-finalized MarkdownFence { overflow-x: auto; scrollbar-size-horizontal: 1; }
    """

    def __init__(self, item: ChatItem) -> None:
        super().__init__("", id=f"row-{_slug(item.key)}")
        self.item = item
        self._stream: MarkdownStream | None = None
        self._written = 0
        self.add_class(f"-{item.role}")
        self.add_class("-streaming")

    async def write_tail(self, text: str) -> None:
        """Append whatever part of `text` has not been written yet."""
        if len(text) <= self._written:
            return
        fragment = text[self._written :]
        self._written = len(text)
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


class ToolCardWidget(Vertical):
    """One tool call: a header line, its body, and any Code Mode sub-dispatches.

    Dispatch rows are mounted as they arrive, keyed by call id, rather than
    composed once. Live, the card mounts on the tick after `tool/call` — before
    any `tool/code-dispatch-start` exists — so compose-time content would show
    the "governed calls" section on replay and never live. Same fold, same
    picture, is the P2-01 promise at the widget level.
    """

    DEFAULT_CSS = """
    ToolCardWidget {
        height: auto; margin: 0 1 1 1; border-left: outer $ph-border; padding-left: 1;
    }
    ToolCardWidget.-error { border-left: outer $ph-tool-error; }
    ToolCardWidget.-settled { border-left: outer $ph-tool-success; }
    ToolCardWidget > .tool-body { color: $ph-muted; height: auto; }
    ToolCardWidget .dispatch { color: $ph-muted; }
    """

    def __init__(self, item: ChatItem) -> None:
        super().__init__(id=f"row-{_slug(item.key)}")
        self.item = item
        self._shown: tuple[Any, ...] | None = None
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

    def _snapshot(self) -> tuple[Any, ...]:
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
        """Redraw if anything changed; answer whether it did.

        The answer is what lets a subclass extend the card without defeating the
        memo: `Static.update` always re-lays-out, so a row that redrew
        unconditionally would force a full arrange on every dirty frame — which
        is the cost this memo exists to remove, and `sync` visits every row.
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
                # Same gap as `TranscriptRow.refresh_text`; compose draws from
                # `self.item` when it runs.
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


def _cell_facts(details: dict[str, Any]) -> str:
    """The `IpythonToolDetails` line — only the facts that are true.

    Read as a plain mapping rather than by importing the model: the payload is
    ph-rlm's, and ph-app does not depend on the bundle. A field this build does not
    know is ignored, which is what lets a tool enrich its own card without the
    transcript learning its schema.

    The dispatch *count* is deliberately absent: the collapsible below already reports
    it, from the `tool/code-dispatch-start` fold that owns those rows. Two projections
    of one number in one widget can disagree, which is what A11 forbids.
    """
    facts: list[str] = []
    attachments = int(details.get("attachments") or 0)
    if attachments:
        facts.append(count_of(attachments, "attachment"))
    if details.get("truncated"):
        facts.append("output truncated")
    if details.get("reset"):
        facts.append("kernel restarted — namespace empty")
    return " · ".join(facts)


class CodeCellWidget(ToolCardWidget):
    """A Code Mode cell: the program, what it printed, and every call it made.

    The card kind `terminal` earns its own widget for one reason: the *program*
    is the interesting half. Every other tool's card is a header and a body,
    because its input is a path or a query that fits on the header line; a cell's
    input is the code the model wrote, and a transcript that showed only its
    first line would hide the thing a reader is trying to follow.

    Everything drawn here comes from the settled record — `ToolCallView.body` for
    the program, `tool/result.meta` for the facts line — so a replayed cell is
    the cell that ran (P2-01).
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

    def _snapshot(self) -> tuple[Any, ...]:
        return (*super()._snapshot(), self._program(), self._facts())

    async def refresh_card(self) -> bool:
        changed = await super().refresh_card()
        if changed:
            # `self._shown` is the tuple `super()` just stored, so the program
            # and the facts are read back rather than recomputed.
            program, facts = self._shown[-2:] if self._shown else ("", "")
            self.query_one(".cell-program", Static).update(Content(str(program)))
            widget = self.query_one(".cell-facts", Static)
            widget.display = bool(facts)
            widget.update(Content(str(facts)))
        return changed


class TranscriptView(VerticalScroll):
    """Renders a list of rows, mounting new ones and updating changed ones.

    Receives rows already filtered — `TuiState.visible_items` owns which toggles
    hide what — so the view has no opinion about the transcript's meaning.
    """

    DEFAULT_CSS = """
    TranscriptView { height: 1fr; background: $ph-background; padding: 1 0 0 0; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._rows: dict[str, Any] = {}
        self._followed = False

    async def sync(self, items: list[ChatItem]) -> None:
        """Bring the view in line with `items`. Idempotent and cheap when nothing changed."""
        for item in items:
            widget = self._rows.get(item.key)
            if widget is None:
                widget = self._build(item)
                self._rows[item.key] = widget
                await self.mount(widget)
            await self._update(widget, item)
        # Stick to the bottom while rows arrive; release when the reader scrolls
        # up. Textual's `anchor()` does that from one call — but engaged only
        # once there is something to scroll: anchoring an underfull pane scrolls
        # it to a *negative* offset and every row renders pushed to the bottom.
        #
        # Re-armed only from the bottom. Textual releases the anchor when the
        # reader scrolls away, and an unconditional re-arm here undid that on
        # the very next dirty frame — which also meant a jump to a record
        # (`scroll_to_seq`) survived exactly until the next event.
        if (
            self.max_scroll_y > 0
            and not self.is_anchored
            and (not self._followed or self.scroll_offset.y >= self.max_scroll_y)
        ):
            self._followed = True
            self.anchor()

    def _build(self, item: ChatItem) -> Any:
        if item.role == "tool":
            kind = item.tool.card if item.tool is not None else "generic"
            return CodeCellWidget(item) if kind == "terminal" else ToolCardWidget(item)
        if item.role in MARKDOWN_ROLES:
            return StreamingMessage(item)
        return TranscriptRow(item)

    async def _update(self, widget: Any, item: ChatItem) -> None:
        if isinstance(widget, ToolCardWidget):
            await widget.refresh_card()
        elif isinstance(widget, StreamingMessage):
            # A settled row is finalized on its first sync, which is what makes
            # a replayed message look identical to one that streamed in.
            if item.streaming:
                await widget.write_tail(item.text)
            else:
                await widget.finalize(item.text)
        elif isinstance(widget, TranscriptRow):
            widget.refresh_text()

    async def rebuild(self, items: list[ChatItem]) -> None:
        """Drop every row and render from scratch — what a toggle needs."""
        await self.remove_children()
        self._rows.clear()
        await self.sync(items)

    # ------------------------------------------------------ cross-navigation --
    # The join with the auditor's view (P4-17): `ChatItem.seq` and
    # `TrajectoryRecord.source_seq` are the same log position, stored on both
    # sides since P3-24. These two methods are all it took to read it.

    def scroll_to_seq(self, seq: int) -> bool:
        """Bring the row for a log seq into view. `False` if none is shown.

        The anchor is released first: a view stuck to the bottom would scroll
        straight back, which is the whole of "the jump did nothing".
        """
        widget = self._row_for_seq(seq)
        if widget is None:
            return False
        self.release_anchor()
        self.scroll_to_widget(widget, top=True, animate=False, immediate=True)
        return True

    def seq_in_view(self) -> int:
        """The seq of the topmost row on screen, or `-1` when nothing is.

        "Where the reader is" without inventing per-row focus: rows are widgets
        in a scroll, not a table with a cursor, so the honest answer to "which
        row am I looking at" is the first one still visible.
        """
        top = self.content_region.y
        for widget in self._rows.values():
            region = widget.region
            if region.height and region.bottom > top:
                return int(widget.item.seq)
        return -1

    def _row_for_seq(self, seq: int) -> Any:
        """The row for `seq`, or the nearest one before it.

        `index_at_or_before` owns the "nearest" rule and why it is nearest, so
        this side and the trajectory's cannot drift apart. `_rows` is in
        transcript order, which is log order.
        """
        rows = list(self._rows.values())
        index = index_at_or_before((row.item.seq for row in rows), seq)
        return rows[index] if index >= 0 else None


def _slug(key: str) -> str:
    return "".join(char if char.isalnum() or char == "-" else "-" for char in key)
