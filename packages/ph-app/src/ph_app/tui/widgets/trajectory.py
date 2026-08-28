"""The auditor's view: a table over the records, and what you can do from one.

The projection (`ph_app.tui.trajectory`) says what happened; this says how to
read it. Three things it offers that the transcript cannot, and each is why the
view exists rather than being a screen inside the chat:

* **It reads any log.** The records come from a `Session`, which a *stored* log
  becomes through `read_session` with nothing mounted — no agent, no provider,
  no answerers. The crashed run and the subagent's log are the cases an
  auditor's view is for, and both are out of reach of a view that can only show
  the session you are in.
* **It searches records *and* sources.** "Every snapshot this plugin injected"
  is a question about attribution, which the transcript does not carry.
* **It forks at a record.** Only at a closed turn (A6), so the table *marks*
  which rows are targets rather than letting a person aim anywhere and be
  refused — a rejection after the fact teaches nothing about where to aim.

@module ph_app.tui.widgets.trajectory
"""

from __future__ import annotations

from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.timer import Timer
from textual.widgets import DataTable, Input, Static

from ..trajectory import TrajectoryRecord
from ..wire import matches_terms

__all__ = ["FORK_MARK", "TrajectoryPanel", "matches", "search_index"]

FILTER_DEBOUNCE = 0.15
"""Seconds a keystroke waits before the table rebuilds."""

FORK_MARK = "⑂"
"""Marks a record a fork may aim at. A6 allows a closed turn and nothing else,
so this is the table's way of saying where the action is available."""


def search_index(record: TrajectoryRecord) -> str:
    """Everything one record is searchable by, lowercased once.

    Text **and** source: a tool call is found by its name, by its arguments and
    by the plugin that produced it, because "which records came from the context
    loader" is the question the transcript cannot answer. Built per record and
    reused across queries — the index is what makes typing feel incremental.
    """
    # `title` is absent deliberately: it is the table's own column and, for
    # every message and tool record, the same text as the source label — one
    # string indexed twice buys nothing and doubles the term.
    return " ".join((record.kind, record.summary, record.detail, record.source.label())).lower()


def matches(index: str, query: str) -> bool:
    """`matches_terms`, named for this call site.

    No ranking, on purpose: an auditor filters to a handful and reads them,
    rather than being handed someone's guess at relevance. The predicate itself
    is the TUI's one definition of filtering, shared with the choice picker."""
    return matches_terms(index, query)


class TrajectoryPanel(Vertical):
    """The table, its filter, and the details of the selected record."""

    DEFAULT_CSS = """
    TrajectoryPanel { height: 1fr; }
    TrajectoryPanel > #trajectory-filter { border: none; height: 3; background: $ph-panel; }
    TrajectoryPanel > #trajectory-body { height: 1fr; }
    TrajectoryPanel #trajectory-table { width: 3fr; height: 1fr; }
    TrajectoryPanel #trajectory-details {
        width: 2fr; height: 1fr; background: $ph-panel; padding: 0 1;
        border-left: vkey $ph-border;
    }
    TrajectoryPanel .details-title { color: $ph-accent; }
    TrajectoryPanel .details-body { color: $ph-foreground; height: auto; }
    """

    def __init__(self, records: list[TrajectoryRecord], **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.records = records
        self._index = [search_index(record) for record in records]
        self.visible_records: list[TrajectoryRecord] = list(records)
        self._table: DataTable[str] | None = None
        self._pending: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Input(placeholder="filter records…", id="trajectory-filter")
        with Horizontal(id="trajectory-body"):
            yield DataTable(id="trajectory-table", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="trajectory-details"):
                yield Static(Content(""), classes="details-title")
                yield Static(Content(""), classes="details-body")

    def on_mount(self) -> None:
        table = self._table = self.query_one("#trajectory-table", DataTable)
        table.add_columns("#", "kind", "turn", "title", "summary")
        self.refresh_rows()

    def focus_filter(self) -> None:
        """Hand focus to the filter. Paired with `focus_table`, so the app does
        not reach through this widget by id for one half of the pair."""
        self.query_one("#trajectory-filter", Input).focus()

    def focus_table(self) -> None:
        """Give the table focus, so the view's single-key actions reach the app.

        The filter is opt-in (`/`) for exactly this reason: an input holding
        focus by default would swallow `f` as a typed character, and the fork
        key would silently do nothing.
        """
        if self._table is not None:
            self._table.focus()

    # ---------------------------------------------------------------- rows --

    def refresh_rows(self, query: str = "") -> None:
        """Rebuild the table for a filter. The record list is the source of both
        the rows and the selection, so they cannot disagree."""
        table = self._table
        if table is None:
            return
        self.visible_records = [
            record
            for record, index in zip(self.records, self._index, strict=True)
            if not query or matches(index, query)
        ]
        table.clear()
        for record in self.visible_records:
            table.add_row(
                # The fork mark is the table's way of saying where the action is
                # available (A6); a plain string, like every other cell, so
                # nothing here can be parsed as markup.
                f"{FORK_MARK if record.fork_point else ' '}{record.index}",
                record.kind,
                str(record.turn),
                # Plain strings: `DataTable` renders a cell literally, so a
                # summary containing `[` needs no escaping and cannot become
                # markup — the same rule the transcript enforces with `Content`.
                record.title,
                record.summary,
                key=str(record.index),
            )
        # Only when the filter emptied the table: otherwise the first `add_row`
        # after a `clear()` moves the cursor and the highlight event draws them.
        if not self.visible_records:
            self.show_details(None)

    def selected(self) -> TrajectoryRecord | None:
        """The record under the cursor, or `None` when the filter emptied the
        table."""
        table = self._table
        if table is None or not self.visible_records:
            return None
        row = min(table.cursor_row, len(self.visible_records) - 1)
        return self.visible_records[row]

    # ------------------------------------------------------------- details --

    def show_details(self, record: TrajectoryRecord | None) -> None:
        title = self.query_one(".details-title", Static)
        body = self.query_one(".details-body", Static)
        if record is None:
            title.update(Content("no record"))
            body.update(Content(""))
            return
        title.update(
            Content.from_markup(
                "[b]#$index $kind[/b] [$ph-muted]$source[/]",
                index=str(record.index),
                kind=record.kind,
                source=record.source.label(),
            )
        )
        body.update(Content("\n\n".join(self._sections(record))))

    def _sections(self, record: TrajectoryRecord) -> list[str]:
        """What the panel shows, in the order an auditor reads it."""
        sections = [record.detail or record.summary]
        if record.timing is not None and record.timing.total_ms is not None:
            timing = record.timing
            parts = [f"{timing.total_ms} ms"]
            if timing.time_to_first_token_ms is not None:
                parts.append(f"{timing.time_to_first_token_ms} ms to first token")
            if timing.decode_tokens_per_second is not None:
                parts.append(f"{timing.decode_tokens_per_second} tok/s decode")
            sections.append("— " + " · ".join(parts))
        if record.tools:
            names = ", ".join(record.tools)
            # The catalog as it was *at call time*: a tool registered later must
            # not appear in the record of a call that could not have used it.
            sections.append(f"tools at call time ({len(record.tools)}): {names}")
        if record.replaced:
            sections.append(f"replaced:\n{record.replaced}")
        if record.fork_point:
            sections.append(f"{FORK_MARK} a fork may start here (closed turn)")
        return sections

    # -------------------------------------------------------------- events --

    def on_input_changed(self, event: Input.Changed) -> None:
        """Coalesce a burst of keystrokes into one rebuild.

        `DataTable` virtualizes *rendering*, not row construction: a refill
        measures every cell of every row, so a thousand-record log costs ~150 ms
        per keystroke and typing four characters cost 625 ms of it. Row surgery
        is worse — removing the rows that drop out reindexes the table each time
        — so the fix is the frequency, not the primitive.
        """
        if self._pending is not None:
            self._pending.stop()
        self._pending = self.set_timer(FILTER_DEBOUNCE, lambda: self.refresh_rows(event.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.show_details(self.selected())
