"""The auditor's view: a table over the records, and what a user can do from a record.

The projection (`ph_app.tui.trajectory`) says what happened, and this module
says how to read it. It offers three things that the transcript cannot, and
each one is a reason that the view exists. `TrajectoryScreen` then makes the
view reachable from both the standalone entry point and the chat, over the same
records.

* **It reads any log.** The records come from a `Session`. A *stored* log
  becomes one through `read_session` with nothing mounted: no agent, no
  provider, no answerers. The crashed run and the log of a subagent are the
  cases that an auditor's view is for. A view that can show only the session
  that you are in reaches neither.
* **It searches records *and* sources.** "Every snapshot that this plugin
  injected" is a question about attribution, which the transcript does not
  carry.
* **It forks at a record.** A fork is allowed only at a closed turn (A6). So
  the table *marks* which rows are targets, and a user does not aim anywhere
  and get refused. A rejection after the fact teaches nothing about where to
  aim.

@module ph_app.tui.widgets.trajectory
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.content import Content
from textual.timer import Timer
from textual.widgets import DataTable, Input, Static

from ph.selectors import Selector, SelectorError, matches_any, parse_all

from ...wire import index_at_or_before, matches_terms, split_terms
from ..trajectory import TrajectoryRecord

__all__ = ["FORK_MARK", "TYPE_TERM", "Query", "TrajectoryPanel", "search_index"]

FILTER_DEBOUNCE = 0.15
"""Seconds a keystroke waits before the table rebuilds."""

FORK_MARK = "⑂"
"""Marks a record that a fork may aim at. A6 allows a closed turn and nothing
else, so this mark is how the table says where the action is available."""


def search_index(record: TrajectoryRecord) -> str:
    """Everything that one record is searchable by, lowercased one time.

    The index holds text **and** source. A tool call is found by its name, by
    its arguments and by the plugin that produced it, because "which records
    came from the context loader" is the question that the transcript cannot
    answer. The index is built for each record and reused across queries. That
    reuse is what makes typing feel incremental.
    """
    # `title` is absent deliberately. It is the table's own column, and for
    # every message and tool record it is the same text as the source label.
    # One string indexed twice gains nothing and doubles the term.
    #
    # `type` is absent for the opposite reason. It is matched *precisely*, by
    # `type:` below, and not by substring. In this index it would make `tool`
    # catch `tools/`-shaped prose, and a namespace could be selected by
    # accident.
    return " ".join((record.kind, record.summary, record.detail, record.source.label())).lower()


TYPE_TERM = "type:"
"""The one query prefix that means "not free text".

It is spelled one time and interpolated into the placeholder below. The syntax
that a user is offered and the syntax that the predicate implements therefore
cannot drift.
"""


@dataclass(frozen=True, slots=True)
class Query:
    """A filter query, tokenized and parsed **one time for each rebuild**.

    The query is compiled here and not inside the per-record predicate.
    `refresh_rows` calls the predicate one time for each record. Parsing inside
    it lexed the query again and parsed every selector again, for every row on
    every keystroke. It also made the *plain free-text* path, which is the
    common one, more expensive than it was before `type:` existed, because the
    tag scan ran whether or not anyone had typed a tag.

    `bad` is how a malformed or foreign `type:` term reaches the caller without
    an exception for each row. Such a query matches nothing, because this
    compiles while a user is still typing: `type:w` on the way to
    `type:workspace` must not throw. The refusal that explains itself belongs
    on a command line, where the input is complete.
    """

    selectors: tuple[Selector, ...] = ()
    free: str = ""
    bad: bool = False

    @classmethod
    def compile(cls, query: str) -> Query:
        tagged, free = split_terms(query, TYPE_TERM)
        try:
            return cls(selectors=tuple(parse_all(tagged, vocabulary="log")), free=free)
        except SelectorError:
            return cls(bad=True)

    def matches(self, record: TrajectoryRecord, index: str) -> bool:
        """Free text over everything that a row shows, `type:` over its namespace.

        There are two predicates because there are two questions. *"Which
        records mention `retry`"* is a substring search. It uses
        `matches_terms`, the one definition of filtering in the TUI, which the
        choice picker shares. *"Which records are `workspace/*`"* is not a
        substring search: asking for a namespace by substring is how `tool`
        comes to select `tools/` (P6-33).

        Several `type:` terms are a **union**, which matches `--type` on the
        command line. `type:workspace type:turn` means either and not both. A
        record cannot be in two namespaces, so an intersection would always be
        empty. The type half is then ANDed with the free text, which is what a
        second term has always meant here.
        """
        if self.bad:
            return False
        return matches_any(record.type, self.selectors) and matches_terms(index, self.free)


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

    def __init__(self, records: list[TrajectoryRecord], **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self.records = records
        self._index = [search_index(record) for record in records]
        self.visible_records: list[TrajectoryRecord] = list(records)
        self._table: DataTable[str] | None = None
        self._pending: Timer | None = None

    def compose(self) -> ComposeResult:
        yield Input(placeholder=f"filter records… or {TYPE_TERM}workspace", id="trajectory-filter")
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
        """Give focus to the filter. This is paired with `focus_table`, so the
        app does not reach through this widget by id for one half of the
        pair."""
        self.query_one("#trajectory-filter", Input).focus()

    def focus_table(self) -> None:
        """Give the table focus, so the single-key actions of the view reach the app.

        The filter is opt-in (`/`) for exactly this reason. An input that held
        focus by default would consume `f` as a typed character, and the fork
        key would silently do nothing.
        """
        if self._table is not None:
            self._table.focus()

    @property
    def filtering(self) -> bool:
        """Whether the filter holds focus.

        This is asked and not remembered. `escape` means "leave the filter"
        while the filter has focus and "leave the screen" otherwise. A flag
        that this widget set would disagree with the focus after a click moved
        it.
        """
        return self.query_one("#trajectory-filter", Input).has_focus

    # ---------------------------------------------------------------- rows --

    def refresh_rows(self, query: str = "") -> None:
        """Rebuild the table for a filter. The record list is the source of both
        the rows and the selection, so the two cannot disagree."""
        table = self._table
        if table is None:
            return
        compiled = Query.compile(query) if query else None
        self.visible_records = [
            record
            for record, index in zip(self.records, self._index, strict=True)
            if compiled is None or compiled.matches(record, index)
        ]
        table.clear()
        for record in self.visible_records:
            table.add_row(
                # The fork mark is how the table says where the action is
                # available (A6). It is a plain string, like every other cell,
                # so nothing here can be parsed as markup.
                f"{FORK_MARK if record.fork_point else ' '}{record.index}",
                record.kind,
                str(record.turn),
                # Plain strings: `DataTable` renders a cell literally. A summary
                # that contains `[` needs no escaping and cannot become markup.
                # The transcript enforces the same rule with `Content`.
                record.title,
                record.summary,
                key=str(record.index),
            )
        # Only when the filter emptied the table. Otherwise the first `add_row`
        # after a `clear()` moves the cursor, and the highlight event draws the
        # details.
        if not self.visible_records:
            self.show_details(None)

    def select_seq(self, seq: int) -> bool:
        """Put the cursor on the record for a log seq, or on the nearest record before it.

        `index_at_or_before` owns the "nearest" rule. The transcript shares it,
        so the two sides of the join cannot come to disagree about which row
        the rule means. Moving the cursor posts `RowHighlighted`, which draws
        the details panel, so this method does not draw it as well.
        """
        table = self._table
        if table is None:
            return False
        row = index_at_or_before((record.source_seq for record in self.visible_records), seq)
        if row is None:
            return False
        table.move_cursor(row=row)
        return True

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
            # The catalog as it was *at call time*. A tool that was registered
            # later must not appear in the record of a call that could not have
            # used it.
            sections.append(f"tools at call time ({len(record.tools)}): {names}")
        if record.replaced:
            sections.append(f"replaced:\n{record.replaced}")
        if record.fork_point:
            sections.append(f"{FORK_MARK} a fork may start here (closed turn)")
        return sections

    # -------------------------------------------------------------- events --

    def on_input_changed(self, event: Input.Changed) -> None:
        """Coalesce a burst of keystrokes into one rebuild.

        `DataTable` virtualizes *rendering* and not row construction. A refill
        measures every cell of every row, so the cost is paid on each keystroke
        over the whole log. Row surgery is worse, because removing the rows
        that drop out reindexes the table each time. So the fix is the
        frequency and not the primitive.
        """
        if self._pending is not None:
            self._pending.stop()
        self._pending = self.set_timer(FILTER_DEBOUNCE, lambda: self.refresh_rows(event.value))

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        self.show_details(self.selected())
