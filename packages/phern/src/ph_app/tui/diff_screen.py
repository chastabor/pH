"""The diff screen: every change the log records, before and after (`/diff`).

The row that contributes it is `ph_app.screen_rows`; what it lists is
`ph_app.tui.diffs`, which this screen draws without knowing where an entry came
from. **Meant to grow more kinds**: context revisions are the one kind today, and
another is a source function there and nothing here. The `kind` column is already
in the table so a mixed list reads as one.

A table of changes over the selected change's details: its producer's records field
by field, then a unified diff. Only the row on screen is rendered, and its text is let
go when the row changes, so opening the screen over a long log costs one row's diff and
holds one row's text. The size column shows what a producer recorded; a change with no
recorded size has its characters counted when it is shown.

**Enter goes to the change in the transcript.** A substitution goes to its own row,
and a rewrite goes to the row it rewrote, since a rewrite draws no row of its own.
It is the same join the trajectory uses (`RevealSeq`, `Revealing`). Opened from the
chat, the screen starts on the nearest change at or before where the reader is.

@module ph_app.tui.diff_screen
"""

from __future__ import annotations

from typing import ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical, VerticalScroll
from textual.content import Content
from textual.screen import Screen
from textual.widgets import DataTable, Static

from ph.seams.tui_screens import ScreenDefinition
from ph.session import Session
from ph.text import count_of

from ..screen_rows import DIFF_KEY, DIFF_SCREEN_ID
from ..wire import index_at_or_before
from .diffs import DiffEntry, Size, build_diffs, unified
from .screens import RevealHost, RevealSeq

__all__ = ["CLIENT_SIDE", "DIFF_LINES", "DiffScreen"]

DIFF_LINES = 400
"""Diff lines drawn for one change before a line saying how many were left out.
The diff is computed whole, so nothing is missed; only the drawing is bounded, as
pi-clm bounds its edits page."""


class DiffScreen(Screen[None]):
    """The changes over one log: a header, the table, and the selected change."""

    DEFAULT_CSS = """
    DiffScreen { background: $ph-background; color: $ph-foreground; }
    DiffScreen > #diff-header {
        height: 1; background: $ph-panel; color: $ph-muted; padding: 0 1;
    }
    DiffScreen #diff-table { height: 1fr; }
    DiffScreen #diff-details {
        height: 2fr; background: $ph-panel; padding: 0 1; border-top: hkey $ph-border;
    }
    DiffScreen #diff-title { color: $ph-accent; }
    DiffScreen #diff-text { height: auto; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "back", "back", id="back"),
    ]
    """`escape` is the screen's own. The table owns the arrows and `enter`, and
    `tab` moves focus to the details so a long diff can be scrolled."""

    def __init__(self, entries: list[DiffEntry], *, session_id: str = "") -> None:
        super().__init__()
        self.entries = entries
        self.session_id = session_id
        self._opening_seq = -1

    # ---------------------------------------------------------------- shape --

    def compose(self) -> ComposeResult:
        yield Static(Content(""), id="diff-header")
        with Vertical(id="diff-body"):
            yield DataTable(id="diff-table", cursor_type="row", zebra_stripes=True)
            with VerticalScroll(id="diff-details"):
                yield Static(Content(""), id="diff-title")
                yield Static(Content(""), id="diff-text")

    def on_mount(self) -> None:
        table = self.table
        table.add_columns("#", "kind", "seq", "change", "by", "size")
        for number, entry in enumerate(self.entries, 1):
            # Plain strings: `DataTable` renders a cell literally, so a producer
            # name with `[` in it cannot become markup.
            table.add_row(
                str(number),
                entry.kind,
                str(entry.seq),
                entry.title,
                entry.producer,
                entry.recorded.label() if entry.recorded is not None else "",
            )
        if not self.entries:
            self.show_details(None)
        elif self._opening_seq >= 0:
            # Here, not after a refresh as the trajectory does: the rows are this
            # method's own, so they exist already. The first `add_row` posted a
            # highlight for row 0, and handled after a later move it would compute
            # row 0's diff for nobody. Moved now, that highlight draws the row the
            # cursor is on.
            self.select_seq(self._opening_seq)
        table.focus()
        self.query_one("#diff-header", Static).update(
            Content(f"{self.session_id or 'session'} · {count_of(len(self.entries), 'change')}")
        )

    @property
    def table(self) -> DataTable[str]:
        return self.query_one("#diff-table", DataTable)

    # ------------------------------------------------------------ selection --

    def selected(self) -> DiffEntry | None:
        """The change under the cursor, or `None` when there are none."""
        return self.entries[self._row()] if self.entries else None

    def _row(self) -> int:
        return min(self.table.cursor_row, len(self.entries) - 1)

    def select_seq(self, seq: int) -> bool:
        """Put the cursor on the change at `seq`, or the nearest one before it.

        `index_at_or_before` is the join's one rule for "nearest", shared with the
        trajectory and the transcript. Moving the cursor posts `RowHighlighted`,
        which draws the details, so this does not draw them too.
        """
        row = index_at_or_before((entry.seq for entry in self.entries), seq)
        if row < 0:
            return False
        self.table.move_cursor(row=row)
        return True

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        """Draw the change the cursor is on *now*, which is not always the row the
        message names: see `on_mount`."""
        event.stop()
        self.show_details(self.selected())

    # ------------------------------------------------------- cross-navigation --

    def reveal(self, seq: int) -> None:
        """Open positioned at a log seq (the `Revealing` protocol).

        Recorded rather than applied: the shell calls this before the screen is
        pushed, when there is no table to move a cursor in.
        """
        self._opening_seq = seq

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A chosen change asks its host for the transcript row that shows it.

        A message, and the host asked structurally, for `TrajectoryScreen`'s
        reasons: this screen does not know what pushed it.
        """
        event.stop()
        entry = self.selected()
        if entry is None:
            return
        if not isinstance(self.app, RevealHost):
            self.notify("no transcript here to go to", title="diff", severity="warning")
            return
        self.post_message(RevealSeq(entry.reveal, screen_id=DIFF_SCREEN_ID))

    def action_back(self) -> None:
        if self.app.screen_stack and self is not self.app.screen_stack[0]:
            # Asked as "am I the one underneath", for the trajectory's reason.
            self.dismiss()

    # -------------------------------------------------------------- details --

    def show_details(self, entry: DiffEntry | None) -> None:
        title = self.query_one("#diff-title", Static)
        text = self.query_one("#diff-text", Static)
        if entry is None:
            title.update(Content("no changes"))
            text.update(Content("Nothing in this session's log has changed what the model sees."))
            return
        before, after = entry.texts()
        size = entry.recorded or Size(len(before), len(after), "chars")
        title.update(
            Content.from_markup(
                "[b]#$number $kind · $title[/b] [$ph-muted]seq $seq · $size[/]",
                number=str(self._row() + 1),
                kind=entry.kind,
                title=entry.title,
                seq=str(entry.seq),
                size=size.label() + ("" if entry.recorded else ", counted here"),
            )
        )
        text.update(_details(entry, unified(before, after)))


def _details(entry: DiffEntry, lines: tuple[str, ...]) -> Content:
    """The selected change: what it was, who wrote it, then its diff."""
    parts: list[str | tuple[str, str]] = [entry.summary, "\n"]
    if entry.producer:
        parts += [(f"by {entry.producer}", "$ph-muted"), "\n"]
    for fact in entry.facts:
        parts += [(fact, "$ph-muted"), "\n"]
    for record in entry.records:
        scope = " · the whole batch's" if record.shared else ""
        parts += ["\n", (f"{record.type} (seq {record.seq}{scope})", "bold"), "\n"]
        for key, value in record.fields:
            parts += [(f"  {key}: ", "$ph-muted"), value, "\n"]
    parts.append("\n")
    if not lines:
        parts.append(("no text change", "$ph-muted"))
        return Content.assemble(*parts)
    for line in lines[:DIFF_LINES]:
        parts += [(line, _style(line)), "\n"]
    if len(lines) > DIFF_LINES:
        parts.append((f"… {count_of(len(lines) - DIFF_LINES, 'more line')}", "$ph-muted"))
    return Content.assemble(*parts)


def _style(line: str) -> str:
    """A unified diff line's color, by its first characters."""
    if line.startswith(("---", "+++")):
        return "bold $ph-muted"
    if line.startswith("@@"):
        return "$ph-accent"
    if line.startswith("+"):
        return "$ph-tool-success"
    if line.startswith("-"):
        return "$ph-error"
    return ""


def _build(session: Session) -> DiffScreen:
    """`build(session)` for this screen: the changes as the log stands now."""
    return DiffScreen(build_diffs(session), session_id=session.id)


CLIENT_SIDE = ScreenDefinition(
    id=DIFF_SCREEN_ID,
    label="Diff",
    order=20,
    key=DIFF_KEY,
    build=_build,
)
"""This screen as the front end draws it. The daemon's `screens/list` supplies the
label, order and key at attach time, so the copies here are defaults
(`trajectory_screen.CLIENT_SIDE` says why `build` cannot travel)."""
