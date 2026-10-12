"""The diff screen: every change the log records, grouped as git groups commits by file
(`/diff`).

The row that contributes it is `ph_app.screen_rows`; what it lists is
`ph_app.tui.diffs`, which this screen draws without knowing where a group came
from. **Meant to grow more kinds**: context revisions are the one kind today, and
another is a source function there and nothing here. The `kind` column is already
in the table so a mixed list reads as one.

A table of groups, one per thing as it is now, over the selected group's details. They
show **each edit**, newest first with its producer's records field by field and its own
diff, as `git log -p -- file` does; or, after `w`, **the whole history** against the
original, as `git diff A..B -- file` does. Only the row on screen is rendered, and its
text is let go when the row changes, so opening the screen over a long log costs one
row's diffs and holds one row's text. The size column shows what the latest change's
producer recorded; a change with no recorded size has its characters counted when it
is shown.

**Enter goes to the thing in the transcript.** A substitution is its own row, and a
rewrite shows on the row it rewrote, since a rewrite draws no row of its own. It is the
same join the trajectory uses (`RevealSeq`, `Revealing`). Opened from the chat, the
screen starts on the nearest group at or before where the reader is.

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
from ph.session import RevisionSize, Session, unified_diff
from ph.text import count_of

from ..screen_rows import DIFF_KEY, DIFF_SCREEN_ID
from ..wire import index_at_or_before
from .diffs import DiffEntry, DiffGroup, build_diffs
from .screens import RevealHost, RevealSeq

__all__ = ["CLIENT_SIDE", "DIFF_LINES", "DiffScreen"]

DIFF_LINES = 400
"""Diff lines drawn for one group before a line saying how many were left out.
The diffs are computed whole, so nothing is missed; only the drawing is bounded, as
pi-clm bounds its edits page."""

EDIT_LABELS = ("before", "after")
WHOLE_LABELS = ("as first written", "now")


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
        Binding("w", "whole", "each edit / whole history", id="whole"),
    ]
    """`escape` and `w` are the screen's own. The table owns the arrows and `enter`,
    and `tab` moves focus to the details so a long diff can be scrolled."""

    def __init__(self, groups: list[DiffGroup], *, session_id: str = "") -> None:
        super().__init__()
        self.groups = groups
        self.session_id = session_id
        self.whole = False
        """Whether the details show the whole history rather than each edit."""
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
        table.add_columns("#", "kind", "seq", "change", "edits", "by", "size")
        for number, group in enumerate(self.groups, 1):
            latest = group.latest
            # Plain strings: `DataTable` renders a cell literally, so a producer
            # name with `[` in it cannot become markup.
            table.add_row(
                str(number),
                group.kind,
                str(group.seq),
                latest.title,
                str(len(group.entries)),
                group.producers,
                latest.recorded.label() if latest.recorded is not None else "",
            )
        if not self.groups:
            self.show_details(None)
        elif self._opening_seq >= 0:
            # Here, not after a refresh as the trajectory does: the rows are this
            # method's own, so they exist already. The first `add_row` posted a
            # highlight for row 0, and handled after a later move it would compute
            # row 0's diff for nobody. Moved now, that highlight draws the row the
            # cursor is on.
            self.select_seq(self._opening_seq)
        table.focus()
        changes = sum(len(group.entries) for group in self.groups)
        self.query_one("#diff-header", Static).update(
            Content(
                f"{self.session_id or 'session'} · {count_of(changes, 'change')} to "
                f"{count_of(len(self.groups), 'node')} · w: each edit / whole history"
            )
        )

    @property
    def table(self) -> DataTable[str]:
        return self.query_one("#diff-table", DataTable)

    # ------------------------------------------------------------ selection --

    def selected(self) -> DiffGroup | None:
        """The group under the cursor, or `None` when there are none."""
        return self.groups[self._row()] if self.groups else None

    def _row(self) -> int:
        return min(self.table.cursor_row, len(self.groups) - 1)

    def select_seq(self, seq: int) -> bool:
        """Put the cursor on the group at `seq`, or the nearest one before it.

        `index_at_or_before` is the join's one rule for "nearest", shared with the
        trajectory and the transcript. Moving the cursor posts `RowHighlighted`,
        which draws the details, so this does not draw them too.
        """
        row = index_at_or_before((group.seq for group in self.groups), seq)
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
        """A chosen group asks its host for the transcript row that shows it.

        A message, and the host asked structurally, for `TrajectoryScreen`'s
        reasons: this screen does not know what pushed it.
        """
        event.stop()
        group = self.selected()
        if group is None:
            return
        if not isinstance(self.app, RevealHost):
            self.notify("no transcript here to go to", title="diff", severity="warning")
            return
        self.post_message(RevealSeq(group.reveal, screen_id=DIFF_SCREEN_ID))

    def action_whole(self) -> None:
        """Between each edit and the whole history, for every group from here on."""
        self.whole = not self.whole
        self.show_details(self.selected())

    def action_back(self) -> None:
        if self.app.screen_stack and self is not self.app.screen_stack[0]:
            # Asked as "am I the one underneath", for the trajectory's reason.
            self.dismiss()

    # -------------------------------------------------------------- details --

    def show_details(self, group: DiffGroup | None) -> None:
        title = self.query_one("#diff-title", Static)
        text = self.query_one("#diff-text", Static)
        if group is None:
            title.update(Content("no changes"))
            text.update(Content("Nothing in this session's log has changed what the model sees."))
            return
        if self.whole:
            before, after = group.whole()
            shown = f"the whole history, {count_of(len(group.entries), 'edit')}"
            size = _counted(before, after)
            body = _diff(unified_diff(before, after, WHOLE_LABELS), DIFF_LINES)
        else:
            shown = f"each edit, {count_of(len(group.entries), 'edit')}, newest first"
            latest = group.latest
            size = latest.recorded.label() if latest.recorded is not None else ""
            body = _edits(group)
        title.update(
            Content.from_markup(
                "[b]#$number $kind · seq $seq[/b] [$ph-muted]$shown$size[/]",
                number=str(self._row() + 1),
                kind=group.kind,
                seq=str(group.seq),
                shown=shown,
                size=f" · {size}" if size else "",
            )
        )
        text.update(body)


def _edits(group: DiffGroup) -> Content:
    """Each edit in a group, newest first, each with its own diff, within one budget of
    drawn lines: an edit past it is counted, not rendered, since a summary's group holds
    every older summary and each one's before can be most of a conversation."""
    parts: list[Content] = []
    left = DIFF_LINES
    newest_first = list(reversed(group.entries))
    for drawn, entry in enumerate(newest_first):
        if left <= 0:
            older = count_of(len(newest_first) - drawn, "older edit")
            parts.append(Content.assemble((f"… {older}; w shows the whole history", "$ph-muted")))
            break
        before, after = entry.texts()
        lines = unified_diff(before, after, EDIT_LABELS)
        parts.append(
            _entry(entry, entry.recorded.label() if entry.recorded else _counted(before, after))
        )
        parts.append(_diff(lines, left))
        left -= len(lines)
    return Content("\n").join(parts)


def _counted(before: str, after: str) -> str:
    """A size counted here, in characters, for a change no record sized."""
    return RevisionSize("chars", len(before), len(after)).label() + ", counted here"


def _entry(entry: DiffEntry, size: str) -> Content:
    """One edit's heading: what it was, who wrote it, and its records field by field."""
    parts: list[str | tuple[str, str]] = [
        (f"{entry.title} · seq {entry.seq} · {size}", "bold $ph-accent"),
        "\n",
        entry.summary,
        "\n",
    ]
    if entry.producer:
        parts += [(f"by {entry.producer}", "$ph-muted"), "\n"]
    for fact in entry.facts:
        parts += [(fact, "$ph-muted"), "\n"]
    for record in entry.records:
        many = len(record.revises) > 1
        scope = f" · one record for {count_of(len(record.revises), 'change')}" if many else ""
        parts += [(f"{record.type} (seq {record.seq}{scope})", "bold"), "\n"]
        for key, value in record.fields:
            parts += [(f"  {key}: ", "$ph-muted"), value, "\n"]
    return Content.assemble(*parts)


def _diff(lines: tuple[str, ...], budget: int) -> Content:
    """A diff's lines, at most `budget` of them, then a line saying how many were left."""
    if not lines:
        return Content.assemble(("no text change", "$ph-muted"), "\n")
    parts: list[str | tuple[str, str]] = []
    for line in lines[:budget]:
        parts += [(line, _style(line)), "\n"]
    if len(lines) > budget:
        parts += [(f"… {count_of(len(lines) - budget, 'more line')}", "$ph-muted"), "\n"]
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
