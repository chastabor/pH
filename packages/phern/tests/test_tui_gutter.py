"""The gutter: the transcript's first column is its scrollbar and its rows' bars.

Positions and colors read off the widgets rather than a picture, so a theme or a
font cannot pass these by accident; the snapshot suite has the pictures.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable

import pytest
from rich.color import Color as RichColor
from rich.segment import Segment
from textual.color import Color
from textual.geometry import Region
from textual.scrollbar import ScrollBar, ScrollBarRender
from textual.widget import Widget
from tui_helpers import TranscriptApp, rows_of

from ph_app.tui.state import ChatItem, ToolCard
from ph_app.tui.widgets.gutter import BAR, ROW_BAR, GutterRender, GutterScroll
from ph_app.tui.widgets.transcript import TranscriptView

__all__: list[str] = []

pytestmark = pytest.mark.anyio


def _card(key: str, *, settled: bool = False, is_error: bool = False) -> ChatItem:
    tool = ToolCard(
        call_id=key,
        name="read",
        arguments="{}",
        title="read",
        subtitle="a.py",
        settled=settled,
        is_error=is_error,
        body="1  from __future__ import annotations",
    )
    return ChatItem(key=key, role="tool", tool=tool)


def _rows() -> list[ChatItem]:
    """One of each row that carries a bar, between rows that do not."""
    return rows_of(
        ChatItem(key="u1", role="user", text="read it", seq=1),
        _card("done", settled=True),
        ChatItem(key="th", role="thinking", text="weighing the options", seq=3),
        _card("failed", settled=True, is_error=True),
        ChatItem(key="c1", role="compaction", text="Earlier: a question.", seq=5),
        _card("pending"),
        ChatItem(key="u2", role="user", text="and then", seq=7),
    )


def _cells(view: TranscriptView) -> list[Segment]:
    """The gutter as drawn: one one-cell segment per row of the scrollbar."""
    bar = view.vertical_scrollbar
    return [segment for y in range(bar.region.height) for segment in bar.render_line(y)]


def _gutter(view: TranscriptView, widget: Widget) -> Iterable[tuple[str, RichColor | None]]:
    """The gutter cells beside `widget`: each one's glyph and foreground."""
    top = view.vertical_scrollbar.region.y
    for cell in _cells(view)[widget.region.y - top : widget.region.bottom - top]:
        yield cell.text, cell.style.color if cell.style else None


def _scrolling() -> TranscriptApp:
    """Fourteen settled cards: a transcript that scrolls in a 60 by 12 terminal."""
    return TranscriptApp(rows_of(*(_card(f"c{n}", settled=True) for n in range(14))))


def _bar_color(widget: Widget) -> RichColor:
    return widget.get_component_styles(ROW_BAR).color.rich_color


async def test_the_gutter_is_column_zero_and_every_row_starts_in_column_one() -> None:
    """The scrollbar is the view's first column, always shown, and no row's text
    is under it: the padding column holds the bar, the content box starts after.

    Sabotage: return `region` unchanged from `GutterScroll._arrange_scrollbars`'s
    vertical branch, and the bar lands on the rows' first character.
    """
    app = TranscriptApp(_rows())
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        box = view.content_region
        bar = view.vertical_scrollbar

        assert bar.display, "the bar is hidden with nothing to scroll; the bars live in it"
        assert bar.region == Region(view.region.x, box.y, 1, box.height)
        assert box.x == view.region.x + 1
        for row in view.children:
            assert row.content_region.x == box.x, f"{row!r} is not in column 1"


async def test_each_rows_bar_is_drawn_in_the_gutter_in_its_own_color() -> None:
    """Beside a card, the bar glyph in the card's `row--bar` color — pending,
    done and failed are three colors — beside reasoning and a compaction marker
    theirs, and beside a plain message nothing at all.

    Sabotage: drop the `-error` rule from `ToolCardWidget`'s CSS, and the failed
    card's bar is the done card's green.
    """
    app = TranscriptApp(_rows())
    async with app.run_test(size=(60, 30)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        rows = {row.id: row for row in view.children}
        colors = {}
        for key in ("row-done", "row-failed", "row-pending", "row-th", "row-c1"):
            cells = list(_gutter(view, rows[key]))
            assert cells, f"{key} has no rows on screen"
            assert all(glyph == BAR for glyph, _ in cells), f"{key} is not barred"
            (color,) = {color for _, color in cells}
            assert color is not None and color.triplet == _bar_color(rows[key]).triplet
            colors[key] = color.triplet
        assert len({colors[k] for k in ("row-done", "row-failed", "row-pending")}) == 3
        assert all(glyph == " " for glyph, _ in _gutter(view, rows["row-u1"])), "a bar beside prose"


async def test_a_card_that_settles_recolors_its_bar() -> None:
    """The one change nothing else repaints the gutter for: a row's color moved
    and its size did not. `TranscriptView.sync` refreshes the bar for it.

    Sabotage: drop the `refresh()` at the end of `TranscriptView._sync`, and the
    bar stays the pending color until something scrolls.
    """
    items = rows_of(ChatItem(key="u1", role="user", text="go", seq=1), _card("c"))
    app = TranscriptApp(items)
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        card = view.query_one("#row-c")
        pending = _bar_color(card).triplet
        assert {color.triplet for _, color in _gutter(view, card) if color} == {pending}

        tool = items[1].tool
        assert tool is not None
        tool.settled = True
        await view.sync(items)
        await pilot.pause()

        done = _bar_color(card).triplet
        assert done != pending, "settling changed nothing; the test proves nothing"
        assert {color.triplet for _, color in _gutter(view, card) if color} == {done}


async def test_the_gutter_scrolls_by_mouse() -> None:
    """The track and the thumb still answer the mouse: Textual's own metadata on
    the cells is kept when a bar is drawn into them.

    Sabotage: build the bar cell's `Style` without `meta=`, and a click on a
    barred cell of the track does nothing.
    """
    app = _scrolling()
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        assert view.max_scroll_y > 0, "nothing to scroll; the test proves nothing"

        await pilot.click(TranscriptView, offset=(0, view.content_region.height))
        await pilot.pause()
        paged = view.scroll_y
        assert paged > 0, "a click on the track did not page"

        view.scroll_to(y=0, animate=False)
        await pilot.pause()
        await pilot.mouse_down(TranscriptView, offset=(0, 2))
        await pilot.hover(TranscriptView, offset=(0, 7))
        await pilot.mouse_up(TranscriptView, offset=(0, 7))
        await pilot.pause()
        assert view.scroll_y > 0, "dragging the thumb did not scroll"


def _thumb(view: TranscriptView) -> list[int]:
    """The gutter rows the thumb covers: the cells a press would grab."""
    return [
        y
        for y, cell in enumerate(_cells(view))
        if cell.style is not None and cell.style.meta.get("@mouse.down") == "grab"
    ]


async def test_the_thumb_keeps_its_size_while_it_scrolls() -> None:
    """Whole cells, one length, from the top row at the top to the last at the end.

    The stock thumb is measured in eighths of a cell and draws a partial block
    at each end; a cell beside a bar can hold one glyph, so each partial end
    became a whole thumb cell and the thumb was one row long when it sat on a
    cell boundary and two when it straddled one — flickering a row on every
    line scrolled.

    Sabotage: compute the length in `GutterRender.thumb` from the position as
    well as the window (round the end, not the length), and the lengths differ.
    """
    app = _scrolling()
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        height = view.vertical_scrollbar.region.height
        seen: dict[int, list[int]] = {}
        for y in range(int(view.max_scroll_y) + 1):
            view.scroll_to(y=y, animate=False)
            await pilot.pause()
            seen[y] = _thumb(view)

        lengths = {len(rows) for rows in seen.values()}
        assert len(lengths) == 1, f"the thumb changes size while scrolling: {seen}"
        assert all(rows == list(range(rows[0], rows[-1] + 1)) for rows in seen.values())
        assert seen[0][0] == 0, "the thumb is not at the top at the top"
        assert seen[int(view.max_scroll_y)][-1] == height - 1, "nor at the bottom at the end"


async def test_the_thumb_is_the_headings_color() -> None:
    """The thumb draws in `$ph-accent`, the role the status line and the side
    panel's headings use — a role, so it follows the theme with them."""
    app = _scrolling()
    async with app.run_test(size=(60, 12)) as pilot:
        await pilot.pause()
        view = app.query_one(TranscriptView)
        accent = Color.parse(app.get_css_variables()["ph-accent"]).rich_color.triplet
        cell = _cells(view)[_thumb(view)[0]]

        assert cell.style is not None and cell.style.bgcolor is not None
        assert cell.style.bgcolor.triplet == accent


def test_the_textual_members_the_gutter_relies_on_still_exist() -> None:
    """The tripwire for a Textual that renames what `GutterScroll` overrides or
    `GutterRender` is constructed with — a failure here names the member, where
    the alternative is a bar drawn on the right with every other test green."""
    for name in ("scrollbar_gutter", "_get_scrollable_region", "_arrange_scrollbars"):
        assert name in vars(Widget), f"Widget.{name} is gone"
        assert name in vars(GutterScroll)
    assert "renderer" in vars(ScrollBar), "ScrollBar.renderer is gone"
    for action in ("scroll_up", "grab", "scroll_down"):
        assert hasattr(ScrollBar, f"action_{action}"), f"a press on the gutter names {action}"
    stock = inspect.signature(ScrollBarRender.__init__).parameters
    ours = inspect.signature(GutterRender.__init__).parameters
    assert set(stock) - {"self"} == set(ours) - {"self", "bars"}
