"""The gutter: one column that is the scrollbar and the rows' highlight bars.

A transcript row that wants marking — a tool card, the model's reasoning, a
compaction marker — used to draw its own left border, which put the bar in
column 1 and the row's text in column 3, two columns right of every other row.
The scrollbar, meanwhile, sat on the right in two columns of its own. Here both
are one column, at the left: the scrollbar's track and thumb are the cell
backgrounds, and each row's bar is the glyph drawn over them in the row's color.
Every row's text starts in column 1.

**A row says what color its bar is; the gutter draws it.** The color is the
row's `row--bar` component style, so it is CSS — themed, and switched by the
same `-settled`/`-error`/`-thinking` classes that style the rest of the row. A
row whose CSS names no color has no bar, which is what a plain message says,
and `transparent` turns off a bar a broader rule gave. `Barred` is the row's
side of that; `GutterScroll` is the view's.

**Why the column is padding.** Textual places a scroll view's children at the
left edge of its content box (`_compositor.add_widget`: `placement_offset =
container_region.offset`) and takes the scrollbar's strip out of that box
*afterwards* (`_get_scrollable_region`). On the right those are the same point;
on the left they are a column the rows and the bar both claim, and the rows'
first character disappears under the bar. So the column is `padding-left`,
which moves the content box, and the bar is arranged into it.

**Three overridden members, two of them private**, because Textual has no style
for which side a scrollbar is on: `scrollbar_gutter` reserves nothing for the
vertical bar, `_get_scrollable_region` carves nothing vertical, and
`_arrange_scrollbars` places the bar one scrollbar-width left of the content
box. `test_tui_gutter` is the tripwire for a Textual that renames one, and
Textual is pinned below 9.

@module ph_app.tui.widgets.gutter
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from functools import partial
from typing import cast

from rich.console import Console, ConsoleOptions, RenderResult
from rich.segment import Segment, Segments
from rich.style import Style, StyleType
from textual.color import Color
from textual.errors import NoWidget
from textual.geometry import Region, Spacing
from textual.scrollbar import ScrollBarRender
from textual.widget import Widget

__all__ = ["BAR", "ROW_BAR", "Barred", "GutterRender", "GutterScroll"]

BAR = "▌"
"""The left half of a cell — Textual's `outer` left border, which is what the rows
drew for themselves before the gutter drew it for them."""

ROW_BAR = "row--bar"
"""The component class a `Barred` row styles: `Row > .row--bar { color: … }`."""


class Barred:
    """A row the gutter draws a bar beside, in its `row--bar` color.

    A bare mixin, not a `DOMNode`, for `_SourceSelection`'s reason: Textual builds
    a widget's default CSS from a single `__bases__` chain and steps over a mixin
    that is not a `DOMNode`, where one that is would truncate the chain. The row
    class itself declares `COMPONENT_CLASSES = {ROW_BAR}`, because Textual
    gathers those from the `DOMNode` bases only.
    """

    def gutter_bar(self) -> Color | None:
        """The bar's color, or `None` when the row's CSS gives it none.

        **No rule is no bar**, rather than a default: an unset component color is
        opaque white, so a row that never mentioned its bar would otherwise draw
        a white one beside every line — and needed a `transparent` rule to say
        what not saying anything already meant."""
        styles = cast(Widget, self).get_component_styles(ROW_BAR)
        if not styles.has_rule("color"):
            return None
        color = styles.color
        return color if color.a else None


_UP = {"@mouse.down": "scroll_up"}
_GRAB = {"@mouse.down": "grab"}
_DOWN = {"@mouse.down": "scroll_down"}
"""What a press on a cell above the thumb, on it, and below it does — the
`ScrollBar` actions Textual's own renderer names, attached the same way."""


class GutterRender(ScrollBarRender):
    """The scrollbar's track and thumb as cell backgrounds, each row's bar drawn
    over them in the row's color.

    **The thumb is whole cells, one length.** Textual's stock thumb is measured in
    eighths of a cell and draws a partial block at each end, so its edges move
    smoothly; a cell beside a bar can hold one glyph, the bar, so each partial
    end had to become a whole thumb cell or none — and counted as whole, a thumb
    sitting on a cell boundary was one row shorter than the same thumb straddling
    one, flickering a row on every line scrolled. Here the length is fixed by the
    window (the stock arithmetic, rounded once) and only the start moves, a whole
    cell at a time, from the top row at the top to the last at the end.
    """

    def __init__(
        self,
        *,
        bars: Callable[[], Sequence[Color | None]],
        virtual_size: int = 100,
        window_size: int = 0,
        position: float = 0,
        thickness: int = 1,
        vertical: bool = True,
        style: StyleType = "bright_magenta on #555555",
    ) -> None:
        # The stock parameters, spelled: `ScrollBar.render` passes them by name.
        super().__init__(virtual_size, window_size, position, thickness, vertical, style)
        self.bars = bars
        """Per gutter row, the color of the bar beside it. Asked at render time,
        which is when the rows' regions are the frame's."""

    def thumb(self, size: int) -> range | None:
        """The rows the thumb covers in a bar `size` rows tall, or `None` when
        there is nothing to scroll — the stock renderer's condition, unchanged."""
        virtual, window = self.virtual_size, self.window_size
        if not (window and size and virtual and size != virtual):
            return None
        length = min(size, max(1, round(window * size / virtual)))
        ratio = min(1.0, max(0.0, self.position / (virtual - window)))
        start = round((size - length) * ratio)
        return range(start, start + length)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        size = options.height or console.height
        pad = " " * ((options.max_width or 1) - 1)
        style = console.get_style(self.style)
        span = self.thumb(size)
        bars = self.bars()
        cells: list[Segment] = []
        for row in range(size):
            on_thumb = span is not None and row in span
            if span is None:
                meta = None  # nothing to scroll, so nothing to press: the stock way
            else:
                meta = _GRAB if on_thumb else _UP if row < span.start else _DOWN
            color = bars[row] if row < len(bars) else None
            cells.append(
                Segment(
                    (BAR if color is not None else " ") + pad,
                    Style(
                        color=color.rich_color if color is not None else None,
                        bgcolor=style.color if on_thumb else style.bgcolor,
                        meta=meta,
                    ),
                )
            )
        yield Segments(cells, new_lines=True)


class GutterScroll:
    """Mixin for a scroll view whose vertical scrollbar is its left padding column.

    The view's own CSS says `padding-left: 1; scrollbar-size-vertical: 1` (the
    two must agree; `test_tui_gutter` checks the arithmetic rather than trusting
    it) and, for a view with `Barred` rows, `overflow-y: scroll`, so the column
    is drawn whether or not there is anything to scroll — the bars live in it.

    **Vertical only.** Both views that use it hide horizontal overflow
    (`VerticalScroll`, and `ListView` with it); a horizontal bar here would also
    need the corner cell Textual's own arrangement draws, so a view that scrolls
    sideways is not one this mixin serves.

    A bare mixin, for `Barred`'s reason.
    """

    def on_mount(self) -> None:
        """Hand the scrollbar a renderer that knows this view's rows.

        An instance attribute over a class-level `renderer`, which is Textual's
        own extension point for the scrollbar's looks, declared as a class
        variable; the bars are this view's, so the binding is per instance.
        """
        widget = cast(Widget, self)
        widget.vertical_scrollbar.renderer = partial(GutterRender, bars=self.gutter_bars)  # type: ignore[assignment, misc]

    def gutter_bars(self) -> list[Color | None]:
        """Per row of the gutter, the bar color of the row beside it, or `None`.

        **Asked of what is on screen, one point per line**: the widget under the
        content box's first column, walked up to this view's child. Never a
        child's `region` — on a frame that only scrolled, Textual has rebuilt
        the visible map and not the full one, and the region of a row off
        screen is looked up in the full one, which rebuilds it: a whole-screen
        arrange per line scrolled, measured at 24 ms a step at 300 rows. The
        point lookup reads the visible map only, and a row spanning many lines
        is asked its color once.
        """
        widget = cast(Widget, self)
        box = widget.content_region
        screen = widget.screen
        colors: dict[int, Color | None] = {}
        bars: list[Color | None] = []
        for y in range(box.y, box.bottom):
            try:
                under, _ = screen.get_widget_at(box.x, y)
            except NoWidget:
                bars.append(None)
                continue
            row = next((node for node in under.ancestors_with_self if node.parent is widget), None)
            if not isinstance(row, Barred):
                bars.append(None)
                continue
            key = id(row)
            if key not in colors:
                colors[key] = row.gutter_bar()
            bars.append(colors[key])
        return bars

    # ---- the three members that put the bar on the left ---------------------

    @property
    def scrollbar_gutter(self) -> Spacing:
        """Nothing: the padding column holds the vertical bar, and there is no
        horizontal one."""
        return Spacing()

    def _get_scrollable_region(self, region: Region) -> Region:
        """The content box whole — the bar takes nothing from it."""
        return region

    def _arrange_scrollbars(self, region: Region) -> Iterable[tuple[Widget, Region]]:
        """The vertical bar, in the column left of the content box `region`."""
        widget = cast(Widget, self)
        size = widget.scrollbar_size_vertical
        if widget.show_vertical_scrollbar and size:
            scrollbar = widget.vertical_scrollbar
            scrollbar.window_virtual_size = widget.virtual_size.height
            scrollbar.window_size = region.height
            yield scrollbar, Region(region.x - size, region.y, size, region.height)
