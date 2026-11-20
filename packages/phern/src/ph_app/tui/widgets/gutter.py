"""The gutter: one column that is the scrollbar and the highlight bars of the rows.

A transcript row that needs a mark is a tool card, the reasoning of the model,
or a compaction marker. Such a row used to draw its own left border. That put
the bar in column 1 and the text of the row in column 3, two columns right of
every other row. The scrollbar was on the right, in two columns of its own.
Here both are one column, at the left. The track and the thumb of the scrollbar
are the cell backgrounds, and the bar of each row is the glyph drawn over them
in the color that the row gives. The text of every row starts in column 1.

**A row says what color its bar is, and the gutter draws it.** The color is the
`row--bar` component style of the row, so it is CSS. It is themed, and the same
`-settled`/`-error`/`-thinking` classes that style the rest of the row switch
it. A row whose CSS names no color has no bar, which is the case for a plain
message. `transparent` turns off a bar that a broader rule gave. `Barred` is
the row side of that, and `GutterScroll` is the view side.

**Why the column is padding.** Textual places the children of a scroll view at
the left edge of its content box (`_compositor.add_widget`: `placement_offset =
container_region.offset`) and takes the strip of the scrollbar out of that box
*afterwards* (`_get_scrollable_region`). On the right those are the same point.
On the left they are a column that the rows and the bar both claim, and the
first character of each row disappears under the bar. So the column is
`padding-left`, which moves the content box, and the bar is arranged into it.

**Three members are overridden, and two of them are private**, because Textual
has no style for the side that a scrollbar is on. `scrollbar_gutter` reserves
nothing for the vertical bar, `_get_scrollable_region` carves nothing vertical,
and `_arrange_scrollbars` places the bar one scrollbar-width left of the
content box. `test_tui_gutter` fails when a Textual release renames one, and
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
"""The left half of a cell. It is Textual's `outer` left border, which the rows
drew for themselves before the gutter drew it for them."""

ROW_BAR = "row--bar"
"""The component class a `Barred` row styles: `Row > .row--bar { color: … }`."""


class Barred:
    """A row that the gutter draws a bar beside, in its `row--bar` color.

    This is a bare mixin, not a `DOMNode`, for the same reason as
    `_SourceSelection`. Textual builds the default CSS of a widget from a single
    `__bases__` chain and steps over a mixin that is not a `DOMNode`. A mixin
    that is a `DOMNode` would truncate the chain. The row class itself declares
    `COMPONENT_CLASSES = {ROW_BAR}`, because Textual gathers those from the
    `DOMNode` bases only.
    """

    def gutter_bar(self) -> Color | None:
        """Return the color of the bar, or `None` when the CSS of the row gives it none.

        **No rule means no bar**, not a default. An unset component color is
        opaque white. A row that never mentioned its bar would otherwise draw a
        white one beside every line, and would need a `transparent` rule to say
        what saying nothing already meant."""
        styles = cast(Widget, self).get_component_styles(ROW_BAR)
        if not styles.has_rule("color"):
            return None
        color = styles.color
        return color if color.a else None


_UP = {"@mouse.down": "scroll_up"}
_GRAB = {"@mouse.down": "grab"}
_DOWN = {"@mouse.down": "scroll_down"}
"""What a press does on a cell above the thumb, on the thumb, and below it. These
are the `ScrollBar` actions that Textual's own renderer names, attached in the
same way."""


class GutterRender(ScrollBarRender):
    """The track and the thumb of the scrollbar as cell backgrounds, with the bar
    of each row drawn over them in the color that the row gives.

    **The thumb is whole cells, and has one length.** Textual's stock thumb is
    measured in eighths of a cell and draws a partial block at each end, so its
    edges move smoothly. A cell beside a bar can hold one glyph, the bar. So
    each partial end had to become a whole thumb cell or none. Counted as whole
    cells, a thumb that sat on a cell boundary was one row shorter than the same
    thumb straddling a boundary, so it flickered by a row on every line
    scrolled. Here the window fixes the length (the stock arithmetic, rounded
    one time), and only the start moves, a whole cell at a time, from the top
    row at the top to the last row at the end.
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
        # The stock parameters are spelled out, because `ScrollBar.render` passes them by name.
        super().__init__(virtual_size, window_size, position, thickness, vertical, style)
        self.bars = bars
        """For each gutter row, the color of the bar beside it. It is asked at
        render time, when the regions of the rows are those of the frame."""

    def thumb(self, size: int) -> range | None:
        """Return the rows that the thumb covers in a bar `size` rows tall, or `None`
        when there is nothing to scroll. The condition is that of the stock
        renderer, unchanged."""
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

    The CSS of the view itself says `padding-left: 1; scrollbar-size-vertical: 1`.
    The two must agree, and `test_tui_gutter` checks the arithmetic and does not
    trust it. For a view with `Barred` rows, the CSS also says
    `overflow-y: scroll`, so the column is drawn whether or not there is
    anything to scroll. The bars are drawn in it.

    **Vertical only.** Both views that use it hide horizontal overflow
    (`VerticalScroll`, and `ListView` with it). A horizontal bar here would also
    need the corner cell that Textual's own arrangement draws. So this mixin
    does not serve a view that scrolls sideways.

    A bare mixin, for the same reason as `Barred`.
    """

    def on_mount(self) -> None:
        """Give the scrollbar a renderer that knows the rows of this view.

        This sets an instance attribute over the class-level `renderer`.
        `renderer` is Textual's own extension point for the looks of the
        scrollbar, and it is declared as a class variable. The bars belong to
        this view, so the binding is per instance.
        """
        widget = cast(Widget, self)
        widget.vertical_scrollbar.renderer = partial(GutterRender, bars=self.gutter_bars)  # type: ignore[assignment, misc]

    def gutter_bars(self) -> list[Color | None]:
        """Return, for each row of the gutter, the bar color of the row beside it, or `None`.

        **The answer comes from what is on screen, one point for each line.**
        The point is the widget under the first column of the content box,
        walked up to the child of this view. It is never the `region` of a
        child. On a frame that only scrolled, Textual has rebuilt the visible
        map and not the full one. The region of a row that is off screen is
        looked up in the full map, which rebuilds it. That is a whole-screen
        arrange for each line scrolled, measured at 24 ms a step at 300 rows.
        The point lookup reads the visible map only, and a row that spans many
        lines is asked for its color one time.
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
        """Return the content box whole. The bar takes nothing from it."""
        return region

    def _arrange_scrollbars(self, region: Region) -> Iterable[tuple[Widget, Region]]:
        """Place the vertical bar in the column left of the content box `region`."""
        widget = cast(Widget, self)
        size = widget.scrollbar_size_vertical
        if widget.show_vertical_scrollbar and size:
            scrollbar = widget.vertical_scrollbar
            scrollbar.window_virtual_size = widget.virtual_size.height
            scrollbar.window_size = region.height
            yield scrollbar, Region(region.x - size, region.y, size, region.height)
