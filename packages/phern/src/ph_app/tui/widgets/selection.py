"""Copying a model's answer gives back the Markdown it wrote.

pH already has the Markdown: `ChatItem.text` is what the model sent, stored as it
arrived, and rendering is only a view of it. So a copy is a *projection* back
onto that text — never a reassembly of what is on screen, and never a character
this module invented.

**The default is a photograph of the screen.** Textual's `Widget.get_selection`
extracts from `self._render()`, so a copy is the rendering by construction —
measured, `## Heading` comes back as `Heading`, `**bold**` as `bold`, `- item` as
`•  item` (Textual's own `MarkdownBullet.get_selection` returns the glyph), a
fenced block as bare code with no fence, and a pipe table as four loose cells.
Every other transcript row is a `Static` holding `item.text`, which copies back
verbatim; the rows that lose their Markdown are the ones that *render* it — the
model's own output, which is the part most worth pasting somewhere else.

**Every block answers with its own span of the document.** Textual gives each
`MarkdownBlock` a `source_range`: the half-open line range it was built from.
Those ranges are exact but not contiguous — a blank line between two blocks
belongs to neither — so a block takes its range *and the blank lines that follow
it*, which puts the paragraph breaks back without inventing any. They matter: two
paragraphs run together are one paragraph when pasted, which changes what the
text means rather than how it looks. The widgets are joined with an empty ending,
because every newline is already in the span.

**One rule decides who speaks**, applied by `_SourceSelection.on_mount` to each
block's own subtree, because Textual's selection walk reaches only leaves
(`textual/selection.py` skips every container) and three of the four shapes a
Markdown document renders keep their text somewhere that is not a block:

* A block with **no blocks under it** is the only one who knows its source, so
  its first leaf speaks for the whole of it and the rest stay silent. That is a
  fence — whose code sits in a `Label` — and a table, whose cells are `Static`s
  built from rows that were never mounted as blocks.
* A block **with blocks under it** lets them answer and silences its own loose
  leaves. That is a list: the item paragraphs carry `- `, and the bullet beside
  them would otherwise add the `●` that is drawn.

Silent means answering `("", "")` rather than leaving the selection: a widget
that drops out of `Screen.selections` also stops *highlighting*, and a person
dragging across a list would watch it come out gap-toothed.

Answering per block is also what keeps the mapping honest. A `Selection` carries
screen offsets, and screen rows are not source lines — they wrap, and a fence is
drawn taller than it reads. Within one block the disagreement is small, because
the block *is* those lines. Where it is not small the block speaks whole: a drag
that stopped inside a fence used to return ```` ```pyth ````, the marker line,
because the `Label`'s first row is the fence's *second* source line.

@module ph_app.tui.widgets.selection
"""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

from textual.geometry import Offset, clamp
from textual.selection import Selection
from textual.widget import Widget
from textual.widgets import Markdown
from textual.widgets.markdown import MarkdownBlock

__all__ = ["SOURCE_BLOCKS"]


def _owner(widget: Widget) -> MarkdownBlock | None:
    """The block this widget draws a piece of — itself, when it is one."""
    return next((n for n in widget.ancestors_with_self if isinstance(n, MarkdownBlock)), None)


def _span(widget: Widget) -> tuple[str, str] | None:
    """This widget's block, and the blank lines separating it from the next one.

    Cut out of the document rather than rebuilt: `MarkdownBlock.source` is the
    same slice, and the only thing added here is the walk forward through blank
    lines, which belong to no block's range and are what hold two paragraphs
    apart. The last block walks to the end, where there is nothing to hold apart.
    """
    nodes = widget.ancestors_with_self
    block = next((n for n in nodes if isinstance(n, MarkdownBlock)), None)
    document = next((n for n in nodes if isinstance(n, Markdown)), None)
    if block is None or document is None:
        return None
    lines = document.source.splitlines(keepends=True)
    start, end = block.source_range
    while end < len(lines) and not lines[end].strip():
        end += 1
    span = "".join(lines[start:end])
    body = span.rstrip("\n")
    return body, span[len(body) :]


def _clip(body: str, separator: str, selection: Selection) -> str:
    """The selection, mapped onto the source rather than onto what was drawn.

    Clamped per line because a rendered row is wrapped and padded where a source
    line is not.

    **`selection.end is None` is the question "did the selection run past me".**
    Textual hands the widget a drag *stopped* on a real `end`, and every widget
    the drag ran through a `None` one (`Screen._watch__select_state`), so that is
    exactly when the blank line after this block belongs in the copy. Comparing
    the extracted text against the whole body instead looks equivalent and is
    not: the widget a drag *starts* in mid-way yields a suffix, so its separator
    went missing and two paragraphs arrived spelled as one.
    """
    lines = body.splitlines()
    if not lines:
        return separator if selection.end is None else ""

    def clip(offset: Offset | None) -> Offset | None:
        if offset is None:
            return None
        row = clamp(offset.y, 0, len(lines) - 1)
        return Offset(clamp(offset.x, 0, len(lines[row])), row)

    text = Selection(clip(selection.start), clip(selection.end)).extract(body)
    return text + separator if selection.end is None else text


def _selected(widget: Widget, selection: Selection) -> tuple[str, str] | None:
    """What this widget contributes to a copy: its block's source, clipped."""
    span = _span(widget)
    return None if span is None else (_clip(*span, selection), "")


def _whole(block: Widget) -> Callable[[Selection], tuple[str, str] | None]:
    """Answer for all of `block`, however much of it was dragged over.

    Unclipped on purpose. This is the answer for a fence and a table, where the
    rows on screen and the lines in the source do not correspond at all — a
    fence draws its code without the markers it is made of, a table draws a grid
    — so all of it or none of it is the only honest offer.
    """

    def answer(_selection: Selection) -> tuple[str, str] | None:
        span = _span(block)
        return None if span is None else ("".join(span), "")

    return answer


def _nothing(_selection: Selection) -> tuple[str, str] | None:
    """Contribute no text, but stay in the selection so the highlight is drawn."""
    return "", ""


class _SourceSelection:
    """Answer a selection with this block's Markdown, not with its rendering.

    **A bare mixin, deliberately not a `MarkdownBlock` subclass.** Textual builds
    a widget's default CSS from a single `__bases__` chain, taking the first
    `DOMNode` base at each step (`DOMNode._css_bases`): a mixin that is not a
    `DOMNode` is stepped over, while one deriving from `MarkdownBlock` would
    *truncate* the chain at itself and drop everything the real block declares.
    Measured through `MarkdownBulletList`, whose `Horizontal`/`Vertical` rules
    would go missing: a two-row list stretched to the full height of the screen.
    """

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return _selected(cast(Widget, self), selection)

    def on_mount(self) -> None:
        """Hand out the answers for the leaves this block owns — see the module.

        On mount because there is no earlier moment: the widgets in question are
        built inside Textual's own `compose`, are reachable through no table pH
        can override, and `MarkdownStream.write` mounts the blocks it parsed only
        after it returns.

        Assigned per widget rather than by subclassing, for the same reason: a
        table cell and a bullet are constructed by library code with no seam. It
        reaches no further than one block's own subtree.
        """
        block = cast(Widget, self)
        inside = block.walk_children(Widget, with_self=False)
        leaves = [node for node in inside if not node.is_container and _owner(node) is block]
        if not leaves:
            return
        speaker = None if any(isinstance(node, MarkdownBlock) for node in inside) else leaves[0]
        for leaf in leaves:
            answer = _whole(block) if leaf is speaker else _nothing
            leaf.get_selection = answer  # type: ignore[assignment]


SOURCE_BLOCKS: dict[str, type[MarkdownBlock]] = {
    token: type(f"Source{block.__name__}", (_SourceSelection, block), {})
    for token, block in Markdown.BLOCKS.items()
}
"""Textual's block table, each entry taught to answer with its own span.

Derived rather than listed: `BLOCKS` has twenty-one entries, and a hand-written
copy would quietly stop covering a block type Textual adds — which is the failure
this table had in its first form, where a pipe table copied as four loose cells
because tables were the one shape nobody had thought of.
"""
