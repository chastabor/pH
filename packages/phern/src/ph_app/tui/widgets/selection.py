"""Copying an answer of the model gives back the Markdown that the model wrote.

pH already has the Markdown. `ChatItem.text` is what the model sent, stored as
it arrived, and rendering is only a view of it. So a copy is a *projection* back
onto that text. It is never a reassembly of what is on screen, and it never
contains a character that this module invented.

**The default is a photograph of the screen.** Textual's `Widget.get_selection`
extracts from `self._render()`, so a copy is the rendering by construction.
Measured:

* `## Heading` comes back as `Heading`.
* `**bold**` comes back as `bold`.
* `- item` comes back as `•  item` (Textual's own `MarkdownBullet.get_selection`
  returns the glyph).
* A fenced block comes back as bare code with no fence.
* A pipe table comes back as four loose cells.

Every other transcript row is a `Static` that holds `item.text`, which copies
back verbatim. The rows that lose their Markdown are the rows that *render* it:
the output of the model, which is the part most worth pasting somewhere else.

**Every block answers with its own span of the document.** Textual gives each
`MarkdownBlock` a `source_range`: the half-open line range that it was built
from. Those ranges are exact but not contiguous, because a blank line between
two blocks belongs to neither. So a block takes its range *and the blank lines
that follow it*, which puts the paragraph breaks back and invents none. The
breaks matter: two paragraphs that run together are one paragraph when pasted,
which changes what the text means and not only how it looks. The widgets are
joined with an empty ending, because every newline is already in the span.

**One rule decides which widget answers.** `_SourceSelection.on_mount` applies
it to the subtree of each block. The rule is needed because Textual's selection
walk reaches only leaves (`textual/selection.py` skips every container), and
three of the four shapes that a Markdown document renders keep their text in a
widget that is not a block:

* A block with **no blocks under it** is the only widget that knows its source.
  Its first leaf answers for the whole block, and the other leaves stay silent.
  That covers a fence, whose code is in a `Label`, and a table, whose cells are
  `Static`s built from rows that were never mounted as blocks.
* A block **with blocks under it** lets them answer and silences its own loose
  leaves. That covers a list: the item paragraphs carry `- `, and the bullet
  beside them would otherwise add the `●` that is drawn.

A silent widget answers `("", "")` and does not leave the selection. A widget
that drops out of `Screen.selections` also stops *highlighting*, and a user who
drags across a list would see gaps in the highlight.

Answering for each block also keeps the mapping correct. A `Selection` carries
screen offsets, and screen rows are not source lines: rows wrap, and a fence is
drawn taller than it reads. Within one block the difference is small, because
the block *is* those lines. Where it is not small, the block answers whole. A
drag that stopped inside a fence used to return ```` ```pyth ````, the marker
line, because the first row of the `Label` is the *second* source line of the
fence.

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
    """Return the block that this widget draws a piece of. A block returns itself."""
    return next((n for n in widget.ancestors_with_self if isinstance(n, MarkdownBlock)), None)


def _span(widget: Widget) -> tuple[str, str] | None:
    """Return the block of this widget, and the blank lines that separate it from the next.

    The span is cut out of the document and not rebuilt. `MarkdownBlock.source`
    is the same slice. The only addition here is the walk forward through blank
    lines. They belong to the range of no block, and they hold two paragraphs
    apart. The last block walks to the end, where there is nothing to hold
    apart.
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
    """Return the selection, mapped onto the source and not onto what was drawn.

    Offsets are clamped for each line, because a rendered row is wrapped and
    padded and a source line is not.

    **`selection.end is None` answers "did the selection run past me".** Textual
    gives the widget where a drag *stopped* a real `end`, and gives every widget
    that the drag ran through a `None` (`Screen._watch__select_state`). So that
    is exactly when the blank line after this block belongs in the copy.
    Comparing the extracted text against the whole body looks equivalent and is
    not. The widget where a drag *starts* part-way yields a suffix, so its
    separator went missing and two paragraphs arrived joined as one.
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
    """Return what this widget contributes to a copy: the source of its block, clipped."""
    span = _span(widget)
    return None if span is None else (_clip(*span, selection), "")


def _whole(block: Widget) -> Callable[[Selection], tuple[str, str] | None]:
    """Answer for all of `block`, however much of it was dragged over.

    The answer is unclipped on purpose. It is the answer for a fence and a
    table, where the rows on screen and the lines in the source do not
    correspond at all. A fence draws its code without the markers that it is
    made of, and a table draws a grid. So the only correct offer is all of the
    block or none of it.
    """

    def answer(_selection: Selection) -> tuple[str, str] | None:
        span = _span(block)
        return None if span is None else ("".join(span), "")

    return answer


def _nothing(_selection: Selection) -> tuple[str, str] | None:
    """Contribute no text, but stay in the selection so the highlight is drawn."""
    return "", ""


class _SourceSelection:
    """Answer a selection with the Markdown of this block, not with its rendering.

    **A bare mixin, deliberately not a `MarkdownBlock` subclass.** Textual builds
    the default CSS of a widget from a single `__bases__` chain, and takes the
    first `DOMNode` base at each step (`DOMNode._css_bases`). A mixin that is
    not a `DOMNode` is stepped over. A mixin that derives from `MarkdownBlock`
    would *truncate* the chain at itself and drop everything that the real block
    declares. This was measured through `MarkdownBulletList`, whose
    `Horizontal`/`Vertical` rules would go missing: a two-row list stretched to
    the full height of the screen.
    """

    def get_selection(self, selection: Selection) -> tuple[str, str] | None:
        return _selected(cast(Widget, self), selection)

    def on_mount(self) -> None:
        """Hand out the answers for the leaves that this block owns. See the module docstring.

        This runs on mount because there is no earlier moment. Textual's own
        `compose` builds the widgets in question, pH can override no table that
        reaches them, and `MarkdownStream.write` mounts the blocks that it parsed
        only after it returns.

        The answer is assigned to each widget and not provided by a subclass,
        for the same reason: library code constructs a table cell and a bullet,
        with no seam. The assignment reaches no further than the subtree of one
        block.
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
"""Textual's block table, with each entry taught to answer with its own span.

The table is derived and not listed. `BLOCKS` has twenty-one entries, and a
hand-written copy would stop covering a block type that Textual adds, with no
warning. This table had that failure in its first form: a pipe table copied as
four loose cells, because tables were the one shape that nobody had thought of.
"""
