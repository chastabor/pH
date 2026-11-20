"""Copying the answer of the model gives back the Markdown that the model wrote.

The tests make real drags and do not call `get_selection` directly, because no
smaller instrument could see the defects that this file guards against.
Textual's selection walk reaches only leaves. An override on the message widget
is the obvious place for one, and it is never called. A hand-written call would
have passed against a TUI that still copied `•  item`. The shape of the drag
matters as much. Every test here used to start at the very first cell, and that
is precisely the drag that cannot see a lost paragraph break.

`ph_app.tui.widgets.selection` carries the argument for the design. This file
holds the design to the one claim that a user can check: paste what you copied,
and it is what the model sent.
"""

from __future__ import annotations

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widget import Widget
from textual.widgets._markdown import MarkdownBlock

from ph_app.tui.state import ChatItem
from ph_app.tui.themes import fallback_variables
from ph_app.tui.widgets.transcript import StreamingMessage

pytestmark = pytest.mark.anyio

ANSWER = """\
## Heading two

A paragraph with **bold** and `code`.

A second paragraph, which must not run into the first.

- first bullet
- second bullet

| left | right |
| ---- | ----- |
| 1    | 2     |

```python
x = 1
```

Closing paragraph.
"""


class _Transcript(App[None]):
    """One assistant row, mounted in the way that the transcript mounts it.

    The app holds the row and not the whole `PHTuiApp`. A widget and a drag are
    under test, and the full app would bring a daemon, a socket and a mount with
    it. The app must still provide `get_theme_variable_defaults`. The CSS of the
    row names `$ph-*`, and Textual treats an unresolved variable as a parse
    failure and not as a missing color.
    """

    def __init__(self, text: str, *, written: str = "") -> None:
        super().__init__()
        self.text = text
        self.written = written
        """What the row streamed before `text` settled it. It is for the case
        where the two are not the same string: attaching in the middle of an
        answer, or a message that is assembled from blocks and not from the
        chunks that this client saw."""

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def compose(self) -> ComposeResult:
        yield StreamingMessage(ChatItem(key="a", role="assistant"))

    async def on_mount(self) -> None:
        row = self.query_one(StreamingMessage)
        if self.written:
            await row.write_tail(self.written)
        await row.finalize(self.text)


async def _copy(
    text: str,
    *,
    target: type[Widget] = StreamingMessage,
    start: Offset = Offset(0, 0),
    release: type[Widget] | None = None,
    end: Offset | None = None,
) -> str:
    """Drag from the first `target` to the last `release`, and return the copy.

    There are two endpoints because the drags worth making are not all inside
    one widget. The press is on the first `target` and the release is on the
    last `release`, which is the same widget unless a test says otherwise.

    `end` defaults to the last cell of the released widget, and not to a corner
    of the screen. The offsets that a pilot takes are relative to the widget,
    and the corner of the screen is outside a row that is shorter than the
    screen. The check is spelled `is not None` because `Offset.__bool__` is
    `!= (0, 0)`. A falsy `Offset(0, 0)` would silently mean "all of it".
    """
    app = _Transcript(text)
    async with app.run_test(size=(60, 60)) as pilot:
        await pilot.pause()
        pressed = app.query(target).first()
        released = app.query(release).last() if release is not None else pressed
        stop = end if end is not None else Offset(released.size.width - 1, released.size.height - 1)
        await pilot.mouse_down(pressed, offset=start)
        await pilot.hover(released, offset=stop)
        await pilot.mouse_up(released, offset=stop)
        await pilot.pause()
        return app.screen.get_selected_text() or ""


async def test_copying_a_whole_answer_gives_back_the_markdown() -> None:
    """The whole claim: what comes out is what went in.

    The assertion is equality with the source, and not a list of things that
    survived, because the failures are all *omissions*: a `##` that is gone, a
    fence that lost its backticks, a table that arrived as four loose cells, a
    blank line that was never there. An assertion that lists them catches only
    the ones that somebody thought of. The blank lines are the half that needs
    stating. Two paragraphs that run together are one paragraph when pasted,
    which changes what the text means and not how it looks.
    """
    assert (await _copy(ANSWER)).strip() == ANSWER.strip()


async def test_a_copy_is_not_what_was_drawn() -> None:
    """The same assertion from the other side, in the terms of the renderings.

    `##`, the fence markers and the pipes of the table are absent from the
    screen by design, because Textual renders them away. Their presence in a
    copy therefore proves that the copy came from the source. `•` is the
    reverse. Textual's `MarkdownBullet` answers a selection with its glyph, so a
    `•` here means that the bullet still answers in place of the `- ` of the
    item.
    """
    copied = await _copy(ANSWER)

    assert "## Heading two" in copied
    assert "```python" in copied
    assert "- first bullet" in copied
    assert "| left | right |" in copied
    assert "•" not in copied, "the rendered bullet glyph reached the clipboard"


async def test_a_drag_that_starts_mid_block_keeps_the_paragraph_break() -> None:
    """The break belongs to the block that the drag *ran past*, not to the one that it filled.

    This is the drag that the other tests in the file could not make: it starts
    inside the first paragraph and ends inside a later one. To ask "is this the
    whole block" in place of "did the selection run past it" looks equivalent
    and is not. The block that a drag starts in yields a suffix and never the
    whole body. So its blank line went missing, and the two paragraphs arrived
    joined as one.
    """
    copied = await _copy(
        "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.\n",
        target=MarkdownBlock,
        start=Offset(6, 0),
        release=MarkdownBlock,
        end=Offset(5, 0),
    )

    # The break must come immediately after the *first* paragraph, and not only
    # somewhere in the copy. An interior block supplies a break of its own, so
    # `"\n\n" in copied` passes against the very bug that this test exists for.
    assert copied.startswith("paragraph here.\n\nSecond"), f"ran together: {copied!r}"


async def test_a_partial_selection_stops_where_the_drag_stopped() -> None:
    """Half a paragraph is half a paragraph, and not the block that it is in.

    The blank line that follows a block is carried only when the selection ran
    past the end of the block. That keeps a partial copy from picking up the
    separation before the next block.
    """
    # The end is a column, and the assertion is not an exact string. The row has
    # a margin and the heading is drawn narrower than it reads, so the renderer
    # decides where the drag lands in the source. What must hold is that the
    # drag stopped inside the heading.
    copied = await _copy(ANSWER, target=MarkdownBlock, end=Offset(5, 0))

    assert copied.startswith("## He")
    assert "paragraph" not in copied
    assert copied == copied.rstrip(), "a half-selected block carried the break after it"


async def test_a_message_with_no_markdown_is_unchanged() -> None:
    """The ordinary case. The text must gain nothing on the way through."""
    plain = "Just a sentence.\n"

    assert (await _copy(plain)).strip() == plain.strip()


async def test_a_row_whose_text_was_replaced_shows_the_replacement() -> None:
    """The cursor into a streamed row is a *string*, not a length (H2).

    `write_tail` used to slice the new text at the number of characters already
    written. That number is a cursor only while every update extends the last
    one, and not every update does. Attaching in the middle of an answer builds
    the row from a snapshot, and the settled `assistant/message` is assembled
    from blocks and not from the concatenated chunks. Sliced at the old length,
    what was appended was the *tail of a string that the row does not hold*. So
    the row showed neither text: it showed the start of one joined to the end
    of the other, at a position that means nothing in either. It stayed that
    way until something forced a rebuild.

    The two strings share a prefix and then diverge, which is the case that a
    length cannot see. A text that only *extends* what was written is handled
    in the same way by both, and that is the case that the old code was written
    for.

    `source` is the Markdown of the row itself. The copy path reads it, and it
    is what was corrupted. The rendering only reflected it.
    """
    app = _Transcript("thought about it, and the answer is no", written="thinking about i")
    async with app.run_test(size=(60, 20)) as pilot:
        await pilot.pause()
        row = app.query_one(StreamingMessage)

        assert row.source == "thought about it, and the answer is no"
