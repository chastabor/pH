"""Copying the model's answer gives back the Markdown it wrote.

Driven with real drags rather than by calling `get_selection` directly, because
the defects this guards were invisible to any smaller instrument. Textual's
selection walk reaches only leaves, so an override on the message widget — the
obvious place for one — is never called, and a hand-written call would have
passed against a TUI that still copied `•  item`. The shape of the drag matters
just as much: every test here once started at the very first cell, and that is
precisely the drag that cannot see a lost paragraph break.

`ph_app.tui.widgets.selection` carries the argument for the design; this file is
what holds it to the one claim a person can check: paste what you copied and it
is what the model sent.
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
    """One assistant row, mounted the way the transcript mounts it.

    The row rather than the whole `PHTuiApp`: what is under test is a widget and
    a drag, and the app would bring a daemon, a socket and a mount with it.
    `get_theme_variable_defaults` is the one thing it must still do — the row's
    CSS names `$ph-*`, and Textual treats an unresolved variable as a parse
    failure rather than a missing colour.
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self.text = text

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def compose(self) -> ComposeResult:
        yield StreamingMessage(ChatItem(key="a", role="assistant"))

    async def on_mount(self) -> None:
        await self.query_one(StreamingMessage).finalize(self.text)


async def _copy(
    text: str,
    *,
    target: type[Widget] = StreamingMessage,
    start: Offset = Offset(0, 0),
    release: type[Widget] | None = None,
    end: Offset | None = None,
) -> str:
    """Drag from the first `target` to the last `release`, and answer with the copy.

    Two endpoints because the drags worth making are not all inside one widget:
    press on the first `target`, release on the last `release` — the same widget
    unless a test says otherwise.

    `end` defaults to the released widget's own last cell rather than a corner of
    the screen: the offsets a pilot takes are widget-relative, and a row shorter
    than the screen puts the screen's corner outside it. Spelled `is not None`
    because `Offset.__bool__` is `!= (0, 0)`, so a falsy `Offset(0, 0)` would
    silently mean "all of it".
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
    """The claim, whole: what comes out is what went in.

    Equality with the source and not a list of things that survived, because the
    failures are all *omissions* — a `##` that is gone, a fence that lost its
    backticks, a table that arrived as four loose cells, a blank line that was
    never there — and an assertion listing them only ever catches the ones
    somebody thought of. The blank lines are the half worth saying out loud: two
    paragraphs run together are one paragraph when pasted, which changes what the
    text means rather than how it looks.
    """
    assert (await _copy(ANSWER)).strip() == ANSWER.strip()


async def test_a_copy_is_not_what_was_drawn() -> None:
    """The same assertion from the other side, in the renderings' own words.

    `##`, the fence markers and the table's pipes are absent from the screen by
    design — Textual renders them away — so their presence in a copy is proof it
    came from the source. `•` is the reverse: Textual's `MarkdownBullet` answers
    a selection with its glyph, and finding one here means the bullet is still
    speaking over the item's own `- `.
    """
    copied = await _copy(ANSWER)

    assert "## Heading two" in copied
    assert "```python" in copied
    assert "- first bullet" in copied
    assert "| left | right |" in copied
    assert "•" not in copied, "the rendered bullet glyph reached the clipboard"


async def test_a_drag_that_starts_mid_block_keeps_the_paragraph_break() -> None:
    """The break belongs to the block the drag *ran past*, not the one it filled.

    This is the drag the rest of the file could not make: starting inside the
    first paragraph and ending inside a later one. Asking "is this the whole
    block" instead of "did the selection run past it" looks equivalent and is
    not — the block a drag starts in yields a suffix, never the whole body, so
    its blank line went missing and the two paragraphs arrived spelled as one.
    """
    copied = await _copy(
        "First paragraph here.\n\nSecond paragraph here.\n\nThird paragraph here.\n",
        target=MarkdownBlock,
        start=Offset(6, 0),
        release=MarkdownBlock,
        end=Offset(5, 0),
    )

    # The break immediately after the *first* paragraph, not merely somewhere in
    # the copy: an interior block supplies one of its own, so `"\n\n" in copied`
    # passes against the very bug this test exists for.
    assert copied.startswith("paragraph here.\n\nSecond"), f"ran together: {copied!r}"


async def test_a_partial_selection_stops_where_the_drag_stopped() -> None:
    """Half a paragraph is half a paragraph — not the block it sits in.

    The blank line that follows a block is carried only when the selection ran
    past its end, which is what keeps a partial copy from picking up the
    separation before the next one.
    """
    # A column and not an exact string: the row is margined and the heading is
    # drawn narrower than it reads, so where the drag lands in the source is the
    # renderer's business. What must hold is that it stopped inside the heading.
    copied = await _copy(ANSWER, target=MarkdownBlock, end=Offset(5, 0))

    assert copied.startswith("## He")
    assert "paragraph" not in copied
    assert copied == copied.rstrip(), "a half-selected block carried the break after it"


async def test_a_message_with_no_markdown_is_unchanged() -> None:
    """The ordinary case, which must not acquire anything on the way through."""
    plain = "Just a sentence.\n"

    assert (await _copy(plain)).strip() == plain.strip()
