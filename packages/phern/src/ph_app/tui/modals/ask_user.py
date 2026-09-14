"""The ask-user modal: a question from the model, answered by a person.

Two shapes, because the seam has two. With `options`, this is a picker and the
answer is one of them. Without, it is free text. Nothing here invents a third
mode — `multi_select` joins the chosen options with a comma, which is what the
seam's `str` return can carry.

@module ph_app.tui.modals.ask_user
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Button, Checkbox, Input, ListItem, ListView, Static

from ph.seams.user_questions import UserQuestion

from .base import PhModal

__all__ = ["AskUserModal"]


class AskUserModal(PhModal[str | None]):
    """Put one question to the human and return their answer."""

    DEFAULT_CSS = """
    AskUserModal { align: center middle; background: $background 60%; }
    AskUserModal > #ask {
        width: 76; max-width: 90%; height: auto; max-height: 80%;
        background: $ph-panel; border: round $ph-accent; padding: 1 2;
    }
    AskUserModal #ask-header { color: $ph-accent; }
    AskUserModal #ask-question { height: auto; margin: 0 0 1 0; }
    AskUserModal #ask-options { height: auto; max-height: 14; background: $ph-panel; }
    AskUserModal #ask-text { border: none; background: $ph-surface; }
    AskUserModal #ask-answer { margin-top: 1; }
    """

    def __init__(self, question: UserQuestion) -> None:
        super().__init__()
        self.question = question
        self.options = list(question.options or [])

    def compose(self) -> ComposeResult:
        with Vertical(id="ask"):
            if self.question.header:
                yield Static(
                    Content.from_markup("[b]$header[/b]", header=self.question.header),
                    id="ask-header",
                )
            yield Static(Content(self.question.question), id="ask-question")
            if not self.options:
                yield Input(placeholder="your answer…", id="ask-text")
            elif self.question.multi_select:
                # A checkbox spends `enter` on toggling, so the answer needs a
                # button of its own rather than a key the boxes already claim.
                with Vertical(id="ask-options"):
                    for index, option in enumerate(self.options):
                        yield Checkbox(option, id=f"ask-opt-{index}")
                yield Button("Answer", id="ask-answer", variant="primary")
            else:
                yield ListView(id="ask-options")

    def on_mount(self) -> None:
        if not self.options:
            self.query_one("#ask-text", Input).focus()
            return
        if self.question.multi_select:
            self.query_one("#ask-opt-0", Checkbox).focus()
            return
        listing = self.query_one("#ask-options", ListView)
        for option in self.options:
            listing.append(ListItem(Static(Content(option))))
        listing.index = 0
        listing.focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        index = event.list_view.index
        if index is not None and index < len(self.options):
            self.dismiss(self.options[index])

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        event.stop()
        chosen = [
            option
            for index, option in enumerate(self.options)
            if self.query_one(f"#ask-opt-{index}", Checkbox).value
        ]
        self.dismiss(", ".join(chosen))
