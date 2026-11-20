"""The modal shapes that everything else in pH is built from.

`PhModal` carries the one thing that every dialog shares, which is a cancel
key. The key is a Textual binding with the id `cancel`, so `App.set_keymap`
rebinds it from `tui.json` together with every other binding. No modal compares
a key literal.

`ChoicePicker` picks one item of a list, and the list is filtered as the user
types. `ConfirmModal` shows a text and then asks for a decision. The two cover
every dialog that Phase 2 needs: model, session, theme, preset and project
trust. Five separate screens would be five places where the escape key, the
focus order and the filter rules could differ slightly.

@module ph_app.tui.modals.base
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, TypeVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Center, Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView, Static

from ...wire import matches_terms

__all__ = ["Action", "Answers", "Choice", "ChoicePicker", "ConfirmModal", "PhModal"]

ResultT = TypeVar("ResultT")


class PhModal(ModalScreen[ResultT]):
    """A modal with the cancel binding of pH. A dismissal returns `cancel_value`."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", id="cancel", show=False)
    ]

    cancel_value: Any = None
    _withdrawn = False

    def action_cancel(self) -> None:
        self.dismiss(self.cancel_value)

    def withdraw(self) -> None:
        """Take this modal down with `cancel_value`, wherever it is in the stack.

        `dismiss` gives its result to the waiter of this screen, but it pops
        whichever screen is on top. With two asks open at the same time, a
        withdrawal of the lower one took the upper one down instead, and the
        waiter of the upper ask never got an answer. A modal that is not on top
        is taken down when it reaches the top.
        """
        if self.is_active:
            self.dismiss(self.cancel_value)
        else:
            self._withdrawn = True

    def on_screen_resume(self) -> None:
        if self._withdrawn:
            self.dismiss(self.cancel_value)


@dataclass(frozen=True, slots=True)
class Choice:
    """One row in a picker."""

    value: str
    label: str
    detail: str = ""
    marked: bool = False
    """Rendered as the current selection: the active model, the live theme."""
    unavailable: str = ""
    """Why this row cannot be chosen. It is for a row that is worth showing but
    cannot be taken, such as the log of a sub-agent among the sessions. Choosing
    the row shows this text and leaves the picker open, so no caller receives a
    value that it would have to refuse."""

    def matches(self, needle: str) -> bool:
        return matches_terms(f"{self.label} {self.detail} {self.value}", needle)


class ChoicePicker(PhModal[str | None]):
    """Pick one value from a filtered list, or dismiss with nothing."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down", "cursor_down", "Next", id="completion_next", show=False),
        Binding("up", "cursor_up", "Previous", id="completion_previous", show=False),
    ]

    DEFAULT_CSS = """
    ChoicePicker { align: center middle; background: $background 60%; }
    ChoicePicker > #picker {
        width: 72; max-width: 90%; height: auto; max-height: 80%;
        background: $ph-panel; border: round $ph-accent; padding: 1 2;
    }
    ChoicePicker #picker-title { color: $ph-accent; }
    ChoicePicker #picker-filter { border: none; background: $ph-surface; margin: 1 0; }
    ChoicePicker #picker-list { height: auto; max-height: 16; background: $ph-panel; }
    ChoicePicker #picker-empty { color: $ph-muted; }
    """

    def __init__(
        self,
        *,
        title: str,
        choices: list[Choice],
        free_text: str = "",
        on_highlight: Callable[[str], None] | None = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.choices = choices
        self.free_text = free_text
        """When set, a filter that is not the value of any matching row is
        offered as a literal value, with the detail `<free_text>, as typed`. The
        user can still name a model id or a session id that pH has no catalog
        for."""
        self.on_highlight = on_highlight
        """Called with the value under the cursor as the cursor moves, so that a
        theme can be applied before it is chosen. The user can only pick a theme
        by seeing it."""
        self._visible: list[Choice] = list(choices)

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Static(
                Content.from_markup("[b]$title[/b]", title=self.title_text), id="picker-title"
            )
            yield Input(placeholder="filter…", id="picker-filter")
            yield ListView(id="picker-list")
            yield Static(Content(""), id="picker-empty")

    def on_mount(self) -> None:
        self._show(self.choices)
        self.query_one("#picker-filter", Input).focus()

    def _show(self, choices: list[Choice]) -> None:
        self._visible = choices
        listing = self.query_one("#picker-list", ListView)
        listing.clear()
        for choice in choices:
            listing.append(
                ListItem(
                    Static(
                        Content.from_markup(
                            "[$ph-accent]$mark[/] $label  [$ph-muted]$detail[/]",
                            mark="●" if choice.marked else " ",
                            label=choice.label,
                            detail=choice.detail,
                        )
                    )
                )
            )
        if choices:
            listing.index = next(
                (index for index, choice in enumerate(choices) if choice.marked), 0
            )
        self.query_one("#picker-empty", Static).update(
            Content("" if choices else "nothing matches")
        )

    async def on_input_changed(self, event: Input.Changed) -> None:
        needle = event.value.strip()
        matches = [choice for choice in self.choices if choice.matches(needle)]
        if self.free_text and needle and not any(choice.value == needle for choice in matches):
            matches.insert(
                0, Choice(value=needle, label=needle, detail=f"{self.free_text}, as typed")
            )
        self._show(matches)

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self._choose(self.query_one("#picker-list", ListView).index)

    async def on_list_view_highlighted(self, event: ListView.Highlighted) -> None:
        index = event.list_view.index
        if self.on_highlight is not None and index is not None and index < len(self._visible):
            self.on_highlight(self._visible[index].value)

    async def on_list_view_selected(self, event: ListView.Selected) -> None:
        self._choose(event.list_view.index)

    def action_cursor_down(self) -> None:
        self.query_one("#picker-list", ListView).action_cursor_down()

    def action_cursor_up(self) -> None:
        self.query_one("#picker-list", ListView).action_cursor_up()

    def _choose(self, index: int | None) -> None:
        if index is None or index >= len(self._visible):
            return
        choice = self._visible[index]
        if choice.unavailable:
            self.notify(choice.unavailable, severity="warning", markup=False)
            return
        self.dismiss(choice.value)


class Answers(Vertical):
    """The answers of a dialog: a column of buttons. Each button is centered, and
    the arrow keys move between the buttons.

    **Each button is in a `Center`.** `align-horizontal` on the column would
    center the answers as one block that is as wide as the widest button. A
    button with a shorter label would then sit against the left edge of the
    block.

    **The arrows are bound here, on the column, and not on the dialog.** They
    therefore act only while an answer has focus, and they move only among the
    answers. A text box handles Left and Right itself, but not Up and Down. If
    Up and Down were bound on the dialog, they would move focus out of the
    reason box of the approval dialog in the middle of a sentence. Tab still
    reaches the box. Down and Up have the ids of the picker (`completion_next`,
    `completion_previous`), so a rebinding in `tui.json` moves both kinds of
    dialog. Right and Left are fixed. They are a courtesy for a user who reads
    the answers as a row.
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("down", "next_answer", "Next", id="completion_next", show=False),
        Binding("up", "previous_answer", "Previous", id="completion_previous", show=False),
        Binding("right", "next_answer", "Next", show=False),
        Binding("left", "previous_answer", "Previous", show=False),
    ]

    DEFAULT_CSS = """
    Answers { height: auto; }
    """

    def __init__(self, *buttons: Button, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(*(Center(button) for button in buttons), **kwargs)

    def action_next_answer(self) -> None:
        self.screen.focus_next("Answers Button")

    def action_previous_answer(self) -> None:
        self.screen.focus_previous("Answers Button")


class Action(NamedTuple):
    """One button on a `ConfirmModal`."""

    value: str
    label: str
    variant: str = "default"


class ConfirmModal(PhModal[str | None]):
    """Show a body, offer named actions, and return the chosen one.

    The body is rendered as plain `Content`. A plan, a diff, a path and a
    command line all contain brackets, and none of them is markup.

    The answers are an `Answers` column. They are centered, and the arrow keys
    move between them as well as Tab.
    """

    DEFAULT_CSS = """
    ConfirmModal { align: center middle; background: $background 60%; }
    ConfirmModal > #confirm {
        width: 84; max-width: 90%; height: auto; max-height: 80%;
        background: $ph-panel; border: round $ph-warning; padding: 1 2;
    }
    ConfirmModal #confirm-title { color: $ph-warning; }
    ConfirmModal #confirm-body { height: auto; max-height: 20; margin: 1 0; }
    """

    def __init__(
        self,
        *,
        title: str,
        body: str,
        actions: list[Action],
        cancel_value: str | None = None,
        initial: str | None = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.body_text = body
        self.actions = actions
        self.cancel_value = cancel_value
        self.initial = initial if initial is not None else (actions[0].value if actions else None)
        """The answer that has focus when the dialog opens, which is what a bare
        Enter chooses. It is the first action unless the caller names a safer
        one."""

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Static(
                Content.from_markup("[b]$title[/b]", title=self.title_text), id="confirm-title"
            )
            yield Static(Content(self.body_text), id="confirm-body")
            yield Answers(
                *(
                    Button(action.label, id=f"act-{action.value}", variant=action.variant)  # type: ignore[arg-type]
                    for action in self.actions
                ),
                id="confirm-actions",
            )

    def on_mount(self) -> None:
        if self.initial is not None:
            self.query_one(f"#act-{self.initial}", Button).focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.dismiss(event.button.id.removeprefix("act-"))
