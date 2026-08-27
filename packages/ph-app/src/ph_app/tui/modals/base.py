"""The modal shapes everything else in pH is built from.

`PhModal` carries the one thing every dialog shares — a cancel key — as a
Textual binding with the id `cancel`, so `App.set_keymap` rebinds it from
`tui.json` along with everything else. No modal compares a key literal.

`ChoicePicker` (pick one of a list, filtered as you type) and `ConfirmModal`
(read this, then decide) cover every dialog Phase 2 needs — model, session,
theme, preset, project trust, plan review. Writing five bespoke screens instead
would mean five places to get the escape key, the focus order and the filter
semantics subtly different.

@module ph_app.tui.modals.base
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, TypeVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Button, Input, ListItem, ListView, Static

__all__ = ["Action", "Choice", "ChoicePicker", "ConfirmModal", "PhModal"]

ResultT = TypeVar("ResultT")


class PhModal(ModalScreen[ResultT]):
    """A modal with pH's cancel binding. `cancel_value` is what dismissal returns."""

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel", "Cancel", id="cancel", show=False)
    ]

    cancel_value: Any = None

    def action_cancel(self) -> None:
        self.dismiss(self.cancel_value)


@dataclass(frozen=True, slots=True)
class Choice:
    """One row in a picker."""

    value: str
    label: str
    detail: str = ""
    marked: bool = False
    """Rendered as the current selection — the active model, the live theme."""

    def matches(self, needle: str) -> bool:
        if not needle:
            return True
        haystack = f"{self.label} {self.detail} {self.value}".lower()
        return all(part in haystack for part in needle.lower().split())


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
        """When set, an unmatched filter is offered as a literal value labelled
        `<free_text>, as typed` — a model id or session id pH has no catalogue
        for is still something the user can name."""
        self.on_highlight = on_highlight
        """Called with the value under the cursor as it moves, so a theme can be
        applied before it is chosen — seeing it is the only way to pick."""
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
        if index is not None and index < len(self._visible):
            self.dismiss(self._visible[index].value)


class Action(NamedTuple):
    """One button on a `ConfirmModal`."""

    value: str
    label: str
    variant: str = "default"


class ConfirmModal(PhModal[str | None]):
    """Show a body, offer named actions, return the chosen one.

    The body is rendered as plain `Content`: a plan, a diff, a path, a command
    line — every one of them contains brackets, and none of them is markup.
    """

    DEFAULT_CSS = """
    ConfirmModal { align: center middle; background: $background 60%; }
    ConfirmModal > #confirm {
        width: 84; max-width: 90%; height: auto; max-height: 80%;
        background: $ph-panel; border: round $ph-warning; padding: 1 2;
    }
    ConfirmModal #confirm-title { color: $ph-warning; }
    ConfirmModal #confirm-body { height: auto; max-height: 20; margin: 1 0; }
    ConfirmModal #confirm-actions { height: auto; }
    ConfirmModal Button { margin-right: 1; }
    """

    def __init__(
        self,
        *,
        title: str,
        body: str,
        actions: list[Action],
        cancel_value: str | None = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.body_text = body
        self.actions = actions
        self.cancel_value = cancel_value

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm"):
            yield Static(
                Content.from_markup("[b]$title[/b]", title=self.title_text), id="confirm-title"
            )
            yield Static(Content(self.body_text), id="confirm-body")
            with Vertical(id="confirm-actions"):
                for action in self.actions:
                    yield Button(action.label, id=f"act-{action.value}", variant=action.variant)  # type: ignore[arg-type]

    def on_mount(self) -> None:
        if self.actions:
            self.query_one(f"#act-{self.actions[0].value}", Button).focus()

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        assert event.button.id is not None
        self.dismiss(event.button.id.removeprefix("act-"))
