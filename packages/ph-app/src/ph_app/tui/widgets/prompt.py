"""The prompt row: input, autocomplete, and the paste placeholder.

`PromptInput` deliberately does not decide what a key *means* — it reads the
configured binding and posts an intent. A widget that compared
`event.key == "escape"` would silently ignore a user's rebinding, which is the
rule prime-agent states and pH adopts (see `ph_app.tui.config`).

A large paste becomes a placeholder rather than thousands of lines in the box:
the text is kept, the display is not, because a terminal re-rendering a 200 KB
paste per keystroke is unusable.

@module ph_app.tui.widgets.prompt
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.message import Message
from textual.widgets import ListItem, ListView, Static, TextArea

from ..autocomplete import CompletionState, build_completion_state
from ..config import TuiKeybindings

__all__ = ["PromptArea", "PromptInput"]

PASTE_PLACEHOLDER_THRESHOLD = 2_000


@dataclass(frozen=True, slots=True)
class _Pasted:
    marker: str
    text: str


class PromptArea(TextArea):
    """The text box, with the prompt's bindings taking precedence.

    A `TextArea` consumes `enter` and `escape` in its own key handler before a
    parent ever runs, so submit and cancel cannot be handled by the container.
    They are decided here, in the public `on_key` hook: Textual walks a widget's
    handlers subclass-first and stops once `prevent_default()` has been called,
    so a key the prompt claims never reaches the editor, and one it does not
    claim is edited as usual.
    """

    def __init__(self, owner: PromptInput, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.owner = owner
        # `tab` accepts a completion, so it must not insert indentation. With a
        # completion list open the prompt claims it first; with none open,
        # moving focus is the sensible remainder.
        self.tab_behavior = "focus"
        self.show_line_numbers = False

    def on_key(self, event: events.Key) -> None:
        self.owner.intercept_key(event)


class PromptInput(Vertical):
    """A multi-line prompt box with a completion list under it."""

    DEFAULT_CSS = """
    PromptInput { height: auto; max-height: 16; }
    PromptInput > TextArea {
        height: auto; max-height: 10; border: round $ph-border; background: $ph-surface;
    }
    PromptInput > TextArea:focus { border: round $ph-accent; }
    PromptInput > #completions {
        height: auto; max-height: 6; display: none; background: $ph-panel; border: round $ph-border;
    }
    PromptInput.-completing > #completions { display: block; }
    PromptInput > #hint { color: $ph-muted; }
    """

    class Submitted(Message):
        """The user asked to send this text."""

        def __init__(self, text: str, *, queue: bool) -> None:
            super().__init__()
            self.text = text
            self.queue = queue

    class Cancelled(Message):
        """The user asked to interrupt the running turn."""

    def __init__(self, keybindings: TuiKeybindings, *, completion_source: Any = None) -> None:
        super().__init__(id="prompt")
        self.keys = keybindings
        self.completion_source = completion_source
        self._pastes: list[_Pasted] = []
        self._completions: CompletionState | None = None

    def compose(self) -> ComposeResult:
        yield PromptArea(self, id="prompt-input", soft_wrap=True)
        yield ListView(id="completions")
        yield Static(Content(""), id="hint")

    # ----------------------------------------------------------------- text --

    @property
    def area(self) -> PromptArea:
        return self.query_one("#prompt-input", PromptArea)

    def text(self) -> str:
        """The prompt as the harness should see it, with pastes restored."""
        text = self.area.text
        for paste in self._pastes:
            text = text.replace(paste.marker, paste.text)
        return text

    def clear(self) -> None:
        self.area.text = ""
        self._pastes.clear()
        self._set_completions(None)

    async def on_paste(self, event: events.Paste) -> None:
        """Keep a large paste out of the display but in the prompt."""
        if len(event.text) < PASTE_PLACEHOLDER_THRESHOLD:
            return
        event.prevent_default()
        event.stop()
        lines = event.text.count("\n") + 1
        marker = f"[#pasted-{len(self._pastes) + 1}: {lines} lines, {len(event.text)} chars]"
        self._pastes.append(_Pasted(marker=marker, text=event.text))
        self.area.insert(marker)

    # ------------------------------------------------------------------ keys --

    def intercept_key(self, event: events.Key) -> bool:
        """Decide the key against the configured bindings; consume it if claimed.

        Never compares a key literal, so a rebound submit or cancel works
        everywhere. (Not named `handle_key` — that is Textual's own dispatch
        hook on `Widget`, and shadowing it breaks every unclaimed key.)
        """
        claimed = self._decide(event.key)
        if claimed:
            event.stop()
            event.prevent_default()
        return claimed

    def _decide(self, key: str) -> bool:
        keys = self.keys
        if key == keys.cancel:
            # An open completion list is what escape closes first; only an
            # already-closed list means "interrupt the turn".
            if self._completions is not None:
                self._set_completions(None)
            else:
                self.post_message(self.Cancelled())
            return True
        if key == keys.submit and self._completions is not None:
            self._accept_completion()
            return True
        if key in (keys.submit, keys.queue_follow_up):
            text = self.text().strip()
            if text:
                self.post_message(self.Submitted(text, queue=key == keys.queue_follow_up))
                self.clear()
            return True
        if self._completions is None:
            return False
        if key == keys.accept_completion:
            self._accept_completion()
            return True
        if key in (keys.completion_next, keys.completion_previous):
            listing = self.query_one("#completions", ListView)
            if key == keys.completion_next:
                listing.action_cursor_down()
            else:
                listing.action_cursor_up()
            return True
        return False

    async def on_text_area_changed(self, _event: TextArea.Changed) -> None:
        self._refresh_completions()

    # ---------------------------------------------------------- completions --

    def _refresh_completions(self) -> None:
        if self.completion_source is None:
            return
        state = build_completion_state(self.area.text, **self.completion_source())
        self._set_completions(state if state.items else None)

    def _set_completions(self, state: CompletionState | None) -> None:
        self._completions = state
        listing = self.query_one("#completions", ListView)
        listing.clear()
        self.set_class(state is not None, "-completing")
        if state is None:
            self.query_one("#hint", Static).update(Content(""))
            return
        for item in state.items:
            listing.append(
                ListItem(
                    Static(
                        Content.from_markup(
                            "[b]$label[/b] [$ph-muted]$detail[/]",
                            label=item.label,
                            detail=item.detail,
                        )
                    )
                )
            )
        listing.index = 0
        self.query_one("#hint", Static).update(
            Content.from_markup(
                "[$ph-muted]$hint[/]", hint=f"{self.keys.accept_completion} to accept"
            )
        )

    def _accept_completion(self) -> None:
        state = self._completions
        listing = self.query_one("#completions", ListView)
        if state is None or listing.index is None or listing.index >= len(state.items):
            return
        chosen = state.items[listing.index]
        self.area.text = state.replace(chosen)
        self.area.move_cursor(self.area.document.end)
        self._set_completions(None)
