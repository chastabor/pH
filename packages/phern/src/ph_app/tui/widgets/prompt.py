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

from collections.abc import Callable, Sequence
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
from ..state import PromptRecord

__all__ = ["PromptArea", "PromptInput"]

PASTE_PLACEHOLDER_THRESHOLD = 2_000


@dataclass(frozen=True, slots=True)
class _Pasted:
    marker: str
    text: str


@dataclass(slots=True)
class _Walk:
    """A walk through history, and the draft it is standing on.

    Snapshotted when the walk starts rather than read per keypress: a turn that
    lands mid-walk would otherwise renumber the entries under the cursor, and the
    next `up` would move somewhere the person did not ask for.

    `draft` is what was in the box before the first `up`. It is also the *filter*
    — a non-empty draft walks only the prompts that start with it, which is what
    every shell does and the reason the ask said "search" rather than "cycle".
    """

    draft: str
    entries: tuple[PromptRecord, ...]
    index: int = -1
    """`-1` is the draft itself; `0` is the newest prompt."""


class PromptArea(TextArea):
    """The text box, with the prompt's bindings taking precedence.

    A `TextArea` consumes `enter` and `escape` in its own key handler before a
    parent ever runs, so submit and cancel cannot be handled by the container.
    They are decided here, in the public `on_key` hook: Textual walks a widget's
    handlers subclass-first and stops once `prevent_default()` has been called,
    so a key the prompt claims never reaches the editor, and one it does not
    claim is edited as usual.
    """

    def __init__(self, owner: PromptInput, **kwargs: Any) -> None:  # noqa: ANN401
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

    class Canceled(Message):
        """The user asked to interrupt the running turn."""

    def __init__(
        self,
        keybindings: TuiKeybindings,
        *,
        completion_source: Any = None,  # noqa: ANN401
        history_source: Callable[[], Sequence[PromptRecord]] | None = None,
    ) -> None:
        super().__init__(id="prompt")
        self.keys = keybindings
        self.completion_source = completion_source
        self.history_source = history_source
        """Where the prompts come from, asked when a walk begins.

        A callable for `completion_source`'s reason: the widget draws, and what
        the person has already sent is the app's to fold. Handed the whole list,
        not a cursor, so the widget owns only the walking."""
        self._pastes: list[_Pasted] = []
        self._completions: CompletionState | None = None
        self._walk: _Walk | None = None
        """The walk in progress, or `None` when the box is the person's own."""

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

    def replace(self, text: str) -> None:
        """Put `text` in the box, cursor at the end — the whole prompt, replaced.

        The public sibling of `text()` and `clear()`. `area.text = …` followed by
        `move_cursor(document.end)` was written in three places, one of them in
        `app.py` reaching across the widget boundary to do what the widget owns —
        and that copy was the one that forgot to close the completion list.
        """
        self._show(text)

    def clear(self) -> None:
        self.area.text = ""
        self._pastes.clear()
        self._walk = None
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
                self.post_message(self.Canceled())
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
            # **Only here**, which is what lets one key mean two things: with a
            # list open the branches below move the *list*, and `up` reaches
            # history only once there is no list to move.
            return self._walk_history(key)
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
        # Unconditional, because a recall does not reach here at all: `_show`
        # writes inside `prevent`, so the person typing is the only thing that
        # posts this. The walk is over — the next `up` starts again from what is
        # in the box now, which is also the new filter. Editing a recalled prompt
        # and keeping the walk position would mean `up` jumping from text nobody
        # can see any more.
        self._walk = None
        self._refresh_completions()

    # -------------------------------------------------------------- history --

    def _walk_history(self, key: str) -> bool:
        """Move through what this person has sent. `True` when the key is spent.

        Claimed **only at the edges of the box**: `up` on the first line, `down`
        on the last. Anywhere else a multi-line draft still navigates, which is
        the behavior a prompt box cannot give up to gain history.
        """
        keys = self.keys
        area = self.area
        if key == keys.history_previous and area.cursor_at_first_line:
            return self._step(1)
        if key == keys.history_next and area.cursor_at_last_line and self._walk is not None:
            # Not claimed when no walk is in progress: `down` on the last line of
            # an untouched box means nothing, and swallowing it would make the
            # key feel broken.
            return self._step(-1)
        return False

    def _step(self, step: int) -> bool:
        """Move `step` entries through the walk. `False` leaves the key alone.

        **An index delta, not a direction**: `+1` is one entry *older*, because
        that is the way the list runs (newest first) and naming it the other way
        made both call sites and the body each invert it once.
        """
        walk = self._walk
        if walk is None:
            if step < 0:
                return False  # nothing to come back down from
            walk = self._begin_walk()
            if walk is None:
                return False
        index = walk.index + step
        if index >= len(walk.entries):
            # Already at the oldest. Claimed anyway, so the cursor does not
            # silently jump out of the box at the end of the walk.
            return True
        if index < 0:
            # Back on the draft: the box is the person's again, and the walk ends
            # rather than sitting on a sentinel.
            self._show(walk.draft)
            self._walk = None
            return True
        walk.index = index
        self._show(walk.entries[index].text)
        return True

    def _begin_walk(self) -> _Walk | None:
        """Snapshot the history, filtered by whatever is already typed."""
        if self.history_source is None:
            return None
        draft = self.area.text
        prefix = draft.strip()
        entries = tuple(
            record
            for record in self.history_source()
            if not prefix or (record.text.startswith(prefix) and record.text != draft)
        )
        if not entries:
            return None
        self._walk = _Walk(draft=draft, entries=entries)
        return self._walk

    def _show(self, text: str) -> None:
        """Put `text` in the box without ending the walk that asked for it.

        **`prevent` rather than a flag or a remembered value**, which is Textual's
        own answer to exactly this and is the only one that is not a race: it adds
        the type to a stack that `post_message` consults *synchronously, at post
        time*, so the `Changed` message is never queued — where a flag set and
        cleared around the write is already back by the time a queued handler
        runs. Its own docstring's example is this line.

        It also makes `_set_completions(None)` mean what it says. Before, the
        `Changed` handler ran straight afterwards and `_refresh_completions` could
        re-open a list over a recalled prompt, so a recall had two writers of the
        completion state and the second one won.
        """
        with self.area.prevent(TextArea.Changed):
            self.area.text = text
            self.area.move_cursor(self.area.document.end)
        self._set_completions(None)

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
        self.replace(state.replace(chosen))
