"""What the transcript *is*, independent of how Textual draws it.

A plain data model, for two reasons. It can be built and asserted without a
running app — which is what makes the resume gate testable — and it keeps the
"what happened" question separate from the "how does it look" one, so a widget
change cannot alter the transcript's meaning.

`ChatItem.key` is the identity a widget mounts against, so a `tool/call` row
becomes its own `tool/result` row rather than a second row beside it.

@module ph_app.tui.state
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

__all__ = ["ChatItem", "ItemRole", "ToolCard", "TuiState"]

ItemRole: TypeAlias = Literal[
    "user", "assistant", "thinking", "context", "tool", "notice", "error", "boundary", "compaction"
]


@dataclass(slots=True)
class ToolCard:
    """The durable facts a tool row renders from.

    Every field comes from the log — never from a live execution — so a replayed
    session draws the identical card (the reason `present_call`/`present_result`
    are pure).
    """

    call_id: str
    name: str
    arguments: str
    title: str = ""
    subtitle: str = ""
    card: str = "generic"
    """The tool's declared `CardKind`. Phase 2 draws every kind the same way;
    P3-19's code cell is the first widget that branches on it."""
    settled: bool = False
    is_error: bool = False
    failure_kind: str = ""
    body: str = ""
    dispatches: list[ToolCard] = field(default_factory=list)
    """Code Mode sub-dispatches, one row each (C2) — forty writes are forty rows."""


@dataclass(slots=True)
class ChatItem:
    """One transcript row."""

    key: str
    role: ItemRole
    text: str = ""
    streaming: bool = False
    tool: ToolCard | None = None
    turn: int = 0
    seq: int = -1
    shadowed: bool = False
    """Replaced by a compaction summary. Distinct from `role == "compaction"`,
    which is the summary itself: the summary is what the model sees *now*, and
    the shadowed rows are what it no longer sees. Conflating them would make a
    compaction indistinguishable from the history it stands in for."""

    @property
    def is_visible_to_model(self) -> bool:
        """Whether this row is part of what the model currently sees.

        A shadowed row stays in the transcript — a person already read it — but
        answers `False`, because the model no longer has it.
        """
        return not self.shadowed


@dataclass(slots=True)
class TuiState:
    """The whole front-end model: rows, live status, and the current posture."""

    items: list[ChatItem] = field(default_factory=list)
    status: Literal["idle", "running"] = "idle"
    turn: int = 0
    queued: int = 0
    preset: str = "read-only"
    sandbox_mode: str = "read-only"
    tokens: int = 0
    """What the last request cost, from the provider's own usage report — the
    same count `ctx.token_meter` calls its `usage` baseline."""
    context_window: int | None = None
    model: str = ""
    provider: str = ""
    todos: list[dict[str, Any]] = field(default_factory=list)
    _cards: dict[str, ToolCard] = field(default_factory=dict, repr=False)
    """Every tool card by call id — top-level calls and Code Mode sub-dispatches
    alike, so a `tool/code-dispatch` finds its row the way a `tool/result` does."""
    _streaming: dict[tuple[int, int], ChatItem] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ rows --

    def reset(self) -> None:
        """Clear everything the log determines, in place.

        In place, and that is the whole point: the app, the frontend and the
        adapter all hold *this* object. A replay that assigned a fresh
        `TuiState` would leave every other holder looking at the empty one —
        which is exactly how the first `--resume` came up blank.
        """
        self.items.clear()
        self._cards.clear()
        self._streaming.clear()
        self.todos.clear()
        self.status = "idle"
        self.turn = 0
        self.queued = 0
        self.tokens = 0
        self.context_window = None

    def add(self, item: ChatItem) -> ChatItem:
        self.items.append(item)
        return item

    def card(self, call_id: str) -> ToolCard | None:
        return self._cards.get(call_id)

    def register_card(self, card: ToolCard) -> ToolCard:
        self._cards[card.call_id] = card
        return card

    def streaming_item(self, turn: int, step: int) -> ChatItem | None:
        return self._streaming.get((turn, step))

    def begin_streaming(self, turn: int, step: int, item: ChatItem) -> ChatItem:
        self._streaming[(turn, step)] = item
        return self.add(item)

    def end_streaming(self, turn: int, step: int) -> ChatItem | None:
        item = self._streaming.pop((turn, step), None)
        if item is not None:
            item.streaming = False
        return item

    @property
    def pressure(self) -> float | None:
        """Fraction of the context window in use, when the window is known."""
        if not self.context_window:
            return None
        return self.tokens / self.context_window

    def visible_items(self, *, thinking: bool = True, tool_results: bool = True) -> list[ChatItem]:
        """The rows a given set of toggles shows."""
        rows: list[ChatItem] = []
        for item in self.items:
            if item.role == "thinking" and not thinking:
                continue
            if item.role == "tool" and not tool_results:
                continue
            rows.append(item)
        return rows
