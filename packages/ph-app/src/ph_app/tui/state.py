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

__all__ = ["ChatItem", "ItemRole", "SubagentRow", "ToolCard", "TuiState"]

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
    """The tool's declared `CardKind`. The `terminal` kind is the code cell
    (P3-19); every other kind draws the same way."""
    input_text: str = ""
    """The call's full input, when the tool offered one (`ToolCallView.body`) —
    a cell's program. Kept apart from `arguments`, which is the raw JSON the
    model emitted and may not even parse."""
    details: dict[str, Any] = field(default_factory=dict)
    """The tool's own durable presentation payload, threaded verbatim from
    `tool/result.meta` — for a cell, `IpythonToolDetails`. The card shows what it
    understands and ignores the rest, so a tool can enrich its own card without
    the transcript learning its schema."""
    settled: bool = False
    is_error: bool = False
    failure_kind: str = ""
    body: str = ""
    dispatches: list[ToolCard] = field(default_factory=list)
    """Code Mode sub-dispatches, one row each (C2) — forty writes are forty rows."""


STATUS_GLYPHS: dict[str, str] = {
    "queued": "○",
    "running": "◐",
    "done": "●",
    "error": "✗",
    "cancelled": "⊘",
}


@dataclass(slots=True)
class SubagentRow:
    """One child, as the panel shows it.

    The *drawn projection* of one row of `TuiState.roster`, which the adapter
    folds through the seam's own `fold_subagent_event`. Keeping the fold in the
    seam and the drawing here is what stops the panel from becoming a second
    projection of one fold (A11) — the failure the seam was factored out to
    prevent. `tokens` is the one field that is genuinely the panel's: usage is
    attributed per child message and is not a roster fact.
    """

    run_id: str
    name: str = ""
    status: str = "queued"
    model: str = ""
    cause: str = ""
    """Why it is in that status — `rehydrated` for a settled child woken by a
    message, which still reads as `running` (P3-13)."""
    deleted: bool = False
    tokens: int = 0

    @property
    def glyph(self) -> str:
        return "⊘" if self.deleted else STATUS_GLYPHS.get(self.status, "○")


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
    status: str = "idle"
    """The root's own word: `idle`, `running`, `waiting`, `retrying`,
    `passivated`. A `str` rather than a two-valued literal because a daemon's
    session has more states than a spinner does, and the alternative — a second
    status field on the remote front end, kept in step by hand — was three
    writers of one fact. Widgets read `busy`, which is the bool they wanted."""
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
    roster: dict[str, dict[str, Any]] = field(default_factory=dict)
    """The seam's own fold of `subagent/*`, kept verbatim so the panel and the
    roster the model reads are one projection rather than two."""
    subagents: dict[str, SubagentRow] = field(default_factory=dict)
    """Children by run id, in admission order — the drawn view of `roster`. A
    live projection beside the transcript rather than rows inside it: eight
    children ticking through `queued → running → done` would push the
    conversation off screen."""
    _cards: dict[str, ToolCard] = field(default_factory=dict, repr=False)
    """Every tool card by call id — top-level calls and Code Mode sub-dispatches
    alike, so a `tool/code-dispatch` finds its row the way a `tool/result` does."""
    _streaming: dict[tuple[int, int], ChatItem] = field(default_factory=dict, repr=False)

    def sync_subagents(self) -> None:
        """Bring the drawn rows in line with the folded roster.

        `tokens` survives, because it is the one field the fold does not carry.
        """
        for run_id, entry in self.roster.items():
            row = self.subagents.get(run_id)
            if row is None:
                row = self.subagents[run_id] = SubagentRow(run_id=run_id)
            row.name = str(entry.get("name") or run_id)
            row.status = str(entry.get("status") or "queued")
            row.model = str(entry.get("model") or "")
            row.cause = str(entry.get("cause") or "")
            row.deleted = bool(entry.get("deleted"))

    # ------------------------------------------------------------------ rows --

    @property
    def busy(self) -> bool:
        """Whether the spinner should turn: work is in flight, or about to be.

        `retrying` counts: a root in P5-04's backoff is between attempts, not
        done. `waiting` does not: a root parked on a person is waiting for the
        screen, and a spinner over a modal says the wrong thing.
        """
        return self.status in ("running", "retrying")

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
        self.roster.clear()
        self.subagents.clear()
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
