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
from enum import IntFlag
from typing import Any, Literal, TypeAlias

from ph.json import JsonValue, as_bool, as_str

from ..payloads import DaemonLifetime

__all__ = [
    "CatalogEntry",
    "ChatItem",
    "ItemRole",
    "PromptRecord",
    "SubagentRow",
    "Surface",
    "ToolCard",
    "TuiState",
]


class Surface(IntFlag):
    """Which part of the screen a change reaches.

    The redraw was one boolean, so every event redrew everything: an
    `assistant/chunk` — which arrives faster than the coalescing window during a
    streaming turn, and cannot change a sidebar — re-ran the session panel, the
    todo list and the subagent fold thirty times a second. A flag lets the draw
    ask what actually moved.

    `ALL` is the default everywhere, and deliberately: a surface that should
    have been redrawn and was not is a pane showing yesterday's answer, which is
    far worse than a redraw nobody needed. An event type earns a narrower
    entry on its own `ph_app.tui.adapter.EventRule`, beside the handler that
    does the folding.
    """

    NOTHING = 0
    """Named rather than `Surface(0)`, because a dataclass default may not be a
    call and a zero flag is worth being able to say out loud."""

    TRANSCRIPT = 1
    FOOTER = 2
    SIDEBAR = 4
    ALL = TRANSCRIPT | FOOTER | SIDEBAR


ItemRole: TypeAlias = Literal[
    "user", "assistant", "thinking", "context", "tool", "notice", "error", "boundary", "compaction"
]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One entry of what this deployment offers the model: a tool, or a skill.

    One type for both because the two panels ask the same two questions of
    them — what is it called, and what is it for — and a `ToolSchema` beside a
    `Skill` would make every renderer here branch on which it had for a field
    both carry. The parts that differ (a tool's parameters, a skill's path and
    version) are not what a catalog panel shows, so they are not carried.
    """

    name: str
    description: str = ""


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
    details: dict[str, JsonValue] = field(default_factory=dict)
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
    "canceled": "⊘",
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


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One prompt a person sent, as the history walk and the picker show it.

    A projection of the `user` rows of `TuiState.items` rather than a second
    store — see `TuiState.prompt_history`. `seq` is what lets a chosen row be
    *revealed* in the transcript, which is the half of this that a history file
    could not do; it is `-1` for a row the log has no position for.
    """

    text: str
    seq: int = -1
    turn: int = 0


@dataclass(slots=True)
class TuiState:
    """The whole front-end model: rows and live status.

    The posture is no longer here — it is a `StatusReading` the seams that own
    it contribute, which the footer and the session panel place by slot."""

    items: list[ChatItem] = field(default_factory=list)
    status: str = "idle"
    """The root's own word: `idle`, `running`, `waiting`, `retrying`,
    `passivated`. A `str` rather than a two-valued literal because a daemon's
    session has more states than a spinner does, and the alternative — a second
    status field on the remote front end, kept in step by hand — was three
    writers of one fact. Widgets read `busy`, which is the bool they wanted."""
    turn: int = 0
    queued: int = 0
    tools: tuple[CatalogEntry, ...] = ()
    """What the model may call here, as `tools/list` projected it.

    Read at draw time rather than folded from events, because it is not in the
    log: a row registers its tools at mount, so this is a fact about the
    deployment the session is running under and not about the session. The
    panel draws the names and `/tools` draws the descriptions — one read, two
    renderings, which is why the description is carried rather than dropped at
    the wire edge."""
    skills: tuple[CatalogEntry, ...] = ()
    """What is installed here, from `skills/list`. `tools`' reasoning, and the
    same catalog the model's own prompt is built from."""
    tokens: int = 0
    """What the last request cost, from the provider's own usage report — the
    same count `ctx.token_meter` calls its `usage` baseline."""
    context_window: int | None = None
    model: str = ""
    provider: str = ""
    lifetime: DaemonLifetime | None = None
    """Why the daemon behind this session is still running, or `None`.

    `None` for a front end that is not on a daemon at all, and the sidebar draws
    no line then: "this process will exit when you close it" is not news about a
    process the person is looking at.

    The wire model itself rather than a copy of its fields: it is read once at
    attach and replaced whole whenever `daemon.lifetime` says the answer moved,
    so a second shape here would be a second declaration of three fields with
    nothing checking they agree."""
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
            row.name = as_str(entry.get("name") or run_id)
            row.status = as_str(entry.get("status"), "queued")
            row.model = as_str(entry.get("model"))
            row.cause = as_str(entry.get("cause"))
            row.deleted = as_bool(entry.get("deleted"))

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

    def prompt_history(self) -> list[PromptRecord]:
        """Every prompt this person sent, newest first.

        **The mirror is the history**, which is why nothing is stored: these rows
        are already here because the transcript is built from them, so the list
        is correct across a resume and costs nothing to keep. A shell-style
        history file would be a second copy of the log, and one that could not
        say *where* in the conversation a prompt sits.

        Consecutive duplicates collapse. Sending the same thing twice in a row is
        a person retrying, and two identical rows in a walk read as the arrow key
        having missed.

        `shadowed` rows are kept: compaction hides a prompt from the *model*, and
        this list is for the person, who still typed it.

        **Not enforced (§5 rule 6): this is *this session's* history, not this
        person's.** It is a fold over the mirror, so it holds exactly what the
        mirror holds — a resume gets that session's prompts back, and a
        reference-forked child gets only its own, because such a child holds only
        its own events. That is the same fact `ph_app.sessions` works around when
        a forked row would otherwise render with no title. A history spanning
        every session would be the second copy of the log this docstring opens by
        refusing.
        """
        history: list[PromptRecord] = []
        for item in reversed(self.items):
            if item.role != "user" or not item.text.strip():
                continue
            if history and history[-1].text == item.text:
                continue
            history.append(PromptRecord(text=item.text, seq=item.seq, turn=item.turn))
        return history

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
