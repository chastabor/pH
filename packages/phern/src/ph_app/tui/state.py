"""What the transcript *is*, independent of how Textual draws it.

This is a plain data model, for two reasons. It can be built and checked without
a running app, which is what makes the resume gate testable. It also keeps the
question "what happened" separate from the question "how does it look", so a
widget change cannot alter the meaning of the transcript.

`ChatItem.key` is the identity that a widget mounts against. A `tool/call` row
therefore becomes its own `tool/result` row, and not a second row beside it.

@module ph_app.tui.state
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import IntFlag
from typing import Any, Literal, TypeAlias

from ph.json import JsonValue
from ph.seams.subagents import child_is_live

from ..payloads import ChildRow, DaemonLifetime

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

    The redraw used to be one boolean, so every event redrew everything. An
    `assistant/chunk` arrives faster than the coalescing window during a
    streaming turn, and it cannot change a sidebar. It still ran the session
    panel, the todo list and the subagent fold again, thirty times a second.
    With a flag, the draw can ask what moved.

    `ALL` is the default everywhere, and that is deliberate. A surface that
    should have been redrawn and was not is a pane that shows an old answer.
    That is much worse than a redraw that nobody needed. An event type gets a
    narrower entry on its own `ph_app.tui.adapter.EventRule`, beside the handler
    that does the fold.
    """

    NOTHING = 0
    """Named, and not written as `Surface(0)`, because a dataclass default may not
    be a call, and because a zero flag is worth a name."""

    TRANSCRIPT = 1
    FOOTER = 2
    SIDEBAR = 4
    ALL = TRANSCRIPT | FOOTER | SIDEBAR


ItemRole: TypeAlias = Literal[
    "user", "assistant", "thinking", "context", "tool", "notice", "error", "boundary", "compaction"
]


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    """One entry of what this deployment offers the model: a tool or a skill.

    One type serves both, because the two panels ask the same two questions of
    them: what is it called, and what is it for. With a `ToolSchema` beside a
    `Skill`, every renderer here would branch on which one it had, for a field
    that both carry. The parts that differ (the parameters of a tool, the path
    and version of a skill) are not what a catalog panel shows, so they are not
    carried.
    """

    name: str
    description: str = ""


@dataclass(slots=True)
class ToolCard:
    """The durable facts that a tool row renders from.

    Every field comes from the log and never from a live execution, so a
    replayed session draws the identical card. That is the reason why
    `present_call` and `present_result` are pure.
    """

    call_id: str
    name: str
    arguments: str
    title: str = ""
    subtitle: str = ""
    card: str = "generic"
    """The declared `CardKind` of the tool. The `terminal` kind is the code cell
    (P3-19). Every other kind draws the same way."""
    input_text: str = ""
    """The full input of the call, when the tool offered one
    (`ToolCallView.body`): the program of a cell. It is kept apart from
    `arguments`, which is the raw JSON that the model emitted and may not even
    parse."""
    details: dict[str, JsonValue] = field(default_factory=dict)
    """The durable presentation payload of the tool, passed through verbatim from
    `tool/result.meta`. For a cell it is `CellDetails`. The card shows what it
    understands and ignores the rest, so a tool can enrich its own card and the
    transcript does not learn its schema."""
    settled: bool = False
    is_error: bool = False
    failure_kind: str = ""
    body: str = ""
    dispatches: list[ToolCard] = field(default_factory=list)
    """Code Mode sub-dispatches, one row each (C2). Forty writes are forty rows."""


STATUS_GLYPHS: dict[str, str] = {
    "queued": "○",
    "running": "◐",
    "done": "●",
    "error": "✗",
    "canceled": "⊘",
}


@dataclass(frozen=True, slots=True)
class SubagentRow:
    """One sub-agent as the panel draws it: the row from the daemon, and its depth.

    **This holds the wire row itself and not a copy of its fields** (P11-09),
    for the reason that `TuiState.lifetime` holds its model whole. The daemon
    folds the log of each child (`ChildRow`). A second shape here would be a
    second declaration of fourteen fields, with nothing to check that the two
    agree. The panel adds only what drawing needs: the glyph, and `depth`. The
    depth is the tree of the family read from `parent_id`, so a grandchild is
    indented under the child that spawned it.
    """

    child: ChildRow
    depth: int = 0
    """0 for a child of the root, 1 for a grandchild, and so on."""

    @property
    def session_id(self) -> str:
        """The log of the child. The panel keys the row by it. It is unique at every
        depth."""
        return self.child.session_id

    @property
    def live(self) -> bool:
        """Whether the child is still working: not revoked, and not settled. This is
        `child_is_live` applied to the row, so the panel and the seam cannot
        disagree."""
        return child_is_live(self.child)

    @property
    def glyph(self) -> str:
        return "⊘" if self.child.deleted else STATUS_GLYPHS.get(self.child.status, "○")


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
    """Whether a compaction summary replaced this row. This is distinct from
    `role == "compaction"`, which is the summary itself. The summary is what the
    model sees *now*, and the shadowed rows are what it no longer sees. If the
    two were merged, a compaction could not be told apart from the history that
    it stands in for."""

    @property
    def is_visible_to_model(self) -> bool:
        """Whether this row is part of what the model currently sees.

        A shadowed row stays in the transcript, because a person already read
        it. It answers `False`, because the model no longer has it.
        """
        return not self.shadowed


@dataclass(frozen=True, slots=True)
class PromptRecord:
    """One prompt that a person sent, as the history walk and the picker show it.

    This is a projection of the `user` rows of `TuiState.items` and not a second
    store: see `TuiState.prompt_history`. With `seq`, a chosen row can be
    *revealed* in the transcript, which a history file could not do. `seq` is
    `-1` for a row that the log has no position for.
    """

    text: str
    seq: int = -1
    turn: int = 0


@dataclass(slots=True)
class TuiState:
    """The whole front-end model: rows and live status.

    The posture is no longer here. It is a `StatusReading` that the seams which
    own it contribute, and the footer and the session panel place it by slot."""

    items: list[ChatItem] = field(default_factory=list)
    status: str = "idle"
    """The status word of the root, for example `idle`, `running`, `waiting`,
    `retrying`, `failed` or `passivated`. It is a `str` and not a two-valued
    literal, because the session of a daemon has more states than a spinner
    does. The alternative was a second status field on the remote front end,
    kept in step by hand, which made three writers of one fact. Widgets read
    `busy`, which is the bool that they wanted."""
    turn: int = 0
    queued: int = 0
    tools: tuple[CatalogEntry, ...] = ()
    """What the model may call here, as `tools/list` projected it.

    It is read from the daemon at attach and not folded from events, because it
    is not in the log. A row registers its tools at mount, so this is a fact
    about the deployment that the session runs under, and not about the session.
    The panel draws the names and `/tools` draws the descriptions: one read, two
    renderings. That is why the description is carried and not dropped at the
    wire edge."""
    skills: tuple[CatalogEntry, ...] = ()
    """What is installed here, from `skills/list`. The reasoning of `tools`
    applies. It is the same catalog that the prompt of the model is built from."""
    tokens: int = 0
    """What the last request cost, from the usage report of the provider. It is the
    same count that `ctx.token_meter` calls its `usage` baseline."""
    context_window: int | None = None
    compacts_at: float | None = None
    """The share of the context window at which the root compacts on its own, or
    `None` when nothing compacts on pressure. The daemon states it in the status
    of the root (`CompactionFacts`), from the compaction seam of the root. It is
    not in the log, so `reset` leaves it as it is."""
    model: str = ""
    provider: str = ""
    model_key: str = ""
    """The name that the profile gives the route in force, or `""` when it lists no
    match."""

    @property
    def route(self) -> str:
        """The `provider/model` in force, in the spelling that `/model` accepts. It
        is `""` before attach."""
        return f"{self.provider}/{self.model}" if self.provider else ""

    lifetime: DaemonLifetime | None = None
    """Why the daemon behind this session is still running, or `None`.

    It is `None` for a front end that is not on a daemon at all, and the sidebar
    then draws no line. "This process will exit when you close it" is not news
    about a process that the user is looking at.

    This holds the wire model itself and not a copy of its fields. It is read
    one time at attach and replaced whole whenever `daemon.lifetime` says that
    the answer changed. A second shape here would be a second declaration of
    four fields, with nothing to check that the two agree."""
    todos: list[dict[str, Any]] = field(default_factory=list)
    subagents: dict[str, SubagentRow] = field(default_factory=dict)
    """Every sub-agent beneath this session, by session id, with a parent before
    its own children. It is the `session.children` of the daemon, drawn
    (P11-09).

    **It is not folded from the log**, and it cannot be. The log of a root holds
    no record of its children (Phase 11), so the daemon reads the log of each
    child and sends the family whole (`take_children`). For the same reason
    `reset` leaves it alone: a replay of the log does not determine it. It is a
    panel beside the transcript and not rows inside it, because eight children
    that move through `queued → running → done` would push the conversation off
    the screen."""
    _cards: dict[str, ToolCard] = field(default_factory=dict, repr=False)
    """Every tool card by call id: top-level calls and Code Mode sub-dispatches
    alike. A `tool/code-dispatch` therefore finds its row in the way that a
    `tool/result` does."""
    _streaming: dict[tuple[int, int, ItemRole], ChatItem] = field(default_factory=dict, repr=False)
    """The open streaming row for each `(turn, step, role)`. **The role is part of
    the key** (H5).

    One step can stream reasoning and then text. When the key was the step
    alone, the second role evicted the first from this map. Nothing then settled
    the evicted row, and it stayed `streaming=True` forever. In the transcript
    that is a `StreamingMessage` that holds a live `MarkdownStream` task: one
    for each reasoning block, for the life of the app. It is also a thinking row
    that never stopped animating, on a turn that had finished long before."""

    def take_children(self, rows: Sequence[ChildRow]) -> None:
        """The children list from the daemon, drawn whole (P11-09).

        **It is replaced and never merged.** Each `session.children` frame is
        the whole family, so a row that is missing from it is a row that the
        daemon no longer lists, and a merge would keep it. The replacement is in
        place, for the reason that `reset` gives: the app holds this dict. Depth
        is read from `parent_id` in one pass. The order of the daemon (a parent
        before its own children) makes that possible. A row whose parent is not
        listed is a child of the root.
        """
        depth: dict[str, int] = {}
        self.subagents.clear()
        for row in rows:
            level = depth[row.parent_id] + 1 if row.parent_id in depth else 0
            depth[row.session_id] = level
            self.subagents[row.session_id] = SubagentRow(child=row, depth=level)

    # ------------------------------------------------------------------ rows --

    def stand_down(self) -> None:
        """Stop claiming work that is not happening, because the daemon has gone.

        This is one member because two callers reach it from two layers. One is
        the `submit` of the front end, whose optimistic `running` started the
        spinner. The other is the watcher of the app, which is the only thing
        that hears about a turn that somebody *else* started. When the rule was
        written inline in both, the two guards had already drifted: one
        preserved `failed` and the other did not. The value that survived
        depended on which guard ran last.

        `failed` is kept because it is a fact that the daemon reported, and
        nothing here has learned otherwise. Every other status is a claim about
        work in flight, and there is none.
        """
        if self.status != "failed":
            self.status = "idle"

    @property
    def busy(self) -> bool:
        """Whether the spinner should turn: work is in flight, or is about to be.

        `retrying` counts. A root in the backoff of P5-04 is between attempts,
        and it is not done. `waiting` does not count. A root that is parked on a
        person waits for the screen, and a spinner over a modal says the wrong
        thing.
        """
        return self.status in ("running", "retrying")

    def reset(self) -> None:
        """Clear everything that the log determines, in place.

        It is in place on purpose: the app, the front end and the adapter all
        hold *this* object. A replay that assigned a new `TuiState` would leave
        every other holder with the empty one. That is exactly how the first
        `--resume` came up blank.
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

    def streaming_item(self, turn: int, step: int, role: ItemRole) -> ChatItem | None:
        return self._streaming.get((turn, step, role))

    def begin_streaming(self, turn: int, step: int, role: ItemRole, item: ChatItem) -> ChatItem:
        self._streaming[(turn, step, role)] = item
        return self.add(item)

    def end_streaming(self, turn: int, step: int | None = None) -> list[ChatItem]:
        """Settle open rows: the rows of one step, or of every step of this turn.

        All the roles of a step are settled together, because they finish
        together. The `assistant/message` that ends a step is everything that
        the step produced, reasoning and text alike. A row that is left open is
        a row that still animates.

        **`step=None` is the end of the turn itself, and it is the backstop.** A
        step that raised (a request error with no retry) never appends an
        `assistant/message`. Nothing then settles what it had already streamed,
        and the row animates for the life of the app. That is the same leak
        that the role key closes on the ordinary path, and it arrives here by
        another path.
        """
        keys = [
            key for key in self._streaming if key[0] == turn and (step is None or key[1] == step)
        ]
        settled = []
        for key in keys:
            item = self._streaming.pop(key)
            item.streaming = False
            settled.append(item)
        return settled

    @property
    def pressure(self) -> float | None:
        """Fraction of the context window in use, when the window is known."""
        if not self.context_window:
            return None
        return self.tokens / self.context_window

    def prompt_history(self) -> list[PromptRecord]:
        """Every prompt that this person sent, newest first.

        **The mirror is the history**, so nothing is stored. These rows are
        already here, because the transcript is built from them. The list is
        therefore correct across a resume, and it costs nothing to keep. A
        shell-style history file would be a second copy of the log, and it could
        not say *where* in the conversation a prompt sits.

        Consecutive duplicates collapse. A person who sends the same thing twice
        in a row is retrying, and two identical rows in a walk look as if the
        arrow key had missed.

        `shadowed` rows are kept. Compaction hides a prompt from the *model*,
        and this list is for the person, who still typed it.

        **Not enforced (§5 rule 6): this is the history of *this session*, and
        not of this person.** It is a fold over the mirror, so it holds exactly
        what the mirror holds. A resume gets the prompts of that session back. A
        reference-forked child gets only its own, because such a child holds
        only its own events. `ph_app.sessions` works around the same fact where
        a forked row would otherwise render with no title. A history that
        spanned every session would be the second copy of the log that the start
        of this docstring refuses.
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
