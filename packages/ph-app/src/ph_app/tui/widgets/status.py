"""Footer and sidebar: what is happening, and under what posture.

The context reading turns amber at the compaction threshold rather than at some
round number, because the number a user needs to see coming is the one where
the harness will act (G4's 0.85 fraction, Phase 4).

@module ph_app.tui.widgets.status
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Static

from ph.seams.tui_status import StatusReading

from ..state import TuiState

__all__ = ["COMPACTION_THRESHOLD", "Sidebar", "StatusBar", "render_subagents"]

COMPACTION_THRESHOLD = 0.85
"""Where Phase 4's `compaction-summarize` triggers. The gauge warns here."""

SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_HOME = str(Path.home())


def render_subagents(state: TuiState) -> str:
    """The delegation panel: one line per child, admission order (P3-19).

    A *panel*, not transcript rows, because a fan-out of eight ticking through
    `queued → running → done` would push the conversation off screen — and
    because the interesting thing about a family is its current shape, which is
    a projection rather than a history. Folded from the same `subagent/*` events
    `subagent_roster` folds; a tombstoned child stays listed, since a parent
    asking what happened to the one it revoked deserves an answer.
    """
    lines: list[str] = []
    for row in state.subagents.values():
        detail = row.cause or (row.model if row.status == "queued" else row.status)
        tokens = f" {row.tokens // 1000}k" if row.tokens >= 1000 else ""
        lines.append(f"{row.glyph} {row.name} {detail}{tokens}".rstrip())
    return "\n".join(lines)


class StatusBar(Vertical):
    """One line: spinner, model, posture, queue depth, context.

    Owns the spinner frame, so the footer and the terminal title read the same
    glyph from one counter.
    """

    DEFAULT_CSS = """
    StatusBar { height: 1; background: $ph-panel; }
    StatusBar > #status-line { height: 1; }
    """

    def __init__(self) -> None:
        super().__init__(id="status")
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static(Content(""), id="status-line")

    @property
    def glyph(self) -> str:
        return SPINNER[self._frame % len(SPINNER)]

    def tick(self) -> None:
        self._frame += 1

    def show(self, state: TuiState, readings: Sequence[StatusReading] = ()) -> None:
        # A frame may land while this widget is mounted and its child is not —
        # compose is asynchronous, and so is teardown. A widget owns its
        # children's lifetime, so it is the widget that says "nothing to draw
        # into yet" rather than every caller guessing at the moment. Checked
        # first, so the string work below is not done for a line that is gone;
        # `query_one` by id is the cheap path, ~45x a generic `query`.
        try:
            line = self.query_one("#status-line", Static)
        except NoMatches:
            return
        parts = [
            "[$ph-accent]$glyph[/]",
            "[$ph-muted]$model[/]",
            "[$ph-muted]·[/] $preset",
        ]
        values: dict[str, Any] = {
            "glyph": self.glyph if state.busy else "●",
            "model": state.model or "no model",
            "preset": state.preset,
        }
        if state.queued:
            parts.append("[$ph-muted]·[/] $queued queued")
            values["queued"] = str(state.queued)
        pressure = state.pressure
        if pressure is not None:
            # Amber from the threshold on: the point is to see it coming.
            style = "$ph-warning" if pressure >= COMPACTION_THRESHOLD else "$ph-muted"
            parts.append(f"[$ph-muted]·[/] [{style}]$context[/]")
            values["context"] = f"context {min(pressure, 1.0) * 100:.0f}%"
        for index, reading in enumerate(readings):
            # Contributed by a row through `ctx.tui_status` — the footer knows
            # what a reading *is* and nothing about what any of them mean, which
            # is what lets `limits` show a budget here without ph-app importing
            # the package that owns one.
            style = "$ph-warning" if reading.level == "warning" else "$ph-muted"
            parts.append(f"[$ph-muted]·[/] [{style}]$field{index}[/]")
            values[f"field{index}"] = reading.text
        line.update(Content.from_markup("  ".join(parts), **values))


class Sidebar(Vertical):
    """Session facts, and (from Phase 4) the todo list."""

    WIDTH = 32
    """Fixed, so a long path is shortened here rather than wrapping into the
    next row's label. Set on the widget, not repeated in CSS, so there is one."""

    DEFAULT_CSS = """
    Sidebar { background: $ph-panel; border-left: vkey $ph-border; padding: 1; }
    Sidebar.-left { border-left: none; border-right: vkey $ph-border; }
    Sidebar > .section-title { color: $ph-accent; }
    Sidebar > .section-body { color: $ph-muted; }
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.styles.width = self.WIDTH
        self._shown: tuple[str, str, str] | None = None

    def compose(self) -> ComposeResult:
        yield Static(Content.from_markup("[b]session[/b]"), classes="section-title")
        yield Static(Content(""), id="session-facts", classes="section-body")
        yield Static(Content.from_markup("[b]todo[/b]"), classes="section-title")
        yield Static(Content(""), id="todo-list", classes="section-body")
        # Hidden until a child exists: an empty heading costs a line of a
        # 32-column panel to say nothing.
        yield Static(
            Content.from_markup("[b]children[/b]"), id="children-title", classes="section-title"
        )
        yield Static(Content(""), id="children", classes="section-body")

    def show(self, state: TuiState, *, session_id: str, cwd: str) -> None:
        facts = "\n".join(
            [
                f"id      {session_id}",
                f"turn    {state.turn}",
                f"model   {state.model or '-'}",
                f"posture {state.preset}",
                f"sandbox {state.sandbox_mode}",
                f"cwd     {_shorten(cwd)}",
            ]
        )
        todos = "\n".join(_todo_line(todo) for todo in state.todos) or "—"
        children = render_subagents(state)
        if (facts, todos, children) == self._shown:
            return
        self._shown = (facts, todos, children)
        try:
            facts_panel = self.query_one("#session-facts", Static)
            todo_panel = self.query_one("#todo-list", Static)
            children_title = self.query_one("#children-title", Static)
            panel = self.query_one("#children", Static)
        except NoMatches:
            # Same reason as `StatusBar.show`: composed asynchronously, torn
            # down asynchronously, and a frame in either gap has nowhere to go.
            return
        facts_panel.update(Content(facts))
        todo_panel.update(Content(todos))
        children_title.display = bool(children)
        panel.display = bool(children)
        panel.update(Content(children))


TODO_GLYPHS = {"pending": "○", "in_progress": "◐", "completed": "●"}

NO_WORK_SEEN = " (no work seen)"
"""What a completed entry says when the harness counted no tools in its window.

**A field, not a rule re-derived here.** `worked` is attached by `tool-todo` when
it writes the list (P7-16), so this reads a number rather than restating what
counts as work — which it could not do anyway: `ph-app` depends on `ph-core` and
not on the bundle that owns the tool, and a copy of the rule on this side is the
drift that boundary exists to prevent.

The sidebar and not only the tool card because the card is one call's and this
panel is the plan a person watches all session. It is a *signal*, not a verdict:
"decide the approach" is a real step with no tool calls, and the point is that a
tick with work behind it and a tick without now look different.
"""


def _todo_line(todo: dict[str, Any]) -> str:
    """One entry, with the receipt when it is empty."""
    glyph = TODO_GLYPHS.get(str(todo.get("status")), "○")
    bare = todo.get("status") == "completed" and not todo.get("worked")
    return f"{glyph} {todo.get('content', '')}{NO_WORK_SEEN if bare else ''}"


def _shorten(path: str, width: int = Sidebar.WIDTH - 10) -> str:
    """A path that fits, keeping the end — the part that identifies it."""
    if path.startswith(_HOME):
        path = f"~{path[len(_HOME) :]}"
    return path if len(path) <= width else f"…{path[-(width - 1) :]}"
