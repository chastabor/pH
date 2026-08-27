"""Footer and sidebar: what is happening, and under what posture.

The context reading turns amber at the compaction threshold rather than at some
round number, because the number a user needs to see coming is the one where
the harness will act (G4's 0.85 fraction, Phase 4).

@module ph_app.tui.widgets.status
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Static

from ..state import TuiState

__all__ = ["COMPACTION_THRESHOLD", "Sidebar", "StatusBar"]

COMPACTION_THRESHOLD = 0.85
"""Where Phase 4's `compaction-summarize` triggers. The gauge warns here."""

SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

_HOME = str(Path.home())


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

    def show(self, state: TuiState) -> None:
        parts = [
            "[$ph-accent]$glyph[/]",
            "[$ph-muted]$model[/]",
            "[$ph-muted]·[/] $preset",
        ]
        values: dict[str, Any] = {
            "glyph": self.glyph if state.status == "running" else "●",
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
        self.query_one("#status-line", Static).update(
            Content.from_markup("  ".join(parts), **values)
        )


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
        self._shown: tuple[str, str] | None = None

    def compose(self) -> ComposeResult:
        yield Static(Content.from_markup("[b]session[/b]"), classes="section-title")
        yield Static(Content(""), id="session-facts", classes="section-body")
        yield Static(Content.from_markup("[b]todo[/b]"), classes="section-title")
        yield Static(Content(""), id="todo-list", classes="section-body")

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
        glyphs = {"pending": "○", "in_progress": "◐", "completed": "●"}
        todos = (
            "\n".join(
                f"{glyphs.get(str(todo.get('status')), '○')} {todo.get('content', '')}"
                for todo in state.todos
            )
            or "—"
        )
        if (facts, todos) == self._shown:
            return
        self._shown = (facts, todos)
        self.query_one("#session-facts", Static).update(Content(facts))
        self.query_one("#todo-list", Static).update(Content(todos))


def _shorten(path: str, width: int = Sidebar.WIDTH - 10) -> str:
    """A path that fits, keeping the end — the part that identifies it."""
    if path.startswith(_HOME):
        path = f"~{path[len(_HOME) :]}"
    return path if len(path) <= width else f"…{path[-(width - 1) :]}"
