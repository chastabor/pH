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

from ph.seams.subagents import child_is_live
from ph.seams.tui_status import StatusReading
from ph.text import thousands

from ..state import CatalogEntry, TuiState

__all__ = ["COMPACTION_THRESHOLD", "Sidebar", "StatusBar", "children_heading", "render_subagents"]

COMPACTION_THRESHOLD = 0.85
"""Where Phase 4's `compaction-summarize` triggers. The gauge warns here."""

SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")


def in_slot(readings: Sequence[StatusReading], slot: str) -> list[StatusReading]:
    """The readings a region draws, by the slot their own field declared.

    This is what `StatusReading.id` and `slot` bought together. The footer used
    to render every reading in order, so a fact that belonged beside the
    session's directory could only get there as a bespoke wire field; the first
    fix for that was a hardcoded set of ids *here*, which put a placement
    decision about a ph-core row in ph-app — where renaming the id silently
    moved the reading back onto the line.

    A slot this build does not know still renders somewhere rather than
    vanishing: `line` is the default and the fallback, which is what keeps a new
    reading additive for a front end that has not been taught about it.
    """
    return [one for one in readings if (one.slot if one.slot in _SLOTS else "line") == slot]


_SLOTS = frozenset({"line", "session"})

_HOME = str(Path.home())


def children_heading(state: TuiState) -> str:
    """`children · 3 running, 5 pending` — the shape of the fan-out at a glance.

    A count, because the panel is one line per child and eight of them is a list
    somebody has to tally by eye; the question a person actually has mid-fan-out
    is how much is moving and how much is waiting.

    **"pending", never "queued"**, and that is the whole reason this reads oddly
    next to the roster it counts. The status bar already says "queued" for the
    person's *own* prompts waiting on a busy agent, and two counts on one screen
    using one word for two different things is worse than a synonym. The roster's
    own vocabulary is untouched — `queued` is what the log says and what every
    other reader folds; this is a heading, and headings are for the reader.

    Settled children are not counted at all. They stay listed, because a parent
    asking what happened to one deserves an answer, but "how busy is this
    fan-out" is a question about the ones still going.

    **`child_is_live` decides which those are, rather than a literal here.** The
    seam counts an *unrecognised* status as live on purpose, so a status this
    package has not heard of lands in `pending` — where a hand-written
    `status in {running, queued}` would have counted it as neither and quietly
    under-reported a fan-out that is still working. Counted off `state.roster`,
    the seam's own fold, for the reason the panel draws from it (A11).
    """
    live = [row for row in state.roster.values() if child_is_live(row)]
    running = sum(1 for row in live if row.get("status") == "running")
    counts = [
        f"{count} {word}"
        for count, word in ((running, "running"), (len(live) - running, "pending"))
        if count
    ]
    return f"children · {', '.join(counts)}" if counts else "children"


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
        tokens = f" {thousands(row.tokens)}" if row.tokens >= 1000 else ""
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
        ]
        values: dict[str, Any] = {
            "glyph": self.glyph if state.busy else "●",
            "model": state.model or "no model",
        }
        if state.turn:
            parts.append("[$ph-muted]·[/] turn $turn")
            values["turn"] = str(state.turn)
        if state.queued:
            parts.append("[$ph-muted]·[/] $queued queued")
            values["queued"] = str(state.queued)
        pressure = state.pressure
        if pressure is not None:
            # Amber from the threshold on: the point is to see it coming.
            style = "$ph-warning" if pressure >= COMPACTION_THRESHOLD else "$ph-muted"
            parts.append(f"[$ph-muted]·[/] [{style}]$context[/]")
            values["context"] = f"context {min(pressure, 1.0) * 100:.0f}%"
        for index, reading in enumerate(in_slot(readings, "line")):
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
        self._shown: tuple[object, ...] | None = None
        # Each catalog with the strings it renders to, keyed on the tuple it was
        # rendered from — see `show`. Empty rather than `None` so the identity
        # check there needs no first-frame special case.
        self._tools: tuple[tuple[CatalogEntry, ...], str, str] = ((), "", "")
        self._skills: tuple[tuple[CatalogEntry, ...], str, str] = ((), "", "")

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
        # What fills the context window, and the reason these are panels rather
        # than a command alone: the catalog is charged to every request, so its
        # size is a thing to be looked at rather than asked after. Each pair
        # hides together — a heading over a hidden body is a wasted line of 32
        # columns, and `/view` is what hides them.
        yield Static(Content.from_markup("[b]tools[/b]"), id="tools-title", classes="section-title")
        yield Static(Content(""), id="tools-list", classes="section-body")
        yield Static(
            Content.from_markup("[b]skills[/b]"), id="skills-title", classes="section-title"
        )
        yield Static(Content(""), id="skills-list", classes="section-body")

    def show(
        self,
        state: TuiState,
        readings: Sequence[StatusReading] = (),
        *,
        session_id: str,
        cwd: str,
        show_tools: bool = True,
        show_skills: bool = True,
    ) -> None:
        # What the rows themselves said belongs here — see `in_slot`. Positional
        # like `StatusBar.show`'s, because it is the same list from one read.
        placed = " · ".join(one.text for one in in_slot(readings, "session"))
        # Model, reasoning effort, turn and posture live on the footer now —
        # they change while a turn runs and the eye is already there. What is
        # left is what does not move: where this session is, and what the
        # kernel will let a command touch.
        facts = "\n".join(
            [
                f"id      {session_id}",
                placed or "sandbox -",
                f"cwd     {_shorten(cwd)}",
            ]
        )
        todos = "\n".join(_todo_line(todo) for todo in state.todos) or "—"
        children = render_subagents(state)
        heading = children_heading(state)
        # **Rendered on identity, not on every frame.** The two catalogs are set
        # once at attach and never move — a row registers its tools at mount —
        # so joining them here would rebuild two byte-identical strings thirty
        # times a second for the life of the session, ahead of the very check
        # that would have thrown them away. `is` rather than `==` because the
        # question is "is this the same tuple", which is what makes it O(1).
        if state.tools is not self._tools[0] or state.skills is not self._skills[0]:
            self._tools = (state.tools, _heading("tools", state.tools), _names(state.tools))
            self._skills = (state.skills, _heading("skills", state.skills), _names(state.skills))
        was = self._shown
        # `_CATALOGS` is where the two rendered catalogs start in this tuple, so
        # the loop below can find last frame's by position rather than by an
        # arithmetic that reads the same forwards and backwards — the first
        # spelling was `7 - index`, which is correct for two entries counted
        # from the wrong end and silently swaps them.
        now = (facts, todos, children, heading, show_tools, show_skills, self._tools, self._skills)
        if now == was:
            return
        self._shown = now
        try:
            facts_panel = self.query_one("#session-facts", Static)
            todo_panel = self.query_one("#todo-list", Static)
            children_title = self.query_one("#children-title", Static)
            panel = self.query_one("#children", Static)
            tools_title = self.query_one("#tools-title", Static)
            tools_panel = self.query_one("#tools-list", Static)
            skills_title = self.query_one("#skills-title", Static)
            skills_panel = self.query_one("#skills-list", Static)
        except NoMatches:
            # Same reason as `StatusBar.show`: composed asynchronously, torn
            # down asynchronously, and a frame in either gap has nowhere to go.
            return
        facts_panel.update(Content(facts))
        todo_panel.update(Content(todos))
        children_title.display = bool(children)
        if was is None or heading != was[3]:
            # Guarded because `from_markup` is the most expensive thing here —
            # 28 µs against `children_heading`'s 1 — and the heading is the one
            # part of this panel that barely changes: a child's token counter
            # ticking redraws the body every frame and leaves this string alone.
            children_title.update(Content.from_markup(f"[b]{heading}[/b]"))
        panel.display = bool(children)
        panel.update(Content(children))
        for index, (title, body, visible, rendered) in enumerate(
            (
                (tools_title, tools_panel, show_tools, self._tools),
                (skills_title, skills_panel, show_skills, self._skills),
            )
        ):
            # A hidden panel hides its heading with it: 32 columns cannot afford
            # a line that names a body nobody can see.
            title.display = visible
            body.display = visible
            # Guarded for `children_title`'s reason, and it bites harder here:
            # `from_markup` is 28 µs against the 1 µs beside it, and these two
            # headings are the parts of this panel that *never* change while a
            # turn's token counter redraws the body on every frame.
            if was is None or rendered is not was[_CATALOGS + index]:
                title.update(Content.from_markup(rendered[1]))
                body.update(Content(rendered[2]))


_CATALOGS = 6
"""Where `Sidebar.show`'s rendered catalogs begin in its change-detection tuple."""


def _heading(what: str, names: tuple[CatalogEntry, ...]) -> str:
    """`tools 5` — the panel's name and how many, which is the glance-value.

    A catalog is charged to every request, so *how many* is the number worth
    seeing without reading the list under it. Bare when empty, because `tools 0`
    reads as a measurement where the `—` below already says it plainly.
    """
    return f"[b]{what}[/b] {len(names)}" if names else f"[b]{what}[/b]"


def _names(names: tuple[CatalogEntry, ...]) -> str:
    """A catalog as its names, wrapped by the panel.

    `—` rather than an empty body, so a panel that is *drawn* always says
    something: no tools is a real answer — a deployment that mounted no tool
    rows — and must not read as a panel that failed to load. The count goes on
    the heading, where it is legible without reading the list.
    """
    return ", ".join(one.name for one in names) if names else "—"


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
