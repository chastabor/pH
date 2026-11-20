"""Footer and sidebar: what is happening, and under what posture.

The context reading turns amber at the compaction threshold and not at a round
number. The number that a user needs to see coming is the one where the harness
will act. That number is the share that the profile of the root configures
(`TuiState.compacts_at`, 0.85 by default). When nothing compacts on pressure,
the reading does not turn amber.

@module ph_app.tui.widgets.status
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.content import Content
from textual.css.query import NoMatches
from textual.widgets import Rule, Static

from ph.json import as_str
from ph.seams.subprocess import first_line
from ph.seams.tui_status import ReadingSlot, StatusReading
from ph.text import duration, elide, thousands

from ...payloads import DaemonLifetime
from ..state import CatalogEntry, TuiState

__all__ = [
    "Sidebar",
    "StatusBar",
    "children_heading",
    "daemon_line",
    "render_subagents",
    "shorten_path",
]

SPINNER = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")

IDLE = "●"
"""The glyph that is shown between turns. During a turn the spinner runs in its place."""

SEPARATOR = " · "
"""The separator between the categories of the footer: `flash/high · turn 3 · context 12%`.

One space on each side. With two spaces, each `·` read as a column gap, and a
footer is one line with no columns. The request was to "tighten" the line, and
these spaces were the only loose part."""

QUALIFIER = "/"
"""The separator within a category: the model and its effort, the gauge and its cache rate.

A reading in the `model` or `context` slot is half of a figure that this footer
already draws. A `·` would make the reading one more category. The effort used
to be drawn that way, three entries from the model that it qualifies."""


def in_slot(readings: Sequence[StatusReading], slot: ReadingSlot) -> list[StatusReading]:
    """Return the readings that a region draws, selected by the slot that their field declared.

    `StatusReading.id` and `slot` together make this possible. The footer used
    to render every reading in order, so a fact that belonged beside the
    directory of the session could only get there as a special wire field. The
    first fix for that was a hardcoded set of ids *here*. That put a placement
    decision about a ph-core row in `ph_app`, where a rename of the id silently
    moved the reading back onto the line.

    **There is no fallback for a slot that this module does not know**, because
    there is no such reading. `ReadingSlot` is closed and `StatusReading`
    validates it. A client therefore refuses an unknown slot at the wire, and an
    in-process reading carries one of the values that this build was typed
    against. A test (`test_every_reading_slot_is_drawn`) checks that every value
    is drawn *somewhere*. No runtime check does.
    """
    return [one for one in readings if one.slot == slot]


def _piece(reading: StatusReading) -> Content:
    """Return one reading as the footer draws it: its own text, amber when it warns.

    A row contributes a reading through `ctx.tui_status`. The footer knows what
    a reading *is* and nothing about what any reading means. That lets `limits`
    show a budget here without `ph_app` importing the package that owns one. The
    piece is `Content`, never markup. The text is a string from a row, so it is
    never parsed, and a `[` in it stays a `[`.
    """
    text = reading.text
    return Content.styled(text, "$ph-warning") if reading.level == "warning" else Content(text)


_HOME = str(Path.home())


def children_heading(state: TuiState) -> str:
    """Return `children · 3 running, 5 pending`: the shape of the fan-out at a glance.

    The heading is a count because the panel has one line for each child, and a
    reader would have to tally eight lines by eye. During a fan-out the user
    wants to know how much is moving and how much is waiting.

    **The word is "pending", never "queued".** That is why the heading reads
    differently from the rows that it counts. The footer already says "queued"
    for the user's *own* prompts that wait on a busy agent. Two counts on one
    screen that use one word for two different things are worse than a synonym.
    The vocabulary of the children does not change: `queued` is what their logs
    say and what every other reader folds. This is a heading, and a heading is
    written for the reader.

    Settled children are not counted. They stay listed, because a parent that
    asks what happened to a child needs an answer. But "how busy is this
    fan-out" is a question about the children that are still going. That holds
    at every depth, because a grandchild that works is the fan-out working.

    **The rule of `child_is_live` decides which children those are, and no
    literal here does** (`SubagentRow.live`). A live child is not revoked and
    is not in the settled set of the seam, so `queued` and any other unsettled
    status are counted as `pending`. A hand-written `status in {running, queued}`
    would count a status added to the seam as neither. It would then under-report
    a fan-out that is still working, with no warning. The count is taken from
    the rows that the panel draws, which are the rows of the daemon (A11).
    """
    live = [row for row in state.subagents.values() if row.live]
    running = sum(1 for row in live if row.child.status == "running")
    counts = [
        f"{count} {word}"
        for count, word in ((running, "running"), (len(live) - running, "pending"))
        if count
    ]
    return f"children · {', '.join(counts)}" if counts else "children"


def render_subagents(state: TuiState) -> str:
    """Render the delegation panel: one line for each sub-agent, with a grandchild
    indented under the child that spawned it (P3-19, P11-09).

    This is a *panel*, not transcript rows, for two reasons. A fan-out of eight
    that ticks through `queued → running → done` would push the conversation off
    screen. And the interesting thing about a family is its current shape, which
    is a projection and not a history. The panel is drawn from the
    `session.children` of the daemon, which the daemon reads from the log of
    each child. The resume sweep and the budgets read the same states. A
    tombstoned child stays listed, because a parent that asks what happened to
    the child that it revoked needs an answer.

    A child that is held for a credential says which credential (T5). `queued`
    alone reads exactly like a child that waits for a slot, and only one of the
    two needs a person.
    """
    lines: list[str] = []
    for row in state.subagents.values():
        child = row.child
        detail = (
            f"needs {child.awaiting}"
            if child.awaiting
            else child.cause or (child.model if child.status == "queued" else child.status)
        )
        tokens = f" {thousands(child.tokens)}" if child.tokens >= 1000 else ""
        indent = "  " * row.depth
        lines.append(f"{indent}{row.glyph} {child.name or child.run_id} {detail}{tokens}".rstrip())
    return "\n".join(lines)


class StatusBar(Horizontal):
    """The footer, in one line: spinner, model and its effort, turn, queue depth,
    context and its cache rate, then every reading that the rows put on the line.

    **The footer is drawn as the top edge of the prompt**, between the
    transcript and the box: `● ──────── flash/high · turn 3 · read-only accepted`.
    The glyph is at the far left, in the column that the gutter of the
    transcript uses. The text is right-aligned. A rule fills the row between
    them. The prompt no longer has a border of its own, because a frame spent
    two rows and two columns of a terminal. This row is therefore what separates
    the conversation from the draft. One column of padding at the right end
    keeps the text off the border of the sidebar, which is drawn in the next
    cell.

    **The glyph is its own cell**, so a spinner frame repaints one cell and
    nothing else. The line is rebuilt only when a fact on it changes. The
    layout, which an auto-width line runs again on every `update`, does not run
    at all for a spinner frame.

    **The footer is drawn in the color of the headings**, rule and text alike:
    `$ph-accent`. The section titles of the sidebar use this role, and the top
    three Markdown heading levels of the transcript resolve to it (the theme
    maps it onto Textual's `primary`). It is a role, not a color, so the footer
    follows the theme as the headings do. The color is set one time on the
    footer and inherited. The only spans that name a color are the spans that
    must differ from it: a warning reading, and the gauge from the compaction
    threshold on.

    **The rule is a widget beside the text, not padding inside it.** The
    horizontal layout of Textual gives the text its own width and gives the rule
    what is left. So nothing here measures a string against the terminal, and
    `#status-line` still renders exactly the line that it was given. A test
    reads that line, and a narrow terminal crops it. When the terminal is too
    narrow for the whole line, the rule keeps its last cell, because a fraction
    never sizes below one. The text then runs past the row, which clips its
    right end. The posture is therefore clipped before the model, which is the
    order in which the line was written. The footer is one row at any width, so
    the prompt never moves down for a long reading.

    The footer owns the spinner frame, so the footer and the terminal title
    read the same glyph from one counter.
    """

    DEFAULT_CSS = """
    StatusBar { height: 1; background: $ph-panel; color: $ph-accent; padding: 0 1 0 0; }
    StatusBar > #status-glyph { width: 1; margin: 0 1 0 0; }
    StatusBar > #status-rule { margin: 0 1 0 0; color: $ph-accent; }
    StatusBar > #status-line { width: auto; height: 1; }
    """

    def __init__(self) -> None:
        super().__init__(id="status")
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static(Content(IDLE), id="status-glyph")
        yield Rule(id="status-rule")
        yield Static(Content(""), id="status-line")

    @property
    def glyph(self) -> str:
        return SPINNER[self._frame % len(SPINNER)]

    def tick(self) -> None:
        """One spinner frame. Called only while a turn runs, so the glyph spins."""
        self._frame += 1
        self._draw_glyph(Content(self.glyph))

    def _part(self, part: Literal["glyph", "line"]) -> Static | None:
        """Return the cell or the line to draw into, or `None` while there is none.

        A frame may arrive while this widget is mounted and its child is not,
        because compose is asynchronous and so is teardown. A widget owns the
        lifetime of its children. So the widget reports "nothing to draw into
        yet", and the callers do not have to guess at the moment. The callers
        ask first, so that no string work is done for a line that is gone.
        `query_one` by id is the cheap path: a generic `query` costs about 45
        times as much.
        """
        try:
            return self.query_one(f"#status-{part}", Static)
        except NoMatches:
            return None

    def _draw_glyph(self, glyph: Content) -> None:
        """Draw the glyph cell. CSS fixes it at one column wide, so no layout runs for it."""
        cell = self._part("glyph")
        if cell is not None:
            cell.update(glyph, layout=False)

    def show(self, state: TuiState, readings: Sequence[StatusReading] = ()) -> None:
        """Draw `flash/high · turn 3 · context 12%/cache 28.8k(98%) · read-only accepted`,
        with the glyph in its own cell at the far left of the footer.

        There is one *segment* for each category, with `SEPARATOR` between
        segments and `QUALIFIER` within one. The segments are the model with
        what qualifies it, the turn, the queue, the gauge with what qualifies it,
        and then the readings of the line in their own order. The line is built
        from `Content` pieces and not from markup, so a string from a row is
        never parsed. A frame also costs no markup parse.
        """
        self._draw_glyph(Content(self.glyph if state.busy else IDLE))
        line = self._part("line")
        if line is None:
            return
        model = [Content(state.model or "no model"), *map(_piece, in_slot(readings, "model"))]
        segments = [Content(QUALIFIER).join(model)]
        if state.turn:
            segments.append(Content(f"turn {state.turn}"))
        if state.queued:
            segments.append(Content(f"{state.queued} queued"))
        context: list[Content] = []
        pressure = state.pressure
        if pressure is not None:
            gauge = f"context {min(pressure, 1.0) * 100:.0f}%"
            # Amber from the threshold on, so that the user sees compaction
            # coming. The threshold is the share that the root compacts at, and
            # a root that does not compact on pressure has none.
            warn = state.compacts_at is not None and pressure >= state.compacts_at
            context.append(Content.styled(gauge, "$ph-warning") if warn else Content(gauge))
        # With no gauge, because the window is unknown, the readings stand alone
        # and do not vanish. A cache rate is still a fact, although it then has
        # no gauge to stand beside.
        context.extend(map(_piece, in_slot(readings, "context")))
        if context:
            segments.append(Content(QUALIFIER).join(context))
        segments.extend(map(_piece, in_slot(readings, "line")))
        line.update(Content(SEPARATOR).join(segments))

    def show_unattached(self, refusal: str | None) -> None:
        """Draw the footer with no session yet: the reason when a start failed, and
        otherwise that a session is on its way.

        The refusal is the line that stays after the modal that reported it is
        gone. Only its first line is drawn: the modal showed all of it, and a
        footer is one row. It is a `Content` piece and not markup, for the reason
        that `ConfirmModal` gives: a refusal quotes paths and options, and `[…]`
        is markup.
        """
        self._draw_glyph(Content(IDLE) if refusal is None else Content.styled(IDLE, "$ph-warning"))
        line = self._part("line")
        if line is None:
            return
        if refusal is None:
            line.update(Content("attaching…"))
            return
        why = first_line(refusal) or "could not start"
        line.update(Content.assemble(("no session", "$ph-warning"), SEPARATOR, why))


class Sidebar(Vertical):
    """The sidebar: session facts, the todo list (from Phase 4), the children, and
    the tool and skill catalogs."""

    WIDTH = 32
    """The fixed width. A long path is shortened here, so that it does not wrap
    into the label of the next row. The width is set on the widget and not
    repeated in CSS, so there is one copy of it."""

    DEFAULT_CSS = """
    Sidebar { background: $ph-panel; border-left: vkey $ph-border; padding: 1; }
    Sidebar.-left { border-left: none; border-right: vkey $ph-border; }
    Sidebar > .section-title { color: $ph-accent; }
    Sidebar > .section-body { color: $ph-muted; }
    """

    def __init__(self, **kwargs: Any) -> None:  # noqa: ANN401
        super().__init__(**kwargs)
        self.styles.width = self.WIDTH
        self._shown: tuple[object, ...] | None = None
        # Each catalog with the strings that it renders to, keyed on the tuple
        # that it was rendered from (see `show`). The tuple is empty and not
        # `None`, so that the identity check there needs no special case for the
        # first frame.
        self._tools: tuple[tuple[CatalogEntry, ...], str, str] = ((), "", "")
        self._skills: tuple[tuple[CatalogEntry, ...], str, str] = ((), "", "")

    def compose(self) -> ComposeResult:
        yield Static(Content.from_markup("[b]session[/b]"), classes="section-title")
        yield Static(Content(""), id="session-facts", classes="section-body")
        yield Static(Content.from_markup("[b]todo[/b]"), classes="section-title")
        yield Static(Content(""), id="todo-list", classes="section-body")
        # Hidden until a child exists. An empty heading would cost a line of a
        # 32-column sidebar and say nothing.
        yield Static(
            Content.from_markup("[b]children[/b]"), id="children-title", classes="section-title"
        )
        yield Static(Content(""), id="children", classes="section-body")
        # These show what fills the context window. They are panels, and not
        # only a command, because the catalog is charged to every request. Its
        # size should therefore be visible without a question. Each pair hides
        # together, because a heading over a hidden body wastes a line of 32
        # columns. `/view` hides them.
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
        # The readings that the rows contributed for this slot belong here. See
        # `in_slot`. The argument is positional like that of `StatusBar.show`,
        # because it is the same list from one read.
        placed = " · ".join(one.text for one in in_slot(readings, "session"))
        # Model, reasoning effort, turn and posture are on the footer now. They
        # change while a turn runs, and the eye is already there. What is left
        # here does not move: where this session is, and what the kernel lets a
        # command touch.
        facts = "\n".join(
            [
                f"id      {session_id}",
                placed or "sandbox -",
                f"cwd     {shorten_path(cwd)}",
                *daemon_line(state.lifetime),
            ]
        )
        todos = "\n".join(_todo_line(todo) for todo in state.todos) or "—"
        children = render_subagents(state)
        heading = children_heading(state)
        # **Rendered on identity, not on every frame.** The two catalogs are set
        # one time at attach and never change, because a row registers its tools
        # at mount. Joining them here would rebuild two byte-identical strings
        # thirty times a second for the life of the session, before the check
        # that would discard them. The test is `is` and not `==` because the
        # question is "is this the same tuple", which makes the test O(1).
        if state.tools is not self._tools[0] or state.skills is not self._skills[0]:
            self._tools = (state.tools, _heading("tools", state.tools), _names(state.tools))
            self._skills = (state.skills, _heading("skills", state.skills), _names(state.skills))
        was = self._shown
        # `_CATALOGS` is the position where the two rendered catalogs start in
        # this tuple. The loop below can then find the catalogs of the last
        # frame by position. The first version computed `7 - index`, an
        # arithmetic that reads the same forwards and backwards. It is correct
        # for two entries counted from the wrong end, so it silently swapped
        # them.
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
            # The same reason as in `StatusBar._part`: the widget is composed
            # asynchronously and torn down asynchronously, and a frame in
            # either gap has nowhere to go.
            return
        facts_panel.update(Content(facts))
        todo_panel.update(Content(todos))
        children_title.display = bool(children)
        if was is None or heading != was[3]:
            # Guarded because `from_markup` is the most expensive call here: 28 µs
            # against 1 µs for `children_heading`. The heading is also the one
            # part of this panel that hardly ever changes. The ticking token
            # counter of a child redraws the body every frame and leaves this
            # string alone.
            children_title.update(Content.from_markup(f"[b]{heading}[/b]"))
        panel.display = bool(children)
        panel.update(Content(children))
        for index, (title, body, visible, rendered) in enumerate(
            (
                (tools_title, tools_panel, show_tools, self._tools),
                (skills_title, skills_panel, show_skills, self._skills),
            )
        ):
            # A hidden panel hides its heading with it. 32 columns cannot afford
            # a line that names a body that nobody can see.
            title.display = visible
            body.display = visible
            # Guarded for the same reason as `children_title`, and the saving is
            # larger here. `from_markup` costs 28 µs against the 1 µs beside it.
            # These two headings *never* change, while the token counter of a
            # turn redraws the body on every frame.
            if was is None or rendered is not was[_CATALOGS + index]:
                title.update(Content.from_markup(rendered[1]))
                body.update(Content(rendered[2]))


_CATALOGS = 6
"""The position where the rendered catalogs begin in the change-detection tuple
of `Sidebar.show`."""


def _heading(what: str, names: tuple[CatalogEntry, ...]) -> str:
    """Return `tools 5`: the name of the panel and the number of entries.

    A catalog is charged to every request, so the number of entries is worth
    seeing without reading the list under it. The heading is bare when the
    catalog is empty, because `tools 0` reads as a measurement and the `—` below
    already says it plainly.
    """
    return f"[b]{what}[/b] {len(names)}" if names else f"[b]{what}[/b]"


def _names(names: tuple[CatalogEntry, ...]) -> str:
    """Return a catalog as its names, which the panel wraps.

    An empty catalog gives `—` and not an empty body, so that a panel that is
    *drawn* always says something. No tools is a real answer: it is a deployment
    that mounted no tool rows. It must not read as a panel that failed to load.
    The count goes on the heading, where it is readable without the list.
    """
    return ", ".join(one.name for one in names) if names else "—"


TODO_GLYPHS = {"pending": "○", "in_progress": "◐", "completed": "●"}

NO_WORK_SEEN = " (no work seen)"
"""What a completed entry says when the harness counted no tools in its window.

**This reads a field and does not derive a rule again.** `tool-todo` attaches
`worked` when it writes the list (P7-16), so this module reads a number and does
not restate what counts as work. It could not restate it anyway: `ph_app`
imports ph-core and may not import the bundle that owns the tool. A copy of the
rule on this side is the drift that the boundary exists to prevent.

The note is in the sidebar, and not only on the tool card, because the card
belongs to one call and this panel is the plan that a user watches for the
whole session. It is a *signal*, not a verdict. "decide the approach" is a real
step with no tool calls. The purpose is that a tick with work behind it and a
tick without now look different.
"""


def daemon_line(lifetime: DaemonLifetime | None) -> list[str]:
    """Return the `daemon` fact, or no line at all. This answers the question of P9-04.

    **The question is "if I close this window, what happens".** A user who runs
    a front end that a daemon was spawned for has no other way to know whether
    the supervisor behind it stays. A client cannot derive the answer, because
    whoever started the process decided the lifetime.

    So the line is one of three sentences, in the order in which the answer
    narrows:

    * `service`: somebody ran `phern daemon`, and closing this window changes
      nothing.
    * `held ·` and the reasons. With `task` or `schedule`, the daemon is
      ephemeral but a turn is in flight or a schedule is recorded, so the daemon
      outlives this window and keeps going. With `client`, the terminal of
      somebody else is on it too.
    * `exits on detach`, or `exits 5m after detach`: nothing but this client
      wants the daemon.

    **`client` is filtered by the count, not by assumption.** It is in `holds`
    whenever *anybody* is connected, and this window is one of them. Printing it
    unconditionally would spend a row of a 32-column sidebar to tell a user that
    they have a window open. Dropping it unconditionally would be worse: with
    two terminals open, both would read `exits on detach` and both would be
    wrong. `clients` is on the frame so that the reader can tell those cases
    apart.

    The result is an empty list and not an empty string, so that a front end
    with no daemon draws no row instead of a blank one.
    `"this process exits when you close it"` is not news about a process that
    the user is looking at.
    """
    if lifetime is None:
        return []
    if lifetime.mode != "ephemeral":
        return [f"daemon  {lifetime.mode}"]
    others = [one for one in lifetime.holds if one != "client" or lifetime.clients > 1]
    if others:
        return [f"daemon  held · {' · '.join(others)}"]
    after = f" {duration(lifetime.keep_alive_ms)} after" if lifetime.keep_alive_ms else " on"
    return [f"daemon  exits{after} detach"]


def _todo_line(todo: dict[str, Any]) -> str:
    """Return one entry, with the `NO_WORK_SEEN` note when it is completed and no
    work was counted."""
    glyph = TODO_GLYPHS.get(as_str(todo.get("status")), "○")
    bare = todo.get("status") == "completed" and not todo.get("worked")
    return f"{glyph} {todo.get('content', '')}{NO_WORK_SEEN if bare else ''}"


def shorten_path(path: str, width: int = Sidebar.WIDTH - 10) -> str:
    """Return a path that fits. It keeps the end, which is the part that identifies it.

    The function is public because the title of the startup picker needs the
    same rule at a different width. Two `~`-substitutions in one package are two
    answers to "how does pH write a path". The second one had no width cap at
    all, so a deep directory overflowed a modal that is 72 columns wide.
    """
    if path.startswith(_HOME):
        path = f"~{path[len(_HOME) :]}"
    return elide(path, width, keep="tail")
