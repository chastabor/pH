"""The auditor's screen, and the row that contributes it (P4-17).

P3-25 shipped this view as its own `App`, which reads a stored log with nothing
mounted — the property the shape was chosen for. What it could not do was open
*over a conversation*: `ph --mode trajectory` was the only route in.

Splitting the screen out of the app fixes that without giving the property up.
`TrajectoryApp` composes one of these with records read from a file;
`ctx.tui_screens` composes one from the live session; both are the same screen,
so the standalone entry point is once again a strict subset of the embedded one
rather than a second implementation of it.

**Forking is the one action, and it is not offered everywhere.** A6 permits a
fork at a closed turn and nowhere else, so the table marks those rows and the
key refuses the others by *saying which rows are targets* rather than by
rejecting after the fact. The fork itself is `ctx.sessions.fork`, which needs a
session store — so the action is available when the view was opened from a live
harness and reports honestly that it is not when it was opened from a file.

@module ph_app.tui.trajectory_screen
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.screen import Screen
from textual.widgets import DataTable, Static

from ph.cordis import Context, plugin
from ph.seams.tui_screens import ScreenDefinition
from ph.session import Session, SessionForkError

from .screens import RevealHost, RevealSeq
from .trajectory import TrajectoryRecord, build_trajectory
from .widgets.trajectory import TrajectoryPanel

__all__ = ["SCREEN_ID", "TRAJECTORY_KEY", "TrajectoryScreen", "apply"]

SCREEN_ID = "trajectory"
"""Its id in `ctx.tui_screens`, and so `/trajectory` and the binding id."""

TRAJECTORY_KEY = "f2"
"""Its default key. A default, not a rule: the id above is the binding id, so
`tui.json` rebinds it like any built-in (see `TuiKeybindings.extra`)."""


class TrajectoryScreen(Screen[None]):
    """The trajectory over one log: a header, the table, and the fork action."""

    DEFAULT_CSS = """
    TrajectoryScreen { background: $ph-background; color: $ph-foreground; }
    TrajectoryScreen > #trajectory-header {
        height: 1; background: $ph-panel; color: $ph-muted; padding: 0 1;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("f", "fork", "fork here", id="fork"),
        Binding("slash", "filter", "filter", id="filter"),
        Binding("escape", "back", "back", id="back"),
    ]
    """The table holds focus, so the single-key actions reach the screen; `/`
    hands focus to the filter and `escape` takes it back — and, when the filter
    does not have it, leaves the screen. A filter box that had focus by default
    would eat every one of these as typed text.

    Every binding carries an `id`, which is what `App.set_keymap` remaps
    against, so a user who rebound one in `tui.json` gets it here too."""

    def __init__(
        self,
        records: list[TrajectoryRecord],
        *,
        session_id: str = "",
        sessions: Any = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.records = records
        self.session_id = session_id
        self.sessions = sessions
        """`ctx.sessions` when a harness is running. `None` when reading a file,
        which is the ordinary case and why the fork action reports rather than
        assumes."""
        self.notice = ""
        """The last action's own words, kept so a test can read what a person
        was told. The person is told by `notify`; this is the record of it."""
        self._opening_seq = -1

    # ---------------------------------------------------------------- shape --

    def compose(self) -> ComposeResult:
        yield Static(Content(""), id="trajectory-header")
        yield TrajectoryPanel(self.records, id="trajectory")

    def on_mount(self) -> None:
        self.panel.focus_table()
        self._show_header()
        if self._opening_seq >= 0:
            # After a refresh: the panel fills its table in its own `on_mount`,
            # and a cursor moved before the rows exist lands nowhere.
            self.call_after_refresh(self.panel.select_seq, self._opening_seq)

    @property
    def panel(self) -> TrajectoryPanel:
        return self.query_one("#trajectory", TrajectoryPanel)

    # ------------------------------------------------- cross-navigation --

    def reveal(self, seq: int) -> None:
        """Open positioned at a log seq (the `Revealing` protocol).

        Recorded rather than applied, because the shell calls this before the
        screen is pushed — there is no table yet to move a cursor in.
        """
        self._opening_seq = seq

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A chosen row asks its host for the transcript counterpart.

        A message rather than a call: this screen does not know what pushed it.
        Whether anyone can answer is asked of the host structurally rather than
        by counting the screen stack — `--mode trajectory` has no transcript
        however deep it is mounted, and a shell that has one can answer from any
        depth.
        """
        event.stop()
        record = self.panel.selected()
        if record is None:
            return
        if not isinstance(self.app, RevealHost):
            self._report("no transcript here — this view was opened on a file", refused=True)
            return
        self.post_message(RevealSeq(record.source_seq, screen_id=SCREEN_ID))

    # -------------------------------------------------------------- actions --

    def action_filter(self) -> None:
        self.panel.focus_filter()

    def action_back(self) -> None:
        """One key, two meanings, decided by what has focus.

        `escape` out of the filter is what a person expects while typing in it;
        `escape` out of the screen is what they expect otherwise. Asking the
        panel which it is keeps the two from needing separate keys.
        """
        if self.panel.filtering:
            self.panel.focus_table()
            return
        if self.app.screen_stack and self is not self.app.screen_stack[0]:
            # A base screen has nothing to go back to. Asked as "am I the one
            # underneath" rather than as a depth, which would also be wrong the
            # first time this opened over anything other than the chat.
            self.dismiss()

    def action_fork(self) -> Session | None:
        """Fork the session at the selected record (A6).

        The refusals are the *store's*, not this view's: `fork` raises
        `SessionForkError` with a code and a sentence, and repeating that rule
        here is how the table came to mark one legal boundary in four while
        citing A6 at the other three. What this owns is only whether a fork is
        possible at all — a view reading a file has no store to fork in.

        Returns the child so a caller has the session rather than a string to
        parse out of a notification.
        """
        record = self.panel.selected()
        if record is None:
            self._report("no record selected", refused=True)
            return None
        if self.sessions is None:
            self._report(
                "forking needs a running harness; this view is reading a file", refused=True
            )
            return None
        try:
            child: Session = self.sessions.fork(self.session_id, record.source_seq)
        except SessionForkError as error:
            self._report(f"#{record.index}: {error}", refused=True)
            return None
        self._report(f"forked at #{record.index} → {child.id}")
        return child

    # -------------------------------------------------------------- header --

    def _show_header(self) -> None:
        forkable = sum(1 for record in self.records if record.fork_point)
        text = (
            f"{self.session_id or 'session'} · {len(self.records)} records · "
            f"{forkable} fork point(s)"
        )
        self.query_one("#trajectory-header", Static).update(
            Content(f"{text} — {self.notice}" if self.notice else text)
        )

    def _report(self, message: str, *, refused: bool = False) -> None:
        """Tell the person, and remember what they were told.

        `notify` is the project's idiom for dynamic text — markup-safe, with
        severity carrying refused-versus-done structurally instead of the reader
        inferring it from prose.
        """
        self.notice = message
        self.notify(
            message,
            title="fork",
            severity="warning" if refused else "information",
            markup=False,
        )
        self._show_header()


@dataclass(frozen=True, slots=True)
class _BuildTrajectory:
    """`build(session)` for the registered screen.

    Holds `ctx.sessions` rather than `ctx`: the factory lives as long as the
    registration, and the store is the only thing it needs from the row's
    activation scope.
    """

    sessions: Any

    def __call__(self, session: Session) -> TrajectoryScreen:
        return TrajectoryScreen(
            build_trajectory(session), session_id=session.id, sessions=self.sessions
        )


@plugin("tui-screen-trajectory", inject=["tui_screens", "sessions"])
async def apply(ctx: Context, config: Any) -> None:
    """Contribute the trajectory to whatever front end is drawing.

    `scope=ctx` is this row's activation scope, and it is what makes the
    registration an effect of *this row* — unloading it takes the screen, its
    `/trajectory` command and its key with it (I2).
    """
    ctx.tui_screens.register(
        ScreenDefinition(
            id=SCREEN_ID,
            label="Trajectory",
            order=10,
            key=TRAJECTORY_KEY,
            build=_BuildTrajectory(sessions=ctx.sessions),
        ),
        scope=ctx,
    )
