"""The auditor's screen (P4-17). The row that contributes it is in `ph_app.screen_rows`.

This is a `Screen` and not an `App`, so the same view serves both entry points.
`TrajectoryApp` composes one with records that are read from a file. That reads
a stored log with nothing mounted, which is the property that this shape was
chosen for. `ctx.tui_screens` composes one from the live session. The
standalone entry point is therefore a strict subset of the embedded one and not
a second implementation of it.

**Forking is the one action, and it is not offered everywhere.** A6 permits a
fork at a closed turn and nowhere else. So the table marks those rows, and the
key refuses the other rows by *saying which rows are targets* and not by a
rejection after the fact. The fork itself is `ctx.sessions.fork`, which needs a
session store. The action is therefore available when the view was opened from
a live harness. When the view was opened from a file, the action reports that
it is not available.

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

from ph.seams.tui_screens import ScreenDefinition
from ph.session import Session, SessionForkError

from ..screen_rows import SCREEN_ID, TRAJECTORY_KEY
from .screens import RevealHost, RevealSeq
from .trajectory import TrajectoryRecord, build_trajectory
from .widgets.trajectory import TrajectoryPanel

__all__ = ["CLIENT_SIDE", "SCREEN_ID", "TRAJECTORY_KEY", "TrajectoryScreen"]


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
    """The table holds focus, so the single-key actions reach the screen. `/`
    gives focus to the filter and `escape` takes it back. When the filter does
    not have focus, `escape` leaves the screen. A filter box that had focus by
    default would consume every one of these keys as typed text.

    Every binding carries an `id`, which is what `App.set_keymap` remaps
    against. A user who rebound a key in `tui.json` therefore gets it here
    too."""

    def __init__(
        self,
        records: list[TrajectoryRecord],
        *,
        session_id: str = "",
        sessions: Any = None,  # noqa: ANN401
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        super().__init__(**kwargs)
        self.records = records
        self.session_id = session_id
        self.sessions = sessions
        """`ctx.sessions` when a harness is running. It is `None` when the view
        reads a file. That is the ordinary case, and it is why the fork action
        reports and does not assume."""
        self.notice = ""
        """The words of the last action, kept so that a test can read what the
        user was told. `notify` tells the user, and this is the record of it."""
        self._opening_seq: int | None = None
        """The log seq to open at, when `reveal` named one before the mount."""

    # ---------------------------------------------------------------- shape --

    def compose(self) -> ComposeResult:
        yield Static(Content(""), id="trajectory-header")
        yield TrajectoryPanel(self.records, id="trajectory")

    def on_mount(self) -> None:
        self.panel.focus_table()
        self._show_header()
        if self._opening_seq is not None:
            # After a refresh, because the panel fills its table in its own
            # `on_mount`, and a cursor that is moved before the rows exist
            # lands nowhere.
            self.call_after_refresh(self.panel.select_seq, self._opening_seq)

    @property
    def panel(self) -> TrajectoryPanel:
        return self.query_one("#trajectory", TrajectoryPanel)

    # ------------------------------------------------- cross-navigation --

    def reveal(self, seq: int) -> None:
        """Open positioned at a log seq (the `Revealing` protocol).

        The seq is recorded and not applied, because the shell calls this
        before the screen is pushed. There is no table yet to move a cursor in.
        """
        self._opening_seq = seq

    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """A chosen row asks its host for the transcript counterpart.

        This is a message and not a call, because this screen does not know
        what pushed it. Whether anyone can answer is asked of the host
        structurally, and not by counting the screen stack. `--mode trajectory`
        has no transcript however deep it is mounted, and a shell that has one
        can answer from any depth.
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
        """One key with two meanings, decided by what has focus.

        `escape` out of the filter is what a user expects while typing in it.
        `escape` out of the screen is what they expect otherwise. The panel
        says which case it is, so the two do not need separate keys.
        """
        if self.panel.filtering:
            self.panel.focus_table()
            return
        if self.app.screen_stack and self is not self.app.screen_stack[0]:
            # A base screen has nothing to go back to. The test is "am I the
            # one underneath" and not a depth. A depth would also be wrong the
            # first time that this opened over anything other than the chat.
            self.dismiss()

    async def action_fork(self) -> Session | None:
        """Fork the session at the selected record (A6), **on disk before the user
        is told that it exists**.

        The `fork` of the store makes the child in memory, and nothing wrote
        it. A crash after "forked at …" lost a session that the user had been
        handed. A child that cannot be written is let go and reported as not
        kept.

        The refusals belong to the *store* and not to this view. `fork` raises
        `SessionForkError` with a code and a sentence. When that rule was
        repeated here, the table marked one legal boundary in four and cited A6
        at the other three. This view owns only whether a fork is possible at
        all: a view that reads a file has no store to fork in.

        Returns the child, so a caller has the session and not a string to
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
        if not await self.sessions.written(child):
            self.sessions.dispose(child.id)
            self._report(
                f"#{record.index}: the fork's log could not be written, so it was not kept",
                refused=True,
            )
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
        """Tell the user, and remember what they were told.

        `notify` is the idiom of the project for dynamic text. It is
        markup-safe, and its severity carries refused-versus-done structurally,
        so the reader does not infer it from prose.
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
    """`build(session)` for this screen. It holds the store that a fork needs, when there is one."""

    sessions: Any

    def __call__(self, session: Session) -> TrajectoryScreen:
        return TrajectoryScreen(
            build_trajectory(session), session_id=session.id, sessions=self.sessions
        )


CLIENT_SIDE = ScreenDefinition(
    id=SCREEN_ID,
    label="Trajectory",
    order=10,
    key=TRAJECTORY_KEY,
    build=_BuildTrajectory(sessions=None),
)
"""This screen as a front end *without* a harness can draw it (P5-14).

`build` is the one field of a `ScreenDefinition` that cannot travel. A remote
front end therefore needs a builder of its own for each screen that the daemon
lists. The place that knows how to build the trajectory is this module and not
the socket client. `sessions=None` is a state that the screen already supports
("`None` when the view reads a file"). Its cost is the fork action, which
`phern --mode trajectory --session <id|path>` gives up in the same way. The
`screens/list` of the daemon supplies `label` and `order` at attach time, so
the copies here are defaults and not a second truth.
"""
