"""`ph --mode trajectory` — the auditor's view over a stored log (P3-25).

**Nothing is mounted.** No agent, no provider, no approval answerers, no prompt
assembly, no plugins. `read_session` gives the events and
`ph_app.tui.trajectory.build_trajectory` gives the records; the whole view is a
fold over a file. That is the property the shape was chosen for: the logs an
auditor most wants to read are the ones nobody can re-open — a crashed run, a
child's transcript, the fixture from a replay report — and a view that could
only show the session you are already in reaches none of them.

**Forking is the one action, and it is not offered everywhere.** A6 permits a
fork at a closed turn and nowhere else, so the table marks those rows and the
key refuses the others by *saying which rows are targets* rather than by
rejecting after the fact. The fork itself is `ctx.sessions.fork`, which needs a
session store — so the action is available when the view was opened from a live
harness and reports honestly that it is not when the view was opened from a
file. Advertising a key that cannot work is the failure the plan names.

@module ph_app.tui.trajectory_app
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.content import Content
from textual.widgets import Static

from ph.paths import resolve_roots
from ph.persistence import read_session, repaired
from ph.persistence.jsonl import session_path
from ph.session import Session, SessionForkError

from .config import TuiSettings, load_tui_settings
from .themes import ThemeCatalog, fallback_variables, load_catalog
from .trajectory import TrajectoryRecord, build_trajectory
from .widgets.trajectory import TrajectoryPanel

__all__ = ["TrajectoryApp", "load_records", "run_trajectory"]


def load_records(
    target: str, *, root: Path | None = None, home: Path | None = None
) -> tuple[str, list[TrajectoryRecord]]:
    """Records for a session id or a path to a stored log.

    A path so a log outside the sessions directory — a fixture, a copy someone
    sent — is readable without being installed anywhere first. `root` is the
    persistence row's own directory when a caller knows it; the default is
    `sessions_dir()`, which is where that row puts logs unless a profile said
    otherwise.
    """
    candidate = Path(target)
    # A target that *looks* like a path is one: reinterpreting a missing file as
    # a session id reported `/nope.jsonl.jsonl` missing, naming a file nobody
    # typed.
    looks_like_path = candidate.suffix != "" or len(candidate.parts) > 1
    if looks_like_path:
        path = candidate
    else:
        roots = resolve_roots() if home is None else None
        base = root or (
            roots.sessions_dir() if roots is not None else Path(home or ".") / "sessions"
        )
        path = session_path(base, target)
    if not path.is_file():
        raise FileNotFoundError(f"no session log at {path}")
    header, events = read_session(path)
    # `repaired`, like a resume: a log whose tail was cut mid-turn is the
    # auditor's headline case, and seeding it raw would refuse the very session
    # someone opened this view to read.
    return header.id, build_trajectory(Session(header.id, seed=repaired(events), header=header))


class TrajectoryApp(App[None]):
    """The trajectory as its own top-level view.

    A separate `App` rather than a screen in `PHTuiApp`: the point is that it
    runs with nothing mounted, and sharing the chat app would mean sharing its
    mount, its trust prompt and its agent. The two are co-equal readers of one
    log, which is what `App.MODES` would express inside a single process — this
    is the same split at the process boundary, and it is what makes the
    harness-free entry point trivial rather than conditional.
    """

    ENABLE_COMMAND_PALETTE = False

    CSS = """
    Screen { background: $ph-background; color: $ph-foreground; }
    #trajectory-header { height: 1; background: $ph-panel; color: $ph-muted; padding: 0 1; }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "quit", id="quit"),
        Binding("f", "fork", "fork here", id="fork"),
        Binding("slash", "filter", "filter", id="filter"),
        Binding("escape", "leave_filter", "leave filter", id="leave_filter", show=False),
    ]
    """The table holds focus, so the single-key actions reach the app; `/` hands
    focus to the filter and `escape` takes it back. A filter box that had focus
    by default would eat every one of these as typed text.

    Every binding carries an `id`, which is what `App.set_keymap` remaps against
    — a user who rebound `quit` in `tui.json` gets it here too, rather than
    finding their key dead and an unremappable `q` in its place."""

    def __init__(
        self,
        records: list[TrajectoryRecord],
        *,
        session_id: str = "",
        sessions: Any = None,
        home: Path | None = None,
    ) -> None:
        super().__init__()
        self.records = records
        self.session_id = session_id
        self.sessions = sessions
        """`ctx.sessions` when a harness is running. `None` when reading a file,
        which is the ordinary case and why the fork action reports rather than
        assumes."""
        self.home = home or resolve_roots().home
        self.settings: TuiSettings = load_tui_settings(self.home)
        self.catalog: ThemeCatalog = load_catalog(self.home)
        self.notice = ""
        """The last action's own words, kept so a test can read what a person
        was told. The person is told by `notify`; this is the record of it."""

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def compose(self) -> ComposeResult:
        yield Static(Content(""), id="trajectory-header")
        yield TrajectoryPanel(self.records, id="trajectory")

    def on_mount(self) -> None:
        # The same chrome the chat app applies, from the same two files: a view
        # that read the theme catalog but not the settings opened in the default
        # theme however the user had set it.
        self.set_keymap(self.settings.keybindings.as_map())
        self.catalog.install(self)
        self.theme = self.catalog.resolve(self.settings.theme).name
        self.panel.focus_table()
        self._show_header()

    def action_filter(self) -> None:
        self.panel.focus_filter()

    def action_leave_filter(self) -> None:
        self.panel.focus_table()

    def _show_header(self) -> None:
        forkable = sum(1 for record in self.records if record.fork_point)
        text = (
            f"{self.session_id or 'session'} · {len(self.records)} records · "
            f"{forkable} fork point(s)"
        )
        self.query_one("#trajectory-header", Static).update(
            Content(f"{text} — {self.notice}" if self.notice else text)
        )

    @property
    def panel(self) -> TrajectoryPanel:
        return self.query_one("#trajectory", TrajectoryPanel)

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

    def _report(self, message: str, *, refused: bool = False) -> None:
        """Tell the person, and remember what they were told.

        `App.notify` is the project's idiom for dynamic text — markup-safe,
        with severity carrying refused-versus-done structurally instead of the
        reader inferring it from prose.
        """
        self.notice = message
        self.notify(
            message,
            title="fork",
            severity="warning" if refused else "information",
            markup=False,
        )
        self._show_header()


async def run_trajectory(target: str, *, home: Path | None = None) -> None:
    """Entry point for `--mode trajectory --session <id|path>`."""
    session_id, records = load_records(target, home=home)
    await TrajectoryApp(records, session_id=session_id, home=home).run_async()
