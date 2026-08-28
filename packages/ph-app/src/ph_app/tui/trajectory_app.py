"""`ph --mode trajectory` — the auditor's view over a stored log (P3-25).

**Nothing is mounted.** No agent, no provider, no approval answerers, no prompt
assembly, no plugins. `read_session` gives the events and
`ph_app.tui.trajectory.build_trajectory` gives the records; the whole view is a
fold over a file. That is the property the shape was chosen for: the logs an
auditor most wants to read are the ones nobody can re-open — a crashed run, a
child's transcript, the fixture from a replay report — and a view that could
only show the session you are already in reaches none of them.

What this file now *is* is the loading and the chrome. The view itself is
`TrajectoryScreen`, which `ctx.tui_screens` also contributes to the chat app
(P4-17) — so this entry point is a strict subset of the embedded one rather
than a second implementation of it, which is what P3-25's separate `App` had
quietly broken.

@module ph_app.tui.trajectory_app
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

from textual.app import App
from textual.binding import Binding, BindingType

from ph.paths import resolve_roots
from ph.persistence import read_session, repaired
from ph.persistence.jsonl import session_path
from ph.session import Session

from .config import TuiSettings, load_tui_settings
from .themes import ThemeCatalog, fallback_variables, load_catalog
from .trajectory import TrajectoryRecord, build_trajectory
from .trajectory_screen import TrajectoryScreen

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

    A separate `App` rather than only a screen in `PHTuiApp`: the point is that
    it runs with nothing mounted, and sharing the chat app would mean sharing
    its mount, its trust prompt and its agent. The two are co-equal readers of
    one log — this app supplies the theme, the keymap and a way out, and
    `TrajectoryScreen` supplies everything else, which is exactly what the chat
    app gets from `ctx.tui_screens`.
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("q", "quit", "quit", id="quit"),
    ]
    """The one binding that is the *app's* rather than the view's. `quit`
    carries its id so a user who rebound it in `tui.json` gets it here too."""

    def __init__(
        self,
        records: list[TrajectoryRecord],
        *,
        session_id: str = "",
        sessions: Any = None,
        home: Path | None = None,
    ) -> None:
        super().__init__()
        self.home = home or resolve_roots().home
        self.settings: TuiSettings = load_tui_settings(self.home)
        self.catalog: ThemeCatalog = load_catalog(self.home)
        self.trajectory = TrajectoryScreen(records, session_id=session_id, sessions=sessions)

    def get_default_screen(self) -> TrajectoryScreen:
        """The view *is* this app's base screen — no push, no second layout."""
        return self.trajectory

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def on_mount(self) -> None:
        # The same chrome the chat app applies, from the same two files: a view
        # that read the theme catalog but not the settings opened in the default
        # theme however the user had set it.
        self.set_keymap(self.settings.keybindings.as_map())
        self.catalog.install(self)
        self.theme = self.catalog.resolve(self.settings.theme).name


async def run_trajectory(target: str, *, home: Path | None = None) -> None:
    """Entry point for `--mode trajectory --session <id|path>`."""
    session_id, records = load_records(target, home=home)
    await TrajectoryApp(records, session_id=session_id, home=home).run_async()
