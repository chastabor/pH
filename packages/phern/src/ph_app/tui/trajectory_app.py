"""`phern --mode trajectory` — the auditor's view over a stored log (P3-25).

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
from typing import ClassVar

from textual.app import App
from textual.binding import Binding, BindingType

from ph.paths import resolve_roots

from ..stored import load_session
from .config import TuiKeybindings, TuiSettings, load_tui_settings
from .themes import ThemeCatalog, ThemeProfile, fallback_variables, load_catalog, load_theme_profile
from .trajectory import TrajectoryRecord, build_trajectory
from .trajectory_screen import TrajectoryScreen

__all__ = ["TrajectoryApp", "load_records", "run_trajectory"]

_DEFAULT_KEYS = TuiKeybindings()
"""The pre-keymap defaults, read rather than retyped in `BINDINGS`.

An instance and not the class: `TuiKeybindings` is `slots=True`, so the class
attribute is the slot descriptor and `Binding(TuiKeybindings.quit, …)` would bind
a repr of one. `set_keymap` replaces both keys from `tui.json` on mount; these
are only what stands before it runs."""


def load_records(
    target: str, *, root: Path | None = None, home: Path | None = None
) -> tuple[str, list[TrajectoryRecord]]:
    """Records for a session id or a path to a stored log (`load_session`)."""
    session = load_session(target, root=root, home=home)
    return session.id, build_trajectory(session)


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
        # **Its own id** (H7). Bound as `quit`, `set_keymap` below rebinds it to
        # whatever `tui.json` says `quit` is — `ctrl+d` by default — so `q`
        # stopped working and the only documented way out of the viewer was
        # gone. The ids are a *shared* namespace across apps and this binding is
        # not the chat app's quit; the configured key is offered beside it, so
        # somebody who remapped `quit` still gets what they asked for.
        Binding(_DEFAULT_KEYS.trajectory_quit, "quit", "quit", id="trajectory_quit"),
        Binding(_DEFAULT_KEYS.quit, "quit", "quit", id="quit", show=False),
    ]

    def __init__(
        self,
        records: list[TrajectoryRecord],
        *,
        session_id: str = "",
        sessions: object = None,
        home: Path | None = None,
    ) -> None:
        super().__init__()
        self.home = home or resolve_roots().home
        self.settings: TuiSettings = load_tui_settings(self.home)
        self.catalog: ThemeCatalog = load_catalog(self.home)
        self.theme_profile: ThemeProfile = load_theme_profile(self.home)
        self.trajectory = TrajectoryScreen(records, session_id=session_id, sessions=sessions)

    def get_default_screen(self) -> TrajectoryScreen:
        """The view *is* this app's base screen — no push, no second layout."""
        return self.trajectory

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def on_mount(self) -> None:
        # The same chrome the chat app applies, from the same files: a view that
        # read the theme catalog but not the preference opened in the default
        # theme however the user had set it. Two files now, since the theme moved
        # out of `tui.json` (P9-02) — the keymap is still the settings'.
        self.set_keymap(self.settings.keybindings.as_map())
        self.catalog.install(self)
        self.theme = self.catalog.resolve(self.theme_profile.chosen).name


async def run_trajectory(target: str, *, home: Path | None = None) -> int:
    """Entry point for `--mode trajectory --session <id|path>`. Answers the exit code.

    The second Textual app in this CLI, and it had `run_tui`'s bug (M4/H3): its
    `return_code` was discarded, so a viewer that crashed exited 0 and anything
    scripting `--mode trajectory` could not tell a crash from a clean close.
    Returned rather than raised for `run_tui`'s reason — the entry point owns
    exiting.
    """
    session_id, records = load_records(target, home=home)
    app = TrajectoryApp(records, session_id=session_id, home=home)
    await app.run_async()
    return app.return_code or 0
