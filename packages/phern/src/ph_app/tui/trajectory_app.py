"""`phern --mode trajectory`: the auditor's view over a stored log (P3-25).

**Nothing is mounted.** There is no agent, no provider, no approval answerers,
no prompt assembly and no plugins. `ph_app.stored.load_session` reads the
events through `read_session`, and `ph_app.tui.trajectory.build_trajectory`
gives the records. The whole view is a fold over a file. The shape was chosen
for that property. The logs that an auditor most wants to read are the ones
that nobody can reopen: a crashed run, the transcript of a child, the fixture
from a replay report. A view that could show only the session that you are
already in reaches none of them.

This file now holds the loading and the chrome. The view itself is
`TrajectoryScreen`, which `ctx.tui_screens` also contributes to the chat app
(P4-17). So this entry point is a strict subset of the embedded one and not a
second implementation of it. The separate `App` of P3-25 had quietly broken
that.

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
"""The defaults that stand before the keymap is applied. `BINDINGS` reads them
from here and does not retype them.

This is an instance and not the class. `TuiKeybindings` is `slots=True`, so the
class attribute is the slot descriptor, and `Binding(TuiKeybindings.quit, …)`
would bind the repr of a descriptor. `set_keymap` replaces both keys from
`tui.json` on mount, so these values apply only until it runs."""


def load_records(
    target: str, *, root: Path | None = None, home: Path | None = None
) -> tuple[str, list[TrajectoryRecord]]:
    """Records for a session id or a path to a stored log (`load_session`)."""
    session = load_session(target, root=root, home=home)
    return session.id, build_trajectory(session)


class TrajectoryApp(App[None]):
    """The trajectory as its own top-level view.

    This is a separate `App` and not only a screen in `PHTuiApp`. Its purpose
    is to run with nothing mounted, and sharing the chat app would mean sharing
    its mount, its trust prompt and its agent. The two apps are equal readers
    of one log. This app supplies the theme, the keymap and a way out.
    `TrajectoryScreen` supplies everything else, which is exactly what the chat
    app gets from `ctx.tui_screens`.
    """

    ENABLE_COMMAND_PALETTE = False

    BINDINGS: ClassVar[list[BindingType]] = [
        # **Its own id** (H7). When this was bound as `quit`, `set_keymap`
        # below rebound it to whatever `tui.json` says `quit` is, which is
        # `ctrl+d` by default. So `q` stopped working, and the only documented
        # way out of the viewer was gone. The ids are a namespace that all apps
        # *share*, and this binding is not the quit of the chat app. The
        # configured key is offered beside it, so a user who remapped `quit`
        # still gets what they asked for.
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
        """The view *is* the base screen of this app. There is no push and no second layout."""
        return self.trajectory

    def get_theme_variable_defaults(self) -> dict[str, str]:
        return fallback_variables()

    def on_mount(self) -> None:
        # This applies the same chrome as the chat app, from the same files. A
        # view that read the theme catalog but not the preference opened in the
        # default theme, however the user had set it. There are two files now,
        # because the theme moved out of `tui.json` (P9-02). The keymap still
        # comes from the settings.
        self.set_keymap(self.settings.keybindings.as_map())
        self.catalog.install(self)
        self.theme = self.catalog.resolve(self.theme_profile.chosen).name


async def run_trajectory(target: str, *, home: Path | None = None) -> int:
    """Entry point for `--mode trajectory --session <id|path>`. Returns the exit code.

    This is the second Textual app in this CLI, and it had the bug of
    `run_tui` (M4/H3). Its `return_code` was discarded, so a viewer that
    crashed exited 0, and a script that ran `--mode trajectory` could not tell
    a crash from a clean close. The code is returned and not raised, for the
    same reason as in `run_tui`: the entry point owns exiting.
    """
    session_id, records = load_records(target, home=home)
    app = TrajectoryApp(records, session_id=session_id, home=home)
    await app.run_async()
    return app.return_code or 0
