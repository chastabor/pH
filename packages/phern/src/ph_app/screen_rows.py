"""The rows that contribute pH's screens: what a front end is told, and no terminal UI.

A presentation row is layered into every named profile, headless included
(decision 24 of `plans/Session_Profiles_Plan.md`), so whatever it imports is paid
by every mount — `phern -p`, `--mode rpc`, and each root a daemon holds. The
screen itself is a Textual `Screen`, and importing it here cost a headless mount
the whole terminal UI: 166 ms became 320 ms.

So the row and the screen are two modules. What the daemon needs is what a client
is told over `screens/list` — the id, the label, the order and the key — and what
the client draws is built by its own copy of the definition
(`ph_app.tui.trajectory_screen.CLIENT_SIDE`), because `build` is the one field
that cannot travel. Outside `ph_app.tui` on purpose: that package's `__init__`
imports the app.

Two rows, one per screen, so each screen's verb and key go away with its own row:
the trajectory (`/trajectory`) and the diff (`/diff`, `ph_app.tui.diff_screen`).

@module ph_app.screen_rows
"""

from __future__ import annotations

from typing import NoReturn

from ph.cordis import Context, plugin
from ph.keys import TUI_SCREENS
from ph.seams.tui_screens import ScreenDefinition

__all__ = ["DIFF_KEY", "DIFF_SCREEN_ID", "SCREEN_ID", "TRAJECTORY_KEY", "diff", "trajectory"]

SCREEN_ID = "trajectory"
"""The trajectory's id in `ctx.tui_screens`, and so `/trajectory` and the binding id."""

TRAJECTORY_KEY = "f2"
"""Its default key. A default, not a rule: the id above is the binding id, so
`tui.json` rebinds it like any built-in (see `TuiKeybindings.extra`)."""


DIFF_SCREEN_ID = "diff"
"""The diff screen's id, and so `/diff` and its binding id."""

DIFF_KEY = "f3"
"""Its default key, rebindable by id like the trajectory's."""


def _built_by_the_front_end(session: object) -> NoReturn:
    """The daemon's `build`, which nothing calls: a client builds from its own copy,
    the screen module's `CLIENT_SIDE`."""
    raise RuntimeError(
        "this screen is built by the front end that draws it (its module's CLIENT_SIDE), "
        "not by the daemon"
    )


@plugin("tui-screen-trajectory", affects="presentation", inject=[TUI_SCREENS])
async def trajectory(ctx: Context, config: None) -> None:
    """Contribute the trajectory to whatever front end is drawing.

    `scope=ctx` is this row's activation scope, and it is what makes the
    registration an effect of *this row* — unloading it takes the screen, its
    `/trajectory` command and its key with it (I2).
    """
    ctx.require(TUI_SCREENS).register(
        ScreenDefinition(
            id=SCREEN_ID,
            label="Trajectory",
            order=10,
            key=TRAJECTORY_KEY,
            build=_built_by_the_front_end,
        ),
        scope=ctx,
    )


@plugin("tui-screen-diff", affects="presentation", inject=[TUI_SCREENS])
async def diff(ctx: Context, config: None) -> None:
    """Contribute the diff screen: every change the log records, before and after.

    The same shape as `trajectory` above, for the same reason: its verb and its key
    are effects of this row, gone when it unloads (I2).
    """
    ctx.require(TUI_SCREENS).register(
        ScreenDefinition(
            id=DIFF_SCREEN_ID,
            label="Diff",
            order=20,
            key=DIFF_KEY,
            build=_built_by_the_front_end,
        ),
        scope=ctx,
    )
