"""The rows that contribute the screens of pH: what a front end is told, with no
terminal UI.

A presentation row is layered into every named profile, headless included
(decision 24 of `plans/Session_Profiles_Plan.md`). So every mount pays for
whatever the row imports: `phern -p`, `--mode rpc`, and each root that a daemon
holds. The screen itself is a Textual `Screen`. When this module imported it, a
headless mount paid for the whole terminal UI: 166 ms became 320 ms.

So the row and the screen are two modules. The daemon needs what a client is
told over `screens/list`: the id, the label, the order and the key. The client
builds what it draws from its own copy of the definition
(`ph_app.tui.trajectory_screen.CLIENT_SIDE`), because `build` is the one field
that cannot travel. This module is outside `ph_app.tui` on purpose: the
`__init__` of that package imports the app.

There are two rows, one for each screen, so the verb and the key of each screen
go away with its own row: the trajectory (`/trajectory`) and the diff (`/diff`,
`ph_app.tui.diff_screen`).

@module ph_app.screen_rows
"""

from __future__ import annotations

from typing import NoReturn

from ph.cordis import Context, plugin
from ph.keys import TUI_SCREENS
from ph.seams.tui_screens import ScreenDefinition

__all__ = ["DIFF_KEY", "DIFF_SCREEN_ID", "SCREEN_ID", "TRAJECTORY_KEY", "diff", "trajectory"]

SCREEN_ID = "trajectory"
"""The id of the trajectory in `ctx.tui_screens`. It is therefore also `/trajectory`
and the binding id."""

TRAJECTORY_KEY = "f2"
"""Its default key. This is a default and not a rule: the id above is the binding
id, so `tui.json` rebinds it like any built-in key (see `TuiKeybindings.extra`)."""


DIFF_SCREEN_ID = "diff"
"""The id of the diff screen. It is therefore also `/diff` and its binding id."""

DIFF_KEY = "f3"
"""Its default key. It is rebindable by id, like the key of the trajectory."""


def _built_by_the_front_end(session: object) -> NoReturn:
    """The `build` of the daemon, which nothing calls. A client builds from its
    own copy, the `CLIENT_SIDE` of the screen module."""
    raise RuntimeError(
        "this screen is built by the front end that draws it (its module's CLIENT_SIDE), "
        "not by the daemon"
    )


@plugin("tui-screen-trajectory", affects="presentation", inject=[TUI_SCREENS])
async def trajectory(ctx: Context, config: None) -> None:
    """Contribute the trajectory to whatever front end is drawing.

    `scope=ctx` is the activation scope of this row. It makes the registration
    an effect of *this row*: unloading the row takes the screen, its
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
    """Contribute the diff screen: every change that the log records, before and
    after.

    This has the same shape as `trajectory` above, for the same reason: its verb
    and its key are effects of this row, and they are gone when it unloads (I2).
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
