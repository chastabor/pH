"""Fixtures for the TUI tests: one app factory, so no test hand-builds `PHTuiApp`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ph_app.profiles import resolve_profile
from ph_app.tui.app import PHTuiApp


@pytest.fixture
def make_tui_app(tmp_path: Path) -> Callable[..., PHTuiApp]:
    """`make_tui_app(**overrides)` → a TUI over the headless profile, homed in `tmp_path`.

    Trusted by default: the trust prompt is a startup gate, so every test that
    is *not* about it would otherwise open a modal before the app exists.

    `$PH_HOME` comes from the root conftest's autouse `_isolated_home`; the copy
    that used to be here set the same variable to the same value and made one
    rule read as three.
    """

    def make(
        *,
        trusted: bool = True,
        profile: str = "headless",
        project: Path | None = None,
        **overrides: Any,
    ) -> PHTuiApp:
        options: dict[str, Any] = {
            "provider": "fake",
            "model": "fake-1",
            "session_id": "pilot",
            "home": tmp_path,
        }
        options.update(overrides)
        # `headless` by default because most of the TUI is not about the
        # posture; `profile="tui"` is what a test asking about a row the
        # interactive profile contributes — a screen, say — passes.
        app = PHTuiApp(resolve_profile(profile), **options)
        if project is not None:
            # `PHTuiApp.project` is `Path.cwd()`, and the sidebar prints it. A
            # snapshot taken here therefore embedded the developer's checkout —
            # `~/Projects/pH` — and could never match CI, whose cwd is
            # `~/work/pH/pH`. Set *before* `trust`, because trust is keyed on
            # this path and a mismatch opens the trust modal over the frame
            # under test.
            app.project = project
        if trusted:
            app.trust.trust(app.project)
        return app

    return make
