"""Shared fixtures for the app tests: one app factory and one set of path roots."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ph.paths import resolve_roots
from ph_app.profiles import compose_profile
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
        app = PHTuiApp(compose_profile(profile), **options)
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


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """All three path roots under `tmp_path`, and the sessions directory.

    Two modules had pinned these same three variables from their own `_roots`
    helper, and one of the three was already redundant — the root conftest's
    `_isolated_home` pins `$PH_HOME` autouse. A fixture rather than a called
    helper so the pinning happens before the test body rather than being
    something each test has to remember to do first.

    The returned path is `PathRoots.sessions_dir()`, created, rather than a
    hand-built `tmp_path / "home" / "sessions"`: a test that re-derives the
    layout is one that keeps passing after the layout moves.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path / "home"))
    monkeypatch.setenv("PH_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "run"))
    sessions = resolve_roots().sessions_dir()
    sessions.mkdir(parents=True, exist_ok=True)
    return sessions
