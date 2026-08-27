"""Fixtures for the TUI tests: one app factory, so no test hand-builds `PHTuiApp`."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from ph_app.profiles import resolve_profile
from ph_app.tui.app import PHTuiApp


@pytest.fixture
def make_tui_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Callable[..., PHTuiApp]:
    """`make_tui_app(**overrides)` → a TUI over the headless profile, homed in `tmp_path`.

    Trusted by default: the trust prompt is a startup gate, so every test that
    is *not* about it would otherwise open a modal before the app exists.
    """
    monkeypatch.setenv("PH_HOME", str(tmp_path))

    def make(*, trusted: bool = True, **overrides: Any) -> PHTuiApp:
        options: dict[str, Any] = {
            "provider": "fake",
            "model": "fake-1",
            "session_id": "pilot",
            "home": tmp_path,
        }
        options.update(overrides)
        app = PHTuiApp(resolve_profile("headless"), **options)
        if trusted:
            app.trust.trust(app.project)
        return app

    return make
