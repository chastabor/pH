"""Shared fixtures for the app tests: one app factory and one set of path roots."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest
from daemon_helpers import daemon_socket, running

from ph.paths import resolve_roots
from ph_app.profiles import compose_profile
from ph_app.tui.app import PHTuiApp


@pytest.fixture
def tui_profile() -> str:
    """Which profile the daemon behind the TUI mounts.

    `headless` because most of the TUI is not about the posture. A file whose
    subject *is* a row the interactive profile contributes — a screen, the
    question tool — overrides this fixture with `"tui"`, which is why it is a
    fixture rather than an argument to the factory: the daemon mounts before any
    app exists, so the choice has to be made before `make_tui_app` is called.
    """
    return "headless"


@pytest.fixture
async def tui_daemon(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tui_profile: str
) -> AsyncIterator[Any]:
    """The daemon the TUI attaches to, in this process and on a real socket.

    After P5-14 the terminal is a protocol client, so a pilot test needs a
    supervisor to talk to — and `$PH_RUNTIME` is pinned here so `ensure_daemon`
    resolves *this* socket, finds it listening, and never spawns a process.

    In-process rather than spawned so a test can still reach the harness: what
    used to be `app.front.ctx` is now `root_of(tui_daemon).ctx`, one side of a
    socket away from the screen asserting on it.
    """
    monkeypatch.setenv("PH_RUNTIME", str(tmp_path / "run"))
    async with running(tmp_path, path=daemon_socket(), profile=compose_profile(tui_profile)) as (
        daemon
    ):
        yield daemon


@pytest.fixture
async def make_tui_app(tmp_path: Path, tui_daemon: Any) -> Callable[..., PHTuiApp]:
    """`make_tui_app(**overrides)` → a TUI attached to `tui_daemon`.

    The construction itself is `tui_helpers.tui_app`, shared with the snapshot
    suite; what this fixture adds is the daemon to attach to. `$PH_HOME` comes
    from the root conftest's autouse `_isolated_home`.
    """
    from tui_helpers import tui_app

    def make(**overrides: Any) -> PHTuiApp:
        return tui_app(home=tmp_path, **overrides)

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
