"""pH's terminal front-end.

Layered so the interesting parts are testable without a terminal:

* `state` — a plain data model of the transcript;
* `adapter` — session events folded into that state (the P2-01 gate);
* `frontend` — the harness bridge: which seams pH answers, and how a prompt
  becomes a turn;
* `app` — widgets, keys, and the worker a modal may be awaited in;
* `themes` / `config` — data under `$PH_HOME`, never code.

@module ph_app.tui
"""

from __future__ import annotations

from .app import PHTuiApp, run_tui

__all__ = ["PHTuiApp", "run_tui"]
