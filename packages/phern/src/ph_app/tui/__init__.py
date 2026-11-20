"""The terminal front end of pH.

It is layered so that the interesting parts are testable without a terminal:

* `state`: a plain data model of the transcript.
* `adapter`: session events folded into that state (the P2-01 gate).
* `frontend`: the two protocols, which say what the terminal needs from a
  harness and what draws its modals.
* `remote`: the implementation of `frontend` that ships, a client of the daemon.
* `app`: widgets, keys, and the worker in which a modal may be awaited.
* `screens`: what a screen that a *row* contributed becomes here: a verb, a key
  and a palette entry, all of them effects of the registration of that row.
* `themes` / `config`: data under `$PH_HOME`, never code.

@module ph_app.tui
"""

from __future__ import annotations

from .app import PHTuiApp, run_tui

__all__ = ["PHTuiApp", "run_tui"]
