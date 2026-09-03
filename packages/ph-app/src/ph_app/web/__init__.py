"""The browser front end (P7-05). See `ph_app.web.serve`.

Nothing is imported here: `textual-serve` is an extra (`ph-app[web]`), so
importing this package must not require it. `ph_app.cli` imports `serve` inside
the `--mode web` branch, which is where a missing extra becomes an install line
rather than an `ImportError` a person has to read.

@module ph_app.web
"""

from __future__ import annotations

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT"]

DEFAULT_HOST = "127.0.0.1"
"""Loopback, so `--mode web` with no arguments exposes nothing to the network.

Here rather than on either of the two things that need it: `cli.py` spells it as
`--host`'s default and `WebServer` as a field default, and while they agreed by
accident a test asserting the *server's* would have passed for a CLI that had
quietly moved to `0.0.0.0`. This module is the one both can import at no cost —
its whole point is that importing it requires no extra."""

DEFAULT_PORT = 8000
"""What upstream's own default is, kept so a person who reads `textual-serve`'s
docs and a person who reads pH's find the same number."""
