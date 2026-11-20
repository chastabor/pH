"""The browser front end (P7-05). See `ph_app.web.serve`.

Nothing is imported here. `textual-serve` is an extra (`phern[web]`), so an
import of this package must not require it. `ph_app.cli` imports `serve` inside
the `--mode web` branch. There a missing extra becomes an install line, and not
an `ImportError` that the user must read.

@module ph_app.web
"""

from __future__ import annotations

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT"]

DEFAULT_HOST = "127.0.0.1"
"""Loopback, so that `--mode web` with no arguments exposes nothing to the network.

The value is here and not in one of the modules that need it. `cli.py` uses it
as the default of `--host`, and `WebServer` uses it as a field default. While
those two agreed only by accident, a test that asserted the default of the
*server* would have passed for a CLI that had silently moved to `0.0.0.0`.
`daemon.probes` also uses it, as the host of a bind that names only a port. Each
of them can import this module at no cost: the purpose of this module is that an
import of it requires no extra."""

DEFAULT_PORT = 8000
"""The default of upstream itself. It is kept so that a person who reads the docs
of `textual-serve` and a person who reads the docs of pH find the same number."""
