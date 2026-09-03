"""The daemon: roots that outlive the clients watching them (P5-01).

**Nothing is re-exported here, deliberately.** A front end needs the *client* and
the protocol; importing either through this package would execute the server and
the supervisor with it — the whole harness, a `Profile`, every seam — for the sake
of one class. `ph_app.web.serve` is where that stopped being theoretical, and
`test_app_layering` is what holds it.

@module ph_app.daemon
"""

from __future__ import annotations

__all__: list[str] = []
