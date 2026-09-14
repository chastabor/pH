"""The daemon: roots that outlive the clients watching them (P5-01).

**Nothing is re-exported here, deliberately.** A front end needs the *client* and
the protocol; importing either through this package would execute the server and
the supervisor with it — the whole harness, a `Profile`, every seam — for the sake
of one class. `ph_app.web.serve` is where that stopped being theoretical, and
`test_app_layering` is what holds it.

@module ph_app.daemon
"""

from __future__ import annotations

from .cancelsafe import apply_cancel_safe_socket_waits

__all__: list[str] = []

# Applied on import, because every socket that reaches anyio's unguarded
# readiness wait is created under this package — `server.create_unix_listener`,
# `client.connect_unix`, `launch.connect_unix` — and the guard has to be in
# force before the first one exists. A side effect in a package `__init__` is
# not this repo's habit, and what earns it an exception is that there is no
# *later* hook covering both directions: one import-time call instead of four
# call sites that must agree, three of them reached only through test helpers.
#
# A comment and not a docstring: a string after a *statement* is a discarded
# expression that no doc tool renders (`verbs.py` says the same of `assert`).
#
# It does not violate the rule above: `cancelsafe` imports `asyncio` and nothing
# of pH's, so no server and no supervisor comes with it, and `test_app_layering`
# still holds.
apply_cancel_safe_socket_waits()
