"""The daemon: roots that outlive the clients that watch them (P5-01).

**Nothing is re-exported here, deliberately.** A front end needs the *client* and
the protocol. An import of either through this package would execute the server
and the supervisor with it: the whole harness, a `Profile` and every seam, for
one class. `ph_app.web.serve` is where that problem became real, and
`test_app_layering` checks the rule.

@module ph_app.daemon
"""

from __future__ import annotations

from .cancelsafe import apply_cancel_safe_socket_waits

__all__: list[str] = []

# The guard is applied on import. Every socket that reaches the unguarded
# readiness wait of anyio is created by a call from this package:
# `server.create_unix_listener`, `client.connect_unix`, and the connect in
# `ph.framing.listening`, which `launch` and `server` call. The guard must be in
# force before the first socket exists. A side effect in a package `__init__` is
# not the habit of this repo. It is an exception here because no *later* hook
# covers both directions: one import-time call replaces four call sites that
# must agree, three of which are reached only through test helpers.
#
# This is a comment and not a docstring. A string after a *statement* is a
# discarded expression that no doc tool renders (`verbs.py` says the same of
# `assert`).
#
# The call does not violate the rule above. `cancelsafe` imports `asyncio` and
# nothing of pH, so no server and no supervisor comes with it, and
# `test_app_layering` still holds.
apply_cancel_safe_socket_waits()
