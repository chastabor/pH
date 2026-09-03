"""What the CLI may drag in — the app layer's half of the layering rule.

Named `test_app_layering` rather than `test_layering`: pytest imports these test
modules by basename, so the two halves of one rule cannot share a filename.

ph-core has `FORBIDDEN`, an AST rule about *presentation* libraries it must never
name. This is the sibling question one layer up, and it is a different one:
`ph_app` may import Textual and aiohttp — that is its job — but importing
`ph_app.cli` must not, because a `ph -p` in a script pays for whatever the CLI
pulls in and may be running somewhere an optional extra was never installed.

The second rule here runs the other way. **A front end imports the daemon's
client, never its server**: the web process proxies tabs and stages blobs, and a
thing that could mount a session is a second supervisor competing for the same
leases (I-5). It was true by intention and false in fact — `ph_app/daemon/
__init__.py` re-exported `DaemonServer` and `Supervisor`, so importing the
*client* executed the whole harness.

**Asserted at runtime, not by AST**, and that is the point: the promise is about
the whole import graph, so a *transitive* pull — a module that imports a module
that imports Textual — is exactly the regression worth catching, and a source
scan of `cli.py` would miss it. It also has to tolerate the deliberate
in-function imports the CLI uses to keep this true, which an AST rule would flag
as violations.
"""

from __future__ import annotations

import subprocess
import sys

OPTIONAL = ("textual", "textual_serve", "aiohttp", "jinja2", "opentelemetry")
"""Heavy or extra-only packages `ph_app.cli` must not import to be loaded.

`textual` is the oldest of these promises and was untested until now: `cli.py`
says "the TUI pulls in Textual, and `ph -p` in a script should not pay for a
terminal UI it will never draw" and then defers the import inside the `--mode
tui` branch. `textual_serve`, `aiohttp` and `jinja2` arrive through
`ph-app[web]`, so for them the cost is not slowness but an `ImportError` in a
deployment that never asked for a web server — which is also why the `--mode web`
branch wraps its import in the one `try` that turns that into an install line.

`opentelemetry` is the odd one: ph-core imports it deliberately, through its own
`otel` extra. It is here for the same reason as the rest — a tracing stack is not
something a one-shot `ph -p` should load — and not because ph-core is wrong to
have it.
"""

FRONT_END_FORBIDS = ("ph_app.daemon.server", "ph_app.daemon.supervisor")
"""What a front end must not be able to reach by importing the web module.

Not a weight argument — it is the boundary. A process serving browser tabs talks
to a daemon over the socket; one that could *mount* a session would be a second
supervisor holding leases the first one owns.
"""


def _dragged_in(entry: str, forbidden: tuple[str, ...]) -> str:
    """Import `entry` in a fresh interpreter and report which of `forbidden` came.

    A subprocess because the promise is about a *cold* import: this interpreter
    has already imported half the tree to collect the tests, so `sys.modules`
    here answers a question nobody asked.
    """
    probe = f"import sys, {entry}; print([n for n in {forbidden!r} if n in sys.modules])"
    found = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    return found.stdout.strip()


def test_the_cli_imports_nothing_optional() -> None:
    """Import the CLI in a fresh interpreter and ask what came with it.

    Sabotage: move any of `ph_app.tui.app` or `ph_app.web.serve` to the top of
    `cli.py`, and `ph -p` starts paying for a UI it will not draw — and fails
    outright wherever the extra is absent.
    """
    dragged = _dragged_in("ph_app.cli", OPTIONAL)

    assert dragged == "[]", f"importing ph_app.cli dragged in {dragged}"


def test_a_front_end_imports_the_daemons_client_and_not_its_server() -> None:
    """The web server proxies tabs; it must not be able to become a daemon.

    Sabotage: re-export `DaemonServer` from `ph_app/daemon/__init__.py` — which
    is what it did until this test — and importing `ph_app.web.serve` loads the
    supervisor, every seam and a `Profile`, for the sake of `DaemonClient`.
    """
    dragged = _dragged_in("ph_app.web.serve", FRONT_END_FORBIDS)

    assert dragged == "[]", f"ph_app.web.serve dragged in {dragged}"


def test_the_human_door_needs_no_daemon_at_all() -> None:
    """`ph_app.attach` reads a person's file and builds a message; that is all.

    It gained `stage_bytes`, which takes a `DaemonClient` — under `TYPE_CHECKING`,
    so the annotation costs no import and the module stays usable by anything
    that has a client rather than only by things that live beside one. Nothing
    tested that guard, and a guard nobody tests is a guard somebody deletes while
    tidying imports.
    """
    dragged = _dragged_in("ph_app.attach", ("ph_app.daemon.client", *FRONT_END_FORBIDS))

    assert dragged == "[]", f"ph_app.attach dragged in {dragged}"
