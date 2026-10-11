"""What the CLI may drag in — the app layer's half of the layering rule.

Named `test_app_layering` rather than `test_layering`: pytest imports these test
modules by basename, so the two halves of one rule cannot share a filename.

ph-core has `FORBIDDEN`, an AST rule about *presentation* libraries it must never
name. This is the sibling question one layer up, and it is a different one:
`ph_app` may import Textual and aiohttp — that is its job — but importing
`ph_app.cli` must not, because a `phern -p` in a script pays for whatever the CLI
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

OPTIONAL = ("textual", "textual_serve", "aiohttp", "jinja2", "opentelemetry", "croniter")
"""Heavy or extra-only packages `ph_app.cli` must not import to be loaded.

`textual` is the oldest of these promises and was untested until now: `cli.py`
says "the TUI pulls in Textual, and `phern -p` in a script should not pay for a
terminal UI it will never draw" and then defers the import inside the `--mode
tui` branch. `textual_serve`, `aiohttp` and `jinja2` arrive through
`phern[web]`, so for them the cost is not slowness but an `ImportError` in a
deployment that never asked for a web server — which is also why the `--mode web`
branch wraps its import in the one `try` that turns that into an install line.

`opentelemetry` is the odd one: ph-core imports it deliberately, through its own
`otel` extra. It is here for the same reason as the rest — a tracing stack is not
something a one-shot `phern -p` should load — and not because ph-core is wrong to
have it.

`croniter` is a ph-core dependency and so always installed, but it costs about 20ms
to import. `ph.seams.schedule` loads with every host, and so with the CLI, while
only a cron schedule needs it, so the module imports it where one is read. N3 once
lifted it to the top, which nothing here caught.
"""

FRONT_END_FORBIDS = ("ph_app.daemon.server", "ph_app.daemon.supervisor")
"""What a front end must not be able to reach by importing the web module.

Not a weight argument — it is the boundary. A process serving browser tabs talks
to a daemon over the socket; one that could *mount* a session would be a second
supervisor holding leases the first one owns.

Two names rather than the subpackage, because a front end *does* import
`ph_app.daemon.client` — talking to the daemon over the socket is the whole
arrangement. What it must not reach is the half that mounts. The human door
below is the one that may name the package, and does.
"""


BUNDLES = ("ph_rlm", "ph_stabilize", "ph_text_index", "ph_code_graph", "ph_clm")
"""The bundles `phern` depends on and `ph_app` may not import (`pyproject.toml`): each
reaches the app as a third-party wheel would, through the `ph.bundles` entry points."""


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
    `cli.py`, and `phern -p` starts paying for a UI it will not draw — and fails
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


def test_the_app_imports_no_bundle() -> None:
    """The rule `pyproject.toml` states, held where it was broken once: the trajectory
    viewer reads a stored log with nothing mounted, and imported ph-clm's kinds leaf by
    name to know its types. It renders a type it does not know from its payload now.

    Sabotage: `from ph_clm import kinds` in `ph_app/__init__.py`.
    """
    dragged = _dragged_in("ph_app.cli, ph_app.daemon.server, ph_app.tui.trajectory_app", BUNDLES)

    assert dragged == "[]", f"the app dragged in {dragged}"


def test_the_human_door_needs_no_daemon_at_all() -> None:
    """`ph_app.attach` reads a person's file and builds a message; that is all.

    It gained `stage_bytes`, which takes a `DaemonClient` — under `TYPE_CHECKING`,
    so the annotation costs no import and the module stays usable by anything
    that has a client rather than only by things that live beside one. Nothing
    tested that guard, and a guard nobody tests is a guard somebody deletes while
    tidying imports.

    **The whole subpackage, because this module needs none of it.** The list was
    `daemon.client` plus the two above — an allowlist wearing a denylist's
    clothes, since importing any *other* module under `ph_app.daemon` passed.
    One did: the request-params models briefly lived at `ph_app.daemon.methods`,
    so typing `stage_bytes`'s sends put a runtime edge from the human door into
    the subpackage, and this test could not see it. The models moved to
    `ph_app.params` — direction places them, not ownership — and naming the
    package is the promise the docstring above was already making in words.
    """
    dragged = _dragged_in("ph_app.attach", ("ph_app.daemon",))

    assert dragged == "[]", f"ph_app.attach dragged in {dragged}"


def test_a_headless_mount_loads_no_terminal_ui() -> None:
    """Every named profile layers pH's presentation rows, headless included, so a
    row that imported its screen made every mount pay for the terminal it will not
    draw — `phern -p` and each daemon root, 166 ms against 320 ms. The row states
    the screen (`ph_app.screen_rows`); the front end builds it.

    A mount, not an import: the cost was paid when the row's entry point resolved,
    which importing `ph_app.cli` never reaches. Sabotage: point the
    `tui-screen-trajectory` entry point back at `ph_app.tui.trajectory_screen`.
    """
    probe = (
        "import sys, anyio\n"
        "from ph_app.profiles import compose_profile\n"
        "from ph_app.runtime import mounted\n"
        "async def main():\n"
        "    async with mounted(compose_profile('headless')):\n"
        "        pass\n"
        "anyio.run(main)\n"
        "print([n for n in ('textual', 'ph_app.tui') if n in sys.modules])\n"
    )
    found = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert found.stdout.strip().splitlines()[-1] == "[]", f"a headless mount loaded {found.stdout}"
