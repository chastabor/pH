"""What the CLI may drag in — the app layer's half of the layering rule.

Named `test_app_layering` rather than `test_layering`: pytest imports these test
modules by basename, so the two halves of one rule cannot share a filename.

ph-core has `FORBIDDEN`, an AST rule about *presentation* libraries it must never
name. This is the sibling question one layer up, and it is a different one:
`ph_app` may import Textual and aiohttp — that is its job — but importing
`ph_app.cli` must not, because a `ph -p` in a script pays for whatever the CLI
pulls in and may be running somewhere an optional extra was never installed.

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
"""


def test_the_cli_imports_nothing_optional() -> None:
    """Import the CLI in a fresh interpreter and ask what came with it.

    Sabotage: move any of `ph_app.tui.app` or `ph_app.web.serve` to the top of
    `cli.py`, and `ph -p` starts paying for a UI it will not draw — and fails
    outright wherever the extra is absent.
    """
    probe = f"import sys, ph_app.cli; print([n for n in {OPTIONAL!r} if n in sys.modules])"
    found = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )

    assert found.stdout.strip() == "[]", f"importing ph_app.cli dragged in {found.stdout.strip()}"
