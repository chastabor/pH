"""`python -m ph_app` — the CLI, reachable without a console script.

Exists for the daemon a UI starts on a person's behalf: `spawn_command` in
`ph_app.cli` says why it must be *this* interpreter rather than whatever `ph`
is on `PATH`.

@module ph_app.__main__
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
