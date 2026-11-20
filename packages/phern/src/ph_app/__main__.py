"""`python -m ph_app`: the CLI, reachable without a console script.

It exists for a process that pH starts for itself, such as the daemon that a UI
starts for the user. `phern_argv` in `ph_app.daemon.launch` builds that command
line. It says why the process must run on *this* interpreter and not on whatever
`phern` is on `PATH`.

@module ph_app.__main__
"""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    main()
