"""Where `ph` writes, and how it refuses (P5-10).

Two consoles and one refusal, in one module, because there are now two command
modules — `cli.py` and `agents.py` — and `cli.py` imports `agents.py`, so
neither can own them without the other importing a command table to get at a
`Console`. A pair per module meant any console setting applied to half the CLI.

**Data does not go through a console.** `--dump-config` writes YAML and `ph
events --json` writes JSON, both meant to be piped; Rich colourizes a plain
string it is asked to print, so with `FORCE_COLOR` set in the environment — as
CI images and many shells do — those two commands emitted ANSI escapes into
their own machine-readable output and `yaml.safe_load` refused it with
"unacceptable character #x001b". `emit` is the plain-bytes path for anything a
program reads; `console` is for anything a person does.

@module ph_app.console
"""

from __future__ import annotations

from typing import NoReturn

import typer
from rich.console import Console

__all__ = ["console", "emit", "err", "fail"]

console = Console(highlight=False)
err = Console(stderr=True, highlight=False)
"""`highlight=False` on both: Rich's automatic highlighter re-colours whatever
in a plain string *looks* like a number, a path or a UUID, which in a CLI's
prose is an arbitrary word coloured for looking like data — `scheduled sch-1 ·
interval 3600000` came out with the interval in cyan and nothing else. Explicit
markup still works; only the guessing is off. It is also a third of the cost of
printing a line, and `ph agents attach` prints one per log event."""


def emit(text: str) -> None:
    """Write machine-readable output, unstyled and unwrapped.

    `print` rather than `console.print`: a console decides colour from the
    environment and width from the terminal, and both are wrong for a document
    another program parses. The one thing this must never do is be clever.
    """
    print(text)


def fail(message: str, *, code: int = 1, cause: BaseException | None = None) -> NoReturn:
    """Print a refusal on stderr and exit with it.

    The two lines were written out at every refusal in both command modules,
    which is one user-facing sentence and one exit code per site to keep in
    step; the `$PH_RUNTIME` refusal had already been copied verbatim between
    them. `cause` keeps the chaining, because a traceback that lost the original
    is the thing `--pdb` was going to be used for.
    """
    err.print(message)
    raise typer.Exit(code=code) from cause
