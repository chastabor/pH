"""Where `ph` writes, how it refuses, and how it lays a report out (P5-10).

Two consoles, one refusal and one section renderer, in one module, because there
are now two command modules — `cli.py` and `agents.py` — and `cli.py` imports
`agents.py`, so neither can own them without the other importing a command table
to get at a `Console`. A pair per module meant any console setting applied to
half the CLI, and the same argument decided `section` after three copies of it
had accumulated: `ph doctor`'s contributed sections, `ph agents status`/`doctor`,
and P5-11's socket lifetime. Every one of them is a title and a list of pairs,
and they were already drifting apart in style.

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

from collections.abc import Iterable, Sequence
from typing import Annotated, NoReturn, TypeAlias

import typer
from rich.console import Console
from rich.table import Table

from ph.selectors import Scheme, Selector, SelectorError, parse_all

__all__ = ["TypeOption", "console", "emit", "err", "fail", "section", "selectors_or_exit"]

console = Console(highlight=False)
err = Console(stderr=True, highlight=False)
"""`highlight=False` on both: Rich's automatic highlighter re-colours whatever
in a plain string *looks* like a number, a path or a UUID, which in a CLI's
prose is an arbitrary word coloured for looking like data — `scheduled sch-1 ·
interval 3600000` came out with the interval in cyan and nothing else. Explicit
markup still works; only the guessing is off. It is also a third of the cost of
printing a line, and `ph agents attach` prints one per log event."""


def section(title: str, rows: Iterable[tuple[str, str]]) -> Table:
    """A report section: a borderless two-column table of label/value pairs.

    The one shape every diagnostic in this CLI has. `PathRoots.describe()`,
    `RuntimeLifetime.describe()`, `DiagnosticsRegistry.report()`'s per-section
    rows and `ph agents status` all produce `list[tuple[str, str]]` already, so
    the renderer is the only thing that was ever per-command — and being
    per-command is exactly how `ph doctor` and `ph agents doctor` came to draw
    the same kind of answer with different titles and, briefly, different styles.
    """
    table = Table(show_header=False, title=title, title_justify="left")
    table.add_column(style="bold")
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


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


TypeOption: TypeAlias = Annotated[
    list[str],
    typer.Option("--type", help="Namespace selector, repeatable: workspace, workspace/acquired."),
]
"""Declared once, for `ProfileOption`'s reason one module over: two commands that
spell their own `--type` can come to disagree about what it means.

Here rather than in `cli.py` because `cli.py` imports the sub-apps, so a sub-app
that wanted the alias would have to import back into it — the cycle this module
was carved out to avoid, and which `documents_or_exit` already sits below.
"""


def selectors_or_exit(patterns: Sequence[str], *, vocabulary: Scheme) -> list[Selector]:
    """Parse `--type` for one command, or exit 2 saying what was wrong.

    The refusal is the command's, not the parser's: `parse_all` raises a
    `SelectorError` that already names the offending selector and the vocabulary
    this surface serves, and every caller wants that sentence on stderr under the
    same exit code — `documents_or_exit`'s argument, applied to the other thing a
    command parses before it can do anything.
    """
    try:
        return parse_all(patterns, vocabulary=vocabulary)
    except SelectorError as error:
        fail(f"[red]{error}[/red]", code=2, cause=error)
