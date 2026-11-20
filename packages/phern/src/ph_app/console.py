"""Where `phern` writes, how it refuses, and how it lays out a report (P5-10).

This module holds two consoles, one refusal and one section renderer. They are
in one module because there are several command modules (`cli.py`, `agents.py`,
`service.py` and others), and `cli.py` imports the others. No command module can
own the consoles without another one importing a command table to reach a
`Console`. When each module had its own pair, a console setting applied to only
half of the CLI. The same argument decided `section`, after three copies of it
had accumulated: the contributed sections of `phern doctor`,
`phern agents status` and `phern agents doctor`, and the socket lifetime of
P5-11. Each of them is a title and a list of pairs, and their styles were
already drifting apart.

**Data does not go through a console.** `--dump-config` writes YAML and
`phern events --json` writes JSON, and both are meant to be piped. Rich
colorizes a plain string that it is asked to print. With `FORCE_COLOR` set in
the environment, as CI images and many shells set it, those two commands wrote
ANSI escapes into their own machine-readable output, and `yaml.safe_load`
refused that output with "unacceptable character #x001b". `emit` is the
plain-bytes path for anything that a program reads. `console` is for anything
that a person reads.

@module ph_app.console
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Annotated, NoReturn, TypeAlias

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from ph.paths import PathRoots, RuntimeDirError, resolve_roots
from ph.selectors import Scheme, Selector, SelectorError, parse_all

__all__ = [
    "TypeOption",
    "console",
    "detail",
    "emit",
    "err",
    "fail",
    "fail_unmounted",
    "nonblank",
    "roots_or_exit",
    "section",
    "selectors_or_exit",
]

console = Console(highlight=False, soft_wrap=True)
err = Console(stderr=True, highlight=False, soft_wrap=True)
"""Two settings are on **both** consoles. That is the reason why the pair is
declared here and not in each module: when a console setting applies to half of
the CLI, the two halves come to disagree about what a line looks like.

`highlight=False`: the automatic highlighter of Rich colors again whatever in a
plain string *looks* like a number, a path or a UUID. In the prose of a CLI that
is an arbitrary word, colored because it looks like data.
`scheduled sch-1 · interval 3600000` came out with the interval in cyan and
nothing else colored. Explicit markup still works, and only the guessing is off.
The highlighter is also a third of the cost of printing a line, and
`phern agents attach` prints one line for each log event.

`soft_wrap=True`: the lines of this CLI *name things*: a profile path, a session
id, a socket, a `loginctl` command to run. The default of Rich folds at the
console width with a hard newline, **inside a word** when the word is longer
than what is left of the line. A path that is broken across two lines cannot be
copied and cannot be found with grep. Under a test runner the width is 80, and a
macOS temp path is a dozen characters longer than the same path on Linux. That
is exactly the margin by which `broken.yaml` stopped appearing in its own
refusal there. The terminal still wraps what does not fit, but it no longer
edits the sentence to do so.

**The setting does not reach `section()` below.** `soft_wrap` is a print-time
setting, and a `Table` folds inside its own cells. A path printed as a table
*value* therefore still breaks: the roots of `phern doctor`, the rows of
`RuntimeLifetime.describe()`. This is stated here and not left to be discovered,
because this module owns both the consoles and the table. It is therefore the
only place that could make "a path that this CLI prints can be copied" true
everywhere. Today that is true of every line and not of table cells."""


def section(title: str, rows: Iterable[tuple[str, str]]) -> Table:
    """Return a report section: a two-column table of label/value pairs, with no header.

    Every diagnostic in this CLI has this one shape. `PathRoots.describe()`,
    `RuntimeLifetime.describe()`, the rows of each section of
    `DiagnosticsRegistry.report()` and `phern agents status` all produce
    `list[tuple[str, str]]` already. The renderer was the only part that was ever
    written for each command. That is exactly how `phern doctor` and
    `phern agents doctor` came to draw the same kind of answer with different
    titles and, for a short time, different styles.
    """
    table = Table(show_header=False, title=title, title_justify="left")
    table.add_column(style="bold")
    table.add_column()
    for label, value in rows:
        table.add_row(label, value)
    return table


def emit(text: str) -> None:
    """Write machine-readable output, unstyled and unwrapped.

    This uses `print` and not `console.print`. A console decides color from the
    environment and width from the terminal, and both are wrong for a document
    that another program parses. This function must never do anything clever.
    """
    print(text)


def detail(text: object) -> str:
    """Make a message from somewhere else safe to interpolate into markup.

    **Rich reads `[...]` in a plain string as a style tag and renders nothing.**
    Every refusal in this CLI is an f-string. It mixes markup that this module
    wrote (`[red]`) with a sentence that it did not write (the text of an
    exception). Rows in six other distributions write the second half. They have
    no reason to know that a square bracket is syntax here, and every reason to
    use one: an *extra* is spelled `phern[local]`, and naming what to install is
    the purpose of those messages.

    `--mode web` without its extra therefore used to advise
    `pip install 'phern'`, which is the command that the user had already run.
    `text-index-local` refused by naming `ph-text-index` and not
    `ph-text-index[local]`. Both printed the one word that mattered as the empty
    string, in the sentence whose only job was to carry it.

    The escape is applied at the interpolation and not inside `fail`, because
    the argument of `fail` *is* markup, and escaping it would print `[red]` to
    the user.
    """
    return escape(str(text))


def fail(message: str, *, code: int = 1, cause: BaseException | None = None) -> NoReturn:
    """Print a refusal on stderr and exit with it.

    One user-facing sentence and one exit code are in one place, and not a pair
    to keep in step at every refusal in the command modules. `cause` keeps the
    chaining, because a traceback that lost the original error is the thing that
    `--pdb` was going to be used for.

    The no-fold property belongs to the console and not to this function. See
    `err` above.
    """
    err.print(message)
    raise typer.Exit(code=code) from cause


def fail_unmounted(profile: str, error: BaseException) -> NoReturn:
    """Print the refusal that every command that mounts a profile owes, and exit.

    Three commands had written this sentence: `phern doctor`,
    `phern workspaces gc` and `phern attachments gc`. Each copied the last. Three
    is also the count that the docstring of this module gives for the extraction
    of `section`. The argument is the same at all three. A profile that refuses
    to mount is the most important thing that the command can report. A user who
    ran the command *because* something is wrong must get the sentence and not a
    traceback.
    """
    fail(f"[red]profile {profile!r} does not mount:[/red] {detail(error)}", cause=error)


def nonblank(value: str | None) -> str | None:
    """The Typer callback for an optional text option: refuse an empty value.

    A flag that was left out is `None`. A typed `""` names nothing, so it is
    refused and never read as "left out". When an empty value was read that way,
    `--session ""` with an unset shell variable widened a command to every
    session. The check is one callback on the option, so a new option gets it
    by declaring it, and the refusal comes at parse time, before any work.
    """
    if value == "":
        raise typer.BadParameter("must not be empty; leave the option out instead")
    return value


TypeOption: TypeAlias = Annotated[
    list[str],
    typer.Option("--type", help="Namespace selector, repeatable: workspace, workspace/acquired."),
]
"""Declared one time, for the same reason as `ProfileOption` one module over: two
commands that each spell their own `--type` can come to disagree about what it
means.

The alias is here and not in `cli.py`, because `cli.py` imports the sub-apps. A
sub-app that wanted the alias would have to import back into `cli.py`. That is
the cycle that this module was made to avoid, and `profile_or_exit` already sits
below that cycle.
"""


def roots_or_exit(*, create: bool = False) -> PathRoots:
    """Return the three path roots, or exit and say why `$PH_RUNTIME` is not usable.

    Every command that finds the socket of the daemon, or binds it, resolves the
    roots first. A runtime directory can fail its checks: it is not ours, it is
    not 0700, or it is a symlink. That is one refusal with one sentence wherever
    it happens.
    """
    try:
        return resolve_roots(create=create)
    except RuntimeDirError as error:
        fail(f"[red]$PH_RUNTIME check failed:[/red] {detail(error)}", cause=error)


def selectors_or_exit(patterns: Sequence[str], *, vocabulary: Scheme) -> list[Selector]:
    """Parse `--type` for one command, or exit 2 and say what was wrong.

    The refusal belongs to the command and not to the parser. `parse_all` raises
    a `SelectorError` that already names the offending selector and the
    vocabulary that this surface serves. Every caller wants that sentence on
    stderr under the same exit code. This is the argument of `profile_or_exit`,
    applied to the other thing that a command parses before it can do anything.
    """
    try:
        return parse_all(patterns, vocabulary=vocabulary)
    except SelectorError as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
