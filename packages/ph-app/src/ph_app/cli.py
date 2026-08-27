"""`ph` — the command line.

Phase 0 ships four things: a one-shot print mode, `--dump-config` (so a
composed profile is inspectable before it runs), `ph doctor` (which prints the
three resolved path roots and how `$PH_RUNTIME` was reached), and `ph events`
(the producer/consumer matrix, generated from the declaration registry rather
than hand-maintained).

@module ph_app.cli
"""

from __future__ import annotations

import json
import sys
from functools import partial
from typing import Annotated

import anyio
import typer
import yaml
from rich.console import Console
from rich.table import Table

from ph.cordis import Loader, import_plugin_modules
from ph.cordis.events import events as event_registry
from ph.paths import RuntimeDirError, resolve_roots

from .modes.print_mode import run_print
from .profiles import BUILTIN_PROFILES, resolve_profile

__all__ = ["app", "main"]

app = typer.Typer(
    name="ph",
    help="pH — a plugin-composed agent harness.",
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)
console = Console()
err = Console(stderr=True)

DEFAULT_PROFILE = "headless"


@app.callback()
def default(
    ctx: typer.Context,
    prompt: Annotated[
        str | None, typer.Option("-p", "--print", help="Run one prompt and print the answer.")
    ] = None,
    profile: Annotated[
        str, typer.Option("--profile", help="Profile name or path to a .yaml.")
    ] = DEFAULT_PROFILE,
    provider: Annotated[str, typer.Option("--provider")] = "fake",
    model: Annotated[str, typer.Option("--model")] = "fake-1",
    session_id: Annotated[
        str | None, typer.Option("--session", help="Session id to create.")
    ] = None,
    dump_config: Annotated[
        bool, typer.Option("--dump-config", help="Print the composed rows and exit.")
    ] = False,
) -> None:
    """Run a prompt, or dump the composed configuration."""
    if ctx.invoked_subcommand is not None:
        return
    try:
        documents = resolve_profile(profile)
    except ValueError as error:
        err.print(f"[red]{error}[/red]")
        raise typer.Exit(code=2) from error

    if dump_config:
        loader = Loader.from_paths(documents)
        console.print(
            yaml.safe_dump(loader.dump(), sort_keys=False, default_flow_style=False).rstrip()
        )
        return

    if prompt is None:
        console.print(ctx.get_help())
        return

    result = anyio.run(
        partial(run_print, documents, prompt, provider=provider, model=model, session_id=session_id)
    )
    console.print(result.text)
    err.print(
        f"[dim]session {result.session_id} · {result.events} events · {result.log_path}[/dim]"
    )


@app.command()
def doctor() -> None:
    """Report the resolved path roots and how each was reached."""
    try:
        roots = resolve_roots()
    except RuntimeDirError as error:
        err.print(f"[red]$PH_RUNTIME check failed:[/red] {error}")
        raise typer.Exit(code=1) from error
    table = Table(title="pH path roots", show_header=True, header_style="bold")
    table.add_column("root")
    table.add_column("resolved")
    for name, value in roots.describe():
        table.add_row(name, value)
    console.print(table)
    console.print(f"[dim]platform: {sys.platform} · python {sys.version.split()[0]}[/dim]")
    console.print(f"[dim]profiles: {', '.join(sorted(BUILTIN_PROFILES))}[/dim]")


@app.command()
def events(as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False) -> None:
    """Print the event producer/consumer matrix.

    Generated from the declaration registry, so it cannot drift from the code
    the way a hand-written table does. Declarations live in the plugin modules
    that own them, so every registered plugin is imported first — third-party
    wheels included.
    """
    import_plugin_modules()
    matrix = event_registry.matrix()
    if as_json:
        console.print_json(json.dumps(matrix))
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("event")
    table.add_column("mode")
    table.add_column("payload")
    table.add_column("producer")
    table.add_column("what it is")
    for row in matrix:
        table.add_row(row["name"], row["mode"], row["payload"] or "", row["producer"], row["doc"])
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
