"""`ph` — the command line.

Four output modes and three diagnostics.

`--mode` decides what reaches stdout, and the choice matters more than a flag
usually does: `json` and `rpc` emit the session log's **own** envelopes rather
than a per-mode rendering (I-7), so a wrapper streaming from a pipe and a tool
reading the stored JSONL parse one format — and dsh's tooling reads both (Q2).
`text` and `transcript` are for people, and `tui` is the interactive one — the
only mode that takes no `--print`, because the prompt is the interface.

`--dump-config` prints the composed rows before anything runs, `ph doctor` the
three resolved path roots, and `ph events` the producer/consumer matrix
generated from the declaration registry rather than hand-maintained.

@module ph_app.cli
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from functools import partial
from typing import Annotated, Any, Literal, TypeAlias

import anyio
import typer
import yaml
from rich.console import Console
from rich.table import Table

from ph.cordis import Loader, import_plugin_modules
from ph.cordis.events import events as event_registry
from ph.paths import RuntimeDirError, resolve_roots

from .modes import run_json, run_print, run_rpc, run_transcript
from .profiles import available_profiles, resolve_profile

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

OutputMode = Literal["text", "json", "transcript", "rpc", "tui", "trajectory"]

ModeRunner: TypeAlias = Callable[..., Awaitable[Any]]
"""Each mode returns its own result type — `json` reports a count, `text` and
`transcript` report text — so the table is typed by what they have in common:
they are awaited, and the caller branches on the mode it asked for."""

_MODES: dict[str, ModeRunner] = {
    "text": run_print,
    "json": run_json,
    "transcript": run_transcript,
}


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
        str | None,
        typer.Option("--session", help="Session id to create, or to read under --mode trajectory."),
    ] = None,
    mode: Annotated[
        OutputMode,
        typer.Option("--mode", help="text (default), json, transcript, rpc, tui, or trajectory."),
    ] = "text",
    resume: Annotated[
        str | None, typer.Option("--resume", help="Session id to reopen (tui only).")
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

    if mode == "trajectory":
        # The auditor's view. Deliberately *before* any profile work: it mounts
        # nothing — no agent, no provider, no answerers — because the logs worth
        # auditing are the ones nobody can reopen (P3-25).
        if session_id is None:
            err.print("[red]--mode trajectory needs --session <id|path>[/red]")
            raise typer.Exit(code=2)
        from .tui.trajectory_app import run_trajectory

        try:
            anyio.run(partial(run_trajectory, session_id))
        except (OSError, ValueError) as error:
            err.print(f"[red]{error}[/red]")
            raise typer.Exit(code=2) from error
        return

    if mode == "rpc":
        # No prompt: the peer drives the session over stdio.
        anyio.run(partial(run_rpc, documents, provider=provider, model=model))
        return

    if mode == "tui":
        # Imported here because the TUI pulls in Textual, and `ph -p` in a
        # script should not pay for a terminal UI it will never draw.
        from .tui.app import run_tui

        anyio.run(
            partial(
                run_tui,
                documents,
                provider=provider,
                model=model,
                session_id=session_id,
                resume=resume,
            )
        )
        return

    if prompt is None:
        console.print(ctx.get_help())
        return

    route = partial(
        _MODES[mode], documents, prompt, provider=provider, model=model, session_id=session_id
    )
    outcome = anyio.run(route)

    if mode == "json":
        # Already written, event by event, as each committed.
        return
    if mode == "transcript":
        console.print(outcome.text)
        return
    console.print(outcome.text)
    err.print(
        f"[dim]session {outcome.session_id} · {outcome.events} events · {outcome.log_path}[/dim]"
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
    # What this install can actually compose — a bundle profile whose
    # distribution is missing is not offered (P3-20).
    console.print(f"[dim]profiles: {', '.join(available_profiles())}[/dim]")


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
