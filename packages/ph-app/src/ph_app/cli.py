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

`ph daemon` runs the supervisor and `ph agents` is the client that talks to it —
the two halves of Phase 5, and the only pair here where the thing you are
addressing is a process rather than this one.

@module ph_app.cli
"""

from __future__ import annotations

import json
import sys
from collections.abc import Awaitable, Callable
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import anyio
import typer
import yaml
from rich.table import Table

from ph.cordis import Loader, import_plugin_modules
from ph.cordis.events import events as event_registry
from ph.paths import RuntimeDirError, resolve_roots

from .agents import agents_app
from .attach import AttachmentUnavailable
from .console import console, emit, err, fail
from .daemon.recovery import PASSIVATE_AFTER
from .modes import run_json, run_print, run_rpc, run_transcript
from .profiles import available_profiles, resolve_profile
from .runtime import mounted

__all__ = ["app", "main"]

app = typer.Typer(
    name="ph",
    help="pH — a plugin-composed agent harness.",
    add_completion=False,
    no_args_is_help=False,
    invoke_without_command=True,
)
# The client half of the daemon, as its own group (P5-10). A sub-app rather
# than seven top-level commands, because every one of them means the same thing
# — "ask the supervisor" — and a person who has not started one should find that
# out from one place.
app.add_typer(agents_app, name="agents")

DEFAULT_PROFILE = "headless"

ProfileOption: TypeAlias = Annotated[
    str, typer.Option("--profile", help="Profile name or path to a .yaml.")
]
"""Declared once, so two commands cannot come to disagree about what `--profile`
means — or, as nearly happened here, about what an unknown one costs."""


def _documents(profile: str) -> list[Path]:
    """The profile's documents, or exit 2 saying which names exist.

    The refusal is the command's, not the resolver's: `resolve_profile` raises a
    `ValueError` that already names the available profiles, and every caller
    wants that sentence on stderr under the same exit code.
    """
    try:
        return resolve_profile(profile)
    except ValueError as error:
        fail(f"[red]{error}[/red]", code=2, cause=error)


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
    profile: ProfileOption = DEFAULT_PROFILE,
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
    attach: Annotated[
        list[Path] | None,
        typer.Option(
            "-a",
            "--attach",
            help="A file to attach to the prompt. Repeatable.",
        ),
    ] = None,
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
    documents = _documents(profile)

    if dump_config:
        loader = Loader.from_paths(documents)
        emit(yaml.safe_dump(loader.dump(), sort_keys=False, default_flow_style=False).rstrip())
        return

    if mode == "trajectory":
        # The auditor's view. Deliberately *before* any profile work: it mounts
        # nothing — no agent, no provider, no answerers — because the logs worth
        # auditing are the ones nobody can reopen (P3-25).
        if session_id is None:
            fail("[red]--mode trajectory needs --session <id|path>[/red]", code=2)
        from .tui.trajectory_app import run_trajectory

        try:
            anyio.run(partial(run_trajectory, session_id))
        except (OSError, ValueError) as error:
            fail(f"[red]{error}[/red]", code=2, cause=error)
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
        _MODES[mode],
        documents,
        prompt,
        provider=provider,
        model=model,
        session_id=session_id,
        attachments=attach or [],
    )
    try:
        outcome = anyio.run(route)
    except (AttachmentUnavailable, OSError) as error:
        # A file that cannot be read fails the *command*: `prompted` ingests
        # before the agent exists, so nothing was logged and there is no partial
        # turn to explain.
        fail(f"[red]{error}[/red]", code=2, cause=error)

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


async def _report(documents: list[Path]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Compose the profile and ask every row what it has to say (P4-12).

    Mounting is the point. Doctor answered from `resolve_roots()` alone until
    now, which meant it could report where the log *would* go and nothing about
    what the process would actually be — and every question this row was written
    for (which rung is in force, what the file rules reach, what runs model code)
    is answered by a row, not by a path. Nothing is created here beyond the
    mount: no session, no agent, no provider call.
    """
    async with mounted(documents) as run:
        registry = run.ctx.get("diagnostics")
        return [] if registry is None else registry.report()


@app.command()
def doctor(profile: ProfileOption = DEFAULT_PROFILE) -> None:
    """Report the path roots, then mount a profile and report what it composed."""
    try:
        roots = resolve_roots()
    except RuntimeDirError as error:
        fail(f"[red]$PH_RUNTIME check failed:[/red] {error}", cause=error)
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

    # Resolved *outside* the catch below, and it matters: `typer.Exit` subclasses
    # `RuntimeError`, so an unknown profile raised inside it would be caught,
    # reported as "does not mount", and re-raised with the wrong exit code.
    documents = _documents(profile)
    try:
        sections = anyio.run(partial(_report, documents))
    except typer.Exit:
        raise
    except Exception as error:
        # Broad on purpose, and only here. A profile that refuses to start is
        # the most important thing doctor can report — `containment.strict` on a
        # host with no sandbox backend is exactly that (E8) — and a person who
        # ran the command *because* the process will not start is owed the
        # sentence rather than a traceback. The exit code says it failed.
        fail(f"[red]profile {profile!r} does not mount:[/red] {error}", cause=error)

    console.print(f"\n[bold]profile:[/bold] {profile}")
    for title, rows in sections:
        section = Table(title=title, show_header=False, title_justify="left")
        section.add_column(style="bold")
        section.add_column()
        for label, value in rows:
            section.add_row(label, value)
        console.print(section)


_DEFAULT_PASSIVATION = f"{PASSIVATE_AFTER / 60:g}"
"""The flag's default, derived from the constant that justifies the number.

It was spelled `"90"` here while `PASSIVATE_AFTER` said ninety minutes in
seconds a module away — two literals in two units for one policy, where editing
the documented one changed nothing for anyone running `ph daemon`, since the CLI
always passes its own value.
"""


def _passivation(value: str) -> float | None:
    """`"off"` or a number of minutes, as seconds (P5-05).

    Refused rather than defaulted when it is neither: a typo in a duration is a
    deployment that silently keeps every root it ever started, and the daemon is
    the one process where that goes unnoticed for a week.
    """
    if value.strip().lower() == "off":
        return None
    try:
        minutes = float(value)
    except ValueError:
        minutes = 0.0
    if minutes <= 0:
        # One message for one mistake: "not a number" and "not a positive
        # number" are the same correction to the same flag.
        raise typer.BadParameter(f'wants positive minutes or "off", not "{value}"')
    return minutes * 60.0


@app.command()
def daemon(
    profile: ProfileOption = DEFAULT_PROFILE,
    provider: Annotated[str, typer.Option("--provider")] = "fake",
    model: Annotated[str, typer.Option("--model")] = "fake-1",
    passivate_after: Annotated[
        str,
        typer.Option(
            "--passivate-after", help='Minutes of quiet before a root is released, or "off".'
        ),
    ] = _DEFAULT_PASSIVATION,
) -> None:
    """Run the supervisor: roots that outlive the clients watching them (P5-01).

    Blocks until a client sends `shutdown`. The socket is
    `$PH_RUNTIME/daemon.sock` — per boot and per user, which is the tier chosen
    for exactly this — and a stale one from a crashed daemon is cleared, while a
    live one is refused rather than stolen.
    """
    from .daemon import serve
    from .daemon.server import DaemonUnavailable

    documents = _documents(profile)
    try:
        socket_path = resolve_roots().ensure().daemon_socket()
    except RuntimeDirError as error:
        fail(f"[red]$PH_RUNTIME check failed:[/red] {error}", cause=error)
    err.print(f"[dim]listening on {socket_path}[/dim]")
    try:
        # The path that was printed, not a second resolution of it: a message
        # naming one socket while the bind takes another is the kind of thing
        # someone debugs for an hour.
        anyio.run(
            partial(
                serve,
                documents,
                provider=provider,
                model=model,
                passivate_after=_passivation(passivate_after),
                path=socket_path,
            )
        )
    except DaemonUnavailable as error:
        # One named type rather than `(RuntimeError, OSError)`, which is two
        # builtins wide enough to swallow a `typer.Exit` — it subclasses
        # `RuntimeError`, and the comment in `doctor` above records this file
        # having been bitten by that already.
        fail(f"[red]{error}[/red]", cause=error)


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
        emit(json.dumps(matrix, indent=2))
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
