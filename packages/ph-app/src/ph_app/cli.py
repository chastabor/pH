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
import shlex
import sys
from collections.abc import Awaitable, Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import anyio
import typer
import yaml
from rich.table import Table

from ph.cordis import LoaderError, Profile, import_plugin_modules
from ph.cordis.catalog import config_catalog
from ph.cordis.events import events as event_registry
from ph.lingering import lifetime
from ph.paths import RuntimeDirError, resolve_roots
from ph.seams.diagnostics import DiagnosticsRegistry
from ph.selectors import matches_any, unknown_namespaces

from .agents import agents_app
from .attach import AttachmentUnavailable
from .console import TypeOption, console, emit, err, fail, section, selectors_or_exit
from .daemon.recovery import PASSIVATE_AFTER
from .modes import run_json, run_print, run_rpc, run_transcript
from .profiles import (
    DEFAULT_PROFILE,
    PatchOption,
    ProfileOption,
    available_profiles,
    profile_or_exit,
)
from .runtime import mounted
from .workspaces import workspaces_app

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
# The cross-session half of `/workspaces` (P6-28). A group of its own rather than
# a verb on the slash command, because the question it answers spans every
# session on disk while `/workspaces` answers for the one a person is in — and
# because collecting checkouts is not something a model should be able to ask
# for by emitting text.
app.add_typer(workspaces_app, name="workspaces")

OutputMode = Literal["text", "json", "transcript", "rpc", "tui", "web", "trajectory"]

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
        typer.Option(
            "--mode", help="text (default), json, transcript, rpc, tui, web, or trajectory."
        ),
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
    no_spawn: Annotated[
        bool,
        typer.Option(
            "--no-spawn",
            help="Refuse rather than start a daemon when none is listening (tui only).",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option("--host", help="Interface to serve the browser UI on (web only).")
    ] = "127.0.0.1",
    port: Annotated[int, typer.Option("--port", help="Port for the browser UI (web only).")] = 8000,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the browser at the token URL (web only).")
    ] = False,
    dump_config: Annotated[
        bool,
        typer.Option(
            "--dump-config",
            help="Print the composed rows and exit — the mount as written; "
            "`ph doctor` shows what activated.",
        ),
    ] = False,
    patch: PatchOption = [],  # noqa: B006 - typer builds the list per invocation
) -> None:
    """Run a prompt, or dump the composed configuration."""
    if ctx.invoked_subcommand is not None:
        return

    if mode == "trajectory":
        # The auditor's view. Deliberately *before* any profile work — it does
        # not read one: it mounts nothing — no agent, no provider, no answerers —
        # because the logs worth auditing are the ones nobody can reopen (P3-25).
        if session_id is None:
            fail("[red]--mode trajectory needs --session <id|path>[/red]", code=2)
        from .tui.trajectory_app import run_trajectory

        try:
            anyio.run(partial(run_trajectory, session_id))
        except (OSError, ValueError) as error:
            fail(f"[red]{error}[/red]", code=2, cause=error)
        return

    if mode == "tui":
        # **Before any profile work**, like the trajectory branch above and for
        # the same reason: the terminal does not mount one. The daemon composes
        # the profile, and `spawn_command` is the command line that says which —
        # so composing one here was a full plugin import per TUI start, thrown
        # away.
        #
        # Imported here because the TUI pulls in Textual, and `ph -p` in a
        # script should not pay for a terminal UI it will never draw.
        from .tui.app import run_tui

        anyio.run(
            partial(
                run_tui,
                # The *name*, not the composed profile: the daemon composes it,
                # and this is the command line that tells it which.
                daemon_argv=spawn_command(
                    profile=profile, provider=provider, model=model, patch=patch
                ),
                # `--resume <id>` is `--session <id>`: the daemon resumes an
                # existing id through the same call that creates a new one.
                session_id=session_id or resume,
                spawn=not no_spawn,
            )
        )
        return

    if mode == "web":
        # Before the profile work, like `tui` above and for the same reason: the
        # terminal each tab runs is the thing that talks to a daemon, and this
        # process only serves them.
        #
        # Imported here because `textual-serve` is an extra: `ph -p` in a script
        # must not require aiohttp, and a person who asked for `--mode web`
        # without it gets the install line rather than a traceback.
        try:
            from .web.serve import WebServer, run_web
        except ImportError as error:
            fail(
                "[red]--mode web needs the web extra:[/red] pip install 'ph-app[web]'",
                cause=error,
            )
        # Deliberately **not** `--session`: `textual-serve` fixes the command at
        # construction and varies nothing per request, so a session id here would
        # be *every* tab's. Without one each tab opens its own new session, and
        # the picker is how a person joins somebody else's.
        tab = reinvoke(
            "--mode",
            "tui",
            *(("--no-spawn",) if no_spawn else ()),
            profile=profile,
            provider=provider,
            model=model,
            patch=patch,
        )
        server = WebServer(command=shlex.join(tab), host=host, port=port)
        # Printed here, before the bind, the way `ph daemon` prints its socket
        # and its linger warning: the sentences are the server's, and saying them
        # is the command's.
        for notice in server.notices():
            err.print(notice)
        anyio.run(partial(run_web, server, open_browser=open_browser))
        return

    composed = profile_or_exit(profile, patch)

    if dump_config:
        emit(yaml.safe_dump(composed.dump(), sort_keys=False, default_flow_style=False).rstrip())
        return

    if mode == "rpc":
        # No prompt: the peer drives the session over stdio.
        anyio.run(partial(run_rpc, composed, provider=provider, model=model))
        return

    if prompt is None:
        console.print(ctx.get_help())
        return

    route = partial(
        _MODES[mode],
        composed,
        prompt,
        provider=provider,
        model=model,
        session_id=session_id,
        attachments=attach or [],
    )
    try:
        outcome = anyio.run(route)
    except (AttachmentUnavailable, LoaderError, OSError) as error:
        # A file that cannot be read fails the *command*: `prompted` ingests
        # before the agent exists, so nothing was logged and there is no partial
        # turn to explain. A row whose plugin will not import is the same kind of
        # failure one step earlier — the loader's one refusal left at mount time,
        # now that `profile_or_exit` composes.
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


NO_DIAGNOSTICS_ROW = "none — this profile mounts no `diagnostics` row, so no row can report"


async def _note_consumers(profile: Profile) -> None:
    """Mount, and let every row's `ctx.on` register itself into the registry.

    Nothing is read back here: `note_consumer` writes into the process-wide
    `EventRegistry` as a side effect of listening, so the mount *is* the query
    and the registry outlives the scope that filled it. Nothing is created
    beyond the mount — no session, no agent, no provider call — for `_report`'s
    reason.
    """
    async with mounted(profile):
        return


async def _report(profile: Profile) -> list[tuple[str, list[tuple[str, str]]]]:
    """Compose the profile and ask every row what it has to say (P4-12).

    Mounting is the point. Doctor answered from `resolve_roots()` alone until
    now, which meant it could report where the log *would* go and nothing about
    what the process would actually be — and every question this row was written
    for (which rung is in force, what the file rules reach, what runs model code)
    is answered by a row, not by a path. Nothing is created here beyond the
    mount: no session, no agent, no provider call. Topology is a row
    (`ph.seams.topology`), so the registry is the only source.
    """
    async with mounted(profile) as ctx:
        registry: DiagnosticsRegistry | None = ctx.get("diagnostics")
        if registry is None:
            # Rule 6, in the seam's place: with no `diagnostics` row nothing can
            # report, and an empty report reads as "nothing wrong".
            return [("Diagnostics", [("sections", NO_DIAGNOSTICS_ROW)])]
        return registry.report()


@app.command()
def doctor(
    profile: ProfileOption = DEFAULT_PROFILE,
    patch: PatchOption = [],  # noqa: B006 - typer builds the list per invocation
) -> None:
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
    # The same subject as the roots table, one question further on: that table
    # says where `$PH_RUNTIME` resolved, and this says whether what a daemon puts
    # there is still there tomorrow (I-6, P5-11).
    #
    # **Not a `ctx.diagnostics` section**, which makes this command's one place
    # with two mechanisms — deliberately. A contributed section is lost exactly
    # when `doctor` bails with "profile does not mount", the case a person most
    # wants it, and this probe needs no profile to answer. What the seam is still
    # owed is the *shape*: a section is a title and a list of pairs, both halves
    # produce `list[tuple[str, str]]`, and both render through `console.section`
    # — so the next profile-free probe is an entry in this list rather than one
    # more `console.print` in a command body.
    #
    # Listed even when the answer is "yes": rule 6 says to state what is not
    # enforced next to where it would be assumed, and the assumption — that
    # `ph daemon` means "until I stop it" — is made by every reader who never
    # sees a warning.
    #
    # `roots=` so this and the table above describe one resolution rather than
    # two independent ones.
    # Imported here rather than at module scope: `ph_app.daemon.supervisor`
    # pulls in the agent, the persistence layer and `filelock`, and `ph --print`
    # has no daemon in it. `daemon()` below reaches for `serve` the same way.
    from .daemon.supervisor import NON_GUARANTEES

    before_mount = [
        ("daemon socket lifetime", lifetime(roots=roots).describe()),
        # N5, and the gate's "doctor prints the worker model". Here and not only
        # in `ph agents doctor` because the two answer different questions: that
        # one describes a daemon somebody already started, and this one is read
        # by the person deciding whether to run six agents under one (I-2).
        ("daemon isolation", list(NON_GUARANTEES)),
    ]
    # What this install can actually compose — a bundle profile whose
    # distribution is missing is not offered (P3-20).
    console.print(f"[dim]profiles: {', '.join(available_profiles())}[/dim]")

    # Resolved *outside* the catch below, and it matters: `typer.Exit` subclasses
    # `RuntimeError`, so an unknown profile raised inside it would be caught,
    # reported as "does not mount", and re-raised with the wrong exit code.
    composed = profile_or_exit(profile, patch)
    try:
        sections = anyio.run(partial(_report, composed))
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
    for title, rows in before_mount + sections:
        console.print(section(title, rows))


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
    ephemeral: Annotated[
        bool,
        typer.Option("--ephemeral", help="Exit once no client, root or appointment needs this."),
    ] = False,
) -> None:
    """Run the supervisor: roots that outlive the clients watching them (P5-01).

    Blocks until a client sends `shutdown`. The socket is
    `$PH_RUNTIME/daemon.sock` — per boot and per user, which is the tier chosen
    for exactly this — and a stale one from a crashed daemon is cleared, while a
    live one is refused rather than stolen.
    """
    from .daemon import serve
    from .daemon.server import DaemonUnavailable

    composed = profile_or_exit(profile)
    try:
        roots = resolve_roots(create=True)
    except RuntimeDirError as error:
        fail(f"[red]$PH_RUNTIME check failed:[/red] {error}", cause=error)
    socket_path = roots.daemon_socket()
    err.print(f"[dim]listening on {socket_path}[/dim]")
    # "names `enable-linger` when a daemon is configured without it" — the row's
    # own wording, and this is the moment it is being configured. Said here as
    # well as in `ph doctor` because the two have different readers: doctor is
    # run by somebody already debugging, and this line is read by somebody who
    # is not, ten seconds before closing the terminal it is printed in.
    life = lifetime(socket_path, roots=roots)
    if life.survives_logout is not True:
        err.print(f"[yellow]this socket does not survive logout:[/yellow] {life.verdict()}")
        err.print(f"[yellow]  {life.advice}[/yellow]")
    try:
        # The path that was printed, not a second resolution of it: a message
        # naming one socket while the bind takes another is the kind of thing
        # someone debugs for an hour.
        anyio.run(
            partial(
                serve,
                composed,
                provider=provider,
                model=model,
                passivate_after=_passivation(passivate_after),
                # Off here, on in `spawn_command`: `DaemonServer.ephemeral` says
                # why the lifetime is decided by who started it (P7-08).
                ephemeral=ephemeral,
                path=socket_path,
            )
        )
    except DaemonUnavailable as error:
        # One named type rather than `(RuntimeError, OSError)`, which is two
        # builtins wide enough to swallow a `typer.Exit` — it subclasses
        # `RuntimeError`, and the comment in `doctor` above records this file
        # having been bitten by that already.
        fail(f"[red]{error}[/red]", cause=error)


def reinvoke(
    *args: str, profile: str, provider: str, model: str, patch: Sequence[str] = ()
) -> list[str]:
    """How pH starts pH: the argv for another process of this one.

    Two callers ask it — the daemon a UI spawns when no socket answers, and the
    terminal a browser tab runs — and they differ only in their leading verb. The
    tail is the same composition every time, so it is written once: an option
    added here reaches both, where two spellings would silently reach one.

    `sys.executable -m ph_app` rather than a bare `ph`: the caller may be running
    from a virtualenv that is not on `PATH`, or from a checkout with no console
    script installed at all, and a pH started from a *different* pH is one whose
    profile, event vocabulary and wire version nobody chose.

    `--patch` travels because the *other* process is the one that composes: a
    patch accepted here and dropped would silently ignore `ph --mode tui --patch
    '{id: tool-ask-user, disabled: false}'`, which is the documented way to arm a
    row anywhere.
    """
    argv = [
        sys.executable,
        "-m",
        "ph_app",
        *args,
        "--profile",
        profile,
        "--provider",
        provider,
        "--model",
        model,
    ]
    for one in patch:
        argv += ["--patch", one]
    return argv


def spawn_command(
    *, profile: str, provider: str, model: str, patch: Sequence[str] = ()
) -> list[str]:
    """The argv for a daemon a UI starts on its own behalf (P7-08).

    Beside the `daemon` command whose options it spells, so renaming one is one
    edit. `--ephemeral` because this daemon was nobody's decision, so it leaves
    when nobody needs it.
    """
    return reinvoke(
        "daemon", "--ephemeral", profile=profile, provider=provider, model=model, patch=patch
    )


@app.command()
def events(
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    type_: TypeOption = [],  # noqa: B006 - typer builds the list per invocation
    profile: ProfileOption = DEFAULT_PROFILE,
    patch: PatchOption = [],  # noqa: B006 - typer builds the list per invocation
) -> None:
    """Print the event producer/consumer matrix.

    Generated from the declaration registry, so it cannot drift from the code
    the way a hand-written table does. Declarations live in the plugin modules
    that own them, so every registered plugin is imported first — third-party
    wheels included.

    **Then a profile is mounted, and that is what makes the consumer half real.**
    Importing a module runs its `declare` calls, so producers are knowable from
    an import alone — but a *consumer* is recorded by `ctx.on`, which runs when a
    row activates. Without a mount this printed a producer matrix under a
    producer/consumer heading, with every consumer list empty and nothing saying
    why. Which rows listen is a property of the profile, so the answer is
    per-profile and the flag is the same one `doctor` takes.

    A profile that will not mount is reported rather than fatal: the declarations
    are still worth printing, and a person debugging a broken profile is exactly
    who is running this.
    """
    import_plugin_modules()
    # Resolved *outside* the guard below: `profile_or_exit` reports an unknown
    # profile by raising `typer.Exit`, which is an `Exception`, so catching
    # broadly around it turned "no such profile" into a full matrix and exit 0 —
    # the answer that looks most like success for the input most likely to be a
    # typo.
    composed = profile_or_exit(profile, patch)
    try:
        anyio.run(partial(_note_consumers, composed))
    except Exception as error:
        err.print(
            f"[yellow]profile {profile!r} does not mount, so no consumers are listed:[/yellow] "
            f"{error}"
        )
    # The bus vocabulary, so a bare `tools` needs no prefix — and `log:workspace`
    # is refused rather than answered emptily, because this registry holds no
    # session-log types and the two share six roots (P6-33).
    selectors = selectors_or_exit(type_, vocabulary="bus")
    matrix = [row for row in event_registry.matrix() if matches_any(row["name"], selectors)]
    if selectors and not matrix:
        # A namespace nothing occupies is a typo far more often than it is an
        # empty one, and this registry knows every name it holds — so it can say
        # which it was rather than printing an empty table.
        unknown = unknown_namespaces(selectors, event_registry.names())
        detail = f": {', '.join(unknown)}" if unknown else ""
        fail(f"[red]no declared event matches{detail}[/red]", code=2)
    if as_json:
        emit(json.dumps(matrix, indent=2))
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("event")
    table.add_column("mode")
    table.add_column("payload")
    table.add_column("producer")
    # The consumers are half of what a producer/consumer matrix is *for*, and
    # the rendered table shipped without them for a round — `matrix()` had
    # carried them all along, so the JSON was complete and only the half a
    # person reads was missing. An event with no listener is a real finding
    # (a declared extension point nobody uses), so an empty cell says so.
    table.add_column("consumers")
    table.add_column("what it is")
    for row in matrix:
        table.add_row(
            row["name"],
            row["mode"],
            row["payload"] or "",
            row["producer"],
            "\n".join(row["consumers"]),
            row["doc"],
        )
    console.print(table)


@app.command()
def config(
    as_json: Annotated[bool, typer.Option("--json", help="Emit JSON.")] = False,
    row: Annotated[list[str], typer.Option("--row", help="Only these rows. Repeatable.")] = [],  # noqa: B006 - typer builds the list per invocation
    all_: Annotated[
        bool, typer.Option("--all", help="Include rows that take no configuration.")
    ] = False,
) -> None:
    """Print what every row accepts as configuration.

    Generated from each plugin's own `config=` model, so it cannot drift from
    the code the way a hand-written options table does — the same argument
    `ph events` makes about the event registry, applied to the other half of
    what a profile is.

    Rows with no options are omitted unless `--all` asks for them: "this row has
    no configuration" is worth being able to look up, but it is not what
    somebody scanning for a knob is reading past sixty of.
    """
    catalog = config_catalog()
    wanted = {name.strip() for name in row if name.strip()}
    if wanted:
        catalog = [entry for entry in catalog if entry["name"] in wanted]
        missing = sorted(wanted - {entry["name"] for entry in catalog})
        if missing:
            # A name that resolves to nothing is a typo far more often than it
            # is an unregistered row, and this catalog knows every name it
            # holds — so it says which, rather than printing an empty table.
            fail(f"[red]no registered row named: {', '.join(missing)}[/red]", code=2)
    if as_json:
        emit(json.dumps(catalog, indent=2))
        return
    shown = [entry for entry in catalog if all_ or entry["config"] or entry.get("error")]
    if not shown:
        emit("no row matched")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("row")
    table.add_column("option")
    table.add_column("type")
    table.add_column("default")
    table.add_column("what it does")
    for entry in shown:
        if error := entry.get("error"):
            table.add_row(entry["name"], "[red]unavailable[/red]", "", "", error)
            continue
        if not entry["config"]:
            table.add_row(entry["name"], "[dim]none[/dim]", "", "", "")
            continue
        for index, field in enumerate(entry["config"]):
            table.add_row(
                entry["name"] if index == 0 else "",
                field["name"],
                field["type"],
                # Required has no default, and printing one would invent a
                # value a profile must actually supply.
                "[bold]required[/bold]" if field["required"] else (field["default"] or ""),
                field["doc"],
            )
    console.print(table)


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
