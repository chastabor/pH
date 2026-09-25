"""`phern` — the command line.

Four output modes and three diagnostics.

`--mode` decides what reaches stdout, and the choice matters more than a flag
usually does: `json` and `rpc` emit the session log's **own** envelopes rather
than a per-mode rendering (I-7), so a wrapper streaming from a pipe and a tool
reading the stored JSONL parse one format — and dsh's tooling reads both (Q2).
`text` and `transcript` are for people, and `tui` is the interactive one — the
only mode that takes no `--print`, because the prompt is the interface.

`--dump-config` prints the composed rows before anything runs, `phern doctor` the
three resolved path roots, and `phern events` the producer/consumer matrix
generated from the declaration registry rather than hand-maintained.

`phern daemon` runs the supervisor and `phern agents` is the client that talks to it —
the two halves of Phase 5, and the only pair here where the thing you are
addressing is a process rather than this one.

@module ph_app.cli
"""

from __future__ import annotations

import json
import shlex
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import anyio
import typer
import yaml
from rich.table import Table

from ph.cordis import LoaderError, MountRefusal, Profile, import_plugin_modules
from ph.cordis.catalog import config_catalog
from ph.cordis.events import events as event_registry
from ph.json import JsonObject, as_obj
from ph.keys import DIAGNOSTICS
from ph.lingering import lifetime
from ph.paths import RuntimeDirError, resolve_roots
from ph.seams.diagnostics import DiagnosticsRegistry
from ph.selectors import matches_any, unknown_namespaces
from ph.session import new_session_id

from .agents import agents_app
from .attach import AttachmentUnavailable
from .attachments import attachments_app
from .console import (
    TypeOption,
    console,
    detail,
    emit,
    err,
    fail,
    fail_unmounted,
    section,
    selectors_or_exit,
)
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
from .web import DEFAULT_HOST, DEFAULT_PORT
from .workspaces import workspaces_app

__all__ = ["app", "main"]

app = typer.Typer(
    name="phern",
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
# The other cross-session accounting command (P7-01). Its own group beside
# `workspaces` rather than a verb there, because the two sweep different things
# under different rules — a retained tree is evidence somebody may want, an
# attachment is content a session cannot open without — and a person reading
# `--help` should not have to infer which rule applies from a shared verb.
app.add_typer(attachments_app, name="attachments")

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
        typer.Option(
            "--session", help="Session id to create or resume, or to read under --mode trajectory."
        ),
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
        str | None, typer.Option("--resume", help="Session id to reopen (tui and web).")
    ] = None,
    keep_alive: Annotated[
        str | None,
        typer.Option(
            "--keep-alive",
            help='Keep a daemon this UI starts up for this long after it detaches ("5m").',
        ),
    ] = None,
    new: Annotated[
        bool,
        typer.Option(
            "--new",
            help="Start a fresh session without offering this directory's previous ones (tui).",
        ),
    ] = False,
    no_spawn: Annotated[
        bool,
        typer.Option(
            "--no-spawn",
            help="Refuse rather than start a daemon when none is listening (tui and web).",
        ),
    ] = False,
    keep_daemon: Annotated[
        bool,
        typer.Option(
            "--keep-daemon",
            help="A daemon this starts is a service, not an ephemeral one (tui and web).",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option("--host", help="Interface to serve the browser UI on (web only).")
    ] = DEFAULT_HOST,
    port: Annotated[
        int, typer.Option("--port", help="Port for the browser UI (web only).")
    ] = DEFAULT_PORT,
    open_browser: Annotated[
        bool, typer.Option("--open", help="Open the browser at the token URL (web only).")
    ] = False,
    dump_config: Annotated[
        bool,
        typer.Option(
            "--dump-config",
            help="Print the composed rows and exit — the mount as written; "
            "`phern doctor` shows what activated.",
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
        # A late load: a command pays only for what it runs (`test_app_layering.py`).
        from .tui.trajectory_app import run_trajectory  # noqa: PLC0415

        try:
            code = anyio.run(partial(run_trajectory, session_id))
        except (OSError, ValueError) as error:
            fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
        if code:
            # The viewer's own code, silent for `--mode tui`'s reason above.
            raise typer.Exit(code)
        return

    # `--resume <id>` is `--session <id>`: the daemon resumes an existing id
    # through the same call that creates a new one, so the two spellings are one
    # answer and both interactive modes read it here.
    wanted = session_id or resume

    if mode == "tui":
        # **Before any profile work**, like the trajectory branch above and for
        # the same reason: the terminal does not mount one. The daemon composes
        # the profile, and `spawn_command` is the command line that says which —
        # so composing one here was a full plugin import per TUI start, thrown
        # away.
        #
        # Imported here because the TUI pulls in Textual, and `phern -p` in a
        # script should not pay for a terminal UI it will never draw.
        from .tui.app import run_tui  # noqa: PLC0415

        code = anyio.run(
            partial(
                run_tui,
                # The *name*, not the composed profile: the daemon composes it,
                # and this is the command line that tells it which.
                daemon_argv=spawn_command(
                    profile=profile,
                    provider=provider,
                    model=model,
                    patch=patch,
                    keep=keep_daemon,
                    keep_alive=_spawned_keep_alive(keep_alive, keep=keep_daemon),
                ),
                session_id=wanted,
                spawn=not no_spawn,
                # `--session`/`--resume` already name one, so they skip the offer
                # by having an answer; `--new` is for the person who never wants
                # to be asked.
                offer_sessions=not new,
            )
        )
        if code:
            # **Silent, and not `console.fail`** (M4). Every other refusal here
            # goes through `fail`, which prints a sentence — but the terminal
            # has already rendered its own crash, and a second line under it
            # would be pH explaining an error the person just watched happen.
            # What `run_tui` returning the code buys is that the exiting lives
            # here rather than inside a library function.
            raise typer.Exit(code)
        return

    if mode == "web":
        # Before the profile work, like `tui` above and for the same reason: the
        # terminal each tab runs is the thing that talks to a daemon, and this
        # process only serves them.
        #
        # Imported here because `textual-serve` is an extra: `phern -p` in a script
        # must not require aiohttp, and a person who asked for `--mode web`
        # without it gets the install line rather than a traceback.
        try:
            from .web.serve import WebServer, run_web  # noqa: PLC0415
        except ImportError as error:
            fail(
                "[red]--mode web needs the web extra:[/red] pip install 'phern\\[web]'",
                cause=error,
            )
        # **One session for every tab of this launch**, minted here when the
        # person did not name one — `serve.py`'s module docstring says why a tab
        # cannot have one of its own.
        shared = wanted or new_session_id()
        tab = tab_command(
            session=shared,
            profile=profile,
            provider=provider,
            model=model,
            patch=patch,
            spawn=not no_spawn,
            keep=keep_daemon,
        )
        server = WebServer(command=shlex.join(tab), session=shared, host=host, port=port)
        # Before the bind, and printed here because `notices()` owns the words.
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
        try:
            anyio.run(partial(run_rpc, composed, provider=provider, model=model))
        except MountRefusal as error:
            fail_unmounted(profile, error)
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
    # Here rather than at the top: `phern --help` must not pay for the persistence
    # layer, and `route` is about to import it anyway.
    from ph.persistence import SessionBusy  # noqa: PLC0415

    try:
        outcome = anyio.run(route)
    except MountRefusal as error:
        # A row that *declined* — `containment.strict` with no backend (E8) — is
        # the sentence doctor prints, with doctor's exit code, and not the
        # traceback a bug in an `apply` still gets. Before this the run path had
        # no name for the difference and printed 191 lines for the one case a
        # person most needs to read (P4-12).
        fail_unmounted(profile, error)
    except (AttachmentUnavailable, LoaderError, OSError, SessionBusy) as error:
        # A file that cannot be read fails the *command*: `prompted` ingests
        # before the agent exists, so nothing was logged and there is no partial
        # turn to explain. A row whose plugin will not import is the same kind of
        # failure one step earlier — the loader's one refusal left at mount time,
        # now that `profile_or_exit` composes. A session another process holds is
        # the same shape again (I-5): refused before a byte is written, as the
        # one sentence the daemon would have sent, not a traceback.
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)

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
    if outcome.ended == "error":
        # Whatever arrived before the failure is still printed above — it is in
        # the log either way, and a truncated answer is worth seeing. What must
        # not happen is exiting 0: a `-p` run is something scripts call, and a
        # provider outage that reports success is indistinguishable from an
        # answer. Only `error`: `blocked` and `max-tokens` are turns that ended
        # the way they were asked to.
        fail(f"[red]the turn failed:[/red] {detail(outcome.failure)}", code=1)


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
        registry: DiagnosticsRegistry | None = ctx.get(DIAGNOSTICS)
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
        fail(f"[red]$PH_RUNTIME check failed:[/red] {detail(error)}", cause=error)
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
    # `phern daemon` means "until I stop it" — is made by every reader who never
    # sees a warning.
    #
    # `roots=` so this and the table above describe one resolution rather than
    # two independent ones.
    # Imported here rather than at module scope: `ph_app.daemon.supervisor`
    # pulls in the agent, the persistence layer and `filelock`, and `phern --print`
    # has no daemon in it. `daemon()` below reaches for `serve` the same way.
    from .daemon.supervisor import NON_GUARANTEES  # noqa: PLC0415

    before_mount = [
        ("daemon socket lifetime", lifetime(roots=roots).describe()),
        # N5, and the gate's "doctor prints the worker model". Here and not only
        # in `phern agents doctor` because the two answer different questions: that
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
        fail_unmounted(profile, error)

    console.print(f"\n[bold]profile:[/bold] {profile}")
    for title, rows in before_mount + sections:
        console.print(section(title, rows))


_DEFAULT_PASSIVATION = f"{PASSIVATE_AFTER / 60:g}"
"""The flag's default, derived from the constant that justifies the number.

It was spelled `"90"` here while `PASSIVATE_AFTER` said ninety minutes in
seconds a module away — two literals in two units for one policy, where editing
the documented one changed nothing for anyone running `phern daemon`, since the CLI
always passes its own value.
"""


def _spawned_keep_alive(value: str | None, *, keep: bool) -> str:
    """The `--keep-alive` a spawned daemon is given, in seconds, refused here on a typo.

    **Parsed at this end, not forwarded as typed.** `phern --keep-alive` does not run
    a daemon, it composes the argv for one — so a string handed straight through
    would be refused by a process whose output goes to the null device, and the
    person would see a UI that could not reach a daemon and no reason why. Same
    parser, same message, whichever end they typed it at; what crosses is a bare
    number of seconds, which `keep_alive_seconds` reads back unambiguously.

    CLI first, then `tui.json`, then zero: a person who sets it once should not
    have to type it, and a person who typed it means this run. The file is read
    only when they did not, which is also why it is not hoisted — settings are
    deliberately re-read for each `PHTuiApp`, so that a `/theme` written during
    a session is picked up when the picker reopens one.

    **`--keep-daemon` and a typed `--keep-alive` are refused together**, because
    they ask for opposite things: one says the daemon stays, the other says how
    long it waits before going. A warning would have been the wrong answer to a
    person who has said two things and meant one of them. A *configured*
    keep-alive is not a contradiction — nobody typed it for this run — so
    `--keep-daemon` simply overrides it, and a service daemon has no window.

    `ph_app.tui.config` is imported inside the function for `run_tui`'s reason:
    it is a front-end module and `phern -p` should not load one to print an
    answer.
    """
    if keep:
        if value is not None and keep_alive_seconds(value) > 0:
            raise typer.BadParameter(
                "--keep-alive and --keep-daemon ask for opposite things: one says when "
                "the daemon leaves, the other says it stays"
            )
        return "0"
    if value is None:
        # A late load: a command pays only for what it runs (`test_app_layering.py`).
        from .tui.config import load_tui_settings  # noqa: PLC0415

        value = load_tui_settings(resolve_roots().home).daemon_keep_alive
    return f"{keep_alive_seconds(value):g}"


UNITS: Mapping[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0}
"""The suffixes a duration option accepts, as seconds each."""


def _duration_seconds(value: str, *, unit: float) -> float:
    """A duration a person typed, as seconds. Negative for "not one".

    One parser for `--keep-alive` and `--passivate-after`, which are the same
    question asked twice and were two parsers with two refusal shapes. `unit` is
    what a *bare* number means — seconds for a keep-alive, minutes for a passivation
    window, which is what each flag's own history established — and a suffix
    overrides it, so `--passivate-after 30s` now means what it reads like
    instead of half an hour.

    Negative rather than raising, because the two callers refuse differently:
    one of them has an `"off"` spelling to check first, and a parser that raised
    would have to know about it.
    """
    text = value.strip().lower()
    if not text:
        return -1.0
    scale = UNITS.get(text[-1])
    number = text[:-1] if scale is not None else text
    try:
        seconds = float(number) * (scale if scale is not None else unit)
    except ValueError:
        return -1.0
    return seconds if seconds >= 0 else -1.0


def keep_alive_seconds(value: str) -> float:
    """`"30s"`, `"5m"`, `"1h"` or a bare number of seconds, as seconds.

    Refused rather than defaulted on a typo, for `_passivation`'s reason: a
    mistyped duration here is a daemon that leaves immediately or never, and
    either is a surprise a person attributes to something else a week later.

    **Public, because both ends of the flag have to agree.** `phern daemon
    --keep-alive` parses it here, and `phern --keep-alive` — which does not run the
    daemon, it *spawns* one — parses it here too rather than forwarding the
    string into an argv where a typo would surface as a daemon that refused to
    start and a UI that said nothing. Same refusal, same message, at whichever
    end the person typed it.

    **A duration and nothing else.** Whether the daemon is ephemeral is the
    caller's question, not this one's: `phern daemon` and `phern` express that
    in opposite directions — one opts into leaving, the other opts into staying
    — so a parser that also decided it would have to know which command it was
    serving.
    """
    if not value.strip() or value.strip() == "0":
        return 0.0
    seconds = _duration_seconds(value, unit=1.0)
    if seconds < 0:
        # `typer.BadParameter` like `_passivation` and `_children_cap` beside it,
        # rather than a hand-built refusal: typer already writes the usage line
        # and the exit code, and three value parsers on one command refusing in
        # two different shapes is how one of them comes to say something else.
        raise typer.BadParameter(f'wants a duration like "30s", "5m" or "90", not "{value}"')
    return seconds


def _passivation(value: str) -> float | None:
    """`"off"`, a number of minutes, or a suffixed duration, as seconds (P5-05).

    Refused rather than defaulted when it is none of those: a typo in a duration
    is a deployment that silently keeps every root it ever started, and the
    daemon is the one process where that goes unnoticed for a week.
    """
    if value.strip().lower() == "off":
        return None
    seconds = _duration_seconds(value, unit=60.0)
    if seconds <= 0:
        # One message for one mistake: "not a number" and "not a positive
        # number" are the same correction to the same flag.
        raise typer.BadParameter(f'wants positive minutes or "off", not "{value}"')
    return seconds


def _children_cap(value: int | None) -> list[str]:
    """The `--max-concurrent-children` override, as the patch it really is.

    The cap *is* row config, so overriding it from the command line is a
    `--patch` and not a second channel: `--dump-config` and `phern doctor` then show
    the number actually in force with `cli` as its provenance, which a field
    threaded past the profile could not do.

    **Unset patches nothing**, which is what lets the flag carry no default of
    its own: whatever the composed profile says stands, so the number lives once,
    in the bundle that ships a subagent provider at all. It also keeps a plain
    `phern daemon` working on a profile with no `jobs` row — a patch names a row by
    id and the loader refuses one that is not there, so a default that always
    patched would have made every such run exit 2 over a cap nobody asked for.
    Name a number *there* and the refusal is earned, and says which row is
    missing.

    The kind comes from `ph.seams.jobs`, imported where the daemon's other
    imports are: this module is read by every `phern` invocation, and the seam is
    not otherwise on that path.
    """
    if value is None:
        return []
    if value < 1:
        raise typer.BadParameter(f"wants a positive number of children, not {value}")
    # A late load: a command pays only for what it runs (`test_app_layering.py`).
    from ph.seams.jobs import CHILDREN_KIND  # noqa: PLC0415

    return [f"{{id: jobs, config: {{concurrency: {{{CHILDREN_KIND}: {value}}}}}}}"]


@app.command()
def daemon(
    profile: ProfileOption = DEFAULT_PROFILE,
    provider: Annotated[str, typer.Option("--provider")] = "fake",
    model: Annotated[str, typer.Option("--model")] = "fake-1",
    max_concurrent_children: Annotated[
        int | None,
        typer.Option(
            "--max-concurrent-children",
            help="Children this deployment runs at once, across every root; the rest queue.",
        ),
    ] = None,
    passivate_after: Annotated[
        str,
        typer.Option(
            "--passivate-after", help='Minutes of quiet before a root is released, or "off".'
        ),
    ] = _DEFAULT_PASSIVATION,
    keep_alive: Annotated[
        str,
        typer.Option(
            "--keep-alive",
            help='Stay up this long after the last client leaves ("5m"); implies --ephemeral.',
        ),
    ] = "0",
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

    `--max-concurrent-children` is the **deployment's** bound and is the one a
    host operator wants: a child is an agent with a model, a workspace and often
    an interpreter of its own, and fan-out across many roots multiplies without
    it. A parent's own fair share is `rlm-subagent-provider`'s `maxConcurrent`,
    and both apply. Neither refuses a delegation — the overflow queues.
    """
    # A late load: a command pays only for what it runs (`test_app_layering.py`).
    from .daemon.server import DaemonUnavailable, serve  # noqa: PLC0415

    # Parsed before anything is composed or bound, so a mistyped duration is a
    # usage error rather than a daemon that got as far as printing a socket path.
    window = keep_alive_seconds(keep_alive)
    composed = profile_or_exit(profile, _children_cap(max_concurrent_children))
    try:
        roots = resolve_roots(create=True)
    except RuntimeDirError as error:
        fail(f"[red]$PH_RUNTIME check failed:[/red] {detail(error)}", cause=error)
    socket_path = roots.daemon_socket()
    err.print(f"[dim]listening on {socket_path}[/dim]")
    # "names `enable-linger` when a daemon is configured without it" — the row's
    # own wording, and this is the moment it is being configured. Said here as
    # well as in `phern doctor` because the two have different readers: doctor is
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
                #
                # **A keep-alive implies it.** "Stay up five minutes after the
                # last client leaves" is a statement about *leaving*, so a person
                # who typed one has said which lifetime they want; the two flags
                # were one concept split across two spellings, and the split is
                # what made `--keep-alive` alone mean nothing and need a warning.
                ephemeral=ephemeral or window > 0,
                keep_alive=window,
                path=socket_path,
            )
        )
    except DaemonUnavailable as error:
        # One named type rather than `(RuntimeError, OSError)`, which is two
        # builtins wide enough to swallow a `typer.Exit` — it subclasses
        # `RuntimeError`, and the comment in `doctor` above records this file
        # having been bitten by that already.
        fail(f"[red]{detail(error)}[/red]", cause=error)


def reinvoke(
    *args: str, profile: str, provider: str, model: str, patch: Sequence[str] = ()
) -> list[str]:
    """How pH starts pH: the argv for another process of this one.

    Two callers ask it — the daemon a UI spawns when no socket answers, and the
    terminal a browser tab runs — and they differ only in their leading verb. The
    tail is the same composition every time, so it is written once: an option
    added here reaches both, where two spellings would silently reach one.

    `sys.executable -m ph_app` rather than a bare `phern`: the caller may be running
    from a virtualenv that is not on `PATH`, or from a checkout with no console
    script installed at all, and a pH started from a *different* pH is one whose
    profile, event vocabulary and wire version nobody chose.

    `--patch` travels because the *other* process is the one that composes: a
    patch accepted here and dropped would silently ignore `phern --mode tui --patch
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


def tab_command(
    *,
    session: str,
    profile: str,
    provider: str,
    model: str,
    patch: Sequence[str] = (),
    spawn: bool = True,
    keep: bool = False,
) -> list[str]:
    """The argv for the terminal a browser tab runs (P7-05).

    A declaration beside `spawn_command`, rather than an argv assembled inline in
    the `--mode web` branch, because the two are the same kind of thing: the two
    ways pH re-invokes itself. Built inline it had no name to test — the argv
    gate pinned a composition *it* wrote, which omitted `--session` and
    `--keep-daemon` because no caller was there to disagree with it — and a
    launch-level flag that must reach a tab had to be remembered in a branch.

    `--session` is on every tab, which is the whole of P7-06's routing: upstream
    fixes the command at construction, so a session id here is necessarily every
    tab's, and a tab that minted its own would leave an upload with nothing to
    stage into.
    """
    return reinvoke(
        "--mode",
        "tui",
        "--session",
        session,
        *(() if spawn else ("--no-spawn",)),
        *(("--keep-daemon",) if keep else ()),
        profile=profile,
        provider=provider,
        model=model,
        patch=patch,
    )


def spawn_command(
    *,
    profile: str,
    provider: str,
    model: str,
    patch: Sequence[str] = (),
    keep: bool = False,
    keep_alive: str = "0",
) -> list[str]:
    """The argv for a daemon a UI starts on its own behalf (P7-08).

    Beside the `daemon` command whose options it spells, so renaming one is one
    edit. `--ephemeral` because this daemon was nobody's decision, so it leaves
    when nobody needs it.

    `keep` is `--keep-daemon`, the other half of the rule that **who started it
    decides**: a person who means the daemon to stay says so, and gets exactly
    what `phern daemon` would have given them, without typing two commands. It is
    the mirror of `phern daemon --ephemeral`, and both exist because the lifetime is
    a decision rather than a property of the process that happened to spawn it.
    """
    ephemeral = () if keep else ("--ephemeral",)
    # **The only route a client's preference has to the daemon.** A daemon may not
    # read a front end's `tui.json` — after P5-14 it may not share a filesystem
    # with it — so the terminal spells its own keep-alive into the command line it
    # composes. Omitted when zero, which is the default on both sides.
    window = ("--keep-alive", keep_alive) if keep_alive not in ("", "0") else ()
    return reinvoke(
        "daemon",
        *ephemeral,
        *window,
        profile=profile,
        provider=provider,
        model=model,
        patch=patch,
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
            f"{detail(error)}"
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
        # The names came off a command line, so they go through `detail` too.
        unmatched = f": {detail(', '.join(unknown))}" if unknown else ""
        fail(f"[red]no declared event matches{unmatched}[/red]", code=2)
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
    profile: ProfileOption = DEFAULT_PROFILE,
    patch: PatchOption = [],  # noqa: B006 - typer builds the list per invocation
    all_: Annotated[
        bool, typer.Option("--all", help="Include rows that take no configuration.")
    ] = False,
) -> None:
    """Print what every row accepts as configuration, and what a profile sets.

    Generated from each plugin's own `config=` model, so it cannot drift from
    the code the way a hand-written options table does — the same argument
    `phern events` makes about the event registry, applied to the other half of
    what a profile is.

    **The `default` column is the code's answer; the profile column is the
    deployment's**, and the second is the one a run actually uses. Composed, not
    mounted: it reads the same layered documents `--dump-config` prints,
    `--patch` included, so what somebody is about to run is what they can check —
    and it starts no agent and opens no session. The answer is the **root**
    agent's, which is what a child inherits. A row the profile does not mount is
    said so rather than shown with a value nothing would apply.

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
    # **After the JSON return, and after `--row`.** The profile column is a thing
    # the *table* prints, so composing above this made `--json` pay ~6 ms to
    # build a map it never reads — and, worse, gave a dump that depends on no
    # profile a way to exit 2 over one.
    #
    # `enabled_rows`, not `dump()`: a dump keeps a row a profile switched off,
    # flagged, so reading it as mounted told a person `disabled: true` was
    # "mounted, the default stands" — which is the third state this command
    # promises to tell apart, got wrong. `enabled_rows` is the predicate
    # `Profile.mount` itself uses.
    # `as_obj` is load-bearing: a mounted row that configures nothing has
    # `config is None`, which is a *different* answer from "not in this profile"
    # and must not share its sentinel — absence is what carries that one. It also
    # settles the shape here, so the renderer below reads presence alone.
    composed = {
        row.name: as_obj(row.config) for row in profile_or_exit(profile, patch).enabled_rows()
    }
    shown = [entry for entry in catalog if all_ or entry["config"] or entry.get("error")]
    if not shown:
        emit("no row matched")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("row")
    table.add_column("option")
    table.add_column("type")
    table.add_column("default")
    table.add_column(f"in {profile}")
    table.add_column("what it does")
    for entry in shown:
        if error := entry.get("error"):
            table.add_row(entry["name"], "[red]unavailable[/red]", "", "", "", error)
            continue
        if not entry["config"]:
            table.add_row(entry["name"], "[dim]none[/dim]", "", "", "", "")
            continue
        # `None`, not `{}`: a row this profile never mounts is a different answer
        # from one it mounts and says nothing about, and one lookup carries both.
        set_here = composed.get(entry["name"])
        for index, field in enumerate(entry["config"]):
            first = index == 0
            table.add_row(
                entry["name"] if first else "",
                field["name"],
                field["type"],
                # Required has no default, and printing one would invent a
                # value a profile must actually supply.
                "[bold]required[/bold]" if field["required"] else (field["default"] or ""),
                _in_profile(field["name"], set_here, first=first),
                field["doc"],
            )
    console.print(table)


def _in_profile(name: str, set_here: JsonObject | None, *, first: bool) -> str:
    """What the composed profile says about one option.

    Three answers, and they are genuinely different: the row is not in this
    profile at all, so nothing here applies; the row is mounted and says nothing
    about this option, so the default stands; or the profile set it, and that is
    what a run uses. `None` carries the first of those, so the caller keeps one
    lookup rather than two facts to hold in step. Blank for that case past its
    first line, because it is the *row's* answer and repeating it once per option
    is noise.
    """
    if set_here is None:
        return "[dim]row not mounted[/dim]" if first else ""
    if name not in set_here:
        return "[dim]—[/dim]"
    return f"[bold]{set_here[name]}[/bold]"


def main() -> None:
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
