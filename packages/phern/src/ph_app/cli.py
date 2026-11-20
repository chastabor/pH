"""`phern`: the command line.

Seven output modes and five diagnostics.

`--mode` decides what reaches stdout, and the choice matters more than a flag
usually does. `json` and `rpc` emit the envelopes of the session log **itself**
and not a rendering for each mode (I-7). A wrapper that streams from a pipe and
a tool that reads the stored JSONL therefore parse one format, and the tooling
of dsh reads both (Q2). `text` and `transcript` are for people. `tui` is the
interactive mode. It takes no `--print`, because the prompt is the interface.
`web` serves that terminal to browser tabs. `trajectory` shows a stored log and
mounts nothing.

`--dump-config` prints the composed rows before anything runs. `phern doctor`
prints the three resolved path roots, then mounts the profile and prints what
its rows report. `phern events` prints the producer/consumer matrix, which is
generated from the declaration registry and is not maintained by hand.
`phern config` prints what each row accepts as configuration. `phern revisions`
prints every revision to what the model saw, from a stored log.

`phern daemon` runs the supervisor, and `phern agents` is the client that talks
to it. They are the two halves of Phase 5. With this pair, the thing that the
user addresses is another process and not this one. The `tui` mode is also a
client of that daemon. `phern service` is the client that an orchestrator, a
cron slot or a person with no init system uses to start, stop, resume and watch
the daemon (P13-13).

@module ph_app.cli
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import sys
from collections.abc import Awaitable, Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Annotated, Any, Literal, TypeAlias

import anyio
import typer
import yaml
from pydantic import ValidationError
from rich.table import Table

from ph.cordis import LoaderError, MountRefusal, Profile, import_plugin_modules
from ph.cordis.catalog import config_catalog
from ph.cordis.events import events as event_registry
from ph.json import JsonObject, as_obj
from ph.keys import DIAGNOSTICS
from ph.lingering import lifetime
from ph.locks import describe_lock
from ph.paths import resolve_roots
from ph.persistence import SessionBusy, damaged_logs, truncate_damaged
from ph.persistence.lease import lease_directory
from ph.resources import run_until_signaled
from ph.seams.diagnostics import DiagnosticsRegistry
from ph.seams.models import ModelChoice, ModelChoiceError, ModelList
from ph.selectors import matches_any, unknown_namespaces
from ph.session import SessionForkError, new_session_id
from ph.session_profile import OverrideNotRecorded
from ph.wire import validation_summary

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
    nonblank,
    roots_or_exit,
    section,
    selectors_or_exit,
)
from .daemon.launch import DETACHED, phern_argv
from .daemon.recovery import PASSIVATE_AFTER
from .modes import run_json, run_print, run_rpc, run_transcript
from .profiles import (
    DEFAULT_PROFILE,
    ModelOption,
    PatchOption,
    ProfileOption,
    ProviderOption,
    available_profiles,
    profile_or_exit,
    unfolded_profiles,
)
from .profiles_cli import profiles_app
from .revisions import revision_rows
from .runtime import mounted
from .service import service_app
from .stored import load_session
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
# The client half of the daemon, as its own group (P5-10). It is a sub-app and
# not nine top-level commands, because every one of them means the same thing:
# "ask the supervisor". A user who has not started a supervisor should find that
# out from one place.
app.add_typer(agents_app, name="agents")
# The half of `/workspaces` that spans sessions (P6-28). It is a group of its own
# and not a verb on the slash command, for two reasons. The question that it
# answers spans every session on disk, while `/workspaces` answers for the one
# session that the user is in. Also, a model should not be able to ask for the
# collection of checkouts by emitting text.
app.add_typer(workspaces_app, name="workspaces")
# The other accounting command that spans sessions (P7-01). It is its own group
# beside `workspaces` and not a verb there, because the two sweep different
# things under different rules. A retained tree is evidence that somebody may
# want. An attachment is content that a session cannot open without. A user who
# reads `--help` should not have to infer from a shared verb which rule applies.
app.add_typer(attachments_app, name="attachments")
# What a named profile sets, and the fold of the older layers into it (session
# profiles, S2). It is a group because a profile is a thing on disk that the user
# manages between sessions. It is not a question about the session they are in.
app.add_typer(profiles_app, name="profiles")
app.add_typer(service_app, name="service")

OutputMode = Literal["text", "json", "transcript", "rpc", "tui", "web", "trajectory"]

ModeRunner: TypeAlias = Callable[..., Awaitable[Any]]
"""Each mode returns its own result type: `json` reports a count, and `text` and
`transcript` report text. The table is therefore typed by what the modes have in
common: they are awaited, and the caller branches on the mode that it asked
for."""

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
    provider: ProviderOption = None,
    model: ModelOption = None,
    session_id: Annotated[
        str | None,
        typer.Option(
            "--session",
            help="Session id to create or resume, or to read under --mode trajectory.",
            callback=nonblank,
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
        str | None,
        typer.Option("--resume", help="Session id to reopen (tui and web).", callback=nonblank),
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
        # The view of the auditor. This branch is deliberately *before* any
        # profile work, because it does not read a profile. It mounts nothing:
        # no agent, no provider, no answerers. The logs most worth auditing are
        # the ones that nobody can reopen (P3-25).
        if session_id is None:
            fail("[red]--mode trajectory needs --session <id|path>[/red]", code=2)
        # A late load: a command pays only for what it runs (`test_app_layering.py`).
        from .tui.trajectory_app import run_trajectory  # noqa: PLC0415

        try:
            code = anyio.run(partial(run_trajectory, session_id))
        except (OSError, ValueError) as error:
            fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
        if code:
            # The exit code of the viewer itself. It is silent for the reason
            # that the `--mode tui` branch gives below.
            raise typer.Exit(code)
        return

    # `--resume <id>` is `--session <id>`. The daemon resumes an existing id
    # through the same call that creates a new one. The two spellings therefore
    # have one answer, and both interactive modes read it here.
    wanted = session_id if session_id is not None else resume
    choice = _choice_or_exit(provider, model)

    if mode == "tui":
        # This branch is **before any profile work**, like the trajectory branch
        # above and for the same reason: the terminal does not mount a profile.
        # The daemon composes the profile, and the terminal only names it. The
        # name goes into `spawn_command` for a daemon that this terminal starts,
        # and into `profile=` for the sessions that it creates (S5). Composing a
        # profile here cost a full plugin import for each TUI start, and the
        # result was discarded.
        #
        # The import is here because the TUI loads Textual, and `phern -p` in a
        # script should not pay for a terminal UI that it never draws.
        from .tui.app import run_tui  # noqa: PLC0415

        code = anyio.run(
            partial(
                run_tui,
                # The *name* and not the composed profile: the daemon composes
                # it. This command line gives a spawned daemon the name.
                daemon_argv=spawn_command(
                    profile=profile,
                    patch=patch,
                    keep=keep_daemon,
                    keep_alive=_spawned_keep_alive(keep_alive, keep=keep_daemon),
                ),
                # The route belongs to the session of this terminal and not to
                # the daemon. It is sent in `session/new`, so a daemon that
                # already serves other sessions keeps them on their own routes.
                model_choice=choice,
                # The profile that a session of this terminal runs on. A spawned
                # daemon also has it as its default. A daemon that already runs
                # with another default still creates the sessions of this
                # terminal on this profile (S5).
                profile=profile,
                session_id=wanted,
                spawn=not no_spawn,
                # `--session` and `--resume` already name a session, so they skip
                # the offer because they have an answer. `--new` is for the user
                # who never wants to be asked.
                offer_sessions=not new,
            )
        )
        if code:
            # **Silent, and not `console.fail`** (M4). Every other refusal here
            # goes through `fail`, which prints a sentence. The terminal has
            # already rendered its own crash. A second line under it would be pH
            # explaining an error that the user just watched happen. `run_tui`
            # returns the code so that the exit is here and not inside a library
            # function.
            raise typer.Exit(code)
        return

    if mode == "web":
        # This branch is before the profile work, like `tui` above and for the
        # same reason. The terminal that each tab runs is the thing that talks
        # to a daemon, and this process only serves those terminals.
        #
        # The import is here because `textual-serve` is an extra. `phern -p` in a
        # script must not require aiohttp. A user who asked for `--mode web`
        # without the extra gets the install line and not a traceback.
        try:
            from .web.serve import WebServer, run_web  # noqa: PLC0415
        except ImportError as error:
            fail(
                "[red]--mode web needs the web extra:[/red] pip install 'phern\\[web]'",
                cause=error,
            )
        # **One session for every tab of this launch.** The id is minted here
        # when the user did not name one. The module docstring of `serve.py`
        # says why a tab cannot have a session of its own.
        shared = wanted if wanted is not None else new_session_id()
        tab = tab_command(
            session=shared,
            profile=profile,
            provider=provider,
            model=model,
            patch=patch,
            spawn=not no_spawn,
            keep=keep_daemon,
            # Only a value that the user typed is forwarded. With none, each
            # tab reads `tui.json` itself, as a terminal does.
            keep_alive=(
                None if keep_alive is None else _spawned_keep_alive(keep_alive, keep=keep_daemon)
            ),
        )
        server = WebServer(command=shlex.join(tab), session=shared, host=host, port=port)
        # The notices are printed before the bind. The print is here because
        # `notices()` owns the words.
        for notice in server.notices():
            err.print(notice)
        anyio.run(partial(run_web, server, open_browser=open_browser))
        return

    composed = profile_or_exit(profile, patch)

    if dump_config:
        emit(yaml.safe_dump(composed.dump(), sort_keys=False, default_flow_style=False).rstrip())
        return

    if mode == "rpc":
        # There is no prompt: the peer drives the session over stdio. A signal
        # stops this mode in the way that it stops the daemon, and a peer that
        # closes it with `SIGTERM` is the common case (`run_until_signaled`).
        # That stop leaves through the signal before the records of the mounts
        # are read. The warning of the last write itself names each log.
        try:
            unwritten = run_until_signaled(
                partial(run_rpc, composed, choice=choice), on_stop=_say_stopped
            )
        except MountRefusal as error:
            fail_unmounted(profile, error)
        _exit_if_unwritten(unwritten)
        return

    if prompt is None:
        console.print(ctx.get_help())
        return

    route = partial(
        _MODES[mode],
        composed,
        prompt,
        choice=choice,
        session_id=session_id,
        attachments=attach or [],
    )
    try:
        outcome = run_until_signaled(route, on_stop=_say_stopped)
    except MountRefusal as error:
        # A row that *declined* gets the sentence that doctor prints, with the
        # exit code of doctor. `containment.strict` with no backend is such a
        # row (E8). A bug in an `apply` still gets a traceback. Before this, the
        # run path had no name for the difference. It printed 191 lines for the
        # one case that a user most needs to read (P4-12).
        fail_unmounted(profile, error)
    except (
        AttachmentUnavailable,
        LoaderError,
        ModelChoiceError,
        OSError,
        OverrideNotRecorded,
        SessionBusy,
        SessionForkError,
    ) as error:
        # A file that cannot be read fails the *command*. `prompted` ingests
        # before the agent exists, so nothing was logged and there is no partial
        # turn to explain. A row whose plugin does not import is the same kind
        # of failure one step earlier. It is the one refusal of the loader that
        # is left at mount time, now that `profile_or_exit` composes. A session
        # that another process holds has the same shape (I-5). It is refused
        # before a byte is written, with the one sentence that the daemon would
        # have sent and not a traceback. The same applies to the log of a
        # sub-agent, which only the mount of its root writes (P11-08). It also
        # applies to a start option that the log could not record: the run does
        # not use an option that is not recorded.
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)

    # `json` has written its output already, event by event, as each committed.
    if mode != "json":
        console.print(outcome.text)
    failed = False
    if mode == "text":
        err.print(
            f"[dim]session {outcome.session_id} · {outcome.events} events · "
            f"{outcome.log_path}[/dim]"
        )
        failed = outcome.ended == "error"
    if failed:
        # Whatever arrived before the failure is still printed above. It is in
        # the log in both cases, and a truncated answer is worth seeing. The run
        # must not exit 0. Scripts call a `-p` run, and a provider outage that
        # reports success cannot be told from an answer. Only `error` fails the
        # run: `blocked` and `max-tokens` are turns that ended the way they were
        # asked to end.
        err.print(f"[red]the turn failed:[/red] {detail(outcome.failure)}")
    # Both are reported. A log that is left short is the more serious exit of
    # the two.
    _exit_if_unwritten(outcome.unwritten)
    if failed:
        raise typer.Exit(code=1)


def _exit_if_unwritten(unwritten: Mapping[str, str]) -> None:
    """Exit 3 when the last write of the mount could not write a log, as
    `phern daemon` does (P13-08). What the run printed stands, and the next run
    resumes a log that is missing its tail."""
    if unwritten:
        fail(
            "\n".join(
                f"[red]session {session_id} could not be written:[/red] {detail(reason)}"
                for session_id, reason in sorted(unwritten.items())
            ),
            code=3,
        )


def _say_stopped(signum: int) -> None:
    """Report on stderr a one-shot run that a signal stopped, after its unwind is over."""
    err.print(f"[yellow]stopped on {signal.Signals(signum).name}[/yellow]")


NO_DIAGNOSTICS_ROW = "none — this profile mounts no `diagnostics` row, so no row can report"


async def _note_consumers(profile: Profile) -> None:
    """Mount, and let the `ctx.on` of every row register itself into the registry.

    Nothing is read back here. `note_consumer` writes into the process-wide
    `EventRegistry` as a side effect of listening. The mount therefore *is* the
    query, and the registry outlives the scope that filled it. Nothing is created
    beyond the mount (no session, no agent, no provider call), for the reason
    that `_report` gives.
    """
    async with mounted(profile):
        return


async def _report(profile: Profile) -> list[tuple[str, list[tuple[str, str]]]]:
    """Mount the profile and ask every row what it reports (P4-12).

    The mount is the purpose. Before this, doctor answered from `resolve_roots()`
    alone. It could report where the log *would* go, and nothing about what the
    process would actually be. Every question that this row was written for is
    answered by a row and not by a path: which rung is in force, what the file
    rules reach, what runs model code. Nothing is created here beyond the mount:
    no session, no agent, no provider call. Topology is a row
    (`ph.seams.topology`), so the registry is the only source.
    """
    async with mounted(profile) as ctx:
        registry: DiagnosticsRegistry | None = ctx.get(DIAGNOSTICS)
        if registry is None:
            # Rule 6, applied here on behalf of the seam. With no `diagnostics`
            # row nothing can report, and an empty report reads as "nothing
            # wrong".
            return [("Diagnostics", [("sections", NO_DIAGNOSTICS_ROW)])]
        return registry.report()


doctor_app = typer.Typer(
    invoke_without_command=True,
    help="Report the path roots, then mount a profile and report what it composed.",
)


@doctor_app.callback()
def doctor(
    ctx: typer.Context,
    profile: ProfileOption = DEFAULT_PROFILE,
    patch: PatchOption = [],  # noqa: B006 - typer builds the list per invocation
) -> None:
    """Report the path roots, then mount a profile and report what it composed.

    `phern doctor sessions` is the one subcommand: a question about the stored logs
    rather than about this install, so it mounts nothing.
    """
    if ctx.invoked_subcommand is not None:
        return
    roots = roots_or_exit()
    table = Table(title="pH path roots", show_header=True, header_style="bold")
    table.add_column("root")
    table.add_column("resolved")
    for name, value in roots.describe():
        table.add_row(name, value)
    console.print(table)
    console.print(f"[dim]platform: {sys.platform} · python {sys.version.split()[0]}[/dim]")
    # This has the same subject as the roots table and asks one further
    # question. The table says where `$PH_RUNTIME` resolved. This says whether
    # what a daemon puts there is still there tomorrow (I-6, P5-11).
    #
    # **It is not a `ctx.diagnostics` section.** This is the one place in this
    # command that has two mechanisms, and that is deliberate. A contributed
    # section is lost exactly when `doctor` stops with "profile does not mount",
    # which is the case where a user most wants it. This probe needs no profile
    # to answer. It still follows the *shape* of the seam. A section is a title
    # and a list of pairs, both halves produce `list[tuple[str, str]]`, and both
    # render through `console.section`. The next probe that needs no profile is
    # therefore an entry in this list, and not one more `console.print` in a
    # command body.
    #
    # The entry is listed even when the answer is "yes". Rule 6 says to state
    # what is not enforced next to where it would be assumed. Every reader who
    # never sees a warning assumes that `phern daemon` means "until I stop it".
    #
    # `roots=` is passed so that this entry and the table above describe one
    # resolution and not two independent ones.
    #
    # The import is here and not at module scope, because `phern --print` has no
    # daemon in it and should not load `ph_app.daemon.supervisor`. `daemon()`
    # below imports `serve` in the same way.
    from .daemon.supervisor import NON_GUARANTEES  # noqa: PLC0415

    before_mount = [
        ("daemon socket lifetime", lifetime(roots=roots).describe()),
        # The lease is a kernel `flock` or nothing (P13-11). A filesystem that
        # cannot hold one would give the next daemon every lease that this one
        # held when it died. The probe takes a real lock in the lease directory,
        # because only the filesystem knows the answer.
        ("session lease", describe_lock(lease_directory(roots.sessions_dir()))),
        # N5, and "doctor prints the worker model" from the gate. The entry is
        # here and not only in `phern agents doctor`, because the two answer
        # different questions. That command describes a daemon that somebody
        # already started. This entry is read by the user who decides whether to
        # run six agents under one daemon (I-2).
        ("daemon isolation", list(NON_GUARANTEES)),
    ]
    # What this install can actually compose. A bundle profile whose distribution
    # is missing is not offered (P3-20).
    console.print(f"[dim]profiles: {', '.join(available_profiles())}[/dim]")
    # These layers are read until they are folded, so nothing is dropped. They
    # are named here, so nothing is read silently (session profiles, item 0).
    unfolded = unfolded_profiles()
    if unfolded:
        console.print(
            f"[yellow]not folded: {', '.join(unfolded)} — "
            "`phern profiles fold` makes each one file[/yellow]"
        )

    # The profile is resolved *outside* the catch below, and that matters.
    # `typer.Exit` subclasses `RuntimeError`. A broad catch around the resolution
    # would catch an unknown profile, report it as "does not mount", and raise it
    # again with the wrong exit code.
    composed = profile_or_exit(profile, patch)
    try:
        sections = anyio.run(partial(_report, composed))
    except typer.Exit:
        raise
    except Exception as error:
        # Broad on purpose, and only here. A profile that refuses to start is
        # the most important thing that doctor can report. `containment.strict`
        # on a host with no sandbox backend is exactly that case (E8). A user who
        # ran the command *because* the process does not start must get the
        # sentence and not a traceback. The exit code says that it failed.
        fail_unmounted(profile, error)

    console.print(f"\n[bold]profile:[/bold] {profile}")
    for title, rows in before_mount + sections:
        console.print(section(title, rows))


@doctor_app.command("sessions")
def doctor_sessions(
    repair: Annotated[
        bool,
        typer.Option(
            "--repair",
            help="Cut each damaged log back to the prefix that reads, keeping the whole "
            "file beside it as `<log>.damaged`. A log a running process holds is left.",
        ),
    ] = False,
) -> None:
    """Name the first line of each stored log the strict reader refuses (P13-09).

    Every log is opened as a resume opens it. Exit 0 when every log reads, 1 while
    any does not.

    A damaged log is one a resume refuses whole: a line that is not a record, or an
    envelope this build cannot validate, with a newline after it. A torn last line is
    not damage — a crash leaves one, and the reader drops it. A sub-agent's damaged
    log leaves its root openable and listed `damaged`; `--repair` is what clears
    that, and the daemon reads the repaired log when it next mounts the root.
    """
    root = roots_or_exit().sessions_dir()
    read, found = damaged_logs(root)
    console.print(f"[dim]{read} stored log(s) under {root}[/dim]")
    if not found:
        console.print("[green]every log reads[/green]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("session")
    table.add_column("line")
    table.add_column("reason")
    table.add_column("repair" if repair else "would keep")
    left = 0 if repair else len(found)
    for damage in found:
        if not repair:
            outcome = (
                f"lines 1 to {damage.line - 1}"
                if damage.kept
                else "nothing: no header reads before the damage"
            )
        else:
            try:
                copy = truncate_damaged(root, damage)
            except (SessionBusy, ValueError, OSError) as refused:
                left += 1
                outcome = f"[yellow]not repaired: {detail(refused)}[/yellow]"
            else:
                outcome = f"cut at line {damage.line}; the whole file is {copy.name}"
        table.add_row(damage.path.stem, str(damage.line), detail(damage.reason), outcome)
    console.print(table)
    if repair and left < len(found):
        console.print(
            "[dim]a daemon reads a repaired log when it next mounts the root; "
            "`phern agents owed --rebuild` re-reads the index[/dim]"
        )
    if left:
        fail(
            f"[yellow]{left} damaged log(s)"
            + ("" if repair else "; `phern doctor sessions --repair` cuts each back")
            + "[/yellow]"
        )


app.add_typer(doctor_app, name="doctor")


_DEFAULT_PASSIVATION = f"{PASSIVATE_AFTER / 60:g}"
"""The default of the flag, derived from the constant that justifies the number.

The default was spelled `"90"` here while `PASSIVATE_AFTER` said ninety minutes
in seconds one module away. That was two literals in two units for one policy.
An edit to the documented literal changed nothing for anyone who ran
`phern daemon`, because the CLI always passes its own value.
"""


def _spawned_keep_alive(value: str | None, *, keep: bool) -> float:
    """Return the `--keep-alive` for a spawned daemon, in seconds. Refuse a typo here.

    **The value is parsed at this end and not forwarded as typed.**
    `phern --keep-alive` does not run a daemon. It composes the argv for one. A
    string forwarded as typed would be refused by the spawned daemon. The user
    would then see a daemon that exited before it listened (`_await_socket` in
    `ph_app.daemon.launch`), and not a usage error for the flag that they typed.
    The parser and the message are the same at whichever end the user typed the
    value. What crosses is a bare number of seconds, which `keep_alive_seconds`
    reads back without ambiguity.

    The order is CLI first, then `tui.json`, then zero. A user who sets the value
    one time should not have to type it, and a user who typed it means this run.
    The file is read only when the user did not type a value. That is also why
    the read is not hoisted. Settings are deliberately read again for each
    `PHTuiApp`, so that a `/theme` written during a session is picked up when the
    picker reopens a session.

    **`--keep-daemon` and a typed `--keep-alive` are refused together**, because
    they ask for opposite things. One says that the daemon stays. The other says
    how long it waits before it leaves. A warning would be the wrong answer to a
    user who has said two things and meant one of them. A *configured*
    keep-alive is not a contradiction, because nobody typed it for this run.
    `--keep-daemon` overrides it, and a service daemon has no window.

    `ph_app.tui.config` is imported inside the function for the same reason as
    `run_tui`: it is a front-end module, and `phern -p` should not load one to
    print an answer.
    """
    if keep:
        if value is not None and keep_alive_seconds(value) > 0:
            raise typer.BadParameter(
                "--keep-alive and --keep-daemon ask for opposite things: one says when "
                "the daemon leaves, the other says it stays"
            )
        return 0.0
    if value is None:
        # A late load: a command pays only for what it runs (`test_app_layering.py`).
        from .tui.config import load_tui_settings  # noqa: PLC0415

        value = load_tui_settings(resolve_roots().home).daemon_keep_alive
    return keep_alive_seconds(value)


UNITS: Mapping[str, float] = {"s": 1.0, "m": 60.0, "h": 3600.0}
"""The suffixes that a duration option accepts, each with its length in seconds."""


def _duration_seconds(value: str, *, unit: float) -> float | None:
    """Return a duration that a user typed, as seconds, or `None` when the text is
    not a duration. A negative number is not a duration.

    This is one parser for `--keep-alive` and `--passivate-after`. The two flags
    ask the same question, and they had two parsers with two refusal shapes.
    `unit` is what a *bare* number means: seconds for a keep-alive, and minutes
    for a passivation window. The history of each flag established its unit. A
    suffix overrides the unit, so `--passivate-after 30s` now means what it reads
    like and not half an hour.

    The function returns `None` and does not raise, because the two callers
    refuse differently. One of them has an `"off"` spelling to check first, and
    a parser that raised would have to know about it.
    """
    text = value.strip().lower()
    if not text:
        return None
    scale = UNITS.get(text[-1])
    number = text[:-1] if scale is not None else text
    try:
        seconds = float(number) * (scale if scale is not None else unit)
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


def keep_alive_seconds(value: str) -> float:
    """Return `"30s"`, `"5m"`, `"1h"` or a bare number of seconds, as seconds.

    A typo is refused and not defaulted, for the same reason as in
    `_passivation`. A mistyped duration here gives a daemon that leaves
    immediately or never. Either is a surprise that a user attributes to
    something else a week later.

    **The function is public, because both ends of the flag must agree.**
    `phern daemon --keep-alive` parses the value here. `phern --keep-alive` does
    not run the daemon, it *spawns* one, and it also parses the value here. If it
    forwarded the string into an argv, a typo would show as a daemon that refused
    to start. The refusal and the message are the same at whichever end the user
    typed the value.

    **A duration and nothing else.** Whether the daemon is ephemeral is a
    question for the caller and not for this function. `phern daemon` and `phern`
    express that in opposite directions: one opts into leaving, and the other
    opts into staying. A parser that also decided it would have to know which
    command it served.
    """
    if not value.strip() or value.strip() == "0":
        return 0.0
    seconds = _duration_seconds(value, unit=1.0)
    if seconds is None:
        # `typer.BadParameter`, like `_passivation` and `_children_cap` beside
        # this function, and not a hand-built refusal. Typer already writes the
        # usage line and the exit code. When three value parsers on one command
        # refuse in two different shapes, one of them comes to say something
        # else.
        raise typer.BadParameter(f'wants a duration like "30s", "5m" or "90", not "{value}"')
    return seconds


def _passivation(value: str) -> float | None:
    """Parse `"off"`, a number of minutes, or a suffixed duration, as seconds (P5-05).

    `"off"` returns `None`. A value that is none of those is refused and not
    defaulted. A typo in a duration gives a deployment that silently keeps every
    root that it ever started, and the daemon is the one process where that goes
    unnoticed for a week.
    """
    if value.strip().lower() == "off":
        return None
    seconds = _duration_seconds(value, unit=60.0)
    if seconds is None or seconds <= 0:
        # One message for one mistake: "not a number" and "not a positive
        # number" are the same correction to the same flag.
        raise typer.BadParameter(f'wants positive minutes or "off", not "{value}"')
    return seconds


def _choice_or_exit(provider: str | None, model: str | None) -> ModelChoice:
    """Return the choice that `--provider` and `--model` spell, or exit 2 with the reason."""
    try:
        return ModelChoice.from_flags(provider, model)
    except ModelChoiceError as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)


def _children_cap(value: int | None) -> list[str]:
    """Return the `--max-concurrent-children` override as the patch that it is.

    The cap *is* row config, so an override from the command line is a patch and
    not a second channel. `--dump-config` and `phern doctor` then show the number
    that is actually in force, with `cli` as its provenance. A field passed
    around the profile could not do that. It is a *deployment* patch and not a
    `--patch`: `jobs` is a row of the host, which the `--patch` of a user may not
    set (`profile_or_exit`).

    **An unset flag patches nothing.** The flag therefore needs no default of its
    own. Whatever the composed profile says stands, so the number is written one
    time, in the bundle that ships a subagent provider. It also keeps a plain
    `phern daemon` working on a profile with no `jobs` row. A patch names a row by
    id, and the loader refuses a patch for a row that is not there. A default
    that always patched would have made every such run exit 2 over a cap that
    nobody asked for. When the user names a number on such a profile, the
    refusal is deserved, and it says which row is missing.

    The kind comes from `ph.seams.jobs`. It is imported late, as the other
    imports of the daemon are, because every `phern` invocation reads this module
    and the seam is not otherwise on that path.
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
    patch: PatchOption = [],  # noqa: B006 - typer builds the list per invocation
    provider: ProviderOption = None,
    model: ModelOption = None,
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
    probe_bind: Annotated[
        str | None,
        typer.Option(
            "--probe-bind",
            help="Answer startup, readiness and liveness probes over HTTP here (host:port).",
        ),
    ] = None,
    resume_owed: Annotated[
        bool,
        typer.Option(
            "--resume-owed",
            help="Mount every stored root that still owes work once the socket is up.",
        ),
    ] = False,
    continue_interrupted: Annotated[
        bool,
        typer.Option(
            "--continue-interrupted",
            help="On resume, continue a person's cut-short turn too, not only a goal's "
            "or a schedule's.",
        ),
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

    `--resume-owed` is the pass `phern agents resume --all` runs, run by the daemon
    itself once its socket is up, for a unit with nothing to run it from (P13-03);
    readiness is no until it ends. `--continue-interrupted` widens what a resume
    continues unasked to a person's own cut-short turn (P13-04).
    """
    # A late load: a command pays only for what it runs (`test_app_layering.py`).
    from .daemon.server import DaemonUnavailable, serve  # noqa: PLC0415

    # The duration is parsed before anything is composed or bound. A mistyped
    # duration is then a usage error, and not a daemon that got as far as
    # printing a socket path.
    window = keep_alive_seconds(keep_alive)
    composed = profile_or_exit(profile, patch, deployment=_children_cap(max_concurrent_children))
    # The choice is refused before the socket is bound. Otherwise every root
    # would refuse a `--model` that the profile does not list, one `session/new`
    # at a time. Only a mount knows whether an adapter serves the provider, so
    # each root refuses that case (`model_for` in `ph_app.daemon.supervisor`).
    choice = _choice_or_exit(provider, model)
    try:
        ModelList.of(composed).resolve(choice)
    except ModelChoiceError as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
    except ValidationError as error:
        said = validation_summary(error, root="config")
        fail(f"[red]the models row: {said}[/red]", code=2, cause=error)
    roots = roots_or_exit(create=True)
    socket_path = roots.daemon_socket()
    err.print(f"[dim]listening on {socket_path}[/dim]")
    # "names `enable-linger` when a daemon is configured without it" is the
    # wording of the row itself, and this is the moment when the daemon is
    # configured. The warning is printed here as well as in `phern doctor`,
    # because the two have different readers. Somebody who is already debugging
    # runs doctor. Somebody who is not debugging reads this line, ten seconds
    # before they close the terminal that it is printed in.
    life = lifetime(socket_path, roots=roots)
    if life.survives_logout is not True:
        err.print(f"[yellow]this socket does not survive logout:[/yellow] {life.verdict()}")
        err.print(f"[yellow]  {life.advice}[/yellow]")
    # Logging is configured before `serve` can log. When this process is
    # detached, its stderr is a file that the launcher reads only until the
    # socket answers, and this log is where its warnings go
    # (`ph_app.daemon.logs`). The call is directly above the `try` whose
    # `finally` releases the logging, so nothing between them can skip the
    # release.
    from .daemon.logs import configure_daemon_logging, release_daemon_logging  # noqa: PLC0415

    log_path = roots.daemon_log()
    configure_daemon_logging(log_path, to_stderr=DETACHED not in os.environ)
    err.print(f"[dim]logging to {log_path}[/dim]")
    try:
        # This is the path that was printed and not a second resolution of it.
        # A message that names one socket while the bind takes another costs
        # somebody an hour of debugging.
        unwritten = anyio.run(
            partial(
                serve,
                composed,
                choice=choice,
                passivate_after=_passivation(passivate_after),
                # Off here, and on in `spawn_command`. `DaemonServer.ephemeral`
                # says why whoever started the daemon decides its lifetime
                # (P7-08).
                #
                # **A keep-alive implies it.** "Stay up five minutes after the
                # last client leaves" is a statement about *leaving*, so a user
                # who typed one has said which lifetime they want. The two flags
                # were one concept split across two spellings. The split made
                # `--keep-alive` alone mean nothing and need a warning.
                ephemeral=ephemeral or window > 0,
                keep_alive=window,
                path=socket_path,
                # `kill`, `systemctl stop` and Ctrl-C end this process in the way
                # that `phern agents shutdown` does, and not in the way that
                # `kill -9` does.
                signals=True,
                probe=probe_bind,
                resume_owed=resume_owed,
                continue_interrupted=continue_interrupted,
            )
        )
        if unwritten:
            # A stop that lost the tail of a log is not a clean stop (P13-08).
            # The unit shows failed, and the resume of the next daemon repairs
            # what this one could not write.
            fail(
                f"[red]stopped with {unwritten} session log(s) that could not be "
                "written; see the daemon log[/red]",
                code=3,
            )
    except DaemonUnavailable as error:
        # One named type and not `(RuntimeError, OSError)`. Those two builtins
        # are wide enough to catch a `typer.Exit`, which subclasses
        # `RuntimeError`. The comment in `doctor` above records that this file
        # already had that bug.
        fail(f"[red]{detail(error)}[/red]", cause=error)
    finally:
        # A host that runs this command in-process (a test runner) gets its
        # logging back the way it was. The real daemon is exiting anyway.
        release_daemon_logging(log_path)


def reinvoke(
    *args: str,
    profile: str,
    provider: str | None = None,
    model: str | None = None,
    patch: Sequence[str] = (),
) -> list[str]:
    """Return the argv with which pH starts another process of itself.

    Two callers use it: the daemon that a UI spawns when no socket answers, and
    the terminal that a browser tab runs. They differ only in their leading
    words. The tail is the same composition every time, so it is written one
    time. An option added here reaches both callers, where two spellings would
    silently reach one. The interpreter and the module in front of the tail come
    from the launcher (`phern_argv`). `phern service start` uses `phern_argv`
    directly, with the daemon options of its caller behind it.

    `--patch` is forwarded because the *other* process is the one that composes.
    A patch that was accepted here and dropped would silently ignore
    `phern --mode tui --patch '{id: tool-ask-user, disabled: false}'`, which is
    the documented way to arm a row anywhere. The route is forwarded only when
    the user typed one. An unset route is the default of the profile, and a
    default written into the argv would be a choice that nobody made.
    """
    argv = phern_argv(*args, "--profile", profile)
    if provider is not None:
        argv += ["--provider", provider]
    if model is not None:
        argv += ["--model", model]
    for one in patch:
        argv += ["--patch", one]
    return argv


def tab_command(
    *,
    session: str,
    profile: str,
    provider: str | None = None,
    model: str | None = None,
    patch: Sequence[str] = (),
    spawn: bool = True,
    keep: bool = False,
    keep_alive: float | None = None,
) -> list[str]:
    """Return the argv for the terminal that a browser tab runs (P7-05).

    This is a declaration beside `spawn_command`, and not an argv assembled
    inline in the `--mode web` branch, because the two are the same kind of
    thing: the two ways in which pH invokes itself again. Built inline, the argv
    had no name to test. The argv test checked a composition that the test
    *itself* wrote. That composition omitted `--session` and `--keep-daemon`,
    because no caller was there to disagree with it. Also, a launch-level flag
    that must reach a tab had to be remembered in a branch.

    `--session` is on every tab, and that is all of the routing of P7-06.
    Upstream fixes the command at construction, so a session id here necessarily
    belongs to every tab. A tab that minted its own id would leave an upload
    with no session to stage into.

    `keep_alive` is the `--keep-alive` that the user typed, in seconds, or
    `None` when they typed none. `--mode web` used to drop the flag, so the
    daemon that a tab started left at once, whatever the user had asked for.
    """
    return reinvoke(
        "--mode",
        "tui",
        "--session",
        session,
        *(() if spawn else ("--no-spawn",)),
        *(("--keep-daemon",) if keep else ()),
        *(() if keep_alive is None else _keep_alive_argv(keep_alive)),
        profile=profile,
        provider=provider,
        model=model,
        patch=patch,
    )


def _keep_alive_argv(seconds: float) -> tuple[str, str]:
    """`--keep-alive` as a command line that pH composes spells it: a bare number
    of seconds, which `keep_alive_seconds` reads back with no unit to guess."""
    return ("--keep-alive", f"{seconds:g}")


def spawn_command(
    *,
    profile: str,
    patch: Sequence[str] = (),
    keep: bool = False,
    keep_alive: float = 0.0,
) -> list[str]:
    """Return the argv for a daemon that a UI starts for itself (P7-08).

    The function is beside the `daemon` command whose options it spells, so that
    a rename is one edit. It passes `--ephemeral` because nobody decided to run
    this daemon, so the daemon leaves when nobody needs it.

    `keep` is `--keep-daemon`. It is the other half of the rule that **whoever
    started the daemon decides**. A user who means the daemon to stay says so,
    and gets exactly what `phern daemon` would have given them, without typing
    two commands. It mirrors `phern daemon --ephemeral`. Both exist because the
    lifetime is a decision, and not a property of the process that happened to
    spawn the daemon.

    **No route.** The terminal names its own route in `session/new`. The model
    that the terminal was started with therefore belongs to its session, and not
    to every session on a daemon that may outlive the terminal.
    """
    ephemeral = () if keep else ("--ephemeral",)
    # **This is the only way that a preference of a client reaches the daemon.**
    # A daemon may not read the `tui.json` of a front end, and after P5-14 it may
    # not share a filesystem with one. The terminal therefore writes its own
    # keep-alive into the command line that it composes. The option is omitted
    # when it is zero, which is the default on both sides.
    window = _keep_alive_argv(keep_alive) if keep_alive > 0 else ()
    return reinvoke(
        "daemon",
        *ephemeral,
        *window,
        profile=profile,
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
    # The profile is resolved *outside* the guard below. `profile_or_exit`
    # reports an unknown profile by raising `typer.Exit`, which is an
    # `Exception`. A broad catch around it turned "no such profile" into a full
    # matrix and exit 0. That is the answer that looks most like success, for
    # the input that is most likely to be a typo.
    composed = profile_or_exit(profile, patch)
    try:
        anyio.run(partial(_note_consumers, composed))
    except Exception as error:
        err.print(
            f"[yellow]profile {profile!r} does not mount, so no consumers are listed:[/yellow] "
            f"{detail(error)}"
        )
    # The bus vocabulary, so a bare `tools` needs no prefix. `log:workspace` is
    # refused and not answered with an empty table, because this registry holds
    # no session-log types and the two vocabularies share six roots (P6-33).
    selectors = selectors_or_exit(type_, vocabulary="bus")
    matrix = [row for row in event_registry.matrix() if matches_any(row["name"], selectors)]
    if selectors and not matrix:
        # A namespace that nothing occupies is a typo far more often than it is
        # an empty one. This registry knows every name that it holds, so it can
        # say which namespace it was and not print an empty table.
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
    # The consumers are half of the purpose of a producer/consumer matrix, and
    # the rendered table shipped without them for one round. `matrix()` had
    # carried them all along. The JSON was complete, and only the half that a
    # person reads was missing. An event with no listener is a real finding (a
    # declared extension point that nobody uses), so an empty cell says so.
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
def revisions(
    target: Annotated[str, typer.Argument(help="A session id, or the path to a stored log.")],
) -> None:
    """Print every revision to what the model saw, one JSON line each.

    What the change replaced and what it is now, its producer's records, the size the
    revision door wrote and the tool calls those records name, for diagnosis and for
    training data (`ph_app.revisions`). Read from the log on disk with nothing mounted,
    as `--mode trajectory` reads one, so a crashed run or a copy someone sent exports
    as well as a live one.
    """
    try:
        session = load_session(target)
    except (OSError, ValueError) as error:
        fail(f"[red]{detail(error)}[/red]", code=2, cause=error)
    for row in revision_rows(session):
        emit(json.dumps(row, ensure_ascii=False))


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
            # is an unregistered row. This catalog knows every name that it
            # holds, so it says which name it was and does not print an empty
            # table.
            fail(f"[red]no registered row named: {', '.join(missing)}[/red]", code=2)
    if as_json:
        emit(json.dumps(catalog, indent=2))
        return
    # **After the JSON return, and after `--row`.** The profile column is a thing
    # that the *table* prints. When the profile was composed above this point,
    # `--json` paid about 6 ms to build a map that it never reads. Worse, a dump
    # that depends on no profile could exit 2 because of one.
    #
    # `enabled_rows`, and not `dump()`. A dump keeps a row that a profile
    # switched off, with a flag. Reading the dump as mounted told a user that
    # `disabled: true` was "mounted, the default stands". This command promises
    # to tell that third state apart, and it got that state wrong.
    # `enabled_rows` is the predicate that `Profile.mount` itself uses.
    # `as_obj` is necessary. A mounted row that configures nothing has
    # `config is None`. That is a *different* answer from "not in this profile",
    # and the two must not share a sentinel: absence carries "not in this
    # profile". `as_obj` also settles the shape here, so the renderer below
    # reads presence alone.
    composed = {
        row.name: as_obj(row.config) for row in profile_or_exit(profile, patch).enabled_rows()
    }
    shown = [entry for entry in catalog if all_ or entry["config"] or entry.get("error")]
    if not shown:
        emit("no row matched")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("row")
    table.add_column("affects")
    table.add_column("option")
    table.add_column("type")
    table.add_column("default")
    table.add_column(f"in {profile}")
    table.add_column("what it does")
    for entry in shown:
        if error := entry.get("error"):
            table.add_row(entry["name"], "", "[red]unavailable[/red]", "", "", "", error)
            continue
        if not entry["config"]:
            table.add_row(entry["name"], entry["affects"], "[dim]none[/dim]", "", "", "", "")
            continue
        # `None`, and not `{}`. A row that this profile never mounts is a
        # different answer from a row that it mounts and says nothing about,
        # and one lookup carries both.
        set_here = composed.get(entry["name"])
        for index, field in enumerate(entry["config"]):
            first = index == 0
            table.add_row(
                entry["name"] if first else "",
                entry["affects"] if first else "",
                field["name"],
                field["type"],
                # A required option has no default. Printing one would invent a
                # value that a profile must actually supply.
                "[bold]required[/bold]" if field["required"] else (field["default"] or ""),
                _in_profile(field["name"], set_here, first=first),
                field["doc"],
            )
    console.print(table)


def _in_profile(name: str, set_here: JsonObject | None, *, first: bool) -> str:
    """Return what the composed profile says about one option.

    There are three answers, and they are different:

    - The row is not in this profile at all, so nothing here applies.
    - The row is mounted and says nothing about this option, so the default
      stands.
    - The profile set the option, and that value is what a run uses.

    `None` carries the first answer, so the caller keeps one lookup and not two
    facts to hold in step. After the first line of a row the first answer is
    blank, because it is the answer of the *row*, and repeating it for each
    option is noise.
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
