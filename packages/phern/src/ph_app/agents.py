"""`phern agents`: the client side of the daemon (P5-10).

The bare command and nine subcommands use one socket. `phern agents` with no
subcommand lists the roots. The subcommands are:

- `attach` follows one root.
- `send` sends a root a prompt.
- `schedule` lists, creates or cancels the schedules of a root.
- `status` asks what one root is doing.
- `doctor` asks the daemon what it is.
- `owed` lists the stored roots that still owe work.
- `resume` mounts stored roots that still owe work.
- `probe` asks the daemon one of the three orchestrator questions.
- `shutdown` stops the daemon.

Each one is a real exchange with a real supervisor. There is no local mode and
no in-process shortcut, because the purpose of this row is that a user can
reach a run that they are not attached to.

**One spine serves all of them.** `_ask` resolves the socket, connects, runs the
pump of the client and closes after itself. Each command is the body of one
exchange plus the code that prints the result. That keeps "no daemon is running"
one sentence and not one sentence for each command. It is also where the two
failures that a user meets are told apart. A socket that is *absent* means that
nothing was started. A socket that is *present but refuses* means that a daemon
crashed and left its path behind.

**`phern service` shares the spine.** `reach` is `_ask` without the exit. It
runs one exchange, or returns a `Down` with the sentence that tells an absent
socket from a dead daemon. `stop_daemon` is `shutdown` without the printing.
`phern service` imports both, and the two tables (`print_owed`, `print_resume`),
so the two clients cannot describe one daemon differently.

**Nothing here derives again what the daemon knows.** `doctor` reports the
socket that the daemon bound, the policy that it was started with, the jobs that
it is running and whether that socket outlives a logout. All of it is read back
over the wire. A client that printed what *it* would have chosen would print the
same thing for a daemon that was started differently, so it would say nothing
about that daemon. The one exception is when the connect itself fails and there
is no daemon to ask (P5-11).

@module ph_app.agents
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import anyio
import typer
from rich.table import Table

from ph.cancel import first_of
from ph.json import JsonObject, as_obj, as_str
from ph.lingering import lifetime
from ph.resources import SHUTDOWN_SECONDS
from ph.seams.schedule import ScheduleKind
from ph.selectors import Selector, matches_any
from ph.session import now_ms
from ph.text import duration, one_line

from . import verbs
from .console import (
    TypeOption,
    console,
    detail,
    fail,
    nonblank,
    roots_or_exit,
    section,
    selectors_or_exit,
)
from .daemon.client import DaemonClient, Exchange, connected
from .daemon.follow import EventFrame, Followed
from .params import (
    CancelScheduleParams,
    CreateScheduleParams,
    OwedParams,
    ProbeParams,
    ResumeParams,
)
from .payloads import DaemonStatusReply, OwedListing, ResumeReport, StatusFacts
from .protocol import (
    Cursor,
    DaemonError,
    DaemonGone,
    NoParams,
    ProbeName,
    SessionParams,
    cursor_text,
    parse_cursor,
)
from .wire import describe, message_of, result_block, text_of_wire

__all__ = ["Down", "agents_app", "print_owed", "print_resume", "reach", "stop_daemon"]

agents_app = typer.Typer(
    help=(
        "Talk to the supervisor: list, attach, send, schedule, status, doctor, owed, "
        "resume, probe, shutdown."
    ),
    add_completion=False,
)
"""`name` and `invoke_without_command` are set in the places where they take
effect: on `add_typer` in `cli.py` and on the callback below. Typer takes those
over anything that is passed here. A second copy here would look like
configuration and would have no effect."""


SESSION_ARGUMENT = typer.Argument(help="The session id of the root.")


# ------------------------------------------------------------------ the spine --


def _unreachable(path: Path, error: OSError) -> str:
    """Why the connect failed, stated as what the user does next.

    Three situations arrive as the same class of `OSError`, and each needs a
    different next step. No socket means that no daemon was ever started. A
    socket that nothing answers on means that a daemon died and left its path
    behind. `phern daemon` clears that path when it starts, and the message says
    so, which saves the reader from deleting a file by hand.

    The third situation is the one from P5-11, and it hides inside the first.
    Take a host where `$PH_RUNTIME` is under `$XDG_RUNTIME_DIR` and this user is
    not lingering. There, "no daemon socket" is also what a user sees after they
    log out and back in, with a daemon **still running**. That daemon still
    holds every session lease that it took (I-5), and it is about to refuse the
    `phern daemon` that this message would otherwise tell the user to start. To
    name this case costs one paragraph and one `stat`. Not to name it costs an
    afternoon on `session_already_active`.
    """
    if not path.exists():
        return f"[red]no daemon socket at {path}[/red]\n{_missing_socket(path)}"
    return (
        f"[red]{path} exists but nothing is listening[/red] ({detail(error)})\n"
        "a daemon crashed and left its socket; [bold]phern daemon[/bold] clears it on start"
    )


def _missing_socket(path: Path) -> str:
    """What to do about an absent socket. The answer depends on whether the
    socket was reaped.

    The linger state is read on this side because there is nothing on the other
    side to ask: the daemon that would have answered is exactly the one that
    cannot be reached. This is the single exception to the rule of this module
    that it never derives again what the daemon knows. The rule was never about
    this case.
    """
    life = lifetime(path)
    if life.survives_logout is not False:
        return "start one with [bold]phern daemon[/bold]"
    return (
        f"[yellow]{life.explanation}[/yellow]\n"
        "if a daemon was running before you logged out it may still be running, "
        "holding this user's session leases — check with "
        f"[bold]pgrep -u {life.user} -f 'phern daemon'[/bold] before starting a second one\n"
        f"[bold]{life.advice}[/bold] makes the next one outlive logout"
    )


@dataclass(frozen=True, slots=True)
class Down:
    """Nothing answered on the socket of the daemon. `why` is the sentence that a
    user reads: an absent socket, the leftover path of a dead daemon, or a
    connection that closed without an answer. `cause` is the error that the
    sentence was read from."""

    why: str
    cause: BaseException


async def reach[T](path: Path, work: Exchange[T]) -> T | Down:
    """Run one exchange against the daemon at `path`, or report that nothing
    answered.

    `Down` is a value and not a refusal, so that each client decides what it
    means. `phern agents` exits with it (`_ask`). `phern service` reports it or
    starts a daemon. A daemon that answered "no" is a refusal in both clients,
    with the sentence of the server, because only the server knows why.
    """
    try:
        return await connected(path, work)
    # These are the plain exceptions and not groups, because `connected` unwraps
    # its own groups. Otherwise a `typer.Exit` that the exchange of a command
    # raised would reach the shell as an `ExceptionGroup` traceback and not as an
    # exit code. `DaemonGone` comes first: it is a `DaemonError`, and nobody
    # refused anything.
    except DaemonGone as gone:
        return Down(f"[yellow]{detail(gone)}[/yellow]", gone)
    except DaemonError as refused:
        fail(f"[red]the daemon refused:[/red] {detail(refused)}", cause=refused)
    except OSError as failed:
        return Down(_unreachable(path, failed), failed)


def _ask[T](work: Exchange[T]) -> T:
    """Run one exchange against the daemon, or exit with a reason.

    Every refusal that a user can meet is named here, so that none reaches them
    as a traceback: an unusable `$PH_RUNTIME`, a daemon that is not there, and a
    daemon that answered "no". The last one keeps the sentence of the server,
    because only the server knows why.
    """
    outcome = anyio.run(partial(reach, roots_or_exit().daemon_socket(), work))
    if isinstance(outcome, Down):
        fail(outcome.why, cause=outcome.cause)
    return outcome


# ------------------------------------------------------------------ rendering --


def _when(moment: int | None) -> str:
    """An epoch-ms instant as local time, or a dash when `moment` is `None`."""
    if moment is None:
        return "—"
    return datetime.fromtimestamp(moment / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _starts_on(facts: DaemonStatusReply) -> str:
    """`main · fake/fake-1`: the key a new root starts on, and its route."""
    starts = facts.starts_on
    if starts is None:
        return "none — the profile lists no models"
    if starts.key is None:
        return f"{starts.route.label} (not in the profile's list)"
    return f"{starts.key} · {starts.route.label}"


def _scheduler(facts: DaemonStatusReply) -> str:
    """When the scheduler next wakes. It sleeps until then and does not tick."""
    if facts.next_wake is None:
        return "idle — nothing scheduled"
    return f"next wake in {duration(max(0, facts.next_wake - now_ms()))}"


def _release(facts: DaemonStatusReply) -> str:
    """When the next quiet root is released. Release sleeps until then, too."""
    if facts.passivate_after is None:
        return "off"
    if facts.next_release is None:
        return "nothing quiet enough to plan for"
    return f"next in {duration(max(0, facts.next_release - now_ms()))}"


def _summary(kind: str, event: JsonObject) -> str:
    """One line that says what an event says.

    This is deliberately a *follower* and not the transcript. `phern agents
    attach` shows the log as it arrives. The two projections that render a
    conversation properly (`tui.adapter` and `tui.trajectory`) both fold whole
    `Session` objects, and a stream of wire frames is not one. So the four types
    that carry a conversation are written out here, and **every other type falls
    through to `describe`** and not to nothing. `describe` is the generic
    reading of the auditor, and it is shared for the reason that it states: a
    view that silently hid an event for which it had no phrase would be the
    omission that A11 forbids. When only four types had a phrase, 57 of the 61
    known types were shown as a bare word. Those included every `supervisor/*`
    record that says why a root stopped, and that is what a user follows a
    remote run to find out.
    """
    data = as_obj(event.get("data"))
    if kind in ("user/message", "assistant/message"):
        return one_line(text_of_wire(message_of(data).get("content")))
    if kind == "tool/call":
        return f"{data.get('name') or '?'} {one_line(as_str(data.get('arguments')), 60)}"
    if kind == "tool/result":
        return one_line(text_of_wire(result_block(message_of(data)).get("content")))
    return describe(data)


def _reachability(facts: DaemonStatusReply) -> list[tuple[str, str]]:
    """The row that exists only when the answer is bad (P5-11).

    The row is absent while the daemon can be reached. That is the rule of
    `Diagnostic.read`, and it is worth following here. A user who can read this
    table at all has just connected, so a permanent "reachable: yes" row would
    be a fact that its own delivery proves. The row that matters would then be
    one more line of a table that nobody scans.

    The case in which the row can be *seen* is the unusual one and the useful
    one. The socket went away, the daemon noticed, and this connection came in
    on a client that was already attached, or through a path that somebody
    restored by hand.
    """
    since = facts.unreachable_since
    if since is None:
        return []
    return [
        ("reachable", f"[red]no[/red] — the socket stopped being this daemon's at {_when(since)}")
    ]


def _line(event: JsonObject) -> str:
    kind = as_str(event.get("type"))
    body = _summary(kind, event)
    seq = f"{event.get('seq', '')!s:>5}"
    if not body:
        return f"[dim]{seq}  {kind}[/dim]"
    return f"[dim]{seq}[/dim]  [bold]{kind}[/bold]  {body}"


# --------------------------------------------------------------------- follow --


NOISE: frozenset[str] = frozenset(
    {
        # The streamed halves of `assistant/message`, which *is* shown. One line
        # for each delta is a keystroke log and not a follow. Here it is also
        # almost all of the rendering cost, for output that nobody reads.
        "assistant/chunk",
        # The opening half of the dispatch. Its settled half carries the content.
        "tool/code-dispatch-start",
        # One event per changed variable per cell.
        "kernel/snapshot",
    }
)
"""The types that a follow leaves out unless it is asked. `--all` includes them.

This is deliberately *not* the `RECORDLESS` set of the trajectory. That set
belongs to that view, for the reasons of that view, and it is under
`ph_app.tui`. This set is small and open on purpose. A type that is not
classified is shown, because a follower that hid something new would be the
silent omission that the fallback in `_summary` exists to prevent.
"""


def _since_cursor(raw: str, current: Cursor) -> Cursor:
    """`--since` as a cursor that the daemon can check, or exit 2.

    The parse is `ph_app.protocol.parse_cursor`. It is beside `cursor_of` and
    `resume_at`, which define what a cursor is. The part that belongs to the CLI
    is the refusal and its code.
    """
    cursor = parse_cursor(raw, current.to_wire())
    if cursor is None:
        fail(
            f"[red]--since {raw!r} is not a cursor; use SEQ or GENERATION:SEQ as "
            "`phern agents status` prints it[/red]",
            code=2,
        )
    return cursor


@dataclass(slots=True)
class _Follow:
    """One attached session as the CLI prints it, built on `Followed`.

    The buffering, the `seq` dedupe and the paging are in `daemon/follow.py`,
    which the socket client of the TUI is built on too. This class holds what
    belongs to the CLI: a console, the `--type` filter, and the `until_idle`
    stop.
    """

    session_id: str
    until_idle: bool = False
    everything: bool = False
    selectors: Sequence[Selector] = ()
    """Namespace selectors from `--type`. Empty means no filter (P6-33)."""
    last_turn: str | None = None
    """How the turn that this follower stopped on ended, so that the exit code
    can say so (P5-04).

    It is kept because the status that *releases* `--until-idle` is the only
    status that decides the exit code, and that status is gone by the time the
    wait returns. It is set in `_status` and nowhere else. That includes a root
    that was already idle when the attach arrived: its only status is the attach
    reply, which `Followed.seed` delivers through the same handler.
    """
    done: anyio.Event = field(default_factory=anyio.Event)
    feed: Followed = field(init=False)

    def __post_init__(self) -> None:
        self.feed = Followed(
            session_id=self.session_id, on_events=self._events, on_status=self._status
        )

    def _events(self, pairs: Sequence[EventFrame], _live: bool) -> None:
        # A follower does not draw the distinction between the phases. It prints
        # records, and a record that was read from a page reads the same as a
        # record that just arrived.
        self.write(event for event, _view in pairs)

    def _status(self, facts: StatusFacts) -> None:
        status, last = facts.status, facts.last_turn
        if status is None:
            # This frame says nothing about where the root is: only its footer
            # or its route moved. A follower prints the log and the status
            # words, so there is no line to print. It used to print a bare `·`.
            return
        # The ending is named when it is not the ordinary one. `idle` after an
        # error turn reads as success, and this line is what `--until-idle` stops
        # on.
        ended = f" · last turn {last}" if status == "idle" and last and last != "completed" else ""
        console.print(f"[dim]· {status}{ended}[/dim]")
        if self.until_idle and status == "idle":
            self.last_turn = last
            self.done.set()

    def write(self, events: Iterable[JsonObject]) -> None:
        """Render a run of events as one write.

        One `console.print` for each event costs several times as much as one
        print of the same events joined. The single print makes catching up on a
        long log bearable, and it is why this method takes a sequence and not
        one event.
        """
        lines = [_line(event) for event in events if self._shows(event)]
        if lines:
            # The lines of a follower are records. A line folded at the terminal
            # width turns a `grep` into two half-matches, which is why the
            # console itself is `soft_wrap` (see `ph_app.console`).
            console.print("\n".join(lines))

    def _shows(self, event: JsonObject) -> bool:
        """Whether this follower prints one event.

        **A `--type` selector replaces the default that hides per-delta events.
        It is not applied on top of that default.** `NOISE` exists because a
        turn is mostly `assistant/chunk`, whose content arrives again in the
        `assistant/message` that closes the turn. So the default is to hide it.
        A namespace that the user names is a stronger signal than that default.
        A user who typed `--type assistant` asked for the assistant traffic, and
        hiding most of it would answer a question that they did not ask. So
        `--all` is not needed when `--type` is given, and is not required on
        top of it.

        This is written as two branches and not as one disjunction, and the type
        is read one time. The earlier form computed `event["type"]` here and
        again in a second predicate. It also computed `bool(self.selectors)`
        again on every frame of a replayed history, although that value is
        fixed at construction.
        """
        kind = as_str(event.get("type"))
        if self.selectors:
            return matches_any(kind, self.selectors)
        return self.everything or kind not in NOISE


# ------------------------------------------------------------------- commands --


@agents_app.callback(invoke_without_command=True)
def agents(ctx: typer.Context) -> None:
    """List the roots this daemon is running."""
    if ctx.invoked_subcommand is not None:
        return
    listed = _ask(lambda client: client.call(verbs.SESSIONS_LIST, NoParams()))
    rows = listed.sessions
    if not rows:
        console.print("[dim]no roots running[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("session")
    table.add_column("status")
    table.add_column("last turn")
    table.add_column("events", justify="right")
    table.add_column("watchers", justify="right")
    for row in rows:
        table.add_row(
            row.session_id,
            row.status,
            str(row.last_turn or ""),
            str(row.cursor.sequence),
            str(row.watchers),
        )
    console.print(table)


@agents_app.command()
def send(
    session: Annotated[str, SESSION_ARGUMENT],
    prompt: Annotated[str, typer.Argument(help="What to say to the agent.")],
) -> None:
    """Queue a turn on a root, starting or resuming it if it is not running.

    Returns as soon as the prompt is *logged*, which is the protocol's own
    contract: the root is a task the daemon owns, and "it is working" is
    something to watch with `phern agents attach`, not something to hold a socket
    open for. The client mints its own idempotence key, so a resend after a
    dropped connection is safe rather than a second turn.
    """
    root = _ask(lambda client: client.prompt(session, prompt))
    console.print(f"[dim]queued on {root.session_id} · {root.status}[/dim]")


@agents_app.command()
def attach(
    session: Annotated[str, SESSION_ARGUMENT],
    since: Annotated[
        str,
        typer.Option(
            "--since",
            help="Resume from a cursor: SEQ for this log, or GENERATION:SEQ as `status` prints it.",
        ),
    ] = "0",
    until_idle: Annotated[
        bool,
        typer.Option(
            "--until-idle",
            help="Stop once the root goes idle; exit 1 if its last turn errored.",
        ),
    ] = False,
    everything: Annotated[
        bool, typer.Option("--all", help="Include streamed chunks and other per-delta events.")
    ] = False,
    type_: TypeOption = [],  # noqa: B006 - typer builds the list per invocation
) -> None:
    """Follow a root's log: its history, then everything as it happens.

    Attaching does not start or stop the work — that is the whole of P5-01's
    gate — so leaving is free, and a root keeps going with nobody watching. It
    *will* start a passivated root, because attaching to a session that was
    released for being quiet should show it to you rather than report it gone.

    Per-delta events are left out unless `--all`: a turn is mostly
    `assistant/chunk`, whose content arrives again as the `assistant/message`
    that closes it, so showing both is a keystroke log wrapped around the thing
    a person came to read.

    **`--since` resumes a reader that already has part of the log.** A cursor is
    `{generation, sequence}`, and a sequence means something only against the log
    that counted it. The full form, `GENERATION:SEQ`, is what `phern agents status`
    prints and what a script should pass back — the daemon checks the generation
    and, for another incarnation of the log, pages from 0 rather than skipping
    events this reader never saw. The bare `SEQ` form is for a person typing at a
    session they were just watching: it is stamped with the *current* generation
    and so cannot be checked. Either way the command says where the history
    actually started when that is not where it was asked to.

    **`--until-idle` exits 1 when the turn it stopped on ended in an error.**
    Idle is how a root that answered and a root whose last answer failed both
    look, so a script chaining on this had no way to tell them apart and read the
    second as success (P5-04). Only `error` — the kind the loop records when a
    turn could not be completed. A turn a person canceled, or one a limit ended,
    exits 0 and says so on its own line: those are outcomes somebody already
    knows about, and turning them into a failed command would make the ordinary
    way of stopping an agent look like a fault.
    """

    # This is the session-log vocabulary. A bare `workspace` needs no prefix, and
    # `bus:tools` is refused and does not silently match nothing (P6-33).
    selectors = selectors_or_exit(type_, vocabulary="log")

    async def work(client: DaemonClient) -> None:
        follow = _Follow(
            session_id=session,
            until_idle=until_idle,
            everything=everything,
            selectors=selectors,
        )
        # The callback is set on the peer, which is where the read loop looks.
        client.peer.on_notify = follow.feed
        # The subscription is made *before* the history is read, so that nothing
        # that happens in between is lost. The feed of `_Follow` holds those
        # frames until the pages are done. No cursor is sent. This reply carries
        # the generation that a snapshot cursor needs, so `from` is 0 here by
        # construction, and catch-up is paged from `--since` against the
        # generation that the daemon just named.
        attached = await client.call(verbs.SESSION_ATTACH, SessionParams(session_id=session))
        cursor = _since_cursor(since, attached.cursor)
        # The reply *is* a status frame. For a root that was already idle it is
        # the only status frame that there will ever be, so it goes through the
        # feed and is not read here. `_status` is then the single place that
        # decides "idle means done", and this command has no `--until-idle`
        # branch of its own.
        follow.feed.seed(attached)
        try:
            # `seen` is *not* set in advance to `asked - 1`. It used to be, so
            # that live frames below the cursor would be dropped. But nothing
            # live can arrive below the head, and the advance setting had a real
            # cost. When the daemon answered a stale generation by paging from
            # 0, the first `asked` events of *this* incarnation were discarded
            # as already seen. `seen` now advances from what arrives, and that
            # shows the fallback.
            started = await follow.feed.catch_up(client, cursor)
            if started < cursor.sequence:
                console.print(
                    f"[yellow]history starts at {started}, not {cursor.sequence}: --since "
                    "named a different incarnation of this log, so nothing was "
                    "skipped[/yellow]"
                )
            follow.feed.live()
            await first_of(follow.done, client.closed)
        finally:
            # Detach only while there is somebody to tell. A daemon that shut
            # down during the follow has already dropped every subscription, and
            # the call would wait for a reply that nobody is left to send.
            if not client.closed.is_set():
                await client.call(verbs.SESSION_DETACH, SessionParams(session_id=session))
        if client.closed.is_set():
            raise DaemonGone
        if follow.last_turn == "error":
            # This runs after the detach above, so that the refusal does not
            # leave a subscription behind on a daemon that is still running.
            fail(f"[red]{session}: the last turn ended in an error[/red]")

    _ask(work)


@agents_app.command()
def schedule(
    session: Annotated[str, SESSION_ARGUMENT],
    prompt: Annotated[
        str | None, typer.Option("--prompt", help="What to say to the agent when it fires.")
    ] = None,
    at: Annotated[
        int | None, typer.Option("--at", help="Fire once, at this epoch-ms instant.")
    ] = None,
    every: Annotated[int | None, typer.Option("--every", help="Fire every N milliseconds.")] = None,
    cron: Annotated[
        str | None,
        typer.Option("--cron", help="Fire on this cron expression.", callback=nonblank),
    ] = None,
    cancel: Annotated[
        str | None, typer.Option("--cancel", help="Cancel this schedule id.", callback=nonblank)
    ] = None,
    schedule_id: Annotated[
        str | None, typer.Option("--id", help="Name the schedule.", callback=nonblank)
    ] = None,
) -> None:
    """List, create, or cancel a root's schedules.

    The timing flag chooses the kind, rather than a `--kind`/`--spec` pair that
    would put the wire's shape in front of the person using it — `--every` is an
    interval, `--cron` an expression, `--at` a single instant. Exactly one, and
    a prompt, because a schedule with nothing to say is claimed, recorded and
    delivers nothing forever.
    """
    # The chosen timing is carried as the pair that it becomes, and not as three
    # bools whose identity a ternary chain must reconstruct further down. The
    # dict is keyed by the literal, so the comprehension keeps the literal. A
    # bare tuple of pairs widens the three to `str`, and `schedule/create` takes
    # the `ScheduleKind` that the seam declares. Without the annotation on the
    # *source*, the CLI hands the typed send a value that the model of the
    # daemon would then refuse.
    #
    # A flag that was left out is `None`. A typed `0` is a value, and it goes to
    # the daemon like any other: the fold of the schedule seam is the one rule
    # for what can fire, so `--every 0` is refused there and `--at 0` fires at
    # once. It is never read as a flag that was left out: that reading made
    # `--every 0 --prompt x` list the schedules and report nothing. An empty
    # string is refused at parse time (`nonblank`).
    offered: dict[ScheduleKind, int | str | None] = {"once": at, "interval": every, "cron": cron}
    timings = [(kind, str(value)) for kind, value in offered.items() if value is not None]
    if cancel is not None:
        if timings:
            raise typer.BadParameter("--cancel takes no timing flag")
        outcome = _ask(
            lambda client: client.call(
                verbs.SCHEDULE_CANCEL,
                CancelScheduleParams(session_id=session, schedule_id=cancel),
            )
        )
        if not outcome.canceled:
            fail(f"[red]no schedule {cancel!r} on {session}[/red]")
        console.print(f"[dim]canceled {cancel}[/dim]")
        return

    if len(timings) > 1:
        raise typer.BadParameter("one of --at, --every or --cron, not several")
    if not timings:
        listed = _ask(
            lambda client: client.call(verbs.SCHEDULE_LIST, SessionParams(session_id=session))
        )
        _print_schedules(session, listed.schedules)
        return
    if not prompt:
        raise typer.BadParameter("a schedule needs --prompt: what to say when it fires")

    ((kind, spec),) = timings
    created = _ask(
        lambda client: client.call(
            verbs.SCHEDULE_CREATE,
            CreateScheduleParams(
                session_id=session,
                schedule_id=(
                    schedule_id if schedule_id is not None else f"sch-{secrets.token_hex(4)}"
                ),
                kind=kind,
                spec=spec,
                prompt=prompt,
            ),
        )
    )
    console.print(f"[dim]scheduled {created.id} · {kind} {spec}[/dim]")


def _print_schedules(session: str, rows: list[dict[str, Any]]) -> None:
    """Print the schedules of a root. The argument is the rows themselves, and
    not a reply to index into.

    The function used to take a whole `schedule/list` envelope and read
    `["schedules"]` out of it. That worked only because the `session/status`
    reply happens to use the same key. A string coupled the printer to two
    unrelated wire shapes.
    """
    if not rows:
        console.print(f"[dim]no schedules on {session}[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("id")
    table.add_column("kind")
    table.add_column("spec")
    table.add_column("next")
    table.add_column("prompt")
    for row in rows:
        # `nextAt` is absent or `None` for a schedule that will not fire again.
        # The row is unvalidated JSON, so anything that is not an instant is
        # read as absent too.
        next_at = row.get("nextAt")
        table.add_row(
            row["id"],
            row["kind"],
            row["spec"],
            _when(next_at if isinstance(next_at, int) else None),
            one_line(row["prompt"], 40),
        )
    console.print(table)


@agents_app.command()
def status(session: Annotated[str, SESSION_ARGUMENT]) -> None:
    """What one root is doing, and what the log says about how it got there."""
    row = _ask(lambda client: client.call(verbs.SESSION_STATUS, SessionParams(session_id=session)))
    console.print(
        section(
            f"root {row.session_id}",
            (
                ("status", row.status),
                ("events", str(row.cursor.sequence)),
                ("generation", str(row.cursor.generation)),
                # The two values above, in the one form that `attach --since` can
                # verify. The module that defines what a cursor is formats it.
                ("resume with", f"--since {cursor_text(row.cursor)}"),
                ("watchers", str(row.watchers)),
                ("retry attempts", str(row.attempts)),
                ("given up", "yes" if row.failed else "no"),
                ("schedules", str(len(row.schedules))),
            ),
        )
    )
    if row.schedules:
        _print_schedules(session, row.schedules)


@agents_app.command()
def doctor() -> None:
    """Ask the daemon what it is: where it listens, and how it was started.

    The client half of `phern doctor`. Everything printed comes back over the wire
    from the running process, so what this reports is what is *in force* rather
    than what this invocation's flags and environment would have produced.
    """
    facts = _ask(lambda client: client.call(verbs.DAEMON_STATUS, NoParams()))
    passivate = facts.passivate_after
    console.print(
        section(
            "pH daemon",
            (
                ("socket", facts.socket),
                ("pid", str(facts.pid)),
                ("uptime", duration(facts.uptime_ms)),
                ("protocol", str(facts.protocol_version)),
                ("capabilities", ", ".join(sorted(facts.capabilities))),
                ("roots", str(facts.roots)),
                ("model", _starts_on(facts)),
                ("passivate after", "off" if passivate is None else duration(passivate * 1000)),
                ("scheduler", _scheduler(facts)),
                ("release", _release(facts)),
                ("socket watch", facts.socket_watch),
                ("invariants", "checked when a root settles" if facts.check_invariants else "off"),
                (
                    "probes",
                    "socket only (`phern agents probe`)" if facts.probe is None else facts.probe,
                ),
                ("unwritten", ", ".join(facts.unwritten) or "none"),
                *_reachability(facts),
            ),
        )
    )
    # This is a loop because the envelope is a list. P5-12 adds the mounted
    # diagnostics of the daemon to the list, and this side needs no change to
    # render them.
    for reported in facts.sections:
        console.print(section(reported.title, [(row.label, row.value) for row in reported.rows]))


@agents_app.command()
def owed(
    rebuild: Annotated[
        bool,
        typer.Option(
            "--rebuild", help="Read every stored root's log instead of trusting the index."
        ),
    ] = False,
    check: Annotated[
        bool,
        typer.Option("--check", help="Exit 2 when any owed root is not mounted."),
    ] = False,
) -> None:
    """Which stored roots still owe work, and why (P13-02).

    What a resumer asks before it asks for anything to be mounted: a root killed
    mid-turn, one with a sub-agent still working in its own log, one with a goal
    nobody settled, one with a prompt no turn claimed. From the daemon's live-roots
    index, rebuilt from the logs when it cannot vouch for itself or `--rebuild` says to.
    """
    listing = _ask(lambda client: client.call(verbs.SESSIONS_OWED, OwedParams(rebuild=rebuild)))
    print_owed(listing)
    if check and any(not row.mounted for row in listing.sessions):
        raise typer.Exit(code=2)


def print_owed(listing: OwedListing) -> None:
    """The owed table and where it came from, as `phern agents owed` and `phern
    service status` both print it."""
    if not listing.sessions:
        console.print("[dim]no root owes work[/dim]")
    else:
        table = Table(show_header=True, header_style="bold")
        table.add_column("session")
        table.add_column("status")
        table.add_column("holds")
        table.add_column("since")
        table.add_column("cwd")
        for row in listing.sessions:
            status = row.status + (" · failed" if row.failed else "")
            if row.damaged:
                status += f" · damaged ({', '.join(row.damaged)})"
            table.add_row(
                row.session_id,
                status,
                ", ".join(row.holds) or "—",
                _when(row.updated),
                row.cwd,
            )
        console.print(table)
        if any(row.damaged for row in listing.sessions):
            console.print(
                "[yellow]damaged: a log beneath the root will not read; "
                "`phern doctor sessions --repair` cuts it back to what does[/yellow]"
            )
    note = (
        "rebuilt from the logs"
        if listing.rebuilt
        else ("index" if listing.trusted else "index, unvouched")
    )
    console.print(f"[dim]source: {note}[/dim]")


@agents_app.command()
def resume(
    sessions: Annotated[
        list[str] | None,
        typer.Argument(help="Roots to mount again; with --all, every owed root as well."),
    ] = None,
    all_owed: Annotated[
        bool,
        typer.Option("--all", help="Every root the live-roots index says owes work."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Mount a root whose ladder is spent as well."),
    ] = False,
) -> None:
    """Mount stored roots that still owe work, so each carries on without a client
    (P13-03).

    What an orchestrator runs after the daemon is up — a unit's `ExecStartPost`, a
    person after a reboot — and the whole of a resume is the mount: the log is
    repaired, the children are readmitted, a turn the crash cut short is continued
    (P13-04), and a prompt nobody claimed is rung. One row per root says what became
    of it; exit 1 if any was refused, so a unit shows the pass did not fully land.
    """
    named = list(sessions or [])
    if not named and not all_owed:
        fail("[red]name a session, or pass --all[/red]", code=2)
    report = _ask(
        lambda client: client.call(
            verbs.SESSIONS_RESUME, ResumeParams(sessions=named, all_owed=all_owed, force=force)
        )
    )
    print_resume(report)
    if any(row.outcome == "refused" for row in report.sessions):
        raise typer.Exit(code=1)


def print_resume(report: ResumeReport) -> None:
    """One row per root a resume pass was asked about, as `phern agents resume` and
    `phern service` both print it."""
    if not report.sessions:
        console.print("[dim]no root owes work[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("session")
    table.add_column("outcome")
    table.add_column("status")
    table.add_column("detail")
    for row in report.sessions:
        table.add_row(
            row.session_id,
            row.outcome,
            "—" if row.status is None else row.status,
            row.detail,
        )
    console.print(table)
    if not report.trusted:
        console.print("[yellow]the live-roots index could not vouch for itself[/yellow]")


@agents_app.command()
def probe(name: Annotated[ProbeName, typer.Argument(help="startup, ready or live.")]) -> None:
    """Ask the daemon one of the three orchestrator questions; exit 0 for yes (P13-12).

    The unix-socket door to the same answers `--probe-bind` serves over HTTP, for a
    systemd `ExecStartPost`, a Kubernetes `exec` probe, or a person. A daemon that
    is not there at all is a no for every probe, with the usual reason on stderr.
    """
    reply = _ask(lambda client: client.call(verbs.DAEMON_PROBE, ProbeParams(name=name)))
    if not reply.ok:
        fail("\n".join(f"[red]{name}: {reason}[/red]" for reason in reply.reasons))
    console.print(f"[green]{name}: yes[/green]")


@agents_app.command()
def shutdown() -> None:
    """Stop the daemon, and wait for it to actually be gone (`stop_daemon`)."""
    if not _ask(stop_daemon):
        fail("[yellow]shutdown sent; the daemon has not stopped yet[/yellow]")
    console.print("[dim]daemon stopped[/dim]")


async def stop_daemon(client: DaemonClient) -> bool:
    """Send `shutdown` and wait for the daemon to close the connection. Returns
    whether the daemon closed it within `SHUTDOWN_SECONDS`.

    `shutdown` is a notification by contract. A request that awaited a reply
    would wait on a frame that the daemon is losing the ability to write at the
    same time. So the confirmation is the connection that the daemon closes as
    it exits. To wait for that, and not to declare success at once, is the
    difference between "I asked" and "it stopped". The daemon flushes its roots
    and releases its leases in exactly that window. The daemon removes its
    socket before its connections close (the teardown of `serve`), so a closed
    connection also means that the next daemon finds the socket path clear.

    The budget is the bound of the teardown itself, plus room for the unwinding
    around it. A daemon that is still inside its grace period has not failed to
    stop. It is stopping. Both halves are declared where they are spent, so
    this budget cannot drift from what the daemon allows.
    """
    await client.notify(verbs.SHUTDOWN, NoParams())
    with anyio.move_on_after(SHUTDOWN_SECONDS):
        await client.closed.wait()
    return client.closed.is_set()
