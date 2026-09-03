"""`ph agents` — the client side of the daemon (P5-10).

Seven commands over one socket: list the roots, `attach` to one and follow it,
`send` it a prompt, `schedule` future work, ask a root's `status`, ask the
daemon's own `doctor`, and `shutdown`. Every one is a real exchange with a real
supervisor — no local mode, no in-process shortcut, because what this row
delivers is precisely that a person can reach a run they are not attached to.

**One spine, seven commands.** `_ask` resolves the socket, connects, runs the
client's pump and closes behind itself; each command is the body of one exchange
plus how to print it. That keeps "no daemon is running" one sentence rather than
seven, and it is where the two failures a person actually meets are told apart: a
socket that is *absent* means nothing was started, and a socket that is *present
but refuses* means something crashed and left its path behind.

**Nothing here re-derives what the daemon knows.** `doctor` reports the socket
the daemon bound, the policy it was started with, the cadences it is running and
whether that socket outlives a logout — all read back over the wire, because a
client printing what *it* would have chosen would agree with a daemon started
differently and say nothing at all. The one exception is when the connect itself
fails and there is no daemon to ask (P5-11).

@module ph_app.agents
"""

from __future__ import annotations

import secrets
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import anyio
import typer
from rich.table import Table

from ph.lingering import lifetime
from ph.paths import RuntimeDirError, resolve_roots
from ph.resources import GRACE_SECONDS
from ph.selectors import Selector, matches_any

from .console import TypeOption, console, fail, section, selectors_or_exit
from .daemon.client import DaemonClient, Exchange, connected
from .daemon.follow import Followed, first_of
from .protocol import DaemonError, DaemonGone
from .wire import describe, message_of, obj, one_line, result_block, seq, text_of_wire

__all__ = ["agents_app"]

agents_app = typer.Typer(
    help="Talk to the supervisor: list, attach, send, schedule, status, doctor, shutdown.",
    add_completion=False,
)
"""`name` and `invoke_without_command` are set where they win — on `add_typer`
in `cli.py` and on the callback below — and Typer takes those over anything
passed here, so spelling them twice reads as configuration and is not."""


SESSION_ARGUMENT = typer.Argument(help="The session id of the root.")


# ------------------------------------------------------------------ the spine --


def _unreachable(path: Path, error: OSError) -> str:
    """Why the connect failed, in the terms of the fix.

    Three situations arrive as the same class of `OSError`, and they want
    different next steps. No socket means no daemon was ever started. A socket
    nothing answers on means one died and left its path behind — which
    `ph daemon` clears on its way up, so saying so saves the reader from
    deleting a file by hand.

    The third is P5-11's, and it hides inside the first: on a host where
    `$PH_RUNTIME` lives under `$XDG_RUNTIME_DIR` and this user is not lingering,
    "no daemon socket" is also what a person sees after logging out and back in
    — with a daemon **still running**, still holding every session lease it took
    (I-5), and about to refuse the `ph daemon` this message would otherwise tell
    them to start. Naming it costs one paragraph and one `stat`; not naming it
    costs an afternoon on `session_already_active`.
    """
    if not path.exists():
        return f"[red]no daemon socket at {path}[/red]\n{_missing_socket(path)}"
    return (
        f"[red]{path} exists but nothing is listening[/red] ({error})\n"
        "a daemon crashed and left its socket; [bold]ph daemon[/bold] clears it on start"
    )


def _missing_socket(path: Path) -> str:
    """What to do about an absent socket — which depends on whether it was reaped.

    The linger state is read on this side because there is nothing on the other
    side to ask: the daemon that would have answered is precisely the one that
    cannot be reached. That is the single exception to this module's rule of
    never re-deriving what the daemon knows, and it is the case the rule was
    never about.
    """
    life = lifetime(path)
    if life.survives_logout is not False:
        return "start one with [bold]ph daemon[/bold]"
    return (
        f"[yellow]{life.explanation}[/yellow]\n"
        "if a daemon was running before you logged out it may still be running, "
        "holding this user's session leases — check with "
        f"[bold]pgrep -u {life.user} -f 'ph daemon'[/bold] before starting a second one\n"
        f"[bold]{life.advice}[/bold] makes the next one outlive logout"
    )


def _ask[T](work: Exchange[T]) -> T:
    """Run one exchange against the daemon, or exit with a reason.

    Every refusal a person can actually hit is named here rather than reaching
    them as a traceback: an unusable `$PH_RUNTIME`, a daemon that is not there,
    and a daemon that answered "no". The last one keeps the server's own
    sentence, because the server is the only thing that knows why.
    """
    try:
        path = resolve_roots().daemon_socket()
    except RuntimeDirError as error:
        fail(f"[red]$PH_RUNTIME check failed:[/red] {error}", cause=error)
    try:
        return anyio.run(partial(connected, path, work))
    # Not `error` again: Python unbinds an `except … as` name at the end of its
    # block, so reusing it here is a read of a deleted variable. And these are
    # the plain exceptions rather than groups because `connected` unwraps its
    # own — a `typer.Exit` raised by a command's own exchange would otherwise
    # reach the shell as an `ExceptionGroup` traceback instead of an exit code.
    except DaemonGone as gone:
        fail(f"[yellow]{gone}[/yellow]", cause=gone)
    except DaemonError as refused:
        fail(f"[red]the daemon refused:[/red] {refused}", cause=refused)
    except OSError as unreachable:
        fail(_unreachable(path, unreachable), cause=unreachable)


# ------------------------------------------------------------------ rendering --


def _when(moment: Any) -> str:
    """An epoch-ms instant as local time, or a dash when there is none."""
    if not isinstance(moment, int) or moment <= 0:
        return "—"
    return datetime.fromtimestamp(moment / 1000).strftime("%Y-%m-%d %H:%M:%S")


def _duration(milliseconds: Any) -> str:
    """`3d 4h`, `1h 30m`, `12m`, `8s` — the two largest units that are not zero.

    Zeros are dropped rather than kept for shape: a sweep every sixty seconds
    reads as `1m`, not `1m 0s`, and an hour and a half of quiet is `1h 30m`
    whether or not the seconds happen to be round.
    """
    if not isinstance(milliseconds, int | float):
        return "—"
    seconds = int(milliseconds // 1000)
    parts: list[str] = []
    for name, size in (("d", 86400), ("h", 3600), ("m", 60), ("s", 1)):
        count, seconds = divmod(seconds, size)
        if count:
            parts.append(f"{count}{name}")
    return " ".join(parts[:2]) or "0s"


def _summary(kind: str, event: Mapping[str, Any]) -> str:
    """One line of what an event says.

    Deliberately a *follower*, not the transcript: `ph agents attach` shows the
    log arriving, and the two projections that render a conversation properly
    (`tui.adapter` and `tui.trajectory`) both fold whole `Session` objects,
    which a stream of wire frames is not. So the four types that carry a
    conversation are spelled out, and **everything else falls through to
    `describe`** rather than to nothing — the auditor's generic reading, shared
    for its stated reason: a view that silently hid an event it had no phrase
    for would be the omission A11 forbids. Naming only four types left 57 of the
    61 known ones as a bare word, including every `supervisor/*` record that
    says why a root stopped — which is the thing a person follows a remote run
    to find out.
    """
    data = obj(event.get("data"))
    if kind in ("user/message", "assistant/message"):
        return one_line(text_of_wire(message_of(data).get("content")))
    if kind == "tool/call":
        return f"{data.get('name') or '?'} {one_line(str(data.get('arguments') or ''), 60)}"
    if kind == "tool/result":
        return one_line(text_of_wire(result_block(message_of(data)).get("content")))
    return describe(data)


def _sections(reported: Any) -> list[tuple[str, list[tuple[str, str]]]]:
    """`daemon/status`' report envelope back into what `console.section` draws.

    Two keys per row rather than a two-element array, for the reason
    `Root.describe` gives about one name per fact: an array of pairs on the wire
    is a shape whose meaning is positional, and positional meaning is what a
    reader gets wrong.

    Through `seq`, not `isinstance(…, list)`. That distinction is load-bearing
    and `wire`'s docstring says why: a payload is a `list` off disk and a
    **tuple** in memory, so a reader that tests for `list` works on resume and
    silently sees nothing live — which for this section would be a socket
    lifetime that renders as an empty table with no error at all.
    """
    return [
        (
            str(obj(one).get("title", "")),
            [
                (str(obj(row).get("label", "")), str(obj(row).get("value", "")))
                for row in seq(obj(one).get("rows"))
            ],
        )
        for one in seq(reported)
    ]


def _reachability(facts: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The row that only exists when the answer is bad (P5-11).

    Absent while the daemon can be reached, which is `Diagnostic.read`'s rule
    and the reason it is worth obeying here: a person who can read this table at
    all just connected, so a permanent "reachable: yes" row would be a fact
    proved by its own delivery — and the row that matters would be one more line
    of a table nobody scans.

    That it can ever be *seen* is the odd case and the useful one: the socket
    went away, the daemon noticed, and this connection came in on a client that
    was already attached — or through a path somebody restored by hand.
    """
    since = facts["unreachableSince"]
    if since is None:
        return []
    return [
        ("reachable", f"[red]no[/red] — the socket stopped being this daemon's at {_when(since)}")
    ]


def _line(event: Mapping[str, Any]) -> str:
    kind = str(event.get("type", ""))
    body = _summary(kind, event)
    seq = f"{event.get('seq', '')!s:>5}"
    if not body:
        return f"[dim]{seq}  {kind}[/dim]"
    return f"[dim]{seq}[/dim]  [bold]{kind}[/bold]  {body}"


# --------------------------------------------------------------------- follow --


NOISE: frozenset[str] = frozenset(
    {
        # The streamed halves of `assistant/message`, which *is* shown. One line
        # per delta is a keystroke log, not a follow — and here it is also almost
        # all of the rendering cost, for output nobody reads.
        "assistant/chunk",
        # The dispatch's opening half; its settled half carries the content.
        "tool/code-dispatch-start",
        # One event per changed variable per cell.
        "kernel/snapshot",
    }
)
"""Types a follow leaves out unless asked. `--all` includes them.

Deliberately *not* the trajectory's `RECORDLESS`, which is that view's set for
that view's reasons and lives under `ph_app.tui`. Small and open on purpose: an
unclassified type is shown, because a follower that hid something new would be
the silent omission the fallback in `_summary` exists to prevent.
"""


@dataclass(slots=True)
class _Follow:
    """One attached session as the CLI prints it, over `Followed`.

    The buffering, the `seq` dedupe and the paging live in `daemon/follow.py`,
    which the TUI's socket client is built on too; what is here is what is the
    CLI's own — a console, the `--type` filter, and the `until_idle` stop.
    """

    session_id: str
    until_idle: bool = False
    everything: bool = False
    selectors: Sequence[Selector] = ()
    """Namespace selectors from `--type`; empty means no filter (P6-33)."""
    done: anyio.Event = field(default_factory=anyio.Event)
    feed: Followed = field(init=False)

    def __post_init__(self) -> None:
        self.feed = Followed(
            session_id=self.session_id, on_events=self._events, on_status=self._status
        )

    def _events(self, pairs: Sequence[tuple[Mapping[str, Any], Any]], _live: bool) -> None:
        # The phase is not a distinction a follower draws: it prints records, and
        # a record read from a page reads the same as one that just arrived.
        self.write(event for event, _view in pairs)

    def _status(self, params: Mapping[str, Any]) -> None:
        console.print(f"[dim]· {params.get('status')}[/dim]", soft_wrap=True)
        if self.until_idle and params.get("status") == "idle":
            self.done.set()

    def write(self, events: Iterable[Mapping[str, Any]]) -> None:
        """Render a run of events as one write.

        One `console.print` per event costs multiples of the same events joined,
        which is what makes catching up on a long log bearable — and why this takes
        a sequence rather than an event.
        """
        lines = [_line(event) for event in events if self._shows(event)]
        if lines:
            # `soft_wrap`, for the reason a machine-readable dump does not go
            # through a console at all: a follower's lines are records, and
            # folding one at the terminal width turns a `grep` into two
            # half-matches.
            console.print("\n".join(lines), soft_wrap=True)

    def _shows(self, event: Mapping[str, Any]) -> bool:
        """Whether this follower prints one event.

        **A `--type` selector replaces the per-delta hush rather than stacking on
        it.** `NOISE` exists because a turn is mostly `assistant/chunk`, whose
        content arrives again in the `assistant/message` that closes it — so the
        default is to hide it. Naming a namespace is a stronger signal than that
        default: somebody who typed `--type assistant` asked for the assistant
        traffic, and hiding most of it would answer a question they did not ask.
        So `--all` stops being necessary once `--type` is given, rather than
        being required on top of it.

        Said as two branches rather than one disjunction, and the type read once:
        the earlier form computed `event["type"]` here and again in a second
        predicate, and recomputed `bool(self.selectors)` — a value fixed at
        construction — on every frame of a replayed history.
        """
        kind = str(event.get("type", ""))
        if self.selectors:
            return matches_any(kind, self.selectors)
        return self.everything or kind not in NOISE


# ------------------------------------------------------------------- commands --


@agents_app.callback(invoke_without_command=True)
def agents(ctx: typer.Context) -> None:
    """List the roots this daemon is running."""
    if ctx.invoked_subcommand is not None:
        return
    listed = _ask(lambda client: client.call("sessions/list"))
    rows = listed["sessions"]
    if not rows:
        console.print("[dim]no roots running[/dim]")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("session")
    table.add_column("status")
    table.add_column("events", justify="right")
    table.add_column("watchers", justify="right")
    for row in rows:
        table.add_row(
            row["sessionId"],
            row["status"],
            str(obj(row.get("cursor")).get("sequence", "")),
            str(row["watchers"]),
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
    something to watch with `ph agents attach`, not something to hold a socket
    open for. The client mints its own idempotence key, so a resend after a
    dropped connection is safe rather than a second turn.
    """
    root = _ask(lambda client: client.prompt(session, prompt))
    console.print(f"[dim]queued on {root['sessionId']} · {root['status']}[/dim]")


@agents_app.command()
def attach(
    session: Annotated[str, SESSION_ARGUMENT],
    since: Annotated[
        int, typer.Option("--since", help="Skip history up to this sequence number.")
    ] = 0,
    until_idle: Annotated[
        bool, typer.Option("--until-idle", help="Stop once the root goes idle.")
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
    """

    # The session-log vocabulary: a bare `workspace` needs no prefix, and
    # `bus:tools` is refused rather than silently matching nothing (P6-33).
    selectors = selectors_or_exit(type_, vocabulary="log")

    async def work(client: DaemonClient) -> None:
        follow = _Follow(
            session_id=session,
            until_idle=until_idle,
            everything=everything,
            selectors=selectors,
        )
        # On the peer, which is where the read loop looks.
        client.peer.on_notify = follow.feed
        # Subscribed *before* the history is read, so nothing that happens in
        # between is lost; `_Follow` holds those frames until the pages are done.
        # No cursor: the generation a snapshot cursor needs is what this reply
        # carries, so `from` is 0 here by construction and catch-up is paged from
        # `--since` against the generation the daemon just named.
        attached = await client.call("session/attach", sessionId=session, cursor=None)
        cursor = {**obj(attached["cursor"]), "sequence": since}
        try:
            # `since` is the sequence to start *at*, so what has been "seen" is
            # everything below it — `- 1`, because seq 0 is a real event and
            # `--since 0` must still print it.
            follow.feed.seen = since - 1
            await follow.feed.catch_up(client, cursor)
            follow.feed.live()
            if until_idle and not follow.done.is_set():
                # One check, after the replay: a root that went idle *during*
                # catch-up announced it into `pending` and has just been drained,
                # and one that was already idle before the attach never announced
                # anything at all. Without this the second case waits forever.
                current = await client.call("session/status", sessionId=session)
                if current["status"] == "idle":
                    follow.done.set()
            await first_of(follow.done, client.closed)
        finally:
            # Only while there is somebody to tell. A daemon that shut down under
            # us has already dropped every subscription, and asking it to would
            # park on a reply nobody is left to send.
            if not client.closed.is_set():
                await client.call("session/detach", sessionId=session)
        if client.closed.is_set():
            raise DaemonGone

    _ask(work)


@agents_app.command()
def schedule(
    session: Annotated[str, SESSION_ARGUMENT],
    prompt: Annotated[
        str, typer.Option("--prompt", help="What to say to the agent when it fires.")
    ] = "",
    at: Annotated[int, typer.Option("--at", help="Fire once, at this epoch-ms instant.")] = 0,
    every: Annotated[int, typer.Option("--every", help="Fire every N milliseconds.")] = 0,
    cron: Annotated[str, typer.Option("--cron", help="Fire on this cron expression.")] = "",
    cancel: Annotated[str, typer.Option("--cancel", help="Cancel this schedule id.")] = "",
    schedule_id: Annotated[str, typer.Option("--id", help="Name the schedule.")] = "",
) -> None:
    """List, create, or cancel a root's schedules.

    The timing flag chooses the kind, rather than a `--kind`/`--spec` pair that
    would put the wire's shape in front of the person using it — `--every` is an
    interval, `--cron` an expression, `--at` a single instant. Exactly one, and
    a prompt, because a schedule with nothing to say is claimed, recorded and
    delivers nothing forever.
    """
    # The chosen timing carried as the pair it becomes, rather than three bools
    # whose identity has to be reconstructed further down by a ternary chain.
    timings = [
        (kind, str(value))
        for kind, value in (("once", at), ("interval", every), ("cron", cron))
        if value
    ]
    if cancel:
        if timings:
            raise typer.BadParameter("--cancel takes no timing flag")
        outcome = _ask(
            lambda client: client.call("schedule/cancel", sessionId=session, scheduleId=cancel)
        )
        if not outcome["cancelled"]:
            fail(f"[red]no schedule {cancel!r} on {session}[/red]")
        console.print(f"[dim]cancelled {cancel}[/dim]")
        return

    if len(timings) > 1:
        raise typer.BadParameter("one of --at, --every or --cron, not several")
    if not timings:
        listed = _ask(lambda client: client.call("schedule/list", sessionId=session))
        _print_schedules(session, listed["schedules"])
        return
    if not prompt:
        raise typer.BadParameter("a schedule needs --prompt: what to say when it fires")

    ((kind, spec),) = timings
    created = _ask(
        lambda client: client.call(
            "schedule/create",
            sessionId=session,
            scheduleId=schedule_id or f"sch-{secrets.token_hex(4)}",
            kind=kind,
            spec=spec,
            prompt=prompt,
        )
    )
    console.print(f"[dim]scheduled {created['id']} · {kind} {spec}[/dim]")


def _print_schedules(session: str, rows: list[dict[str, Any]]) -> None:
    """The rows themselves, not a reply to index into.

    It took a whole `schedule/list` envelope and read `["schedules"]` out of it,
    which worked only because the `session/status` reply happens to use the same
    key — a printer coupled to two unrelated wire shapes by a string.
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
        table.add_row(
            row["id"],
            row["kind"],
            row["spec"],
            _when(row.get("nextAt")),
            one_line(row["prompt"], 40),
        )
    console.print(table)


@agents_app.command()
def status(session: Annotated[str, SESSION_ARGUMENT]) -> None:
    """What one root is doing, and what the log says about how it got there."""
    row = _ask(lambda client: client.call("session/status", sessionId=session))
    cursor = obj(row.get("cursor"))
    schedules = row["schedules"]
    console.print(
        section(
            f"root {row['sessionId']}",
            (
                ("status", str(row["status"])),
                ("events", str(cursor.get("sequence", ""))),
                ("generation", str(cursor.get("generation", ""))),
                ("watchers", str(row["watchers"])),
                ("retry attempts", str(row["attempts"])),
                ("given up", "yes" if row["failed"] else "no"),
                ("schedules", str(len(schedules))),
            ),
        )
    )
    if schedules:
        _print_schedules(session, schedules)


@agents_app.command()
def doctor() -> None:
    """Ask the daemon what it is: where it listens, and how it was started.

    The client half of `ph doctor`. Everything printed comes back over the wire
    from the running process, so what this reports is what is *in force* rather
    than what this invocation's flags and environment would have produced.
    """
    facts = _ask(lambda client: client.call("daemon/status"))
    passivate = facts["passivateAfter"]
    console.print(
        section(
            "pH daemon",
            (
                ("socket", facts["socket"]),
                ("pid", str(facts["pid"])),
                ("uptime", _duration(facts["uptimeMs"])),
                ("protocol", str(facts["protocolVersion"])),
                ("capabilities", ", ".join(sorted(facts["capabilities"]))),
                ("roots", str(facts["roots"])),
                ("provider", f"{facts['provider']} · {facts['model']}"),
                ("passivate after", "off" if passivate is None else _duration(passivate * 1000)),
                ("tick", _duration(facts["tickEvery"] * 1000)),
                ("sweep", _duration(facts["sweepEvery"] * 1000)),
                ("heartbeat", _duration(facts["heartbeatEvery"] * 1000)),
                ("socket watch", _duration(facts["watchEvery"] * 1000)),
                *_reachability(facts),
            ),
        )
    )
    # A loop, because the envelope is a list: P5-12 adds the daemon's mounted
    # diagnostics to it and this side needs no change to render them.
    for title, rows in _sections(facts["sections"]):
        console.print(section(title, rows))


@agents_app.command()
def shutdown() -> None:
    """Stop the daemon, and wait for it to actually be gone.

    `shutdown` is a notification by contract — a request awaiting a reply would
    be waiting on a frame the daemon is concurrently losing the ability to write
    — so the confirmation is the connection the daemon closes on its way out.
    Waiting for it rather than declaring success is the difference between "I
    asked" and "it stopped", and roots being flushed and leases released is
    exactly what happens in that window.
    """

    async def work(client: DaemonClient) -> None:
        await client.notify("shutdown")
        # The same budget teardown itself is bounded by, plus room for the
        # unwinding around it: a daemon still inside its grace period has not
        # failed to stop, it is stopping.
        with anyio.move_on_after(GRACE_SECONDS + 5.0):
            await client.closed.wait()
        if not client.closed.is_set():
            fail("[yellow]shutdown sent; the daemon has not stopped yet[/yellow]")

    _ask(work)
    console.print("[dim]daemon stopped[/dim]")
