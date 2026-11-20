"""`phern service`: the thin client that an orchestrator drives (P13-13).

**phern never restarts itself.** An init system that already exists (systemd,
launchd, Kubernetes) owns the process, its environment and its backoff.
`contrib/` has systemd units and a Kubernetes manifest. This module is the
client for a host that has no init system: a shell, a cron slot, a container
entrypoint, a person who wants one word for "bring it up and resume what it
owes". It has six commands over one socket and one spawner. Every command
answers in the vocabulary of the orchestrator: exit 0 when the daemon is ready,
1 when it is down, and 2 when it is degraded. A degraded daemon has one of
these: a root that is held on a person, a root whose ladder is spent, a root
with a log that does not read, a root that owes work and is not mounted, or a
root that the daemon could not write.

`start` spawns `phern daemon --resume-owed` with whatever follows it on the
command line, and waits for readiness. Readiness covers the resume pass of the
daemon itself. `start` then also runs the pass from here and prints it. For a
daemon that was already listening, this is the pass. `stop` sends `shutdown` and
waits for the daemon to go. `restart` is `stop`, then `start`. `status` is one
look: the readiness answer and the owed table. `resume` is the pass alone.
`watch` is the loop: start when down, resume what is owed when ready, report
when degraded. Between failing passes it waits on the retry ladder of the
scheduler.

**It passes no environment and reads no files.** The daemon that it starts
composes its own configuration from `$PH_HOME`, `$PH_RUNTIME` and the options
that it was given. The `EnvironmentFile` of a unit owns those. A credential
reaches the daemon in the same way and is never written down. This module knows
where the socket is. It resolves the socket exactly as `phern agents` does, and
it reaches the daemon through the same function (`agents.reach`).

@module ph_app.service
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Literal, TypeAlias

import anyio
import typer
from rich.console import Console

from ph.paths import PathRoots
from ph.resources import SHUTDOWN_SECONDS
from ph.text import duration

from . import verbs
from .agents import Down, print_owed, print_resume, reach, stop_daemon
from .console import console, detail, err, fail, roots_or_exit
from .daemon.client import DaemonClient
from .daemon.launch import SPAWN_TIMEOUT, DaemonAbsent, ensure_daemon, phern_argv
from .daemon.recovery import WAKE_RETRY_DELAYS
from .params import OwedParams, ProbeParams, ResumeParams
from .payloads import OwedListing, ProbeReply
from .protocol import NoParams

__all__ = ["EXIT", "Observed", "service_app"]

service_app = typer.Typer(
    help="Run the daemon as a service: start, stop, restart, status, resume, watch.",
    add_completion=False,
)

Verdict: TypeAlias = Literal["ready", "degraded", "down"]
"""What one look at the daemon concluded, in the three words that an orchestrator reads."""

EXIT: Mapping[Verdict, int] = {"ready": 0, "down": 1, "degraded": 2}
"""The exit status for each verdict: what a unit, a cron slot or a script branches on."""

PASS_THROUGH: Mapping[str, bool] = {"allow_extra_args": True, "ignore_unknown_options": True}
"""Typer settings for the commands that start a daemon. Everything after the
command name that is not an option of the command itself goes to `phern daemon`
as typed. The options of the daemon are therefore spelled one time, by the
daemon."""


# ------------------------------------------------------------------ the look --


@dataclass(frozen=True, slots=True)
class Observed:
    """What one look at a running daemon found. It is folded one time into the
    reasons that an operator reads and the roots that a resume pass is for."""

    pid: int
    uptime_ms: int
    owed: OwedListing
    unmounted: tuple[str, ...]
    """Roots that owe work and are not mounted. A resume pass is for these. A root
    whose ladder is spent is left out. It is a reason of its own, and a pass
    without `force` would leave it where it is."""
    degraded: tuple[str, ...]
    """Why this daemon is not ready, one sentence for each reason. Empty when it is ready."""

    @classmethod
    def of(cls, *, pid: int, uptime_ms: int, ready: ProbeReply, owed: OwedListing) -> Observed:
        """Take the reasons of the readiness probe itself, then add what the owed
        listing shows: a root whose ladder is spent, a root that is held on a
        person, a root with a log that does not read, a root that nobody has
        mounted."""
        rows = owed.sessions
        unmounted = tuple(
            row.session_id for row in rows if not row.mounted and row.left_by_a_pass() is None
        )
        reasons = list(ready.reasons)
        if failed := [row.session_id for row in rows if row.failed]:
            reasons.append(
                f"root(s) gave up and wait for a person, or for `phern agents resume "
                f"--force`: {', '.join(failed)}"
            )
        if held := [f"{row.session_id} ({row.status})" for row in rows if row.held]:
            reasons.append(f"root(s) held on a person: {', '.join(held)}")
        damaged = [f"{row.session_id} ({', '.join(row.damaged)})" for row in rows if row.damaged]
        if damaged:
            reasons.append(
                f"root(s) with a log that will not read: {', '.join(damaged)}; "
                "`phern doctor sessions --repair` cuts it back to what does"
            )
        if unmounted:
            reasons.append(
                f"root(s) owe work and are not mounted: {', '.join(unmounted)}; "
                "`phern service resume` mounts them"
            )
        return cls(
            pid=pid, uptime_ms=uptime_ms, owed=owed, unmounted=unmounted, degraded=tuple(reasons)
        )

    @property
    def verdict(self) -> Verdict:
        return "degraded" if self.degraded else "ready"


async def _probe_ready(client: DaemonClient) -> ProbeReply:
    return await client.call(verbs.DAEMON_PROBE, ProbeParams(name="ready"))


async def _look(client: DaemonClient) -> Observed:
    """Ask the three questions on one connection."""
    status = await client.call(verbs.DAEMON_STATUS, NoParams())
    ready = await _probe_ready(client)
    owed = await client.call(verbs.SESSIONS_OWED, OwedParams(rebuild=False))
    return Observed.of(pid=status.pid, uptime_ms=status.uptime_ms, ready=ready, owed=owed)


def _bullets(printer: Console, reasons: Iterable[str]) -> None:
    for reason in reasons:
        printer.print(f"  [yellow]-[/yellow] {detail(reason)}")


def _down(down: Down) -> Verdict:
    """Say why nothing answered, and return the verdict `"down"`."""
    err.print(down.why)
    return "down"


def _say(observed: Observed, *, table: bool) -> None:
    """Print the verdict line and its reasons. With `table`, print the owed table too."""
    verdict = observed.verdict
    color = "green" if verdict == "ready" else "yellow"
    console.print(
        f"[{color}]{verdict}[/{color}] · pid {observed.pid} · up {duration(observed.uptime_ms)}"
    )
    _bullets(console, observed.degraded)
    if table:
        print_owed(observed.owed)
    elif not observed.owed.trusted:
        console.print("[dim]the live-roots index could not vouch for itself[/dim]")


# --------------------------------------------------------------- the actions --


async def _await_ready(path: Path) -> ProbeReply | Down:
    """Return the readiness answer when it is yes, or the last no at `SPAWN_TIMEOUT`.

    The wait is as long as the wait of the spawn, for the same reason: a wait
    that ends early tells a user that their daemon failed while it is still
    starting."""
    deadline = anyio.current_time() + SPAWN_TIMEOUT
    while True:
        reply = await reach(path, _probe_ready)
        if isinstance(reply, Down) or reply.ok or anyio.current_time() >= deadline:
            return reply
        await anyio.sleep(0.25)


async def _start(roots: PathRoots, options: Sequence[str]) -> Verdict:
    """Spawn, wait for ready, resume, and look. The function is idempotent: a
    daemon that already listens is not started again, and is still asked to
    resume what it owes.

    A daemon that this function starts runs its own resume pass
    (`--resume-owed`). Readiness is what a probe port reports while this function
    waits. It therefore means that the roots of the daemon are mounted, and not
    only that its socket is bound. The pass from here then reports the roots."""
    path = roots.daemon_socket()
    try:
        started = await ensure_daemon(
            argv=phern_argv("daemon", "--resume-owed", *options), roots=roots
        )
    except DaemonAbsent as refused:
        err.print(f"[red]{detail(refused)}[/red]")
        return "down"
    said = "started a daemon on" if started.spawned else "a daemon is already listening on"
    console.print(f"[dim]{said} {path}[/dim]")
    ready = await _await_ready(path)
    if isinstance(ready, Down):
        return _down(ready)
    if not ready.ok:
        err.print(f"[yellow]listening, but not ready after {SPAWN_TIMEOUT:g}s:[/yellow]")
        _bullets(err, ready.reasons)
        return "degraded"
    return await _resume(path, force=False)


async def _resume(path: Path, *, force: bool) -> Verdict:
    """Run the pass over every owed root, print it, and look at the daemon after it."""
    report = await reach(
        path,
        lambda client: client.call(
            verbs.SESSIONS_RESUME, ResumeParams(sessions=[], all_owed=True, force=force)
        ),
    )
    if isinstance(report, Down):
        return _down(report)
    print_resume(report)
    observed = await reach(path, _look)
    if isinstance(observed, Down):
        return _down(observed)
    _say(observed, table=False)
    return observed.verdict


async def _stop(path: Path) -> bool:
    """Send `shutdown`. Return whether the socket is now clear: the daemon is gone,
    or there was none.

    A closed connection means that the socket is already removed
    (`agents.stop_daemon`), so the next `start` finds nothing to clear."""
    stopped = await reach(path, stop_daemon)
    if isinstance(stopped, Down):
        console.print(f"[dim]nothing to stop: no daemon answers at {path}[/dim]")
        return True
    if not stopped:
        err.print(
            f"[yellow]shutdown sent; the daemon has not stopped after "
            f"{SHUTDOWN_SECONDS:g}s[/yellow]"
        )
        return False
    console.print("[dim]daemon stopped[/dim]")
    return True


# ------------------------------------------------------------------ the loop --


Said: TypeAlias = tuple[Verdict, tuple[str, ...]]
"""What a watcher last reported. With it, a daemon that stays ready for a week is
one line in the journal and not one line for each pass."""


async def _pass(
    roots: PathRoots, options: Sequence[str], said: Said | None
) -> tuple[Verdict, Said | None]:
    """Run one pass: start when down, resume when something is owed, and report
    the rest when it differs from what was `said` last. Return the verdict and
    what is now said. An action is reported in full and clears what was said, so
    the next look is always reported."""
    path = roots.daemon_socket()
    observed = await reach(path, _look)
    if isinstance(observed, Down):
        console.print(f"[yellow]down:[/yellow] no daemon answers at {path}; starting one")
        return await _start(roots, options), None
    if observed.unmounted:
        console.print(f"[yellow]owed:[/yellow] {', '.join(observed.unmounted)}; resuming")
        return await _resume(path, force=False), None
    now = (observed.verdict, observed.degraded)
    if now != said:
        _say(observed, table=observed.verdict != "ready")
    return observed.verdict, now


def _delay(verdict: Verdict, rung: int) -> tuple[float, int]:
    """Return how long to wait after a pass that ended `verdict`, and the rung
    that the next pass starts on.

    The wait comes from `rung` of the retry ladder of the scheduler
    (`WAKE_RETRY_DELAYS`). The rung is the first one while the daemon is ready.
    It goes one up for each pass that found the daemon not ready, and the last
    rung repeats. The ladder of the scheduler is used because the wait has the
    same shape: a thing that was tried, that failed, and that will not be
    different in the next second."""
    if verdict == "ready":
        return WAKE_RETRY_DELAYS[0], 0
    return WAKE_RETRY_DELAYS[rung], min(rung + 1, len(WAKE_RETRY_DELAYS) - 1)


async def _watch(roots: PathRoots, options: Sequence[str], *, once: bool) -> Verdict:
    said: Said | None = None
    rung = 0
    while True:
        verdict, said = await _pass(roots, options, said)
        if once:
            return verdict
        delay, rung = _delay(verdict, rung)
        await anyio.sleep(delay)


# ------------------------------------------------------------------- commands --


@service_app.command(context_settings=dict(PASS_THROUGH))
def start(ctx: typer.Context) -> None:
    """Start the daemon, wait until it is ready, and resume every root that owes work.

    Everything after `start` goes to `phern daemon` as typed — `--profile`,
    `--probe-bind`, `--max-concurrent-children`, a `--patch` — so the daemon's
    options are spelled once, by the daemon, and its refusal of one is the daemon's.
    A daemon already listening is not started again; it is still asked to resume.
    Exit 0 ready, 1 down, 2 degraded.
    """
    verdict = anyio.run(partial(_start, roots_or_exit(create=True), list(ctx.args)))
    raise typer.Exit(code=EXIT[verdict])


@service_app.command()
def stop() -> None:
    """Stop the daemon and wait for it to go.

    Nothing listening is not a failure: the state asked for is the state found.
    """
    clear = anyio.run(partial(_stop, roots_or_exit().daemon_socket()))
    raise typer.Exit(code=0 if clear else 1)


@service_app.command(context_settings=dict(PASS_THROUGH))
def restart(ctx: typer.Context) -> None:
    """`stop`, then `start` with everything after `restart` handed to the daemon.

    A daemon that has not stopped is not restarted: its socket is already gone,
    so a second one would bind beside a first still giving its leases back.
    """
    if not anyio.run(partial(_stop, roots_or_exit().daemon_socket())):
        fail("[red]not starting another daemon beside one still stopping[/red]")
    start(ctx)


@service_app.command()
def status() -> None:
    """One look: ready, degraded or down, why, and the owed table.

    Exit 0 ready, 1 down, 2 degraded — a root held on a person, one whose ladder is
    spent, one that owes work and is not mounted, or one the daemon could not write.
    """
    observed = anyio.run(partial(reach, roots_or_exit().daemon_socket(), _look))
    if isinstance(observed, Down):
        raise typer.Exit(code=EXIT[_down(observed)])
    _say(observed, table=True)
    raise typer.Exit(code=EXIT[observed.verdict])


@service_app.command()
def resume(
    force: Annotated[
        bool,
        typer.Option("--force", help="Mount a root whose ladder is spent as well."),
    ] = False,
) -> None:
    """Resume every root that owes work, then look.

    `phern agents resume --all` in the service's vocabulary: exit 0 ready, 1 down,
    2 degraded.
    """
    verdict = anyio.run(partial(_resume, roots_or_exit().daemon_socket(), force=force))
    raise typer.Exit(code=EXIT[verdict])


@service_app.command(context_settings=dict(PASS_THROUGH))
def watch(
    ctx: typer.Context,
    once: Annotated[
        bool,
        typer.Option(
            "--once",
            help="One pass, then exit with its verdict — for a cron slot rather than a loop.",
        ),
    ] = False,
) -> None:
    """Keep the daemon up: start it when down, resume when owed, report when degraded.

    Everything after `watch` that is not `--once` goes to `phern daemon` on each start.
    Between passes it waits on the scheduler's retry ladder — its first rung while
    the daemon is ready, climbing while it is not — and says something only when
    the verdict or its reasons change.

    A signal ends the watcher the default way and nothing else: it holds nothing a
    signal would need to unwind, and the daemon is a process of its own, which
    `phern service stop` stops.
    """
    verdict = anyio.run(partial(_watch, roots_or_exit(create=True), list(ctx.args), once=once))
    raise typer.Exit(code=EXIT[verdict])
