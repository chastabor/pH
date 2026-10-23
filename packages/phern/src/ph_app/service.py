"""`phern service` — the thin client an orchestrator drives (P13-13).

**phern never restarts itself.** An init system that already exists — systemd,
launchd, Kubernetes — owns the process, its environment and its backoff, and
`contrib/` has a unit and a manifest for each. This is the client for the host with
none of them: a shell, a cron slot, a container entrypoint, a person who wants one
word for "bring it up and resume what it owes". Six commands over one socket and
one spawner, and every one answers in the orchestrator's vocabulary: exit 0 when the
daemon is ready, 1 when it is down, 2 when it is degraded — a root held on a person,
one whose ladder is spent, one that owes work and is not mounted, or one the daemon
could not write.

`start` spawns `phern daemon --resume-owed` with whatever follows it on the command
line, waits for readiness — which covers the daemon's own resume pass — then runs
the pass from here as well and prints it, which for a daemon already listening is
the pass. `stop` sends `shutdown` and waits for the daemon to go. `restart` is the
two. `status` is one look: the readiness answer and the owed table. `resume` is the
pass alone. `watch` is the loop — start when down, resume what is owed when ready,
report when degraded — with the scheduler's own retry ladder between failing passes.

**It passes no environment and reads no files.** The daemon it starts composes its
own configuration from `$PH_HOME`, `$PH_RUNTIME` and the options it was handed; a
unit's `EnvironmentFile` owns those, and a credential reaches the daemon the same
way and is never written down. What this module knows is where the socket is, which
it resolves exactly as `phern agents` does, and it reaches the daemon through the
same spine (`agents.reach`).

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
"""What one look at the daemon concluded, in the three words an orchestrator reads."""

EXIT: Mapping[Verdict, int] = {"ready": 0, "down": 1, "degraded": 2}
"""The exit status for each verdict: what a unit, a cron slot or a script branches on."""

PASS_THROUGH: Mapping[str, bool] = {"allow_extra_args": True, "ignore_unknown_options": True}
"""Typer settings for the commands that start a daemon: everything after the
command name that is not the command's own option goes to `phern daemon` as typed,
so the daemon's options are spelled once, by the daemon."""


# ------------------------------------------------------------------ the look --


@dataclass(frozen=True, slots=True)
class Observed:
    """What one look at a running daemon found, folded once into the reasons an
    operator reads and the roots a resume pass is for."""

    pid: int
    uptime_ms: int
    owed: OwedListing
    unmounted: tuple[str, ...]
    """Roots that owe work and are not mounted — what a resume pass is for. A root
    whose ladder is spent is left out: it is a reason of its own, and a pass without
    `force` would leave it where it is."""
    degraded: tuple[str, ...]
    """Why this daemon is not simply ready, one sentence each; empty when it is."""

    @classmethod
    def of(cls, *, pid: int, uptime_ms: int, ready: ProbeReply, owed: OwedListing) -> Observed:
        """The readiness probe's own reasons, then what the owed listing adds: a
        root whose ladder is spent, one held on a person, one nobody has mounted."""
        rows = owed.sessions
        unmounted = tuple(row.session_id for row in rows if not row.mounted and not row.failed)
        reasons = list(ready.reasons)
        if failed := [row.session_id for row in rows if row.failed]:
            reasons.append(
                f"root(s) gave up and wait for a person, or for `phern agents resume "
                f"--force`: {', '.join(failed)}"
            )
        if held := [f"{row.session_id} ({row.status})" for row in rows if row.held]:
            reasons.append(f"root(s) held on a person: {', '.join(held)}")
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
    """The three questions, on one connection."""
    status = await client.call(verbs.DAEMON_STATUS, NoParams())
    ready = await _probe_ready(client)
    owed = await client.call(verbs.SESSIONS_OWED, OwedParams(rebuild=False))
    return Observed.of(pid=status.pid, uptime_ms=status.uptime_ms, ready=ready, owed=owed)


def _bullets(printer: Console, reasons: Iterable[str]) -> None:
    for reason in reasons:
        printer.print(f"  [yellow]-[/yellow] {detail(reason)}")


def _down(down: Down) -> Verdict:
    """Say why nothing answered, and conclude so."""
    err.print(down.why)
    return "down"


def _say(observed: Observed, *, table: bool) -> None:
    """The verdict line, its reasons, and with `table` the owed table under it."""
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
    """The readiness answer once it is yes, or the last no at `SPAWN_TIMEOUT` — the
    spawn's own patience, for the spawn's own reason: a wait that fires early tells
    a person their daemon failed while it is still starting."""
    deadline = anyio.current_time() + SPAWN_TIMEOUT
    while True:
        reply = await reach(path, _probe_ready)
        if isinstance(reply, Down) or reply.ok or anyio.current_time() >= deadline:
            return reply
        await anyio.sleep(0.25)


async def _start(roots: PathRoots, options: Sequence[str]) -> Verdict:
    """Spawn, wait for ready, resume, and look. Idempotent: a daemon already
    listening is not started again, and is still asked to resume what it owes.

    A daemon this starts runs its own resume pass (`--resume-owed`), so readiness
    — what a probe port reports while this waits — means its roots are mounted,
    not merely its socket bound; the pass from here then reports them."""
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
    """The pass over every owed root, printed, and the look after it."""
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
    """`shutdown`, and whether the door is now clear: the daemon gone, or none there.

    A closed connection is a socket already removed (`agents.stop_daemon`), so the
    next `start` finds nothing to clear."""
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
"""What a watcher last reported, so a daemon that stays ready for a week is one
line in the journal rather than one per pass."""


async def _pass(
    roots: PathRoots, options: Sequence[str], said: Said | None
) -> tuple[Verdict, Said | None]:
    """One pass: start when down, resume when something is owed, report the rest
    when it differs from what was `said` last. Returns the verdict and what is now
    said; an action is said in full, so the next look is reported afresh."""
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
    """How long to wait after a pass that ended `verdict`, from `rung` of the
    scheduler's retry ladder (`WAKE_RETRY_DELAYS`), and the rung the next pass
    starts on: the first rung while the daemon is ready, one up per pass that found
    it otherwise, the last repeated. The scheduler's ladder because it is the same
    shape of wait — a thing that was tried, failed, and will not be different next
    second."""
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
