"""`$PH_RUNTIME/daemon.sock` — the supervisor's front door (P5-01).

A unix socket rather than stdio, which is the whole point: stdio has exactly one
peer and dies with it, and a daemon exists to be reconnected to. The method names
extend `--mode rpc`'s shape rather than starting a second vocabulary.

**The socket is state, and stale state is a lie.** A path left behind by a crashed
daemon makes every client hang on a connect that will never be answered, so
binding removes an unresponsive one first and **refuses a responsive one** — the
second is another daemon, which is P5-03's lease to arbitrate rather than this
row's to overwrite.

The same sentence read the other way is P5-11: a path that stopped being *this*
daemon's socket is a lie about this daemon. `$PH_RUNTIME` sits under
`$XDG_RUNTIME_DIR`, which logind reaps at logout for a user who is not lingering,
so the door can be removed while the process behind it keeps running — and every
later client is told "no daemon socket" and to start one, which the leases the
first is still holding will refuse. `watch` compares the socket's inode against
the one bound here, and says so in each root's own log, because by then the
surfaces that could carry the news are exactly the ones that went away.

@module ph_app.daemon.server
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream

from ph.cordis import ProfileLayer
from ph.lingering import RuntimeLifetime, lifetime, socket_identity
from ph.paths import resolve_roots
from ph.resources import GRACE_SECONDS
from ph.seams.schedule import Schedule
from ph.session import now_ms

from ..protocol import (
    SNAPSHOT_EVENTS,
    Refusal,
    capabilities,
    cursor_of,
    notification,
    respond,
    resume_at,
)
from .framing import FramingError, read_frames, write_frame
from .recovery import PASSIVATE_AFTER, WAKE_WITHIN
from .supervisor import NON_GUARANTEES, Supervisor

__all__ = ["DaemonServer", "serve"]

log = logging.getLogger("ph_app.daemon")

HEARTBEAT_EVERY = 5 * 60.0
"""Seconds between liveness records for a root that has work scheduled.

Not a keep-alive and not a health check: a record, so an operator reading a
cron-driven trace can tell "waiting for Wednesday" from "died on Tuesday". A
schedule that fires monthly otherwise leaves a log whose last line is a month
old, which is indistinguishable from a log nobody is writing.

Beside the other two cadences rather than in the seam, where it was: ph-core
held a constant only this loop read."""

TICK_EVERY = 5.0
"""How often due schedules are checked. The floor on a schedule's resolution.

Five seconds rather than the sweeper's sixty: this decides when work *starts*,
and a minute of slack on "run at 09:00" is a minute somebody notices."""

SWEEP_EVERY = 60.0
"""How often the passivation sweep runs. A coarse tick, not a second timeout."""

WATCH_EVERY = 30.0
"""How often the daemon checks that the socket at its path is still its own (P5-11).

The thing being watched changes at most once in the life of a process — a
logout, a reboot's worth of directory — so this is not a poll on a hot fact. It
is a bound on how long the log takes to say what happened, and thirty seconds
means the record's timestamp still lines up with the logout a person is trying
to correlate it with. Cheaper than the sweep it sits beside: one `lstat`, no
roots walked.
"""

CAPABILITIES = ("roots", "attach", "cursors", "snapshots")
"""What this transport adds to the two both of them have.

A constant because two callers say it now — `initialize`, and the `daemon/status`
a client runs when it wants to know what it is talking to — and a capability
block that disagreed with itself depending on which method you asked would be
worse than having none.
"""


class DaemonUnavailable(Refusal):
    """This daemon cannot take its socket, and the reason is worth a sentence.

    One type over both startup preconditions — a socket another daemon is
    listening on, and one the kernel will not bind (a `$PH_RUNTIME` past
    `AF_UNIX`'s 107-byte path limit, a directory this user cannot write). The
    CLI caught `(RuntimeError, OSError)` for them, which is two builtins wide
    enough to swallow a `typer.Exit` — `typer.Exit` subclasses `RuntimeError`,
    and `cli.py` already carries a comment about being bitten by exactly that.
    """

    code = "daemon_unavailable"


class UnknownMethod(Refusal):
    """This server does not serve that name."""

    code = "unknown_method"


class NoSuchSession(Refusal):
    """No root is running under that id — which is not the same as it being
    busy, and a client that wants to start one branches differently."""

    code = "no_such_session"


@dataclass(slots=True)
class _Connection:
    """One client, and the roots it is watching.

    The attachment table is per connection so a disconnect can undo exactly what
    that client did — a root keeps running, and stops sending to a socket nobody
    is reading.
    """

    stream: ByteStream
    server: DaemonServer
    attached: set[str] = field(default_factory=set)
    outbox: Any = None

    async def serve(self) -> None:
        send, receive = anyio.create_memory_object_stream[dict[str, Any]](max_buffer_size=1024)
        self.outbox = send
        try:
            async with anyio.create_task_group() as group:
                # Writes go through one task, so a notification arriving while a
                # reply is half-written cannot interleave two frames on the wire.
                group.start_soon(self._pump, receive)
                await self._read()
                group.cancel_scope.cancel()
        finally:
            for root_id in self.attached:
                root = self.server.supervisor.roots.get(root_id)
                if root is not None:
                    root.unsubscribe(self.notify)
            self.attached.clear()
            await send.aclose()

    async def _pump(self, receive: Any) -> None:
        async with receive:
            async for payload in receive:
                try:
                    await write_frame(self.stream, payload)
                except (anyio.BrokenResourceError, anyio.ClosedResourceError):
                    return

    async def _read(self) -> None:
        try:
            async for request in read_frames(self.stream):
                await self._handle(request)
        except FramingError as error:
            # Unreadable framing ends the connection: after a bad frame there is
            # no way to know where the next one starts.
            log.info("ph_app.daemon: closing a connection — %s", error)
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Queue a notification, or *raise* so the root drops this watcher.

        **Raising is the point.** Catching `WouldBlock` here and logging "dropped"
        drops nothing, and the watcher that cannot keep up re-pays the whole fan-out
        for every later event. The subscriber list belongs to the root, so the root
        is what removes from it; this only has to fail loudly enough to be noticed.
        """
        if self.outbox is None:
            raise RuntimeError("this connection is closed")
        self.outbox.send_nowait(notification(method, params))

    async def _handle(self, request: dict[str, Any]) -> None:
        reply = await respond(request, self._dispatch)
        if reply is not None and self.outbox is not None:
            self.outbox.send_nowait(reply)

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        """The daemon's half of the vocabulary — dsh's names (P5-02).

        `session/*` rather than P5-01's `root/*`: a supervised root *is* a
        session here, and the dsh client already ships against these names. The
        supervisory additions (`daemon/hello`, `session/attach`) are declared in
        the capability block rather than inferred from which socket answered.
        """
        supervisor = self.server.supervisor
        if method in ("initialize", "daemon/hello"):
            return capabilities(*CAPABILITIES)
        if method == "sessions/list":
            return {"sessions": supervisor.describe()}
        if method == "session/new":
            root = await supervisor.start(str(params["sessionId"]))
            return root.describe()
        if method == "session/prompt":
            # Joined here, at the wire edge, so the key has one construction.
            command_id = str(params.get("commandId", ""))
            client_id = str(params.get("clientId", ""))
            root = await supervisor.prompt(
                str(params["sessionId"]),
                str(params.get("prompt", "")),
                command=f"{client_id}:{command_id}" if command_id else "",
            )
            return root.describe()
        if method == "session/attach":
            # Through `start`, so attaching to a *passivated* root brings it
            # back rather than reporting it gone (P5-05). `start` returns the
            # live root untouched when there is one, and resumes from the log
            # when there is not — the same path `session/prompt` takes, which is
            # what keeps rehydration one mechanism instead of two.
            return self._attach(
                await supervisor.start(str(params["sessionId"])), params.get("cursor")
            )
        if method == "session/status":
            return self._status(str(params["sessionId"]))
        if method == "session/detach":
            return self._detach(str(params["sessionId"]))
        if method == "session/snapshot":
            return self._snapshot(str(params["sessionId"]), params.get("cursor"))
        # The schedule seam over the wire (P5-06, P5-10). Create and cancel go
        # through the supervisor rather than the seam directly: both need the
        # root mounted and the append flushed, and a schedule that lives only in
        # a buffer is one a restart forgets.
        if method == "schedule/create":
            created = await supervisor.schedule(
                str(params["sessionId"]),
                Schedule(
                    id=str(params["scheduleId"]),
                    kind=params["kind"],
                    spec=str(params["spec"]),
                    prompt=str(params["prompt"]),
                ),
            )
            return created.to_wire()
        if method == "schedule/cancel":
            session_id, schedule_id = str(params["sessionId"]), str(params["scheduleId"])
            cancelled = await supervisor.unschedule(session_id, schedule_id)
            return {"sessionId": session_id, "scheduleId": schedule_id, "cancelled": cancelled}
        if method == "schedule/list":
            root = self._root(str(params["sessionId"]))
            return {"sessionId": root.id, "schedules": supervisor.scheduled(root)}
        if method == "daemon/status":
            return self.server.status()
        if method == "shutdown":
            # Actually stops it, and takes no id by contract: a client awaiting
            # a reply would be waiting on a frame the daemon is concurrently
            # losing the ability to write. "Stop" is not a question.
            self.server.stop.set()
            return {"ok": True}
        raise UnknownMethod(f'unknown method "{method}"')

    def _root(self, session_id: str) -> Any:
        root = self.server.supervisor.roots.get(session_id)
        if root is None:
            raise NoSuchSession(f'no session "{session_id}"')
        return root

    def _attach(self, root: Any, cursor: Any) -> dict[str, Any]:
        """Subscribe to what happens *next*, and say where that starts.

        **Attach does not replay.** Streaming the gap here — one `session.event` frame per
        event into a fixed-size outbox with no await point — makes a client reattaching to
        a root that has moved on get a `WouldBlock` out of its own attach, *after* the
        subscription has been made.

        So catch-up has one mechanism, and it is the paged one: the reply says where the
        live stream begins, and the client reads `session/snapshot` from its cursor up to
        that point. That also makes the 512 KiB-class bound apply to replay.
        """
        # The root, not an id to look up again: `start` has just returned it, and
        # re-deriving it would keep a `no_such_session` branch `start` has already
        # made unreachable.
        if root.id not in self.attached:
            self.attached.add(root.id)
            root.subscribe(self.notify)
        return {
            **root.describe(),
            # Where this client is being resumed from, said out loud: a stale
            # generation silently means "from the beginning", and a client
            # should not have to infer that from sequence numbers arriving in
            # an order it did not expect.
            "from": resume_at(root.session, cursor),
        }

    def _snapshot(self, session_id: str, cursor: Any) -> dict[str, Any]:
        """One bounded page of a session's history, and the cursor for the next."""
        root = self._root(session_id)
        start = resume_at(root.session, cursor)
        events = [
            event.to_wire(thaw=False) for event in root.session.events_from(start)[:SNAPSHOT_EVENTS]
        ]
        return {
            "sessionId": session_id,
            "events": events,
            "cursor": cursor_of(root.session, start + len(events)),
            "more": start + len(events) < root.session.seq,
        }

    def _status(self, session_id: str) -> dict[str, Any]:
        """One root in detail — what `sessions/list` says, and why it says it.

        The listing carries what a table needs for every root; this carries what
        a person asks about *one*. The retry ladder is the reason it exists:
        `status` collapses to `"retrying"` or `"failed"`, and the two questions
        that follow — how many attempts, and what is still going to fire — have
        no other way to be asked.
        """
        root = self._root(session_id)
        return {**root.detail(), "schedules": self.server.supervisor.scheduled(root)}

    def _detach(self, session_id: str) -> dict[str, Any]:
        was_attached = session_id in self.attached
        self.attached.discard(session_id)
        root = self.server.supervisor.roots.get(session_id)
        if root is not None:
            root.unsubscribe(self.notify)
        # Deliberately *not* an error when nothing was attached: detach is what a
        # client does while tidying up, often twice, and a teardown path that
        # raises is one nobody can write correctly.
        return {"sessionId": session_id, "detached": was_attached}


@dataclass(slots=True)
class DaemonServer:
    """The supervisor behind the socket, and the event that ends the run."""

    supervisor: Supervisor
    stop: anyio.Event
    path: Path
    """The socket this is answering on, so `daemon/status` can say so.

    A client resolves the same path to connect, but "which socket am I actually
    talking to" is the first question anyone debugging two daemons asks, and an
    answer derived a second time on the client side would agree with the server
    by assumption rather than by evidence."""
    tick_every: float = TICK_EVERY
    sweep_every: float = SWEEP_EVERY
    heartbeat_every: float = HEARTBEAT_EVERY
    watch_every: float = WATCH_EVERY
    """The four cadences, named rather than a tuple: `serve` already threads
    them past each other positionally into `start_soon`, and this is the one
    place they are read back by a person."""
    started: int = field(default_factory=now_ms)
    """When this daemon came up, for the uptime a status reply carries."""
    identity: tuple[int, int] | None = None
    """`(st_dev, st_ino)` of the socket at bind time — what `watch` compares against.

    Captured by `serve` rather than read here, because "the socket I bound" is a
    fact about a moment and this object outlives it: reading the path on first
    use would adopt whatever is there by then, which is the exact substitution
    the watch exists to catch."""
    unreachable_since: int | None = None
    """When the socket stopped being this daemon's, or `None` while it still is.

    A latch, not a sample. The transition is one-way by construction — an
    unlinked socket's inode does not come back, and a path re-created by anyone
    else is somebody else's — so this is set once, announced once, and read
    thereafter by `status` for whoever eventually gets to ask."""

    def status(self) -> dict[str, Any]:
        """What this daemon is, for `ph agents doctor`.

        Everything here is read from the running process rather than re-derived
        by the client: the socket it bound, the policy it was started with, the
        cadences it is actually running. A doctor that reported what the *client*
        would have chosen would agree with a daemon started differently and say
        nothing at all.
        """
        supervisor = self.supervisor
        return {
            **capabilities(*CAPABILITIES),
            "pid": os.getpid(),
            "socket": str(self.path),
            "uptimeMs": now_ms() - self.started,
            "roots": len(supervisor.roots),
            "provider": supervisor.provider,
            "model": supervisor.model,
            "passivateAfter": supervisor.passivate_after,
            "tickEvery": self.tick_every,
            "sweepEvery": self.sweep_every,
            "heartbeatEvery": self.heartbeat_every,
            "watchEvery": self.watch_every,
            "unreachableSince": self.unreachable_since,
            # `DiagnosticsRegistry.report()`'s shape verbatim — a list of
            # sections, each a title and `(label, value)` rows — carrying one
            # built-in section today (P5-11's socket lifetime). One encoding of
            # one fact, so nothing on the wire can disagree with itself and the
            # client needs no bespoke decoder; P5-12 fills the same envelope from
            # the daemon's *mounted* registry with no client change.
            "sections": [
                {
                    "title": title,
                    "rows": [{"label": label, "value": value} for label, value in rows],
                }
                for title, rows in self.report()
            ],
        }

    def report(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """The daemon's own diagnostic sections, in `report()`'s shape.

        A list because there is already more than one, and P5-12's arrival is
        what the shape was built for: `sections` reached the client through one
        loop, so the isolation section cost a tuple entry here and nothing at
        all on the other side.
        """
        return [
            ("socket lifetime", self.socket_lifetime().describe()),
            # The daemon's own non-guarantees (N5, I-2), printed by the command a
            # person runs to ask what this daemon is. Rule 6 wants them beside
            # where they would be assumed, and this reply *is* that place: it
            # says "roots: 7" two rows up, which is the sentence that invites
            # every assumption these rows correct.
            ("isolation", list(NON_GUARANTEES)),
        ]

    def socket_lifetime(self) -> RuntimeLifetime:
        """Whether *this* socket survives logout — asked of the path it bound.

        `serve()` takes an explicit path, so the socket a daemon is answering on
        and the one `resolve_roots()` derives are not always the same file. A
        daemon that reported the lifetime of a directory it is not using would
        be the client-side re-derivation `daemon/status` exists to prevent,
        moved inside the server.

        Read per call rather than captured at start: lingering can be enabled
        while a daemon runs, and a doctor that answered from a snapshot taken at
        boot would keep telling a person to run the command they just ran.
        """
        return lifetime(self.path)

    async def check_reachable(self) -> str:
        """One watch pass: is the socket at our path still ours? (P5-11)

        `""` when it is. Two shapes of no, wanting the same record and not the same
        sentence: `removed` is logout reaping `$XDG_RUNTIME_DIR` out from under a daemon
        that keeps running, and `replaced` is what happens next — the person logs back in,
        `ph daemon` binds a *new* socket at the same path, and two supervisors now believe
        they own this user's roots. An existence check reads the second as a recovery,
        which is why the identity is a `(dev, inode)` pair rather than a boolean.

        **Deliberately not a shutdown.** The roots keep working — their tasks hold no
        reference to a connection, which is P5-01's whole inversion — and ending an hour
        of in-flight work over a socket problem would be this row's own failure mode
        arriving from the other side.
        """
        if self.identity is None or self.unreachable_since is not None:
            # Nothing to compare against (a caller that built this by hand), or
            # already latched. Either way there is no transition to find, and the
            # `lstat` below is skipped for the life of the process.
            return ""
        current = socket_identity(self.path)
        if current == self.identity:
            return ""
        reason = "removed" if current is None else "replaced"
        life = self.socket_lifetime()
        self.unreachable_since = now_ms()
        note = {
            "reason": reason,
            "socket": str(self.path),
            # Which daemon, and which incident. Ten roots get ten records with
            # ten `Session`-assigned timestamps, and without these the only way
            # to ask "were these two sessions in the same incident" afterwards is
            # to correlate on clock times and a payload that happens to be equal.
            "pid": os.getpid(),
            "since": self.unreachable_since,
            "tier": life.tier,
            "linger": life.linger,
            "advice": life.advice,
        }
        # `error`, not `warning`: for a daemon started from a terminal this line
        # is the only place the news lands at the moment it happens, and it is
        # the difference between "the agents stopped answering" and a sentence
        # naming the command that prevents it next time.
        log.error(
            "ph_app.daemon: %s is no longer this daemon's socket (%s) — "
            "clients cannot reach %d root(s); %s",
            self.path,
            reason,
            len(self.supervisor.roots),
            life.advice or "the roots keep running",
        )
        await self.supervisor.announce_unreachable(note)
        return reason

    async def _handle(self, stream: ByteStream) -> None:
        async with stream:
            await _Connection(stream=stream, server=self).serve()


async def _every(
    seconds: float, stop: anyio.Event, work: Callable[..., Awaitable[Any]], what: str
) -> None:
    """Run `work` on a fixed cadence until the daemon stops.

    **One reading of `stop`, not three**: the timeout falls through to the work
    and a set event returns, so the loop condition and a trailing guard cannot
    disagree about what "stopped" means. That reasoning was written once and
    then depended on twice — the sweeper and the ticker were the same seven
    lines with different bodies — so any change to shutdown semantics needed
    two edits and nothing would have noticed one of them being missed.

    A failing pass is logged and the cadence continues: work that raised would
    otherwise take the task group with it, and with it every root, over a
    housekeeping pass. That is the reasoning `_drive` contains a crash for.
    """
    while True:
        with anyio.move_on_after(seconds):
            await stop.wait()
            return
        try:
            await work()
        except Exception:
            log.exception("ph_app.daemon: %s failed", what)


async def _clear_stale(path: Path) -> None:
    """Remove a socket nobody is listening on; refuse one somebody is.

    A stale path is the ordinary aftermath of a crash and makes every client
    hang on a connect that is never answered. A *live* one is another daemon,
    and taking its socket would leave two supervisors both believing they own
    this user's roots — which is I-5's question and P5-03's to answer, so here
    it is a refusal rather than a race.
    """
    if not path.exists():
        return
    try:
        stream = await anyio.connect_unix(str(path))
    except (ConnectionRefusedError, FileNotFoundError, OSError):
        path.unlink(missing_ok=True)
        return
    await stream.aclose()
    raise DaemonUnavailable(f"a daemon is already listening on {path}")


async def serve(
    documents: Sequence[ProfileLayer],
    *,
    provider: str = "fake",
    model: str = "fake-1",
    passivate_after: float | None = PASSIVATE_AFTER,
    wake_within: float | None = WAKE_WITHIN,
    sweep_every: float = SWEEP_EVERY,
    tick_every: float = TICK_EVERY,
    heartbeat_every: float = HEARTBEAT_EVERY,
    watch_every: float = WATCH_EVERY,
    path: Path | None = None,
    ready: anyio.Event | None = None,
    started: Callable[[DaemonServer], None] | None = None,
) -> None:
    """Run the supervisor until `shutdown`.

    `ready` is an `anyio.Event` set once the socket is accepting, so a caller —
    a test, or `ph agents` starting a daemon on demand — can wait for the door
    to open rather than poll for the file to appear. The file exists before it
    is listening, which is exactly the window a poll would land in.
    """
    socket_path = path or resolve_roots().ensure().daemon_socket()
    await _clear_stale(socket_path)
    # Bound *before* the task group, beside the stale check it belongs with:
    # binding is a precondition, and a precondition that fails inside a group
    # comes back wrapped in an `ExceptionGroup` that `ph daemon`'s `except`
    # cannot see. A `$PH_RUNTIME` deep enough to exceed `AF_UNIX`'s 107-byte
    # path limit printed a full traceback for exactly that reason. Nothing has
    # been built yet at this point, so there is nothing for the teardown below
    # to have cleaned up either.
    try:
        listener = await anyio.create_unix_listener(socket_path)
        # The socket carries every command this user's agents will take, so it
        # is theirs alone — the same reasoning `$PH_RUNTIME` is 0o700 for.
        os.chmod(socket_path, 0o600)
    except OSError as error:
        raise DaemonUnavailable(f"cannot listen on {socket_path}: {error}") from error
    async with anyio.create_task_group() as tasks:
        # Built inside the group so `tasks` is a required field rather than an
        # Optional with a "not serving" guard: a supervisor that cannot start a
        # root is a state that should not be representable.
        supervisor = Supervisor(
            documents=list(documents),
            tasks=tasks,
            provider=provider,
            model=model,
            passivate_after=passivate_after,
            wake_within=wake_within,
        )
        try:
            server = DaemonServer(
                supervisor=supervisor,
                stop=anyio.Event(),
                path=socket_path,
                tick_every=tick_every,
                sweep_every=sweep_every,
                heartbeat_every=heartbeat_every,
                watch_every=watch_every,
                # Taken here, immediately after the bind and before anything can
                # have replaced it — the one moment at which "the socket at this
                # path" and "the socket this daemon is listening on" are the
                # same file by construction rather than by assumption (P5-11).
                identity=socket_identity(socket_path),
            )
            # Four cadences, four tasks, one primitive: a cadence riding another's
            # counter advances only when that one *succeeds*, so a run of failing
            # ticks would starve an unrelated record.
            if passivate_after is not None:
                tasks.start_soon(_every, sweep_every, server.stop, supervisor.sweep, "the sweep")
            if tick_every > 0:
                # Woken before fired, and on the tick's own cadence rather than
                # once at boot: a session can gain an appointment at any moment
                # from a `ph -p` run in another process, and a root the sweeper
                # released is unmounted again by the time its next one is due
                # (P6-23). One small file read per pass buys both.
                await supervisor.rehydrate()
                tasks.start_soon(
                    _every, tick_every, server.stop, supervisor.wake_and_tick, "the tick"
                )
                tasks.start_soon(
                    _every, heartbeat_every, server.stop, supervisor.heartbeat, "the heartbeat"
                )
            if watch_every > 0:
                # Its own cadence and its own `if`, not a rider on the tick: a test
                # that turns the scheduler off to keep a timer out of its assertions
                # must not thereby turn off the thing that notices the daemon has no
                # door.
                tasks.start_soon(
                    _every, watch_every, server.stop, server.check_reachable, "the socket watch"
                )
            if started is not None:
                # Handed out rather than reachable through the socket: a test
                # whose subject is the supervisor's own concurrency has no wire
                # question to ask, and reaching it through a client would be
                # testing the transport to get at something behind it.
                started(server)
            async with listener:
                tasks.start_soon(listener.serve, server._handle)
                if ready is not None:
                    ready.set()
                await server.stop.wait()
        finally:
            # Shielded, and bounded. `shutdown` is a notification — the caller
            # does not wait for it — so a cancel from whoever started `serve()`
            # routinely arrives *while* this is unwinding, and an unwinding cut
            # short here loses everything teardown is for: sessions unflushed,
            # worktrees unreclaimed (F6), and P5-03's leases never released, so
            # the next daemon refuses a session whose holder is already gone.
            # `move_on_after` rather than a bare shield, because a root that
            # will not unwind must not become a process that will not exit.
            # `GRACE_SECONDS` rather than a number of its own: it is the same
            # budget `install_lifecycle` spends on `ctx.dispose()`, with the
            # same `move_on_after(shield=True)`, and two constants for one
            # tunable is how an inner budget silently becomes dead code (when
            # it exceeds the outer one) or the only one that ever fires.
            with anyio.move_on_after(GRACE_SECONDS, shield=True):
                await supervisor.aclose()
            socket_path.unlink(missing_ok=True)
            # Last: the accept loop and any root task still in flight. Roots are
            # unwound above by their own channels closing, so this cancels a
            # listener rather than a turn.
            tasks.cancel_scope.cancel()
