"""`$PH_RUNTIME/daemon.sock` — the supervisor's front door (P5-01).

A unix socket rather than stdio, which is the whole point: stdio has exactly one
peer and dies with it, and a daemon exists to be reconnected to. The method
names extend `--mode rpc`'s shape rather than starting a second vocabulary, so
the SDK client dsh already ships stays usable and P5-02 adds capabilities to
*this* surface instead of a parallel one.

**The socket is state, and stale state is a lie.** A path left behind by a
crashed daemon makes every client hang on a connect that will never be answered,
so binding removes an unresponsive one first and refuses a responsive one — the
second is another daemon, which is P5-03's lease to arbitrate rather than this
row's to overwrite.

@module ph_app.daemon.server
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream

from ph.paths import resolve_roots
from ph.resources import GRACE_SECONDS

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
from .recovery import PASSIVATE_AFTER
from .supervisor import Supervisor

__all__ = ["DaemonServer", "serve"]

log = logging.getLogger("ph_app.daemon")

SWEEP_EVERY = 60.0
"""How often the passivation sweep runs. A coarse tick, not a second timeout."""


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

        Raising is the point. An earlier draft caught `WouldBlock` here, logged
        "dropped", and returned — so nothing was dropped, and the watcher that
        could not keep up re-paid the whole fan-out for every later event. The
        subscriber list belongs to the root, so the root is what removes from
        it; this only has to fail loudly enough to be noticed.
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
            return capabilities("roots", "attach", "cursors", "snapshots")
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
        if method == "session/detach":
            return self._detach(str(params["sessionId"]))
        if method == "session/snapshot":
            return self._snapshot(str(params["sessionId"]), params.get("cursor"))
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

        **Attach does not replay.** The first draft streamed the whole gap here,
        one `session.event` frame per event, straight into a 1024-slot outbox
        with no await point — so a client reattaching to a root that had moved
        on by more than a thousand events got a `WouldBlock` out of its own
        attach, after the subscription had already been made. Measured: it
        failed at exactly 1 025. The gate test passed only because its log was
        three events long.

        So catch-up has one mechanism, and it is the paged one: the reply says
        where the live stream begins, and the client reads `session/snapshot`
        from its cursor up to that point. That also makes the 512 KiB-class
        bound apply to replay, which it never did before.
        """
        # The root, not an id to look up again: `start` has just returned it,
        # and re-deriving it kept a `no_such_session` branch `start` had already
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

    async def _handle(self, stream: ByteStream) -> None:
        async with stream:
            await _Connection(stream=stream, server=self).serve()


async def _sweeper(supervisor: Supervisor, every: float, stop: anyio.Event) -> None:
    """Release idle roots on a timer, until the daemon stops (P5-05).

    The interval is not the timeout: `after` decides eligibility from the log's
    own clock, so a coarse sweep releases a root a little late rather than
    letting the two numbers drift into one meaning. That also makes the sweep
    cheap enough to be uninteresting — it reads a status, a set and a tail
    event per root.

    Bounded by the same `stop` the accept loop waits on, so shutting down does
    not depend on a sleep elapsing.
    """
    while True:
        # One reading of `stop`, not three: the timeout falls through to the
        # sweep and a set event returns, so the loop condition and a trailing
        # guard cannot disagree about what "stopped" means.
        with anyio.move_on_after(every):
            await stop.wait()
            return
        try:
            # Not logged again here: `passivate` already says which root went
            # and how long it had been quiet. The ids come back so a test — and
            # P5-10's `ph agents` — can ask without parsing a log line.
            await supervisor.sweep()
        except Exception:
            # A sweep that raised would take the task group with it — every
            # root, over a housekeeping pass. One bad root must not be a dead
            # daemon, which is the same reasoning `_drive` contains a crash for.
            log.exception("ph_app.daemon: the passivation sweep failed")


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
    raise RuntimeError(f"a daemon is already listening on {path}")


async def serve(
    documents: Sequence[Path],
    *,
    provider: str = "fake",
    model: str = "fake-1",
    passivate_after: float | None = PASSIVATE_AFTER,
    sweep_every: float = SWEEP_EVERY,
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
        )
        try:
            listener = await anyio.create_unix_listener(socket_path)
            # The socket carries every command this user's agents will take, so
            # it is theirs alone — the same reasoning `$PH_RUNTIME` is 0o700 for.
            os.chmod(socket_path, 0o600)
            server = DaemonServer(supervisor=supervisor, stop=anyio.Event())
            if passivate_after is not None:
                tasks.start_soon(_sweeper, supervisor, sweep_every, server.stop)
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
