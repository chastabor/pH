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
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anyio
from anyio.abc import ByteStream

from ph.paths import resolve_roots

from .framing import FramingError, read_frames, write_frame
from .supervisor import Supervisor

__all__ = ["DaemonServer", "serve"]

log = logging.getLogger("ph_app.daemon")

PROTOCOL_VERSION = 1


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
        self.outbox.send_nowait({"method": method, "params": params})

    async def _handle(self, request: dict[str, Any]) -> None:
        request_id = request.get("id")
        method = str(request.get("method", ""))
        params = request.get("params") or {}
        try:
            result = await self._dispatch(method, params)
        except Exception as error:
            if request_id is not None and self.outbox is not None:
                self.outbox.send_nowait(
                    {"id": request_id, "error": {"code": -32000, "message": str(error)}}
                )
            return
        if request_id is not None and self.outbox is not None:
            self.outbox.send_nowait({"id": request_id, "result": result})

    async def _dispatch(self, method: str, params: dict[str, Any]) -> Any:
        supervisor = self.server.supervisor
        if method == "initialize":
            return {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {"roots": True, "attach": True, "streaming": True},
            }
        if method == "roots/list":
            return {"roots": supervisor.describe()}
        if method == "root/start":
            root = await supervisor.start(str(params["rootId"]))
            return root.describe()
        if method == "root/prompt":
            root = await supervisor.prompt(str(params["rootId"]), str(params.get("prompt", "")))
            return root.describe()
        if method == "root/attach":
            return self._attach(str(params["rootId"]), int(params.get("replay", 0)))
        if method == "root/detach":
            return self._detach(str(params["rootId"]))
        if method == "shutdown":
            # Actually stops it. An earlier draft set a flag nothing read, so
            # `serve`'s "runs until shutdown" and the CLI's "blocks until a
            # client sends shutdown" were both false, and every test cancelled
            # the scope by hand to terminate.
            #
            # Sent without an id, by contract: a client awaiting a reply would
            # be waiting on a frame the daemon is concurrently tearing down the
            # ability to write. "Stop" is not a question.
            self.server.stop.set()
            return {"ok": True}
        raise ValueError(f'unknown method "{method}"')

    def _attach(self, root_id: str, replay: int) -> dict[str, Any]:
        root = self.server.supervisor.roots.get(root_id)
        if root is None:
            raise ValueError(f'no root "{root_id}"')
        if root_id in self.attached:
            return root.describe()
        self.attached.add(root_id)
        root.subscribe(self.notify)
        if replay:
            # What happened while nobody was watching. A cursor rather than a
            # count is P5-02's; this is enough to prove a reattachment sees the
            # work it missed.
            # `events_from`, not a slice of `events`: the latter materializes a
            # snapshot of the whole log to keep its tail, which is 1.3 ms on a
            # 200 000-event root and is what P5-02's cursor will index into.
            for event in root.session.events_from(max(0, root.session.seq - replay)):
                self.notify("session.event", {"rootId": root_id, "event": event.to_wire()})
        return root.describe()

    def _detach(self, root_id: str) -> dict[str, Any]:
        was_attached = root_id in self.attached
        self.attached.discard(root_id)
        root = self.server.supervisor.roots.get(root_id)
        if root is not None:
            root.unsubscribe(self.notify)
        # Deliberately *not* an error when nothing was attached: detach is what a
        # client does while tidying up, often twice, and a teardown path that
        # raises is one nobody can write correctly.
        return {"rootId": root_id, "detached": was_attached}


@dataclass(slots=True)
class DaemonServer:
    """The supervisor behind the socket, and the event that ends the run."""

    supervisor: Supervisor
    stop: anyio.Event

    async def _handle(self, stream: ByteStream) -> None:
        async with stream:
            await _Connection(stream=stream, server=self).serve()


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
    path: Path | None = None,
    ready: anyio.Event | None = None,
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
            documents=list(documents), tasks=tasks, provider=provider, model=model
        )
        try:
            listener = await anyio.create_unix_listener(socket_path)
            # The socket carries every command this user's agents will take, so
            # it is theirs alone — the same reasoning `$PH_RUNTIME` is 0o700 for.
            os.chmod(socket_path, 0o600)
            server = DaemonServer(supervisor=supervisor, stop=anyio.Event())
            async with listener:
                tasks.start_soon(listener.serve, server._handle)
                if ready is not None:
                    ready.set()
                await server.stop.wait()
        finally:
            await supervisor.aclose()
            socket_path.unlink(missing_ok=True)
            # Last: the accept loop and any root task still in flight. Roots are
            # unwound above by their own channels closing, so this cancels a
            # listener rather than a turn.
            tasks.cancel_scope.cancel()
