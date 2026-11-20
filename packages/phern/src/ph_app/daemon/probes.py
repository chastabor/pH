"""Startup, readiness and liveness over TCP, for whatever supervises the daemon (P13-12).

**phern never restarts itself.** An orchestrator that already exists (systemd,
launchd, Kubernetes) owns the process, its environment and its backoff. The
orchestrator needs three answers from the daemon, and it must be able to poll
them without a client library: has the daemon started, may it be sent work, is it
still alive. Kubernetes asks over HTTP, and this module is the smallest responder
that speaks enough HTTP. The same three answers are a verb on the unix socket
(`daemon/probe`), for a host with no port to spare. `DaemonServer.probe` is the
one place that decides each answer, so the TCP responder and the socket verb
cannot disagree.

**Loopback by default, read only, nothing secret.** A body is a `ProbeReply` or
the daemon status, and neither names a credential or a prompt. `/status` is the
whole `daemon/status` reply, for a person or a dashboard. For polling, use the
three probes, which read only what the process holds. A bind beyond loopback is
the choice of the operator in `--probe-bind`. Nothing here authenticates: the
answers are the same ones that `phern agents doctor` prints to anyone at the
socket.

**A heartbeat gives liveness its meaning.** `keep_heartbeat` stamps the clock
from the event loop on a short interval. A probe that is answered from a loop
that has not moved in three intervals shows a daemon that is up and not working.
That is the one state that a `tcpSocket` probe cannot see. The same task feeds
the watchdog of systemd when a unit asked for one.

@module ph_app.daemon.probes
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from http import HTTPStatus
from typing import TYPE_CHECKING

import anyio
from anyio.abc import SocketStream
from anyio.streams.buffered import BufferedByteReceiveStream

from ph.json import JsonObject, dumps
from ph.session import now_ms
from ph.wire import literal_lookup

from ..protocol import PROBES, ProbeName
from ..web import DEFAULT_HOST
from .notify import notify, watchdog_interval

if TYPE_CHECKING:
    from .server import DaemonServer

__all__ = [
    "HEARTBEAT_SECONDS",
    "SCHEDULER_GRACE_SECONDS",
    "STALE_HEARTBEATS",
    "answer_probe",
    "keep_heartbeat",
    "parse_bind",
]

log = logging.getLogger("ph_app.daemon")

HEARTBEAT_SECONDS = 5.0
"""How often the event loop stamps `DaemonServer.heartbeat_at`."""

STALE_HEARTBEATS = 3
"""How many heartbeats may be missed before `live` is no. A loop that has not run
in fifteen seconds is not merely busy."""

SCHEDULER_GRACE_SECONDS = 60.0
"""How late the planned pass of the scheduler may be before `live` reports that
it is stuck. A pass is a mount, which can take seconds. A pass that is a minute
behind its own plan is a task that will not come back."""

_REQUEST_BYTES = 4096
"""The most of a request that is read. A probe is one line and a few headers."""

_REQUEST_SECONDS = 2.0
"""How long a connection may take to send its request."""

_PROBE_NAMES: Mapping[str, ProbeName] = literal_lookup(ProbeName)
"""Maps the path that a request names to a probe. A path with no entry is not a
probe."""


def parse_bind(spec: str) -> tuple[str, int]:
    """Split `host:port` into its halves. `[::1]:8080` and `:8080` are allowed. A
    port alone binds loopback. The loopback address is the one that the web front
    end uses (`DEFAULT_HOST`), so the two cannot disagree."""
    host, _, port_text = spec.rpartition(":")
    host = host.strip("[]") or DEFAULT_HOST
    try:
        port = int(port_text)
    except ValueError as error:
        raise ValueError(f'"{spec}" is not host:port') from error
    if not 0 <= port <= 65535:
        raise ValueError(f'"{spec}" names a port outside 0..65535')
    return host, port


async def answer_probe(server: DaemonServer, stream: SocketStream) -> None:
    """One HTTP exchange: read the request line, answer, close."""
    async with stream:
        request = b""
        # The whole head is read, so that nothing is left unread when the socket
        # closes. Only the first line of the head decides anything.
        with (
            anyio.move_on_after(_REQUEST_SECONDS),
            suppress(anyio.EndOfStream, anyio.DelimiterNotFound, anyio.IncompleteRead),
        ):
            request = await BufferedByteReceiveStream(stream).receive_until(
                b"\r\n\r\n", _REQUEST_BYTES
            )
        line = request.split(b"\r\n", 1)[0].decode("latin-1", "replace")
        parts = line.split()
        code: int
        body: JsonObject
        if len(parts) < 2:
            code, body = 400, {"error": "a request line was expected"}
        else:
            method, target = parts[0], parts[1]
            path = target.split("?", 1)[0].strip("/")
            name = _PROBE_NAMES.get(path)
            if method not in ("GET", "HEAD"):
                code, body = 405, {"error": "GET or HEAD"}
            elif path == "status":
                code, body = 200, server.status().to_wire()
            elif name is not None:
                reply = server.probe(name)
                code, body = (200 if reply.ok else 503), reply.to_wire()
            else:
                code, body = 404, {"error": f"no probe at /{path}", "probes": list(PROBES)}
        payload = dumps(body).encode("utf-8")
        head = (
            f"HTTP/1.1 {code} {HTTPStatus(code).phrase}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(payload)}\r\n"
            "Cache-Control: no-store\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii")
        try:
            await stream.send(head + (payload if parts[:1] != ["HEAD"] else b""))
        except (anyio.BrokenResourceError, anyio.ClosedResourceError):
            return


async def keep_heartbeat(server: DaemonServer, stop: anyio.Event) -> None:
    """Stamp the clock from the event loop until `stop`, and feed systemd meanwhile.

    `READY=1` is sent one time, the first time that readiness is yes.
    `WATCHDOG=1` is sent on every beat while liveness is yes. A daemon that knows
    it is stuck therefore stops feeding the watchdog and does not give it a false
    answer.
    """
    watchdog = watchdog_interval()
    interval = HEARTBEAT_SECONDS if watchdog is None else min(HEARTBEAT_SECONDS, watchdog)
    announced = False
    while not stop.is_set():
        server.heartbeat_at = now_ms()
        if not announced and server.probe("ready").ok:
            announced = True
            notify("READY=1")
        if watchdog is not None and server.probe("live").ok:
            notify("WATCHDOG=1")
        with anyio.move_on_after(interval):
            await stop.wait()
