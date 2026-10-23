"""Startup, readiness and liveness over TCP, for whatever supervises the daemon (P13-12).

**phern never restarts itself.** An orchestrator that already exists — systemd,
launchd, Kubernetes — owns the process, its environment and its backoff, and what
it needs from the daemon is three answers it can poll without a client library:
has it come up, may it be sent work, is it still alive. Kubernetes asks over HTTP;
this is the smallest responder that speaks enough of it. The same three answers
are a verb on the unix socket (`daemon/probe`) for a host with no port to spare,
and `DaemonServer.probe` is the one place each is decided, so the two doors cannot
disagree.

**Loopback by default, read only, nothing secret.** A body is a `ProbeReply` or the
daemon status, neither of which names a credential or a prompt. `/status` is the
whole `daemon/status` reply, for a person or a dashboard; poll the three probes,
which read only what the process holds. Binding beyond
loopback is the operator's choice in `--probe-bind`, and nothing here authenticates:
the answers are the same ones `phern agents doctor` prints to anyone at the socket.

**A heartbeat is what makes liveness mean anything.** `keep_heartbeat` stamps the
clock from the event loop on a short interval; a probe answered from a loop that
has not moved in three intervals is a daemon that is up and not working, which is
the one state a `tcpSocket` probe cannot see. The same task feeds systemd's watchdog
when a unit asked for one.

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
"""How many heartbeats may be missed before `live` is no: a loop that has not run
in fifteen seconds is not merely busy."""

SCHEDULER_GRACE_SECONDS = 60.0
"""How late the scheduler's planned pass may be before `live` says it is stuck. A
pass is a mount, which can take seconds; a minute past its own plan is a task that
is not coming back."""

_REQUEST_BYTES = 4096
"""The most of a request this reads: a probe is one line and a few headers."""

_REQUEST_SECONDS = 2.0
"""How long a connection may take to say what it wants."""

_PROBE_NAMES: Mapping[str, ProbeName] = literal_lookup(ProbeName)
"""The path a request names, read back as a probe or nothing."""


def parse_bind(spec: str) -> tuple[str, int]:
    """`host:port` to its halves. `[::1]:8080` and `:8080` are allowed; a port alone
    binds loopback — the web door's (`DEFAULT_HOST`), so the two cannot disagree."""
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
        # The whole head is read, so nothing is left unread when the socket closes;
        # only its first line decides anything.
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
    """Stamp the clock from the event loop until `stop`; feed systemd meanwhile.

    `READY=1` goes out once, the first time readiness is yes; `WATCHDOG=1` on every
    beat while liveness is yes, so a daemon that knows it is stuck stops feeding the
    watchdog rather than lying to it.
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
