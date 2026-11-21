"""`sandbox_egress`: the network allowlist, enforced from outside the sandbox.

A confined command runs in an unshared network namespace: one loopback
interface, and no route anywhere. That is the boundary. It is a namespace and
not a filter, so nothing inside can talk past it. This module adds the **one
way out**: a filtering HTTP proxy on the host. The proxy listens on a unix
socket that the sandbox can reach through its read-only view of `/`. Inside the
sandbox, a `socat` shim turns `127.0.0.1:<port>` into that socket, so that the
proxy variables that every HTTP client honors have something to point at.
Seatbelt on macOS is a *deny list* and not a namespace, so it has no separate
loopback to put a shim on. There, the same proxy also listens on
`127.0.0.1:<port>` of the host, and the profile allows that one remote and
nothing else (`EgressProxy.loopback`).

**The proxy decides, and the namespace enforces.** A `CONNECT example.com:443`
is checked against the allowances of the deployment. The proxy asks the seam
for each connection, so `/sandbox allow host` takes effect on the next request
and nothing restarts. The request is then refused with a 403, or tunneled byte
for byte. A plain `http://` request in absolute form is forwarded in the same
way, one request per connection. The proxy inspects nothing past the host and
the port: TLS stays end to end, and the proxy holds no certificate for
anything.

**What the proxy does not do is a stated limit, not a gap.** A command that
ignores `HTTPS_PROXY` reaches nothing: `ssh`, a raw socket, anything that
speaks a protocol that is not HTTP. The seam records what the command printed
when it tried. That is the closed direction, and the only correct one for a
filter. A proxy that tried to be transparent would have to become a router, and
then the namespace would no longer be the enforcement.

Refusals are **records**, not log lines. The proxy reports each refusal to
whoever mounted it, with the host, the port and the agent that the proxy URL
named. The seam appends `sandbox/denied` to the session of that agent, and that
is how the TUI shows a user exactly which host to allow.

**This module uses plain `asyncio` streams on purpose, in a codebase that
otherwise uses anyio.** The server outlives every call and ends with its row.
So it is not a detached coroutine that `ctx.drain` waits on (the drain would
wait forever). asyncio serves it, with one task for each connection. A task
that anyio did not start is a *foreign* task to anyio, and a task group with
half-closes and deadlines inside such a task fails in the machinery of anyio
(`InvalidStateError`, measured). The stream API that the standard library gives
every asyncio task has no such condition, and the first working spike used it.
Only `start` and `aclose` are reached from anyio code.

Measured before the module was written:

- From inside `bwrap --unshare-net`, a unix socket under `$XDG_RUNTIME_DIR` is
  connectable through the read-only bind with no mount of its own.
- `curl` through the shim reached an allowed host, and was refused an unlisted
  host with the 403 of the proxy.
- A raw `connect()` to an address got `Network is unreachable`.
- The shim died with the PID namespace when the command exited.

@module ph.seams.sandbox_egress
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from http import HTTPStatus
from pathlib import Path
from urllib.parse import unquote, urlsplit

__all__ = ["PROBE_HOST", "EgressProxy", "ProxyRequest", "origin_form", "parse_head"]

log = logging.getLogger("ph.seams.sandbox_egress")

PROBE_HOST = "ph-egress-probe.invalid"
"""The one host that the proxy refuses whatever the allowances say, and never
records.

`sandbox-local` sends a `CONNECT` to it from *inside* a confined command, to
prove the whole path to the proxy before it claims the bridge. Under `bwrap`
that path is the shim, the socket and the proxy. `.invalid` is reserved
(RFC 2606), so to refuse it costs nobody anything. Because the refusal is
unconditional, the verdict of the probe does not depend on the allowlist of the
deployment. Under `full`, the proxy would otherwise try to resolve a name that
cannot resolve and answer 502, which is not the answer that the probe checks.
"""

HEAD_LIMIT = 16 * 1024
"""How much request head the proxy reads before answering 431."""
HEAD_TIMEOUT = 15.0
"""How long a client has to finish sending its request head."""
CONNECT_TIMEOUT = 10.0
"""How long the upstream connect may take before the client is told 502."""
LINGER_AFTER_EOF = 5.0
"""How long the other direction of a tunnel has to finish after one side has
hung up. A client that has closed will not read what arrives, and a server that
has closed will not read what is sent."""

HOP_BY_HOP = frozenset(
    {
        "proxy-authorization",
        "proxy-connection",
        "connection",
        "keep-alive",
        "te",
        "trailer",
        "upgrade",
    }
)
"""Headers that were addressed to the proxy. They are dropped before a request
is forwarded. `proxy-authorization` is the identity of the agent and must not
reach the origin."""

Permits = Callable[[str, int], bool]
"""`permits(host, port)`: the answer of the seam, asked for each connection."""
OnDenied = Callable[[str, int, str | None], None]
"""`on_denied(host, port, agent)`: the callback that a refusal is reported to."""


@dataclass(frozen=True, slots=True)
class ProxyRequest:
    """One request head, as the proxy read it."""

    method: str
    target: str
    headers: tuple[tuple[str, str], ...]
    """The header lines as sent, with the case of each name preserved. Names are
    matched case-insensitively."""

    @property
    def agent(self) -> str | None:
        """The agent that the proxy URL named, from `Proxy-Authorization: Basic
        <user>:<pw>`.

        The backend puts the agent id in the user part of the proxy URL. It adds
        a fixed password, because `urllib` sends the header only when both parts
        are present. Every client that honors the URL sends the two parts back
        as Basic credentials. The password is ignored. This is best effort: when
        a client sends nothing, the refusal cannot be assigned to an agent, and
        the seam logs the refusal and does not drop it.
        """
        for name, value in self.headers:
            if name.lower() != "proxy-authorization":
                continue
            scheme, _, credential = value.strip().partition(" ")
            if scheme.lower() != "basic":
                return None
            try:
                decoded = base64.b64decode(credential.strip(), validate=True).decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                return None
            user, _, _ = decoded.partition(":")
            return unquote(user) or None
        return None

    def route(self) -> tuple[str, int] | None:
        """The host and the port that this request wants, or `None` for a request
        that the proxy will not carry.

        `CONNECT host:port` is the HTTPS form and the ordinary one. An
        absolute-form `GET http://host/path` is the plain-HTTP form. `https://`
        in absolute form is refused, because no client sends it and to honor it
        would mean terminating TLS here.
        """
        if self.method == "CONNECT":
            return _authority(self.target, default_port=443)
        parts = urlsplit(self.target)
        if parts.scheme.lower() != "http" or not parts.hostname:
            return None
        try:
            port = parts.port
        except ValueError:
            return None
        return parts.hostname, port or 80


def _authority(text: str, *, default_port: int) -> tuple[str, int] | None:
    host, separator, port = text.rpartition(":")
    if not separator or not port.isdigit():
        host, port = text, str(default_port)
    host = host.strip("[]").lower()
    return (host, int(port)) if host else None


def parse_head(head: bytes) -> ProxyRequest | None:
    """The request line and headers, or `None` for something that is not HTTP."""
    lines = head.decode("latin-1").split("\r\n")
    request_line = lines[0].split(" ")
    if len(request_line) != 3 or not request_line[2].startswith("HTTP/"):
        return None
    method, target, _ = request_line
    headers: list[tuple[str, str]] = []
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator or not name.strip():
            return None
        headers.append((name.strip(), value.strip()))
    return ProxyRequest(method=method.upper(), target=target, headers=tuple(headers))


def origin_form(request: ProxyRequest) -> bytes:
    """An absolute-form request, rewritten for the origin that it names.

    The path replaces the URL, the hop-by-hop headers are removed, and
    `Connection: close` is set, because this proxy carries one request per
    connection. That is the simplest form that is correct, and the `fork` of the
    shim already assumes it.
    """
    parts = urlsplit(request.target)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    lines = [f"{request.method} {path} HTTP/1.1"]
    has_host = False
    for name, value in request.headers:
        lowered = name.lower()
        if lowered in HOP_BY_HOP:
            continue
        if lowered == "host":
            has_host = True
        lines.append(f"{name}: {value}")
    if not has_host:
        port = f":{parts.port}" if parts.port and parts.port != 80 else ""
        lines.append(f"Host: {parts.hostname}{port}")
    lines.append("Connection: close")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


@dataclass(slots=True)
class EgressProxy:
    """The filtering proxy: one unix socket (and one loopback port, when asked
    for), and one `permits` question per connection.

    **The server is not a `ctx.detach` coroutine.** `ctx.drain` waits for a
    coroutine that is detached to be stopped by its scope, and the hosts drain
    *before* they dispose. So a server that runs until its row unmounts would
    hold every shutdown open. The lifetime of the listener is the lifetime of
    the row. `aclose`, called from the disposer of the row, ends it, and nothing
    else needs to know that it exists.
    """

    path: Path
    permits: Permits
    on_denied: OnDenied | None = None
    """Attached after the bridge has been probed, so that the refusal of the
    probe itself is never assigned to anyone."""
    loopback: bool = False
    """Also listen on `127.0.0.1:<ephemeral>`, for a backend that has no network
    namespace to put a shim in.

    Seatbelt confines by *denying* and not by unsharing. A confined command on
    macOS shares the loopback of the host, so it can reach the proxy there
    directly. The profile allows exactly one remote, `localhost:<tcp_port>`, and
    refuses every other address. Measured 2026-09-07: the allowed port answers,
    the next port is `Operation not permitted`, DNS is refused, and a `curl`
    that honors `HTTPS_PROXY` gets the 403 of the proxy.

    The listener exposes nothing that the host does not already have. It tunnels
    only to hosts that the deployment allows, and every process on this machine
    can reach those hosts directly. The cost is attribution: a local process
    that dials the port with an invented user is recorded under that name. The
    `0700` directory of the unix socket prevents that, and this listener does
    not. It is off by default, so a `bwrap` host opens no port. The sandboxes of
    such a host could not reach the port anyway."""
    denied: int = 0
    """How many connections this proxy has refused, for `phern doctor`."""
    _server: asyncio.AbstractServer | None = field(default=None, repr=False)
    _tcp: asyncio.Server | None = field(default=None, repr=False)
    """The type is `Server` and not `AbstractServer`. `tcp_port` reads
    `.sockets`, which only the concrete class has, and `start_server` returns
    exactly that class."""
    _connections: set[asyncio.Task[object]] = field(default_factory=set, repr=False)
    """Every connection that is being served, so that `aclose` can end the
    tunnels that the `close` of the server leaves running. Each task removes
    itself when it ends."""
    _closing: bool = field(default=False, repr=False)
    """Set when `aclose` begins. A connection that the listener spawned and the
    loop had not yet stepped is not in `_connections`, so `aclose` cannot cancel
    it. `_handle` closes it unserved. Before this flag existed, such a
    connection reported refusals to a row that had unmounted."""

    @property
    def tcp_port(self) -> int | None:
        """The port where the loopback listener landed, or `None` when there is
        no listener.

        It is derived from the listener and not copied beside it. The port
        belongs to the socket, and a field that `start` assigned and `aclose`
        cleared was two places to keep one fact true. It is `None` before
        `start` and after `aclose`. `sandbox_local` reads it to choose the port
        that a command dials.
        """
        if self._tcp is None:
            return None
        return int(self._tcp.sockets[0].getsockname()[1])

    async def start(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with suppress(FileNotFoundError):
            self.path.unlink()
        self._server = await asyncio.start_unix_server(
            self._handle, path=str(self.path), limit=HEAD_LIMIT
        )
        # The directory is already private. This is a second protection for a
        # socket that gives access to the network of the deployment.
        os.chmod(self.path, 0o600)
        if self.loopback:
            self._tcp = await asyncio.start_server(
                self._handle, host="127.0.0.1", port=0, limit=HEAD_LIMIT
            )
        # There is no `serve_forever` task. `start_unix_server` defaults to
        # `start_serving=True`, so the loop already accepts and already spawns
        # `_handle` for each connection. A task parked on the future of
        # `serve_forever` runs nothing, and managing it was half of `aclose`.

    async def aclose(self) -> None:
        """Stop accepting, end every tunnel, remove the socket, and remove its
        directory if the directory is empty.

        `server.close()` stops the listener but leaves the connections that it
        already spawned running. For a tunnel, that means until the far end
        hangs up. So the connections in flight are canceled here and do not
        outlive the row.
        """
        self._closing = True
        servers = [server for server in (self._server, self._tcp) if server is not None]
        # Cleared before the close, so that `tcp_port` reads `None` from the
        # moment the row starts to let go, and not after the last tunnel drains.
        self._server = self._tcp = None
        for server in servers:
            server.close()
        pending = list(self._connections)
        for task in pending:
            # `cancel()` on a finished task is a documented no-op, so no guard.
            task.cancel()
        if pending:
            # `wait`, not `await task`. `await task` re-raises the cancellation
            # into whoever disposes the row.
            await asyncio.wait(pending, timeout=5)
        for server in servers:
            with suppress(Exception):
                await server.wait_closed()
        with suppress(OSError):
            self.path.unlink()
        with suppress(OSError):
            # The directory is empty only when this was the last socket in it,
            # or when it was the private temp directory that
            # `egress_socket_path` fell back to.
            self.path.parent.rmdir()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        if self._closing:
            # The connection was accepted before the close and runs after it.
            # Nothing is served for a row that has let go, and nothing is
            # reported to it.
            await _close(writer)
            return
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        try:
            await self._serve_one(reader, writer)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("ph.seams.sandbox_egress: a connection failed")
        finally:
            if task is not None:
                self._connections.discard(task)
            await _close(writer)

    async def _serve_one(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), HEAD_TIMEOUT)
        except asyncio.LimitOverrunError:
            await _respond(writer, 431, "ph sandbox: request head too large")
            return
        except (TimeoutError, asyncio.IncompleteReadError, ConnectionError, OSError):
            # The readiness check of the shim connects and hangs up, and so does
            # a client that gave up. Neither is a request.
            return
        request = parse_head(head)
        if request is None:
            await _respond(writer, 400, "ph sandbox: not an HTTP proxy request")
            return
        route = request.route()
        if route is None:
            await _respond(
                writer, 400, f"ph sandbox: cannot route {request.method} {request.target}"
            )
            return
        host, port = route
        if host == PROBE_HOST:
            await _respond(writer, 403, f"ph sandbox: {host} is the probe host; always refused")
            return
        if not self.permits(host, port):
            self._refuse(host, port, request.agent)
            await _respond(
                writer,
                403,
                f"ph sandbox: {host}:{port} is not in the network allowlist; "
                f"/sandbox allow host {host} permits it",
            )
            return
        try:
            upstream_reader, upstream_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port), CONNECT_TIMEOUT
            )
        except (OSError, TimeoutError) as error:
            await _respond(writer, 502, f"ph sandbox: could not reach {host}:{port} ({error})")
            return
        try:
            if request.method == "CONNECT":
                writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
                await writer.drain()
            else:
                upstream_writer.write(origin_form(request))
                await upstream_writer.drain()
            # Anything that the client sent after its head is still in `reader`:
            # a body, or the TLS hello that a client sends without waiting. It
            # goes upstream first, through the pump.
            await _tunnel(reader, writer, upstream_reader, upstream_writer)
        finally:
            await _close(upstream_writer)

    def _refuse(self, host: str, port: int, agent: str | None) -> None:
        self.denied += 1
        if self.on_denied is None:
            return
        try:
            self.on_denied(host, port, agent)
        except Exception:
            log.exception("ph.seams.sandbox_egress: recording a refusal failed")


async def _respond(writer: asyncio.StreamWriter, status: int, text: str) -> None:
    body = (text + "\n").encode("utf-8")
    head = (
        f"HTTP/1.1 {status} {HTTPStatus(status).phrase}\r\n"
        "Content-Type: text/plain; charset=utf-8\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: close\r\n\r\n"
    ).encode("latin-1")
    with suppress(Exception):
        writer.write(head + body)
        await writer.drain()


async def _close(writer: asyncio.StreamWriter) -> None:
    with suppress(Exception):
        writer.close()
        await writer.wait_closed()


async def _pump(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy until EOF, then half-close the other side, so that a server that
    reads to EOF before it answers gets its EOF."""
    try:
        while chunk := await reader.read(65536):
            writer.write(chunk)
            await writer.drain()
    except (ConnectionError, OSError):
        pass
    finally:
        with suppress(Exception):
            if writer.can_write_eof():
                writer.write_eof()


async def _tunnel(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    upstream_reader: asyncio.StreamReader,
    upstream_writer: asyncio.StreamWriter,
) -> None:
    """Copy bytes both ways until both sides are done.

    When one direction ends, the other has `LINGER_AFTER_EOF` to finish. A peer
    that has hung up will not read what is still arriving, and a tunnel that
    waited on it would outlive the command that it served.
    """
    up = asyncio.ensure_future(_pump(client_reader, upstream_writer))
    down = asyncio.ensure_future(_pump(upstream_reader, client_writer))
    try:
        _, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        if pending:
            await asyncio.wait(pending, timeout=LINGER_AFTER_EOF)
    finally:
        # This is the one cancellation point, and it covers whatever the
        # deadline left running. A second loop over `pending` before this point
        # canceled the same tasks, and it read as though it canceled different
        # tasks.
        for task in (up, down):
            task.cancel()
        await asyncio.wait({up, down})
