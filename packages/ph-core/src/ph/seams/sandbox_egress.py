"""`sandbox_egress` — the network allowlist, enforced from outside the sandbox.

A confined command lives in an unshared network namespace: one loopback
interface, no route anywhere. That is the boundary, and it is a namespace rather
than a filter, so nothing inside can talk past it. What this module adds is the
**one door**: a filtering HTTP proxy on the host, listening on a unix socket the
sandbox can reach through its read-only view of `/`, and inside the sandbox a
`socat` shim that turns `127.0.0.1:<port>` into that socket so the proxy variables
every HTTP client honours have something to point at. Where the sandbox is a
*deny list* rather than a namespace — Seatbelt on macOS — there is no separate
loopback to put a shim on, so the same proxy also listens on the host's
`127.0.0.1:<port>` and the profile allows that one remote and nothing else
(`EgressProxy.loopback`).

**The proxy decides; the namespace enforces.** A `CONNECT example.com:443` is
checked against the deployment's allowances — asked of the seam per connection,
so `/sandbox allow host` takes effect on the next request with nothing restarted
— and refused with a 403 or tunnelled byte for byte. A plain `http://` request in
absolute form is forwarded the same way, one request per connection. There is no
inspection past the host and port: TLS stays end to end, and the proxy holds no
certificate for anything.

**What it does not do is a stated limit, not a gap.** A command that ignores
`HTTPS_PROXY` — `ssh`, a raw socket, anything speaking a protocol that is not
HTTP — reaches nothing, and the seam records what it printed when it tried. That
is the closed direction, and the only honest one for a filter: a proxy that tried
to be transparent would have to become a router, and then the namespace would no
longer be the enforcement.

Refusals are **records**, not log lines. The proxy reports each one to whoever
mounted it, with the host, the port and the agent the proxy URL named, and the
seam appends `sandbox/denied` to that agent's session — which is how the TUI
comes to show a person exactly which host to allow.

**Plain `asyncio` streams, deliberately, in a codebase that is otherwise anyio.**
The server outlives every call and dies with its row, so it cannot be a detached
coroutine (`ctx.drain` would wait on it forever) and runs as a task of its own.
A task anyio did not start is a *foreign* task to anyio, and a task group with
half-closes and deadlines inside one fails in anyio's own machinery
(`InvalidStateError`, measured); the stream API the standard library gives every
asyncio task has no such condition, and it is what the first working spike used.
Nothing here is reached from anyio code except `start` and `aclose`.

Measured before it was written: from inside `bwrap --unshare-net`, a unix socket
under `$XDG_RUNTIME_DIR` is connectable through the read-only bind with no mount
of its own; `curl` through the shim reached an allowed host and was refused an
unlisted one with the proxy's own 403; a raw `connect()` to an address got
`Network is unreachable`; and the shim died with the PID namespace when the
command exited.

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
"""The one host the proxy refuses whatever the allowances say, and never records.

`sandbox-local` sends a `CONNECT` to it from *inside* a confined shell to prove
the whole path — shim, socket, proxy — before claiming the bridge. `.invalid` is
reserved (RFC 2606), so refusing it costs nobody anything, and refusing it
unconditionally is what makes the probe's verdict independent of the deployment's
allowlist: under `full` the proxy would otherwise try to resolve a name that
cannot resolve and answer 502, which is not the answer being checked.
"""

HEAD_LIMIT = 16 * 1024
"""How much request head the proxy reads before answering 431."""
HEAD_TIMEOUT = 15.0
"""How long a client has to finish sending its request head."""
CONNECT_TIMEOUT = 10.0
"""How long the upstream connect may take before the client is told 502."""
LINGER_AFTER_EOF = 5.0
"""How long the other direction of a tunnel has to finish once one side has hung
up. A client that has closed is not going to read what arrives; a server that has
closed is not going to read what is sent."""

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
"""Headers that were addressed to the proxy, dropped before a request is forwarded.
`proxy-authorization` is the agent's identity and must not reach the origin."""

Permits = Callable[[str, int], bool]
"""`permits(host, port)` — the seam's answer, asked per connection."""
OnDenied = Callable[[str, int, str | None], None]
"""`on_denied(host, port, agent)` — what a refusal is reported to."""


@dataclass(frozen=True, slots=True)
class ProxyRequest:
    """One request head, as the proxy read it."""

    method: str
    target: str
    headers: tuple[tuple[str, str], ...]
    """Header lines as sent, name case preserved; matched case-insensitively."""

    @property
    def agent(self) -> str | None:
        """Who the proxy URL named, from `Proxy-Authorization: Basic <user>:<pw>`.

        The backend puts the agent id in the proxy URL's user part (with a fixed
        password, because `urllib` sends the header only when both parts are
        there), and every client that honours the URL sends it back as Basic
        credentials. The password is ignored. Best effort: a client that sends
        nothing is an agent the refusal cannot be charged to, and the seam logs
        rather than drops it.
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
        """The host and port this request wants, or `None` for one the proxy will
        not carry.

        `CONNECT host:port` is the HTTPS shape and the ordinary one. An absolute-form
        `GET http://host/path` is the plain-HTTP shape; `https://` in absolute form
        is refused because no client sends it and honouring it would mean
        terminating TLS here.
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
    """An absolute-form request rewritten for the origin it names.

    The path replaces the URL, the hop-by-hop headers go, and `Connection: close`
    is set because this proxy carries one request per connection — the simplest
    shape that is correct, and the one the shim's `fork` already assumes.
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
    """The filtering proxy: one unix socket (and, asked for, one loopback port), one
    `permits` question per connection.

    **Its own task, not `ctx.detach`.** A detached coroutine is one `ctx.drain`
    waits for, and the hosts drain *before* they dispose — so a server that runs
    until its row unmounts would hold every shutdown open. The listener's lifetime
    is the row's, ended by `aclose` from the row's disposer, and nothing else
    needs to know it exists.
    """

    path: Path
    permits: Permits
    on_denied: OnDenied | None = None
    """Attached once the bridge has been probed, so the probe's own refusal is
    never charged to anyone."""
    loopback: bool = False
    """Also listen on `127.0.0.1:<ephemeral>`, for a backend with no network
    namespace to put a shim in.

    Seatbelt confines by *denying* rather than by unsharing: a confined command on
    macOS shares the host's loopback, so the proxy can be reached there directly
    and the profile allows exactly one remote — `localhost:<tcp_port>` — and refuses
    every other address. Measured 2026-09-07: the allowed port answers, the next
    port is `Operation not permitted`, DNS is refused, and `curl` honouring
    `HTTPS_PROXY` gets the proxy's own 403.

    Nothing the host does not already have is exposed by the listener: it tunnels
    only to hosts the deployment allows, and every process on this machine can
    reach those directly. What it costs is attribution — a local process that dials
    the port with a made-up user is charged to that name — which the unix socket's
    `0700` directory does not allow and this does. Off by default, so a `bwrap`
    host (whose sandboxes cannot reach it anyway) opens no port."""
    denied: int = 0
    """How many connections this proxy has refused, for `ph doctor`."""
    _server: asyncio.AbstractServer | None = field(default=None, repr=False)
    _tcp: asyncio.Server | None = field(default=None, repr=False)
    """`Server`, not `AbstractServer`: `tcp_port` reads `.sockets`, which only the
    concrete class carries — and `start_server` returns exactly that."""
    _connections: set[asyncio.Task[object]] = field(default_factory=set, repr=False)
    """Every connection being served, so `aclose` can end the tunnels the server's
    own `close` leaves running."""

    @property
    def tcp_port(self) -> int | None:
        """Where the loopback listener landed, or `None` when there is not one.

        Derived from the listener rather than copied beside it: the port *is* the
        socket's, and a field assigned in `start` and cleared in `aclose` was two
        places to keep one fact true. `None` before `start` and after `aclose`,
        which is what `sandbox_local` reads to pick the port a command dials.
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
        # The directory is already private; this is belt and braces for a socket
        # that hands out the deployment's network.
        os.chmod(self.path, 0o600)
        if self.loopback:
            self._tcp = await asyncio.start_server(
                self._handle, host="127.0.0.1", port=0, limit=HEAD_LIMIT
            )
        # No `serve_forever` task: `start_unix_server` defaults to
        # `start_serving=True`, so the loop is already accepting and already spawns
        # `_handle` per connection. A task parked on `serve_forever`'s future runs
        # nothing, and managing it was half of `aclose`.

    async def aclose(self) -> None:
        """Stop accepting, end every tunnel, drop the socket, and take an empty
        directory with it.

        `server.close()` stops the listener but leaves the connections it already
        spawned running, which for a tunnel means until the far end hangs up — so
        the in-flight ones are cancelled here rather than outliving the row.
        """
        servers = [server for server in (self._server, self._tcp) if server is not None]
        # Cleared before the close, so `tcp_port` reads `None` from the moment the
        # row starts letting go rather than after the last tunnel drains.
        self._server = self._tcp = None
        for server in servers:
            server.close()
        pending = list(self._connections)
        for task in pending:
            # `cancel()` on a finished task is a documented no-op, so no guard.
            task.cancel()
        if pending:
            # `wait`, not `await task`: the latter re-raises the cancellation into
            # whoever is disposing the row.
            await asyncio.wait(pending, timeout=5)
        self._connections.clear()
        for server in servers:
            with suppress(Exception):
                await server.wait_closed()
        with suppress(OSError):
            self.path.unlink()
        with suppress(OSError):
            # Empty only when this was the last socket in it — or when it was the
            # private temp directory `egress_socket_path` fell back to.
            self.path.parent.rmdir()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
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
            # The shim's readiness check connects and hangs up; so does a client
            # that gave up. Neither is a request.
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
            # Whatever the client sent past its head — a body, or the TLS hello a
            # client sends without waiting — is still in `reader` and goes up
            # first, through the pump.
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
    """Copy until EOF, then half-close the other side so a server that reads to EOF
    before answering gets its EOF."""
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

    When one direction ends, the other has `LINGER_AFTER_EOF` to finish: a peer
    that has hung up is not going to read what is still arriving, and a tunnel
    left waiting on it would outlive the command it served.
    """
    up = asyncio.ensure_future(_pump(client_reader, upstream_writer))
    down = asyncio.ensure_future(_pump(upstream_reader, client_writer))
    try:
        _, pending = await asyncio.wait({up, down}, return_when=asyncio.FIRST_COMPLETED)
        if pending:
            await asyncio.wait(pending, timeout=LINGER_AFTER_EOF)
    finally:
        # The one cancellation point, and it covers whatever the deadline left
        # running — a second loop over `pending` beforehand cancelled the same
        # tasks and read as though it cancelled different ones.
        for task in (up, down):
            task.cancel()
        await asyncio.wait({up, down})
