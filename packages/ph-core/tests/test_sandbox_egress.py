"""P6-38 — the network allowlist: a namespace with one door, and the door decides.

`--unshare-net` leaves a confined command one loopback interface and no route.
What this row adds is a filtering proxy on the host, reached from inside over a
unix socket through a `socat` shim, so `HTTPS_PROXY` has something to point at
and everything not on the allowlist is refused *by the proxy* — with a record.
Seatbelt has no namespace to shim, so there the same proxy listens on the host's
loopback and the profile allows that one remote (`EgressProxy.loopback`).

Two halves, like `test_sandbox_local.py`. The proxy is exercised over a real
unix socket (and its loopback door) against a real TCP server on this host, which
needs no kernel that confines. The end-to-end half runs a confined command through
the whole path — door, proxy, host server — and skips cleanly where no backend
enforces, or where a `bwrap` host has no `socat`.
"""

from __future__ import annotations

import base64
import http.server
import json
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any

import anyio
import pytest

from ph.agent.types import AgentOptions
from ph.keys import AGENTS, SANDBOX, SESSIONS, SHELL, WORKSPACE
from ph.seams.sandbox import DENIED, Allowances, NetworkAllowance
from ph.seams.sandbox_egress import PROBE_HOST, EgressProxy, origin_form, parse_head
from ph.testing import MountProfile, report_section

pytestmark = pytest.mark.anyio

ROW = {"id": "sandbox-local", "disabled": False}


# ------------------------------------------------------------------ parsing --


def test_a_connect_names_its_host_and_port() -> None:
    request = parse_head(b"CONNECT example.com:8443 HTTP/1.1\r\nHost: example.com:8443\r\n\r\n")
    assert request is not None
    assert request.route() == ("example.com", 8443)


def test_a_connect_without_a_port_means_https() -> None:
    request = parse_head(b"CONNECT example.com HTTP/1.1\r\n\r\n")
    assert request is not None
    assert request.route() == ("example.com", 443)


def test_an_absolute_form_request_routes_to_its_origin() -> None:
    request = parse_head(b"GET http://Example.com/a?b=1 HTTP/1.1\r\nHost: example.com\r\n\r\n")
    assert request is not None
    assert request.route() == ("example.com", 80)


def test_https_in_absolute_form_is_not_carried() -> None:
    """No client sends it, and honouring it would mean terminating TLS here."""
    request = parse_head(b"GET https://example.com/ HTTP/1.1\r\n\r\n")
    assert request is not None
    assert request.route() is None


def test_something_that_is_not_http_is_refused_at_the_parser() -> None:
    assert parse_head(b"SSH-2.0-OpenSSH\r\n\r\n") is None
    assert parse_head(b"GET / HTTP/1.1\r\nno-colon-here\r\n\r\n") is None


def test_the_agent_rides_the_proxy_credentials() -> None:
    """The backend puts the agent in the proxy URL's user part; clients send it
    back as Basic credentials, and the password beside it is ignored."""
    credential = base64.b64encode(b"agent%2F7:ph").decode()
    request = parse_head(
        f"CONNECT x:443 HTTP/1.1\r\nProxy-Authorization: Basic {credential}\r\n\r\n".encode()
    )
    assert request is not None
    assert request.agent == "agent/7"


def test_origin_form_drops_what_was_addressed_to_the_proxy() -> None:
    """`Proxy-Authorization` is the agent's identity and must not reach the origin."""
    request = parse_head(
        b"GET http://example.com/p?q=1 HTTP/1.1\r\n"
        b"Proxy-Authorization: Basic abc\r\nProxy-Connection: keep-alive\r\n"
        b"Accept: */*\r\n\r\n"
    )
    assert request is not None
    head = origin_form(request).decode()
    assert head.startswith("GET /p?q=1 HTTP/1.1\r\n")
    assert "Proxy-Authorization" not in head
    assert "Proxy-Connection" not in head
    assert "Accept: */*" in head
    assert "Host: example.com" in head
    assert head.rstrip().endswith("Connection: close")


# --------------------------------------------------------------- the proxy --


class _Hello(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        body = b"hello from host"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: Any) -> None:
        return


class _HostServer:
    """A plain HTTP server on this host's loopback, for the proxy to reach."""

    def __enter__(self) -> _HostServer:
        self.server = http.server.HTTPServer(("127.0.0.1", 0), _Hello)
        self.port = self.server.server_address[1]
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.server.shutdown()
        self.server.server_close()


async def _through(proxy: EgressProxy, request: bytes) -> bytes:
    """One request through the proxy's socket; everything it sent back.

    Bounded by `move_on_after` rather than by EOF: a refusal closes, but a tunnel to
    a keep-alive server stays open, and the bytes are the answer either way.
    """
    async with await anyio.connect_unix(proxy.path) as client:
        await client.send(request)
        received = bytearray()
        with anyio.move_on_after(5):
            try:
                while True:
                    received += await client.receive(65536)
            except anyio.EndOfStream:
                pass
        return bytes(received)


async def _proxy(permits: Any, *, loopback: bool = False) -> EgressProxy:
    """A started proxy on a socket path short enough to bind.

    Not under `tmp_path`: `sun_path` is 108 bytes and pytest's tree on macOS —
    `/private/var/folders/<xx>/<32 chars>/T/pytest-of-<user>/pytest-<n>/<test>0/` — is
    already past it (`AF_UNIX path too long`, measured). A private directory of its
    own instead, which `aclose` removes once the socket is gone — the fallback
    `egress_socket_path` takes for the same reason.
    """
    home = Path(tempfile.mkdtemp(prefix="ph-px-"))
    proxy = EgressProxy(path=home / "px.sock", permits=permits, loopback=loopback)
    await proxy.start()
    return proxy


async def test_the_loopback_door_is_the_same_proxy_over_tcp() -> None:
    """Seatbelt's door (`EgressProxy.loopback`): the proxy also answers on the
    host's `127.0.0.1:<port>`, with the same 403 and the same refusal charged to the
    same agent, and the port is closed with the row. Off by default, so a `bwrap`
    host — whose sandboxes could not reach it anyway — opens nothing."""
    refused: list[tuple[str, int, str | None]] = []
    proxy = await _proxy(lambda h, p: False, loopback=True)
    proxy.on_denied = lambda h, p, a: refused.append((h, p, a))
    try:
        assert proxy.tcp_port is not None
        port = proxy.tcp_port
        credential = base64.b64encode(b"a1:ph").decode()
        request = (
            f"CONNECT example.invalid:443 HTTP/1.1\r\n"
            f"Proxy-Authorization: Basic {credential}\r\n\r\n"
        ).encode()
        async with await anyio.connect_tcp("127.0.0.1", port) as client:
            await client.send(request)
            with anyio.fail_after(5):
                answer = await client.receive(65536)
        assert answer.startswith(b"HTTP/1.1 403")
        assert refused == [("example.invalid", 443, "a1")]
    finally:
        await proxy.aclose()
    assert proxy.tcp_port is None
    with pytest.raises(OSError):
        await anyio.connect_tcp("127.0.0.1", port)

    plain = await _proxy(lambda h, p: True)
    try:
        assert plain.tcp_port is None, "no door was asked for, so no port is open"
    finally:
        await plain.aclose()


async def test_an_allowed_origin_is_reached_and_a_refused_one_gets_the_proxys_403() -> None:
    """The gate. Both directions in one test for `probe_sandbox`'s reason: a proxy
    that refused everything would pass the refusal half while carrying nothing."""
    refused: list[tuple[str, int, str | None]] = []
    with _HostServer() as host:
        proxy = await _proxy(lambda h, p: h == "127.0.0.1" and p == host.port)
        proxy.on_denied = lambda h, p, agent: refused.append((h, p, agent))
        try:
            reply = await _through(
                proxy, f"GET http://127.0.0.1:{host.port}/ HTTP/1.1\r\nHost: x\r\n\r\n".encode()
            )
            assert reply.startswith(b"HTTP/1.0 200") or reply.startswith(b"HTTP/1.1 200"), reply
            assert reply.endswith(b"hello from host")

            credential = base64.b64encode(b"a9:").decode()
            reply = await _through(
                proxy,
                (
                    "CONNECT example.invalid:443 HTTP/1.1\r\n"
                    f"Proxy-Authorization: Basic {credential}\r\n\r\n"
                ).encode(),
            )
            assert reply.startswith(b"HTTP/1.1 403"), reply
            assert b"/sandbox allow host example.invalid" in reply
            assert refused == [("example.invalid", 443, "a9")]
            assert proxy.denied == 1
        finally:
            await proxy.aclose()
    assert not proxy.path.exists(), "the socket goes with the proxy"


async def test_a_connect_to_an_allowed_host_is_a_tunnel() -> None:
    """Bytes after `200 Connection Established` go to the origin untouched — the
    HTTPS shape, exercised with a plain request so the answer is readable."""
    with _HostServer() as host:
        proxy = await _proxy(lambda h, p: True)
        try:
            reply = await _through(
                proxy,
                (
                    f"CONNECT 127.0.0.1:{host.port} HTTP/1.1\r\n\r\n"
                    "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
                ).encode(),
            )
        finally:
            await proxy.aclose()
    assert reply.startswith(b"HTTP/1.1 200 Connection Established\r\n\r\n")
    assert reply.endswith(b"hello from host")


async def test_the_probe_host_is_always_refused_and_never_recorded() -> None:
    """What lets `sandbox-local` check the bridge whatever the allowlist says."""
    refused: list[Any] = []
    proxy = await _proxy(lambda h, p: True)
    proxy.on_denied = lambda *args: refused.append(args)
    try:
        reply = await _through(proxy, f"CONNECT {PROBE_HOST}:443 HTTP/1.1\r\n\r\n".encode())
    finally:
        await proxy.aclose()
    assert reply.startswith(b"HTTP/1.1 403")
    assert refused == [] and proxy.denied == 0


async def test_a_connection_that_hangs_up_is_not_a_request() -> None:
    """The shim's readiness check connects and closes; the proxy must not answer,
    record, or fall over."""
    refused: list[Any] = []
    proxy = await _proxy(lambda h, p: False)
    proxy.on_denied = lambda *args: refused.append(args)
    try:
        async with await anyio.connect_unix(proxy.path):
            pass
        reply = await _through(proxy, b"CONNECT x:1 HTTP/1.1\r\n\r\n")
    finally:
        await proxy.aclose()
    assert reply.startswith(b"HTTP/1.1 403"), "still serving after the empty connection"
    assert len(refused) == 1


async def test_the_allowlist_is_asked_live() -> None:
    """`permits` is consulted per connection, which is what lets `/sandbox allow`
    take effect without the proxy restarting."""
    allowed: set[str] = set()
    with _HostServer() as host:
        proxy = await _proxy(lambda h, p: h in allowed)
        try:
            request = f"GET http://127.0.0.1:{host.port}/ HTTP/1.1\r\nHost: x\r\n\r\n".encode()
            assert (await _through(proxy, request)).startswith(b"HTTP/1.1 403")
            allowed.add("127.0.0.1")
            assert (await _through(proxy, request)).endswith(b"hello from host")
        finally:
            await proxy.aclose()


# ------------------------------------------------------------- end to end --


async def _bridged(mount: MountProfile, *rows: dict[str, Any]) -> Any:
    """A mount with the backend and a live bridge, or a skip that says why not."""
    ctx = await mount(ROW, *rows)
    if ctx.require(SANDBOX).provider is None:
        pytest.skip("no enforcing sandbox backend on this host")
    if ctx.require(SANDBOX).egress is None:
        pytest.skip(f"no egress bridge: {report_section(ctx, 'Local confinement').get('egress')}")
    return ctx


def _fetch(url: str) -> str:
    """A shell command that fetches `url` with the interpreter this test runs on —
    which honours `http_proxy` the way every client does. `json.dumps` for the URL:
    double quotes inside the single-quoted `-c` argument, which is what `sh` needs
    and what `repr` does not give."""
    fetch = f"urllib.request.urlopen({json.dumps(url)}, timeout=10).read().decode()"
    code = f"import urllib.request;print({fetch})"
    return f"{sys.executable} -c '{code}'"


async def test_a_confined_command_reaches_an_allowed_host_and_only_that(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The row's gate.** Through `ctx.shell`, as an agent's command runs: the
    shim comes up, the socket is reached through the read-only bind, the proxy
    carries the allowed origin and refuses the other — and the refusal lands in
    the agent's own session with the line that lifts it.

    The allowed host is this machine's loopback *as the proxy sees it*: the
    proxy connects from the host side, so `127.0.0.1` there is the test server,
    while inside the namespace the same address is the sandbox's own empty
    loopback. That is the whole mechanism in one address.
    """
    with _HostServer() as host:
        allow = {
            "id": "sandbox-allow",
            "config": {"network": {"mode": "allowlist", "hosts": ["127.0.0.1"]}},
        }
        ctx = await _bridged(mount, allow)
        session = ctx.require(SESSIONS).create("egress")
        agent = ctx.require(AGENTS).create(session, AgentOptions(provider="fake", model="f"))
        # The lifecycle row acquires at the agent's first step; a command run outside
        # a turn needs its workspace acquired by hand, as the ladder tests do.
        await ctx.require(WORKSPACE).acquire(
            session_id=session.id, agent_id=agent.id, base=tmp_path
        )

        reached = await ctx.require(SHELL).run(
            _fetch(f"http://127.0.0.1:{host.port}/"), agent=agent
        )
        assert reached.exit_code == 0, reached.stderr
        assert reached.stdout.strip() == "hello from host"
        assert reached.confined_by == ctx.require(SANDBOX).provider.backend

        refused = await ctx.require(SHELL).run(_fetch("http://example.invalid/"), agent=agent)
        assert refused.exit_code != 0
        assert "403" in refused.stderr

    denials = [event for event in session.events if event.type == DENIED]
    assert [event.data.get("host") for event in denials] == ["example.invalid"]
    assert denials[0].data["via"] == "proxy"
    assert denials[0].data["agent"] == agent.id
    assert "/sandbox allow host example.invalid" in denials[0].data["message"]


async def test_a_command_that_ignores_the_proxy_reaches_nothing_and_says_so(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The closed direction, and the record read from the command's own words:
    a raw `connect()` finds no route, and the kernel's silence becomes a
    `sandbox/denied` because the output said so — `Network is unreachable` from a
    namespace with no route, `Operation not permitted` from Seatbelt's deny."""
    ctx = await _bridged(mount)
    session = ctx.require(SESSIONS).create("raw")
    agent = ctx.require(AGENTS).create(session, AgentOptions(provider="fake", model="f"))
    await ctx.require(WORKSPACE).acquire(session_id=session.id, agent_id=agent.id, base=tmp_path)
    raw = f"{sys.executable} -c 'import socket; socket.create_connection((\"192.0.2.1\", 80), 2)'"

    result = await ctx.require(SHELL).run(raw, agent=agent)

    assert result.exit_code != 0
    words = "Operation not permitted" if sys.platform == "darwin" else "Network is unreachable"
    assert words in result.stderr
    denials = [event for event in session.events if event.type == DENIED]
    assert len(denials) == 1
    assert denials[0].data["kind"] == "network" and denials[0].data["via"] == "output"


async def test_the_bridge_is_claimed_only_after_its_probe(mount: MountProfile) -> None:
    """`ph doctor` says the bridge is up because a confined command reached it
    through the door, not because a proxy was started."""
    ctx = await _bridged(mount)
    section = report_section(ctx, "Local confinement")
    assert section["egress"].startswith("proxy at ")
    assert PROBE_HOST in section["egress"]
    assert section["egress refused"] == "0"
    # And what the seam says follows from it.
    assert ctx.require(SANDBOX).allowances == Allowances(network=NetworkAllowance())
    assert "reachable through the egress proxy" in ctx.require(SANDBOX).network_posture()
