"""`ph --mode web` — the same terminal, in a browser tab (P7-05).

**One layout, not two.** `textual-serve` runs a real `PHTuiApp` as a subprocess
and streams its frames to a canvas over a websocket, so the browser shows the
terminal — the same widgets, the same fold, the same verbs — and layout parity is
by construction rather than by two implementations kept in step. A native HTML
renderer on the same view model is P7-07; this is what makes a browser useful
before that exists.

Each tab is its own subprocess and so its own front end on the one daemon. By
default each opens its own *new* session: upstream fixes the command at
construction and varies nothing per request, so there is nowhere to put a session
id that would not be every tab's. Joining a session somebody else is in is what
the picker is for, exactly as it is from a second terminal.

**We compose the aiohttp app rather than calling `Server.serve()`.** Upstream
builds its own `web.Application` inside `serve()` with no hook to add a route or
a middleware, and this needs both: a token gate, and later an upload endpoint
(P7-06). What it does expose is the three handlers, the two lifecycle hooks and
the two asset paths, so this module is `_make_app` re-stated with our additions —
and `test_textual_serve_still_exposes_what_we_compose` fails at test time if a
release renames any of them, rather than at `ph --mode web` in front of a person.
Overriding the private `_make_app` from a subclass would fail the other way: a
rename there leaves the override uncalled and the app built *without the gate*,
which is a security hole that comes up green.

Three details of that re-statement are load-bearing, and all three are upstream's
requirements rather than choices:

* `aiohttp_jinja2.setup` with upstream's own template loader, because
  `handle_index` is decorated with `@aiohttp_jinja2.template("app_index.html")`
  and renders nothing without it;
* the route **names** `websocket` and `static`, because `handle_index` builds the
  page's websocket URL and asset prefix by looking them up in the router;
* upstream's `statics_path`/`templates_path`, which it resolves relative to its
  own `server.py` — so they are read off the `Server` instance rather than
  guessed.

Every import here is at module scope, and that is the point: this module is only
reached from the `--mode web` branch, which wraps the import in the one `try`
that turns a missing extra into an install line. A lazily-imported
`textual_serve` would slip past that `try` and fail later, in front of a person
who had already been told the server was starting.

**Exposure is a token, and that is all it is** (§5 rule 6). No TLS, no users, no
revocation: one secret minted per launch, carried in the URL, exchanged for a
cookie so the page's own websocket and asset URLs — which upstream builds, and
which cannot carry a query parameter this module chose — are authorised too.
Anyone holding it has whatever authority the terminal has, which includes
approving tool calls and running `!!` shell commands. `notices()` is where that
is said; the command prints it.

@module ph_app.web.serve
"""

from __future__ import annotations

import logging
import secrets
import threading
import webbrowser
from collections.abc import Awaitable, Callable, Iterator

import aiohttp_jinja2
import anyio
import jinja2
from aiohttp import web
from textual_serve.server import Server

from ph.resources import GRACE_SECONDS

__all__ = ["COOKIE", "TOKEN_QUERY", "WebServer", "run_web"]

log = logging.getLogger("ph_app.web")

TOKEN_QUERY = "token"
"""The query parameter the launch URL carries the secret in."""

COOKIE = "ph_web_token"
"""The cookie the first authorised request sets.

The page's websocket and static URLs are built by upstream's own template from
the router, so they cannot carry a query parameter — a query-only gate would
serve the shell and then refuse the socket it opens. So the token is exchanged
once, on any request that presents it, and every later one presents the cookie.
"""

FAVICON = "/favicon.ico"
"""The one path served without a token.

Upstream's page declares no icon, so browsers ask for this unprompted; refusing
it would put a line in the log on every page load and protect nothing.
"""

TOKEN: web.AppKey[str] = web.AppKey("ph_web_token")
"""This launch's secret, read by the gate off the request's own app.

On the app rather than captured in a closure, so the middleware is a plain module
function: one that closed over a `WebServer` would hold the whole server for the
process's life to read one string.
"""


@web.middleware
async def _refuse_untokened(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    """Refuse anything that does not present the launch token.

    A middleware rather than a check per handler: the handlers are upstream's and
    cannot be edited, and a gate that had to be remembered at each route is one
    that will be forgotten at the next.

    The refusal is logged at `debug`, not `warning`. On a non-loopback bind this
    is the one per-request cost an unauthenticated caller controls, and a scanner
    probing the port should not be able to make pH format and write a line per
    probe on the event loop thread.
    """
    if request.path == FAVICON:
        return await handler(request)
    token = request.app[TOKEN]
    offered = request.query.get(TOKEN_QUERY) or request.cookies.get(COOKIE, "")
    # `compare_digest`, because a token is a secret and `==` returns early on the
    # first differing byte — a prefix oracle for anyone who can measure.
    if not secrets.compare_digest(offered, token):
        log.debug("ph_app.web: refused %s %s — no token", request.method, request.path)
        raise web.HTTPForbidden(text="pH: this page needs the token from its launch URL\n")
    response = await handler(request)
    if request.query.get(TOKEN_QUERY):
        response.set_cookie(COOKIE, token, httponly=True, samesite="Strict", path="/")
    return response


class WebServer:
    """The composed application, its token, and where it is listening."""

    __slots__ = ("command", "host", "port", "title", "token")

    def __init__(
        self, command: str, *, host: str = "127.0.0.1", port: int = 8000, title: str = "pH"
    ) -> None:
        self.command = command
        """What `textual-serve` runs per tab — one `ph --mode tui`, one subprocess."""
        self.host = host
        self.port = port
        self.title = title
        """What the browser calls the app. Not a knob: upstream falls back to
        `title or command`, which would put the whole `python -m ph_app --mode
        tui …` line in the page's own intro dialog."""
        self.token = secrets.token_urlsafe(32)
        """Minted per launch, so closing the process ends the grant — there is no
        session store to revoke one in."""

    @property
    def url(self) -> str:
        """The address a person opens, token included."""
        return f"http://{self.host}:{self.port}/?{TOKEN_QUERY}={self.token}"

    def notices(self) -> Iterator[str]:
        """What a person needs to be told before this starts serving.

        Sentences rather than prints, so the *command* does the printing — the
        shape `ph daemon` already uses for its socket path and its linger
        warning, where `ph.lingering` owns the words and `cli.py` owns the
        console.

        Upstream prints a banner of its own from `on_startup`, naming the command
        it runs per tab and the public URL **without** the token — which is the
        half that does not work. These are the halves that do.
        """
        yield f"[bold]open[/bold] {self.url}"
        yield (
            "[dim]anyone with that URL has this terminal's authority: approvals, "
            "shell commands, the workspace[/dim]"
        )
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            yield (
                f"[yellow]{self.host} is not loopback: that authority reaches anyone "
                "who can route to this port[/yellow]"
            )

    def application(self) -> web.Application:
        """`Server._make_app`, re-stated with our gate. See the module docstring."""
        server = Server(command=self.command, host=self.host, port=self.port, title=self.title)
        app = web.Application(middlewares=[_refuse_untokened])
        app[TOKEN] = self.token
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(server.templates_path))
        app.add_routes(
            [
                # Names matter: `handle_index` resolves `websocket` and `static`
                # out of the router to build the page.
                web.get("/", server.handle_index, name="index"),
                web.get("/ws", server.handle_websocket, name="websocket"),
                web.get("/download/{key}", server.handle_download, name="download"),
                web.static("/static", server.statics_path, show_index=True, name="static"),
            ]
        )
        # Upstream's hooks are kept even though `on_startup` is only a banner and
        # `on_shutdown` is empty *today*: a release that moved subprocess cleanup
        # into either would otherwise leak a process per tab, and that is not a
        # bet worth taking to save two lines.
        app.on_startup.append(server.on_startup)
        app.on_shutdown.append(server.on_shutdown)
        return app


async def run_web(server: WebServer, *, open_browser: bool = False) -> None:
    """Serve the terminal over HTTP until cancelled.

    Under anyio, like every other mode: `cli.py` has one shape for "go do the
    thing", and `web.run_app` would be a second — it installs its own signal
    handlers and owns the loop. It matters ahead of P7-06 too, whose upload route
    needs a `DaemonClient` on the socket with this server's lifetime, which is a
    task group rather than an addition.

    `GRACE_SECONDS` and not aiohttp's own 60: two budgets for one shutdown is how
    the inner one silently becomes dead code, and a hung tab must not hold this
    process six times longer than every other pH teardown allows.
    """
    runner = web.AppRunner(server.application(), shutdown_timeout=GRACE_SECONDS)
    await runner.setup()
    await web.TCPSite(runner, server.host, server.port).start()
    try:
        if open_browser:
            # **After the bind, and off this thread.** `webbrowser.open` scans
            # `PATH` for every known browser before launching one, and a
            # `$BROWSER` naming a plain command makes it *wait for that process
            # to exit* — either of which, done first, hands out a URL the socket
            # is not answering on yet, or never starts the server at all.
            threading.Thread(target=webbrowser.open, args=(server.url,), daemon=True).start()
        await anyio.sleep_forever()
    finally:
        # Runs `on_shutdown`, which is where upstream will put per-tab cleanup if
        # it ever does.
        await runner.cleanup()
