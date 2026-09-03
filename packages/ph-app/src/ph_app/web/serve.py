"""`ph --mode web` — the same terminal, in a browser tab (P7-05).

**One layout, not two.** `textual-serve` runs a real `PHTuiApp` as a subprocess
and streams its frames to a canvas over a websocket, so the browser shows the
terminal — the same widgets, the same fold, the same verbs — and layout parity is
by construction rather than by two implementations kept in step. A native HTML
renderer on the same view model is P7-07; this is what makes a browser useful
before that exists.

Each tab is its own subprocess and so its own front end — and **every tab of one
launch is on one session**, which is the multiplex the plan's decision 2
describes: several UIs on one conversation, each with a private composer, a
submitted prompt visible to all. That is not a preference; upstream fixes the
command at construction and varies nothing per request, so a session id here is
necessarily every tab's. Rather than leave it unset — which would make each tab a
*different* new session and leave an upload with no session to stage into — the
launch mints one and says so. `--session` names an existing one instead, and the
picker still moves a tab to another conversation.

A browser tab therefore differs from a second terminal, which does start a new
session. The difference is honest: two terminals are two programs a person ran,
and two tabs are two windows onto one server they opened.

**We compose the aiohttp app rather than calling `Server.serve()`.** Upstream
builds its own `web.Application` inside `serve()` with no hook to add a route or
a middleware, and this needs both: a token gate and an upload endpoint
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

**An upload is the browser's half of the human door (I-9).** `/attach` reads the
filesystem the *front end* is on, which for a tab is the server's — useless for a
file on the person's laptop. So the page takes bytes and this posts them to the
daemon: `attachment/put` stores the blob content-addressed, `session/stage` puts
it on the root's shared tray, and `session.staged` makes the chip appear in every
attached UI including the terminal on the server. The next prompt from any of
them carries it. The model never learns a path, and nothing here reads a file.

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
from dataclasses import KW_ONLY, dataclass, field
from functools import partial
from typing import cast

import aiohttp_jinja2
import anyio
import jinja2
from aiohttp import web
from aiohttp.multipart import BodyPartReader
from textual_serve.server import Server

from ph.paths import resolve_roots
from ph.resources import GRACE_SECONDS
from ph.seams.attachments import mime_of

from ..attach import stage_bytes
from ..daemon.client import DaemonClient, connected
from ..daemon.framing import MAX_ATTACHMENT_BYTES
from ..protocol import DaemonError

__all__ = ["CLOSING_BODY", "COOKIE", "TOKEN_QUERY", "WebServer", "run_web"]

log = logging.getLogger("ph_app.web")

CLOSING_BODY = "</body>"
"""Where the drop zone is inserted into a page this module does not own."""

_DROP_ZONE = """
<div id="ph-drop"><span>drop a file to attach it</span></div>
<style>
#ph-drop { position: fixed; inset: 0; display: none; place-items: center;
  background: rgba(0,0,0,.6); color: #fff; font: 600 1rem/1.4 system-ui, sans-serif;
  z-index: 9999; pointer-events: none; }
#ph-drop.on { display: grid; }
</style>
<script>
(() => {
  const zone = document.getElementById("ph-drop");
  const label = zone.querySelector("span");
  const ready = label.textContent;
  const show = (on) => zone.classList.toggle("on", on);
  let depth = 0;
  addEventListener("dragenter", (e) => { e.preventDefault(); if (++depth === 1) show(true); });
  addEventListener("dragleave", () => { if (--depth <= 0) { depth = 0; show(false); } });
  addEventListener("dragover", (e) => e.preventDefault());
  addEventListener("drop", async (e) => {
    e.preventDefault(); depth = 0; show(false);
    for (const file of e.dataTransfer.files) {
      const body = new FormData();
      body.append("file", file, file.name);
      // Same-origin, so the token cookie the shell set rides along; nothing
      // here knows the token itself.
      const sent = await fetch("/api/attachments", { method: "POST", body });
      // Restored, or the next drag prompts with the last failure's sentence.
      if (!sent.ok) { label.textContent = await sent.text(); show(true);
        setTimeout(() => { show(false); label.textContent = ready; }, 4000); }
    }
  });
})();
</script>
"""
"""The drop zone, inline rather than a static file.

Upstream owns `/static`, and adding a file to a directory it ships would put pH's
asset inside the package's own tree — so this travels with the page it is
inserted into. It posts to `/api/attachments` same-origin, which is what lets it
carry the token cookie without ever seeing the token."""

DROPPABLE = _DROP_ZONE + CLOSING_BODY
"""The insertion, joined once rather than per page load."""

EXPECTED = "an upload is one multipart part named 'file'"
"""What the route takes, said the one way, to whoever got it wrong."""

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


@dataclass(slots=True)
class WebServer:
    """The composed application, its token, and where it is listening."""

    command: str
    """What `textual-serve` runs per tab — one `ph --mode tui`, one subprocess."""
    _: KW_ONLY
    session: str
    """The session every tab of this launch is on, and where an upload stages.

    Known here because this process built the command that carries it — which is
    the only reason an upload can be routed at all: nothing upstream
    distinguishes one tab from another."""
    host: str = "127.0.0.1"
    port: int = 8000
    title: str = "pH"
    """What the browser calls the app. Not a knob: upstream falls back to
    `title or command`, which would put the whole `python -m ph_app --mode tui …`
    line in the page's own intro dialog."""
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32), init=False)
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
        # Said out loud because it is the surprise: a second *terminal* is a new
        # session and a second *tab* is not. It is also the id for `ph agents
        # attach` and for reopening the conversation after this process ends.
        yield f"[dim]every tab lands on session {self.session}[/dim]"
        yield (
            "[dim]anyone with that URL has this terminal's authority: approvals, "
            "shell commands, the workspace[/dim]"
        )
        if self.host not in ("127.0.0.1", "localhost", "::1"):
            yield (
                f"[yellow]{self.host} is not loopback: that authority reaches anyone "
                "who can route to this port[/yellow]"
            )

    async def _index(self, server: Server, request: web.Request) -> web.StreamResponse:
        """Upstream's page with a drop zone inserted into it.

        **Inserted, not templated.** `app_index.html` is upstream's and has no
        jinja blocks to extend, so the choices were to fork the template — which
        is half of a client protocol whose other half ships in the wheel — or to
        add to what it rendered. Insertion keeps upstream's page authoritative:
        a release that changes it changes what a person sees, and the only thing
        that can break here is the `</body>` this looks for, which is checked.
        """
        # A `Response`, not a `StreamResponse`: the `aiohttp_jinja2.template`
        # decorator is what turns upstream's context dict into one, which is also
        # why its body can be read back and rewritten.
        rendered = cast("web.Response", await server.handle_index(request))
        body = rendered.text or ""
        if CLOSING_BODY not in body:
            # Upstream's page changed shape. Serving it unmodified is the right
            # failure: the terminal still works, and a person who cannot drop a
            # file can still use `/attach`. Said out loud because nothing else
            # would say it — a `replace` that matches nothing is silent.
            log.warning("ph_app.web: no %s in the page; no drop zone added", CLOSING_BODY)
            return rendered
        rendered.text = body.replace(CLOSING_BODY, DROPPABLE)
        return rendered

    async def _upload(self, request: web.Request) -> web.StreamResponse:
        """Bytes from the browser onto this session's tray.

        Read in chunks against the same ceiling the daemon enforces, so a
        too-large file is refused before it is buffered whole rather than after
        `attachment/put` has already had it. Nothing else applies that ceiling on
        this path: aiohttp checks `client_max_size` in `read()` and `post()`, and
        `read_chunk` — the only reader here — is deliberately not one of them.

        The declared `Content-Type` wins over the name, because the browser knows
        what it picked up and pH is guessing from an extension; `mime_of` is that
        guess, shared with `--attach` so one file is one kind of thing whichever
        door it came through.

        **A malformed body is answered, not raised through.** The only producer
        is the drop zone above, so the first part is taken rather than scanned
        for — but the caller might also be a person with `curl`, and
        `request.multipart()` *asserts* on a body that is not multipart, which
        reaches them as a 500 and an `AssertionError`. The same class of defect
        as a daemon refusal arriving as one: an authorised person getting it
        slightly wrong is not a server error.
        """
        if not request.content_type.startswith("multipart/"):
            raise web.HTTPUnsupportedMediaType(text=f"pH: {EXPECTED}\n")
        part = await (await request.multipart()).next()
        if not isinstance(part, BodyPartReader) or part.name != "file":
            raise web.HTTPBadRequest(text=f"pH: {EXPECTED}\n")
        name = part.filename or "upload"
        content = bytearray()
        # 64 KiB rather than the 8 KiB default: a 5 MiB drop is 80 awaits on the
        # loop that is streaming every open tab's terminal, not 640.
        while chunk := await part.read_chunk(1 << 16):
            content += chunk
            if len(content) > MAX_ATTACHMENT_BYTES:
                raise web.HTTPRequestEntityTooLarge(
                    max_size=MAX_ATTACHMENT_BYTES,
                    actual_size=len(content),
                    text=(
                        f"pH: {name} is over the {MAX_ATTACHMENT_BYTES // (1024 * 1024)} MiB "
                        "an attachment frame carries\n"
                    ),
                )

        mime = part.headers.get("Content-Type") or mime_of(name)
        # `attachmentId` is the reference's own wire field — echoed rather than
        # renamed, so a browser and the log call one blob by one name.
        return web.json_response(
            {"attachmentId": await self._stage(name, mime, bytes(content)), "name": name},
            status=201,
        )

    async def _stage(self, name: str, mime: str, content: bytes) -> str:
        """Put the blob on the daemon and on this session's tray, or say why not.

        **A fresh connection per request, not a held one**: an upload is rare and
        already costs a file read, a unix connect is sub-millisecond, and a client
        kept for the process's life is one that goes stale when the daemon
        passivates or an ephemeral one exits — a staleness this route would then
        have to detect and repair for no gain.

        **A daemon refusal is not a server error.** Every reachable one here says
        the same thing to a person — no daemon, or no session yet, because the
        *tabs* are what start both — and the daemon's own sentence is better than
        anything this route could invent. Without the mapping they arrived as a
        500 and an `ExceptionGroup` traceback in the log, which is what happens
        when a person uploads before opening a tab.

        `except*`, because `connected` runs the calls inside a task group — the
        pump has to be reading for a reply to arrive — and anyio wraps whatever
        leaves one, so an `except DaemonError` here would see nothing (the
        `_alone` hazard). The `OSError` is raised before that group and so is not
        wrapped, which is why the two are nested rather than side by side.
        """

        async def stage(client: DaemonClient) -> str:
            return await stage_bytes(client, self.session, name, mime, content)

        try:
            try:
                return await connected(resolve_roots().daemon_socket(), stage)
            except OSError as gone:
                raise web.HTTPServiceUnavailable(
                    text="pH: no daemon is running; open a tab first\n"
                ) from gone
        except* DaemonError as refused:
            raise web.HTTPServiceUnavailable(
                text=f"pH: {refused.exceptions[0]}; open a tab first\n"
            ) from refused

    def application(self) -> web.Application:
        """`Server._make_app`, re-stated with our gate. See the module docstring."""
        server = Server(command=self.command, host=self.host, port=self.port, title=self.title)
        # A backstop, not the gate: aiohttp applies `client_max_size` to
        # `read()`/`post()` and never to the `read_chunk` loop `_upload` uses, so
        # the refusal a person sees is that loop's. One number rather than two,
        # because a second budget is how the first silently becomes dead code.
        app = web.Application(middlewares=[_refuse_untokened], client_max_size=MAX_ATTACHMENT_BYTES)
        app[TOKEN] = self.token
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(server.templates_path))
        app.add_routes(
            [
                # Names matter: `handle_index` resolves `websocket` and `static`
                # out of the router to build the page.
                web.get("/", partial(self._index, server), name="index"),
                web.post("/api/attachments", self._upload, name="attachments"),
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
    handlers and owns the loop.

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
