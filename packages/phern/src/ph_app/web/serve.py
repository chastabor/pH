"""`phern --mode web`: the same terminal, in a browser tab (P7-05).

**One layout, not two.** `textual-serve` runs a real `PHTuiApp` as a subprocess
and streams its frames to a canvas over a websocket. The browser therefore shows
the terminal: the same widgets, the same fold, the same verbs. Layout parity
comes from the construction, and not from two implementations that must be kept
in step. A native HTML renderer on the same view model is P7-07. This module
makes a browser useful before that renderer exists.

Each tab is its own subprocess and so its own front end. **Every tab of one
launch is on one session.** That is the multiplex that decision 2 of the plan
describes: several UIs on one conversation, each with a private composer, and a
submitted prompt that all of them see. This is not a preference. Upstream fixes
the command at construction and varies nothing for each request, so a session id
here necessarily belongs to every tab. If the id were left unset, each tab would
be a *different* new session, and an upload would have no session to stage into.
The launch therefore mints one id and reports it. `--session` names an existing
session instead, and the picker still moves a tab to another conversation.

A browser tab therefore differs from a second terminal, which does start a new
session. The difference matches what the user did: two terminals are two
programs that the user ran, and two tabs are two windows onto one server that
the user opened.

**This module composes the aiohttp app and does not call `Server.serve()`.**
Upstream builds its own `web.Application` inside `serve()` with no hook to add a
route or a middleware, and this module needs both: a token gate and an upload
endpoint (P7-06). Upstream does expose the three handlers, the two lifecycle
hooks and the two asset paths. This module is therefore `_make_app` written
again with the additions of pH. `test_textual_serve_still_exposes_what_we_compose`
fails at test time if a release renames any of them, and not at
`phern --mode web` in front of the user. An override of the private `_make_app`
from a subclass would fail in the other direction: a rename there leaves the
override uncalled and the app built *without the gate*. That is a security hole
that passes every test.

Three details of that copy are necessary. All three are requirements of upstream
and not choices:

- `aiohttp_jinja2.setup` with the template loader of upstream itself, because
  `handle_index` is decorated with `@aiohttp_jinja2.template("app_index.html")`
  and renders nothing without it.
- The route **names** `websocket` and `static`, because `handle_index` builds
  the websocket URL and the asset prefix of the page by looking them up in the
  router.
- The `statics_path`/`templates_path` of upstream, which upstream resolves
  relative to its own `server.py`. They are therefore read from the `Server`
  instance and not guessed.

Every import here is at module scope, on purpose. This module is reached only
from the `--mode web` branch, which wraps the import in the one `try` that turns
a missing extra into an install line. A lazily imported `textual_serve` would
pass that `try` and fail later, in front of a user who had already been told
that the server was starting.

**An upload is the browser's half of the human door (I-9).** `/attach` reads the
filesystem that the *front end* is on. For a tab, that is the filesystem of the
server, which is useless for a file on the laptop of the user. The page
therefore takes bytes, and this module posts them to the daemon.
`attachment/put` stores the blob content-addressed. `session/stage` puts it on
the shared tray of the root. `session.staged` makes the chip appear in every
attached UI, including the terminal on the server. The next prompt from any of
them carries the blob. The model never learns a path, and nothing here reads a
file.

**Exposure is a token, and nothing more** (§5 rule 6). There is no TLS, no users
and no revocation. One secret is minted for each launch and carried in the URL.
It is exchanged for a cookie, so that the websocket and asset URLs of the page
are authorized too. Upstream builds those URLs, and they cannot carry a query
parameter that this module chose. Anyone who holds the token has whatever
authority the terminal has, which includes approving tool calls and running `!!`
shell commands. `notices()` states that, and the command prints it.

@module ph_app.web.serve
"""

from __future__ import annotations

import logging
import secrets
import threading
import webbrowser
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass, field
from functools import partial
from typing import cast

import aiohttp_jinja2
import anyio
import jinja2
from aiohttp import web
from aiohttp.multipart import BodyPartReader
from textual_serve.server import Server

from ph.llm.types import AttachmentRef
from ph.paths import resolve_roots
from ph.resources import GRACE_SECONDS
from ph.seams.attachments import mime_for

from ..attach import stage_bytes
from ..daemon.client import DaemonClient, connected
from ..daemon.framing import MAX_ATTACHMENT_BYTES, MAX_ATTACHMENT_SIZE
from ..protocol import DaemonError
from . import DEFAULT_HOST, DEFAULT_PORT

__all__ = ["CLOSING_BODY", "COOKIE", "TOKEN_QUERY", "WebServer", "run_web"]

log = logging.getLogger("ph_app.web")

CLOSING_BODY = "</body>"
"""Where the drop zone is inserted into a page that this module does not own."""

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
"""The drop zone, inline and not a static file.

Upstream owns `/static`. A file added to a directory that upstream ships would
put an asset of pH inside the tree of that package. The drop zone therefore
travels with the page that it is inserted into. It posts to `/api/attachments`
same-origin, so it carries the token cookie and never sees the token."""

EXPECTED = "pH: an upload is one multipart part named 'file'\n"
"""What the route accepts, in one wording, for a caller that sent something else."""

TITLE = "pH"
"""The name that the browser shows for the app.

This is not a setting, so it is not a field. Upstream falls back to
`title or command`, so an unset title would put the whole `python -m ph_app
--mode tui …` line in the intro dialog of the page."""

NO_SESSION = "no_such_session"
"""The name that the daemon itself gives to the one refusal that has a next step.

`DaemonError.reason` is `Refusal.code` after a round trip. This is therefore the
vocabulary of the wire, and not a second vocabulary invented at the HTTP edge. A
match on the message text is a contract that nobody writes down and that every
rewording breaks. Advice is attached to this reason alone. Telling somebody to
open a tab helps when there is no session, and is noise when their file was
only too big."""

TOKEN_QUERY = "token"
"""The query parameter in which the launch URL carries the secret."""

COOKIE = "ph_web_token"
"""The cookie that the first authorized request sets.

The template of upstream builds the websocket and static URLs of the page from
the router, so they cannot carry a query parameter. A gate that accepted only
the query would serve the shell and then refuse the socket that the shell opens.
The token is therefore exchanged for the cookie on any request that presents it
in the query, and every later request presents the cookie.
"""

FAVICON = "/favicon.ico"
"""The one path that is served without a token.

The page of upstream declares no icon, so browsers request this path without
being asked. A refusal would put a line in the log on every page load and would
protect nothing.
"""

TOKEN: web.AppKey[str] = web.AppKey("ph_web_token")
"""The secret of this launch, which the gate reads from the app of the request.

It is on the app and not captured in a closure, so the middleware is a plain
module function. A middleware that closed over a `WebServer` would hold the
whole server for the life of the process to read one string.
"""


@web.middleware
async def _refuse_untokened(
    request: web.Request, handler: Callable[[web.Request], Awaitable[web.StreamResponse]]
) -> web.StreamResponse:
    """Refuse anything that does not present the launch token.

    This is a middleware and not a check in each handler. The handlers belong to
    upstream and cannot be edited, and a gate that must be remembered at each
    route will be forgotten at the next route.

    The refusal is logged at `debug`, not `warning`. On a non-loopback bind this
    is the one per-request cost that an unauthenticated caller controls. A
    scanner that probes the port should not be able to make pH format and write
    a line for each probe on the event loop thread.
    """
    if request.path == FAVICON:
        return await handler(request)
    token = request.app[TOKEN]
    offered = request.query.get(TOKEN_QUERY) or request.cookies.get(COOKIE, "")
    # `compare_digest` is used because a token is a secret. `==` returns early
    # at the first differing byte, which is a prefix oracle for anyone who can
    # measure.
    if not secrets.compare_digest(offered, token):
        log.debug("ph_app.web: refused %s %s — no token", request.method, request.path)
        raise web.HTTPForbidden(text="pH: this page needs the token from its launch URL\n")
    response = await handler(request)
    if request.query.get(TOKEN_QUERY):
        response.set_cookie(COOKIE, token, httponly=True, samesite="Strict", path="/")
    return response


@dataclass(slots=True, kw_only=True)
class WebServer:
    """The composed application, its token, and where it is listening."""

    command: str
    """What `textual-serve` runs for each tab: one `phern --mode tui`, in one
    subprocess."""
    session: str
    """The session that every tab of this launch is on, and where an upload is
    staged. The module docstring explains why it cannot be the session of one
    tab."""
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    token: str = field(default_factory=lambda: secrets.token_urlsafe(32), init=False)
    """Minted for each launch, so the grant ends when the process closes. There
    is no session store in which to revoke a grant."""

    @property
    def url(self) -> str:
        """The address that the user opens, with the token included."""
        return f"http://{self.host}:{self.port}/?{TOKEN_QUERY}={self.token}"

    def notices(self) -> Iterator[str]:
        """What the user must be told before this starts serving.

        The method yields sentences and does not print, so the *command* does
        the printing. `phern daemon` already uses this arrangement for its
        socket path and its linger warning: `ph.lingering` owns the words and
        `cli.py` owns the console.

        Upstream prints its own banner from `on_startup`. The banner names the
        command that it runs for each tab and the public URL **without** the
        token, and that URL does not work. These notices give the parts that do
        work.
        """
        yield f"[bold]open[/bold] {self.url}"
        # This is stated because it is the surprise: a second *terminal* is a new
        # session and a second *tab* is not. It is also the id for `phern agents
        # attach`, and for reopening the conversation after this process ends.
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
        """Return the page of upstream with a drop zone inserted into it.

        **Inserted, not templated.** `app_index.html` belongs to upstream and
        has no jinja blocks to extend. One choice was to fork the template,
        which is half of a client protocol whose other half ships in the wheel.
        The other choice was to add to what the template rendered. Insertion
        keeps the page of upstream authoritative: a release that changes the
        page changes what the user sees. The only thing that can break here is
        the `</body>` that this method looks for, and that is checked.
        """
        # This is a `Response`, not a `StreamResponse`. The
        # `aiohttp_jinja2.template` decorator turns the context dict of upstream
        # into one, and that is also why its body can be read back and rewritten.
        rendered = cast("web.Response", await server.handle_index(request))
        body = rendered.text or ""
        if CLOSING_BODY not in body:
            # The page of upstream changed shape. To serve it unmodified is the
            # right failure: the terminal still works, and a user who cannot
            # drop a file can still use `/attach`. The warning is logged because
            # nothing else would report it: a `replace` that matches nothing is
            # silent.
            log.warning("ph_app.web: no %s in the page; no drop zone added", CLOSING_BODY)
            return rendered
        rendered.text = body.replace(CLOSING_BODY, _DROP_ZONE + CLOSING_BODY)
        return rendered

    async def _upload(self, request: web.Request) -> web.StreamResponse:
        """Put bytes from the browser onto the tray of this session.

        **A malformed body gets an answer, and does not raise through.** The
        only producer is the drop zone above, so the first part is taken and not
        searched for. But the caller might also be a person with `curl`, and
        `request.multipart()` *asserts* on a body that is not multipart. That
        would reach the caller as a 500 and an `AssertionError`. It is the same
        class of defect as a daemon refusal that arrives as a 500: an authorized
        person who gets the request slightly wrong is not a server error.

        **There are two ceilings, and both are necessary.** `Content-Length` is
        refused before a byte of the body is read, which stops a 40 MB drop from
        crossing the wire only to be discarded. The chunk loop is the backstop
        for a chunked body, which has no length to check. Nothing else applies
        the limit here: aiohttp consults `client_max_size` in `read()` and
        `post()` and never in `read_chunk`, the only reader on this path.
        """
        if not request.content_type.startswith("multipart/"):
            raise web.HTTPUnsupportedMediaType(text=EXPECTED)
        if request.content_length and request.content_length > MAX_ATTACHMENT_BYTES:
            raise self._too_large("the upload", request.content_length)
        part = await (await request.multipart()).next()
        if not isinstance(part, BodyPartReader) or part.name != "file":
            raise web.HTTPBadRequest(text=EXPECTED)
        name = part.filename or "upload"
        content = bytearray()
        # 64 KiB and not the 8 KiB default: a 5 MiB drop is then 80 awaits, not
        # 640, on the loop that streams the terminal of every open tab.
        while chunk := await part.read_chunk(1 << 16):
            content += chunk
            if len(content) > MAX_ATTACHMENT_BYTES:
                raise self._too_large(name, len(content))

        # The browser knows what it picked up, and pH guesses from a name, so a
        # declared type wins. The exception is the type that declares nothing,
        # which a browser sends for an extension that it does not know.
        # `mime_for` is that rule, and `--attach` follows the same rule.
        mime = mime_for(part.headers.get("Content-Type"), name)
        try:
            reference = await self._stage(name, mime, content)
        except OSError as gone:
            raise web.HTTPServiceUnavailable(
                text="pH: no daemon is running; open a tab first\n"
            ) from gone
        except DaemonError as refused:
            raise self._refused(refused, name, len(content)) from refused
        # Return the wire form of the reference itself, not a subset of it. A
        # browser and the log then describe one blob in the same way, down to
        # the field names.
        return web.json_response(reference.to_wire(), status=201)

    @staticmethod
    def _too_large(what: str, size: int) -> web.HTTPRequestEntityTooLarge:
        """Build the one message for the limit, for each place that detects it."""
        return web.HTTPRequestEntityTooLarge(
            max_size=MAX_ATTACHMENT_BYTES,
            actual_size=size,
            text=f"pH: {what} is over the {MAX_ATTACHMENT_SIZE} an attachment frame carries\n",
        )

    async def _stage(self, name: str, mime: str, content: bytes | bytearray) -> AttachmentRef:
        """Put the blob on the daemon and on the tray of this session, or raise
        the refusal.

        **Each request uses a fresh connection, not a held one.** An upload is
        rare and already costs a file read, and a unix connect takes less than a
        millisecond. A client kept for the life of the process goes stale when
        the daemon passivates or when an ephemeral daemon exits. This route
        would then have to detect and repair that staleness for no gain.

        **The method raises what the daemon refused, directly.** `connected`
        unwraps its own task group, so a caller no longer has to know that a
        frame crossed one. `_refused` turns the refusal into an answer.
        """

        async def stage(client: DaemonClient) -> AttachmentRef:
            return await stage_bytes(client, self.session, name, mime, content)

        return await connected(resolve_roots().daemon_socket(), stage)

    def _refused(self, refused: DaemonError, name: str, size: int) -> web.HTTPException:
        """Turn the "no" of the daemon into an answer that a browser understands.

        **Each refusal gets its own sentence and its own status.** An upload
        before a tab is opened is the ordinary state between launch and use. It
        first arrived as a 500 with an `ExceptionGroup` traceback. To answer
        *every* refusal with one status and one sentence was the next version of
        the same mistake. A file of exactly `MAX_ATTACHMENT_BYTES` reaches
        `attachment_too_large` here: the ceiling of this route is on the raw
        bytes, and the ceiling of the daemon is on their base64 expansion, which
        is slightly larger. For that file, "open a tab first" is advice about
        the wrong problem.
        """
        if refused.reason == "attachment_too_large":
            return self._too_large(name, size)
        advice = "; open a tab first" if refused.reason == NO_SESSION else ""
        return web.HTTPServiceUnavailable(text=f"pH: {refused}{advice}\n")

    def application(self) -> web.Application:
        """`Server._make_app`, written again with the gate of pH. See the module
        docstring."""
        server = Server(command=self.command, host=self.host, port=self.port, title=TITLE)
        # This is a backstop, not the gate. aiohttp applies `client_max_size` to
        # `read()`/`post()` and never to the `read_chunk` loop that `_upload`
        # uses, so the refusal that the user sees comes from that loop. There is
        # one number and not two, because a second budget is how the first
        # silently becomes dead code.
        app = web.Application(middlewares=[_refuse_untokened], client_max_size=MAX_ATTACHMENT_BYTES)
        app[TOKEN] = self.token
        aiohttp_jinja2.setup(app, loader=jinja2.FileSystemLoader(server.templates_path))
        app.add_routes(
            [
                # The names matter: `handle_index` looks up `websocket` and
                # `static` in the router to build the page.
                web.get("/", partial(self._index, server), name="index"),
                web.post("/api/attachments", self._upload, name="attachments"),
                web.get("/ws", server.handle_websocket, name="websocket"),
                web.get("/download/{key}", server.handle_download, name="download"),
                web.static("/static", server.statics_path, show_index=True, name="static"),
            ]
        )
        # The hooks of upstream are kept, although `on_startup` is only a banner
        # and `on_shutdown` is empty *today*. A release that moved subprocess
        # cleanup into either hook would otherwise leak a process for each tab.
        # That risk is not worth the two lines saved.
        app.on_startup.append(server.on_startup)
        app.on_shutdown.append(server.on_shutdown)
        return app


async def run_web(server: WebServer, *, open_browser: bool = False) -> None:
    """Serve the terminal over HTTP until canceled.

    The server runs under anyio, like every other mode. `cli.py` has one shape
    for "go do the thing", and `web.run_app` would be a second shape: it installs
    its own signal handlers and owns the loop.

    The shutdown timeout is `GRACE_SECONDS` and not the 60 of aiohttp, by the
    rule that `application()` states: a second budget is how the first becomes
    dead code. A hung tab must not hold this process six times longer than every
    other pH teardown allows.
    """
    runner = web.AppRunner(server.application(), shutdown_timeout=GRACE_SECONDS)
    await runner.setup()
    await web.TCPSite(runner, server.host, server.port).start()
    try:
        if open_browser:
            # **After the bind, and off this thread.** `webbrowser.open` scans
            # `PATH` for every known browser before it launches one. A
            # `$BROWSER` that names a plain command makes it *wait for that
            # process to exit*. If either happened first, the user would get a
            # URL that the socket does not answer on yet, or the server would
            # never start.
            threading.Thread(target=webbrowser.open, args=(server.url,), daemon=True).start()
        await anyio.sleep_forever()
    finally:
        # This runs `on_shutdown`, which is where upstream will put per-tab
        # cleanup if it ever adds any.
        await runner.cleanup()
