"""P7-05 — the terminal in a browser tab, and the token that gates it.

Two kinds of claim, and they fail for different reasons.

**What we compose is upstream's**, so a `textual-serve` release that renames a
handler must fail here rather than at `ph --mode web` in front of a person.
`_make_app` has no hook for a route or a middleware, so this module re-states it;
that re-statement is only correct while the pieces it names still exist.

**Nothing is served without the token.** `--host` can open an interface, and
whoever reaches it holds whatever authority the terminal has — approving tool
calls, running `!!`. So the gate is asserted on the shell, the websocket and an
unknown path, and asserted as a *refusal* rather than as a redirect.

No browser and no subprocess: the tab's command is a stub, because what is under
test is the HTTP surface rather than Textual's frame streaming.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import pytest
from aiohttp import FormData, web
from aiohttp.test_utils import TestClient, TestServer
from daemon_helpers import serving, until

from ph_app.daemon.framing import MAX_ATTACHMENT_BYTES
from ph_app.web.serve import CLOSING_BODY, COOKIE, TOKEN_QUERY, WebServer

pytestmark = pytest.mark.anyio

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
"""Enough of a header that the store recognises it; the bytes do not matter."""


def a_server(session: str = "served", **options: Any) -> WebServer:
    """A composed server whose tabs would run a command that never runs."""
    return WebServer(command="/bin/true", session=session, **options)


@pytest.fixture
def server() -> WebServer:
    return a_server()


@asynccontextmanager
async def browser_on(server: WebServer) -> AsyncIterator[Any]:
    """A client on this server, torn down with the block.

    Takes the server rather than making one: the daemon-backed tests need a
    session id that does not exist until the test has a root, and the token
    assertions need the *same* object they read `.token` off."""
    async with TestClient(TestServer(server.application())) as opened:
        yield opened


@pytest.fixture
async def client(server: WebServer) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """The common case as a fixture: a server nobody named, and a client on it."""
    async with browser_on(server) as opened:
        yield opened


async def drop(
    server: WebServer,
    browser: Any,
    content: bytes = PNG,
    name: str = "diagram.png",
    mime: str = "image/png",
) -> Any:
    """One file, dropped on the page. What every upload test does."""
    body = FormData()
    body.add_field("file", content, filename=name, content_type=mime)
    return await browser.post("/api/attachments", params={TOKEN_QUERY: server.token}, data=body)


# ------------------------------------------------------- what upstream owes us --


def test_textual_serve_still_exposes_what_we_compose() -> None:
    """The three handlers, the two hooks, and the two paths — by name.

    `application()` re-states `Server._make_app` because upstream offers no way
    to add a middleware to the app it builds itself. That is fine while these
    names hold and silently wrong the moment one changes, so the names are the
    assertion: an upstream rename fails *this* fast, named test rather than
    surfacing as a 500 inside the end-to-end one.

    `statics_path` and `templates_path` are in the list because upstream resolves
    them relative to its *own* `server.py` — a path this module must not guess.
    One instance for all of them, since an instance answers for class attributes
    too and building a `Server` costs a `rich.Console` and a `DownloadManager`.
    """
    from textual_serve.server import Server

    upstream = Server(command="/bin/true")
    for name in (
        "handle_index",
        "handle_websocket",
        "handle_download",
        "on_startup",
        "on_shutdown",
        "statics_path",
        "templates_path",
    ):
        assert hasattr(upstream, name), name


# --------------------------------------------------------------- the token --


async def test_the_shell_is_refused_without_the_token(client: Any) -> None:
    """A 403, not a redirect and not a login page.

    There is nothing to log in to: one secret per launch is the whole scheme
    (§5 rule 6), so the honest answer to a caller without it is that this is not
    for them.
    """
    refused = await client.get("/")

    assert refused.status == 403
    assert "token" in await refused.text()


async def test_the_websocket_is_refused_without_the_token(client: Any) -> None:
    """The socket is the interesting door, and it is the one a gate can miss.

    `/` is obvious; `/ws` is where the terminal's keystrokes go, and its URL is
    built by upstream's template rather than by this module — so it cannot carry
    a query parameter and would have been the path a cookie-only scheme left
    open. Asserted separately for that reason.
    """
    assert (await client.get("/ws")).status == 403
    assert (await client.get("/static/does-not-matter.js")).status == 403
    assert (await client.get("/nothing-here")).status == 403


async def test_the_token_is_exchanged_for_a_cookie(server: WebServer, client: Any) -> None:
    """One paste authorises the page *and* everything the page then fetches.

    The websocket and asset URLs come out of upstream's template, so they carry
    no token — the cookie is what makes them work. Asserted end to end: the
    shell renders with the query token, and a *second* request with no query at
    all is allowed because the first set the cookie.

    Sabotage: drop the `set_cookie` and the page loads with no terminal in it,
    which is the failure nobody would attribute to the gate.
    """
    page = await client.get("/", params={TOKEN_QUERY: server.token})

    assert page.status == 200
    assert COOKIE in client.session.cookie_jar.filter_cookies(page.url)
    # The shell really is upstream's, and really rendered — the template puts the
    # websocket URL into the page for the browser to open. Which also makes this
    # the test that catches a missing `aiohttp_jinja2.setup` or a renamed route:
    # either one 500s here.
    assert "ws" in (await page.text()).lower()

    assert (await client.get("/static/", allow_redirects=False)).status != 403


async def test_a_wrong_token_is_refused_like_none_at_all(server: WebServer, client: Any) -> None:
    """No hint, and no timing signal.

    Compared with `compare_digest`, because a token is a secret and `==` returns
    early on the first differing byte — which is a prefix oracle for anyone who
    can measure the reply.
    """
    assert (await client.get("/", params={TOKEN_QUERY: server.token[:-1]})).status == 403
    assert (await client.get("/", params={TOKEN_QUERY: ""})).status == 403


def test_the_launch_url_carries_the_token_and_the_bind() -> None:
    """What `--open` opens, and what a person pastes.

    The token has to be *in* the URL: there is nowhere else to put it, since the
    first request is the one that needs authorising.
    """
    server = a_server(port=7777)

    assert server.url == f"http://127.0.0.1:7777/?token={server.token}"


def test_every_launch_mints_its_own_token() -> None:
    """Per launch, so closing the process ends the grant.

    A fixed or derived token would be one a person could reuse against a later
    run they were not meant to reach — and there is no session store to revoke
    it in.
    """
    assert a_server().token != a_server().token


def test_a_non_loopback_bind_says_what_it_costs() -> None:
    """The exposure sentence knows its own host, and its own session.

    Both are facts a person needs *before* the bind and neither is one `cli.py`
    can be trusted to re-derive — it used to compare against a `"127.0.0.1"`
    literal it did not own. The session line is the surprise: a second tab is not
    a second conversation, unlike a second terminal.
    """
    local = list(a_server("shared-id").notices())
    public = list(a_server(host="0.0.0.0").notices())

    assert any("authority" in one for one in local), "the token's cost is always said"
    assert any("shared-id" in one for one in local), "and which session the tabs land on"
    assert not any("loopback" in one for one in local)
    assert any("loopback" in one for one in public), "and a public bind says more"


# ------------------------------------------------------- the browser's bytes --


async def test_the_page_offers_somewhere_to_drop_a_file(server: WebServer, client: Any) -> None:
    """The drop zone is inserted into a page this module does not own.

    The marker is asserted because it is the one thing insertion can break —
    `_index`'s docstring says why insertion rather than a fork.

    Sabotage: stop inserting, and a person can see the terminal and has no way to
    give it a file from their own machine.
    """
    page = await client.get("/", params={TOKEN_QUERY: server.token})
    body = await page.text()

    assert CLOSING_BODY in body, "upstream's page no longer ends the way we insert into"
    assert 'id="ph-drop"' in body
    assert "/api/attachments" in body, "and the zone knows where to post"


async def test_an_upload_needs_the_token_like_everything_else(client: Any) -> None:
    """The interesting door, again: this one *writes*.

    A route that took bytes from anyone would be worse than a readable shell —
    it stages a file onto a live session's tray, which the next prompt carries
    to the model.
    """
    assert (await client.post("/api/attachments")).status == 403


async def test_a_post_that_is_not_a_dropped_file_is_refused_not_traced(
    server: WebServer, client: Any
) -> None:
    """One multipart part named `file` — anything else gets a sentence.

    The route takes the *first* part rather than scanning for one, because the
    only producer is the drop zone above it and a scan invents a shape nothing
    sends. So the narrowing has to be **answered**, not raised through: the
    caller might be a person with `curl`, and getting it slightly wrong is not a
    server error.

    The empty-body case is the one that bit, found by hand rather than here:
    `request.multipart()` *asserts* on a body that is not multipart, so a plain
    POST arrived as a 500 and an `AssertionError` — the same defect as a daemon
    refusal arriving as one, one route over.

    Sabotage: drop either guard and this is a 500 whose text is a traceback.
    """
    named_wrong = FormData()
    named_wrong.add_field("upload", PNG, filename="diagram.png", content_type="image/png")

    refused = await client.post(
        "/api/attachments", params={TOKEN_QUERY: server.token}, data=named_wrong
    )
    assert refused.status == 400
    assert "'file'" in await refused.text()

    # No body at all — `FormData()` with no fields is not even multipart.
    plain = await client.post("/api/attachments", params={TOKEN_QUERY: server.token})
    assert plain.status == 415
    assert "'file'" in await plain.text()


async def test_an_upload_with_no_daemon_says_so(server: WebServer, client: Any) -> None:
    """No daemon means no session to stage onto, and that is a sentence.

    The *tabs* are what start a daemon, so a person who has not opened one has
    nothing for a file to land on. Starting one here would mount a session
    nobody is attached to in order to hold a file nobody asked for.
    """
    refused = await drop(server, client)

    assert refused.status == 503
    assert "no daemon" in await refused.text()


async def test_an_upload_before_any_tab_exists_says_what_to_do(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A daemon refusal is not a server error, and this is the one that bit.

    A daemon is running but nothing has created the session yet — the state a
    person is in between launching `--mode web` and opening a tab, since the tabs
    are what create it. It arrived as a **500** and an `ExceptionGroup`
    traceback; `connected`'s docstring says why the obvious `except` saw nothing.

    The daemon's *own* sentence is asserted, not a sentence this route invents:
    `_refused` keeps it and adds advice only for this reason.

    Sabotage: drop the mapping and this is a 500 whose text tells a person
    nothing.
    """
    server = a_server("not-open-yet")
    async with serving(tmp_path, monkeypatch), browser_on(server) as browser:
        refused = await drop(server, browser)

        assert refused.status == 503
        text = await refused.text()
        assert "not-open-yet" in text, "the daemon's own sentence reaches the person"
        assert "open a tab" in text


async def test_an_oversized_upload_is_refused_before_it_is_buffered(
    server: WebServer, client: Any
) -> None:
    """The same ceiling the daemon enforces, applied where the bytes arrive.

    `attachment/put` refuses over 5 MiB because one frame cannot carry it — and a
    refusal that arrives *after* the whole file has been read into this process
    and base64'd is one that spent the memory anyway. So the read is chunked
    against the limit and stops at it. Nothing else would: aiohttp applies
    `client_max_size` to `read()` and `post()`, never to `read_chunk`.
    """
    refused = await drop(
        server,
        client,
        b"\x00" * (MAX_ATTACHMENT_BYTES + 1024),
        "big.bin",
        "application/octet-stream",
    )

    assert refused.status == 413
    assert "MiB" in await refused.text()


async def test_a_browser_uploaded_file_reaches_the_model_as_a_media_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate this increment exists for**, end to end (P7-06).

    Bytes leave a browser, land on the root's shared tray, and the next prompt —
    from *any* attached front end, here a plain client — carries them to the
    model as a `media` block. Asserted on the `user/message` the log kept,
    because that is what `derive_messages` sends: a 201 reply would pass for a
    server that dropped the file on the floor.

    Sabotage: stop draining `staged` in `Supervisor.prompt`, and the turn goes as
    plain text with nothing saying the picture never went.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
        root = await daemon.root("dropped")
        server = a_server(root.id)
        async with browser_on(server) as browser:
            sent = await drop(server, browser)

            assert sent.status == 201
            reference = await sent.json()
            assert reference["attachmentId"].startswith("sha256:")
            assert reference["name"] == "diagram.png"

            # On the *root's* tray, which is what every attached UI sees.
            assert [one.name for one in root.staged.refs] == ["diagram.png"]

        # Any client's next prompt carries it — the tray is the root's, not a
        # front end's.
        client = await daemon.client()
        await client.prompt(root.id, "look at this")
        await until(
            lambda: root.session.latest("user/message") is not None,
            what="the prompt to reach the log",
        )
        message = root.session.latest("user/message")
        assert message is not None
        content = message.data["content"]
        assert [one["type"] for one in content] == ["text", "media"]
        assert content[1]["attachment"]["attachmentId"] == reference["attachmentId"]


async def test_a_browser_that_cannot_name_the_type_still_lands_an_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dropped file is classified the same way `--attach` would classify it.

    Browsers send `application/octet-stream` for any extension they do not know,
    and a route that took the declared type literally would store a `.png` as a
    document — wrong extension out of `EXTENSIONS`, missed by `IMAGE_MIMES`, and
    a picture reaching the model as a file. `mime_for` is the one ladder both
    doors climb.

    Asserted on the *stored* reference rather than on the reply, because that is
    what a later turn reads.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
        root = await daemon.root("shrugged")
        server = a_server(root.id)
        async with browser_on(server) as browser:
            sent = await drop(server, browser, mime="application/octet-stream")

            assert sent.status == 201
            assert (await sent.json())["mime"] == "image/png"
            assert [one.mime for one in root.staged.refs] == ["image/png"]


async def test_the_same_bytes_dropped_twice_are_one_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Content-addressed, so re-dropping a file is cheap rather than a second copy.

    Its own test because it is its own claim: the end-to-end one above is about
    the *route* — bytes to a model — and this is about the *store*. Folded
    together, a failure here read as a failure of the gate the increment is named
    for.
    """
    async with serving(tmp_path, monkeypatch) as daemon:
        root = await daemon.root("twice")
        server = a_server(root.id)
        async with browser_on(server) as browser:
            once = await drop(server, browser)
            again = await drop(server, browser)

            assert (await once.json())["attachmentId"] == (await again.json())["attachmentId"]
            assert [one.name for one in root.staged.refs] == ["diagram.png"], "one chip, not two"
