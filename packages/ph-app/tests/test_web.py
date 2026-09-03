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
from typing import Any

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from ph_app.web.serve import COOKIE, TOKEN_QUERY, WebServer

pytestmark = pytest.mark.anyio


@pytest.fixture
def server() -> WebServer:
    """One composed server whose tabs would run a command that never runs."""
    return WebServer(command="/bin/true")


@pytest.fixture
async def client(server: WebServer) -> AsyncIterator[TestClient[web.Request, web.Application]]:
    """A client on that server, torn down with the test.

    A fixture because four tests want the same three lines, and because the
    server has to be the *same* object the test asserts the token of."""
    async with TestClient(TestServer(server.application())) as opened:
        yield opened


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
    server = WebServer(command="/bin/true", port=7777)

    assert server.url == f"http://127.0.0.1:7777/?token={server.token}"


def test_every_launch_mints_its_own_token() -> None:
    """Per launch, so closing the process ends the grant.

    A fixed or derived token would be one a person could reuse against a later
    run they were not meant to reach — and there is no session store to revoke
    it in.
    """
    assert WebServer(command="/bin/true").token != WebServer(command="/bin/true").token


def test_a_non_loopback_bind_says_what_it_costs() -> None:
    """One owner for the exposure sentence, and it knows the host.

    `notices()` holds the words and `cli.py` prints them — the shape `ph daemon`
    uses for its socket path and its linger warning. It was said in both places,
    in two styles, on two sides of the bind; and the CLI's copy compared against
    a `"127.0.0.1"` literal it did not own.
    """
    local = list(WebServer(command="/bin/true").notices())
    public = list(WebServer(command="/bin/true", host="0.0.0.0").notices())

    assert any("authority" in one for one in local), "the token's cost is always said"
    assert not any("loopback" in one for one in local)
    assert any("loopback" in one for one in public), "and a public bind says more"
