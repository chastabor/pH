"""P7-03's gate — a file sent by reference, and a dead handle that costs one retry.

Two claims, and the second is the one the row is really about.

**Upload once, reference thereafter.** A route that declares `uploads` sends the
bytes to a provider's file API and puts an id on the wire, so a 4 MB file is
uploaded once instead of base64-encoded onto every step of the session.

The worked example here is a PDF through Anthropic's Files API, and that is a
deliberate narrowing of the row's own framing. Video is what makes
upload-and-reference *necessary* rather than merely cheaper — but **neither
provider pH has an adapter for accepts video**, so a route declaring it would be
a fiction, and this file would be asserting against a wire shape that does not
exist. The last test below pins what actually happens to one today. The mechanism
is the same one Gemini's Files API needs; what is missing is that adapter, not
this path.

**An expired handle is a retry, not a lost turn.** The cache is a prediction, and
providers expire files early, delete them from another session, or forget them.
That surfaces mid-request, and what must not happen is an hour of conversation
ending because a cache entry went stale. The adapter forgets the handle and
raises `FILE_EXPIRED`, which `llm-retry` already treats as transient — so the next
attempt rebuilds the request against a fresh upload.

The provider is simulated: there is no Anthropic key in CI, and a file API that
returns ids and then stops honouring one is exactly what a test needs to be able
to *cause*. Everything above the wire is real — the seam, its on-disk cache, the
adapter, the retry row and the agent loop.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.bundles import BASE, HEADLESS
from ph.cordis import Context
from ph.llm.types import FILE_EXPIRED, MediaBlock, create_user_message
from ph.testing import anthropic_reply
from ph_app.adapters._http import HttpClient, failure_from_status
from ph_app.adapters.anthropic import _is_missing_file, _is_overflow

pytestmark = pytest.mark.anyio

PROFILE = [BASE, HEADLESS]
PDF = b"%PDF-1.7\n" + b"pages" * 64
ROUTE = {
    "insert": [
        {
            "id": "llm-anthropic",
            "name": "llm-anthropic",
            "config": {
                "provider": "anthropic",
                "apiKeyEnv": "ANTHROPIC_API_KEY",
                "accepts": ["application/pdf"],
                "uploads": ["application/pdf"],
            },
        }
    ]
}
OPTIONS = AgentOptions(provider="anthropic", model="claude-test")


class _FileApi:
    """A provider that hands out file ids and can be made to forget one."""

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.forgotten: set[str] = set()
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []

    def issue(self) -> str:
        handle = f"file_{len(self.uploaded) + 1:03d}"
        self.uploaded.append(handle)
        return handle

    def referenced(self, body: dict[str, Any]) -> list[str]:
        return [
            str(block["source"]["file_id"])
            for message in body.get("messages") or []
            for block in message.get("content") or []
            if isinstance(block.get("source"), dict) and block["source"].get("type") == "file"
        ]

    def events(self, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        self.bodies.append(body)
        gone = [one for one in self.referenced(body) if one in self.forgotten]
        if gone:
            # Through the real classifier rather than a hand-made `LlmError`:
            # this stub stands in for the transport, and the transport is what
            # turns a body into a code. A stub that skipped it would let the
            # adapter's half pass while the shared half was broken.
            raise failure_from_status(
                404,
                f'{{"type":"error","error":{{"type":"not_found_error",'
                f'"message":"File {gone[0]} not found"}}}}',
                is_overflow=_is_overflow,
                is_missing_file=_is_missing_file,
            )
        return anthropic_reply("seen", usage={"input_tokens": 5, "output_tokens": 0})


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _FileApi:
    api = _FileApi()

    async def post_multipart(self: HttpClient, url: str, **kwargs: Any) -> dict[str, Any]:
        return {"id": api.issue()}

    async def stream_sse(
        self: HttpClient, url: str, **kwargs: Any
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        api.headers.append(dict(kwargs["headers"]))
        for event in api.events(kwargs["json"]):
            yield event

    monkeypatch.setattr(HttpClient, "post_multipart", post_multipart)
    monkeypatch.setattr(HttpClient, "stream_sse", stream_sse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return api


async def _attached(ctx: Context, mime: str = "application/pdf", name: str = "paper.pdf") -> Any:
    ref = await ctx.attachments.save_bytes(content=PDF, mime=mime, name=name)
    return create_user_message(
        content=[{"type": "text", "text": "what happens here?"}, MediaBlock(attachment=ref)],
        source={"kind": "user"},
    )


async def test_a_file_is_uploaded_once_and_then_referenced(mount: Any, wire: _FileApi) -> None:
    """The half of the gate that is about shape.

    The file goes up once and the request carries an id. Asserted on the body
    rather than on the store, because "was it uploaded" and "was it *referenced*"
    are different questions and only the second is what gets a format onto a wire
    that will not take it inline.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("video")
    agent = ctx.agents.create(session, OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.uploaded == ["file_001"]
    (body,) = wire.bodies
    assert wire.referenced(body) == ["file_001"]
    blob = str(body)
    assert "base64" not in blob, "a referenced file must not also be inlined"
    # And the fact is in the log, while the handle is not: a person auditing
    # where their data went reads this; nothing reads it back to find an id.
    (record,) = [one for one in session.events if one.type == "attachment/uploaded"]
    assert record.data["provider"] == "anthropic" and record.data["name"] == "paper.pdf"
    assert "file_001" not in str(record.data)


async def test_the_second_turn_reuses_the_handle(mount: Any, wire: _FileApi) -> None:
    """What the cache is for, and the reason it is keyed on the digest.

    The same bytes are not uploaded twice — not on the next step of this turn,
    not on the next turn, and (because the key is the content digest rather than
    the session) not from a second session either.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("reuse"), OPTIONS)
    agent.followup(await _attached(ctx))
    await agent.run()

    await agent.prompt("and now?")
    other = ctx.agents.create(ctx.sessions.create("second-session"), OPTIONS)
    other.followup(await _attached(ctx))
    await other.run()

    assert wire.uploaded == ["file_001"], "one upload for one set of bytes"
    assert all(wire.referenced(body) == ["file_001"] for body in wire.bodies if body["messages"])


async def test_an_expired_handle_is_re_uploaded_rather_than_failing_the_turn(
    mount: Any, wire: _FileApi
) -> None:
    """The gate's other half, and the failure it is written against.

    The provider forgets the file between one turn and the next — early expiry, a
    deletion from elsewhere, a bad day. The turn must still answer. `FILE_EXPIRED`
    is in `TRANSIENT_CODES` because by the time it is raised the dead entry is
    already gone from the cache, so the retry rebuilds against a fresh upload
    instead of repeating a request that cannot work.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("expired")
    agent = ctx.agents.create(session, OPTIONS)
    agent.followup(await _attached(ctx))
    await agent.run()

    wire.forgotten.add("file_001")
    await agent.prompt("still there?")

    assert wire.uploaded == ["file_001", "file_002"], "the dead handle was not replaced"
    assert wire.referenced(wire.bodies[-1]) == ["file_002"]
    assert session.events[-1].type == "turn/end"
    assert not [one for one in session.events if one.type == "turn/error"]
    # The retry is recorded under the code that explains it, not as an unknown.
    (retry,) = [one for one in session.events if one.type == "llm/retry"]
    assert retry.data["code"] == FILE_EXPIRED


async def test_a_404_that_names_no_handle_of_ours_is_not_retried(
    mount: Any, wire: _FileApi
) -> None:
    """The half of the classification that keeps it honest.

    A `not_found` from a gateway in front of the API is about the *route*, and
    retrying it would be the "unknown failure billed twice" the retry policy
    exists to refuse. So the check is both halves: the provider said a file is
    missing, **and** it named one this request sent.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("stranger")
    agent = ctx.agents.create(session, OPTIONS)
    agent.followup(await _attached(ctx))
    await agent.run()

    wire.forgotten.add("file_999")  # somebody else's id
    wire.bodies.clear()
    await agent.prompt("and now?")

    assert wire.uploaded == ["file_001"], "nothing was re-uploaded for a stranger's 404"
    assert not [one for one in session.events if one.type == "llm/retry"]


async def test_a_route_that_declares_no_uploads_sends_bytes_as_before(
    mount: Any, wire: _FileApi
) -> None:
    """The default, and the one every existing deployment keeps.

    `uploads` is empty unless a deployment opts in, because the Files API is a
    beta an account may not have — so with it unset nothing is uploaded, no beta
    header is sent, and a PDF rides inline exactly as it did before this row.
    """
    route = {
        "insert": [
            {
                "id": "llm-anthropic",
                "name": "llm-anthropic",
                "config": {
                    "provider": "anthropic",
                    "apiKeyEnv": "ANTHROPIC_API_KEY",
                    "accepts": ["application/pdf"],
                },
            }
        ]
    }
    ctx: Context = await mount(route, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("inline"), OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.uploaded == []
    assert "base64" in str(wire.bodies[-1])
    assert ctx.uploads.uploaders == {}, "no uploader is registered for a route that wants none"


async def test_a_format_this_wire_cannot_express_degrades_even_when_uploaded(
    mount: Any, wire: _FileApi
) -> None:
    """Where video actually stands today, asserted rather than implied.

    A deployment can declare `accepts: [video/mp4]` on this route and the upload
    will happen — but Anthropic has no video content block, so the renderer has no
    shape to put the id in, and P7-01's rule takes over: *total over its own
    vocabulary*, a MIME it cannot express becomes a pointer rather than being
    dressed as an image. The model is told, the turn survives, and nothing claims
    a capability the provider does not have.

    This is the honest state of the row's video gate: the transport is built and
    the wire that needs it is a provider pH has no adapter for yet.
    """
    route = {
        "insert": [
            {
                "id": "llm-anthropic",
                "name": "llm-anthropic",
                "config": {
                    "provider": "anthropic",
                    "apiKeyEnv": "ANTHROPIC_API_KEY",
                    "accepts": ["video/mp4"],
                    "uploads": ["video/mp4"],
                },
            }
        ]
    }
    ctx: Context = await mount(route, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("video"), OPTIONS)

    agent.followup(await _attached(ctx, mime="video/mp4", name="clip.mp4"))
    await agent.run()

    assert wire.uploaded == ["file_001"], "the transport works"
    assert wire.referenced(wire.bodies[-1]) == [], "and this wire has nowhere to put the id"
    assert "was not sent" in str(wire.bodies[-1]), "so the model reads a pointer"


async def test_only_the_handle_the_provider_named_is_forgotten(mount: Any, wire: _FileApi) -> None:
    """The other half of the invalidation, and a bug the first draft shipped.

    A request carries two files and the provider says one of them is gone. A
    draft invalidated *every* handle the request referenced on a match — which on
    a twenty-file conversation throws away nineteen live uploads to replace one
    dead one, and pays for all twenty again on the retry.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("pair")
    agent = ctx.agents.create(session, OPTIONS)
    first = await ctx.attachments.save_bytes(content=PDF, mime="application/pdf", name="a.pdf")
    second = await ctx.attachments.save_bytes(
        content=PDF + b"other", mime="application/pdf", name="b.pdf"
    )
    agent.followup(
        create_user_message(
            content=[
                {"type": "text", "text": "both please"},
                MediaBlock(attachment=first),
                MediaBlock(attachment=second),
            ],
            source={"kind": "user"},
        )
    )
    await agent.run()
    assert wire.uploaded == ["file_001", "file_002"]

    wire.forgotten.add("file_001")
    await agent.prompt("still there?")

    # Exactly one replacement: the survivor keeps its id across the retry.
    assert wire.uploaded == ["file_001", "file_002", "file_003"]
    assert wire.referenced(wire.bodies[-1]) == ["file_003", "file_002"]


async def test_a_referenced_file_is_never_also_encoded(mount: Any, wire: _FileApi) -> None:
    """What makes an upload cheaper rather than merely different.

    Before this, a referenced attachment was still read and base64-encoded for a
    request that would not carry the bytes — so the wire payload shrank and
    nothing else did, and the discarded 1.37x-sized string stayed in the store's
    encode cache for the life of the process. Asserted against that cache, which
    is where the cost actually landed.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("unencoded"), OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.uploaded == ["file_001"], "it went up"
    assert ctx.attachments._encoded == {}, "and was never encoded to be thrown away"


async def test_the_files_beta_rides_only_requests_that_reference_one(
    mount: Any, wire: _FileApi
) -> None:
    """A capability the deployment declared must not become an outage on requests
    that never used it.

    Keying the beta header on `config.uploads` put it on *every* request a
    configured route sent, so an account without the beta would fail the plain
    text turns too. It is a fact about one request — does this body carry a file
    id — not about the route.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("beta"), OPTIONS)

    await agent.prompt("no attachment here")
    agent.followup(await _attached(ctx))
    await agent.run()

    plain, with_file = wire.headers[0], wire.headers[-1]
    assert "anthropic-beta" not in plain, "a text turn carried a beta it did not need"
    assert with_file["anthropic-beta"] == "files-api-2025-04-14"


async def test_a_second_provider_re_uploads_from_the_same_stored_blob(
    mount: Any, wire: _FileApi
) -> None:
    """Running the same attachment against another route costs an upload, not a file.

    The store keeps originals and nothing about uploading touches them: the blob
    is content-addressed under `$PH_HOME/attachments` and the handle cache is
    keyed on `(provider, digest)`, so a second provider is a *miss* that
    re-uploads from bytes already on disk. That is what makes "try this
    conversation on a different model" a thing a person can do after the fact.

    The failure this pins is silent and bad: a cache keyed on the digest alone
    would hand provider B an id that provider A issued — a wire error at best,
    and at worst a reference to somebody else's file.
    """
    two_routes = {
        "insert": [
            {
                "id": f"llm-{name}",
                "name": "llm-anthropic",
                "config": {
                    "provider": name,
                    "apiKeyEnv": "ANTHROPIC_API_KEY",
                    "accepts": ["application/pdf"],
                    "uploads": ["application/pdf"],
                },
            }
            for name in ("first", "second")
        ]
    }
    ctx: Context = await mount(two_routes, profile=PROFILE)

    for name in ("first", "second"):
        agent = ctx.agents.create(ctx.sessions.create(name), AgentOptions(provider=name, model="m"))
        agent.followup(await _attached(ctx))
        await agent.run()

    assert wire.uploaded == ["file_001", "file_002"], "each route needs its own copy"
    assert [wire.referenced(body) for body in wire.bodies] == [["file_001"], ["file_002"]]
    # One set of bytes, kept: the originals are not the provider's copy.
    assert len(list(ctx.attachments.root.iterdir())) == 1
    assert sorted(path.parent.name for path in ctx.uploads.root.rglob("*.json")) == [
        "first",
        "second",
    ]
