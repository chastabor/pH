"""P7-03 on the OpenAI wire — the second of the three file APIs the row names.

`test_uploads.py` proves the *seam*: upload once, reference thereafter, and an
expired handle costs one retry rather than the turn. None of that is re-proved
here. What is this wire's own, and what a second provider is the only way to find
out, is four things:

* `purpose` — the one form field this Files API requires, which is why
  `post_multipart` grew a `data` argument instead of this adapter building its own
  request and inheriting none of the status→code classification;
* `expires_at` — stated per file here where Anthropic states none, in unix
  *seconds* against a `FileHandle` in milliseconds;
* the `file` content part, which has **two** spellings — a `file_id` for an
  uploaded document and inline `file_data` for one that was not;
* and the intersection rule, which is where this row deliberately diverges from
  the Anthropic one.

The provider is simulated for `test_uploads.py`'s reason: there is no OpenAI key
in CI, and a file API that hands out ids and then forgets one is exactly what a
test needs to be able to *cause*.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.bundles import BASE, HEADLESS
from ph.cordis import Context
from ph.llm.types import FILE_EXPIRED, MediaBlock, create_user_message
from ph.seams.attachments import digest_of
from ph_app.adapters._http import HttpClient, failure_from_status
from ph_app.adapters.openai_compatible import _is_missing_file, _is_overflow

pytestmark = pytest.mark.anyio

PROFILE = [BASE, HEADLESS]
PDF = b"%PDF-1.7\n" + b"pages" * 64
OPTIONS = AgentOptions(provider="openai", model="gpt-test")


def _route(**config: Any) -> dict[str, Any]:
    profile = {
        "provider": "openai",
        "apiKeyEnv": "OPENAI_API_KEY",
        "accepts": ["application/pdf"],
        **config,
    }
    return {
        "insert": [
            {
                "id": "llm-openai",
                "name": "llm-openai-compatible",
                "config": {"profiles": [profile]},
            }
        ]
    }


REFERENCING = _route(uploads=["application/pdf"])


def _reply(text: str) -> list[tuple[str, dict[str, Any]]]:
    """One text answer in this wire's streaming shape."""
    return [
        ("", {"choices": [{"delta": {"content": text}}]}),
        ("", {"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        ("", {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 1}}),
    ]


class _FileApi:
    """A provider that hands out file ids and can be made to forget one."""

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.forms: list[dict[str, Any]] = []
        self.forgotten: set[str] = set()
        self.bodies: list[dict[str, Any]] = []
        self.expires_at: int | None = None

    def issue(self) -> str:
        handle = f"file-{len(self.uploaded) + 1:03d}"
        self.uploaded.append(handle)
        return handle

    def referenced(self, body: dict[str, Any]) -> list[str]:
        return [
            str(part["file"]["file_id"])
            for message in body.get("messages") or []
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if isinstance(part.get("file"), dict) and part["file"].get("file_id")
        ]

    def events(self, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        self.bodies.append(body)
        gone = [one for one in self.referenced(body) if one in self.forgotten]
        if gone:
            # Through the real classifier rather than a hand-made `LlmError`: this
            # stub stands in for the transport, and the transport is what turns a
            # body into a code.
            raise failure_from_status(
                400,
                f'{{"error":{{"message":"No such File object: {gone[0]}",'
                f'"type":"invalid_request_error"}}}}',
                is_overflow=_is_overflow,
                is_missing_file=_is_missing_file,
            )
        return _reply("seen")


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _FileApi:
    api = _FileApi()

    async def post_multipart(self: HttpClient, url: str, **kwargs: Any) -> dict[str, Any]:
        api.forms.append({"url": url, **kwargs})
        reply: dict[str, Any] = {"id": api.issue()}
        if api.expires_at is not None:
            reply["expires_at"] = api.expires_at
        return reply

    async def stream_sse(
        self: HttpClient, url: str, **kwargs: Any
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        for event in api.events(kwargs["json"]):
            yield event

    monkeypatch.setattr(HttpClient, "post_multipart", post_multipart)
    monkeypatch.setattr(HttpClient, "stream_sse", stream_sse)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    return api


async def _attached(ctx: Context, mime: str = "application/pdf", name: str = "paper.pdf") -> Any:
    ref = await ctx.attachments.save_bytes(content=PDF, mime=mime, name=name)
    return create_user_message(
        content=[{"type": "text", "text": "what happens here?"}, MediaBlock(attachment=ref)],
        source={"kind": "user"},
    )


async def test_a_document_is_uploaded_with_a_purpose_and_then_referenced(
    mount: Any, wire: _FileApi
) -> None:
    """The shape half of the gate on this wire.

    `purpose` rides the multipart body, the request carries a `file` part naming
    the id, and the bytes are nowhere in it — which is the whole point, since a
    4 MB PDF is 5.5 MB of base64 on every step of the session otherwise.
    """
    ctx: Context = await mount(REFERENCING, profile=PROFILE)
    session = ctx.sessions.create("referenced")
    agent = ctx.agents.create(session, OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.uploaded == ["file-001"]
    assert wire.forms[0]["data"] == {"purpose": "user_data"}
    assert wire.forms[0]["url"].endswith("/files")
    (body,) = wire.bodies
    assert wire.referenced(body) == ["file-001"]
    assert "base64" not in str(body), "a referenced file must not also be inlined"
    # The fact is in the log and the handle is not.
    (record,) = [one for one in session.events if one.type == "attachment/uploaded"]
    assert record.data["provider"] == "openai" and record.data["name"] == "paper.pdf"
    assert "file-001" not in str(record.data)


async def test_a_stated_expiry_is_read_in_the_units_the_provider_used(
    mount: Any, wire: _FileApi
) -> None:
    """Seconds on the wire, milliseconds in the cache.

    Anthropic states no per-file expiry, so this wire is the first to exercise the
    field at all — and the conversion is the kind of mistake that is invisible in
    the cheap direction: a handle stamped in seconds reads as expired in 1970, so
    every request re-uploads and everything still *works*, slowly and expensively.
    """
    wire.expires_at = 2_000_000_000  # seconds
    ctx: Context = await mount(REFERENCING, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("expiry"), OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()
    await agent.prompt("again")

    stored = ctx.uploads.cached("openai", digest_of(PDF))
    assert stored is not None and stored.expires_at == 2_000_000_000_000
    assert wire.uploaded == ["file-001"], "a live handle was treated as expired"


async def test_an_expired_handle_is_re_uploaded_rather_than_failing_the_turn(
    mount: Any, wire: _FileApi
) -> None:
    """The same recovery as the Anthropic row, through this wire's own prose.

    What differs between the two is only how a provider *says* a file is gone,
    which is why the phrases stay per-adapter and the action they trigger is
    shared. The turn must survive either way.
    """
    ctx: Context = await mount(REFERENCING, profile=PROFILE)
    session = ctx.sessions.create("expired")
    agent = ctx.agents.create(session, OPTIONS)
    agent.followup(await _attached(ctx))
    await agent.run()

    wire.forgotten.add("file-001")
    await agent.prompt("still there?")

    assert wire.uploaded == ["file-001", "file-002"]
    assert wire.referenced(wire.bodies[-1]) == ["file-002"]
    assert session.events[-1].type == "turn/end"
    (retry,) = [one for one in session.events if one.type == "llm/retry"]
    assert retry.data["code"] == FILE_EXPIRED


async def test_a_404_that_names_no_handle_of_ours_is_not_retried(
    mount: Any, wire: _FileApi
) -> None:
    """A missing-file message about somebody else's id is somebody else's problem.

    Retrying it would be the "unknown failure billed twice" the retry policy
    exists to refuse, so the check is both halves: the provider said a file is
    missing, **and** it named one this request sent.
    """
    ctx: Context = await mount(REFERENCING, profile=PROFILE)
    session = ctx.sessions.create("stranger")
    agent = ctx.agents.create(session, OPTIONS)
    agent.followup(await _attached(ctx))
    await agent.run()

    wire.forgotten.add("file-999")
    await agent.prompt("and now?")

    assert wire.uploaded == ["file-001"]
    assert not [one for one in session.events if one.type == "llm/retry"]


async def test_a_declared_mime_this_wire_cannot_reference_is_not_uploaded(
    mount: Any, wire: _FileApi
) -> None:
    """Where this row diverges from the Anthropic one, and why it must.

    There, a declared MIME with no wire shape (video) is uploaded and then
    degrades to a pointer — the pointer was the outcome either way, because
    Anthropic has no video content block at all, and only the upload is wasted.
    Here an image *is* expressible inline, so uploading one this wire cannot
    reference would turn a picture that works today into a pointer, since
    `load_media` skips the bytes of anything that has a handle. A configuration
    mistake must not be able to remove a capability, so the declaration is
    intersected with what the renderer can express.
    """
    route = _route(accepts=["image/png"], uploads=["image/png"])
    ctx: Context = await mount(route, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("image"), OPTIONS)

    agent.followup(await _attached(ctx, mime="image/png", name="shot.png"))
    await agent.run()

    assert wire.uploaded == [], "nothing this wire cannot reference should be uploaded"
    assert "image_url" in str(wire.bodies[-1]), "and the picture still reaches the model"


async def test_a_route_that_declares_no_uploads_sends_bytes_as_before(
    mount: Any, wire: _FileApi
) -> None:
    """The default every existing deployment keeps.

    Most servers speaking this shape implement `/chat/completions` and not
    `/files`, so an uploader registered for every profile would put a file API
    behind a provider name that has none. With `uploads` unset the document rides
    inline as `file_data` — which is also the fallback when an upload fails.
    """
    ctx: Context = await mount(_route(), profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("inline"), OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.uploaded == []
    assert ctx.uploads.uploaders == {}, "no uploader for a route that wants none"
    (part,) = [
        block
        for message in wire.bodies[-1]["messages"]
        if isinstance(message.get("content"), list)
        for block in message["content"]
        if block.get("type") == "file"
    ]
    assert part["file"]["file_data"].startswith("data:application/pdf;base64,")


async def test_a_failed_upload_falls_back_to_the_bytes(
    mount: Any, wire: _FileApi, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The seam's contract, kept by this wire's inline spelling.

    "An upload that fails for any reason leaves the id out and the attachment goes
    inline, which is what every route did before this row" — that promise is only
    true where the renderer *has* an inline form. Without one, a file API having a
    bad minute would degrade a document the route can send perfectly well.
    """

    async def failing(self: HttpClient, url: str, **kwargs: Any) -> dict[str, Any]:
        raise OSError("the file API is down")

    monkeypatch.setattr(HttpClient, "post_multipart", failing)
    ctx: Context = await mount(REFERENCING, profile=PROFILE)
    session = ctx.sessions.create("fallback")
    agent = ctx.agents.create(session, OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert session.events[-1].type == "turn/end", "a file API outage is not a lost turn"
    assert "file_data" in str(wire.bodies[-1]), "the document went inline"
    assert not [one for one in session.events if one.type == "attachment/uploaded"]
