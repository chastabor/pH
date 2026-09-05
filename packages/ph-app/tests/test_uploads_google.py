"""P7-03's gate, finally driven through a route that accepts a video.

The row's own note said what was missing: *"neither provider pH has an adapter
for accepts video, so a shipped route declaring it would be a fiction… that half
of the gate lands with a Gemini adapter."* This is that half.

**A video reaches a route that accepts one** — through the Files API, because
this provider bounds a whole request at 20 MB and a clip is not under it, and
because a `fileUri` cannot be referenced until the file it names is `ACTIVE`.
That last part is what makes this uploader different in kind from its two
siblings: it is not finished when the bytes have landed, so a test here has to be
able to keep a file `PROCESSING` and watch what the adapter does about it.

The wire itself is exercised alongside, because every name in it is different and
three of the mappings are places where the obvious guess is wrong: thinking
tokens counted outside the answer's, a function call with no id, and a thought
that has no way back.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.bundles import BASE, HEADLESS
from ph.cordis import Context
from ph.llm.types import (
    FILE_EXPIRED,
    MediaBlock,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    create_assistant_message,
    create_tool_result_message,
    create_user_message,
)
from ph.seams.attachments import digest_of
from ph_app.adapters._http import HttpClient, failure_from_status
from ph_app.adapters.google import (
    _call_names,
    _is_missing_file,
    _is_overflow,
    _StreamState,
    _to_google,
    _to_usage,
)

pytestmark = pytest.mark.anyio

PROFILE = [BASE, HEADLESS]
CLIP = b"\x00\x00\x00\x18ftypmp42" + b"frames" * 128
OPTIONS = AgentOptions(provider="google", model="gemini-test")
ROUTE = {
    "insert": [
        {
            "id": "llm-google",
            "name": "llm-google",
            "config": {"provider": "google", "apiKeyEnv": "GEMINI_API_KEY", "uploadReadyMs": 4000},
        }
    ]
}


def _reply(text: str) -> list[tuple[str, dict[str, Any]]]:
    """One text answer in this wire's streaming shape."""
    return [
        ("", {"candidates": [{"content": {"role": "model", "parts": [{"text": text}]}}]}),
        (
            "",
            {
                "candidates": [{"finishReason": "STOP", "content": {"parts": []}}],
                "usageMetadata": {"promptTokenCount": 9, "candidatesTokenCount": 2},
            },
        ),
    ]


class _FileApi:
    """The resumable upload, the state machine after it, and the wire."""

    def __init__(self) -> None:
        self.uploaded: list[str] = []
        self.forgotten: set[str] = set()
        self.bodies: list[dict[str, Any]] = []
        self.starts: list[dict[str, str]] = []
        self.polls = 0
        self.processing = 0
        """How many polls a fresh upload spends in `PROCESSING` before it is ready."""
        self.state: dict[str, int] = {}

    def issue(self) -> str:
        name = f"files/clip{len(self.uploaded) + 1}"
        self.uploaded.append(name)
        self.state[name] = self.processing
        return name

    def record(self, name: str) -> dict[str, Any]:
        remaining = self.state.get(name, 0)
        return {
            "name": name,
            "uri": f"https://files.example/{name}",
            "state": "PROCESSING" if remaining > 0 else "ACTIVE",
            "expirationTime": "2033-05-18T03:33:20Z",
        }

    def referenced(self, body: dict[str, Any]) -> list[str]:
        return [
            str(part["fileData"]["fileUri"])
            for content in body.get("contents") or []
            for part in content.get("parts") or []
            if isinstance(part.get("fileData"), dict)
        ]

    def events(self, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        self.bodies.append(body)
        gone = [one for one in self.referenced(body) if one in self.forgotten]
        if gone:
            raise failure_from_status(
                403,
                f'{{"error":{{"code":403,"message":"You do not have permission to access the '
                f'File {gone[0]} or it may not exist.","status":"PERMISSION_DENIED"}}}}',
                is_overflow=_is_overflow,
                is_missing_file=_is_missing_file,
            )
        return _reply("watched it")


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _FileApi:
    api = _FileApi()

    async def post_raw(self: HttpClient, url: str, **kwargs: Any) -> tuple[dict[str, Any], Any]:
        if url.endswith("/files"):
            api.starts.append(dict(kwargs["headers"]))
            name = api.issue()
            return {}, {"x-goog-upload-url": f"https://upload.example/{name}"}
        name = url.rsplit("/upload.example/", 1)[-1]
        return {"file": api.record(name)}, {}

    async def get_json(self: HttpClient, url: str, **kwargs: Any) -> dict[str, Any]:
        api.polls += 1
        name = "files/" + url.rsplit("/files/", 1)[-1]
        if api.state.get(name):
            api.state[name] -= 1
        return api.record(name)

    async def stream_sse(
        self: HttpClient, url: str, **kwargs: Any
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        assert ":streamGenerateContent" in url, "this wire streams from a method, not a path"
        for event in api.events(kwargs["json"]):
            yield event

    monkeypatch.setattr(HttpClient, "post_raw", post_raw)
    monkeypatch.setattr(HttpClient, "get_json", get_json)
    monkeypatch.setattr(HttpClient, "stream_sse", stream_sse)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    return api


async def _attached(ctx: Context, mime: str = "video/mp4", name: str = "clip.mp4") -> Any:
    ref = await ctx.attachments.save_bytes(content=CLIP, mime=mime, name=name)
    return create_user_message(
        content=[{"type": "text", "text": "what happens in this?"}, MediaBlock(attachment=ref)],
        source={"kind": "user"},
    )


async def test_a_video_reaches_a_route_that_accepts_one(mount: Any, wire: _FileApi) -> None:
    """P7-03's gate, whole, for the first time.

    The transport was built and proven a phase ago against a PDF; what could not
    be proven was the case that made upload-and-reference a row rather than an
    optimisation. The clip goes up through the resumable protocol, the request
    carries a `fileData` part naming it, and the bytes are nowhere in the body.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("video")
    agent = ctx.agents.create(session, OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.uploaded == ["files/clip1"]
    assert wire.starts[0]["X-Goog-Upload-Protocol"] == "resumable"
    assert wire.starts[0]["X-Goog-Upload-Header-Content-Type"] == "video/mp4"
    (body,) = wire.bodies
    assert wire.referenced(body) == ["https://files.example/files/clip1"]
    assert "inlineData" not in str(body), "a referenced file must not also be inlined"
    assert session.events[-1].type == "turn/end"
    # And the fact is in the log while the handle is not.
    (record,) = [one for one in session.events if one.type == "attachment/uploaded"]
    assert record.data["mime"] == "video/mp4"
    assert "files.example" not in str(record.data)


async def test_the_upload_waits_for_the_file_to_become_usable(mount: Any, wire: _FileApi) -> None:
    """The way this uploader differs in kind from its two siblings.

    A video is *processed* after it is stored, and a `fileUri` referenced before
    its file is `ACTIVE` is refused. An uploader that returned when the transfer
    finished would cache a handle whose first use fails — indistinguishable, from
    the cache's point of view, from a provider that expired it, and so a retry
    loop rather than a wait.
    """
    wire.processing = 2
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("processing"), OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert wire.polls == 2, "it waited exactly as long as the file was not ready"
    assert wire.referenced(wire.bodies[-1]) == ["https://files.example/files/clip1"]
    stored = ctx.uploads.cached("google", digest_of(CLIP))
    assert stored is not None and stored.expires_at == 2_000_000_000_000


async def test_a_file_that_never_becomes_ready_falls_back_to_the_bytes(
    mount: Any, wire: _FileApi
) -> None:
    """The budget, and why running out of it is not a lost turn.

    `load_handles` swallows an upload failure and the attachment goes inline —
    which is what every route did before this row. For a clip under the inline cap
    that is a working turn; over it, `media-degrade` turns it into a pointer. Both
    are better than a turn that fails because a transcoder was slow.
    """
    wire.processing = 1_000  # never ready inside the budget
    # A budget of one poll rather than the shipped two minutes: what is under test
    # is what happens when it runs out, and `uploadReadyMs` being row config is
    # exactly what lets a test say so without waiting.
    impatient = {
        "insert": [
            {**ROUTE["insert"][0], "config": {**ROUTE["insert"][0]["config"], "uploadReadyMs": 600}}
        ]
    }
    ctx: Context = await mount(impatient, profile=PROFILE)
    session = ctx.sessions.create("slow")
    agent = ctx.agents.create(session, OPTIONS)

    agent.followup(await _attached(ctx))
    await agent.run()

    assert session.events[-1].type == "turn/end"
    assert wire.referenced(wire.bodies[-1]) == [], "nothing was referenced"
    assert "inlineData" in str(wire.bodies[-1]), "the clip went inline instead"
    assert not [one for one in session.events if one.type == "attachment/uploaded"]


async def test_a_revoked_file_is_re_uploaded_rather_than_failing_the_turn(
    mount: Any, wire: _FileApi
) -> None:
    """This provider's own way of saying a file is gone.

    A file deleted from another session answers `PERMISSION_DENIED` rather than a
    not-found, which is why the phrases stay per-adapter — a shared list would
    have had to be the union, and a `permission` denial about an *API key* would
    then read as a missing file and retry for ever.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("revoked")
    agent = ctx.agents.create(session, OPTIONS)
    agent.followup(await _attached(ctx))
    await agent.run()

    wire.forgotten.add("https://files.example/files/clip1")
    await agent.prompt("and now?")

    assert wire.uploaded == ["files/clip1", "files/clip2"]
    assert session.events[-1].type == "turn/end"
    (retry,) = [one for one in session.events if one.type == "llm/retry"]
    assert retry.data["code"] == FILE_EXPIRED


async def test_an_image_rides_inline_because_only_video_is_referenced(
    mount: Any, wire: _FileApi
) -> None:
    """`uploads` defaults to the video family, not to everything it accepts.

    An upload is a round trip, and a screenshot under the inline cap does not
    need one. What the default says is the narrower true thing: video cannot go
    inline, so video goes by reference.
    """
    ctx: Context = await mount(ROUTE, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("image"), OPTIONS)

    agent.followup(await _attached(ctx, mime="image/png", name="shot.png"))
    await agent.run()

    assert wire.uploaded == []
    assert "inlineData" in str(wire.bodies[-1])


# ------------------------------------------------------------ the mappings --


def test_thinking_tokens_are_added_into_the_output_they_sit_outside_of() -> None:
    """The mapping that under-reports if it is taken at face value.

    This provider counts `thoughtsTokenCount` *outside* `candidatesTokenCount`,
    where pH documents `reasoning_tokens` as a subset of `output_tokens` — `total`
    leaves it out for exactly that reason. Mapping across directly would make
    every thinking turn's output smaller than the bill.
    """
    usage = _to_usage(
        {
            "promptTokenCount": 1_000,
            "cachedContentTokenCount": 400,
            "candidatesTokenCount": 30,
            "thoughtsTokenCount": 70,
        }
    )

    assert usage.output_tokens == 100 and usage.reasoning_tokens == 70
    # Cached input is inside the prompt count here, so it is subtracted out — the
    # same correction DeepSeek needs, and for the same disjointness rule (D15).
    assert usage.input_tokens == 600 and usage.cache_read_tokens == 400
    assert usage.total == 1_100


def test_a_thought_and_an_answer_are_two_blocks() -> None:
    """A thought part is `thought: true` on a text part, not a separate kind.

    Folding it into visible text would make the transcript claim the model said
    what it was only considering — the same rule DeepSeek's `reasoning_content`
    and Anthropic's `thinking` blocks are mapped under.
    """
    state = _StreamState()
    chunks = [
        *state.consume(
            {"candidates": [{"content": {"parts": [{"text": "hmm", "thought": True}]}}]}
        ),
        *state.consume({"candidates": [{"content": {"parts": [{"text": "the answer"}]}}]}),
        *state.finish(),
    ]

    ends = [chunk for chunk in chunks if chunk.type == "block-end"]
    assert [end.block.type for end in ends] == ["reasoning", "text"]
    assert ends[0].block.text == "hmm" and ends[1].block.text == "the answer"


def test_a_function_call_is_given_the_id_this_wire_does_not_have() -> None:
    """pH pairs a result to its call by id; this wire addresses one by name.

    So an id is minted, and the finish kind is derived from what was streamed
    rather than from `finishReason` — which is `STOP` for a tool call here, and
    would have ended the turn with the model's request unanswered.
    """
    state = _StreamState()
    chunks = [
        *state.consume(
            {
                "candidates": [
                    {
                        "content": {
                            "parts": [{"functionCall": {"name": "read", "args": {"p": 1}}}]
                        },
                        "finishReason": "STOP",
                    }
                ]
            }
        ),
        *state.finish(),
    ]

    (end,) = [chunk for chunk in chunks if chunk.type == "block-end"]
    assert end.block.name == "read" and end.block.id == "call-1"
    assert end.block.arguments == '{"p": 1}'
    assert chunks[-1].reason.kind == "tool-calls"


def test_a_tool_result_goes_back_addressed_by_name() -> None:
    """The pairing this wire uses, rebuilt from the history it is given.

    A `functionResponse` carries the function's **name**; pH's `ToolResultBlock`
    carries the call's **id**, because that is what every other wire pairs on. The
    map is rebuilt per request from the assistant messages already in history
    rather than kept as adapter state — a resumed session's first request has no
    state to have kept, and a map right only for calls this process saw would
    silently mis-address every result after a restart.
    """
    called = create_assistant_message(
        content=[ToolCallBlock(id="call-1", name="read", arguments='{"path": "x"}')],
        provider="google",
        model="m",
    )
    answered = create_tool_result_message(
        call_id="call-1", content=[{"type": "text", "text": "the file"}], is_error=False
    )
    names = _call_names([called, answered])

    request = _to_google(called, {}, {}, names)
    reply = _to_google(answered, {}, {}, names)

    assert request == {
        "role": "model",
        "parts": [{"functionCall": {"name": "read", "args": {"path": "x"}}}],
    }
    assert reply == {
        "role": "user",
        "parts": [{"functionResponse": {"name": "read", "response": {"output": "the file"}}}],
    }


def test_a_thought_in_history_is_not_sent_back_as_speech() -> None:
    """There is no input shape for a thought on this wire.

    Rendering one as a text part would put the model's private reasoning into the
    conversation as something it said — the same rule that keeps `thinking` and
    `reasoning_content` out of visible text on the other two wires. The log still
    holds it; the request does not. A message that was *only* a thought produces
    no entry at all, because an entry with no parts is a request error.
    """
    thinking = create_assistant_message(
        content=[ReasoningBlock(text="hmm"), TextBlock(text="the answer")],
        provider="google",
        model="m",
    )
    only_thought = create_assistant_message(
        content=[ReasoningBlock(text="hmm")], provider="google", model="m"
    )

    assert _to_google(thinking, {}, {}, {}) == {"role": "model", "parts": [{"text": "the answer"}]}
    assert _to_google(only_thought, {}, {}, {}) is None
