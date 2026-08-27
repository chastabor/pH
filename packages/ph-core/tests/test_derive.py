"""P0-10 — `derive_messages()` and the incremental folds.

Gate: *the cache invalidates only on generation change; the folds are
incremental.*

`derive_messages()` is the only path to model context (A2). Anything else that
built a message list would be a second answer to "what did the model see", and
the invariant in P0-14 exists to make sure there isn't one.
"""

from __future__ import annotations

from ph.llm.types import LlmCallConfig, ToolSchema
from ph.session import Session, SurfaceIntent, SurfaceReplace
from ph.session.request_header import (
    EpochHeader,
    canonical_header,
    fold_request_header,
    header_equals,
)
from ph.testing import assistant_payload, user_payload


def test_derivation_follows_the_surface() -> None:
    session = Session("s")
    session.append("turn/start", {"turn": 1})
    session.append("user/message", user_payload("hi", "m1"), SurfaceIntent("append"))
    session.append("assistant/chunk", {"turn": 1, "step": 1, "chunk": {"type": "usage"}})
    session.append(
        "assistant/message", assistant_payload("hey", "m2"), SurfaceIntent("append", (2,))
    )
    messages = session.derive_messages()
    # Boundaries and raw chunks are trace data, so they are correctly absent.
    assert [(m.role, m.content[0].text) for m in messages] == [
        ("user", "hi"),
        ("assistant", "hey"),
    ]


def test_empty_assistant_message_derives_to_nothing() -> None:
    session = Session("s")
    session.append("assistant/message", assistant_payload("", "m1"), SurfaceIntent("append", ()))
    # It exists only to host a max-tokens step's usage; a content-less
    # assistant turn is rejected by several providers.
    assert session.derive_messages() == ()
    assert session.surface.nodes == (0,)


def test_a_holder_never_sees_later_appends() -> None:
    session = Session("s")
    session.append("user/message", user_payload("a", "m1"), SurfaceIntent("append"))
    held = session.derive_messages()
    session.append("user/message", user_payload("b", "m2"), SurfaceIntent("append"))
    assert len(held) == 1
    assert len(session.derive_messages()) == 2


def test_cache_rebuilds_only_on_a_surface_rewrite() -> None:
    session = Session("s")
    session.append("user/message", user_payload("a", "m1"), SurfaceIntent("append"))
    session.append("assistant/message", assistant_payload("b", "m2"), SurfaceIntent("append", ()))
    first = session.derive_messages()
    second = session.derive_messages()
    # Each node is projected once: identity proves the cache was reused — and
    # the whole tuple is the same object when nothing was appended.
    assert first is second

    session.append(
        "user/message",
        user_payload("summary", "m3"),
        SurfaceIntent(SurfaceReplace(start=0, end=1), (0, 1)),
    )
    rebuilt = session.derive_messages()
    assert [m.content[0].text for m in rebuilt] == ["summary"]
    assert session.surface.replace_generation == 1


def test_transcript_keeps_what_the_surface_shadows() -> None:
    session = Session("s")
    session.append("user/message", user_payload("a", "m1"), SurfaceIntent("append"))
    session.append("assistant/message", assistant_payload("b", "m2"), SurfaceIntent("append", ()))
    session.append(
        "user/message",
        user_payload("summary", "m3"),
        SurfaceIntent(SurfaceReplace(start=0, end=1), (0, 1)),
    )
    # The model sees the summary; the human still sees the conversation.
    assert [m.content[0].text for m in session.derive_messages()] == ["summary"]
    assert [m.content[0].text for m in session.transcript()] == ["a", "b"]


def test_request_header_folds_incrementally_and_matches_the_pure_fold() -> None:
    session = Session("s")
    assert session.request_header() is None

    first = canonical_header(
        EpochHeader(config=LlmCallConfig(provider="fake", model="m1"), system="be brief")
    )
    session.append("request/header", {"header": first.to_wire(), "reason": "initial"})
    assert session.request_header() == first

    second = canonical_header(
        EpochHeader(
            config=LlmCallConfig(provider="fake", model="m2"),
            tools=[ToolSchema(name="read", description="d", parameters={})],
        )
    )
    session.append("request/header", {"header": second.to_wire(), "reason": "change"})
    live = session.request_header()
    assert live == second
    # The live incremental fold and the offline one must agree.
    assert live == fold_request_header(session.events)


def test_canonical_header_drops_empty_system_and_tools() -> None:
    header = canonical_header(
        EpochHeader(config=LlmCallConfig(provider="p", model="m"), system="", tools=[])
    )
    assert header.system is None
    assert header.tools is None
    # Otherwise `[]` and `None` would compare unequal and append a header on
    # every step, which is exactly what breaks prefix caching.
    assert header_equals(header, EpochHeader(config=LlmCallConfig(provider="p", model="m")))


def test_header_equality_compares_tool_schemas_in_order() -> None:
    config = LlmCallConfig(provider="p", model="m")
    a = ToolSchema(name="read", description="d", parameters={})
    b = ToolSchema(name="write", description="d", parameters={})
    assert not header_equals(
        EpochHeader(config=config, tools=[a, b]), EpochHeader(config=config, tools=[b, a])
    )


def test_request_context_folds() -> None:
    session = Session("s")
    assert session.request_context() is None
    session.append("request/context", {"provider": "fake", "model": "m", "contextWindow": 100})
    context = session.request_context()
    assert context is not None
    assert context.context_window == 100
