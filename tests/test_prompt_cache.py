"""P6-13's gate — the *number*, not the shape (A12, P4-03).

`test_adapters.py` pins where the `cache_control` markers land. That is a shape
assertion, and a shape assertion cannot fail for the defect this row exists to
fix: markers placed at indices no later request ever marks again are written
once, read never, and look perfectly correct in a body. So the gate is a count —
`cacheReadTokens > 0` on the *second* request of a session, and on the
compaction call — read from the log, which is where `driver.py` and
`compaction.py` already put it.

**The provider is simulated, and that is the honest description of it.** There is
no Anthropic key in CI, so `_PromptCache` below implements Anthropic's
*documented* prefix caching — longest marked prefix wins, a write needs 1024
tokens, `cache_creation` is the span between the read point and the last marker —
and every request that reaches it is the real `AnthropicAdapter._body`. What this
licenses is a claim about pH's marker placement under those rules. It is not a
measurement of Anthropic, and a change in what Anthropic actually does would not
fail it.

Everything else is real: the base profile, the stabilize bundle, the agent loop,
`/compact`, and the session log the assertions read.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import AsyncIterator
from typing import Any

import pytest

from ph.agent.types import AgentOptions
from ph.bundles import BASE, HEADLESS
from ph.session import Session
from ph.testing import anthropic_reply
from ph_app.adapters._http import HttpClient
from ph_app.adapters.anthropic import MIN_CACHEABLE_TOKENS
from ph_stabilize import BUNDLE

pytestmark = pytest.mark.anyio

PROFILE = [BASE, HEADLESS, BUNDLE]
ROUTE = {
    "insert": [
        {
            "id": "llm-anthropic",
            "name": "llm-anthropic",
            # A window a test session can actually fill: compaction keeps the
            # last tenth of it, so at the route default of 200k nothing here
            # would ever be old enough to summarize.
            "config": {
                "provider": "anthropic",
                "apiKeyEnv": "ANTHROPIC_API_KEY",
                "contextWindow": 8192,
            },
        }
    ]
}
OPTIONS = AgentOptions(provider="anthropic", model="claude-test")


def _prompt(turn: int) -> str:
    """A question big enough that the prefix clears Anthropic's write floor.

    `headless` mounts six tools and **no system prompt** — about 900 tokens of
    prefix before a word is said — so a one-line question leaves the marked
    prefix under `MIN_CACHEABLE_TOKENS` and the provider stores nothing. That is
    a property of this profile rather than of the placement (`rlm-stable` opens
    with a far larger system prompt), but a gate that quietly measured the floor
    instead of the markers would be worth nothing.
    """
    return f"question {turn} " + "detail " * 200


def _strip(value: Any) -> Any:
    """The same value without its markers.

    A cache key is the *content* of a prefix; Anthropic does not make a message
    a different message for having been marked. Hashing the marker too would let
    this file pass for a placement that only ever agrees with itself.
    """
    if isinstance(value, dict):
        return {key: _strip(item) for key, item in value.items() if key != "cache_control"}
    if isinstance(value, list):
        return [_strip(item) for item in value]
    return value


def _units(body: dict[str, Any]) -> list[tuple[str, bool]]:
    """Every cacheable unit in wire order, and whether it carries a marker.

    Order is the wire's, not the body dict's: Anthropic's prefix runs tools →
    system → messages, which is exactly why the adapter spends a marker on the
    tool list rather than letting `system` cover it.
    """
    units: list[tuple[str, bool]] = []
    for tool in body.get("tools") or []:
        units.append((json.dumps(_strip(tool), sort_keys=True), "cache_control" in tool))
    system = body.get("system")
    if isinstance(system, str):
        units.append((system, False))
    else:
        for block in system or []:
            units.append((json.dumps(_strip(block), sort_keys=True), "cache_control" in block))
    for message in body.get("messages") or []:
        marked = any("cache_control" in block for block in message["content"])
        units.append((json.dumps(_strip(message), sort_keys=True), marked))
    return units


class _PromptCache:
    """Anthropic's documented prefix caching, as a wire.

    Holds only digests, so a test can assert on `reads` and `writes` without the
    fixture becoming a second copy of the adapter's arithmetic.
    """

    def __init__(self) -> None:
        self.stored: set[str] = set()
        self.reads: list[int] = []
        self.writes: list[int] = []
        self.reply = "answer"

    def usage(self, body: dict[str, Any]) -> dict[str, int]:
        prefix, total = "", 0
        marked: list[tuple[str, int]] = []
        for text, is_breakpoint in _units(body):
            prefix += text
            total += len(text) // 4
            if is_breakpoint:
                marked.append((hashlib.sha256(prefix.encode()).hexdigest(), total))
        # Longest marked prefix that is already stored — the list is in wire
        # order, so the last match is the longest.
        read = next((tokens for digest, tokens in reversed(marked) if digest in self.stored), 0)
        written = 0
        for digest, tokens in marked:
            if digest not in self.stored and tokens >= MIN_CACHEABLE_TOKENS:
                self.stored.add(digest)
                written = tokens
        created = max(0, written - read)
        self.reads.append(read)
        self.writes.append(created)
        return {
            "input_tokens": total - read - created,
            "output_tokens": 0,
            "cache_read_input_tokens": read,
            "cache_creation_input_tokens": created,
        }

    def events(self, body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        return anthropic_reply(self.reply, usage=self.usage(body))


@pytest.fixture
def wire(monkeypatch: pytest.MonkeyPatch) -> _PromptCache:
    """The simulated provider, patched onto the class the adapter builds itself.

    On `HttpClient` rather than on the adapter instance, because the row
    constructs its own client — reaching past the row to swap one out would be a
    test of a wiring this profile does not use.
    """
    cache = _PromptCache()

    async def stream_sse(
        self: HttpClient, url: str, **kwargs: Any
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        for event in cache.events(kwargs["json"]):
            yield event

    monkeypatch.setattr(HttpClient, "stream_sse", stream_sse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    return cache


def _usage(session: Session, event_type: str) -> list[dict[str, Any]]:
    return [
        dict(event.data["usage"])
        for event in session.events
        if event.type == event_type and event.data.get("usage")
    ]


async def test_the_second_request_of_a_session_reads_cache(mount: Any, wire: _PromptCache) -> None:
    """The gate, and the defect it is here for.

    Anthropic's caching is opt-in: before P6-13 this adapter sent no markers, so
    every prefix-stability decision in the harness — A12's byte-identical system
    prompt, its append-only message list — paid off on the routes that cache
    implicitly and nowhere here. The first request writes; the second has to
    *read*, and the log is where that is answerable.
    """
    ctx = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("cache")
    agent = ctx.agents.create(session, OPTIONS)

    await agent.prompt(_prompt(0))
    await agent.prompt(_prompt(1))

    first, second = _usage(session, "assistant/message")[:2]
    # `to_wire` omits a `None`, so "did not happen" is an absent key rather than a
    # zero — which is also how a reader of the log meets it.
    assert "cacheReadTokens" not in first, "nothing was cached before the first request"
    assert first["cacheWriteTokens"] > 0, "the first request cached nothing to read"
    assert second["cacheReadTokens"] > 0, "the second request re-read no prefix"


async def test_every_later_request_keeps_reading(mount: Any, wire: _PromptCache) -> None:
    """The property a marker on "the newest message" would fail.

    A conversation grows past several checkpoint boundaries here, so this covers
    the request that *advances* the checkpoint — the one a single quantized
    marker leaves with nothing to read. Asserted over every request after the
    first rather than at a chosen depth, because the failure is periodic and a
    spot check lands between the periods.
    """
    ctx = await mount(ROUTE, profile=PROFILE)
    agent = ctx.agents.create(ctx.sessions.create("deep"), OPTIONS)

    for turn in range(8):
        await agent.prompt(_prompt(turn))

    assert len(wire.reads) == 8
    assert all(read > 0 for read in wire.reads[1:]), wire.reads
    # And the tail is what gets written, not the conversation: after the first
    # request every write is a fraction of what a full re-cache would cost.
    assert max(wire.writes[1:]) < wire.reads[-1]


async def test_the_compaction_call_reads_the_conversations_own_prefix(
    mount: Any, wire: _PromptCache
) -> None:
    """P4-03's replayed envelope, priced — the second half of the gate.

    `_replay` reuses the session's own `system` and `tools` so the summarize
    request is a strict *prefix* of the conversation's, which is the entire
    reason that shape exists. `compaction/summarized` already carries the usage
    beside the shape it used; this asserts the two agree.
    """
    ctx = await mount(ROUTE, profile=PROFILE)
    session = ctx.sessions.create("compact")
    agent = ctx.agents.create(session, OPTIONS)
    for turn in range(6):
        await agent.prompt(_prompt(turn))

    wire.reply = "## SESSION INTENT\n\nthe scripted summary"
    await ctx.commands.dispatch("/compact", session=session, agent=agent)

    (record,) = [event for event in session.events if event.type == "compaction/summarized"]
    assert record.data["shape"] == "replay", "a direct shape cannot hit the conversation's cache"
    assert record.data["usage"]["cacheReadTokens"] > 0
