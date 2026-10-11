"""What every live local-server suite shares: the opt-in, the server, the model.

`test_local_server.py` asks whether the route is hooked up, `test_local_rlm.py`
whether Code Mode works over it, and `test_local_clm.py` whether a model edits its own
context. All three are opt-in by the same variable, probe the same server, discover the
same model and mount their profiles the same way, so those live here once rather than
drifting apart in three files — the reason `./test.sh smoke` reads its URL out of
`llama.yaml` rather than restating it. The `route` fixture is here too, registered by
the root conftest's `pytest_plugins` as the per-package fixture sets are.

@module local_server
"""

from __future__ import annotations

import os
from collections.abc import Collection
from typing import Any

import httpx
import pytest

from ph.agent.types import AgentOptions
from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.keys import AGENTS, SESSIONS
from ph.llm.types import text_of
from ph.session import Session, SessionEvent
from ph.testing import MountProfile, result_text
from ph_app.profiles import PROFILE_DIR, resolve_profile

__all__ = [
    "LIVE",
    "TEMPERATURE",
    "answer_of",
    "cells_after",
    "get_json",
    "live_agent",
    "local_model",
    "output_of",
    "route",
    "server_props",
    "server_root",
]

BASE_URL = os.environ.get("LLAMA_BASE_URL", "")

LIVE = pytest.mark.skipif(
    not BASE_URL, reason="LLAMA_BASE_URL is not set; the live local-server gate is opt-in"
)
"""**Opt-in, and the opt-in is the base URL**: nothing marked with this runs unless
`LLAMA_BASE_URL` is set, so a developer with something unrelated listening on the
port never has a suite talk to it."""

TEMPERATURE = 0.6
"""Pinned on every live request, so the gate does not depend on how the server was
started — and a child a live agent delegates to inherits it with the rest of its
parent's options.

Measured on `test_local_server.py`'s requests, each replayed 25 times: at the
server's earlier 1.0 there was a runaway tail — one "reply with ok" thought for 7,446
tokens, about sixteen minutes at a 27B model's speed, and then called a tool nobody
asked for. At 0.6, the setting Qwen's model cards give for thinking, the longest of
175 replies was 462 tokens and none called a stray tool."""

PROBE_TIMEOUT = 30.0
"""For the metadata endpoints only. Generation gets the adapter's own timeout, which
is minutes — a 27B model on a laptop takes them."""


def get_json(url: str, *, absent: Collection[int] = ()) -> Any:  # noqa: ANN401
    """One metadata read — `None` for a status in `absent` — or a failure that names
    the URL.

    A failure and not a skip: the person who set `LLAMA_BASE_URL` asked whether the
    server is reachable, and answering "skipped" to that question is how a hookup test
    comes to be green against nothing at all. `absent` is for the endpoints whose
    absence is genuinely inconclusive — a fact about the server, not a defect.
    """
    try:
        reply = httpx.get(url, timeout=PROBE_TIMEOUT)
        if reply.status_code in absent:
            return None
        reply.raise_for_status()
        return reply.json()
    except httpx.HTTPError as error:
        pytest.fail(f"{url} did not answer: {error}")


def server_root() -> str:
    """The server's own root, which is *not* the OpenAI base URL.

    llama.cpp serves the compatible wire under `/v1` and its own endpoints —
    `/props`, `/slots`, `/health` — beside it rather than under it, so one
    variable cannot address both without this. Derived rather than asked for as a
    second variable: two URLs for one server is how a gate comes to probe a
    different process than it prompts.
    """
    base = BASE_URL.rstrip("/")
    return base.removesuffix("/v1").rstrip("/")


def server_props() -> Any | None:  # noqa: ANN401
    """llama.cpp's `/props`, or `None` when this server does not publish it: a 404
    means "not llama.cpp, or started with the endpoint off"."""
    return get_json(f"{server_root()}/props", absent=(httpx.codes.NOT_FOUND,))


def local_model(monkeypatch: pytest.MonkeyPatch) -> str:
    """The profile's credential, defaulted — and the model to ask for.

    `LLAMA_API_KEY` is a formality llama.cpp ignores unless it was started with
    `--api-key`, but the route resolves it at the request edge like any other, so
    unset means `MISSING_CREDENTIAL` rather than an anonymous request. Defaulted
    here rather than required of the person, who already said which server.

    The model is *discovered* rather than configured, because `--model` names a
    `.gguf` path on one host and an alias on the next. `LLAMA_MODEL` overrides for a
    server holding more than one.
    """
    if not os.environ.get("LLAMA_API_KEY"):
        monkeypatch.setenv("LLAMA_API_KEY", "local")
    configured = os.environ.get("LLAMA_MODEL")
    if configured:
        return configured
    models = get_json(f"{BASE_URL.rstrip('/')}/models").get("data") or []
    if not models:
        pytest.fail(f"{BASE_URL} has no model loaded; /models listed none")
    return str(models[0]["id"])


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> str:
    """The model the live gate asks for, with the route's credential defaulted
    (`local_model`). Only the opt-in live suites take it."""
    return local_model(monkeypatch)


async def live_agent(
    mount: MountProfile, route: str, profile: str, name: str
) -> tuple[Context, ReactLoopAgent]:
    """A shipped `profile` over the shipped llama route, and an agent on it.

    Resolved here, inside the test, and never at import: the suite's own home is
    pinned per test, and a module-level resolution reads the person's real
    `$PH_HOME/profiles/` instead — where a file named after a shipped profile is
    theirs, not what the gate is about.
    """
    ctx = await mount(profile=[*resolve_profile(profile), PROFILE_DIR / "llama.yaml"])
    session = ctx.require(SESSIONS).create(name)
    agent = ctx.require(AGENTS).create(
        session,
        AgentOptions(provider="llama", model=route, max_tokens=2048, temperature=TEMPERATURE),
    )
    assert isinstance(agent, ReactLoopAgent)
    return ctx, agent


def answer_of(session: Session) -> str:
    """What the person was shown — `print_mode`'s own extraction.

    The transcript rather than the model surface, and joined across messages
    because a model that thinks out loud lands more than one.
    """
    return "\n".join(
        text_of(message.content)
        for message in session.transcript()
        if message.role == "assistant" and text_of(message.content)
    )


def cells_after(session: Session, seq: int) -> list[SessionEvent]:
    """The model's own top-level calls logged after `seq` — at least one, since every
    live prompt names the cell to run. A dispatch inside a cell is a
    `tool/code-dispatch`, not one of these."""
    calls = [event for event in session.select("tool/call") if event.seq > seq]
    assert calls, "the model was told to run a cell by name and ran none"
    return calls


def output_of(session: Session, call: SessionEvent) -> str:
    """What one of those calls returned to the model, as text."""
    return result_text(session, str(call.data.get("callId")))
