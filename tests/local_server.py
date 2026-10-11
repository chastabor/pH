"""What every live local-server suite shares: the opt-in, the server, the model.

`test_local_server.py` asks whether the route is hooked up, `test_local_rlm.py`
whether Code Mode works over it, and `test_local_clm.py` whether a model edits its own
context. Both are opt-in by the same variable,
probe the same server and discover the same model, so those live here once rather
than drifting apart in two files — the reason `./test.sh smoke` reads its URL out of
`llama.yaml` rather than restating it.

@module local_server
"""

from __future__ import annotations

import os
from typing import Any

import httpx
import pytest

from ph.llm.types import text_of
from ph.session import Session, SessionEvent
from ph.testing import result_text

__all__ = [
    "BASE_URL",
    "LIVE",
    "PROBE_TIMEOUT",
    "answer",
    "get",
    "local_model",
    "props",
    "server_root",
]

BASE_URL = os.environ.get("LLAMA_BASE_URL", "")

LIVE = pytest.mark.skipif(
    not BASE_URL, reason="LLAMA_BASE_URL is not set; the live local-server gate is opt-in"
)
"""**Opt-in, and the opt-in is the base URL**: nothing marked with this runs unless
`LLAMA_BASE_URL` is set, so a developer with something unrelated listening on the
port never has a suite talk to it."""

PROBE_TIMEOUT = 30.0
"""For the metadata endpoints only. Generation gets the adapter's own timeout, which
is minutes — a 27B model on a laptop takes them."""


def get(url: str) -> Any:  # noqa: ANN401
    """One metadata read, or a failure that names the URL.

    A failure and not a skip: the person who set `LLAMA_BASE_URL` asked whether
    the server is reachable, and answering "skipped" to that question is how a
    hookup test comes to be green against nothing at all. The one endpoint whose
    absence is genuinely inconclusive is handled by its own reader below.
    """
    try:
        reply = httpx.get(url, timeout=PROBE_TIMEOUT)
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


def props() -> Any | None:  # noqa: ANN401
    """llama.cpp's `/props`, or `None` when this server does not publish it.

    The one inconclusive answer in the file: a 404 here means "not llama.cpp, or
    started with the endpoint off", which is a fact about the server rather than a
    defect. Anything else — a refused connection, a 500 — is still a failure,
    because `/models` just answered and something is wrong if this cannot.
    """
    url = f"{server_root()}/props"
    try:
        reply = httpx.get(url, timeout=PROBE_TIMEOUT)
        if reply.status_code == httpx.codes.NOT_FOUND:
            return None
        reply.raise_for_status()
        return reply.json()
    except httpx.HTTPError as error:
        pytest.fail(f"{url} did not answer: {error}")


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
    models = get(f"{BASE_URL.rstrip('/')}/models").get("data") or []
    if not models:
        pytest.fail(f"{BASE_URL} has no model loaded; /models listed none")
    return str(models[0]["id"])


def answer(session: Session) -> str:
    """What the person was shown — `print_mode`'s own extraction.

    The transcript rather than the model surface, and joined across messages
    because a model that thinks out loud lands more than one.
    """
    return "\n".join(
        text_of(message.content)
        for message in session.transcript()
        if message.role == "assistant" and text_of(message.content)
    )


def calls_after(session: Session, seq: int) -> list[SessionEvent]:
    """The model's own top-level calls logged after `seq` — a dispatch inside a cell is
    a `tool/code-dispatch`, not one of these."""
    return [event for event in session.select("tool/call") if event.seq > seq]


def output_of(session: Session, call: SessionEvent) -> str:
    """What one of those calls returned to the model, as text."""
    return result_text(session, str(call.data.get("callId")))
