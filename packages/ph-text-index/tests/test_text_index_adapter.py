"""`text-index-adapter` — embeddings through one of the session's own model routes.

Gates: *a text index embeds through a profile's OpenAI-compatible route; an
adapter without the capability is refused at the row with a sentence naming it;
the route's credential never reaches the command.*

## Where the key is

The tools run in a command inside the calling agent's sandbox, so anything that
command holds is in reach of the agent's own tools. The route's key is therefore
not handed in: the command sends its passages to the session over a socket, the
session's adapter embeds them, and vectors come back. The tests below start the
shipped command, as a session does, and look for the key in the two places it
could have leaked to — what the command was given, and what it could send.

## The provider

The fixture server of `text_index_helpers`, behind phern's real adapter rows: the
OpenAI-compatible one for the gates, and the Google one for its own request
shape. A key in a variable the harness resolves through `ctx.credentials`.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from text_index_helpers import (
    KEY_ENV,
    Endpoint,
    command_environment,  # noqa: F401 - autouse, by being here
    docs_tree,
    embeddings_server,  # noqa: F401 - the `endpoint` fixture, by being here
    long_document,
    new_agent,
)

from ph.cordis import MountRefusal
from ph.json import as_int, as_str
from ph.keys import LLM, SANDBOX, WORKSPACE
from ph.llm.adapter import EmbeddingAdapter, EmbeddingRequest, LlmError
from ph.llm.types import text_of
from ph.seams.token_meter import event_tokens
from ph.testing import MountProfile, report_section, run_tool, session_of
from ph.toolcli import CONFIG_ENV
from ph.workspace_caches import digest
from ph_text_index._config import EMBEDDER_ENV
from ph_text_index.cli import CLI
from ph_text_index.row import TEXT_INDEX
from ph_utcp import UTCP

pytestmark = pytest.mark.anyio


PROPS = {"path": "/props", "field": ["default_generation_settings", "n_ctx"]}
"""How a profile tells a route to ask a llama.cpp for its window."""


def _route(url: str, **profile: object) -> dict[str, Any]:
    """An OpenAI-compatible route at the fixture server, keyed by `KEY_ENV`."""
    route = {"provider": "fx", "baseUrl": url, "apiKeyEnv": KEY_ENV, **profile}
    return {
        "id": "llm-openai-compatible",
        "name": "llm-openai-compatible",
        "config": {"profiles": [route]},
    }


def _session(url: str, **config: object) -> tuple[dict[str, Any], ...]:
    """A session whose text index embeds through that route."""
    return (
        _route(url),
        {"id": "utcp", "name": "utcp"},
        {"id": "text-index", "name": "text-index"},
        {
            "id": "text-index-adapter",
            "name": "text-index-adapter",
            "config": {"provider": "fx", "model": "fx-embed", **config},
        },
    )


# ------------------------------------------------------------ the capability --


async def test_the_openai_compatible_adapter_embeds_and_says_what_it_cost(
    mount: MountProfile, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ctx.llm.embed`, through the route's own adapter: one vector an input, in
    the inputs' order though the wire returned them reversed, and the provider's
    count of what it read."""
    monkeypatch.setenv(KEY_ENV, "sekrit")
    ctx = await mount(_route(endpoint.url))
    llm = ctx.require(LLM)

    result = await llm.embed(
        EmbeddingRequest(provider="fx", model="fx-embed", inputs=("alpha beta", "gamma"))
    )
    alone = await llm.embed(EmbeddingRequest(provider="fx", model="fx-embed", inputs=("gamma",)))

    assert llm.embeds("fx") and isinstance(llm.adapter_for("fx"), EmbeddingAdapter)
    assert len(result.vectors) == 2 and len(result.vectors[0]) == 96
    assert result.vectors[1] == alone.vectors[0], "the vectors came back out of order"
    assert result.input_tokens == 3
    assert endpoint.asked[0] == {
        "path": "/v1/embeddings",
        "model": "fx-embed",
        "input": ["alpha beta", "gamma"],
    }
    assert set(endpoint.keys) == {"Bearer sekrit"}


async def test_the_google_adapter_embeds_in_its_own_shape(
    mount: MountProfile, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`models/<model>:batchEmbedContents`: a `content` of one text part an input,
    the width asked as `outputDimensionality`, and the key in this API's header."""
    monkeypatch.setenv(KEY_ENV, "sekrit")
    ctx = await mount(
        {
            "id": "llm-google",
            "name": "llm-google",
            "config": {"provider": "gx", "baseUrl": endpoint.url, "apiKeyEnv": KEY_ENV},
        }
    )

    result = await ctx.require(LLM).embed(
        EmbeddingRequest(provider="gx", model="gx-embed", inputs=("alpha", "beta"), dimensions=64)
    )

    assert len(result.vectors) == 2 and result.input_tokens is None
    [asked] = endpoint.asked
    assert asked["path"] == "/v1/models/gx-embed:batchEmbedContents"
    assert asked["requests"] == [
        {
            "model": "models/gx-embed",
            "outputDimensionality": 64,
            "content": {"parts": [{"text": text}]},
        }
        for text in ("alpha", "beta")
    ]
    assert endpoint.keys == ["sekrit"]


async def test_a_route_that_cannot_embed_says_so_by_name(mount: MountProfile) -> None:
    """An optional capability, asked of the Protocol: the suite's fake adapter
    streams and does not embed, and a route nobody registered has no adapter."""
    llm = (await mount()).require(LLM)

    assert not llm.embeds("fake") and not llm.embeds("nobody")
    with pytest.raises(LlmError, match='provider "fake" does not embed') as refused:
        await llm.embed(EmbeddingRequest(provider="fake", model="m", inputs=("x",)))
    assert refused.value.code == "NO_EMBEDDINGS"


# --------------------------------------------------------------- the session --


async def test_a_text_index_embeds_through_a_route_and_the_key_stays_in_the_harness(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** The shipped command indexes and searches, and every vector
    came from the session's route: the provider saw the key, and the command was
    given none — not by value, not by name — only the socket to ask on."""
    docs_tree(tmp_path, documents=3)
    monkeypatch.setenv(KEY_ENV, "sekrit")
    ctx = await mount(
        *_session(endpoint.url, documentPrefix="search_document: ", queryPrefix="search_query: ")
    )
    agent = new_agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    found = await run_tool(
        ctx, "text_search", {"query": "which sandbox backend confines", "k": 1}, agent=agent
    )

    assert not indexed.is_error, text_of(indexed.content)
    assert [hit["path"] for hit in found.value["hits"]] == ["docs/w.md"]
    assert set(endpoint.keys) == {"Bearer sekrit"}
    assert endpoint.asked[0]["input"][0].startswith("search_document: ")
    assert endpoint.asked[-1]["input"] == ["search_query: which sandbox backend confines"]

    seam = ctx.require(TEXT_INDEX)
    assert seam.provider is not None and seam.manual is not None
    given = seam.manual.model_dump()
    assert seam.manual.pass_env == [], "the command was handed a variable"
    assert "sekrit" not in json.dumps(given) and KEY_ENV not in json.dumps(given)
    assert set(seam.manual.env) == {CONFIG_ENV, EMBEDDER_ENV}
    rows = report_section(ctx, "Text index")
    assert rows["embedder"] == "fx-embed through fx"
    assert rows["embedded"].startswith("4 texts in ") and "tokens" in rows["embedded"]
    # And what the route spent is in the log of the agent it was spent for,
    # where a budget reads every other spend.
    spent = [one for one in session_of(agent).events if one.type == "llm/embedded"]
    assert sum(as_int(one.data["texts"]) for one in spent) == 4
    assert {(one.data["provider"], one.data["model"]) for one in spent} == {("fx", "fx-embed")}
    assert sum(event_tokens(one) for one in spent) > 0
    [hosted] = ctx.require(UTCP).hosts.values()
    assert hosted.home is not None
    assert {one.name for one in hosted.home.iterdir() if one.is_dir()} == {
        digest("adapter:fx/fx-embed:search_query: |search_document: ")
    }


async def test_a_key_that_is_not_set_is_said_by_the_tool_as_a_model_call_would_say_it(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The adapter's own sentence and code, across the socket: the variable is
    named to the agent and to `phern doctor`, and nothing is sent anywhere."""
    docs_tree(tmp_path)
    monkeypatch.delenv(KEY_ENV, raising=False)
    ctx = await mount(*_session(endpoint.url))
    agent = new_agent(ctx)

    refused = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert refused.is_error
    assert refused.error is not None and refused.error.info is not None
    assert refused.error.info["code"] == "MISSING_CREDENTIAL"
    assert f"{KEY_ENV} is not set" in refused.error.message
    assert "MISSING_CREDENTIAL" in report_section(ctx, "Text index")["last failure"]
    assert endpoint.asked == []
    # The reason is in the agent's log as well: the request that was not made.
    failed = [one for one in session_of(agent).events if one.type == "llm/embedding-failed"]
    assert failed and {one.data["code"] for one in failed} == {"MISSING_CREDENTIAL"}
    assert all(f"{KEY_ENV} is not set" in as_str(one.data["reason"]) for one in failed)


async def test_a_route_whose_adapter_cannot_embed_is_refused_at_the_row(
    mount: MountProfile,
) -> None:
    """**The gate.** At mount, by name, to whoever wrote the profile — and not to
    an agent, on the first call, as a tool that fails."""
    with pytest.raises(MountRefusal, match='names provider "fake", whose adapter does not embed'):
        await mount(
            {"id": "utcp", "name": "utcp"},
            {"id": "text-index", "name": "text-index"},
            {
                "id": "text-index-adapter",
                "name": "text-index-adapter",
                "config": {"provider": "fake", "model": "m"},
            },
        )


@pytest.mark.needs_git
async def test_a_confined_agent_embeds_with_no_network_of_its_own(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command is in a sandbox whose deployment allows it no host at all. It
    asks the session, on a socket both were granted, and the session — which is
    not confined — asks the provider. So an agent can embed through a route
    without being given a way to reach it, or the key to it."""
    monkeypatch.setenv(KEY_ENV, "sekrit")
    ctx = await mount({"id": "sandbox-local", "disabled": False}, *_session(endpoint.url))
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    repo = docs_tree(tmp_path / "repo", documents=3)
    agent = new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=repo
    )

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    found = await run_tool(ctx, "text_search", {"query": "containment tier", "k": 1}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["total_documents"] == 3
    assert [hit["path"] for hit in found.value["hits"]] == ["docs/w.md"]
    assert set(endpoint.keys) == {"Bearer sekrit"}


def test_from_a_shell_the_command_says_there_is_no_session_to_ask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """This embedder is a route of a running session. A command a person types
    has none, and says which rows it can use instead."""
    monkeypatch.setenv(
        CONFIG_ENV, json.dumps({"text-index-adapter": {"provider": "fx", "model": "m"}})
    )
    monkeypatch.delenv(EMBEDDER_ENV, raising=False)

    done = subprocess.run(
        [*CLI.invocation(), "call", "text_search", "--query", "x"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    answer = json.loads(done.stdout)
    assert answer["error"]["code"] == "MOUNT_REFUSED"
    assert f"${EMBEDDER_ENV} is not set" in answer["error"]["message"]
    assert "text-index-endpoint" in answer["error"]["message"]


# ----------------------------------------------------- how long an input is --


async def test_passages_are_cut_to_the_routes_window_and_a_setting_wins(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command asks the session what its route reads, as it asks it for
    vectors: the route's window, which for a llama.cpp is the embedding model's.
    A hosted provider's window is its chat model's, so a profile writes
    `max_tokens` there — and what it writes is used without asking anybody."""
    long_document(tmp_path)
    monkeypatch.setenv(KEY_ENV, "sekrit")
    endpoint.window, endpoint.reads = 60, 60
    # The route learns its window the way a profile tells it to: by asking.
    asking = _route(endpoint.url, contextWindowProbes=[PROPS])
    ctx = await mount(asking, *_session(endpoint.url)[1:])
    by_window = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))
    # Which a session can say for itself, to `phern doctor`, with no command.
    assert report_section(ctx, "Text index")["limit"] == "60 tokens (the route's window)"
    await ctx.drain()
    await ctx.dispose()

    set_ = await mount(*_session(endpoint.url, maxTokens=30))
    by_setting = await run_tool(set_, "text_index", {"paths": ["docs"]}, agent=new_agent(set_))

    assert not by_window.is_error, text_of(by_window.content)
    assert by_window.value["limit"] == "60 tokens (the route's window)"
    # Two tokens of each limit are the markers every passage carries.
    assert by_window.value["passage_tokens"] == 58
    assert by_setting.value["limit"] == "30 tokens (configured)"
    assert report_section(set_, "Text index")["limit"] == "30 tokens (configured)"
    assert by_setting.value["passage_tokens"] == 28
    assert max(endpoint.took) <= 60


async def test_a_provider_that_refuses_a_passage_is_sent_a_smaller_one(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** The refusal crosses two processes and keeps its meaning: the
    adapter classifies the server's sentence as an input that did not fit, the
    session hands that code down the socket, and the command packs the document
    smaller and asks again."""
    long_document(tmp_path)
    monkeypatch.setenv(KEY_ENV, "sekrit")
    endpoint.reads = 40
    ctx = await mount(*_session(endpoint.url))

    with pytest.raises(LlmError) as refused:
        await ctx.require(LLM).embed(
            EmbeddingRequest(provider="fx", model="m", inputs=(" ".join(["w"] * 41),))
        )
    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))

    assert refused.value.code == "CONTEXT_WINDOW_EXCEEDED"
    assert not indexed.is_error, text_of(indexed.content)
    assert endpoint.took and max(endpoint.took) <= 40
    assert indexed.value["cut_smaller"] == ["docs/long.md"]
