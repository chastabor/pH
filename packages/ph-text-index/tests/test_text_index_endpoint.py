"""`text-index-endpoint` — an embedder that is somebody else's process.

Gates: *against a fixture HTTP server, indexing and search work with no model on
this machine; an unset key variable and a host that cannot be reached each
produce a named failure and no traceback.*

## The server

A thread serving the OpenAI-compatible `/v1/embeddings` request with the suite's
hashing embedder behind it, so the vectors mean something and retrieval can be
asserted. It doubles every vector on the way out: not every server normalizes,
and the index scores by inner product. It keeps what it was asked, which is how
a test sees the key a command was handed without the key ever being printed.

## Two levels

The engine's row mounted in this process proves the embedder. The session's row
proves the join: that naming the endpoint in a profile is enough for the
package's own command to mount that embedder, and that the one variable the row
names reaches a command whose environment drops everything that looks like a
credential.
"""

from __future__ import annotations

import importlib.util
import json
import sys
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

from ph.keys import SANDBOX, WORKSPACE
from ph.llm.types import text_of
from ph.testing import MountProfile, report_section, run_tool, session_of
from ph.workspace_caches import digest
from ph_text_index._embed import EndpointEmbedder
from ph_text_index.row import TEXT_INDEX
from ph_utcp import UTCP

pytestmark = pytest.mark.anyio


def _engine(url: str, **config: object) -> tuple[dict[str, Any], dict[str, Any]]:
    """The engine's rows, as the command mounts them for an endpoint."""
    return (
        {"id": "text-index", "name": "text-index-engine"},
        {
            "id": "text-index-endpoint",
            "name": "text-index-endpoint-engine",
            "config": {"baseUrl": url, **config},
        },
    )


def _session(url: str, **config: object) -> tuple[dict[str, Any], ...]:
    """The session's rows: `ph-utcp`, the text index, and the endpoint named."""
    return (
        {"id": "utcp", "name": "utcp"},
        {"id": "text-index", "name": "text-index"},
        {
            "id": "text-index-endpoint",
            "name": "text-index-endpoint",
            "config": {"baseUrl": url, **config},
        },
    )


# ------------------------------------------------------------ the embedder --


async def test_an_endpoint_embeds_and_the_index_answers(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** Indexing and search with no model on this machine: the
    documents go to the endpoint a batch at a time with the passage prefix, the
    question goes with its own, the key is the named variable's, and vectors a
    server did not normalize are normalized before they are scored."""
    docs_tree(tmp_path, documents=3)
    monkeypatch.setenv(KEY_ENV, "sekrit")
    ctx = await mount(
        *_engine(
            endpoint.url,
            model="fx-embed",
            apiKeyEnv=KEY_ENV,
            documentPrefix="search_document: ",
            queryPrefix="search_query: ",
            batchSize=2,
        )
    )
    agent = new_agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    found = await run_tool(
        ctx, "text_search", {"query": "which sandbox backend confines", "k": 1}, agent=agent
    )

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["total_documents"] == 3
    assert [hit["path"] for hit in found.value["hits"]] == ["docs/w.md"]
    assert 0.0 < found.value["hits"][0]["score"] <= 1.01, "the vectors were not normalized"
    assert all(
        one["path"] == "/v1/embeddings" and one["model"] == "fx-embed" for one in endpoint.asked
    )
    assert all(len(one["input"]) <= 2 for one in endpoint.asked)
    assert endpoint.asked[0]["input"][0].startswith("search_document: ")
    assert endpoint.asked[-1]["input"] == ["search_query: which sandbox backend confines"]
    assert set(endpoint.keys) == {"Bearer sekrit"}
    assert "sentence_transformers" not in sys.modules, "an endpoint loaded the local model"


async def test_a_key_that_is_not_set_is_said_by_name_and_nothing_is_sent(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** A named failure and no traceback — and no request, so a
    document's text is never sent to an endpoint that would have refused it."""
    docs_tree(tmp_path)
    monkeypatch.delenv(KEY_ENV, raising=False)
    ctx = await mount(*_engine(endpoint.url, apiKeyEnv=KEY_ENV))

    refused = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))

    assert refused.is_error
    assert refused.error is not None and refused.error.info is not None
    assert refused.error.info["code"] == "MISSING_CREDENTIAL"
    assert f"${KEY_ENV} is not set" in refused.error.message
    assert endpoint.asked == []


async def test_an_endpoint_that_cannot_be_reached_or_will_not_answer_says_which(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """**The gate.** Two failures a person fixes in two places, so two codes: the
    address reaches nothing, and the server answered with something that is not
    embeddings — whose own words are kept."""
    docs_tree(tmp_path)
    nowhere = await mount(*_engine("http://127.0.0.1:9/v1", timeoutS=5))
    unreachable = await run_tool(nowhere, "text_search", {"query": "x"}, agent=new_agent(nowhere))
    await nowhere.drain()
    await nowhere.dispose()

    endpoint.status = 503
    ctx = await mount(*_engine(endpoint.url))
    failing = await run_tool(ctx, "text_search", {"query": "x"}, agent=new_agent(ctx))

    assert unreachable.error is not None and unreachable.error.info is not None
    assert unreachable.error.info["code"] == "TEXT_INDEX_ENDPOINT_UNREACHABLE"
    assert "sandbox-allow" in unreachable.error.message
    assert failing.error is not None and failing.error.info is not None
    assert failing.error.info["code"] == "TEXT_INDEX_ENDPOINT_FAILED"
    assert "503" in failing.error.message and "the model is loading" in failing.error.message


def test_the_identity_is_the_model_and_not_the_address() -> None:
    """What moves the vector space moves the index: the model, both prefixes and
    an asked-for width. The address does not — the same model served somewhere
    else is the same space, and moving a server must not cost an index."""
    here = EndpointEmbedder(base_url="http://a/v1", model="m")
    there = EndpointEmbedder(base_url="http://b/v1", model="m")
    other = EndpointEmbedder(base_url="http://a/v1", model="n")
    prefixed = EndpointEmbedder(base_url="http://a/v1", model="m", query_prefix="q: ")
    narrow = EndpointEmbedder(base_url="http://a/v1", model="m", dimensions=256)

    assert here.name == there.name
    assert len({here.name, other.name, prefixed.name, narrow.name}) == 4


# --------------------------------------------------------------- the session --


async def test_naming_the_endpoint_in_a_profile_is_enough_and_its_key_reaches_the_command(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** The shipped command, with no local model to be had.

    `sentence-transformers` is hidden from this process, as it is on an install
    without the `[local]` extra: the endpoint's row mounts all the same. The
    command learns which embedder to mount from the row being named to it, and
    is handed the one variable the row names — though a command's environment
    drops every name that looks like a credential, which this one does.
    """
    docs_tree(tmp_path, documents=3)
    monkeypatch.setenv(KEY_ENV, "sekrit")
    real = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name, *rest: None if name == "sentence_transformers" else real(name, *rest),
    )
    ctx = await mount(*_session(endpoint.url, model="fx-embed", apiKeyEnv=KEY_ENV))
    agent = new_agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    found = await run_tool(ctx, "text_search", {"query": "containment tier", "k": 1}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert [hit["path"] for hit in found.value["hits"]] == ["docs/w.md"]
    assert set(endpoint.keys) == {"Bearer sekrit"}
    seam = ctx.require(TEXT_INDEX)
    assert seam.provider is not None and seam.manual is not None
    assert seam.provider.label == f"fx-embed at {endpoint.url}"
    assert seam.manual.pass_env == [KEY_ENV]
    assert "sekrit" not in json.dumps(seam.manual.model_dump()), "the key is held in the manual"
    assert report_section(ctx, "Text index")["embedder"] == f"fx-embed at {endpoint.url}"
    [hosted] = ctx.require(UTCP).hosts.values()
    assert hosted.home is not None
    assert {one.name for one in hosted.home.iterdir() if one.is_dir()} == {
        digest("endpoint:fx-embed:|")
    }


@pytest.mark.needs_git
async def test_a_confined_agent_is_told_when_its_endpoints_host_is_not_allowed(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """**The gate.** A confined command has the network its deployment allows and
    no more. An endpoint outside it is a failure with a code and the way out,
    not a traceback: the host goes in `sandbox-allow`."""
    ctx = await mount({"id": "sandbox-local", "disabled": False}, *_session(endpoint.url))
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    repo = docs_tree(tmp_path / "repo")
    agent = new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=repo
    )

    refused = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert refused.is_error
    assert refused.error is not None and refused.error.info is not None
    assert refused.error.info["code"] in (
        "TEXT_INDEX_ENDPOINT_UNREACHABLE",
        "TEXT_INDEX_ENDPOINT_FAILED",
    ), text_of(refused.content)
    assert endpoint.asked == [], "a confined command reached a host nobody allowed"


@pytest.mark.needs_git
async def test_a_confined_agent_reaches_an_endpoint_whose_host_its_deployment_allows(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """The other half: named in `sandbox-allow`, host and port, the endpoint is
    reached from inside the sandbox — through the deployment's egress proxy,
    whose variables the standard library's client honors."""
    port = endpoint.url.rsplit(":", 1)[1].split("/")[0]
    allowed = {"network": {"mode": "allowlist", "hosts": [f"127.0.0.1:{port}"]}}
    ctx = await mount(
        {"id": "sandbox-local", "disabled": False},
        {"id": "sandbox-allow", "config": allowed},
        *_session(endpoint.url),
    )
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    repo = docs_tree(tmp_path / "repo")
    agent = new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=repo
    )

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["total_documents"] == 2
    assert endpoint.asked, "the endpoint was never asked"


# ----------------------------------------------------- how long an input is --


def test_an_endpoint_says_what_it_reads_where_it_can_be_asked(endpoint: Endpoint) -> None:
    """A llama.cpp answers `/props` with its window, so nobody has to write it
    down; a server with nothing to ask has no limit known ahead of time; and a
    setting is the answer wherever one is written."""
    from ph_text_index._config import Limit

    endpoint.window = 512
    assert EndpointEmbedder(base_url=endpoint.url).limit() == Limit(512, "server")
    assert EndpointEmbedder(base_url=endpoint.url, max_tokens=64).limit() == Limit(64, "configured")
    endpoint.window = None
    assert EndpointEmbedder(base_url=endpoint.url).limit() is None


async def test_passages_are_cut_to_the_window_a_server_reports(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """**The gate.** The shipped command, asking a server that reads 60 tokens:
    it is asked once what it reads, passages are packed to that, and it is never
    sent an input it would refuse — where a passage cut at `max_chars` was a 500
    that ended the run."""
    long_document(tmp_path)
    endpoint.window, endpoint.reads = 60, 60
    ctx = await mount(*_session(endpoint.url))

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["limit"] == "60 tokens (the server's)"
    # 58 tokens of text: two of the 60 are the markers every passage carries.
    assert indexed.value["passage_tokens"] == 58
    assert max(endpoint.took) <= 60
    assert "Passages are held to 58 tokens of text" in text_of(indexed.content)


async def test_a_server_that_refuses_a_passage_is_sent_a_smaller_one(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """**The gate.** A server with nothing to ask, which reads 40 tokens and says
    so only by refusing: its own sentence is recognized as "too long", the
    document is packed smaller and sent again, and the run ends with an index
    and a line naming the document."""
    long_document(tmp_path)
    endpoint.reads = 40
    ctx = await mount(*_session(endpoint.url))

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    assert endpoint.took and max(endpoint.took) <= 40
    assert indexed.value["cut_smaller"] == ["docs/long.md"]
    assert indexed.value["total_documents"] == 1
