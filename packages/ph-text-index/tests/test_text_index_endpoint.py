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

A third, for a harness that is not pH: a UTCP client of the library's own, which
hands a command the variables its template names and no others.
"""

from __future__ import annotations

import importlib.util
import io
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
    prose,
)

from ph.keys import SANDBOX, SHELL, WORKSPACE
from ph.llm.types import text_of
from ph.testing import MountProfile, report_section, run_tool, session_of
from ph.toolcli import CONFIG_ENV, INHERITED_ENV, manual_call_template, reporting_to
from ph.workspace_caches import digest
from ph_text_index._embed import EndpointEmbedder
from ph_text_index.cli import CLI, MODEL_ENV
from ph_text_index.row import TEXT_INDEX
from ph_utcp import UTCP
from ph_utcp._library import UtcpClient

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
    unreachable = await run_tool(
        nowhere, "text_index", {"paths": ["docs"]}, agent=new_agent(nowhere)
    )
    await nowhere.drain()
    await nowhere.dispose()

    endpoint.status = 503
    ctx = await mount(*_engine(endpoint.url))
    failing = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))

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
    home = ctx.require(UTCP)._state(seam.manual, agent)
    assert home is not None
    assert {one.name for one in home.iterdir() if one.is_dir()} == {digest("endpoint:fx-embed:|")}


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


async def test_the_key_reaches_the_tools_command_and_not_the_agents_shell(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The gate.** The key's value is set on the manual's own command and on
    nothing else. Under a real sandbox the tool embeds with it, and the same
    agent's `bash` has no such variable and cannot read it off another process:
    its environment is the scrubbed one, and its `/proc` is its own."""
    monkeypatch.setenv(KEY_ENV, "sekrit-value")
    port = endpoint.url.rsplit(":", 1)[1].split("/")[0]
    allowed = {"network": {"mode": "allowlist", "hosts": [f"127.0.0.1:{port}"]}}
    ctx = await mount(
        {"id": "sandbox-local", "disabled": False},
        {"id": "sandbox-allow", "config": allowed},
        *_session(endpoint.url, apiKeyEnv=KEY_ENV),
    )
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    repo = docs_tree(tmp_path / "repo")
    agent = new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=repo
    )

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    looked = await ctx.require(SHELL).run(
        "env; cat /proc/[0-9]*/environ 2>/dev/null | tr '\\0' '\\n'", agent=agent
    )

    assert not indexed.is_error, text_of(indexed.content)
    assert set(endpoint.keys) == {"Bearer sekrit-value"}, "the tool's command had no key"
    assert looked.confined_by is not None
    assert "PATH=" in looked.stdout, "the shell printed no environment to look through"
    assert KEY_ENV not in looked.stdout and "sekrit-value" not in looked.stdout


# ----------------------------------------------------- how long an input is --


def test_an_endpoint_says_what_it_reads_where_it_can_be_asked(endpoint: Endpoint) -> None:
    """A llama.cpp answers `/props` with its window, so nobody has to write it
    down; a server with nothing to ask has no limit known ahead of time; and a
    setting is the answer wherever one is written."""
    from ph_text_index._config import Limit

    endpoint.window = 512
    assert EndpointEmbedder(base_url=endpoint.url).limit() == Limit(tokens=512, source="server")
    assert EndpointEmbedder(base_url=endpoint.url, max_tokens=64).limit() == Limit(
        tokens=64, source="configured"
    )
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


async def test_a_key_whose_name_the_scrub_does_not_match_is_said_at_mount(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """The key is this tool's alone because every other command starts from the
    scrubbed environment — and the scrub goes by name. A variable it does not
    drop is in an agent's `bash` already, so `ph-utcp` says so of any manual
    that passes one, where a person reads it. One the scrub does drop is said
    nothing about."""
    docs_tree(tmp_path)
    loud = await mount(*_session(endpoint.url, apiKeyEnv="FX_EMBEDDINGS_AUTH"))
    quiet = await mount(*_session(endpoint.url, apiKeyEnv=KEY_ENV))

    said = report_section(loud, "UTCP tools")["ph_text_index"]

    assert "$FX_EMBEDDINGS_AUTH is passed to its commands by name" in said
    assert "every command an agent runs has it already" in said
    assert "scrub" not in report_section(quiet, "UTCP tools")["ph_text_index"]


# ------------------------------------------- what the server said, kept --


def _log(ctx: Any, agent: Any) -> list[dict[str, Any]]:  # noqa: ANN401
    """The embedder log in the sidecar of `agent`'s index, oldest first."""
    seam = ctx.require(TEXT_INDEX)
    home = ctx.require(UTCP)._state(seam.manual, agent)
    assert home is not None
    [sidecar] = home.glob("*/chunks.json")
    log: list[dict[str, Any]] = json.loads(sidecar.read_text(encoding="utf-8"))["embedder"]
    return log


def _record(ctx: Any, agent: Any) -> dict[str, Any]:  # noqa: ANN401
    """The newest entry of the embedder log: the one that is used."""
    return _log(ctx, agent)[-1]


async def test_what_the_server_said_is_kept_beside_the_index_and_a_search_asks_nothing(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """**The gate.** An index run asks the server once what it reads and keeps
    the answer in the index itself, saved with the passages it was used to cut.
    Every search is a command of its own, and each reads it there rather than
    putting the question again. There is no file beside the index."""
    long_document(tmp_path)
    endpoint.window, endpoint.reads = 60, 60
    ctx = await mount(*_session(endpoint.url))
    agent = new_agent(ctx)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    assert not indexed.is_error, text_of(indexed.content)
    assert endpoint.props == 1
    for _ in range(2):
        found = await run_tool(ctx, "text_search", {"query": "w7l1n3", "k": 1}, agent=agent)
        assert found.value["hits"]

    assert endpoint.props == 1, "a search asked the server what it reads"
    kept = _record(ctx, agent)
    assert kept["askedOf"] == endpoint.url.removesuffix("/v1")
    assert kept["reported"] == {"n_ctx": 60, "total_slots": 4}
    assert kept["limit"] == {"tokens": 60, "source": "server"}
    assert kept["learned"] == {}
    assert len(_log(ctx, agent)) == 1


async def test_what_a_refusal_taught_is_kept_until_the_server_says_something_else(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """**The gate.** A llama.cpp reports its window and refuses past its batch.
    The first run learns the real limit from one refusal and keeps it, so the
    next run — another command — is refused nothing. When the server is given
    another window the next run notices: the result says so, what was learned
    under the old answer is dropped, and passages are packed to the new one.
    The log keeps both answers, the newest last: what changed, and when."""
    long_document(tmp_path)
    endpoint.window, endpoint.reads = 200, 40
    ctx = await mount(*_session(endpoint.url))
    agent = new_agent(ctx)

    first = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    assert not first.is_error, text_of(first.content)
    assert endpoint.refused == 1 and first.value["embedder_changed"] is None
    assert _record(ctx, agent)["learned"]["tokens"] == 40

    (tmp_path / "docs" / "more.md").write_text(prose(12), encoding="utf-8")
    second = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    assert not second.is_error, text_of(second.content)
    assert endpoint.refused == 1, "a second run was refused what the first had learned"
    assert max(endpoint.took) <= 40

    endpoint.window, endpoint.reads = 400, 400
    sent = len(endpoint.took)
    (tmp_path / "docs" / "third.md").write_text(prose(12), encoding="utf-8")
    third = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    assert not third.is_error, text_of(third.content)
    changed = (
        "The embedder now reads 400 tokens (the server's); when last asked it read "
        "200 tokens (the server's)."
    )
    assert third.value["embedder_changed"] == changed
    assert changed in text_of(third.content)
    assert _record(ctx, agent)["learned"] == {}
    assert max(endpoint.took[sent:]) > 40, "passages were still packed to the old limit"
    old, new = _log(ctx, agent)
    assert old["limit"]["tokens"] == 200 and old["learned"]["tokens"] == 40
    assert new["limit"]["tokens"] == 400 and new["askedAtMs"] > old["askedAtMs"]


async def test_a_server_asked_at_another_address_is_said_and_logged(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """The index is keyed by the model and not the address, so a moved server
    keeps its index. The next run asks the new address, and its result says the
    embedder is asked somewhere else now; the log keeps both."""
    docs_tree(tmp_path)
    endpoint.window = 60
    first = await mount(*_engine(endpoint.url))
    await run_tool(first, "text_index", {"paths": ["docs"]}, agent=new_agent(first))
    await first.drain()
    await first.dispose()

    moved = endpoint.url.replace("127.0.0.1", "localhost")
    (tmp_path / "docs" / "new.md").write_text(prose(2), encoding="utf-8")
    second = await mount(*_engine(moved))
    again = await run_tool(second, "text_index", {"paths": ["docs"]}, agent=new_agent(second))

    assert not again.is_error, text_of(again.content)
    said = again.value["embedder_changed"]
    assert said is not None and f"now asked at {moved.removesuffix('/v1')}" in said
    assert f"was asked at {endpoint.url.removesuffix('/v1')}" in said


async def test_a_runs_progress_says_what_the_endpoint_spent(
    mount: MountProfile, tmp_path: Path, endpoint: Endpoint
) -> None:
    """Every progress line carries what the run has spent so far, and the last
    one the whole of it: requests, texts, and the tokens the server's `usage`
    reported — so whoever reads the stream can track what an index cost. The
    result says it too, for a caller that waited and read no stream."""
    docs_tree(tmp_path, documents=3)
    ctx = await mount(*_engine(endpoint.url))
    out = io.StringIO()

    with reporting_to(out):
        indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=new_agent(ctx))

    assert not indexed.is_error, text_of(indexed.content)
    last = json.loads(out.getvalue().splitlines()[-1])
    texts = sum(len(one["input"]) for one in endpoint.asked)
    tokens = sum(len(text.split()) for one in endpoint.asked for text in one["input"])
    assert last["state"] == "done"
    spent = {"requests": len(endpoint.asked), "texts": texts, "tokens": tokens}
    assert last["spent"] == spent
    assert indexed.value["spent"] == spent
    assert f"Spent embedding: {len(endpoint.asked)} request" in text_of(indexed.content)


# ---------------------------------------------------------- another harness --


async def test_a_utcp_client_of_another_harness_hands_the_key_to_the_command(
    tmp_path: Path, endpoint: Endpoint, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Outside pH nothing hands a variable across by `pass_env`: a UTCP client
    passes a command what its template names. So the manual names the variable
    the endpoint's config names — after the ones the package always reads, and
    by name alone — and the library's own client, with no pH around it, embeds
    with the key. Indexing is the proof; a search through such a client is
    `test_text_index_cli`'s.
    """
    docs_tree(tmp_path, documents=3)
    # A UTCP call runs in the client's working directory, which is the workspace.
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv(KEY_ENV, "sekrit")
    settings = {"baseUrl": endpoint.url, "model": "fx-embed", "apiKeyEnv": KEY_ENV}
    monkeypatch.setenv(CONFIG_ENV, json.dumps({"text-index-endpoint": settings}))
    client = await UtcpClient.create(config={"manual_call_templates": [manual_call_template(CLI)]})

    tools = await client.config.tool_repository.get_tools()
    indexed = await client.call_tool("ph_text_index.text_index", {"paths": ["docs"]})

    assert [one.name for one in tools] == ["ph_text_index.text_index", "ph_text_index.text_search"]
    for tool in tools:
        assert tool.tool_call_template.inherit_env_vars == [*INHERITED_ENV, *MODEL_ENV, KEY_ENV]
    # Reading the manual embeds nothing, so its own template names no key.
    assert KEY_ENV not in str(manual_call_template(CLI))
    assert indexed["ok"] is True, indexed
    assert indexed["value"]["total_documents"] == 3
    assert set(endpoint.keys) == {"Bearer sekrit"}
    assert "sekrit" not in json.dumps([tool.model_dump() for tool in tools])
