"""`ph-text-index` — the engine's tools from a command line, and through a UTCP client.

Gates: *the command answers with the values the engine answers with; an index
built by one process is searched by the next, where the caller said; a real UTCP
client registers the manual and calls a tool, with an argument left out and an
argument that is a list.*

## Why real processes

What this file is about only exists between processes: an index one command
builds and another searches, and a UTCP client that reaches the tool by running
a shell command with a cut-down environment. Calling `run()` in this process
would prove the argument parsing and none of that.

## Which command

`python -m text_index_stub` for everything that embeds — the package's own
engine, tools and manual with a deterministic embedder (`text_index_stub`) — and
the shipped `ph-text-index` for what it can say without loading its model: which
tools it has, how a harness registers it, and where its model stands.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from text_index_helpers import (
    command_environment,  # noqa: F401 - autouse, by being here
    docs_tree,
    new_agent,
)
from text_index_stub import STUB_CLI, HashingEmbedder

from ph.paths import resolve_roots
from ph.testing import MountProfile, run_tool
from ph.toolcli import (
    CONFIG_ENV,
    HOST_ENV,
    PROGRESS_FILE,
    STATE_ENV,
    ToolCli,
    manual_call_template,
    read_progress,
)
from ph.workspace_caches import digest
from ph_text_index.cli import CLI
from ph_text_index.engine import TEXT_INDEX
from ph_utcp._library import UtcpClient

pytestmark = pytest.mark.anyio


def _run(cli: ToolCli, root: Path, *argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*cli.invocation(), *argv],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def _call(root: Path, tool: str, *argv: str) -> dict[str, Any]:
    done = _run(STUB_CLI, root, "call", tool, *argv)
    assert done.stderr == "", f"a call writes nothing to stderr, not {done.stderr!r}"
    lines = done.stdout.splitlines()
    assert len(lines) == 1, f"a call prints exactly one line, not {done.stdout!r}"
    answer: dict[str, Any] = json.loads(lines[0])
    return answer


# ---------------------------------------------------------------- the command --


async def test_the_command_answers_with_the_values_the_engine_answers_with(
    mount: MountProfile, tmp_path: Path
) -> None:
    """One body per tool. The engine is mounted here and the command mounts it
    again in its own process; both read one index, so the answers are equal or
    there are two implementations."""
    docs_tree(tmp_path, documents=3)
    ctx = await mount({"id": "text-index", "name": "text-index-engine"})
    ctx.require(TEXT_INDEX).register(HashingEmbedder())
    agent = new_agent(ctx)

    indexed = await anyio.to_thread.run_sync(
        lambda: _call(tmp_path, "text_index", "--paths", "docs")
    )
    assert indexed["ok"] is True, indexed
    assert indexed["value"]["total_documents"] == 3

    arguments = {"query": "which sandbox backend confines a command", "k": 2}
    flags = [f"--{name}={value}" for name, value in arguments.items()]
    from_command = await anyio.to_thread.run_sync(partial(_call, tmp_path, "text_search", *flags))
    from_engine = await run_tool(ctx, "text_search", arguments, agent=agent)

    assert from_command["ok"] is True, from_command
    assert from_command["value"] == from_engine.value
    assert from_command["text"] == from_engine.content[0].text  # type: ignore[union-attr]


async def test_the_index_goes_where_the_caller_says_and_is_found_there_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--state`, and `$PH_TOOL_STATE` under it: the directory a session names for
    each agent, and a person names to try one. The processes share nothing but
    that directory, so the second finding the first's passages is the index being
    reused — which is what a restarted agent gets, because its directory is the
    same one.

    And with neither, the index is where it has always been: under `$PH_CACHE`,
    keyed by the workspace.
    """
    docs_tree(tmp_path, documents=2)
    state = tmp_path.parent / f"{tmp_path.name}-state"

    built = await anyio.to_thread.run_sync(
        lambda: _call(tmp_path, "text_index", "--paths", "docs", "--state", str(state))
    )
    monkeypatch.setenv(STATE_ENV, str(state))
    found = await anyio.to_thread.run_sync(
        lambda: _call(tmp_path, "text_search", "--query", "containment tier", "--k", "1")
    )
    monkeypatch.delenv(STATE_ENV)
    apart = await anyio.to_thread.run_sync(
        lambda: _call(tmp_path, "text_search", "--query", "containment tier")
    )

    assert built["value"]["total_documents"] == 2
    assert sorted(one.name for one in state.iterdir()) == sorted(
        [digest(HashingEmbedder().name), PROGRESS_FILE]
    )
    assert [hit["path"] for hit in found["value"]["hits"]] == ["docs/w.md"]
    assert apart["value"]["searched"] == 0, "with no state named, it is a different index"
    assert (resolve_roots().cache / "text-index" / digest(str(tmp_path))).is_dir()


async def test_a_failure_from_the_tool_is_in_the_answer(tmp_path: Path) -> None:
    """The exit code is ignored by a UTCP client, so the refusal has to be in what
    the command prints, with a code a caller can route on."""
    answer = await anyio.to_thread.run_sync(lambda: _call(tmp_path, "text_search", "--k", "3"))

    assert answer["ok"] is False
    assert answer["error"]["code"] == "TOOL_FAILED"
    assert "query" in answer["error"]["message"]


# ------------------------------------------------------- the shipped command --


async def test_the_shipped_command_names_its_tools_without_loading_its_model(
    tmp_path: Path,
) -> None:
    """Discovery, and `status`: what a harness and a person ask before anything is
    embedded. Neither loads the model, so neither downloads it — and `status`
    says so, rather than "not loaded", which is true of every process that has
    only just started."""
    manual = await anyio.to_thread.run_sync(lambda: _run(CLI, tmp_path, "manual"))
    status = await anyio.to_thread.run_sync(lambda: _run(CLI, tmp_path, "status"))

    assert manual.returncode == 0, manual.stderr
    assert [one["name"] for one in json.loads(manual.stdout)["tools"]] == [
        "text_index",
        "text_search",
    ]
    assert status.returncode == 0, status.stderr
    assert "all-MiniLM-L6-v2" in status.stdout
    assert "not downloaded — `install` fetches it" in status.stdout
    assert str(resolve_roots().cache / "models") in status.stdout
    assert not (resolve_roots().cache / "models").exists(), "asking fetched something"


def test_the_registration_names_this_interpreter_and_the_model_settings() -> None:
    """What another harness registers: the module run by the interpreter that
    printed it, and the variables an operator steers a local model with — a
    mirror, no network — among those the command inherits."""
    template = manual_call_template(CLI)

    assert template["name"] == "ph_text_index"
    assert "-m ph_text_index manual" in str(template["commands"])
    inherited = template["inherit_env_vars"]
    assert isinstance(inherited, list)
    assert {"HF_HUB_OFFLINE", "HF_ENDPOINT", "PH_CACHE", STATE_ENV} <= set(inherited)


# ----------------------------------------------------------------- UTCP, real --


async def test_a_real_utcp_client_registers_the_manual_and_calls_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path another harness takes: discovery by running `manual`, then a
    call through the tool's own template, in the client's cut-down environment.

    `glob`, `exclude` and `forget` are left out of the first call and `paths` of
    the last on purpose. The client substitutes `MISSING_ARG_<name>` for a
    placeholder with no argument, so a command that read its flags as plain text
    would index a file of that name.
    """
    docs_tree(tmp_path, documents=3)
    # A UTCP call runs in the client's working directory, which is the workspace.
    monkeypatch.chdir(tmp_path)
    client = await UtcpClient.create(
        config={"manual_call_templates": [manual_call_template(STUB_CLI)]}
    )

    names = {tool.name for tool in await client.config.tool_repository.get_tools()}
    assert names == {"ph_text_index.text_index", "ph_text_index.text_search"}

    # A list argument: it arrives as Python's repr of the list.
    indexed = await client.call_tool("ph_text_index.text_index", {"paths": ["docs"]})
    assert indexed["ok"] is True, indexed
    assert indexed["value"]["total_documents"] == 3

    # An integer, and a list that narrows.
    narrowed = await client.call_tool(
        "ph_text_index.text_search",
        {"query": "invoices issued monthly", "k": 1, "paths": ["docs/billing0.md"]},
    )
    assert narrowed["ok"] is True, narrowed
    assert [hit["path"] for hit in narrowed["value"]["hits"]] == ["docs/billing0.md"]

    # Optional arguments left out.
    found = await client.call_tool("ph_text_index.text_search", {"query": "containment tier"})
    assert found["ok"] is True, found
    assert found["value"]["hits"][0]["path"] == "docs/w.md"
    assert found["value"]["searched"] == indexed["value"]["total_chunks"]


# -------------------------------------------------------------------- a host --


@contextmanager
def _host(root: Path, *argv: str, config: dict[str, Any] | None = None) -> Iterator[Path]:
    """`serve`, as whoever wants a host starts one: a process of its own, on a
    socket, ended with its owner. Yields the socket once something listens."""
    socket_path = root.parent / f"{root.name}.sock"
    env = {**os.environ, **({CONFIG_ENV: json.dumps(config)} if config else {})}
    process = subprocess.Popen(
        [*STUB_CLI.invocation(), "serve", "--socket", str(socket_path), *argv],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not socket_path.exists():
            assert process.poll() is None, f"the host left at once: {process.communicate()}"
            assert time.monotonic() < deadline, "the host never listened"
            time.sleep(0.02)
        yield socket_path
    finally:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=30)


def _asked(root: Path, socket_path: Path, tool: str, *argv: str) -> dict[str, Any]:
    """`_call`, with the host named as `ph-utcp` names it."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(HOST_ENV, str(socket_path))
        return _call(root, tool, *argv)


def _until(done: Callable[[], bool], what: str) -> None:
    deadline = time.monotonic() + 60
    while not done():
        assert time.monotonic() < deadline, f"never saw {what}"
        time.sleep(0.05)


def test_a_host_loads_once_what_every_call_would_load(tmp_path: Path) -> None:
    """**The gate.** A second `text_search` does not load the model again.

    The stub's row writes a line each time it is mounted — each time a process
    "loads the model". Three calls through a host leave the one line the host
    wrote; the same three calls with no host leave three more, one a process.
    And a call that names a host nobody is running is still answered.
    """
    docs_tree(tmp_path, documents=2)
    record = tmp_path.parent / f"{tmp_path.name}-loads"
    config = {"text-index-stub": {"record": str(record)}}

    with (
        pytest.MonkeyPatch.context() as patch,
        _host(tmp_path, config=config) as socket_path,
    ):
        patch.setenv(CONFIG_ENV, json.dumps(config))
        indexed = _asked(tmp_path, socket_path, "text_index", "--paths", "docs")
        first = _asked(tmp_path, socket_path, "text_search", "--query", "containment tier")
        second = _asked(tmp_path, socket_path, "text_search", "--query", "containment tier")
        hosted = len(record.read_text(encoding="utf-8").splitlines())

        alone = [
            _call(tmp_path, "text_search", "--query", "containment tier"),
            _asked(
                tmp_path, tmp_path / "nobody.sock", "text_search", "--query", "containment tier"
            ),
        ]
        every = len(record.read_text(encoding="utf-8").splitlines())

    assert indexed["value"]["state"] == "done"
    assert first == second
    assert first["value"]["hits"][0]["path"] == "docs/w.md"
    assert hosted == 1, "a call through the host mounted the rows again"
    assert [one["value"] for one in alone] == [first["value"], first["value"]]
    assert every == 3


def test_a_host_answers_at_once_and_a_search_says_the_index_is_not_finished(
    tmp_path: Path,
) -> None:
    """**The gate.** In a host `text_index` does not wait for the run: it answers
    `indexing` with how far it has got, and `text_search` answers meanwhile from
    what is indexed so far and says so. When the run is over the progress file
    says `done`, the warning is gone, and asking again finds nothing to embed.
    """
    docs_tree(tmp_path, documents=20)
    # Slow enough that the run is still going two calls later: each waits a
    # second for it before it answers.
    config = {"text-index": {"waitS": 1.0}, "text-index-stub": {"delay": 0.25}}

    with _host(tmp_path, config=config) as socket_path:
        began = time.monotonic()
        started = _asked(tmp_path, socket_path, "text_index", "--paths", "docs")
        waited = time.monotonic() - began
        joined = _asked(tmp_path, socket_path, "text_index", "--paths", "docs")
        during = _asked(tmp_path, socket_path, "text_search", "--query", "invoices issued monthly")

        home = resolve_roots().cache / "text-index" / digest(str(tmp_path))
        _until(lambda: getattr(read_progress(home), "state", "") == "done", "the run finish")
        after = _asked(tmp_path, socket_path, "text_search", "--query", "invoices issued monthly")
        again = _asked(tmp_path, socket_path, "text_index", "--paths", "docs")

    assert started["value"]["state"] == "indexing"
    assert started["value"]["total"] == 20
    assert waited < 3, "the call waited for the run"
    assert "Indexing in the background" in started["text"]
    # The same request, asked again while its run is going, joins it.
    assert joined["value"]["state"] == "indexing"
    assert joined["value"]["done"] >= started["value"]["done"]
    assert 0 < during["value"]["indexing"]["done"] < 20
    assert during["value"]["hits"], "a search during the run found nothing indexed so far"
    assert "Not finished indexing" in during["text"]
    assert after["value"]["indexing"] is None
    assert after["value"]["searched"] == 20
    assert again["value"]["state"] == "done"
    assert (again["value"]["unchanged"], again["value"]["chunks_added"]) == (20, 0)


def test_a_host_that_is_killed_keeps_what_it_saved_and_the_next_run_goes_on(
    tmp_path: Path,
) -> None:
    """**The gate.** A worker killed with SIGKILL embeds nothing twice — up to its
    last save. The run saves as it goes, so the documents it had saved are found
    by the next run, by their digests, and only the rest are embedded.
    """
    docs_tree(tmp_path, documents=12)
    config = {"text-index": {"saveEveryS": 0.2}, "text-index-stub": {"delay": 0.3}}
    home = resolve_roots().cache / "text-index" / digest(str(tmp_path))

    with _host(tmp_path, "--warm", "text_index", config=config):
        _until(lambda: getattr(read_progress(home), "done", 0) >= 5, "five documents")
        host = int(subprocess.check_output(["pgrep", "-f", f"{tmp_path.name}.sock"]).split()[0])
        os.kill(host, signal.SIGKILL)

    resumed = _call(tmp_path, "text_index")

    assert resumed["ok"] is True, resumed
    assert resumed["value"]["unchanged"] >= 3, "nothing the killed host embedded was kept"
    assert resumed["value"]["unchanged"] + len(resumed["value"]["documents"]) == 12
    assert resumed["value"]["total_documents"] == 12


def test_a_host_leaves_when_nobody_has_asked_and_a_second_start_stands_down(
    tmp_path: Path,
) -> None:
    """A host costs a process and, for a model, memory — so it goes when it has
    been idle, and whoever wants one starts another. And two starts for one
    socket end with one host: the second finds it answering and leaves."""
    with _host(tmp_path, "--idle", "30") as socket_path:
        second = _run(STUB_CLI, tmp_path, "serve", "--socket", str(socket_path), "--idle", "30")
        assert second.returncode == 0
        assert socket_path.exists(), "the second start took the first one's socket"

    brief = subprocess.run(
        [*STUB_CLI.invocation(), "serve", "--socket", str(socket_path), "--idle", "0.3"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert brief.returncode == 0, brief.stderr
    assert not socket_path.exists(), "an idle host left its socket behind"
