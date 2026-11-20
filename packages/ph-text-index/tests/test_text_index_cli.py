"""`ph-text-index`: the tools of the engine from a command line, and through a UTCP client.

Gates:

* The command answers with the values that the engine answers with.
* An index that one process builds is searched by the next process, in the place
  that the caller named.
* A real UTCP client registers the manual and calls a tool, with one argument
  left out and one argument that is a list.

## Why real processes

The subject of this file exists only between processes: an index that one
command builds and another command searches, and a UTCP client that reaches the
tool by running a shell command with a reduced environment. A call to `run()` in
this process would prove the argument parsing and none of that.

## Which command

`python -m text_index_stub` runs everything that embeds. It is the package's own
engine, tools, and manual, with a deterministic embedder (`text_index_stub`).
The shipped `ph-text-index` runs what it can answer without loading its model:
which tools it has, how a harness registers it, and what state its model is in.
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
from ph.toolcli import CONFIG_ENV, STATE_ENV, ToolCli, manual_call_template
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
    """There is one body for each tool. The engine is mounted here, and the
    command mounts it again in its own process. Both read one index. The answers
    are therefore equal, or there are two implementations."""
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
    """`--state`, and `$PH_TOOL_STATE` below it. A session names this directory
    for each agent, and a person names one to try the command. The processes
    share nothing except that directory. When the second process finds the
    passages of the first, the index was reused. A restarted agent gets the same
    result, because its directory is the same one.

    With neither setting, the index is where it has always been: under
    `$PH_CACHE`, keyed by the workspace.
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
    assert [one.name for one in state.iterdir()] == [digest(HashingEmbedder().name)]
    assert [hit["path"] for hit in found["value"]["hits"]] == ["docs/w.md"]
    assert apart["value"]["searched"] == 0, "with no state named, it is a different index"
    # A search of an index that nobody has built creates nothing. It only reads.
    assert not (resolve_roots().cache / "text-index" / digest(str(tmp_path))).exists()


async def test_a_failure_from_the_tool_is_in_the_answer(tmp_path: Path) -> None:
    """A UTCP client ignores the exit code. The refusal must therefore be in the
    text that the command prints, with a code that a caller can route on."""
    answer = await anyio.to_thread.run_sync(lambda: _call(tmp_path, "text_search", "--k", "3"))

    assert answer["ok"] is False
    assert answer["error"]["code"] == "TOOL_FAILED"
    assert "query" in answer["error"]["message"]


# ------------------------------------------------------- the shipped command --


async def test_the_shipped_command_names_its_tools_without_loading_its_model(
    tmp_path: Path,
) -> None:
    """Discovery and `status`: what a harness and a person ask before anything is
    embedded. Neither loads the model, so neither downloads it. `status` says
    "not downloaded" and not "not loaded", because "not loaded" is true of every
    process that has just started."""
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
    """What another harness registers: the module, run by the interpreter that
    printed the registration, and the variables that an operator uses to steer a
    local model, for example to a mirror or to no network. These variables are
    among the variables that the command inherits."""
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
    """The whole path that another harness takes: discovery by running `manual`,
    then a call through the template of the tool, in the reduced environment of
    the client.

    `glob`, `exclude`, and `forget` are left out of the first call on purpose,
    and `paths` is left out of the last call. The client substitutes
    `MISSING_ARG_<name>` for a placeholder that has no argument. A command that
    read its flags as plain text would index a file of that name.
    """
    docs_tree(tmp_path, documents=3)
    # A UTCP call runs in the working directory of the client, which is the workspace.
    monkeypatch.chdir(tmp_path)
    client = await UtcpClient.create(
        config={"manual_call_templates": [manual_call_template(STUB_CLI)]}
    )

    names = {tool.name for tool in await client.config.tool_repository.get_tools()}
    assert names == {"ph_text_index.text_index", "ph_text_index.text_search"}

    # A list argument. It arrives as the Python repr of the list.
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


# ----------------------------------------------------------- a run, reported --


@contextmanager
def _running(
    root: Path, *argv: str, config: dict[str, Any] | None = None
) -> Iterator[subprocess.Popen[str]]:
    """`call text_index --progress`, as a session starts a run in the background.
    The run is a separate process, and its stderr is a pipe that the report is
    read from."""
    env = {**os.environ, **({CONFIG_ENV: json.dumps(config)} if config else {})}
    process = subprocess.Popen(
        [*STUB_CLI.invocation(), "call", "text_index", "--progress", *argv],
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=30)


def _reported(process: subprocess.Popen[str], until: Callable[[dict[str, Any]], bool]) -> None:
    """Read the run's report, a line at a time, up to the line `until` accepts."""
    assert process.stderr is not None
    for line in process.stderr:
        if until(json.loads(line)):
            return
    raise AssertionError("the report ended before the line that was waited for")


def test_a_run_reports_on_stderr_and_a_search_meanwhile_reads_what_it_has_saved(
    tmp_path: Path,
) -> None:
    """**The gate.** A run reports its progress on stderr and ends with one last
    line. While the run works, nothing waits for it. A second `text_index` that
    will not wait answers `indexing`. `text_search` answers immediately from
    what the run has saved so far, and says that the index is not finished. When
    the run is over, the warning is gone, and a new `text_index` finds nothing
    to embed.
    """
    docs_tree(tmp_path, documents=20)
    config = {"text-index": {"saveEveryS": 0.2}, "text-index-stub": {"delay": 0.25}}

    with (
        pytest.MonkeyPatch.context() as patch,
        _running(tmp_path, "--paths", "docs", config=config) as run,
    ):
        patch.setenv(CONFIG_ENV, json.dumps(config))
        _reported(run, lambda line: line["done"] >= 6)
        began = time.monotonic()
        second = _call(tmp_path, "text_index", "--paths", "docs", "--wait", "0")
        during = _call(tmp_path, "text_search", "--query", "invoices issued monthly")
        waited = time.monotonic() - began
        out, err = run.communicate(timeout=120)
        after = _call(tmp_path, "text_search", "--query", "invoices issued monthly")
        again = _call(tmp_path, "text_index", "--paths", "docs")

    last = json.loads(err.splitlines()[-1])
    assert (last["state"], last["done"], last["total"]) == ("done", 20, 20)
    assert last["changed"] is True and "20 documents" in last["note"]
    assert json.loads(out)["value"]["state"] == "done"
    assert second["value"]["state"] == "indexing"
    assert "already building this index" in second["text"]
    assert waited < 4, "a call waited for the run"
    assert during["value"]["indexing"] is True
    assert during["value"]["hits"], "a search during the run found nothing saved so far"
    assert 0 < during["value"]["searched"] < 20
    assert "Not finished indexing" in during["text"]
    assert after["value"]["indexing"] is False
    assert after["value"]["searched"] == 20
    assert again["value"]["state"] == "done"
    assert (again["value"]["unchanged"], again["value"]["chunks_added"]) == (20, 0)


def test_a_run_that_is_killed_writes_no_last_line_and_the_next_one_goes_on(
    tmp_path: Path,
) -> None:
    """**The gate.** A run is killed with SIGKILL. Nothing that it embedded up to
    its last save is embedded a second time. Its report ends without a last
    line, and that is how a reader of the report knows that the run did not
    finish. The next run finds the saved documents by their digests and embeds
    only the others.
    """
    docs_tree(tmp_path, documents=12)
    config = {"text-index": {"saveEveryS": 0.2}, "text-index-stub": {"delay": 0.3}}

    with _running(tmp_path, config=config) as run:
        _reported(run, lambda line: line["done"] >= 5)
        run.send_signal(signal.SIGKILL)
        _out, err = run.communicate(timeout=30)

    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(CONFIG_ENV, json.dumps(config))
        resumed = _call(tmp_path, "text_index")

    assert all(json.loads(line)["state"] == "working" for line in err.splitlines())
    assert resumed["ok"] is True, resumed
    assert resumed["value"]["unchanged"] >= 3, "nothing the killed run embedded was kept"
    assert resumed["value"]["unchanged"] + len(resumed["value"]["documents"]) == 12
    assert resumed["value"]["total_documents"] == 12
