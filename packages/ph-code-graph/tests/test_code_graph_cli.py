"""`ph-code-graph`: the tools of the row from a command line, and through a UTCP client.

Gates:

* The command answers with the values that the row answers with.
* Two processes that index one tree report counts that add up.
* A real UTCP client registers the manual and calls a tool, with one argument
  left out and one argument that is a list.

## Why real processes

The subject of this file exists only between processes. The lock that orders two
indexing runs is a file, because a lock inside one process ordered nothing across
processes. A UTCP client reaches the tool by running a shell command with a
reduced environment. A call to `run()` in this process would prove the argument
parsing and none of that. The gates below therefore start the interpreter in the
way that a client does.

## Why the templates of the client itself

The UTCP test registers `manual_call_template(CLI)` unchanged, which is the
object that `ph-code-graph manual --registration` prints. It then calls what
discovery returned. A test that built its own template would check a manual that
nobody ships.
"""

from __future__ import annotations

import json
import subprocess
from functools import partial
from pathlib import Path
from typing import Any

import anyio
import pytest
from code_graph_helpers import (
    grammar_cache,  # noqa: F401 - autouse, by being here
    shared_tree,
)

from ph.locks import acquire_file_lock
from ph.testing import FAKE_OPTIONS, MountProfile, run_tool
from ph.toolcli import manual_call_template
from ph.workspace_caches import digest
from ph_code_graph.cli import CLI
from ph_utcp._library import UtcpClient

pytestmark = pytest.mark.anyio


def _command(*argv: str) -> list[str]:
    return [*CLI.invocation(), *argv]


def _call(root: Path, tool: str, *argv: str) -> dict[str, Any]:
    done = subprocess.run(
        _command("call", tool, *argv),
        cwd=root,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.stderr == "", f"a call writes nothing to stderr, not {done.stderr!r}"
    lines = done.stdout.splitlines()
    assert len(lines) == 1, f"a call prints exactly one line, not {done.stdout!r}"
    answer: dict[str, Any] = json.loads(lines[0])
    return answer


# ---------------------------------------------------------------- the command --


async def test_the_command_answers_with_the_values_the_row_answers_with(
    mount: MountProfile, tmp_path: Path
) -> None:
    """There is one body for each tool. The row is mounted here, and the command
    mounts it again in its own process. Both read one index. The answers are
    therefore equal, or there are two implementations."""
    from ph.keys import AGENTS, SESSIONS

    shared_tree(tmp_path, callers=3)
    ctx = await mount({"id": "code-graph", "name": "code-graph-engine"})
    agent = ctx.require(AGENTS).create(ctx.require(SESSIONS).create("cli-parity"), FAKE_OPTIONS)

    indexed = await anyio.to_thread.run_sync(lambda: _call(tmp_path, "code_index", "--paths", "."))
    assert indexed["ok"] is True
    assert indexed["value"]["indexed"] == 4

    for arguments in (
        {"mode": "callers", "query": "shared"},
        {"mode": "define", "query": "shared"},
        {"mode": "entities"},
    ):
        flags = [f"--{name}={value}" for name, value in arguments.items()]
        from_command = await anyio.to_thread.run_sync(
            partial(_call, tmp_path, "code_graph", *flags)
        )
        from_row = await run_tool(ctx, "code_graph", arguments, agent=agent)

        assert from_command["ok"] is True
        assert from_command["value"] == from_row.value
        assert from_command["text"] == from_row.content[0].text  # type: ignore[union-attr]

    # The other direction: the row finds the index of the command already current.
    again = await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)
    assert again.value["indexed"] == 0
    assert again.value["unchanged"] == 4


async def test_the_index_goes_where_the_caller_says_and_is_found_there_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--state`, and `$PH_TOOL_STATE` below it. A session names this directory
    for each agent, and a person names one to try the command. Three processes
    share nothing except that directory. When the second process parses nothing,
    the index was reused. A restarted agent gets the same result, because its
    directory is the same one.

    With neither setting, the index is where it has always been: under
    `$PH_CACHE`, keyed by the workspace.
    """
    from ph.paths import resolve_roots
    from ph.toolcli import STATE_ENV

    shared_tree(tmp_path, callers=3)
    state = tmp_path.parent / f"{tmp_path.name}-state"

    def call(*argv: str) -> dict[str, Any]:
        return _call(tmp_path, "code_index", "--paths", ".", *argv)

    first = await anyio.to_thread.run_sync(lambda: call("--state", str(state)))
    monkeypatch.setenv(STATE_ENV, str(state))
    again = await anyio.to_thread.run_sync(call)
    monkeypatch.delenv(STATE_ENV)
    apart = await anyio.to_thread.run_sync(call)

    assert first["value"]["indexed"] == 4
    assert (state / "graph.db").exists()
    assert (again["value"]["indexed"], again["value"]["unchanged"]) == (0, 4)
    assert apart["value"]["indexed"] == 4, "with no state named, it is a different index"
    assert (resolve_roots().cache / "code-graph" / digest(str(tmp_path)) / "graph.db").exists()


async def test_a_failure_from_the_tool_keeps_its_routable_code(tmp_path: Path) -> None:
    answer = await anyio.to_thread.run_sync(
        lambda: _call(
            shared_tree(tmp_path, callers=3), "code_graph", "--mode", "define", "--query", "shared"
        )
    )

    assert answer == {
        "ok": False,
        "error": {"code": "CODE_GRAPH_FAILED", "message": answer["error"]["message"]},
    }
    assert "code_index" in answer["error"]["message"]


# ------------------------------------------------------------------- the lock --


@pytest.mark.parametrize("verb", [("manual",), ("status",), ("call", "code_index")])
def test_a_module_in_the_tree_is_not_imported_by_the_command(
    tmp_path: Path, verb: tuple[str, ...]
) -> None:
    """The command runs in the tree that it works on, and imports nothing from it.

    `python -m` puts the working directory first on `sys.path`. A `shlex.py` at
    the root of a repository therefore ran in place of the module of the standard
    library. That happened in discovery and in `status`, which no sandbox
    confines. `invocation()` adds `-P` to prevent it. Without `-P`, this test
    prints the marker."""
    marker = "IMPORTED FROM THE TREE"
    (tmp_path / "shlex.py").write_text(f"raise SystemExit({marker!r})\n", encoding="utf-8")
    done = subprocess.run(
        _command(*verb), cwd=tmp_path, capture_output=True, text=True, timeout=120, check=False
    )
    assert marker not in done.stdout + done.stderr
    assert "-P" in CLI.invocation()


async def test_two_processes_indexing_one_tree_report_counts_that_add_up(tmp_path: Path) -> None:
    """Each file is parsed by exactly one of the processes, and the other finds
    it current.

    Before the lock was a file, two processes each stayed atomic for each file,
    and both parsed whatever the other had not yet written. The two `indexed`
    counts then added up to more than the number of files, and neither count
    described what happened.
    """
    files = 41
    shared_tree(tmp_path, callers=files - 1)
    # Index one time, alone, on another tree. The first parse of a language
    # materializes its grammar into the cache, and the pack is responsible for
    # the concurrency of that step.
    warm = shared_tree(tmp_path.parent / f"{tmp_path.name}-warm", callers=1)
    assert (await anyio.to_thread.run_sync(lambda: _call(warm, "code_index")))["ok"]

    def both() -> list[dict[str, Any]]:
        started = [
            subprocess.Popen(
                _command("call", "code_index", "--paths", "."),
                cwd=tmp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for _ in range(2)
        ]
        return [json.loads(one.communicate(timeout=120)[0]) for one in started]

    first, second = (one["value"] for one in await anyio.to_thread.run_sync(both))

    assert first["indexed"] + second["indexed"] == files
    assert first["unchanged"] + second["unchanged"] == files
    assert first["total_files"] == second["total_files"] == files


async def test_an_indexing_run_waits_for_the_writer_that_holds_the_index(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The run waits and is not refused. A second run is expected to queue."""
    from ph.keys import AGENTS, SESSIONS
    from ph.paths import resolve_roots

    shared_tree(tmp_path, callers=3)
    ctx = await mount({"id": "code-graph", "name": "code-graph-engine"})
    agent = ctx.require(AGENTS).create(ctx.require(SESSIONS).create("cli-lock"), FAKE_OPTIONS)
    directory = resolve_roots().cache / "code-graph" / digest(str(tmp_path))
    release = acquire_file_lock(directory / "graph.db.lock", timeout=0, what="the test")
    finished = anyio.Event()

    async def index() -> None:
        await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)
        finished.set()

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(index)
        await anyio.sleep(0.4)
        assert not finished.is_set(), "the run went ahead while another writer held the index"
        release()
        with anyio.fail_after(60):
            await finished.wait()


async def test_a_call_that_will_not_wait_says_another_run_holds_the_index(
    tmp_path: Path,
) -> None:
    """After its wait, a second run answers and does not continue to wait behind
    the first. It answers `state: indexing`, with what the graph holds so far,
    and it parses nothing. `code_graph` never waits. It answers from what is
    indexed and says that the index is still being built. `--wait 0` is how a
    caller asks for an immediate answer, and a session uses it while its own run
    works."""
    shared_tree(tmp_path, callers=3)
    state = tmp_path.parent / f"{tmp_path.name}-state"
    built = _call(tmp_path, "code_index", "--state", str(state))
    assert built["value"]["state"] == "done" and built["value"]["indexed"] == 4

    release = acquire_file_lock(state / "graph.db.lock", timeout=0, what="the test")
    try:
        again = _call(tmp_path, "code_index", "--state", str(state), "--wait", "0")
        asked = _call(
            tmp_path, "code_graph", "--mode", "callers", "--query", "shared", "--state", str(state)
        )
    finally:
        release()

    assert again["ok"] is True and again["value"]["state"] == "indexing"
    assert again["value"]["indexed"] == 0 and again["value"]["total_files"] == 4
    assert "already building this graph" in again["text"]
    assert asked["value"]["indexing"] is True and len(asked["value"]["edges"]) == 3
    assert "[Not finished indexing." in asked["text"]


# ----------------------------------------------------------------- UTCP, real --


async def test_a_real_utcp_client_registers_the_manual_and_calls_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path that another harness takes: discovery by running `manual`,
    then a call through the template of the tool, in the reduced environment of
    the client.

    `query` and `path` are left out of the last call on purpose. The client
    substitutes `MISSING_ARG_<name>` for a placeholder that has no argument. A
    command that read its flags as plain text would search for that literal.
    """
    shared_tree(tmp_path, callers=3)
    # A UTCP call runs in the working directory of the client, which is the workspace.
    monkeypatch.chdir(tmp_path)
    client = await UtcpClient.create(config={"manual_call_templates": [manual_call_template(CLI)]})

    names = {tool.name for tool in await client.config.tool_repository.get_tools()}
    assert names == {"ph_code_graph.code_index", "ph_code_graph.code_graph"}

    # A list argument. It arrives as the Python repr of the list.
    indexed = await client.call_tool("ph_code_graph.code_index", {"paths": ["pkg"]})
    assert indexed["ok"] is True, indexed
    assert indexed["value"]["indexed"] == 4

    callers = await client.call_tool(
        "ph_code_graph.code_graph", {"mode": "callers", "query": "shared", "limit": 2}
    )
    assert callers["ok"] is True, callers
    assert len(callers["value"]["edges"]) == 2
    assert callers["value"]["truncated"] is True

    # Optional arguments are left out, and the one that is given is an integer.
    entities = await client.call_tool("ph_code_graph.code_graph", {"mode": "entities", "limit": 3})
    assert entities["ok"] is True, entities
    assert entities["value"]["query"] is None
    assert len(entities["value"]["symbols"]) == 3
