"""`ph-code-graph` — the row's tools from a command line, and through a UTCP client.

Gates: *the command answers with the values the row answers with; two processes
indexing one tree report counts that add up; a real UTCP client registers the
manual and calls a tool, with an argument left out and an argument that is a list.*

## Why real processes

What this file is about only exists between processes. The lock that orders two
indexing runs is a file because an in-process one ordered nothing across them,
and a UTCP client reaches the tool by running a shell command with a cut-down
environment. Calling `run()` in this process would prove the argument parsing and
none of that, so the gates below start the interpreter the way a client does.

## Why the client's own templates

The UTCP test registers `manual_call_template(CLI)` unchanged — the object
`ph-code-graph manual --registration` prints — and calls what discovery returned.
A test that built its own template would be checking a manual nobody ships.
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
    """One body per tool. The row is mounted here and the command mounts it again
    in its own process; both read one index, so the answers are equal or there are
    two implementations."""
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

    # And the other direction: the row finds the command's index already current.
    again = await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)
    assert again.value["indexed"] == 0
    assert again.value["unchanged"] == 4


async def test_the_index_goes_where_the_caller_says_and_is_found_there_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--state`, and `$PH_TOOL_STATE` under it: the directory a session names for
    each agent, and a person names to try one. Three processes share nothing but
    that directory, so the second parsing nothing is the index being reused —
    which is what a restarted agent gets, because its directory is the same one.

    And with neither, the index is where it has always been: under `$PH_CACHE`,
    keyed by the workspace.
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


async def test_two_processes_indexing_one_tree_report_counts_that_add_up(tmp_path: Path) -> None:
    """Each file is parsed by exactly one of them, and the other finds it current.

    Before the lock was a file, two processes each stayed atomic per file and both
    parsed whatever the other had not yet written — so the two `indexed` counts
    summed past the number of files, and neither described what happened.
    """
    files = 41
    shared_tree(tmp_path, callers=files - 1)
    # Once, alone, on another tree: the first parse of a language materializes its
    # grammar into the cache, and that is the pack's own concurrency to answer for.
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
    """Waits, rather than being refused: a second run is expected to queue."""
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


# ----------------------------------------------------------------- UTCP, real --


async def test_a_real_utcp_client_registers_the_manual_and_calls_a_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The whole path another harness takes: discovery by running `manual`, then a
    call through the tool's own template, in the client's cut-down environment.

    `query` and `path` are left out of the last call on purpose. The client
    substitutes `MISSING_ARG_<name>` for a placeholder with no argument, so a
    command that read its flags as plain text would search for that literal.
    """
    shared_tree(tmp_path, callers=3)
    # A UTCP call runs in the client's working directory, which is the workspace.
    monkeypatch.chdir(tmp_path)
    client = await UtcpClient.create(config={"manual_call_templates": [manual_call_template(CLI)]})

    names = {tool.name for tool in await client.config.tool_repository.get_tools()}
    assert names == {"ph_code_graph.code_index", "ph_code_graph.code_graph"}

    # A list argument: it arrives as Python's repr of the list.
    indexed = await client.call_tool("ph_code_graph.code_index", {"paths": ["pkg"]})
    assert indexed["ok"] is True, indexed
    assert indexed["value"]["indexed"] == 4

    callers = await client.call_tool(
        "ph_code_graph.code_graph", {"mode": "callers", "query": "shared", "limit": 2}
    )
    assert callers["ok"] is True, callers
    assert len(callers["value"]["edges"]) == 2
    assert callers["value"]["truncated"] is True

    # Optional arguments left out, and the one given is an integer.
    entities = await client.call_tool("ph_code_graph.code_graph", {"mode": "entities", "limit": 3})
    assert entities["ok"] is True, entities
    assert entities["value"]["query"] is None
    assert len(entities["value"]["symbols"]) == 3
