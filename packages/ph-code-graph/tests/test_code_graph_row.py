"""`code-graph` — the row a session mounts, whose tools run in the command.

Gates: *a session has `code_index` and `code_graph` under those names; an agent
working in its own tree indexes that tree and no other; the session's process
never loads the parser.*

## What is being held equal

`test_code_graph.py` proves the tools right, mounted in-process as the command
mounts them. This file proves that reaching them through the manual changes
nothing an agent can see — the names, the values, the rendered text, the error
codes, the skill — and changes the two things it was meant to: whose tree is
read, and which process pays for tree-sitter.

## Why the parser gate is a fresh interpreter

`sys.modules` is the whole question, and this process has imported tree-sitter a
hundred tests ago. Asked of an interpreter that has mounted only the session's
rows, the answer means something.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from code_graph_helpers import (
    cache_beside_the_tree,  # noqa: F401 - autouse, by being here
    grammar_cache,  # noqa: F401 - autouse, by being here
    new_agent,
    shared_tree,
)

from ph.agent.types import PreStepDecision, PreStepRequest
from ph.cordis import DEPLOYMENT
from ph.cordis.loader import MOUNT
from ph.keys import (
    COMMANDS,
    SANDBOX,
    SKILLS,
    SYSTEM_PROMPT,
    TOOLS,
    WORKSPACE,
)
from ph.llm.types import text_of
from ph.paths import resolve_roots
from ph.seams.workspace import scratch_of
from ph.testing import MountProfile, report_section, run_tool, session_of, settled
from ph.testing.git import git, git_repo
from ph.toolcli import CONFIG_ENV
from ph.tools.registry import RUN_CODE
from ph_code_graph.row import Config, manual
from ph_utcp import UTCP

pytestmark = pytest.mark.anyio

ROWS: tuple[dict[str, Any], ...] = (
    {"id": "utcp", "name": "utcp"},
    {"id": "code-graph", "name": "code-graph"},
)
"""What the `utcp` and `code-graph` bundles add to a profile, in their order."""


def _index(ctx: Any, agent: Any) -> Path:  # noqa: ANN401
    """Where `agent`'s index is: the row's directory in that agent's scratch."""
    return scratch_of(ctx, agent.session.id, agent) / "code-graph" / "graph.db"


# -------------------------------------------------------------- what is equal --


async def test_a_session_has_both_tools_under_their_own_names(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** Not `ph_code_graph_code_index`: the skill, the prompts and
    every program a model has written say `code_graph`."""
    ctx = await mount(*ROWS)

    schemas = {one.name: one for one in ctx.require(TOOLS).schemas(scope=DEPLOYMENT)}

    assert {"code_index", "code_graph"} <= set(schemas)
    assert schemas["code_graph"].description.startswith("Ask about a codebase's structure")
    assert set(schemas["code_graph"].parameters["properties"]) == {
        "mode",
        "query",
        "path",
        "kind",
        "distance",
        "limit",
        "offset",
    }
    assert ctx.require(UTCP).states["ph_code_graph"].tools == ("code_graph", "code_index")


async def test_a_call_gives_the_value_the_text_and_the_code_the_tool_always_gave(
    mount: MountProfile, tmp_path: Path
) -> None:
    shared_tree(tmp_path)
    ctx = await mount(*ROWS)
    agent = new_agent(ctx)

    early = await run_tool(ctx, "code_graph", {"mode": "define", "query": "shared"}, agent=agent)
    assert early.is_error
    assert early.error is not None and early.error.info is not None
    assert early.error.info["code"] == "CODE_GRAPH_FAILED", "the tool's own code, carried across"
    assert "code_index" in early.error.message

    indexed = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    assert not indexed.is_error, text_of(indexed.content)
    assert indexed.value["indexed"] == 3
    assert text_of(indexed.content).startswith("Indexed 3 files")

    callers = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)
    assert not callers.is_error, text_of(callers.content)
    assert {one["name"] for one in callers.value["edges"]} == {"caller0", "caller1"}
    # The tool's own rendering, written in its process — not pH's default JSON.
    assert "calling 'shared'" in text_of(callers.content)
    assert not text_of(callers.content).lstrip().startswith("{")


async def test_code_mode_hands_the_model_both_tools_through_the_sdk(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The claim the package rests on for its intended caller.

    Under `tools.mode: code` the model gets one callable and an SDK listing, so
    "can the RLM use this" is exactly "are these in the listing". A tool that
    arrives through a manual is registered like any other, so it is.
    """
    ctx = await mount(
        {"id": "tools", "config": {"mode": "code"}},
        {"id": "tools-code-mode", "name": "tools-code-mode"},
        {"id": "code-runtime-stub", "name": "code-runtime-stub"},
        *ROWS,
    )
    agent = new_agent(ctx)

    view = ctx.require(TOOLS).view(scope=agent.ctx)
    assert view.mode == "code"
    assert [schema.name for schema in view.schemas] == [RUN_CODE]
    assert {"code_index", "code_graph"} <= set(view.visible)

    prompt = await ctx.require(SYSTEM_PROMPT).assemble(agent=agent, scope=agent.ctx)
    sdk = dict(prompt.sections)["tools:sdk"]
    assert "async def tools.code_index(" in sdk
    assert "async def tools.code_graph(" in sdk
    assert "Ask about a codebase's structure" in sdk


async def test_the_skill_arrives_with_the_tools(mount: MountProfile, tmp_path: Path) -> None:
    """What the RLM reads before it reaches for these tools. Registered by the row,
    so the catalog entry exists exactly when the tools do."""
    ctx = await mount(*ROWS)

    assert "code-graph" in {one.name for one in ctx.require(SKILLS).list(scope=DEPLOYMENT)}
    skill = ctx.require(SKILLS).get("code-graph", DEPLOYMENT)
    assert skill is not None
    assert {"code_index", "code_graph"} <= set(skill.allowed_tools)
    assert len(skill.description) < 300
    body = ctx.require(SKILLS).body("code-graph", DEPLOYMENT)
    assert body is not None and 'code_graph(mode="impact"' in body


async def test_without_the_utcp_row_the_row_stands_down_and_offers_nothing(
    mount: MountProfile,
) -> None:
    """A catalog entry for tools nobody has is the failure this must not have. The
    row waits on `ctx.utcp`; with nothing providing it, it never activates."""
    ctx = await mount({"id": "code-graph", "name": "code-graph"})

    assert "code-graph" in ctx.require(MOUNT).inactive()
    assert "code_graph" not in ctx.require(TOOLS).names(scope=DEPLOYMENT)
    assert "code-graph" not in {one.name for one in ctx.require(SKILLS).list(scope=DEPLOYMENT)}


# ------------------------------------------------------------- what changed --


async def test_an_agent_in_its_own_tree_indexes_that_tree(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** Two agents, two trees, one session's rows.

    The command runs in the calling agent's workspace, so the index it builds is
    that workspace's: the child's own files, kept in the child's own scratch.
    Registered in-process, the same tool read whichever root `ctx.fs` resolved
    for the agent — and this is the proof the command line kept that.
    """
    shared_tree(tmp_path, callers=2)
    worktree = shared_tree(tmp_path.parent / f"{tmp_path.name}-worktree", callers=5)
    ctx = await mount(*ROWS)
    parent, child = new_agent(ctx), new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(child).id, agent_id=child.id, base=worktree
    )

    for agent in (parent, child):
        done = await run_tool(ctx, "code_index", {"paths": ["."]}, agent=agent)
        assert not done.is_error, text_of(done.content)

    theirs = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=child)
    ours = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=parent)

    assert len(theirs.value["edges"]) == 5
    assert len(ours.value["edges"]) == 2
    indexes = {_index(ctx, parent), _index(ctx, child)}
    assert len(indexes) == 2
    assert all(index.exists() for index in indexes)
    assert not (resolve_roots().cache / "code-graph").exists(), "nothing is shared"


@pytest.mark.needs_git
async def test_a_confined_agent_that_may_not_write_its_tree_still_indexes_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** Read-only bars the repository, and not what a tool builds
    about it.

    The default posture is read-only, and on a host with a sandbox backend that
    is enforced. The command keeps its index in the agent's scratch — writable
    under every mode — and its grammars and log in the directories the manual
    grants. So the index is built, by an agent whose own `bash` could not create a
    file beside the code, and the tree is exactly as it was.

    A git repository, so the command asks git what changed: the path that spawns a
    child of its own from inside the sandbox. In a directory of its own, because
    the suite's `$PH_HOME` — and so the scratch — is `tmp_path`.
    """
    ctx = await mount({"id": "sandbox-local", "disabled": False}, *ROWS)
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    repo = await git_repo(ctx, shared_tree(tmp_path / "repo"))
    await git(ctx, repo, "add", "-A")
    await git(ctx, repo, "commit", "-qm", "base")
    agent = new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=repo
    )

    first = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    again = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    asked = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)

    assert not first.is_error, text_of(first.content)
    assert first.value["indexed"] == 3
    assert again.value["unchanged"] == 3
    assert len(asked.value["edges"]) == 2
    assert _index(ctx, agent).exists()
    _, changed, _ = await git(ctx, repo, "status", "--porcelain")
    assert changed == "", "the tree was written"


def test_the_sessions_process_never_loads_the_parser(tmp_path: Path) -> None:
    """**The gate.** tree-sitter, and a crash in it, belong to the command."""
    shared_tree(tmp_path)
    probe = """
import json, sys
from pathlib import Path
import anyio
from ph.cordis import DEPLOYMENT, Context, Profile
from ph.cordis.loader import ProfileDocument
from ph.keys import TOOLS
from ph.tools.definition import ToolExecutionInput

ROWS = [
    {"id": "tools", "name": "tools"},
    {"id": "subprocess", "name": "subprocess-local"},
    {"id": "fs", "name": "fs-local"},
    {"id": "shell", "name": "shell-local"},
    {"id": "utcp", "name": "utcp"},
    {"id": "code-graph", "name": "code-graph"},
]

async def main():
    ctx = Context()
    await Profile.from_documents([ProfileDocument("probe", ROWS)]).mount(ctx, project=Path.cwd())
    tools = ctx.require(TOOLS)
    calls = (("code_index", {"paths": ["pkg"]}), ("code_graph", {"mode": "entities"}))
    for name, arguments in calls:
        result = await tools.execute(
            ToolExecutionInput(call_id=name, name=name, arguments=arguments, scope=DEPLOYMENT)
        )
        assert not result.is_error, result.content
    heavy = ("tree_sitter", "tree_sitter_language_pack")
    ours = ("ph_code_graph.engine", "ph_code_graph._extract", "ph_code_graph._store")
    loaded = sorted(n for n in sys.modules if n.split(".")[0] in heavy or n in ours)
    print(json.dumps({"symbols": len(result.value["symbols"]), "loaded": loaded}))
    await ctx.drain()
    await ctx.dispose()

anyio.run(main)
"""

    done = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout.splitlines()[-1])
    assert answer["symbols"] == 3, "the tools ran, in the command"
    assert answer["loaded"] == [], "and the session's process loaded neither it nor the engine"


# ------------------------------------------------------- what the row itself does --


async def test_the_rows_config_reaches_the_command(mount: MountProfile, tmp_path: Path) -> None:
    """A setting in the session's profile is the setting the tool indexes with.
    `languages: [rust]` makes a Python tree index nothing."""
    shared_tree(tmp_path)
    ctx = await mount(ROWS[0], {**ROWS[1], "config": {"languages": ["rust"]}})

    done = await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=new_agent(ctx))

    assert not done.is_error, text_of(done.content)
    assert done.value["indexed"] == 0


async def test_a_profile_points_the_row_at_a_package_and_the_command_indexes_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The `index` block, written under the session's row and read by the command:
    a bare `code_index` indexes the named package, less what is excluded."""
    shared_tree(tmp_path / "ours", callers=3)
    shared_tree(tmp_path / "theirs", callers=3)
    scope = {"paths": ["ours"], "exclude": ["**/mod0.py"]}
    ctx = await mount(ROWS[0], {**ROWS[1], "config": {"index": scope}})
    agent = new_agent(ctx)

    done = await run_tool(ctx, "code_index", {}, agent=agent)
    asked = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)

    assert not done.is_error, text_of(done.content)
    assert done.value["indexed"] == 3, "helpers.py, mod1.py and mod2.py of `ours`"
    assert {one["path"] for one in asked.value["edges"]} == {
        "ours/pkg/mod1.py",
        "ours/pkg/mod2.py",
    }


def test_the_row_is_named_to_the_command_with_what_differs_from_its_defaults() -> None:
    assert json.loads(manual(Config()).env[CONFIG_ENV]) == {"code-graph": {}}

    configured = manual(Config(languages=["python"], max_files=7))

    assert json.loads(configured.env[CONFIG_ENV]) == {
        "code-graph": {"languages": ["python"], "maxFiles": 7}
    }


def test_the_manual_grants_what_the_command_writes_outside_the_workspace(
    tmp_path: Path,
) -> None:
    """What agents share is granted — the grammar cache and the log, both under
    `$PH_CACHE`. The index is not: it is named as state, so it goes in the calling
    agent's own scratch and needs no grant."""
    own = manual(Config())
    granted = set(own.writable)
    cache = resolve_roots().cache

    assert own.state == "code-graph"
    assert str(cache / "logs") in granted
    assert any("tree-sitter" in path for path in granted)
    assert str(cache / "code-graph") not in granted


async def test_the_command_answers_through_the_packages_own_program(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`/code-graph status` — a person asks before an agent does, and it costs no
    model turn. The sentence is the command's, which is the one that knows."""
    ctx = await mount(*ROWS)

    said = await ctx.require(COMMANDS).dispatch("/code-graph status")

    assert said is not None
    assert "grammars under" in said and "ready" in said


async def test_the_row_reports_itself_to_doctor(mount: MountProfile, tmp_path: Path) -> None:
    shared_tree(tmp_path)
    ctx = await mount(*ROWS)

    rows = report_section(ctx, "Code graph")
    assert rows["tools"] == "code_graph, code_index (through UTCP)"
    assert "scratch" in rows["index"]
    assert rows["details"] == "/code-graph status"


async def test_status_counts_the_index_of_the_agent_that_asked(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Each agent has its own index, so "is it built" has an answer per agent: the
    command is told the asking agent's state directory, as its tools are."""
    shared_tree(tmp_path)
    ctx = await mount(*ROWS)
    builder, other = new_agent(ctx), new_agent(ctx)
    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=builder)

    commands = ctx.require(COMMANDS)
    built = await commands.dispatch("/code-graph status", agent=builder, scope=builder.ctx)
    unbuilt = await commands.dispatch("/code-graph status", agent=other, scope=other.ctx)

    assert built is not None and unbuilt is not None
    assert "files: 3" in built
    assert str(_index(ctx, builder)) in built
    assert "not built" in unbuilt


# -------------------------------------------------------------------- the host --


async def test_a_session_keeps_one_host_for_a_tree_and_its_calls_go_through_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The command that answers the tools stays up between calls, one for each
    tree: started at the first call, by `ctx.shell`, for the agent that made it."""
    shared_tree(tmp_path)
    ctx = await mount(*ROWS)
    agent = new_agent(ctx)
    utcp = ctx.require(UTCP)

    await run_tool(ctx, "code_index", {"paths": ["pkg"]}, agent=agent)
    [first] = utcp.hosts.values()
    asked = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)
    [again] = utcp.hosts.values()

    assert len(asked.value["edges"]) == 2
    assert again is first and first.running
    assert first.owner is agent and first.socket.exists()
    assert first.host.label == "code graph"


async def test_a_root_agents_first_step_builds_the_graph_and_it_is_told_when_it_ends(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** No tool is called: the row's `agent/pre-step` listener starts
    the host with `code_index` to run, the graph is built in the background, and
    the agent is told with a note that waits for its next step."""
    shared_tree(tmp_path, callers=4)
    ctx = await mount(*ROWS)
    agent = new_agent(ctx)

    async def entered() -> PreStepDecision:
        return PreStepDecision(kind="enter")

    request = PreStepRequest(agent=agent, session=session_of(agent), messages=(), turn=1, step=1)
    await ctx.require(UTCP).eager(request, entered)
    await settled(lambda: bool(agent.inbox.next_step), "the completion note", seconds=60)

    [note] = agent.inbox.next_step
    assert "The code graph is finished: 5 files" in text_of(note.content)
    asked = await run_tool(ctx, "code_graph", {"mode": "callers", "query": "shared"}, agent=agent)
    assert len(asked.value["edges"]) == 4
    assert asked.value["indexing"] is None


async def test_on_mount_false_starts_nothing_until_a_tool_is_called(
    mount: MountProfile, tmp_path: Path
) -> None:
    shared_tree(tmp_path)
    ctx = await mount(ROWS[0], {**ROWS[1], "config": {"index": {"onMount": False}}})
    agent = new_agent(ctx)

    async def entered() -> PreStepDecision:
        return PreStepDecision(kind="enter")

    request = PreStepRequest(agent=agent, session=session_of(agent), messages=(), turn=1, step=1)
    await ctx.require(UTCP).eager(request, entered)

    assert not ctx.require(UTCP).hosts
    assert manual(Config.model_validate({"index": {"onMount": False}})).host is not None
