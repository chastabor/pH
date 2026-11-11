"""The `text-index` rows a session mounts: the manual, registered through `ph-utcp`.

Gates: *a session offers `text_index` and `text_search` under their own names; a
child agent in a worktree indexes its own tree; the session's process never
loads the model; a profile that names a different embedder gets a separate
index.*

## What is under test

`test_text_index.py` proves the engine and `test_text_index_cli.py` the command.
This file proves the join: that mounting the rows a profile gets from the
`utcp` and `text-index` bundles gives an agent the same tools, values, text and
error codes it had when they were registered in-process — and that what changed
is what was meant to change.

## Real processes, and a stub embedder

Every call below starts a command, as `ph-utcp` does for an agent. The command
is `python -m text_index_stub`: the package's own engine, tools and manual, with
a deterministic embedder where the shipped one downloads a model
(`text_index_stub`). The session's side names it with a row of its own, as
`text-index-local` names a model — which is the seam a deployment's own embedder
uses, exercised.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from text_index_helpers import (
    ROWS,
    command_environment,  # noqa: F401 - autouse, by being here
    docs_tree,
    new_agent,
)
from text_index_stub import STUB_ROW

from ph.agent.types import PreStepDecision, PreStepRequest
from ph.cordis import DEPLOYMENT, MountRefusal
from ph.cordis.loader import MOUNT
from ph.json import as_str
from ph.keys import AGENTS, COMMANDS, SANDBOX, SKILLS, SYSTEM_PROMPT, TOOLS, WORKSPACE
from ph.llm.types import text_of
from ph.paths import resolve_roots
from ph.seams.workspace import WorkspaceAccess, scratch_of
from ph.testing import (
    FAKE_OPTIONS,
    MountProfile,
    admitted_child,
    report_section,
    run_tool,
    session_of,
    settle_background,
    settled,
)
from ph.testing.git import git, git_repo
from ph.testing.stub_workspace import StubWorkspaceProvider
from ph.toolcli import CONFIG_ENV
from ph.tools.registry import RUN_CODE
from ph.workspace_caches import digest
from ph_text_index._config import Config, LocalConfig
from ph_text_index.cli import CLI
from ph_text_index.row import TEXT_INDEX, Provider, manual
from ph_utcp import UTCP

pytestmark = pytest.mark.anyio


def _indexes(ctx: Any, agent: Any) -> Path:  # noqa: ANN401
    """Where `agent`'s indexes are: the rows' directory in that agent's scratch."""
    return scratch_of(ctx, agent.session.id, agent) / "text-index"


# -------------------------------------------------------------- what is equal --


async def test_a_session_has_both_tools_under_their_own_names(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** Not `ph_text_index_text_search`: the skill, the prompts and
    every program a model has written say `text_search`."""
    ctx = await mount(*ROWS)

    schemas = {one.name: one for one in ctx.require(TOOLS).schemas(scope=DEPLOYMENT)}

    assert {"text_index", "text_search"} <= set(schemas)
    assert schemas["text_search"].description.startswith("Search the text index by meaning")
    assert set(schemas["text_search"].parameters["properties"]) == {"query", "k", "paths"}
    assert set(schemas["text_index"].parameters["properties"]) == {
        "paths",
        "glob",
        "exclude",
        "forget",
    }
    assert ctx.require(UTCP).states["ph_text_index"].tools == ("text_index", "text_search")


async def test_a_call_gives_the_value_and_the_text_the_tool_always_gave(
    mount: MountProfile, tmp_path: Path
) -> None:
    docs_tree(tmp_path, documents=3)
    ctx = await mount(*ROWS)
    agent = new_agent(ctx)

    early = await run_tool(ctx, "text_search", {"query": "sandbox"}, agent=agent)
    assert not early.is_error, text_of(early.content)
    assert early.value["hits"] == []
    assert "Index the documents first" in text_of(early.content)

    indexed = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    assert not indexed.is_error, text_of(indexed.content)
    assert sorted(indexed.value["documents"]) == [
        "docs/billing0.md",
        "docs/billing1.md",
        "docs/w.md",
    ]
    assert text_of(indexed.content).startswith("Indexed 3 documents")

    found = await run_tool(
        ctx, "text_search", {"query": "which sandbox backend confines", "k": 1}, agent=agent
    )
    assert not found.is_error, text_of(found.content)
    [hit] = found.value["hits"]
    assert hit["path"] == "docs/w.md"
    lines = (tmp_path / hit["path"]).read_text(encoding="utf-8").splitlines()
    assert hit["text"].splitlines()[0] == lines[hit["start_line"] - 1]
    # The tool's own rendering, written in its process — not pH's default JSON.
    assert text_of(found.content).startswith("1 passage for")


async def test_an_argument_the_tool_refuses_keeps_the_tools_own_words(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Validation is the tool's, in its process, and the sentence comes back."""
    ctx = await mount(*ROWS)

    refused = await run_tool(ctx, "text_search", {"query": "x", "k": 500}, agent=new_agent(ctx))

    assert refused.is_error
    assert "k" in text_of(refused.content)


async def test_code_mode_hands_the_model_both_tools_through_the_sdk(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Under Code Mode "can the RLM use this" is "are these in the SDK listing".
    A tool that arrives through a manual is registered like any other, so it is."""
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
    assert {"text_index", "text_search"} <= set(view.visible)

    prompt = await ctx.require(SYSTEM_PROMPT).assemble(agent=agent, scope=agent.ctx)
    sdk = dict(prompt.sections)["tools:sdk"]
    assert "async def tools.text_index(" in sdk
    assert "async def tools.text_search(" in sdk
    assert "Search the text index by meaning" in sdk


# ------------------------------------------------- the manual waits for a claim --


async def test_no_embedder_means_no_tools_no_skill_and_a_doctor_line(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The absence is complete, not partial. A tool named in every prompt and
    refused on every call would teach a capability the deployment does not have,
    so the manual is registered by the embedder's claim and not before."""
    ctx = await mount(*ROWS[:2])
    agent = new_agent(ctx)

    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is None
    assert ctx.require(TOOLS).get("text_index", scope=DEPLOYMENT) is None
    assert "text-search" not in {one.name for one in ctx.require(SKILLS).list(scope=DEPLOYMENT)}
    assert "none registered" in report_section(ctx, "Text index")["embedder"]
    said = await ctx.require(COMMANDS).dispatch("/text-index install", agent=agent, scope=agent.ctx)
    assert said is not None and "No embedder is registered" in said


async def test_unmounting_the_embedder_takes_the_tools_and_the_skill_with_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Registered on the provider's scope, so they leave on it — and the slot is
    free for another embedder to claim."""
    ctx = await mount(*ROWS)
    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is not None

    await ctx.require(MOUNT).forks[STUB_ROW].dispose()

    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is None
    assert "text-search" not in {one.name for one in ctx.require(SKILLS).list(scope=DEPLOYMENT)}
    seam = ctx.require(TEXT_INDEX)
    assert seam.provider is None and seam.manual is None
    assert "ph_text_index" not in ctx.require(UTCP).states


async def test_a_second_embedder_is_refused(mount: MountProfile, tmp_path: Path) -> None:
    """One index is one vector space."""
    ctx = await mount(*ROWS)
    seam = ctx.require(TEXT_INDEX)
    held = seam.provider
    assert held is not None

    with pytest.raises(RuntimeError, match="already registered"):
        await seam.register(held)


async def test_without_the_utcp_row_the_rows_stand_down_and_offer_nothing(
    mount: MountProfile,
) -> None:
    """A catalog entry for tools nobody has is the failure this must not have. The
    rows wait on `ctx.utcp`; with nothing providing it, they never activate."""
    ctx = await mount(*ROWS[1:])

    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is None
    assert "text-search" not in {one.name for one in ctx.require(SKILLS).list(scope=DEPLOYMENT)}


async def test_the_skill_arrives_with_the_tools(mount: MountProfile, tmp_path: Path) -> None:
    """What the RLM reads before it reaches for these tools. Registered by the
    embedder's claim, so the catalog entry exists exactly when the tools do — and
    only its description rides the prompt (G9); the body is a page, on disk."""
    ctx = await mount(*ROWS)

    skill = ctx.require(SKILLS).get("text-search", DEPLOYMENT)
    assert skill is not None
    assert "meaning" in skill.description and len(skill.description) < 300
    assert {"text_index", "text_search"} <= set(skill.allowed_tools)
    body = ctx.require(SKILLS).body("text-search", DEPLOYMENT)
    assert body is not None and len(body) > 1_500
    assert "text_search(query=" in body


# ------------------------------------------------------------- what changed --


async def test_an_agent_in_its_own_tree_indexes_that_tree(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** Two agents, two trees, one session's rows.

    The command runs in the calling agent's workspace, so the index it builds is
    that workspace's: the child's own documents, kept in the child's own scratch.
    """
    docs_tree(tmp_path, documents=2)
    worktree = docs_tree(tmp_path.parent / f"{tmp_path.name}-worktree", documents=4)
    ctx = await mount(*ROWS)
    parent, child = new_agent(ctx), new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(child).id, agent_id=child.id, base=worktree
    )

    counts = {}
    for name, agent in (("parent", parent), ("child", child)):
        done = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
        assert not done.is_error, text_of(done.content)
        counts[name] = done.value["total_documents"]

    assert counts == {"parent": 2, "child": 4}
    assert _indexes(ctx, parent) != _indexes(ctx, child)
    assert _indexes(ctx, parent).is_dir() and _indexes(ctx, child).is_dir()
    assert not (resolve_roots().cache / "text-index").exists(), "nothing is shared"


@pytest.mark.needs_git
async def test_a_confined_agent_that_may_not_write_its_tree_still_indexes_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** Read-only bars the repository, and not what a tool builds
    about it.

    The default posture is read-only, and on a host with a sandbox backend that
    is enforced. The command keeps its index in the agent's scratch — writable
    under every mode — so it is built by an agent whose own `bash` could not
    create a file beside the documents, the tree is exactly as it was, and a
    second run finds the first one's work: git vouches for every document.

    In a directory of its own, because the suite's `$PH_HOME` — and so the
    scratch — is `tmp_path`.
    """
    # Beside the command's log, which is the one shared directory this manual is
    # granted: a confined command can write nothing else outside its scratch.
    record = resolve_roots().cache / "logs" / "loads"
    embedder = {**ROWS[2], "config": {"record": str(record)}}
    ctx = await mount({"id": "sandbox-local", "disabled": False}, *ROWS[:2], embedder)
    if not ctx.require(SANDBOX).available:
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    repo = await git_repo(ctx, docs_tree(tmp_path / "repo", documents=3))
    await git(ctx, repo, "add", "-A")
    await git(ctx, repo, "commit", "-qm", "base")
    agent = new_agent(ctx)
    await ctx.require(WORKSPACE).acquire(
        session_id=session_of(agent).id, agent_id=agent.id, base=repo
    )

    first = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    again = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    asked = await run_tool(ctx, "text_search", {"query": "containment tier", "k": 1}, agent=agent)

    assert not first.is_error, text_of(first.content)
    assert len(first.value["documents"]) == 3
    assert (again.value["unchanged"], again.value["chunks_added"]) == (3, 0)
    assert [hit["path"] for hit in asked.value["hits"]] == ["docs/w.md"]
    assert _indexes(ctx, agent).is_dir()
    _, changed, _ = await git(ctx, repo, "status", "--porcelain")
    assert changed == "", "the tree was written"
    # And the three calls were answered by one host: a confined command reached a
    # confined host, on a socket both sandboxes were granted.
    assert len(record.read_text(encoding="utf-8").splitlines()) == 2


async def test_a_profile_that_names_a_different_embedder_gets_a_separate_index(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate, and E1.** The embedder's settings are the session's profile's,
    carried to the command; and the index is keyed by the embedder's identity, so
    changing them starts a second index beside the first and changing back would
    find the first intact."""
    docs_tree(tmp_path)
    ctx = await mount(*ROWS[:2], {**ROWS[2], "config": {"dim": 64}})
    agent = new_agent(ctx)

    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)

    built = {one.name for one in _indexes(ctx, agent).iterdir() if one.is_dir()}
    assert built == {digest("stub-hashing:64")}
    assert report_section(ctx, "Text index")["embedder"] == "stub-hashing:64"


async def test_the_rows_config_reaches_the_command(mount: MountProfile, tmp_path: Path) -> None:
    """A profile that points the row at a tree and leaves a file out is obeyed by
    a process the session never configured directly."""
    docs_tree(tmp_path, documents=3)
    (tmp_path / "notes.md").write_text("# Elsewhere\n\nNot under docs.\n", encoding="utf-8")
    scope = {"paths": ["docs"], "exclude": ["**/billing1.md"]}
    ctx = await mount(ROWS[0], {**ROWS[1], "config": {"index": scope}}, ROWS[2])

    done = await run_tool(ctx, "text_index", {}, agent=new_agent(ctx))

    assert not done.is_error, text_of(done.content)
    assert sorted(done.value["documents"]) == ["docs/billing0.md", "docs/w.md"]


def test_the_manual_names_both_rows_so_the_command_mounts_that_embedder() -> None:
    """Being named in `$PH_TOOL_CONFIG` is how the command learns which embedder
    row to mount, so a default embedder is named too. `preload` travels with the
    rest of the embedder's config, and the command does not read it: the session
    asks for the load once, by running `install`."""
    provider = Provider(
        row="text-index-local", config=LocalConfig(preload=True), cli=CLI, label="m"
    )

    own = manual(Config(), provider)

    assert json.loads(own.env[CONFIG_ENV]) == {
        "text-index": {},
        "text-index-local": {"preload": True},
    }
    assert own.state == "text-index"
    assert own.tools["text_index"].timeout_ms == 30 * 60 * 1000


def test_the_manual_grants_what_agents_share_and_not_the_index() -> None:
    """The weights and the log are granted, both under `$PH_CACHE`. The index is
    not: it is named as state, so it goes in the calling agent's own scratch."""
    cache = resolve_roots().cache
    provider = Provider(
        row="text-index-local",
        config=LocalConfig(),
        cli=CLI,
        label="m",
        writable=(str(cache / "models"),),
    )

    granted = set(manual(Config(), provider).writable)

    assert granted == {str(cache / "models"), str(cache / "logs")}


# ------------------------------------------------------ provisioning, doctor --


async def test_status_counts_the_index_of_the_agent_that_asked(
    mount: MountProfile, tmp_path: Path
) -> None:
    """Each agent has its own index, so "is it built" has an answer per agent: the
    command is told the asking agent's state directory, as its tools are."""
    docs_tree(tmp_path, documents=3)
    ctx = await mount(*ROWS)
    builder, other = new_agent(ctx), new_agent(ctx)
    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=builder)

    commands = ctx.require(COMMANDS)
    built = await commands.dispatch("/text-index status", agent=builder, scope=builder.ctx)
    unbuilt = await commands.dispatch("/text-index status", agent=other, scope=other.ctx)

    assert built is not None and unbuilt is not None
    assert "documents: 3" in built
    assert str(_indexes(ctx, builder)) in built
    assert "not built" in unbuilt


async def test_the_rows_report_themselves_to_doctor(mount: MountProfile, tmp_path: Path) -> None:
    ctx = await mount(*ROWS)

    rows = report_section(ctx, "Text index")

    assert rows["tools"] == "text_index, text_search (through UTCP)"
    assert rows["embedder"] == "stub-hashing:96"
    # No setting, and nothing a session can ask without starting the command.
    assert rows["limit"] == "asked of the embedder when it embeds; `/text-index status` shows it"
    assert "scratch" in rows["index"]
    assert rows["details"] == "/text-index status"


# -------------------------------------------------- the shipped embedder's row --


async def test_the_local_row_names_the_model_and_grants_its_weights(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`text-index-local`, as the bundle mounts it. The session's process decides
    nothing about the model but its name: the row hands its config across and
    grants the one shared directory the model's weights are in. Discovery mounts
    the real engine rows and loads no model, so this downloads nothing."""
    ctx = await mount(
        *ROWS[:2],
        {
            "id": "text-index-local",
            "name": "text-index-local",
            "config": {"model": "nomic-ai/nomic-embed-text-v1.5", "query_prefix": "search_query: "},
        },
    )
    seam = ctx.require(TEXT_INDEX)

    assert seam.provider is not None and seam.manual is not None
    assert seam.provider.label == "nomic-ai/nomic-embed-text-v1.5"
    assert json.loads(seam.manual.env[CONFIG_ENV])["text-index-local"] == {
        "model": "nomic-ai/nomic-embed-text-v1.5",
        "queryPrefix": "search_query: ",
    }
    assert str(resolve_roots().cache / "models") in seam.manual.writable
    assert ctx.require(TOOLS).get("text_search", scope=DEPLOYMENT) is not None
    said = await ctx.require(COMMANDS).dispatch("/text-index status")
    assert said is not None
    assert "nomic-ai/nomic-embed-text-v1.5:search_query: |: not downloaded" in said


async def test_preload_refuses_the_mount_on_the_commands_refusal(
    mount: MountProfile, tmp_path: Path
) -> None:
    """The cookbook's rule, and the whole reason `preload` exists: an unattended
    run has nobody to type `/text-index install`, so it finds out at mount.

    The session's row asks the command, which is where the model loads. A model
    named by a path that is not one fails to load without touching the network,
    and the command's refusal is the session's.
    """
    with pytest.raises(MountRefusal, match="asked to preload and could not"):
        await mount(
            *ROWS[:2],
            {
                "id": "text-index-local",
                "name": "text-index-local",
                "config": {"model": str(tmp_path / "no-such-model"), "preload": True},
            },
        )


def test_the_sessions_process_never_loads_the_model(tmp_path: Path) -> None:
    """**The gate.** The model, torch and the index library belong to the command:
    several hundred megabytes a session that never searches does not hold."""
    probe = """
import json, sys
from pathlib import Path
import anyio
from ph.cordis import DEPLOYMENT, Context, Profile
from ph.cordis.loader import ProfileDocument
from ph.keys import TOOLS

ROWS = [
    {"id": "tools", "name": "tools"},
    {"id": "subprocess", "name": "subprocess-local"},
    {"id": "fs", "name": "fs-local"},
    {"id": "shell", "name": "shell-local"},
    {"id": "commands", "name": "commands"},
    {"id": "diagnostics", "name": "diagnostics"},
    {"id": "utcp", "name": "utcp"},
    {"id": "text-index", "name": "text-index"},
    {"id": "text-index-local", "name": "text-index-local"},
]

async def main():
    ctx = Context()
    await Profile.from_documents([ProfileDocument("probe", ROWS)]).mount(ctx, project=Path.cwd())
    names = [n for n in ctx.require(TOOLS).names(scope=DEPLOYMENT) if n.startswith("text_")]
    heavy = ("sentence_transformers", "torch", "transformers", "turbovec", "numpy")
    ours = ("ph_text_index.engine", "ph_text_index._store", "ph_text_index._embed")
    loaded = sorted(n for n in sys.modules if n.split(".")[0] in heavy or n in ours)
    print(json.dumps({"tools": names, "loaded": loaded}))
    await ctx.drain()
    await ctx.dispose()

anyio.run(main)
"""

    done = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    answer = json.loads(done.stdout.splitlines()[-1])
    assert answer["tools"] == ["text_index", "text_search"]
    assert answer["loaded"] == []


# ---------------------------------------------------- the bundle, and the RLM ----


def test_the_bundle_is_discoverable_without_importing_it() -> None:
    """The entry point is the whole coupling between a profile and this package."""
    from ph.bundles import installed_bundles, resolve_bundle
    from ph_text_index import BUNDLE

    assert "text-index" in installed_bundles()
    assert resolve_bundle("text-index") == BUNDLE


def test_every_row_in_the_bundle_names_a_resolvable_plugin() -> None:
    from ph.cordis import Profile
    from ph.cordis.loader import resolve_plugin
    from ph_text_index import BUNDLE

    rows = Profile.from_paths([BUNDLE]).dump()
    assert rows, "the bundle declares no rows"
    for row in rows:
        name = as_str(row["name"])
        assert resolve_plugin(name) is not None, name


def test_the_bundle_ships_the_provider_beside_the_seam() -> None:
    """**The subtlety that would otherwise be a silence.**

    The manual is registered by an embedder's claim, so a bundle carrying only
    `text-index` would mount a service and advertise nothing — and nothing would
    fail. Both rows, or the bundle means nothing.
    """
    from ph.cordis import Profile
    from ph_text_index import BUNDLE

    assert [row.id for row in Profile.from_paths([BUNDLE]).enabled_rows()] == [
        "text-index",
        "text-index-local",
    ]


async def test_the_rlm_indexed_profile_layers_this_bundle() -> None:
    from ph_app.profiles import available_profiles, resolve_profile
    from ph_text_index import BUNDLE

    assert "rlm-indexed" in available_profiles()
    assert BUNDLE in resolve_profile("rlm-indexed")


# -------------------------------------------------------------------- the host --


ENTERED = PreStepDecision(kind="enter")


async def _entered() -> PreStepDecision:
    """The rest of the `agent/pre-step` chain, for a test that calls one listener."""
    return ENTERED


def _step(agent: Any) -> PreStepRequest:  # noqa: ANN401
    """The request a driver makes before `agent`'s first step."""
    return PreStepRequest(agent=agent, session=agent.session, messages=(), turn=1, step=1)


def _loads(record: Path) -> int:
    """How many processes mounted the stub embedder — "loaded the model"."""
    return len(record.read_text(encoding="utf-8").splitlines())


def _recording(tmp_path: Path, **config: object) -> tuple[Path, dict[str, Any]]:
    """The stub embedder's row, configured to write a line for every process
    that mounts it."""
    record = tmp_path.parent / f"{tmp_path.name}-loads"
    return record, {**ROWS[2], "config": {"record": str(record), **config}}


async def test_a_session_keeps_one_host_and_every_call_is_answered_by_it(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** A second `text_search` does not load the model again.

    Two processes mount the embedder: the one that printed the manual, and the
    host. Three tool calls add none — where, with no host, each is a process
    that loads what the last one loaded.
    """
    docs_tree(tmp_path)
    record, embedder = _recording(tmp_path)
    ctx = await mount(*ROWS[:2], embedder)
    agent = new_agent(ctx)

    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=agent)
    first = await run_tool(ctx, "text_search", {"query": "containment tier"}, agent=agent)
    again = await run_tool(ctx, "text_search", {"query": "containment tier"}, agent=agent)

    assert [hit["path"] for hit in first.value["hits"]][:1] == ["docs/w.md"]
    assert again.value == first.value
    assert _loads(record) == 2
    assert [name for name, _ in ctx.require(UTCP).hosts] == ["ph_text_index"]


async def test_a_child_that_shares_its_parents_tree_shares_its_index_and_its_host(
    mount: MountProfile, tmp_path: Path
) -> None:
    """One host a tree, not one an agent. A child with no worktree of its own
    reads its parent's files, so it asks its parent's index — and a fan-out of
    such children is one index and one model, not one of each a child."""
    docs_tree(tmp_path, documents=3)
    record, embedder = _recording(tmp_path)
    ctx = await mount(*ROWS[:2], embedder)
    workspace = ctx.require(WORKSPACE)
    parent = new_agent(ctx)
    child = ctx.require(AGENTS).create(admitted_child(ctx, session_of(parent), "c1"), FAKE_OPTIONS)
    for agent in (parent, child):
        log = agent.session
        assert log is not None
        held = await workspace.acquire(session_id=log.id, agent_id=agent.id, base=tmp_path)
        assert held.root == tmp_path, "the two do not share a tree"

    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=parent)
    found = await run_tool(ctx, "text_search", {"query": "containment tier", "k": 1}, agent=child)

    utcp = ctx.require(UTCP)
    assert utcp.owner_of(child) is parent
    assert [hit["path"] for hit in found.value["hits"]] == ["docs/w.md"]
    assert list(utcp.hosts) == [("ph_text_index", parent.id)]
    assert _loads(record) == 2
    assert not _indexes(ctx, child).exists(), "the child built an index of its own"


async def test_a_root_agents_first_step_starts_the_index_and_it_is_told_when_it_ends(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The gate.** The index is being built before anything asks for it; the
    footer reading moves while it is; and when it ends the agent is told, with a
    note that waits in its inbox for its next step and starts no turn.

    No tool is called here. The row's `agent/pre-step` listener starts the host
    with the index tool to run, and the host does the rest in the background.
    """
    docs_tree(tmp_path, documents=12)
    _, embedder = _recording(tmp_path, delay=0.15)
    ctx = await mount(*ROWS[:2], embedder)
    utcp = ctx.require(UTCP)
    agent = new_agent(ctx)
    readings: list[str] = []

    assert await utcp.eager(_step(agent), _entered) is ENTERED

    def finished() -> bool:
        reading = utcp.reading(session_of(agent))
        if reading is not None and reading.text not in readings:
            readings.append(reading.text)
        return bool(agent.inbox.next_step)

    await settled(finished, "the completion note", seconds=60)

    [note] = agent.inbox.next_step
    assert "The text index is finished: 12 documents" in text_of(note.content)
    # With what the tool's result would have said to a caller that waited for it.
    assert "Indexed 12 documents: 12 passages added, 0 removed." in text_of(note.content)
    assert "The index now holds 12 passages from 12 documents." in text_of(note.content)
    # Once for the tree: a later step neither starts another host nor indexes
    # the tree again for nobody.
    [hosted] = utcp.hosts.values()
    await utcp.eager(_step(agent), _entered)
    assert list(utcp.hosts.values()) == [hosted]
    assert ("ph_text_index", agent.id) in utcp.warmed
    assert len(readings) >= 2, f"the reading did not move: {readings}"
    assert all(one.startswith("text index ") and one.endswith("/12 documents") for one in readings)
    assert utcp.reading(session_of(agent)) is None, "a finished index still has a reading"
    searched = await run_tool(ctx, "text_search", {"query": "containment tier"}, agent=agent)
    assert searched.value["searched"] == 12


async def test_on_mount_false_leaves_indexing_to_the_tool_and_a_child_waits_for_a_call(
    mount: MountProfile, tmp_path: Path
) -> None:
    """`index.on_mount: false`, and the other half of the rule for free: a
    sub-agent's step starts nothing either way, so a child in a throwaway
    worktree that never searches never holds a model."""
    docs_tree(tmp_path)
    off = {**ROWS[1], "config": {"index": {"onMount": False}}}
    ctx = await mount(ROWS[0], off, ROWS[2])
    utcp = ctx.require(UTCP)
    agent = new_agent(ctx)

    await utcp.eager(_step(agent), _entered)
    assert not utcp.hosts

    eager = await mount(*ROWS)
    parent = new_agent(eager)
    child = eager.require(AGENTS).create(
        admitted_child(eager, session_of(parent), "c1"), FAKE_OPTIONS
    )
    await eager.require(UTCP).eager(_step(child), _entered)
    assert not eager.require(UTCP).hosts


async def test_unmounting_the_embedder_ends_its_host(mount: MountProfile, tmp_path: Path) -> None:
    """A host is a job of the agent it serves, and it is the manual's too: the
    tools go with the embedder's row, and so does the process that answers them."""
    docs_tree(tmp_path)
    ctx = await mount(*ROWS)
    utcp = ctx.require(UTCP)
    await run_tool(ctx, "text_search", {"query": "x"}, agent=new_agent(ctx))
    [hosted] = utcp.hosts.values()
    assert hosted.socket.exists()

    await ctx.require(MOUNT).forks[STUB_ROW].dispose()
    await settle_background(ctx)

    assert not utcp.hosts
    assert not hosted.running
    assert not hosted.socket.exists()


async def test_a_child_builds_its_own_index_only_when_it_has_a_tree_it_writes_for_keeps(
    mount: MountProfile, tmp_path: Path
) -> None:
    """**The rule a read-only fan-out rests on.** A child granted only to read the
    project asks its parent's index: it has no tree of its own — its directory
    is an empty one beside a repository it cannot write — so it embeds nothing,
    holds no model, and is told which tree the paths it is handed belong to. A
    child with a worktree it writes for keeps has files that diverge, so it is
    the owner of that tree and builds its own.

    Sabotage: decide by the root alone, and the reader indexes its own empty
    directory into its parent's index and sweeps every document out as gone.
    """
    record, embedder = _recording(tmp_path)
    ctx = await mount(*ROWS[:2], embedder)
    workspace, utcp = ctx.require(WORKSPACE), ctx.require(UTCP)
    workspace.register_provider(
        StubWorkspaceProvider(root=tmp_path / "trees", kinds=("worktree", "readonly-scratch"))
    )
    parent = new_agent(ctx)
    reader = ctx.require(AGENTS).create(admitted_child(ctx, session_of(parent), "r"), FAKE_OPTIONS)
    writer = ctx.require(AGENTS).create(admitted_child(ctx, session_of(parent), "w"), FAKE_OPTIONS)
    held = {}
    asked: tuple[tuple[Any, WorkspaceAccess], ...] = (
        (parent, "write"),
        (reader, "read"),
        (writer, "write"),
    )
    for agent, access in asked:
        log = agent.session
        assert log is not None
        held[agent.id] = await workspace.acquire(
            session_id=log.id, agent_id=agent.id, base=tmp_path, access=access, session=log
        )
    docs_tree(held[parent.id].root, documents=3)
    docs_tree(held[writer.id].root, documents=5)
    assert held[reader.id].kind == "readonly-scratch"
    assert held[writer.id].kind == "worktree"

    await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=parent)
    read = await run_tool(ctx, "text_search", {"query": "containment tier", "k": 1}, agent=reader)
    swept = await run_tool(ctx, "text_index", {}, agent=reader)
    wrote = await run_tool(ctx, "text_index", {"paths": ["docs"]}, agent=writer)

    assert utcp.owner_of(reader) is parent and utcp.owner_of(writer) is writer
    assert [hit["path"] for hit in read.value["hits"]] == ["docs/w.md"]
    assert f"[Paths here are relative to {held[parent.id].root}," in text_of(read.content)
    assert swept.value["total_documents"] == 3, "the reader swept its parent's index"
    assert wrote.value["total_documents"] == 5
    assert "[Paths here" not in text_of(wrote.content)
    assert sorted(owner for _, owner in utcp.hosts) == sorted([parent.id, writer.id])
    assert not _indexes(ctx, reader).exists()
    assert _loads(record) == 3, "the manual, the parent's host and the writer's"
