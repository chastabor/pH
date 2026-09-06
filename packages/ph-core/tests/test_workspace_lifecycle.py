"""P4-08 — who acquires a workspace, and what then resolves against it (D21, E2).

The seam by itself is inert; this row is what makes a tier bite. Two properties
carry the whole thing:

**Every agent has a workspace before it does anything**, taken once and released
with the agent's own scope. It is lazy, at the first `agent/pre-step`, because
`agent/created` is an `emit` — a listener there could not hold the agent up
while `git worktree add` ran, and the first tool call would race the checkout.

**`ctx.fs` resolves per agent, not per process.** That is what makes two children
of one session write two different trees, and it is the difference between the
`worktree` tier isolating a fan-out and merely renaming the shared one. What it
does *not* do is bound an absolute path — `resolve` passes those through on
purpose, and only the `sandbox` tier refuses them (E13).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ph.seams.workspace import PROJECT_PROVISION_FILE, discover_provisioning
from ph.testing import FAKE_OPTIONS, StubWorkspaceProvider, run_tool

pytestmark = pytest.mark.anyio


def _tier(tmp_path: Path) -> StubWorkspaceProvider:
    """A tier giving each agent its own directory, and an env to prove it rode
    along. The checkout itself is `test_workspace_git.py`'s claim."""
    return StubWorkspaceProvider(
        root=tmp_path / "trees", env={"PH_TEST_REDIRECT": str(tmp_path / "scratch")}
    )


async def _run(ctx: Any, session_id: str = "s") -> Any:
    """One agent, one prompt — the least that reaches `agent/pre-step`."""
    session = ctx.sessions.create(session_id)
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    await agent.prompt("hello")
    return agent


# ------------------------------------------------------------- acquisition --


async def test_an_agent_holds_a_workspace_by_the_time_it_steps(mount: Any) -> None:
    """The row's whole job, at the default tier: nothing changes except that the
    question "where does this agent write" now has an answer."""
    ctx = await mount()

    agent = await _run(ctx)

    workspace = ctx.workspace.of(agent.id)
    assert workspace is not None
    assert workspace.kind == "shared"
    # The process's own directory, which is what `shared` means — so a profile
    # that names no tier behaves exactly as it did before this row existed.
    assert workspace.root == ctx.fs.root


async def test_a_child_with_no_workspace_refuses_instead_of_inventing_one(mount: Any) -> None:
    """§6.5, at the one place it was reachable by omission.

    A child's base and `access` are its parent's decision and arrive with the spawn.
    The lazy acquire below has neither, so the only thing it could do for a child is
    substitute the *root's*: the process's own directory as `base`, and whatever the
    profile configured for the person's own agent as `access` — `write` by default.
    A child admitted as `read` would come back holding a writable checkout, and
    every event about it would report that honestly while being wrong about what was
    asked for.

    So the omission fails loudly. It is a **missing call in a spawn path**, and the
    loud version costs a failed turn naming the child; the quiet version costs a
    child holding more than its admission recorded, which nothing downstream can
    detect — `workspace/acquired` says what it got, not what it should have got.
    """
    ctx = await mount()
    orphan = ctx.sessions.create("child", meta={"origin": "subagent", "parentSession": "root"})
    agent = ctx.agents.create(orphan, FAKE_OPTIONS)

    await agent.prompt("hello")

    # Loud *and* durable: the turn ends in error and the log says which, which is
    # what a raise from `agent/pre-step` becomes. A bare exception would reach this
    # record as `UNKNOWN`, so the code is the half worth pinning.
    (ended,) = [one for one in orphan.events if one.type == "turn/end"]
    error = ended.data["reason"]["error"]
    assert ended.data["reason"]["kind"] == "error"
    assert error["code"] == "CHILD_WORKSPACE_MISSING"
    assert agent.id in str(error["message"])
    assert ctx.workspace.of(agent.id) is None, "the refusal must not leave a workspace behind"


async def test_a_child_whose_parent_did_acquire_steps_normally(mount: Any, tmp_path: Path) -> None:
    """The other half, and the one that must not regress: the refusal is about an
    *absent* workspace, never about being a child.

    A spawn acquires on the child's behalf before it runs — with the base and access
    the parent chose — and this row must then find that one rather than refuse it or
    overwrite it.
    """
    ctx = await mount()
    child = ctx.sessions.create("child", meta={"origin": "subagent", "parentSession": "root"})
    agent = ctx.agents.create(child, FAKE_OPTIONS)
    # What a spawn does: the parent decides, and it decided `read`.
    granted = await ctx.workspace.acquire(
        session_id=child.id, agent_id=agent.id, base=tmp_path, access="read", session=child
    )

    await agent.prompt("hello")

    assert ctx.workspace.of(agent.id) is granted


async def test_the_workspace_is_taken_once_not_once_per_turn(mount: Any) -> None:
    """`git worktree add` per turn would be both slow and wrong — the second
    call would find the first's tree and the branch already taken."""
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await agent.prompt("first")
    await agent.prompt("second")

    acquired = [event for event in session.events if event.type == "workspace/acquired"]
    assert len(acquired) == 1


async def test_an_acquire_that_names_a_live_agent_unwinds_with_it(mount: Any) -> None:
    """P4-16's note, closed: `agent_id` already says whose workspace this is.

    A hand-rolled `acquire` with no `scope=` handed the seam a checkout that
    outlived the agent by the whole process — the footgun the containment-ladder
    module had to remember by hand. For an agent the registry knows, the agent's
    own scope is the owner whether or not the caller said so.
    """
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)

    await ctx.workspace.acquire(session_id=session.id, agent_id=agent.id, base=ctx.fs.root)
    assert ctx.workspace.of(agent.id) is not None

    await ctx.agents.dispose(agent.id)

    assert ctx.workspace.of(agent.id) is None


async def test_disposing_the_agent_releases_its_workspace(mount: Any) -> None:
    """I2, end to end: the agent's scope owns the checkout, so an agent that
    goes away does not leave one behind for a reconciler to find."""
    ctx = await mount()
    session = ctx.sessions.create("s")
    agent = ctx.agents.create(session, FAKE_OPTIONS)
    await agent.prompt("hello")

    await ctx.agents.dispose(agent.id)

    assert ctx.workspace.of(agent.id) is None
    assert [event.type for event in session.events if event.type.startswith("workspace/")] == [
        "workspace/acquired",
        "workspace/disposed",
    ]


# ------------------------------------------------------------- resolution --


async def test_relative_paths_resolve_against_the_agents_own_root(
    mount: Any, tmp_path: Path
) -> None:
    """The tier biting: a relative write lands in this agent's tree.

    A *relative* one. `resolve` passes an absolute path through untouched, which
    is why the tier table calls this collision isolation rather than
    confinement — and why a test asserting otherwise would be the regression
    §12 Q10 exists to prevent.
    """
    ctx = await mount()
    ctx.workspace.register_provider(_tier(tmp_path))
    agent = await _run(ctx)
    root = ctx.workspace.of(agent.id).root

    assert ctx.fs.resolve("notes.txt", agent=agent) == root / "notes.txt"
    assert ctx.fs.root_for(agent) == root
    # Absolute paths are the tier's stated limit, not an oversight.
    assert ctx.fs.resolve("/etc/hosts", agent=agent) == Path("/etc/hosts")


async def test_two_agents_resolve_to_two_different_trees(mount: Any, tmp_path: Path) -> None:
    """E2 at the layer that makes it true.

    A fan-out is only isolated if `edit("x.py")` means a different file for each
    child. With one process-wide root it would mean the same file, and the tier
    would be a rename of the hazard rather than a fix for it.
    """
    ctx = await mount()
    ctx.workspace.register_provider(_tier(tmp_path))

    one = await _run(ctx, "s1")
    two = await _run(ctx, "s2")

    assert ctx.fs.resolve("x.py", agent=one) != ctx.fs.resolve("x.py", agent=two)


async def test_an_agent_with_no_workspace_still_reads_the_process_root(
    mount: Any,
) -> None:
    """`ph doctor`, a CLI probe, a test — callers with no agent at all are real
    and must not be the ones that raise."""
    ctx = await mount()

    assert ctx.fs.root_for(None) == ctx.fs.root
    assert ctx.fs.resolve("x.py") == ctx.fs.root / "x.py"


async def test_a_resolver_that_breaks_falls_back_rather_than_failing_the_read(
    mount: Any,
) -> None:
    """An agent whose workspace lookup broke still has to be able to read a
    file: the wrong-but-working directory is a better failure than a traceback
    out of `read`, and the log carries the reason either way."""
    ctx = await mount()

    class _Exploding:
        id = "boom"

    ctx.fs._rebase = _raise  # type: ignore[assignment]

    assert ctx.fs.root_for(_Exploding()) == ctx.fs.root


def _raise(_agent: Any) -> Path:
    raise RuntimeError("the workspace seam is gone")


# ------------------------------------------------------------------ command --


async def test_bash_runs_in_the_agents_workspace(mount: Any, tmp_path: Path) -> None:
    """The other half of "cwd resolves to `workspace.root`" (§4.8).

    A shell command is the shortest path from a model to a relative-path write,
    so a `bash` that ran in the process's directory would leave the tier bounding
    the tools and nothing else.
    """
    ctx = await mount()
    ctx.workspace.register_provider(_tier(tmp_path))
    agent = await _run(ctx)
    root = ctx.workspace.of(agent.id).root

    result = await run_tool(ctx, "bash", {"command": "pwd && echo $PH_TEST_REDIRECT"}, agent=agent)

    stdout = result.value["stdout"]
    assert str(root) in stdout
    # And the workspace's own environment rides the call, which is what keeps a
    # test run's caches out of the tree (E12).
    assert str(tmp_path / "scratch") in stdout


async def test_the_model_is_told_when_its_command_said_more_than_was_kept(
    mount: Any, tmp_path: Path
) -> None:
    """P7-13's model-facing half: a truncated result must not read as a whole one.

    `tool-bash` had no cap at all and put a child's whole output straight into a
    `tool/result` — the more serious half of the row, because the reader is the
    model. It inherits the seam's bound now without asking; what it still owes is
    the *fact*, in the text the model reads, or a prefix taken for the whole
    output sends it hunting a failure further down that it cannot see.

    The sentence is `truncation_marker`'s, which the RLM kernel and the guest
    runner already say for this event — "a reader comparing a transcript to a log
    must not find two different sentences for the same event" (D4).

    Sabotage: drop the marker from `_render` and the model gets 4 KiB of output
    with nothing saying there was more.
    """
    ctx = await mount({"id": "subprocess", "config": {"maxOutputBytes": 4096}})
    ctx.workspace.register_provider(_tier(tmp_path))
    agent = await _run(ctx)

    result = await run_tool(
        ctx, "bash", {"command": f"printf 'x%.0s' $(seq {4096 + 100})"}, agent=agent
    )

    assert result.value["exit_code"] == 0, "the child finished; nothing blocked on a full pipe"
    assert len(result.value["stdout"]) == 4096
    assert result.value["dropped"] == 100
    assert "100 bytes dropped, cap 4096 bytes" in result.content[0].text


# -------------------------------------------------------------- provisioning --


def test_a_project_states_its_own_materials(tmp_path: Path) -> None:
    """A repository is the only party that knows it keeps dependencies in
    `vendor/` rather than `node_modules/`, so it gets to say so — as **data**.

    That is the whole trade the `command` hook was refused for: a list of paths
    cannot execute, so cloning a repository and starting pH runs nothing that
    repository authored. `wtp` accepts shell in the equivalent file on the
    grounds that it is "controlled by developer"; that assumption is not
    available here.
    """
    project = tmp_path / "repo" / "packages" / "api"
    project.mkdir(parents=True)
    (tmp_path / "repo" / PROJECT_PROVISION_FILE).write_text(
        "provision:\n  - source: .env\n  - source: vendor\n    mode: hardlink\n",
        encoding="utf-8",
    )

    entries = discover_provisioning(project)

    assert [(e.source, e.mode) for e in entries] == [(".env", "copy"), ("vendor", "hardlink")]


def test_the_nearest_file_wins(tmp_path: Path) -> None:
    """`memory-agents-md`' rule, for the same reason: the file beside the code
    is the one that knows what the code needs, and a monorepo root should not
    override a package that states its own."""
    root, package = tmp_path / "repo", tmp_path / "repo" / "packages" / "api"
    package.mkdir(parents=True)
    (root / PROJECT_PROVISION_FILE).write_text(
        "provision: [{source: root-thing}]", encoding="utf-8"
    )
    (package / PROJECT_PROVISION_FILE).write_text(
        "provision: [{source: package-thing}]", encoding="utf-8"
    )

    assert [e.source for e in discover_provisioning(package)] == ["package-thing"]


@pytest.mark.parametrize(
    "text",
    [
        "provision: [{source: .env, command: rm -rf /}]",
        "provision: [{mode: copy}]",
        "provision: not-a-list",
        ":::not yaml at all",
    ],
    ids=["a-key-that-would-execute", "no-source", "wrong-shape", "malformed"],
)
def test_a_project_file_that_does_not_parse_is_ignored(tmp_path: Path, text: str) -> None:
    """Every failure is a shrug, and the first case is the point: `extra="forbid"`
    means a `command:` key is not silently dropped but *rejects the file*, so a
    repository cannot smuggle one past a reader who assumed the model was
    data-only. Refusing to start would make an optional file load-bearing."""
    (tmp_path / PROJECT_PROVISION_FILE).write_text(text, encoding="utf-8")

    assert discover_provisioning(tmp_path) == []


async def test_the_project_list_is_guarded_like_any_other(mount: Any, tmp_path: Path) -> None:
    """The end of the claim the row is built on.

    A repository may state what its worktrees need and may **not** name anything
    outside its own tree, in either direction. The guards do not care where an
    entry came from — a discovered one is checked exactly like a profile's — and
    the refusal is per entry, so the workspace is still handed over.
    """
    ctx = await mount()
    base = tmp_path / "project"
    base.mkdir()
    (base / PROJECT_PROVISION_FILE).write_text(
        "provision: [{source: ../../etc/passwd, dest: stolen}]", encoding="utf-8"
    )
    ctx.workspace.provision(discover_provisioning(base))
    ctx.workspace.register_provider(_tier(tmp_path))

    workspace = await ctx.workspace.acquire(session_id="s", agent_id="a1", base=base)

    assert not (workspace.root / "stolen").exists()
    assert len(workspace.provision_failures) == 1
