"""P6-05 — `readonly-scratch`: the one kind whose repository is genuinely unwritable.

Gate: *a read child cannot write the repo; the degradation is reported in doctor.*

**The kind was vocabulary-complete and implementation-absent.** `readonly-scratch`
has been in `WorkspaceKind` since P6-20, exhaustively classified by all four kind
predicates, named in §4.8's tier table as what `sandbox` hands out — and nothing
produced it. So the top rung of the ladder existed as a row in a table and as a
`match` arm, which is the shape a claim takes when nobody can reach it.

Two halves, tested where each can be:

* the **vocabulary** half runs everywhere — which kind is handed out, what
  `repo_writable` says, where the root is, and what policy `workspace_policy`
  derives from it. That policy is the whole mechanism: the repository is
  unwritable because it is not in the writable set, not because this module
  enforces anything.
* the **enforcement** half needs a kernel that will refuse, and skips where there
  is none. It is the same shape as `test_sandbox_local.py`'s gate and deliberately
  a bare `ctx.subprocess` spawn: an `open()` that never consults a cwd is what the
  rung's sentence is actually about.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from ph.seams.sandbox import writable_paths
from ph.seams.subprocess import SubprocessSpawnSpec, scrub_env
from ph.seams.workspace import project_access, workspace_policy
from ph.seams.workspace_scratch import ReadonlyScratchProvider
from ph.testing import StubSandboxProvider, report_section

pytestmark = pytest.mark.anyio

ROW = {"insert": [{"id": "workspace-readonly-scratch", "name": "workspace-readonly-scratch"}]}
SANDBOX_ROW = {"insert": [{"id": "sandbox-local", "name": "sandbox-local"}]}
SECTION = "Read-only scratch workspaces"


async def _with_backend(ctx: Any, enforcement: str) -> None:
    """Register a backend the way a profile that layers one *after* this row does."""
    ctx.sandbox.register_provider(StubSandboxProvider(enforcement=enforcement))
    await ctx.serial("profile/mounted")


async def _acquire(tmp_path: Path, access: str = "write") -> Any:
    """One workspace straight from the provider, for the vocabulary half."""
    scratch = tmp_path / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    return await ReadonlyScratchProvider().acquire(
        session_id="s", agent_id="a", base=tmp_path / "repo", scratch=scratch, access=access
    )


async def _enforcing(mount: Any, tmp_path: Path) -> tuple[Any, Any]:
    """A mounted rung over a host whose kernel enforces, and a workspace — or a skip.

    Through `ctx.workspace.acquire` rather than the provider directly, so what the
    kernel half exercises is the workspace the *seam* produces: `_scratch_for` owns
    the layout, and a test that built its own would be confining a tree production
    never hands out.
    """
    ctx = await mount(SANDBOX_ROW, ROW)
    if ctx.workspace.provider is None:
        # The reason is already in `sandbox-local`'s own section; re-probing to
        # build a skip message costs a second bwrap spawn for a string the row
        # computed at mount.
        pytest.skip(f"no enforcing sandbox backend: {report_section(ctx, 'Local confinement')}")
    workspace = await ctx.workspace.acquire(session_id="s", agent_id="a", base=tmp_path / "repo")
    return ctx, workspace


async def _confined_write(ctx: Any, workspace: Any, target: Path) -> tuple[int, str]:
    """A raw `open()` under the workspace's own confinement.

    `scrub_env(extra=workspace.env)` matches `test_containment_ladder.py:46` — the
    workspace carries `redirection_env`, and a spawn that dropped it would confine
    a process the harness never runs.
    """
    argv = ctx.sandbox.confine(
        (sys.executable, "-c", f"open({str(target)!r}, 'w').write('agent')"),
        workspace_policy(workspace),
    ).argv
    outcome = await ctx.subprocess.run(
        SubprocessSpawnSpec(argv=argv, cwd=workspace.root, env=scrub_env(extra=workspace.env))
    )
    return outcome.exit_code, outcome.stdout + outcome.stderr


# ------------------------------------------------------------- the vocabulary --


async def test_the_repository_is_not_in_the_writable_set(tmp_path: Path) -> None:
    """**The gate, and the whole mechanism.**

    Nothing in this module refuses a write. The repository is unwritable because
    the policy `workspace_policy` derives from this workspace does not name it —
    the writable set is the agent's root and its scratch, both inside scratch —
    and `ctx.shell` hands that policy to a backend that does the refusing.
    """
    workspace = await _acquire(tmp_path)
    policy = workspace_policy(workspace)

    writable = writable_paths(policy)
    assert str(tmp_path / "repo") not in writable
    assert all(Path(path).is_relative_to(workspace.scratch) for path in writable), writable


async def test_asking_for_write_does_not_widen_the_tier(tmp_path: Path) -> None:
    """**A caller asking `write` gets the kind anyway, and is told.**

    The ordinary lifecycle asks for `write` on the person's own agent, so a tier
    that refused the request would be unusable at the rung it exists for; one that
    quietly widened would be a rung nobody could rely on. `repo_writable` carries
    the difference, which is exactly the field's job (§12 Q10).
    """
    for access in ("read", "write"):
        workspace = await _acquire(tmp_path, access)
        assert workspace.kind == "readonly-scratch", access
        assert workspace.repo_writable is False, access
        assert project_access(workspace.kind) == "read", access


async def test_the_root_is_inside_scratch_and_exists(tmp_path: Path) -> None:
    """ "Rooted at scratch" is the row's own phrase for the mechanism, and the
    directory has to be there before an agent is pointed at it."""
    workspace = await _acquire(tmp_path)

    assert workspace.root.is_relative_to(workspace.scratch)
    assert workspace.root.is_dir()
    assert workspace.env["TMPDIR"] == str(workspace.scratch), "the toolchain is redirected too"


# -------------------------------------------------------------- registration --


async def test_the_rung_is_not_claimed_without_a_backend(mount: Any) -> None:
    """**Gated on enforcement, not on acquisition**, and the direction matters.

    `register_provider` is `claim_slot`: exclusive. A row that claimed the slot
    and then declined every acquire would fall back to `shared` — *advisory* — so
    a deployment that asked for the strongest rung would silently get the weakest.
    Worse, the claim this kind makes is `repo_writable=False`, which without a
    backend behind it is E1's failure in the field E1 is about.
    """
    ctx = await mount(ROW)

    assert ctx.workspace.provider is None, "no backend, no rung"
    assert "no sandbox backend" in report_section(ctx, SECTION)["declined"]


async def test_a_partial_backend_is_declined_by_default(mount: Any) -> None:
    """A backend enforcing part of a policy cannot be known to make a repository
    unwritable, so the default is to decline and say which backend and why."""
    ctx = await mount(ROW)
    await _with_backend(ctx, "partial")

    assert ctx.workspace.provider is None
    assert "partial" in report_section(ctx, SECTION)["declined"]


# --------------------------------------------------------------- enforcement --


async def test_a_read_child_really_cannot_write_the_repo(mount: Any, tmp_path: Path) -> None:
    """**The gate against a kernel that will refuse**, or a skip.

    A bare `ctx.subprocess` spawn rather than `ctx.shell`, for
    `test_containment_ladder.py`'s reason: a shell-based write would pass or fail
    on what the profile mounted and say nothing about the rung. What the sentence
    is about is a raw `open()` that never consults a cwd — refused here, where
    every rung below reports it as an escape.
    """
    ctx, workspace = await _enforcing(mount, tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    target = repo / "untouched.txt"
    target.write_text("host", encoding="utf-8")

    code, output = await _confined_write(ctx, workspace, target)

    assert code != 0, output
    assert target.read_text(encoding="utf-8") == "host", "the repository was written"


async def test_the_agents_own_scratch_still_takes_writes(mount: Any, tmp_path: Path) -> None:
    """The half that stops "refuses everything" reading as success.

    A rung that refused the agent's own root would be unusable, and the first
    person to meet it would turn the tier off — the same pairing
    `test_sandbox_local.py` makes for the same reason.
    """
    ctx, workspace = await _enforcing(mount, tmp_path)

    code, output = await _confined_write(ctx, workspace, workspace.root / "produced.txt")

    assert code == 0, output
    assert (workspace.root / "produced.txt").read_text(encoding="utf-8") == "agent"


async def test_the_sandbox_rung_is_now_reachable_end_to_end(mount: Any) -> None:
    """**What P6-05 actually closes**, asserted through the seam a person reads.

    `effective_tier` could never return `sandbox`: no `WorkspaceProvider` declared
    that tier, so the top rung of §4.8's table was a row nothing could produce and
    `containment.strict` with `tier: sandbox` had nothing to satisfy it on the
    workspace side. Both halves are checked here because the tier and the kind are
    separate claims — one is what `ph doctor` prints per role, the other is what
    the agent is actually handed.
    """
    ctx = await mount(ROW)
    await _with_backend(ctx, "full")

    assert ctx.workspace.effective_tier(child=False) == "sandbox"
    assert ctx.workspace.effective_tier(child=True) == "sandbox"

    acquired = await ctx.workspace.acquire(session_id="s", agent_id="a", base=Path("/repo"))
    assert acquired.kind == "readonly-scratch"
    assert acquired.repo_writable is False
