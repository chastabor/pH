"""`ctx.workspace` — where an agent's writes land, and how honestly that is stated.

The seam the containment ladder hangs off (D21, §4.8). Its consumer is the
**agent lifecycle**, not a tool: an agent acquires a workspace, and `ctx.fs`'s
root and `ctx.subprocess`'s cwd resolve to `workspace.root`, which is what makes
a tier bound *authored* code rather than merely observe it. A tool the model
calls could never do that job — a permission row can deny `edit`, but it cannot
deny `open(path, "w")`, because a deny-list needs a registered name to match.

**`repo_writable` is a claim, and the seam refuses to overstate it.** The
`worktree` tier gives an agent its own checkout: collisions are isolated and a
run is revertible, but an absolute-path `open()` never consults a cwd, so that
tier bounds nothing about `/etc/passwd`. Only `sandbox` can refuse that write, at
the kernel. So a caller asking for `access="read"` gets the strongest kind the
mounted tier can actually provide, and `repo_writable` records **which guarantee
was obtained** rather than which was requested. `False` means a tier is enforcing
it. Any wording here, in `ph doctor`, or in a config comment that blurs the two
is a defect (§12 Q10) — the whole point of the field is that a caller must not
have to infer a guarantee that is not there.

**There is always a workspace.** `acquire` never fails and never returns `None`:
an agent needs a working directory and somewhere to write, and "no workspace" is
not a state the lifecycle can be in. A provider that cannot serve a request —
`workspace-git-worktree` handed a `base` that is not a repository — *declines*,
and the seam falls back to `shared` with a logged notice. A hard failure there
would turn "this directory is not a git repo" into "pH will not start".

**A workspace is an effect of the scope that took it (I2).** `acquire` registers
its teardown through `ctx.effect`, which is the mechanism `Context.effect`'s own
docstring names a worktree for: a disposed agent scope unwinds the workspace
instead of leaving it held with nobody to release it. That is the *in-process*
half of cleanup; the `workspace/acquired`/`disposed` pair is the crash half,
reconciled at session open (§4.9), and a build that had only the second would
report a leak for every ordinary error path.

**`scratch` is always present and always writable**, on every kind and every
tier — and the *seam* creates it, so that guarantee has one implementation rather
than one per provider. It lives in pH's own state directory rather than in the
workspace, so it survives disposal as a session artifact: a child whose worktree
is discarded still leaves behind what it was asked to produce. A provider is
handed the path and may substitute its own — a sandbox tier has to put it
somewhere the container can reach — but never has to invent the layout.

@module ph.seams.workspace
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

import anyio

from ..cordis import Context, Disposer, maybe_await, plugin
from ..paths import default_home_path
from ..session import Session
from ..wire import WireModel
from ._registry import claim_slot

__all__ = [
    "ContainmentTier",
    "SharedWorkspaceProvider",
    "Workspace",
    "WorkspaceAccess",
    "WorkspaceKind",
    "WorkspaceProvider",
    "WorkspaceSeam",
    "apply",
]

log = logging.getLogger("ph.seams.workspace")

ContainmentTier: TypeAlias = Literal["advisory", "worktree", "sandbox"]
"""The ladder, named once.

`ph doctor` prints it (P4-12) and `containment.tier` selects it (P4-11); a
vocabulary spelled out at each of those sites is one where a typo reads as a
tier nobody has.
"""

WorkspaceKind: TypeAlias = Literal["shared", "worktree", "worktree-ephemeral", "readonly-scratch"]
"""What an agent actually got.

Four kinds rather than a boolean, because the interesting distinctions are not
writable-or-not. `shared` is today's behaviour — one checkout, no isolation.
`worktree` is that agent's own branch, merged back deliberately.
`worktree-ephemeral` is a full checkout the agent may write and whose writes
**reach nobody**: discarded on disposal, never merged. `readonly-scratch` is the
only one where the repository is genuinely unwritable, and only a sandbox backend
can deliver it.
"""

WorkspaceAccess: TypeAlias = Literal["write", "read"]
"""What the *caller* needs of `base` — a request, not a guarantee.

A research child asking for `read` should not be handed a checkout it might
mutate; but "read-only" is an enforcement claim, and the tier decides whether one
can be made. The answer comes back as `kind` and `repo_writable`.
"""


@dataclass(frozen=True, slots=True)
class Workspace:
    """One agent's working directory, and the truth about what it bounds.

    A value: what the provider decided, and nothing about who is holding it. The
    seam keeps the bookkeeping — which agent, which session, how to end it — so a
    provider cannot half-implement the lifecycle.
    """

    root: Path
    """The agent's cwd: `ctx.fs`'s root and `ctx.subprocess`'s default cwd."""
    scratch: Path
    """Always writable, on every kind and tier, and outside `root` on purpose —
    so it survives disposal even when the workspace itself is discarded. Handed
    to the provider already created; a provider substitutes only if its tier
    needs the path somewhere else."""
    kind: WorkspaceKind
    repo_writable: bool
    """Whether `root` can actually be written. **`False` only when a tier is
    enforcing it** — never as a statement of intent, and never inferred from
    `access`."""
    ref: str | None = None
    """The git branch, when the kind has one."""
    env: Mapping[str, str] = field(default_factory=dict)
    """Environment a runner should apply, for a kind that needs redirecting.

    Empty for `shared`. The read-only kinds point `TMPDIR`, `PYTEST_ADDOPTS` and
    friends inside `scratch`, because build tools write into the tree they are
    run against and an enforced-read-only repo would otherwise make "run the
    tests" impossible rather than merely safe. Best-effort by construction: a
    toolchain that insists on writing beside its sources will still fail, and the
    answer to that is `access="write"` for that agent, not a weaker tier.
    """
    release: Callable[[], Awaitable[bool]] | None = None
    """The provider's teardown, returning whether anything was **kept**.

    The answer is only knowable here: P4-08's policy is "keep dirty, remove
    clean, discard ephemeral even if dirty", so a field set at acquire time could
    not carry it and `kind` cannot be asked instead — a `worktree` is either. The
    seam records the answer on `workspace/disposed` so a reader can tell "nothing
    changed, so it was removed" from "these writes were thrown away by design".
    """


@runtime_checkable
class WorkspaceProvider(Protocol):
    """A tier's implementation. `None` declines, and declining is normal.

    Typed rather than duck-typed, and `tier` is a member rather than a `getattr`
    probe: a provider whose method drifted would otherwise fail at runtime inside
    the seam's `except`, be reported to the operator as `shared`, and take the
    containment with it silently.
    """

    tier: ContainmentTier

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None: ...


@dataclass(frozen=True, slots=True)
class SharedWorkspaceProvider:
    """`workspace-shared` — today's behaviour, and the floor under every tier.

    Returns `base` itself, so mounting the seam changes nothing: no checkout, no
    copy, no cost. It is also the fallback the seam keeps for a provider that
    declines, which is why it lives beside the seam rather than in a row of its
    own — "there is always a workspace" is a promise the seam makes, and a
    promise kept by a row a profile might not layer is not one.

    `access="read"` is honoured by *saying so*: the kind stays `shared` and
    `repo_writable` stays `True`, because nothing here enforces anything. A
    provider that returned `repo_writable=False` for a shared checkout would be
    telling the caller a lie it would then act on.
    """

    tier: ContainmentTier = "advisory"

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace:
        return Workspace(root=base, scratch=scratch, kind="shared", repo_writable=True)


@dataclass(slots=True)
class _Held:
    """One live workspace and the disposer that ends it.

    Paired here rather than on `Workspace` because the disposer is the *scope's*,
    not the provider's: a value handed to a caller should not carry the seam's
    bookkeeping, and `Job.release` next door is the same split.
    """

    workspace: Workspace
    dispose: Disposer | None = None


@dataclass(slots=True)
class WorkspaceSeam:
    """The service published as `ctx.workspace`."""

    ctx: Context
    shared: SharedWorkspaceProvider
    scratch_root: Path
    provider: WorkspaceProvider | None = None
    _held: dict[str, _Held] = field(default_factory=dict)
    """Live workspaces by agent id.

    Held here rather than on the agent because the *question* is asked by things
    that have an agent id and no agent object — the prompt's workspace line, and
    `ph doctor`'s per-agent report. Emptied by the effect disposer, so an entry
    surviving its agent is the same leak `workspace/acquired` without a
    `disposed` records durably.
    """

    def of(self, agent_id: str) -> Workspace | None:
        """The workspace this agent holds, if it has acquired one.

        `None` is a real answer and the common one today: nothing acquires until
        the agent lifecycle does (P4-08), and a caller must say what is true
        rather than describing a workspace nobody took.
        """
        held = self._held.get(agent_id)
        return None if held is None else held.workspace

    def register_provider(
        self, provider: WorkspaceProvider, *, scope: Context | None = None
    ) -> Disposer:
        """Claim the tier. One at a time; `shared` remains the fallback."""
        return claim_slot(scope or self.ctx, self, "provider", provider, label="workspace.provider")

    @property
    def tier(self) -> ContainmentTier:
        """The *effective* tier, which is what `ph doctor` must report.

        Effective, not configured: a `worktree` row over a directory that is not
        a repository declines on every acquire, and a doctor that read the config
        would tell an operator they have containment they do not have.
        """
        return "advisory" if self.provider is None else self.provider.tier

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        access: WorkspaceAccess = "write",
        session: Session | None = None,
        scope: Context | None = None,
    ) -> Workspace:
        """Take a workspace for one agent. Never fails, never returns `None`.

        `scope` is what bounds the workspace's life — the agent's own scope, for
        the lifecycle that lands in P4-08. Disposing it releases the workspace
        and writes the closing event, so an error path that never reaches an
        explicit `dispose` is not a leak.
        """
        scratch = await self._scratch_for(session_id, agent_id)
        workspace = None
        if self.provider is not None:
            try:
                workspace = await self.provider.acquire(
                    session_id=session_id,
                    agent_id=agent_id,
                    base=base,
                    scratch=scratch,
                    access=access,
                )
            except Exception:
                # A tier that broke is a tier that is not in force. Falling back
                # is the honest outcome and `workspace/acquired` will say
                # `shared`, which is what an operator needs to see.
                log.exception("ph.seams.workspace: provider failed; falling back to shared")
            else:
                if workspace is None:
                    log.info(
                        "ph.seams.workspace: provider declined %s for agent %s; using a shared "
                        "workspace, so this agent is not contained",
                        base,
                        agent_id,
                    )
        if workspace is None:
            workspace = await self.shared.acquire(
                session_id=session_id,
                agent_id=agent_id,
                base=base,
                scratch=scratch,
                access=access,
            )
        return await self._record(workspace, agent_id, session, scope)

    async def dispose(self, agent_id: str) -> None:
        """Release this agent's workspace early.

        "Early" because the scope owns it either way (I2); this is the same
        teardown reached deliberately rather than by unwinding, which is what
        `jobs.forget` and `subagents.delete` are to their own effects. Calling it
        twice is a no-op — the disposer deregisters itself.
        """
        held = self._held.get(agent_id)
        if held is None or held.dispose is None:
            return
        await maybe_await(held.dispose())

    async def _scratch_for(self, session_id: str, agent_id: str) -> Path:
        """Per session *and* per agent, created rather than merely named.

        Owned by the seam so the layout has one implementation: two children of
        one session writing notes into one directory is the collision this
        avoids, and a provider that got it wrong would break it for its tier
        alone with nothing checking.
        """
        scratch = self.scratch_root / session_id / agent_id
        await anyio.to_thread.run_sync(lambda: scratch.mkdir(parents=True, exist_ok=True))
        return scratch

    async def _record(
        self,
        workspace: Workspace,
        agent_id: str,
        session: Session | None,
        scope: Context | None,
    ) -> Workspace:
        """Register the teardown as an effect and log the acquisition.

        Both halves of the durable pair are written here rather than by the
        provider, because a pair only reconciles if one place owns both — a
        provider that forgot the second would leave every workspace looking
        leaked.
        """
        held = _Held(workspace=workspace)
        self._held[agent_id] = held

        def enter() -> Disposer:
            async def release() -> None:
                # Identity, not presence: an agent that re-acquired must not have
                # its live workspace evicted by the previous handle's disposal.
                if self._held.get(agent_id) is not held:
                    return
                del self._held[agent_id]
                kept = True if workspace.release is None else await workspace.release()
                if session is not None:
                    session.append(
                        "workspace/disposed", self._payload(workspace, agent_id, kept=kept)
                    )

            return release

        held.dispose = await (scope or self.ctx).effect(enter, label=f"workspace({agent_id})")
        if session is not None:
            session.append(
                "workspace/acquired",
                self._payload(
                    workspace,
                    agent_id,
                    kind=workspace.kind,
                    root=str(workspace.root),
                    repoWritable=workspace.repo_writable,
                ),
            )
        return workspace

    def _payload(self, workspace: Workspace, agent_id: str, **extra: Any) -> dict[str, Any]:
        """The keys both halves of the pair share, spelled once.

        `ref` rides both so a reader can say which branch a turn ran against
        without inspecting the repository, and is omitted rather than sent as
        `null` for the kinds that have none.
        """
        data: dict[str, Any] = {"agentId": agent_id, **extra}
        if workspace.ref is not None:
            data["ref"] = workspace.ref
        return data


class Config(WireModel):
    """Row config for the shared provider."""

    scratch: str | None = None
    """Where scratch directories live. `$PH_HOME/scratch` by default — the idiom
    `default_home_path` exists for, and outside the workspace on purpose."""


@plugin("workspace-shared", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the seam with the shared provider as its floor.

    Named for the provider rather than for the seam, because there is no useful
    "seam with no behaviour" state here: `sandbox-policy` can mount a seam whose
    `confine()` refuses, since refusing is a real answer, but an agent with no
    working directory is not.
    """
    ctx.provide(
        "workspace",
        WorkspaceSeam(
            ctx=ctx,
            shared=SharedWorkspaceProvider(),
            scratch_root=default_home_path(config.scratch, "scratch"),
        ),
    )
