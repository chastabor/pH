"""`ctx.containment` — which rung of the ladder this deployment asked for (E1, E8).

§4.8's ladder is `advisory` → `worktree` → `sandbox`, and every rung has been
built: the seam and its fallback (P4-07), the git tier (P4-08), provisioning and
`/workspaces` (P4-08b), restore points (P4-09), the default write scope (P4-10).
None of it is *in force* until something chooses, and this is the choosing.

**Two tiers, because the interesting default is not uniform.** A person running
`rlm` is working in their own checkout and would be surprised to find their
agent editing a copy somewhere else — so the root agent stays `advisory`. Its
*children* are the fan-out hazard the tier exists for: eight of them writing one
tree concurrently is the case §4.8 opens with, so `child_tier` defaults to
`worktree`. One knob would have forced the wrong answer on one of them.

**Selection is per acquire, not per mounted row.** The provider is registered
whenever its row is layered; what `tier` decides is whether a given caller
*asks* for it. That keeps one provider slot (two answers to "what runs this" is
a contradiction) while letting a parent and its children sit on different rungs,
and it is why `WorkspaceSeam.acquire` takes a tier rather than this module
deciding which rows exist.

**`strict` refuses to start, and refuses on `partial`.** dsh's fail-closed
`SANDBOX_UNAVAILABLE` posture, lifted from per-call to profile start (Q10). An
operator who sets it is saying "I do not want to run at all unless confinement
is real", so a `partial` backend is a refusal rather than a downgrade to accept
quietly — the whole point being that a downgrade nobody notices is
indistinguishable from the thing they were trying to prevent.

The check runs **after** the profile is mounted rather than inside this row's
`apply`, because a backend may be layered after this row and a verdict computed
at mount would be the wrong one for exactly the profile that orders things that
way — the same reasoning E9's live `reach` sentence is built on.

@module ph.seams.containment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..cordis import Context, plugin
from ..wire import WireModel
from .diagnostics import Diagnostic, contribute
from .sandbox import enforcement_of
from .workspace import ContainmentTier

__all__ = [
    "STRICT_REFUSAL",
    "TIERS",
    "Config",
    "ContainmentService",
    "ContainmentUnavailableError",
    "DescribingProvider",
    "TierDescription",
    "apply",
]

STRICT_REFUSAL = (
    "containment.strict is set, so pH refuses to start unless confinement is real: {because}. "
    "Mount a sandbox backend (sandbox-local, P6-04) and set containment.tier: sandbox, "
    "or unset containment.strict to run at the tier you have."
)
"""Why the process is not starting, and the two ways out.

Both ways, because a refusal that names only the problem leaves an operator
guessing whether the answer is to install something or to change a setting — and
one of those is a decision they may be entitled to make.
"""


@dataclass(frozen=True, slots=True)
class TierDescription:
    """One rung of §4.8's table: what it bounds, what it does not, what it buys.

    Three columns and no severity, which is §12 Q10's own instruction for
    `ph doctor` — *"prints the same three columns rather than a severity
    colour"*. A colour invites a reader to skip the sentence, and the sentence is
    the entire point of E1: the failure being prevented is a **tier name**
    overstating what the tier does, and only prose can correct that.
    """

    bounds: str
    does_not_bound: str
    buys: str


@runtime_checkable
class DescribingProvider(Protocol):
    """A provider whose bargain differs from the stock description of its rung.

    **The columns belong to whoever occupies the rung, not to its name.** `TIERS`
    is keyed by rung, so every provider at `worktree` inherited "buys: collision
    isolation and revertibility (fan-out safety, per-run checkpoints, /revert)" —
    true of a checkout and false of an overlay, which has no git tree to hash and
    therefore never writes a restore point. `ph doctor` was advertising a
    mechanism the mounted tier does not have, in the one place a person looks to
    check exactly that, which is the single failure E1 exists to prevent.

    Optional, like `ReclaimingProvider` and `ExportingProvider`: a provider whose
    bargain *is* its rung's says nothing and gets the stock row. `tier` is on the
    Protocol so the override can be matched to the rung it describes — a report
    with two rungs must not let a child-tier provider rewrite the root's columns.
    """

    tier: ContainmentTier

    def describe_tier(self) -> TierDescription: ...


TIERS: dict[ContainmentTier, TierDescription] = {
    "advisory": TierDescription(
        bounds="nothing",
        does_not_bound="anything — the whole host, at the user's own permissions",
        buys="convention only",
    ),
    "worktree": TierDescription(
        bounds="every tool-mediated write, and every relative-path raw write, "
        "since both resolve against the agent's cwd",
        does_not_bound='an absolute-path raw write — open("/etc/passwd", "w") never consults a cwd',
        buys="collision isolation and revertibility (fan-out safety, per-run checkpoints, /revert)",
    ),
    "sandbox": TierDescription(
        bounds="every write, absolute paths included, refused at the kernel",
        does_not_bound="side effects that are not filesystem writes — network, "
        "already-published artifacts",
        buys="confinement",
    ),
}
"""§4.8's table, once, where every reader can reach it.

E1's gate is a docs test asserting no tier is described as bounding writes it
does not bound (P6-06). That is only worth running against a single home: a
table in prose and a second copy in `ph doctor` would drift, and the drift would
be invisible precisely because each copy looks right on its own. So the
sentences live here, `ph doctor` prints them verbatim, and the docs row when it
lands has one thing to check rather than two things to reconcile.
"""


class ContainmentUnavailableError(RuntimeError):
    """`strict` was set and the deployment cannot honour it.

    A refusal to start, raised where the profile is composed rather than at the
    first call that would have been unconfined: by then the agent is running and
    "refuse to start" has already been disobeyed.
    """


class Config(WireModel):
    """Row config: which rung, for whom, and how strictly."""

    tier: ContainmentTier | None = None
    """The rung the *root* agent runs on — the person's own session.

    `None` is **no opinion**, not `advisory`: mounting this row must not opt
    every profile out of a provider it deliberately layered. A profile says
    `advisory` when it means "this agent stays in the person's own checkout",
    which is a different statement from never having chosen.
    """
    child_tier: ContainmentTier | None = None
    """The rung spawned children run on. `None` follows `tier`.

    A separate knob because the fan-out is the hazard: children are the case
    where "eight agents writing one tree" happens, and a profile wanting them
    isolated should not have to move the person's own agent to get it.
    """
    strict: bool = False
    """Refuse to start unless confinement is real (E8)."""


@dataclass(slots=True)
class ContainmentService:
    """The service published as `ctx.containment`."""

    ctx: Context
    tier: ContainmentTier | None = None
    child_tier: ContainmentTier | None = None
    strict: bool = False

    def for_role(self, *, child: bool) -> ContainmentTier | None:
        """The rung this role gets. The one place the two knobs are read."""
        return (self.child_tier or self.tier) if child else self.tier

    def describe(self) -> list[tuple[str, str]]:
        """What `ph doctor` prints about this deployment (E1, E10).

        Three claims, and each one is a defect this row could otherwise hide.

        **Effective, not configured.** It asks the workspace seam rather than
        reading its own config back to the operator: a `worktree` row over a
        directory that is not a repository declines on every acquire, and being
        told your own setting is not being told what you have.

        **Both rungs, each with its own three columns.** The shipped `rlm`
        posture puts the person's agent on `advisory` and its children on
        `worktree`, so a report printing one rung's table would say "bounds:
        nothing" about a process where most of the writing is bounded — or the
        reverse, which is worse.

        **No severity, by §12 Q10's own instruction.** The three columns *are*
        the finding; a colour would let a reader skip the sentence, and the
        sentence is what stops a tier's name from overstating it (E1).
        """
        workspace = self.ctx.get("workspace")
        root: ContainmentTier = "advisory"
        child: ContainmentTier = "advisory"
        if workspace is not None:
            root, child = (
                workspace.effective_tier(child=False),
                workspace.effective_tier(child=True),
            )
        rows: list[tuple[str, str]] = [("tier (effective)", root)]
        if self.tier is not None and self.tier != root:
            rows.append(("tier (configured)", f"{self.tier} — not in force here"))
        rungs = [root]
        if child != root:
            rows.append(("tier for children", child))
            rungs.append(child)
        provider = None if workspace is None else workspace.provider
        for rung in rungs:
            # Prefixed only when there are two, so the ordinary single-rung
            # report reads as the table §4.8 prints rather than as a matrix.
            prefix = "" if len(rungs) == 1 else f"{rung} "
            description = TIERS[rung]
            if isinstance(provider, DescribingProvider) and provider.tier == rung:
                description = provider.describe_tier()
            rows.append((f"{prefix}bounds", description.bounds))
            rows.append((f"{prefix}does NOT bound", description.does_not_bound))
            rows.append((f"{prefix}buys", description.buys))
        rows.append(("strict", "yes" if self.strict else "no"))
        if self.strict:
            rows.append(("strict is satisfied", "no" if self._unconfined() else "yes"))
        return rows

    def _unconfined(self) -> str | None:
        """Why confinement is not real under `strict`, or `None`.

        A clause, not the whole refusal: the "here are your two ways out"
        paragraph only makes sense when refusing to start, and stitching it on
        at each branch is three places one sentence could drift from the one it
        wraps. `ph doctor` wants the clause; `verify` wants both.
        """
        if not self.strict:
            return None
        if self.tier != "sandbox":
            named = f'"{self.tier}"' if self.tier else "unset"
            return f"containment.tier is {named}, which enforces nothing"
        enforcement = enforcement_of(self.ctx)
        if enforcement is None:
            return "no sandbox backend is mounted"
        if enforcement != "full":
            return (
                f'the mounted backend enforces "{enforcement}", and a partial '
                "boundary is a refusal rather than a downgrade"
            )
        return None

    def verify(self) -> None:
        """Raise unless this deployment can honour what it asked for.

        Registered on `profile/mounted` rather than run in this row's `apply`: a
        backend may be layered *after* the row that asked for it, so a verdict
        computed at mount would be wrong for exactly the profile that orders
        things that way — E9's live-sentence reasoning, one seam over.
        """
        because = self._unconfined()
        if because is not None:
            raise ContainmentUnavailableError(STRICT_REFUSAL.format(because=because))


@plugin("containment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Publish the chosen rungs, and refuse the run if they cannot be honoured."""
    containment = ContainmentService(
        ctx=ctx,
        tier=config.tier,
        child_tier=config.child_tier,
        strict=config.strict,
    )
    ctx.provide("containment", containment)
    # A listener rather than a call here, so the verdict is taken once the whole
    # profile is composed — and so a *second* row wanting to refuse a deployment
    # registers one too, instead of another `if` in whoever starts the process.
    ctx.on("profile/mounted", lambda: containment.verify())

    contribute(
        ctx, Diagnostic(id="containment", title="Containment", read=containment.describe, order=10)
    )
