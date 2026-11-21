"""`ctx.containment`: which rung of the ladder this deployment asked for (E1, E8).

The ladder of §4.8 is `advisory` → `worktree` → `sandbox`, and every rung has
been built: the seam and its fallback (P4-07), the git tier (P4-08),
provisioning and `/workspaces` (P4-08b), restore points (P4-09), and the
default write scope (P4-10). None of it is *in force* until something chooses.
This row makes that choice.

**There are two tier settings, because the useful default is not uniform.** A
user who runs `rlm` works in their own checkout and would be surprised to find
their agent editing a copy somewhere else. So the root agent stays `advisory`.
Its *children* are the fan-out hazard that the tier exists for: eight children
that write one tree at the same time are the case that §4.8 opens with. So the
`rlm` bundle sets `child_tier` to `worktree`. One setting would have forced the
wrong answer on one of the two.

**Selection is per acquire, not per mounted row.** The provider is registered
whenever its row is layered and, for a tier that probes its host, the probe
passes. `tier` decides whether a given caller *asks* for it. That keeps one
provider slot, because two answers to "what runs this" is a contradiction. It
also lets a parent and its children be on different rungs. It is the reason why
`WorkspaceSeam.acquire` takes a tier and this module does not decide which rows
exist.

**`strict` refuses to start, and it refuses on `partial`.** This is the
fail-closed `SANDBOX_UNAVAILABLE` posture of dsh, moved from each call to
profile start (Q10). An operator who sets it says "I do not want to run at all
unless confinement is real". So a `partial` backend is a refusal and not a
downgrade to accept quietly. A downgrade that nobody notices cannot be told
from the thing that the operator tried to prevent.

The check runs **after** the profile is mounted and not inside the `apply` of
this row. A backend may be layered after this row, and a verdict computed at
mount would be wrong for exactly the profile that orders its rows that way.
The live `reach` sentence of E9 is built on the same reasoning.

@module ph.seams.containment
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from ..cordis import Context, MountRefusal, plugin
from ..keys import CONTAINMENT, WORKSPACE
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
"""Why the process does not start, and the two ways out.

It names both ways. A refusal that names only the problem leaves an operator
to guess whether the answer is to install something or to change a setting.
One of those is a decision that the operator may be entitled to make.
"""


@dataclass(frozen=True, slots=True)
class TierDescription:
    """One rung of the table in §4.8: what it bounds, what it does not, what it buys.

    There are three columns and no severity. That is the instruction of §12 Q10
    for `phern doctor`: it *"prints the same three columns rather than a
    severity color"*. A color invites a reader to skip the sentence, and the
    sentence is the purpose of E1. The failure that E1 prevents is a **tier
    name** that overstates what the tier does, and only prose can correct that.
    """

    bounds: str
    does_not_bound: str
    buys: str


@runtime_checkable
class DescribingProvider(Protocol):
    """A provider that offers something different from the stock description of its rung.

    **The columns belong to the provider that occupies the rung, not to the
    name of the rung.** `TIERS` is keyed by rung. So every provider at
    `worktree` inherited "buys: collision isolation and revertibility (fan-out
    safety, per-run checkpoints, /revert)". That is true of a checkout and
    false of an overlay, which has no git tree to hash and therefore never
    writes a restore point. `phern doctor` advertised a mechanism that the
    mounted tier does not have, in the one place where a user looks to check
    exactly that. That is the single failure that E1 exists to prevent.

    The capability is optional, like `ReclaimingProvider` and
    `ExportingProvider`. A provider that offers exactly what its rung
    describes declares nothing and gets the stock row. `tier` is on the
    Protocol so that the override can be matched to the rung that it
    describes. A report with two rungs must not let a child-tier provider
    rewrite the columns of the root.
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
        bounds="every write, absolute paths included, refused at the kernel; and "
        "network egress, to whatever `phern doctor`'s Sandbox allowances section reports "
        "as in force",
        does_not_bound="what an allowed request does at the far end — an artifact "
        "published or a message sent to a reachable host is not taken back",
        buys="confinement",
    ),
}
"""The table of §4.8, in one place that every reader can reach.

The gate of E1 is a docs test which asserts that no tier is described as
bounding writes that it does not bound (P6-06). That test is only worth
running against a single source. A table in prose and a second copy in
`phern doctor` would drift, and nobody would see the drift, because each copy
looks right on its own. So the sentences live here, and `phern doctor` prints
them verbatim. When the docs row lands, it has one thing to check and not two
things to reconcile.
"""


class ContainmentUnavailableError(MountRefusal):
    """`strict` was set and the deployment cannot honor it.

    This is a refusal to start. It is raised where the profile is composed and
    not at the first call that would have been unconfined. By then the agent is
    running, and "refuse to start" has already been disobeyed. It is a
    `MountRefusal`, so every command that mounts prints it as a sentence. It
    was a bare `RuntimeError` before, and `phern -p` under `strict` answered
    with a traceback while only `phern doctor` caught it (P4-12).
    """


class Config(WireModel):
    """Row config: which rung, for whom, and how strictly."""

    tier: ContainmentTier | None = None
    """The rung that the *root* agent runs on, which is the user's own session.

    `None` means **no opinion**, not `advisory`. To mount this row must not opt
    every profile out of a provider that it deliberately layered. A profile
    says `advisory` when it means "this agent stays in the user's own
    checkout". That is a different statement from never having chosen.
    """
    child_tier: ContainmentTier | None = None
    """The rung that spawned children run on. `None` follows `tier`.

    It is a separate setting because the fan-out is the hazard. Children are
    the case where "eight agents writing one tree" happens. A profile that
    wants them isolated should not have to move the user's own agent to get
    that.
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
        """The rung that this role gets. The one place that reads the two settings."""
        return (self.child_tier or self.tier) if child else self.tier

    def describe(self) -> list[tuple[str, str]]:
        """What `phern doctor` prints about this deployment (E1, E10).

        It makes three claims. Each one is a defect that this row could
        otherwise hide.

        **Effective, not configured.** It asks the workspace seam and does not
        read its own config back to the operator. A `worktree` row over a
        directory that is not a repository declines on every acquire. To be
        told your own setting is not to be told what you have.

        **Both rungs, each with its own three columns.** The shipped `rlm`
        posture puts the user's agent on `advisory` and its children on
        `worktree`. A report that printed the table of one rung would say
        "bounds: nothing" about a process where most of the writing is bounded.
        Or it would say the reverse, which is worse.

        **No severity, by the instruction of §12 Q10.** The three columns
        *are* the finding. A color would let a reader skip the sentence, and
        the sentence stops the name of a tier from overstating the tier (E1).
        """
        workspace = self.ctx.get(WORKSPACE)
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
            # The prefix is added only when there are two rungs. So the ordinary
            # single-rung report reads as the table that §4.8 prints and not as
            # a matrix.
            prefix = "" if len(rungs) == 1 else f"{rung} "
            description = TIERS[rung]
            if isinstance(provider, DescribingProvider) and provider.tier == rung:
                description = provider.describe_tier()
            rows.append((f"{prefix}bounds", description.bounds))
            rows.append((f"{prefix}does NOT bound", description.does_not_bound))
            rows.append((f"{prefix}buys", description.buys))
        # **Both boundaries are in one section, because they contradicted each
        # other across two sections.** The rows above describe the *workspace*
        # tier, which `effective_tier` reads from the workspace provider. So a
        # deployment with a mounted sandbox backend and no sandbox-tier
        # workspace provider printed `bounds: nothing` here and `confines: yes`
        # five lines down, with `strict is satisfied: yes` between them. They
        # are two different boundaries: one is the tree that an agent works in,
        # and the other is every command that the harness wraps. The row below
        # says so next to them, so that a reader does not have to reconcile two
        # sections.
        confinement = enforcement_of(self.ctx)
        if confinement is not None:
            rows.append(
                (
                    "confined commands",
                    f'a sandbox backend is mounted and enforces "{confinement}"; every '
                    "command the harness wraps is bounded by the kernel, whatever the "
                    "workspace rows above say about the tree. Whether *code cells* are "
                    "is a further question — it needs the agent to have a workspace — "
                    "and the Code runtime section answers it",
                )
            )
        rows.append(("strict", "yes" if self.strict else "no"))
        if self.strict:
            rows.append(("strict is satisfied", "no" if self._unconfined() else "yes"))
        return rows

    def _unconfined(self) -> str | None:
        """Why confinement is not real under `strict`, or `None`.

        This is a clause and not the whole refusal. The "here are your two ways
        out" paragraph only makes sense in a refusal to start. To append it at
        each branch would make three places where one sentence could drift
        from the sentence that it wraps. `phern doctor` needs the clause, and
        `verify` needs both.
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
        """Raise unless this deployment can honor what it asked for.

        It is registered on `profile/mounted` and does not run in the `apply`
        of this row. A backend may be layered *after* the row that asked for
        it, so a verdict computed at mount would be wrong for exactly the
        profile that orders its rows that way. This is the live-sentence
        reasoning of E9, applied to this seam.
        """
        because = self._unconfined()
        if because is not None:
            raise ContainmentUnavailableError(STRICT_REFUSAL.format(because=because))


@plugin("containment", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Publish the chosen rungs, and refuse the run if they cannot be honored."""
    containment = ContainmentService(
        ctx=ctx,
        tier=config.tier,
        child_tier=config.child_tier,
        strict=config.strict,
    )
    ctx.provide(CONTAINMENT, containment)
    # This is a listener and not a call here, so that the verdict is taken when
    # the whole profile is composed. A *second* row that wants to refuse a
    # deployment then registers a listener too, and nobody adds another `if` to
    # the code that starts the process.
    ctx.on("profile/mounted", lambda: containment.verify())

    contribute(
        ctx, Diagnostic(id="containment", title="Containment", read=containment.describe, order=10)
    )
