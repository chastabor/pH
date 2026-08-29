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

from ..cordis import Context, plugin
from ..wire import WireModel
from .sandbox import enforcement_of
from .workspace import ContainmentTier

__all__ = [
    "STRICT_REFUSAL",
    "Config",
    "ContainmentService",
    "ContainmentUnavailableError",
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
