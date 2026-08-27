"""`ctx.sandbox` — confinement, and the refusal to pretend.

The one rule that makes this seam worth having: **`confine()` never passes
through**. A caller asking to confine an argv is asking for a guarantee; a
policy-only provider that returned the argv unchanged would hand back an
unconfined command that *looks* confined, and every layer above would then be
reasoning about a boundary that does not exist.

So the Phase 1 provider is honest and useless: it resolves and records policy,
and raises `SANDBOX_UNAVAILABLE` when asked to actually confine (dsh's fail-
closed posture). `sandbox-local` with `bwrap`/Landlock/Seatbelt lands in P6-04,
and that is the *only* tier that bounds an absolute-path write (N2, E13).

Mode resolution is explicit > last logged `sandbox/mode` event > deployment
default, so a per-call decision wins, a session-level change persists in the
log, and neither is guessed.

@module ph.seams.sandbox
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Literal, TypeAlias, cast, get_args

from ..cordis import Context, Disposer, plugin
from ..session import Session
from ..tools.errors import HarnessError
from ..wire import WireModel
from ._registry import claim_slot

__all__ = [
    "ConfinedArgv",
    "SandboxError",
    "SandboxMode",
    "SandboxPolicy",
    "SandboxSeam",
    "apply",
]

SandboxMode: TypeAlias = Literal["read-only", "workspace-write", "danger-full-access"]
Enforcement: TypeAlias = Literal["full", "partial"]


class SandboxError(HarnessError):
    """Confinement was requested and could not be provided.

    A denial, not a failure: the *policy* said this must be confined, and pH
    refuses rather than running unconfined.
    """

    denies: ClassVar[bool] = True

    def __init__(self, message: str) -> None:
        super().__init__(message, "SANDBOX_UNAVAILABLE")


class SandboxPolicy(WireModel):
    """The complete per-call confinement request."""

    mode: SandboxMode = "read-only"
    workspace_root: str | None = None
    """The one writable root under `workspace-write`."""
    writable_extra: list[str] | None = None
    network: bool = False


@dataclass(frozen=True, slots=True)
class ConfinedArgv:
    """An argv wrapped by a real backend, and how much it actually enforces.

    `enforcement: "partial"` is reported rather than smoothed over: under
    `containment.strict` (Phase 4) a partial backend is a refusal to start, not
    a downgrade to accept quietly.
    """

    argv: tuple[str, ...]
    enforcement: Enforcement
    backend: str


@dataclass(slots=True)
class SandboxSeam:
    """The service published as `ctx.sandbox`."""

    ctx: Context
    default_mode: SandboxMode = "read-only"
    provider: Any = None

    def register_provider(self, provider: Any, *, scope: Context | None = None) -> Disposer:
        return claim_slot(scope or self.ctx, self, "provider", provider, label="sandbox.provider")

    def resolve_mode(
        self, session: Session | None = None, *, explicit: SandboxMode | None = None
    ) -> SandboxMode:
        """Explicit beats the log; the log beats the deployment default."""
        if explicit is not None:
            return explicit
        if session is not None:
            event = session.latest("sandbox/mode")
            if event is not None and event.data.get("mode") in get_args(SandboxMode):
                return cast(SandboxMode, event.data["mode"])
        return self.default_mode

    def set_mode(self, session: Session, mode: SandboxMode) -> None:
        session.append("sandbox/mode", {"mode": mode})

    @property
    def available(self) -> bool:
        return self.provider is not None

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv:
        """Wrap `argv` so the kernel enforces `policy`.

        :raises SandboxError: when no backend can. Never returns `argv`
            unchanged — a caller must not be able to mistake absence for
            confinement.
        """
        if self.provider is None:
            raise SandboxError(
                "no sandbox backend is mounted, so this command cannot be confined; "
                "mount sandbox-local (P6-04) or run at the worktree tier, which "
                "bounds relative writes only"
            )
        confined = self.provider.confine(argv, policy)
        if not isinstance(confined, ConfinedArgv):  # pragma: no cover - provider bug
            raise SandboxError("sandbox provider did not return a ConfinedArgv")
        return confined


class Config(WireModel):
    """Row config for the policy-only sandbox provider."""

    default_mode: SandboxMode = "read-only"


@plugin("sandbox-policy", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the sandbox seam with policy resolution and no backend."""
    ctx.provide("sandbox", SandboxSeam(ctx=ctx, default_mode=config.default_mode))
