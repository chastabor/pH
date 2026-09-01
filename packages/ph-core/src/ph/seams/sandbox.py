"""`ctx.sandbox` — confinement, and the refusal to pretend.

The one rule that makes this seam worth having: **`confine()` never passes
through**. A caller asking to confine an argv is asking for a guarantee; a
policy-only provider that returned the argv unchanged would hand back an
unconfined command that *looks* confined, and every layer above would then be
reasoning about a boundary that does not exist.

So the Phase 1 provider is honest and useless: it resolves and records policy,
and raises `SANDBOX_UNAVAILABLE` when asked to actually confine (dsh's fail-
closed posture). `sandbox-local` landed in P6-04 with `bwrap` (Landlock and Seatbelt still owed),
and that is the *only* tier that bounds an absolute-path write (N2, E13).

Mode resolution is explicit > last logged `sandbox/mode` event > deployment
default, so a per-call decision wins, a session-level change persists in the
log, and neither is guessed.

@module ph.seams.sandbox
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypeAlias, cast, get_args, runtime_checkable

from ..cordis import Context, Disposer, Running, plugin, running
from ..session import Session
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel
from ._registry import claim_slot

__all__ = [
    "ConfinedArgv",
    "Enforcement",
    "SandboxError",
    "SandboxMode",
    "SandboxPolicy",
    "SandboxProvider",
    "SandboxSeam",
    "apply",
    "enforcement_of",
    "writable_paths",
]

SandboxMode: TypeAlias = Literal["read-only", "workspace-write", "danger-full-access"]
Enforcement: TypeAlias = Literal["full", "partial"]


class SandboxError(HarnessError):
    """Confinement was requested and could not be provided.

    A denial, not a failure: the *policy* said this must be confined, and pH
    refuses rather than running unconfined.
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, message: str) -> None:
        super().__init__(message, "SANDBOX_UNAVAILABLE")


class SandboxPolicy(WireModel):
    """The complete per-call confinement request."""

    mode: SandboxMode = "read-only"
    workspace_root: str | None = None
    """The one writable root under `workspace-write`."""
    writable_extra: list[str] | None = None
    network: bool = False


def writable_paths(policy: SandboxPolicy) -> list[str]:
    """Every path this policy permits writing, in the order a backend binds them.

    **On the seam, beside the value it describes**, for `workspace_policy`'s
    reason one module over: two backends need this set and a third arrives with
    P6-04's own Landlock, so a rule private to one of them is a rule the next one
    copies. `writable_roots` answers the same question of a `Workspace`; this
    answers it of the policy that workspace produced.

    Exhaustive over `SandboxMode` rather than an `if`, which is how every other
    closed Literal here is answered — `project_access`, `restorable` and their
    siblings all argue for it, and `restorable`'s docstring is specifically about
    a membership test that nearly went wrong. A fourth mode should fail to
    compile rather than silently inherit "workspace and extras".

    `read-only` names no workspace root and no implicit `/tmp`: a temp directory
    the caller did not ask for is a writable hole nobody declared, and
    `redirection_env` already points the toolchain's scratch at one that was.
    `danger-full-access` returns nothing here because it is not a *set* of
    writable paths — it is "everything", which each backend spells its own way.
    """
    extra = list(policy.writable_extra or ())
    match policy.mode:
        case "read-only" | "danger-full-access":
            return extra
        case "workspace-write":
            roots = [policy.workspace_root] if policy.workspace_root else []
            return [*roots, *extra]


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


@runtime_checkable
class SandboxProvider(Protocol):
    """A confinement backend.

    `enforcement` is a *descriptor*, readable before any call, because
    `containment.strict` has to decide at startup whether this deployment is
    actually confined — and a property only discoverable by confining something
    would make that check "run a command and see", which is not a thing a
    refusal-to-start can do. `partial` is a refusal under strict, not a
    downgrade (E8).

    Typed rather than duck-typed for the reason `WorkspaceProvider` is: a
    backend whose method drifted would fail inside a caller's `except` and be
    reported as *unconfined*, which is the one direction this seam must never
    fail in silently.
    """

    enforcement: Enforcement

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv: ...


@dataclass(slots=True)
class SandboxSeam:
    """The service published as `ctx.sandbox`."""

    ctx: Context
    default_mode: SandboxMode = "read-only"
    provider: SandboxProvider | None = None
    provider_by: Running | None = None
    """Who registered the backend (P6-29). See `ph.seams.compaction`."""

    def register_provider(
        self, provider: SandboxProvider, *, scope: Context | None = None
    ) -> Disposer:
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "provider",
            provider,
            label="sandbox.provider",
        )

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

    @property
    def enforcement(self) -> Enforcement | None:
        """How much the mounted backend actually enforces, or `None` for none.

        Read at startup by `containment.strict`, which refuses on anything but
        `full` — including on `None`, since "no backend" and "a backend that
        bounds some of it" are both "not confined" to a deployment that asked to
        be sure.
        """
        return None if self.provider is None else self.provider.enforcement

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
        # As the row that registered the provider (P6-29); the layer is the
        # registration's, for the reason `CompactionSeam.register` states.
        with running(self.provider_by):
            confined = self.provider.confine(argv, policy)
        if not isinstance(confined, ConfinedArgv):  # pragma: no cover - provider bug
            raise SandboxError("sandbox provider did not return a ConfinedArgv")
        return confined


def enforcement_of(ctx: Context) -> Enforcement | None:
    """How much confinement this deployment actually has. `None` is none at all.

    Asked of a seam that may not be mounted, the way `workspace_of` is — and
    written once because the copies had begun to *disagree*: `permissions-fs`
    read "a backend exists" while `containment` read "a backend that says
    `full`", so a `partial` backend would have had E9's reach sentence telling
    an operator a sandbox bounds their code cells while E8's refusal told them a
    partial boundary is not confinement at all. Two operator-facing statements
    about one fact, in two packages, is the drift this seam exists to prevent.
    """
    seam = ctx.get("sandbox")
    return None if seam is None else seam.enforcement


class Config(WireModel):
    """Row config for the policy-only sandbox provider."""

    default_mode: SandboxMode = "read-only"


@plugin("sandbox-policy", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the sandbox seam with policy resolution and no backend."""
    ctx.provide("sandbox", SandboxSeam(ctx=ctx, default_mode=config.default_mode))
