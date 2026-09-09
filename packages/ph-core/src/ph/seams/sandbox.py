"""`ctx.sandbox` — confinement, and the refusal to pretend.

The one rule that makes this seam worth having: **`confine()` never passes
through**. A caller asking to confine an argv is asking for a guarantee; a
policy-only provider that returned the argv unchanged would hand back an
unconfined command that *looks* confined, and every layer above would then be
reasoning about a boundary that does not exist.

So the Phase 1 provider is honest and useless: it resolves and records policy,
and raises `SANDBOX_UNAVAILABLE` when asked to actually confine (dsh's fail-
closed posture). `sandbox-local` landed in P6-04 with `bwrap`, and P6-40 verified
Seatbelt (Landlock still owed) — that is the *only* tier that bounds an
absolute-path write (N2, E13).

Mode resolution is explicit > last logged `sandbox/mode` event > deployment
default, so a per-call decision wins, a session-level change persists in the
log, and neither is guessed.

**What a confined command may reach beyond its workspace is the deployment's to
say, and it is said once** (P6-38). `Allowances` — directories writable beyond the
workspace, and the network hosts reachable through the egress proxy — are
registered by the `sandbox-allow` row and merged into every policy *here*, in
`confine`, so the shell, the probe and whatever confines next all get one
boundary, and a change to the row reaches the next command without anything
restarting. Without the row the seam fails closed: no extra directory, no network.

**Network is three states, not a bool.** `off` is `--unshare-net` and nothing
else; `full` is the host's network unfiltered; `allowlist` is the namespace kept
and a single door punched through it — a filtering proxy on the host, reached from
inside over a unix socket (`ph.seams.sandbox_egress`). A command that ignores the
proxy variables finds nothing to connect to, which is the direction a filter must
fail in: the namespace is the enforcement, the proxy only decides what it lets
through.

A boundary that refuses in silence teaches nobody anything, so a refusal is a
**record**: `sandbox/denied` in the agent's session, from the proxy when it
refused a host and from the command's own output when the kernel refused a
write. The TUI renders it with the `/sandbox` line that lifts it.

@module ph.seams.sandbox
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, cast, get_args, runtime_checkable

from pydantic import Field

from ..cordis import Context, Disposer, Running, plugin, running
from ..keys import AGENTS, SANDBOX
from ..paths import canonical
from ..session import Session
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel
from ._registry import claim_slot

__all__ = [
    "DEFAULT_HOSTS",
    "DENIED",
    "NOTHING",
    "Allowances",
    "ConfinedArgv",
    "Denial",
    "DenialKind",
    "DenialReader",
    "DenialSource",
    "Egress",
    "Enforcement",
    "NetworkAllowance",
    "NetworkMode",
    "SandboxError",
    "SandboxMode",
    "SandboxPolicy",
    "SandboxProvider",
    "SandboxSeam",
    "allowed_paths_of",
    "apply",
    "enforcement_of",
    "host_allowed",
    "writable_paths",
]

log = logging.getLogger("ph.seams.sandbox")

SandboxMode: TypeAlias = Literal["read-only", "workspace-write", "danger-full-access"]
Enforcement: TypeAlias = Literal["full", "partial"]
NetworkMode: TypeAlias = Literal["off", "allowlist", "full"]

DENIED = "sandbox/denied"
"""The session event a refused boundary leaves behind. See `Denial`."""

DEFAULT_HOSTS: tuple[str, ...] = (
    # Python
    "pypi.org",
    "files.pythonhosted.org",
    "docs.python.org",
    # JavaScript
    "registry.npmjs.org",
    "www.npmjs.com",
    "npmjs.com",
    # Rust
    "crates.io",
    "static.crates.io",
    "index.crates.io",
    "docs.rs",
    # Go
    "proxy.golang.org",
    "sum.golang.org",
    "pkg.go.dev",
    # Source hosting
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "*.githubusercontent.com",
    # Reference
    "developer.mozilla.org",
)
"""What a confined command may reach when a deployment has not said otherwise.

Package indexes and the places their documentation lives — an agent that cannot
`uv add` or read `docs.rs` is one somebody switches the sandbox off for, and a
sandbox switched off bounds nothing. **A starting point, not a judgement**: the
row it seeds is the user's to trim or extend, from the profile or from `/sandbox`,
and `ph doctor` prints what is in force.

Exact hosts, with one wildcard. `*.githubusercontent.com` is the one because
GitHub serves raw files, release assets and avatars from a family of
subdomains that changes; every other entry is a single host, because a wildcard
is a widening somebody should have to write on purpose. See `host_allowed` for
the two spellings.
"""


class SandboxError(HarnessError):
    """Confinement was requested and could not be provided.

    A denial, not a failure: the *policy* said this must be confined, and pH
    refuses rather than running unconfined.
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, message: str) -> None:
        super().__init__(message, "SANDBOX_UNAVAILABLE")


# ----------------------------------------------------------------- policy --


class Egress(WireModel):
    """The one door out of a confined command's network, to the proxy.

    Two roles, one value. The backend row **registers** one on the seam once its
    probe has shown a confined command can reach the proxy; `SandboxSeam.effective`
    then hands a **copy carrying the agent** to each policy it merges. A separate
    `EgressBridge` type held the same two fields for the registration half and was
    copied into this one at three sites.

    A value rather than the proxy object, because these are the only facts a
    backend needs and it must not be able to reach the listener from a policy
    merge. A caller has nothing to say about any of it: the proxy's socket is a
    fact about the deployment, not about one command.
    """

    socket: str
    """The proxy's unix socket on the host, reachable inside through the read-only
    bind of `/` — no extra mount, and nothing writable. The `shim` door's target;
    the `loopback` door does not use it."""
    port: int
    """Where the command dials on `127.0.0.1`, and what `proxy_url` names. Under a
    network namespace it is the shim's fixed port inside, private to that sandbox;
    under Seatbelt, which shares the host's loopback, it is the proxy's own
    listener there — see `sandbox_local.Door`."""
    agent: str | None = None
    """Who this command runs for, carried to the proxy as the proxy URL's user, so a
    refusal is recorded in the right session."""


class SandboxPolicy(WireModel):
    """The complete per-call confinement request."""

    mode: SandboxMode = "read-only"
    workspace_root: str | None = None
    """The one writable root under `workspace-write`."""
    writable_extra: list[str] | None = None
    refuse_network: bool = False
    """The caller's veto: this command has no business online, whatever the
    deployment permits.

    **A request can only ever narrow.** There is deliberately no way to spell "give
    me the network" — §6.5 caps a caller at what the deployment allows, so an
    asking field would have exactly one reachable behaviour (no effect) and a
    docstring claiming otherwise. What a caller *can* do is want less, which is
    this.

    Read by `SandboxSeam.effective` and nowhere else; `network` below is what a
    backend reads."""
    network: bool = False
    """Whether this command has the host's network, **resolved**.

    Set by `SandboxSeam.effective` from the deployment's `Allowances`, and read by
    the backend. `False` by default, so a policy handed straight to a backend
    without passing through the seam — as the backend's own tests do — is closed
    rather than open.

    This and `refuse_network` are two fields because they are two facts: what the
    caller asked of the seam, and what the seam resolved for the backend. They were
    one tri-state field, and its documented third state was unreachable —
    `True` and `None` took the same branch everywhere."""
    egress: Egress | None = None
    """The proxy door, when the effective network mode is `allowlist`. See `Egress`:
    the seam sets this; a caller never does."""


def writable_paths(policy: SandboxPolicy) -> list[str]:
    """Every path this policy permits writing, in the order a backend binds them.

    **On the seam, beside the value it describes**: two backends need this set and a
    third arrives with Landlock, so a rule private to one of them is a rule the next
    one copies. `writable_roots` answers the same question of a `Workspace`; this
    answers it of the policy that workspace produced.

    Exhaustive over `SandboxMode` rather than an `if`, so a fourth mode fails to
    compile rather than silently inheriting "workspace and extras".

    `read-only` names no workspace root and **no implicit `/tmp`**: a temp directory
    the caller did not ask for is a writable hole nobody declared, and
    `redirection_env` already points the toolchain's scratch at one that was.
    `danger-full-access` returns nothing here because it is not a *set* of writable
    paths — it is "everything", which each backend spells its own way.
    """
    extra = list(policy.writable_extra or ())
    match policy.mode:
        case "read-only" | "danger-full-access":
            return extra
        case "workspace-write":
            roots = [policy.workspace_root] if policy.workspace_root else []
            return [*roots, *extra]


# ------------------------------------------------------------- allowances --


class NetworkAllowance(WireModel):
    """Which network a confined command gets, and through which door."""

    mode: NetworkMode = "allowlist"
    """`off`: none. `allowlist`: the hosts below, through the egress proxy. `full`:
    the host's network, unfiltered — the mode `danger-full-access` implies."""
    hosts: list[str] = Field(default_factory=lambda: list(DEFAULT_HOSTS))
    """What `allowlist` lets through. Two spellings — see `host_allowed`."""


class Allowances(WireModel):
    """What a confined command may reach beyond its workspace and scratch.

    The deployment's statement, registered once by `sandbox-allow` and merged into
    every policy by `SandboxSeam.effective`. The defaults are the row's *shipped*
    posture; a profile that spells the row's config replaces the whole of it
    (`ph.cordis.loader`'s patch rule), so a deployment that wants one more host
    writes the list it wants rather than a delta.
    """

    paths: list[str] = Field(default_factory=list)
    """Directories writable beyond the workspace and its scratch, `~` allowed.

    Empty by default on purpose. `redirection_env` already points the toolchain's
    caches inside the agent's scratch, so the ordinary build needs nothing here;
    what belongs here is a directory a deployment *shares* across agents and
    accepts the consequences of — a package cache, a model store. Never `/`: that
    is `danger-full-access` spelled to look like an allowlist, and the command
    refuses it."""
    network: NetworkAllowance = Field(default_factory=NetworkAllowance)


NOTHING = Allowances(paths=[], network=NetworkAllowance(mode="off", hosts=[]))
"""What a deployment gets when no `sandbox-allow` row is mounted: no extra
directory and no network. The closed direction, stated as a value so `effective`
has no `if allowances is None` branch to get wrong."""


def host_allowed(hosts: Sequence[str], host: str, port: int) -> bool:
    """Whether `host:port` is named by `hosts`.

    **Two spellings, and neither implies the other.** `example.com` allows that one
    host; `*.example.com` allows every host *under* it and not the apex, because a
    wildcard that quietly covered its apex is a widening nobody wrote down. A
    deployment that wants both lists both, as `DEFAULT_HOSTS` does for GitHub.

    An entry may carry `:port`, and then allows only that port; without one it
    allows any. Matching is case-insensitive and ignores a trailing dot, which is
    the same host to DNS.
    """
    wanted = host.lower().rstrip(".")
    for entry in hosts:
        name, allowed_port = _split_entry(entry)
        if allowed_port is not None and allowed_port != port:
            continue
        if name.startswith("*."):
            if wanted.endswith(name[1:]) and wanted != name[2:]:
                return True
        elif wanted == name:
            return True
    return False


def _split_entry(entry: str) -> tuple[str, int | None]:
    """`host[:port]` → the host and the port, if one was written."""
    text = entry.strip().lower()
    name, separator, port = text.rpartition(":")
    if separator and port.isdigit():
        return name.rstrip("."), int(port)
    return text.rstrip("."), None


# ---------------------------------------------------------------- denials --


DenialKind: TypeAlias = Literal["filesystem", "network"]
DenialSource: TypeAlias = Literal["proxy", "output"]


@dataclass(frozen=True, slots=True)
class Denial:
    """One refused boundary, with the sentence a person reads about it.

    `via` says how it is known. `proxy` is certain: the proxy refused a host and
    knows which. `output` is read from what a confined command printed — the kernel
    refuses in silence, so `Read-only file system` in stderr is the only account
    there is — and the sentence says "reported" rather than "blocked" for that
    reason.
    """

    kind: DenialKind
    via: DenialSource
    host: str | None = None
    port: int | None = None
    evidence: str | None = None

    def message(self) -> str:
        """What the TUI shows and the log keeps — one sentence, with the way out."""
        if self.kind == "network" and self.via == "proxy":
            return (
                f"Sandbox blocked network access to {self.host}:{self.port}. "
                f"Allow it with /sandbox allow host {self.host}."
            )
        if self.kind == "network":
            return (
                f'A confined command reported "{self.evidence}". It did not go through the '
                "sandbox's proxy, so nothing was reachable; hosts on the allowlist are "
                "reachable through HTTPS_PROXY, and /sandbox lists them."
            )
        return (
            f'A confined command reported "{self.evidence}". Writes outside the workspace '
            "are refused; allow a directory with /sandbox allow path <dir>."
        )

    def record(self, agent: str | None) -> dict[str, Any]:
        """The `sandbox/denied` payload. Carries the sentence, so every reader —
        the TUI, the trajectory, `ph agents attach` — prints one account."""
        data: dict[str, Any] = {"kind": self.kind, "via": self.via, "message": self.message()}
        if self.host is not None:
            data["host"] = self.host
        if self.port is not None:
            data["port"] = self.port
        if self.evidence is not None:
            data["evidence"] = self.evidence
        if agent is not None:
            data["agent"] = agent
        return data


@runtime_checkable
class DenialReader(Protocol):
    """A backend that can recognise its own kernel's refusals in a command's output.

    **Optional, and the backend's rather than the seam's**, because every string
    such a reader matches is a fact about one platform: `bwrap` on Linux leaves
    `Read-only file system` and `Network is unreachable`, and Seatbelt on macOS
    says `Operation not permitted` for both boundaries and leaves the path on the
    line to tell them apart. A table of Linux sentences living in this module would
    be the seam asserting one kernel's dialect on every backend's behalf —
    `DescribingProvider` next door exists for exactly this shape, and for the same
    reason.

    A backend that has not been measured implements nothing and the seam records
    nothing, which is the honest answer: the kernel refuses in silence, and a
    guess about *how* it phrases that is not evidence.
    """

    def read_denial(self, output: str, *, network: bool) -> Denial | None: ...


# ------------------------------------------------------------------ seam --


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
    forwards_signals: bool = True
    """Whether a signal sent to this argv reaches the command inside it.

    **Declared, because it cannot be discovered by trying.** `bwrap` does not
    forward: it dies of `SIGINT` itself and, having unshared the PID namespace,
    takes everything inside down with it — so a caller that signals a confined
    child to *interrupt* it destroys it instead. `sandbox-exec` execs its target
    and adds no namespace, so signals land normally.

    A caller with a cooperative stop of its own (the RLM kernel's `cancel` frame)
    reads this to decide whether the signal is a second route or a demolition. It
    is `enforcement`'s shape and for `enforcement`'s stated reason: a property only
    discoverable by confining something and seeing what happens is not one a caller
    can act on beforehand.

    Defaults to `True` — the unwrapped truth — so a backend that adds no process
    between the caller and its child says nothing."""
    policy: SandboxPolicy | None = None
    """The *effective* policy the backend enforced — the caller's request with the
    deployment's allowances merged in. Stamped by the seam, so a caller reading the
    result back knows what was actually bounded rather than what it asked for."""


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

    @property
    def enforcement(self) -> Enforcement: ...

    @property
    def backend(self) -> str:
        """What this backend is called — `bwrap`, `sandbox-exec`.

        A *declared* name, beside `enforcement` and readable for the same
        reason: a consumer that wants to tell a person what is confining them
        had only `type(provider).__name__`, which is a Python class name derived
        rather than stated, and which no seam contract promises.
        `ConfinedArgv.backend` is this field rather than a literal each
        `confine` repeats.

        Both are properties rather than attributes — `CodeRuntime` says why —
        because both backends expose them as read-only, which a settable
        Protocol member refuses.
        """
        ...

    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv: ...


@dataclass(slots=True)
class SandboxSeam:
    """The service published as `ctx.sandbox`."""

    ctx: Context
    default_mode: SandboxMode = "read-only"
    provider: SandboxProvider | None = None
    provider_by: Running | None = None
    """Who registered the backend (P6-29). See `ph.seams.compaction`."""
    allowances: Allowances | None = None
    allowances_by: Running | None = None
    """What the deployment lets a confined command reach, and who said so. `None`
    reads as `NOTHING` — see `effective`."""
    egress: Egress | None = None
    egress_by: Running | None = None
    """The proxy door, once the backend row has proved a confined command can reach
    it. `None` with `allowlist` in force means no network at all, and
    `network_posture` says so."""

    @property
    def _allowed(self) -> Allowances:
        """What this deployment permits, with the unmounted case already answered.

        The one spelling of `allowances or NOTHING`, which is what `NOTHING` is for:
        it was stated as a value so no reader would have an `is None` branch to get
        wrong, and then three readers each wrote one anyway. A pydantic model is
        always truthy, so `or` is safe here.

        `network_posture` keeps its own `None` branch deliberately — "no
        sandbox-allow row is mounted" is a different sentence from "the row says
        off", and a person reading `ph doctor` needs to be able to tell them apart.
        """
        return self.allowances if self.allowances is not None else NOTHING

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

    def register_allowances(
        self, allowances: Allowances, *, scope: Context | None = None
    ) -> Disposer:
        """Say what confined commands may reach. One slot: two statements of one
        boundary is a contradiction, and re-applying the row is how it changes.

        **The paths are canonicalised here, because this is where they are minted.**
        A deployment writes `~/.cache/uv`; the kernel matches the path it resolves,
        and so must the prompt boundary `permissions-fs` draws from the same set
        (E6). Doing it here rather than in `allowed_paths` is the difference between
        once per mount and once per confined command *and* per gated write — the
        cost the memoised helper this replaced was written to avoid, and it belongs
        at the mint, not behind a cache. `expanduser` comes with it: `~` is a
        spelling too, and a consumer comparing against a literal `~` compares
        against nothing.
        """
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "allowances",
            allowances.model_copy(
                update={"paths": [str(canonical(Path(p).expanduser())) for p in allowances.paths]}
            ),
            label="sandbox.allowances",
        )

    def register_egress(self, bridge: Egress, *, scope: Context | None = None) -> Disposer:
        """Say that a proxy is up and a confined command can reach it."""
        return claim_slot(
            self.ctx.running_for(scope), self, "egress", bridge, label="sandbox.egress"
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

    # ------------------------------------------------------------ allowances --

    def allowed_paths(self) -> tuple[Path, ...]:
        """The deployment's extra writable directories — expanded, and only the
        ones that exist.

        A missing directory is skipped rather than bound, because `bwrap` refuses
        to start when a bind source is absent and every confined command would
        fail with it. Skipping is the closed direction — less is writable, not
        more — and `sandbox-allow`'s diagnostic marks the entry `missing` so the
        omission is visible rather than silent.
        """
        # Already canonical and expanded — `register_allowances` did it once, at the
        # mint. This is a filter and nothing more: it runs inside `effective`, so on
        # every confined command and every gated write, and a `realpath` here would
        # be a syscall per configured directory per command.
        found = [Path(entry) for entry in self._allowed.paths]
        existing = tuple(path for path in found if path.is_dir())
        for path in found:
            if path not in existing:
                log.debug("ph.seams.sandbox: allowed path %s does not exist; not bound", path)
        return existing

    def permits(self, host: str, port: int) -> bool:
        """What the egress proxy asks, per connection, against the allowances *now*.

        Read live rather than captured when the proxy started, which is what lets
        `/sandbox allow host` take effect on the next request without the proxy —
        and the tunnels it is carrying — being restarted.
        """
        network = self._allowed.network
        match network.mode:
            case "off":
                return False
            case "full":
                return True
            case "allowlist":
                return host_allowed(network.hosts, host, port)

    def network_posture(self) -> str:
        """One sentence on what confined commands can reach, for `ph doctor` and
        `/sandbox` — written once so the two cannot disagree."""
        if self.allowances is None:
            return "off — no sandbox-allow row is mounted, so a confined command has no network"
        network = self.allowances.network
        match network.mode:
            case "off":
                return "off — confined commands have no network"
            case "full":
                return "full — confined commands use the host's network, unfiltered"
            case "allowlist":
                if self.egress is None:
                    return (
                        "allowlist, but no egress bridge is mounted (see Local confinement), "
                        "so confined commands have no network"
                    )
                count = len(network.hosts)
                return (
                    f"allowlist — {count} host{'s' if count != 1 else ''} reachable through "
                    "the egress proxy; everything else is refused"
                )

    def effective(self, policy: SandboxPolicy, *, agent: str | None = None) -> SandboxPolicy:
        """The caller's request with the deployment's allowances merged in.

        **The one place the two meet.** A backend enforces what it is handed and
        knows nothing about rows; a caller states what its command needs and knows
        nothing about the deployment. Merging here means `ctx.shell`, the probe and
        the next confiner cannot drift from one another about what is allowed.

        Extra directories are not added under `read-only`: that mode is the
        session saying *nothing* is writable, and a deployment's cache directory
        is not an exception to a posture a person chose for one session.

        **Network and its door are decided in one chain**, each case stated once.
        They were two passes — an `if/elif/else` for the network, then a four-clause
        guard for the door that re-tested two of the same conditions plus `not
        network`, which encoded the `danger-full-access` case while looking like it
        said something else. The cases: the mode that means no confinement gets the
        host's network; a caller that refused gets nothing; then the deployment
        decides, and `allowlist` opens the door only when a bridge is actually
        mounted — `allowlist` without one is no network, which is the closed
        direction and what `network_posture` reports.
        """
        extra = list(policy.writable_extra or ())
        if policy.mode != "read-only":
            extra += [str(path) for path in self.allowed_paths() if str(path) not in extra]
        allowed = self._allowed.network
        network, egress = False, None
        if policy.mode == "danger-full-access":
            network = True
        elif policy.refuse_network:
            pass
        elif allowed.mode == "full":
            network = True
        elif allowed.mode == "allowlist" and self.egress is not None:
            egress = self.egress.model_copy(update={"agent": agent})
        return policy.model_copy(
            update={"writable_extra": extra or None, "network": network, "egress": egress}
        )

    # --------------------------------------------------------------- confine --

    def confine(
        self, argv: tuple[str, ...], policy: SandboxPolicy, *, agent: str | None = None
    ) -> ConfinedArgv:
        """Wrap `argv` so the kernel enforces `policy`, plus the deployment's allowances.

        `agent` is who the command runs for — carried to the proxy so a refused host
        is recorded in that agent's session rather than nowhere.

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
        effective = self.effective(policy, agent=agent)
        # As the row that registered the provider (P6-29); the layer is the
        # registration's, for the reason `CompactionSeam.register` states.
        with running(self.provider_by):
            confined = self.provider.confine(argv, effective)
        if not isinstance(confined, ConfinedArgv):  # pragma: no cover - provider bug
            raise SandboxError("sandbox provider did not return a ConfinedArgv")
        return replace(confined, policy=effective)

    # --------------------------------------------------------------- denials --

    def report_denial(
        self, confined: ConfinedArgv, streams: Sequence[str], agent: str | None
    ) -> None:
        """Read what a confined command's own words say was refused, and record it.

        **The seam's rule, in the seam.** This module's docstring states that a
        refusal is a record rather than a silence, and `ctx.shell` had the only
        implementation — so when the RLM kernel became the second thing confined it
        adopted the boundary and not the rule: a cell refused by the kernel produced
        a traceback with no `/sandbox allow path` line, while the identical refusal
        from `tool-bash` produced one. One statement, both callers.

        `streams` are tried in order and the first refusal wins; pass stderr before
        stdout, which is where these sentences come from. **Only pass the output of
        a command that actually failed** — the signatures match nothing an ordinary
        command prints on success, so a `grep` that finds "Read-only file system"
        and exits 0 has not been refused anything.
        """
        network = bool(confined.policy.network) if confined.policy is not None else False
        for stream in streams:
            denial = self.read_denial(stream, network=network)
            if denial is not None:
                self.record_denial(denial, agent=agent)
                return

    def read_denial(self, output: str, *, network: bool) -> Denial | None:
        """What the mounted backend recognises in a confined command's output.

        `None` when no backend is mounted or the one that is has not been measured
        on this platform — see `DenialReader`. Asked here rather than by the caller
        so a consumer never learns which backend answered (I5), and run as the row
        that registered it (P6-29), like every other provider body.
        """
        if not isinstance(self.provider, DenialReader):
            return None
        with running(self.provider_by):
            return self.provider.read_denial(output, network=network)

    def record_denial(self, denial: Denial, *, agent: str | None) -> None:
        """Put a refused boundary in the agent's log, where the TUI will find it.

        The agent's *session*, looked up through `ctx.agents`, because that is the
        transcript the person is reading. A denial with no agent to charge it to —
        a connection that arrived without the proxy user, an agent already gone —
        is logged rather than dropped, so the fact is somewhere.
        """
        session = self._session_of(agent)
        if session is None:
            log.warning(
                "ph.seams.sandbox: %s (no session to record it in; agent=%r)",
                denial.message(),
                agent,
            )
            return
        session.append(DENIED, denial.record(agent))

    def _session_of(self, agent: str | None) -> Session | None:
        agents = self.ctx.get(AGENTS)
        if agents is None or not agent:
            return None
        found = agents.get(agent)
        session = getattr(found, "session", None)
        return session if isinstance(session, Session) else None


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
    seam = ctx.get(SANDBOX)
    return None if seam is None else seam.enforcement


def allowed_paths_of(ctx: Context) -> tuple[Path, ...]:
    """The deployment's extra writable directories, asked of a seam that may not
    be mounted.

    For `permissions-fs`, whose `outside-workspace` rule prompts about what the
    backend would refuse — and must therefore stop prompting about what the
    backend now allows, or the prompt boundary and the enforced one describe two
    different sets (E6).
    """
    seam = ctx.get(SANDBOX)
    return () if seam is None else seam.allowed_paths()


class Config(WireModel):
    """Row config for the policy-only sandbox provider."""

    default_mode: SandboxMode = "read-only"


@plugin("sandbox-policy", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the sandbox seam with policy resolution and no backend."""
    ctx.provide(SANDBOX, SandboxSeam(ctx=ctx, default_mode=config.default_mode))
