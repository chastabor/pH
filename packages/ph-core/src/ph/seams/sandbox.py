"""`ctx.sandbox`: confinement, and the refusal to pretend.

One rule gives this seam its value: **`confine()` never passes through**. A
caller that asks to confine an argv asks for a guarantee. A policy-only
provider that returned the argv unchanged would return an unconfined command
that *looks* confined. Every layer above would then reason about a boundary
that does not exist.

So the `sandbox-policy` row in this module is policy-only. It resolves and
records the policy, and the seam raises `SANDBOX_UNAVAILABLE` when a caller
asks it to confine (the fail-closed posture of dsh). `sandbox-local` is the
backend. It arrived in P6-04 with `bwrap`, and P6-40 verified Seatbelt.
Landlock is not written yet. `sandbox-local` is the *only* tier that bounds an
absolute-path write (N2, E13).

The mode is resolved in this order: an explicit mode, then the last logged
`sandbox/mode` event, then the default of the deployment. So a per-call
decision wins, a session-level change persists in the log, and neither is
guessed.

**The deployment decides what a confined command may reach beyond its
workspace, and it states that in one place** (P6-38). `Allowances` holds the
directories that are writable beyond the workspace, and the network hosts that
are reachable through the egress proxy. The `sandbox-allow` row registers the
value, and this module merges it into every policy in `confine`. So the shell,
the code kernel and whatever confines next all get one boundary, and a change
to the row reaches the next command with no restart. Without the row the seam
fails closed: no extra directory and no network.

**The network has three states, not two.** `off` is `--unshare-net` and nothing
else. `full` is the network of the host, unfiltered. `allowlist` keeps the
namespace and adds one way out: a filtering proxy on the host, which a command
reaches from inside over a unix socket (`ph.seams.sandbox_egress`). A command
that ignores the proxy variables finds nothing to connect to. That is the
direction in which a filter must fail: the namespace enforces, and the proxy
only decides what it lets through.

A boundary that refuses in silence teaches nobody anything, so a refusal is a
**record**: `sandbox/denied` in the session of the agent. The record comes from
the proxy when the proxy refused a host, and from the output of the command
when the kernel refused a write. The TUI renders it with the `/sandbox` line
that lifts the refusal.

@module ph.seams.sandbox
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import Field

from ..agent.types import AgentHandle
from ..cordis import (
    ChildLimit,
    ChildReach,
    Context,
    Disposer,
    NarrowingRefused,
    Running,
    ScopedEntries,
    plugin,
    running,
)
from ..json import JsonValue, as_str
from ..keys import AGENTS, SANDBOX, TUI_STATUS
from ..paths import canonical, is_under
from ..session import Session, SessionBatch
from ..session.writers import log_writer
from ..tools.errors import FailureKind, HarnessError
from ..wire import WireModel, literal_lookup
from ._registry import claim_slot, contribute_item
from .subprocess import SubprocessSpawnSpec
from .tui_status import StatusField, StatusReading

_LOG = log_writer(__name__)

__all__ = [
    "DEFAULT_HOSTS",
    "DENIED",
    "NETWORK_MODES",
    "NOTHING",
    "SANDBOX_MODES",
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
    "allowance_path",
    "allowed_paths_of",
    "apply",
    "enforcement_of",
    "grants_apart",
    "held_under",
    "host_allowed",
    "narrower",
    "writable_paths",
]

log = logging.getLogger("ph.seams.sandbox")

SandboxMode: TypeAlias = Literal["read-only", "workspace-write", "danger-full-access"]
Enforcement: TypeAlias = Literal["full", "partial"]
NetworkMode: TypeAlias = Literal["off", "allowlist", "full"]

_MODE_READINGS: dict[str, StatusReading] = {
    mode: StatusReading(text=f"sandbox {mode}")
    for mode in ("read-only", "workspace-write", "danger-full-access")
}
"""One reading for each mode, built one time. There are three modes, and the
footer reads the field on every refresh."""

_MODE_RANK: Mapping[SandboxMode, int] = {
    "read-only": 0,
    "workspace-write": 1,
    "danger-full-access": 2,
}
"""How much each mode permits, so that two modes can be compared (N1).

The ranks are written out and are not derived from the order of `SandboxMode`.
The order of a `Literal` is a detail of its declaration, and this table is a
statement about what the kernel allows. The key type is `SandboxMode`, so mypy
refuses a key that is not a mode, and `SANDBOX_MODES` below is built from the
same alias."""


def narrower(requested: SandboxMode | None, resolved: SandboxMode) -> SandboxMode:
    """The more restrictive of what a caller asked for and what the session has.

    **A request can only narrow.** `SandboxPolicy.refuse_network` states that
    rule for the network, and this function is the same rule for the mode. The
    rule was not kept before. `effective` read the posture of the session and
    *replaced* the mode of the caller with it, so a logged `danger-full-access`
    widened a policy that said `read-only`. Widening is the one direction in
    which a confinement seam must never be wrong, and only the uniformity of the
    callers prevented it.

    `None` means that the caller has no opinion. The caller then takes the
    posture of the session unchanged. See `SandboxPolicy.mode`.
    """
    if requested is None:
        return resolved
    return min(requested, resolved, key=_MODE_RANK.__getitem__)


SANDBOX_MODES: Mapping[str, SandboxMode] = literal_lookup(SandboxMode)
"""Every `SandboxMode`, keyed by its own spelling. This is the read-side check
for a value that arrived as a `str`. See `literal_lookup`."""

NETWORK_MODES: Mapping[str, NetworkMode] = literal_lookup(NetworkMode)
"""Every `NetworkMode`, keyed by its own spelling. It is the same read-side
check.

It has its own docstring and does not share one with `SANDBOX_MODES` above,
because a string literal that follows two assignments documents only the second
assignment."""

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

The list holds package indexes and the sites that host their documentation. If
an agent cannot run `uv add` or read `docs.rs`, somebody switches the sandbox
off for it, and a sandbox that is switched off bounds nothing. **The list is a
starting point, not a judgment.** It seeds the `sandbox-allow` row. The user can
trim or extend that row from the profile or from `/sandbox`, and `phern doctor`
prints what is in force.

The entries are exact hosts, with one wildcard. `*.githubusercontent.com` is the
wildcard, because GitHub serves raw files, release assets and avatars from a
family of subdomains that changes. Every other entry is a single host, because
a wildcard widens the list, and somebody should have to write that on purpose.
See `host_allowed` for the two spellings.
"""


class SandboxError(HarnessError):
    """Confinement was requested and could not be provided.

    This is a denial and not a failure. The *policy* said that the command must
    be confined, and pH refuses to run it unconfined.
    """

    failure_kind: FailureKind = "denied"

    def __init__(self, message: str) -> None:
        super().__init__(message, "SANDBOX_UNAVAILABLE")


# ----------------------------------------------------------------- policy --


class Egress(WireModel):
    """The egress bridge: the one route from a confined command to the proxy.

    One value has two roles. The backend row **registers** an `Egress` on the
    seam after its probe has shown that a confined command can reach the proxy.
    `SandboxSeam.effective` then gives a **copy that carries the agent** to each
    policy that it merges. Before, a separate `EgressBridge` type held the same
    two fields for the registration, and three sites copied it into this type.

    It is a value and not the proxy object. These are the only facts that a
    backend needs, and a backend must not be able to reach the listener from a
    policy merge. A caller sets none of it: the socket of the proxy is a fact
    about the deployment, not about one command.
    """

    socket: str
    """The unix socket of the proxy on the host. A confined command reaches it
    through the read-only bind of `/`, so it needs no extra mount and nothing
    writable. The `socat` shim of `bwrap` connects to it. Seatbelt reaches the
    proxy on the loopback of the host and does not use it."""
    port: int
    """The port that the command dials on `127.0.0.1`, and the port that
    `proxy_url` names. Under a network namespace it is the fixed port of the
    shim inside the sandbox, which is private to that sandbox. Seatbelt shares
    the loopback of the host, so there it is the port of the proxy's own
    listener. See `needs_loopback` on the backends in `sandbox_local`."""
    agent: str | None = None
    """The agent that this command runs for. It is carried to the proxy as the
    user of the proxy URL, so that a refusal is recorded in the right session."""


class SandboxPolicy(WireModel):
    """The complete per-call confinement request."""

    mode: SandboxMode | None = None
    """The ceiling of the caller, or `None` for the mode that the session resolved.

    **This field holds two facts, and `effective` is where they meet.** On the
    way in, it is the most that the caller accepts. On the way out, it is what
    the backend enforces, because `effective` replaces it with `narrower(mode,
    resolve_mode(session))`. A request can only narrow. That is the rule of
    `refuse_network`, two fields below.

    The default is `None` and not `read-only`, because "no opinion" and "the
    tightest opinion" are different answers, and a default of `read-only` made
    them one answer. `workspace_policy` is the caller with no opinion, and N1 is
    the reason. It used to guess `workspace-write`. That capped a session that
    logged `danger-full-access` at workspace-write, *and* it raised a deployment
    that set `defaultMode: read-only` to workspace-write. One guess was wrong in
    both directions, and the footer reported a posture that was not in force.

    A policy that reaches a backend without passing through the seam keeps
    `None`. `writable_paths` then returns the extras alone. That is the closed
    direction, for the same reason as the default of `network`."""
    workspace_root: str | None = None
    """The one writable root under `workspace-write`."""
    writable_extra: list[str] | None = None
    refuse_network: bool = False
    """The veto of the caller: this command must not go online, whatever the
    deployment permits.

    **A request can only narrow.** On purpose, a caller has no way to ask for
    the network. §6.5 caps a caller at what the deployment allows. So a field
    that asked would have exactly one reachable behavior (no effect), and its
    docstring would claim otherwise. A caller *can* want less, and this field
    says so.

    Only `SandboxSeam.effective` reads this field. A backend reads `network`
    below."""
    network: bool = False
    """Whether this command has the network of the host, **resolved**.

    `SandboxSeam.effective` sets it from the `Allowances` of the deployment, and
    the backend reads it. The default is `False`. So a policy that goes straight
    to a backend without passing through the seam, as the tests of the backend
    do, is closed and not open.

    This field and `refuse_network` are two fields because they are two facts:
    what the caller asked of the seam, and what the seam resolved for the
    backend. They were one tri-state field before, and its documented third
    state was unreachable: `True` and `None` took the same branch everywhere."""
    egress: Egress | None = None
    """The egress bridge, when the effective network mode is `allowlist`. See
    `Egress`. The seam sets this field, and a caller never does."""


def writable_paths(policy: SandboxPolicy) -> list[str]:
    """Every path that this policy permits writing, in the order in which a
    backend binds them.

    **The rule is on the seam, beside the value that it describes.** Two
    backends need this set, and Landlock will be a third. A rule that is private
    to one backend is a rule that the next backend copies. `writable_roots`
    answers the same question for a `Workspace`. This function answers it for
    the policy that the workspace produced.

    The `match` is exhaustive over `SandboxMode` and is not an `if`. So a fourth
    mode fails the type check, and does not silently inherit "workspace and
    extras".

    `read-only` names no workspace root and **no implicit `/tmp`**. A temp
    directory that the caller did not ask for is a writable hole that nobody
    declared, and `redirection_env` already points the scratch files of the
    toolchain at a directory that was declared. `danger-full-access` returns
    nothing here because it is not a *set* of writable paths. It means
    "everything", and each backend expresses that in its own way.

    `None` is a policy that never passed through `effective`, which resolves the
    mode. It returns the extras alone, for the same reason as `read-only`.
    """
    extra = list(policy.writable_extra or ())
    match policy.mode:
        case None | "read-only" | "danger-full-access":
            return extra
        case "workspace-write":
            roots = [policy.workspace_root] if policy.workspace_root else []
            return [*roots, *extra]


# ------------------------------------------------------------- allowances --


class NetworkAllowance(WireModel):
    """Which network a confined command gets, and by which route."""

    mode: NetworkMode = "allowlist"
    """The network mode. `off` gives no network. `allowlist` gives the hosts
    below, through the egress proxy. `full` gives the network of the host,
    unfiltered, which is the mode that `danger-full-access` implies."""
    hosts: list[str] = Field(default_factory=lambda: list(DEFAULT_HOSTS))
    """The hosts that `allowlist` lets through. An entry has one of two
    spellings: see `host_allowed`."""


class Allowances(WireModel):
    """What a confined command may reach beyond its workspace and its scratch
    directory.

    This is the statement of the deployment. The `sandbox-allow` row registers
    it one time, and `SandboxSeam.effective` merges it into every policy. The
    defaults are the posture that the row *ships* with. A profile that sets the
    config of the row replaces the whole config (the patch rule of
    `ph.cordis.loader`). So a deployment that wants one more host writes the
    whole list that it wants, and not a delta.
    """

    paths: list[str] = Field(default_factory=list)
    """Directories that are writable beyond the workspace and its scratch
    directory. `~` is allowed.

    The list is empty by default on purpose. `redirection_env` already points
    the caches of the toolchain inside the scratch directory of the agent, so an
    ordinary build needs nothing here. This list is for a directory that a
    deployment *shares* across agents and accepts the consequences of, such as a
    package cache or a model store. Never list `/`: that is `danger-full-access`
    written to look like an allowlist, and the `/sandbox` command refuses it."""
    network: NetworkAllowance = Field(default_factory=NetworkAllowance)


NOTHING = Allowances(paths=[], network=NetworkAllowance(mode="off", hosts=[]))
"""What a deployment gets when no `sandbox-allow` row is mounted: no extra
directory and no network. This is the closed direction. It is a value so that
`effective` has no `if allowances is None` branch that could be wrong."""


def allowance_path(entry: str) -> str:
    """A directory as `sandbox-allow` names it, spelled as the kernel will match
    it: `~` is expanded and every symlink is resolved (`canonical`). The
    directories of the deployment and the narrower list of a child both use this
    one spelling."""
    return str(canonical(Path(entry).expanduser()))


def held_under(path: str, roots: Sequence[str]) -> bool:
    """Whether the canonical `path` is one of `roots` or is inside one. A
    directory of a narrowed child must be held under the directories of its
    parent, at admission and at every bind."""
    return any(is_under(Path(path), Path(root)) for root in roots)


def grants_apart(workspace_root: str, grants: Sequence[str]) -> tuple[list[str], list[str]]:
    """The writable grants of a caller, split into the grants that are honored
    and the grants that are refused.

    **Read-only is a barrier for the workspace, not for the working state of a
    tool.** A grant is honored under every mode (`writable_paths` returns the
    extras under `read-only` too). That is what lets a tool that changes no file
    of a repository still build an index of it. This function holds the other
    half of the rule. A grant that *is* the workspace root, is inside it, or
    contains it is refused. To honor it would make the repository writable in a
    session whose user said that it is not: a manual would only have to name the
    tree.

    Such a grant is refused whatever the mode. Under `workspace-write` the root
    is already writable, so the refusal loses nothing. And a rule that depended
    on the mode would silently widen when the posture changes in the middle of
    a session.

    Each grant is spelled as the kernel will match it (`allowance_path`) before
    it is compared, and the honored grants are returned in that spelling. The
    workspace root is taken as given, because the workspace seam hands out
    canonical roots.
    """
    root = Path(workspace_root)
    honored: list[str] = []
    refused: list[str] = []
    for grant in grants:
        spelled = allowance_path(grant)
        overlaps = is_under(Path(spelled), root) or is_under(root, Path(spelled))
        (refused if overlaps else honored).append(spelled)
    return honored, refused


def host_allowed(hosts: Sequence[str], host: str, port: int) -> bool:
    """Whether `hosts` names `host:port`.

    **There are two spellings, and neither implies the other.** `example.com`
    allows that one host. `*.example.com` allows every host *under* it and not
    the apex, because a wildcard that silently covered its apex would widen the
    list in a way that nobody wrote down. A deployment that wants both lists
    both.

    An entry may carry `:port`, and then it allows only that port. Without a
    port it allows any port. Matching is case-insensitive and ignores a trailing
    dot, because DNS treats a name with a trailing dot as the same host.
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
    """Split `host[:port]` into the host and the port. The port is `None` when
    the entry has none."""
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
    """One refused boundary, with the sentence that a user reads about it.

    `via` says how the refusal is known. `proxy` is certain: the proxy refused a
    host and knows which host. `output` is read from what a confined command
    printed. The kernel refuses in silence, so `Read-only file system` in stderr
    is the only account of the refusal. For that reason the sentence says
    "reported" and not "blocked".
    """

    kind: DenialKind
    via: DenialSource
    host: str | None = None
    port: int | None = None
    evidence: str | None = None

    def message(self) -> str:
        """What the TUI shows and the log keeps: one sentence that also says how
        to lift the refusal."""
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

    def record(self, agent: str | None) -> dict[str, JsonValue]:
        """The `sandbox/denied` payload. It carries the sentence, so every
        reader (the TUI, the trajectory, `phern agents attach`) prints the same
        account."""
        data: dict[str, JsonValue] = {
            "kind": self.kind,
            "via": self.via,
            "message": self.message(),
        }
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
    """A backend that can recognize the refusals of its own kernel in the output
    of a command.

    **The capability is optional, and it belongs to the backend and not to the
    seam**, because every string that such a reader matches is a fact about one
    platform. `bwrap` on Linux leaves `Read-only file system` and `Network is
    unreachable`. Seatbelt on macOS says `Operation not permitted` for both
    boundaries, and the path on the line tells them apart. A table of Linux
    sentences in this module would be the seam stating the wording of one kernel
    for every backend. `DescribingProvider` in `ph.seams.containment` exists for
    the same case, and for the same reason.

    A backend that has not been measured implements nothing, and the seam
    records nothing. That is the correct answer: the kernel refuses in silence,
    and a guess about *how* it words a refusal is not evidence.
    """

    def read_denial(self, output: str, *, network: bool) -> Denial | None: ...


# ------------------------------------------------------------------ seam --


@dataclass(frozen=True, slots=True)
class ConfinedArgv:
    """An argv that a real backend wrapped, and how much the backend actually
    enforces.

    `enforcement: "partial"` is reported and not hidden. Under
    `containment.strict`, a partial backend is a refusal to start, not a
    downgrade to accept quietly.
    """

    argv: tuple[str, ...]
    enforcement: Enforcement
    backend: str
    forwards_signals: bool = True
    """Whether a signal sent to this argv reaches the command inside it.

    **It is declared, because a caller cannot discover it by trying.** `bwrap`
    does not forward signals. It dies of `SIGINT` itself, and because it has
    unshared the PID namespace, everything inside dies with it. So a caller that
    signals a confined child to *interrupt* it destroys it. `sandbox-exec` execs
    its target and adds no namespace, so signals arrive normally.

    A caller that has its own cooperative stop (the `cancel` frame of the RLM
    kernel) reads this field to decide whether the signal is a second route or
    would destroy the child. `spawn_spec` gives the field to the spawn, and the
    spawn kills such a child and does not ask it to stop. The field has the
    shape of `enforcement`, for the reason that `enforcement` states: a caller
    cannot act beforehand on a property that it can discover only by confining
    something and watching the result.

    The default is `True`, which is the truth for an unwrapped argv. So a
    backend that adds no process between the caller and its child declares
    nothing."""
    policy: SandboxPolicy | None = None
    """The *effective* policy that the backend enforced: the request of the
    caller with the allowances of the deployment merged in. The seam stamps it,
    so a caller that reads the result knows what was actually bounded, and not
    only what it asked for."""

    def spawn_spec(
        self,
        cwd: Path,
        *,
        env: dict[str, str] | None = None,
        timeout_ms: int | None = None,
        on_stderr_line: Callable[[str], None] | None = None,
    ) -> SubprocessSpawnSpec:
        """This argv as a spawn, with `forwards_signals` carried along. This
        method is the one path from a confined argv to a spawn through
        `ctx.subprocess`, so no such spawn can drop the flag that decides how the
        child is stopped (`SubprocessSpawnSpec.forwards_signals`)."""
        return SubprocessSpawnSpec(
            argv=self.argv,
            cwd=cwd,
            env=env,
            timeout_ms=timeout_ms,
            on_stderr_line=on_stderr_line,
            forwards_signals=self.forwards_signals,
        )


@runtime_checkable
class SandboxProvider(Protocol):
    """A confinement backend.

    `enforcement` is a *descriptor* that can be read before any call.
    `containment.strict` must decide at startup whether this deployment is
    actually confined. If the property could be discovered only by confining
    something, that check would be "run a command and see", and a refusal to
    start cannot do that. Under strict, `partial` is a refusal and not a
    downgrade (E8).

    The Protocol is typed and not duck-typed, for the same reason as
    `WorkspaceProvider`. A backend whose method drifted would fail inside the
    `except` of a caller and be reported as *unconfined*. That is the one
    direction in which this seam must never fail silently.
    """

    @property
    def enforcement(self) -> Enforcement: ...

    @property
    def backend(self) -> str:
        """The name of this backend: `bwrap`, `sandbox-exec`.

        The name is *declared*, beside `enforcement`, and it is readable for the
        same reason. A consumer that wanted to tell a user what confines them
        had only `type(provider).__name__`. That is a Python class name, which
        is derived and not stated, and no seam contract promises it.
        `ConfinedArgv.backend` is this field, and not a literal that each
        `confine` repeats.

        Both are properties and not attributes, because both backends expose
        them as read-only, and a settable Protocol member refuses that.
        `CodeRuntime` explains the reason.
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
    """What the deployment lets a confined command reach, and the row that
    registered it. `None` reads as `NOTHING`: see `allowed`."""
    egress: Egress | None = None
    egress_by: Running | None = None
    """The egress bridge, after the backend row has proved that a confined
    command can reach the proxy. `None` with `allowlist` in force means no
    network at all, and `network_posture` says so."""
    _path_limits: ScopedEntries[tuple[str, ...]] = field(default_factory=ScopedEntries)
    """The writable directories that a narrowed child keeps (`restrict_paths`),
    keyed by the scope that each limit binds (`ScopedEntries`, as `ctx.skills`
    keeps its filters). A command reads only the limits along the
    `isolation_chain()` of its agent."""

    @property
    def allowed(self) -> Allowances:
        """What this deployment permits, with the unmounted case already answered.

        This is the one place that answers a missing `allowances` with
        `NOTHING`, which is the purpose of `NOTHING`. `NOTHING` was made a value
        so that no reader would have an `is None` branch that could be wrong,
        and then three readers each wrote one anyway.

        `network_posture` keeps its own `None` branch on purpose. "no
        sandbox-allow row is mounted" is a different sentence from "the row says
        off", and a user who reads `phern doctor` must be able to tell them
        apart.
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
        """Say what confined commands may reach. There is one slot: two
        statements of one boundary would contradict each other, and the boundary
        changes when the row is applied again.

        **The paths are made canonical here, because this is where they enter
        the seam.** A deployment writes `~/.cache/uv`. The kernel matches the
        path that it resolves, and so must the prompt boundary that
        `permissions-fs` draws from the same set (E6). Done here, the work
        happens one time per mount. Done in `allowed_paths`, it would happen for
        every confined command *and* for every gated write. The memoized helper
        that this replaced was written to avoid that cost, and the work belongs
        where the paths enter, not behind a cache. `expanduser` is applied here
        too: `~` is a spelling, and a consumer that compares against a literal
        `~` matches nothing.
        """
        return claim_slot(
            self.ctx.running_for(scope),
            self,
            "allowances",
            allowances.model_copy(update={"paths": [allowance_path(p) for p in allowances.paths]}),
            label="sandbox.allowances",
        )

    def register_egress(self, bridge: Egress, *, scope: Context | None = None) -> Disposer:
        """Say that a proxy is up and a confined command can reach it."""
        return claim_slot(
            self.ctx.running_for(scope), self, "egress", bridge, label="sandbox.egress"
        )

    def logged_mode(self, session: Session | None) -> SandboxMode | None:
        """The posture that this session recorded, or `None` if it never set one.

        It is separate from `resolve_mode` so that the *absence* of a record can
        be read. `None` means that nobody chose a posture, which is a different
        answer from "the deployment says workspace-write". `effective` used to
        read this method, and it let the logged mode replace the mode of the
        caller. It now reads `resolve_mode` through `mode_for`, because the
        default of the deployment is a posture too (N1). `resolve_mode` is the
        only caller of this method.

        **Who wrote it is this seam** (T6, closing F12). `sandbox/mode` is
        written only through the writer of this module, which is the writer that
        `WRITERS` grants the type to. So a third-party row that tries to write
        one is refused at the write. Two bypasses are not prevented at run time:
        a module that imports the writer of this module, and a module that calls
        `Session._append` directly. Both are deliberate acts, and
        `test_log_writers.py` fails on them.
        """
        if session is None:
            return None
        event = session.latest("sandbox/mode")
        if event is None:
            return None
        return SANDBOX_MODES.get(as_str(event.data.get("mode")))

    def resolve_mode(
        self, session: Session | None = None, *, explicit: SandboxMode | None = None
    ) -> SandboxMode:
        """An explicit mode wins over the log, and the log wins over the default
        of the deployment."""
        return explicit or self.logged_mode(session) or self.default_mode

    def mode_for(self, agent: str | None) -> SandboxMode:
        """The posture that the session of `agent` runs in. Without a session,
        it is the posture of the deployment."""
        return self.resolve_mode(self._session_of(agent))

    def set_mode(self, log: Session | SessionBatch, mode: SandboxMode) -> None:
        """Record a posture change. Pass a batch when the change is one of
        several records (S14)."""
        _LOG.append(log, "sandbox/mode", {"mode": mode})

    def mode_reading(self, session: Session) -> StatusReading:
        """`sandbox workspace-write`: what a confined command may write.

        The reading is labeled, because the preset beside it is *named* for the
        sandbox mode, and the two answer different questions. A preset says what
        runs without anybody being asked. This reading says what the kernel lets
        through at all.

        It is a bound method and not a lambda over the seam. So the
        `StatusField` that holds it keeps one reference to this object, and not
        a closure cell over everything else that was in scope when the row
        mounted.
        """
        return _MODE_READINGS[self.resolve_mode(session)]

    @property
    def available(self) -> bool:
        return self.provider is not None

    @property
    def enforcement(self) -> Enforcement | None:
        """How much the mounted backend actually enforces, or `None` when there
        is no backend.

        `containment.strict` reads it at startup and refuses on anything except
        `full`. That includes `None`: "no backend" and "a backend that bounds
        some of it" are both "not confined" to a deployment that asked to be
        sure.
        """
        return None if self.provider is None else self.provider.enforcement

    # ------------------------------------------------------------ allowances --

    def restrict_paths(self, paths: Sequence[str], *, scope: Context) -> Disposer:
        """Bind only `paths` of the extra writable directories for what runs in
        `scope` and in every scope inside it. This is for a child that an
        assigned profile narrowed (S7b).

        The paths are canonical (`allowance_path`), as the paths of the
        deployment are, and they are within the paths of the deployment. The
        `narrows` function of `sandbox-allow` checks that at admission.
        `writable_for` keeps a limit to what the reach of its parent still
        holds, so a child does not keep a directory that the deployment has
        stopped allowing. The limit is lifted with `scope`, as
        `ctx.skills.restrict` is.
        """
        # The table becomes empty when its last limit is removed. That keeps the
        # fast path of `_limits_of` for every command after the narrowed children
        # are gone.
        by = self.ctx.running_for(scope)
        return self._path_limits.add(by, tuple(paths), label="sandbox.path-limit")

    def writable_for(self, agent: str | None = None) -> tuple[str, ...]:
        """The extra writable directories that `agent` may use, canonical,
        whether or not they exist. These are the directories of the deployment,
        less what a narrowing along the scope of the agent holds back.

        The limits are read outermost first, and each keeps only what the limit
        before it left. So a grandchild holds no more than its parent holds
        *now*.
        """
        paths = tuple(self.allowed.paths)
        for limit in self._limits_of(agent):
            paths = tuple(path for path in limit if held_under(path, paths))
        return paths

    def _limits_of(self, agent: str | None) -> list[tuple[str, ...]]:
        """The path limits along the scope of `agent`, outermost first."""
        if not self._path_limits or not agent:
            # The common case, answered before the lookup. This runs on every
            # confined command (inside `effective`) and on every gated write.
            return []
        found = self._agent(agent)
        if found is None:
            return []
        return self._path_limits.gathered(reversed(found.ctx.isolation_chain()))

    def allowed_paths(self, agent: str | None = None) -> tuple[Path, ...]:
        """The extra writable directories that `agent` may use (`writable_for`),
        but only the directories that exist.

        A missing directory is skipped and not bound, because `bwrap` refuses to
        start when a bind source is absent, and every confined command would
        then fail. To skip is the closed direction: less is writable, not more.
        The diagnostic of `sandbox-allow` marks the entry `missing`, so the
        omission is visible.
        """
        # The paths are already canonical and expanded: `register_allowances`
        # did that one time, when it took them. This is only a filter. It runs
        # on every confined command (inside `effective`) and on every gated
        # write, and a `realpath` here would be a syscall per configured
        # directory per command.
        found = [Path(entry) for entry in self.writable_for(agent)]
        existing = tuple(path for path in found if path.is_dir())
        for path in found:
            if path not in existing:
                log.debug("ph.seams.sandbox: allowed path %s does not exist; not bound", path)
        return existing

    def permits(self, host: str, port: int) -> bool:
        """What the egress proxy asks for each connection, answered from the
        allowances as they are *now*.

        The allowances are read live and are not captured when the proxy starts.
        So `/sandbox allow host` takes effect on the next request, and neither
        the proxy nor the tunnels that it carries are restarted.
        """
        network = self.allowed.network
        match network.mode:
            case "off":
                return False
            case "full":
                return True
            case "allowlist":
                return host_allowed(network.hosts, host, port)

    def network_posture(self) -> str:
        """One sentence on what confined commands can reach, for `phern doctor`
        and `/sandbox`. It is written in one place so that the two cannot
        disagree."""
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
        """The request of the caller with the allowances of the deployment merged in.

        **This is the one place where the two meet.** A backend enforces what it
        is given and knows nothing about rows. A caller states what its command
        needs and knows nothing about the deployment. Because the merge is here,
        `ctx.shell`, the code kernel and the next confiner cannot drift from one
        another about what is allowed.

        The extra directories of the deployment are not added under `read-only`.
        That mode is the session saying that *nothing* is writable, and the
        cache directory of a deployment is not an exception to a posture that a
        user chose for one session.

        **The network and the egress bridge are decided in one chain**, and each
        case is stated one time. Before, there were two passes: an
        `if/elif/else` for the network, then a four-clause guard for the bridge.
        The guard tested two of the same conditions again, plus `not network`,
        which encoded the `danger-full-access` case while it looked like it said
        something else. The cases, in order:

        - The mode that means no confinement gets the network of the host.
        - A caller that refused the network gets nothing.
        - Otherwise the deployment decides. `allowlist` gives the bridge only
          when a bridge is actually mounted. `allowlist` without a bridge is no
          network, which is the closed direction and what `network_posture`
          reports.
        """
        # **The posture that the session chose is applied here** (J2).
        # `apply_preset` logs `sandbox/mode`, and before this nothing read it
        # back. `workspace_policy` hard-coded `workspace-write`, both
        # confinement callers passed that through unchanged, and the only reader
        # of `resolve_mode` was the footer. So the status bar reported
        # `read-only` and `danger-full-access` as in force while the backend was
        # given `workspace-write` in both cases. The two halves of a preset
        # disagreed, and the visible half was wrong.
        #
        # **The mode is bounded, not replaced** (N1). To replace it was wrong in
        # the widening direction: a logged `danger-full-access` overruled a
        # policy that said `read-only`. The bound is `resolve_mode` and not
        # `logged_mode`, because the default of the deployment is a posture too.
        # The disagreement of J2 survived for `defaultMode: read-only` with no
        # preset chosen: the footer said `read-only` and the backend got
        # `workspace-write`. A caller that states nothing takes the posture
        # unchanged. A caller that states something can only ask for less.
        mode = narrower(policy.mode, self.mode_for(agent))
        extra = list(policy.writable_extra or ())
        if mode != "read-only":
            extra += [str(path) for path in self.allowed_paths(agent) if str(path) not in extra]
        allowed = self.allowed.network
        network, egress = False, None
        if mode == "danger-full-access":
            network = True
        elif policy.refuse_network:
            pass
        elif allowed.mode == "full":
            network = True
        elif allowed.mode == "allowlist" and self.egress is not None:
            egress = self.egress.model_copy(update={"agent": agent})
        return policy.model_copy(
            update={
                "mode": mode,
                "writable_extra": extra or None,
                "network": network,
                "egress": egress,
            }
        )

    # --------------------------------------------------------------- confine --

    def confine(
        self, argv: tuple[str, ...], policy: SandboxPolicy, *, agent: str | None = None
    ) -> ConfinedArgv:
        """Wrap `argv` so that the kernel enforces `policy`, plus the allowances
        of the deployment.

        `agent` is the agent that the command runs for. It is carried to the
        proxy so that a refused host is recorded in the session of that agent,
        and is not lost.

        :raises SandboxError: when no backend can confine. This method never
            returns `argv` unchanged, because a caller must not be able to
            mistake absence for confinement.
        """
        if self.provider is None:
            raise SandboxError(
                "no sandbox backend is mounted, so this command cannot be confined; "
                "mount sandbox-local (P6-04) or run at the worktree tier, which "
                "bounds relative writes only"
            )
        effective = self.effective(policy, agent=agent)
        # Run as the row that registered the provider (P6-29). The layer is that
        # of the registration, for the reason that `CompactionSeam.engine_by`
        # states.
        with running(self.provider_by):
            confined = self.provider.confine(argv, effective)
        if not isinstance(confined, ConfinedArgv):  # pragma: no cover - provider bug
            raise SandboxError("sandbox provider did not return a ConfinedArgv")
        return replace(confined, policy=effective)

    # --------------------------------------------------------------- denials --

    def report_denial(
        self, confined: ConfinedArgv, streams: Sequence[str], agent: str | None
    ) -> None:
        """Read what the output of a confined command says was refused, and
        record it.

        **The rule of the seam is implemented in the seam.** The docstring of
        this module says that a refusal is a record and not a silence, and
        `ctx.shell` had the only implementation. So when the RLM kernel became
        the second thing that is confined, it adopted the boundary and not the
        rule. A cell that the operating-system kernel refused produced a
        traceback with no `/sandbox allow path` line, while the identical
        refusal from `tool-bash` produced one. Now there is one statement, and
        both callers use it.

        `streams` are tried in order, and the first refusal wins. Pass stderr
        before stdout, because stderr is where these sentences come from. **Pass
        only the output of a command that actually failed.** The signatures
        match nothing that an ordinary command prints on success, so a `grep`
        that finds "Read-only file system" and exits 0 has not been refused
        anything.
        """
        network = bool(confined.policy.network) if confined.policy is not None else False
        for stream in streams:
            denial = self.read_denial(stream, network=network)
            if denial is not None:
                self.record_denial(denial, agent=agent)
                return

    def read_denial(self, output: str, *, network: bool) -> Denial | None:
        """What the mounted backend recognizes in the output of a confined
        command.

        Returns `None` when no backend is mounted, or when the mounted backend
        has not been measured on this platform: see `DenialReader`. The seam
        asks, and not the caller, so that a consumer never learns which backend
        answered (I5). The call runs as the row that registered the backend
        (P6-29), like every other provider body.
        """
        if not isinstance(self.provider, DenialReader):
            return None
        with running(self.provider_by):
            return self.provider.read_denial(output, network=network)

    def record_denial(self, denial: Denial, *, agent: str | None) -> None:
        """Put a refused boundary in the log of the agent, where the TUI will
        find it.

        The record goes to the *session* of the agent, looked up through
        `ctx.agents`, because that is the transcript that the user reads. A
        denial may have no agent to assign it to: a connection that arrived
        without the proxy user, or an agent that is already gone. Such a denial
        is written to the process log and not dropped, so that the fact is
        recorded somewhere.
        """
        session = self._session_of(agent)
        if session is None:
            log.warning(
                "ph.seams.sandbox: %s (no session to record it in; agent=%r)",
                denial.message(),
                agent,
            )
            return
        _LOG.append(session, DENIED, denial.record(agent))

    def _session_of(self, agent: str | None) -> Session | None:
        """The session that the records and the posture of an agent belong to.

        It has two readers. `record_denial` records a denial where a user will
        look for it. `mode_for` reads the mode that the session chose (J2), for
        `effective` and for the `sandbox-policy` of a child. The second reader
        is why the lookup is here and is not passed through every caller.
        `confine` already takes the agent id, because a denial must land in the
        transcript of that agent, and the mode is a fact about the same session.
        A `session=` parameter would have to be added to `ShellService.run`, to
        the confiner of the kernel and to everything that builds one, to carry
        something that the seam can already ask for.
        """
        found = self._agent(agent)
        return found.session if found is not None else None

    def _agent(self, agent: str | None) -> AgentHandle | None:
        """The live agent by id. A command runs for the session and the scope of
        this agent."""
        agents = self.ctx.get(AGENTS)
        if agents is None or not agent:
            return None
        return agents.get(agent)


def enforcement_of(ctx: Context) -> Enforcement | None:
    """How much confinement this deployment actually has. `None` means none at all.

    The question is asked of a seam that may not be mounted, as `workspace_of`
    asks its question. It is written one time because the copies had begun to
    *disagree*. `permissions-fs` read "a backend exists", while `containment`
    read "a backend that says `full`". So with a `partial` backend, the reach
    sentence of E9 would have told an operator that a sandbox bounds their code
    cells, while the refusal of E8 told them that a partial boundary is not
    confinement at all. Two statements to the operator about one fact, in two
    packages, is the drift that this seam exists to prevent.
    """
    seam = ctx.get(SANDBOX)
    return None if seam is None else seam.enforcement


def allowed_paths_of(ctx: Context, agent: str | None = None) -> tuple[Path, ...]:
    """The extra writable directories that `agent` may use, asked of a seam that
    may not be mounted.

    This is for `permissions-fs`. Its `outside-workspace` rule prompts about
    what the backend would refuse. So it must stop prompting about what the
    backend now allows, or the prompt boundary and the enforced boundary
    describe two different sets (E6). The answer is per agent for the same
    reason: a child that is narrowed to fewer directories is asked about the
    directories that it lost.
    """
    seam = ctx.get(SANDBOX)
    return () if seam is None else seam.allowed_paths(agent)


class Config(WireModel):
    """Row config for the policy-only sandbox provider."""

    default_mode: SandboxMode = "read-only"


def narrows(_mounted: Config, asked: Config, reach: ChildReach) -> ChildLimit:
    """The `sandbox-policy` row of a child may start the child in a narrower
    posture than its parent runs in, and never in a wider one (S7b). `read-only`
    is the posture that a child holds on its own:
    `SubagentService.resolve_profile` refuses it `write` access."""
    seam = reach.ctx.get(SANDBOX)
    parent = seam.mode_for(reach.agent) if seam is not None else "danger-full-access"
    mode = asked.default_mode
    if narrower(mode, parent) != mode:
        raise NarrowingRefused(
            f"its sandbox-policy lets a child run {mode}, where its parent runs {parent}"
        )
    return ChildLimit(read_only=mode == "read-only")


@plugin("sandbox-policy", affects="environment", config=Config, narrows=narrows)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the sandbox seam with policy resolution and no backend."""
    seam = SandboxSeam(ctx=ctx, default_mode=config.default_mode)
    ctx.provide(SANDBOX, seam)
    # The posture is published as a reading, not as a field on every status
    # frame. Only this row can answer it: the mode is the default of this config
    # until the log overrides it, and `resolve_mode` holds that precedence. A
    # front end that had to be *told* the posture needed a wire field of its
    # own, plus a projection, plus a reader, for a string that this row already
    # has.
    #
    # The id is `sandbox-mode`, because `ph.commands.sandbox` already holds
    # `sandbox` for the refusal *count*. These are two facts about one seam, and
    # the registry refuses a second claim on one id. It does not let one claim
    # overwrite the other.
    contribute_item(
        ctx,
        TUI_STATUS,
        StatusField(id="sandbox-mode", read=seam.mode_reading, order=11, slot="session"),
        label="sandbox(mode)",
    )
