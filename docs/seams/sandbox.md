# `ctx.sandbox` — confinement, and the refusal to pretend

**Module:** `ph/seams/sandbox.py` · **Rows:** `sandbox-policy` (definition),
`sandbox-local` (the bwrap or Seatbelt backend, plus the egress proxy), `sandbox-allow` (what a
confined command may reach beyond its workspace), `sandbox-commands` (`/sandbox`) ·
**Consumers:** `ctx.shell`, `permissions-fs`, `containment`

Read [`ctx.containment`](containment.md) alongside this: that seam decides *which
rung* a deployment asked for, and this one is the mechanism at the top rung.

## The one rule that makes the seam worth having

**`confine()` never passes through.**

A caller asking to confine an argv is asking for a *guarantee*. A policy-only
provider that returned the argv unchanged would hand back an unconfined command
that looks confined, and every layer above would then reason about a boundary
that does not exist.

So the shipped definition-only provider is honest and useless: it resolves and
records policy, and raises `SandboxError` (`SANDBOX_UNAVAILABLE`) when asked to
actually confine.

`SandboxError` is a **denial, not a failure** — policy said this must be
confined, and pH refuses rather than running unconfined. Under Code Mode that
ends the program (C3) rather than being catchable.

## Modes and enforcement

```text
SandboxMode  = "read-only" | "workspace-write" | "danger-full-access"
Enforcement  = "full" | "partial"
NetworkMode  = "off" | "allowlist" | "full"
```

Mode resolution is **explicit > last logged `sandbox/mode` event > deployment
default** — so a per-call decision wins, a session-level change persists in the
log, and neither is guessed.

`enforcement` is a **descriptor, readable before any call**, and that is
load-bearing: `containment.strict` has to decide *at startup* whether this
deployment is actually confined, and a property discoverable only by confining
something would make that check "run a command and see", which is not something a
refusal-to-start can do. **`partial` is a refusal under strict, not a
downgrade.**

## What a confined command may reach (P6-38)

The workspace and its scratch are writable; everything else is the deployment's
to allow, and it says so in one place. `sandbox-allow` registers an `Allowances`
value — directories writable beyond the workspace, and a network posture with its
host list — and `ctx.sandbox.effective()` merges it into every policy inside
`confine()`, so `ctx.shell`, the probe and whatever confines next cannot drift
from one another. Without the row the seam **fails closed**: no extra directory,
no network.

```yaml
- id: sandbox-allow
  config:
    paths: [~/.cache/uv]            # writable beyond the workspace; `/` is refused
    network:
      mode: allowlist               # off | allowlist | full
      hosts: [pypi.org, files.pythonhosted.org, github.com, "*.githubusercontent.com"]
```

A profile that spells `config:` replaces the whole statement (the loader's patch
rule), so write the list you want rather than the difference. With no `config:`
the row ships `allowlist` over `DEFAULT_HOSTS` — package indexes and the places
their documentation lives — which is a starting point and the user's to trim.

**Two spellings for a host, and neither implies the other.** `example.com` is that
one host; `*.example.com` is every host under it and not the apex. An entry may
carry `:port`. `read-only` mode takes no extra directories: that mode is the
session saying nothing is writable, and a deployment's cache is not an exception
to a posture a person chose. A directory that does not exist is skipped rather
than bound — `bwrap` refuses to start over a missing bind source — and `ph doctor`
marks it `missing`.

**A caller can only ever narrow.** `SandboxPolicy.refuse_network` is the veto —
this command has no business online, whatever the deployment permits — and
`SandboxPolicy.network` is the *resolved* fact the backend reads, filled in by
`effective()`. There is deliberately no way to spell "give me the network": §6.5
caps a caller at what the deployment allows, so an asking field would have exactly
one reachable behaviour. `network` defaults to `False`, so a policy handed straight
to a backend without passing through the seam is closed.

## Network: a namespace, then a door

`off` is `--unshare-net` and nothing else. `full` is the host's network,
unfiltered. `allowlist` keeps the namespace and punches one door through it: a
filtering HTTP proxy on the host (`ph/seams/sandbox_egress.py`), listening on a
unix socket under `$PH_RUNTIME/sandbox/` that the sandbox reaches through its
read-only view of `/`, and inside the sandbox a `socat` shim that turns
`127.0.0.1:3128` into that socket so `HTTPS_PROXY` has something to point at.

**Each backend declares its door** (`sandbox_local.Door`), because the two
mechanisms are not the same shape. `bwrap` *unshares*: the sandbox has a loopback
of its own, so the shim above is the door. Seatbelt *denies*: a confined command on
macOS shares the host's loopback, there is no PID namespace for a shim to die with
— one left by the first design was still holding port 3128 an hour later — so the
proxy also listens on the host's `127.0.0.1:<port>` and the profile allows exactly
`(remote ip "localhost:<port>")` and nothing else. No shim, no `socat`. Measured:
the allowed port answers, the port beside it is `Operation not permitted`, DNS is
refused, and `curl` honouring `HTTPS_PROXY` gets the proxy's own 403. The loopback
listener exposes nothing the host does not already have — it tunnels only to hosts
the deployment allows, which every local process can reach directly — and costs
attribution only, since a local caller can name any agent in the proxy URL.

**The proxy decides; the namespace enforces.** A `CONNECT example.com:443` is
checked against the allowances — asked of the seam per connection, so `/sandbox
allow host` takes effect on the next request with nothing restarted — and refused
with the proxy's own 403 or tunnelled byte for byte. A command that ignores the
proxy variables (`ssh`, a raw socket) reaches nothing, which is the direction a
filter must fail in.

The bridge is **claimed only when it works**: `sandbox-local` starts the proxy and
sends one `CONNECT` from inside a confined command — `socat` through the shim under
`bwrap`, this interpreter dialling the port under Seatbelt — to
`ph-egress-probe.invalid`, the host the proxy always refuses, and registers the
bridge only if that 403 comes back. Without `socat` on a `bwrap` host, or with a
bridge that does not answer, `allowlist` means no network — and `network_posture()`
says so.

## Refusals are records

A boundary that refuses in silence teaches nobody anything, so a refusal is a
`sandbox/denied` event in the agent's session, carrying one sentence with the
`/sandbox` line that lifts it. The proxy knows the host it refused (`via: proxy`).
The kernel refuses in silence, so the filesystem half is read from what the
command printed — `Read-only file system`, `Network is unreachable` — and the
record says so (`via: output`), and only for a command that actually failed.

**Which sentences count is the backend's, not the seam's.** Those strings are
facts about one platform, so each backend implements the optional `DenialReader`
protocol over its own table: `Bubblewrap` owns `DENIAL_SIGNATURES` (`Read-only
file system`, `Network is unreachable`, the glibc resolver texts) and `Seatbelt`
owns `SEATBELT_SIGNATURES`, measured on macOS on 2026-09-07. Seatbelt has one word
for both boundaries — `Operation not permitted` — so its reader looks at the
*line*: a path on it is a file refusal (every tool measured puts one there), a bare
one is a socket when the command had no network, and a bare one under the host's
network is neither boundary this seam bounds and is not recorded. A backend with no
reader means no filesystem record, which is the honest answer; until the
measurement Seatbelt was one.

The TUI renders the sentence as a notice; the footer counts the session's
refusals; `/sandbox` lists them.

## `/sandbox`

```text
/sandbox                          # the posture in force, and this session's refusals
/sandbox allow host <host[:port]> # `example.com`, `*.example.com`, `example.com:443`
/sandbox allow path <dir>         # absolute or `~`; must exist; never `/`
/sandbox revoke host <host>
/sandbox revoke path <dir>
/sandbox network off|allowlist|full
```

An edit is **applied, then kept**. `Mount.reconfigure` re-applies the
`sandbox-allow` row with the new config — one slot released and refilled, no
provider swapped, no probe rerun, no proxy restarted, and an agent mid-command
notices only when its next command is bounded by the new statement. Then the row
is written to `$PH_HOME/profiles/<name>.d/sandbox.yaml`, a drop-in the next start
composes after the profile's own overlay. A drop-in rather than an edit of
`<name>.yaml`, because that file is the person's and a YAML rewrite drops every
comment in it. A deployment run from a profile *file* is applied and told the
change was not saved.

Because a row's config is replaced whole rather than merged, the first edit
freezes the whole host list for that profile — hosts added to pH's defaults in a
later release will not appear until the drop-in is deleted or edited. The file pH
writes says so at the top.

## The surface

```text
ctx.sandbox.register_provider(provider)     -> Disposer
ctx.sandbox.register_allowances(allowances) -> Disposer   # one slot; re-apply the row to change it
ctx.sandbox.register_egress(egress)         -> Disposer   # the backend row, once its probe held
ctx.sandbox.confine(argv, policy, agent=)   -> ConfinedArgv     # raises if it cannot; stamps the effective policy
ctx.sandbox.effective(policy, agent=)       -> SandboxPolicy    # the request with the allowances merged in
ctx.sandbox.permits(host, port)             -> bool             # what the proxy asks, per connection
ctx.sandbox.allowed_paths()                 -> tuple[Path, ...] # existing directories only
ctx.sandbox.network_posture()               -> str              # one sentence, for doctor and /sandbox
ctx.sandbox.read_denial(output, network=)   -> Denial | None    # asks the backend; None if it has no reader
ctx.sandbox.record_denial(denial, agent=)                       # appends sandbox/denied to the agent's session
ctx.sandbox.resolve_mode(...)               -> SandboxMode
ctx.sandbox.set_mode(session, mode)
ctx.sandbox.available                       -> bool
ctx.sandbox.enforcement                     -> Enforcement | None
```

## Providing a backend

```python
@runtime_checkable
class SandboxProvider(Protocol):
    enforcement: Enforcement
    backend: str          # the declared name — "bwrap", "sandbox-exec"
    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv: ...

@runtime_checkable
class DenialReader(Protocol):        # optional; a backend that knows its kernel's words
    def read_denial(self, output: str, *, network: bool) -> Denial | None: ...
```

One slot — two answers to "what confines this" is a contradiction. Typed rather
than duck-typed, for `WorkspaceProvider`'s reason: a backend whose method drifted
would fail at runtime inside a caller's `except` and be reported as "no
provider".

A backend reads `policy.egress` when it is set — the proxy's socket and the
port to dial, filled in by the seam — and builds its door around the command;
`ph/seams/sandbox_local.py` does it for `bwrap` (a `socat` shim on the sandbox's
loopback) and `sandbox-exec` (an `env` prefix and one `remote ip` line). A backend
owns its door in three pieces: `needs_loopback`, which the row reads *before* it
has confined anything to decide whether the proxy opens a TCP listener
(`enforcement`'s shape, and its reason); and `egress_blocker` plus `egress_probe`,
the prerequisite and the command that proves the door works (`DenialReader`'s
shape — behaviour the backend owns). So a third backend adds a door without
editing the probe.

`sandbox-local` is the shipped one, and **both backends are verified against a
real kernel**: `bwrap` on 2026-09-01, Seatbelt on 2026-09-07. The Seatbelt profile
was written blind and deny-by-default so a rule somebody forgot would fail closed —
and it did: Seatbelt matches the path the kernel *resolves*, and on macOS `/var`
and `/tmp` are symlinks into `/private`, so a workspace under `$TMPDIR` was refused
its own writes and the probe declined the tier rather than claiming it. The
profile now names canonical paths. **Landlock is still owed.**

## What is confined

Everything the harness spawns for an agent, from one policy:

| spawned by | bounded to |
|---|---|
| `ctx.shell` (`!!`, `tool-bash`, an autonomous gate) | the agent's workspace and scratch |
| `code-runtime-python` (the Python kernel behind `run_code`) | the same, per agent |

Both build `workspace_policy(workspace)`, so a deployment cannot end up with one
confined and the other not, and both decline the same way — no backend, or no
workspace to be the writable root, means no confinement rather than a passthrough.
Neither is gated on the containment tier: `ph doctor`'s containment section says
so in its own row, because the tree an agent works in and the commands the harness
wraps are genuinely different boundaries.

**Confining the kernel is what makes E9's sentence true.** `permissions-fs` bounds
tool calls through `ctx.fs` and tells operators that "a sandbox provider bounds
what a code cell can reach directly" — which was aspirational while the kernel was
spawned unwrapped. A cell's raw `open()` still reaches no rule, by construction
(N1); it is now refused by the OS instead.

Two consequences worth knowing:

- **fd 3 crosses the boundary.** The kernel's framed channel is an inherited
  descriptor, and `bwrap`, the egress shim's `sh -c … exec "$@"`, `sandbox-exec`
  and its `env` prefix all pass it through. Measured on both platforms, because a
  wrapper that closed it would look like a dead runtime rather than a lost channel.
- **Cancelling a confined cell has one cooperative route under `bwrap`, two under
  Seatbelt.** `bwrap` does not forward signals — it dies of `SIGINT` itself and
  takes the namespace with it — so there the kernel sends the `cancel` frame and
  skips the signal. `sandbox-exec` execs its target, so a signal lands on the cell
  (measured) and both routes stay; `ConfinedArgv.forwards_signals` is how the
  kernel knows which host it is on. The guest installs `SIGINT` as a loop callback
  either way, so both routes always needed the same running loop.
- **A kernel that never boots confines nothing**, and on macOS none did until
  2026-09-07: the guest cannot set `RLIMIT_AS` there and reported `RLIM_INFINITY`,
  which the host's lossless-integer codec refused, so every `boot-ack` was dropped
  as junk and every start waited out `boot_timeout`. The guest now reports `None`
  for a limit it could not apply and the host faults on an unreadable first frame
  (`test_boot_report.py`). The address-space limit itself is unenforceable on
  macOS, so `ph doctor`'s per-child limits row is read from the guests that
  actually started and says "address space not applied" rather than repeating the
  number the host asked for — E1's rule, one layer down from the tier table.

## What only this seam can claim

`sandbox` is the **only** tier that bounds an absolute-path write (N2, E13), and
the only one that bounds network egress — to the hosts `sandbox-allow` names, and
through the proxy only.

Everything below it moves the *cwd*: a `worktree` bounds a relative write and
does nothing about `/etc/passwd`, and `permissions-fs` bounds tool calls through
`ctx.fs` and nothing about a code cell's raw `open()`. Those are honest, stated
limits of their layers — and the reason the ladder exists rather than one
mechanism claiming to be enough.

## See also

[`ctx.containment`](containment.md) · [`ctx.workspace`](workspace.md) ·
[`ctx.fs`](fs.md) · `test_sandbox_local.py`, `test_sandbox_egress.py`,
`test_sandbox_allow.py`, `test_commands_sandbox.py`, `test_containment_ladder.py`
