# `ctx.sandbox` — confinement, and the refusal to pretend

**Module:** `ph/seams/sandbox.py` · **Rows:** `sandbox-policy` (definition),
`sandbox-local` (the bwrap backend) · **Consumers:** `ctx.shell`,
`ctx.subprocess`, `ctx.code_runtime` providers

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

## The surface

```text
ctx.sandbox.register_provider(provider)     -> Disposer
ctx.sandbox.confine(argv, policy)           -> ConfinedArgv     # raises if it cannot
ctx.sandbox.resolve_mode(...)               -> SandboxMode
ctx.sandbox.set_mode(session, mode)
ctx.sandbox.available()                     -> bool
ctx.sandbox.enforcement()                   -> Enforcement | None
```

## Providing a backend

```python
@runtime_checkable
class SandboxProvider(Protocol):
    enforcement: Enforcement
    def confine(self, argv: tuple[str, ...], policy: SandboxPolicy) -> ConfinedArgv: ...
```

One slot — two answers to "what confines this" is a contradiction. Typed rather
than duck-typed, for `WorkspaceProvider`'s reason: a backend whose method drifted
would fail at runtime inside a caller's `except` and be reported as "no
provider".

`sandbox-local` is the shipped one (bwrap, verified against a real kernel).
**Landlock and Seatbelt are still owed** — the Seatbelt profile is written blind
and deny-by-default so a rule somebody forgot fails closed.

## What only this seam can claim

`sandbox` is the **only** tier that bounds an absolute-path write (N2, E13).

Everything below it moves the *cwd*: a `worktree` bounds a relative write and
does nothing about `/etc/passwd`, and `permissions-fs` bounds tool calls through
`ctx.fs` and nothing about a code cell's raw `open()`. Those are honest, stated
limits of their layers — and the reason the ladder exists rather than one
mechanism claiming to be enough.

## See also

[`ctx.containment`](containment.md) · [`ctx.workspace`](workspace.md) ·
[`ctx.fs`](fs.md) · `test_sandbox_local.py`, `test_containment_ladder.py`
