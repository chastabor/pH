# `ctx.shell` — bash over `ctx.subprocess`, confined when a backend exists

**Module:** `ph/seams/shell.py` · **Row:** `shell-local` · **Consumers:**
`tool-bash`, `!!` in the composer, `ctx.goals`' gates

A **thin layer, deliberately.**

Everything that makes a command safe — the scrubbed environment, the explicit
cwd, termination and reaping — already belongs to
[`ctx.subprocess`](subprocess.md). Everything that makes it *bounded* belongs to
[`ctx.sandbox`](sandbox.md).

What is left here is turning a command string into an argv and **asking for
confinement when the policy says to**.

```text
await ctx.shell.run(command, *, agent=..., ...)   # -> ShellResult
```

## Why it is a seam at all

Because the *asking* is the part that must not be forgotten. A caller that built
its own argv and went straight to `ctx.subprocess` would get the scrubbed
environment and skip confinement entirely — and nothing about that call would
look wrong.

Routing every shell command through one place means the sandbox consult happens
once, in code that is read once, rather than at every call site that happens to
run a command.

`workspace_policy(...)` is how it learns what to ask for: the agent's workspace
kind decides the `SandboxPolicy`, so a read-only tier and a writable one produce
different confinement without the caller choosing.

## What a caller gets

`ShellResult` carries the subprocess result plus `dropped` — the output bound is
[`ctx.subprocess`](subprocess.md)'s and surfaces here unchanged, so a model
running a chatty build sees a truncation notice rather than a silent tail.

`platform_shell()` picks the interpreter, which is the one place a Windows
difference lands rather than being spelled at each caller.

## What it does not do

* **It does not parse.** The command is handed to a shell, which is what a person
  and a model both mean by "run this". Anything that needs structure should build
  an argv and use [`ctx.subprocess`](subprocess.md) directly.
* It does not decide whether a command is dangerous — that is a policy row on
  `tools/pre-execute` returning `ask`, and `ph-stabilize`'s `destructive.py`
  parses the command properly rather than matching patterns against JSON.
* It does not promise confinement. If no backend is mounted, the command runs
  unconfined and the containment tier says so.

## See also

[`ctx.subprocess`](subprocess.md) · [`ctx.sandbox`](sandbox.md) ·
[`ctx.approval`](approval.md) · `test_shell.py`, `test_daemon_shell.py`
