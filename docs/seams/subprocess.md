# `ctx.subprocess` — spawning, with nothing implicit

**Module:** `ph/seams/subprocess.py` · **Row:** `subprocess-local` ·
**Consumers:** [`ctx.shell`](shell.md), the git worktree tier, the runtime venv
builder, `ph-rlm`'s kernel

Two properties matter more than convenience.

## The spec is fully explicit

```text
SubprocessSpawnSpec = argv | cwd | stdio | env | grace_ms | timeout_ms | max_output
```

**No hidden defaults.** A caller that did not think about the working directory
of a process the *model* asked for should have to say so — and a default
inherited from the harness process is exactly the wrong answer once
[`ctx.workspace`](workspace.md) starts handing out per-agent roots.

## The environment is scrubbed (I-4)

A child runs code the model wrote, so it does **not** inherit `*KEY*`,
`*SECRET*`, `*TOKEN*` or `*PASSWORD*`.

Credentials reach an adapter as a `CredentialRef` and are resolved at the edge
(I-3, [`ctx.credentials`](credentials.md)); **a child never needs the value**, and
the one that does is asking for it. `scrub_env` is the function, and it is
exported because the shell and the kernel both spawn.

## Both bounds live here (P7-13)

A child is somebody else's program: it decides how much it prints and how long it
runs. Every caller that trusted it was one runaway command away from taking the
process down.

* **`max_output`** — an oversized buffer spills to
  [`ctx.spill_store`](spill_store.md) rather than into the model's context, and
  `SubprocessResult.dropped` says how much did not survive.
* **`timeout_ms`** — with `timed_out` on the result.

The result is `stdout`, `stderr`, `exit_code`, `dropped`, `timed_out`. **A
non-zero exit is not a tool error**: `is_error` is about the tool raising, and a
command that failed usefully has an exit code the model should read.

## Readers are offset-based

Not streams. A tool reading a long build log needs to say *"give me from byte
N"*, and an offset is a thing a model can put in an argument.

## The surface

```text
await ctx.subprocess.spawn(spec)     # -> a handle
await ctx.subprocess.run(spec)       # -> SubprocessResult
ctx.subprocess.max_output
```

Every spawn is an **effect of its scope** (§4.9, I2), so a disposed scope
terminates and reaps the child — `grace_ms` is how long the polite signal is
given before the impolite one. A lint fails the build on a raw
`subprocess.Popen` outside the seams, because the fiftieth plugin author will not
have read §4.9.

## What it does not do

* It does not confine — see [`ctx.sandbox`](sandbox.md). A scrubbed environment
  is not a boundary.
* It does not interpret a command string; that is [`ctx.shell`](shell.md).
* It does not choose a cwd for you. That is the point.

## See also

[`ctx.shell`](shell.md) · [`ctx.sandbox`](sandbox.md) ·
[`ctx.credentials`](credentials.md) · `test_subprocess.py`, `test_shell.py`
