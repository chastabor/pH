# `ctx.subprocess` — spawning, with nothing implicit

**Module:** `ph/seams/subprocess.py` · **Row:** `subprocess-local` ·
**Consumers:** [`ctx.shell`](shell.md), the workspace tiers (git, jj, agentfs)
and the probes of the local sandbox backend

Two properties matter more than convenience.

## The spec is fully explicit

```text
SubprocessSpawnSpec = argv | cwd | stdio | env | grace_ms | timeout_ms | max_output | on_stderr_line | forwards_signals
```

**No hidden defaults.** A caller that did not think about the working directory
of a process the *model* asked for should have to say so — and a default
inherited from the harness process is exactly the wrong answer once
[`ctx.workspace`](workspace.md) starts handing out per-agent roots.

## The environment is scrubbed (I-4)

A child runs code the model wrote, so it does **not** inherit `*KEY*`,
`*SECRET*`, `*TOKEN*`, `*PASSWORD*`, `*PASSWD*` or `*CREDENTIAL*`. Those are the
default marks. `subprocess-local` takes `scrub` and `keep` to change them.

Credentials reach an adapter as a `CredentialRef` and are resolved at the edge
(I-3, [`ctx.credentials`](credentials.md)); **a child never needs the value**, and
the one that does is asking for it. `ctx.subprocess.env()` is the one function
that builds the environment of a child under the scrub of the deployment.
`ph-rlm`'s kernel does not spawn through this seam (its guest must not inherit
the terminal), and it applies the same scrub to the guest itself. `scrub_env` is
the default scrub, for a caller with no `ctx` to ask.

## Both bounds live here (P7-13)

A child is somebody else's program: it decides how much it prints and how long it
runs. Every caller that trusted it was one runaway command away from taking the
process down.

* **`max_output`** — the head of each stream is kept up to the cap, and
  `SubprocessResult.dropped` says how many bytes past it were read and not kept.
* **`timeout_ms`** — with `timed_out` on the result.

A child whose stderr is a report of how far it has got — a tool run with
`--progress` — names a listener (`on_stderr_line`). That stream is then handed
over a line at a time and not kept: `max_output` keeps the head of a stream, and
the line that matters most of a progress stream is its last.

The result is `stdout`, `stderr`, `exit_code`, `dropped`, `timed_out`. **A
non-zero exit is not a tool error**: `is_error` is about the tool raising, and a
command that failed usefully has an exit code the model should read.

## The surface

```text
await ctx.subprocess.spawn(spec)     # -> a handle
await ctx.subprocess.run(spec)       # -> SubprocessResult
ctx.subprocess.max_output
```

Every spawn is an **effect of its scope** (§4.9, I2), so a disposed scope
terminates and reaps the child — `grace_ms` is how long the polite signal is
given before the impolite one. A child whose wrapper passes no signal on
(`forwards_signals: false`, which `ConfinedArgv.spawn_spec` sets for a command
under `bwrap`)
gets the impolite one at once: asked first, a command stopped in its first
milliseconds was left running with no parent. The impolite one goes to the whole
group a stopped child was spawned as leader of, whether or not the child itself
is still there, so a member that ignored the asking does not outlive a leader
that heeded it. A child seen through — its output read to the end (`pump`) and
its exit awaited (`wait`), as `run` does (`SubprocessHandle.finished`) — is reaped
and its group left alone; a timeout or a cancel is a stop even when the leader had
exited. A lint fails the build on a raw
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
