# `ctx.jobs` — background work with a handle, a cancel and a completion

**Module:** `ph/seams/jobs.py` · **Row:** `jobs-local` · **Consumers:** the
subagent drive loop, `/refine`, the daemon's cadences

Anything long-running that is not a turn: a planner pass, a watcher, a background
build.

## A job is not a tool call

It **outlives the step that started it**, so it needs its own identity and its
own cancellation rather than borrowing the turn's. A tool call that tried to be a
job would be cancelled when its step ended, which is the opposite of the point.

## A job is an effect of the scope that owns it (I2)

It outlives the *step* — not the agent, and not the session:

| the work | its owner |
|---|---|
| a subagent's drive job | the delegation |
| a `/refine` pass | the session that asked |
| a daemon sweeper | the process |

So `start` takes a `scope=` like every other registration here, and **disposing
that scope cancels the job and drops its entry**. Without an owner the table only
ever grew, and the bound would have had to be a number somebody picked.

## Two halves of "this job is over"

Deliberately distinct, and picking the wrong one produces a wrong record:

* **abandoned** — the owning scope went away while the work was still running.
  Cancel it, then forget it.
* **released** (`forget`) — the owner knows the work is finished and wants the
  entry gone. **Forget it, cancel nothing.**

A job whose own body triggers its owner's teardown would otherwise report
`cancelled` for work that had in fact completed.

## Cancellation is cooperative

`Job.cancel()` sets a token. A body that never reads it **will run to
completion**, and `ctx.drain()` still waits for it.

That is a deliberate limit rather than an oversight: pre-emption in Python means
killing a thread or a process, and a seam that promised it would be promising
something it cannot deliver for an in-process body. A long body should read its
token at the points where stopping is safe.

## The surface

```text
ctx.jobs.start(work, *, scope=..., kind=..., label=...)   # -> Job
ctx.jobs.get(job_id)      ctx.jobs.list(...)
ctx.jobs.cancel(job_id)   ctx.jobs.forget(job_id)
ctx.jobs.bind(...)                                        # attach to a task group
```

A `Job` carries `id`, `kind`, `label`, `state`, `token`, `result`, `error` and
`release`.

`bind` is how the seam gets a task group to run in. **Without one bound, a job
runs inline** — which is what makes `ctx.jobs` usable in a one-shot `ph -p` run
where there is no supervisor, and is also why `Context.detach()` exists: an
admission path that started a job inline would block on the very work it was
trying to detach.

## Events

| event | |
|---|---|
| `job/started` | with its kind and label |
| `job/settled` | how it ended — completed, cancelled, failed |

## What it does not do

* It does not retry. A job that should retry does so in its body, where it knows
  what a retry means.
* It does not schedule. Work that should happen *later* rather than *now* is
  [`ctx.schedule`](schedule.md).
* It does not survive the process. A job is in-process work with an owner; a
  daemon that restarts has no jobs, and anything that must outlive a restart is
  in the log.

## See also

[`ctx.schedule`](schedule.md) · [`ctx.subagents`](subagents.md) · `test_seams.py`
