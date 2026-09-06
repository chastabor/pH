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
ctx.jobs.start(work, *, scope=..., kind=..., label=...,
               slot=(key, limit), on_queued=...)          # -> Job
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

## Slots: more work than should run at once

`slot=(key, limit)` queues the overflow rather than refusing it. The caller asked
for all of it, and a refusal answers a question about *resources* with one about
*intent*.

`start` still returns at once and the handle is real — what waits is the **body**,
so a producer's admission, its records and its identity are untouched by how busy
the deployment is. A job waiting for a slot is `queued`, takes one in admission
order, and frees it on **every** ending — done, failed or cancelled — so one
failure cannot wedge everything behind it.

* **The key is the producer's to choose**, and is namespaced by job kind, so the
  obvious key — a session id — cannot collide with another producer's. The first
  caller of a key fixes that key's limit, and a later `(key, other)` joins the
  queue rather than resizing it: one producer disagreeing with itself is not
  something this seam can settle for it.
* **`on_queued` fires only when there is really a wait**, so a producer records
  *that it waited* instead of inferring it — the subagent provider writes `queued`
  to the parent's log there.
* **Cancelling a queued job stops the wait.** A body parked on a limiter is not
  running, so it reads no cancel token; `Job.cancel` cancels the wait itself.
  Without that a cancelled job sits behind work that may never settle and
  `ctx.drain()` waits for it — a cancellation that hangs the shutdown.
* **A queue is dropped once nobody holds a place in it**, by a refcount rather
  than a scan of the job table: `forget` is the owner's call, so a settled job may
  legitimately still be in that table and a scan would answer "somebody is still
  here" forever.

Here rather than in each producer because every one of those is a subtlety
discovered once and then re-discovered by the next one. A subagent provider, a
background build and a watcher queue the same way, and a second provider inherits
this instead of re-deriving it.

## The deployment's own bound

`slot=` is one caller's fair share. What the *machine* can carry is a different
question with a different owner, so it is row config rather than an argument a
producer passes — one a producer could forget to pass, or quote differently from
its neighbour:

```yaml
- id: jobs
  config:
    concurrency: { subagent: 8 }     # per job kind, across every producer
```

Empty by default. A shipped default naming `subagent` would be this seam — which
treats `kind` as a free string everywhere else — knowing one bundle's vocabulary,
and `ph-base` mounts no subagent provider for it to bound. The `rlm` bundle,
which does ship one, carries the number.

Both apply, and a job takes its producer's slot **before** the deployment's. That
order is what makes holding two safe: the other way round, one parent's queued
children would sit on every deployment slot while waiting for a bound of their
own and starve every other parent. No cycle can form either, because every job
acquires in that same order.

A kind that is absent from the table is uncapped, which is why the shipped
default names only `subagent`: a sweeper or a watcher queued behind a fan-out of
children is a housekeeping pass that stops happening. `ph daemon
--max-concurrent-children` overrides the one kind people meet, and does it by
patching this row so `--dump-config` reports the number actually in force.

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
