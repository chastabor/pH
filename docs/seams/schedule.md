# `ctx.schedule` — work a root will do later, folded from its own log

**Module:** `ph/seams/schedule.py` · **Row:** `schedule` · **Consumers:** the
daemon's tick cadence, `/autonomous`

Three kinds, one mechanism: `once` at a moment, `interval` every so often, and
`cron` on an expression.

What makes this a **seam rather than a timer** is that a schedule outlives the
process holding it.

## Everything is in the log, including the claim

A schedule is `schedule/created` until a matching `schedule/cancelled`. A firing
is `schedule/tick`, appended **before** the work is delivered.

That ordering is A10's write-ahead applied to time, and the asymmetry is the
whole argument:

| | cost |
|---|---|
| tick recorded, then lost to a crash | **one skipped run** |
| tick delivered, then lost to a crash | a **repeated** run |

For a schedule whose payload sends a prompt, repeating bills twice and confuses
the transcript. So: **at-most-once, deliberately**, and the log says which.

## Missed ticks coalesce

`due_at` takes the **last claimed time** rather than counting from creation.

A five-minute schedule on a laptop that slept for three hours has thirty-six fire
times behind it. Delivering all thirty-six turns a nap into a stampede;
delivering the *oldest* works through the backlog for another three hours. So the
answer is **one tick naming the most recent due moment** — and the gap stays
visible in the log, because the previous tick is still there.

## The clock is a parameter

`now` is passed in rather than read here. A test can advance three hours without
sleeping, and a caller can drive the whole thing from one stamp.

## The surface

```text
ctx.schedule.create(schedule, session=...)     # -> Schedule
ctx.schedule.cancel(schedule_id, session=...)
ctx.schedule.claim(..., now=...)               # the tick, written ahead
ctx.schedule.live(session)                     # what is still scheduled
ctx.schedule.states(session)                   # the fold
ctx.schedule.heartbeat(...)   ctx.schedule.index()   ctx.schedule.reindex()
```

A `Schedule` is `id`, `kind`, `spec`, `prompt`.

## Waking a session that is not mounted

A schedule belongs to a session, and a session that nobody has open is not
running to notice its own appointment. `schedule_index.py` answers *which stored
logs hold a live appointment and when each is next due*, so the daemon can wake
one — a small file read per pass, rather than a scan of every stored log.

That index is a **derived cache**: losing it costs a late wake, not a lost
schedule, because the schedule itself is in the session's log.

## What it does not do

* **Nothing here creates a schedule from the model's side by default.** There is
  no tool; `/autonomous` is what creates one, so a model cannot give itself
  appointments.
* It does not guarantee promptness. A tick fires on the daemon's cadence, and a
  process that is not running fires nothing until it is.
* It is not cron. The OS already ships cron, anacron and systemd timers — pH
  schedules *inside a conversation* and is not trying to out-cron them.
  `wake_within` stays a knob defaulting to `None`, because a schedule attached to
  a conversation can be abandoned in a way a crontab entry cannot.

## Events

| event | |
|---|---|
| `schedule/created` | with its kind and spec |
| `schedule/cancelled` | the matching end |
| `schedule/tick` | **before** the work is delivered |
| `schedule/heartbeat` | the cadence is alive |

## See also

[`ctx.goals`](goals.md) · [`ctx.jobs`](jobs.md) · `test_schedule.py`
