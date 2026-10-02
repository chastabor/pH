# `ctx.schedule` — work a root will do later, folded from its own log

**Module:** `ph/seams/schedule.py` · **Row:** `schedule` · **Consumers:** the
daemon's scheduler (`Supervisor.scheduler`), `/autonomous`

Three kinds, one mechanism: `once` at a moment, `interval` every so often, and
`cron` on an expression.

What makes this a **seam rather than a timer** is that a schedule outlives the
process holding it.

## Everything is in the log, including the claim

A schedule is `schedule/created` until a matching `schedule/canceled`. A firing
is `schedule/tick`, appended **before** the work is delivered.

That ordering is A10's write-ahead applied to time. The tick and the prompt it
delivers are appended together and reach disk in the flush that ends the
scheduler's pass for that root (`Supervisor.tick`). What a crash costs depends on
where it lands:

| crash lands | on the next start | cost |
|---|---|---|
| before that flush | neither record is on disk, so the moment is still due and fires | a **late** run, never a lost one |
| after the flush, before the turn's first model request | the tick is on disk and the prompt is in the inbox, which `_start` rings | the run happens, late |
| during the turn | repair closes the turn as interrupted; the tick is on disk, so the moment is not fired again | **one skipped run** |

The model request is itself a barrier (`session-checkpoint-policy` flushes before
every request), so no row can bill a run twice. Putting the record before the
work is what rules out the fourth row, a turn that ran and left no tick, which
would fire again and bill twice. So: **at-most-once, deliberately**, and the log
says which.

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
ctx.schedule.index()   ctx.schedule.reindex()
```

A `Schedule` is `id`, `kind`, `spec`, `prompt`.

## Waking a session that is not mounted

A schedule belongs to a session, and a session that nobody has open is not
running to notice its own appointment. `schedule_index.py` answers *which stored
logs hold a live appointment and when each is next due*, so the daemon can wake
one — a small file read per pass, rather than a scan of every stored log.

That index is a **derived cache**: losing it costs a late wake, not a lost
schedule, because the schedule itself is in the session's log.

**The daemon sleeps until something is due; it does not poll.** It knows from the
index and its roots' logs when it next has work (`Supervisor.next_wake`): a
schedule due on a mounted root, or an appointment of a session it has not mounted.
It sleeps until then, and no pass comes sooner than a second after the last
(`PASS_FLOOR`). A schedule is made or canceled in a root's log, which the daemon
watches, and a root mounting moves the plan too: either wakes it to plan again.
**With nothing scheduled it has no deadline at all**, so making a schedule is the
opt-in. The first pass is at boot.

**A session the pass could not mount for its appointment is tried again.** The usual
cause is a lease: a `phern -p` running on that session at the appointed minute holds
its log, and that process letting go is nothing this daemon can be told. So the
pass plans a retry, `WAKE_RETRY_DELAYS` later (30 s, then a minute, two, five, ten,
and ten thereafter, one step per pass in a row that leaves an appointment behind),
and drops it the first time a pass wakes everything it meant to. That is a bounded
backoff, the shape of the retry ladder, not a cadence: nothing is planned while no
wake has failed. An appointment declined as too stale (`wake_within`) is not a
failure and is not retried.

**The sleep is on the wall clock** (`ph.wall_clock`). The loop's own deadlines run
on the monotonic clock, which stops while the machine is suspended, so a laptop
closed overnight used to wake the scheduler late by however long it slept. A
wall-clock timer fires on resume for anything that came due during the suspend.
`docs/dev-notes/linux-macos-differences.md` §9 has the per-platform detail and how
to check it by hand.

**A daemon rebuilds it from the logs when it can't vouch for itself** (S18), in the
background. Each pass surveys the index, and a rebuild starts only on one of these:
* **Incomplete.** Only a rebuild writes `complete: true`, and a writer keeps what it
  found. So a missing, corrupt or older file, or one a writer made from any of
  those, is incomplete.
* **An abandoned claim.** `create` leaves a locked mark in `schedules.claims/` until
  the new appointment is written. A write that fails, or a host that dies before
  it, leaves the mark unlocked.

In the ordinary case that is a guess at the first boot over a fresh `$PH_HOME`,
then the rebuild the first root's store vouches for, and none after it.

**It reads through a live root's store**, whatever its kind
(`SessionArchive.holding`). It keeps any entry for a log it didn't read, and any
entry a writer changed while it read.

**A rebuild that leaves the index in doubt is tried again for a reason, never on a
timer** (`Supervisor._rebuild`): a schedule made or canceled, one come due, or the
first store to read through. A rebuild that failed on a read-only `$PH_HOME` fails
the same way on the next pass, so it waits for something it depends on to have
moved. With nothing mounted the daemon reads its own `$PH_HOME/sessions` as JSONL
to wake what is due, and never vouches for that guess: the root it wakes brings
the store that can.

## What it does not do

* **Nothing here creates a schedule from the model's side by default.** There is
  no tool; `/autonomous` is what creates one, so a model cannot give itself
  appointments.
* It does not guarantee promptness beyond the daemon being up: it wakes at the
  moment a schedule is due, and a process that is not running fires nothing until
  it is.
* It is not cron. The OS already ships cron, anacron and systemd timers — pH
  schedules *inside a conversation* and is not trying to out-cron them.
  `wake_within` stays a knob defaulting to `None`, because a schedule attached to
  a conversation can be abandoned in a way a crontab entry cannot.

## Events

| event | |
|---|---|
| `schedule/created` | with its kind and spec |
| `schedule/canceled` | the matching end |
| `schedule/tick` | **before** the work is delivered |

There is no liveness record. Nothing here appends because time passed, and
whether a daemon is still watching is a question asked of the daemon:
`phern agents doctor` prints when the scheduler next wakes (`nextWake`). A log
written before 0.7.0 may still hold `schedule/heartbeat` records. They were
written `ignorable`, so the log still opens and nothing renders them.

## See also

[`ctx.goals`](goals.md) · [`ctx.jobs`](jobs.md) · `test_schedule.py`
