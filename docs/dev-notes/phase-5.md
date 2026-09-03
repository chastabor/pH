# Phase 5 — Long-running

**Status:** the sixteen rows are settled — fourteen landed, one dropped after measurement, one landed in half with the other half scoped as P7-07.

**Gate:** `ruff` + `ruff format` + `mypy --strict` across all five packages + 1 984 tests, green.

Every mode before this phase ties an agent's life to a connection. `ph -p` exits
with the turn, the TUI's root dies with the terminal, `--mode rpc` lives as long
as stdin. Phase 5 is the inversion: **a run that takes an hour must not stop
because a laptop lid closed.**

Almost everything below follows from that one sentence, and so does the phase's
recurring correction: *the thing that survives and the thing that watches it are
not the same thing.* A log survives; the process reading it may not. A socket
path survives; the socket it named may not. A schedule survives; the root that
would keep it may not. Each row that got this wrong got it wrong in the same
shape, and the last one is still open — deliberately, and printed.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P5-01 | The supervisor: a root that owns its own task and refers to no client | `ph_app/daemon/{framing,supervisor,server,client}.py` |
| P5-02 | One protocol over two transports — cursors, snapshots, a command journal | `ph_app/protocol.py` |
| P5-03 | Leases: one writer per session log, one daemon per socket (I-5) | `Supervisor._lease`, `SessionBusy` |
| P5-04 | The retry ladder, folded from the log rather than remembered | `ph_app/daemon/recovery.py` |
| P5-05 | Passivation: an idle root released, and rehydrated by the ordinary path | `Supervisor.passivatable`/`sweep`/`passivate` |
| P5-06 | Schedules over the log, at-most-once, with missed ticks coalesced | `ph/seams/schedule.py`, `Supervisor.tick` |
| P5-07 | Autonomous goals, and the three ways a run is allowed to end | `ph/seams/goals.py`, `ph/commands/autonomous.py` |
| P5-08 | `SessionPersistence` as a Protocol, and Turso behind it | `ph/persistence/{protocol,turso}.py` |
| P5-09 | OTLP as a *sink*, downstream of the redaction waterfall | `ph/seams/telemetry_otel.py` |
| P5-10 | `ph agents` — seven commands, every one a real exchange | `ph_app/agents.py` |
| P5-11 | Lingering detection: the socket that logout takes with it (I-6) | `ph/lingering.py`, `DaemonServer.check_reachable` |
| P5-12 | The non-guarantees, printed rather than documented (N5, I-2); these notes | `Supervisor.NON_GUARANTEES` |
| P5-13 | The ask direction: one `Peer` at both ends, `AskDesk` fanning one ask to every front end | `ph_app/daemon/{duplex,frontend}.py` |
| P5-14 | The TUI as a daemon client — `PHTuiApp` with a `daemon_argv` and no profile | `ph_app/tui/remote.py`, `ph_app/daemon/{launch,follow,projections}.py` |
| P5-15 | **Half.** A screen's *schema* travels and its `build()` cannot; declarative bodies are P7-07 | `ScreenSchema`, `screens/list` |
| P5-16 | **Dropped.** Built, measured, removed — the prompt layer already delivered it | — |

---

## The four things worth knowing

### 1. A watcher must never become the thing the work waits on

`Supervisor._run` is the sentence the phase is built on, and the useful thing
about it is what it does *not* contain: nothing in that loop mentions a
connection. Attaching subscribes a connection to a root's events; detaching
unsubscribes it; neither starts nor stops work. That is what makes "closing the
terminal leaves the root running" a property rather than a promise — you can
check it by reading the function.

The same inversion repeats one layer down. Watchers are held on the root by
token rather than on the connection by identity, because the root is what is
still there between attachments. A subscriber whose socket died is *dropped*
rather than raised through, and dropped where the subscriber list is, so the
policy has one owner. A client that cannot keep up is disconnected rather than
allowed to slow the work down.

### 2. Fold the state, don't remember it

Three rows arrived at this independently and one of them arrived at it twice.
The retry ladder's attempt count is folded from `supervisor/retry` records, not
held on the supervisor — because a daemon that crashed mid-ladder and came back
would otherwise start from zero and retry a failing turn for as long as the
process lives. Idle time is measured from the log's own last event rather than a
timer, so it means "nothing has happened" and survives a restart. Schedules are
`schedule/created` until a matching `schedule/cancelled`, in the log, so an
appointment outlives the process holding it.

The counter-pressure is real and was measured: a whole-log scan per read is
4.9 ms at 200 000 events, and `Root.status` is read for every root on every
`sessions/list`. So the shape that survived is **fold once at start, maintain
through the methods that append** — derived, but not re-derived.

The sharpest correction in the phase came from the same place. P5-04's first
draft reset the ladder on any `turn/end`, which the retry itself *manufactures*:
a re-entered `run()` finds an empty inbox, appends `turn/start` and
`turn/end{completed}`, and clears the counter that bounds it. Measured against a
persistently failing flush: **165 retries in two seconds, no give-up, the fold
pinned at one attempt, and the root reporting "idle"**. A marker only *success*
can write cannot be forged by the failure.

### 3. Stale state is a lie, and there are two ways to tell it

P5-01 handled the first: a socket path left by a crashed daemon makes every
client hang on a connect nobody will answer, so binding removes an unresponsive
one — while a *responsive* one is another daemon, refused rather than stolen.

P5-11 is the same sentence read the other way. `$PH_RUNTIME` lives under
`$XDG_RUNTIME_DIR`, which logind reaps at logout for a user who is not
lingering, so the door can be removed while the process behind it keeps running.
Every later client is told "no daemon socket" and to start one — which the
leases the first is still holding will refuse. The detection is a `(dev, inode)`
pair rather than an existence check, because the *second* thing that happens is
the person logging back in and starting the daemon that was recommended, which
puts somebody else's socket at the same path. An existence check reads that as a
recovery. It is two supervisors believing they own this user's roots.

Its diagnostic has no connected reader by construction — nothing can connect to
be told — so it goes to the daemon's log and, flushed, into every root's own
transcript. The reader is whoever opens the log afterwards asking why nothing
answered.

### 4. What the phase does not promise, and why that is printed

`ph doctor` and `ph agents doctor` both print an **isolation** section, because
rule 6 says a caveat only in the docs is a defect and the assumption being
corrected is made by the reply itself: `daemon/status` says "roots: 7" a few
rows above it, and seven roots reads as seven things that cannot hurt each
other. They are one `anyio` task each, in one process. There is no per-root
memory cap; a root that allocates without bound is OOM-killed as one process and
takes the others with it. CPU is shared. Restart is not rolling.

Two of those rows are corrections to sentences other rows had already written,
and both were found in the act of writing them down — which is the argument for
P5-12 existing at all:

* §3's N5 says pH does not "contain crashes between roots." **P5-04 landed after
  that sentence and does.** A root whose task raises climbs the ladder, gives up
  in its own log, and leaves every other root running. What stays uncontained is
  the *process* — a segfault in a C extension, an OOM kill, a `SIGKILL`. That is
  a narrower claim, and a reader who took the broad one would provision against
  the wrong failure.
* P5-06 argues that "a machine that reboots between Tuesday and Wednesday must
  not lose Wednesday's run." **True of the log; not true of the daemon.**
  `Supervisor.tick` iterates `self.roots`, and nothing re-mounts a root at boot —
  so the schedule is still in the log and nothing is watching it. A person can
  only discover this by missing a run.

The second is a gap rather than a design choice, and closing it is a row rather
than a sentence (P6-23). Enumeration is not what is missing — `stored()` lists
what is on record — but a `StoredSession` cannot say whether a log holds an
appointment, so choosing which roots to wake means reading every log; mounting a
root is a whole profile, workspace and lease, which is the cost P5-05 exists to
*release*; and a daemon that auto-mounted every stored root would hold every
session's lease and refuse the `ph -p` its owner runs next. The shape that
answers all three is an index of live schedules rather than a scan. Until then it
is asserted — `test_non_guarantees.py` fails the day somebody adds rehydration,
which is the correct way for a non-guarantee to end.

---

## What is left

Nothing in the phase, and two things it started.

**P5-15's other half.** A row contributes a screen's *schema* — id, label, order,
key — and cannot contribute its body, because `build()` returns a widget and the
front end may be in another process. So a screen pH ships is drawable by any pH
client and a screen a third-party row contributes reaches a remote front end as
nothing at all. That is a gate rather than a note
(`test_a_screen_this_build_cannot_draw_is_not_offered`), and P7-07's `ScreenData`
is what closes it.

**`repaired()` still closes a turn parked on a human as interrupted.** The ask is
in the log and `pending_approvals`/`pending_questions` fold it; nothing reads that
fold on *resume*, only across an attach. Leaving the turn open is unsafe until
something does, because a parked turn holds a dangling `tool/call` by
construction (B4 appends it before the pipeline gates it) and several providers
reject such a log outright. Closing it is the honest behaviour until the resume
half lands.

The one open gap, as above, is boot-time rehydration. It is printed in both
doctors and asserted in a test; it is not hidden.

**Where the phase went next.** Landing P5-13/14/15 made the daemon the only place
the harness runs, which is what a second front end needs — so `--mode web`
(P7-05), the browser upload (P7-06), the ephemeral daemon a UI starts for itself
(P7-08), `ask_user` (P7-09), `!!` (P7-10) and one daemon serving many
repositories (P7-14) all follow from this phase and are recorded in Phase 7.
