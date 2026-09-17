# Phase 9 — The front end a person lives in

**Status:** eight rows, all landed.

**Gate:** `ruff` + `ruff format` + `mypy --strict` across 444 source files + 2 941 tests, green.

Phase 5 made the daemon the only place the harness runs, and Phase 7 put a second
front end on it. What neither did was ask what the terminal is like to *live in* —
to open twenty times a day, in the same repository, for months. This phase is the
four answers a person notices on the first morning and never thinks about again:
the colors are theirs, the arrow key brings back what they typed, opening pH in a
directory offers the work they did there, and closing the window does not leave a
supervisor behind.

The recurring correction is smaller than Phase 5's and turns up in five of the
eight rows: **a preference is not configuration, and a fact about the process is
not a fact about a session.** Four rows reached for a mechanism the repository
already had — a profile row, a history file, a per-session notice, a per-client
count — and in each case the mechanism was the wrong shape for the thing being
carried, in a way that only showed up when somebody drew it.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P9-01 | Themes in YAML or JSON, a `vars` palette, every fault reported at once | `ph_app/tui/themes/__init__.py` |
| P9-02 | `$PH_HOME/themes/theme-profile.yaml` — sole owner of the theme, written by `/theme` | `ThemeProfile`, `load_theme_profile` |
| P9-03 | Prompt history on the arrow keys, searched, and revealed in the transcript | `TuiState.prompt_history`, `tui/widgets/prompt.py` |
| P9-04 | `sessions/browse` filtered by working directory, tagged at the directory level | `ph_app/sessions.py`, `ph/session/session.py`, `ph/persistence/families.py` |
| P9-05 | The startup session picker | `PHTuiApp._offer_sessions` |
| P9-06 | An ephemeral daemon leaves when the last front end does | `DaemonServer.holds`/`spent`/`check_lifetime` |
| P9-07 | The lifetime on the wire, in the side panel, and in the doctor | `DaemonLifetime`, `daemon/lifetime`, `widgets/status.daemon_line` |
| P9-08 | The non-guarantees, `Implementation_Plan.md` §4, `DESIGN.md`, these notes | `Supervisor.NON_GUARANTEES` |

---

## The six things worth knowing

### 1. A preference file is not a profile row

The obvious place for a theme is a `cordis` row: pH composes profiles, rows carry
config, `/theme` writes config. It is the wrong place, and the reason generalizes
past themes.

A profile row is part of a **deployment** — the thing that decides which seams
mount, which tools exist, what the model may touch. A theme is a fact about the
person sitting at one terminal. Putting it in the profile makes changing a color
a change to the deployment, which means it composes, layers, and can be
overridden by a drop-in somebody installed; it also means the *daemon* would have
to carry it, because the daemon composes the profile — and after P5-14 the daemon
may not even be on the same machine as the terminal whose colors these are.

So `$PH_HOME/themes/theme-profile.yaml` is its own file with its own reader, and
it is the **sole** owner: `/theme` writes it and nothing else does. What made this
concrete was the migration question. The first draft carried a precedence ladder —
the profile row, then `tui.yaml`, then the new file — and the honest answer was
that there was nothing to migrate from. `tui.yaml` is a plugin profile and
`tui.json` is a preferences file, and conflating them is how a person's color
choice would have ended up deciding which tools mount.

The same split runs through the phase. `tui.json` holds what a person prefers;
the profile holds what a deployment is; and P9-06's `--keep-alive` is a third
case, below.

### 2. The mirror is the history

A prompt history looks like it wants a file. Shells have one; every editor has
one; the obvious implementation is `$PH_HOME/history` and an append.

The TUI already holds every prompt the person sent, because the transcript is
*built* from them — `TuiState.items` is a live mirror of the session log, which
P6-44 made a reader of the same shape as the daemon's. So the history is a fold
over rows that are already in memory, and it costs nothing to keep, is correct
across a resume without a second durability story, and cannot drift from the
transcript because it *is* the transcript.

The half that a file could not have done at all is the one that turned out to
matter most: because each row knows its `seq`, choosing one can **reveal** it in
the transcript rather than only retype it. "Where in this conversation did I ask
that" is a question a history file has no way to answer.

What it costs is stated where it would be assumed: the history is *this
session's*, not this person's. A resume gets that session's prompts back; a
reference-forked child gets only its own, because such a child holds only its own
events — the same fact `ph_app.sessions` already works around when a forked row
would otherwise render with no title.

### 3. A client's preference reaches the daemon through the argv it composed

`--keep-alive` is how long an auto-started daemon waits after the last client
leaves. It is a *preference* — it belongs with the theme, in `tui.json` — and it
has to be obeyed by a *process the person is not running*, which is the part that
decides the design.

There is exactly one route. The daemon cannot read the front end's `tui.json`: it
is a different process, started by a different command, and after P5-14 it may not
share a filesystem with the terminal at all. So the terminal reads its own
preference and spells it into the command line it composes for the daemon it
spawns — beside the `--ephemeral` that argv already carried.

That also settles where the *validation* lives, which the first version got
wrong. `phern --keep-alive 5min` forwarded the string as typed; the spawned daemon
refused it; and the daemon's output goes to the null device, so the person got a
UI that could not reach a daemon and nothing that said why. The parse belongs at
whichever end the person typed at, and what crosses is a bare number of seconds.

**And two flags describing one concept want to imply each other.** `--keep-alive`
without `--ephemeral` meant nothing at all, and the only thing between a person
and that surprise was a warning line they had to read. "Stay up five minutes after
the last client leaves" is a statement about *leaving*, so it carries the lifetime
with it. Merging them into a single flag was tried and is not available — typer
has no optional-value option, and the two commands express the lifetime in
opposite directions on purpose (`phern daemon` opts into leaving, `phern` opts
into staying), so one merged flag could not read the same on both.

### 4. When a scan is slow, the answer was a tag and not a faster scan

`sessions/browse` filtered by working directory starts as a comprehension over
the result. That is wrong before it is slow: `limit` takes the newest rows, so
filtering afterwards lets a directory's own sessions fall off the window behind
newer work elsewhere, and a repository somebody last touched a month ago lists as
empty on a busy machine.

Filtering *during* the scan fixes the correctness and makes the cost worse — the
walk no longer stops at `limit` files, because it has to read headers until it has
`limit` matches, and for any directory with fewer sessions than the limit that is
the whole store. Measured end to end: 9.66 ms for one directory's list at a few
hundred logs, on the path to the first prompt.

The fix is not a faster header read. A session's `family` — its lineage directory —
now carries the first six hex characters of the `sha256` of its `cwd`, so a
directory's whole lineage is selected by a prefix match on a **directory name**,
before a single file is opened or `stat`ed. 9.66 ms → 0.21 ms, 46×, and the
header check that remains is confirming a 24-bit match rather than sifting the
store.

Two method notes came out of measuring it. The first A/B reported a *negative*
saving, because the control arm was the pre-fix bug — it scanned 50 files, not 500
— and an A/B whose arms differ in two ways measures neither. And the
filename-prefix variant that was considered first is worse than the
directory-level one for a reason that is invisible until you write it down: a
per-file prefix still costs one `stat` per file to sort by mtime, and a
per-directory tag skips the directory entirely.

### 5. A probe is not a client, and the exit predicate is where that belongs

P9-06 makes an auto-started daemon exit when the last connection closes. The
launch path checks whether a daemon is there by connecting to the socket and
immediately closing it — `launch.listening()`, which every spawn, every
stale-socket check and every `_await_socket` poll runs. Reading that close as "the
last client left" stopped the daemon the poll had just declared ready, so the UI
that started it connected to nothing. It surfaced as three failures in
`test_daemon_launch.py`, none of which named the line responsible.

The first fix guarded the *teardown*: only a connection that had sent a frame
could ask for the exit. It worked, and it was at the wrong altitude, which a
review pass caught. `spent()` still answered "yes" from the daemon's first
instant, and two other callers ask it in the same window — the passivation sweep,
and a scheduled root finishing a turn. Guarding one caller leaves the next one to
find the same hole by another door.

So the term moved into the predicate: an auto-started daemon may not exit until
somebody has spoken to it or `launch.SPAWN_TIMEOUT` has elapsed, which is the
launcher's own number — past it, `_await_socket` has already given up, so there is
nobody left to protect. Every caller inherits it, the per-connection flag narrows
to the one job it genuinely has (a *client* leaving is what arms the keep-alive
window, and a probe is not one), and the case a per-connection guard could not
cover — a probe still open when the last real client leaves — closes as a
side effect.

It is in `spent()` and deliberately not in `holds()`. A daemon waiting for the
client that spawned it is not *held* by anything; nobody could name a reason. Put
among the reasons it would have appeared on the wire and rendered as
`held · starting` in a sidebar for the first thirty seconds of every session,
which is a word about the daemon's plumbing in the row that answers a person's
question.

### 6. What a frame must carry is decided by what the reader has to say

P9-07 puts the lifetime on the wire. The plan's model carried an `exits_at`, for a
sidebar countdown. It cannot be honest, and the mechanism is why: the keep-alive
window is armed when the last connection closes and cleared by the first frame of
the next one, so a client is *by construction* never connected while a deadline is
running — `phern agents doctor` cancels the very window it would have printed. The
field could only ever have carried `null`. What a connected client can say
truthfully is not "it leaves at 9:41" but "it leaves five minutes after you do",
so `keep_alive_ms` stays and `exits_at` was never built.

The opposite happened one field over. `holds` reports `client` whenever *anybody*
is connected, and the sidebar filtered that reason out before rendering — it is
always true of the window drawing the line, so printing it would spend a row of a
32-column panel telling a person they have a window open. With two terminals on
one daemon, both then read `exits on detach`, which is false for whichever closes
first. The subtraction needed a fact the frame did not carry, so `clients` is on
it, and a client *arriving* is announced as well as one leaving — the transition
nothing had been sent for.

Both are the same rule from opposite ends: a field earns its place on the wire by
what a reader can honestly say with it, and neither the producer's convenience nor
the plan's first sketch is the test.

---

## What the phase does not promise

Four of these rows touch the TUI, where the reflex is to reach for `ctx.settings`,
so the caveats are stated where they would be assumed (§5 rule 6) and two of them
are printed by both doctors:

* **A theme file is read at start and never again.** Editing one while pH runs
  changes nothing until the next launch; there is no watcher, and `/theme` picks
  from the catalog rather than re-reading. Stated on `ThemeCatalog`.
* **Prompt history is this session's**, per §2 above. Stated on
  `TuiState.prompt_history`.
* **The startup picker matches the `cwd` *string* a header recorded**, not the
  directory it names. One checkout reached through two paths — a symlink, `/tmp`
  against `/private/tmp` — lists as two directories with two sets of sessions.
  Resolving would be the wrong fix rather than a missing one: the header records
  where a session *said* it was working, and `/sessions` lists the store without
  the filter. Stated on `session_summaries`.
* **An auto-started daemon leaves when nothing needs it**, so a schedule keeps it
  resident and the ordinary case holds — but a `kill -9`, a logout reap or a
  reboot ends the process and the appointment with it until a UI opens again.
  Printed: `NON_GUARANTEES`.
* **`--keep-alive` is a floor on how long the process stays, not a ceiling.** Work
  that outruns it keeps the daemon up, which is the point. Printed:
  `NON_GUARANTEES`.

Two lifetime terms are not event-driven and fall to the sweep, which is written
down in `DaemonServer.holds` itself: a root parking on a person (`Root.status`
becomes `waiting` without the agent's status moving) and a keep-alive expiring,
which ends on a clock with nobody left in the process to notice. Each costs at
most one `sweep_every` of a stale sidebar line.

**No seam page, because nothing here adds a seam.** Said explicitly for the same
reason as the paragraph above it.

---

## What is left

Nothing in the phase. Two notes for whoever picks the area up next.

**The theme profile's `dark`/`light` pair was dropped, not deferred by accident.**
An earlier draft carried a per-appearance pair in the profile so a terminal could
switch with the system. Nothing read it, no verb toggled on terminal appearance,
and a field carried but never read is the shape rule 6 exists to forbid — so it
came out, and `TuiTheme.dark` (which Textual does read, per theme) is the only
`dark` left. Adding it back means adding the reader in the same row.

**The daemon's lifetime is now a fact three readers share**, through one builder:
the verb a front end reads at attach, the notification it is sent when the answer
moves, and the doctor's section. A fourth reader — the web front end — needs
nothing new; it is the same `DaemonSession`. What it would need is a decision this
phase did not take: whether a browser tab closing should count as a detach the
same way a terminal does, given that a tab can be restored.
