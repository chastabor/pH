# `ctx.session_telemetry` — records, redaction, and no span tracer

**Module:** `ph/seams/telemetry.py` · **Row:** `telemetry` (`enabled: false` by
default) · **Sink:** `session-telemetry-otel` (P5-09)

## The session log *is* the trace (§8)

There is deliberately **no span hierarchy**. Spans would be a second, lossier
account of what already exists event-by-event in the log, and **two accounts of
one run diverge**.

What this seam adds is *export*: a record stream a sink can ship somewhere.

## Redaction is an ordering guarantee, not a convention

> **Every record passes the `session-telemetry/record` redaction waterfall before
> any sink sees it.**

A sink registered as an ordinary listener *alongside* redaction could observe an
unredacted record by winning a race. A sink registered through `add_sink`
cannot, because it runs after the waterfall settles.

That is the whole reason `add_sink` exists rather than telling people to use
`ctx.on`.

## The surface

```text
ctx.session_telemetry.record(record)       # -> passes redaction, then sinks
ctx.session_telemetry.add_sink(sink)       # after the waterfall, by construction
ctx.session_telemetry.observe(session, events)  # mirror session events
ctx.session_telemetry.ops
```

A `SessionTelemetryRecord` is `time`, `channel`, `severity`, `body`,
`attributes` — a log-record shape rather than a span shape, which is what lets it
map onto OTel logs without inventing a hierarchy.

## A ledger record ships once the log holds its event

A ledger record names its event by `(session.id, event.seq)`. `session/event`
fires on the in-memory append, and a crash loses whatever no flush wrote. The
resumed log appends at its own length, so a record shipped straight off the
firehose could name one event while the log, after the resume, put another at
that seq.

So the row keeps a cursor per session and never listens to the append at all.
`SessionStore.flush` emits `session/durable` once every backend's flush has
returned, with the log's length as it was before the dispatch, and the row reads
the events up to there off the log. A flush that raised releases nothing. A
session disposed before a flush reached its tail never ships that tail, because
nothing says the log holds it. With no sink registered, nothing is read.

The cost is lag, not loss: a record ships at the next flush, which every step,
tool call and turn end makes. It still carries its event's own `time`.

**A new session's cursor starts at what its store already held**
(`Session.durable_length`), not at the end of its seed. A fork's prefix was its
parent's to ship, and a stored log was the run's that wrote it. A resume's repair
closers, which close a crashed turn, are in the seed but not yet in the store, so
this process writes them and the ledger ships them, with the reason the repair
gives.

## The shutdown tail ships before the mount lets go

A mount's rows unwind in reverse mount order, each layer before the ones it stands
on. So this row sits in **the log layer**, mounted right after `session` and beneath
every row that writes to the log:

1. `session-persistence`, the log's storage;
2. `telemetry`, this row, and its JSONL sink;
3. any export row, such as the OTel sink `session-telemetry-otel`;
4. `checkpoint-policy`, which decides when the log reaches disk;
5. then `agent`, and everything above it.

**Each row's plugin declares its place** (`log_layer=` on `@plugin`, one of
`LogLayer`'s `storage`, `telemetry`, `export` and `checkpoint`), and the composition
gathers the layer where its first row is written, in that order. `ph-base` writes
the first one right after `session`. So a deployment adds a row by naming it,
wherever it writes it: the OTel sink is
`{id: telemetry-otel, name: session-telemetry-otel}` in `daemon.yaml`, which puts it
at the end, and it mounts third.

Unwinding runs that list backwards. Every row above the agent goes, then the agent,
whose scopes append what they append on the way out, such as `workspace/disposed`,
a child's tombstone or a canceled turn's `turn/end`. The checkpoint policy goes
next, and its last barrier flushes every live session through the store.
Persistence is still mounted, so the flush writes, and `session/durable` says so.
This row hears it and ships the tail while both sinks are still there. The mount's
last write (`ph.persistence.write_on_unwind`) stays as a backstop and finds nothing
owed.

* **The barrier waits for what its flush set off.** The ledger ships off the flush
  path, in detached work the unwind would cancel: the JSONL sink takes a thread per
  record, and the OTel row shuts its provider down as it goes. So the barrier waits
  out that work (`Context.drain`) before the rows beneath it unwind, within the
  unwind's own budget.
* **Telemetry goes before persistence.** A flush after persistence had gone would
  have no backend to dispatch to, answer at once, and announce a write that never
  happened. Beneath persistence, this row is never mounted while that can be true.
* `test_config_owners` holds every shipped profile to writing the layer's first
  row beneath `agent`, with the OTel sink added the way a host adds it.

**A row that redacts or sinks records declares `log_layer="export"`.** One mounted
above the agent is gone before the tail ships, so the tail would skip that redactor
or miss that sink. No redacting row ships with pH. A crash still loses whatever no
flush had written, as it always has.

## One exception to mirroring

Ledger records mirror session events one-to-one, except that **only the first
`assistant/chunk` per step ships**.

A token-by-token export is thousands of records saying the same thing, and the
first is the one that carries the latency signal.

## A record carries what its type exports

A ledger record names its event, and of its data carries only the fields its type
exports: why the work failed, was refused or stopped. A `turn/end` ships its
`reason`, which says who canceled the turn, what a stop from outside said of itself,
or a failure's code and message. A declined compaction ships its `code` and
`reason`, a refused profile its `reason`, a denied sandbox boundary its `kind` and
`message`, a supervisor's retry or failure its `reason`. An operator looking for the
work that was cut or failed needs that without the log at hand.

Each field ships as `event.<field>`, so one attribute name, `event.reason`, finds
every reason whatever its event, and `event.type` says which kind it is. The OTel
sink ships it as `ph.event.reason`, in JSON text as it ships any value that is not
a scalar. It passes redaction like any attribute.

**The vocabulary declares the fields, not this seam.** ph-core's types are listed in
`EXPORTED_SESSION_EVENT_FIELDS` beside their other properties, and a package
declares its own with `declare_log_type(..., exported=)`, as `ph_clm` does for
`clm/declined`. So the seam never names a type or a payload key, and a bundle opts
in without ph-core learning about it. A type is listed rather than every field
named `reason` shipped, because a field's name does not say what it holds:
`request/header`'s `reason` says which prompt snapshot it is, and much of a payload
is what the model or a person wrote. `test_log_writers` holds each exported name to
a payload its writers write, so a misspelled one fails there.

## What it does not do

* It does not trace. See above.
* It does not decide what is sensitive — a redaction row does, on the waterfall.
* **Ledger records are the seam's own mirroring of the session log**, so what an
  operator sees there is what the log already holds. The `ops` channel says what
  the log cannot. Most of it comes through the module's `ops_record` helper, which
  is a no-op without the seam: a session open refused because another process
  holds it, a store that cannot take the I-5 lease, a workspace provider that
  failed and left an agent uncontained, and a log the mount's last write could not
  write. A flush that raised is heard by this row itself, from the store's
  `session/unwritten`, and ships as an `error` record with its `reason`. None of it
  is made from inside the sink loop, where a failing sink reporting itself would
  recurse. Stated because a telemetry seam
  that looked wired but reported nothing would be worse than one that is off.

## The row

```yaml
- id: telemetry
  name: session-telemetry
  config:
    enabled: false      # ph-base default: pay nothing until asked
    # path: the JSONL sink's file, $PH_HOME/telemetry.jsonl unless set
```

## See also

[`ctx.diagnostics`](diagnostics.md) · `test_telemetry.py`,
`test_telemetry_otel.py`
