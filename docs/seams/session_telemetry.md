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
ctx.session_telemetry.observe(...)         # mirror session events
ctx.session_telemetry.wants(channel)       # is anyone listening?
ctx.session_telemetry.ops
```

A `SessionTelemetryRecord` is `time`, `channel`, `severity`, `body`,
`attributes` — a log-record shape rather than a span shape, which is what lets it
map onto OTel logs without inventing a hierarchy.

`wants(channel)` exists so a producer can skip building a record nobody will
receive; with telemetry off, that is every record.

## One exception to mirroring

Ledger records mirror session events one-to-one, except that **only the first
`assistant/chunk` per step ships**.

A token-by-token export is thousands of records saying the same thing, and the
first is the one that carries the latency signal.

## What it does not do

* It does not trace. See above.
* It does not decide what is sensitive — a redaction row does, on the waterfall.
* **Nothing produces ledger records yet beyond the seam's own mirroring**, so what
  an operator sees is what the session log already says, and the `ops` channel has
  no producer at all. Stated because a telemetry seam that looked wired but
  reported nothing would be worse than one that is off.

## The row

```yaml
- id: telemetry
  name: session-telemetry
  config:
    enabled: false      # ph-base default: pay nothing until asked
```

## See also

[`ctx.diagnostics`](diagnostics.md) · `test_telemetry.py`,
`test_telemetry_otel.py`
