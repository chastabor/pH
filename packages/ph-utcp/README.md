# ph-utcp

*Tools published as [UTCP](https://utcp.io) manuals, as tools a pH agent can call.*

One row, `utcp`. Give it a manual call template — the object any UTCP harness
registers — and every tool the manual names becomes a pH tool: in the prompt,
through the pipeline, in the log, and in Code Mode's listing.

```yaml
- id: utcp
  name: utcp
  config:
    manuals:
      - template:
          name: weather
          call_template_type: cli
          commands:
            - command: weather-tools manual
        writable: []          # paths outside the workspace its commands may write
        tools:
          forecast:
            concurrency_safe: true
            effect_free: true
```

## A UTCP tool sees what its agent sees

A `cli` tool's command is started by `ctx.shell` for the agent making the call:
in that agent's workspace, under that agent's sandbox, and ended by the call's
own cancellation. That makes a UTCP tool bounded the way `bash` is bounded.

**`permissions-fs` does not apply to it.** Those rules govern calls through
`ctx.fs`, and a subprocess is not one. What bounds a UTCP tool's reads and writes
is the sandbox. A manual whose tool keeps state outside the workspace names those
paths in `writable`; under confinement the tool cannot write them otherwise.

## What a manual cannot say

UTCP describes a tool with a name, a description and two schemas. pH's scheduler
and policy rows also read whether two calls may overlap and whether a call has
effects. Each defaults to the careful answer — not safe to overlap, has effects —
and the manual's `tools` table states otherwise per tool.

A UTCP tool has no idempotency key and no `reconcile`, so after a crash mid-call
its outcome is unknown to the model, as `DESIGN.md` says of any such tool.

## A provider that is down

A manual whose discovery fails is skipped: a warning in the log, a line under
"UTCP tools" in `phern doctor`, and a note in the system prompt naming the
provider, so the model does not go looking for a tool that is not there. The
mount goes ahead with the manuals that answered.

## Protocols

`cli` is allowed by default, because it is the one this row runs through pH's
own doors. Another protocol — `http`, `mcp` — runs inside the harness process by
the UTCP library's own client, outside the sandbox and its egress rules. To use
one, install its plugin (`utcp-http`, `utcp-mcp`) and name it in `protocols`.

## Credentials

A template's `${VAR}` references are resolved from the environment when a call is
made. This row stores no values.

## Publishing a pH tool as a manual

`ph.toolcli` turns a package's rows into a command line with a `manual`
subcommand; see `docs/cookbook/adding-a-tool.md`. A manual written that way sets
`envelope: true`, and its tools then read to an agent exactly as they would
registered in-process: the same value, the same rendered text, the same error
codes.
