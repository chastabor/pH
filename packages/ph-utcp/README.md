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
        state: weather        # a directory of the calling agent's scratch, for what it builds
        writable: []          # shared paths outside the workspace its commands may write
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
is the sandbox.

## Where a tool keeps what it builds

A tool that builds something *about* a tree — an index, a graph — changes no
file of it, so a read-only session should not stop it, and should still stop it
writing the tree. Two settings say where such a tool writes:

- **`state`** names a directory for the tool's working state. Each call's command
  is told `$PH_TOOL_STATE=<the calling agent's scratch>/<state>`. The scratch is
  writable under every sandbox mode; the workspace only under `workspace-write`.
  It is per session and agent, so two agents never share state, and an agent
  that restarts is told the same directory and finds what it built. A tool made
  with `ph.toolcli` reads it with `state_dir()`.
- **Temporary files go in the scratch too.** Every confined command gets
  `TMPDIR=<the agent's scratch>` from the shell seam: the sandbox leaves it no
  writable `/tmp`, and a model library that makes one temporary file would
  otherwise fail there.
- **`writable`** names paths outside the workspace that agents *share*: a grammar
  cache, a model's weights. A grant holds under every sandbox mode. One that
  overlaps the workspace — inside it, the root itself, or a directory containing
  it — is refused and reported in `phern doctor`, because honoring it would make
  the repository writable in a session whose person said it is not. So a
  deployment whose workspace contains `$PH_CACHE` (a session started in the home
  directory) gets no grant under the cache.

## A host, for a manual whose tools are slow to start

```yaml
      - template: { name: notes, call_template_type: cli, commands: [{command: ph-notes manual}] }
        state: notes
        host:
          command: [ph-notes, serve]    # a `ph.toolcli` command, up to and including `serve`
          label: notes index            # what the footer and the completion note call it
          warm: [notes_index]           # tools it calls once when started unasked
          eager: true                   # start it at a root agent's first step
          idle_s: 900
```

A manual can name a command that stays up and answers its tools on a socket, so
what they load — an embedding model — is loaded once, and a long piece of work
goes on past the call that asked for it. The row keeps one running for each tree
an agent works in:

- **Started as every command is**: by `ctx.shell`, for the agent, in its
  workspace and under its sandbox. Each call is told the socket in
  `$PH_TOOL_HOST`; a command that finds nobody there runs by itself, so a host is
  never required for a call to be answered.
- **A job of the agent it serves** (`ctx.jobs`). It ends when the agent does and
  when the manual's row unmounts, leaves by itself when idle, and is started
  again by the next call. A daemon that restarts starts a new one; nothing is
  picked up, and nothing needs to be, because what the tools build is on disk.
- **One a tree, not one an agent.** A child builds its own only when it has a
  tree it writes for keeps: a worktree, an overlay. One in its parent's checkout,
  or granted only to read the project — `readonly-scratch`, and the two
  ephemeral kinds — uses its parent's host and its parent's `state`, and its
  commands run in that tree, which is how a `ph.toolcli` command is told its
  root. So a read-only fan-out is one index and one host however many children
  it has. An answer about a tree other than the asking agent's own says so.
- **`eager`** starts it at a root agent's first step with the `warm` tools to
  call, so an index is being built before anything asks for it — once for each
  tree, and without holding the step up. A host that leaves when idle is started
  again by the next call, not the next step, so an unchanged tree is not indexed
  again for nobody.
- **Progress and its end.** A host's tools report to `progress.json` in the state
  directory. While one works the footer says how far it has got ("notes index
  120/539 documents"); when it ends, and somebody is owed word of it — a call was
  answered "still going", or the host began the work unasked — the agent is told,
  with a note that waits for its next step and starts no turn.

Sockets are under `$PH_RUNTIME/utcp`, granted to the host and to each call. A
deployment with no job runner or no usable runtime directory runs every call by
itself.

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

A command's environment starts as the deployment's scrubbed one, which drops
every variable whose name looks like a credential, and a template cannot name
its way past that. A manual's `pass_env` is how a deployment says one may pass:
the names of variables read from the harness's own environment each time one of
the manual's commands is started — a tool call, or its host — and set on that
command alone.

## Publishing a pH tool as a manual

`ph.toolcli` turns a package's rows into a command line with a `manual`
subcommand; see `docs/cookbook/adding-a-tool.md`. A manual written that way sets
`envelope: true`, and its tools then read to an agent exactly as they would
registered in-process: the same value, the same rendered text, the same error
codes.
