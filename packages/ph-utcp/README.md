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

## A run in the background, for a tool whose work is long

```yaml
      - template: { name: notes, call_template_type: cli, commands: [{command: ph-notes manual}] }
        envelope: true
        state: notes
        background:
          command: [ph-notes, call]     # a `ph.toolcli` command, up to and including `call`
          label: notes index            # what the footer and the completion note call a run
          tools: [notes_index]          # the tools that run this way, by their own names
          eager: true                   # run each once before a root agent's first step
          wait_s: 5                     # how long a call waits for its run before answering
```

Every call is one command; nothing stays up and nothing listens on a socket. A
manual can name tools whose work is long, an indexing run, and the row then
starts such a tool itself and watches it:

- **Started as every command is**: by `ctx.shell`, for the agent whose tree the
  run is about, in its workspace and under its sandbox. The row adds the tool's
  name, `--progress` and `--args-json` to `command`, and the state directory is
  told as each call's is, in `$PH_TOOL_STATE`.
- **A job of the agent it works for** (`ctx.jobs`). It ends when the agent does
  and when the manual's row unmounts. A daemon that restarts picks no run up,
  and none needs to be, because what the tools build is on disk: the next run
  finds it and does only what is left.
- **Progress is read from stderr.** Under `--progress` the command writes JSONL
  there: `{"state":"working","done":4,"total":8,"what":"files"}` as it goes and
  exactly one last line, `done`, `busy` or `failed` (`ph.toolcli.Progress`),
  each saying what the run has spent so far where the tool counts it. That stream
  is everything the row knows of the run. While one works the footer says how
  far it has got ("notes index 120/539 documents").
- **A call starts the run and waits `wait_s`.** A call of one of `tools` starts
  a run with its arguments, or joins the same request already going. A run that
  ends in time is the call's answer: the run's own result. Past `wait_s` the
  command is asked with `--wait 0` and answers what it says while another run
  holds its work, `state: indexing`, with one line added to the text: "[The
  notes index is still working: 4/8 files.]". The run goes on.
- **A manual that passes a variable the scrub does not drop is said.**
  `pass_env` hands a credential to the manual's commands and no others — but
  only if the deployment's environment scrub drops its name everywhere else.
  One whose name it does not match is in every command's environment already,
  and `phern doctor` says so in this row's section.
- **A different request during a run is not queued.** It waits for the run
  that is going: if that ends within `wait_s` its own run starts, and otherwise
  it gets the same `state: indexing` answer. The agent calls the tool again
  once it is told the run ended.
- **The agent is told when a run ends**, with a note that waits for its next
  step and starts no turn. A run that finished says its count and what the
  tool's result would have told a caller that waited (`Progress.note`). A run whose command
  could not be started at all — a program that is not there, a sandbox that
  refused it — writes nothing, and is told with why; a call that was waiting
  for it fails with the same. One
  that failed says the tool's own sentence. One that found another run holding
  its work ended `busy` having done nothing; an agent answered "still going" is
  told to call again once that run has finished. One whose stream ended with no
  last line was killed or died, and is told as that: "stopped: its command ended
  (exit N) without finishing", with its last words. A call that took the run's
  result is told nothing more.
- **`eager`** runs each of `tools` once, with no arguments, before a root
  agent's first step, so an index is being built before anything asks for it.
  Started and not waited for: the step is not held up. Once for each tree, so
  an unchanged tree is not indexed again at the next step, and the agent is
  told of its end only if the run changed something.
- **One run a tree at a time, not one an agent.** A child builds its own only
  when it has a tree it writes for keeps: a worktree, an overlay. One in its
  parent's checkout, or granted only to read the project — `readonly-scratch`,
  and the two ephemeral kinds — uses its parent's `state`, and its commands run
  in that tree, which is how a `ph.toolcli` command is told its root. So a
  read-only fan-out is one index however many children it has. An answer about
  a tree other than the asking agent's own says so.

A background tool answers in `ph.toolcli`'s envelope, so the manual sets
`envelope: true`. In a deployment with no job runner each call does the work
itself, to its end.

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
the manual's commands is started — a tool call, or a run in the background —
and set on that command alone. An agent's `bash` does not have it.

## Publishing a pH tool as a manual

`ph.toolcli` turns a package's rows into a command line with a `manual`
subcommand; see `docs/cookbook/adding-a-tool.md`. A manual written that way sets
`envelope: true`, and its tools then read to an agent exactly as they would
registered in-process: the same value, the same rendered text, the same error
codes.
