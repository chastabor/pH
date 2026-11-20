# ph-utcp

*Use tools published as [UTCP](https://utcp.io) manuals as tools a pH agent can call.*

A UTCP manual is a document in which a tool provider lists its tools. This
package has one row, `utcp`. A row is an entry in a pH profile that mounts one
plugin. You give the row a manual call template, which is the object that any
UTCP harness registers. The row then makes each tool in the manual a pH tool.
The tool appears in the prompt, goes through the tool pipeline, is recorded in
the log, and is listed in Code Mode.

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
        state: weather        # a directory in the calling agent's scratch directory, for what the tool builds
        writable: []          # shared paths outside the workspace that its commands may write
        tools:
          forecast:
            concurrency_safe: true
            effect_free: true
```

## A UTCP tool sees what its agent sees

The shell service (`ctx.shell`) starts the command of a `cli` tool for the agent
that makes the call. The command runs in the workspace of that agent and under
the sandbox of that agent. If the call is canceled, the command stops. A UTCP
tool therefore has the same limits as the `bash` tool.

**`permissions-fs` does not apply to a UTCP tool.** The `permissions-fs` rules
apply to calls through the filesystem service (`ctx.fs`). A subprocess does not
go through that service. The sandbox is what limits the files that a UTCP tool
can read and write.

## Where a tool keeps what it builds

Some tools build data about a file tree, such as an index or a graph. A tree is
the directory of project files that a tool works on. Such a tool does not change
the files in the tree. A read-only session must let this tool work, and must
still stop it from writing to the tree. Two settings, `state` and `writable`,
tell the row where the tool can write. Temporary files have their own rule.

- **`state`** is the name of a directory for the working state of the tool. For
  each call, the row sets `$PH_TOOL_STATE=<the calling agent's scratch
  directory>/<state>` on the command. The scratch directory is writable in every
  sandbox mode. The workspace is writable only in the `workspace-write` mode.
  Each session and agent has its own scratch directory, so two agents never
  share state. After a restart, an agent gets the same directory and finds what
  it built. A tool made with `ph.toolcli` reads the directory with `state_dir()`.
- **Temporary files** also go in the scratch directory. The shell seam sets
  `TMPDIR=<the agent's scratch directory>` on every confined command. The
  sandbox gives a confined command no writable `/tmp`. Without this setting, a
  model library that creates one temporary file would fail.
- **`writable`** lists paths outside the workspace that agents *share*, such as
  a grammar cache or model weights. A grant applies in every sandbox mode. The
  row refuses a grant that overlaps the workspace: a path inside the workspace,
  the workspace root, or a directory that contains the workspace. `phern doctor`
  reports the refusal. If the row accepted such a grant, the repository would be
  writable in a session that the user made read-only. For example, a session
  started in the home directory has a workspace that contains `$PH_CACHE`. That
  deployment gets no grant under the cache.

## Background runs, for a tool whose work is long

```yaml
      - template: { name: notes, call_template_type: cli, commands: [{command: ph-notes manual}] }
        envelope: true
        state: notes
        background:
          command: [ph-notes, call]     # a `ph.toolcli` command, up to and including `call`
          label: notes index            # the name of a run in the footer and in the completion note
          tools: [notes_index]          # the tools that run this way, by their own names
          eager: true                   # run each tool once before the first step of a root agent
          wait_s: 5                     # how long a call waits for its run before it answers
```

Each tool call is one command. No process stays running between calls, and no
process listens on a socket. Some tools do long work, such as an indexing run. A
manual can name those tools under `background`. The row then starts the command
for such a tool itself and monitors it. One such execution is a *background
run*.

- **The row starts a background run as it starts every command.** It uses
  `ctx.shell`, for the agent that owns the tree the run works on. The command
  runs in the workspace of that agent and under its sandbox. The row adds the
  tool name, `--progress`, and `--args-json` to `command`. The command gets the
  state directory in `$PH_TOOL_STATE`, as each call does.
- **A background run is a job of the agent it works for** (`ctx.jobs`). The run
  stops when the agent ends, and when the row of the manual unmounts. A
  restarted daemon does not resume a background run. It does not need to,
  because the tools keep what they build on disk. The next run finds that data
  and does only the remaining work.
- **The row reads progress from stderr.** With `--progress`, the command writes
  JSON Lines to stderr. While it works, it writes lines such as
  `{"state":"working","done":4,"total":8,"what":"files"}`. At the end it writes
  exactly one last line, with the state `done`, `busy`, or `failed`
  (`ph.toolcli.Progress`). If the tool counts what it spends, each line also
  gives the amount the run has spent so far. This stream is the only information
  the row has about the run. While a run works, the footer shows its progress,
  for example "notes index 120/539 documents".
- **A call starts a background run and waits up to `wait_s` seconds.** A call to
  one of the `tools` starts a run with the arguments of the call. If a run for
  the same request is already in progress, the call joins that run. If the run
  ends within `wait_s`, the result of the run is the answer to the call. If it
  does not, the row runs the command with `--wait 0`. The command then gives the
  answer that it gives while another run holds its work: `state: indexing`. The
  row adds one line to the text of that answer: "[The notes index is still
  working: 4/8 files.]". The background run continues.
- **The row does not queue a different request.** A call for a different tool,
  or with different arguments, waits for the run that is in progress. If that
  run ends within `wait_s`, the row starts a run for the new request. If it does
  not, the call gets the same `state: indexing` answer. The agent calls the tool
  again after the row tells it that the run ended.
- **The row tells the agent when a background run ends.** The row adds a note
  that the agent reads at its next step. The note does not start a turn. If a
  call waited and took the result of the run, the row sends no note. The content
  of the note depends on how the run ended:
  - *The run finished.* The note gives the count. It also gives what the tool
    result would have told a caller that waited (`Progress.note`).
  - *The command could not start.* For example, the program is not installed, or
    the sandbox refused it. Such a run writes nothing. The note gives the
    reason. A call that was waiting for the run fails with the same reason.
  - *The run failed.* The note gives the failure message of the tool.
  - *The run ended `busy`.* Another run held its work, so this run did nothing.
    If a call got the "still working" answer, the note tells the agent to call
    the tool again after the other run finishes.
  - *The stream ended with no last line.* The command was killed or it crashed.
    The note says so: "stopped: its command ended (exit N) without finishing".
    The note includes the last lines that the command wrote.
- **`eager`** runs each of the `tools` one time, with no arguments, before the
  first step of a root agent. The index is then already in progress before a
  call asks for it. The row starts the run and does not wait for it, so the step
  is not delayed. The row does this one time for each tree. It therefore does
  not index an unchanged tree again at the next step. The row tells the agent
  about the end of this run only if the run changed something.
- **One background run for each tree at a time, not one for each agent.** A
  child agent builds its own state only when it has a tree that it writes to
  permanently: a worktree or an overlay. Other children use the `state` of their
  parent. This applies to a child that works in the checkout of its parent. It
  also applies to a child that can only read the project: `readonly-scratch` and
  the two ephemeral workspace kinds. The commands of such a child run in the
  tree of the parent, because the working directory is how a `ph.toolcli`
  command learns its root. As a result, a group of read-only children uses one
  index, whatever the number of children. When an answer is about a tree that is
  not the tree of the asking agent, the answer says so.

A background tool gives its answer in the `ph.toolcli` envelope, so the manual
must set `envelope: true`. A deployment with no job runner has no background
runs. Each call then does all of the work itself before it answers.

## What a manual cannot say

A UTCP manual describes a tool with a name, a description, and two schemas. The
scheduler and the policy rows of pH need two more facts: whether two calls can
overlap, and whether a call has effects. A manual cannot give these facts. The
row uses the careful default for each: calls cannot overlap, and a call has
effects. To change a default for one tool, set it in the `tools` table of the
manual.

A UTCP tool has no idempotency key and no `reconcile`. If the process crashes
during a call, the model cannot know the outcome of that call. `DESIGN.md` says
the same about every tool of this type.

## A provider that is down

If discovery fails for a manual, the row skips that manual. The row writes a
warning to the log. `phern doctor` shows a line under "UTCP tools". The system
prompt gets a note that names the provider, so the model does not look for a
tool that is not available. The mount continues with the manuals whose discovery
succeeded.

## Protocols

By default the row allows only the `cli` protocol. `cli` is the only protocol
whose commands this row runs through the pH shell service and its sandbox. Other
protocols, such as `http` and `mcp`, run inside the harness process. The client
of the UTCP library runs them, outside the sandbox and outside its egress rules.
To use one of these protocols, install its plugin (`utcp-http`, `utcp-mcp`) and
add its name to `protocols`.

## Credentials

A template can contain `${VAR}` references. The UTCP client resolves them from
the environment each time a call is made. This row stores no values.

The environment of a command starts as the scrubbed environment of the
deployment. The scrub removes every variable whose name looks like a credential.
A template cannot get such a variable by naming it. The `pass_env` setting of a
manual is how a deployment lets one pass. `pass_env` lists variable names. Each
time the row starts a command of the manual (a tool call or a background run),
it reads those variables from the environment of the harness and sets them on
that command only. The `bash` tool of an agent does not get them.

**`phern doctor` reports a passed variable that the scrub does not remove.**
`pass_env` keeps a credential away from other commands only if the scrub removes
its name from them. If the scrub does not match the name, every command already
has the variable. `phern doctor` reports this in the section for this row.

## Publishing a pH tool as a manual

`ph.toolcli` turns the rows of a package into a command-line program with a
`manual` subcommand. See `docs/cookbook/adding-a-tool.md`. A manual made this
way sets `envelope: true`. Its tools then look the same to an agent as tools
registered in-process: the same value, the same rendered text, and the same
error codes.
