# UTCP tool port — phern calls tools through UTCP, starting with code graph and text index

*2026-11-10. A review of `ph-code-graph` and `ph-text-index` against the UTCP Python
library cloned at `sources/python-utcp` (utcp 1.2.0, utcp-cli 1.1.5), and a plan to
make UTCP the way phern reaches both packages' tools and any tool published later.
Revised twice the same day on the user's direction; every decision that was the user's
has been taken.*

**Status, 2026-11-10.** U1 to U4 are built: the command line and the file lock, the
manual and the lenient decoder, the `ph-utcp` row, and the code graph reached through
it. Of U7, the `index` block's `paths`, `include` and `exclude` are built on both rows,
with `exclude` on both tools' arguments; `on_mount` waits for the worker that reads it.
Not started: U5 (the worker), U6 (the text index through `ph-utcp`), U8 and U9 (the two
embedder stages). The text index is still registered in-process and works as before.
The READMEs, the shell seam page and the tool cookbook describe what is built. What
each built step settled, and what it found that the plan had wrong, is under "What U1
to U4 settled" below; what was learned about the worker is under "Before U5".

## Before U5

Three facts found while building U3 and U4 that the worker design above gets wrong or
leaves open. U5 starts from these.

- **A worker cannot be started from inside a tool call.** A confined command runs in
  its own process namespace, and everything in it is killed when the command ends. So
  "the worker starts on the first tool call" cannot mean the command starts it. The
  session's row has to: `ph-utcp` needs a hook that runs before a manual's call, in the
  harness, for the calling agent.
- **There is no agent when a row mounts.** "Through the same door as any command for
  that agent" has no agent to name at mount, and `ctx.shell` confines only a command
  run for an agent that holds a workspace. A worker started at mount is confined by a
  policy the row states itself: the session's root as the workspace, plus the paths
  the manual grants.
- **Where the socket goes.** The caller and the worker are in two sandboxes, so the
  socket has to be on a path both can reach, and short enough for a socket address
  (108 bytes). A directory under `$PH_RUNTIME`, granted to the manual, fits both; the
  index directory under `$PH_CACHE` does not fit the length in a test tree.

A design that follows from them, not yet built: `ph.toolcli` gains `serve`, a process
that mounts the profile once and answers calls on that socket; `call` uses a running
host when there is one and runs directly otherwise, and never starts one. The engine
row indexes in the background only when it is mounted in a host, and otherwise runs to
completion as now, which keeps the engine's tests as they are. Progress goes to a file
in the index directory, which the session's row reads for the footer and for the
completion notice — a message delivered with `inject`, so it waits for the agent's
next step and starts no turn.

## What U1 to U4 settled

- **utcp 1.2.0 and utcp-cli 1.1.5 are on PyPI**, and are now test dependencies of the
  workspace and runtime dependencies of `ph-utcp`.
- **The headless profile is five rows**: the registry, the local subprocess row,
  `fs-local`, `commands` and `diagnostics`. No model, session or agent row is needed to
  run a tool.
- **A granted path is writable in a read-only session.** The plan expected indexing to
  be refused there. The sandbox reads a policy's extra paths under either mode, so a
  confined agent that may not write its tree still builds an index under `$PH_CACHE`.
  A test holds this under a real `bwrap`, in both modes.
- **A granted path has to exist.** A backend grants a path by binding it. `ph-utcp`
  makes each one when a manual is added, and says which it could not.
- **`ctx.shell.run` is a login shell**, so it cannot carry a command whose stdout is
  data: a profile that prints would corrupt the answer. `ctx.shell` gained `run_argv`,
  with the same placement and confinement, and `ph-utcp` runs `/bin/bash -c` through
  it. `run_argv` can narrow the environment to named variables and add to it; it
  cannot widen it.
- **A credential cannot be inherited by name.** The deployment's scrubbed environment
  drops every variable whose name contains `KEY`, `TOKEN`, `SECRET` and the like, and a
  template's `inherit_env_vars` selects from that environment. A credential reaches a
  command through the template's `env_vars` with a `${VAR}` reference, which the UTCP
  client resolves. E2 has to use that route for an endpoint's key.
- **The override of the `cli` protocol is process-wide**, because the library's
  registry is. It applies only inside one of `ph-utcp`'s calls, marked by a context
  variable; a plain `UtcpClient` in the same process gets the library's own behavior.
- **Templates use `--name=value`.** A value that starts with `-` is otherwise read as a
  flag.
- **The engine is a second row in each package**, `code-graph-engine`, mounted by the
  command under the session row's id so one config serves both. The session's row
  imports neither it nor the parser; a test in a fresh interpreter holds that.
- **`phern` is no longer the only console script.** `tests/test_packaging.py` names the
  tool commands as decisions.

Open from these steps:

- `ph-utcp` allows only the `cli` protocol by default. `http` and `mcp` manuals need
  their plugin installed and the type named in the row's `protocols`, and then run
  inside the harness process, outside the sandbox. Routing those through pH's egress
  rules is not designed.
- The session's `phern doctor` section for the code graph no longer shows counts or
  grammar readiness; it points at `/code-graph status`, which asks the command.
- Both READMEs still describe the tools as registered in-process (U10).

## The goal

phern can call any tool published as a UTCP manual, through one row, `ph-utcp`.
`code_index`, `code_graph`, `text_index` and `text_search` are the first tools it
reaches that way, and they reach other UTCP harnesses through the same manuals. A UTCP
tool reads and writes the filesystem the agent calling it sees.

## Decided with the user

1. **`ph-utcp` is in this work and is the main piece.** It serves future UTCP tools and
   pH's own code-graph and text-index tools in the same way.
2. **A UTCP tool sees the agent's filesystem.** Its command runs in the agent's
   workspace, under the agent's sandbox.
3. **Each package's command line is an extra for people**, for trying a query by hand
   and for debugging. It is also what the package's manual calls.
4. **Each package keeps a thin row** that registers its manual with `ph-utcp`, so the
   bundle check, the skill, the slash command and the doctor section survive.
5. **The worker starts at mount for a root session**, and on the first tool call for a
   child in its own worktree.
6. **The row config says what to index**: which packages, which files to take and which
   to leave out.
7. **The text index can choose its embedding model, and reach local and remote models
   through an adapter.** This is added in stages, E1 to E3 below.

## What exists today

**Neither package's tools can be called from a command line.** Checked four ways:

- Neither `pyproject.toml` has a `[project.scripts]` table. Both register only
  `ph.plugins` and `ph.bundles` entry points, so each is a row mounted inside a harness
  process.
- `phern` has no command that runs a named tool. Its commands are the default prompt
  runner, `daemon`, `doctor`, `events`, `revisions`, `config`, and the `agents`,
  `workspaces`, `attachments`, `profiles` and `service` groups.
- `--mode rpc` and the daemon protocol both have `tools/list`, which returns schemas.
  Neither has a verb that executes a tool; the only way to run one is `session/prompt`,
  where the model decides.
- What a person can reach without a model turn is provisioning and reporting only:
  `/code-graph install|status` and `/text-index install|status` inside a session, and
  the two sections in `phern doctor`.

The one place a tool runs outside the agent loop is `ph.testing.run_tool`, which calls
`ctx.require(TOOLS).execute(ToolExecutionInput(...))` on a mounted profile. That call is
the shape a command-line door takes.

**What the tool bodies take from the mount.** A CLI has to supply all of it:

| need | where it comes from now |
|---|---|
| the workspace root, the walk, ignore rules, reads | `ctx.fs` (`collect`, `read`, `skip_reason`, `resolve`, `named`, `root_for`) |
| "which files changed" from git or jj | `ph.seams.changes.tree_state`, through the subprocess seam |
| cache paths, per-workspace index directories, the sweep | `ph.paths`, `ph.workspace_caches` |
| the per-agent root, the policy scope, cancellation | `ToolRunContext` |

**No process maintains either index.** Both READMEs say so: no watcher, no daemon. An
index changes when its index tool runs. Re-running is cheap because git or jj vouches
for unchanged files, and the code graph also compares a sha256 per file.

**How a command already runs "as the agent".** `ctx.shell.run(command, agent=...)`
resolves the working directory to the agent's workspace root and, when a sandbox
backend exists, confines the command with that workspace as its one writable root.
`SandboxPolicy.writable_extra` names further writable paths. `permissions-fs` says of
itself that it covers tool calls through `ctx.fs` and not a subprocess.

## What the measurements say about processes

Measured on this machine, 2026-11-10, in the workspace venv:

| step | cost |
|---|---|
| `import ph_code_graph` | 0.44 s |
| `import ph_text_index` (no model) | 0.29 s |
| `phern --help` | 0.59 s |
| `import sentence_transformers`, warm file cache | 3.6 s, about 400 MB resident |
| `import sentence_transformers`, cold | 11.4 s |

From `packages/ph-text-index/README.md`: the model loads in 19 s cold (MiniLM) or 26 s
(nomic), and indexing this repo's `docs/` takes 20 s (MiniLM) or 185 s (nomic).

The UTCP CLI protocol in utcp-cli 1.1.5 starts a fresh `bash -c` per call and kills it
after a fixed 120 seconds (60 for discovery). That limit binds other harnesses. Inside
phern the command is run by `ph-utcp`'s own executor, whose limit is pH's to set.

- **Code graph works as one process per call.** Half a second of startup, SQLite in WAL
  mode with a connection per operation, and each file committed in its own transaction,
  so a call that is killed resumes where it stopped.
- **Text index does not.** Every `text_search` would pay the import and the model load
  before embedding one query, and `text_index` saves once, at the end, so a killed call
  loses everything it embedded. It needs a process that stays up with the model loaded.

Two gaps show up only once two processes share an index:

- The code graph's index lock is an in-process `anyio.Lock` (`CodeGraphSeam._lock`).
  The text index already has the fix: a `writer.lock` file through
  `ph.locks.acquire_file_lock`.
- The text index has no per-document digest. In a tree with no git or jj, a repeated
  `text_index` re-embeds every document. The code graph's sha256 column is the model.

## What the UTCP CLI protocol requires

Read from `plugins/communication_protocols/cli/src/utcp_cli/`.

1. **Discovery is a command whose stdout is a manual.** The client runs the manual
   template's commands and parses stdout as JSON with `utcp_version` and `tools`. Each
   tool carries its own `tool_call_template` of type `cli`.
2. **Arguments arrive one at a time, as `str(value)`.** A `UTCP_ARG_name_UTCP_END`
   placeholder becomes one environment variable. A list arrives as Python's repr
   (`['a', 'b']`), a boolean as `True`, and an optional argument the model left out
   arrives as the literal `MISSING_ARG_name`. There is no placeholder for the whole
   argument object.
3. **stderr is merged into stdout.** Each step runs as `$(command 2>&1)`, so one log
   line on stderr corrupts the JSON result.
4. **The exit code is ignored on Unix.** The result is stdout, parsed as JSON when it
   starts with `{` or `[`. A failure has to be reported in the body.
5. **The environment is cut to an allowlist** (`PATH`, `HOME`, `LANG`, `LC_ALL`,
   `LC_CTYPE`, `USER`, `LOGNAME`, `SHELL`, `TZ`, `TERM`). `PH_HOME`, `PH_CACHE`,
   `PH_RUNTIME` and the `XDG_*` variables are dropped unless the template names them in
   `inherit_env_vars`.
6. **The working directory is the client's**, unless the template sets `working_dir`.
7. **Tool names are prefixed with the manual's name**: `ph_code_graph.code_index`.
8. **The protocol has tools and nothing else.** No hook that runs when a manual is
   loaded, no progress, no notification. Anything of that kind is pH's own addition and
   another harness will not see it.
9. **Importing `utcp` calls `logging.basicConfig(level=INFO)`**, as does every protocol
   module.

Both packages are POSIX-only by their classifiers, so the PowerShell half of the
protocol is out of scope.

## The design

### `ph-utcp`: the row that makes a manual's tools phern tools

A new package with one row. Its config lists manual call templates, and other rows can
register a manual with it at mount. For each manual it runs discovery and registers
every tool with `define_tool` on the raw-schema path, which exists for declarations pH
did not author. The manual's `inputs` and `outputs` schemas become the tool's schemas,
so Code Mode's generated listing shows them like any other tool.

- **Commands run through `ctx.shell` for the calling agent.** The `cli` protocol is a
  subclass of `CliCommunicationProtocol` that overrides only `_execute_shell_script`.
  The command gets the agent's workspace as its working directory, the agent's sandbox,
  pH's cancellation, the orphan journal and a timeout pH sets. The library's argument
  substitution, which carries its security fixes, is kept as is.
- **A manual can be given extra writable paths**, passed as `writable_extra`. A tool
  that keeps state outside the workspace cannot work under confinement without this.
- **A manual that fails discovery is skipped** with a warning, a line in `phern doctor`
  and a note to the model. It does not refuse the mount.
- **No values are stored.** UTCP's `${VAR}` references resolve from the environment at
  call time.
- **What UTCP cannot say, the row's config says.** A manual has no word for "safe to
  run two at once" or "has no effect". Each defaults to the careful answer and a
  manual's entry in the config can state otherwise per tool.
- **A UTCP tool has no idempotency key and no `reconcile`**, so after a crash its
  outcome is unknown to the model. `DESIGN.md` already states this for MCP tools.
- **The result is shown as the tool's `text` when the result has one**, and as JSON
  otherwise. pH's own tools use that field; see the envelope below.
- Logging is configured before `utcp` is imported, so its `basicConfig` call is a no-op.

### Each package: a command line, a manual, and a thin row

**The tool bodies move behind the package's command.** A generic module in `ph-core`,
working name `ph.toolcli`, builds the command-line app from a list of rows:

- `call <tool>` mounts a small headless profile (the tools registry, `fs-local`, the
  local subprocess row, the package's rows) and runs the tool through the registry, as
  `run_tool` does. The existing bodies and their tests stay; what changes is which
  process mounts them. `--root` defaults to the working directory, which under
  `ph-utcp` is the agent's workspace.
- `manual` prints a UTCP manual generated from the registry's `ToolSchema` list.
- `install` and `status` are the package's provisioning command. They stay out of the
  manual, for the reason the packages give now: provisioning is something a person
  asks for.
- `--args-json` takes a whole argument object, for people and scripts.

**Argument decoding is driven by the tool's pydantic model.** One flag per top-level
field. For a field that is not a string the decoder tries JSON, then Python's
`True`/`False`/`None` and list repr, then a comma split for arrays. A value starting
with `MISSING_ARG_` is treated as absent.

**The result is one envelope on stdout, and nothing on stderr**:

```json
{"ok": true, "value": {"mode": "define", "...": "..."}, "text": "2 matches for 'claim_slot': ..."}
{"ok": false, "error": {"code": "CODE_GRAPH_FAILED", "message": "no code index exists ..."}}
```

`value` is the tool's declared output and `text` is the tool's own `render`. Logging
goes to a file under `$PH_CACHE`, and each generated template ends in `2>/dev/null`.

**The row a session mounts becomes a registration.** It keeps its name and its bundle,
so `rlm-indexed` and the check that hides a profile whose bundles do not resolve work
as now. At mount it:

- registers the package's manual with `ph-utcp`, under the tool names used today
  (`code_index`, not `ph_code_graph.code_index`), so skills and prompts do not change;
- hands its row config to the command through the environment;
- names the index directory under `$PH_CACHE` as an extra writable path;
- offers the package's skill, its slash command and its `phern doctor` section, with
  the last two answered by running the command's `status`.

### Indexing as a background worker with visible progress

This is the user's proposal, and it replaces the earlier "call again to collect"
design. An index tool never waits for indexing to finish.

- **A worker process per package and workspace.** The registration row starts it as a
  `ctx.jobs` job when it mounts, through the same door as any command for that agent,
  so it sees the agent's filesystem. The job belongs to the row's scope, so unmounting
  cancels it. A child agent in its own worktree gets its worker on its first tool
  call, because each text-index worker holds a loaded model (about 400 MB) and a
  throwaway worktree may never search. A harness with no such row also gets the worker
  on the first call.
- **It loads what it needs once and stays up.** For the text index that is the
  embedding model. It indexes, then answers `text_search` queries without loading the
  model again, and exits after an idle period.
- **It reports progress** to a small file in the index directory, written atomically:
  state, documents done, documents total, and the failure text when there is one. It
  holds the writer lock while it works, so "is it alive" is asked of the lock and not
  of a pid.
- **The tools answer at once.** `text_index` on a new project starts or queues the work
  and returns `state: indexing` with the counts. `text_search` during indexing answers
  from what has been saved so far and carries a warning, "Not finished indexing: 120 of
  539 documents". The code graph follows the same contract.
- **The front end shows it.** The row contributes a reading to `ctx.tui_status`,
  "indexing 120/539", from the last progress it read. When the job settles
  (`job/settled`) the person gets a completion notice and the model gets a note. Which
  transcript mechanism carries the notice is chosen in U5.
- **After a daemon restart** the remounted row reads the progress file and the lock: a
  live worker is picked up, a dead one is started again, and the index's own
  increments mean nothing is embedded twice.

Three changes inside the text index make this honest: a per-document digest, saves at
intervals instead of once at the end, and a search that reads a saved copy while a
writer works (which `TextIndexSeam.index` already does).

### What the worker indexes: the row's `index` block

Both rows gain the same block. It is what the worker indexes at mount, and the default
for a tool call that names no paths.

```yaml
- id: code-graph
  name: code-graph
  config:
    index:
      on_mount: true                   # root sessions; a worktree child waits for a call
      paths: [packages/ph-core]        # directories or files under the root; default ["."]
      include: ["**/*.py"]             # default: the tool's own glob
      exclude: ["**/tests/**", "**/vendor/**"]
    languages: [python]                # code graph only, as now
```

- **`paths` is how a row points at a package.** Several are allowed, so a monorepo
  names the packages it wants and stays under the 20,000-file bound.
- **`include` replaces the single `glob`** each row has now, and takes a list.
- **`exclude` is new.** Neither tool can leave files out today except through
  `fs-local`'s ignore rules. It is applied to the workspace-relative names the walk
  returns, before any file is opened.
- The tools' own arguments gain `exclude` beside `paths` and `glob`, so a call can
  narrow in the same words as the config.
- `on_mount: false` turns the worker's first run off and leaves indexing to tool calls.

### Choosing the embedder, in stages

What exists: an `Embedder` protocol (a name and `encode`) and one provider row,
`text-index-local`, which already takes a `model`, the two prefixes, `batch_size` and
`trust_remote_code`. The module docstring and README describe an endpoint-backed
embedder as a row a deployment writes itself; none ships. pH's `LlmAdapter` protocol
has one method, `stream`, so no adapter can embed today.

**E1. Pick the local model.** The thin row passes `text-index-local`'s config through
to the worker, so `model` and the prefixes are set in the session's profile as now.
Nothing new is built; this is U6 carrying the existing config across the process
boundary. The index stays keyed by the embedder's identity, so changing the model
starts a separate index and changing back finds the old one.

**E2. An endpoint embedder.** A new provider row, working name `text-index-endpoint`,
that speaks the OpenAI-compatible `/v1/embeddings` request. One row covers a local
server (llama.cpp, Ollama, vLLM) and a hosted API:

```yaml
- id: text-index-endpoint
  name: text-index-endpoint
  config:
    base_url: http://127.0.0.1:8080/v1
    model: nomic-embed-text-v1.5
    api_key_env: ""                    # the NAME of a variable; empty for a local server
    query_prefix: "search_query: "
    document_prefix: "search_document: "
```

- **No torch.** A deployment with an endpoint installs `ph-text-index` without the
  `[local]` extra, and the worker holds no model in memory.
- **No key is stored.** The config names an environment variable. The row forwards
  that name to the command, and a worker that finds it unset reports "missing
  credential" with the name as its state.
- **The sandbox has to permit the host.** A confined command has no network unless the
  deployment's allowances name the host and port. The row asks for that one allowance.
- **A remote endpoint is sent the documents' text.** That is opt-in by writing the row
  and the README says so in those words.
- **The identity includes the endpoint's model name, both prefixes and the dimension.**
  A hosted model that changes under an unchanged name cannot be detected beyond a
  dimension check.
- **An embedder that cannot be reached does not switch vector spaces silently.** The
  tool answers with the reason and the reading shows it. A row may list a second
  embedder to fall back to; a search served by it says which embedder answered, since
  it is searching a different index.

**E3. Embeddings through pH's adapters.** An optional `EmbeddingAdapter` protocol
beside `LlmAdapter`, following the tree's rule that an optional capability is its own
protocol. The OpenAI-compatible and Google adapters in phern implement it, and a
provider row, working name `text-index-adapter`, names a `provider` and `model` the way
a session's route does. This reuses the routes, the credentials seam and the usage
accounting. The worker's profile mounts the adapter row, so this stage needs phern
installed beside `ph-text-index`, where E2 does not.

### What stays as it is

- **No file watcher.** Re-indexing an unchanged tree costs 0.03 s for 137 files. The
  worker could grow a watch mode later; it is not in this plan.
- **A model is never downloaded without being asked.** A worker that finds no weights
  reports "model not installed" as its state and the reading says to run
  `/text-index install`.

## Steps

**U1. The command line.** `ph.toolcli` with `call`, `--args-json`, the envelope and the
silent stderr. `ph-code-graph` gets a console script. The code graph's index lock
becomes a file lock. *Gate: `ph-code-graph call code_index` then `call code_graph` on a
fixture tree return the same values as the row's own tests, and two concurrent
`code_index` processes each report counts that add up.*

**U2. The manual.** `manual`, the per-field flags, the lenient decoder and the
`MISSING_ARG_` rule. `utcp` and `utcp-cli` become test dependencies. *Gate: a real
`UtcpClient` registers the manual template and calls `ph_code_graph.code_graph` end to
end, including an omitted optional argument and a list argument.*

**U3. `ph-utcp`.** The client row, the `cli` protocol through `ctx.shell`, registration
by other rows, extra writable paths, the per-tool declarations, the diagnostics
section. *Gate: a session calls a fixture manual's tool and the command's working
directory is the agent's workspace; under a sandbox backend it cannot write outside
that workspace and its extra paths; a manual whose command fails leaves the mount up
with a warning.*

**U4. Code graph through `ph-utcp`.** The registration row, the bundle, the skill, the
slash command, the doctor section. *Gate: `rlm-indexed` offers `code_index` and
`code_graph` under their present names, a child agent in a worktree indexes its own
tree, and the session process never imports `tree_sitter`.*

**U5. The worker and its progress.** The text index's digest and interval saves, the
worker, the answer-at-once contract, the progress file, the status reading, the
completion notice. *Gate: a second `text_search` does not load the model again; a
search during indexing returns saved passages with the warning; the footer reading
moves; a worker killed with SIGKILL is replaced at the next mount and embeds nothing
twice.*

**U6. Text index through `ph-utcp`, with E1.** Its command, manual and registration
rows, and the local model's config passed through. *Gate: U4's gate for the text tools
with the suite's stub embedder; the session process never imports
`sentence_transformers`; a profile that names a different model gets a separate
index.*

**U7. The `index` block.** `paths`, `include`, `exclude` and `on_mount` on both rows,
and `exclude` on the tools' arguments. *Gate: a worker on a fixture monorepo indexes
only the named package, an excluded directory contributes no symbol or passage, and a
worktree child's worker does not start until its first call.*

**U8. E2, the endpoint embedder.** *Gate: against a fixture HTTP server, indexing and
search work with no torch installed; an unset key variable and an unpermitted host
each produce a named state and no traceback.*

**U9. E3, the adapter embedder.** The protocol, the two adapters, the provider row.
*Gate: a text index embeds through a profile's OpenAI-compatible route, and an adapter
without the capability is refused at the row with a sentence naming it.*

**U10. Documentation.** Both READMEs, a `ph-utcp` README, and a section in
`docs/cookbook/adding-a-tool.md` on publishing a tool as a UTCP manual.

U7 can move ahead of U5 if choosing what to index is wanted before the worker exists;
it depends only on U4.

## What breaks

- **The three rows stop running tools in the session process.** `ctx.code_graph` and
  `ctx.text_index` are no longer provided to a session's mount. A deployment that
  registered its own embedder on `ctx.text_index` configures it in the tool's own
  profile instead.
- **Index tools return before the work is done.** A caller that read `IndexValue` as a
  finished count reads a state and a progress count.
- **`permissions-fs` no longer screens these tools' reads**, and `fs/read-intent` no
  longer fires for them. The sandbox bounds them, as it bounds `bash`.
- **Indexing needs a writable index directory.** In a session whose sandbox mode is
  read-only the index tools are expected to be refused while search still works; U3
  confirms what the seam does with an extra writable path in that mode.
- **Row config reaches the tool through the environment**, so a change to it applies
  to the next command and to the next worker start.
- **`glob` on both rows becomes `index.include`**, a list. The old key is removed, not
  aliased.
- New: two console scripts, a lock file in each code-graph index directory, a digest
  and a progress file in each text index (an index built before U5 is rebuilt), one
  worker process per indexed workspace.

## Left for the step that meets it

Nothing waits on the user. These are settled where the code is read:

- Which transcript mechanism carries the completion notice (U5).
- Whether the worker's framing reuses pieces of `ph_app.daemon` moved down into
  `ph-core`, or is a small JSON-lines protocol of its own (U5).
- Whether E2's fallback list is worth having, or one embedder per row is enough (U8).

## Not verified

- That utcp 1.2.0 and utcp-cli 1.1.5 are the versions published on PyPI. The clone is
  what was read; `sources/` is ignored by git, so the dependency has to come from an
  index.
- How UTCP clients in other languages stringify list and boolean arguments. The decoder
  accepts JSON, Python repr and comma-separated forms for that reason.
- The model load time on this machine. No weights are cached here, and none were
  downloaded for this review; the 19 s and 26 s figures are the README's.
- Which rows the headless profile needs beyond the registry, `fs-local` and the local
  subprocess row. U1 finds out by mounting it.
- What a confined command can write when the session's sandbox mode is read-only and
  the policy names an extra path. U3 finds out.
