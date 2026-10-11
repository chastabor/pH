# pH

*phern* — a Python Harness, in earnest.

An experiment in breaking a harness into small, replaceable components. It
borrows deliberately and says where: the **Deepseek harness** (`dsh`) for the
space/time split and the wire envelope, **LangChain Deep Agents** for the todo
list as a cognitive anchor, **OpenMono's playbooks** for structured-reply repair
and step ordering, and **prime-agent** for the RLM programming model. Each is
credited in the source at the point it is used, not just here.

**There is no privileged core.** The agent loop, the model adapter, the tool
registry and the session log are all *rows in a YAML profile*, mounted by the
same loader in file order — so any of them can be replaced without forking
anything else.

**A session is an append-only log, and what the model sees is a projection of
it.** Because the log is never rewritten, the prompt prefix stays byte-identical
and a provider's cache keeps hitting across a turn. And because state is derived
rather than held, a restart is a replay: an agent that crashed, was paused, or
had its terminal closed rebuilds from its log and carries on from where it got
to. Side effects *outside* the harness still have to be idempotent — a log can
only promise what it recorded.

**Results are checked rather than taken on trust.** Values that cross a boundary
are declared pydantic shapes, validated before anything renders or records them.
The same principle steers the loop instead of driving it: a `SKILL.md` that
declares `steps:` seeds them as todo entries the model may complete but may not
delete, and a listener on the turn boundary objects while any remain — steering
the agent rather than reaching into loop state. A finished entry carries
`worked`, the number of tools the harness *saw* run, because a receipt the
claimant issues is not a receipt.

[`DESIGN.md`](DESIGN.md) is the specification; [`docs/`](docs/README.md) is how
to extend it.

## Install

**From PyPI**, to use it — after this, `phern` is a command like any other:

```bash
uv tool install phern
```

That is the whole harness: one name, every profile. `phern` is the distribution
a person installs and the command they get; the seven libraries it is assembled
from publish under their own names — `ph-core`, `ph-rlm`, `ph-runtime-guest`,
`ph-stabilize`, `ph-text-index`, `ph-code-graph`, `ph-clm` — because each is usable
on its own. A front end of your own over `ph-core`, or a deployment that wants
`ph-code-graph`'s two tools and nothing else, installs exactly that and never
sees this package.

The *import* names are a third namespace again, and unchanged: `ph`, `ph_app`,
`ph_rlm`, `ph_runtime`. A distribution name, a module name and a directory name
are allowed to differ, and here all three do — `packages/ph-core/src/ph` builds
`ph-core`, and `packages/phern/src/ph_app` builds `phern`.

**In the checkout**, to work on it:

```bash
uv sync                                       # every member plus the dev group
uv run phern --help
uv tool install ./packages/phern              # a PATH `phern` from this checkout
uv tool install --editable ./packages/phern   # ...that tracks it
```

The repository root is a *virtual* workspace root — members and tooling
configuration, no distribution of its own — so it is `./packages/phern` that
gets installed rather than `.`, and there is no longer an empty wheel in the
middle holding a dependency list.

uv prints where it put `phern` — `~/.local/bin` by default — and `uv tool
update-shell` fixes a PATH that misses it. Afterwards the deployment answers to
its own name: `uv tool upgrade phern`, `uv tool uninstall phern`. `--editable`
reaches every member through the workspace, so a `git pull` is the whole
upgrade. Coming from 0.6, read [*Upgrading from 0.6*](#upgrading-from-06) first.

`phern doctor` reports what you actually got — which rows mounted, which
profiles this install can compose, and why any of them refused.

### The extras are yours to decide

Three things this harness can do are large enough, or specific enough, that
installing them is a choice rather than a default. None of them is a feature
flag — each is a *row* that either mounts or does not, and `phern doctor`
reports which ones actually activated.

| extra | what it adds | what it costs |
|---|---|---|
| `phern[local]` | the local embedding model, so `text-index-local` mounts and `rlm-indexed` composes | `sentence-transformers` and torch — about 4.2 GB with the default CUDA wheels |
| `phern[web]` | `--mode web`, the browser tab | `textual-serve`, with aiohttp and jinja2 behind it |
| `phern[otel]` | exporting the session log to an OpenTelemetry collector | the OTel SDK and its OTLP exporter |

**The embedder is the one worth thinking about.** Without `[local]`,
`text_index` and `text_search` are still registered and the index still works —
you supply the vectors. What is missing is the row that *computes* them, and it
refuses to mount with a message naming the extra. A deployment that embeds
through an endpoint instead — llama.cpp's `/v1/embeddings`, a provider's API —
registers its own embedder on `ctx.text_index` and never wants torch at all.
That is why it is not a dependency: a gigabyte for one provider of one seam is
not a cost this package gets to impose on everyone who installs it.

If you do want it and have no GPU, the CPU wheels are a third of the size:

```bash
pip install "phern[local]" --extra-index-url https://download.pytorch.org/whl/cpu
```

The checkout picks that index up on its own, from `[[tool.uv.index]]` in the
root `pyproject.toml` — uv configuration, which is why it cannot travel in the
published metadata and has to be a flag you pass.

Python 3.12 or newer. Every example below says `phern`: literal after a tool
install, `uv run phern` in the checkout.

## Point it at a model

A provider is a profile plus an environment variable holding the **name** of a
credential — the value is resolved at the request edge and never enters a row, an
event, or a child process.

| `--profile` | route | credential |
|---|---|---|
| `llama` | a local llama.cpp server (`LLAMA_BASE_URL`, default `http://127.0.0.1:9931/v1`); its context window is read from `/props` at mount | `LLAMA_API_KEY` — a formality llama.cpp ignores without `--api-key`, but it must be set |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `anthropic` | Anthropic messages API | `ANTHROPIC_API_KEY` |
| `google` | Gemini | `GEMINI_API_KEY` |
| `base` · `headless` | no model route at all — the fake adapter, for wiring work | — |
| `tui` | `headless` plus a writable workspace: a person is present to answer approvals | — |
| `rlm` · `rlm-stable` | Code Mode, and Code Mode with the stabilization gates on | — |
| `rlm-clm` | `rlm` plus `ph-clm`: the model edits its own context, with the context tools or as a file | — |

`phern doctor` prints the list this install can actually compose. Profiles compose,
so `--profile llama` is `base` plus the llama route; the interactive profiles
layer `headless` and then their own rows.

**Every one of them also layers `ph-stabilize`**, so a conversation is compacted
at 85% of the window rather than growing until the provider refuses it, and
`/compact` is there to do it by hand. That layer is *optional*: an install
without the distribution composes the same profiles and simply never compacts —
which is the one place a profile's behavior depends on what is installed, and
`phern doctor` reports what actually activated.

## One prompt

```bash
export LLAMA_API_KEY=local
export LLAMA_MODEL="$(the model llama-server loaded)"   # the llama profile's `main`
phern --profile llama -p "what is in this repo?"
```

- Each profile lists the models it runs on (`phern config --row models`), and a
  root runs on the default. `--model fast` picks another listed one by key;
  `--provider P --model M` runs a route the list does not hold. A route no
  adapter in the profile serves is refused before anything runs.
- `-a/--attach FILE` sends a file with the prompt; repeatable.
- `--session <id>` continues (or resumes from disk) instead of starting fresh.
- `--mode json` and `--mode transcript` change the shape of what is printed;
  `text` is the default.

## The TUI

```bash
phern --profile tui --provider llama --model <model> --mode tui       # new session
phern --mode tui --resume 20260908T041607-1c5576                      # reopen one
phern --mode tui --no-spawn                                           # refuse rather than start a daemon
phern --mode tui --keep-daemon                                        # the daemon it starts is a service
```

The front end talks to a daemon and starts an ephemeral one if nothing is
listening, so closing the TUI does not end the turn — the root keeps working and
`phern agents attach` will show it to you again. `/model` (`ctrl+p`) moves the
session to another model the profile lists, or to a `provider/model` you type,
from its next request; `--model` on the command line asks the same for the
session this terminal opens, and leaves the daemon's other sessions alone.

## The browser UI, on localhost

```bash
phern --mode web --profile llama                              # 127.0.0.1:8000
phern --mode web --port 8080 --open
```

It prints, before it binds:

```
open http://127.0.0.1:8000/?token=…
every tab lands on session 20260908T041607-1c5576
anyone with that URL has this terminal's authority: approvals, shell commands, the workspace
```

Three things that line says, spelled out:

- **The token in the URL is the whole authentication story** — no TLS, no users.
  Treat the URL like the terminal it came from.
- **Every tab of one launch shares one session.** A second tab joins the
  conversation; a second `phern --mode web` is a new one.
- `--host` anything other than loopback prints a further warning, because that
  authority then reaches anyone who can route to the port.

## The daemon

```bash
phern daemon --profile llama                  # the profile new sessions start on
phern daemon --max-concurrent-children 6      # across every root; the rest queue
phern daemon --passivate-after 30             # minutes of quiet before a root is released, or `off`
phern daemon --ephemeral                      # exit once no client, root or appointment needs it
```

The socket is `$PH_RUNTIME/daemon.sock`, per boot and per user. A stale socket
from a crashed daemon is cleared; a live one is refused rather than stolen.

## What is running, and stopping it

```bash
phern agents                                  # the roots this daemon is running
phern agents status <session>                 # what one root is doing, and a --since cursor
phern agents attach <session>                 # its history, then everything as it happens
phern agents attach <session> --until-idle    # exits 1 if the turn it stopped on errored
phern agents send <session> "carry on"        # queue a turn, starting or resuming the root
phern agents schedule <session> --every 3600000 --prompt "check the build"
phern agents doctor                           # socket, pid, uptime, provider, roots, capabilities
phern agents shutdown                         # stop it, and wait until it is actually gone
```

Attaching neither starts nor stops the work, so leaving is free. `phern agents
doctor` reports what is **in force** in the running process, unlike `phern doctor`,
which reports what this invocation's flags and environment would produce.

## Configuring

Every row says what its settings shape, and each kind has one owner:

| kind | what it shapes | where a person sets it |
|---|---|---|
| environment | the model, tools, skills, sandbox, workspace | a session profile, `$PH_HOME/profiles/<name>.yaml` |
| deployment | persistence, telemetry, the job bound, invariants | the daemon's `$PH_HOME/daemon.yaml` |
| presentation | screens and footer readings | the TUI's `$PH_HOME/tui.json` (hide a screen, remap its key) |

`phern config` shows each row's kind. A row set in the wrong place is refused by
name, with the file it belongs in. The layers, applied in this order:

1. the shipped bundle documents — `ph-core/src/ph/bundles/*.yaml` and
   `packages/phern/src/ph_app/profiles/*.yaml`, which may set any kind;
2. `rows:` in `$PH_HOME/daemon.yaml` (deployment), then the daemon's own flags
   such as `--max-concurrent-children`;
3. **your named profile**, `$PH_HOME/profiles/<name>.yaml`: the shipped profile it
   `extends` and only the rows that differ (environment);
4. `--patch`, this run only, same grammar as a profile's rows (environment).

```yaml
# $PH_HOME/profiles/work.yaml — run it with `--profile work`
extends: tui                     # a shipped profile; left out of a file named after one
rows:
  - id: tool-bash
    disabled: true
```

```bash
phern profiles show work            # what the file sets
phern profiles show work --full     # every setting a session on it runs with, defaults included
phern profiles fold                 # fold drop-ins and old list-format files into one file each
phern profiles session <id>         # the environment a session ran in, and the skills it read
phern profiles session <id> --at N  # the same at seq N of its log (--full: every row)
phern profiles diff work            # what each session on it would change, to take its current version
phern profiles adopt work           # take it, at each session's next start (--session <id>, --yes)
```

A file that is still a bare list of rows (the format before named profiles), and
the `<name>.d/` drop-ins `/sandbox` used to write, are read until `phern profiles
fold` folds them into the named file; `phern doctor` names each one until then.

What a session changes while it runs — `/sandbox allow`, `/model`, and each
`--patch` or `--model` that differs from its profile — is an override in the
session's log, written before the change takes effect. A session is mounted from
its own log whenever it starts again — its profile as it was when it began, then
its overrides — so a daemon's `--profile` is only the one *new* sessions start on,
and one daemon holds sessions on as many profiles as were asked for.

A session keeps the version of its named profile it started on. If you edit the file
while it is not running, its next start from the TUI stops and shows each changed
setting, old and new: use the new version, keep the session's (and not be asked about
that version again), or decide next time. The session's overrides still apply over the
new version. A start nobody is there to answer — a schedule's, or `phern -p` — keeps
the session's version and says so, and so does a change that is only pH's own defaults
moving under a file you did not edit. `phern profiles adopt` takes a new version on
purpose, for every session on the profile, without starting any of them. A terminal
that attaches to a session kept on an older version is told so once, with whose the
changed settings are and the commands that list them and take them.

A version that will not mount — a row that no longer resolves, one that declines, a
config its model rejects — is taken back at the start that tried it, whichever start
that is: a daemon's, `phern -p --session`, rpc. The session starts on the version it
had, is not offered that one again, and says why: in the reply to `/profile use`, to
a terminal that attaches, on stderr, and in `/profile show`. Any other failure — a
full disk, a bug in a row — fails the start and takes nothing back, so the next start
can mount it.

Inside a session, `/profile` manages its profile:

```text
/profile show [--full]        the base, each override and what asked for it, and what they change
/profile diff                 how the session's named profile has moved since it started
/profile save <name>          the session's environment as a named profile (--replace to overwrite)
/profile use <name> [--clear] run the session on another named profile, its overrides kept or cleared
/profile clear [row]          stop one override, or all of them, from applying
```

So a profile is tuned in a TUI session and saved to run headless: `/sandbox allow` and
`/model` there, `/profile save work`, then `phern -p … --profile work` starts on the same
environment. `/profile use` restarts the session on the new profile; `/profile use` of
the name it already runs on takes that profile's current version.

**A child agent can be given a profile too.** A parent that delegates — the `task`
tool, or `rlm.run` under Code Mode — names the child's `model` by a key its own
profile lists, or lets a skill it hands over name one (`model:` in the skill's front
matter), and may assign it a named profile, `profile="reviewer"`. A child runs on its
parent's mount, so that profile is read as a narrowing rather than mounted:

- a row it runs that the parent does not is refused, by name;
- a row it leaves out takes the tools and skills that row gave with it;
- a row it keeps holds the child to what it says: the `models` default, a `read-only`
  sandbox, fewer `skills-progressive` paths, fewer writable directories in
  `sandbox-allow` — whose network must be the parent's, since one egress proxy
  serves every agent.

A child holds less than its parent and never more, and its admission records what it
got, so a restart brings it back no wider.

`daemon.yaml` can also move where sessions and profiles are kept. It is read
when a process starts:

```yaml
paths:
  sessions: ~/work/ph-sessions   # relative paths are under $PH_HOME
rows:
  - id: jobs
    config:
      concurrency: {subagent: 8}
```

```bash
phern config --profile llama                            # every knob every row accepts
phern config --row autonomous --profile tui             # one row, by the name it is registered under
phern --dump-config --profile llama                     # the mount as written, in order
phern doctor --profile llama                            # what actually activated
phern --patch '{id: fs, config: {root: /tmp/scratch}}' --profile tui --mode tui
```

A patch entry is `{id: …, config: {…}}` to reconfigure, `{id: …, disabled:
false}` to arm a row a bundle ships off, `{id: …, remove: true}` to drop one, or
`{insert: [...]}` to add one. Note that a **list** in row config is replaced
rather than merged — an overlay that restates one field of a route must restate
the whole route entry, or it inherits that field's default.

Three roots, each overridable by its variable: `PH_HOME` (`~/.ph` — sessions,
attachments, your profile overlays, `daemon.yaml`), `PH_CACHE` (`~/.cache/ph` — safe to delete
wholesale) and `PH_RUNTIME` (the daemon socket). `phern doctor` prints where all
three resolved, and which tier `PH_RUNTIME` landed in.

## Upgrading from 0.7

0.8 asks the session store less: a stored log is found once and read once, never on
the daemon's event loop, a new session is never searched for, and a kernel cell's
spilled variables are written together. What that changes for an 0.7 setup:

- **Restart the daemon.** It speaks protocol 8: `session/new` may omit `sessionId`,
  and the daemon names the new session in its reply. A 0.7 client always sends an
  id and is unaffected; the 0.8 TUI omits it for a fresh session, which a 0.7
  daemon refuses as `invalid_params`. `phern agents shutdown`, and the next command
  that needs one starts the new one.
- **Sessions carry on.** The log format is still 3.
- **Spilled output is written, then named.** A blob reaches disk before the record
  that points at it, with no staging step. A 0.7 run that died mid-spill may have
  left a `.staging` directory beside its blobs under `$PH_HOME/spill`; 0.8 neither
  finishes nor collects it, so a record naming such a blob reads as one that is not
  there.
- **For code built on pH:**
  - `Context.emit` no longer takes `contained=`: every `emit` logs a failing
    listener and runs the rest, and a listener that must refuse belongs on
    `serial` or `waterfall`.
  - `SpillStore` writes only through `try_save`, `try_save_text` and
    `try_save_all`, before the record naming the blob is appended; `plan` names a
    blob before it is written. `reserve`, `commit`, `save`, `save_text` and
    `locator_for` are gone, and so are `ph.paths.holds` and `is_atomic_temp`.
  - `ph.persistence.resume_session(ctx, session_id, header, events)` takes the log
    its caller read; `open_session` reads and resumes. A `SessionPersistence`
    backend raises `NoStoredSession` when nothing is stored under an id, and its
    `exists` takes `family=`.

## Upgrading from 0.6

0.7 takes the daemon off the clock — nothing wakes up to ask whether something
changed; a wait ends when the thing it waits on happens, or at a deadline named in
advance — and gives every host the same orderly stop on a signal. What that
changes for an 0.6 setup:

- **Restart the daemon.** It speaks protocol 7: `daemon/status` drops
  `sweepEvery`, `heartbeatEvery`, `watchEvery` and `invariantsEvery` for
  `nextRelease`, `socketWatch` and `checkInvariants`, and a client and a daemon
  from different releases refuse each other's reply. `phern agents shutdown`, and
  the next command that needs one starts the new one.
- **Sessions carry on.** The log format is still 3. A root with a live schedule no
  longer writes `schedule/heartbeat` every five minutes, and a log that already
  holds heartbeats still opens, since each was written `ignorable`. Whether a
  daemon is alive is a question `phern agents doctor` or the OpenTelemetry sink
  asks it, not a record in the log.
- **A session has one writer, whoever it is.** The lease is the store's now, so
  `phern -p --session x` against a session a daemon or another `phern -p` holds is
  refused with `session_already_active` instead of appending beside it. The lease
  is an `flock`, so a process that crashed gives it back by dying; there is no
  stale lock to clear.
- **A signal stops a run the way `phern agents shutdown` does.** The first
  `SIGTERM` or Ctrl-C to `phern daemon`, `phern -p`, `--mode json`, transcript or
  `--mode rpc` suspends each sub-agent in its own log (resumable, no restart
  attempt spent), stops child processes and writes the logs; a one-shot run prints
  `stopped on SIGTERM` and exits with the signal's status. A stop still running
  15 seconds later is ended, killing what the process still owns, and a second
  signal does that at once.
- **The one-shot modes resume sub-agents.** `phern -p --session`, `--mode json`,
  transcript and `--mode rpc` run the daemon's resume sweep before the first
  prompt, so a child an interrupted run left `running` is readmitted, or ended if
  it is spent, rather than left as it was.
- **The daemon logs to `$PH_HOME/logs/daemon.log`.** A detached daemon's warnings
  used to go to the null device.
- **For code built on pH:** `ph.resources.install_lifecycle` is gone. A host takes
  signals with `stop_on_signals` (async, for its whole life) or
  `run_until_signaled` (a one-shot run from synchronous code). Also gone:
  `ph.cancel.POLL_SECONDS`, `ph_runtime.lifecycle.POLL_SECONDS`, and the schedule
  seam's `HEARTBEAT` and `ScheduleService.heartbeat`. `ph.cancel.first_of` takes
  any `Waitable`, a `ph.wall_clock.Alarm` among them.

## Upgrading from 0.5

0.6 gives every session its own log — a sub-agent's records are in the sub-agent's,
so a restarted daemon brings each child back where it stopped — and lets the
daemon's scheduler sleep until something is due. What that changes for an 0.5 setup:

- **Restart the daemon.** It speaks protocol 6, and a client and a daemon from
  different releases refuse each other's new fields. `phern agents shutdown`, and
  the next command that needs one starts the new one.
- **Sessions from 0.5 do not carry over.** The session log is format 3, and a
  format-2 log is refused when it is opened (`session header version must be 3, got
  2`) rather than migrated: a child's records moved out of its parent's log, and an
  old log read as a new one would put them in the wrong place. Finish what is
  running on 0.5 first. The old logs stay where they are, as plain JSONL.
- **The code graph and the text index are rebuilt once.** Each keeps one index per
  workspace now (`$PH_CACHE/code-graph/<workspace>/graph.db`,
  `$PH_CACHE/text-index/<workspace>/<embedder>`), so the first `code_index` or
  `text_index` after the upgrade starts from scratch. The 0.5 indexes are not read,
  and go with the rest of `$PH_CACHE` when you clear it. A `path:` on the
  `text-index` row is now the directory the indexes go under, not the index itself.
  A worktree's index is removed once the worktree has been gone a week.
- **Schedules fire on time, and nothing polls.** The daemon sleeps until the next
  appointment instead of checking every five seconds, so a schedule runs at its
  moment, and a daemon with nothing scheduled wakes for none. `phern agents doctor`
  shows when the scheduler next wakes where it used to show the tick.
- **The `settings` row is gone.** Nothing read `ctx.settings`. A profile that still
  patches `settings` or names `settings-local` is refused at load; delete the row.
- **Telemetry ships a record once the log holds its event** — at the next flush, not
  on the append — so an exported `(session, seq)` always names an event the log
  keeps. What only the last write at shutdown puts on disk is not exported
  ([`docs/seams/session_telemetry.md`](docs/seams/session_telemetry.md)).

## Upgrading from 0.4

0.5 gives each kind of setting one owner and makes a session's profile its own, so
some of what an 0.4 setup did now lands somewhere else:

- **Restart the daemon.** It speaks protocol 5, and a client and a daemon from
  different releases refuse each other's new fields — which an upgrade reaches,
  since a command connects to whatever daemon is already listening. `phern agents
  shutdown`, and the next command that needs one starts the new one.
- **Your profile files keep working, then fold.** A file still in the old list
  format, and the `<name>.d/` drop-ins `/sandbox` wrote, are read as before, and
  `phern doctor` names each one; `phern profiles fold` turns them into one named
  file — `extends` plus the rows that differ — and changes nothing a session runs
  with.
- **A row in the wrong file is refused, by name.** Persistence, telemetry and the
  job bound are the daemon's (`$PH_HOME/daemon.yaml`, under `rows:`), and screens
  and footer readings the TUI's (`$PH_HOME/tui.json`); a profile that still sets
  one says which file to move it to. `phern config` shows each row's kind.
- **`--model` alone is a key** of the profile's `models` list (`phern config --row
  models`). A model the list does not hold needs `--provider` beside it. A child's
  `model` is a key the same way, where it was a model name.
- **Sessions carry on.** The log format is unchanged, and a session from 0.4
  records the profile it starts on at its next start; from then on it comes back as
  its log says, whatever `--profile` a later run gives.

## Optional plugins

Packages in this workspace that ship rows no profile mounts by default — because
each brings a third-party dependency not every deployment wants, or, for
`ph-clm`, changes how the model treats its own context. Add a row and they are
there:

| package | rows | tools |
|---|---|---|
| [`ph-code-graph`](packages/ph-code-graph/) | `code-graph` | `code_index` / `code_graph` — a tree-sitter code graph: search by prose, find definitions, callers, callees, transitive impact, biggest symbols. Every answer is a `path:start-end`. |
| [`ph-text-index`](packages/ph-text-index/) | `text-index`, `text-index-local` | `text_index` / `text_search` — semantic retrieval over documents on a local turbovec index, answering with the passage **and** its `path:start-end`. |
| [`ph-clm`](packages/ph-clm/) | `clm-context`, `clm-mirror` | `context_sections` / `context_tombstone` / `context_replace` / `context_rewrite` / `context_recall` / `context_diff` — the model edits its own context, section by section, with these tools or by editing a file of its context with the tools it already has. Every edit is a surface replace, so the log keeps the originals. |

Each registers a **bundle**, so `--profile rlm-indexed` is `rlm-stable` plus
the first two — the RLM asking a codebase and a corpus about themselves instead
of reading them. Under Code Mode they arrive as `await tools.code_graph(...)`
and `await tools.text_search(...)`, because every registered tool is in the
generated SDK listing:

```bash
phern --profile rlm-indexed --provider llama --model <model> --mode tui
```

An install missing either distribution is simply not offered that profile —
`phern doctor` lists what this install can compose — rather than being offered one
that fails at mount. Layer one on its own with `--patch`:

```bash
phern --patch '{insert: [{id: code-graph, name: code-graph}]}' --profile llama -p "..."
```

`phern config --row code-graph` prints what each row accepts, and all of them
report to `phern doctor`. Each package's README is the honest account of what its
dependencies can and cannot do — worth reading before relying on one.

**Provision before an agent needs it.** Both plugins fetch something on first
use — grammars, an embedding model — and finding that out mid-turn is the wrong
time. Each ships a command a person triggers, and reports readiness to
`phern doctor`:

```bash
/code-graph status     /code-graph install      # tree-sitter grammars
/text-index status     /text-index install      # the embedding model
```

`ph-text-index` additionally takes `preload: true`, which loads the model at
mount and refuses the mount if it cannot — for a daemon or a scheduled run,
where nobody is present to type either command. Both plugins also install a
`SKILL.md`, so an RLM sees one catalog line and reads the page only when it
needs it.

Anything either plugin caches lives under `$PH_CACHE` — grammars, the model
weights, the indexes — so `phern doctor` can name it and deleting that root
reclaims it.

**Neither plugin watches the filesystem.** An index changes when its tool runs,
and at no other moment. Re-running is cheap because both ask **git or jj** what
changed first ([`ph.seams.changes`](docs/seams/README.md)) — a file the version
control vouches for is never re-read, and for `text_index` never re-embedded.
Which of the two answers is the workspace provider's to state, so a `jj` tier
gets jj and a `git` worktree gets git; a tree under neither falls back to
content hashing and is merely slower.

## Diagnostics

```bash
phern doctor                                  # roots, platform, available profiles
phern doctor --profile rlm-stable             # then mount it and let every row report
phern events                                  # the event producer/consumer matrix
phern --mode trajectory --session <id|path>   # audit a log, mounting nothing
```

`phern doctor` is the topology answered by the rows themselves — which containment
rung is in force, what the file rules reach, what runs model code, which
invariants hold. It creates nothing: no session, no agent, no provider call. A
profile that refuses to start is reported as a sentence rather than a traceback,
which is the case it exists for.

Housekeeping, both of which report by default and collect only with `--remove`:

```bash
phern workspaces gc          # the git trees agents left behind, across every stored session
phern attachments gc         # media no stored session references
```

## Tests

```bash
./test.sh                 # lint, format, types, tests — CI's four gates, in CI's order
./test.sh test -k prefix  # anything after the gate goes to pytest
./test.sh smoke           # the live local-server gate; needs llama-server running
./test.sh --help
```

`./test.sh` rather than bare `pytest`: on macOS it also exports a `$TMPDIR`
short enough for a unix socket, reports which optional backends are installed
and what each missing one costs, and fails on any failure that is not on its
enumerated list of platform gaps — which is empty on both platforms today.

## Building a release

Every member is released together, at one version: each pins the others exactly
(`ph-core==0.8.0`), so a change in any of them is a release of all eight.

```bash
# 1. The version, everywhere it is written: `version` and the `==` pins between
#    members in packages/*/pyproject.toml, and `__version__` in
#    packages/ph-core/src/ph/__init__.py. Then:
uv lock                                          # moves the eight workspace entries
./test.sh                                        # the four gates, on the new version

# 2. A wheel and a source distribution per member. dist/ keeps only its
#    .gitignore between releases.
rm -f dist/*.whl dist/*.tar.gz
uv build --all-packages --out-dir dist

# 3. The set installs together from the files alone, as PyPI will serve it.
uv venv /tmp/phern-release
uv pip install --python /tmp/phern-release/bin/python dist/*.whl
/tmp/phern-release/bin/phern --help

# 4. PyPI: the wheels and the sdists.
uv publish dist/*
```

Both carry the package's README as the PyPI description, so README changes —
the upgrade notes above among them — go in before step 2. A release that changes
what a daemon and a client say to each other bumps `PROTOCOL_VERSION`
(`packages/phern/src/ph_app/protocol.py`), and one that an older build could no
longer read a log of bumps `SESSION_FORMAT_VERSION`
(`packages/ph-core/src/ph/session/events.py`). Each records its reason beside the
number, and each belongs in the upgrade notes.

## Where to read more

| | |
|---|---|
| [`DESIGN.md`](DESIGN.md) | What pH **is**, verified against the source, with `file:line` citations. Start here for the architecture, the invariants, and the two axes (topology and log). |
| [`docs/README.md`](docs/README.md) | The components: the cookbook (add a plugin, a tool, an adapter, a seam), reference for every seam, skill authoring, and the per-phase dev notes. |
| [`plans/`](plans/) | Why each decision fell where it did, and what remains. |

Where the documentation and `DESIGN.md` disagree, `DESIGN.md` wins and the
documentation is a bug.

## License

MIT — see [`LICENSE`](LICENSE).
