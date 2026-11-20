# phern

*The `phern` command and its seven output modes, with the Textual TUI and the
browser tab among them, the daemon and its client, and the three real model
wires.*

`ph-core` is the harness, and `phern` is what a user runs. It owns no seam, and
it publishes no service that a profile requires. It contributes four things:

- **Front ends**: text, json, transcript, rpc, tui, web and trajectory.
- **The supervisor**, which lets a run outlive the terminal that started it.
- **The adapters**, which speak Anthropic, Gemini and the OpenAI-compatible
  wire.
- **The profile table**, which composes everything that is installed into a
  name that you can type.

```bash
uv tool install ./packages/phern        # the CLI, the core and the six bundle packages
LLAMA_MODEL=<model> phern --profile llama -p "what is in this repo?"
phern --profile tui --provider llama --model <model> --mode tui
```

`phern` depends on the six bundle packages, so an install composes all eleven
profiles. Seven of them need no bundle: `anthropic`, `base`, `deepseek`,
`google`, `headless`, `llama` and `tui`. The `rlm*` profiles need bundles. See
*Profiles* below.

## The commands

| Command | What it does |
|---|---|
| `phern -p "…"` | Runs one prompt and prints one answer. `-a/--attach FILE` sends a file with the prompt, and it is repeatable. |
| `phern --mode json \| transcript \| rpc \| tui \| web \| trajectory` | Selects what reaches stdout, or which front end runs. |
| `phern doctor [--profile …]` | Prints the three path roots. It then mounts a profile, and every row reports what activated. |
| `phern events [--json]` | Prints the event producer/consumer matrix, which is generated from the declaration registry. |
| `phern config [--row id] [--all]` | Prints what every row accepts, its default, and what this profile sets. The profile is composed and not mounted: the command starts no agent and opens no session. |
| `phern --dump-config` | Prints the composed rows in order, before anything runs. |
| `phern daemon` | Runs the supervisor. |
| `phern agents …` | The client that talks to the supervisor. With no command, it lists the roots. The nine commands are `status`, `attach`, `send`, `schedule`, `doctor`, `owed` (which stored roots still owe work), `resume` (mount them), `probe` (`startup`, `ready` or `live`, exit 0 for yes) and `shutdown`. |
| `phern profiles …` | Named profiles: `show [--full]`, `session <id>`, `fold`, `diff` and `adopt`. `diff` shows what the current version of a profile would change in each session that is on it. `adopt` takes that version at the next start of those sessions. |
| `phern workspaces gc` | Finds the git trees that agents left behind, across every stored session. It reports by default, and collects with `--remove`. |
| `phern attachments gc` | Finds the media that no stored session references. The same rule applies. |

`phern doctor` reports what the flags and the environment of *this invocation*
would produce. `phern agents doctor` reports what is **in force** in the running
daemon. These are different questions, so know which one you asked.

### The modes, and why the choice matters

`json` and `rpc` emit the envelopes of the session log **itself**, and not a
rendering for each mode (I-7). So a wrapper that streams from a pipe and a tool
that reads the stored JSONL parse one format. `transcript` reads
`session.transcript()`, which is what the user saw. A compacted conversation
therefore still shows the turns that the user had, where the model surface
deliberately shadows the replaced ranges. `text` is the default. `trajectory`
audits a stored log and mounts nothing:
`phern --mode trajectory --session <id|path>`.

## The TUI

```bash
phern --mode tui --provider llama --model <model>     # new session
phern --mode tui --resume 20260908T041607-1c5576      # reopen one
phern --mode tui --no-spawn                           # refuse rather than start a daemon
phern --mode tui --keep-daemon                        # the daemon it starts is a service
```

The front end is a client of the daemon. It talks to the daemon over
`$PH_RUNTIME/daemon.sock`, and it starts an ephemeral daemon if nothing is
listening. So closing the TUI does not end the turn.

Every front-end action is a `TuiVerb`, and it is reachable in three ways:

- As a slash command. `local_commands` builds one definition for each verb, so
  the palette lists it and the prompt completes it.
- As a Textual action.
- As a key.

| Slash | Key | |
|---|---|---|
| `/commands` | `ctrl+k` | Browse every command, those of the daemon and those of the client. |
| `/model [<key> \| <provider>/<model>]` | `ctrl+p` | Choose the provider and model. With a key or a route, move to it without the picker. |
| `/theme` | `ctrl+y` | Choose a theme: `ph-dark`, `ph-light`, `high-contrast`, four catppuccin flavors, or one of your own in `$PH_HOME/themes`. |
| `/sessions` | `ctrl+r` | Reopen a stored session. |
| `/history` | `alt+r` | Search the prompts that you have sent. |
| `/permissions` | `ctrl+g` | Change what pH may do without asking. |
| `/view <what>` | | Show or hide part of the view. See below. |
| `/tools` | | List what the model may call. |
| `/skills` | | List the skills that are installed here. |
| `/login` | | Provide a provider credential for this process. |
| `/profile …` | | The profile of the session: `show`, `diff`, `save <name>`, `use <name>`, `clear [row]`. This command belongs to the daemon, which registers it on every root. |
| `/attach <path> …` | | Attach files to the next prompt. |
| `/quit` | `ctrl+d` | |

A verb with no argument in this table refuses one. `/theme dark` opens nothing
and says `/theme takes no argument`.

`/view` takes one word. Three of the words have keys of their own:

| `/view …` | Key | |
|---|---|---|
| `results` | `ctrl+o` | Show or hide tool calls. Each card is whole: its title, its input and what it returned. |
| `thinking` | `ctrl+t` | Show or hide the reasoning of the model. |
| `sidebar` | `ctrl+b` | Show or hide the sidebar. |
| `tools` | | Show or hide the tools panel of the sidebar. |
| `skills` | | Show or hide the skills panel of the sidebar. |
| `all` | | Both panels at once. It shows them, or hides them when both are already shown. |

Every toggle except `sidebar` is saved to `tui.json`, so a view that you hid
stays hidden in the next session. `sidebar` belongs to this window, and it is
forgotten on exit.

**Copying.** Drag across the transcript, then press `ctrl+c`. The copy key of
the terminal itself (`ctrl+shift+c` in most Linux terminals) finds nothing,
because the TUI holds the mouse. The answer of a model is copied as the Markdown
that the model wrote, and not as what was drawn: a heading keeps its `#`, a
fence keeps its backticks, and a table keeps its pipes. The text reaches the
system clipboard through an OSC 52 escape. Most terminals honor that escape, and
macOS Terminal.app does not. To copy what is on the screen instead, hold `shift`
while you drag. That hands the selection to the terminal.

Rows contribute their own screens and commands through `ctx.tui_screens` and
`ctx.commands`, and these arrive with the same three routes. `/trajectory`
(`f2`) and `/diff` (`f3`) are two such screens. `/diff` shows every change to
what the model sees, grouped by what it changed, for each edit or for the whole
history. A row contributes each of the two screens, and neither is built in. If
its row is removed, a screen takes its key and its palette entry with it.
Screens are presentation, so every named profile ships them
(`profiles/presentation.yaml`), and a profile may not remove one.
`hidden_screens` in `tui.json` hides a screen from this terminal.

### `$PH_HOME/tui.json`

This file holds keybindings and preferences. **Never hard-code a key check.**
Every binding is a named field, and the name of the field is also the Textual
binding id. So one `set_keymap` call rebinds the whole app, screens and modals
included. The key of a contributed screen can be remapped exactly like a
built-in key.

```json
{
  "sidebar": "right",
  "turn_notification": "bell",
  "show_thinking": true,
  "show_tool_results": true,
  "show_tools": true,
  "show_skills": true,
  "daemon_keep_alive": "0",
  "hidden_screens": [],
  "keybindings": { "command_palette": "ctrl+k", "quit": "ctrl+d" }
}
```

`daemon_keep_alive` is how long a daemon that this TUI starts stays after the
terminal closes. It is written like `--keep-alive` (`"30s"`, `"5m"`). With `"0"`
the daemon leaves at once.

The theme is not in this file. `/theme` writes it to
`$PH_HOME/themes/theme-profile.yaml`. A `theme` key that an older pH left here
is ignored, and it is dropped on the next write.

A file that fails to parse does not stop the TUI from starting: the TUI starts
on the defaults and says so. An unrecognized binding id under `keybindings` is
kept and not dropped, because it may belong to a plugin screen. Any other
unrecognized key is ignored.

## The browser tab

```bash
phern --mode web --profile llama                      # 127.0.0.1:8000
phern --mode web --port 8080 --open
```

`textual-serve` runs a real `PHTuiApp` as a subprocess and streams its frames.
So the browser shows the terminal: there is one layout, and not two. It prints
three things before it binds, and each of them matters:

- **The token in the URL is the whole of the authentication.** There is no TLS
  and there are no users. Treat the URL like the terminal that it came from.
- **Every tab of one launch is on one session.** A second tab joins the
  conversation, and a second `phern --mode web` is a new conversation.
- A `--host` that is not loopback reaches anyone who can route to the port.

It needs the `web` extra: `uv tool install "phern[web]"`, or
`uv tool install "./packages/phern[web]"`. Without the extra, `--mode web` fails
with the install line and not with an `ImportError`.

## The daemon

```bash
phern daemon --profile llama                  # the profile new sessions start on
phern daemon --max-concurrent-children 6      # across every root; the rest queue
phern daemon --passivate-after 30             # minutes of quiet before a root is released, or `off`
phern daemon --ephemeral                      # exit once no client, root or appointment needs it
```

There is one `anyio` task for each root, and the client is not that task. A root
owns a mounted profile, a session, an agent and a queue. Its task drains that
queue whether or not anybody is attached. Attaching subscribes a connection to
the events of the root, and detaching unsubscribes it. **Neither starts or stops
the work**, which is why leaving costs nothing.

The socket is per boot and per user. A stale socket from a crashed daemon is
cleared. A live socket is refused and not taken over. On Linux, logind removes
`$XDG_RUNTIME_DIR` at logout for a user who is not lingering. A daemon can then
keep running and *lose its socket*. After that, every client is told to start a
daemon, and the leases that the first daemon still holds refuse the new one.
`phern doctor` and `phern daemon` say so in advance, and
`loginctl enable-linger` is the fix.

## Profiles

The table is in `src/ph_app/profiles.py`, and the documents are in
`src/ph_app/profiles/`.

| `--profile` | Layers | Credential |
|---|---|---|
| `base` | `ph-base` | — |
| `headless` | `base` + the fake adapter | — |
| `tui` | `headless` + `tui.yaml` (writable workspace, `ask_user` armed) | — |
| `llama` | `base` + a local llama.cpp route | `LLAMA_API_KEY` (a formality that llama.cpp ignores, but it must be set) |
| `deepseek` | `base` + DeepSeek over the OpenAI-compatible wire | `DEEPSEEK_API_KEY` |
| `anthropic` | `base` + the messages API | `ANTHROPIC_API_KEY` |
| `google` | `base` + Gemini (the one route that declares video, so `uploads` has a provider) | `GEMINI_API_KEY` |
| `rlm` | `tui` + the `rlm` bundle | needs `ph-rlm` |
| `rlm-stable` | `rlm` + `stabilize`, with the gates on | needs `ph-rlm`, `ph-stabilize` |
| `rlm-indexed` | `rlm-stable` + `utcp` + `code-graph` + `text-index` | also needs `ph-utcp`, `ph-code-graph` and `ph-text-index` |
| `rlm-clm` | `rlm` + the `clm` bundle: the model edits its own context. It is on `rlm` and not on `rlm-stable` | needs `ph-rlm`, `ph-clm` |

Every row of that table also layers the **`stabilize` bundle**, so every profile
compacts at 85% of its window and offers `/compact`. The bundle is layered
*optionally*. An install without `ph-stabilize` composes each of these profiles
unchanged and never compacts. `phern` depends on `ph-stabilize`, so that is an
install from which someone removed the package. `phern doctor` reports which
rows activated.

**A profile is offered only if every layer that it names resolves.**
`available_profiles()` asks exactly the question that `resolve_profile` will
answer, so a `--help` line and a command line cannot disagree. An install that
lacks a bundle sees no `rlm-indexed`, and not an `rlm-indexed` that fails at
mount. The refusal names the package to install. A `--profile` value that is a
path to a `.yaml` file is used directly. So a scenario test or a one-off
deployment is one file and not an install step.

## Adjusting it

Every pH row uses the same layers:

- The shipped documents.
- `rows:` in `$PH_HOME/daemon.yaml`, for the machinery of the host itself.
- Your named profile at `$PH_HOME/profiles/<name>.yaml` (`extends` plus the rows
  that differ), for what an agent runs with.
- `--patch`, for one run.

Each row belongs to one of those files by what it shapes. A row that is set in
the wrong file is refused, and the refusal names the file that it belongs in.
`phern config` shows the kind of each row. Screens and footer readings belong to
`tui.json`. What is specific to this package is the rows that it registers:

| Row | Config | Default |
|---|---|---|
| `llm-anthropic` | `provider`, `baseUrl`, `apiKeyEnv`, `contextWindow`, `defaultMaxTokens`, `accepts`, `maxAttachmentBytes`, `uploads`, `filesBeta`, `maxImageEdge`, `usableImageEdge`, `cacheControl` | `anthropic`, `ANTHROPIC_API_KEY`, `200000`, `8192`, images + PDF, 5 MiB, prompt caching on |
| `llm-google` | the same fields without `filesBeta` and `cacheControl`, plus `uploadReadyMs` | `1048576` window, images/audio/video/PDF, 20 MiB, video routed through the Files API |
| `llm-openai-compatible` | `profiles: [ProviderProfile, …]` | one entry for each route. `llama` and `deepseek` insert this row |
| `tui-screen-trajectory` | — | in every named profile (`presentation.yaml`). `tui.json` hides it for one terminal |
| `tui-screen-diff` | — | in every named profile (`presentation.yaml`). `tui.json` hides it for one terminal |

```yaml
# $PH_HOME/profiles/anthropic.yaml: a different model ceiling, caching off
extends: anthropic
rows:
  - id: llm-anthropic
    config:
      provider: anthropic
      apiKeyEnv: ANTHROPIC_API_KEY
      contextWindow: 200000
      defaultMaxTokens: 16384
      cacheControl: false
```

**A list is replaced and not merged.** On these rows that is a trap. An overlay
that restates one field of an OpenAI-compatible route must restate the whole
route entry. Otherwise the route inherits `api.openai.com` from the row default,
and a local deployment quietly calls a hosted provider. The comments of
`llama.yaml` have the worked example.

An `apiKeyEnv` is a **name**, and never an interpolation. The adapter resolves
it at the request edge, so the value never enters a row, an event, or a child
process (I-3). `${env:…}` interpolation is available for everything that is not
a secret, with `${env:VAR:-default}` for a fallback.

## Upgrading from 0.9

0.10 makes a restart something that an orchestrator can own. The daemon says
which stored roots still owe work, and resumes them when it is asked. It
continues a turn that a crash cut short. It answers startup, readiness and
liveness probes. It says so when a session log could not be written. For a 0.9
setup, this changes the following:

- **Restart the daemon.** It speaks protocol 10. `daemon/status` carries
  `unwritten` and `probe`, so a 0.9 client refuses the status of a 0.10 daemon.
  `sessions/owed`, `sessions/resume` and `daemon/probe` are new. Run
  `phern agents shutdown`, and the next command that needs a daemon starts the
  new one.
- **Sessions from 0.9 carry over.** The session log is still format 4.
- **A log that could not be written is exit 3.** `phern -p`, `--mode json` and
  `--mode transcript` exit 3 when a session log could not be written, whether it
  is the log of the root or of a child. They first print what the run produced,
  and stderr names each log and the reason. `--mode rpc` does the same when its
  peer closes the pipe. `phern daemon` exits 3 in the same way, and counts the
  logs. Its readiness is no while a log that it holds is unwritten. A script
  that branches on the exit code of `-p` sees 3 beside 1 and 2: the answer
  stands, and the log behind it is missing records.
- **New: a restart that an orchestrator owns.** `phern daemon --resume-owed`
  resumes what the last daemon owed, while the new daemon starts.
  `--continue-interrupted` continues a turn of a user that was cut short, as
  well as an unattended turn. `--probe-bind host:port` answers the startup,
  ready and live probes over HTTP. Under systemd the daemon sends `READY=1` and
  `WATCHDOG=1`. `phern agents owed`, `phern agents resume` and
  `phern agents probe` ask the same from a shell.
  `phern service start|stop|restart|status|resume|watch` does the part of the
  orchestrator on a host that has none.
  [`contrib/`](https://github.com/chastabor/pH/blob/main/contrib/README.md) has
  a system unit, a user unit and a Kubernetes manifest.
- **New: `phern doctor sessions [--repair]`.** It reads every stored log
  strictly and names the first line that it refuses. `--repair` cuts each
  damaged log back to what reads, and keeps the original beside it. One damaged
  child log no longer takes its family offline.
- **The lease is `flock` or nothing.** A session store on a filesystem without
  `flock(2)` is refused (`LockUnsupported`). It is not given a lock that a crash
  cannot release. `phern doctor` names the lock mechanism and the filesystem of
  the lease directory. ph-core needs `filelock>=3.32`.
- **The ending of a child wakes an idle parent**, in the daemon and in rpc mode.
  A child that the sweep gives up on tells its parent so. A root that is held
  for a credential holds its children, and the key that arrives readmits them.
- **Rows in the log layer place themselves.** A row that keeps, exports or
  flushes the log goes beneath the agents, wherever a profile writes it. So a
  telemetry sink that is added at the end of `daemon.yaml`, such as
  `{id: telemetry-otel, name: session-telemetry-otel}`, ships the shutdown tail
  too. A ledger record carries why its work ended (`event.reason`). A failed
  write is an `error` ops record with its reason. A profile that keeps the log
  with no checkpoint row gets a warning at mount.
- **For code built on pH:**
  - A plugin that keeps, exports or flushes the log says where with
    `@plugin(..., log_layer=...)`: `storage`, `telemetry`, `export` or
    `checkpoint`.
  - `AgentCancelCause.kind` is `user`, `parent` or `disposed`. `hook`, `legacy`
    and `reason` are gone.
  - `SessionStore.flush` says `session/unwritten`, with the error, for a flush
    that raised. `ph.persistence.unwritten_at_unwind(scope)` answers every log
    that a mount left incomplete, for a host that mounts its own.
  - `run_rpc` returns that record, and `PrintResult`, `JsonResult` and
    `TranscriptResult` carry it as `unwritten`.

## Upgrading from 0.6

- **Restart the daemon** (`phern agents shutdown`). It speaks protocol 7, and a
  client and a daemon from different releases refuse the `daemon/status` of each
  other.
- **Sessions carry on.** The log format is still 3. A root with a schedule no
  longer writes a heartbeat every five minutes, and logs that hold heartbeats
  still open.
- **Nothing in the daemon wakes on a clock.** The daemon releases quiet roots,
  checks invariants and watches the socket when something happens or when a
  deadline arrives, and not on a cadence. `phern agents doctor` shows the next
  release where it showed the sweep.
- **A session has one writer.** `phern -p --session x` against a session that a
  daemon or another run holds is refused with `session_already_active`.
- **A signal stops a run in the same way as `phern agents shutdown`.** `SIGTERM`
  or Ctrl-C suspends each sub-agent so that it can be resumed, and writes the
  logs before the exit. The one-shot modes now resume the sub-agents of an
  interrupted session first.
- **The daemon logs to `$PH_HOME/logs/daemon.log`.**
- **For code built on pH:** `ph.resources.install_lifecycle` is gone. Use
  `stop_on_signals` or `run_until_signaled`.

## Upgrading from 0.5

- **Restart the daemon** (`phern agents shutdown`). It speaks protocol 6, and a
  client and a daemon from different releases refuse the new fields of each
  other.
- **Sessions from 0.5 do not carry over.** The session log is format 3. A 0.5
  log is refused when it is opened, and it is not migrated. Finish what is
  running on 0.5 first. The old logs stay on disk as plain JSONL.
- **The records of a sub-agent are in its own log.** A front end gets the family
  of a root from the daemon: from `session/children`, and from the
  `session.children` notice whenever a row moves. It does not get the family
  from the stream of the root. A restarted daemon brings each child back where
  it stopped.
- **The scheduler sleeps until something is due**, and does not check every five
  seconds. So a schedule runs at its moment. `phern agents doctor` shows its
  next wake where it showed the tick.
- **The code graph and the text index are rebuilt one time.** There is now one
  index for each workspace. A `path:` on the `text-index` row is the directory
  that the indexes go under.

## Upgrading from 0.4

- **Restart the daemon** (`phern agents shutdown`). It speaks protocol 5, and a
  client and a daemon from different releases refuse the new fields of each
  other.
- **Fold your profiles.** A file in the old list format, and the `<name>.d/`
  drop-ins that `/sandbox` wrote, are read until `phern profiles fold` folds
  them into one named file. `phern doctor` names each one until then.
- **A row in the wrong file is refused, by name.** Persistence, telemetry and
  the job bound belong in `daemon.yaml`, and screens belong in `tui.json`.
- **`--model` alone is a key** of the `models` list of the profile. A model that
  the list does not hold needs `--provider` beside it. The `model` of a child is
  a key too.
- **Sessions carry on.** The log format is unchanged. A session from 0.4 records
  the profile that it starts on at its next start, and after that it comes back
  as its log says.

A session keeps the version of its named profile that it started on.
`/profile diff` and `phern profiles diff` say how the profile has moved, and
`/profile use` or `phern profiles adopt` takes the new version. A version that
will not mount is taken back at the start that tried it, and the session runs on
the version that it had.

## Limitations, and things that are deliberate

- **`ph_app` must not import a bundle**: `ph_rlm`, `ph_stabilize`,
  `ph_text_index`, `ph_code_graph`, `ph_utcp` or `ph_clm`. The `phern`
  distribution depends on all six wheels. `ph_app` composes their profiles
  through the `ph.bundles` entry-point group, and it reads their events
  (`subagent/*`) without importing the rows that emit them.
  `tests/test_app_layering.py` imports the entry points of the app in a fresh
  interpreter and fails if a bundle was loaded. Depending on a wheel decides
  what is installed. It does not give this module a name that it may write
  down. Keeping the two separate lets a bundle that nobody here ships compose a
  profile on the same footing.
- **Textual is bounded at both ends (`>=8.2.5,<9`), and both bounds are
  needed.** The floor is where the suite passes. `MarkdownStream` (the streaming
  append of the transcript) does not exist before Textual 5, and the committed
  SVG snapshots then narrow the floor further: 8.2.4 fails one of them. The
  ceiling guards `add_binding`. That function writes through a private
  `BindingsMap`, because the public `bind()` drops the id that `set_keymap`
  matches on. That id is what makes the key of a contributed screen rebindable
  like every other key.
- **The web UI is the terminal in a canvas.** It is not an HTML renderer on the
  same view model. That trade gives layout parity by construction.
- **A client reads no session file at all.** After P5-14 the daemon holds the
  session files and answers `sessions/browse`. That makes a front end on another
  machine possible, and it stops a client and a daemon from disagreeing about
  which `$PH_HOME` they meant.
- **The method vocabulary of the daemon is typed and closed** (`ph_app.verbs`,
  `ph_app.params`, `ph_app.payloads`). A field that a method does not take is a
  refusal that names the field, and not a silent drop. A typed edge exists to
  end one failure: a client that believed it had said something.

## Tests

`tests/` has 59 modules. They cover the CLI, the four non-interactive modes, the
daemon (framing, methods, mutations, lifetime, recovery, asks), the TUI (pilot
runs, remote verbs, screens, state) and the three adapters. There are also
committed Textual SVG snapshots. Read `test_non_guarantees.py` first. It tests
the *claims* and not a mechanism: the sentences that a user reads before
deciding whether to run six agents under one daemon. It also checks that
`phern doctor` and `phern agents doctor` still print those sentences.
