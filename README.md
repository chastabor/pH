# pH

*A plugin-composed Python agent harness.*

There is no privileged core. The agent loop, the model adapter, the tool
registry and the session log are all **rows in a YAML profile**, mounted by the
same loader in file order — so any of them can be replaced without forking
anything else. [`DESIGN.md`](DESIGN.md) is the specification;
[`docs/`](docs/README.md) is how to extend it.

## Install

```bash
uv sync                 # the workspace (packages/*) plus the dev group
uv run ph --help
```

Python 3.12 or newer. Every example below says `ph`; prefix it with `uv run`, or
activate `.venv`, whichever you prefer. `--mode web` additionally needs the web
extra (`ph-app[web]`).

## Point it at a model

A provider is a profile plus an environment variable holding the **name** of a
credential — the value is resolved at the request edge and never enters a row, an
event, or a child process.

| `--profile` | route | credential |
|---|---|---|
| `llama` | a local llama.cpp server (`LLAMA_BASE_URL`, default `http://127.0.0.1:9931/v1`) | `LLAMA_API_KEY` — a formality llama.cpp ignores without `--api-key`, but it must be set |
| `deepseek` | DeepSeek | `DEEPSEEK_API_KEY` |
| `anthropic` | Anthropic messages API | `ANTHROPIC_API_KEY` |
| `google` | Gemini | `GEMINI_API_KEY` |
| `base` · `headless` | no model route at all — the fake adapter, for wiring work | — |
| `tui` | `headless` plus a writable workspace: a person is present to answer approvals | — |
| `rlm` · `rlm-stable` | Code Mode, and Code Mode with the stabilization gates on | — |

`ph doctor` prints the list this install can actually compose. Profiles compose,
so `--profile llama` is `base` plus the llama route; the interactive profiles
layer `headless` and then their own rows.

## One prompt

```bash
export LLAMA_API_KEY=local
ph --profile llama --provider llama --model "$(the model llama-server loaded)" \
   -p "what is in this repo?"
```

- `-a/--attach FILE` sends a file with the prompt; repeatable.
- `--session <id>` continues (or resumes from disk) instead of starting fresh.
- `--mode json` and `--mode transcript` change the shape of what is printed;
  `text` is the default.

## The TUI

```bash
ph --profile tui --provider llama --model <model> --mode tui       # new session
ph --mode tui --resume 20260908T041607-1c5576                      # reopen one
ph --mode tui --no-spawn                                           # refuse rather than start a daemon
ph --mode tui --keep-daemon                                        # the daemon it starts is a service
```

The front end talks to a daemon and starts an ephemeral one if nothing is
listening, so closing the TUI does not end the turn — the root keeps working and
`ph agents attach` will show it to you again.

## The browser UI, on localhost

```bash
ph --mode web --provider llama --model <model>            # 127.0.0.1:8000
ph --mode web --port 8080 --open
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
  conversation; a second `ph --mode web` is a new one.
- `--host` anything other than loopback prints a further warning, because that
  authority then reaches anyone who can route to the port.

## The daemon

```bash
ph daemon --profile tui --provider llama --model <model>
ph daemon --max-concurrent-children 6      # across every root; the rest queue
ph daemon --passivate-after 30             # minutes of quiet before a root is released, or `off`
ph daemon --ephemeral                      # exit once no client, root or appointment needs it
```

The socket is `$PH_RUNTIME/daemon.sock`, per boot and per user. A stale socket
from a crashed daemon is cleared; a live one is refused rather than stolen.

## What is running, and stopping it

```bash
ph agents                                  # the roots this daemon is running
ph agents status <session>                 # what one root is doing, and a --since cursor
ph agents attach <session>                 # its history, then everything as it happens
ph agents attach <session> --until-idle    # exits 1 if the turn it stopped on errored
ph agents send <session> "carry on"        # queue a turn, starting or resuming the root
ph agents schedule <session> --every 3600000 --prompt "check the build"
ph agents doctor                           # socket, pid, uptime, provider, roots, capabilities
ph agents shutdown                         # stop it, and wait until it is actually gone
```

Attaching neither starts nor stops the work, so leaving is free. `ph agents
doctor` reports what is **in force** in the running process, unlike `ph doctor`,
which reports what this invocation's flags and environment would produce.

## Configuring

Three layers, applied in this order:

1. the shipped bundle documents — `ph-core/src/ph/bundles/*.yaml` and
   `ph-app/src/ph_app/profiles/*.yaml`;
2. **your overlay**, `$PH_HOME/profiles/<name>.yaml`, which patches a row by id
   without forking a bundle;
3. `--patch`, this run only, same grammar as a profile document.

```bash
ph config --profile llama                            # every knob every row accepts
ph config --row autonomous --profile tui             # one row, by the name it is registered under
ph --dump-config --profile llama                     # the mount as written, in order
ph doctor --profile llama                            # what actually activated
ph --patch '{id: fs, config: {root: /tmp/scratch}}' --profile tui --mode tui
```

A patch entry is `{id: …, config: {…}}` to reconfigure, `{id: …, disabled:
false}` to arm a row a bundle ships off, `{id: …, remove: true}` to drop one, or
`{insert: [...]}` to add one. Note that a **list** in row config is replaced
rather than merged — an overlay that restates one field of a route must restate
the whole route entry, or it inherits that field's default.

Three roots, each overridable by its variable: `PH_HOME` (`~/.ph` — sessions,
attachments, your profile overlays), `PH_CACHE` (`~/.cache/ph` — safe to delete
wholesale) and `PH_RUNTIME` (the daemon socket). `ph doctor` prints where all
three resolved, and which tier `PH_RUNTIME` landed in.

## Diagnostics

```bash
ph doctor                                  # roots, platform, available profiles
ph doctor --profile rlm-stable             # then mount it and let every row report
ph events                                  # the event producer/consumer matrix
ph --mode trajectory --session <id|path>   # audit a log, mounting nothing
```

`ph doctor` is the topology answered by the rows themselves — which containment
rung is in force, what the file rules reach, what runs model code, which
invariants hold. It creates nothing: no session, no agent, no provider call. A
profile that refuses to start is reported as a sentence rather than a traceback,
which is the case it exists for.

Housekeeping, both of which report by default and collect only with `--remove`:

```bash
ph workspaces gc          # the git trees agents left behind, across every stored session
ph attachments gc         # media no stored session references
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
