# ph-app

*The `ph` command: four output modes, the Textual TUI, the browser tab, the
daemon and its client — and the three real model wires.*

`ph-core` is the harness; this is the thing a person runs. It owns no seam and
publishes no service that a profile could not do without: what it contributes is
**front ends** (print, json, transcript, rpc, tui, web, trajectory), the
**supervisor** that lets a run outlive the terminal that started it, the
**adapters** that speak Anthropic, Gemini and the OpenAI-compatible wire, and
the **profile table** that composes everything installed into a name you can
type.

```bash
uv tool install ./packages/ph-app        # the CLI and core alone, about 60 MB
ph --profile llama --provider llama --model <model> -p "what is in this repo?"
ph --profile tui --provider llama --model <model> --mode tui
```

Installed on its own it composes seven profiles — `anthropic`, `base`,
`deepseek`, `google`, `headless`, `llama` and `tui`. The `rlm*` profiles need
bundles it deliberately does not depend on; see *Profiles* below.

## The commands

| command | what it does |
|---|---|
| `ph -p "…"` | one prompt, one answer. `-a/--attach FILE` sends a file with it, repeatable |
| `ph --mode json \| transcript \| rpc \| tui \| web \| trajectory` | what reaches stdout, or which front end runs |
| `ph doctor [--profile …]` | the three path roots, then mount a profile and let every row report what activated |
| `ph events [--json]` | the event producer/consumer matrix, generated from the declaration registry |
| `ph config [--row id] [--all]` | what every row accepts, its default, and what this profile sets. Composed, not mounted: it starts no agent and opens no session |
| `ph --dump-config` | the composed rows in order, before anything runs |
| `ph daemon` | run the supervisor |
| `ph agents …` | the client that talks to it: bare (list roots), `status`, `attach`, `send`, `schedule`, `doctor`, `shutdown` |
| `ph workspaces gc` | the git trees agents left behind, across every stored session. Reports by default; collects with `--remove` |
| `ph attachments gc` | media no stored session references. Same rule |

`ph doctor` reports what *this invocation's* flags and environment would
produce; `ph agents doctor` reports what is **in force** in the running daemon.
They are different questions and it is worth knowing which one you asked.

### The modes, and why the choice matters

`json` and `rpc` emit the session log's **own** envelopes rather than a per-mode
rendering (I-7), so a wrapper streaming from a pipe and a tool reading the
stored JSONL parse one format. `transcript` reads `session.transcript()` — what
the person saw — so a compacted conversation still shows the turns they actually
had, where the model surface deliberately shadows replaced ranges. `text` is the
default. `trajectory` audits a stored log and mounts nothing.

## The TUI

```bash
ph --mode tui --provider llama --model <model>     # new session
ph --mode tui --resume 20260908T041607-1c5576      # reopen one
ph --mode tui --no-spawn                           # refuse rather than start a daemon
ph --mode tui --keep-daemon                        # the daemon it starts is a service
```

The front end talks to the daemon over `$PH_RUNTIME/daemon.sock` and starts an
ephemeral one if nothing is listening, so closing the TUI does not end the turn.

Every front-end action is a `TuiVerb` reachable three ways — a slash command
registered into `ctx.commands` (so the palette lists it, the prompt completes
it, and `command/run` records it), a Textual action, and a key:

| slash | key | |
|---|---|---|
| `/commands` | `ctrl+k` | browse every command, the daemon's and the client's |
| `/model` | `ctrl+p` | choose the provider and model |
| `/theme` | `ctrl+y` | `ph-dark`, `ph-light`, `high-contrast` |
| `/sessions` | `ctrl+r` | reopen a stored session |
| `/permissions` | `ctrl+g` | change what pH may do without asking |
| `/thinking` | `ctrl+t` | show or hide the model's reasoning |
| `/tools` | `ctrl+o` | show or hide tool results |
| `/sidebar` | `ctrl+b` | show or hide the sidebar |
| `/login` | | provide a provider credential for this process |
| `/attach <path> …` | | attach files to the next prompt |
| `/quit` | `ctrl+d` | |

Rows contribute their own screens and commands through `ctx.tui_screens` and
`ctx.commands`, and they arrive with the same three routes — `/trajectory` is
one such screen, contributed by `tui.yaml` rather than built in, and it takes
its key and palette entry away with it if the row is removed.

### `$PH_HOME/tui.json`

Keybindings, theme and preferences. **Never hard-code a key check**: every
binding is a named field whose name doubles as the Textual binding id, so one
`set_keymap` rebinds the whole app, screens and modals included — a contributed
screen's key is remappable exactly like a built-in.

```json
{
  "theme": "ph-dark",
  "sidebar": "right",
  "turn_notification": "bell",
  "show_thinking": true,
  "show_tool_results": true,
  "keybindings": { "command_palette": "ctrl+k", "quit": "ctrl+d" }
}
```

A file that fails to parse does not stop the TUI starting: it launches on
defaults and says so. Unrecognized keys are kept rather than dropped, because
one of them is a plugin screen's binding id.

## The browser tab

```bash
ph --mode web --provider llama --model <model>     # 127.0.0.1:8000
ph --mode web --port 8080 --open
```

`textual-serve` runs a real `PHTuiApp` as a subprocess and streams its frames,
so the browser shows the terminal — one layout, not two. Three things it prints
before it binds, each of which is load-bearing:

- **the token in the URL is the whole authentication story** — no TLS, no users;
  treat the URL like the terminal it came from;
- **every tab of one launch is on one session** (a second tab joins the
  conversation; a second `ph --mode web` is a new one);
- `--host` anything but loopback reaches anyone who can route to the port.

Needs the `web` extra: `uv tool install --with "ph-app[web]" .`, or
`uv tool install "./packages/ph-app[web]"`. Without it, `--mode web` fails with
the install line rather than an `ImportError`.

## The daemon

```bash
ph daemon --profile tui --provider llama --model <model>
ph daemon --max-concurrent-children 6      # across every root; the rest queue
ph daemon --passivate-after 30             # minutes of quiet before a root is released, or `off`
ph daemon --ephemeral                      # exit once no client, root or appointment needs it
```

One `anyio` task per root, and the client is not it: a root owns a mounted
profile, a session, an agent and a queue, and its task drains that queue whether
or not anybody is attached. Attaching subscribes a connection to the root's
events; detaching unsubscribes it. **Neither starts nor stops the work**, which
is why leaving is free.

The socket is per boot and per user. A stale socket from a crashed daemon is
cleared; a live one is refused rather than stolen. On Linux, note that logind
reaps `$XDG_RUNTIME_DIR` at logout for a user who is not lingering — a daemon
can keep running and *lose its socket*, after which every client is told to
start one and the leases the first still holds will refuse it. `ph doctor` and
`ph daemon` say so in advance; `loginctl enable-linger` is the fix.

## Profiles

The table lives in `src/ph_app/profiles.py`, the documents in
`src/ph_app/profiles/`.

| `--profile` | layers | credential |
|---|---|---|
| `base` | `ph-base` | — |
| `headless` | `base` + the fake adapter | — |
| `tui` | `headless` + `tui.yaml` (writable workspace, `/trajectory`, `ask_user` armed) | — |
| `llama` | `base` + a local llama.cpp route | `LLAMA_API_KEY` (a formality llama.cpp ignores, but it must be set) |
| `deepseek` | `base` + DeepSeek over the OpenAI-compatible wire | `DEEPSEEK_API_KEY` |
| `anthropic` | `base` + the messages API | `ANTHROPIC_API_KEY` |
| `google` | `base` + Gemini (the one route declaring video, so `uploads` has a provider) | `GEMINI_API_KEY` |
| `rlm` | `tui` + the `rlm` bundle | needs `ph-rlm` |
| `rlm-stable` | `rlm` + `stabilize`, gates on | needs `ph-rlm`, `ph-stabilize` |
| `rlm-indexed` | `rlm-stable` + `code-graph` + `text-index` | needs both plugin distributions too |

**A profile is offered only if every layer it names resolves.**
`available_profiles()` asks exactly the question `resolve_profile` will answer,
so a `--help` line and a command line cannot disagree; an install missing a
bundle sees no `rlm-indexed` rather than one that fails at mount, and the
refusal names the package to install. A `--profile` value that is a path to a
`.yaml` is used directly, which is what makes a scenario test or a one-off
deployment one file rather than an install step.

## Adjusting it

The same three layers every pH row uses — the shipped documents, your overlay at
`$PH_HOME/profiles/<name>.yaml`, then `--patch` for one run. What is specific to
this package is the rows it registers:

| row | config | default |
|---|---|---|
| `llm-anthropic` | `provider`, `baseUrl`, `apiKeyEnv`, `contextWindow`, `defaultMaxTokens`, `accepts`, `maxAttachmentBytes`, `uploads`, `filesBeta`, `maxImageEdge`, `usableImageEdge`, `cacheControl` | `anthropic`, `ANTHROPIC_API_KEY`, `200000`, `8192`, images + PDF, 5 MiB, prompt caching on |
| `llm-google` | the same shape plus `uploadReadyMs` | `1048576` window, images/audio/video/PDF, 20 MiB, video routed through the Files API |
| `llm-openai-compatible` | `profiles: [ProviderProfile, …]` | one entry per route; this is the row `llama` and `deepseek` insert |
| `tui-screen-trajectory` | — | contributed by `tui.yaml` |

```yaml
# $PH_HOME/profiles/anthropic.yaml — a different model ceiling, caching off
- id: llm-anthropic
  config:
    provider: anthropic
    apiKeyEnv: ANTHROPIC_API_KEY
    contextWindow: 200000
    defaultMaxTokens: 16384
    cacheControl: false
```

**A list is replaced, not merged**, and on these rows that is the trap worth
naming: an overlay that restates one field of an OpenAI-compatible route must
restate the whole route entry, or it inherits `api.openai.com` from the row
default — a local deployment quietly calling a hosted provider. `llama.yaml`'s
own comments carry the worked example.

An `apiKeyEnv` is a **name**, never an interpolation. The adapter resolves it at
the request edge, so the value never enters a row, an event, or a child process
(I-3). `${env:…}` interpolation is available for everything that is not a
secret, with `${env:VAR:-default}` for a fallback.

## Limitations, and things that are deliberate

- **`ph-app` must not depend on `ph-rlm` or `ph-stabilize`.** It composes their
  profiles through the `ph.bundles` entry-point group and reads their events
  (`subagent/*`) without importing the rows that emit them —
  `tests/test_app_layering.py` enforces it.
- **Textual is pinned at both ends (`>=8.2.5,<9`), and both ends are
  load-bearing.** The floor is where the suite actually passes: `MarkdownStream`
  (the transcript's streaming append) does not exist before Textual 5, and the
  committed SVG snapshots then narrow it further — 8.2.4 fails one of them. The
  ceiling guards `add_binding`, which writes through a private `BindingsMap`
  because the public `bind()` drops the id that `set_keymap` matches on — and
  that id is what makes a contributed screen's key rebindable like every other.
- **The web UI is the terminal in a canvas**, not an HTML renderer on the same
  view model. That is the trade that buys layout parity by construction.
- **A client reads no session file at all.** After P5-14 the daemon holds them
  and answers `sessions/browse`, which is what makes a front end on another
  machine possible and stops a client and a daemon disagreeing about which
  `$PH_HOME` they meant.
- **The daemon's method vocabulary is typed and closed** (`ph_app.verbs`,
  `ph_app.params`, `ph_app.payloads`): a field a method does not take is a
  refusal that names the field, not a silent drop. A client that believed it had
  said something is the failure a typed edge exists to end.

## Tests

`tests/` — 36 modules covering the CLI, the four non-interactive modes, the
daemon (framing, methods, mutations, lifetime, recovery, asks), the TUI (pilot
runs, remote verbs, screens, state) and the three adapters, plus committed
Textual SVG snapshots. `test_non_guarantees.py` is worth reading first: it pins
the *claims* rather than a mechanism — the sentences a person reads before
deciding whether to run six agents under one daemon, and that `ph doctor` and
`ph agents doctor` still print them.
