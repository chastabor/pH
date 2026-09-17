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
a person installs and the command they get; the six libraries it is assembled
from publish under their own names — `ph-core`, `ph-rlm`, `ph-runtime-guest`,
`ph-stabilize`, `ph-text-index`, `ph-code-graph` — because each is usable on its
own. A front end of your own over `ph-core`, or a deployment that wants
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
upgrade.

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
phern --profile llama --provider llama --model "$(the model llama-server loaded)" \
   -p "what is in this repo?"
```

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
`phern agents attach` will show it to you again.

## The browser UI, on localhost

```bash
phern --mode web --provider llama --model <model>            # 127.0.0.1:8000
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
phern daemon --profile tui --provider llama --model <model>
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

Three layers, applied in this order:

1. the shipped bundle documents — `ph-core/src/ph/bundles/*.yaml` and
   `packages/phern/src/ph_app/profiles/*.yaml`;
2. **your overlay**, `$PH_HOME/profiles/<name>.yaml`, which patches a row by id
   without forking a bundle;
3. `--patch`, this run only, same grammar as a profile document.

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
attachments, your profile overlays), `PH_CACHE` (`~/.cache/ph` — safe to delete
wholesale) and `PH_RUNTIME` (the daemon socket). `phern doctor` prints where all
three resolved, and which tier `PH_RUNTIME` landed in.

## Optional plugins

Packages in this workspace that ship rows no profile mounts by default, because
each brings a third-party dependency not every deployment wants. Add a row and
they are there:

| package | rows | tools |
|---|---|---|
| [`ph-code-graph`](packages/ph-code-graph/) | `code-graph` | `code_index` / `code_graph` — a tree-sitter code graph: search by prose, find definitions, callers, callees, transitive impact, biggest symbols. Every answer is a `path:start-end`. |
| [`ph-text-index`](packages/ph-text-index/) | `text-index`, `text-index-local` | `text_index` / `text_search` — semantic retrieval over documents on a local turbovec index, answering with the passage **and** its `path:start-end`. |

Each registers a **bundle**, so `--profile rlm-indexed` is `rlm-stable` plus
both of them — the RLM asking a codebase and a corpus about themselves instead
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
