# ph-text-index

*`text_index` and `text_search`: semantic retrieval over the agent's own
documents, on a local [turbovec][turbovec] index.*

`grep` finds the string you typed. This finds the passage you meant — and hands
back the `path:start-end` it came from, so the model can answer from the passage
or `read` the file around it.

```bash
ph-text-index call text_index --paths docs --format text
ph-text-index call text_search --query "how a worktree is cleaned up" --format text
```

Two tools, a command that runs them, and a [UTCP](https://utcp.io) manual that
describes them. The same two tools reach a model three ways: inside a
[phern](https://github.com/chastabor/pH) session, typed at a shell, or registered with any other
harness that speaks UTCP. This package needs phern for the first only.

## Installing

Python 3.12 or newer, on Linux or macOS. **What turns text into vectors is your
choice, and it decides what you install:**

| embedder | install | what it costs |
|---|---|---|
| a model on this machine (`text-index-local`, the default) | `ph-text-index[local]` | `sentence-transformers` and torch, about 4.2 GB with the default CUDA wheels, and the model's weights (90 MB for the default) |
| an embeddings server or API (`text-index-endpoint`) | `ph-text-index` | turbovec and numpy; nothing is loaded here |
| one of a phern session's model routes (`text-index-adapter`) | `ph-text-index`, inside phern | the same; only a session can use it |

**With phern**, `uv tool install 'phern[local]'` is the first row and
`uv tool install phern` the other two. The `rlm-indexed` profile mounts the
tools with the local model.

**By itself**, for a shell or for another harness:

```bash
uv tool install 'ph-text-index[local]'    # or: pip install 'ph-text-index[local]'
ph-text-index install                     # fetch and load the model now
ph-text-index status
```

With no GPU, torch's CPU wheels are a third of the size:

```bash
pip install 'ph-text-index[local]' --extra-index-url https://download.pytorch.org/whl/cpu
```

For an endpoint, install without the extra and name the server (see "From a
command line" for the file):

```bash
uv tool install ph-text-index
```

That is the `ph-text-index` command and nothing of phern's: `ph-core` (the
registry the tools are written against), `ph-utcp` and the two UTCP libraries it
names, turbovec and numpy.

**Without the extra and with no other embedder named, the command refuses** —
`status`, `tools`, `manual` and every `call` — with one sentence naming the
extra. The tools exist only once something can embed for them.
`sentence-transformers` is an extra because a gigabyte of torch for one
provider is a decision that belongs to whoever is installing, and a deployment
that embeds through an endpoint never wanted it.

**The command arrives in 0.11.** `ph-text-index` 0.10.0 is the plugin
that ran inside a phern session; it has no command and no manual.

**From a checkout**, `uv sync` puts the command in the workspace's environment
with the local model beside it (`uv run ph-text-index ...`), and
`uv tool install ./packages/ph-text-index` puts the command on `PATH`.

## Three ways in

### In a phern session

The `text-index` row and an embedder's row register this package's manual with
`ph-utcp`, and the two tools appear under their own names. A profile layers the
`utcp` bundle and then this one; `rlm-indexed` does:

```bash
phern --profile rlm-indexed -p "tell me how the workspace seam decides a containment tier"
```

On another profile, add the three rows:

```bash
phern --profile llama --provider llama --model <model> \
   --patch '{insert: [{id: utcp, name: utcp},
                      {id: text-index, name: text-index},
                      {id: text-index-local, name: text-index-local}]}' \
   -p "index docs/ then tell me how the workspace seam decides a containment tier"
```

The index is built for you: a host is started at the agent's first step and
indexes the row's `index` scope in the background (see "Where the tools run").
`/text-index status` and `/text-index install` ask the command about its model,
and `phern doctor` reports the embedder, its limit and where the index is.
Another embedder is another row in place of `text-index-local`: see "An endpoint
instead of a local model" and "One of the session's own model routes".

### From a command line

To try a query by hand, or to see why one came back empty. Run it **in the tree
you are asking about**: the workspace is the working directory, or `--root`.

```bash
ph-text-index call text_index --paths docs --format text
ph-text-index call text_search --query "how a worktree is cleaned up" --k 3 --format text
ph-text-index tools      # the tools and their flags
ph-text-index status     # is the model downloaded? what is indexed?
```

| command | what it does |
|---|---|
| `call TOOL [--field value ...]` | run one tool; each of its fields is a flag, or pass them all as `--args-json '{...}'` |
| `tools` | the tools and their flags |
| `status` | the embedder, whether its model is downloaded, its limit, and what the index holds |
| `install` | fetch and load the model now; an endpoint has nothing to fetch |
| `manual` | the UTCP manual; `manual --registration` is what a harness registers |
| `serve --socket PATH` | stay up and answer `call`s on a unix socket, with the model loaded once |

**A call prints one JSON object** and nothing else: `{"ok": true, "value": ...,
"text": ...}`, or `{"ok": false, "error": {"code": ..., "message": ...}}`.
`value` is the tool's result as data and `text` is the same result written for a
model to read; `--format text` prints only that. The exit code is 0, or 1 for a
failure, and the failure is in the object either way.

**Each call by itself loads the model first**: four to six seconds with MiniLM
on disk, measured here, against 0.3 for a search a host answers. For more than
a query or two, keep one up:

```bash
ph-text-index serve --socket /tmp/text-index.sock &
export PH_TOOL_HOST=/tmp/text-index.sock
ph-text-index call text_search --query "how a worktree is cleaned up" --format text
```

It leaves by itself after fifteen idle minutes (`--idle SECONDS`), and a call
that finds nobody on the socket runs by itself.

**Settings** are the session rows' own, by row id, read from three places with
the later winning: `$PH_HOME/tools/ph-text-index.yaml`, then `$PH_TOOL_CONFIG`
(JSON), then `--config FILE`. Naming an embedder's row is what chooses it:

```yaml
# ~/.ph/tools/ph-text-index.yaml
text-index:
  index:
    paths: [docs]
    include: ["**/*.md", "**/*.rst"]
text-index-endpoint:                 # leave this out for the local model
  base_url: http://127.0.0.1:8080/v1
  model: nomic-embed-text-v1.5
```

**Where things go** is said by five variables, none of which has to be set:

| variable | default | what the command keeps there |
|---|---|---|
| `PH_HOME` | `~/.ph` | `tools/ph-text-index.yaml`, the settings above |
| `PH_CACHE` | `$XDG_CACHE_HOME/ph`, so `~/.cache/ph` | the model's weights, the log, and the index when nothing else says where |
| `PH_TOOL_STATE` | unset | the directory for the index; `--state DIR` is the same |
| `PH_TOOL_CONFIG` | unset | row settings as one JSON object, for a caller that writes no file |
| `PH_TOOL_HOST` | unset | the socket of a `serve` to ask instead of running the call here |

`HF_ENDPOINT`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `OMP_NUM_THREADS` and
`CUDA_VISIBLE_DEVICES` reach the local model as you set them. See "On disk" for
each path.

### From another harness, over UTCP

Install the package where the harness runs its commands, choose an embedder as
above, then register what this prints:

```bash
ph-text-index manual --registration
```

```json
{"name": "ph_text_index",
 "call_template_type": "cli",
 "commands": [{"command": "/path/to/python -m ph_text_index manual"}],
 "inherit_env_vars": ["PATH", "HOME", "...", "PH_HOME", "PH_CACHE", "PH_TOOL_STATE", "PH_TOOL_HOST"]}
```

The harness runs that command once to read the manual, and from then on runs the
command each tool names. With the Python UTCP client, which this package
installs:

```python
import asyncio, json, subprocess

from utcp.utcp_client import UtcpClient


async def main() -> None:
    registration = json.loads(
        subprocess.run(
            ["ph-text-index", "manual", "--registration"], check=True, capture_output=True
        ).stdout
    )
    client = await UtcpClient.create(config={"manual_call_templates": [registration]})
    await client.call_tool("ph_text_index.text_index", {"paths": ["docs"]})
    found = await client.call_tool(
        "ph_text_index.text_search", {"query": "how a worktree is cleaned up", "k": 3}
    )
    print(found["text"] if found["ok"] else found["error"])


asyncio.run(main())
```

What a harness needs to know:

- **The tools are `ph_text_index.text_index` and `ph_text_index.text_search`.**
  A client prefixes each tool with the manual's name.
- **The registration names this install.** Its command is the absolute path of
  the interpreter the package is installed in, so it needs no `PATH`, and it is
  printed again after the package moves.
- **The workspace is the directory the command is started in.** Start the
  harness in the tree, or have it run its commands there. A `working_dir` on
  the registered template covers reading the manual and not the tool calls, so
  it does not move the workspace.
- **Give the model `text`.** Each answer is the envelope above, parsed: `text`
  is what a model in a phern session reads, and `value` is for code. A failure
  is `ok: false` with a `code`; the UTCP command-line protocol ignores exit
  codes, so the envelope is the only place it is said.
- **Optional arguments may be left out.** The Python client logs "Missing
  argument" for each one and the call runs.
- **Nothing watches the tree.** Have the model call `text_index` again after it
  edits; a document that has not changed is not embedded again.
- **One index a tree and embedder.** Without `PH_TOOL_STATE` the index is kept
  under `$PH_CACHE`, keyed by the workspace. A harness whose agents work in
  separate trees needs nothing more; one that wants an index per agent sets
  `PH_TOOL_STATE` for each.
- **Start a host if the model is local.** A call by itself loads the model
  every time. Start `ph-text-index serve --socket PATH --warm text_index` in
  the tree and set `PH_TOOL_HOST=PATH` on the calls: the model is loaded once,
  the tree is indexed in the background, `text_index` answers
  `state: indexing` after `wait_s` while the run goes on, and `text_search`
  answers meanwhile from what is indexed so far. Keep the path short; a unix
  socket's is limited to about a hundred characters.
- **In a sandbox, three places are written**: the index's directory, the
  weights under `$PH_CACHE/models` and `$PH_CACHE/logs`. Nothing in the tree
  is. A local model needs the network only to download, so run
  `ph-text-index install` outside the sandbox first; an endpoint needs its host
  reachable on every call.
- **An endpoint's key travels by name.** `api_key_env` names a variable, and the
  command reads it from its own environment. A UTCP client passes a command
  only the variables its template names, so the manual names that one in each
  tool's template: set the variable where the harness runs and the client hands
  it across. The manual holds the name and never the key. Settings a harness
  changes after it read the manual need the manual read again.
- **`text-index-adapter` is phern's.** It asks a running phern session to embed
  and refuses anywhere else; use the local model or an endpoint.
- **The model's instructions ship with the package**, as
  `ph_text_index/skills/text-search/SKILL.md`: how to scope a search, and that a
  score is not a confidence.

phern's own `ph-utcp` row is one such harness, and what it adds is described
next.

## Where the tools run

In the `ph-text-index` command's own process — started, in a pH session, by
`ctx.shell` for the agent that called the tool: in that agent's workspace, under
that agent's sandbox. So a child agent in its own worktree indexes that worktree,
and the harness process never loads the model, torch or turbovec.

**One process stays up for each tree**, and holds the model once it has loaded
it. Run for one call the command loads the model before it embeds the question —
six seconds with MiniLM cached, measured here; answered by the host a search
takes 0.3. `ph-utcp` keeps the host (`ph-text-index serve`) running for the
agent whose tree it is, starts it as it starts every command — in that agent's
workspace, under its sandbox — and ends it with the agent. It leaves by itself
after fifteen idle minutes, and the next call starts another.

**A child builds its own index only when it has a tree it writes for keeps.** A
child in its parent's checkout, or one granted only to read the project — every
child of a read-only fan-out, which is what `rlm-indexed` runs — asks its
parent's index through its parent's host: it embeds nothing and holds no model,
and its results say which tree their paths belong to. A child with a worktree or
an overlay of its own has files that diverge, so it gets its own index and host,
at its first call.

**Indexing does not hold a call up.** In a host `text_index` waits `wait_s` (five
seconds) for its run and then answers `state: indexing` with how far it has got,
while the run goes on in the background. `text_search` answers meanwhile from
what is indexed so far and says so: "Not finished indexing: 120 of 539
documents". The footer shows the count moving, and when the run ends the agent
is told, with a note that waits for its next step and starts no turn.

**It starts unasked.** When a root session's agent takes its first step the host
is started and indexes the row's `index` scope in the background, so the index
is being built before anything asks for it. `index.on_mount: false` leaves
indexing to `text_index` calls.

**A run that is stopped keeps what it embedded.** It saves every `save_every_s`
(thirty seconds) and on the way out, and each document carries a digest of its
text — so the next run, in the next host or after a restart, embeds only what the
last one had not saved. The digest is also what says "unchanged" in a tree with
no git and no jj, where every call used to embed every document again.

**`permissions-fs` does not screen these tools' reads.** Its rules govern calls
through a session's `ctx.fs`, and a command is not one. What bounds the command
is the sandbox, as it bounds `bash`.

**Read-only bars the repository, not the index.** These tools change no file of
the tree; they build data to probe it. So the index is kept in the calling
agent's scratch — `$PH_HOME/scratch/<session>/<agent>/text-index/` — which is
writable under every sandbox mode, while the tree is writable only under
`workspace-write`. An agent that may not write a tree can still index it, and
cannot write it by doing so. The model's weights and the command's log are
shared by every agent, under `$PH_CACHE`, and the rows grant those two paths.

## Two rows, and why

| row | what it is in a session | what the command mounts for it |
|---|---|---|
| `text-index` | registers the manual, the skill, `/text-index` and the doctor section | `text-index-engine`: the chunker, the turbovec store, and the two tools |
| `text-index-local` | names a `sentence-transformers` model as the embedder | `text-index-local-engine`: the model, loaded on first use |

The split is the shape `subprocess`/`subprocess-local` and
`code-runtime`/`code-runtime-python` already have in this tree, and it pays for
itself three times:

* a deployment with an embeddings **endpoint** — llama.cpp's `/v1/embeddings`, a
  provider's API — names another embedder row and keeps the tools;
* the **tests** run the real turbovec index against a deterministic stub
  embedder, so the suite proves chunking, paging and persistence without
  downloading a model;
* the row that pulls in **torch** is one a profile can leave out.

**The tools appear only once an embedder is claimed.** In a session the manual
is registered by the embedder row's claim, on that row's own scope, so a profile
with `text-index` and no embedder advertises nothing at all — the rule
`subagent-task` states, for the same reason: a tool named in every prompt and
refused on every call spends context teaching the model a capability the
deployment does not have. The command keeps the same rule one process down.

**The session's profile still chooses the model.** `text-index-local`'s config —
`model`, the two prefixes, `batch_size`, `trust_remote_code`, `cache` — is
handed to the command with `text-index`'s own, and being named there is how the
command knows which embedder row to mount. A person at a shell writes the same
two rows' configs in `$PH_HOME/tools/ph-text-index.yaml`.

## The two tools

**`text_index(paths?, glob?, exclude?, forget?)`** reads each document through `ctx.fs`,
cuts it into passages, embeds them and stores them. Re-indexing a file replaces
its passages, so running it again after an edit is correct. `forget: true`
removes instead.

What it looks at when a call does not say is the row's `index` block:

```yaml
- id: text-index
  name: text-index
  config:
    index:
      paths: [docs, notes]             # directories or files; default ["."]
      include: ["**/*.md", "**/*.rst"] # default ["**/*.md"]
      exclude: ["**/drafts/**"]
      on_mount: true                   # index this when a root session starts; default true
    wait_s: 5                          # how long text_index waits before answering "indexing"
    save_every_s: 30                   # how often a run saves what it has embedded
```

`exclude` is read against the workspace root and a directory it covers is never
entered. A call's `glob` replaces `include` for that call; a call's `exclude` is
*added* to the row's. A file named outright in `paths` is taken whatever is
excluded.

**Nothing watches the filesystem** — no daemon, no watcher. The index changes
when `text_index` runs and at no other moment, so run it again after edits.

That is cheap now because **git or jj is asked first** (`ph.seams.changes`): a
document the version control vouches for is not read, not re-cut, and — the part
that matters — **not re-embedded**. This loop has no content digest of its own,
so before the filter every call re-embedded the whole corpus: 20 s of MiniLM for
this repo's `docs/`, or 185 s under nomic. Which backend answers is the
workspace provider's to state; a tree with no version control behaves exactly as
it did before, which is a test rather than a hope. A document past `max_bytes` is *skipped and
reported*, not an error — a call over a directory should not fail because it
found a minified bundle, and a caller who never learns what was skipped cannot
tell a quiet corpus from a quiet failure.

**`text_search(query, k?, paths?)`** returns the passages with their line
ranges and scores. `paths` restricts the search to a subtree.

An earlier version of this paragraph claimed a filtered search costs *less* than
an unfiltered one, because turbovec filters inside the SIMD kernel. Measured,
that is wrong: the kernel does, but building the id list on the Python side
dominates it by roughly fifteen times (0.11 ms against a 0.22 ms unfiltered
search on a 10 000-chunk index, and selectivity changes nothing). Use `paths`
for **precision**, not for speed.

## Everything is read through `ctx.fs`

Every indexed byte arrives via the command's `ctx.fs.read`: the walk's ignore
rules, the size bound and the skip reasons are that seam's, said once. In a
session that `ctx.fs` is the command's own and not the session's, so what bounds
an indexing run's reads there is the calling agent's sandbox — see "Where the
tools run".

The index is **per agent** in a session: each agent's own, in its scratch. See
"On disk" for where it lives and why it is keyed that way.

## Chunking

Paragraphs, not a fixed character stride. A stride is simpler and reliably
splits the one sentence that answers the query across two chunks, so neither
retrieves; blank lines are where the document's own author already said "new
idea". Every chunk carries its 1-based line span, because a chunk that does not
know its place in the file is a wall of prose the agent must then go and locate.
`overlap_chars` carries the previous chunk's tail forward, which is the
concession to a paragraph whose meaning depends on the one before it.

## On disk

`index.tvim` is turbovec's own format, written with `sync()` — incremental, one
fsync per call, crash-safe at any byte. `chunks.json` is the sidecar: path, line
span and the passage text.

**The sidecar holds the text on purpose.** Storing only pointers would keep the
file small and make every hit a second tool call before the model learns whether
the hit was any good — answered from a file that may have changed since it was
indexed. It is also the ceiling: the sidecar is rewritten whole on every save,
which is fine for a corpus of documents and wrong for millions.

A crash between the two writes can leave them diverged, so vectors are committed
first and `open` reconciles toward the sidecar both ways. It drops any chunk the
index cannot answer for, since a chunk nobody can retrieve is invisible. It
removes any vector the sidecar does not name, since a vector with no text would
surface as a hit this row could not describe, and its id is one the next add
would allocate again.

**One index per tree and embedder**, because what an index holds belongs to one
tree: documents named relative to it, one version-control token, and a sweep
that drops what that tree no longer has.

- **In a session** it is in the scratch of the agent whose tree it is,
  `$PH_HOME/scratch/<session>/<agent>/text-index/<embedder>`: the calling agent,
  or the ancestor whose tree it shares. A child in a worktree of its own indexes
  its own tree. A resumed session has the
  same session and agent ids, so the same index, and `text_index` then embeds
  only what version control says changed. The cost is that a new session embeds
  the corpus again rather than finding the last one's.
- **From a shell**, `--state DIR` puts it in `DIR/<embedder>`, and so does
  `$PH_TOOL_STATE` — the variable `ph-utcp` sets per call, and another harness
  can set. With neither it is `$PH_CACHE/text-index/<workspace>/<embedder>`.
  That workspace directory records which workspace it is, and one whose
  workspace has been gone a week is removed on the next change
  (`ph.workspace_caches`).

**One writer at a time.** Processes that share a directory share its lock. A change is made under `writer.lock` in it, starting from what is on
disk: a copy another writer has changed since it was loaded is loaded again. A
second writer waits for the first. A search loads a changed copy again only when
the lock is free, and otherwise answers from the copy it has.

### Calibration

turbovec's TQ+ calibration is worth 2.5 to 8.7 points of R@10, and upstream is
emphatic about the one way to get it wrong: the sample must be a uniform random,
representative draw of what the index will hold, and a clustered prefix "fits a
calibration that actively destroys recall". An incremental indexer has no such
draw at the moment it would have to commit one — the first document is the most
clustered prefix there is.

So it is committed in exactly the situation upstream describes: the index is
**empty** and the incoming batch is large enough (1 024 rows) to sample from,
in which case a random sample of that batch *is* a representative draw and
calibrating before the add is the documented order. Otherwise the index stays
uncalibrated, which is plain TurboQuant — good rather than wrong. `calibrate:
false` turns even that off.

## Using it from the RLM

Nothing to do: under Code Mode every registered tool is in the generated SDK
listing, so these arrive as `await tools.text_index(...)` and
`await tools.text_search(...)` with no work from this package. The
[`rlm-indexed`](../ph-app/src/ph_app/profiles.py) profile is `rlm-stable` plus
this bundle and `ph-code-graph`'s:

```bash
phern --profile rlm-indexed --provider llama --model <model> --mode tui
```

This package registers a `ph.bundles` entry point so that profile is *hidden*
on an install without this distribution rather than offered and then failing at
mount — `available_profiles()` gates on bundle resolution, and the refusal names
the package to install.

The bundle carries **both** rows, and that matters: the seam registers no tools
until an embedder is claimed, so a bundle shipping only `text-index` would mount
a service and advertise nothing, with no error anywhere. A deployment bringing
its own embedder disables `text-index-local` and mounts its own provider row.

## Switching models

The default is **`sentence-transformers/all-MiniLM-L6-v2`** — 384-dimensional,
about 90 MB, symmetric so it needs no prefixes, and fast enough on a CPU that
indexing a documentation tree is a coffee break. It is the *small* choice, not
the best one.

The index directory is keyed by a digest of the embedder's identity — model
name **and** both prefixes, because all three move the vector space — so
switching is safe to try: a new model gets its own index and switching back
finds the old one intact. An index opened by an embedder other than the one
that built it raises `IndexMismatch` rather than returning neighbors computed in
a space nothing shares.

### A worked upgrade: `nomic-embed-text-v1.5` — measured

Run against this repository's `docs/seams` (30 documents, 235 passages), same
queries, same chunking:

| | MiniLM-L6-v2 | nomic-embed-text-v1.5 |
|---|---|---|
| dimensions | 384 | 768 |
| load, cold | 19 s | 26 s |
| index 235 passages | **5.4 s** | 41.7 s |
| *"stop an agent writing in my own checkout"* | workspace.md (0.319) | workspace.md (**0.599**) |
| *"what confines a shell command on linux"* | shell.md (0.529) | shell.md (**0.753**) |

On a harder corpus — all 57 files of `docs/`, 539 passages, eight paraphrase
queries scored on whether the top hit was the right *file* — nomic got **3/8**
against MiniLM's **1/8**, at 185 s of indexing against 20 s.

Read that honestly in both directions. nomic retrieves better and separates hits
far more confidently; it also costs roughly **8× the indexing time**, and 3/8 is
not good retrieval in absolute terms. Some of those misses are the scoring being
crude (a filename substring), but not all of them: a corpus of long design notes
is genuinely hard, and *narrowing the corpus* helped more than changing the model
— MiniLM found `workspace.md` over `docs/seams` and missed it over all of
`docs/`. Use `paths=` before reaching for a bigger model.

**MiniLM stays the default** for that reason: the cheap thing is good enough for
a scoped corpus, and the expensive thing does not rescue an unscoped one.



768-dimensional, 8192-token context, materially better retrieval than MiniLM.
Three things it needs, and each is a field rather than a special case:

```bash
pip install 'ph-text-index[nomic]'   # einops, which its remote code imports
```

```yaml
- id: text-index-local
  config:
    model: nomic-ai/nomic-embed-text-v1.5
    # Asymmetric: it wants to know whether it is embedding a question or a
    # passage. Both prefixes are part of the index identity, so this gets its
    # own directory automatically.
    queryPrefix: "search_query: "
    documentPrefix: "search_document: "
    # Its config.json carries an `auto_map` pointing at nomic-ai/nomic-bert-2048,
    # so sentence-transformers will not load it without this. Read the next
    # paragraph before setting it.
    trustRemoteCode: true
```

**`trustRemoteCode` is a trust decision, not a compatibility flag.** It
downloads Python from the model's repository — here from a *second* repository
via `auto_map` — and executes it in the harness process. It is off by default
and nothing infers it from a load failure, because "retry with arbitrary code
execution enabled" is not a fallback. It is deliberately *not* part of the index
identity: it governs what may load, not where a vector lands, so granting or
revoking it does not invalidate an index.

A model whose weights are safetensors and whose architecture `transformers`
already knows needs none of this — `BAAI/bge-base-en-v1.5` and the `e5` family
are drop-in with prefixes alone.

**And the `[nomic]` extra is not optional politeness.** Without `einops` the
model downloads its weights *and* its remote code successfully and then dies at
import:

```
ImportError: This modeling file requires the following packages that were not
found in your environment: einops
```

A missing transitive dependency that only the model's own code knows about, and
that no amount of pre-downloading would have caught. It is the reason
provisioning **loads** the model rather than fetching it — see below.

## A passage fits its embedder

A model reads so many tokens and no more. Past that a local model drops the rest
without a word — the tail of the passage is in the index and can never be found
— and a server refuses the request. Measured on this repository's `docs/seams`
with the default model, which reads 256 tokens: 144 of 233 passages cut at 1,200
characters were longer, and the last eleventh of the text was never embedded.

So passages are packed by what the embedder counts as well as by `max_chars`,
and each is as long as both allow:

| embedder | where its limit comes from | how text is counted |
|---|---|---|
| `text-index-local` | the model's own length; `max_tokens` can lower it | the model's tokenizer, exactly |
| `text-index-endpoint` | `max_tokens`, else a llama.cpp's `/props` | its bytes, three to a token |
| `text-index-adapter` | `max_tokens`, else the route's window | the same |

- **Packed in the embedder's unit.** `max_chars` says how big a passage should
  be; the limit says how much the embedder reads, in its tokens, and a
  character is a quarter of a token in English prose, a third in code, and one
  or more in a script that is three bytes a character. So the packer keeps two
  bounds: `max_chars`, and the limit less the passage prefix and the model's
  start and end markers. Each paragraph and line is weighed for itself, so a
  table in a page of prose is cut small and the prose around it is not, and
  text in any script fits at the first cut. On this repository's `docs/seams`
  with the default model: 316 passages, none over 256 tokens, where sizing in
  characters gave 388 and packed three documents twice.
- **A larger limit changes nothing.** Smaller passages retrieve better, so
  `max_chars` stays the ceiling: under an 8,192-token window the same documents
  are 233 passages, as with no limit at all.
- **Checked as it is sent.** A packing adds up the weights of a passage's
  parts and an embedder counts the whole, so each document's passages are
  measured before they go. One over the limit — a tokenizer whose count of a
  whole is more than its parts' — has its document packed again with less room.
  Re-packed rather than split in place, so every passage still starts and ends
  on the lines it says.
- **A refusal is not the end of the run.** A passage the embedder turns away as
  too long is packed with less room and sent again. Where the refusal says in
  numbers what it takes and how long the input was — a llama.cpp and OpenAI
  both do — the first is the limit for the rest of the run and the second is
  the corpus's own density, so one refusal settles it. One that names no number
  halves the document's passages and teaches nothing about the next. A document
  that will not fit even at 16 tokens is skipped, with the embedder's words, and
  listed.
- **It says so.** When the limit ended a passage sooner than `max_chars` would
  have, `text_index` says what passages were held to and by what ("Passages are
  held to 254 tokens of text, to fit the embedder's limit of 256 tokens (the
  model's)"), and it names each document that had to be packed again. A run
  that finished in the background says the same in the note that tells the
  agent it ended.
- **A question is held to it too.** One longer than the embedder reads is
  searched by its head — as much of its start as fits, to the end of a word —
  and the answer says how many of its characters were searched. A local model
  did that without a word, and a server refused the search.
- **`phern doctor` and `/text-index status` say the limit.** `phern doctor`
  shows a setting, or the route's window for `text-index-adapter`; a model's own
  length or a server's is learned by asking, which of a local model means
  loading it, so `/text-index status` shows what the last run to embed found.

What a server reports is not always what it takes. A llama.cpp's `/props` gives
its window per slot — `-c 8192` with four slots reports 2,048 — and it refuses
an embedding longer than its *batch* (`-b`, 512 by default), which `/props` does
not mention. The index learns the real number from the first refusal; to use a
model's whole length, start the server with `-b` and `-ub` as large as the
window. A hosted API has nothing to ask, so write `max_tokens` there (8191 for
OpenAI's embedding models): the route's window is its chat model's.

Changing `max_tokens` cuts every document again. Passages already in an index
stay as they were cut until their document changes.

## An endpoint instead of a local model

`text-index-endpoint` is an embedder that is somebody else's process: the
OpenAI-compatible `/v1/embeddings` request, so one row covers a local server —
llama.cpp, Ollama, vLLM — and a hosted API. Mount it in place of
`text-index-local`:

```yaml
- id: text-index-local
  disabled: true
- id: text-index-endpoint
  name: text-index-endpoint
  config:
    base_url: http://127.0.0.1:8080/v1
    model: nomic-embed-text-v1.5
    api_key_env: ""                    # the NAME of a variable; empty for a local server
    max_tokens: 2048                   # optional; a llama.cpp is asked, a hosted API is not
    query_prefix: "search_query: "
    document_prefix: "search_document: "
```

- **No torch.** Nothing is loaded on this machine, so a deployment that embeds
  this way installs `ph-text-index` without the `[local]` extra.
- **A remote endpoint is sent the text of every document indexed and of every
  question asked.** Writing this row is how a deployment says it means that.
- **No key is stored.** `api_key_env` names a variable. A command's environment
  drops every name that looks like a credential, so the row has `ph-utcp` hand
  that one variable across from the environment pH was started in, each time it
  starts the command. Another harness is told the same name by the manual,
  which lists it among the variables each tool inherits. A command that finds
  it empty fails with `MISSING_CREDENTIAL` — an adapter's code for the same
  thing — names the variable, and sends nothing.
- **A confined agent needs the host allowed.** A sandboxed command has the
  network its deployment allows: name the endpoint's host and port in
  `sandbox-allow` (`hosts: ["127.0.0.1:8080"]`). One that is not allowed, or is
  down, fails with `TEXT_INDEX_ENDPOINT_UNREACHABLE`; a server that answers with
  an error fails with `TEXT_INDEX_ENDPOINT_FAILED` and the server's own words, or
  `CONTEXT_WINDOW_EXCEEDED` when the words say an input was too long.
- **The index is keyed by the model, the prefixes and an asked-for
  `dimensions`** — not by the address, so moving a server keeps the index. A
  hosted model that changes under an unchanged name cannot be told apart beyond
  its width, which is checked on every add.
- **One embedder a session.** There is no fallback to a second one: an index is
  one vector space, and a search quietly answered from another would be a
  different index answering. An endpoint that cannot be reached is the answer.

From a shell, `$PH_HOME/tools/ph-text-index.yaml` names the same row:

```yaml
text-index-endpoint:
  base_url: http://127.0.0.1:8080/v1
  model: nomic-embed-text-v1.5
```

## One of the session's own model routes

`text-index-adapter` embeds through a provider the session already has a route
to, named the way a session names a model:

```yaml
- id: text-index-local
  disabled: true
- id: text-index-adapter
  name: text-index-adapter
  config:
    provider: openai                   # a provider one of this profile's adapter rows registers
    model: text-embedding-3-small
    dimensions: 512                    # optional, where the model can shorten its vectors
    max_tokens: 8191                   # what the embedding model reads; the route's window is its chat model's
```

- **The key stays in the harness.** The tools run in a command inside the
  calling agent's sandbox, and everything that command holds is in reach of the
  agent's own tools. So the route's credential is not handed to it: the command
  sends its passages to the session over a unix socket, the session's adapter
  embeds them — resolving the credential at its own edge, through
  `ctx.credentials`, as it does for a model call — and vectors come back.
  Nothing an agent can read, print or put in a model's context ever held the key.
- **A confined agent needs no network for it.** The session asks the provider;
  the command asks only the session, on a socket under `$PH_RUNTIME/utcp` that
  the rows grant. No host goes in `sandbox-allow`.
- **A route whose adapter cannot embed is refused at mount**, by name. phern's
  OpenAI-compatible and Google adapters embed (`ph.llm.adapter.EmbeddingAdapter`,
  `ctx.llm.embed`); the Anthropic API has no embeddings call.
- **A failure is the adapter's own.** A key that is not set fails the tool with
  `MISSING_CREDENTIAL` and the variable's name, as a model call would; a server that will not embed with its own words. `phern doctor` shows
  the last one, and how much the session has embedded.
- **What it spends is in the log.** An embedding is not a turn's request, so
  each one is its own record, `llm/embedded`, in the log of the agent whose
  index it is: the route, how many texts, and the tokens the provider counted. A
  request that failed is `llm/embedding-failed`, with its code and reason. A
  sub-agent's account includes them, and a goal's budget charges them when its
  `tokenSources` names `embedding`; by default it charges the agent's own
  replies and nothing else. A child that shares its parent's index spends on
  its parent's account, since the index and the host that embeds are the
  parent's.
- **The provider is sent the text of every document indexed and every question
  asked**, as it is sent the conversation.
- **Only from a session.** `ph-text-index` typed at a shell has no session to
  ask and says so; from a shell use `text-index-local` or `text-index-endpoint`.

`text-index-endpoint` reaches the same servers without a route, with the key in
the command. Use the adapter when the key matters; use the endpoint when the
embeddings server is local and wants none.

## Where the weights live, and installing on purpose

`$PH_CACHE/models`, set by the row (`cache:` overrides it). Left to
`sentence-transformers` they would go to `$HF_HOME` or `~/.cache/huggingface` —
outside all three of pH's roots, so a gigabyte of weights would sit somewhere
`phern doctor` never mentions and `rm -rf $PH_CACHE` would not reclaim. Rebuildable
and large is the lifecycle `$PH_CACHE` names (Q1), and the runtime venv is there
for the same reason.

**Provision before the agent needs it**, three ways, in increasing order of
"nobody is watching":

```
/text-index status     # is the model ready? costs nothing, downloads nothing
/text-index install    # fetch and load it now, and say what happened
```

A **command**, not a tool: a person asks the harness to do this, and routing it
through a model turn would put the model in the log as having decided it. It
costs no turn, and it runs outside the sandbox, because `install` needs the
network. `ph-text-index status` and `ph-text-index install` are the same two from
a shell. `status` says one of three things about the model: `loaded`,
`downloaded, loads on first use`, or `not downloaded`.

**A tool call does not need the network once the weights are on disk.** The
command loads a downloaded model without asking the hub, so a confined agent —
whose network is the deployment's allowlist — can search. A model that is not
downloaded is fetched by the first call that needs it where the network allows;
where it does not, the call fails with `TEXT_INDEX_MODEL_MISSING` and names
`/text-index install`.

For an **unattended** run — a daemon, a scheduled tick, `phern -p` in CI — there is
nobody to type either:

```yaml
- id: text-index-local
  config:
    preload: true
```

The session's row then asks the command to load the model once, at **mount**,
and refuses the mount with a `MountRefusal` sentence if it cannot, which is the cookbook's own rule: refuse at mount, not at
first use, because by then the agent is running and "refuse to start" has already
been disobeyed. Off by default, because a person at a TUI would rather the
harness start now and pay for the model when they use it.

All three call the same thing, and it **loads** the model rather than fetching
it — the `einops` failure above is exactly why. Whatever upstream says on
failure is passed through verbatim, because that sentence names the fix.

## The skill

The package installs a `text-search` skill, so an RLM under progressive
disclosure sees one line in its catalog and can read the page when it needs
it — how to scope a search, that a score is not a confidence, that a passage can
be an argument the document goes on to reject.

Registered by the row rather than found by a directory scan, so it arrives
exactly when the tools do and leaves with them: `skills-progressive` ships an
empty `paths` on purpose, because scanning a well-known directory would make
"install a skill" mean "drop a file somewhere", and a skill is something a
distribution installs deliberately (I7).

[turbovec]: https://pypi.org/project/turbovec/
