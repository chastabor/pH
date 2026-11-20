# ph-text-index

*`text_index` and `text_search`: semantic retrieval over the agent's own
documents, on a local [turbovec][turbovec] index.*

`grep` finds the string that you typed. This package finds the passage that you
meant. It also returns the `path:start-end` that the passage came from, so the
model can answer from the passage or `read` the file around it.

```bash
ph-text-index call text_index --paths docs --format text
ph-text-index call text_search --query "how a worktree is cleaned up" --format text
```

This package has two tools, a command that runs them, and a
[UTCP](https://utcp.io) manual that describes them. The same two tools reach a
model in three ways: inside a [phern](https://github.com/chastabor/pH) session,
typed at a shell, or registered with any other harness that supports UTCP. Only
the first way needs phern.

## Installing

The package needs Python 3.12 or newer, on Linux or macOS. **You choose what
turns text into vectors, and that choice decides what you install:**

| Embedder | Install | What it costs |
|---|---|---|
| A model on this machine (`text-index-local`, the default) | `ph-text-index[local]` | `sentence-transformers` and torch, about 4.2 GB with the default CUDA wheels, and the weights of the model (90 MB for the default) |
| An embeddings server or API (`text-index-endpoint`) | `ph-text-index` | turbovec and numpy. Nothing is loaded here |

**With phern**, `uv tool install 'phern[local]'` is the first row and
`uv tool install phern` is the second. The `rlm-indexed` profile mounts the
tools with the local model.

**Without phern**, for a shell or for another harness:

```bash
uv tool install 'ph-text-index[local]'    # or: pip install 'ph-text-index[local]'
ph-text-index install                     # fetch and load the model now
ph-text-index status
```

With no GPU, the CPU wheels of torch are a third of the size:

```bash
pip install 'ph-text-index[local]' --extra-index-url https://download.pytorch.org/whl/cpu
```

For an endpoint, install without the extra and name the server (see "From a
command line" for the file):

```bash
uv tool install ph-text-index
```

That installs the `ph-text-index` command and nothing from phern. The
dependencies are `ph-core` (the registry that the tools are written against),
`ph-utcp` and the two UTCP libraries that it names, turbovec, and numpy.

**Without the extra, and with no other embedder named, the command refuses.**
`status`, `tools`, `manual`, and every `call` refuse with one sentence that
names the extra. The tools exist only when something can embed for them.
`sentence-transformers` is an extra because a gigabyte of torch for one provider
is a decision, and that decision belongs to whoever installs the package. A
deployment that embeds through an endpoint never wanted torch.

**The command arrives in 0.11.** `ph-text-index` 0.10.0 is the plugin that ran
inside a phern session. It has no command and no manual.

**From a checkout**, `uv sync` puts the command in the environment of the
workspace, with the local model beside it (`uv run ph-text-index ...`), and
`uv tool install ./packages/ph-text-index` puts the command on `PATH`.

## Three ways in

### In a phern session

The `text-index` row and the row of an embedder register the manual of this
package with `ph-utcp`, and the two tools appear under their own names. A
profile layers the `utcp` bundle and then this bundle. `rlm-indexed` does that:

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

The index is built for you. Before the first step of the agent, an indexing run
starts in the background over the `index` scope of the row (see "Where the tools
run"). `/text-index status` and `/text-index install` ask the command about its
model, and `phern doctor` reports the embedder, its limit, and where the index
is. To use another embedder, name its row in place of `text-index-local`. See
"An endpoint instead of a local model".

### From a command line

Use the command line to try a query by hand, or to see why a query returned
nothing. Run the command **in the tree that you ask about**. The workspace is
the working directory, or the value of `--root`.

```bash
ph-text-index call text_index --paths docs --format text
ph-text-index call text_search --query "how a worktree is cleaned up" --k 3 --format text
ph-text-index tools      # the tools and their flags
ph-text-index status     # is the model downloaded? what is indexed?
```

| Command | What it does |
|---|---|
| `call TOOL [--field value ...]` | Runs one tool. Each field of the tool is a flag, or pass all fields as `--args-json '{...}'` |
| `tools` | Lists the tools and their flags |
| `status` | Shows the embedder, whether its model is downloaded, its limit, and what the index holds |
| `install` | Fetches and loads the model now. An endpoint has nothing to fetch |
| `manual` | Prints the UTCP manual. `manual --registration` prints what a harness registers |

**A call prints one JSON object** and nothing else: `{"ok": true, "value": ...,
"text": ...}`, or `{"ok": false, "error": {"code": ..., "message": ...}}`.
`value` is the result of the tool as data. `text` is the same result, written
for a model to read. `--format text` prints only `text`. The exit code is 0, or
1 for a failure. A failure is also in the object in both cases.

**Every call is one command.** No process stays running between calls, and no
process listens on a socket. Two flags of `call` are for a run that takes a long
time:

- **`--progress`** reports the progress of the run on stderr as JSON Lines, one
  JSON object on each line. While the run works, it writes
  `{"state":"working","done":120,"total":539,"what":"documents"}` at most every
  0.5 s. It then writes exactly one last line:
  - `"state":"done"`, with `changed` and a `note`.
  - `"state":"busy"`, when another run held the index and this run did nothing.
  - `"state":"failed"`, with a `code` and a `failure`.

  If the stream ends with no last line, the command was killed or it died.
  stdout does not change: it carries one object.

  **Every line reports what the run has spent so far**, and the last line
  reports the total: `"spent":{"requests":40,"texts":242,"tokens":66774}`.
  - `requests` and `texts` count what the embedder was sent.
  - `tokens` is what the `usage` of the server reported. It is absent where the
    server reports none. A local model costs nothing and counts no tokens.

  A run that failed reports what it spent before it failed. The result of
  `text_index` reports the spend too: as `spent`, and as a line ("Spent
  embedding: 40 requests, 242 texts, 66,774 tokens.").
- **`--wait SECONDS`** is how long the call waits for another run that holds the
  index. It overrides the `wait_s` of the row. `--wait 0` answers immediately.

**With the local model, every call loads the model first.** That takes about
3.9 s for a search with MiniLM on disk, and 6 s under `bwrap`. No process holds
a model between calls. For more than one or two queries, serve the model from a
local llama.cpp and name it with `text-index-endpoint`. The command then loads
nothing, and that is the fast local path (see "An endpoint instead of a local
model").

**Settings** are the settings of the session rows, under each row id. They are
read from three places, and a later place overrides an earlier one:
`$PH_HOME/tools/ph-text-index.yaml`, then `$PH_TOOL_CONFIG` (JSON), then
`--config FILE`. To choose an embedder, name its row:

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

**Four variables control where things go.** None of them has to be set:

| Variable | Default | What the command keeps there |
|---|---|---|
| `PH_HOME` | `~/.ph` | `tools/ph-text-index.yaml`, the settings above |
| `PH_CACHE` | `$XDG_CACHE_HOME/ph`, which is usually `~/.cache/ph` | The weights of the model, the log, and the index when nothing else says where |
| `PH_TOOL_STATE` | unset | The directory for the index. `--state DIR` does the same |
| `PH_TOOL_CONFIG` | unset | The row settings as one JSON object, for a caller that writes no file |

`HF_ENDPOINT`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `OMP_NUM_THREADS`, and
`CUDA_VISIBLE_DEVICES` reach the local model as you set them. See "On disk" for
each path.

### From another harness, over UTCP

Install the package where the harness runs its commands, and choose an embedder
as described above. Then register what this command prints:

```bash
ph-text-index manual --registration
```

```json
{"name": "ph_text_index",
 "call_template_type": "cli",
 "commands": [{"command": "/path/to/python -P -m ph_text_index manual"}],
 "inherit_env_vars": ["PATH", "HOME", "...", "PH_HOME", "PH_CACHE", "PH_TOOL_STATE"]}
```

The harness runs that command one time to read the manual. After that, it runs
the command that each tool names. This example uses the Python UTCP client,
which this package installs:

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
  A client puts the name of the manual before each tool name.
- **The registration names this install.** Its command is the absolute path of
  the interpreter in which the package is installed, so it needs no `PATH`.
  Print the registration again after the package moves. `-P` keeps the working
  directory off the module path, so the command imports no file from the tree.
- **The workspace is the directory in which the command starts.** Start the
  harness in the tree, or make it run its commands there. A `working_dir` on the
  registered template applies to reading the manual and not to the tool calls,
  so it does not move the workspace.
- **Give the model `text`.** Each answer is the envelope above, parsed. `text` is
  what a model in a phern session reads, and `value` is for code. A failure is
  `ok: false` with a `code`. The UTCP command-line protocol ignores exit codes,
  so the envelope is the only place that reports a failure.
- **Optional arguments can be left out.** The Python client logs "Missing
  argument" for each one, and the call still runs.
- **Nothing watches the tree.** Make the model call `text_index` again after it
  edits. A document that has not changed is not embedded again.
- **One index for each tree and embedder.** Without `PH_TOOL_STATE`, the index
  is kept under `$PH_CACHE`, keyed by the workspace. A harness whose agents work
  in separate trees needs nothing more. A harness that wants an index for each
  agent sets `PH_TOOL_STATE` for each agent.
- **Every call loads a local model.** Each call is one command, and no process
  holds the model between calls. A search takes about 3.9 s with MiniLM. A
  harness that searches often names `text-index-endpoint` and a local llama.cpp,
  and the command then loads nothing.
- **One indexing run at a time.** A `text_index` call is one command, and it
  finishes before it answers. A call that is made while another run holds the
  index waits `wait_s` for that run. It then answers `state: indexing`: another
  run is building the index, and this call did nothing. `text_search` never
  waits. It answers from what the run has saved so far, with `indexing: true` in
  its value.
- **Progress is for a harness that starts the command itself.** A standard UTCP
  client merges stderr into stdout, so no template of the manual carries
  `--progress`. A harness that wants indexing to run in the background runs
  `ph-text-index call text_index --progress` in the tree and reads stdout and
  stderr separately. Its model can call `text_search` during the run.
- **In a sandbox, the command writes three places**: the directory of the index,
  the weights under `$PH_CACHE/models`, and `$PH_CACHE/logs`. It writes nothing
  in the tree. A local model needs the network only for the download, so run
  `ph-text-index install` outside the sandbox first. An endpoint needs its host
  to be reachable on every call.
- **The key of an endpoint travels by name.** `api_key_env` names a variable,
  and the command reads that variable from its own environment. A UTCP client
  passes a command only the variables that its template names. The manual
  therefore names that variable in the template of each tool. Set the variable
  where the harness runs, and the client passes it to the command. The manual
  holds the name and never the key. If a harness changes settings after it read
  the manual, it must read the manual again.
- **The instructions for the model ship with the package**, as
  `ph_text_index/skills/text-search/SKILL.md`: how to scope a search, and that a
  score is not a confidence.

The `ph-utcp` row of phern is one such harness. The next section describes what
it adds.

## Where the tools run

The tools run in the process of the `ph-text-index` command. In a pH session,
`ctx.shell` starts that command for the agent that called the tool, in the
workspace of that agent and under its sandbox. A child agent in its own worktree
therefore indexes that worktree, and the harness process never loads the model,
torch, or turbovec.

**Every call is one command, and no process stays running.** With
`text-index-local`, every call therefore loads the model before it embeds. That
takes about 3.9 s for a search by itself with MiniLM on disk, and 6 s under
`bwrap`. A deployment that searches often serves the model from a local
llama.cpp and mounts `text-index-endpoint`, which loads nothing in the command.

**A child agent builds its own index only when it has a tree that it writes to
permanently.** A child that works in the checkout of its parent, or that can
only read the project, asks the index of its parent. That includes every child
of a read-only fan-out, which is what `rlm-indexed` runs. Such a child embeds no
document, and its results say which tree their paths belong to. A child with its
own worktree or overlay has files that diverge, so it gets its own index at its
first call.

**Indexing does not delay a call.** A `text_index` call starts a run in the
background, or joins the same request if it is already in progress. The run is a
job of the agent that owns the tree, under the sandbox of that agent. The call
waits `wait_s` (five seconds) for the run:

- If the run ends in time, its result is the answer to the call.
- If not, the call answers `state: indexing`, with a line that gives the
  progress of the run ("[The text index is still working: 120/539 documents.]").
  The run continues.

The footer shows the count as it changes. When the run ends, the agent is told
with a note. The note waits for the next step of the agent and starts no turn.

**One run at a time for each index.** A run holds the writer lock of the index
while it works. `state: indexing` means that another run holds the index and
this call did nothing. The totals beside it are what the index holds so far. A
different request that is made during a run is not queued, so call `text_index`
again after the run ends.

**A search never waits.** `text_search` opens the index as a reader. It takes no
lock, and it writes neither the vectors nor the sidecar. During a run, it
answers from what the run has saved so far, and a run saves every
`save_every_s`. The search reports this: the `indexing` field of its value is
`true`, and its text says "Not finished indexing".

**Indexing starts without a request.** Before the agent of a root session takes
its first step, one `text_index` run starts over the `index` scope of the row.
The index is therefore being built before anything asks for it. The agent is
told when the run ends only if the run changed the index.
`index.on_mount: false` leaves indexing to `text_index` calls.

**A run that is stopped keeps what it embedded.** A run saves every
`save_every_s` (thirty seconds) and when it exits, and each document carries a
digest of its text. The next run, in the same session or after a restart,
therefore embeds only what the last run had not saved. The digest also
identifies an unchanged document in a tree with no git and no jj, where every
call previously embedded every document again. When a run fails or is killed,
the agent is told that the run stopped. The note gives the sentence of the tool.
When the command ended with no last line, the note gives its exit code and its
last output instead.

**`permissions-fs` does not screen the reads of these tools.** Its rules govern
calls through the `ctx.fs` of a session, and a command is not such a call. The
sandbox limits the command, as it limits `bash`.

**A read-only session protects the repository, not the index.** These tools
change no file in the tree. They build data that is used to query the tree. The
index is therefore kept in the scratch directory of the calling agent,
`$PH_HOME/scratch/<session>/<agent>/text-index/`. That directory is writable in
every sandbox mode, and the tree is writable only in `workspace-write`. An agent
that may not write a tree can still index it, and cannot write the tree by doing
so. Every agent shares the weights of the model and the log of the command,
under `$PH_CACHE`, and the rows grant those two paths.

## A seam row and an embedder row, and why

A profile mounts `text-index` and one embedder row:

| Row | What it is in a session | What the command mounts for it |
|---|---|---|
| `text-index` | Publishes the seam that an embedder row claims, and adds `/text-index` and the doctor section | `text-index-engine`: the chunker, the turbovec store, and the two tools |
| `text-index-local` | Names a `sentence-transformers` model as the embedder. Its claim registers the manual and the skill | `text-index-local-engine`: the model, loaded on first use |
| `text-index-endpoint` | Names an OpenAI-compatible embeddings endpoint as the embedder. Its claim registers the manual and the skill | `text-index-endpoint-engine`: the client for the endpoint. Nothing is loaded |

`subprocess`/`subprocess-local` and `code-runtime`/`code-runtime-python` already
have this split in this tree. The split has three benefits:

- A deployment with an embeddings **endpoint** (the `/v1/embeddings` of
  llama.cpp, or the API of a provider) mounts `text-index-endpoint` in place of
  `text-index-local` and keeps the tools. See "An endpoint instead of a local
  model". A deployment with another source of vectors writes its own embedder
  row.
- The **tests** run the real turbovec index against a deterministic stub
  embedder. The suite therefore proves chunking, paging, and persistence without
  a model download.
- A profile can leave out the row that brings in **torch**.

**The tools appear only when an embedder is claimed.** In a session, the claim
of the embedder row registers the manual, on the scope of that row. A profile
with `text-index` and no embedder therefore advertises nothing. `subagent-task`
states the same rule, for the same reason: a tool that is named in every prompt
and refused on every call spends context to teach the model a capability that
the deployment does not have. The command keeps the same rule one process down.

**The profile of the session still chooses the model.** The config of the
embedder row is passed to the command with the config of `text-index`. For
`text-index-local`, that is `model`, the two prefixes, `batch_size`,
`trust_remote_code`, and `cache`. For `text-index-endpoint`, it is the settings
in "An endpoint instead of a local model". The command knows which embedder row
to mount because that row is named there. A person at a shell writes the configs
of the same two rows in `$PH_HOME/tools/ph-text-index.yaml`.

## The two tools

**`text_index(paths?, glob?, exclude?, forget?)`** reads each document through
`ctx.fs`, cuts it into passages, embeds them, and stores them. Indexing a file
again replaces its passages, so it is correct to run the tool again after an
edit. `forget: true` removes a document instead.

When a call does not say what to index, the `index` block of the row decides:

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

How the scope settings work:

- **`exclude`** is read against the workspace root. A directory that it covers
  is never entered.
- **The `glob` of a call** replaces `include` for that call.
- **The `exclude` of a call** is *added* to the `exclude` of the row.
- **A file that is named directly in `paths`** is indexed whatever is excluded.

**Nothing watches the filesystem.** There is no daemon and no file watcher. The
index changes when `text_index` runs, and at no other time. After edits, run it
again.

That is cheap because **git or jj is asked first** (`ph.seams.changes`). A
document that the version control vouches for is not read, not cut again, and,
most important, **not embedded again**. A document that is read is compared with
a digest of its text, and a document whose digest has not changed is not
embedded again either. Before these filters, every call embedded the whole
corpus again: 20 s of MiniLM for the `docs/` of this repository, or 185 s under
nomic. The workspace provider states which backend answers. A tree with no
version control pays for the read and relies on the digest, and a test covers
that case.

A document over `max_bytes` is *skipped and reported*. It is not an error. A
call over a directory should not fail because it found a minified bundle. A
caller who never learns what was skipped cannot tell a quiet corpus from a quiet
failure.

**`text_search(query, k?, paths?)`** returns the passages with their line ranges
and scores. `paths` restricts the search to a subtree.

An earlier version of this paragraph claimed that a filtered search costs *less*
than an unfiltered search, because turbovec filters inside the SIMD kernel.
Measurement shows that the claim is wrong. The kernel does filter, but building
the id list on the Python side dominates it by roughly fifteen times (0.11 ms
against a 0.22 ms unfiltered search on a 10 000-chunk index). Selectivity
changes nothing. Use `paths` for **precision**, not for speed.

## Everything is read through `ctx.fs`

Every indexed byte arrives through the `ctx.fs.read` of the command. The ignore
rules of the walk, the size bound, and the skip reasons belong to that seam, and
are stated one time. In a session, that `ctx.fs` belongs to the command and not
to the session. The sandbox of the calling agent therefore limits the reads of
an indexing run there. See "Where the tools run".

In a session, the index is **per agent**: each agent has its own index, in its
scratch directory. See "On disk" for where the index is and why it is keyed that
way.

## Chunking

Passages are cut at paragraphs, not at a fixed character stride. A stride is
simpler, but it reliably splits the one sentence that answers the query across
two chunks, and then neither chunk retrieves. A blank line is where the author
of the document already marked a new idea. Every chunk carries its 1-based line
span, because a chunk that does not know its place in the file is a block of
prose that the agent must then locate. `overlap_chars` carries the tail of the
previous chunk forward. That is for a paragraph whose meaning depends on the
paragraph before it.

## On disk

`index.tvim` is in the format of turbovec, written with `sync()`: incremental,
one fsync for each call, and crash-safe at any byte. `chunks.json` is the
sidecar. It holds the path, the line span, and the text of each passage. It also
holds the embedder log: what the embedder reported about itself each time an
index run found that the answer had changed. See "What the embedder said".

**The sidecar holds the text on purpose.** Storing only pointers would keep the
file small. But every hit would then need a second tool call before the model
learns whether the hit was any good, and that call would be answered from a file
that may have changed since it was indexed. The text is also the ceiling: the
sidecar is rewritten whole on every save. That is fine for a corpus of documents
and wrong for millions.

A crash between the two writes can leave the two files different. The vectors
are therefore committed first, and `open` reconciles toward the sidecar in both
directions:

- It drops any chunk that the index cannot answer for, because a chunk that
  nobody can retrieve is invisible.
- It removes any vector that the sidecar does not name. A vector with no text
  would appear as a hit that this row could not describe, and the next add would
  allocate its id again.

**One index for each tree and embedder**, because what an index holds belongs to
one tree: documents that are named relative to it, one version-control token,
and a sweep that drops what that tree no longer has.

- **In a session**, the index is in the scratch directory of the agent that owns
  the tree, `$PH_HOME/scratch/<session>/<agent>/text-index/<embedder>`. The
  owner is the calling agent, or the ancestor whose tree it shares. A child in
  its own worktree indexes its own tree. A resumed session has the same session
  id and agent id, and therefore the same index. `text_index` then embeds only
  what version control says changed. The cost is that a new session embeds the
  corpus again and does not find the index of the last session.
- **From a shell**, `--state DIR` puts the index in `DIR/<embedder>`, and so
  does `$PH_TOOL_STATE`. `ph-utcp` sets that variable for each call, and another
  harness can set it. With neither, the index is
  `$PH_CACHE/text-index/<workspace>/<embedder>`. That workspace directory
  records its workspace. When the workspace has been gone for a week, the next
  change removes the directory (`ph.workspace_caches`).

**One writer at a time.** Processes that share a directory share its lock. A
change is made under the `writer.lock` in that directory, and it starts from
what is on disk: a copy that another writer has changed since it was loaded is
loaded again. A second writer waits `wait_s` for the first, and then answers
`state: indexing`. A search takes no lock. It reads what is saved, and every
file that a writer saves is replaced whole.

### What the embedder said

The sidecar keeps an embedder log, `embedder`, oldest entry first. The log is
written in the same save as the passages, under the same writer lock. The log
is therefore always saved together with the passages that it was used to cut:

```json
"embedder": [
  {"askedOf": "http://127.0.0.1:8080", "askedAtMs": 1762905600000,
   "reported": {"n_ctx": 2048, "total_slots": 4},
   "limit": {"tokens": 2048, "source": "server"},
   "learned": {"tokens": 512, "density": 2.7}},
  {"askedOf": "http://127.0.0.1:8080", "askedAtMs": 1763510400000,
   "reported": {"n_ctx": 8192, "total_slots": 4},
   "limit": {"tokens": 8192, "source": "server"},
   "learned": {}}
]
```

| Field | What it holds |
|---|---|
| `askedOf` | The address that was asked. Absent for a model in the process of the command |
| `askedAtMs` | When this answer was first given, in milliseconds since the epoch |
| `reported` | What a server reported, unchanged: the `n_ctx`, `total_slots`, `model_path`, and `model_alias` of the `/props` of a llama.cpp, where present |
| `limit` | How long an input the embedder reads, and the source of that number: `configured`, `model`, or `server` |
| `learned` | What refusals taught: `tokens`, a limit lower than the reported limit, and `density`, the bytes of this corpus that make a token |

- **The newest entry is the one that is used.** The entries before it show what
  changed, and when. The `model` of the sidecar says which embedder the index
  belongs to, as it always did, and an index of another model is refused.
- **An index run that embeds asks the embedder.** `text-index-endpoint` asks the
  `/props` of the server unless `max_tokens` is set. `text-index-local` reads
  the length of the model that the run has loaded. A run that embeds nothing
  asks nothing. An answer that matches the newest entry is not added again.
- **A changed answer is a new entry, and `learned` starts again.** A server can
  be moved or given another window, and what its refusals taught was learned
  under the old answer. The result of the run reports the change in
  `embedder_changed`: "The embedder now reads 8192 tokens (the server's); when
  last asked it read 2048 tokens (the server's)." An entry from another address
  is not used.
- **A search reads the newest entry and asks nothing.** Each search is its own
  command, and would otherwise ask the server a question every time. A search
  writes nothing. Only an index run writes the log, under its lock, so what a
  refused question teaches stays with that search. A search of an index that
  holds nothing embeds nothing: it answers immediately.
- **`status` reads the log.** `ph-text-index status` and `/text-index status`
  report the limit from the newest entry. They load no model and ask no server.

### Calibration

The TQ+ calibration of turbovec is worth 2.5 to 8.7 points of R@10. Upstream is
emphatic about the one way to get it wrong. The sample must be a uniform random,
representative draw of what the index will hold, and a clustered prefix "fits a
calibration that actively destroys recall". An incremental indexer has no such
draw at the moment when it would have to commit one. The first document is the
most clustered prefix that there is.

The calibration is therefore committed in exactly the situation that upstream
describes: the index is **empty**, and the first add is large enough (1 024
rows) to sample from. A random sample of that add *is* a representative draw,
and calibrating before the add is the documented order. An add is one document,
not one indexing run, so this is rare: the first document that is indexed must
be cut into 1 024 passages or more by itself. In every other case the index
stays uncalibrated. That is plain TurboQuant, which is good and not wrong.
`calibrate: false` turns off even that.

## Using it from the RLM

There is nothing to do. In Code Mode, every registered tool is in the generated
SDK listing. These tools therefore arrive as `await tools.text_index(...)` and
`await tools.text_search(...)`, with no work in this package. The
[`rlm-indexed`](../phern/src/ph_app/profiles.py) profile is `rlm-stable` plus
this bundle and the bundle of `ph-code-graph`:

```bash
phern --profile rlm-indexed --provider llama --model <model> --mode tui
```

This package registers a `ph.bundles` entry point. On an install without this
distribution, that profile is therefore *hidden*. It is not offered and then
left to fail at mount. `available_profiles()` gates a profile on whether its
bundles resolve, and the refusal names the package to install.

The bundle carries **both** rows, and that matters. The seam registers no tools
until an embedder is claimed. A bundle with only `text-index` would mount a
service and advertise nothing, with no error anywhere. A deployment that brings
its own embedder disables `text-index-local` and mounts its own provider row.

## Switching models

The default is **`sentence-transformers/all-MiniLM-L6-v2`**. It is
384-dimensional and about 90 MB. It is symmetric, so it needs no prefixes. It is
fast enough on a CPU that indexing a documentation tree takes about as long as a
coffee break. It is the *small* choice, not the best choice.

The index directory is keyed by a digest of the identity of the embedder: the
model name **and** both prefixes, because all three move the vector space.
Switching is therefore safe to try. A new model gets its own index, and
switching back finds the old index intact. When an embedder opens an index that
another embedder built, the index raises `IndexMismatch`. It does not return
neighbors that were computed in a space that nothing shares.

### A worked upgrade: `nomic-embed-text-v1.5`, measured

Run against the `docs/seams` of this repository (30 documents, 235 passages),
with the same queries and the same chunking:

| | MiniLM-L6-v2 | nomic-embed-text-v1.5 |
|---|---|---|
| Dimensions | 384 | 768 |
| Load, cold | 19 s | 26 s |
| Index 235 passages | **5.4 s** | 41.7 s |
| *"stop an agent writing in my own checkout"* | workspace.md (0.319) | workspace.md (**0.599**) |
| *"what confines a shell command on linux"* | shell.md (0.529) | shell.md (**0.753**) |

On a harder corpus, nomic got **3/8** against the **1/8** of MiniLM, at 185 s of
indexing against 20 s. The corpus was all 57 files of `docs/`, 539 passages. The
test was eight paraphrase queries, scored on whether the top hit was the right
*file*.

Read that result fairly in both directions. nomic retrieves better, and it
separates hits with much more confidence. It also costs roughly **8× the
indexing time**, and 3/8 is not good retrieval in absolute terms. Some of those
misses come from the crude scoring (a filename substring), but not all of them.
A corpus of long design notes is hard, and *narrowing the corpus* helped more
than changing the model: MiniLM found `workspace.md` over `docs/seams` and
missed it over all of `docs/`. Use `paths=` before you choose a bigger model.

**MiniLM stays the default** for that reason. The cheap model is good enough for
a scoped corpus, and the expensive model does not rescue an unscoped corpus.

**To switch to nomic, install its extra and set three fields.**
`nomic-embed-text-v1.5` is 768-dimensional, has an 8192-token context, and
retrieves materially better than MiniLM. Each thing that it needs is a field of
the row and not a special case:

```bash
pip install 'ph-text-index[nomic]'   # einops, which its remote code imports
```

```yaml
- id: text-index-local
  config:
    model: nomic-ai/nomic-embed-text-v1.5
    # The model is asymmetric: it must know whether it embeds a question or a
    # passage. Both prefixes are part of the index identity, so this model gets
    # its own directory automatically.
    queryPrefix: "search_query: "
    documentPrefix: "search_document: "
    # Its config.json carries an `auto_map` that points at
    # nomic-ai/nomic-bert-2048, so sentence-transformers does not load it
    # without this field. Read the next paragraph before you set it.
    trustRemoteCode: true
```

**`trustRemoteCode` is a trust decision, not a compatibility flag.** It
downloads Python from the repository of the model (here from a *second*
repository, through `auto_map`) and executes it in the process of the command.
It is off by default, and nothing infers it from a load failure, because "retry
with arbitrary code execution enabled" is not a fallback. It is deliberately
*not* part of the index identity. It governs what may load, not where a vector
lands, so granting or revoking it does not invalidate an index.

A model whose weights are safetensors, and whose architecture `transformers`
already knows, needs none of this. `BAAI/bge-base-en-v1.5` and the `e5` family
are drop-in with prefixes alone.

**The `[nomic]` extra is not optional.** Without `einops`, the model downloads
its weights *and* its remote code successfully and then fails at import:

```
ImportError: This modeling file requires the following packages that were not
found in your environment: einops
```

This is a missing transitive dependency that only the code of the model knows
about. No amount of pre-downloading would have caught it. It is the reason that
provisioning **loads** the model and does not only fetch it. See "Where the
weights live, and installing on purpose".

## A passage fits its embedder

A model reads a fixed number of tokens and no more. Past that number, a local
model drops the rest silently: the tail of the passage is in the index and can
never be found. A server refuses the request. Measured on the `docs/seams` of
this repository with the default model, which reads 256 tokens: 144 of 233
passages cut at 1,200 characters were longer, and the last eleventh of the text
was never embedded.

Passages are therefore packed by what the embedder counts as well as by
`max_chars`, and each passage is as long as both allow:

| Embedder | Where its limit comes from | How text is counted |
|---|---|---|
| `text-index-local` | The length of the model. `max_tokens` can lower it | With the tokenizer of the model, exactly |
| `text-index-endpoint` | `max_tokens`, or else the `/props` of a llama.cpp | By its bytes, three to a token (`ph.llm.tokens.tokens_of`, the estimate that also sizes the state of the decision model) |

- **Passages are packed in the unit of the embedder.** `max_chars` says how big
  a passage should be. The limit says how much the embedder reads, in its
  tokens. A character is a quarter of a token in English prose, a third in code,
  and one or more in a script that takes three bytes for each character. The
  packer therefore keeps two bounds: `max_chars`, and the limit minus the
  passage prefix and the start and end markers of the model. Each paragraph and
  line is weighed separately. A table in a page of prose is therefore cut small
  while the prose around it is not, and text in any script fits at the first
  cut. Every cut inside a line falls between characters as a reader sees them
  (`ph.characters.cut_before`): an accent, an emoji sequence, or a flag is never
  split across two passages. On the `docs/seams` of this repository with the
  default model, this gives 316 passages, none over 256 tokens. Sizing in
  characters gave 388 passages and packed three documents twice.
- **A larger limit changes nothing.** Smaller passages retrieve better, so
  `max_chars` stays the ceiling. Under an 8,192-token window the same documents
  are 233 passages, the same as with no limit.
- **Passages are checked as they are sent.** A packing adds up the weights of
  the parts of a passage, and an embedder counts the whole. The passages of each
  document are therefore measured before they are sent. A passage can be over
  the limit when a tokenizer counts more for the whole than for its parts. The
  document of such a passage is packed again with less room. It is packed again
  and not split in place, so every passage still starts and ends on the lines
  that it reports.
- **A refusal does not end the run.** When the embedder refuses a passage as too
  long, the passage is packed with less room and sent again. Some refusals state
  in numbers what the embedder takes and how long the input was. A llama.cpp and
  OpenAI both do that. The first number is then the limit for the rest of the
  run, and the second number gives the density of the corpus, so one refusal
  settles both. Both are kept in the index (`learned`, in the embedder log), so
  the next run does not spend a refusal to learn them again. A refusal that
  names no number halves the passages of the document and teaches nothing about
  the next. A document that does not fit even at 16 tokens is skipped, with the
  words of the embedder, and listed.
- **The tool reports it.** When the limit ended a passage sooner than
  `max_chars` would have, `text_index` reports what the passages were held to,
  and by what ("Passages are held to 254 tokens of text, to fit the embedder's
  limit of 256 tokens (the model's)"). It also names each document that had to
  be packed again. A run that finished in the background reports the same in the
  note that tells the agent that it ended.
- **A question is held to the limit too.** A question that is longer than the
  embedder reads is searched by its head: as much of its start as fits, up to
  the end of a word. The answer says how many of its characters were searched.
  Previously, a local model did that silently, and a server refused the search.
- **`phern doctor` and `/text-index status` report the limit.** `phern doctor`
  shows a setting. The length of a model or of a server is learned by asking,
  and asking a local model means loading it. `/text-index status` therefore
  shows what the last run that embedded found and kept in the embedder log.

What a server reports is not always what it takes. The `/props` of a llama.cpp
gives its window for each slot: `-c 8192` with four slots reports 2,048. A
llama.cpp also refuses an embedding that is longer than its *batch* (`-b`, 512
by default), and `/props` does not mention the batch. The index learns the real
number from the first refusal and keeps it. To use the whole length of a model,
start the server with `-b` and `-ub` as large as the window. A hosted API has
nothing to ask, so write `max_tokens` there (8191 for the embedding models of
OpenAI).

Changing `max_tokens` changes how a document is cut from then on. It does not
cut the whole index again. A document that the version control vouches for is
skipped before it is read, so its passages stay as they were until the document
changes. A document that is read is cut again, because its digest covers the
`max_tokens` setting and no longer matches. To cut everything at the new length,
drop the documents with `text_index(forget: true)` and index again.

## An endpoint instead of a local model

`text-index-endpoint` is the remote embedder: an embedder that runs in the
process of somebody else. It sends the OpenAI-compatible `/v1/embeddings`
request, so one row covers a local server (llama.cpp, Ollama, vLLM) and a hosted
API. Mount it in place of `text-index-local`:

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
- **The fast local path.** Every call is one command, and with
  `text-index-local` each call loads the model. A local llama.cpp behind this
  row holds the model instead, and a search loads nothing.
- **A remote endpoint is sent the text of every indexed document and of every
  question.** By writing this row, a deployment states that it intends that.
- **No key is stored.** `api_key_env` names a variable. The environment of a
  command drops every name that looks like a credential. The row therefore makes
  `ph-utcp` pass that one variable from the environment in which pH was started,
  each time it starts the command. The manual gives another harness the same
  name: it lists the name among the variables that each tool inherits. A command
  that finds the variable empty fails with `MISSING_CREDENTIAL` (the code that
  an adapter uses for the same failure), names the variable, and sends nothing.
- **Only the tool gets the key.** Its value is passed only to the commands of
  this manual. The `bash` of an agent starts from the scrubbed environment and
  does not have the key. That does not hold in three cases:
  - **A name that the scrub does not match.** The scrub drops a variable whose
    name contains `KEY`, `SECRET`, `TOKEN`, `PASSWORD`, `PASSWD`, or
    `CREDENTIAL`. A variable with any other name is already in the environment
    of every command. `ph-utcp` warns at mount about any manual that passes such
    a variable, and reports it in `phern doctor`. That includes a variable that
    `api_key_env` names.
  - **No sandbox backend.** Then nothing protects the key: `bash` runs as the
    same user as the harness and can read the environment of the harness.
  - **A key kept in a readable file.** If the commands of the agent can read a
    shell profile or a `.env` file, `bash` can read it.
- **A confined agent needs the host to be allowed.** A sandboxed command has the
  network that its deployment allows. Name the host and port of the endpoint in
  `sandbox-allow` (`network: {hosts: ["127.0.0.1:8080"]}`). The list replaces
  the shipped list, and a top-level `hosts` is refused at mount. An endpoint
  that is not allowed, or is down, fails with `TEXT_INDEX_ENDPOINT_UNREACHABLE`.
  A server that answers with an error fails with `TEXT_INDEX_ENDPOINT_FAILED`
  and the words of the server, or with `CONTEXT_WINDOW_EXCEEDED` when the words
  say that an input was too long.
- **An allowed host is allowed to every confined command.** `sandbox-allow`
  belongs to the deployment, not to this tool: the `bash` of an agent reaches
  the same host. A hosted API refuses `bash` because the key is missing. A
  server that needs no key answers `bash` too.
- **What a call spends is not recorded.** The request belongs to the command, so
  the cost of an embedding is not in the log of a session and is not charged to
  the budget of a goal.
- **The index is keyed by the model, the prefixes, and a requested
  `dimensions`.** It is not keyed by the address, so moving a server keeps the
  index. A hosted model that changes under an unchanged name cannot be
  distinguished except by its width, which is checked on every add.
- **One embedder for each session.** There is no fallback to a second embedder.
  An index is one vector space, and a search that was silently answered from
  another space would be an answer from a different index. When an endpoint
  cannot be reached, that failure is the answer.

The embeddings API of a provider is the same row with its address, the name of
its key, and its host in `sandbox-allow`
(`network: {hosts: ["api.openai.com:443"]}`):

```yaml
- id: text-index-endpoint
  name: text-index-endpoint
  config:
    base_url: https://api.openai.com/v1
    model: text-embedding-3-small
    api_key_env: OPENAI_API_KEY        # the NAME of the variable that holds the key
    dimensions: 512                    # optional, where the model can shorten its vectors
    max_tokens: 8191                   # what the embedding model reads; an API has nothing to ask
```

From a shell, `$PH_HOME/tools/ph-text-index.yaml` names the same row:

```yaml
text-index-endpoint:
  base_url: http://127.0.0.1:8080/v1
  model: nomic-embed-text-v1.5
```

## Where the weights live, and installing on purpose

The weights are in `$PH_CACHE/models`, which the row sets (`cache:` overrides
it). `sentence-transformers` alone would put them in `$HF_HOME` or
`~/.cache/huggingface`. Those are outside all three roots of pH, so a gigabyte
of weights would be in a place that `phern doctor` never mentions and that
`rm -rf $PH_CACHE` would not reclaim. `$PH_CACHE` is the root for artifacts that
are large and can be rebuilt (Q1), and the runtime venv is there for the same
reason.

**Provision before the agent needs the model.** There are three ways, in
increasing order of how unattended the run is:

```
/text-index status     # is the model ready? costs nothing, downloads nothing
/text-index install    # fetch and load it now, and say what happened
```

This is a **command**, not a tool. A person asks the harness to do this, and a
route through a model turn would record the model in the log as the one that
decided it. The command costs no turn, and it runs outside the sandbox, because
`install` needs the network. From a shell, `ph-text-index status` and
`ph-text-index install` are the same two commands. `status` reports one of three
things about a local model: `loaded`, `downloaded, loads on first use`, or
`not downloaded`. For an endpoint it reports `nothing to load`.

**A tool call does not need the network after the weights are on disk.** The
command loads a downloaded model without asking the hub. A confined agent, whose
network is the allowlist of the deployment, can therefore search. When a model
is not downloaded, the first call that needs it fetches it where the network
allows. Where the network does not allow it, the call fails with
`TEXT_INDEX_MODEL_MISSING` and names `/text-index install`.

For an **unattended** run (a daemon, a scheduled tick, `phern -p` in CI), nobody
is present to type either command:

```yaml
- id: text-index-local
  config:
    preload: true
```

The row of the session then asks the command to load the model one time, at
**mount**. If the command cannot load it, the row refuses the mount with a
`MountRefusal` sentence. That is the cookbook's own rule: refuse at mount, not
at first use, because by then the agent is running and "refuse to start" has
already been disobeyed. `preload` is off by default, because a person at a TUI
prefers that the harness starts now and pays for the model when they use it.

All three ways do the same thing: they **load** the model and do not only fetch
it. The `einops` failure above is exactly the reason. Whatever upstream reports
on a failure is passed through verbatim, because that sentence names the fix.

## The skill

The package installs a `text-search` skill. An RLM under progressive disclosure
sees one line in its catalog and can read the page when it needs it. The page
says how to scope a search, that a score is not a confidence, and that a passage
can be an argument that the document later rejects.

The row registers the skill. A directory scan does not find it. The skill
therefore arrives exactly when the tools do, and is removed with them.
`skills-progressive` ships an empty `paths` on purpose. Scanning a well-known
directory would make "install a skill" mean "drop a file somewhere", and a
distribution installs a skill deliberately (I7).

[turbovec]: https://pypi.org/project/turbovec/
