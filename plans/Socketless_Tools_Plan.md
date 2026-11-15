# Socketless tools — drop the adapter row, the host and both sockets

*2026-11-12. Follows the design review of 0.11.0 (`plans/Utcp_Tool_Port_Plan.md` is
the record of what this changes).*

**Status, 2026-11-13.** S1 to S7 are built, with their gates: the documents are
written and every member is 0.12.0. Nothing is committed. What building them settled,
and where it differs from the design below, is under "What building it settled".

## The goal

`ph-code-graph` and `ph-text-index` run as plain commands and nothing else: no
process that stays up, no socket in `$PH_RUNTIME`. A tool reaches an embeddings API
itself, with a key the profile names and a host the deployment allows, and keeps
everything it knows — the index and what the embedder said about itself — in its
state directory. A long run reports on its own stderr, and that stream is how the
session shows progress and learns that a run stopped.

## Decided with the user

1. **`text-index-adapter` is removed, and its socket with it.** `text-index-endpoint`
   is the remote embedder: the key by name (`api_key_env`, `Manual.pass_env`), the
   request through the egress proxy to a host in `sandbox-allow`.
2. **The host is removed, and its socket with it.** Every call is one command.
3. **What the embedder reported is saved beside the index** — the `/props` answer,
   asked when an index run starts — and read back by later calls.
4. **`--progress` makes a command report on stderr.** The session reads it for the
   footer and the completion note, and to detect a run that stopped.

## What exists, and what each socket was for

| | Embedding socket | Host socket |
|---|---|---|
| Where | `ph_text_index/row.py:474` (`_Embedding`) | `ph/toolcli.py:1140` (`_serve`), `ph_utcp/__init__.py:615` |
| Bought | the key stays in the harness; spend in the log; Google's API | a model loaded once; indexing past the call; progress |
| Cost | reachable by any confined command, unauthenticated (review, finding 2) | the same reach; 500 lines of host, watcher and joining |

What the host did that still has to happen without it:

- **Indexing unasked at a root agent's first step** (`index.on_mount`).
- **An index tool that answers before the work ends** (`wait_s`, `state: indexing`).
- **The footer reading and the note when a run ends.**

None of the three needs a process that answers calls. Each needs a command that
runs for a while and says how far it has got.

## The design

### Progress on stderr

- **`call <tool> --progress`** writes JSONL to stderr: one complete JSON object a
  line, UTF-8, each line flushed as it is written. The first report, then at most
  one every `REPORT_EVERY_S`, then exactly one last line.

  ```json
  {"state": "working", "done": 120, "total": 539, "what": "documents"}
  {"state": "done", "done": 539, "total": 539, "what": "documents", "changed": true, "note": "…"}
  {"state": "failed", "done": 120, "total": 539, "what": "documents", "code": "…", "failure": "…"}
  ```

  Each record is today's `Progress` model without `owed` and `started`, with
  `code` beside `failure` and `changed` on the last line. Under `--progress`
  nothing else is written to that stream; a reader that meets a line that is not
  one of these records logs it and goes on.
- **Written to the real descriptor.** `_stderr_discarded` points fd 2 at nothing so
  a model library's progress bar cannot reach a client. With `--progress` the
  original descriptor is kept open and only these lines are written to it.
- **Never on a manual's template.** A UTCP client runs each command as
  `$(<command> 2>&1)`, so a line on stderr would be in front of the answer. The
  flag is for a caller that reads the two streams apart, which `ph-utcp` is when
  it starts a run itself. stdout is unchanged: one envelope.
- **A run that stops is known three ways**, and all three reach the agent:
  1. a last line with `state: failed` — the tool's own code and sentence;
  2. a nonzero exit with no last line — said with the exit code and the log file;
  3. the stream ending with no last line and no exit code of its own — killed, out
     of memory, a native crash.
  The traceback of a failed run is written to the command's log by the same door
  that writes the last line (review, code-graph finding 7).
- **Core door.** The subprocess seam gains a line listener for one stream
  (`SubprocessSpawnSpec.on_stderr_line`), called by `pump` as lines arrive and not
  counted against `max_output`: `pump` keeps the head of a stream, and a progress
  stream's last line is the one that matters. `ctx.shell.run_argv` passes it on.

### The background run replaces the host

- **`Manual.host` becomes `Manual.background`**: `command` (up to and including
  `call`), `label`, `tools` (the tools that run this way), `eager`.
- **One run a tree at a time**, keyed as hosts are by manual and the tree's owner
  (`UtcpSeam.owner_of` is unchanged). It is a job of the owner's (`ctx.jobs`),
  started by `ctx.shell.run_argv` for that agent — in its workspace, under its
  sandbox — with `--progress` and the listener.
- **Unasked, at a root agent's first step**, for each of `tools`, when `eager`.
  As today, once a tree for the life of the mount.
- **Asked for by a model.** A call of one of `tools` starts the run with the
  call's arguments (`--args-json`) and waits `wait_s`:
  - ended in time: its stdout envelope is the answer, as any call's is;
  - still going: the command is asked again, finds the writer lock held, and
    answers `state: indexing` in its own schema. `ph-utcp` adds the count it holds
    to the text.
  - a run already going for that tree: the same `state: indexing` answer. The call
    is not queued. The note at the end tells the agent, and a call made then is
    the cheap incremental one.
- **The reading and the note come from the lines.** `ph-utcp` keeps the last
  report in memory for the footer. When the run ends it injects the note: the
  count and the tool's `note`, or why it stopped.
- **No note for an unasked run that changed nothing.** The last line says whether
  anything was added or removed; a resumed session is not told twice a start that
  its indexes are still current.
- **A search during a run** is an ordinary call. The command sees the lock held
  and marks its answer `indexing`; `ph-utcp` adds the count.

### The key is the tool's alone

The claim to hold: an agent's `bash` cannot use the embeddings API, because it
never has the key. The key's value is read from the harness's environment when
`ph-utcp` starts one of the manual's commands and is set on that command only
(`UtcpSeam._surroundings`, `Manual.pass_env`). Checked on 2026-11-12:

- **`bash` does not inherit it.** Every other command starts from the scrubbed
  environment, which drops a name containing `KEY`, `SECRET`, `TOKEN`,
  `PASSWORD`, `PASSWD` or `CREDENTIAL`. Probed: `EMBED_API_KEY` and `NOMIC_TOKEN`
  are dropped.
- **`bash` cannot read it off the tool's process.** Each confined command has its
  own PID namespace and `/proc`. Probed under the sandbox's flags: a variable of a
  process outside is not readable, and four pids are visible.
- **The tool's own children do not get it.** `git` and `jj` are started by the
  command's subprocess row from the scrubbed environment.

Where it does not hold, and what this plan does about each:

- **A variable whose name the scrub does not match is in every command's
  environment already.** Probed: `EMBED_AUTH` is kept. The endpoint row says so at
  mount when `api_key_env` names one — a warning and a `phern doctor` line.
- **Code from the workspace can run inside the tool's process** until S1 (`-P`).
- **With no sandbox backend nothing holds.** `bash` runs as the same user and can
  read the harness's own environment from `/proc`. True of every credential
  today, and said in the README beside the endpoint row.
- **A key kept in a file the user can read** — a shell profile, a `.env`, a unit
  file — is readable by `bash`, since `/` is bound read-only. The deployment's to
  avoid; said in the same place.
- **The host is reachable without the key.** An allowance in `sandbox-allow` is the
  deployment's, for every confined command: a policy can refuse the network and
  cannot add a host. So a hosted API refuses `bash` for want of the key, and a
  local server with no key (`api_key_env: ""`) answers it. See "Open", 4.

### What the embedder said, saved beside the index

- **`embedder.json`** in the index directory, beside `index.tvim` and
  `chunks.json`, written with `write_atomic` under the writer lock:

  ```json
  {"asked_at": 1762905600, "asked_of": "http://127.0.0.1:9931", "model": "nomic-embed-text-v1.5",
   "reported": {"n_ctx": 8192}, "limit": {"tokens": 8192, "source": "server"},
   "learned": {"tokens": 512, "density": 2.7}}
  ```

- **Asked when an index run starts**, by a run that has something to embed, and
  saved before the first document is sent. That is the one request for
  `text-index-endpoint`; for `text-index-local` it is the loaded model's own
  length, which that run has loaded anyway.
- **Read by every later call.** A search reads the file and asks nothing. Only a
  call that finds no file, or one asked of another address or model, asks — and
  saves what it hears. This removes the `/props` request every one-shot search
  would otherwise make.
- **What a refusal taught is saved too** (`learned`). With no host there is no
  process for it to live in, and each run would otherwise spend a refusal
  relearning it.
- **A changed answer replaces the old one and drops `learned`.** The old limit was
  learned under the old configuration; this is the stale-number case the last
  plan gave as the reason not to persist. The result says the change in a line:
  "the server now reads 8,192 tokens; it read 512".
- **`TextIndex.limit`, the phrase in the sidecar, is removed.** `status` and
  `phern doctor` read `embedder.json`.

### What is removed

- **`ph-text-index`**: `text-index-adapter` and its engine row, `AdapterConfig`,
  `_Embedding`, `SessionEmbedder`, `$PH_TEXT_INDEX_EMBEDDER`, their entry points,
  `test_text_index_adapter.py`, the README sections.
- **`ph-core`, embeddings**: `EmbeddingAdapter`, `EmbeddingRequest`,
  `LlmRuntime.embed` and `embeds`, `vectors_of`; the `llm/embedded` and
  `llm/embedding-failed` records; the `embedding` token source. `input_too_long`
  and `too_long_facts` stay, for the endpoint embedder; `openai_embeddings_body`
  and `openai_vectors` move to `_embed.py`, their one user.
- **`phern`**: `embed` on the OpenAI-compatible and Google adapters.
- **`ph.toolcli`**: `serve`, `ToolHost`, `TOOL_HOST`, `$PH_TOOL_HOST`,
  `$PH_TOOL_AGENT`, `_asked_of_host`, `_UNASKED`, `progress.json` and its two
  functions, `BackgroundRuns`' joining and grace. What is left is `RunProgress`
  and one function that runs work with its end written.
- **`ph.framing`**: `ask` and `answer_one`. `listening`, `read_frames` and
  `write_frame` stay for the daemon.
- **`ph-utcp`**: `Host`, `_Hosted`, `socket_dir`, `_sockets`, `_host_for`,
  `_listening`, `_hosting`, `_watch`, the socket directory's grant.

## Steps

**S1. `-P`.** `ToolCli.invocation()` spells `python -P -m <module>`. First, because
after S6 the only remote embedder puts a key's value in a confined command.
*Gate: a `shlex.py` at the workspace root does not run under `manual`, `call` or
`status`; without `-P` the same test prints its marker.*

**S2. `--progress`.** The flag, the kept descriptor, the last-line rule, the
traceback in the log; the subprocess seam's line listener and `run_argv`'s
argument. *Gate: a run's stderr is only progress lines with a model library's
noise still discarded; stdout is one envelope; a failing run ends with a
`failed` line carrying the tool's code; a listener sees the last line of a
stream longer than `max_output`.*

**S3. The background run.** `Manual.background`, the job, the reading, the note,
the three ways a stop is known; both rows declare it. The host still exists
beside it until S4. *Gate: the footer moves and the note is delivered from stderr
alone; a run killed with SIGKILL produces a "stopped" note; an unasked run over
an unchanged tree produces none; a call during a run answers `state: indexing`
and starts no second process that indexes.*

**S4. The host goes.** Everything under "What is removed" for `ph.toolcli`,
`ph.framing` and `ph-utcp`. *Gate: no file is created under `$PH_RUNTIME/utcp`
in a session that indexes and searches with both packages; U5's gates that still
apply — nothing embedded twice after a kill, a search during indexing returns
saved passages with the warning — hold with no host.*

**S5. `embedder.json`.** *Gate, against the fixture server: an index run makes one
`/props` request and a search after it makes none; a server whose `/props`
changes is noticed by the next index run, the result says so, and `learned` is
dropped; a refusal's limit is in the file and the next run sends nothing that is
refused.*

**S6. The adapter row goes.** Everything under "What is removed" for
`ph-text-index`, the embeddings half of `ph-core`, and `phern`. *Gate: a profile
naming `text-index-adapter` is refused by name; a 0.11.0 session log that holds
`llm/embedded` records still opens. Under a real `bwrap`, with the fixture server allowed and requiring a
key: `text_index` embeds; the agent's `bash` has no such variable, cannot read it
from `/proc`, and its request to the server is answered 401; a `git` the command
starts has no such variable; a row whose `api_key_env` the scrub does not match
warns at mount.*

**S7. Documents and version.** Both READMEs, `ph-utcp`'s, the cookbook, `DESIGN.md`,
"Upgrading from 0.11" in the root README; 0.12.0 in every member.

## What building it settled

- **S2 to S4 were built as one change.** The host and the progress file went in
  the same edit that added `--progress`, so there was never a tree with both.
- **`--wait SECONDS` on `call`**, beside `--progress`. A run waits `wait_s` for
  another that holds the index and then answers `state: indexing`; `--wait 0` is
  how `ph-utcp` asks for that answer at once after it has done the waiting
  itself. Without it a call during a run waited twice.
- **A search takes no lock.** With no host, every search is a command with no
  copy of the index, and under the old rule it stood behind a run for as long as
  the run embedded. A reader now opens the index as a reader: it changes no
  file and answers for what the sidecar names. That is also the end of the
  review's finding that `status` could remove a writer's unsaved vectors.
- **"Is a run going" is asked with a shared lock** (`ph.locks.file_lock_held`).
  With no progress file to say `working` first, two searches that each tried the
  lock at one moment would have told each other an index was being built.
- **The last line of a failed run is the command's**, written from the envelope
  the failure becomes, so it carries the code and sentence a caller reads; a
  refusal before any work — an unknown tool, a row that will not mount — ends
  the stream the same way.
- **`ph.toolcli.reporting_to(out)`** is public: a caller that runs a tool in its
  own process gets the report `--progress` writes.
- **A profile that names `text-index-adapter` gets the loader's own refusal**:
  `cannot resolve plugin "text-index-adapter"`. It names the row and not
  `text-index-endpoint`; a row kept only to say so would be a shim. The upgrade
  notes say where to go.
- **A 0.11.0 log with `llm/embedded` records opens.** Run against a log written
  by the 0.11.0 tag's own code: both records are kept, and neither counts as
  spend. Not run: a 0.11 session whose recorded profile names the adapter row.
- **`ph.framing`'s parser refuses every line that is not a frame.** Bytes that
  are not UTF-8 and nesting past the parser's depth raised errors of their own;
  both are `FramingError` now. The daemon's socket reads through the same door.
- **The endpoint row warns at mount** when `api_key_env` names a variable the
  scrub does not drop, and `phern doctor` shows the line.
- **`embedder.json` keeps plain values**: `n_ctx`, and `total_slots`,
  `model_path` and `model_alias` where a `/props` answer has them.

Left as it was, with the reason:

- **The three open decisions stand as the plan assumed.** The local model loads
  on every call; embedding spend is in no log; `progress.json` is gone.
- **Open 4, a manual's own allowed hosts, is not built.**

## A `/simplify` pass

Four reviews — reuse, simplification, efficiency, altitude. Fixed:

- **A run that found its work held by another ends `busy`**, a fourth last
  line (`Progress.state`, `reported(busy=)`). It had been written as `done` with
  nothing done, and an agent owed a note was told "finished: 0 documents.
  Searches now cover all of it". It is now told to call again; one nobody
  waited for is told nothing.
- **A call that will answer `state: indexing` walks nothing.** Both engines
  take the lock before `collect` and `claim_home`, so the `--wait 0` answer no
  longer walks the tree it then throws away.
- **A search of an empty index embeds nothing.** It loaded the model — about
  4 s and 450 MB under MiniLM — to answer "0 passages".
- **The search retry loop is gone.** Every file a save writes is replaced
  whole; a probe loaded the vectors 115,000 times beside a writer with no
  failure, and the loop retried any failure at all while a run held the lock.
- **The kept record is written under the writer lock.** A search writes it
  only if it can take the lock that instant, so it cannot overwrite what a run
  just kept.
- **`TextIndexSeam`'s record state is two fields**, `_kept` (directory and
  record) and `_saved`; `EmbedderRecord.effective()` is the one place a refusal
  wins over what was reported. `Limit` is a `WireModel` and `Reads` is gone.
  `embedder.json` is camelCase with absent fields left out, as every wire form
  is, and `asked_at` is `askedAtMs`.
- **The key-name warning is `ph-utcp`'s, for every manual's `pass_env`**, in
  that row's `phern doctor` section; the endpoint row's copy is gone.
- **`ph-utcp` runs a manual's command through one door** (`_command`), and a
  `--wait 0` answer that times out is `UTCP_TIMEOUT` rather than "did not
  answer". `_unwrapped` asks `_tree` instead of re-deriving it.
- Smaller: `run_lock` is one bounded wait; `RunProgress` loses the cancellation
  plumbing the host needed and is handed its sink; `_answered` turns a refusal
  into an envelope and has one tail; `waited_file_lock` is deleted; `HOST_ROWS`
  is `COMMAND_ROWS`; the stderr's last words are a bounded `deque`; the code
  graph's busy answer uses `CodeGraphStore.stats()`; the endpoint's request
  helpers are private.

Left as they were, with the reason:

- **`file_lock_held` and the schedule index's own probe stay two.** They
  disagree on an unopenable file: "not held" here, "cannot judge" there.
- **The index open and the query embedding stay sequential.** Overlapping them
  saves at most the first open against an endpoint, and the empty-index answer
  needs the open first.
- **The text index's guards and `TextIndex.readonly` stay.** The engine rows
  can still be mounted in a session, where a run and a search share a copy;
  and a reader's copy can hold vectors no record names without being stale.
- **Tests that reach `_start`, `_Reports` and `_WAIT` stay.** No public seam
  is worth adding for them.
- **Reading the busy answer's counts without loading the vectors** — about
  160 ms of a `--wait 0` call — is not done.

## After the `/simplify` pass, at the user's word

- **A background run whose command cannot be started is said.** `run_argv`
  raising — a program that is not there, a sandbox that refused it — left the
  run with no result and nothing on its stderr, and `_ended` sent nothing: an
  eager run's failure went unseen. `_Run.error` keeps why; the agent is told
  "could not be started", and a call waiting on the run fails with it instead
  of starting a probe that fails the same way.
- **The embedder record is a log in the sidecar**, not `embedder.json`. The
  user asked for it in "the database storing the index", as table rows. The
  text index has no database — the vectors are turbovec's file and everything
  else is `chunks.json` — so the log is a list in the sidecar
  (`TextIndex.embedder`, `note_embedder`), written by the same atomic save as
  the passages, under the same lock. An entry is added only when the answer
  differs from the newest, which is the one used; the sidecar's `model` is the
  identity, as before, so an entry carries none. A search reads the newest and
  writes nothing: what a refused question teaches is that search's alone.
  Moving the text index to SQLite would make the log a table; not done.
- **Progress lines carry `spent`** (`Progress.spent`, `RunProgress.spent`): a
  generic count by unit, which the text index fills from its embedder
  (`Metered`) — `requests` and `texts`, and `tokens` where the server's `usage`
  reports them. Each run reports its own share of the embedder's counts. A
  failed run's last line carries what it spent, read from the run and not from
  the last line written. `text_index`'s result carries it too (see the second
  `/simplify` pass).

## A second `/simplify` pass

Over what was added after the first, with the same four reviews. Fixed:

- **What a run spent is asked as each line is written** (`RunProgress.spent`
  is a getter the work sets once). It had been set at the top of each
  document, so a run that failed part-way through one left that document's
  requests out of its last line. The getter asks the embedder and not the
  seam: a failed run's last line is written after the mount has gone.
- **Spend is a `Counter`** in both embedders and in the run's share, so a unit
  spent none of is left out: a run that embedded nothing says nothing, where it
  had said "0 requests, 0 texts".
- **`text_index`'s result carries `spent`**, and a line of its text. The spend
  had travelled only on the progress stream, so a call that waited for its
  run, and every call made in-process, never saw it. The note for a run nobody
  waited for carries the line through `Progress.note`; `ph-utcp`'s own
  "Spent:" line is gone.
- **`ph-utcp`'s per-manual spend total is gone.** `phern doctor` mounts a fresh
  profile and the daemon reports only its own sections, so the row was never
  seen outside a test.
- **"The embedder now reads …" is said by the run that saw the change**,
  derived from the log's newest entry before the run and the record after it.
  It had been a field of the seam, set once and never cleared, so an engine
  mounted in a session said it on every later run. A server asked at another
  address is now said too ("is now asked at …"), where its old entry had been
  dropped without a word; and `limit` takes the kept entry when the answer is
  the same rather than copying what it learned onto a new one.
- **One shielded save at the end of a run**, done, stopped or failed — the
  record noted and the passages saved together — where there were two copies.
- **A run that cannot start keeps the error itself** (`_Run.error`), so a
  sandbox's refusal reaches a waiting call with the sandbox's code rather than
  as `UTCP_FAILED`; another error is said with `error_message`.
- Smaller: `_Reports.last` is an `ended` flag, and `RunProgress.line` builds
  every line, `end`'s included; `EmbedderRecord.says_the_same_as` is gone, the
  store comparing dumps without `askedAtMs` so a field added later is compared
  too; `TextIndex.embedder_log`, which nothing called, is gone; `readonly`'s
  docstring is back beside it; "1 request", not "1 requests".

Left as they were, with the reason:

- **The embedder log is not capped.** An entry is added only on a change, and
  each is about 300 bytes and 2 µs to read. The case that grows it without end
  — two profiles alternating addresses over one cache directory — is inferred,
  and a cap would drop the history the user asked for.
- **Texts and calls are still counted by each embedder**, not at the engine's
  one embed door. The door would need a counter the run owns, beside a search
  in the same mount; nothing reads the difference today.
- **`run_command`, a person's `install` and `status`, is not wrapped** to say a
  command that cannot start the way a background run does. It is outside this
  change, and its callers say the error already.

## What breaks

- **`text-index-adapter` is gone.** A profile that names it moves to
  `text-index-endpoint` with `base_url`, `model` and `api_key_env`, and adds the
  endpoint's host to `sandbox-allow`.
- **`text-index-local` loads its model on every call.** Measured in the last
  plan: a search 3.9 s by itself against 0.26 s through a host, 6 s under `bwrap`.
- **Embedding spend is no longer in a session's log or a budget.** The
  `llm/embedded` records and the `embedding` token source are removed.
- **Google's native embeddings API is no longer reached.** Its OpenAI-compatible
  endpoint may serve; not verified.
- **An allowed embeddings host is allowed to every confined command**, the agent's
  `bash` included. The key is handed only to the tool's command.
- **`Manual.host` is `Manual.background`**, with different fields. `serve`,
  `$PH_TOOL_HOST` and `$PH_TOOL_AGENT` are removed, so another harness can no
  longer start a host.
- **`progress.json` is gone.** A search's `indexing` value loses its counts: a
  command knows a run holds the lock and not how far it is. Inside pH the count
  is added to the text.
- **A call during a run is not queued.** In 0.11 a different request took its
  turn behind the running one.
- **`text_index`'s sidecar loses `limit`** and gains an embedder log. An index
  built by 0.11 has none, and its next run that embeds starts one.

## Open, with a recommendation

1. **The local model with no host.** Accept the load on each search (recommended:
   a local llama.cpp behind `text-index-endpoint` is then the fast local path), or
   keep a host for that one row and so keep one socket.
2. **Spend.** Lose it (as planned above), or have the endpoint embedder put the
   provider's `usage` in `text_index`'s value and its note (recommended: cheap,
   and the only place a person would see what a run cost).
3. **`progress.json` for other harnesses.** Drop it (as planned), or keep writing
   it beside the stderr lines so a stock UTCP client's search can say a count.
4. **A keyless embeddings server and `bash`.** Leave it reachable (as planned: the
   key is the barrier, and a server with none was opened by the deployment), or
   give a manual its own allowed hosts, granted to its commands as `writable`
   paths are. That is a new rule in the sandbox seam — today a policy can only
   narrow the network — and is worth it only if local keyless servers should be
   out of an agent's reach.

## Not in this plan

From the review, and untouched by this: a person's `$PH_HOME/tools/<program>.yaml`
layering into session calls; a state grant followed through a symlink for a
sharing child; the version-control shortcut that skips the digest; the code
graph's token advancing on a narrowed run; indexes per session with nothing
sweeping the scratch.
