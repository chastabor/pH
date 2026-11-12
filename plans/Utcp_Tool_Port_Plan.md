# UTCP tool port — phern calls tools through UTCP, starting with code graph and text index

*2026-11-10. A review of `ph-code-graph` and `ph-text-index` against the UTCP Python
library cloned at `sources/python-utcp` (utcp 1.2.0, utcp-cli 1.1.5), and a plan to
make UTCP the way phern reaches both packages' tools and any tool published later.
Revised three times the same day on the user's direction; every decision that was the
user's has been taken.*

**Status, 2026-11-10.** U1 to U7 are built, with U4a: the command line and the file
lock, the manual and the lenient decoder, the `ph-utcp` row, the code graph and the
text index reached through it, each agent's tool state in its scratch, the host that
stays up for each tree with its background indexing, progress and completion note, and
the whole `index` block. A `/simplify` pass over U1 to U4 is applied, and so is the one
fix it deferred (`ToolRunContext.present`).

U8 and U9 are built too: `text-index-endpoint`, the OpenAI-compatible `/v1/embeddings`
embedder, and `text-index-adapter`, which embeds through one of the session's own
model routes. U10's documents were written with each step: both READMEs, `ph-utcp`'s,
the cookbook and the two skills. **Every step of the plan is built.**

What each built step settled, and what it found that the plan had wrong, is under
"What U9 settled", "What U8 settled", "What U5 settled", "What U4a and U6 settled" and
"What U1 to U4 settled" below.

**2026-11-11.** Passages are packed in the embedder's own unit, and the four gaps the
plan had recorded as not done are closed: a long question is cut, `phern doctor` says
the limit, a background run's note says what its result would have, and what a route
spends on embeddings is in the session's log. See "Packed in the embedder's unit, and
four gaps closed", and "A fourth `/simplify` pass" over it.

**2026-11-11, before publishing as 0.11.0.** Both READMEs say how to install and use
the tools three ways, the manual names an endpoint's key variable, and four of the
records under "Not verified" were run for real. Three of those runs found something.
See "Before publishing".

## Before publishing

**Built.**

- **The manual names the variable an endpoint's key is in.** A UTCP client hands a
  command only the variables its template names, and `api_key_env` is known only once
  the endpoint's row has its config. The config field is marked
  (`Annotated[str, ph.wire.EnvName()]`; a `ToolEnv` registration came first), and `manual` lists
  it in each tool's `inherit_env_vars` after the package's own: the name, never the
  value. The registration that reads the manual names no key. Inside pH nothing
  changes: a session's row hands the variable across by `pass_env`, and the scrubbed
  environment still drops a credential a template names.
  - Gate: the library's own UTCP client, with no pH around it, indexes and searches
    against the fixture server and the server sees the key.
  - Live, against the nomic server: 30 documents, 233 passages in 2.9 s with
    `LOCAL_LLAMA_API_KEY` set where the client runs; `MISSING_CREDENTIAL` with it
    unset.
- **`/code-graph install` fetches on both pack versions.** See the second finding.
- **The code graph's background run says what it did**, as the text index's does
  (`said=`): its counts, and the first five files it skipped with why.
- **Version 0.11.0** in every member, the pins between them and `uv.lock`; "Upgrading
  from 0.10" in the root README. Building into `dist/`, the tag and `uv publish` are
  not done.

**Verified, from "Not verified".**

- **A real daemon, a real agent loop, a restart.** `phern daemon` on the `headless`
  profile with its scripted model, the `utcp`, `code-graph`, `text-index` and
  `text-index-endpoint` rows, and `sandbox-allow` naming the embeddings server.
  - `phern agents send` took a step: `agent/pre-step` fired, both hosts started under
    `bwrap`, the text index embedded 30 documents (233 passages) through the
    deployment's egress proxy, and both notes were spliced into the root's log.
  - `phern agents shutdown` ended both hosts.
  - A second daemon, a second turn on the same root: new hosts, the same scratch, and
    the text index's note read "0 passages added, 0 removed; 30 unchanged and not
    re-embedded". `chunks.json` was not rewritten.
- **A tree that moved a long way.** A clone of this repository indexed at `0.9.0`, then
  checked out at `f870c9f`: 211 files changed under `packages` and `docs`.

  | | First index | Unchanged | After the move |
  |---|---|---|---|
  | Code graph, `packages` | 8.9 s, 516 files | 0.39 s | 3.0 s: 174 parsed, 380 current |
  | Text index, `docs`, nomic endpoint | 7.3 s, 60 documents | 0.46 s | 1.6 s: 6 embedded, 55 unchanged |

- **The model's load time here.** MiniLM with its weights on disk: a search by itself
  3.9 s, the same search answered by a host 0.26 s, the host's first 3.4 s.
- **Both packages alone.** Wheels of `ph-core`, `ph-utcp` and `ph-code-graph` in an
  empty environment are 24 packages and about 30 MB, with no phern. The command, the
  settings file, a host on a socket and the Python UTCP client were each run there.

**Found, and fixed.**

- **The grammars are downloaded, and a confined command cannot.** This plan and the
  README both said the pack's 26 bundled grammars unpack from the wheel with no
  network. Measured with 1.16.2 and 1.25.0, an empty cache and no network: no parser,
  python's included. With the network the bundled set arrives as one archive from
  GitHub the first time a parser is asked for. In 0.10 the tools ran in the session's
  process and fetched it unnoticed; a confined host is refused (`sandbox/denied`,
  `github.com:443`), indexed nothing, and told the agent "The code graph is finished:
  34 files".
  - `ensure` now calls the pack's `prefetch` for what is not on disk, and asks for each
    one alone only if the batch fails. On 1.16.2 `download()` fetched none of them, so
    `/code-graph install` reported 0 of 18 ready with the network up; it now fetches
    all 18 in about 3 s on both pack versions, and an index with no network works
    after it.
  - The note says what was skipped and why, and the reason names `/code-graph install`.
  - Both READMEs and the upgrade notes say to run it once before the first session.
  - The docstrings, the skip reason, the install message and `pyproject.toml`'s comment
    no longer say the bundled grammars need no network.

- **A sandboxed command ended in its first few milliseconds outlived pH.** pH ended
  a command with SIGTERM to its group, and sent SIGKILL only if its own child was
  still there after the grace. Under `bwrap` that child is the wrapper's outer
  process; the command is under an inner one, which is init of a PID namespace. For
  the first milliseconds the inner one ignores SIGTERM, as a namespace's init does,
  and has not yet armed `--die-with-parent`. The outer one dies, pH sees its child
  gone and stops, and the inner one goes on to start the command.
  - Measured with `bwrap … --unshare-pid --new-session --unshare-net sleep`, the
    group signaled after a delay, 8 trials at each of 0, 1, 2, 3, 4, 5, 6, 8, 12, 20
    and 50 ms:

    | How it was ended | Trials that left the command running |
    |---|---|
    | SIGTERM to the group, as today | 30 of 88, all between 1 and 6 ms |
    | SIGTERM, then SIGKILL to the pid `bwrap` reported on `--json-status-fd` | 0 of 88 |
    | SIGKILL to the group | 0 of 88 |

  - `--unshare-pid` is what it takes: without it nothing is left, and without
    `--new-session` the same trials are.
  - How it showed: `phern -p` with the scripted model ends its turn in 0.3 s, just as
    the eager hosts start. Five runs of five left both hosts up under
    `systemd --user`; the journal shows both wrappers spawned and reaped. A host
    leaves by itself after its idle time; a `bash` command would run to its end.
  - It is `ph.seams.subprocess` and `sandbox_local`, not this plan's code, and 0.10.0
    has the same two lines. A turn of a real model lasts longer than the window; a
    first step that fails at once — a missing key — does not.
  - *An earlier version of this record said SIGKILL left it too, over 3 to 100 ms.
    That run was counting a second copy of itself.*
  - **Fixed, the way the user chose: such a command is killed, not asked.** A spawn
    says whether a signal reaches its command (`SubprocessSpawnSpec.forwards_signals`,
    carried from `ConfinedArgv.forwards_signals` by `ConfinedArgv.spawn_spec`, the one
    door from a confined argv to a spawn: `ctx.shell` and both sandbox probes), and
    `terminate` sends SIGKILL at once to one whose wrapper passes none on. `bwrap`
    never passed SIGTERM to the command, so the grace was not the command's to use,
    and a stop that leaves nothing running is what a clean restart needs. An
    unconfined command and one under `sandbox-exec`, which do hear SIGTERM, are
    still asked first. Reporting the sandbox's pid on `--json-status-fd` and ending
    that was the other design; it is not built.
  - Gates: a confined command stopped 0 to 8 ms after its spawn leaves no mark, under
    a real `bwrap`, and the same test with the child asked first leaves three; a
    handle sends `kill` alone to such a child. Eight `phern -p` runs left no host.

- **A daemon restarted with the `--patch '{insert: …}'` it was started with refused
  its own roots.** The root's log holds the insert as a `profile/override`, the
  daemon's patch declared the rows again, and `phern agents send` answered "row id
  "utcp" is already declared by profile/override".
  - **Fixed: a row a start option declares that the log already holds is not declared
    again** (`ph.cordis.loader.new_declarations`, used by `ph.session_profile.rebuilt`
    where a root is mounted from its log and by `opened` where its start options are
    logged). The log's row stands as the log has it; its config is changed by id, as
    the refusal always said.
  - Gate: `-p` twice with the same insert exits 0 both times and logs one override,
    and a third with a patch by id beside it logs that one; with the skip switched
    off the second start exits with the old refusal.

**Still not verified.** Other languages' UTCP clients; a model with remote code under
a sandbox (`einops` and nomic's weights are not installed here); a hosted embeddings
API and Google's; a fan-out of real sub-agents under `rlm-indexed`. The last three
need a key or a chat model this machine does not have.

## A fifth `/simplify` pass, over "Before publishing"

Fixed:

- **`ToolEnv` is registered through the shared door** (`Registers`, `contribute_item`),
  so the endpoint row needs no `ctx.get` and a name leaves with its row. `manual`
  builds the inherited list once (`_inherited`), and de-duplicates the names.
- **`ConfinedArgv.spawn_spec`** is the one way from a confined argv to a spawn. The
  sandbox's two mount probes built their specs by hand and dropped `forwards_signals`;
  they now go through it, with `ctx.shell` and the test.
- **`BackgroundRuns.run` requires `said`.** Defaulting to nothing is how the code graph
  told an agent "finished: 34 files" over a graph that held none.
- **`ensure` calls `prefetch` once** for what `readiness` says is missing, and asks for
  each one alone only if the batch fails (one unknown name fails it). Cold, 2.9 s on
  1.16.2 and 3.2 s on 1.25.0, against 6.2 s one language at a time.
- **The code graph's summary is one function with a count** (`_index_summary(shown=)`),
  its note names 8 skipped files as the text index's does, and `said` is a `partial`.
- **The "bundled grammars need no network" text is corrected** in `_extract`, the
  skip reason, the install message, `row.py` and `pyproject.toml`.
- Tests: one recording fake process in `test_subprocess.py`; the real-sandbox test
  waits 0.5 s rather than 1.2 s and still fails three times of three with the child
  asked first.

Left as they were, with the reason:

- Nothing. The five left here at first were built at the user's word; see below.

**Built after the pass, at the user's word.**

- **The endpoint's key variable is said once.** `api_key_env` is marked
  `Annotated[str, EnvName()]` (`ph.wire`). The command's `manual` reads the mark off
  each mounted fork's config, and `config_env` returns it beside the config for the
  session's row to pass as `pass_env` (the final form; see the sixth and seventh
  passes). `ToolEnv`, `TOOL_ENV`, `Provider.pass_env` and the engine row's registration
  are gone.
- **A stopped command's whole group is killed, whether or not its leader is still
  there.** `terminate` sent SIGKILL only while its own child lived, so a member that
  ignored SIGTERM outlived a leader that heeded it. Once the child is reaped the group
  is named by the pid it was spawned with. Only on the stop path: a command that
  exited by itself is reaped and its group left alone, since `run` releases every
  child this way and a `&` job may be on purpose.
  - Gates: a shell that heeds SIGTERM over a child that traps it — the child is gone
    after `terminate`, and with the old condition the test times out; a command that
    exits leaving a `sleep` running still finds it running.
- **A fetch that fails is the last an indexing run tries** (`_extract.obtained`). A
  confined command asked GitHub once for each language it had no grammar for, a
  `sandbox/denied` apiece on every call. What is on disk still loads, and an
  unconfined run on a fresh cache still fetches.
- **`BRIEFLY` and the "[and N more]" line are `ph.text`'s** (`first_lines`), used by
  both packages.
- **The UTCP-client test indexes and stops.** The key reaching the server is its proof;
  a search through a stock client is `test_text_index_cli`'s.

## A sixth `/simplify` pass

Fixed:

- **A stop is what the caller meant, not whether the leader had exited.**
  `SubprocessHandle.finished` is set by `run` only when its drain ended — the child
  exited and its output reached the end — and every other release kills the group.
  Reading the leader's exit instead left a timed-out `sh -c 'sleep 30 & echo hi'`'s
  `sleep` running and forgotten by the journal. Gate: that run, `timeout_ms=500`; with
  the old rule the test times out waiting for the job to die.
- **`signal_group(pid, kill=, reaped=)`** names the group by the pid once the child is
  reaped, for the seam and the RLM kernel alike, and refuses a pid of 0 or less
  (`killpg(0)` is the caller's own group). The kernel's `_guest_pgid`, always equal to
  `process.pid`, is gone; the sweep's `_kill` passes nothing extra.
- **Start options are filtered against everything before them** — the host's rows
  too, which the host profile was composed with the options over — not only the log.
  `new_declarations`' docstring says a held row keeps its config.
- **The manual reads the mounted forks** (`fork.spec.resolve_config(fork.config)`, as
  `ph.seams.subagent_profiles` does), so a private copy and a live reconfigure are read;
  `resolved_config` is back inside `resolve_row`. `env_names(*configs)` de-duplicates,
  and both packages' session rows pass what every config they hand the command marks.
- **A refused grammar fetch is remembered for the process** (`_extract._fetch_refused`,
  first on the seam), so a host answers its next call without asking GitHub again, and
  only a fetch that was tried and failed sets it — a grammar on disk that will not load
  does not.
- **`first_lines(items, shown, line)`** formats only the lines it shows.
- Smaller: `_beyond` reads its documents as `opened` does; `new_declarations` returns
  a copy; `utcp_manual` takes `named` without a default.

Left as they were, with the reason:

- **`EnvName` and phern's credential picker (`apiKeyEnv`) stay two rules.** They name
  different facts: a variable a tool's command must be handed, and a key the harness
  holds. An adapter's key must not carry the mark — it never leaves the harness.
- **A start option's dropped declaration is not noted to the person.** The log's row
  stands, as the user chose; saying so at each start would be a new message.
- **Other "and N more" spellings** (attachments, the RLM harness, stabilize) keep their
  own words.
- *`test_subprocess`'s `_alive` was listed here at first; done since, at the user's
  word: it and `ph-rlm`'s `test_lifecycle` copy were the two suites left over when
  `ph.orphans.process_alive` was made public for them, and both now call it. Their
  other reading of `PermissionError` is unreachable there: every pid they ask about
  is a child the test started, under its own user.*

## A seventh `/simplify` pass

Fixed:

- **`SubprocessHandle.finished` is the handle's own fact**: true once `pump` read the
  output to the end and `wait` returned. `run` no longer sets it, so a caller that
  drives `spawn` → `pump` → `wait` itself also keeps what its command left running, and
  a caller cannot set it to skip the kill. Gate: the "seen through" test, by `run` and
  by `spawn`.
- **`ph.orphans.signal_child(process, kill=)`** — the group, or the child alone, with
  `reaped` worked out from the process — is the one call for the seam and the RLM
  kernel, whose `_kill_group` lost its `reaped` argument. A stale comment about the
  group id "captured at spawn" is gone.
- **`config_env` returns the variables to pass beside the config**, so a session row
  cannot hand one and forget the other; `env_names` reads nested models too. `EnvName`
  lives in `ph.wire`, beside the models it marks.
- **The refused-fetch flag is `_extract`'s**, process-wide beside `_configured`, and
  `obtained(language) -> bool` again; `_reset` clears it.
- **`opened` filters start options through `_beyond`**, as `rebuilt` does.
- **`ph.seams.fs.skipped_lines`** is the one spelling of a skipped file, beside
  `skip_reason`, for both index engines.
- **The stopped-job test cancels once the job has started** instead of waiting out a
  500 ms timeout.
- Smaller: `MOUNT` from `ph.keys`; docstrings that told history or repeated a
  neighbor's are cut to the rule.

Left as they were, with the reason:

- **The two `SkippedValue` models stay per package.** Each is part of its tool's output
  schema; moving one into core renames a `$defs` entry in both manuals.
- **`test_session_overrides`' third CLI run stays.** It is the only check that a patch by
  id beside a skipped declaration is logged through the real start path.

## Decided with the user after U8

10. **The adapter stays in the harness, and the command asks it for vectors over a
    socket** — so a route's key is held where the seam holds it, and an agent cannot
    reveal it to a model's context by accident.
11. **Children share their parent's index and host.** A child builds its own only when
    it is writable and would need one. A read-only fan-out, which is what `rlm-indexed`
    runs, shares one index.

## What U9 settled

- **`EmbeddingAdapter` is an optional Protocol beside `LlmAdapter`**, asked with
  `isinstance`; `ctx.llm.embed` routes a request to it and `ctx.llm.embeds` says
  whether a provider can. phern's OpenAI-compatible adapter posts `/embeddings` and
  its Google adapter `models/<model>:batchEmbedContents`. The Anthropic API has no
  embeddings call, so that adapter is not one.
- **The command does not mount an adapter row.** The plan had it mount one, with phern
  installed beside it. The session's row opens a unix socket under `$PH_RUNTIME/utcp`
  and the command's embedder sends its passages there, a batch a connection; the
  session calls `ctx.llm.embed` and sends vectors back. The command is told the socket
  in `$PH_TEXT_INDEX_EMBEDDER` and is given no variable to pass, by value or by name.
- **A confined agent needs no network for it.** The session asks the provider. Held
  under a real `bwrap` with no host allowed.
- **A failure crosses the socket as the adapter's own code and sentence**, under this
  package's prefix: `TEXT_INDEX_MISSING_CREDENTIAL` names the variable, as a model
  call does.
- **A route whose adapter cannot embed is refused at the row's mount**, when the route
  is there to ask. One registered later is met at the first call instead.
- **Usage is counted on the row and shown in `phern doctor`**, texts, requests and the
  tokens the provider reports. It was not in a session's log or its budget: an
  embedding is not a turn's request, and the log had no record for one. Built on
  2026-11-11 as `llm/embedded`; see "Packed in the embedder's unit, and four gaps
  closed".
- **Not available from a shell.** The command says so and names the two rows that are.
- **Run against the user's llama.cpp at `127.0.0.1:9931`**, serving
  `bge-large-en-v1.5` with `--embeddings --pooling mean`. Both embedders indexed pH's
  `docs/seams` — 30 documents, 233 passages, 1024-wide vectors — in under five
  seconds, the adapter path from inside a read-only `bwrap` with no host allowed. Of
  eight questions with a known document, seven had it first and all eight had it in
  the first three; each search took 0.3 s. A second run embedded nothing. The command
  on the adapter path was given the socket and its config, and no variable.
- **What the earlier attempts on that server found.** Without `--embeddings` it
  answers 501, without a pooling type 400, and with a wrong key 401: each reached the
  agent as a coded failure in the server's own words. A chat model served with
  `--pooling last` returned vectors that depended only on a text's first word, so
  every search returned the same documents; asked directly, the server gave the same
  vectors. An index cannot tell that from a working embedder.
- **That model reads 512 tokens.** A longer input is refused with a 500, which fails
  the indexing run that sent it. None of this corpus's passages reached it at the
  default `max_chars` of 1,200 (254 tokens a passage on average); a corpus of dense
  text could, and the setting to lower is `max_chars`.

## Passages are cut to what the embedder reads

Asked for by the user after the live runs, and not a step of the original plan.

- **The failure was silent.** Passages were cut by characters and the embedder's own
  length was nobody's business. The default local model reads 256 tokens and truncates
  without a word: on `docs/seams` 144 of 233 passages were longer, and an eleventh of
  the text was never embedded. A server refused instead, which ended the run.
- **`max_tokens` on each embedder row, detected when unset**: the local model's own
  length, a llama.cpp's `/props` for the endpoint, the route's window for the adapter
  (`_embed.TokenBounded`). Passages are packed to what the limit leaves for their text,
  at three bytes a token, when that is under `max_chars`; measured — exactly, by the
  local model's tokenizer; by the bytes elsewhere — and a document denser than the
  estimate is packed again, smaller.
  Re-packed, not split in place, so line spans stay true.
- **A refusal is read, not only survived.** A passage refused as too long is packed
  smaller and sent again. Where the refusal names numbers, the limit it names becomes
  the run's limit and the token count it names becomes the corpus's density. A first
  version learned the limit from the size a document happened to be taken at; on the
  live server that shrank every later document to fit the densest one, and was
  replaced.
- **What a llama.cpp reports is not what it takes.** `/props` gives the window per
  slot — `-c 8192` with four slots reports 2,048 — and an embedding is refused past
  the batch size, 512 by default, which `/props` does not give. Run against the user's
  server with `max_chars` raised to 20,000: one refusal, the limit learned as 512, the
  other 29 documents packed to fit and none refused.
- **Run with the real local model**: 381 passages packed to 768 characters, the
  longest 244 tokens against a limit of 256, none over.
- **A passage is packed to what the limit leaves for its text**, after its prefix
  and the model's start and end markers (`TextIndexSeam.fit`, measured as an empty
  passage). Packed to the limit's worth of text, every full passage measured a few
  tokens over it once measuring used the corpus's density, and the live server run
  reported 14 of 30 documents as needing smaller passages. After: 4 — the one the
  server refused, and three measured denser than the density that refusal set —
  and with the real local model 762 characters, 388 passages, none over 256.
- **Run again with the user's server at an 8,192-token window** — `nomic-embed-text-v1.5`,
  four slots, `/props` reporting 8,192 a slot and the batch raised to match. Asked
  directly, it embedded a 4,463-token input and refused a 9,090-token one, naming a
  batch size of 8,192. Indexing `docs/seams` from inside a read-only `bwrap`, both
  embedders, each at three passage sizes:

  | `max_chars` | passages (longest) | limit used | indexing | first / top three |
  |---|---|---|---|---|
  | 1,200, the default | 233 (1,200 chars) | none needed | 2.8–3.0 s | 7/8 / 7/8 |
  | 20,000 | 32 (19,968 chars) | none needed | 2.9–3.2 s | 7/8 / 8/8 |
  | 100,000 | 31 (24,213 chars) | 8,192: the route's window, and `/props` | 3.0–3.1 s | 7/8 / 8/8 |

  The two embedders gave the same passages and the same ranking in every row. At
  100,000 the limit was read before anything was sent — no refusal, no document cut
  smaller — and passages were capped at 24,552 characters, so only `subagents.md`
  (40 KB) was split. The one first-place miss in every row is "approve a tool call
  before it runs", which ranks `tools.md` or `user_questions.md` over `approval.md`.
- **A larger window is not a reason to raise `max_chars`.** A search returns each
  hit's whole passage, so whole-document passages at `k=3` put up to 70,000
  characters in front of the model a search. What the window buys is that a long
  paragraph is embedded whole rather than refused or truncated; the default stays.
- **The result says it**: the size packed to, the limit that set it and where that
  came from, and each document that needed smaller passages still. A document that
  does not fit at 64 characters is skipped with the embedder's words.
- **The detected limit is asked for late**, only by a run with something to embed: of
  a local model it means loading it. So the digest that says "unchanged" covers a
  limit a profile *set* and not one that was detected.
- **Not done here, done on 2026-11-11**: a question longer than the limit was not cut,
  `phern doctor` did not show the limit, and a background run's completion note did
  not mention what the result of a waited-for run does. And passages are no longer
  sized in characters at all: what this section says of `fit`, of 762 characters and
  of documents "packed again" describes the first version. See "Packed in the
  embedder's unit, and four gaps closed".

## A second `/simplify` pass

Over everything after U4, with four reviews (reuse, simplification, efficiency,
altitude). What changed:

- **`ph.framing`** is the daemon's JSONL framing moved down into core, with a
  blocking `ask` and a one-request `answer_one`. The tool host, its client and the
  session's embedding socket use it in place of three hand-rolled copies that
  lacked the second length check.
- **`BackgroundRuns.run`** is the one door for a long piece of work, host or not.
  It writes the run's start, end and failure itself; the work only says `begin`
  and `tick` (`RunProgress`). The code graph had reported a failure as "0 of N";
  a run that ended just before its caller gave up was announced to nobody; and a
  person's call beside a warm one was counted as unasked. All three are gone with
  the copies.
- **A command runs in the tree it is about.** `$PH_TOOL_ROOT` is removed; the note
  that paths belong to another tree is keyed on where the command ran.
- **`UtcpSeam.run_command`** runs a manual's own `install` or `status` with what its
  tools are given; both rows' copies are gone. `ManualState.said` replaces two
  copies of the doctor line.
- **`preload` runs the command's `install`**, which now exits failing when the model
  will not load. The command no longer reads `preload`, so a tool call is not told
  to skip it and a person's config file does not load the model on every call.
- **The shell seam gives every confined command a `TMPDIR`** in the agent's
  scratch; `ph-utcp`'s copy of that rule is gone.
- **"Input too long" is one list in `ph.llm.adapter`** (`input_too_long`), and one
  code, `CONTEXT_WINDOW_EXCEEDED`. A missing key is `MISSING_CREDENTIAL` from every
  embedder. The session passes the adapter's code through unprefixed.
- **The session socket sends vectors as base64 float32**, a quarter the size, and
  the harness does no per-number work for them beyond the adapter's own check.
- **Hosts are warmed once a tree**, and the step that warms one does not wait for
  it. The owner is resolved once a call; the socket directory once a mount; the
  progress file is read only when its stat changes.
- **A measured passage uses the corpus's density** where the embedder has no
  tokenizer, so a refusal that said how dense the text is shortens the next cut.
- Smaller: one batching helper for both remote embedders, the limit cached once,
  the text-index engine registers its tools at mount, `vectors_of` in core, shared
  test builders.

Left as they were, with the reason:

- A call answered by a host still starts an interpreter (about 0.25 s). Sending it
  to the host from `ph-utcp` would bypass the manual's own command, which is what
  another harness runs.
- Embedding stays one request a document; batching across documents would have to
  reckon with a document re-packed after a refusal.
- `text_search` loads the index and embeds the query one after the other: run side
  by side, a failure arrived as an exception group in place of the tool's code.
- The engines' own tool traits stay beside the manual's: the engine rows are public
  and can be mounted in a session.
- The owner rule stays in `ph-utcp` rather than the workspace seam, which would
  have to keep each workspace's base.
- A confined pH process still sees the host's runtime directory read-only.
- The command's log stays under `$PH_CACHE` with its grant: moved into the state
  directory it landed inside the tree in a layout where the scratch is.
- A host's `preload` still loads the model twice, once in `install`.

## A third `/simplify` pass

Over the whole diff, with the same four reviews, after the token-limit work.
What changed:

- **The caller's cancellation is `RunProgress`'s.** `BackgroundRuns.run` takes
  `raise_if_canceled` and asks it at every `tick` where the run is the caller's
  alone, with no host. Both engines' `detached` flag and its `if` are gone, so a
  third package cannot forget it.
- **A document is embedded or given a reason, not both.** `_embedded` returns a
  cut or the reason it could not make one. A document refused even at the smallest
  passage had been reported as both skipped and "needed smaller passages".
- **The digest is compared before `triage`.** The same text gets the same answer,
  so an unchanged document is no longer classified only for the result to be
  thrown away (about 110 µs a document, every run with no version control).
- **The seam's limit is one public field, `known`.** `learn` takes a `Limit`, and
  `fit` measures the empty passage itself instead of caching it by density.
- **One module reads the servers' wording.** `too_long_facts` sits in
  `ph.llm.adapter` beside `input_too_long`, and the OpenAI `/embeddings` request
  and answer (`openai_embeddings_body`, `openai_vectors`) are shared by the
  adapter and `EndpointEmbedder`. The two remote embedders share one base for
  estimating, batching, the setting-first limit and the name.
- **Shared doors in core:**
  - `ph.framing.listening` for the daemon launcher and a tool host;
  - `ph.resources.run_until_signaled` for `serve`, so a host stops on a signal
    as every host does;
  - `ph.toolcli.failure_envelope` and `value_envelope` for the session's
    embedding socket;
  - `ph.paths.file_stamp` for ph-utcp's watcher and the text index's store.
- **`confined_env` in the workspace seam.** It gives a confined process a
  `TMPDIR` in its scratch. `ctx.shell` and the code kernel (`ph-rlm`) both call
  it, because a kernel confined in a shared workspace had no writable temporary
  directory either.
- **A read-only `$PH_RUNTIME` is logged at debug.** Inside a sandbox it is the
  expected state, and it had been a warning on every call. The `subprocess`
  diagnostic still says the journal is unavailable.
- Smaller:
  - an earlier host's report is dropped as it is read, instead of being filtered
    in two places;
  - `UtcpSeam.state_of` is gone;
  - `ph.testing.new_agent` replaces copies in three packages' tests.

Left as they were, with the reason:

- **Passages are sized in bytes and cut in characters.** `fit` turns the limit
  into bytes at the corpus's density, and the chunker counts characters, so
  non-ASCII text is over on its first cut. The measure and the refusal loops
  catch it. Packing in the embedder's own unit needs the chunker to take a size
  function, which changes how every passage is cut. *Done since, at the user's
  word: the next section. The density table below is of the version it replaced.*
- **A counting embedder still packs at the careful density, and measured, that is
  right.** The review guessed prose at 4 bytes a token, so MiniLM's 762-character
  passages looked like 30% more vectors than needed. Measured on 2026-11-11 with
  MiniLM (256 tokens) over `docs/seams` and the 8 questions:
  - the corpus is 3.66 bytes a token by MiniLM's tokenizer, 3.01 to 3.95 per
    document, so 3.0 is close to the truth;
  - the density was set before each run:

    | Density | Passages | Packed to | Cut again | Mean tokens a passage | Time | Top-1 / Top-3 |
    |---|---|---|---|---|---|---|
    | 3.0 (today) | 388 | 762 chars | 3 | 162 | 6.9 s | 7/8 / 7/8 |
    | 3.5 | 358 | 889 | 14 | 176 | 6.5 s | 7/8 / 8/8 |
    | 4.0 | 374 | 1,016 | 26 | 167 | 6.6 s | 7/8 / 8/8 |
    | 4.3 | 371 | 1,092 | 28 | 169 | 7.8 s | 7/8 / 8/8 |

  - **Past the real density it backfires.** Most documents come out over on
    the first cut and are cut again, and a second cut shrinks the whole document
    to fit its densest passage, so 4.0 ends with more passages than 3.5.
  - **The best case saves 30 of 388 passages and no time.** The model's cost
    follows tokens, and the total is the same.
  - **Ranking is the same within noise.** The one top-3 difference is the
    approval question.
  - **More notes for the model.** Every document cut again is named in the
    result as needing smaller passages: 14 at 3.5 against 3 today.
  - It bears only on a local model whose window is below `max_chars`. The
    endpoint and session embedders estimate rather than count, and at nomic's
    8,192 tokens `max_chars` binds first.
- **Discovering a manual costs about 0.3 s at mount, one row after another.**
  Overlapping them needs discovery moved out of mount, since tools must be
  registered before the first step, or a cache keyed on the installed command.
- **`vectors_of` checks every number on the daemon's loop**, about 3.5 ms for 32
  vectors. Carrying float32 arrays in `Embeddings` would change a core type for one
  consumer.
- **Where the limit comes from stays as it is.** The route's window and
  llama.cpp's `n_ctx` answer a different question than the embedding limit, and
  making the embedding route own that fact is a profile design.
- **The learned limit is not persisted.** Learning only goes downward, so a server
  reconfigured upward, as this one was from 512 to 8,192, would stay at the stale
  number.
- **Refusal numbers are still read from the error's text, not carried as fields.**
  The reading itself moved to core.
- **Some per-package code stays per package:**
  - each tool's "still indexing" model and its sentences;
  - `Scope.on_mount`, since `IndexScope` is the fs seam's and knows nothing of
    hosts;
  - the remote configs' fields, whose docstrings differ by row;
  - the two rows' manuals, which share only four arguments.
- **Vector packing stays in `row.py`.** The session's process must never import
  `_embed`, and a test holds it to that.
- **Two test helpers stay copied in both packages' tests:** the runtime-and-cache
  fixture and the CLI `_call` check. In `ph.testing` they would be core source
  calling `subprocess.run` and `tempfile.mkdtemp`, which core's resource gate
  forbids.
- **A "still going" answer still asks the runs for the current progress.** The
  run's own progress would report 0/0 while it waits behind another request's run.

## Packed in the embedder's unit, and four gaps closed

Asked for by the user on 2026-11-11: the one skipped item worth doing if a corpus
is not English, and the four small things this plan had listed as not done.

**Passages are packed by what the embedder counts.**

- **Two bounds, in two units.** `max_chars` still says how big a passage should
  be. Beside it the packer takes a `Weight` (`_chunk`): what a text weighs, and
  the most a chunk may. The text index gives it the embedder's tokens — the local
  model's tokenizer, the text's own tokens with no prefix or markers; bytes over
  the corpus's density for an endpoint or a route (`TokenBounded.weigh`) — and
  the limit less what the embedder puts around a passage (`TextIndexSeam.room`).
  `fit`, its conversion to characters and `MIN_PASSAGE_CHARS` are gone; the floor
  is `MIN_PASSAGE_TOKENS`, 16.
- **Each passage for itself.** Paragraphs and lines are weighed a batch at a
  time and packed while both bounds hold, so a table in a page of prose is cut
  small and the prose is not, and a script of three bytes a character fits at
  the first cut. Before, a dense document came out over, was measured, and all
  of it was packed again to fit its densest passage.
- **The check stays.** A packing adds the weights of a passage's parts with a
  joint between them; an embedder counts the whole. For bytes that sum is never
  under. For a tokenizer it is exact where blank space is not a token, and on the
  careful side by one a joint. So what is sent is still measured, and a document
  over the limit is packed again with less room: a guard that does not fire for
  the models tried, and is tested with a stub whose blank lines cost tokens no
  paragraph weighed.
- **A refusal packs to what it named.** With both numbers read, the texts are
  weighed again at the density the refusal stated and packed to the limit it
  named, exactly. Only where that changes nothing — the refusal named no number,
  or by this seam's weighing the texts fit already — is the room aimed in
  proportion, or halved. Aimed at what the heaviest text weighed and not at the
  room: under an 8,192-token window with passages ending at 300 tokens, half the
  room is still thirteen passages.
- **A line cut inside itself stays in pieces.** Pieces of one line are never put
  back in a chunk with a blank line between them, which the unequal pieces a
  weight makes would otherwise allow.
- **The result says tokens.** `passage_chars` is gone from `text_index`'s value;
  `passage_tokens` and `limit` are set when the limit ended a passage sooner than
  `max_chars` would have, which the packer reports (`Weight.bound`).
  `cut_smaller` now names only documents packed a second time.
- **Measured, MiniLM (256 tokens) over `docs/seams`, the same 8 questions**:

  | | Passages | Mean tokens a passage | Over the limit | Packed again | Time | Top-1 / Top-3 |
  |---|---|---|---|---|---|---|
  | characters, density 3.0 (before) | 388 | 162 | 0 | 3 | 6.9 s | 7/8 / 7/8 |
  | the model's tokens (now) | 316 | 199 | 0 | 0 | 6.3 s | 7/8 / 7/8 |

  Packing and checking all 30 documents takes 0.12 s of the 6.3. The longest
  passage is 254 tokens as sent.
- **Measured, the user's nomic server at 8,192 tokens**, both embedders, from
  inside a read-only `bwrap`. The two gave the same passages and ranking in
  every row, and no document was packed again, refused or skipped:

  | `max_chars` | `max_tokens` | Passages (longest) | Indexing | Top-1 / Top-3 |
  |---|---|---|---|---|
  | 1,200 | unset | 233 (1,200 chars) | 2.8–3.2 s | 7/8 / 7/8 |
  | 100,000 | unset | 31 (24,213 chars) | 2.9–3.1 s | 7/8 / 8/8 |
  | 1,200 | 128 | 802 (360 chars) | 4.1–4.5 s | 5/8 / 8/8 |
  | 100,000 | 512 | 182 (1,498 chars) | 2.6–3.0 s | 7/8 / 7/8 |

  The first two rows are what the version before gave.
- **An index built before keeps its passages.** The digest that says "unchanged"
  covers the text and the settings, not how the packer works, so a document is
  cut the new way when it next changes. Its old passages fit; they are only
  smaller than they need be.

**A question is cut to fit** (`engine._asked`, `_chunk.head_of`).

- One that measures over the limit, with the *question's* prefix, is searched by
  its head: as much of its start as weighs what the limit leaves, back to the
  end of a word. `text_search`'s value carries `query_chars` and its text says
  how many characters were searched.
- One refused all the same is cut and asked again, by the rule a passage is, and
  what the refusal taught is kept for the next. At 16 tokens the refusal is the
  search's failure, in the embedder's words.
- **Its head, not its paragraphs.** The first version packed the question as a
  document and took the first chunk; on the live server a 16,887-character page
  asked under a 128-token limit was searched by its 57-character title.
- A search now asks the embedder what it reads, as an index run does: of a
  server, one request a process, which a host makes once.
- A result repeats at most 200 characters of the question back.

**`phern doctor` says the limit.**

- The session's section says a setting, or for `text-index-adapter` the route's
  window (`Provider.limit`). A model's own length or a server's is the command's
  to ask — of a local model that means loading it — so for those it points at
  `/text-index status`.
- `/text-index status` says what the last run to embed found. The index's
  sidecar keeps that phrase (`TextIndex.limit`), for people only: it is never
  read back as the limit, which is why learning it was not persisted before.
- `Limit` moved to `_config`, which a session's process may import.

**A background run's note says what its result would have.**

- `Progress.note`, made from the run's value by the `said` a tool passes to
  `BackgroundRuns.run`, and written with the run's last report — so only a run
  that ended has one. `ph-utcp` adds it to the note that tells the agent the run
  ended.
- The text index says the lines of its result without the list of documents,
  and with what it skipped counted past eight. The code graph passes none yet.

**What a route spends on embeddings is in the log.**

- `LlmRuntime.embed(request, session=…)` records each request in the session it
  is for: `llm/embedded`, with the route, how many texts and the provider's
  `usage`; and `llm/embedding-failed`, with the code and the reason. Both are
  ignorable and audit-only. Asked for nobody, it records nothing.
- `llm/embedded` is in `SPENDING_TYPES`, so a sub-agent's account includes it. A
  goal's fold counts it as a source of its own, `embedding`, which a budget
  charges only where its `token_sources` names it.
- **Whose log.** `ph-utcp` sets `$PH_TOOL_AGENT` beside `$PH_TOOL_STATE`, naming
  the agent the state belongs to; the command sends it with each request and
  the session's row finds that agent's log. A child that shares its parent's
  index therefore spends on its parent's account. The name comes from inside a
  sandbox, so it only chooses a log among this session's agents; a request that
  names none is embedded and counted on the row alone.
- One record a request: 40 for the 242 texts of the live run, 66,774 tokens,
  the same total `phern doctor` shows.
- Only the adapter embedder spends a route. The endpoint is the command's own
  request and the local model spends nothing, so neither writes a record.

**What breaks**: `text_index`'s `passage_chars` is `passage_tokens`;
`MIN_PASSAGE_CHARS` and `TextIndexSeam.fit` are gone; an embedder that is
`TokenBounded` needs `weigh`, and its `measure` takes `query`; `Limit` is
imported from `_config`; `TokenSource` has a fourth value, `embedding`.

## A fourth `/simplify` pass

Over the diff of the section above, with the same four reviews. What changed:

- **A long line is cut in linear time.** `_lightened` weighed all that was left
  of a line after every cut, a tokenizer's pass over the line for each piece.
  Each cut is now aimed by the stretch just ahead of it, twice the last piece,
  and a stretch that fits is taken whole. One 100,000-character line under
  MiniLM's tokenizer, with `max_chars` 100,000 and the heaviest passage 254
  tokens every time:

  | Line | Before | After |
  |---|---|---|
  | CJK | 19.5 s, 394 passages | 0.39 s, 394 passages |
  | base64 | 13.5 s, 256 passages | 0.47 s, 246 passages |

  At the default `max_chars` it was bounded already (0.44 s). A paragraph of
  one line is weighed once, with the document's paragraphs, and not again as a
  line and as a piece. `_fitting` is the one cut, for `_lightened` and `head_of`.
- **A run's note is made from its value.** `BackgroundRuns.run` takes `said`, and
  the end is written with what it says; `RunProgress.note` and the filter that
  kept it off a failed run are gone.
- **The goal fold's sources are keyed by `SPENDING_TYPES`**, and refused at import
  when they are not. Adding `llm/embedded` took two edits, and a record put in
  one set alone would have reached children's accounts and no budget. The
  table's key column, `usage` in every row, is gone.
- **A failed embedding has the code a failed stream has** (`_failure_of`): an
  error that is not an `LlmError` is `UNKNOWN` with its message, where it had
  been the exception's class name.
- **No empty-string absences:** `TextIndex.limit` and `SessionEmbedder.agent` are
  `None` when nothing is known.
- Smaller:
  - the echoed question is `ph.text.one_line`;
  - `_embed` no longer re-exports `Limit`;
  - `query_chars` is set when the question was cut, not when it measured over;
  - `_refused`'s test is one condition;
  - `_embedded` reports the bound of the packing it sent;
  - `estimate_tokens` is `weigh_by_bytes` rounded.
- `DESIGN.md` said a child's log holds what its tools embedded. That holds when
  the index is the child's own; a child sharing its parent's index embeds on
  the parent's account, as the README says.
- **Live, after the pass**, against the nomic server at 8,192 tokens with both
  embedders: the same as before it in every row of the table above — 233, 31,
  802 and 182 passages; the same rankings; the long question searched by 343 and
  1,495 characters, with `sandbox.md` first; nothing packed again, refused or
  skipped. The adapter's first run still logged 40 `llm/embedded` records, 242
  texts and 66,774 tokens, as `phern doctor` said.

Left as they were, with the reason:

- **Pieces of one cut line are kept apart by a flush rule** (`_of_one_line`),
  not by each block carrying the separator it had in the source. That deeper
  fix, with a line cut by both bounds in one pass, lets a line's pieces share a
  passage: a change to how passages are packed, not a cleanup.
- **`Provider.max_tokens` still copies the config's.** `Provider` is how another
  package registers an embedder with a config of its own; typing the config to
  a base in this package would make every such config inherit from it.
- **A search asks the limit before it opens the index, not beside it.** In a
  task group a refusal would come out as an `ExceptionGroup`, and the cost is
  one-call mode against an endpoint with no `max_tokens` set: one `/props`
  request a search.
- **The agent-to-session lookup is open-coded once more.** The same few lines
  are in the sandbox, workspace and subagents seams and `ph-utcp`; a public
  helper in core would collect all of them.
- **`$PH_TOOL_AGENT` names whose state it is, not who called.** A child sharing
  its parent's index charges even its own searches to the parent. Naming the
  caller would mean passing it with each call.
- **`Progress.note` is `""` when there is nothing to add.** It is text, and an
  empty one says nothing; it is not standing in for a missing value.

## What decision 11 changed

- **Who owns a tree is decided by what the child was granted, not only by its root**
  (`UtcpSeam.owner_of`). A child shares its parent's index and host when it works in
  the parent's checkout, or when its workspace grants only reading the project:
  `readonly-scratch` and the two ephemeral kinds (`ph.seams.workspace.project_access`).
  A `worktree` or an `overlay` is a tree written for keeps, and gets its own.
- **A sharing child's command is told which tree it works on** (`$PH_TOOL_ROOT`). A
  `readonly-scratch` child's own directory is empty: without this it would have indexed
  nothing into its parent's index and swept every document out as gone. Its results
  carry a line saying which tree their paths belong to.
- **An ephemeral worktree shares too**, though its holder can write it: those writes
  reach nobody, so what it was granted of the project is reading. A child that edits a
  throwaway checkout and asks the index is answered from its parent's tree.

## What U8 settled

- **One row, one request.** `text-index-endpoint` posts `{model, input}` to
  `<base_url>/embeddings` with the standard library, a batch at a time, and normalizes
  what comes back. No dependency was added, and nothing of the local model is imported.
- **The key travels by name, through `ph-utcp`.** The plan had the template's
  `env_vars` carry a `${VAR}` reference. A host is started by `ph-utcp` and not by the
  UTCP client, so that reference would not have been resolved for it. `Manual.pass_env`
  names variables the row reads from the harness's environment each time it starts a
  command, a call or a host, and holds nowhere.
- **Three named failures**: `TEXT_INDEX_CREDENTIAL_MISSING`, which sends nothing;
  `TEXT_INDEX_ENDPOINT_UNREACHABLE`, which names `sandbox-allow`; and
  `TEXT_INDEX_ENDPOINT_FAILED`, with the server's own words.
- **The row does not ask for a network allowance.** Allowances are the deployment's
  statement (`sandbox-allow`), and no row adds to them. The README says which host to
  name. Checked under a real `bwrap`: with `127.0.0.1:<port>` allowed a confined
  command indexes through the egress proxy, and without it the call fails with a code
  and the server is never reached.
- **The identity is the model, the prefixes and an asked-for width, and not the
  address**, so moving a server keeps the index.
- **No fallback embedder.** One embedder a session; an endpoint that cannot be reached
  is the answer. A second embedder would be a second index answering under the first
  one's name.

## What U5 settled

The worker is built, and it is called a **host**. It differs from "Before U5" in four
places, each because the code allowed something simpler.

- **No new spawn door.** A host is started for an agent, after that agent holds its
  workspace, so `ctx.shell.run_argv` already places and confines it; a job
  (`ctx.jobs`) owns it, and waiting for the command to end is what a job body does.
- **It does not start at mount.** At mount there is no agent, so no tree and no
  scratch to keep an index in. `ph-utcp` listens on `agent/pre-step` and starts a root
  agent's hosts before its first step, with the index tool to run. That is the first
  moment "the root session" has a workspace; it is also every later step, so a host
  that left when idle is started again by the next one.
- **One host a tree, not one an agent.** A child that shares its parent's workspace
  root uses its parent's host and its parent's state (`UtcpSeam.owner_of`). Decision 9
  names the agent, and for a root agent and for a child in its own worktree that is
  what happens. For a child in its parent's tree the literal reading was the same
  documents embedded again and a 400 MB model held for every child of a fan-out, which
  is what `rlm-indexed` does. The state path is still `<session>/<agent>/<tool>`; the
  agent is the tree's owner.
- **The footer reading is polled**, twice a second while a host runs, and kept in
  memory. One small file, and `ph.path_watch` would have needed the same fallback.

What it is:

- **`ph.toolcli serve`**: mounts the rows once and answers `call`s on a unix socket, a
  line of JSON each way. `call` uses the host in `$PH_TOOL_HOST` when one is there and
  runs alone when not, and when the host says the call is for another tree or index.
  The host leaves when idle and removes its socket.
- **Background work** (`BackgroundRuns`): in a host an index tool waits `wait_s` for
  its run — 5 s for the text index, 15 s for the code graph — and then answers
  `state: indexing` with how far it has got. The same request asked again joins the
  run. Searches and graph queries carry "Not finished indexing: 120 of 539 documents"
  while one is going. A command run for one call still runs to the end.
- **`progress.json`** in the state directory. Each report carries which run it is and
  whether its end is owed to anybody: a call was answered "still going", or the host
  began the run unasked. `ph-utcp` tells the agent on that, with `inject`, so a run
  whose own caller got its counts back is not announced twice.
- **The text index keeps a digest per document and saves as it goes** (`save_every_s`,
  30 s, and on the way out). A tree with no git or jj no longer embeds every document
  on every call.
- **`index.on_mount`** on both rows, default true.
- **Sockets under `$PH_RUNTIME/utcp`**, granted to the host and to each call.

Measured with MiniLM under a read-only `bwrap`: 30 documents indexed from a cold host
in 9.5 s, the footer moving from 0/30 to 28/30, the note delivered, and each search
after it 0.3 s — against 6 s for a command that loads the model itself.

Left open by this step:

- **Killing a host loses up to `save_every_s` of embedding.** The gate holds "embeds
  nothing twice" up to the last save.
- **A stale `progress.json` is told apart by the lock and by when its run began**, not
  removed. A session that never starts a host again leaves it.
- **A host started for a tree is not told when the row's config changes.** A changed
  config applies to the next host; the running one keeps what it was started with
  until it leaves or the row remounts.
- **Other harnesses get none of this.** `serve` and `$PH_TOOL_HOST` are in the manual's
  inherited variables, so another harness could start a host; nothing in UTCP starts
  one for it. *Both READMEs now say how a harness starts one, and it was run from a
  shell: see "Before publishing".*

## Before U5

*Kept as written before U5, for what it weighed. "What U5 settled" above says which of
these held.*

What building U1 to U4 found that the worker design below gets wrong or leaves open.
Checked against the code; U5 starts from these.

- **U6 comes first** (decided). The worker runs the text index's tools, which are not a
  command until U6, and U4a decides where both packages keep their state.
- **A worker cannot be started from inside a tool call.** `bwrap` runs every confined
  command with `--unshare-pid --die-with-parent`, so nothing a command starts outlives
  it. The session's row starts the worker, in the harness. A child in its own worktree
  gets its worker lazily, which needs a `ph-utcp` hook that runs in the harness, for
  the calling agent, before a manual's command.
- **A long-lived confined process needs a door of its own.** `run_argv` waits for its
  command to finish, and `ctx.shell` confines only a command run for an agent that holds
  a workspace; at mount no agent exists. U5 adds a spawn door with `run_argv`'s
  placement, environment and grants, taking an explicit policy — the root as the
  workspace, the agent's scratch, and the manual's grants — and returning a handle owned
  by the row's scope. The subprocess seam journals it like any child.
- **A worker lives and dies with its mount.** Every spawn is in the orphan journal,
  which is swept at each pH start, and a child dies with the scope that owns it. So a
  restarted daemon does not pick up a live worker; the remounted row starts a new one.
  That is cheap because the indexes are incremental — the code graph's sha256 and the
  text index's per-document digest and interval saves — and because U4a keeps each
  index where the same agent finds it again.
- **One worker per root, and per child that asks; never shared.** Sharing one across
  roots or processes is the cross-root coordination the "roots are self-contained" rule
  forbids. A daemon with N roots on one workspace runs N workers, each text-index worker
  holding a model of about 400 MB.
- **The worker needs the agent it indexes for.** U4a puts state in an agent's scratch,
  and at mount there is a root session but no agent yet. U5 settles which ids the root's
  worker uses: the root agent's, known from the session's own log when it is resumed.
- **The footer reading is cached.** A status field is read on every redraw and must not
  do I/O. The row watches the progress file (`ph.path_watch`) and keeps the last reading
  in memory.
- **The completion notice has a recipient.** The root agent for a root's worker, the
  child for a child's; delivered with `inject`, so it waits for the agent's next step,
  starts no turn, and is in that agent's log.
- **The tools' outputs change shape.** `code_index` and `text_index` gain a state
  (`indexing`, `done`, `failed`) and progress counts; a search gains its "not finished
  indexing" warning. Only a worker answers early: a direct `ph-code-graph call` still
  runs to completion, so trying a query by hand keeps working.
- **Where the socket goes.** The caller and the worker are in two sandboxes, so the
  socket is on a path both reach and short enough for a socket address (108 bytes): a
  directory under `$PH_RUNTIME`, granted to the manual.

A design that follows from them: `ph.toolcli` gains `serve`, a process that mounts the
profile once and answers calls on that socket; `call` uses a running host when there is
one and runs directly otherwise, and never starts one. The engine row indexes in the
background only when it is mounted in a host. Progress goes to a file beside the index,
which the session's row watches for the footer and the completion notice.

## What U4a and U6 settled

- **The index location is the caller's to say, and nothing else's.** `--state` and
  `$PH_TOOL_STATE`, read by a row with `ph.toolcli.state_dir()`. Both rows' `path`
  setting is removed: it named a place outside the scratch, which a session could not
  grant without reopening the question decision 9 closed.
- **A grant that overlaps the workspace is refused**, whatever the mode
  (`ph.seams.sandbox.grants_apart`): inside it, the root itself, or a directory that
  contains it. `ctx.shell` drops it per call, against the calling agent's root, and
  `ph-utcp` says so in `phern doctor` when a manual is added.
- **A session started where the workspace contains `$PH_CACHE` loses its grants.** The
  home directory is the case. The index is unaffected, being in the scratch; the code
  graph's grammar cache and both commands' logs are refused. Under `workspace-write`
  nothing is lost, since the root is writable. Under `read-only` the grammars cannot be
  materialized and `code_index` fails. Not handled; the fix would be a grammar cache
  that falls back to the state directory.
- **A child that shares its parent's workspace has its own index**, because the scratch
  is per agent. For the code graph that is about a second per package. For the text
  index it is every document embedded again, per child. The user's decision 9 names the
  agent, so this is as decided; a profile that fans out children over one tree
  (`rlm-indexed`) pays it until something lets a child read its parent's index.
- **`/code-graph status` and `/text-index status` answer for the agent that typed
  them.** `install_or_status` now hands its body the invocation, and the row passes the
  asking agent's state directory to the command.
- **A `2>/dev/null` on a template does nothing.** The UTCP client wraps each command as
  `$(<command> 2>&1)` and the later redirection wins, so a model library's progress bar
  was in front of the answer. The command now closes its own stderr for the length of a
  `call` or a `manual` (`--verbose` keeps it), and the templates carry no redirection.
- **A confined command has no temporary directory** under a read-only sandbox when the
  workspace is the person's own checkout: `/tmp` is not writable and that kind of
  workspace gets no redirection. `ph-utcp` sets `TMPDIR` to the calling agent's scratch.
- **A downloaded model is loaded without asking the hub.** `sentence-transformers`
  checks the repository on every load, and a confined command's network is the
  deployment's allowlist, which answered `403`. A model that is not downloaded is still
  fetched by the first call where the network allows; where it does not, the call fails
  with `TEXT_INDEX_MODEL_MISSING` and names `/text-index install`, which runs outside
  the sandbox.
- **A tool's command does not sweep the orphan journal.** Inside a sandbox it is in
  another PID namespace, where every process the journal names looks dead. And
  `host_journal` now treats a `$PH_RUNTIME` it cannot write as no journal, which is what
  a confined command sees: one warning a call, in a log that is now rotated at 1 MiB.
- **Which embedder a command mounts is chosen by naming its row** (`ToolCli.providers`):
  the one a config names, or the first. `config_env` therefore names every row it is
  given, a default one as `{}`. This is the door E2 and E3 use.
- **The session's embedder row is a claim, not a model.** `text-index` publishes
  `ctx.text_index`; an embedder's row claims it with which row of which command makes
  the vectors, and that row's config. The manual is registered by the claim, on the
  claiming row's scope, so no embedder still means no tools and no skill.
- **`preload` is asked of the command, once, at the session row's mount.** Tool calls
  are never told to preload.
- **The library logged every omitted optional argument at ERROR.** Demoted to DEBUG
  inside a pH call.
- **Verified with the real model**, MiniLM, under a read-only `bwrap`: indexed into the
  scratch, searched, and `status` counted it. The download was 20 s and a search 6 s.

## What U1 to U4 settled

- **utcp 1.2.0 and utcp-cli 1.1.5 are on PyPI**, and are now test dependencies of the
  workspace and runtime dependencies of `ph-utcp`.
- **The headless profile is five rows**: the registry, the local subprocess row,
  `fs-local`, `commands` and `diagnostics`. No model, session or agent row is needed to
  run a tool.
- **A granted path is writable in a read-only session.** The plan expected indexing to
  be refused there. The sandbox reads a policy's extra paths under either mode, so a
  confined agent that may not write its tree still builds an index under `$PH_CACHE`.
  A test holds this under a real `bwrap`, in both modes. The user has since made it a
  rule rather than a side effect, and moved the state it protects into the agent's
  scratch; see "Where the tools keep their state".
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
- **Fixed after the `/simplify` review:** an enveloped tool's text was put in place by
  `finalize_content`, after `tools/post-execute`, so it overwrote what those rows wrote
  (ph-clm's receipt, offload's preview). A body now says what the value it returns
  renders as (`ToolRunContext.present`); the registry uses that wherever it renders
  that value, and a row that replaces the value gets the definition's rendering of the
  replacement.

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
8. **U6 comes before U5.**
9. **Read-only is a barrier for the repo, not for a tool's working state.** These tools
   change no file of the repo; they build data to probe it. So under a read-only
   sandbox they may not write the workspace and may write a designated area of their
   own — the agent's scratch — where an agent that restarts finds what it built.

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
- names the directories it writes outside the workspace — the grammars and its log —
  as extra writable paths, and from U4a keeps its index in the agent's scratch;
- offers the package's skill, its slash command and its `phern doctor` section, with
  the last two answered by running the command's `status`.

### Where the tools keep their state

Decision 9, against what pH already has. Every agent has a **scratch** directory,
`$PH_HOME/scratch/<session>/<agent>` (`ph.seams.workspace.scratch_of`): per session and
agent, under `$PH_HOME` so it survives a restart, outside the worktree by design, and in
the workspace's writable set in every sandbox mode — "the one place a read-only or
ephemeral agent is told it may write". The workspace root is writable only under
`workspace-write`. So the barrier the user asked for exists, and the designated area is
the scratch.

- **Each tool keeps its state in a directory of the scratch named for it:**
  `<scratch>/code-graph/` holds the graph database and its lock, and
  `<scratch>/text-index/<embedder>/` the vectors, the sidecar and the writer lock.
- **The command is told where.** `ph.toolcli` gains `--state DIR`, and reads
  `$PH_TOOL_STATE` when the flag is absent. The engine rows take it as their index
  location in place of the per-workspace cache directory. A manual declares the
  directory name it wants (`Manual.state: "code-graph"`), and `ph-utcp` sets
  `PH_TOOL_STATE=<scratch of the calling agent>/<name>` on each call — the session and
  the agent are on the call, so this needs no per-agent manual.
- **Reused on restart.** A resumed session has the same session and agent ids, so the
  same scratch, so the same index: a re-run parses or embeds only what changed. A child
  in its own worktree has its own agent, so its own index of its own tree.
- **With no session** — a person at a shell typing `ph-code-graph call …` — the state
  goes where it goes today, `$PH_CACHE/<tool>/<digest of the root>`, unless `--state`
  names a directory.
- **What stays shared:** the tree-sitter grammars and the embedding model's weights.
  Both are package content keyed by version, not data about a repo, and copying them
  per agent would cost the 4 s unpack and the 90 MB download each time. They stay under
  `$PH_CACHE` as grants, with the command's log.
- **The rule, stated in the sandbox seam:** a grant (`writable_extra`) is honored in
  every mode, and a grant inside the workspace root is refused — otherwise a manual
  could name the repo and make it writable in a read-only session.
- **The cost:** two sessions on one checkout no longer share an index, so a new session
  builds its own — about a second for a package's code graph, 20 s to 185 s of
  embedding for this repo's `docs/`. A resumed one does not.

### Indexing as a background worker with visible progress

This is the user's proposal, and it replaces the earlier "call again to collect"
design. An index tool never waits for indexing to finish.

- **A worker process per package and root, and per child that asks.** The registration
  row starts it as a `ctx.jobs` job when it mounts, through the spawn door "Before U5"
  describes, confined to the root's workspace and the agent's scratch. The job belongs
  to the row's scope, so unmounting cancels it. A child agent in its own worktree gets its worker on its first tool
  call, because each text-index worker holds a loaded model (about 400 MB) and a
  throwaway worktree may never search. A harness with no such row also gets the worker
  on the first call.
- **It loads what it needs once and stays up.** For the text index that is the
  embedding model. It indexes, then answers `text_search` queries without loading the
  model again, and exits after an idle period.
- **It reports progress** to a small file beside the index, written atomically:
  state, documents done, documents total, and the failure text when there is one. It
  holds the writer lock while it works, so "is it alive" is asked of the lock and not
  of a pid.
- **The tools answer at once.** `text_index` on a new project starts or queues the work
  and returns `state: indexing` with the counts. `text_search` during indexing answers
  from what has been saved so far and carries a warning, "Not finished indexing: 120 of
  539 documents". The code graph follows the same contract.
- **The front end shows it.** The row contributes a reading to `ctx.tui_status`,
  "indexing 120/539", from the last progress it read — watched, not read per redraw.
  When the job settles (`job/settled`) the person gets a completion notice and the
  agent the worker indexes for gets a note through `inject`.
- **After a daemon restart** the old worker is gone — it died with its mount, or the
  orphan sweep ended it — and the remounted row starts a new one. The index's own
  increments mean nothing is parsed or embedded twice.

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

**U4a. Each agent's tool state in its scratch.** `--state` and `$PH_TOOL_STATE` in
`ph.toolcli`, the engine rows reading them, `Manual.state` and the per-call variable in
`ph-utcp`, the sandbox rule that refuses a grant inside the workspace, and the code
graph's grants reduced to the grammars and the log. *Gate: under a read-only sandbox an
agent's `code_index` cannot write the repo and builds its index in its scratch; a
resumed session's `code_index` parses nothing unchanged; a child in a worktree has its
own index; `ph-code-graph call` with no session still indexes under `$PH_CACHE`, and
with `--state` where it says.*

**U5. The worker and its progress.** The text index's digest and interval saves, the
worker, the answer-at-once contract, the progress file, the status reading, the
completion notice. *Gate: a second `text_search` does not load the model again; a
search during indexing returns saved passages with the warning; the footer reading
moves; a worker killed with SIGKILL is replaced at the next mount and embeds nothing
twice.*

**U6. Text index through `ph-utcp`, with E1.** Its command, manual and registration
rows, and the local model's config passed through, with its state in the agent's
scratch from the start (U4a) and the model's weights a shared grant. *Gate: U4's gate for the text tools
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

The order is U4a, U6, U5, then the rest of U7 (`on_mount`, which only the worker reads),
then U8 to U10.

## What breaks

- **The three rows stop running tools in the session process.** `ctx.code_graph` and
  `ctx.text_index` are no longer provided to a session's mount. A deployment that
  registered its own embedder on `ctx.text_index` configures it in the tool's own
  profile instead.
- **Index tools may return before the work is done.** In a session, where a host
  answers, a run longer than the row's `wait_s` answers `state: indexing` with a
  progress count, and its final counts are in no tool result: the agent is told it
  ended, with the total. A caller that read `IndexValue` as a finished count reads
  `state` first.
- **`permissions-fs` no longer screens these tools' reads**, and `fs/read-intent` no
  longer fires for them. The sandbox bounds them, as it bounds `bash`.
- **Indexes move into the agent's scratch (U4a).** An index under
  `$PH_CACHE/<tool>/<workspace digest>` built by a session is not found by the next
  session; it is rebuilt in that session's scratch. Indexing works in a read-only
  session, and never writes the repo.
- **A manual's `writable` grant inside the workspace is refused (U4a).**
- **Row config reaches the tool through the environment**, so a change to it applies
  to the next command and to the next worker start.
- **`glob` on both rows becomes `index.include`**, a list. The old key is removed, not
  aliased.
- New: two console scripts, a lock file in each code-graph index directory, a progress
  file in each state directory, a digest per document in each text index (one built
  before has none, and each document is embedded once more), and one host process per
  package for each tree an agent works in, started at a root agent's first step unless
  `index.on_mount` is false.
- **Both rows' `path` setting is removed.** `--state` and `$PH_TOOL_STATE` say where an
  index goes.
- **`install_or_status`'s body takes the invocation as well as the verb.**
- **`config_env` names every row it is given**, a default one as `{}`, where it used
  to leave a default row out.

## Left for the step that meets it

Nothing is left. What each step settled where the code was read is recorded above.

## Not verified

*2026-11-11: the second, third, fourth and sixth of these were run; see "Before
publishing". The rest stand.*

- How UTCP clients in other languages stringify list and boolean arguments. The decoder
  accepts JSON, Python repr and comma-separated forms for that reason.
- The model load time on this machine. No weights are cached here, and none were
  downloaded for this review; the 19 s and 26 s figures are the README's.
- How long a resumed session's re-index takes when its scratch survived but the tree
  moved a long way under it — a rebase across hundreds of files. U4a's gate measures
  the unchanged case only.
- A resumed session finding its index through a real daemon restart. The gates hold the
  two halves: the state directory is a function of the session and agent ids, and a
  second process given the same directory parses or embeds nothing unchanged.
- `nomic-embed-text-v1.5`, or any model with remote code, under a sandbox. Only MiniLM
  was run.
- A host across a real daemon restart, and a real agent loop taking a step: the gates
  call the `agent/pre-step` listener themselves.
- A hosted embeddings API, and Google's. The OpenAI-compatible request was run against
  a real llama.cpp; Google's only against a fixture server that speaks it.
- A fan-out of real sub-agents on one tree, spawned by `rlm-indexed`. The gates build
  the parent and its children by hand, one reader and one writer.
