# ph-text-index

*`text_index` and `text_search`: semantic retrieval over the agent's own
documents, on a local [turbovec][turbovec] index.*

`grep` finds the string you typed. This finds the passage you meant — and hands
back the `path:start-end` it came from, so the model can answer from the passage
or `read` the file around it.

```bash
ph --profile llama --provider llama --model <model> \
   --patch '{insert: [{id: text-index, name: text-index},
                      {id: text-index-local, name: text-index-local}]}' \
   -p "index docs/ then tell me how the workspace seam decides a containment tier"
```

## Two rows, and why

| row | what it is |
|---|---|
| `text-index` | the seam: `ctx.text_index`, the chunker, the turbovec store, and the two tools |
| `text-index-local` | the provider: a `sentence-transformers` model, loaded on first use |

The split is the shape `subprocess`/`subprocess-local` and
`code-runtime`/`code-runtime-python` already have in this tree, and it pays for
itself three times:

* a deployment with an embeddings **endpoint** — llama.cpp's `/v1/embeddings`, a
  provider's API — writes its own row and keeps the tools;
* the **tests** run the real turbovec index against a deterministic stub
  embedder, so the suite proves chunking, paging and persistence without
  downloading a model;
* the row that pulls in **torch** is one a profile can leave out.

**The tools appear only once an embedder is claimed.** `TextIndexSeam.register`
registers them on the provider's own scope, so a profile with the seam and no
provider advertises nothing at all — the rule `subagent-task` states, for the
same reason: a tool named in every prompt and refused on every call spends
context teaching the model a capability the deployment does not have.

## The two tools

**`text_index(paths, glob?, forget?)`** reads each document through `ctx.fs`,
cuts it into passages, embeds them and stores them. Re-indexing a file replaces
its passages, so running it again after an edit is correct. `forget: true`
removes instead.

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

Every indexed byte arrives via `ctx.fs.read`, so `fs/read-intent` fires,
`permissions-fs` decides, the workspace tier bounds the path and every
registered screen gets its say — the same door `read` goes through, and for the
reason `tool-attach` insists on it (I-9). It matters more here, because
indexing is a *bulk* read: a tool that walked the tree with `Path.open` would be
an exfiltration primitive with a glob argument.

The corollary: the index is **per-deployment, not per-agent**. Two agents share
`$PH_CACHE/text-index/<embedder digest>` unless a profile says otherwise, so a
passage one indexed is retrievable by another. Right for a documentation corpus,
wrong for anything private — `path:` in the row's config is how you separate
them.

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
first and `open` reconciles by trusting the sidecar and dropping any id the index
cannot answer for — a chunk nobody can retrieve is invisible, where a vector
with no text would surface as a hit this row could not describe.

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
ph --profile rlm-indexed --provider llama --model <model> --mode tui
```

This package registers a `ph.bundles` entry point so that profile is *hidden*
on an install without this distribution rather than offered and then failing at
mount — `available_profiles()` gates on bundle resolution, and the refusal names
the package to install.

The bundle carries **both** rows, and that matters: the seam registers no tools
until an embedder is claimed, so a bundle shipping only `text-index` would mount
a service and advertise nothing, with no error anywhere. A deployment bringing
its own embedder disables `text-index-local` and mounts its own provider row.

## Installing

`sentence-transformers` is a **hard dependency**, so installing this package is
how a deployment gets a working index. That is most of a gigabyte of torch, in
every environment that resolves the package, CI included. Moving it to an
extra is one line in `pyproject.toml`; the cost of that is that
`pip install ph-text-index` no longer gives you something that runs. The
seam/provider split keeps the choice cheap either way — nothing but
`text-index-local` imports it.

## Switching models

The default is **`sentence-transformers/all-MiniLM-L6-v2`** — 384-dimensional,
about 90 MB, symmetric so it needs no prefixes, and fast enough on a CPU that
indexing a documentation tree is a coffee break. It is the *small* choice, not
the best one.

The index directory is keyed by a digest of the embedder's identity — model
name **and** both prefixes, because all three move the vector space — so
switching is safe to try: a new model gets its own index and switching back
finds the old one intact. Pointing `path:` at a fixed directory and then
changing the model raises `IndexMismatch` rather than returning neighbours
computed in a space nothing shares.

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

## Where the weights live, and installing on purpose

`$PH_CACHE/models`, set by the row (`cache:` overrides it). Left to
`sentence-transformers` they would go to `$HF_HOME` or `~/.cache/huggingface` —
outside all three of pH's roots, so a gigabyte of weights would sit somewhere
`ph doctor` never mentions and `rm -rf $PH_CACHE` would not reclaim. Rebuildable
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
costs no turn. `ph doctor` answers the same question without mounting an agent —
the section reports `model: not loaded — /text-index install`.

For an **unattended** run — a daemon, a scheduled tick, `ph -p` in CI — there is
nobody to type either:

```yaml
- id: text-index-local
  config:
    preload: true
```

That loads the model at **mount** and refuses the mount with a `MountRefusal`
sentence if it cannot, which is the cookbook's own rule: refuse at mount, not at
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
