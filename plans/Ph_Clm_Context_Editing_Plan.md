# ph-clm: the model edits its own context, over the session surface

A review of `sources/context-language-models` (CLM) and of its Pi port (`pi-clm`), and a plan
for a `ph-clm` plugin package shaped like `ph-rlm`. The model manages its context with the
tools it already has, and every edit lands in the session log as a **tombstone**, a
**replace** (a compressed version) or a **rewrite** (part of a section changed). The log keeps
the originals, so the person and the model can diff what changed.

## The short answer

- **ph already has the primitive.** A `user/message`, `assistant/message` or `tool/result`
  appended with `surfaceOp: {op: "replace", replaces: [seq, ...]}` takes the named nodes out
  of what the model sees, and the log keeps every one of them (I4). Compaction, argument
  elision, the overflow clip and paste offload all use it today. `derive_messages()` follows the
  surface, while `transcript()` keeps what the person saw. A ph-clm edit is one more producer
  of that operation, not a new mechanism.
- **The session log needs no new envelope fields and no format bump.** A section's identity
  is the seq of its first surface node, and seqs never move (A1). A replacement's lineage is
  already on the event (`surfaceOp.replaces` plus `sourceEventSeqs`), and the originals stay in
  the log, so a diff is a read. ph-clm adds two **ignorable** record types for its own
  bookkeeping (`clm/revised`, `clm/declined`) and a writers-table row. A log written with
  ph-clm opens and derives the same context on a build without it.
- **Sections come from the surface's tool-pairing balance, not from step events.**
  `cuts_over` in ph-stabilize already computes where the surface can be cut without orphaning
  a tool call or a result, and its docstring says why step boundaries are the wrong source:
  "a landed replacement moves positions". It moves to ph-core so ph-clm and compaction share
  one rule.
- **Marking every section for caching cannot work, and would not help if it could.**
  Anthropic allows 4 breakpoints per request, and ph already spends all four. A cache entry is
  keyed by the *whole prefix* up to its breakpoint, so removing a middle section misses every
  entry after it, however many marks there are. What can be done is in
  [Caching](#caching-what-a-middle-edit-costs-and-what-helps). Show the model what each edit
  costs, keep one warm **floor** breakpoint, and on self-hosted SGLang there is
  CLM's Suffix Cache Reuse.
- **License:** the CLM repository is **CC BY-NC 4.0**, and ph is MIT. Take the ideas, not the
  code or the prompt text. pi-clm is MIT and can be borrowed from with attribution.

## What CLM is

The paper's claim is that a model manages its own context best when the context is **a file
it may edit without restriction**. Three implementations were reviewed.

### `clm/clm_harness` (Harbor; the paper's harness)

The model gets one `bash` tool. Per turn, `ContextEnv.step` (`context_env/env.py:182`) does the
following:

1. **Mirror.** It writes the editable context to `/tmp/.live_ctx/LIVE_CTX_MAIN.txt`, one
   `[[CTX_TURN i role=…]]` block per message, with the system prompt and task hidden and
   pinned (`render_editable`, `context_utils/context_string.py:90`). The turn numbers are
   display-only and are renumbered on every render.
2. **Command.** The model runs any command. Editing means `sed`, `python3` or `cat >` on the
   mirror.
3. **Read back.** If the file changed, `parse_back` (`context_string.py:104`) turns whatever
   headers survived into a message list:
   - assistant text stays assistant, and **everything else folds to a plain user turn**;
   - tool-call structure is not rebuilt;
   - stray text becomes a user note;
   - empty turns drop, and consecutive same-role turns merge.
4. **Edit gate** (`context_env/edit_gate.py`). Under `fit`, a grown edit is accepted if it
   still fits the budget. Under `shrink`, an edit must shrink the context. A rejected edit is
   not applied, and the model is told why in a one-line receipt.
5. **Budget** (`utils/budget.py`):
   - nudges at 25/50/75% of the budget, plus a persistent urgent nudge near the limit;
   - over the limit, it **rolls back** the newest turns and tells the model to compact (up to
     50 times), then gives one final turn.

   A turn that only edits the mirror and prints nothing is free against the step budget.
6. **Trajectory.** `ATIF-CTX` (`agent_trajectory_format/models.py`) records an append-only list
   of **segments**:
   - each segment starts with the literal `input_context` and holds only the new steps after it;
   - a new segment opens on every edit, carrying a `BranchTrigger` (`kind`, `removed_tokens`,
     `summary`, …);
   - sub-agents hang off the parent as `subtrajectories` with spawn/fold links.

The system prompt (`clm_agent/prompts.yaml`) carries the cost model, which matters for ph:
"an edit forces everything *after* it to be re-read, so cost grows with how much text FOLLOWS
the edit". From that it advises:

- batch edits;
- don't compact a small early region above a long, still-useful tail;
- be generous in the summary, since the tail is re-read anyway.

### `pi-clm` (the Pi coding-agent port; MIT)

This is the closest analogue to ph: a harness with ordinary file tools and an append-only
session JSONL.

- The mirror carries a `LIVE_CONTEXT` header: version, revision, document nonce, and baseline
  digest.
- Blocks look like `[[CTX_TURN … id=… role=… protected=…]]`, with stable ids between accepted
  edits. A `new-*` id inserts a block.
- Edits are read back at `turn_end`:
  - untouched blocks keep their original message objects;
  - an edited assistant body becomes text-only and loses its tool calls;
  - an edited tool result becomes a custom message;
  - **broken tool-call groups are flattened to text rather than refused**.
- **Pi's history is never rewritten.** Each accepted edit is stored as a full-projection
  checkpoint (a `live-context-state` custom entry), which is revalidated against the raw
  prefix on resume, fork and `/tree`.
- Pi's automatic compaction is paused because it would throw away the model's edits.
- An overflow guard withholds the oldest tool results behind files.
- `/clm` opens a panel with a per-revision side-by-side diff.
- Its docs say nothing about prompt caching.

### `suffix_cache_reuse` (SCR; an SGLang patch)

SCR exists because a CLM edits the middle of its prompt, and standard prefix caching then
re-prefills everything after the first changed token.

- SCR diffs each prompt against the session's previous one.
- It relocates the K (6) longest surviving spans: it copies their KV entries, re-rotates the
  RoPE keys, and splices them in after the edit.
- Only the new text is prefilled. The relocated tokens keep slightly stale states, which the
  README argues can help.
- It matches requests to conversations by content and `cache_salt`, so no client change is
  needed.
- It supports only SGLang 0.5.16 with Qwen3.6-27B. It is a self-hosting option, not something
  ph implements.

## What ph has today

| piece | where | what it gives ph-clm |
|---|---|---|
| Surface-eligible types | `SURFACE_EVENT_TYPES`, `session/events.py:74` | Only `user/message`, `assistant/message` and `tool/result` reach the model. |
| `SurfaceReplace` | `session/events.py:107` | **An id-set, not a range**: every named seq must be a node *now*. The replacement lands where the earliest was. |
| Fold rules | `session/surface.py:144-302` | `sourceEventSeqs` must cite every shadowed node. A shadowed node cannot be named again. A `tool/result` replace may change **only content**, of exactly one result (`:202`). |
| Substitution vs in-place rewrite | `is_in_place_rewrite`, `surface.py:66` | A one-node near-copy reads as a rewrite, which the TUI updates in place. A range reads as a substitution, so the originals are dimmed and the replacement is shown. |
| The model's view | `Session.derive_messages`, `session/session.py:733` | Cached per node and rebuilt on `replace_generation`. |
| The person's view | `derive_transcript`, `session/derive.py:58` | Every append-origin message, so the originals stay visible. |
| One request source | `_build_request`, `agent_loop/driver.py:615`; invariant I3 | Every request's `messages` *is* `derive_messages()`, and `agent-loop-invariant` refuses anything else. A history edit **must** go through the surface. There is nowhere else to put it. |
| Precedent: compaction | `ph_stabilize/compaction.py` (`_land` :1270, `compaction/declined` :1383) | The summary and its replace land in one `session.batch()`, and failed attempts are recorded. Compaction summarizes the **surface**, so it keeps ph-clm's edits rather than discarding them (the reverse of Pi). |
| Precedent: paste offload | `ph_stabilize/input_offload.py:212` | `user/message` from a `PluginSource`, with `SurfaceReplace(replaces=(seq,))`, in one batch beside its own record. |
| Tool-pair balance | `cuts_over` / `balanced_cuts` / `safe_cutoff`, `compaction.py:392-440` | Cut `i` is balanced when no call or result straddles it, computed over the current surface. |

## Mapping the operations onto the surface

The user-facing verbs and the one log operation behind each:

| verb | what it names | event appended | shape |
|---|---|---|---|
| **tombstone** | one or more whole sections | `user/message` (`PluginSource`, `plugin: "clm"`, `form: "notice"`) whose text is a one-line marker, e.g. `[S412–S431 removed: dead-end grep for the parser bug]` | substitution |
| **replace** (compress) | a contiguous run of whole sections | `user/message` (`PluginSource`) holding the model's or person's revised text | substitution |
| **rewrite** a tool result | one `tool/result` node | `tool/result` with the same call id and error flag and new content. Core already allows exactly this. | in-place rewrite |
| **rewrite** assistant text | one `assistant/message` node | `assistant/message` with new text and **the same tool calls**, the shape `compaction/args-truncated` already uses | in-place rewrite |

Rules ph-clm enforces before it appends anything (core enforces the rest):

- **Whole sections only for a substitution.** The shadowed set must be a union of balanced
  intervals, so a tool call and its result are always both in or both out. This is a refusal,
  not pi-clm's flatten-to-text repair. The model gets told which section it split.
- **Contiguous.** A substitution lands at its earliest node, so its sections must be
  adjacent on the surface. Two separate runs are two substitutions in one batch.
- **Protected:**
  - the context snapshot message, which `context_message` re-adds anyway when it is missing;
  - optionally the first user task;
  - the newest section, which holds the step that is editing.
- **One batch.** Each edit's replacements are committed in a single `session.batch()` with
  their `clm/revised` record. This is the durability door that compaction and input-offload
  already use.
- **Back-to-back user messages are fine.** A user-role replacement can sit next to a user
  message. Compaction already produces that, and the Messages API combines adjacent same-role
  turns.

One replace event lands **one** node, so every substitution collapses its range to one
message. That matches CLM, whose `parse_back` also gives each block one plain message.
Restoring a multi-message range as its original separate messages is the one thing current
ops cannot express. See open decision 3.

## Sections

A **section** is the run of surface nodes between two consecutive balanced cuts: a user
message on its own, or an assistant message together with every tool result it opened. It is
the smallest unit that can leave the surface without orphaning anything.

- **Id `S<seq>`**, the seq of its first node. Unlike CLM's turn numbers, it does not renumber
  between renders. Unlike pi-clm's per-document nonce ids, it needs nothing stored: A1 makes it
  stable for the life of the log.
- A replacement node is itself a section with its own id. A section map shows what it stands
  for: "S2210, replaces S412–S431, compressed, 9.1k→640 tokens".
- The section map is a fold over the surface (`SessionFoldCache`). It gets the same "rebuild on
  `replace_generation`, extend on append" treatment as `derive_messages`.

**What to move:** `cuts_over`, `balanced_cuts`, `safe_cutoff` and `_open_call_delta` go from
`ph_stabilize/compaction.py` to ph-core (`ph.session.surface` or a small `ph.session.balance`).
ph-clm must not depend on ph-stabilize, and two copies of "where may the surface be cut" would
drift.

## Using the ordinary tools: the mirror

The model edits a rendered file with the tools it already has: `read`/`edit`/`write`, a shell
`sed`, or, under the `rlm` profile, Python in a Code Mode cell. ph-clm then **compiles the
diff into the verbs above**.

```text
[[LIVE_CONTEXT session=<id> base=<replace_generation>:<last node seq>]]
[[SECTION S412 role=assistant calls=2 tokens=5.2k after=38.4k]]
<assistant text>
[[RESULT S413 call=read]]
<tool output>
[[RESULT S414 call=grep]]
<tool output>
[[SECTION S415 role=user tokens=210 after=33.1k]]
...
```

`after=` is the number of tokens that follow the section, which is the cost of editing it (see
[Caching](#caching-what-a-middle-edit-costs-and-what-helps)). Lines in a body that start with
`[[` are escaped, as pi-clm does.

**Read-back happens in `tools/post-execute` of the top-level call that wrote the
file**, before that call's result is logged. ph-clm hashes the file after every
top-level call and compares it with the render it wrote. Shell writes and kernel-side
Python writes are caught the same way: a Code Mode cell's call is the `ipython` call.

Not on `agent/pre-step`, which was the first draft. The log has to say what the
model was told. Suppose the model's `sed` succeeds and its result is logged, and the
daemon then dies before the next step reads the file back. On resume the mirror is
re-rendered from the log, so the edit is gone, while the log still says the write
succeeded. With post-execute, the edit lands before the result does. A crash between
the two leaves an interrupted call beside an edit that did land, which is exactly
what happened.

The diff compiles to edits as follows:

| what changed in the file | becomes |
|---|---|
| nothing | nothing |
| a `SECTION` block deleted (header and body) | tombstone (consecutive deletions become one marker) |
| text inside a `RESULT` sub-block changed | tool-result rewrite |
| assistant text changed, `RESULT` headers intact | assistant rewrite (calls kept) |
| several adjacent blocks' headers removed and their text replaced | one substitution over those sections |
| a whole-file rewrite with no headers | one substitution over the editable region (CLM's "replace everything with notes") |
| blocks reordered, a sub-header damaged, a section split, a protected section touched | **refused**: the receipt names the block, and `clm/declined` records it for the auditor |
| `replace_generation` moved since the render, because another replace landed meanwhile (an offload or a compaction) | refused and re-rendered. The model edits against the current surface. |

Only `replace_generation` decides staleness. The last node seq in `base=` is informational,
the way pi-clm treats its `baseline=` digest. Sections appended since the render are not in
the file and read as untouched. Otherwise every edit made in a step, after that step's own
results landed, would be refused.

The call that made the edit carries a one-line receipt in its own result, in CLM's
style: "edit applied: 3 sections → 1, ~41.2k→29.8k tokens, re-reads 12.4k following".
`post-execute` returns the result with the receipt appended (`Accept` with new
content, as tool-result offload does), so the receipt is logged with the result. No
separate notice message is added to the context, and the explicit tools already
work the same way.

**Where the file lives.** The tools run in different places: the host, a sandbox, or
ph-runtime-guest. pi-clm's stated limitation is that "remote tool backends cannot see the local
mirror". So ph-clm writes the mirror **through `ctx.fs`**, the same backend the fs tools use,
to a path its fs permission rule allows and the runtime can see. Where that is per profile
is the first thing to settle in Phase 2.

## Calling the operations directly

The same door, the `Editor` that `clm-context` provides as `ctx.clm` (Phase 1b), also backs explicit tools. The mirror is one front end
and these are another, so there is one implementation of the rules:

| tool | does |
|---|---|
| `context_sections(range?)` | the section map: id, role, tokens, `after=`, first line, and what a replacement stands for |
| `context_tombstone(sections, reason)` | tombstone |
| `context_replace(sections, text)` | substitution (compress, or a free rewrite of a span) |
| `context_rewrite(section, old, new)` | `str_replace` inside one result or assistant body: an in-place rewrite |
| `context_diff(section)` | unified diff, current versus the original messages it stands for, read from the log |
| `context_recall(section, max_tokens)` | the originals behind a replacement, returned as this call's result and bounded like pi-clm's `live_context_recall`. It appends at the tail, so it costs no middle edit (open decision 3). |

Under Code Mode they are registered as a code namespace (`tools.register_code_namespace`, as
`rlm-messaging` does). A cell can then compute an edit in Python, which is CLM's
`python3 - <<PY` idiom with a typed API instead of a regex.

## Comparing the changes

- **The log keeps both versions**, so comparing them is a read:
  - the current section is `derive_event_message(log[seq])`;
  - the original is `log[s]` for each `s` in its `surfaceOp.replaces`;
  - this applies recursively, because a replacement can replace a replacement.

  `context_diff`, a `ph clm diff <session> <section>` command and any TUI view all read it
  the same way.
- **The TUI already draws both shapes.** An in-place rewrite updates its row. A substitution
  dims the shadowed rows and shows the replacement (`is_in_place_rewrite`'s docstring). A
  per-revision diff panel like pi-clm's `/clm` is later work.
- **Trying an alternative context.** `SessionStore.fork` at the tip, then edit in the child.
  The parent is untouched, and the two surfaces can be compared or both continued. A reference
  fork cannot *start* from rewritten history (its seed is the parent's literal prefix), but it
  does not need to: the edit is just the child's first appends. ATIF-CTX opens a new segment
  per edit. In ph the replace event *is* that boundary, so no new log is needed unless the
  person wants a branch.

## Caching: what a middle edit costs, and what helps

**The rules** (official Anthropic docs, read 2026-10-09):

- A request may carry up to **4** breakpoints.
- The cache key is the exact prefix in the order `tools` → `system` → `messages`, and a change
  invalidates everything after it.
- From each breakpoint, the lookback checks at most **20** earlier block positions, and it
  finds only entries an earlier request wrote at its own breakpoints.
- TTL is 5 min (refreshed on each read) or 1 h.
- Price multipliers on base input:
  - a 5 min write costs **1.25×**, a 1 h write **2×**;
  - a read costs **0.1×** (**0.05×** on Opus 5.5 and Sonnet 5.5, **0.025×** on Fable 5.1).
- OpenAI caches the exact prefix automatically, in 128-token increments, from 1,024 tokens.

**What ph does today** (`ph_app/adapters/anthropic.py`):

- `CACHE_BREAKPOINTS = 4` (:116) is spent as tools (1) + system (1) + two message checkpoints
  quantized to `CHECKPOINT_EVERY = 4` (`_checkpoints`, :211; applied in `_body`, :325), so that
  consecutive requests mark the same index.
- There is no `ttl`, so the default applies.
- The OpenAI-compatible and Google routes send no markers and cache implicitly.
- P6-13 already notes that "no marker moves with input-offload's or compaction's own boundary —
  the adapter cannot see retention".

**Why per-section marks fail.** There are only four slots, and ph uses all of them. And an
entry for "everything up to section k" is useless once anything before k changes. After an
edit at surface position *p*:

- every entry ending after *p* misses;
- the tail is written again at 1.25× instead of read at 0.1×;
- if *p* is more than 20 blocks behind the new breakpoints, or the entries before *p* have
  gone cold, even the unchanged prefix is written again.

**The cost of an edit, in plain numbers.** Remove *R* tokens at a point with *T* tokens after
it. On a 5 min Anthropic cache:

- the one-off cost is ≈ *T* × (1.25 − 0.1);
- the saving is ≈ *R* × 0.1 per later request;
- so the edit pays for itself on price after **≈ 11.5 · T / R** requests (≈ 24 · T / R at
  0.05× reads).

Two examples:

- Dropping 20k tokens with 10k after them pays back in about 6 requests.
- Dropping 2k with 40k after them takes about 230.

That is CLM's prompt advice as arithmetic. It is also `docs/dev-notes/prefix-cache-benchmark.md`
finding 6 again: on a discounting provider, editing buys **headroom and focus** more than
money. On a route that does not discount cached input, the saving is the full *R* per request.

**What ph-clm should do:**

1. **Show the cost before the edit.** Each section carries `after=` in the map and the mirror.
   The receipt reports the re-read. The prompt states CLM's three rules in ph's own words:
   batch; don't edit far above a long useful tail; a generous summary is nearly free.
2. **One floor breakpoint instead of two quantized ones.** Keep tools, system and the moving
   tail checkpoint. Spend the fourth slot on a **floor**: a section boundary that stays put
   across many requests, so every request reads it and it stays warm.
   - Edits above the floor keep the prefix up to it.
   - The section map marks the floor, so the model knows which sections are cheap.
   - The floor moves forward rarely, for example when the tail beyond it passes a threshold,
     and it lands on the first request after an edit at the seam.

   This takes a provider-neutral hint from ph-clm to the adapter (a field on the request
   config, which `agent/request` already carries), and the Anthropic adapter honors it. It is
   the "mark sections for caching" idea in the one form the 4-slot limit allows. Measure it
   before adopting it: extend `tests/prefix_bench.py` with an edit scenario.
3. **Optional 1 h TTL** on the tools and system breakpoints, and on the floor, for sessions
   with pauses. A config knob on the Anthropic row.
4. **Self-hosted Qwen on SGLang:** SCR reuses the suffix after an edit, which is the thing the
   other providers cannot do. The client needs a per-session `cache_salt` (isolation) and,
   optionally, a `rid` per request. Check whether the OpenAI-compatible adapter can pass
   `extra_body` through.

**Not adopted: Anthropic's server-side context editing** (`clear_tool_uses_20250919`, beta
`context-management-2025-06-27`). It edits after the request leaves ph, so the log would no
longer say what the model saw, which I3 and the agent-loop invariant exist to guarantee. The
docs also say it invalidates the cached prefix, so it brings no cache advantage to trade
against that.

## Does the session log need to change?

| need | already there | to add |
|---|---|---|
| stable section ids | event seq (A1) | nothing |
| tombstone / replace / rewrite | `surfaceOp: replace` on the three surface types | nothing in core |
| keep the originals | I4: the log is never mutated, and `transcript()` keeps them | nothing |
| diff | `replaces` and `sourceEventSeqs` link to the originals | nothing |
| survive restart | the surface fold runs on resume like any other | nothing (pi-clm has to revalidate checkpoints; ph does not) |
| where the surface may be cut | `cuts_over` (ph-stabilize) | **move it to ph-core** |
| why, how, by whom, at what cost | — | **`clm/revised`** (ignorable): `{via: mirror\|tool\|person, ops: [{verb, sections, replacement, tokensBefore, tokensAfter}], reread, reason}`, in the same batch as the replacements |
| refused edits | — | **`clm/declined`** (ignorable): `{via, code, reason, block?}`. This is compaction's "record the attempts that fail". |
| write permission | `_WRITTEN_BY` writers table, `test_log_writers.py` | a `ph_clm.edits` row for `clm/revised` (and `clm/declined` in Phase 2), beside ph-rlm's types. Phase 1 also granted it the three surface types. From Phase 1b, core's revision door (`ph.session.revise`) writes every replacement, and the row keeps only ph-clm's own records. |
| bring an original back | every original is in the log by seq | nothing for `context_recall` (at the tail) or a flattened restore |
| restore a multi-message range as separate messages, or insert a new block | no op inserts a node after another | **deferred**: a surface op would mean `SESSION_FORMAT_VERSION` 4 (open decision 3) |
| a tombstone that leaves *no* message | no type does this honestly | **deferred**: see open decision 1 |

## The package, shaped like ph-rlm

```text
packages/ph-clm/
  pyproject.toml        ph-clm (same version as the workspace), ph-core==<version>, MIT
                        [project.entry-points."ph.bundles"]  clm = "ph_clm:BUNDLE"
                        [project.entry-points."ph.plugins"]  clm-context = "ph_clm.context:apply"
                                                             clm-mirror  = "ph_clm.mirror:apply"
  src/ph_clm/
    __init__.py         BUNDLE = Path(__file__).parent / "bundle.yaml"
    bundle.yaml         rows: clm-context, clm-mirror (and a prompt row)
    keys.py             CLM: ServiceKey[ClmService]
    sections.py         the section map: a pure fold over the surface plus the balance rule
    edits.py            THE door: verbs → validated ops → one batch (+ clm/revised); _LOG here
    tools.py            context_sections / _tombstone / _replace / _rewrite / _diff, plus the code namespace
    mirror.py           render through ctx.fs, read back on agent/pre-step, compile the diff → edits
    prompt.py           prompt.section: the protocol and the cost rules, written fresh (not CLM's text)
    py.typed
  tests/
```

- Plugins use `@plugin("clm-context", affects=…, inject=[TOOLS, SESSIONS, …], config=Config)`
  (`cordis/plugin.py:130`).
- Tools use `define_tool(...)` (`tools/definition.py:856`) and `ctx.require(TOOLS).register`.
- The root `pyproject.toml` lists the package under `[tool.uv.sources]`, `ruff src`, `mypy files`
  and `pytest testpaths`.
- `phern` pins it `==<version>` (`test_packaging.py:115`).
- Profiles in `ph_app/profiles.py`: `rlm-clm` = `RLM_LAYERS` + `Bundle("clm")`, and a coding
  `clm` profile if wanted.
- Compaction stays on as the backstop at 0.85 of the window. It composes with ph-clm because
  both are surface replaces.

## The durable log and one mechanism

This section reviews the plan against ph's two ground rules: **the session log is
durable and append-only**, and **a rule several producers share lives in one core
door and is tested there once**.

**What already holds:**
- Every edit is an appended event. The log is never mutated (I4), and `transcript()`
  keeps the originals.
- A resumed session folds to the same view, and a test pins it.
- Each edit and its record land in one batch, and a crash mid-call reconciles from the
  record.
- The mirror is a derived cache: rendered from the log, never trusted as a source,
  and refused if a surface rewrite moved underneath it.
- The explicit tools and the mirror share one `Editor`. Compaction and ph-clm share one
  balance rule (`ph.session.balance`) and one stand-in predicate (`is_stand_in`).

**What did not hold, now changed in this plan:**
- **Mirror read-back.** Moved from `agent/pre-step` to `tools/post-execute`, for the
  durability reason given in the mirror section.
- **Budget readouts.** Phase 3's readouts now travel the receipt's channel: a trailer
  that `post-execute` appends to a tool result, as pi-clm's `sizeTrailer` does. Under
  I3, anything the model is shown must be logged. A separate notice message per
  threshold would be one more surface node to edit away, and a second mechanism for
  the same job.
- **The cache floor.** Phase 4's floor hint goes in the request config, so the
  `request/header` event that records a config change records every floor move, and a
  replay reproduces the cache markers. The markers never enter the messages, so I3
  holds as it is.

**What still does not hold: replacements are built in three places.** Compaction
(summary, argument elision, overflow clip), input-offload (paste preview) and ph-clm
(substitute, rewrite) each:
- construct `SurfaceReplace` and `SurfaceIntent` by hand;
- copy the payload rules: D8's dropping of usage/turn/step, content-only result
  rewrites, message-id handling;
- hold the three surface types in the writers table.

ph's convention for a rule several producers share is a core door that keeps it.
`ph.seams.subagents`' `record_*` doors are the precedent: providers report a child
through them, and the doors keep each record's durability rule. Phase 1b is that door
for surface replacements.

### Phase 1b: one door for every surface revision

None of this bumps `SESSION_FORMAT_VERSION`: the door writes the shapes the log
already holds.

| change | where | replaces |
|---|---|---|
| `substitute(log, shadowed, message)` and `rewrite(log, event, message)`, the only two ways to append a replacement. A **substitution** mints a new message id and cites every shadowed seq. A **rewrite** keeps the message id, keeps a result's call id and error flag, and drops usage/turn/step from an assistant payload. Both write into the caller's batch, so each producer's own record lands beside the edit. | new `ph.session.revise`, with its own `_LOG` | compaction's `_land`, elision and clip; input-offload's `_append_preview`; ph-clm's `Editor._substitute` and `Editor.rewrite` |
| `is_in_place_rewrite` decided **by type**: an `assistant/message` or `tool/result` replacing one node cited as its source. The door makes the type equal to message identity. Every rewrite is one of those two types and keeps the id; every substitution is a `user/message` with a new id. So the predicate is exact by construction, and it still reads the event alone. | `ph.session.surface` | today's structural test (one seq, cited as the source), which also matches a single-node substitution |
| Lineage: `shadowed_by(event)`, `origin_of(session, seq)`, `originals(session, seq)` | `ph.session.revise` | ph-clm's private copies. Any phern view of revisions needs these too (the Phase 5 diff panel, a `phern clm diff` command), and `ph_app` may not import `ph_clm`. |
| `one_line(text, limit)` | `ph.text` | `ph_app.wire.one_line` and ph-clm's `clip` |

**Governance.** `ph.session.revise` becomes the one writer of replacement surface
events. Compaction, input-offload and ph-clm keep only their own record types
(`compaction/*`, `offload/input-spilled`, `clm/revised`). A static walk, like
`test_log_writers.py`'s, holds that no shipped module but the door constructs a
`SurfaceReplace`.

**Testing the shared parts once:**
- The door's suite covers every rule it keeps, each one sabotage-checked.
- Producers keep only their policy tests (what to shadow, what text to write) and stop
  asserting payload mechanics.
- The section map's fold cache gets its invariant row (`contribute_fold_cache`), as
  the six other fold caches do. Its tests run it through `ph.testing.VerifyingFoldCache`,
  which checks every read against a cold fold.

**ph-clm's side, in ph-rlm's shape.** A `keys.py` with `CLM: ServiceKey[Editor]`, which
`clm-context` provides. The Phase 2 `clm-mirror` row injects it rather than building
a second map with its own cache and settings.

**Not advised:**
- **Declaring `clm/revised` with `declare_log_type`.** ph-rlm keeps its types in
  core's tables. Once the door owns the surface types, ph-clm's core row is only
  `clm/revised`. Moving that type out is worth it only together with a generic
  trajectory fallback for declared types, which would help every plugin.
- **A shared core message renderer.** Compaction's `render_for_summary` and ph-clm's
  `render_messages` differ on purpose, and core has no second consumer.
- **Sections or spans in core.** Compaction cuts a prefix and does not need sections.
  Revisit if its planner wants the memoized sizes.

## Phases

| phase | work | gate |
|---|---|---|
| **0. Core prep** ✅ | Move `cuts_over`/`balanced_cuts`/`safe_cutoff` to ph-core (`ph.session.balance`), with compaction importing them. | compaction tests unchanged; the three balance tests moved to `packages/ph-core/tests/test_session_balance.py` |
| **1. Sections and the door** ✅ | `sections.py`, `edits.py`, the six tools; `clm/revised` and the `ph_clm.edits` writers row | Each verb lands as a surface replace; `derive_messages` shows the replacement and `transcript` the original; a resumed session derives the same messages; refusals for an unbalanced or non-contiguous set, a stale seq, a protected section, or a result rewrite that touches more than content; `clm/revised` is in the same batch; the agent-loop invariant holds. Sabotage-check each refusal. |
| **1b. One revision door** ✅ | `ph.session.revise`; `is_in_place_rewrite` by message identity; lineage in `ph.session.surface`; `one_line` in `ph.text`; compaction, input-offload and ph-clm moved onto the door; `ph_clm/keys.py`; the fold-cache invariant row | the door's own suite, sabotage-checked; the static walk shows no `SurfaceReplace` outside the door; compaction, input-offload and ph-clm suites green without asserting payload mechanics; `VerifyingFoldCache` over the section map |
| **2. The mirror** | render through `ctx.fs`; read back in `tools/post-execute` of the writing call, with the receipt on its result; the diff compiler; `clm/declined` | `sed` and a Code Mode cell each produce the expected ops; a reorder, a damaged header, or a moved `base=` are each refused with a receipt; an edit-only step leaves no trace beyond its own call and result; a crash between an edit and its result resumes with the edit landed and the call interrupted |
| **3. Budget and prompt** | budget readouts as a `post-execute` trailer on tool results, from `ctx.token_meter`; the protocol section; the profiles | a recorded `rlm-clm` session on a real provider shows readouts at their thresholds and edits that land, with no notice message added to the surface |
| **4. Caching** | `after=` costs; the floor breakpoint hint as a request-config field, so `request/header` records each move, plus the Anthropic adapter; optional 1 h TTL | `prefix_bench` with an edit scenario: cache reads after an edit, floor against the two quantized checkpoints; a replay reproduces the markers |
| **5. Later** | a restore op (format 4); a revisions diff panel in the TUI; sub-agent briefs as files (CLM's `SUBCTX`) | — |

### Phase 1b as built

- **The door is `ph.session.revise`.** It holds `substitute`, `rewrite`,
  `editable_message`, `origin_of` and `originals`. `originals` lives here rather than in
  `ph.session.surface` because it derives messages, and `ph.session.derive` already
  imports `surface`. `shadowed_by` needs nothing of that, so it sits in `surface`, where
  `is_in_place_rewrite` and the TUI adapter use it too.
- **`is_in_place_rewrite` decides by type, and the commit makes the type exact.** A
  message-id comparison would need the replaced event, and the predicate's callers
  (repair, the todo row, the TUI) hand it the event alone. Instead:
  - `surface._assert_assistant_rewrite` sits beside the existing tool-result check. An
    `assistant/message` replacement must name exactly one current
    `assistant/message`, cite only it, and keep its message id.
  - Every other replacement is therefore a `user/message`, for every writer and on
    every replay, not only for the door's own writes.
  - Every log this format reads already agrees.
- **What the door adds, and what it leaves to the core.** The door builds the payload
  from the original, drops D8's fields, and writes the citation. It refuses only what
  the commit cannot: rewriting a user message, and a substitution in assistant role.
  The id rule, the content-only rule and the empty-run rule are the core's.
- **The writers table now has one writer of replacements**, `ph.session.revise`.
  Compaction, input-offload and ph-clm keep only `compaction/*`,
  `offload/input-spilled` and `clm/revised`. The writers walk in `test_log_writers.py`,
  which already polices who writes to the log, also fails on any `SurfaceReplace(...)`
  outside the door.
- **Compaction's `truncated_assistant_payload` is now `truncated_assistant_message`.**
  It returns the message, and the door builds the payload around it and drops D8's
  fields. The D8 reasoning moved into the door's docstring.
- **`one_line` is in `ph.text`.** phern's five importers use it, and ph-clm's
  `clip` is gone.
- **ph-clm:**
  - `keys.py` holds `CLM`, which `clm-context` provides;
  - the section map's fold cache has its invariant row (`clm-section-map`);
  - `SectionMap.fold`/`extend` are public, so a test runs `assert_fold_laws` over a
    log with revisions in it, and compares a map kept across edits with a fresh one.
- **Tests now sit where their rules live:**
  - the door's 8 tests cover its rules, and `test_surface.py` covers the commit-time
    assistant rule;
  - sabotage checks: 7 against the door and commit (one of them the static walk) and 9
    against ph-clm;
  - ph-clm's rewrite test asserts only its own policy (the section keeps its name);
  - compaction's duplicate of the door's D8 test is gone. Its end-to-end meter test
    stays, and its other tests pass unchanged.
  - A todo fixture that built a "rewrite" with a new message id now keeps the id, as a
    real rewrite does.

### Phases 0 and 1 as built

Where the build departs from the sections above, the build is what holds:

- **The log types arrived with Phase 1, not Phase 0.**
  `test_the_writers_table_is_what_the_writers_write` refuses a grant that nothing
  writes, deliberately, so a type can land only together with its writer.
- **`clm/declined` waits for Phase 2.** A refused tool call is already in the log as an
  error `tool/result`. Only the mirror, which has no call of its own, needs a separate
  record, and the same governance rule keeps the type out until its writer exists.
- **Substitutions use `form: "compaction"`, not `"notice"`.** `PluginSource` reserves
  that form for exactly this claim: the text stands in for conversation that has left
  the surface. Two things follow for free:
  - the TUI draws the edit as a revision row, with the originals dimmed;
  - compaction treats the edit as a prior summary, so it won't spend a model call
    summarizing it again (`ph.session.is_stand_in`, which the simplify pass promoted out
    of compaction).
- **A section's id is its first node's origin seq** (`sections.origin_of`). Without
  this, an in-place rewrite of a single-message section would rename it (`S3` became
  `S8` in a test), which breaks "ids never change" for the edit meant to change
  nothing. Only a substitution retires an id now, and the refusal for a retired id
  names what stands for it.
- **Spans are written `S412..S431`** in ASCII, in labels as well as input. Ruff flags
  an en dash as ambiguous, and the model has to type the span back.
- **The explicit tools apply their edit inside the call.** The call's own result is the
  receipt, which is the review's first placement for receipts, so no notice is added to
  the context. The step in flight is protected, and `reconcile` reports a crash between
  the edit and its result as `Done`, because the record lands in the edit's batch.
- **No code namespace.** Under Code Mode, every registered tool is already
  `tools.context_*`.
- **No `ctx.clm` service yet.** The door is `ph_clm.edits.Editor`, built once per row
  over a `SectionMap`. The map folds each event's facts once (`SessionFoldCache`), so a
  call pays only for what was appended since the last one. The Phase 2 mirror is in
  the same package and uses the same `Editor`.
- **A section keeps its id only through a true in-place rewrite**, judged by the
  replacement keeping the original's message id. Core's `is_in_place_rewrite` reads
  structure alone: one shadowed seq, cited as the source. That also matches a
  substitution standing for a single node, such as a one-message tombstone. Today only
  ph-clm meets that case: the core readers ask only about assistant messages, and
  every assistant replacement is a rewrite. Phase 1b moves the message-identity rule
  into core.
- **The open decisions follow their recommendations:**
  - visible tombstones, with adjacent ones absorbed into one marker;
  - user role for substitutions, and the original role for in-place rewrites;
  - `context_recall` at the tail, plus `context_diff`;
  - gate `none` by default, with `fit` and `shrink` available.

## Review against pi-clm's source

The source is at `sources/pi-clm` (v1.0.0, `b84a9d7`), read through its codegraph index. The
earlier sections came from its published docs. This section separates what the source
confirms from what should change. Section numbers (§) refer to `docs/architecture.md`.

**Confirmed**

- **Raw history is never rewritten.** The effective context is a projection (§1). ph gets the
  same thing from the surface without pi-clm's fragility. A pi-clm revision anchors to a
  SHA-256 of the raw prefix and is discarded when another extension's transform changes that
  prefix (§6, `projection.ts`). ph's replacements are ordinary log events that the fold
  replays.
- **No-op renders are never edits.** Untouched blocks reuse the original message objects, so
  re-rendering and re-reading the mirror never counts as a change (§4). The ph equivalent: an
  untouched section produces no event.
- **A whole-file rewrite keeps the first user turn** (§5). This matches the plan's protected
  task.
- **Pi's threshold compaction is cancelled in CLM mode,** because it summarizes the raw
  transcript and would throw away the projection (§7a). ph's compaction summarizes the
  surface, so keeping it as a backstop holds.
- **pi-clm has no caching logic.** `cacheRead`/`cacheWrite` are only summed into a request's
  size (`timeline.ts:66`). Everything in the plan's caching section goes beyond what pi-clm
  does.

**Should change**

1. **Staleness comes from surface rewrites only.** pi-clm checks revision and nonce, and treats
   `baseline=` as informational, so a header read earlier in the turn stays valid as new
   messages arrive (§3). Applied above, in the mirror section.
2. **Recall, not only restore.** `live_context_recall` returns an annotated entry's exact
   source as a tool result, bounded to 128–8,000 tokens (`continuity.ts:302`, `index.ts:919`).
   - It appends at the tail, so it never causes a middle edit.
   - ph already has every original in the log by seq, so `context_recall` needs no
     annotation store.

   Added to the tools table. See decision 3.
3. **Insertion is first-class in pi-clm.**
   - A block with an `id=new-*` header adds a note, tracker or scratchpad anywhere (§3–4).
     Any role label is lowered to user text.
   - Reordering blocks is an edit like any other.

   The plan has no insertion (a surface replace can land only where a shadowed node was), and
   it refuses reorders. This is the same missing operation as a multi-message restore. Folded
   into decision 3.
4. **Continuity annotations** come in three kinds: `pin`, `continuity` and `archive`.
   - Each is a durable pointer carrying a reason and a next action.
   - They are rendered in an extension-managed message after the editable region, so
     compressing the mirror cannot drop them (`continuity.ts:261`).
   - Under I3, ph would have to log that message, so re-rendering it on every request is not
     free. A `pin` maps more naturally onto "this section is protected".

   Later work, not Phase 1.
5. **Notices are request-only in Pi but logged in ph.** pi-clm appends outcome and budget
   notices to each request and never stores them (the `context` hook, `index.ts:1266`). In ph,
   every notice stays a surface node until something edits it away. Two cheaper placements:
   - **Receipt in the result.** Apply the edit in `tools/post-execute` of the call that wrote
     the mirror, and append the receipt to that call's result, so there is no extra node.
     pi-clm's `sizeTrailer` edits tool results the same way (`index.ts:1532`). First check
     that a surface replace may be appended while the step's batch is still open.
   - **Otherwise keep receipts small.** Keep each to one line and list it in the section map,
     so the next batch edit clears it.
6. **Token estimates need calibrating.** pi-clm found that chars/4 undercounts dense content
   by 1.8–2.5×. It keeps an EMA correction factor against the provider's counts (`budget.ts`,
   §7a). ph's `after=` costs, and any size gate, should come from `ctx.token_meter`'s
   provider-reported numbers. Check what the token meter offers before Phase 3.

## Open decisions

### 1. Tombstone: a visible marker or nothing

**What the references do.** Neither leaves a marker for the model.
- CLM drops an emptied turn (`_normalize`).
- pi-clm omits a removed block and records `removed` only in the revision's edit trace, for
  its panel (`types.ts:49`). Its continuity annotations are the opt-in pointer.

| option | pros | cons |
|---|---|---|
| **A. Visible marker.** A one-line user-role `PluginSource` notice replaces the run. | No core change. The model knows work happened there and why, which is the gap pi-clm's annotations exist to fill. Matches the word "tombstone". The reason is in context, not only in the log. | About 15–30 tokens each, accumulating unless merged. A user-role line in mid-conversation can read as an instruction. Neither reference implementation does it. |
| **B. Silent: an empty-content `assistant/message` as the replacement.** | Zero tokens, and it works today: it derives to `None` (`derive.py:48`). Compaction's `_plan` already skips nodes that derive to nothing, and older builds read it the same way, so no format bump. | Overloads a shape whose documented purpose is hosting a max-tokens step's usage. Every reader keyed on `assistant/message` has to be checked. Someone reading the log sees an empty model turn, not a deletion. |
| **C. Silent: absorb into the next section.** Replace the deleted run plus the following single-node section with a copy of that section. | Zero tokens. No core change and no new meaning for an existing shape. The edit point, and so the cache cost, is unchanged. | Works only when the next section is a single node (a user message, or text-only assistant text), so a deletion in mid-turn falls back to A: two behaviors for one verb. The copy duplicates a message in the log. The TUI dims a message that did not change. |
| **D. Silent and first-class:** a `remove` surface op, or a surface-eligible type that derives no message. | Explicit, and one behavior everywhere. | A surface-mechanism change, so `SESSION_FORMAT_VERSION` 4, and older builds refuse the log. Touches the fold, validation and the TUI. |

**Recommendation: A** for Phase 1, with consecutive tombstones merged into one marker.
Revisit **D** together with decision 3's operation if the measured marker cost matters. Skip
B, which overloads an existing shape. C is an optimization at most.

### 2. Replacement role

**What the references do.**
- CLM keeps assistant text as assistant and folds everything else into user turns
  (`parse_back`).
- pi-clm keeps the assistant role when an existing assistant block is edited. The edited
  message becomes text-only and loses its tool calls (`changedMessage`,
  `context-document.ts:318`).
- pi-clm lowers every new or relabeled block to user-role text prefixed with
  `[context role=X]` (`contextNote`, `:307`). Its rule: "authored roles are lowered to
  non-authoritative text" (§11).

The plan already keeps the assistant role for a one-node in-place rewrite. The open question
is only about substitutions over a range.

| option | pros | cons |
|---|---|---|
| **A. Always user role** (a `PluginSource` notice) | Never puts words in the model's mouth. Provenance is visible (plugin `clm`). No reasoning blocks or signatures to strip. Compaction and input-offload already do this. | The model's own conclusions ("I found X") arrive as user text and may be weighed as the person's claims. First-person summaries read oddly. |
| **B. Assistant role when the range held only assistant text** | The model keeps its own voice, as both CLM and pi-clm do for edited assistant text. Turns alternate naturally. | A plugin-written `assistant/message` needs a provenance convention, since there is no model and no usage. Reasoning blocks and signatures must be dropped. Applies only to the rare range with no tool steps and no user turns. |
| **C. The editor chooses the role** (pi-clm's `role=` label) | The most expressive. | pi-clm itself lowers any authored label to user text, so in practice this is A with a label. |

**Recommendation: A for substitutions, and assistant role for one-node in-place rewrites.**
This is exactly pi-clm's split. An edit to something the model said stays the model's. Text
that stands for several turns, or that is newly written, is a note.

### 3. Restore and insert

New from the source review: inserting new blocks (scratchpads, trackers, notes anywhere) needs
the same missing operation as restoring a multi-message range as separate messages. They are
one decision.

| option | pros | cons |
|---|---|---|
| **A. Flattened restore.** Replace the replacement with one message holding the originals' text. | No core change. Puts the text back where it was. | Tool structure is lost and the role becomes user. It is another middle edit, so the tail is read again. |
| **B. Recall at the tail.** `context_recall(section, max_tokens)` returns the originals as that call's result, as pi-clm's `live_context_recall` does. | No core change. Append-only, so it costs nothing in cache beyond the recalled tokens. Bounded. ph needs no annotation store, because every original is in the log by seq. | The text arrives at the tail, not in its place. The replacement stays, so the context holds both. |
| **C. An insert-after or unshadow surface op.** | Exact restore with structure. Also enables pi-clm-style insertion and reordering. | `SESSION_FORMAT_VERSION` 4. Adds to the fold, validation, the TUI, and tool-pairing checks for inserted structure. Every use is a middle edit. |
| **D. Fork from before the edit.** | Exists today, and exact. | Discards everything after the fork point. It is a branch, not a restore. |
| **E. Insert without C:** append the new text to an adjacent section's in-place rewrite (a user message's content, or a tool result's content, which core allows). | No core change. | A scratchpad glued onto someone else's message, or a tool result that is no longer the tool's output. |

**Recommendation: B in Phase 1, with A available as an explicit undo.** Hold off on C until
recorded sessions show that scratchpads or insertion matter. If it is adopted, one format bump
covers restore, insertion and reordering.

### 4. Edit gate default

**What the references and ph do.**
- CLM ships `fit` (`edit_gate.py`), with `shrink` as the strict mode.
- pi-clm's CLM mode has **no size gate**: "strategy is the model's (or a steering document's)
  concern, not the validator's" (§5, §11). Its conservative mode requires an edit to shrink
  the context.
- ph's limits ship unset, because "a limit nobody chose fires on somebody's longest
  legitimate turn" (`ph_stabilize/limits.py:284`). Compaction is the backstop at 0.85 of the
  window.

| option | pros | cons |
|---|---|---|
| **None** (pi-clm's CLM mode) | Needs no token estimate when the edit is made. Allows scratchpads, restores and clarifying rewrites. Matches ph's unset-by-default stance. Compaction catches overflow. | A careless edit can grow the context until compaction's lossy summary fires. |
| **Fit** (CLM's default) | Growth is allowed but bounded. It is the paper's default. | Needs an accurate count at the moment of the edit, and pi-clm saw chars/4 undercount by 1.8–2.5×. A wrong estimate refuses a good edit or lets a bad one through. |
| **Shrink** | Can never make the context bigger. | Blocks restores and every edit that adds clarity. Both reference implementations rejected it as the default. |

**Recommendation: no gate by default, with `fit` and `shrink` as config.** The receipt carries
CLM's "edit applied but it GREW context" note. This changes the plan's earlier recommendation
of `fit`, based on pi-clm's reasoning and ph's own stance on limits.

### 5. Floor breakpoint or two quantized checkpoints

pi-clm has nothing here, and the CLM harness relies on server-side prefix caching and SCR.
This is purely a ph question.

| option | pros | cons |
|---|---|---|
| **Two quantized checkpoints** (today, `anthropic.py:211`) | Shipped and measured. No plumbing. | Neither mark survives an edit before it. Before a far-back edit, the entries have gone cold, since the 5-minute TTL refreshes only on reads. |
| **A floor plus one quantized tail mark** | Every request reads the floor, so it stays warm, and edits above it keep it. The section map can mark which sections are cheap to edit. | Needs a hint from ph-clm to the adapter (a core request field plus the Anthropic adapter) and a floor-placement policy to tune. An edit below the floor loses it. Each move costs a 1.25× write. |
| **Automatic caching** (top-level `cache_control`) | One line of code. | It moves with the tail, does nothing for middle edits, and still uses a slot. |

Two facts tilt this decision:
- **The second checkpoint may be redundant.** The adapter's case for it is that "a single
  quantized mark goes dark exactly when it advances" (`anthropic.py:135`). That case doesn't
  mention the 20-position lookback, which should find the previous checkpoint's entry from a
  new mark four messages later. If a probe confirms this (`cache_read_input_tokens` on the
  request that crosses a boundary), the second slot can become the floor at no cost.
- **Only Anthropic needs it.** On OpenAI-compatible routes, the implicit cache already serves
  the longest unchanged prefix, in 128-token increments.

**Recommendation: unchanged. Measure first** (Phase 4), starting with the lookback probe.

Aside: the docstring on `MIN_CACHEABLE_TOKENS` (`anthropic.py:145`) quotes 1,024 and 2,048
tokens. The current docs list 512 for the 5.x models and up to 4,096 for some older ones.
Nothing enforces the value, so only the comment is stale.

### 6. Free edit steps

**What the references and ph do.**
- CLM exempts a turn that only edits the mirror (no output, exit 0) from `max_steps`. It
  still counts that turn toward `lm_call_cap` (`2*max_steps+24`).
- pi-clm has no step budget to exempt anything from. Its conservative mode allows one mirror
  write per turn (`mirror-guard.ts`).
- ph ships every ceiling unset. `modelCalls` counts each `step/start` and each `step/retry`
  (`limits.py`), so this question matters only where a deployment sets a limit.

| option | pros | cons |
|---|---|---|
| **Count every step** (today) | No change. An edit step is a real model call that costs money, and limits exist to bound that. | A model near its ceiling has a reason not to compact. |
| **Exempt edit-only steps** | CLM's incentive: keeping the context clean is free. | A model can loop on edits indefinitely, and the breaker counts failures, not successful edits. `limits` would have to classify a step as edit-only (its only effect a `clm/revised`). |
| **Exempt from the turn limit but keep a session cap** (CLM's split) | Keeps the incentive and still bounds a runaway. | Two numbers to explain, and it still needs the classification. |

**Recommendation: count every step for now,** since limits are unset by default. Add CLM's
split as a `limits` option only if a deployment with ceilings asks for it.
