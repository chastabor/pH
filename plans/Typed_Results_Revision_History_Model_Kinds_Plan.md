# Typed child results, revision history, and model kinds

Three follow-ups from the 2026-10-11 review of what the `/simplify` passes skipped, in the
order they will be built. Each is its own change with its own gate.

1. **[Typed child results](#1-typed-child-results)**: a child hands back a checked JSON value,
   not only prose, so a skill that delegates "categorize this text, image, audio or video"
   gets a structure its parent can rely on.
2. **[Revision history](#2-revision-history-like-git-log-for-the-context)**: every change to
   the context is paired exactly with the record of why it was made, and changes are grouped
   by the part of the context they touched, the way `git log -p -- file` groups a file's
   commits. For diagnosis, and for training data.
3. **[Model kinds](#3-model-kinds)**: the settings a model maker recommends, a shared set of
   kind names a skill can ask for on any profile, and a check that a child's model can take
   the media it is handed.

**Status:** section 1 done 2026-10-11, sections 2 and 3 done 2026-10-12 (see each
one's Status). **Done first, 2026-10-11** (recorded under S7b in
`Session_Profiles_Plan.md`):

- A child named onto a whole route (`provider`/`model`) resolves through the list like a
  key does.
- A skill's model key that the profile cannot run **falls back to the parent's model with
  a warning** rather than refusing (`ModelFallback`): the graceful-degradation pattern.
  The parent's model does the work as well as it can until the person sets up a profile
  and a provider for the key. Sections 1 and 3 build on that rule.
- **A readmitted child's route is worked out again**, not read back from its admission
  (the readmission open question in `docs/seams/subagents.md`, decided). The request
  carries what the spawn asked for (`choice: ModelChoice`) apart from what it resolved to
  (`entry: ModelEntry | None`), and so does the admission. A readmission degrades every
  choice that cannot run, the spawn's own and skills that disagree (`contested`)
  included, since nobody is there to choose again. So a model set up after a fallback
  reaches the child at its next start.

**Why this order:**

- **Typed results come first.** A categorization skill needs checked JSON whichever model
  runs it. With the fallback, it can run on the parent's model, so it works before model
  kinds exist.
- **Revision history is next.** Nothing in it depends on the other two, and diagnosis and
  training both need its exact pairing.
- **Model kinds come last.** They pick a better model for the job, but neither of the other
  two depends on them.

## 1. Typed child results

### What exists

- **A child returns text.** `SubagentResult` carries `status`, `answer` and `error`.
  `task`'s value carries `answer`. An `rlm.run` child replies by a message in its parent's
  inbox (`ChildNotice`, carried on the child's ending record).
- **It was promised and never built.** The port plan says `task` returns "structured JSON
  when an output schema is given" (`Python_Harness_Port_Plan.md`, `subagent-task`), but no
  spawn takes a schema.
- **Most of the pieces exist:**
  - A skill's `parameters:` is a JSON Schema, checked when the skill loads
    (`_parameter_schema`) and checked against the values at use (`validate_json_schema_value`).
  - A tool call's arguments are checked against the tool's schema, and a bad call goes
    back to the model as an error it can fix.
  - A tool result can end the turn (`ToolExecutionResult.concludes_turn`).
  - `ph.llm.structured` asks a single model call for a shape: `ask_for_shape`,
    `validated_shape`, and `structural_warning` for a schema too loose to constrain.
    `schema_of` accepts a pydantic model or a schema dict. But it refuses a call that also
    offers tools, so it cannot be used for an agent that works with tools.

### Design

**Declaring the shape:**

- A skill declares `output:` in its front matter, a JSON Schema read the way `parameters:`
  is (`Skill.output`), with `structural_warning` logged when it loads.
- `task` and `rlm.run` take an `output` schema too, for a one-off request with no skill
  behind it. A spawn's own schema beats its skill's.
- Python callers, such as an rlm cell or a plugin, may pass a pydantic model class wherever
  a schema is taken. `schema_of` turns it into the schema, and the value comes back as an
  instance of that class. So pydantic checks the structure before the caller sees it, as
  `ask_for_shape` does for a single call.
- `SubagentRequest.output` carries the schema, and the admission records it, so a
  readmitted child still owes the same shape.

**The child returns its value through a tool:**

- A child with an `output` is given one extra tool whose argument schema *is* the output
  schema. A valid call records the value and ends the child's turn. An invalid call is a
  tool error naming each violation, which the model fixes on its next step. That is the
  path every tool's arguments already take, so checking and correction come free.
- It works beside the child's other tools: the child researches, reads the image, then
  answers through the tool.
- The child's prompt says it must finish by calling the tool.
- Where a route can enforce a tool's schema on the wire (OpenAI's `strict` tools, a llama.cpp
  grammar), the tool may ask for it. That is optional, because the check runs either way.

**A child that ends without calling the tool** gets one reminder, as `ask_for_shape` sends
one correction. If it still has not called it, the child ends as an error ("no value"), with
its text kept as `answer`. The parent gets the prose and knows the structure is missing,
and it decides what to do.

**Where the value lives** (each session owns its log):

- The child's own log records the value on its ending record. `ChildNotice` carries it to
  the parent, and `SubagentResult.value`, `task`'s `value` and the rlm reply all hold it.
- A crash between the child's ending and the parent's read loses nothing: the sweep that
  brings the parent back delivers what its log lacks, as it does for an answer today.
- An rlm cell does not receive inbox messages, so it needs a way to get the value. Either
  a binding returns the checked value (an instance, given a pydantic class), or the roster
  row carries it. See the decisions.

**The check** is `validate_json_schema_value`, the same checker as tool arguments and skill
parameters, so a keyword it does not support is unsupported everywhere at once
(`CONSTRAINING_KEYWORDS`).

### Status (2026-10-11)

B1 to B3 are built, as `docs/seams/subagents.md` ("A value in a set shape") describes.
What changed from the design above, decided in the building:

- **A value also comes back as JSON text** (decided by Charles). A child that ends
  without calling the tool has its last reply read as the object (`validated_shape`, a
  fence allowed), since a JSON value is text in the end; only a child with neither, after
  the one reminder, fails.
- **The value has a record of its own, not a field on the ending.** `subagent/returned`
  (`record_returned`), written by the tool or by `settle_value`, so one reader serves a
  native call, a Code Mode dispatch and a text reply alike. The fold carries it to
  `ChildState.value` and the roster row; the notice carries it as text.
- **An `output` is an object at its root, with only keywords pH checks**
  (`object_schema_problems`), refused otherwise, at a skill's load and at a spawn, because
  the value is also a tool's arguments and a half-checked contract is none.
- **The tool is registered on the child's own scope**, which its own filter cannot mask,
  and `held_by` keeps it out of what that child can hand its own children.
- **The shape is read back at a readmission, not worked out again** (unlike the model):
  the parent was promised it.
- **The one reminder is an `agent/turn-stopping` steer**, the door a listener objects to
  a stop through, rather than the seam running the child's agent again; so
  `settle_value` only reads the log.
- **The ending door holds every provider to the contract.** `record_ended` asked for
  `done` settles the child itself and ends one with no value as `error`, and the seam
  makes the waiter's result follow the log (`_settle_waiter`), so a provider gets typed
  results by spawning through the seam and ending through the door; ph-rlm asks
  `settle_value` first only to word its own notices.

**B4 and the live gate are built too.** A cell passes a pydantic model class as `output`
and gets an instance back from `rlm.wait`, a new binding (`rlm_wait`) that waits for
one of its own children; the validator gained `anyOf`, so a model's `X | None` fields
check; and a cell's annotations are its own now (`compile_cell` stopped inheriting the
guest's `from __future__ import annotations`, which left every pydantic model in a cell
unbuildable). The live gate, `tests/test_local_typed_results.py` in `./test.sh smoke`,
sorts a report and a picture on the local Qwen through skills that declare `output:`;
the picture is attached by the child, on a route that takes images. On it the local Qwen
returned its values through the tool (`via: tool`). It also found that the harness's
refine job ignored its cancel token, so a refinement ran on past a disposed scope; it is
raced against the token now.

### Steps and gates

- **B1: Declared.** Skill `output:`, the spawn argument, the request and the admission.
  *Gate:* a skill's schema reaches its child's request and is in the admission; a schema
  that cannot constrain is warned about when the skill loads.
- **B2: Returned.** The tool, its check, the end of the turn, and the one reminder. *Gate:*
  a categorizing skill run through `task` returns a `value` that matches its schema; an
  invalid call is corrected; a child that never calls the tool ends as an error with its
  text kept.
- **B3: Durable.** The value is on the ending record, the notice, a crash's reconcile, and
  a readmission. *Gate:* a child that answered before a restart still gives its parent the
  value.
- **B4: In Python.** rlm cell access, and pydantic instances. *Gate:* a cell gets an
  instance of its own model class back.
- **Live gate:** a categorizing skill over a text and an image, on the local Qwen, added to
  `./test.sh smoke`.

### Decisions (made 2026-10-11)

- **The tool's name:** `submit_result`.
- **A child that never returns a value** fails, with its text kept (decided by Charles:
  the value is a contract the skill may require).
- **Two skills that declare different `output` schemas** are refused at the spawn, and the
  parent passes `output` to settle it.
- **How an rlm cell gets the value:** from the ending notice and the roster row for now; a
  binding that waits for it belongs to B4.

### What breaks

The admission gains `output`; a child's log gains `subagent/returned`, which the state and
the roster row read as `value`; `task` takes `output` and its value gains `value`; and
`rlm.run` takes `output`. All are new. A log written before this opens as it did. A
failed `task` now carries the child's last text in its error.

## 2. Revision history, like `git log` for the context

### What exists

- **The history is already in the log.** Every replacement names what it replaced
  (`replaces`, `sourceEventSeqs`), and the originals stay in the log (I4). `ph.session.revise`
  reads them back: `shadowed_by`, `origin_of`, `originals`, `messages_of`, `render_messages`.
- **Section names never move.** A ph-clm section is named by the seq its first node was
  written at (`S412`), so a section has a stable identity, as a file path does in git.
- **Records are paired with their replacements by guesswork.** `/diff` infers the pairing
  per batch (`ph_app.tui.diffs._batch`), and ph-clm reads its own record at `seq + 1`
  (`ph_clm.edits._edit_at`). Each producer writes its record in its own place:

  | Producer | Replacement | Record | Where the record is |
  |---|---|---|---|
  | compaction's summary | `substitute` | `compaction/summarized` | before it, same batch |
  | compaction's argument elision | many `rewrite`s | one `compaction/args-truncated` | after all of them |
  | the overflow clip | `rewrite` | `offload/spilled` (by call id) | before it, same batch |
  | input offload | `substitute` | `offload/input-spilled` (by the original's seq) | before it, same batch |
  | ph-clm | `substitute` or `rewrite` | `clm/revised` (by the replacement's seq) | right after each |

- **Sizes:** `/diff` reads any `<unit>Before`/`<unit>After` pair of integers, and only ph-clm
  writes one.
- **Two copies of the diff call:** ph-clm's `context_diff` (`tools.py`) and `/diff`
  (`diffs.unified`).

### Design

**1. Pair records exactly, when they are written.**

- `substitute` and `rewrite` take the producer's record (its type and payload). They write
  it right after the replacement in the caller's batch, and stamp it with the replacements
  it describes: `revises: [seq, ...]`.
- A record that covers several replacements, such as argument elision's one record for
  every argument it cut, is written once through a third door function naming all of
  them.
- One core reader, `records_of(session, seq)`, is indexed once per session (a fold cache).
  `/diff`'s inferred pairing and its "shared" flag are deleted, and ph-clm's `_edit_at`
  becomes a lookup in that reader.
- That is the door pattern: the rule lives in core, and the writers table holds every
  producer to it.
- **Decision:** where the link lives. The proposal is a payload key that core reserves on
  records written through the door, which needs no log format change. The alternative is
  an envelope field on records, which is cleaner to read but changes an envelope rule
  (`sourceEventSeqs` is forbidden on non-message types today).
- Compaction's summary record moves from before its replacement to after it. The readers
  that find it (the token meter's `SPENDING_TYPES`, goals, rlm's auto harness, the TUI
  adapter) look for it by type, and each is checked in H1.

**2. Sizes on the record.** The door's record takes an optional size (`unit`, `before`,
`after`), and `/diff` reads it in place of the field-name convention. Each producer fills
in what it knows: tokens for ph-clm and compaction, bytes for offload.

**3. One history reader.** `history(session, seq)` returns every replacement among a node's
ancestors, oldest first, following `shadowed_by`. Unlike git's one line per file, this
history merges: a summary of five nodes has five lines of history behind it. It is cached
per session.

**4. `/diff` grouped as git groups files:**

- Changes are grouped by where they live now: the node on today's surface, or the ph-clm
  section holding it.
- Each group switches between **each edit** (like `git log -p -- file`) and **the whole
  history against the original** (like `git diff A..B -- file`).
- ph-clm's `context_diff` reads the same history.

**5. One diff helper** in `ph.session.revise`, beside `render_messages`, taking its labels as
arguments, used by `/diff` and `context_diff`. The pairing passages in `diffs.py`'s
docstrings are rewritten along with the reader.

**6. Optional: an export for training.**

- One JSON line per change: before, after, producer, the record's fields, size, time, and
  for a ph-clm edit the model's call that asked for it.
- It reads a stored log through `ph.persistence.read_session`, as `diffs` already can, from
  a `phern` command.

### Status (2026-10-12)

H1 to H4 are built. **The decision was taken as proposed:** the link is a payload key
core reserves (`revises`), not an envelope field, so the log format is unchanged.

- **H1.** `substitute`, `rewrite` and a third door, `rewrite_many`, take the producer's
  record as a `RevisionRecord` (its own writer, its type, its payload) and write it
  right after the replacement, stamped `revises: [seq, ...]`. `LogWriter.append` refuses
  any other write that carries the key; the door writes through `_append_revising`.
  `Session.records_of(seq)` answers from an index (`RevisionIndex`) extended as the log
  grows. `/diff`'s batch pairing (`_landed`, `_batch`) and its `shared` flag are gone: a
  record that explains several changes says so by its own `revises`. ph-clm's
  `_edit_at` is a lookup, and `clm/revised` lost `replacement`. Every producer's record
  now follows its replacement, compaction's summary and the clip's `offload/spilled`
  included. The writers walk reads `RevisionRecord(<writer>, type, …)` as a write.
- **H2.** `RevisionRecord.size` (`RevisionSize`: tokens or bytes, before, after) is
  written by the door as the record's `size`: compaction's summary and argument elision
  and ph-clm in tokens, the clip and input-offload in bytes. `/diff` reads it in place
  of the `<unit>Before`/`<unit>After` convention, which is gone, and lists the door's two
  keys apart from the producer's fields. The gate is in `tests/test_revision_history.py`,
  which drives each producer for real.
- **H3.** `Session.history(seq)` lists every replacement behind a node, oldest first,
  from the same index, kept per node once worked out (what a node stands for never
  changes). `/diff` lists one group per surface node a revision led to, and `w` switches
  its details between each edit, newest first with its records and its own diff, and the
  whole history, the node as first written against the node now. `unified_diff` in
  `ph.session.revise` is the one diff helper, with its labels as arguments, for `/diff`
  and ph-clm's `context_diff`, which now lists the section's edits, oldest first, above
  its diff. *Departure from the design:* groups are surface nodes only, not ph-clm
  sections, because the app may not read a bundle's section map
  (`test_app_layering.py`); a section is a run of these nodes.
- **H4.** `phern revisions <id|path>` prints one JSON line per change, read from the log
  on disk with nothing mounted (`ph_app.stored.load_session`, which the trajectory
  viewer now shares): the change's shape, what it replaced and what it is now as
  `render_messages` renders both, its producer, its records whole, the door's size,
  and the tool calls its records name by `callId`. *Departure:* "the model's call that
  asked for it" is read generically, since the app may not import ph-clm, so the
  clip's row names the call whose result was clipped. `ph_app.revisions` reads a
  revision once for `/diff` and the export (`read_revision`); core holds what every
  reader shares: `shape_of`, `revises_of`, `RevisionSize.of` and `producer_data`. The gate is
  `tests/test_revision_history.py`, which also holds H2's.

### Steps and gates

- **H1: Records through the door.** Every producer moves, and `records_of` replaces the
  guessing. *Gate:* for each producer in the table above, `records_of` finds the record for
  its replacement, including argument elision's one record for many; `diffs._batch` is
  deleted; ph-clm's `_edit_at` uses the reader.
- **H2: Sizes.** *Gate:* `/diff` shows recorded sizes for compaction and offload as well as
  for ph-clm.
- **H3: History.** The reader, grouped `/diff`, the shared diff helper, and `context_diff`
  reading the history. *Gate:* a summary over two sections that ph-clm had edited shows both
  edits in its group; the whole-history diff equals the originals against the current text.
- **H4 (optional): Export.** *Gate:* the export of a session with every producer has one
  row per change, each with its record.

### What breaks

- Logs written before H1 have records with no `revises`, so `/diff` shows their changes
  with no records beside them, and ph-clm cannot restore an edit made before H1 (it
  answers that the section has not been revised). There is no shim.
- Compaction's summary record comes after its replacement instead of before it, and so
  does the overflow clip's `offload/spilled`.
- `clm/revised.replacement` repeats `revises` and is dropped from ph-clm's own record, and
  `tokensBefore`/`tokensAfter` move into the door's `size`.
- `substitute` and `rewrite` take a required record; a caller outside the shipped
  producers (a test, a plugin) passes one (`ph.testing.revision_record`).
- `/diff` no longer reads a `<unit>Before`/`<unit>After` pair off any record.

## 3. Model kinds

### What exists

- **A profile's `models` list maps keys to routes with settings.** `ModelRoute` carries
  `reasoningEffort`, `temperature` and `maxTokens`. A skill asks for a key (`model: classify`),
  a spawn names one, and a child runs with its route's settings (`child_options`).
- **The fallback already exists** (see Status): a skill's key that cannot run gives a
  `ModelFallback` whose reason is the code `choose` refused with (`UnrunnableChoice.reason`).
- **Media is declared per provider row** (`accepts`) and reported per route as
  `ResolvedModel.accepts`. Only the Google adapter accepts video. `choose` never asks
  whether a child's route can take what it will be handed.
- **The adapters send a setting only when it is set.** A local server's own flags decide
  the rest, which is how a llama-server started at temperature 0.6 runs at 0.6 while pH
  sends nothing.
- **A model kind beyond chat** (an embedding model for the text index) is noted under S7b,
  item 8, in `Session_Profiles_Plan.md`.

### Design

**1. More settings.**

- `ModelRoute` and `GenerateOptions` gain `topP`, `topK`, `minP`, `presencePenalty` and
  `frequencyPenalty`, which are what model makers' recommended settings consist of.
- Each adapter sends what its API takes:
  - **OpenAI-compatible:** `top_p` and the penalties; llama.cpp and vLLM also take `top_k`
    and `min_p` as extra body fields.
  - **Anthropic:** `top_p` and `top_k`.
  - **Google:** `topP`, `topK` and the penalties.
- A setting a route's API cannot carry is dropped with a warning and a `phern doctor` line,
  rather than refused (graceful degradation). Unset stays unsent.
- Recommended settings go on the key's entry, next to the model they were tuned for. A
  table of settings by model lineage in pH would duplicate the server's own flags and go
  stale.

**2. A shared set of kind names.**

- Document the kind names shipped skills may ask for, so a skill works on any profile, for
  example:
  - `classify`: short replies, low effort, steady settings;
  - `light`: cheap general work;
  - `vision`, `audio`, `video`: models that take that media.
- A profile lists the kinds it can serve. A kind it does not list falls back to the
  parent's model with a warning, which is the cue to set it up.
- Shipped profiles do not alias the kinds to their default model, because that would hide
  the warning.

**3. Media a child needs.**

- A skill declares the media it hands its child (`accepts: [image/*]`), or the spawn's
  attachments imply it. `resolve_model` asks `ctx.llm.resolve_model(provider, model).accepts`
  of the chosen route.
- A route that cannot take the media falls back to the parent's route if that one can
  (`ModelFallback.reason` gains `cannot-accept`, decided by the subagent seam rather
  than by `choose`).
- If neither can, the child runs with text placeholders and a warning. The `llama.yaml`
  comment records a model with no image input spending many steps trying to see one, so
  the warning has to reach the parent.

### Status (2026-10-12)

**A1 to A3 are built.** The decisions are below; the graceful-degradation rule settled
two of them as proposed, and Charles settled the third.

- `ModelRoute`, `AgentOptions`, `LlmCallConfig` and `GenerateOptions` gain `top_p`,
  `top_k`, `min_p`, `presence_penalty` and `frequency_penalty` beside `temperature`,
  flat as `temperature` is, so a logged header is unchanged where none is set. Each
  layer copies them as it does `temperature` (`ModelEntry.options`, `seed_config`,
  `_build_request`, compaction's `_replay`); `sampling_of` reads them by the name a
  route spells (`SamplingSetting`).
  - *Since consolidated (2026-10-13):* the six are one value, `Sampling`, which each
    layer carries whole as `sampling` and which holds the rules over them
    (`settings`, `uncarried`, `without`, `on_wire`). The drop rule the seam wrote twice,
    for the logged header and for a request reaching it another way, is
    `Sampling.without` for both. `Sampled`, `SamplingFields`, `FIELD_OF` and the free
    functions are gone. On the wire it nests (`sampling: {temperature, topP}`), and an
    unset one is absent, so a route or header that sets none is written as before; the
    session format is 4, and a format-3 log is refused.
- `ResolvedModel.sampling` says what a route's API takes; `resolved()` requires it.
  OpenAI-compatible: `temperature`, `topP` and the penalties, plus `topK` and `minP`
  where a profile lists them in `samplingExtras` (the shipped `llama.yaml` does);
  Anthropic: `temperature`, `topP`, `topK`; Google: all but `minP`. Each adapter
  sends only those, under its wire's names.
- The `llm` seam drops any other before the request (so `llm/send` sees what is
  sent) and warns once per route and setting; the `models` row adds a "Model
  settings" section to `phern doctor` naming each listed key that drops one. *Since
  moved* ahead of the logged header, which now records what it withheld; see
  "Recorded, not planned".
- *Found:* no adapter sends `reasoningEffort`; see "Recorded, not planned" (since
  built as a route's `efforts`).
- **A2.** `MODEL_KINDS` (`ph.seams.models`) names the shared kinds, `classify` and
  `light`, with the job each is for; the `models` row adds a "Model kinds" section to
  `phern doctor` naming the route each is listed on, or that a skill asking for it runs
  on its parent's model. ph-core ships `categorize` (`ph/shipped_skills`), which asks
  for `classify` and owes `{category, reason}`, offered by a row that registers a
  subagent provider (`offer_shipped_skills`, from ph-rlm's `rlm-subagent-provider`),
  so its catalog line appears only where a child can be started. The skills README's
  front-matter table gains `model`, `output` and `accepts`, and `docs/seams/models.md`
  the kinds.
- **A3.** A skill states the media it hands its child (`accepts: [image/*]`,
  `Skill.accepts`, at most 16, each a type or family). `resolve_model` asks them of the
  route it resolved to (`_with_media`): a route that cannot take them gives way to the
  parent's where that can (`ModelFallback.reason` `cannot-accept`, with `media`); one
  the spawn named itself is refused, the rule for every spawn choice; where neither
  can, the child runs and sees placeholders, and a `MediaShortfall` reaches the
  admission (`mediaShortfall`) and the parent's note. A readmission asks again and
  degrades. *Not built:* media "the spawn's attachments imply", since a spawn request
  carries none. `categorize` states no `accepts`, since it sorts text as well; a skill
  that sorts only pictures would.

### Steps and gates

- **A1: Settings.** *Gate:* each adapter's request body carries what its API takes and
  nothing unset; `phern doctor` reports a setting a route drops.
- **A2: Kind names.** Documented, with the skills cookbook updated. *Gate:* a shipped
  categorizing skill names `classify`; on a profile without it, the child runs on the
  parent's model and the parent is told.
- **A3: Media.** *Gate:* an image-categorizing child whose kind's route takes no images
  falls back to a parent that does; with neither able, it runs and the parent is warned.

### Decisions (made 2026-10-12)

- **Media are requirements, not kinds** (decided by Charles): kinds name the job
  (`classify`, `light`), and a skill states its media with `accepts`, so a cheap
  classifier that takes images is one key and one list.
- **A setting a route cannot carry is dropped with a warning**, as proposed, by the
  graceful-degradation rule.
- **A child no route here lets see its media runs with placeholders and its parent is
  warned**, as proposed, by the same rule; a route the spawn itself named that cannot
  take them is refused, since the parent can choose again at once.

### What breaks

- `ph.llm.adapter.resolved()` requires `sampling`, so an adapter outside this repository
  that calls it must say what its wire takes.
- `ModelRoute`, `AgentOptions`, `LlmCallConfig` and `GenerateOptions` gain five optional
  fields; a header with none set is written as before. *Then (2026-10-13)* one
  `sampling` field instead: a route's sampling moves under `sampling:` in its YAML, a
  flat `temperature` on a route is refused, and every log from before is refused as
  format 3. Code building these types passes `sampling=Sampling(...)`.
- A route on the OpenAI wire sends `top_k` and `min_p` only where its profile lists them
  in `samplingExtras`; the shipped `llama.yaml` does, and a person's own overlay of a
  llama.cpp or vLLM route has to add them.
- Wherever ph-rlm's subagent provider is mounted, the skill catalog gains `categorize`,
  so those profiles' prompts gain a line.
- `ModelFallback.reason` gains `cannot-accept` (with `media`), and the admission
  `mediaShortfall`; a log from before opens as it did.
- A skill's front matter gains `accepts`; a malformed one refuses the skill.

## Recorded, not planned here

- **No adapter sent `reasoningEffort`: built 2026-10-12 as a route's `efforts`** (found
  building A1). A route's or a spawn's effort reached `GenerateOptions` and the footer
  (`reasoning_reading`) and stopped there: none of the three adapters put it in a body,
  and nothing warned. Each API spells it differently (OpenAI's `reasoning_effort`,
  Anthropic's `thinking.budget_tokens`, Google's `thinkingConfig.thinkingBudget`,
  llama.cpp's `chat_template_kwargs`), and the spelling is the model's as much as the
  wire's.
  - *What was built* (Charles asked for the warning and a profile field per model):
    `ModelRoute.efforts` maps each effort name to request body fields, per model in the
    `models` row. The agent loop looks them up for the route it sends on, after the
    `agent/request` waterfall, in the session's own list (`ModelList.efforts_for`: the
    agent's key where it names that route, else the first key that does), sets the
    chosen effort's fields on `GenerateOptions.effort_fields`, and each adapter merges
    them into its body key by key after its own (`wire_effort`). An effort the route's
    `efforts` do not define, its own or one a spawn asks of a child, is not sent: a
    warning once per route and effort, the header's `withheld` (below), and a line under
    "Model settings" in `phern doctor`. A route a plugin moved the request to is
    answered as the list holds it, and one it does not list has no efforts. An
    `llm/stream` listener that moves the route owns the `effort_fields` it passes on, as
    it owns `max_tokens`, since the request names no effort to look up there.
    `ResolvedModel.reasoning`, a field nothing read, is gone, and so are
    `GenerateOptions.reasoning_effort` (the fields say what is sent; the header keeps
    the name) and `EpochHeader.adapter_defaults` with `LlmCallConfigAdapterDefaults`
    (no adapter ever reported a default).
  - *The provider's default* (Charles, the same day): leaving the field out is how
    each API falls back to its own default, so an effort nobody asked for sends nothing
    and warns of nothing, and `default` (`DEFAULT_EFFORT`) asks for it by name, over an
    effort the route sets: a route's `reasoningEffort`, a spawn's effort (rlm's
    `thinking`, whose description now says so) or a skill's new `effort` front matter
    may name it. It sends nothing, is not withheld or warned about, and `efforts` cannot
    define it, since it is the provider's and not pH's. The warning stays for an effort
    someone asked for that the route has no mapping for. A child runs at the spawn's
    effort, else one its skills agree on (`SubagentRequest.skill_effort`, resolved by
    `resolve_model` at a spawn and a readmission), else its route's; skills that
    disagree leave it to the route, with a warning. Effort names are words, as model
    keys are.
  - *Not built:* any default per wire. No shipped profile defines `efforts`; the
    `llama.yaml` comment shows a Qwen3 overlay. A spawn's parent is not told that the
    effort it asked of a child was withheld; the child's header and the warning say so.
    A route no `models` list holds (a mount without the row, or a whole route a spawn
    named that the list does not match) has no efforts, so its effort is withheld.
  - *What breaks:* a route that sets `reasoningEffort` without `efforts` no longer shows
    the effort in the footer or the logged header (it never reached the server); it
    shows under `withheld` instead. A profile whose `efforts` define `default`, or name
    an effort by anything but a word, is refused; so is a skill whose `effort` is not a
    word. rlm's spawn tool schema gains a description for `thinking`, so prompts that
    carry it gain a sentence. A plugin reading `GenerateOptions.reasoning_effort`, or
    building an `EpochHeader` with `adapter_defaults`, fails; a header logged before
    reads as before, since nothing ever wrote the field.

- **A dropped sampling setting left no record, and the header claimed it was sent:
  fixed 2026-10-12.** The `llm` seam dropped it at the edge, after the agent loop had
  logged `request/header` with it, so the log said it was sent.
  - *What was built:* the loop asks the route before it logs the header
    (`LlmRuntime.withhold`), so `config` is what was sent and `EpochHeader.withheld`
    holds each sampling setting or effort asked for and not sent, by value. The next
    request asks for them again (`_request_proposal`) and checks them against the route
    then, so a route that comes to carry one sends it. The seam's edge drop stays for
    requests that do not come through the loop, and both warn through one set, once.
  - *What breaks:* a logged header no longer holds a setting its route cannot carry;
    readers of `config` see what was sent, and `withheld` is a new optional field, so a
    log from before reads as it did.

- **A request larger than the window was sent before it was shrunk: fixed 2026-10-12**
  (found by section 1's live gate). On the local llama slot (16,384 tokens), one cell's
  output, capped at a fixed 64 KB (`max_log_bytes`), can outgrow the window by itself,
  and a summary leaves the latest turn whole, so the next request (about 26,000 tokens)
  went to the server, which refused it; only then did the overflow recovery clip it.
  - *What was built* (the option chosen, of three: scaling `max_log_bytes` to the
    window, checking the request against the window before sending, or mounting the
    offload row in `rlm`): `compaction-summarize` listens on `agent/request`, where
    input-offload revises too, once the step's batch is in the log and before the loop
    derives the messages, and asks whether the token meter's estimate of the request
    fills the whole window (`_past_the_window`, trigger `window`). If so it runs the
    overflow remedy first: the clip of the step's results, then a summary in the direct
    shape if it still does not fit. The step's own pressure check leaves such a request
    to it, so the clip comes before a summary is paid for. No refusal is faked, so
    `limits` counts no model call and the log shows no retry; the estimate is the
    meter's, the provider's last count plus what was added since.
  - *Found on the way, fixed with it:* the recovery after a refusal clipped only results
    at the very end of the context, but a step's batch can come after them (rlm's
    session facts, added again once a summary has shadowed them), so the clip found
    nothing and the turn ended in the refusal. `_trailing_results` now looks past the
    batch's user messages.
  - *Live check* (a throwaway probe, not kept): a cell printing 65 KB on the local Qwen
    was clipped before the next request, which was about 2,700 tokens, with nothing
    refused; with the check disabled, the refusal was recovered by the clip past the
    session facts.
  - *The prefix benchmark moved with it* (`docs/dev-notes/prefix-cache-benchmark.md`,
    findings 5 and 6, and the headline): at the 8,192 window `rlm-stable` now clips
    each oversized cell result before sending instead of summarizing every turn, so
    it sends 88 % fewer tokens, keeps its cached prefix (75.9 %, was 22.9 %), and its
    uncached tokens fall from about 36,000 to 4,120. The summary's trade against the
    prefix still applies when the history itself outgrows the window.
  - *Cost of the estimate:* without `tiktoken` the meter counts four characters a
    token, which under-counts digit-heavy output on Qwen; a request it misses is still
    refused once and recovered as before.

- **A drain that times out on a running refinement fails the teardown: fixed
  2026-10-12.** Seen in the live probe above: a summary makes an automatic refinement
  due (`harness.auto.due`), `turn/end` starts it, and with no task group bound to the
  jobs seam it runs on `ctx.detach`, so `ctx.drain()` waits on its model call (minutes
  on a local 27B model) until its deadline. A `CancelledError` then reached pytest's
  runner (and a `ClosedResourceError` after it).
  - *The cause:* `drain` waited with `asyncio.gather`, which hands its waiter's
    cancellation (the deadline, as anyio's cancel of the draining task) to every task
    it waits on and, once they end, raises a `CancelledError` made from the last of
    them. The pass sat in `raced`'s anyio task group, which claims a cancel that reads
    as anyio's and returns, so the job ended as if finished and gather's error carried
    no message. The drain's scope claims only its own cancellation, so the error
    escaped, and `ph_app.runtime.mounted` skipped the `dispose` behind it: leases,
    worktrees and kernels left held. Any detached task running a task group at the
    deadline did the same; the shielded scope around a plain `gather` that was checked
    before had no such task in it.
  - *What was built:* `drain` waits with `asyncio.wait`, which hands its cancellation
    to nothing and raises nothing a task did. Its deadline now ends the wait and
    nothing else: what is still running is left to its owner, whose `dispose` stops
    it (a refinement through its token, a child's drive by the provider's `suspend`,
    which keeps its inbox). A stop added in drain at its deadline was tried and
    dropped: it cancelled owned work before its owner's own stop, so a child's drive
    would dispose its agent, clearing its inbox, before `suspend` reached it (read
    from the code, not reproduced). Found with
    it: a pass stopped at a mount's unwind ends after the jobs row has gone, and its
    `require(JOBS)` made it `failed`; it reads the row with `get`, as `_quiesce` does.
    The gates: `test_cordis_context.py` and `test_harness_planner.py`, which holds an
    automatic pass at its model call through a drain and a `dispose`.
  - *What changes:* a teardown that runs out still spends the whole drain share on a
    pass (`DRAIN_SECONDS`) before `dispose` stops it.
  - *Then every detached task got an owner: built 2026-10-12.* Without one, a task
    still running past the drain's budget outlived the teardown, as an async listener
    that never ended would. `Context.detach` takes a required `at_dispose`
    (`AtDispose`: `cancel`, `finish`, `leave`) and registers the stop as an effect of
    the scope it is called on, so it unwinds in its place. `emit` detaches a
    listener's task on the scope that registered the listener, with `cancel`; the
    one shipped async `emit` listener is the harness's auto-refine check, on its row.
    The schedule index writer uses `finish`, so its row's dispose waits for a batch
    already promised. Jobs use `leave`: their owner stops them through the token.
    This replaced the planned last-resort sweep, since every listener has a scope.
    - *What it needed:* a scope still hears events while it unwinds, until its
      listeners' `off` runs, so the stop is appended past `add_disposer`'s refusal of
      a scope that is down; `_unwind` then removes the effect it ran rather than
      `pop()`ing whatever came last. `emit` spawns a listener's task inside
      `_invoke`'s binding instead of wrapping it in `_as_owner`, since a task copies
      the context it is created in: a stop before its first step then means it never
      runs, where a wrapped one started only to be cancelled, or was reported never
      awaited.
    - *The gates:* four tests in `test_cordis_context.py`; `test_schedule.py`, where
      a write held behind the lock lands before the row's dispose returns; and
      `test_runtime.py`, where the never-settling listener is stopped by the root.
  - *Then jobs left the drain's pool: built 2026-10-13.* A teardown spent up to
    `DRAIN_SECONDS` on every running job before its owner stopped it anyway, and a job
    rarely finished in that window (a refinement is a review call and a plan call,
    minutes on a local model). The wait was not a graceful shutdown: the graceful stop
    is the owner's, in `dispose`, and keeps the work. A child is suspended with its
    inbox and readmitted later, and a refinement writes nothing and is considered
    again at the next turn. `detach(..., at_dispose="leave")` now keeps a task out of
    what `drain` waits for, since whoever stops it decides how long it gets; jobs are
    the one `leave` caller. Nothing new waits for a stopped job to wind down, and
    nothing did before: the owners' stops do not wait, and the wind-down has nothing
    durable left to do.
    - *What changes:* `phern -p` no longer waits on a working child or a pass; the
      print-run test now asserts the run leaves before the drain window. Work that
      would have finished inside the window is redone instead: the child's
      interrupted step on readmission, the pass at the next turn.
    - *The wait, for a caller that wants one:* `JobService.settled()` waits out every
      body started, counted rather than read from the table, which drops an abandoned
      job while its body may still run. `Context.settled()` is drain's wait without
      its shield and deadline, so a test's own deadline reaches it, and
      `ph.testing.settle_background(ctx)` loops the two under one; the job,
      subagent, messaging and planner tests that used `drain` to wait use it now.
    - *The gates:* `test_cordis_context.py` (a drain does not wait on `leave` work),
      `test_seams.py` (`settled` counts bodies), `test_harness_planner.py` (a
      teardown stops an automatic pass without waiting) and `test_modes.py`.

- **A fallback badge on a child's row in the TUI.** `ChildRow` shows a downgrade today;
  a model fallback reaches the parent's tool result and the log, but not the panel.
- **"Fork from before this change" in `/diff`:** recorded in
  `Ph_Clm_Context_Editing_Plan.md`, Phase 5 as built.
- **A record of what each readmission resolved to.** A readmission writes no admission,
  so its fallback reaches the warning and the readmitted run, and the child's own
  `request/header` records each request's route; but the roster (`ChildState.to_wire`)
  still shows the admission's model and fallback.
  - *The change:* the resolved route (`modelKey`, `route`, `modelFallback`, the run's
    model) rides on the `resumed` status every restart already writes
    (`record_started`), and the fold reads the latest. That door is called by the
    provider, so the seam would hand it the facts on the run rather than the provider
    spelling them.
  - *Cost of waiting:* a roster row can name the model a restarted child was admitted on
    after the person changed the list. Nothing reads a readmission's fallback yet; the
    TUI badge above would.
- **An assigned profile's model key is fixed at admission.** `resolve_profile` folds the
  profile's key into the spawn's `choice`, so a readmission resolves that key again but
  does not read the profile again, while it does ask the skills again. Reading the named
  profile at each start would make it follow the profile as it does the skills; it would
  need the profile composed again at readmission.
