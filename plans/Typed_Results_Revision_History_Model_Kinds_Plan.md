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

**Status:** not started. **Done first, 2026-10-11** (recorded under S7b in
`Session_Profiles_Plan.md`):

- A child named onto a whole route (`provider`/`model`) resolves through the list like a
  key does.
- A skill's model key that the profile cannot run **falls back to the parent's model with
  a warning** rather than refusing (`ModelFallback`): the graceful-degradation pattern.
  The parent's model does the work as well as it can until the person sets up a profile
  and a provider for the key. Sections 1 and 3 build on that rule.

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

### Decisions to make

- The tool's name. It must not collide with any shipped tool, and should read as "this is
  my answer".
- A child that never returns a value: an error with its text kept (proposed), or `done`
  with a note.
- Two skills given to one child that declare different `output` schemas: refuse, as two
  different models are refused (proposed), or require the spawn to pass `output`.
- How an rlm cell gets the value: a binding that waits for it, or a field on the roster row.

### What breaks

The admission gains `output`, the ending record and `ChildNotice` gain `value`, and `task`'s
value and the rlm reply gain `value`. All are new fields. A log written before this opens
as it did.

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
  with no records beside them. There is no shim.
- Compaction's summary record comes after its replacement instead of before it.
- `clm/revised.replacement` repeats `revises` and is dropped from ph-clm's own record.

## 3. Model kinds

### What exists

- **A profile's `models` list maps keys to routes with settings.** `ModelRoute` carries
  `reasoningEffort`, `temperature` and `maxTokens`. A skill asks for a key (`model: classify`),
  a spawn names one, and a child runs with its route's settings (`child_options`).
- **The fallback already exists** (see Status): a skill's key that cannot run gives a
  `ModelFallback` whose reason is the code `choose` refused with (`ModelChoiceError.reason`).
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

### Steps and gates

- **A1: Settings.** *Gate:* each adapter's request body carries what its API takes and
  nothing unset; `phern doctor` reports a setting a route drops.
- **A2: Kind names.** Documented, with the skills cookbook updated. *Gate:* a shipped
  categorizing skill names `classify`; on a profile without it, the child runs on the
  parent's model and the parent is told.
- **A3: Media.** *Gate:* an image-categorizing child whose kind's route takes no images
  falls back to a parent that does; with neither able, it runs and the parent is warned.

### Decisions to make

- The kind names, and whether `vision`, `audio` and `video` are kinds or requirements a
  skill states with `accepts`.
- A setting a route cannot carry: drop it with a warning (proposed), or refuse it.
- What a child does when no route here takes its media: run with placeholders and warn
  (proposed), or refuse.

## Recorded, not planned here

- **A fallback badge on a child's row in the TUI.** `ChildRow` shows a downgrade today;
  a model fallback reaches the parent's tool result and the log, but not the panel.
- **"Fork from before this change" in `/diff`:** recorded in
  `Ph_Clm_Context_Editing_Plan.md`, Phase 5 as built.
- **A spawn request that carries a `ModelChoice` instead of loose model fields.**
  `SubagentRequest`'s `provider`, `model`, `model_key` and `route` mean both "what was
  asked" and "what it resolved to". So `child_route`, `child_model_key`, `child_options`
  and `resolve_model` each work out which case they are in, and a half-named route needs
  a guard that `ModelChoice` would refuse when built.
  - *The change:* the request carries `choice: ModelChoice` and `entry: ModelEntry | None`,
    plus `model_fallback`.
  - *Blocked on a decision:* a readmitted child on its parent's route is rebuilt as a
    pinned model with its parent's settings as they are now. Neither type can say that
    without pinning the parent's settings at admission, which reverses the documented
    "inherits again" (the open question on readmission in `docs/seams/subagents.md`).
  - *Cost of waiting:* none a user sees. No shipped tool names a whole route.
- **A readmitted run's handle has no `model_fallback`.** The admission keeps it, and a
  crash's reconcile reads it from there, but `_request_of` and `_readmit_one` do not
  carry it onto the run a readmission builds.
