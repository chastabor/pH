# Self-steering skills

A **self-steering skill** is an ordinary `SKILL.md` that also declares `steps:`.
Reading it seeds those steps into the session's todo list as entries the model
**may mark done and may not delete**, and a listener on the agent loop's own
`agent/turn-stopping` boundary objects while any of them can still be started.
The model does not stop at step two and declare victory, because the turn that
tried to stop is told what is left.

Nothing here is a new mechanism. `/autonomous` already steers on that boundary,
`tool-todo` already owns an ordered list with dependencies, and G9 already keeps
a skill's body on disk until the model asks for it. This is those three wired
together, plus the one thing that had to be added: **provenance**, so the list a
skill contributed to survives the model's next whole-list write.

Rows: `skills-progressive` (ph-core), `tool-todo` and `skill-steps`
(ph-stabilize). Implementation: `ph/seams/skills.py`,
`ph_stabilize/todo.py`, `ph_stabilize/skill_steps.py`. Plan rows P7-16, P7-17,
P7-18.

---

## 1. Turn it on

Three rows. `skills-progressive` is mounted by `base.yaml` and **scans nothing**
until you name a directory; `tool-todo` and `skill-steps` ship **disabled** and a
profile arms them.

```yaml
# your-profile.yaml

# G9. Empty `paths` is deliberate: a well-known directory scanned at every start
# would make "install a skill" mean "drop a file somewhere", and a skill is
# something a distribution or a person installs on purpose (I7).
- id: skills-progressive
  config:
    paths:
      - ~/.ph/skills
      - ./.ph/skills          # later paths shadow earlier ones by name

# The list itself. `skill-steps` seeds into *this* row's list and is useless
# without it.
- id: tool-todo
  disabled: false

# The seeding and the steer.
- id: skill-steps
  disabled: false
```

Or, without editing a profile:

```
ph -p "…" \
  --patch '{id: tool-todo, disabled: false}' \
  --patch '{id: skill-steps, disabled: false}' \
  --patch '{id: skills-progressive, config: {paths: [~/.ph/skills]}}'
```

`skill-steps` ships `disabled: true` for `tool-todo`'s reason and one more: **it
can keep a turn going.** A row that spends model calls on the deployment's behalf
is a posture a profile chooses rather than one it inherits by mounting a bundle.

Check what you got with `ph --dump-config`, and what the model was offered with
the deployment's `tools/list`.

---

## 2. Write the skill

A skill is a directory whose name matches the `name:` in its frontmatter:

```
~/.ph/skills/
  port-a-feature/
    SKILL.md
```

A `SKILL.md` is a **Markdown file with a YAML frontmatter block** — `---`, the
declaration, `---`, then ordinary prose. The frontmatter is what pH parses at
discovery; the prose below it is the body the model reads when it calls the
`skill` tool, and pH never interprets it beyond substituting `{{parameters.x}}`.
The example below is mostly frontmatter because that is the half this document is
about; a real skill is mostly body.

```markdown
---
# ── frontmatter: YAML, parsed at discovery ──────────────────────
name: port-a-feature
description: Port one feature from a vendored upstream onto pH's own seams.
version: 2
argument-hint: <upstream-path> [--dry-run]
allowed-tools: [read, grep, glob, edit, write, bash]
parameters:
  upstream:
    type: string
    required: true
    hint: Path under sources/ holding the implementation to port.
  gate:
    type: string
    default: "uv run pytest -q"
    hint: The command that must exit zero before the port counts as done.
steps:
  - "Read the upstream implementation under {{parameters.upstream}} and write down what it actually does."
  - "Name what pH already has under another name, and what is genuinely missing."
  - "Implement the missing half on pH's seams — not the upstream's shape."
  - "Write a gate for each new rule, and prove it fails under a named sabotage."
  - "Run {{parameters.gate}} and report the result verbatim."
# ── end of frontmatter; everything below is the body ─────────────
---

# Porting a feature

Read the upstream at `{{parameters.upstream}}` before you change anything. The
steps above are the procedure; everything here is how to carry each one out —
which files to look at first, what "already has it under another name" tends to
look like in this codebase, and what a good sabotage is for a gate.

## 1. Read the upstream

…
```

### Frontmatter reference

| Key | Required | What it is |
|---|---|---|
| `name` | **yes** | Slug, ≤ 64 chars, and it **must equal the directory name** — otherwise the catalog addresses one string and the disk holds another. |
| `description` | **yes** | 1–1024 chars. This is what rides the prompt every turn; make it say when the skill *applies*. |
| `version` | no | `[0-9A-Za-z][0-9A-Za-z.+_-]{0,31}`. Reported, never compared — it tells the model which of the shadowed copies it just read. |
| `argument-hint` | no | ≤ 200 chars, rendered in the catalog as ``Usage: `name <hint>` ``. The author's one line for the common case. |
| `allowed-tools` | no | ≤ 32 names. **A declaration, not a boundary** — see §6. |
| `parameters` | no | ≤ 16 inputs. See below. |
| `steps` | no | ≤ 32 non-empty, **distinct** strings. This is the half that makes the skill self-steering. |

**A malformed value refuses the whole skill** rather than being dropped, and the
reason is diagnostic: a skill installed with `allowed-tools` silently empty
gives its author no way to tell "pH ignored my list" from "pH does not support
lists". The refusal is a `WARNING` on the `ph.seams.skills` logger naming the
file and the field — one bad directory never costs a deployment its other
skills, so if a skill does not appear in the catalog, read the log first.

### `parameters`

Authored in the friendly per-input form; converted to JSON Schema, so the checker
is `validate_json_schema_value` — the one pH already has — rather than a second
answer to "is this input acceptable".

```yaml
parameters:
  tag:
    type: string          # string | number | boolean
    required: true
    hint: Shown to the model when its arguments are refused.
  mode:
    type: string
    enum: [fast, thorough]
    default: thorough     # a default satisfies `required`, so give one or the other
```

The model calls `skill(name="port-a-feature", arguments={"upstream": "sources/x"})`.
Before it reads a word of the body, four things are refused: a missing required
input, a value outside its `enum`, a wrong type, and **an argument the skill
never heard of** — `additionalProperties: False` is the point of declaring at
all, since silently ignoring a `--dry-run` misspelled `dryrun` is how it runs for
real. The refusal names what the skill declares, so the second call is right
without opening the file.

`{{parameters.<name>}}` is substituted in the body **and in each step's text**,
with defaults applied first. The interpolation is deliberately narrow — one
prefix, one regex — because a `SKILL.md` is prose *and* code samples, and a
general template pass would make a skill unable to contain an example of itself.
Every other brace in your file survives byte for byte. A placeholder naming a
parameter the frontmatter never declared is refused at read time — anywhere in the
file, since the model is handed the whole of it — and read time is the first
moment that holds both halves (G9 keeps the body on disk until then).

---

## 3. What happens when the model reads it

1. The `skill` tool returns the body with parameters filled in, plus `version`,
   `parameters`, `allowed_tools` and `missing_tools` — the last **resolved
   against the caller's own scope**, so it answers "can I actually follow these
   instructions from here?" rather than echoing the declaration back.
2. ph-core emits `skills/read`. It emits rather than acts, because turning a
   declaration into a plan belongs to `tool-todo`, which lives in a package this
   seam cannot import and a row a deployment may not have mounted.
3. `skill-steps` appends each step to the end of the todo list as
   `{content, status: "pending", requires: [<previous step>], source: "skill"}`
   and writes one `todo/write` event.

Appended rather than merged into position: the model's own entries stay where it
put them, and a procedure that arrives mid-session is work added to the end
rather than a plan rewritten underneath somebody. `requires` chains **within one
skill only** — two procedures read in one session are two orderings, not one
queue.

Seeding is **idempotent by content**. Re-reading a skill adds nothing; editing a
skill to add a step mid-session adds only the new one. If the model has already
written a todo whose text is identical to one of the steps, the skill seeds
*nothing* and logs why — `requires` names entries by their content, so two
entries with one name is a plan the next `write_todos` cannot reproduce.

Seeding is also **all or nothing, and bounded by what the list can hold**. A step
longer than 500 characters, or one that would push the list past 100 entries,
seeds nothing and logs which skill. That is not fussiness: an entry appended past
those bounds makes *every* later `write_todos` fail validation while the
no-dropping rule refuses any write that removes it — the model would be locked
out of its own plan with the row steering it onward. Note that a step's length is
not entirely yours to control, since `{{parameters.x}}` renders a caller-supplied
value into it.

### What progress looks like

**You never write a checklist — you get two, and they are the same list.** Both
are folds of the `todo/write` events, so they cannot disagree (A11); they differ
only in what each reader can act on.

The `port-a-feature` skill above, seeded into a session that already had one
entry of its own, at four moments:

```
 the person's sidebar          the model's prompt section
 (TUI, "todo" panel)           (rebuilt every turn)

 ─── just after the skill is read ────────────────────────────────
 ● read the plan row           [x] read the plan row
 ○ survey the callers          [ ] survey the callers
 ○ port the row                [ ] port the row (waiting on: survey the callers)
 ○ gate it                     [ ] gate it (waiting on: port the row)

 ─── the model starts step 1 ─────────────────────────────────────
 ● read the plan row           [x] read the plan row
 ◐ survey the callers          [~] survey the callers
 ○ port the row                [ ] port the row (waiting on: survey the callers)
 ○ gate it                     [ ] gate it (waiting on: port the row)

 ─── step 1 done, step 2 running ─────────────────────────────────
 ● read the plan row           [x] read the plan row
 ● survey the callers          [x] survey the callers
 ◐ port the row                [~] port the row
 ○ gate it                     [ ] gate it (waiting on: port the row)

 ─── finished ────────────────────────────────────────────────────
 ● read the plan row           [x] read the plan row
 ● survey the callers          [x] survey the callers
 ● port the row                [x] port the row
 ● gate it (no work seen)      [x] gate it
```

Three states, one vocabulary: `○ [ ]` pending, `◐ [~]` in progress, `● [x]`
completed. The first entry is the model's own — seeded steps sit beside it and
are not marked out, because the plan is one plan.

Each side carries what its reader can do something about, and only that:

* **The sidebar carries the receipt.** `(no work seen)` on the last line means
  the harness counted zero tool calls in the window that step was finished in —
  the `worked` field (P7-16). It is a **signal, not a verdict**: "decide the
  approach" is a real step with no tool calls. The point is that a tick with work
  behind it and a tick without now look different to the person watching.
* **The prompt carries the blockers.** `(waiting on: …)` is what the model needs
  at the moment it chooses what to do next, and it disappears as its blocker
  completes. The receipt is deliberately *not* repeated here — it would spend
  tokens every turn telling the model something it cannot act on.

And when a turn tries to end at the first frame above, this arrives as a notice
tagged to the row rather than to you:

```
A skill you read set out a procedure and it is not finished: 'survey the
callers' can be started now. 2 further steps wait on these. Continue with it,
or if a step genuinely does not apply, mark it completed and say why in your
next message.
```

### What the model may and may not do

| | |
|---|---|
| Mark a seeded step `in_progress` or `completed` | **allowed** |
| Add its own entries before, between or after them | **allowed** |
| Reorder its *own* entries | **allowed** |
| Delete a seeded step | **refused** |
| Reorder seeded steps relative to each other | **refused** |
| Rewrite a seeded step's text | **refused** (it reads as a delete plus an add) |

The refusal is a `PlanError` the model reads as a tool result, and it says what
to do: *keep them, in order — you may add your own entries around them and mark
these done.* Without this rule the whole feature is decorative: `write_todos`
replaces the *entire* list, so seeded steps would be the model's to delete on its
next write, and a listener enforcing against that list would be enforcing against
nothing. `source` and `requires` are re-attached from the log on every write, so
the model never has to echo fields it did not author.

### When it steers, and when it does not

At `agent/turn-stopping` — the boundary the loop already fires, the same one
`/autonomous` uses — `skill-steps` asks one question: **is a seeded step
startable?** A step is startable when it is not completed and nothing it
`requires` is outstanding.

* **Something startable** → `agent.steer(...)` with a message naming up to three
  of them and counting the rest, tagged `PluginSource(plugin=
  "ph_stabilize.skill_steps", form="notice")` so the transcript does not
  attribute the harness's nudge to the person reading it. The turn continues.
* **No seeded steps outstanding** → the turn ends, as it always did. A skill's
  steps are a sequential chain, so this is the ordinary way one finishes: while
  anything is unfinished, its earliest link is waiting on nothing, and there is
  always exactly one step to point at.
* **Nothing startable** → the turn ends. This cannot happen to a chain
  `skill-steps` wrote, for the reason just given; it is a guard, because a steer
  naming no step is worse than no steer, and the list is a fold of a log that
  something else could put another arrangement into.

The nudge is a pointer, not a second copy of the plan: the model already has the
list in its context, and a twenty-line reminder every time a turn tries to end is
how a steer becomes noise the model learns to skim.

**And it stands down.** Three nudges with the todo list unchanged and the row
stops steering that session until something moves. It is counted since the list
last *changed*, not since the turn began, so any `write_todos` — marking a step
done, adding an entry, re-planning — resets it: a run making progress is never
cut off, and one going in circles hands the turn back rather than spending model
calls on a model that is not going to comply. The count is folded from the log
(the nudges are messages in it), so it survives a resume.

Everything lives in the log — `todo/write` is a fold — so a seeded procedure
survives a resume, a passivation and a fork for free, and the TUI sidebar and the
model's view are one projection rather than two that can disagree (A11).

---

## 4. Writing steps that work

* **One step, one outcome.** "Read the upstream and write down what it does" is a
  step. "Do the port" is a plan.
* **Name the gate in the step's text.** `skill-steps` holds the model to its own
  claim, not to reality — see §6 — so a step that must be *checked* should say
  what checks it: *"Run `uv run pytest -q` and report the result verbatim."*
* **Order by genuine dependency.** The chain is sequential by construction, which
  is right for a procedure. If two steps are genuinely independent, that is a
  sign they belong to two skills.
* **Fewer than you think.** Thirty-two is the cap, not a target. Every step is an
  entry the model cannot delete, so the cap is also a bound on how much of
  somebody's plan one skill may commandeer.
* **A step the model can decline.** The steer tells it: if a step genuinely does
  not apply, mark it completed and say why. That is deliberate — a procedure that
  cannot be exited is one that traps a session on a false premise.
* **Do not write the procedure as a checklist in the body.** `steps:` is the only
  place pH reads a procedure from; a `- [ ]` list below the frontmatter is prose,
  passed to the model verbatim and tracked by nothing. It is worse than merely
  inert: the body is re-read from disk on every `skill` call and never changes, so
  those boxes are *permanently unchecked* — a model that re-reads the skill
  halfway through gets a document saying nothing is done beside a live list saying
  three are. The model already sees its progress as a checklist, rendered from the
  log each turn, and so do you — see *What progress looks like* in §3. Use the body to explain the steps — a heading per step reads well — and let
  `steps:` be the only place they are declared.

---

## 5. Checking it works

```
# the skill loaded at all
ph --dump-config | grep -A3 skills-progressive
# …and watch the log for `ph.seams.skills` warnings if it did not

# end to end
ph --mode tui --patch '{id: tool-todo, disabled: false}' \
               --patch '{id: skill-steps, disabled: false}'
```

Eight worked skills, ported from OpenMono's playbook examples, live in
[`self-steerings-examples/`](self-steerings-examples/) — point `paths:` at that
directory to install all of them. Its README also records what a Playbook can
declare that pH cannot.

Ask the model to read the skill. The sidebar should fill with the steps, each
after the first marked *waiting on* its predecessor. Ask it to stop early; the
turn should keep going with a notice naming what is left. Ask it to rewrite the
list without one of the steps; the tool call should be refused with the sentence
above.

---

## 6. What this does not do

DESIGN §5 rule 6 — a caveat that lives only in a doc is a defect — so each of
these is gated as well as written down.

* **A loop step is not a plan step.** `agent/turn-stopping` fires when a turn is
  about to end, and a plan step spans many turns. The listener asserts a boundary
  invariant — *work remains that can begin* — and never claims to know which step
  is running. There is no "you are on step 3".
* **It enforces the model against its own accepted plan, not against reality**
  (P5-16). Marking a step done is still the model's word. `worked` — the count of
  tool calls the harness saw in the window a step was finished in — makes a bare
  tick *visible* to a person reading the card, and a model can generate tool
  calls. Real gates check the world: `ctx.goals` (a process exiting zero) and
  `ctx.approval` (a person). A Playbook's per-step `Confirm`/`Approve` maps onto
  approval, not onto this.
* **`allowed-tools` is a declaration, not a boundary.** pH has the enforcing
  mechanism — `ctx.tools.restrict` — and deliberately does not wire it here: a
  restriction is a scope boundary with a disposer, restrictions intersect, and a
  *turn is not a scope*. Two skills read in one turn could narrow to nothing with
  no moment to widen back out. What you get is `missing_tools`, resolved, at the
  one moment it is worth saying.
* **A skill cannot install itself.** I7: the model cannot mint a capability, and
  `/refine` writes procedure rather than capability (Q13). `paths` is a
  deployment's decision, written down.
* **Steps are seeded, not scheduled.** Nothing runs a step for the model, nothing
  times one out, and nothing notices a step that has been `in_progress` for an
  hour. The only pressure is the steer at the end of a turn, and that pressure
  stops after three nudges the plan does not answer.
* **It does not know what `/autonomous` decided.** Both are listeners on the same
  `agent/turn-stopping` boundary and `ctx.serial` runs them in registration
  order; neither returns a value, so neither can outrank the other. With a goal
  open *and* a procedure unfinished, `/autonomous` can settle `budget_limited`
  and this row will still steer — the run continues past the budget that was
  meant to bound it, and the model gets two notices for one step. A deployment
  that wants both should treat the goal's budget as advisory, or mount one of
  them. Making a "stop" outrank a "continue" is a change to the boundary itself
  and is not in this row.
