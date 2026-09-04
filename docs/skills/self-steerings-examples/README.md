# Self-steering examples

Eight skills ported from OpenMono's playbook examples
(`sources/OpenMonoAgent.ai/docs/playbooks-examples`), rewritten as pH `SKILL.md`
files. They are here to be read *and* run — point a profile at this directory and
all eight install:

```yaml
- id: skills-progressive
  config:
    paths: [docs/skills/self-steerings-examples]
- id: tool-todo
  disabled: false
- id: skill-steps
  disabled: false
```

See [../README.md](../README.md) for what a self-steering skill is and how the
seeding and steering work.

| Skill | Steps | What it is |
|---|---|---|
| `commit` | 3 | Stage, write a conventional message, commit. |
| `file-scan` | 2 | Create the notes files, scan for TODO/FIXME markers. |
| `graphify` | 3 | Query or rebuild the graphify knowledge graph. |
| `deploy-ftp` | 4 | Build, diff against the remote, upload what changed. |
| `pr-ready` | 6 | Sync check, tests, lint, commit, describe, open the PR. |
| `release` | 6 | Pre-flight, analyse, changelog, bump, validate, tag. |
| `db-migrate` | 7 | Validate, dry-run, review, staging, smoke, prod, verify. |
| `incident-response` | 7 | Gather, blast radius, confirm, mitigate, verify, close, write up. |

Every one of them loads and renders under this build — `test_the_ported_examples_
all_load_and_render` in `packages/ph-core/tests/test_skills_progressive.py` is
the gate, so an example that stops parsing fails the suite rather than failing a
person.

---

## What the port changed

Both formats are Markdown with a YAML frontmatter block, so the file *shape*
carried over unchanged. What differed:

| Playbook | pH | Note |
|---|---|---|
| `{{params.x}}` | `{{parameters.x}}` | Straight rename. |
| `type: String` / `Boolean` | `type: string` / `boolean` | pH's three types are `string`, `number`, `boolean`. |
| `steps: [{id, requires, gate, file, script, output}]` | `steps: ["…"]` | A pH step is a string. See the table below for each dropped key. |
| `file: steps/01-x.md` | a `## 1. …` section in the body | 23 step files inlined; nothing else in a skill directory is read. |
| `script: scripts/x.sh` | the commands, inline in the step's section | 1 258 lines of shell not ported — see gap 3. |
| `constraints.inline: [...]` | a `## Rules` section | Prompt text on both sides; only the location changed. |
| `trigger`, `trigger-patterns`, `user-invocable` | — | pH has no pattern-triggered invocation; the catalog `description` is what makes the model reach for a skill. |
| `context-mode`, `max-context-tokens`, `tags`, `depends-on` | — | See gaps 8 and 9. |

Two adaptations worth naming because they are not mechanical: `release` was
.NET-only (`dotnet test`, `*.csproj`) and now detects the project type, since a pH
user's repo generally is not a .NET solution; and `commit` had no `steps:` at all
— its body was already a numbered procedure, so the port promoted it.

---

## What is missing, in order of how much it costs

### 1. Per-step gates — `gate: Review | Confirm | Approve` (33 uses)

**The largest gap.** In a playbook the executor *halts* at a gate; it is a
runtime mechanism the model cannot talk its way past. In pH the step text says
"stop for approval" and that is an instruction, not a mechanism.

pH's real gates — `ctx.approval` and `ctx.goals` — are the right ones, but they
attach to a **tool call** and to a **goal**, not to a step boundary. There is no
way for a step to declare "do not pass here without a person". So the three
distinct gate strengths collapse into prose, and the model can walk past them.

Every example that pauses (`release` step 3, `deploy-ftp` step 3, `db-migrate`
steps 3/4/6, `incident-response` steps 3/4/6, `pr-ready` steps 5/6) is prose in
the port. This is the one gap where a ported example is genuinely weaker than its
source.

### 2. Step outputs — `output: name` and `{{state.name}}` (23 declared, 8 read)

A playbook step names a value and later steps interpolate it. pH has nothing: the
value lives in the conversation and the later step relies on the model still
remembering it.

That mostly works, and then stops working precisely where it matters — **after
compaction**, when a summariser may drop the version string step 4 computed and
step 6 needs. `release` shows the shape: its step 4 says *"say the new version out
loud — the later steps need it and there is no slot that holds it for you"*, which
is a workaround, not a fix.

### 3. Shipped assets — `{{playbook.base-path}}` (14 uses, 14 scripts, 1 258 lines)

A playbook is a *directory*: steps reference `scripts/validate.sh` relative to
the playbook's own path. pH's `discover_skills` globs `*/SKILL.md` and reads only
that file — nothing else in the directory is discovered, and the skill's own
location is not reliably visible to the model either (`SkillValue.path` is in the
structured value, but the tool renders only `instructions` as the result text).

So there is no way to ship a script beside a skill and have a step run it. Every
script in the source became either inline commands or a description of what the
command must do.

### 4. Command substitution — `{{shell:...}}` (11 uses)

The playbook executor runs the command and splices its stdout into the prompt
*before* the model sees the step. pH has no equivalent, and **should not**: I3
says model-visible implies logged, and this would put command output into a
context window without a `tool/call` or `tool/result` behind it. The pH shape is
that the model calls `bash` itself — one more round trip, and on the log.

Listed as a gap because it changes how the examples read, not because it should
be closed.

### 5. Per-step bodies — `file: steps/NN-x.md` (23 uses)

One `SKILL.md` per skill, so a seven-step procedure is one document and the whole
of it enters context when the skill is read. G9 defers the body until the model
asks — but it cannot defer *per step*. `db-migrate`'s source is 349 lines across
seven files; the port is one 4.2 KB body.

### 6. A step that invokes another skill — `playbook: commit` (1 use)

`pr-ready`'s fourth step calls the `commit` playbook. pH has no compositional
call: the model can call the `skill` tool again by itself, but a step cannot
declare the dependency, and the sub-skill's steps would seed into the same flat
list with no relationship to the parent's.

### 7. Per-step `allowed-tools` (1 use)

pH declares tools skill-wide, and **as a declaration rather than a boundary** —
documented, with the reason: `ctx.tools.restrict` is a scope with a disposer,
restrictions intersect, and a turn is not a scope.

### 8. `context-mode: Selective` / `max-context-tokens` (6 and 5 uses)

pH renders the whole todo list every turn. Collapsing completed steps was
examined during P7-16 and rejected: after compaction the `PromptContext` is the
only surviving statement of what was done, so hiding finished entries loses the
account of the run. Bounds were added instead.

### 9. `trigger: auto` / `trigger-patterns` (8 uses), `tags` (6), `depends-on` (5)

No pattern-triggered invocation in pH — the model chooses from the catalog
`description`, and a person invokes a verb through `ctx.commands`. No tag
filtering of the catalog. `depends-on` is empty in every source example.

### 10. Two small bounds

`MAX_TODO_CONTENT` caps a step at 500 characters, and one source `inline-prompt`
was 510 — a step is an entry in a todo list here, not a paragraph of prompt.
`MAX_STEPS` caps a procedure at 32; the longest source playbook has 7.

---

## What pH has that the source does not

Worth stating, since the gaps above are one-directional by construction:

- **The steps are a real todo list the model cannot quietly drop.** A playbook's
  steps are executor state the model never holds; pH's are entries in the model's
  own plan, and `_carried` refuses a write that drops, reorders or rewords one.
- **A receipt per completion.** `worked` counts the tool calls the harness saw in
  the window a step was ticked in, and the sidebar marks a bare tick `(no work
  seen)`.
- **It is all a fold of the session log**, so a procedure survives a resume, a
  passivation and a fork, and the person's sidebar and the model's prompt are one
  projection rather than two that can disagree.
- **The steering stands down.** Three nudges with the plan unmoved and the row
  stops, rather than spending model calls on a model that is not complying.
