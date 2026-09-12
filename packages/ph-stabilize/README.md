# ph-stabilize

*The things that keep a long session from falling over: a plan, offloading in
both directions, compaction, limits, a human in the loop, and path rules — each
a row on a seam that already exists.*

Deep Agents' stabilization features are *algorithms plus prompts, not runtime*,
so none of them lands as a parameter on the agent loop. Todo planning is a tool
plus a prompt section. Offloading is a `tools/post-execute` listener.
Summarization is `agent/pre-step`. The driver does not know any of them is
there — which is the whole integration thesis (D12), and the reason a deployment
that wants the plain harness gets the plain harness.

```bash
ph --profile rlm-stable --provider llama --model <model> --mode tui
```

The package registers the `stabilize` **bundle** (`src/ph_stabilize/bundle.yaml`),
which `ph-app` composes through the `ph.bundles` entry-point group without
depending on this distribution.

**Every profile `ph-app` offers layers it** — `base` included — as an *optional*
bundle: an install without this distribution composes the same profiles and
simply never compacts. Compaction is the row that earns that, because a session
that grows until the provider refuses it is a defect in any posture; the rest
come along because a bundle is the unit a profile can name, and they are inert
until configured. `ph doctor` reports which of them actually activated.

## The rows

| row | what it does | shipped |
|---|---|---|
| `tool-todo` | `write_todos`, the `todo/write` event, the prompt section, and the one-call-per-turn rule | **off** |
| `skill-steps` | a `SKILL.md` that declares `steps:` becomes work the turn must finish | **off** |
| `tool-result-offload` | a result over the threshold goes to the spill store; the model gets a head-and-tail preview and the path | on |
| `input-offload` | the other direction: a pasted build log or dumped table | on |
| `compaction-summarize` | the conversation replaced by a summary when it stops fitting | on |
| `command-compact` | `/compact`, the human verb for the same thing | on |
| `limits` | model-call, tool-call and child ceilings, plus a consecutive-failure breaker | on, every ceiling unset |
| `hitl` | a person between the model and what it cannot take back | on, asking about nothing |
| `permissions-fs` | path rules over `ctx.fs` | on, with one default write rule |

**"On" mostly means "inert."** Layering this bundle must not, by itself, change
what a deployment does: `limits` mounts with every ceiling unset, `hitl` with an
empty `interruptOn`, `permissions-fs` with rules that allow everything inside
the workspace. A harness that starts asking on first run teaches its user to
approve without reading, and a limit nobody chose is a limit that fires on
somebody's longest legitimate turn.

The two that ship **off** are off for a different reason: they hand the model a
tool or keep a turn going, and that is a posture a profile chooses rather than
inherits. `rlm-stable` is the profile that chooses it — and the reason the
bundle reaching every profile does not thereby give every deployment a todo
tool.

The one row that is **not** inert on arrival is `permissions-fs`: it ships a
rule sending writes outside the agent's workspace to `interrupt`. That is E6's
intended default rather than an oversight, but it is a behaviour change a
deployment should know it inherited — `ph config --row permissions-fs` prints
what is in force.

## The tool, and the command

- **`write_todos`** (row `tool-todo`) — the whole list, replaced. The list lives
  in the log and nowhere else, so the TUI sidebar and the model's own view are
  one projection rather than two that can disagree, and it survives a resume and
  a fork for free. Two deliberate forks from upstream: `requires` declares what
  a step waits on (upstream works around the gap in *prose*, which is a
  dependency graph nothing can read or check), and a completed entry carries
  `worked` — the number of tools the harness saw run while it was being
  finished — which is the one field the model does not write, because a receipt
  the claimant issues is not a receipt.
- **`/compact`** (row `command-compact`) — replace older history with a summary,
  optionally told what you are about to work on. A command, not a turn: it
  dispatches directly, records `command/run`/`command/done`, and insists the
  agent is idle first, because a compaction landing mid-turn would move the
  surface underneath a request the loop had already derived.

Every failure is a sentence rather than a traceback — `CompactionError.code` is
a closed set precisely so a front end can phrase each one.

## Including it, and adjusting it

`rlm-stable` is `rlm` plus this bundle plus the rows it arms. Without a profile
that names the bundle, insert the rows you want by name — they are ordinary
plugin entry points, so this works against any profile:

```bash
ph --patch '{insert: [{id: tool-result-offload, name: tool-result-offload}]}' \
   --profile llama -p "…"
```

```yaml
# $PH_HOME/profiles/rlm-stable.yaml — ceilings, and a narrower file rule
- id: limits
  config:
    modelCalls: {turnLimit: 40, sessionLimit: 400, exit: end}
    toolCalls: {turnLimit: 100, perTool: {bash: {turnLimit: 20}}, exit: continue}
    breaker: {consecutiveFailures: 5}

- id: permissions-fs
  config:
    rules:
      - operations: [read]
        paths: ["**/.env", "**/*.pem"]
        mode: deny
      - operations: [write]
        paths: ["**"]
        scope: outside-workspace
        mode: interrupt
```

| row | config | default |
|---|---|---|
| `tool-result-offload` | `tokenLimit`, `maxInlineBytes`, `excludedTools` | `20000`; `None` disables offloading |
| `input-offload` | `tokenLimit` | `50000` |
| `compaction-summarize` | `auto`, `triggerFraction`, `keepFraction`, `triggerTokens`, `keepMessages`, `maxTokens`, `summaryInputTokens`, `truncateArgs`, `overflowClipTokens` | `0.85` / `0.10` of a known window; `170000` tokens / `6` messages when it is not — upstream's numbers |
| `limits` | `modelCalls` (`exit: end \| error`), `toolCalls` (plus `perTool`, `exit: continue \| end \| error`), `children`, `breaker.consecutiveFailures` — each budget takes `turnLimit` and `sessionLimit` | all ceilings unset; breaker `5` |
| `hitl` | `mode` (`manual` \| `auto` \| `yolo`), `declared`, `interruptOn` | `auto`, nothing configured |
| `permissions-fs` | `rules` — each `operations`, `paths`, `scope` (`anywhere` \| `outside-workspace`), `mode` (default `deny`), `description` | one rule: `write` anywhere `outside-workspace` → `interrupt` |
| `tool-todo`, `skill-steps`, `command-compact` | — | no configuration |

### Writing a `hitl` rule

Keys are tool names, and each takes a `preset`, a `when:` list of regexes, a
`description` and an allowed decision set. `rlm-stable`'s own rules are the
worked example:

```yaml
- id: hitl
  config:
    interruptOn:
      bash:
        preset: destructive
        when: ["\\b(npm|pnpm|yarn|uv|pip)\\s+publish\\b", "\\bsudo\\b"]
        description: >-
          This command changes something a worktree cannot undo.
      run_code:
        preset: destructive
        when: ["\\bsubprocess\\b"]
        allowedDecisions: [approve, reject]
```

Three things that rule says, spelled out:

- **Name the preset rather than retyping it.** `destructive` is the set this
  package ships and tests. The first draft of `rlm-stable.yaml` retyped a subset
  and had already widened `git push` from force-only to *every* push, against a
  shipped test pinning an ordinary push as ordinary — one security judgement
  beats two that disagree.
- **Key Code Mode on the *reserved* name.** The registry renames the transport
  to whatever the presentation calls it (`ipython`, in the `rlm` bundle) and
  `hitl` resolves that. Keying on the presented name would not turn the gate off
  — the tool declares `is_irreversible`, so `declared` still catches it — but
  the `when:` additions are keyed by name and would silently stop applying.
- **Four decisions, not two.** `approve` and `reject` were always reachable;
  this row adds `edit` (run it with these arguments instead) and `respond` (do
  not run it — tell the model this), because stopping a turn to say "wrong path"
  or "you don't need that, the answer is X" costs a round trip that answering in
  place does not. `allowedDecisions` narrows the set — a program is not
  something to hand-patch in a modal, so `run_code` above drops `edit`.

**The classifier parses; it does not pattern-match.** `preset: destructive` runs
`ph_stabilize.destructive`, which reads each string argument in its own dialect
— shell through `shlex`, SQL as statements, Python through `ast` — and judges
the *structure*. The twelve regexes it replaced were run over the call's
arguments rendered as JSON, so a real newline became the two characters `\` and
`n` and every `\b`-anchored pattern stopped matching on a second line: `rm -rf`,
`DROP TABLE`, `shutil.rmtree` and `curl | sh` were all ungated inside a
multi-line cell, which is *every* `run_code` cell. Nothing failed; the gate
simply did not fire. A `when:` list is still regex, as the deployment's own
escape hatch, but it is matched against the decoded strings.

## Limitations, and things that are deliberate

- **`permissions-fs` bounds seam-mediated access only** (E9, N1). The rules
  attach to `ctx.fs`'s intent waterfalls, so any tool going through the seam is
  covered without this module knowing a tool name — and a model-authored
  `open(path, "w")` inside a code cell, or a `subprocess` that shells out, never
  fires an intent and is not touched by *these rules*. What bounds those is the
  rung below: `code-runtime-python` confines the kernel itself. The row says so
  at mount and carries the sentence on `ctx.fs_permissions.reach`, so the
  statement toggles with the sandbox rather than being a paragraph in a README
  that is wrong half the time.
- **Offloading never deletes.** Both directions are a surface `replace`: the log
  keeps every event, `derive_messages()` yields the preview, and `transcript()`
  still shows the person what they sent. Rewriting a message *before* logging it
  was the alternative and is refused — the log would then attribute
  harness-authored text to the human.
- **Offloading fails open.** If the spill write fails, the original result is
  kept: an offload that cannot store the content must not be the reason the
  model loses it.
- **Self-limiting tools are left alone, and they say so themselves.** A tool
  that bounds its own output and offers a way to page (`read` with an offset,
  `grep` with a match cap) sets `ToolDefinition.self_limits`. Upstream matches a
  hardcoded list of *its* tool names; pH asks the tool, because a deployment
  renames them and an MCP server adds its own. `excludedTools` is the escape
  hatch for a third-party tool whose author has not declared.
- **Compaction can decline.** The cut is moved back to the nearest balanced
  boundary so a call is never split from its result; if the only balanced cut is
  the start of the conversation, nothing is compacted and the attempt **says
  so**. Shipping an orphaned `tool-result` to a provider that rejects it is not
  a repair. Two cheaper remedies run first — eliding over-long call arguments in
  retained history, then spilling the trailing tool-result batch.
- **Limits are a fold over the log, not a counter in memory.** A limit that
  lives in a field is a limit a resume forgets. Upstream's *thread* and *run*
  map to pH's **session** and **turn**, renamed once here so a diff against
  `model_call_limit.py` stays readable.
- **`skill-steps` holds the model to its own accepted plan, not to reality.**
  Marking a step done is still the model's word; it asserts only that work
  remains which *can* begin, and never claims to know which step is running.
  Gates that check the world are `ctx.goals` and `ctx.approval`. It is also
  useless without `tool-todo` — the list it seeds is that row's — so a profile
  enabling one enables both.

## Tests

`tests/` — nine modules, one per feature plus the classifier. The destructive-command
tables are a starting point and are meant to grow: neither list claims to be
complete, and the honest posture is that the gate reports what it found, in the
parser's own words, so a person is told what the harness saw and an auditor can
tune it.
