# ph-stabilize

*The rows that keep a long session stable: a plan, offloading in both
directions, compaction, limits, a human in the loop, and path rules. Each one is
a row on a seam that already exists.*

A row is an entry in a pH profile that mounts one plugin. The stabilization
features of Deep Agents are *algorithms plus prompts, not runtime*, so none of
them becomes a parameter on the agent loop. Todo planning is a tool plus a
prompt section. Offloading is a `tools/post-execute` listener. Summarization is
`agent/pre-step`. The driver does not know that any of them is there. That is
the integration thesis (D12). It is also the reason why a deployment that wants
the plain harness gets the plain harness.

```bash
phern --profile rlm-stable --provider llama --model <model> --mode tui
```

The package registers the `stabilize` **bundle**
(`src/ph_stabilize/bundle.yaml`). A bundle is a named list of rows that a
profile can layer. `phern` composes this bundle through the `ph.bundles`
entry-point group, and its module `ph_app` does not import this package.

**Every profile that `phern` offers layers this bundle**, `base` included. Each
profile layers it as an *optional* bundle: an install without this distribution
composes the same profiles and never compacts. Compaction is the row that
justifies this, because a session that grows until the provider refuses it is a
defect in any posture. The other rows come with it because a bundle is the unit
that a profile can name, and they are inert until they are configured.
`phern doctor` reports which rows activated.

## The rows

| Row | What it does | Shipped |
|---|---|---|
| `tool-todo` | Adds `write_todos`, the `todo/write` event, the prompt section, and the rule of one call in each assistant message | **Off** |
| `skill-steps` | Makes a `SKILL.md` that declares `steps:` into work that the turn must finish | **Off** |
| `tool-result-offload` | Sends a result over the threshold to the spill store. The model gets a head-and-tail preview and the path | On |
| `input-offload` | Does the same in the other direction, for a pasted build log or a dumped table | On |
| `compaction-summarize` | Replaces the conversation with a summary when it no longer fits | On |
| `command-compact` | Adds `/compact`, the command that a user gives for the same thing | On |
| `limits` | Sets ceilings on model calls, tool calls and children, plus a consecutive-failure breaker | On, every ceiling unset |
| `hitl` | Puts a person between the model and what it cannot take back | On, asks only about a destructive command |
| `permissions-fs` | Applies path rules over `ctx.fs` | On, with one default write rule |

**"On" mostly means "inert".** Layering this bundle must not, by itself, change
what a deployment does:

- `limits` mounts with every ceiling unset.
- `hitl` mounts with an empty `interruptOn` and one curated rule for the tools
  that declare themselves irreversible.
- `permissions-fs` mounts with rules that allow everything inside the workspace.

A harness that starts to ask on the first run teaches its user to approve
without reading. A limit that nobody chose fires on the longest legitimate turn
of some user.

The two rows that ship **off** are off for a different reason. They give the
model a tool or keep a turn going, and that is a posture that a profile chooses
and does not inherit. `rlm-stable` chooses the first: it turns `tool-todo` on.
That is why the bundle can reach every profile and not give every deployment a
todo tool. No shipped profile turns `skill-steps` on. A profile or a `--patch`
that wants `skill-steps` turns on both rows.

Three things are **not** inert on arrival:

- `permissions-fs` ships a rule that sends writes outside the workspace of the
  agent to `interrupt`. That is the intended default of E6 and not an oversight.
  But it is a behavior change, and a deployment should know that it inherited
  the change. `phern config --row permissions-fs` prints the rules that are in
  force.
- `hitl` ships its `declared` rule as the `destructive` preset. A tool that
  declares itself irreversible (`bash`, a code cell) asks when the parsed
  classifier finds something that it cannot take back (`rm -rf`, `DROP TABLE`,
  `shutil.rmtree`, `curl | sh`). Otherwise the tool runs without an ask. With no
  approval answerer mounted, an ask is a refusal.
- `limits` ships its breaker on, although every ceiling is unset. The breaker
  trips after five consecutive failures of one tool. Set
  `breaker.consecutiveFailures` to `null` to turn it off.

## The tool, and the command

- **`write_todos`** (row `tool-todo`) replaces the whole list. The list is in
  the log and nowhere else. The TUI sidebar and the view of the model are
  therefore one projection, and not two that can disagree. The list also
  survives a resume and a fork with no extra work. The tool differs from
  upstream in two deliberate ways:
  - `requires` declares what a step waits on. Upstream works around the gap in
    *prose*, which is a dependency graph that nothing can read or check.
  - A completed entry carries `worked`: the number of tools that the harness saw
    run while the entry was being finished. It is the one field that the model
    does not write, because a receipt that the claimant issues is not a receipt.
- **`/compact`** (row `command-compact`) replaces older history with a summary.
  You can tell it what you are about to work on. It is a command, not a turn: it
  dispatches directly and records `command/run` and `command/done`. It requires
  the agent to be idle first. A compaction that landed in the middle of a turn
  would move the surface underneath a request that the loop had already derived.

Every failure is a sentence and not a traceback. `CompactionError.code` is a
closed set so that a front end can phrase each one.

## Including it, and adjusting it

`rlm-stable` is `rlm` plus this bundle plus the rows that it turns on. Without a
profile that names the bundle, insert the rows that you want by name. They are
ordinary plugin entry points, so this works against any profile:

```bash
phern --patch '{insert: [{id: tool-result-offload, name: tool-result-offload}]}' \
   --profile llama -p "…"
```

```yaml
# $PH_HOME/profiles/rlm-stable.yaml: ceilings, and a narrower file rule
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

| Row | Config | Default |
|---|---|---|
| `tool-result-offload` | `tokenLimit`, `maxInlineBytes`, `excludedTools` | `20000`; `None` disables offloading |
| `input-offload` | `tokenLimit` | `50000` |
| `compaction-summarize` | `auto`, `triggerFraction`, `keepFraction`, `triggerTokens`, `keepMessages`, `maxTokens`, `summaryInputTokens`, `truncateArgs`, `overflowClipTokens` | `0.85` / `0.10` of a known window. `170000` tokens / `6` messages when the window is not known. These are the numbers of upstream |
| `limits` | `modelCalls` (`exit: end \| error`), `toolCalls` (plus `perTool`, `exit: continue \| end \| error`), `children` (`exit: continue \| end \| error`), `breaker.consecutiveFailures`. Each budget takes `turnLimit` and `sessionLimit` | All ceilings unset; breaker `5` |
| `hitl` | `mode` (`manual` \| `auto` \| `yolo`), `declared`, `interruptOn` | `auto`. `declared` is `{preset: destructive}`, so `bash` and a code cell ask when the classifier finds a destructive command, and run without an ask otherwise. `interruptOn` is empty |
| `permissions-fs` | `rules`. Each rule has `operations`, `paths`, `scope` (`anywhere` \| `outside-workspace`), `mode` (default `deny`), and `description` | One rule: `write` anywhere `outside-workspace` → `interrupt` |
| `skill-steps` | `maxNudges`, `decision` (`provider`, `model`, `holdAt`, `timeoutMs`) | `3`. No decision model, so every unfinished procedure is nudged. With a decision model, a turn that ends to wait for a person is allowed to end, and a route that times out rests for a minute. See [docs/skills](../../docs/skills/README.md#asking-a-decision-model-first) |
| `tool-todo`, `command-compact` | None | No configuration |

### Writing a `hitl` rule

Keys are tool names. Each key takes a `preset`, a `when:` list of regexes, a
`description`, and a set of allowed decisions. The rules of `rlm-stable` itself
are the worked example:

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

That rule says three things:

- **Name the preset and do not retype it.** `destructive` is the set that this
  package ships and tests. The first draft of `rlm-stable.yaml` retyped a
  subset. It had already widened `git push` from force-only to *every* push,
  against a shipped test that holds an ordinary push to be ordinary. One
  security judgment is better than two that disagree.
- **Key Code Mode on the *reserved* name.** The registry renames the transport
  to the name that the presentation gives it (`python`, in the `rlm` bundle),
  and `hitl` resolves that name. A key on the presented name would not turn the
  gate off: the tool declares `is_irreversible`, so `declared` still catches it.
  But the `when:` additions are keyed by name, and they would stop applying with
  no error.
- **Four decisions, not two.** `approve` and `reject` were always reachable.
  This row adds `edit` (run the call with these arguments instead) and `respond`
  (do not run the call, and tell the model this). To stop a turn to say "wrong
  path" or "you don't need that, the answer is X" costs a round trip, and an
  answer in place does not. `allowedDecisions` narrows the set. A program is not
  something to patch by hand in a modal, so `run_code` above drops `edit`.

**The classifier parses. It does not match patterns.** `preset: destructive`
runs `ph_stabilize.destructive`. That module reads each string argument in its
own dialect (shell through `shlex`, SQL as statements, Python through `ast`) and
judges the *structure*. The twelve regexes that it replaced ran over the
arguments of the call rendered as JSON. A real newline therefore became the two
characters `\` and `n`, and every pattern anchored with `\b` stopped matching on
a second line. `rm -rf`, `DROP TABLE`, `shutil.rmtree` and `curl | sh` were all
ungated inside a multi-line cell, which is *every* `run_code` cell. Nothing
failed. The gate did not fire. A `when:` list is still regex, as the escape
hatch of the deployment, but it is matched against the decoded strings.

## Limitations, and things that are deliberate

- **`permissions-fs` bounds only the access that goes through the seam** (E9,
  N1). The rules attach to the intent waterfalls of `ctx.fs`. Any tool that goes
  through the seam is therefore covered, and this module does not need to know a
  tool name. A model-authored `open(path, "w")` inside a code cell, or a
  `subprocess` that runs a shell command, never fires an intent, and *these
  rules* do not touch it. The rung below bounds those: `code-runtime-python`
  confines the kernel itself. The row states its reach at mount and keeps the
  sentence on `ctx.fs_permissions.reach`. The statement therefore changes with
  the sandbox. A paragraph in a README would be wrong half the time.
- **Offloading never deletes, and each direction keeps the original in its own
  way.** A paste is a surface `replace`. The log keeps the `user/message` as it
  was sent, `derive_messages()` yields the preview, and `transcript()` still
  shows the user what they sent. The alternative was to rewrite a message
  *before* it is logged, and that is refused: the log would then attribute text
  that the harness wrote to the human. A tool result is already written by the
  harness, so it is swapped before it is logged. `tool/result` carries the
  preview that the model saw, the full text is in the spill store, and
  `offload/spilled` records where the text went.
- **Offloading fails open.** If the spill write fails, the row keeps the
  original result. An offload that cannot store the content must not be the
  reason that the model loses it.
- **Self-limiting tools are left alone, and they declare it themselves.** A tool
  that limits its own output and offers a way to page (`read` with an offset,
  `grep` with a match cap) sets `ToolDefinition.self_limits`. Upstream matches a
  hardcoded list of *its* tool names. pH asks the tool, because a deployment
  renames tools and an MCP server adds its own. `excludedTools` is the escape
  hatch for a third-party tool whose author has not declared.
- **Compaction can decline.** The cut is moved back to the nearest balanced
  boundary, so a call is never split from its result. If the only balanced cut
  is the start of the conversation, nothing is compacted. The automatic path
  records nothing, and `/compact` **says that** there is no compactable history
  yet. To send an orphaned `tool-result` to a provider that rejects it is not a
  repair. Two cheaper remedies run first: the row elides over-long
  call arguments in retained history, and then spills the trailing tool-result
  batch.
- **Limits are a fold over the log, not a counter in memory.** A limit that is
  kept in a field is lost on a resume. The *thread* and *run* of upstream map to
  the **session** and **turn** of pH. The row renames them one time, so that a
  diff against `model_call_limit.py` stays readable.
- **`skill-steps` holds the model to the plan that it accepted, not to
  reality.** When the model marks a step done, that is still only the word of
  the model. The row asserts only that work remains which *can* begin, and it
  never claims to know which step is running. The gates that check the world are
  `ctx.goals` and `ctx.approval`. The row also needs `tool-todo` beside it, and
  nothing checks that `tool-todo` is there. The list that `skill-steps` seeds
  belongs to `tool-todo`. Without `write_todos`, the steps are still seeded and
  the turn is still nudged, but the model cannot mark a step done. The row then
  spends its nudges and stands down. A profile that turns `skill-steps` on turns
  on both rows.
- **Nothing shipped changes the mode of `hitl` during a session.** The mode in
  force is the last `approval/mode` record, or else the `mode` of the row.
  `set_mode` writes that record, but no command or key calls it. Today the mode
  is the one that the row of the profile sets.
- **A decision model is asked, never relied on.** `skill-steps` asks a decision
  model whether a turn is ending to wait for a person (`ctx.llm.decide`). When
  it is, the row holds its nudge. Sometimes a route cannot be asked: it is
  unregistered, down, slow, or it answers in another shape. The row then sends
  the nudge that it always sent, with a warning and a `llm/decision-failed`
  record. The row has no `inject` on `llm`, on purpose.

## Tests

`tests/` has nine modules: one for each feature, plus one for the classifier.
The destructive-command tables are a starting point and are meant to grow. None
of them claims to be complete. The correct posture is for the gate to report
what it found, in the words of the parser. A person is then told what the
harness saw, and an auditor can tune the gate.
