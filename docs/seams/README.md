# Seam reference

**Status (P6-10):** **every service key has a page**, and both halves are
pinned by a test — no seam module is missing from this index, and no page names a
method that does not exist.

That is the row's gate met for the seams. What P6-10 also asks for is the
[cookbook](../cookbook/), which is written. `ctx.subagent_presets` has no page of
its own: it is a three-method service whose design only makes sense beside the
delegation it configures, so it is documented in
[`subagents.md`](subagents.md).

Each line is the module's own first docstring line. **The module is the
authority**: every seam in this tree argues its design at length in its own
docstring, and a page here that restated it would be a second copy free to drift.
What a page adds, when written, is the part a module docstring is the wrong place
for — how to *provide* one, what a consumer may assume, and the worked example.

`test_docs_seams.py` fails when a seam module has no entry here, so this list
cannot silently fall behind the code — and for each page that exists, it checks
that every `ctx.<key>.<method>` the page names is still a real method.

## Core services

Not in `ph/seams/` — one implementation, extended by registering into it rather
than by replacing it.

| service | module | |
|---|---|---|
| [`ctx.tools`](tools.md) | `ph/tools/registry.py` | The registry, the visibility rules, and the pipeline. |

## Files, execution and confinement

| service | module | |
|---|---|---|
| [`ctx.fs`](fs.md) | `seams/fs.py` | Filesystem access with an interception point before every access. |
| [`ctx.subprocess`](subprocess.md) | `seams/subprocess.py` | Spawning, with nothing implicit. |
| [`ctx.shell`](shell.md) | `seams/shell.py` | Bash over `ctx.subprocess`, confined when a backend exists. |
| [`ctx.sandbox`](sandbox.md) | `seams/sandbox.py` | Confinement, and the refusal to pretend. |
| [`ctx.containment`](containment.md) | `seams/containment.py` | Which rung of the ladder this deployment asked for (E1, E8). |
| [`ctx.code_runtime`](code_runtime.md) | `seams/code_runtime.py` | The seam definition only (P1-06, C1). |
| [`ctx.workspace`](workspace.md) | `seams/workspace.py` | Where an agent's writes land, and how honestly that is stated. |

Providers, which register into the seams above rather than publishing their own
key: `seams/sandbox_local.py` (bwrap, P6-04), `seams/sandbox_egress.py` (the
filtering proxy behind the network allowlist), `seams/sandbox_allow.py` (what a
confined command may reach beyond its workspace), `seams/workspace_git.py` (the
`worktree` tier), `seams/workspace_jj.py` (the same tier over Jujutsu, where a
child starts from its parent's work in progress), `seams/workspace_agentfs.py` (a
copy-on-write overlay), `seams/workspace_scratch.py` (the `sandbox` rung's kind),
`seams/workspace_provision.py` (making a fresh tree usable).

## What reaches the model

| service | module | |
|---|---|---|
| [`ctx.attachments`](attachments.md) | `seams/attachments.py` | Media the log points at but cannot reconstruct. |
| [`ctx.uploads`](uploads.md) | `seams/uploads.py` | A provider's copy of an attachment, and the handle for it (P7-03). |
| [`ctx.spill_store`](spill_store.md) | `seams/spill.py` | Oversized content out of context, with a way back. |
| [`ctx.compaction`](compaction.md) | `seams/compaction.py` | Replacing history with a summary, without losing it. |
| [`ctx.token_meter`](token_meter.md) | `seams/token_meter.py` | The provider's count is the truth; ours is for pressure. |
| [`ctx.skills`](skills.md) | `seams/skills.py` | The capability layer, and the boundary it must not cross. |

## Asking a person

| service | module | |
|---|---|---|
| [`ctx.approval`](approval.md) | `seams/approval.py` | Asking a human, and failing closed when you cannot. |
| [`ctx.user_questions`](user_questions.md) | `seams/user_questions.py` | Asking the human something that is not an approval. |
| [`ctx.commands`](commands.md) | `seams/commands.py` | Human slash commands that spend no model turn. |
| [`ctx.permission_presets`](permission_presets.md) | `seams/permission_presets.py` | One name for a sandbox mode *and* an approval policy. |
| [`ctx.settings`](settings.md) | `seams/settings.py` | Durable user preferences, read as data. |

## Agents and work

| service | module | |
|---|---|---|
| [`ctx.subagents`](subagents.md) | `seams/subagents.py` | Delegation to a child agent, and the handle it returns. |
| [`ctx.subagent_presets`](subagents.md) | `seams/subagents.py` | The named grants a child may be admitted under — documented with the delegation it configures. |
| [`ctx.jobs`](jobs.md) | `seams/jobs.py` | Background work with a handle, a cancel and a completion. |
| [`ctx.schedule`](schedule.md) | `seams/schedule.py` | Work a root will do later, folded from its own log (P5-06). |
| [`ctx.goals`](goals.md) | `seams/goals.py` | An objective, a budget, and the gates that decide it (P5-07). |

`seams/schedule_index.py` answers which sessions hold a live appointment (P6-23).

## The front end

| service | module | |
|---|---|---|
| [`ctx.tui_screens`](tui_screens.md) | `seams/tui_screens.py` | The front end's registration seam (P4-17). |
| [`ctx.tui_status`](tui_status.md) | `seams/tui_status.py` | A live reading in the footer, contributed by a row. |
| [`ctx.diagnostics`](diagnostics.md) | `seams/diagnostics.py` | What a row wants `ph doctor` to say about it. |

`seams/topology.py` contributes doctor's Topology section: what the mount *became*.

## Secrets, records and invariants

| service | module | |
|---|---|---|
| [`ctx.credentials`](credentials.md) | `seams/credentials.py` | References travel, values do not (I-3). |
| [`ctx.session_telemetry`](session_telemetry.md) | `seams/telemetry.py` | Records, redaction, and no span tracer. |
| [`ctx.invariants`](invariants.md) | `seams/invariants.py` | Which invariants this deployment enforces, and whether they hold. |

`seams/telemetry_otel.py` ships the ledger to an OTel collector (P5-09).
`seams/scope_invariant.py` and `seams/skills_invariant.py` are runtime checks registered into
`ctx.invariants` (P6-01) rather than seams of their own.

## Writing a page

When one is written, it belongs here as `<key>.md` and should cover what the
module docstring does not:

* **what a consumer may assume** — including the no-provider answer;
* **how to provide one** — the Protocol, and what registration path;
* **what it refuses**, and what it explicitly does not enforce;
* a worked example that exists in the tree.

[Adding a seam](../cookbook/adding-a-seam.md) is the how-to; these are the
reference.
