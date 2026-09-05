# Seam reference

**Status (P6-10):** this index is complete and pinned by a test; the per-seam
pages are not written. That is the honest state — the row's gate is "every seam
has a page", and what exists so far is every seam having an *entry*.

Each line is the module's own first docstring line. **The module is the
authority**: every seam in this tree argues its design at length in its own
docstring, and a page here that restated it would be a second copy free to drift.
What a page adds, when written, is the part a module docstring is the wrong place
for — how to *provide* one, what a consumer may assume, and the worked example.

`test_docs_seams.py` fails when a seam module has no entry here, so this list
cannot silently fall behind the code.

## Files, execution and confinement

| service | module | |
|---|---|---|
| `ctx.fs` | `seams/fs.py` | Filesystem access with an interception point before every access. |
| `ctx.subprocess` | `seams/subprocess.py` | Spawning, with nothing implicit. |
| `ctx.shell` | `seams/shell.py` | Bash over `ctx.subprocess`, confined when a backend exists. |
| `ctx.sandbox` | `seams/sandbox.py` | Confinement, and the refusal to pretend. |
| `ctx.containment` | `seams/containment.py` | Which rung of the ladder this deployment asked for (E1, E8). |
| `ctx.code_runtime` | `seams/code_runtime.py` | The seam definition only (P1-06, C1). |
| `ctx.workspace` | `seams/workspace.py` | Where an agent's writes land, and how honestly that is stated. |

Providers, which register into the seams above rather than publishing their own
key: `seams/sandbox_local.py` (bwrap, P6-04), `seams/workspace_git.py` (the `worktree` tier),
`seams/workspace_agentfs.py` (a copy-on-write overlay), `seams/workspace_scratch.py` (the
`sandbox` rung's kind), `seams/workspace_provision.py` (making a fresh tree usable).

## What reaches the model

| service | module | |
|---|---|---|
| `ctx.attachments` | `seams/attachments.py` | Media the log points at but cannot reconstruct. |
| `ctx.uploads` | `seams/uploads.py` | A provider's copy of an attachment, and the handle for it (P7-03). |
| `ctx.spill_store` | `seams/spill.py` | Oversized content out of context, with a way back. |
| `ctx.compaction` | `seams/compaction.py` | Replacing history with a summary, without losing it. |
| `ctx.token_meter` | `seams/token_meter.py` | The provider's count is the truth; ours is for pressure. |
| `ctx.skills` | `seams/skills.py` | The capability layer, and the boundary it must not cross. |

## Asking a person

| service | module | |
|---|---|---|
| `ctx.approval` | `seams/approval.py` | Asking a human, and failing closed when you cannot. |
| `ctx.user_questions` | `seams/user_questions.py` | Asking the human something that is not an approval. |
| `ctx.commands` | `seams/commands.py` | Human slash commands that spend no model turn. |
| `ctx.permission_presets` | `seams/permission_presets.py` | One name for a sandbox mode *and* an approval policy. |
| `ctx.settings` | `seams/settings.py` | Durable user preferences, read as data. |

## Agents and work

| service | module | |
|---|---|---|
| `ctx.subagents` | `seams/subagents.py` | Delegation to a child agent, and the handle it returns. |
| `ctx.subagent_presets` | `seams/subagents.py` | The named grants a child may be admitted under. |
| `ctx.jobs` | `seams/jobs.py` | Background work with a handle, a cancel and a completion. |
| `ctx.schedule` | `seams/schedule.py` | Work a root will do later, folded from its own log (P5-06). |
| `ctx.goals` | `seams/goals.py` | An objective, a budget, and the gates that decide it (P5-07). |

`seams/schedule_index.py` answers which sessions hold a live appointment (P6-23).

## The front end

| service | module | |
|---|---|---|
| `ctx.tui_screens` | `seams/tui_screens.py` | The front end's registration seam (P4-17). |
| `ctx.tui_status` | `seams/tui_status.py` | A live reading in the footer, contributed by a row. |
| `ctx.diagnostics` | `seams/diagnostics.py` | What a row wants `ph doctor` to say about it. |

`seams/topology.py` contributes doctor's Topology section: what the mount *became*.

## Secrets, records and invariants

| service | module | |
|---|---|---|
| `ctx.credentials` | `seams/credentials.py` | References travel, values do not (I-3). |
| `ctx.session_telemetry` | `seams/telemetry.py` | Records, redaction, and no span tracer. |
| `ctx.invariants` | `seams/invariants.py` | Which invariants this deployment enforces, and whether they hold. |

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
