# Phase 4 — The stabilization bundle

**Status:** complete. Every row has landed.

**Gate:** `ruff` + `ruff format` + `mypy --strict` across all five packages + 1 250 tests, green.

Phase 3 gave pH the RLM's semantics. Phase 4 gives it the things that keep a
long run from falling over: planning, offload, compaction, limits, a human in
the loop, filesystem rules — and, taking most of the phase, **containment**: the
question of where an agent's writes actually land.

The rule the phase kept arriving back at is §4.8's: **a tier's name must not
promise more than the tier delivers.** Almost every correction below is a
version of it.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P4-01 | The bundle itself, and `write_todos` + `todo/write` + the prompt section (G1) | `ph_stabilize/todo.py`, `bundle.yaml` |
| P4-02 | Large-result and input offload to the spill store, with previews (G2, G3) | `ph_stabilize/{offload,input_offload}.py` |
| P4-03 | Threshold summarization, RLM-aware (G4, G10) | `ph_stabilize/compaction.py` |
| P4-04 | `/compact` | `ph_stabilize/compact_command.py` |
| P4-05 | Model- and tool-call limits (G5) | `ph_stabilize/limits.py` |
| P4-06 | `permissions-fs`: first-match-wins path rules over `ctx.fs` (G7, E9) | `ph_stabilize/permissions_fs.py` |
| P4-07 | The `ctx.workspace` seam, the `shared` floor, `scratch` (D21, E2, E5) | `ph/seams/workspace.py` |
| P4-08 | The `worktree` tier over real `git worktree` | `ph/seams/workspace_git.py` |
| P4-08b | Materials-only provisioning, `.ph-workspace.yml`, `/workspaces` (E14, E15) | `ph/seams/workspace_provision.py`, `ph/commands/workspaces.py` |
| P4-09 | Per-run checkpoints and `/revert` | `ph/seams/workspace_git.py`, `ph/commands/revert.py` |
| P4-10 | The default write scope, as a `permissions-fs` rule field (E6) | `ph_stabilize/permissions_fs.py` |
| P4-11 | The containment selector and `strict` (E1, E8) | `ph/seams/containment.py` |
| P4-12 | `ctx.diagnostics` and a `ph doctor` that mounts (E9, E10) | `ph/seams/diagnostics.py`, `ph_app/cli.py` |
| P4-13 | Memory after the cache, progressive skills, blocking delegation (G8, G9) | `ph/system_prompt/memory.py`, `ph/seams/skills.py`, `ph/tools/builtin/subagent_task.py` |
| P4-13b | Per-agent skills and tools, bounded by the parent (I7, B7) | `ph/seams/{skills,subagents}.py` |
| P4-14 | Paired-event reconciliation at session open (F6) | `ph/seams/workspace.py` |
| P4-15 | The `rlm-stable` profile; these notes | `ph_app/profiles/rlm-stable.yaml` |
| P4-16 | The ladder's claims asserted, escape included (E1, E13) | `ph-core/tests/test_containment_ladder.py` |
| P4-17 | `ctx.tui_screens` and the trajectory as its first registrant (I1, I2) | `ph/seams/tui_screens.py` |

---

## The five things worth knowing

### 1. `worktree` is not confinement, and saying so is the feature

The ladder is `advisory → worktree → sandbox`, and the middle rung is the one
that invites a wrong belief. A worktree bounds every tool-mediated write and
every *relative* raw write, because both resolve against the agent's cwd. It
does not bound `open("/etc/passwd", "w")`, which never consults a cwd at all.

What it buys is **collision isolation and revertibility** — eight children
writing one tree concurrently is the case the tier exists for — and that is
worth having on its own. What it does not buy is confinement, which only the
sandbox rung can claim. `ph doctor` prints the three columns rather than a
severity colour precisely because a colour invites a reader to skip the
sentence, and the sentence is the whole point.

The table has one home (`containment.TIERS`) so the prose and the command
cannot drift; P6-06's docs test inherits one thing to check.

### 2. A child may never hold more than its parent

P4-13b's ruling, and it shaped the mechanism rather than just the policy. The
obvious design scopes skills by owner, the way prompt sections are scoped — but
owner-scoped *registration* is a way to give one agent something another cannot
see, which is the widening the rule forbids. So the seam offers **narrowing
only**: filters that intersect, and no mechanism to add. The widening case is
not guarded against; it is absent.

To give a child a skill, grant it to the parent first. Applied down the tree,
the root's grant bounds every descendant and capability narrows monotonically —
which is what makes a fan-out auditable from one place.

Two consequences fell out of writing it. `AgentRegistry.create` scopes every
agent under the *registry*, so a child's scope is its parent's **sibling** and a
parent's filter does not reach it — inheritance has to be materialized, not
assumed. And a spawn may only ever narrow: registering on a child's scope is
unmaskable by design, so restrictions are the only instrument a spawn may use.

### 3. Placement is a design decision, not a detail

Three rows in this phase turned out to be about *where* something goes.

`AGENTS.md` moved from the cached prefix to a post-cache snapshot, because a
file whose purpose is to be edited must not re-bill every token before it when
someone edits it — and the same move made memory live, since the old row read
once at mount and an edit did nothing until a restart. Cheap and live were the
same change.

Skill bodies stay out of the prompt (a catalog of twenty is twenty bodies the
model probably will not need) — *except* the one a child was spawned for, which
it will certainly need. That inverts G9 for exactly the case where the question
G9 defers is already answered.

And a tool advertised in every prompt and refused on every call teaches the
model a capability the deployment does not have, so `task` registers only where
a subagent provider is mounted and `skill` only where a skill is installed.

### 4. The pair is what distinguishes a leak from a feature

A crashed process leaves a worktree behind. So does the disposal policy, when it
keeps a dirty tree for review. `git worktree list` reports them identically, so
neither the filesystem nor `/workspaces` can tell them apart — but the log can: a
`disposed` with `kept: true` is a decision, and no `disposed` at all is a
process that died holding the tree.

That asymmetry is F6, and it is why the seam writes both halves of the pair
rather than each provider writing one.

### 5. Six defects the tests did not find, and one they did

Worth recording because the pattern repeated: **the review that measures finds
what the review that reads does not.**

- `os.walk(followlinks=False)` silently dropped symlinked *directories* — most
  of a pnpm `node_modules` — from provisioning. `shutil.copytree(symlinks=True)`
  fixes it and is 2.2× faster.
- Git has no per-worktree `info/exclude`; the path a plan sentence named
  resolves to the shared file. Verified before building on it, and the mechanism
  moved out of git entirely.
- `branch -D` force-deleted committed work, because a clean worktree is not
  evidence that its branch was merged. `-d`, and the refusal names the flag.
- The memory provider read every `AGENTS.md` *before* checking whether anything
  had changed — the cache saved a `str.join` and cost an extra `stat`, while its
  own docstring promised the opposite. It was 97% of prompt assembly.
- Skill restrictions were a flat list, so a fan-out of sixteen children made the
  **parent's** assembly 3.9× slower with filters that could not affect it.
- Reconciliation folded the whole log, so forking a session reported the
  parent's *live* worktree as the child's leak — and would have removed a tree
  an agent was working in.

The one the tests did find was the worst: the suite created real `ph/*`
worktrees and branches inside the developer's own checkout, three times, because
`fs.root` defaults to the process cwd. Fixed once, in the `mount` fixture, which
now pins `fs.root` at `tmp_path` for the same reason it already pinned
`PH_HOME` — and `fs.root` is the stronger of the two, since it is what a tier
branches from.

---

## What `rlm-stable` is

The profile the phase builds toward: `rlm` plus `stabilize`, plus the two rows
both bundles deliberately ship **disabled**. A bundle that armed them on
layering would make "I want offload" mean "and also a tool, and also a corpus" —
so each row says in its own comment that a profile flips it, and this is that
profile.

The human gate **names `ph-stabilize`'s own pattern set** rather than retyping
it. The first draft did retype it, and had already widened `git push` from
force-only to every push — against a shipped test that pins an ordinary push as
ordinary. A security judgement with two homes is one that disagrees with
itself.

It is also the first profile that composes both capability bundles at once, so
it is where their rows meet for the first time: Code Mode's transport beside the
tool gates, one `permissions-fs` over both, one containment tier under both.

Children get worktrees; the person's own agent stays in the checkout they
opened. The human gate names two calls rather than everything, because a harness
that asks about everything teaches its user to approve without reading.

---

## What is left

Nothing in Phase 4. P4-16 closed it by asserting the ladder's claims rather than
its mechanisms — including, deliberately, that an absolute-path raw write
**escapes** `worktree`, so the tier table cannot quietly become a lie. The
`sandbox` half of that argument — the same write, refused — is P6-06, together
with the docs test that checks no tier is described as bounding writes it does
not bound.
