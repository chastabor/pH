# Phase 11 — Each session owns its log

**Status:** ten rows, all landed (2026-09-30). Four fixes landed after the phase:
the grandchildren of a child that did not come back, the descendants a child that
ends here leaves behind, the spawn-cap race and the sibling-name race (under "What
was traded").

**Gate:** `ruff` + `ruff format` + `mypy` across the tree, and `./test.sh test`: 3,747
passed after the `/simplify` pass, 8 opt-in skips, green on linux (from 3,713 before
the phase). The four later fixes add 18 tests, and the suite stays green. Every new
gate was sabotage-checked by reverting the mechanism it holds; the plan and the tests'
docstrings name each sabotage.

A sub-agent used to be a row in its parent's log. Its admission, its statuses, its
tombstone and a copy of what it spent were written there — by the child's own drive,
into another session's log. So a second account of every child sat beside the child's
own, and the seam-logging audit's durability fixes for children were all about keeping
the two in step: flush the parent before the child's gate opens (S2), flush the parent
before a restart's attempt (S10), write the child's log before the parent says `done`
(F1), copy the child's answers back into the parent after a crash (L5). Each was right,
and each was an order between two logs that nothing could make atomic.

This phase removes the copy. Every record of a child is in the child's own log, written
through `ph.seams.subagents`' doors, and everything that wants a child's state reads it
from there: `ChildState` is a pure fold of one child's log, cached per child, and
`SubagentService.children` joins them for a parent. One writer per log — a child never
appends to or flushes its parent's.

---

## What has landed

| Item | Delivered | Where |
|---|---|---|
| P11-01 | A parent's children listed from the store, past the survey limit | `SessionArchive.descendants_of`, `families.children_under` |
| P11-02 | Records and doors in the child's log; format 3 | `record_admitted/waiting/started/ended/deleted` |
| P11-03 | A child's state, folded from its own log | `ChildState`, `child_state`, `children`, `load_children` |
| P11-04 | The seam decides about a child from its own log | `_admit`, `resume_children`, `_held_for_credential`, `delete` |
| P11-05 | ph-rlm writes only its children's logs | `RlmChildProvider`, `RevokingProvider.revoke` |
| P11-06 | The `task` check, goal budgets and spawn caps read children | `admitted_by`, `GoalService.spent`, `SubagentService.child_counts` |
| P11-07 | The family, pushed to clients; protocol 6 | `session/children`, `session.children`, `ChildRow` |
| P11-08 | A sub-agent's log is never a root | `NotARoot`, `mount_session`, `SessionSummary.origin` |
| P11-09 | The panel from the notice | `TuiState.subagents`, `take_children` |
| P11-10 | Docs and bookkeeping | `DESIGN.md` §6, `docs/seams/subagents.md` |

---

## The things worth knowing

**The seam writes the admission, not the provider.** The plan had the provider write it
and the seam check that it had. The seam already held both halves — the resolved
request and the provider's run — and was the one that knew `owner`, so it writes the
record itself and the check disappears: a provider cannot forget what it never writes.
The admission also names the provider now. It never did, which meant a readmission
after a restart could only resolve "the one provider mounted", and with two, none.

**A child's log must name its parent twice.** The header's `parentSession` is what
decides, and the id's `<parent>-` prefix is how `descendants_of` narrows the family
directory before it reads a header. A spawn whose child log misses either is refused,
since a restart could never find it. ph-rlm passes its parent's `family` explicitly: the
store files a child with its parent only while the parent is live in it.

**F1 is one flush now.** `record_ended` flushes the child's ending before its caller
wakes whoever waits on the result, so a crash leaves at worst a child that ended under a
parent that never heard — never a parent holding an answer from a child that reads as
unfinished. That order is the one fact about a child that still spans two logs, and it
is kept by a door rather than by two logs' flush schedules.

**No catch-up step.** The ladder counts restarts and answers in the child's own log,
which is where the answers are, so the resume sweep decides with nothing to reconcile
first. `AttributingProvider`, `usage_mirror` and `reconcile_usage` are gone, and so is
`subagent/usage-attributed`.

**Opening a stored child twice in one mount.** The sweep ends a spent child, or records
a credential hold, in the child's own log without resuming it (`stored_session`), then
lets it go — and may readmit it later in the same mount. A lease is an `flock` on a
descriptor, so a second claim from this process refused its own holder. A claim from a
scope that already holds the id is now a no-op; the store still refuses a second live
`Session` for an id, so nothing inside one mount gets two writers.

**Statuses replace, they do not merge.** The roster folded each status into its row with
`update`, so a child woken after a restart went on reading `resumed`, and a refused
spawn's `sessionId: None` overwrote the admission's. Each field is now set by its own
record, whole.

**A sub-agent's log was mountable as a root.** `session/attach` on a child's id mounted
it and appended to it — a second writer on what is now the child's only record. The
daemon refuses with `not_a_root`, naming the top of the delegation line; `phern -p` and
rpc refuse through `mount_session` with `SESSION_IS_SUBAGENT`.

---

## What was traded

- **A stored child costs a read per mount.** `load_children` reads each child this
  process is not running once per parent per mount, off the loop. The resume sweep, the
  spawn guards, `delete` and the crash checks all go through it first.
- **Grandchildren under a child that did not come back** — *closed after the phase.*
  A restart first read children one level at a time, and only beneath children it
  readmitted, so a grandchild beneath a child that had ended was never read: its spend
  reached no goal, it was not listed, and nothing ended it — four readers each had to
  know to look past it. Now the store lists a whole tree in one read
  (`descendants_of`), `load_children` files every level, and the sweep revokes what an
  ended child left unfinished beneath it, in each descendant's own log
  (`PARENT_TEARDOWN`), finishing the teardown a crash cut short. In this process too:
  the doors that end a child (`record_ended`, `record_deleted`) take its unfinished
  descendants that nothing is running with it — one held for a credential, or one a
  delete reached on disk, used to stay `queued` and hold its root.
- **A fork starts with no children and no spawn counts.** It used to fold its source's
  roster from its seeded prefix, and so inherit children that were not its own.
- **The spawn-cap race** — *closed after the phase.* Guards ran before the provider
  built the child, and a child counted only once its admission was written, so two
  spawns from one step could both pass a cap one of them crossed. A spawn is now on
  its parent's list of spawns on their way from its last guard until its admission
  lands or it is refused, and the seam's count includes it (`SubagentService.child_counts`,
  which the limits row now asks rather than joining the children itself). The same race let
  two siblings take one name, since the rlm provider named a child from the admitted
  ones: the seam now names every child right after the guards, and a spawn in flight
  holds its name on the same list.
- **A 0.5 client and a 0.6 daemon do not talk**, and a format-2 log is refused.

---

## The version bump

Format 3 (`SESSION_FORMAT_VERSION`) and protocol 6 (`PROTOCOL_VERSION`) move together.
A format-2 reader would read a child's `subagent/admitted` as the child having admitted
a child of its own, so format-2 logs are refused rather than migrated. Protocol 6
carries the family projection and notice, `not_a_root`, and `origin` on browse rows.
