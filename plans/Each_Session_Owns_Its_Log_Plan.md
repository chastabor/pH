# Each session owns its log — a sub-agent's record moves out of its parent's

*Phase 11. Decided with the user on 2026-09-30: every fact about a sub-agent lives in the
sub-agent's own session log, written through core's doors. A parent, the resume sweep,
budgets, caps and front ends work out a child's state by reading the child's log. The
parent's log keeps only the parent's own acts.*

## Context

Checked against the source at `818dff8`, plus the uncommitted `open_deferred` change,
which this plan does not touch. Line numbers are at that commit, and paths are under
`packages/`.

### What exists today

A child is a row in its parent's *roster*, which is folded from four record types in the
**parent's** log:

| Record (in the parent's log) | Written by | Rule it carries | Tag |
|---|---|---|---|
| `subagent/admitted` | the provider, through `record_admitted` (`ph/seams/subagents.py:758`) | the parent is flushed before the child's gate opens, fail-closed (`:1426-1434`) | S2 |
| `subagent/status` `running` (+ `cause`) | `record_started` (`:711`) | the parent is flushed before a *resumed* attempt | S10 |
| `subagent/status` `done` / `error` / `canceled` | `record_settled` (`:733`) | the child's log is flushed first, then the status goes to the parent | F1 |
| `subagent/status` `queued` / `error` | `record_status` (`:686`), the resume sweep, `_settle_unadmitted` (`:1334`) | none; the sweep flushes once before it readmits (`:1662`) | S10, K3 |
| `subagent/deleted` (+ `canceled`) | `record_deleted` (`:769`) | one batch; never `canceled` over `done` | S14 |
| `subagent/usage-attributed` | `usage_mirror` (`:815`), `reconcile_usage` (`:827`) | none while running; on resume, missing answers are copied back from the child's stored log | L5 |

`ph_rlm.subagents` has no log writer of its own since S10's follow-up. But its child's
drive calls these doors on the parent's `Session`, so every child writes into its
parent's log and sometimes flushes it. A child's flush never writes its parent
(`session/store.py:247-273`, P10-04). That is why S2, S10 and F1 each needed an explicit
order between two logs, and why L5 needed a catch-up step.

### What the child's own log already holds

From ph-rlm `subagents.py:422-452` and `agent_loop/driver.py`:
- a header with `parentSession`, `origin: "subagent"`, `delegationDepth` and
  `agentPreset`;
- the workspace records;
- the task, as an `agent/inbox/spliced` from `_TASK_SOURCE`;
- every `turn/start`, every `assistant/message` (with `usage`) and every `turn/end`.

The checkpoint policy flushes it before each model request and before each tool body. It
is not flushed after the child's last answer.

**What it lacks, compared with the admission record:** the name, run id and call id; the
requested access and downgrade reason; preset, profile, model key, skills, tools, paths
and effort; the raw prompt; and every status.

**The header reaches disk only at the first flush** (`persistence/jsonl.py:474-478`,
`turso.py:171-187`), and it is never rewritten after that (`session/session.py:160-164`).
So anything about a child that changes has to be an event, not a header field.

### Who reads the roster

All of these fold the parent's log.

- **ph-core:**
  - `SubagentService.roster` (`:1533`), a `SessionFoldCache` with no `extend`;
  - `resume_children` (`:1584`), `_reconcile_answers` (`:1665`), `_readmit_children`
    (`:1728`), `_request_of` (`:1840`), `_readmit_one` (`:1871`) and `readmit_waiting`
    (`:1701`);
  - `name_of` and `roster_name` (`:1554`, `:2306`), `admitted_by` (`:868`) and
    `child_is_live` (`:270`).
- **The `task` tool's crash check** (`tools/builtin/subagent_task.py:240-256`) finds the
  child of a cut-short call by `callId`. It reads a copy of the stored parent log, before
  any child is open.
- **Goals** (`seams/goals.py:51,225-277`): the `children` token source is charged to
  whichever goal is open where each usage record lands in the parent's log.
- **ph-stabilize spawn caps** (`ph_stabilize/limits.py:317-414`) count `subagent/admitted`
  per session and per parent turn.
- **ph-rlm:**
  - sibling-name uniqueness (`subagents.py:240`);
  - `bindings.py:201-208` (`reconciled_spawn`, `list_subagents`);
  - `messaging.py:469-479,613-620` (message recipients and names);
  - `prompt.py:194-211` (the "your children" line).
- **phern:**
  - the TUI roster (`tui/adapter.py:1037-1101`, `tui/state.py:248-282`,
    `tui/widgets/status.py:68-118`);
  - passivation (`daemon/supervisor.py:2192`);
  - the credential rows (`:829-837`).

  Clients receive only the root's stream (`daemon/follow.py:122`,
  `supervisor.py:1232-1235`). That is the only reason the panel works today.

**Already reading child headers, so nothing changes for them:** `reachable_family`,
`descendants` and `family_reach` (`seams/subagents.py:2228-2336`), the workspace
`stored_survivors` sweep (`seams/workspace.py:2141-2187`), and the delegation depth gate.

### Found while researching: defects this plan removes or must not repeat

1. **A child's provider is never recorded.** `SubagentRun.owner` is left out of
   `to_wire` (`:544-551`). `_readmitter` therefore resolves `None`, which only works when
   exactly one provider is mounted (`:1074`, `:1909`).
2. **A refused spawn nulls the row's `sessionId`.** `start` calls `_admit` without
   `session_id`, and `_settle_unadmitted` writes `sessionId: None` over the admission's
   value (`:1433`, `:1358`).
3. **`cause` and `detail` carry over.** The fold merges each status with `row.update`
   (`:2171`), so a later status keeps an earlier status's `cause`.
4. **The sweep changes cached rows.** `resume_children` makes a shallow copy of the cached
   roster and then sets `row["status"]` on the shared row objects (`:1629`, `:1649`).
5. **Only an in-memory child can be deleted.** ph-rlm's `delete` pops `_children`
   (`subagents.py:903`), so a child settled by an earlier process answers `False`.
6. **A child's id can be attached as a root.** `session/attach` goes through
   `Supervisor.start`, which has no `is_subagent` guard (`daemon/server.py:1007`,
   `supervisor.py:1009-1056`). Not run to confirm.
7. **A fork inherits its source's roster.** The fold reads a fork's whole log, including
   the prefix it copied from its source (`roster_of(session.events)`, `:2122`), and the
   spawn caps do the same (`limits.py:441-443`). Not checked: whether a fork's resume
   sweep can then reach its source's running children.
8. **The spawn-cap guard races admission.** Guards run (`seams/subagents.py:1489-1493`)
   before the provider awaits the child's session and workspace, so two concurrent spawns
   can both pass (`limits.py:749-793`). This predates the plan; see Non-goals.

## Decided with the user (2026-09-30)

1. **Each session owns its log.** A child records its admission, attempts, waits, ending
   and deletion in its own log, through doors in `ph.seams.subagents` that keep each
   record's durability rule. Nothing about a child is written to its parent's log. A
   parent, the resume sweep, budgets, caps and front ends work out a child's state by
   reading the child's log. One writer per log also means one writer per lock: no child
   appends to or flushes its parent's log.
2. **A delete goes in the child's log.** The tombstone is the child's record, landing in
   one batch with the `canceled` that ends a child that hadn't ended (S14). So the child's
   log alone tells its whole story. The parent's log keeps the parent's act only as the
   tool call that asked for it.
3. **No backwards compatibility** (standing since 2026-09-24).
   `SESSION_FORMAT_VERSION` moves to 3 and `PROTOCOL_VERSION` to 6, once each. Logs and
   daemons from before are refused, not migrated. What breaks is listed below.

## Decided with the user (2026-09-30): the four recommendations

Taken as recommended; the alternatives are kept for the record.

1. **One id or two.** Today a child has a run id (`child-<12hex>`) and a session id
   (`<parent>-<run id>`), one to one.
   - *Recommended:* keep `runId` as a field of the child's admission. Bindings, messaging,
     the TUI and the `task` result all address children by it. Merging the two ids can be
     its own change.
   - *Alternative:* the session id is the only id. DESIGN already says "an agent's id is
     its session's id".
2. **Which goal a child's spend is charged to, and how deep.** Today a child's answers
   are charged to whichever parent goal is open when each usage record lands, and only one
   level down: a grandchild's spend, and a child's compaction spend, never reach a root's
   goal.
   - *Recommended:* stamp the parent's open goal id on the child's admission. Charge that
     goal with the spend of every descendant admitted under it (`descendants` already
     walks the headers). A goal is a budget for a whole delegation tree; stopping at one
     level was a limit of the mirror, not a choice anyone made.
   - The default `token_sources` stays `["own"]` either way.
   - *Alternative:* keep one level.
3. **How clients learn child state.** Clients see only the root's stream.
   - *Recommended:* the daemon builds the children list from the child logs and serves it
     two ways, both with precedents: a `session/children` projection (like `readings_of`),
     and a pushed `session.children` notice (like `SessionScreensNotice`).
   - *Alternative:* forward each child's event stream to clients. That is heavier (every
     child chunk goes to every watcher) and the panel doesn't need it.
   - Opening a child's transcript live is a later row either way.
4. **Spawn caps and forks.**
   - *Recommended:* the admission carries the parent's turn, as the seq of the parent's
     latest `turn/start`, so "children spawned this turn" stays exact.
   - A fork is a new root, so it starts with no children and no spawn counts. Today it
     inherits both (defect 7).

## What breaks

- **Every log written before format 3 is refused** by the header check
  (`session.py:193-202`), just as format-1 logs were at the bump to 2. Sessions from
  0.5.x cannot be resumed.
- **A 0.5 front end and a 0.6 daemon won't talk** (`PROTOCOL_VERSION` 6). Restarting the
  daemon fixes it, as before.
- **A root's log no longer shows its children.** Its transcript shows the calls that
  spawned and deleted them; the panel comes from the daemon's notice.
- **Forks no longer inherit their source's children or spawn counts** (decision 4).
- **Goal budgets that count `children` now count every descendant**, charged to the goal
  that was open at admission (decision 2).
- **A sub-agent's log can no longer be attached as a root** (P11-08).
- **Public API:**
  - Removed: `subagent_roster`, `roster_of`, `fold_subagent_event`, `usage_mirror`,
    `reconcile_usage` and `AttributingProvider`.
  - Changed: the doors take the child's `Session`, not the parent's.

## The design

### The child's records

All of these live in the child's own log and are written through `ph.seams.subagents`,
which stays their writer of record in `_WRITTEN_BY` (`known_event_types.py:571-585`).
All are **required reading**: a reader that skipped one would miscount the retry ladder
or bring back a deleted child.

| Type | When | Durability rule (the door keeps it) | Replaces |
|---|---|---|---|
| `subagent/admitted` | once, when the provider has resolved the child | on the child's disk before its gate opens; fail-closed | S2's parent flush |
| `subagent/status` `queued` | a wait: for a slot (`slots`), or suspended (`SUSPENDED_DETAIL`) | none | `record_status` |
| `subagent/status` `running` | an attempt starts; `cause` is `resumed` or `rehydrated` | on the child's disk before a `resumed` attempt | S10's parent flush |
| `subagent/status` `done` / `error` / `canceled` | the ending | on the child's disk **before the result is handed to the parent** | F1's two-log order |
| `subagent/deleted` | a delete | one batch with `canceled` if the child hadn't ended; on disk before the child's runtime is released | S14, now in the child's log |

`subagent/usage-attributed` is **removed**: every answer's `usage` is already on the
child's `assistant/message`. The type names stay the same. The format bump means an old
log that used them the old way is refused rather than misread.

**The admission payload** is today's `admission_payload` (`:634-671`), minus the fields
the header now carries (`parentId`, `sessionId`), plus:
- `owner`: the plugin row that provides the child, so readmission finds the right
  provider (defect 1);
- `callId`: already there; it is now how the `task` crash check finds the child;
- `parentTurn`: the seq of the parent's latest `turn/start` (decision 4);
- `goalId`: the parent's open goal, if there is one (decision 2).

**F1 becomes a rule inside one log, plus an order.** `record_ended` flushes the child's
ending before `run.result` resolves. The parent then records whatever it keeps of the
result, such as a tool result, in its own log. A crash can leave an ended child whose
parent never heard; the parent's call is then settled by P11-06's crash check. It can
never leave a parent holding an answer while its child still reads as unfinished.

**A child log with no admission is not a child.** The workspace records can reach disk
before the admission, because `workspace/acquiring` is made durable before the tier acts.
Every reader skips such a log, just as the fold today skips records with no admission row
(`:2156`). The workspace survivors sweep reclaims its tree, as it does now.

### The doors (core, `ph.seams.subagents`)

```python
def record_admitted(child: Session, run: SubagentRun, request: SubagentRequest, *,
                    owner: str, parent_turn: int | None, goal_id: str | None) -> SessionEvent
def record_waiting(child: Session, **extra: JsonValue) -> None     # queued: slots / suspended
async def record_started(ctx: Context, child: Session, *, cause: StatusCause | None) -> SessionEvent
async def record_ended(ctx: Context, child: Session, status: SettledStatus,
                       **extra: JsonValue) -> SessionEvent
async def record_deleted(ctx: Context, child: Session, reason: str) -> None
```

- Every door takes the **child's** session, and none takes a parent.
- `record_deleted` reads "already ended" from the child's own state, not from a caller's
  flag.
- No door needs the child's scope, which is already gone on the parent-teardown path
  (ph-rlm `subagents.py:884-893`). The session belongs to the store, and the flush is
  `session_written(ctx, …)` on the mount.

### Working out a child's state

- **`child_state(log) -> ChildState`** is a pure fold over *one* child's own log. It
  holds:
  - the admission payload;
  - `status`, `cause` and `detail`, **set per status and never merged** (defect 3);
  - `deleted` and `deletedReason`;
  - `starts`, `resumes` and `resumesAtLastAnswer`, where an answer is an
    `assistant/message` in the child's own log;
  - usage totals.

  It is typed (a dataclass), not today's `dict[str, Any]` row. Each child session gets
  one `SessionFoldCache` **with `extend`**, contributed as `subagent-fold-cache` and held
  to `check_fold_laws`. Because each fold reads only its own log, the rule that a cached
  fold always equals a fresh one (I6) holds per child, and the staleness poll needs no
  key spanning several logs.
- **`SubagentService.children(parent_id) -> Mapping[str, ChildState]`** is a join, not a
  fold. It takes the parent's children — live ones from `SESSIONS.list()` by
  `delegating_parent`, stored ones from P11-01's listing — and runs each through
  `child_state`. A stored child that isn't live is read once per mount, off the event
  loop, and cached until it is opened. Callers use this where they used
  `roster(parent_session)`.
- **Budgets and caps combine at read time, and their own folds stay pure.**
  - `GoalService`'s fold keeps folding the parent's log. `charged_tokens` adds the
    children's spend from `children` (decision 2).
  - The caps' fold keeps counting the parent's turns. The child counts come from
    `children`, matched by `parentTurn`.

### The resume sweep, reading each child's own log

`resume_children` keeps its shape and loses its first step:

1. List the parent's children (P11-01) and read each one's state.
2. **No catch-up step.** The ladder reads answers from the child's own log, which is
   where they are.
3. A `running` child is judged `spent` or `recoverable` as now.
   - A resumable child is readmitted. Its drive writes `running, cause resumed` on its
     own disk before the attempt.
   - A spent or unrecoverable child is opened (its lease claimed), ended with `error` and
     the detail, flushed, and let go.
4. `queued` children are readmitted as now. One that is waiting for a credential records
   the hold **in its own log**, as `CREDENTIAL_WAIT` with `SESSION_HOLDER`. That is what
   a session waiting on its own route already writes.
5. A readmitted child's own children are swept before its gate opens (L5b), unchanged.

Nothing orders one child's records against another child's, or against the parent's:
each decision lives in the log it is about. A crash mid-sweep leaves each child either
decided on its own disk or undecided, and the next start decides the undecided ones the
same way.

## Phase 11 rows

| Row | What it lands | Depends on |
|---|---|---|
| P11-01 | **Landed.** The store lists a session's children | — |
| P11-02 | **Landed.** The child's records and doors; format 3 | decisions 1, 2, 4 |
| P11-03 | **Landed.** `child_state` and `SubagentService.children` | P11-02 |
| P11-04 | **Landed.** The seam reads children, not a roster: admission, sweep, readmission, credential holds, delete | P11-01, P11-03 |
| P11-05 | **Landed.** ph-rlm writes only its children's logs | P11-04 |
| P11-06 | **Landed.** The `task` crash check, goals and spawn caps read children | P11-03, decisions 2, 4 |
| P11-07 | **Landed.** The daemon: passivation, credential rows, the children projection and notice; protocol 6 | P11-04, decision 3 |
| P11-08 | **Landed.** A sub-agent's log is never mounted as a root | — |
| P11-09 | **Landed.** The TUI panel from the notice | P11-07 |
| P11-10 | **Landed.** Docs, `NON_GUARANTEES`, bookkeeping | all |

**Order.**
- P11-01 and P11-08 don't depend on anything and can go first.
- P11-02 → P11-03 → P11-04 → P11-05 is one path. Until P11-05 lands, ph-rlm still writes
  the old records, so these four go on one branch, reviewed row by row. The tree is
  green again only once P11-05 is in.
- P11-06 and P11-07 come next, then P11-09 after P11-07. P11-10 is last.

---

## P11-01 — the store lists a session's children

*(Landed. `SessionArchive.children_of(parent_id, family)` — since renamed `descendants_of`,
which lists every level beneath the parent — with the rule shared by both
backends and both test stores in `protocol.children_among` (now `descendants_among`), and the scan in
`families.children_under`. Turso answers per candidate, not by one query: each session
is its own database, so no table holds two headers. `read_own(id, family=)` already
existed and is what a child is read with. An empty family is refused rather than
scanning the sessions root.)*

**Why.** Every reader now starts from "who are this session's children", including
children settled by an earlier process.
- `SessionArchive.stored(limit=…)` stats every log in the store, then truncates
  (`persistence/jsonl.py:556-569`, `families.py:62-88`, `SURVEY_LIMIT` 500).
- So a parent's children can fall below the cut. That is fine for the picker, but not for
  a ladder, a budget or a cap.

**What.** `children_of(parent_id: str, family: str) -> tuple[StoredSession, ...]` on
`SessionArchive`, in both backends:
- **JSONL:**
  - One `scandir` of the family directory: children inherit their parent's family
    (`session/store.py:197-200`).
  - Filter by the id prefix `f"{parent_id}-"`: child ids are `<parent>-child-<hex>`
    (ph-rlm `subagents.py:239,249`).
  - Then peek at each header to confirm `delegating_parent == parent_id`. The prefix only
    narrows the search, because a grandchild shares it; the header decides.
- **Turso:** the same answer from one query over the headers, rather than one connection
  per peek (`turso.py:310-343`).

Reads by child id pass the family (`read_own(id, family=…)`), instead of paying
`locate_under`'s scan of every family (`families.py:91-111`).

**Files:** `ph/persistence/protocol.py`, `jsonl.py`, `turso.py`, `families.py`;
`tests/test_persistence_backends.py`.

**Gates** (both backends):
- `test_a_parent_lists_every_child_past_the_survey_limit`: `SURVEY_LIMIT + 1` children,
  and every one is listed.
- `test_a_grandchild_is_not_its_grandparents_child`.
- `test_a_fork_is_not_a_child`.

*Sabotage:* answer with `stored()` filtered by parent, and the first gate misses the
children below the cut.

## P11-02 — the child's records and doors; format 3

*(Landed. Two changes to the design above. **The seam writes the admission**, not the
provider — `SubagentService._admit` holds both the resolved request and the provider's
run, and it is the one that knows `owner` — so a provider cannot forget it and the check
the design gave P11-04 became the write itself. And `record_admitted` stays synchronous;
the flush is `_admit`'s. A child's log must name its parent twice over — the header's
`parentSession` and the `<parent>-` id prefix `children_of` narrows by — or the spawn
is refused; ph-rlm now passes the parent's `family` in the child's header meta, since
the store files a child with its parent only while the parent is live in it.)*

**Why.** The rules S2, S10, F1 and S14 carry are kept by doors today. This row moves them
to the log the records are about.

**What.**
- The five doors above, in `ph.seams.subagents`, each taking the child's `Session`.
  `record_admitted` builds on `admission_payload`, adding `owner`, `parentTurn` and
  `goalId` and dropping `parentId` and `sessionId`.
- `subagent/usage-attributed` leaves the vocabulary (`known_event_types.py:273-284`).
- `subagent/status` leaves the ignorable set (`:389-394`): the ladder reads it, so it is
  required now.
- `_WRITTEN_BY` keeps `ph.seams.subagents` as the one writer of the three remaining types.
- `SESSION_FORMAT_VERSION` goes to 3 (`session/events.py:41`).
- Durability per door:
  - `record_ended` awaits the child's flush before it returns.
  - `record_started` awaits it for `cause: "resumed"`.
  - `record_deleted` writes one batch and awaits the flush.
  - `record_admitted` and `record_waiting` only append. The admission's flush belongs to
    the seam (P11-04) and is fail-closed.

**Files:** `ph/seams/subagents.py`, `ph/session/known_event_types.py`,
`ph/session/events.py`; `tests/test_subagent_status.py` (rewritten against the child's
log), `tests/test_log_writers.py`.

**Gates:**
- `test_a_resumed_attempt_is_on_the_childs_disk_before_it_runs` (S10).
- `test_an_ending_is_on_the_childs_disk_before_the_result_is_handed_over` (F1).
- `test_a_delete_lands_whole_in_the_childs_log` and
  `test_a_delete_never_cancels_an_ended_child` (S14).
- `test_log_writers`: the three types are pinned to the one module.

*Sabotage:* drop the flush from `record_ended`, and the F1 gate reads the child's store
and finds no ending.

## P11-03 — `child_state` and `SubagentService.children`

*(Landed. `ChildState` is a frozen dataclass; `extend_child_state` is the cache's
extender; `load_children(parent_id, family)` reads the stored children once per parent
per mount and `_let_go` keeps a child's last state when its session is disposed.
`ChildState.awaiting` folds the child's own credential hold. `tokens` counts a child's
compactions as well as its answers. A lease claimed twice by one scope is now a no-op
(`lease.claim_session`), since the sweep opens a stored child to write its ending or
hold and may readmit it later in the same mount.)*

**Why.** One pure fold per child replaces the roster fold over the parent, so every
reader asks the same question of the same log.

**What.**
- `ChildState`, `child_state(log)` and `fold_child_event`:
  - `status` is a `SubagentStatus`, and `cause` a `StatusCause | None`;
  - fields are set per status and never merged (defect 3);
  - an answer is an `assistant/message` in the child's own log;
  - usage is summed with `reported_usage`.
- `restarts_since_progress(state)` keeps its meaning: `resumes - resumesAtLastAnswer`.
- `SubagentService.children(parent_id)`:
  - live children by `delegating_parent`, and stored ones through P11-01;
  - each through a per-child `SessionFoldCache` with `extend` (the roster cache had none,
    `:1017`);
  - a child that is only stored is read once per mount, off the event loop.
- `child_is_live(state)` keeps its rule: an unrecognized status counts as live.
- `name_of`, `roster_name` and `admitted_by` are rewritten over `children`.
- The states handed out are immutable (defect 4).

**Files:** `ph/seams/subagents.py`, `ph/seams/invariants.py` (the cache row);
`tests/test_fold_laws.py`, new `tests/test_child_state.py`.

**Gates:**
- `check_fold_laws(child_state)`.
- `test_a_later_status_does_not_keep_an_earlier_cause`.
- `test_an_answer_forgives_the_restarts_before_it`.
- `test_a_stored_child_is_read_once_per_mount`.
- `test_the_child_cache_is_polled_as_an_invariant`.

*Sabotage:* merge statuses with `update`, and the second gate fails.

**Measure before closing the row:** the cost of reading N stored children at mount, for
N = 32 and 256, with logs of 5k and 50k events (mostly `assistant/chunk`). If it matters,
read a stored child's state on first use rather than at mount.

## P11-04 — the seam reads children, not a roster

*(Landed. `SubagentService.delete` takes the parent's `Session`, not its id: it loads the
parent's stored children by the parent's family before looking. `start` also loads them
before the spawn guards run, so a cap and a provider's sibling-name check count children
from an earlier process even when no resume sweep ran. A parent with no log of its own
records nothing for its child — the one spawn that is not durable. The credential hold is
checked against the route first and the child's log opened only when the hold starts or
ends. The stub opens a child session named for its parent.)*

**Why.** Every decision the seam makes about a child — admit, resume, readmit, hold,
delete — now reads and writes that child's log.

**What.**
- **Admission (S2, fail-closed).** `_admit` checks that the child's log holds its
  admission (`child.latest(ADMITTED)`) and refuses a provider that didn't write one;
  that's one lookup. It then flushes the **child**. A write that fails refuses the spawn,
  as today.
- **`_settle_unadmitted`** ends the child in its own log (`record_ended`). There is no
  `sessionId` field left to null (defect 2).
- **Readmission** reads the request from the child's admission (`_request_of`), finds the
  provider through the admission's `owner` (defect 1), and takes `starts` from
  `child_state`.
- **The sweep** works as designed above: no `_reconcile_answers`, no parent writes, no
  parent flush. A spent child is ended in its own log.
- **Credential holds** go in the child's log with `SESSION_HOLDER`
  (`seams/credentials.py:146-183`). `waiting_for` covers a root and all its descendants.
- **Delete moves into the seam.** `SubagentService.delete(parent, run_id, reason)` writes
  the tombstone into the child's log, whether the child is live or was settled by an
  earlier process (defect 5). It then asks the provider to release a live runtime: the
  provider's existing `delete`, minus its record.
- **Removed:** `AttributingProvider`, `_reconcile_answers`, `usage_mirror`,
  `reconcile_usage`, `_record_usage`, `_answer_usage`, `subagent_roster`, `roster_of`,
  `fold_subagent_event` and `_rosters`.
- **The stub provider** (`ph/testing/stub_subagent.py`) opens a child session and writes
  its admission through the door, so a stubbed spawn meets the same check.

**Files:** `ph/seams/subagents.py`, `ph/seams/credentials.py`,
`ph/testing/stub_subagent.py`; `tests/test_subagent_grant.py`,
`tests/test_subagent_task.py`, `tests/test_seams.py`.

**Gates:**
- `test_a_child_is_on_its_own_disk_before_its_gate_opens` (S2). Crash-injected: hold the
  gate, snapshot the store, resume.
- `test_a_provider_that_wrote_no_admission_is_refused`.
- `test_a_spent_child_is_ended_in_its_own_log`.
- `test_readmission_finds_the_provider_that_admitted_the_child`: two providers mounted.
  This fails on today's code (defect 1).
- `test_a_child_settled_by_an_earlier_process_can_be_deleted` (defect 5).
- `test_a_held_child_waits_in_its_own_log`.

*Sabotage:* flush the parent instead of the child in `_admit`, and the first gate resumes
a store with no child in it.

## P11-05 — ph-rlm writes only its children's logs

*(Landed. The provider calls no admission door — the seam writes it (P11-02) — and its
`delete` became `revoke(run_id, reason)` (`RevokingProvider`), reached through
`SubagentService.delete`. The `turn/start` fallback for counting restarts went: a
child's `running` is in its own log ahead of its turns. ph-rlm's suite, 422 tests, and
the SIGKILL mass-restart tests pass; the S2, S10 and F1 gates now read the store at the
moment the act starts, since a child's checkpoint flush would otherwise make them pass
for free. The one-writer gate allows the parent inbox splices a child's notices and
messages make — messages, not records about the child.)*

**Why.** The provider is the one module left that hands a parent's `Session` to a door.

**What.**
- **`_admit`** calls `record_admitted(child_session, …)` with `owner`, `parentTurn` and
  `goalId`. Sibling-name uniqueness reads names from `children` (`subagents.py:240`).
- **`_attach`** loses the usage mirror and `unobserve`. `on_queued` becomes
  `record_waiting(child, slots=…)`.
- **`_drive`** calls `record_started(ctx, child, cause=…)` and
  `record_ended(ctx, child, …)`. The `turn/start` fallback for counting restarts (`:383`)
  goes, since `starts` is the child's own count now.
- **`delete` and `_release`** call the seam's delete. The parent-teardown path writes the
  tombstone without needing the child's scope.
- **`suspend`** writes `record_waiting(child, detail=SUSPENDED_DETAIL)`.
- **`reconcile_answers`** goes.
- **`bindings.py`** (`reconciled_spawn`, `list_subagents`), **`messaging.py`**
  (recipients among children and siblings, and names) and **`prompt.py`** (the family
  line) read `children`.

**Files:** `ph-rlm/src/ph_rlm/{subagents,bindings,messaging,prompt}.py`;
`ph-rlm/tests/test_{subagents,bindings,messaging,prompt,subagent_profiles,bundle}.py`.

**Gates.** The existing admission, restart, ending, delete, ladder, readmit, rehydrate
and suspend tests, pointed at the child's log (`test_subagents.py` 176–427, 530–682 and
1151–1706; `test_messaging.py` 281–463). Plus:
- `test_a_parents_log_holds_no_record_of_its_child`. Spawn, answer, rehydrate and delete:
  the parent's log gains no `subagent/*` record, and nothing is appended to it except by
  its own turn. **This is the one-writer gate.**
- `test_a_crash_after_an_answer_is_counted_exactly`. The child answers, only the child's
  log is written, and the mount restarts. The ladder forgives the restart and the budget
  counts the answer, with no catch-up step. This is the L5 case, now fixed by
  construction.

*Sabotage:* put a mirror into the parent's log back, and the first gate fails.

## P11-06 — the `task` crash check, goals and spawn caps read children

*(Landed. `GoalService` holds `ctx` and `spent(session, state)` joins
`SubagentService.delegated_tokens`; `/autonomous` reads the joined spend. ph-stabilize's
caps join `children` at read time in `refuse_child` (`child_counts`), and a deleted child
still counts, as before. The known limit this row shipped with — the children of a
child that was not readmitted after a restart went unread — was closed after the
phase: `descendants_of` reads a whole tree, and the sweep revokes what an ended child
left unfinished.)*

**Why.** Three readers outside the seam fold the parent's log for facts about children.

**What.**
- **The `task` crash check** (`tools/builtin/subagent_task.py:240-256`) runs at resume,
  before any child is open. It lists the children with `children_of(parent, family)` and
  matches each child's admission by `callId`. That needs only the start of each log (the
  header and first events), not the whole file. `NotDone`, `Unknown` and `Done` work as
  now.
- **Goals** (`seams/goals.py`): when `children` is in `token_sources`, `charged_tokens`
  adds the usage of every descendant whose admission carries the goal's id (decision 2).
  The goal fold stays pure over the parent's log, and `CHILD_USAGE` leaves
  `_TOKEN_RECORDS`.
- **Spawn caps** (`ph_stabilize/limits.py`):
  - the session count is the parent's admitted children;
  - the turn count is those whose `parentTurn` is the parent's current turn;
  - the caps' fold keeps counting the parent's `turn/start` records, and `ADMITTED`
    leaves `_COUNTED`;
  - a fork starts at zero (decision 4).

**Files:** as named; `tests/test_subagent_task.py`, `tests/test_goals.py`,
`tests/test_fold_laws.py`, `ph-stabilize/tests/test_limits.py`.

**Gates:**
- `test_a_cut_short_task_finds_its_child_on_disk`.
- `test_a_goal_counts_a_grandchilds_spend`.
- `test_spend_is_charged_to_the_goal_open_at_admission`.
- `test_the_turn_cap_counts_children_admitted_this_turn`.
- `test_a_fork_starts_with_no_children`.

*Sabotage:* count `subagent/admitted` in the parent's log, and the cap gates see nothing.

## P11-07 — the daemon

*(Landed. `session/children` and `session.children` carry `ChildRow`s for the whole
family, a parent before its own children. The supervisor listens on the store-wide
`session/event` for ids under its root and pushes when a child's cached state changes,
coalesced to one frame per checkpoint. Passivation counts every live member of the
family — once the sweep revokes what an ended child left unfinished (closed after the
phase), none of them is an orphan. Credential rows walk the family. `PROTOCOL_VERSION` 6 also carries P11-08's `not_a_root` and
`SessionSummary.origin`.)*

**What.**
- **Passivation** (`daemon/supervisor.py:2151-2199`): a root is held while
  `any(child_is_live(s) for s in children(root).values())`.
- **`_resume_children` and `credential_supplied`** make the same calls as today.
  `_waiting_on` and `awaited` read credential holds across the root and all its
  descendants.
- **A `session/children` projection and a `session.children` notice** (decision 3).
  - Each row carries `runId`, `sessionId`, `name`, `model`, `grantedAccess`,
    `downgradeReason`, `status`, `cause`, `detail`, `deleted`, `deletedReason` and
    `tokens`. Grandchildren are included, with a `parent` field.
  - The notice is pushed when a child's log changes. The supervisor listens on the root
    context's `session/event`, filtered to sessions descending from the root; today it
    deliberately doesn't (`supervisor.py:1232-1234`). Pushes are coalesced like the
    status notices.
- **Wire:** both are added to `VOCABULARY` (`ph_app/verbs.py:226-257`) and routed
  (`test_daemon_methods`). `PROTOCOL_VERSION` goes to 6, with its docstring entry
  (`ph_app/protocol.py:82`).
- **`NON_GUARANTEES`:** the "facts across two logs" row (`supervisor.py:300-308`) drops
  the child-usage lag it describes.

**Files:** `ph_app/daemon/{supervisor,server,projections}.py`, `ph_app/payloads.py`,
`ph_app/verbs.py`, `ph_app/protocol.py`; `tests/test_daemon.py` (`:1397` rewritten over
child logs), `tests/test_daemon_methods.py`, `tests/test_payloads.py`.

**Gates:**
- `test_a_root_with_a_live_child_is_not_released`, over child logs.
- `test_a_childs_status_reaches_the_client_as_a_notice`.
- `test_the_children_projection_lists_grandchildren`.
- `test_a_held_child_is_listed_as_awaiting_a_credential`.

*Sabotage:* stop listening to child events, and the notice gate times out.

## P11-08 — a sub-agent's log is never mounted as a root

*(Landed. Defect 6 was real: `Supervisor.start` mounted a stored child and appended to
its file. Refused in `Supervisor._start` with a new code, `not_a_root` (`NotARoot`),
naming the **top** root of the delegation line (`RecordedStart.owner`). The same refusal
for `phern -p --session <child>` and rpc, in `runtime.mount_session`, as a
`SessionForkError` coded `SESSION_IS_SUBAGENT`. The picker shows a child as a child
(`SessionSummary.origin`) and refuses to attach it rather than opening it read-only —
the TUI reads no session files — telling the person to use `--mode trajectory`. A
child's header `kind` was never `"fork"`; the picker drew any non-segment parent link as
a branch.)*

**Why.** Once the child's log is its only record, a second writer on it is the one thing
that can corrupt it.
- `session/attach` mounts any id (defect 6).
- The picker shows a sub-agent's log as a fork of its parent. Its `kind` is `"fork"`
  (ph-rlm `subagents.py:515`), and `SessionSummary` has no `origin`
  (`ph_app/sessions.py:67-106`).

**What.**
- `Supervisor.start` refuses a log whose header says `origin: "subagent"`, and names the
  root that owns it.
- `SessionSummary` gains `origin`. The picker nests a child under its parent as a child
  rather than a fork, and opens it read-only (`--mode trajectory`) instead of attaching.

**Files:** `ph_app/daemon/supervisor.py`, `ph_app/sessions.py`,
`ph_app/tui/modals/pickers.py`; `tests/test_daemon.py`, `tests/test_tui_state.py`.

**Gates:**
- `test_a_subagents_log_cannot_be_attached_as_a_root`. Confirm defect 6 first: the gate
  is written to fail on today's code.
- `test_the_picker_shows_a_child_as_a_child`.

## P11-09 — the TUI panel from the notice

*(Landed. `TuiState.subagents` holds the daemon's rows, keyed by session id, replaced
whole by each frame; an attach's projection is drawn only if no frame arrived while it
was in flight. The adapter's `subagent/*` handlers and the "Delegated to" / "Revoked
child" rows are gone — the spawn and delete calls' own cards show those — and the
trajectory view of a child's own log renders all three records.)*

**What.**
- The adapter's `subagent/*` handlers go (`tui/adapter.py:1037-1101`, and `RULES` at
  1244-1251). The panel folds the `session.children` notice into `TuiState.subagents`
  (`state.py:248-282`), and `SubagentRow` gains `session_id`.
- `children_heading` and `render_subagents` (`widgets/status.py:68-118`) read the rows as
  now.
- The "Delegated to …" and "Revoked child …" transcript rows are either drawn from the
  spawn and delete calls' own results or dropped, since the call cards already show
  them. That is decided in review, against
  `test_delegation_records_produce_no_transcript_rows`.
- The vocabulary gates (`test_tui_adapter.py:880-888`, `test_trajectory.py:74,176-181`)
  follow the new vocabulary. The trajectory view of a child's own log renders its
  `subagent/*` records.

**Files:** `ph_app/tui/{adapter,state,trajectory,remote}.py`, `ph_app/tui/widgets/status.py`;
`tests/test_tui_code_cell.py` (the panel tests at `:145-283`, rewritten over the notice),
`tests/test_tui_adapter.py`, `tests/test_trajectory.py`.

**Gates:**
- `test_the_panel_is_the_daemons_children_field_for_field`.
- `test_attributed_usage_is_summed_per_child`, now from the children's own usage.
- `test_a_revoked_child_stays_listed`.

## P11-10 — docs, `NON_GUARANTEES`, bookkeeping

*(Landed: `DESIGN.md` §2.5, §5.2, §5.4, §6.1–6.4 and the I2/I4 paragraphs;
`docs/seams/subagents.md`; `docs/cookbook/adding-a-seam.md`; L5 and L5b marked
superseded in `plans/Two_Log_Facts_Todo.md`; dated notes in
`reviews/11-seam-logging-audit.md`; `Implementation_Plan.md` Phase 11;
`docs/dev-notes/phase-11.md`.)*

- `DESIGN.md`:
  - §6.2–6.3: the roster, "usage is attributed upward", and the admission order;
  - the paragraph on facts across two logs (`:1418-1432`).
- Engineering rule 2 in `Implementation_Plan.md` §5 still says `rlm/child-admitted`.
- `docs/seams/subagents.md:163-238` and `docs/cookbook/adding-a-seam.md:97`.
- Mark as superseded by this phase: L5 and L5b in `plans/Two_Log_Facts_Todo.md`; S2, S10,
  F1, S14 and L5 in `reviews/11-seam-logging-audit.md`.
- A Phase 11 section in `Implementation_Plan.md`, and `docs/dev-notes/phase-11.md`.

## Reuse (do not rewrite)

| Need | Already there |
|---|---|
| Opening, resuming and leasing a child's log | `open_session` (`persistence/opening.py:26-150`) |
| A cached fold per log, checked for drift | `SessionFoldCache` with `extend`, `check_fold_laws`, `contribute_fold_cache` |
| A flush that doesn't raise | `session_written(ctx, session)` |
| Records that land whole | `Session.batch()` |
| A session waiting on its own route | `CREDENTIAL_WAIT` with `SESSION_HOLDER` (`session/kinds.py:534-552`) |
| Walking a family by header | `descendants`, `reachable_family`, `delegating_parent` |
| A parent's view built from child logs | `stored_survivors` (`seams/workspace.py:2141-2187`) |
| Pushing a derived view to clients | `SessionScreensNotice` and `_projection` (`daemon/supervisor.py:1168-1192`, `daemon/server.py`) |
| Crash injection without killing a process | `test_subagents.py`'s `_restart`, over a store snapshot |
| Reading a usage payload | `reported_usage` (`seams/token_meter.py:308`) |

## Non-goals

- **Atomicity across logs.** Two logs still have two schedules. This phase removes the
  facts that spanned them rather than tying them together. The one order left, a child's
  ending on its disk before its parent is handed the result, is kept by a door.
- **Live child transcripts in the TUI.** Forwarding a child's stream, or attaching
  read-only to a child over the daemon, can follow P11-07. P11-08's read-only trajectory
  view already covers reading one.
- **Merging run id and session id** (decision 1).
- **The spawn-cap race** (defect 8). It predates this phase and this phase doesn't change
  it: a guard still runs before admission. The fix, counting a spawn as in flight from
  the moment its guard passes, is its own row. *(Landed after the phase:
  `SubagentService.child_counts`, which counts the spawns on their way. The same race let two siblings
  take one name, so the seam now names every child right after the guards, and a spawn
  in flight holds its name on the same list.)*
- **Migrating logs** from format 2.

## Verification

- **Every row** ends with four gates green: `ruff check`, `ruff format --check`,
  `./test.sh types` and `./test.sh test`. The baseline is 3,713 passed with 8 opt-in
  skips (2026-09-30).
- **Every new gate is sabotage-checked** by reverting the mechanism it holds and watching
  it fail.
- **Crash injection** works as in Phase 10. Hold the act open on an `anyio.Event`, read
  the store while it is held, and resume that snapshot in a second mount.

**Definition of done:**
- A parent's log holds no record of any child.
- Every child's log alone says what the child was asked, how often it was started, what
  it spent, how it ended and whether it was deleted.
- A daemon restarted at any point brings every child back from its own log, with the
  ladder counting exactly and no catch-up step.
- The panel, budgets, caps and the `task` crash check agree with the children's logs,
  because they read nothing else.
