# `ctx.subagents` — delegation to a child agent, and the handle it returns

**Module:** `ph/seams/subagents.py` · **Rows:** `subagents`, `subagent-presets` ·
**Also documents:** `ctx.subagent_presets`, `ctx.named_profiles` ·
**Provider:** `rlm-child` (ships with the `rlm` bundle) · **Consumers:**
`subagent-task`, `rlm-*`

**Definition only in `ph-base`.** The seam mounts everywhere and registers no
provider, because "run a child agent" has genuinely different answers in
different deployments. `subagent-task` therefore registers *nothing* until a
provider is mounted — a tool advertised in every prompt and refused on every call
has taught the model a capability the deployment does not have.

## The handle returns before the child answers

`start()` resolves once the child is **admitted** — session created, admission
logged in the child's own log, task detached — not once it has finished.

That is the non-blocking fan-out an RLM parent depends on: it spawns eight
children, keeps working, and their replies arrive as ordinary inbox messages on
later turns. A contract where `start()` awaited completion would make the
parent's control loop serial and force it to poll.

Completion is available but **separate**: `SubagentRun.result()` awaits
quiescence, for the caller that genuinely wants to block — a generic `task` tool
returning the child's last text. Both callers use one provider, which is why the
answer is reachable but never the thing `start()` gives back.

```text
run = await ctx.subagents.start(name, request)   # returns at admission
answer = await run.result()                      # only if you want to block
```

## `access` defaults to read (E4)

A child asks for the workspace guarantee it needs, and the default is the
conservative one — a delegation that never mentions `access` **cannot silently
receive a writable repo**.

The handle reports both, and they may differ:

| field | |
|---|---|
| `requested_access` | what the parent asked for |
| `granted_access` | what the available tier could actually honor |

`granted` is what the parent's list of its children and the child's own prompt
report, because a child told nothing about its workspace attempts writes and reads
the failures as bugs.
This is the same request-versus-guarantee rule [`ctx.workspace`](workspace.md)
states about `repo_writable`, seen from the delegation side.

`model_provider` is named for what it is: the *subagent* provider is a different
thing on the same handle, and one field called `provider` for both is how
`rehydrate` came to look up the wrong one.

## Presets: a name a deployment is willing to spawn

```text
ctx.subagent_presets.get(name)     # -> SubagentPreset | None
ctx.subagent_presets.names()       # what a profile offers
ctx.subagent_presets.presets()     # the whole table
```

That is the whole of `ctx.subagent_presets` — a small service with no page of its
own, because everything worth saying about it is *why* a preset is shaped this
way, and that belongs beside the delegation it configures.

A preset binds a name to a **capability** — `skills`, `tools` — and nothing else.

**No prompt field, deliberately.** The obvious design gives a preset its own
standing instructions, and then a directing skill and a preset are two channels
saying what a child is for: competing where they disagree, duplicated where they
do not. A skill body already *is* a standing instruction (P4-13b), so `reviewer`
is `skills: [code-review]`, and what a reviewer does is written once, in the
skill, where a human edits it.

`ph-base` ships the row with **no presets**: a preset that widened whatever
selected it would put an escalation one indirection away and under the model's
control. A profile names its own.

## Grants cannot widen

`check_grant` and `grant_for` resolve what a child may have against what its
*parent* holds. A child cannot be granted a capability its parent lacks, and the
ceiling is computed at the boundary the parent actually occupies — `held_by`
walks the parent scope rather than trusting a claim on the request.

The `task` tool refuses to widen rather than silently narrowing, so a parent
asking for more than it has gets an error it can act on.

## A model from the list, and a profile a parent assigns (S7b)

A spawn names a model by **key** (`model="classify"`, `SubagentRequest.model_key`),
and a skill it gives the child may name the one it needs in its front matter
(`model: classify`). `resolve_model` turns the key into a route through the
parent's own `models` list (`ctx.models`, the profile as it runs); a key the list
does not hold is refused, so the models an agent can reach are the ones its
profile says. The child's `subagent/admitted` records the key beside the route it
resolved to.
Two skills naming different models are refused — name one with `model=`.

`profile="reviewer"` assigns the child a named profile, composed by the host
(`ctx.named_profiles`, which `ph_app.runtime.mounted` provides) and read by
`ph/seams/subagent_profiles.py` as a **narrowing on the parent's mount** — a child
never gets a mount of its own:

- a row the profile runs that the parent's mount does not is refused, naming it;
- the tools and skills of each row the parent runs and the profile does not are
  taken away, found by the row that registered them (`ToolRuntime.registrants`,
  `SkillService.registrants`);
- each row it keeps is asked of its own plugin, which declares how a child holds
  less of it (`plugin(..., narrows=)`, `ph.cordis.child_limit`):
  - `models` — the default must be a key the parent lists, and is the child's model;
  - `sandbox-policy` — a `read-only` default makes the child read-only; one wider
    than the parent's posture is refused;
  - `skills-progressive` — the paths must be a subset of the parent's, and only
    skills found under them stay;
  - `sandbox-allow` — the writable directories must be the parent's or inside them,
    and are all the child's sandbox binds (`SandboxSeam.restrict_paths`, recorded
    as the admission's `paths`). Its network must be the parent's exactly: one
    egress proxy serves every agent, so a profile with fewer hosts is refused
    rather than given its parent's.

What a row with no narrower says is the parent's, since it is the parent's mount the
child runs on. A new row joins by declaring one, and nothing in the narrowing names it, as long as what it holds back is one of `ChildLimit`'s kinds; a new kind is a field there and on the child's grant. The narrowing is written into the request before the ceiling, so
`check_grant` checks it and the admission records it: a child's reach is fixed when
it is admitted. What a spawn names beside a profile may narrow it further, and
naming more than it gives is refused.

## Guards: a policy asked before the child exists

`ctx.subagents.guard(check)` registers a deny-only policy the seam asks on every
admission, before the provider is: `check(request)` returns a reason to refuse or
`None`. A refused spawn raises `SubagentSpawnError` with that reason and produces
no session, no log and no artifact. The same shape `ctx.tools.guard` has, and
monotonic for the same reason: a guard can narrow what a deployment allows and
never widen it. The limits row's child caps are the first registrant. How many
children a turn or a session may spawn is a count the seam can *ask* and a policy
row must *decide*, which is why it is a registration here rather than a field.

**A cap asks the seam for the count** (`child_counts(parent)`, a `ChildCounts` of
this turn's children and all of them). It counts a spawn from the moment its guards
pass. The guards run before the provider builds the child, and a child is in
`children` only once its admission is written, so a spawn is on its parent's list
from its last guard until that admission lands or the spawn is refused. Each
hand-off happens with no await, so the count holds the spawn exactly once. Without
it, two spawns from one step (the driver runs a step's tool calls side by side) each
saw the other missing, and both passed a cap that only one of them fit under.

**The seam names the child, at the same moment.** Right after the guards, `start`
gives the child the name the spawn asked for, or one made from its task
(`default_child_name`), and holds it on the same list. The name must be unique among
the parent's children and its spawns on their way: names address children
(`agent_message`, the roster), and a provider that chose them from the admitted
children alone let two spawns from one step take one. A name that is taken, empty or
longer than `MAX_NAME_CHARS` is refused before the provider is asked. The provider
receives the request with its name filled in, and the seam stamps the name on the
run as it stamps the owner.

## Providing one

```python
@runtime_checkable
class SubagentProvider(Protocol):
    async def start(self, request: SubagentRequest) -> SubagentRun: ...
```

**One method.** A `capabilities` set was drafted here — whether children survive
the parent, whether they can be messaged — and removed again: with one provider
there is nothing to branch on, and the vocabulary a second provider needs is not
guessable from the first.

Resuming is a separate Protocol, `RehydratableProvider.rehydrate(run_id) ->
bool`, because not every way of running a child can resume one — and as its own
Protocol rather than a `getattr` probe, since a provider whose method is misnamed
or has the wrong arity would otherwise fail silently as "cannot rehydrate".
Stopping a live child is one too, `RevokingProvider.revoke(run_id, reason) ->
bool`, for the same reason (see [Revoking a child](#revoking-a-child)).

`ctx.subagents.register_provider(name, provider)` — per name, so a deployment may
offer more than one kind of child.

Obligations a provider carries, in order:

1. **Open the child's own log, naming its parent.** The header carries
   `parentSession` and `origin: "subagent"`, the id starts `<parent id>-`, and the
   log is filed in the parent's family. That log is the child's only record, and it
   is how the parent finds the child after a restart (`SessionArchive.descendants_of`),
   so the seam refuses a child whose log does not name its parent.
2. **Validate the kwargs, gate the depth** (`RLM_MAX_DEPTH`), and preflight the
   model **with no fallback**: a child silently downgraded to another model is a
   result nobody can interpret.
3. **Hand back the run, and hold the child at its gate.** The provider doesn't write
   the admission. The seam does, into the child's log, because it holds both the
   resolved request and the run, and it stamps the `owner` a readmission finds the
   provider by. The seam then bounds the child, flushes the admission, and only
   then opens `SubagentRun.ready`, which the drive waits on before its first step.
4. **Detach.** The child runs in its own task; `start` returns.
5. **Record the child's life through the seam's doors, into the child's log:**
   `record_started`, `record_waiting`, `record_ended` and `record_deleted`. Each door
   keeps its record's durability rule, so no provider hand-writes a flush.
   `ph.seams.subagents` is the one writer of these types (`_WRITTEN_BY`), so a
   provider that appends one itself fails `test_log_writers`. Nothing goes into the
   parent's log but what an ending tells it: a `ChildNotice` the provider hands
   `record_ended`, which rides on the ending and is delivered into the parent's own
   inbox. The resume sweep delivers one a crash kept from the parent, and never one
   the parent's log already has.

A child's spend needs no obligation: its answers carry their `usage` in its own
log.

## Records, in the child's own log

Each session owns its log (Phase 11). Every record about a child is in the child's
log, and the parent's log holds none: the parent keeps its own acts, such as the tool
call that spawned the child. All three types are required reading, because a reader
that skipped one would miscount the retry ladder or bring back a deleted child.

| record | carries | door | durable when |
|---|---|---|---|
| `subagent/admitted` | `admission_payload(run, request)` — run id, name, route, both accesses and any downgrade, the prompt, and the narrowing (`preset`, `profile`, `modelKey`, `skills`, `tools`, `paths`, `reasoningEffort`) and `callId` where given — plus `owner`, `parentTurn` and `goalId` | `record_admitted`, written by the seam alone | on the child's disk before its gate opens; a write that fails refuses the spawn (S2) |
| `subagent/status` | `status`, and `cause`, `detail`, `answerPreview`, `notice`, `slots` or `reason` where they apply | `record_waiting` (`queued`), `record_started` (`running`), `record_ended` (`done`, `error`, `canceled`) | a `resumed` start before its attempt (S10); an ending before its parent is handed the result (F1); a wait rides the child's next flush |
| `subagent/deleted` | `reason` | `record_deleted` | in one batch with `canceled` if the child had not ended, and flushed (S14) |

- **The admission** has no `sessionId` or `parentId`: the log's own id is one, and
  its header names the other. `owner` is the provider row that runs the child, so a
  readmission finds that provider rather than whichever one is mounted alone.
  `parentTurn` is the seq of the parent's latest `turn/start`, which the spawn caps
  count a turn by. `goalId` is the parent's open goal, which the child's spend is
  charged to. The admission is written before the ceiling is applied, so a child the
  ceiling refuses reads as admitted and then ended in its own log.
- **A status carries its own reason or none.** `cause` is `resumed` for a restart
  and `rehydrated` for a woken child; `detail` says why a child waits or failed.
  Neither carries over to the next status.
- **`subagent/usage-attributed` is gone.** It was a copy of each answer's usage in
  the parent's log. The answer's own `assistant/message` in the child's log already
  carries it.

Until format 3 these records sat in the parent's log, and every durability rule
above was an order kept *between* two logs: the parent flushed before the child ran
(S2), the parent flushed before a restart's attempt (S10), the child's log written
before the parent's `done` (F1), and the parent's usage copy caught up from the
child's log on resume (L5). Two logs have two write schedules, so each of those was
a way for a crash to leave the two accounts disagreeing. With one account, each is a
rule about one log. Format 3 refuses a format-2 log rather than misreading it.

## Reading a child

`child_state(session)` folds one child's log, and nothing else, into a
`ChildState`. It's frozen, and a pure function of that log:

- the admission, read through properties (`run_id`, `name`, `owner`, `call_id`,
  `goal_id`, `parent_turn`, `model`, the two accesses);
- the latest status with its `cause` and `detail`, each status replacing them whole;
- `deleted` and `deleted_reason`;
- `awaiting`, the credential the child is held for;
- the ladder's counters (below);
- `tokens`, what its answers and compactions spent.

A child with an admission and no status reads `queued`. A log with no admission is
not a child (a workspace record can reach the disk first), and every reader skips
it. `to_wire()` is the row the model's list tool returns; `run()` rebuilds the handle
a spawn returned, for a tool answering after a crash.

A parent's children are a **join over their logs**, not a table and not a fold of
the parent:

```text
ctx.subagents.children(parent_id)                      # {run_id: ChildState}, admission order
await ctx.subagents.load_children(parent_id, family)   # the same, with the stored ones read in
ctx.subagents.state(session_id)                        # one child, live or stored
ctx.subagents.name_of(agent_id)                        # its admission's name, or the id
```

- **`children`** reads each live child through its own cached fold
  (`subagent-fold-cache`, checked per child against a fresh fold, I6), beside the
  stored children already read.
- **`load_children`** reads everything beneath the parent off the store — the whole
  tree, in one read (`SessionArchive.descendants_of`, in the parent's family), each
  state filed under the parent its header names — once per mount, on a worker
  thread, parsing only the records a child's state is folded from
  (`CHILD_EVENT_TYPES`). `open_session` calls it as a resumed session opens, on every
  host; the resume sweep, a spawn, a `task` call's crash check, messaging and a
  delete call it again, which then costs a lookup.
- **`family`** walks the whole tree beneath a parent, each child before its own
  children — the one walk a goal's spend, the daemon's panel and what holds a root
  all read.
- **A child that is let go** keeps its last state as its stored copy.

The prompt, the list tool, messaging, passivation, the spawn caps (the session count
is the parent's children, and the turn count those whose `parentTurn` is its current
turn) and a goal budget with `children` among its `token_sources` all read this join.
The budget reads `delegated_tokens(parent_id, goal_id)`: the children admitted under
that goal, and every level beneath them. A fork is a new root, so it starts with no
children.

`queued` is a child admitted and waiting for a slot. The rlm provider caps how
many of one parent's children run at once (`maxConcurrent`, four in the shipped
`rlm` bundle), and the rest wait in admission order rather than being refused:
the parent asked for them, and a refusal answers a question about resources with
one about intent. A queued child is live (`child_is_live`), so a parent is not
passivated while it waits, and a slot is freed on `done`, `error` or `canceled`
alike. Deleting a queued child cancels its wait and takes no slot.

**The queue itself is [`ctx.jobs`](jobs.md)**, not this provider's — `slot=` on
the drive job, keyed by the parent's session. A provider chooses *whether* to cap
and what to key it by; the waiting, the release on every ending and the queue's
own lifetime belong to the work seam, so a second provider gets them without
re-deriving forty lines of async.

That per-parent number is a **fair share**, not a bound on the host: ten roots at
four apiece is forty children. What the machine can carry is the work seam's
`concurrency` config. The `rlm` bundle sets `subagent: 8` beside its per-parent
number, so the two figures an operator compares sit in one file; `ph-core` ships
no default naming a kind it does not produce. `phern daemon
--max-concurrent-children` overrides it, and both apply, the parent's first.

## Across a restart

A harness that stopped between a child's admission and its end left that work
described in the child's own log and running nowhere. `resume_children(parent, *,
retry_limit)` is what the next one calls — the daemon does, at the point it resumes
a root. It reads the parent's children first (`load_children`), and gives the two
states opposite answers:

| the log says | what happens | why |
|---|---|---|
| `queued` | **re-driven**, under its own run id | it claimed nothing and spent nothing; this is the work happening once |
| `running` | put back on the **ladder**, its task presented again | its turn was cut short; three attempts, then failed |
| `running`, nothing mounted that can resume it | **settled** with its own reason | a child left `queued` for a provider that never comes holds the root out of passivation for good |

Which of those a child gets is decided where the capability probe answers, and
written once. Marking it `queued` and discovering afterwards that nothing can
readmit it is the state this sweep exists to end, recreated.

Leaving the second alone is not neutral: `child_is_live` counts an unsettled
child, so a child nothing will ever move keeps its whole root out of passivation
for the life of the process while the parent waits on a reply nobody is writing.

**Each decision is written in the log it is about.** A child the sweep gives up on
is ended `error` in its own log: opened from the store without being resumed
(`stored_session`), written, flushed and let go. A child it puts back is readmitted
by the provider its admission's `owner` names, and the child's drive writes
`running` with `cause: resumed` on its own disk before the attempt. Nothing orders
one child's records against another's or against the parent's, so a crash
mid-sweep leaves each child decided on its own disk or undecided, and the next start
decides the undecided ones the same way.

**A child waiting for a credential is held, not readmitted** (T5). If its route
names a credential this deployment cannot supply, readmitting it would spend its
ladder on a key a person could supply in a second. So its own log records the hold
(`CREDENTIAL_WAIT` with `SESSION_HOLDER`, what a root waiting on its own route
writes). It stays live, and no start is counted. `readmit_waiting` asks again when
a credential arrives.

**Re-presenting the task is what makes a retry real.** Starting a turn *claims*
the task from the child's inbox, and the claim is a logged splice — so a resumed
child that was merely driven again finds an empty inbox, ends at step zero and
reports `completed` for work it never did. The task goes back in, saying the
turn was cut short, so the transcript reads as one interrupted attempt and not
as one instruction given twice.

**The ladder's whole state is folded, not carried.** The child's own log already
records one `subagent/status running` per drive and one `assistant/message` per
model answer, so `ChildState` derives its counters from facts that are already
there rather than maintaining a number that could disagree with them:

| on `ChildState` | what it counts | cleared by |
|---|---|---|
| `starts` | every time this child has been driven | nothing |
| `resumes` | the drives that were restarts (`cause: resumed`) | nothing |
| `resumes_at_last_answer` | `resumes` as of the child's latest model answer | — |

`restarts_since_progress(state)`, the difference of the last two, is the ladder's
count: restarts that achieved nothing since the child last answered. After
`retry_limit` of them the child is failed and the reason says which bound it hit.

**No catch-up step.** The answers the ladder counts are in the child's own log,
where the child wrote them, so the sweep reads them there. When the parent's log
held a copy of each answer's usage (`subagent/usage-attributed`), the copy waited
in the parent's memory while the child's log reached disk before every request. A
crash could drop the copy and keep the answer, and the ladder then read a child that
had answered as stuck. The sweep had to copy the difference back before deciding
(L5, `reconcile_answers`). Phase 11 removed the copy, and the step went with it.

**Every level** (L5b). A readmitted child may have children of its own, so each
readmitted child is swept the same way, with the same `retry_limit`, before it takes
its first step: `SubagentRun.ready` is the gate its drive waits on, opened once the
child is bounded and its children swept. `readmit_waiting` descends the same way when a credential
arrives. There are no delays, unlike the root's ladder, because this one only ever
runs while a harness is starting — which is already the wait.

**A child that ends takes what it left unfinished beneath it.** Its running children
are artifacts of its scope, revoked (`PARENT_TEARDOWN`) as it unwinds. That never
reaches a descendant this process is not running: one held for a credential, one
beneath a child a restart readmitted, one a delete reached on disk. So the two doors
that end a child, `record_ended` and `record_deleted`, tombstone every such
descendant in its own log, every level in one walk (`_revoke_beneath`), whichever
provider or seam path called them. Left alone, such a descendant read as working for
good: it held its root out of passivation, and a credential it waited for would have
readmitted it under a parent that had ended. A crash between a child's ending and its
children's leaves the rest to the resume sweep, which does the same for a child that
had ended. Nothing left live beneath a root is then an orphan, and passivation counts
every level alike.

**The bound is the host's, and `resume_children` takes it with no default.** This
seam owns the sweep, the fold and the records; how many attempts work is worth is
policy, and a seam that answered it for a caller who said nothing would be
choosing one (P6-32's rule). The daemon states it beside the root's own ladder,
in `ph_app.daemon.recovery`, which is where somebody tuning restart behavior is
already looking.

**Progress clears the count**, so the ladder bounds *consecutive* interruptions
rather than a lifetime's: a child stopped once, working for an hour, then stopped
again met two separate incidents, and reading that as one child running out of
attempts fails work that was going fine. Progress is a **model answer** and never
a turn ending — P5-04's forged-marker problem one level down, since a retry whose
task somehow never reached the inbox ends a turn having done nothing, and counting
turns would let exactly the broken case clear the count that bounds it.

**The two counters are separate because they part company.** Which attempt a
restart is comes from `starts`, never from the ladder's count: progress clears it,
so a child that got somewhere and was stopped again would otherwise be readmitted
as though it had never run, its restart go unrecorded as one, and the ladder never
count it again.

**The ceiling is re-derived, not restored from memory.** A readmit goes through
the same `check_grant` and `grant_for` a fresh admission does, against a request
rebuilt from the child's admission — which is why that record carries the
child's `preset`, `skills` and `tools` (`admission_payload`). Without them a
child would come back holding its parent's whole reach: §6.5 broken by a power
cut. It is checked against what the deployment holds **now**, so a child admitted
with a skill since removed is refused rather than quietly readmitted without it.

The spawn **guards** do not run again, deliberately: a guard answers "may this
delegation happen", and this one already did — its admission is in its log, so
asking again would count the child against a cap its own record fills and refuse
to restore work that was once allowed. Guards gate new work.

**Open question: which model and reasoning effort a readmitted child runs on.** It
is decided two different ways today:

* **The route is pinned at admission.** The admission records the provider and
  model the spawn resolved to, and `resolve_model` leaves a request that already
  carries a route as it is. So a readmitted child runs on the model it was
  admitted with, even if the skill that chose it (`model:` in its front matter)
  or the parent's `models` list has since moved to another.
* **The reasoning effort is pinned only when something named it.** The admission
  records the effort the spawn asked for, or the one the route a model key
  resolved to carries (`Admission.reasoning_effort`). An effort nothing named was
  the parent's, and a readmitted child takes its parent's effort as it is at the
  restart (`RlmChildProvider._resolve_model`), not the one it first ran on.

The direction to decide: whether what a child runs on is the **parent's and the
skill's to say at each start**, rather than something the child's record fixes. A
skill updated to fix or improve a delegation — a better model, another reasoning
level — would then reach a child readmitted after it, as it reaches a child
spawned after it. The admission would keep recording what was *asked* (the model
key, the skills, a named effort), and a readmit would resolve the route and effort
again from the parent and the skills as they are then. What it would cost: a task
begun on one model could be finished on another, which the re-presented task
already tells the child is a new attempt (`restarts`). Until this is decided, a
readmitted child keeps its admitted route, and an effort it left unnamed follows
its parent.

A provider opts in by implementing `ReadmittingProvider.readmit(request, *,
run_id, session_id, restarts)` — its own Protocol, like `RehydratableProvider`, because
resuming an un-run child is not something every way of running one can do.

## Asking a settled child something else

Addressing a child that has finished rehydrates it (`RehydratableProvider`), and
it comes back **to its own work**. That falls out of the `worktree` tier rather
than being arranged: disposal commits the checkout to the child's branch before
removing it, and a re-acquire attaches that branch rather than resetting it — so
the same run id resolves to the same branch and the second question starts where
the first stopped.

The workspace is re-acquired with the access the *admission* recorded, so nothing
about being asked again can widen what the first question was allowed. Before
this, rehydration rebuilt the agent and its grant but took no workspace at all,
which left a re-addressed child writing into its parent's tree.

## Revoking a child

```text
await ctx.subagents.delete(parent_session, run_id, reason="…")   # -> bool
```

The one revocation door, for **any** child of the parent, live or not. A child this
process runs is its provider's to stop (`RevokingProvider.revoke`): its job, its
agent. The provider writes the tombstone into the child's log as it lets go. A child
nothing here runs, because an earlier process settled it, is tombstoned by the seam
in its stored log. Before Phase 11 a revocation reached only children a provider
held in memory, so one an earlier process had settled answered `False`. Now `False`
means there is no such child, or it was deleted already.

`record_deleted` writes the `canceled` that ends an unfinished child and the
`subagent/deleted` tombstone in **one batch** (S14). Written apart, a flush between
them left a child canceled and not deleted. Whether the child had already ended is
read from its own log, not from a caller's flag, and a child that ended is not
ended again: a `canceled` over its `done` turned a finished child into a revoked
one. A child tombstoned already is left as it is, so a teardown and a cascade that
both reach one child write one tombstone. The transcript stays on disk, because a
parent looking for what a revoked child did should find the revocation, not a gap.
A deleted child is never rehydrated.

A mount going away is not a revocation. The provider suspends each unfinished
child instead: `queued` with `SUSPENDED_DETAIL` in its own log, no tombstone, and
the resume sweep readmits it.

## What it does not do

* It does not run the child. That is the provider's, and `ph-base` has none.
* It does not deliver replies. Those arrive as inbox messages through the
  messaging row; addressing a settled child fails with the `agent_observe` route
  named.
* It does not bound spend or fan-out by itself — `ctx.goals` and the limits row
  do, the latter through `guard`.
* It does not retry a child *turn* that failed on its own — that is the model's
  outcome and the transcript's. The ladder above counts interruptions, meaning a
  harness that stopped, which is a different thing entirely.

## See also

[`ctx.workspace`](workspace.md) · [Adding a seam](../cookbook/adding-a-seam.md) ·
`test_child_logs.py`, `test_subagent_status.py`, `test_subagent_task.py`,
`test_subagent_grant.py`, `test_subagents.py` (ph-rlm)
