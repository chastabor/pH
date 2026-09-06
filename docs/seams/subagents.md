# `ctx.subagents` — delegation to a child agent, and the handle it returns

**Module:** `ph/seams/subagents.py` · **Rows:** `subagents`, `subagent-presets` ·
**Also documents:** `ctx.subagent_presets` ·
**Provider:** `rlm-child` (ships with the `rlm` bundle) · **Consumers:**
`subagent-task`, `rlm-*`

**Definition only in `ph-base`.** The seam mounts everywhere and registers no
provider, because "run a child agent" has genuinely different answers in
different deployments. `subagent-task` therefore registers *nothing* until a
provider is mounted — a tool advertised in every prompt and refused on every call
has taught the model a capability the deployment does not have.

## The handle returns before the child answers

`start()` resolves once the child is **admitted** — session created, admission
logged, task detached — not once it has finished.

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
| `granted_access` | what the available tier could actually honour |

`granted` is what the roster and the child's own prompt report, because a child
told nothing about its workspace attempts writes and reads the failures as bugs.
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

## Guards: a policy asked before the child exists

`ctx.subagents.guard(check)` registers a deny-only policy the seam asks on every
admission, before the provider is: `check(request)` returns a reason to refuse or
`None`. A refused spawn raises `SubagentSpawnError` with that reason and produces
no session, no log and no artifact. The same shape `ctx.tools.guard` has, and
monotonic for the same reason: a guard can narrow what a deployment allows and
never widen it. The limits row's child caps are the first registrant. How many
children a turn or a session may spawn is a count the seam can *ask* and a policy
row must *decide*, which is why it is a registration here rather than a field.

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

`ctx.subagents.register_provider(name, provider)` — per name, so a deployment may
offer more than one kind of child.

Obligations a provider carries, in order:

1. **Log admission first.** `subagent/admitted` is appended *before* the handle
   returns — the seam owns the neutral event names, and `ph-app` reads them
   without depending on `ph-rlm`.
2. **Validate the kwargs, gate the depth** (`RLM_MAX_DEPTH`), and preflight the
   model **with no fallback**: a child silently downgraded to another model is a
   result nobody can interpret.
3. **Detach.** The child runs in its own task; `start` returns.
4. **Attribute usage** — `subagent/usage-attributed`, so a parent's budget sees
   what its children spent.
5. **Report terminal states**, and emit `subagent/status` as it moves.

## Events

| event | |
|---|---|
| `subagent/admitted` | a child was created — logged *before* the handle returns |
| `subagent/status` | it moved |
| `subagent/usage-attributed` | what it spent, charged to the parent |
| `subagent/deleted` | a tombstone; the roster folds these |

The roster is a **fold over these events**, not a table — `roster(session)` — so
passivation and rehydration need no second source of truth, and a crash leaves a
roster that still reconstructs.

`queued` is a child admitted and waiting for a slot. The rlm provider caps how
many of one parent's children run at once (`maxConcurrent`, four in the shipped
`rlm` bundle), and the rest wait in admission order rather than being refused:
the parent asked for them, and a refusal answers a question about resources with
one about intent. A queued child is live to the roster, so a parent is not
passivated while it waits, and a slot is freed on `done`, `error` or `cancelled`
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
no default naming a kind it does not produce. `ph daemon
--max-concurrent-children` overrides it, and both apply, the parent's first.

## Across a restart

A harness that stopped between a child's admission and its first turn left that
work described in the parent's log and running nowhere. `resume_children(parent)`
is what the next one calls — the daemon does, at the point it resumes a root —
and it gives the two states opposite answers:

| the log says | what happens | why |
|---|---|---|
| `queued` | **re-driven**, under its own run id | it claimed nothing and spent nothing; this is the work happening once |
| `running` | put back on the **ladder**, its task presented again | its turn was cut short; three attempts, then failed |
| `running`, nothing mounted that can resume it | **settled** with its own reason | a row left `queued` for a provider that never comes holds the root out of passivation for good |

Which of those a child gets is decided where the capability probe answers, and
written once. Marking it `queued` and discovering afterwards that nothing can
readmit it is the state this sweep exists to end, recreated.

Leaving the second alone is not neutral: `child_is_live` counts an unsettled
child, so a row nothing will ever move keeps its whole root out of passivation
for the life of the process while the parent waits on a reply nobody is writing.

**Re-presenting the task is what makes a retry real.** Starting a turn *claims*
the task from the child's inbox, and the claim is a logged splice — so a resumed
child that was merely driven again finds an empty inbox, ends at step zero and
reports `completed` for work it never did. The task goes back in, saying the
turn was cut short, so the transcript reads as one interrupted attempt and not
as one instruction given twice.

**The ladder's whole state is folded, not carried.** The parent's log already
records one `subagent/status running` per drive and one
`subagent/usage-attributed` per model answer, so the roster derives two counters
from facts that are already there rather than maintaining a number that could
disagree with them:

| on the row | what it counts | cleared by |
|---|---|---|
| `starts` | every time this child has been driven | nothing |
| `attempts` | restarts that achieved nothing since | a model answer |

After `CHILD_RETRY_LIMIT` attempts the child is failed and the reason says which
bound it hit. There are no delays, unlike the root's ladder, because this one only
ever runs while a harness is starting — which is already the wait.

**Progress clears `attempts`**, so the ladder bounds *consecutive* interruptions
rather than a lifetime's: a child stopped once, working for an hour, then stopped
again met two separate incidents, and reading that as one child running out of
attempts fails work that was going fine. Progress is a **model answer** and never
a turn ending — P5-04's forged-marker problem one level down, since a retry whose
task somehow never reached the inbox ends a turn having done nothing, and counting
turns would let exactly the broken case clear the count that bounds it.

**The two counters are separate because they part company.** Which attempt a
restart is comes from `starts`, never from `attempts`: progress clears the ladder,
so a child that got somewhere and was stopped again would otherwise be readmitted
as though it had never run, its restart go unrecorded as one, and the ladder never
count it again.

**The ceiling is re-derived, not restored from memory.** A readmit goes through
the same `check_grant` and `grant_for` a fresh admission does, against a request
rebuilt from the admission record — which is why that record carries the
child's `preset`, `skills` and `tools` (`admission_payload`). Without them a
child would come back holding its parent's whole reach: §6.5 broken by a power
cut. It is checked against what the deployment holds **now**, so a child admitted
with a skill since removed is refused rather than quietly readmitted without it.

The spawn **guards** do not run again, deliberately: a guard answers "may this
delegation happen", and this one already did — its admission is in the log, so
asking again would count the child against a cap its own record fills and refuse
to restore work that was once allowed. Guards gate new work.

A provider opts in by implementing `ReadmittingProvider.readmit(request, *,
run_id, session_id, restarts)` — its own Protocol, like `RehydratableProvider`, because
resuming an un-run child is not something every way of running one can do.

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
`test_subagent_task.py`, `test_subagent_grant.py`, `test_subagents.py` (ph-rlm)
