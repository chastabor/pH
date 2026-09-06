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

## What it does not do

* It does not run the child. That is the provider's, and `ph-base` has none.
* It does not deliver replies. Those arrive as inbox messages through the
  messaging row; addressing a settled child fails with the `agent_observe` route
  named.
* It does not bound spend by itself — `ctx.goals` and the limits row do.

## See also

[`ctx.workspace`](workspace.md) · [Adding a seam](../cookbook/adding-a-seam.md) ·
`test_subagent_task.py`, `test_subagent_grant.py`, `test_subagents.py` (ph-rlm)
