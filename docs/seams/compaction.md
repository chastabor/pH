# `ctx.compaction` — replacing history with a summary, without losing it

**Module:** `ph/seams/compaction.py` · **Row:** `compaction` (definition only) ·
**Provider:** `compaction-summarize` (`ph-stabilize`) · **Consumers:** `/compact`,
whatever policy row decides a session is under pressure

**Definition only in `ph-base`.** *When* a conversation is too long and *what* a
summary should say are policy, and a deployment that wants the plain harness
should get a session that grows until the provider says no.

## Compaction is a surface `replace`, and that is the whole safety story (A3)

The engine appends **one message** whose `surfaceOp` shadows the range it stands
for:

* `derive_messages()` yields the summary — that is what the model sees;
* the log keeps **every shadowed event** — nothing is deleted;
* `transcript()` still shows the person the conversation they actually had.

So a summary that turns out to have dropped something important is a bad
*reading* of the log rather than a hole in it (I4). This is why compaction is
allowed to be lossy at all: the loss is in the derivation, never in the record.

## Notes are the part that is pH's own

A summary replaces **conversation**. pH has state that is not conversation — an
RLM session's kernel namespace survives the cut completely untouched, because
compaction rewrites a surface and a REPL is not on it.

A summary that did not say so would leave the model believing its variables went
wherever the conversation went. So a plugin owning such state registers a note:

```text
ctx.compaction.note(CompactionNote(name=..., text=..., order=...))
ctx.compaction.notes()          # what the engine splices into the summary prompt
```

The engine puts them in the summary prompt (G10). **The state itself is never
touched from here** — a note is a statement about what survived, not an
instruction to preserve anything.

## The surface

```text
ctx.compaction.register(engine)          -> Disposer
ctx.compaction.require()                 # the engine, or a refusal
await ctx.compaction.compact_now(...)    # /compact
await ctx.compaction.compact_if_needed(...)  # the pressure path
```

## Providing an engine

```python
@runtime_checkable
class CompactionEngine(Protocol):
    async def compact_if_needed(
        self, agent: Any, trigger: CompactionTrigger
    ) -> CompactionResult | None: ...

    async def compact_now(
        self, agent: Any, *, instructions: str = ""
    ) -> CompactionResult | None: ...
```

`None` from either means *there was nothing worth compacting* — not a failure.

`instructions` is the person's own account of what they are about to work on. An
engine may weight the summary towards it; one that cannot is free to ignore it,
which is why it has a default rather than being a second method.

Both methods, because the two callers are genuinely different: `/compact` is a
person saying *do it now*, and the pressure path is a policy row asking *is it
time* — carrying the `CompactionTrigger` that says which pressure. An engine
implementing only the second would leave the command with nothing to call.

The loop knows nothing about any of this — compaction attaches to
`agent/pre-step` and `agent/request-error` like every other stabilization feature
(D12).

## Two things an engine must get right

* **Append the summary and the replace atomically with respect to durability.**
  The pair is what makes the surface consistent; an interleave leaves a log whose
  derivation depends on when it was read.
* **Record the attempts that fail.** `compaction/declined` exists because the
  manual path once recorded *nothing* while telling the person "the attempt is
  recorded" — the failed attempt has to be visible in the log, which is one of
  the benefits a surface replace is supposed to buy.

## What it does not do

* It does not decide when. That is a policy row reading
  [`ctx.token_meter`](token_meter.md) pressure.
* It does not touch plugin state. See notes, above.
* It does not summarise *as an agent*. The summarizer is one model call today;
  making it a full agent turn — able to chunk a long range or read the spilled
  history file — is its own row, deferred because *whose loop*, *whose log* and
  *whose namespace* all have to be settled first.

## See also

[`ctx.token_meter`](token_meter.md) · [`ctx.spill_store`](spill_store.md) ·
`test_compaction.py` (ph-stabilize)
