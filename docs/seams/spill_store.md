# `ctx.spill_store` — oversized content out of context, with a way back

**Module:** `ph/seams/spill.py` · **Row:** `spill-local` · **Consumers:**
`tool-result-offload`, `input-offload` (both in `ph-stabilize`)

An offloaded tool result is not deleted, it is **relocated**: the model gets a
preview and a locator, and the locator resolves to the full text.

That is what makes G2/G3 offloading an optimization rather than a lie — **the
harness never tells the model something is gone when it is on disk.**

## The surface

```text
await ctx.spill_store.reserve_text(text, ...)      # -> SpillRef, staged
await ctx.spill_store.try_reserve_text(text, ...)  # -> SpillRef | None
await ctx.spill_store.commit(ref)                  # publish, after the append
await ctx.spill_store.save_text(text, ...)         # -> SpillRef, unordered
await ctx.spill_store.load_text(ref)               # -> str
ctx.spill_store.locator_for(...)                   # the name, before the write
ctx.spill_store.claim(...)
await ctx.spill_store.sweep_session(session_id)
```

A `SpillRef` is three fields, and the third is the interesting one:

| field | |
|---|---|
| `locator` | where it went |
| `bytes` | how much there was |
| `retrieval_hint` | **how to get the rest, in the model's own vocabulary** |

`retrieval_hint` exists so the preview can say *"`read` this path, offset N"*
rather than making the model guess. A locator with no hint is a reference the
model has to reverse-engineer, and it will reverse-engineer it wrongly.

`try_reserve_text` is the non-raising form: a spill that fails is an
optimization that did not happen, and the caller keeps the content inline rather
than losing the turn. It is the *write*, so that fallback is still on the table
when it answers `None` — nothing has been logged yet.

## Reserve, append, commit

A blob is garbage exactly when the log does not name it. That is what makes the
sweep below safe, and it is a promise about **ordering** that only producers can
keep: one that writes the file first and appends the locator second leaves its
own blob indistinguishable from garbage for as long as that takes, and the sweep
runs on another task. It collected a live history file often enough to fail a
test under load.

So a producer that records a locator writes in two steps:

```python
ref = await ctx.spill_store.try_reserve_text(
    owner=session.id, source="tool result", suggested_name=name, content=text
)
if ref is None:
    return None  # fail open; nothing has been logged
session.append("offload/spilled", {"callId": call_id, "locator": ref.locator})
await ctx.spill_store.commit(ref)  # the blob appears, already named
```

`reserve` stages the bytes where the sweep does not look; `commit` is a rename
within the owner's directory, so the blob appears whole, at a locator the log
already names, or not at all. `save_text` remains for a caller with no log entry
to keep in step — a test planting a blob, or a producer that appends nothing.

`locator_for` is the derivation underneath both, public for the same reason:
a caller that must record a reference before writing needs the name first.

## Why this is not `ctx.attachments`

Identical mechanics — content-addressed, digest-named files — and the lifecycles
differ in the one way that matters.

| | spill | attachment |
|---|---|---|
| what it is | a **forwarding address** | **content** the log points at |
| losing it costs | a reader's way back to the original | a piece of the conversation nothing can rebuild |
| collection | per-owner sweep at session open (F7) | a fold over every session, manual only |
| lives in | the cache | `$PH_HOME` |

Someone will eventually clear a cache directory to reclaim space, and that must
not be able to delete conversation. So [`ctx.attachments`](attachments.md) is its
own seam and does **not** participate in this sweep.

## Collection

`sweep_session(session_id)` runs at session open — the per-owner sweep F7
describes. It is safe to be automatic *because* a spill has an owner: the session
that produced it. That is exactly the property an attachment lacks, which is why
its collection is a command a person runs.

## The row

```yaml
- id: spill
  name: spill-local
  config:
    root: /var/cache/ph/spill    # optional
```

## See also

[`ctx.compaction`](compaction.md) · [`ctx.attachments`](attachments.md) ·
[`ctx.token_meter`](token_meter.md) · `test_spill_sweep.py`,
`test_offload.py` (ph-stabilize)
