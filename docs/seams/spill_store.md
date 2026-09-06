# `ctx.spill_store` — oversized content out of context, with a way back

**Module:** `ph/seams/spill.py` · **Row:** `spill-local` · **Consumers:**
`tool-result-offload`, `input-offload` (both in `ph-stabilize`)

An offloaded tool result is not deleted, it is **relocated**: the model gets a
preview and a locator, and the locator resolves to the full text.

That is what makes G2/G3 offloading an optimisation rather than a lie — **the
harness never tells the model something is gone when it is on disk.**

## The surface

```text
await ctx.spill_store.save_text(text, ...)      # -> SpillRef
await ctx.spill_store.try_save_text(text, ...)  # -> SpillRef | None
await ctx.spill_store.load_text(ref)            # -> str
ctx.spill_store.locator_for(...)                # the name, before the write
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

`try_save_text` is the non-raising form: a spill that fails is an optimisation
that did not happen, and the caller keeps the content inline rather than losing
the turn.

## `locator_for` before the write

The name is derived before the bytes are written, which is what lets a caller
record the reference and the content in one consistent step — the same
"log first, act second" shape the tool pipeline uses for `tool/call`.

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
