# `ctx.spill_store` — oversized content out of context, with a way back

**Module:** `ph/seams/spill.py` · **Row:** `spill-local` · **Consumers:**
`tool-result-offload`, `input-offload` (both in `ph-stabilize`), compaction
(`ph/seams/compaction.py`), and the kernel snapshot writer (`ph-rlm`)

An offloaded tool result is not deleted, it is **relocated**: the model gets a
preview and a locator, and the locator resolves to the full text.

That is what makes G2/G3 offloading an optimization rather than a lie — **the
harness never tells the model something is gone when it is on disk.**

## The surface

```text
await ctx.spill_store.try_save_text(text, ...)  # -> SpillRef | None, before the append
await ctx.spill_store.try_save(planned, ...)    # a blob already planned
await ctx.spill_store.try_save_all(blobs)       # several, one directory sync
await ctx.spill_store.load_text(ref)            # -> str
ctx.spill_store.plan(...)                       # the name and the bytes, together
ctx.spill_store.claim(...)
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

Every write is non-raising: a spill that fails is an optimization that did not
happen, and the caller keeps the content inline rather than losing the turn. It is
the *write*, so that fallback is still on the table when it answers `None` — nothing
has been logged yet. There is no raising form: every producer wanted the fallback,
and the one writer underneath is `try_save_all`.

## Write, then append

A blob is garbage exactly when the log does not name it, which is what makes the
sweep below safe. So a producer that records a locator writes the blob first and
appends the record naming it second:

```python
ref = await ctx.spill_store.try_save_text(
    owner=session.id, source="tool result", suggested_name=name, content=text
)
if ref is None:
    return None  # fail open; nothing has been logged
_LOG.append(session, "offload/spilled", {"callId": call_id, "locator": ref.locator})
```

The blob is durable before any record names it, so the log never names bytes that
are not there, and a write that fails does so before anything is logged. A run that
dies between the two leaves a file nothing names, which the next read collects.

It used to take three steps — reserve, append, commit; `SpillStore.try_save` says
why it no longer does.

`plan` derives the name a blob will have, for a caller that must put it in the
wording or the record that points at it before the write.

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

The sweep runs when a stored session is read, awaited before the session is handed
out (`session/loaded`) — the per-owner sweep F7 describes. It is
safe to be automatic *because* a spill has an owner: the session that produced it.
That is exactly the property an attachment lacks, which is why its collection is a
command a person runs.

It collects a dead run's leftovers along with the rest — a blob whose record never
landed, a `write_atomic` temp a kill interrupted — and reports a record naming a blob
that is not there. Collecting rests on nothing writing the session while it runs,
which is why it has one caller and a test that says so.

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
