# `ctx.attachments` — media the log points at but cannot reconstruct

**Module:** `ph/seams/attachments.py` · **Row:** `attachments-local` ·
**Consumers:** `media-degrade`, every adapter, `tool-attach`, `--attach`, the
browser drop zone

The log is lossless JSON (A1), so an image never enters it. A `MediaBlock`
carries an `AttachmentRef` and the bytes live here, named by their own SHA-256.

## Why this is not `ctx.spill_store`

The mechanics are identical — content-addressed, digest-named files — and the
lifecycles differ in the one way that matters.

A spilled tool result is a **forwarding address**: the log already holds the
preview the model saw, and losing the file costs a reader the way back to the
original. An attachment is content the log only **points at** — lose it and the
conversation is missing a piece nothing can rebuild.

Someone will eventually clear a cache directory to reclaim space, and that must
not be able to delete conversation. So: its own seam, its own directory under
`$PH_HOME`, and no participation in the spill sweep.

## Addressed globally, with no owner directory

A digest is a digest. Two sessions attaching the same photo share one file, and a
fork references exactly the digests its parent does rather than a directory the
parent owns — which is what stops deleting a parent session from breaking its
children.

The cost is that collection needs a fold over *every* session, so nothing here
collects automatically. That is `ph attachments gc`, deliberately unlike the
spill store's per-owner sweep, and the rule it obeys was written down before it
existed: **a blob any stored log still references must not be collected, however
old it is.**

## Reading a path is the caller's business (I-9)

The store takes **bytes**. Who is allowed to turn a path into bytes is a security
question with two different answers, and a store that read paths itself would
answer it once, wrongly, for both:

| door | how | bound by |
|---|---|---|
| a **person** | `save_path(path)` | the harness's own permissions — they may attach anything they can already open |
| a **model** | `ctx.fs.read_bytes(...)` then `save_bytes(...)` | `permissions-fs`, the workspace tier, every `fs/read-intent` listener |

`save_path` exists for the human door and says so. A tool that reached it would
be an exfiltration primitive with a friendly name — attach a private key as a
"document" and let a provider read it out. `tool-attach` is the model door and is
the only caller this seam expects.

## The surface

```text
await ctx.attachments.save_bytes(content=..., mime=..., name=..., width=..., ...)  # -> AttachmentRef
await ctx.attachments.save_path(path, mime=None)      # the human door only
await ctx.attachments.load_bytes(ref)                 # -> bytes
await ctx.attachments.load_b64(ref)                   # -> str, encoded once per process
ctx.attachments.exists(ref)                           # bytes still there?
ctx.attachments.path_for(ref)                         # pure: the digest *is* the name
```

`path_for` is a pure function of the ref — the digest is the name and the suffix
comes from the ref's own MIME — so a reference read back out of a stored log
resolves to the same path on a machine that has never seen the session.

`load_b64` caches, bounded at `ENCODED_CACHE_BYTES`, and **the cache cannot go
stale**: `attachment_id` is the SHA-256 of the content, so two refs with one id
have one body by definition. It is worth having because a media block stays in
derived history for the life of the session — without it, an image attached at
the first turn is re-read and re-encoded for every request after it.

`exists()` is worth asking before building a request: a reference whose blob is
gone must degrade to a pointer the model can read, not a wire error it cannot act
on.

## Dimensions are measured here

`save_bytes` fills in `width`/`height` for an image when the caller did not,
using `ph.llm.dimensions` — `int.from_bytes` and a format dispatch, no Pillow, no
ImageMagick. A supplied argument still wins: an ingester that already decoded the
image knows at least as much as its header does.

**Not enforced** (§5 rule 6): this fills in an attachment as it is *stored*, so a
reference in a log written before that landed keeps `width: None` for ever, and
neither pixel ceiling can fire for it. Backfilling would mean rewriting stored
references, which A1 does not allow.

## What decides whether the model actually sees it

Not this seam. `media-degrade` (`ph/llm/media.py`) sits above every adapter and
answers whether *this route* can take *this block* — MIME, byte ceiling, pixel
edge, and whether the bytes are still on disk. A block it refuses becomes a text
pointer with a logged `attachment/degraded` notice, never a silent drop and never
a hard failure: a session begun on a vision model and resumed on a text one must
still open.

## Collection

`ph attachments gc` folds **every** stored session for referenced digests. Three
rules make it safe:

* references are matched by the **digest's shape**, not a list of key names —
  three rows already spell one three ways, and a key list maintained elsewhere is
  a blob collected out from under a session;
* the listing is every log, and the cap is checked — a bounded answer to "does
  anyone still need this" is not a smaller answer but a wrong one;
* a log that will not parse, or a truncated listing, collects **nothing**.

`--min-age` only ever *refuses*; age never authorises collection. It covers the
window where a person has dropped a file on the composer and nothing references
it until they send the prompt.

## The row

```yaml
- id: attachments
  name: attachments-local
  config:
    root: /var/lib/ph/attachments   # optional; else $PH_HOME/attachments
```

## See also

[`ctx.uploads`](uploads.md) — the provider-side copy · [Adding a
tool](../cookbook/adding-a-tool.md) · `test_attachments.py`,
`test_attachments_gc.py`, `test_attach_tool.py`, `test_media_degrade.py`
