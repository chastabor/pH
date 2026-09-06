# `ctx.uploads` — a provider's copy of an attachment, and the handle for it

**Module:** `ph/seams/uploads.py` · **Row:** `uploads-local` · **Providers:** each
adapter row that declares `uploads` · **Consumers:** `_media.load_handles`

Some things cannot be sent inline. A provider's file API takes the bytes once and
gives back an id to reference on every later request — which video effectively
requires everywhere, and which large documents are worth on any route that offers
it: a 4 MB PDF re-encoded as base64 on every step of a fifty-step session is the
same 5.5 MB uploaded fifty times.

## Where the state lives, and why it is not the log

This is the one piece of genuinely non-derivable state in the media phase, so
departing from "state lives in the log" needs the argument written down:

* **The handle is not in the log; the upload is.** A handle is a *prediction* —
  "this id will work until roughly T" — and an append-only log (A1) that recorded
  one would be asserting something false the moment a provider expired it early,
  with no way to take it back. That bytes left this machine for a named provider
  is a **fact**, is privacy-relevant, and is exactly what an audit wants — so
  `attachment/uploaded` is appended and the handle is not.
* **Keyed on the digest, not the session.** Two sessions attaching one photo
  share one blob already; making them share one upload follows. A per-session
  record would make one session's log the authority for another's uploads.
* **Losing it costs an upload, not information.** Every entry is reconstructible
  from bytes the store still holds — the definition of a cache — so it lives
  under `$PH_CACHE`, where deleting the directory costs one slow turn.

**One file per entry, not an index.** Several daemon roots and several processes
share a cache; a single index is a contended write whose corrupt read loses every
handle at once.

## Expiry is checked twice, and the second check matters

An entry past its own `expires_at` is re-uploaded before a request is built. But
providers expire files early, delete them from another session, or forget them —
and that surfaces **mid-request**.

The adapter then invalidates the handle and raises `FILE_EXPIRED`, which is in
`TRANSIENT_CODES` for the strictest possible reason: **the state that caused the
failure is already gone**, so the retry rebuilds rather than repeating.

Classification takes **both halves** — the provider said a file is missing *and*
it named one this request sent. A `not_found` from a gateway in front of the API
is somebody else's 404, and retrying it would be the "unknown failure billed
twice" the retry policy exists to refuse. `forget_named_handle` in
`ph_app/adapters/_media.py` is that logic, shared by all three wires, and it
forgets **the named handle, not every handle**: a twenty-file request must not
throw away nineteen live uploads to replace one dead one.

## The surface

```text
ctx.uploads.register_uploader(provider, uploader)          -> Disposer
await ctx.uploads.handle_for(ref, provider=..., session_id=...)  # -> FileHandle | None
ctx.uploads.cached(provider, attachment_id)                # -> FileHandle | None
ctx.uploads.invalidate_handle(provider, handle)
ctx.uploads.stale(referenced)                              # -> paths gc may remove
ctx.uploads.prune(paths)                                   # -> int
```

`handle_for` answers `None` when this provider has no uploader — **not an
error**, because the caller's alternative is to send the bytes inline, which is
what every route did before this seam existed.

It takes a `session_id` and resolves the session itself. That is deliberate: the
first shape took the `Session`, and three adapters had written the same five-line
lookup by the time the third wire landed — an omission that shape invites is
silent in the worst way, since a forgotten argument loses `attachment/uploaded`,
the only record that bytes left the machine, while every request still succeeds.

## Providing one

```python
class Uploader(Protocol):
    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle: ...
```

Register **only when the route declares `uploads`**, so a file API is never put
behind a provider that has none:

```python
if uploads is not None and config.uploads:
    ctx.add_disposer(uploads.register_uploader(config.provider, adapter), label=...)
```

The three shipped uploaders are three protocols, not three spellings — plain
multipart (Anthropic), multipart plus a required `purpose` (OpenAI), and a
resumable two-step with a readiness poll (Google). `_http`'s `post_multipart`,
`post_raw` and `get_json` are the shared floor under them.

**An uploader may not be finished when the bytes have landed.** Google processes
video after storing it, and a `fileUri` referenced before its file reaches
`ACTIVE` is refused — so that uploader polls, and remembers what it transferred
if it runs out of patience, so the next attempt resumes rather than re-sending.
A *pending* id must never reach this seam: `FileHandle` is a **usable** id, and
caching a `PROCESSING` one would hand the next request a reference the provider
refuses, turning a re-upload into a retry loop.

## What makes it actually cheaper

`load_media(..., skip=handles.keys())`. Without it a referenced file was still
read and base64-encoded and then discarded by the renderer — the wire payload
shrank and nothing else did, while a 5.5 MB string sat in the encode cache for
the life of the process. Uploading is supposed to remove that work, not move it.

## Pruned by `ph attachments gc`

Off the same reference set as the blobs, because the question is the same one —
*does any stored session still point at this digest* — and answering it twice
would be two folds that can disagree. What is **not** shared is the reason:
losing an entry costs one upload and losing a blob costs the conversation, which
is why blobs are aged and guarded by a completeness check and these are not.

Still not enforced (§5 rule 6): nothing prunes it automatically. The command is
manual, and the directory is safe to delete wholesale.

## The row

```yaml
- id: uploads
  name: uploads-local
  config:
    root: /var/cache/ph/uploads    # optional; else $PH_CACHE/uploads
```

Mounted with **no uploader** — each adapter row registers its own — so a profile
with no file API pays nothing and every route sends bytes inline.

## See also

[`ctx.attachments`](attachments.md) · [Adding an
adapter](../cookbook/adding-an-adapter.md) · `test_uploads.py`,
`test_uploads_openai.py`, `test_uploads_google.py`
