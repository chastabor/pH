# Adding an adapter

An adapter is one provider's wire. It turns a `GenerateOptions` into HTTP and the
response back into pH's chunk vocabulary — and nothing above it learns which
adapter answered.

Three ship, and they are the worked examples: `ph_app/adapters/anthropic.py`,
`openai_compatible.py`, `google.py`. Read one alongside this page; they differ in
message shape, in how usage is reported and in how a file API is shaped, and
those three axes are most of the work.

## The contract

`LlmAdapter` requires one method:

```python
async def stream(self, options: GenerateOptions) -> AsyncIterator[Any]: ...
```

`resolve_model` is optional but you almost always want it — it is how the layers
above learn the route's context window, what media it accepts and whether it
enforces a response schema.

Register in `apply`:

```python
@plugin("llm-acme", config=Config, inject=["llm", "credentials"])
async def apply(ctx: Context, config: Config) -> None:
    adapter = AcmeAdapter(ctx=ctx, config=config)
    handle = ctx.llm.register_adapter([config.provider], adapter)
    ctx.add_disposer(handle.dispose, label=f"llm({config.provider})")
    ctx.add_disposer(adapter.http.aclose, label=f"http({config.provider})")
```

One adapter may claim several provider names — `llm-openai-compatible` registers
one per configured route, which is why its config is a list of profiles.

## What `stream` must yield

The vocabulary is in `ph.llm.types`, and `BlockAssembler` reconstructs a message
from it:

```
BlockStart(index, block_type)      # "text" | "reasoning" | "tool-call"
TextDelta / ReasoningDelta / ToolCallDelta
BlockEnd(index, block)             # the completed block
UsageChunk(usage)                  # before the finish
Finish(reason=FinishReason(kind))  # exactly one, last
```

Two rules that are easy to break:

* **Stream deltas, not completed blocks.** `assistant/chunk` promises
  token-level replay; an adapter that only emitted finished calls would make that
  promise false. Where a provider sends whole parts (Google does), emit each part
  as a delta on the block it continues — same fidelity, larger pieces.
* **Thinking is `reasoning`, never text.** DeepSeek's `reasoning_content`,
  Anthropic's `thinking` blocks, Google's `thought: true` parts all map to
  `reasoning`. Folding them into visible text makes the transcript claim the model
  said what it was only considering.

## Usage counts are disjoint (D15)

`TokenUsage` counts do not overlap. Providers disagree about this, so each adapter
corrects in its own direction:

* DeepSeek and Google fold cache hits *into* the prompt total — subtract them out,
  or every cache hit is billed twice in pH's accounting.
* Google reports `thoughtsTokenCount` *outside* `candidatesTokenCount`, where pH
  documents `reasoning_tokens` as a **subset** of `output_tokens` — add it in, or
  every thinking turn under-reports the part that did the work.

`TokenUsage.total` deliberately omits `reasoning_tokens` for that reason. Get the
mapping wrong and nothing fails; the bill is just wrong.

## Credentials are resolved at the edge, and nowhere else (I-3)

```python
secret = resolve_secret(self.ctx, self.config.api_key_env, self.config.provider)
```

The seam passes a credential *reference* around; only the adapter turns one into
a header, into a local that goes out of scope with the request. Config holds the
**name** of an environment variable, never an interpolation of its value — so the
secret never enters a row, an event or a child process.

## Sharing the transport

`adapters/_http.py` holds what every wire has in common: one long-lived
`httpx.AsyncClient` per adapter, `resolve_secret`, and — most importantly — the
status→code classification the retry policy routes on. Wire-specific judgements
are callbacks, because each provider phrases them differently and both are
expensive to get wrong:

* `is_overflow(body)` — a missed context overflow retries forever; a false one
  compacts a conversation that fit.
* `is_missing_file(body)` — for a route with a file API.

Use `stream_sse` for the request. `post_multipart`, `post_raw` and `get_json` are
there for file APIs.

## Declaring the route

```python
def resolve_model(self, provider: str, model: str) -> ResolvedModel:
    return resolved(self.config, structured_output=False)
```

`resolved(route, structured_output=)` projects the six route facts —
`context_window`, `default_max_tokens`, `accepts`, `max_attachment_bytes`,
`max_image_edge`, `usable_image_edge` — onto `ResolvedModel`. Your config
satisfies the `MediaRoute` Protocol by having those names; if it does not, mypy
says so at the call rather than the route silently reporting defaults.

`structured_output` is passed rather than read from config because it is a claim
about the *wire*, not a setting: say `True` only if the server actually enforces
the schema.

`accepts` and the limits belong in **row config**, not module constants. One
`openai-compatible` row serves a hosted gateway, Groq, Together and a local
llama.cpp, and they do not agree about media — a constant made every one of them
promise images.

## Media, if the route takes any

`ph.llm.media` (`media-degrade`) answers the *policy* question above every
adapter, so by the time a request reaches you, any `MediaBlock` still on it is one
this route said it accepts. What is left is the wire shape:

* `load_media(store, messages, skip=handles.keys())` — base64 by attachment id;
* `load_handles(uploads, messages, provider=, mimes=, session_id=)` — provider
  file ids for what this route references rather than inlines;
* `media_pointer(attachment)` — the text block that stands in for what you cannot
  express.

**Be total over your own vocabulary.** A MIME you have no shape for becomes a
pointer, never something dressed as an image. That is not a second copy of the
accept policy; it is the narrower honesty an adapter still owes.

For a file API, implement `Uploader.upload` and register it with
`ctx.uploads.register_uploader(provider, self)` — only when the route declares
`uploads`, so a file API is never put behind a provider that has none. On a
mid-request failure naming a handle you sent, call `forget_named_handle(...)`:
`FILE_EXPIRED` is transient precisely because the dead entry is cleared before
the retry.

## Checklist

- [ ] deltas, not completed blocks; thinking maps to `reasoning`
- [ ] usage counts disjoint, in this provider's direction
- [ ] the credential resolved only in `_headers`, from a variable *name*
- [ ] `is_overflow` matches this provider's phrasing and nothing broader
- [ ] `resolve_model` goes through `resolved(...)`
- [ ] a test against a stubbed transport — `test_uploads_openai.py` is the shape
