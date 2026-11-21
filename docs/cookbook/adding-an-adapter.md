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
    return resolved(
        self.config,
        structured_output=False,
        credential=self.config.api_key_env,
        sampling=frozenset(SAMPLING_WIRE),
        max_decision_tokens=None,
        uploads=referenced_mimes(self.ctx, self.config.uploads),
    )
```

`resolved(route, ...)` projects the six route facts (`context_window`,
`default_max_tokens`, `accepts`, `max_attachment_bytes`, `max_image_edge`,
`usable_image_edge`) onto `ResolvedModel`. Your config satisfies the
`MediaRoute` Protocol by having those names. If it does not, mypy says so at the
call, and the route does not silently report defaults.

The five keyword arguments are required, so an adapter cannot forget one:

- `structured_output` is a claim about the *wire*, not a setting. Say `True`
  only if the server enforces the schema.
- `credential` is the name of the environment variable that the adapter
  resolves, or `None` for a route that needs none.
- `sampling` is the set of sampling settings that the API takes.
- `max_decision_tokens` is `None` for an adapter that does not decide.
- `uploads` is the set of MIME types that the adapter sends by reference
  through a file API. Pass `frozenset()` if the provider has none. A type in
  this set is not held to `max_attachment_bytes` by `media-degrade`, so the
  adapter must pass that limit to `load_media(..., inline_limit=)`. A file with
  no uploaded handle then goes inline only if it fits.

`accepts` and the limits belong in **row config**, not module constants. One
`openai-compatible` row serves a hosted gateway, Groq, Together and a local
llama.cpp, and they do not agree about media — a constant made every one of them
promise images.

## Deciding, if the provider has a decision model

A decision model is asked typed questions about a state and answers with
probabilities rather than text. An adapter whose provider has one implements a
second method, and that is the whole of the opt-in:

```python
async def decide(self, request: DecisionRequest) -> Decisions: ...
```

`DecisionAdapter` is its own `runtime_checkable` Protocol beside `LlmAdapter`,
asked with `isinstance` — an adapter without the method is simply not one, and
`ctx.llm.decides(provider)` says which is which. Callers go through
`ctx.llm.decide(request, session=, timeout_s=)`, which runs the adapter as the
row that registered it, records the spend (`llm/decided`) or the failure
(`llm/decision-failed`) in the session it was for, and holds every adapter to
one rule: each question gets the kind of answer it is owed.

The three kinds of question, and their fields, carry the wire's own names:

| question | answer | |
|---|---|---|
| `Noul(instructions)` | `NoulAnswer(noul)` | How likely yes is. Never a verdict: the line is the caller's. |
| `Choice(instructions, criteria)` | `ChoiceAnswer(choice, probabilities, confidence)` | `criteria` maps each key to what it means; shares come back in that order. |
| `Score(instructions, criteria)` | `ScoreAnswer(score, probabilities, confidence)` | `criteria` lowest first; shares in that order. |

A caller reads each answer as the kind it asked for — `decided.answer("team",
ChoiceAnswer)` — and gets `BAD_DECISION` for any other.

A route that decides should also say how much one decision may carry, where its
server will not: `resolved(..., max_decision_tokens=)`, from row config —
required, so an adapter that does not decide passes `None`. The seam lowers it
by what a refusal names, and askers size their state by
`ctx.llm.decision_limit(provider, model)`, through `ph.llm.tokens` (`bytes_of`,
`tokens_of`) where they have no tokenizer to ask. Text cut to such a size is cut
with `ph.characters`, which only stops between characters, never inside an
accent or an emoji sequence — and so is every other cut in the tree: `head` and
`tail` by code points, `head_bytes` and `tail_bytes` by bytes, `capped_text` for
a buffer a byte cap threw the rest of away, `stream_decoder` for bytes read in
pieces.

`openai_compatible.py` is the worked example: `/systemone` beside
`/chat/completions`, answered by llama.cpp and Ollama. Three things it does that
yours should:

* **Refuse a reply in another shape** with `BAD_DECISION`, and say what was
  wrong. A caller draws a line through these numbers; a string coerced to a
  float is a decision made on nothing.
* **Leave an unanswered question out** rather than inventing an answer. The seam
  refuses the reply for it.
* **Classify "the state did not fit" as `CONTEXT_WINDOW_EXCEEDED`**, with the
  server's sentence kept: `too_long_facts` reads the two numbers out of it, and a
  caller aims a shorter state by them. `input_too_long` is the phrase list for
  servers that read a decision in one batch, as llama.cpp does.

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
- [ ] if it decides: a reply in another shape is `BAD_DECISION`, an overflow
      `CONTEXT_WINDOW_EXCEEDED`
- [ ] a test against a stubbed transport — `test_uploads_openai.py` is the shape
