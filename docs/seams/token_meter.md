# `ctx.token_meter` — the provider's count is the truth; ours is for pressure

**Module:** `ph/seams/token_meter.py` · **Row:** `token-meter` · **Consumers:**
compaction policy, the offload rows, the TUI context gauge, `ctx.goals`

Two numbers, and conflating them causes real bugs.

| | |
|---|---|
| **provider-reported `usage`** | **authoritative** — what gets billed, what the window is measured against |
| **an estimate** | exists only to decide *"should we compact **before** asking"* |

There is no usage number for a request that has not been made yet, so something
has to guess — and a guess that is 15% off is fine for a threshold.

## The baseline switches once, and never drifts back

The baseline starts as an estimate and becomes **reported usage the moment the
first response lands** (D15). It does not alternate.

That one-way switch is the design: a meter that fell back to estimating after a
request that reported nothing would make the gauge move for reasons the
conversation cannot explain, and a compaction trigger that fired on an estimate
*after* real numbers were available would be second-guessing the provider.

## The surface

```text
ctx.token_meter.measure(message)          # one message
ctx.token_meter.measure_text(text)
ctx.token_meter.estimate_messages(msgs)
ctx.token_meter.baseline(...)             # what pressure is judged against
ctx.token_meter.last_usage(...)           # what the provider actually reported
```

`tiktoken` is used when installed and `len/4` otherwise. The fallback is
deliberately crude: a harness that refused to start without an optional tokenizer
would be worse than one that occasionally compacts a turn early.

## Media is not free

`measure` reads `.text`, `.arguments` and nested `.content` — and a `MediaBlock`
has none of them, so for a while an image contributed **zero**. A conversation of
forty pictures reported no pressure at all and G2/G3's character thresholds never
fired on it.

Media now carries a per-MIME estimate: images ≈ `w×h/750` when the dimensions
were measured, audio per second, PDFs per page, and a flat floor otherwise. The
numbers are approximate on purpose — what matters is that the answer is never
zero, because zero is the only value that is wrong in a way nothing recovers
from.

## Reading pressure

`TokenBaseline.pressure` is what a policy row and the TUI footer both read, so
the number a person sees and the number that triggers compaction are the same
number. A second definition of pressure is how a footer comes to disagree with
the behaviour it is describing.

## What it does not do

* It does not decide to compact. It reports; [`ctx.compaction`](compaction.md)
  and a policy row decide.
* It does not bill. `ctx.goals` and the telemetry ledger do that from the
  reported usage, not from estimates.
* It does not know the window. `ResolvedModel.context_window` does, and a route
  that publishes none leaves pressure undefined rather than invented.

## See also

[`ctx.compaction`](compaction.md) · [`ctx.spill_store`](spill_store.md) ·
`test_attachments.py` (the media estimates), `test_seams.py`
