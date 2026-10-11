# `ctx.token_meter` — the provider's count is the truth; ours is for pressure

**Module:** `ph/seams/token_meter.py` · **Row:** `token-meter` · **Consumers:**
compaction policy, the offload rows, ph-clm's gate, receipts, readouts and section
map, `ctx.goals`

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

## Until the next response, it follows the surface

The provider counted the request it answered — system prompt and tool schemas
included — and nothing after it. So the baseline is that count **moved by what the
surface did since**:

* every surface event after the reply that reported adds a node, estimated — the
  step's tool results, a message spliced in;
* a replacement takes off what it shadows: a node the provider counted is estimated
  and subtracted, one added since simply never counts — a compaction summary, a
  model's own context edit, an elided argument;
* `pending`, messages not yet logged, goes on top.

Only the events since that reply are walked, and each node is estimated once
(`node_tokens`, kept per session and let go on `session/disposed`) — the same
estimate the branch before any reply sums over the surface, and the one ph-clm's
section map reads rather than measuring a node again. Before this, the baseline was the last
count plus `pending` alone: a step's tool results were not in it until the next
response, and an edit or an elision did not move it at all — so compaction's
re-measure after eliding arguments could never spare the summary it was meant to.

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

`TokenBaseline.pressure` is what compaction's trigger, ph-clm's `fit` gate and its
readouts read, so the number the model is told and the number that triggers
compaction are the same number. A second definition of pressure is how a readout
comes to disagree with the behavior it is describing.

**The TUI footer is the exception, and it agrees at every response.** It shows the
provider's count of the last request (`TuiEventAdapter._count_usage`): the adapter
folds events one at a time on the front end — a remote one holds only a mirror of
the log and no meter — and does not estimate. Between a response and the next it does
not move for a tool result or an edit, where the baseline does. The way to put it on
the baseline is a daemon-side `StatusField` reading, as the meter already contributes
`reasoning` and `cache`, re-published per step rather than only when the agent's
status moves, with compaction's own threshold as its warning level.

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
