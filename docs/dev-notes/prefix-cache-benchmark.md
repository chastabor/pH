# Prefix-cache hit rate and tokens/turn, with and without stabilization (P6-03)

A cache miss looks exactly like a cache hit, only on the invoice. A12 asserts
prefix stability *structurally* — `test_prefix_stability.py` checks that the
system prompt is byte-identical and that each request extends the last — but it
puts no number on what that is worth, and no number on what `stabilize` costs
or saves. This is that measurement.

**Headline: stabilization cuts tokens sent by 67 % under context pressure and
changes what a provider bills by under 2 %.** The saving is almost entirely a
saving on tokens the prefix cache would have served anyway. What stabilization
buys at these sizes is headroom, not price.

## The numbers

| context window | profile | requests | tokens/turn | sent | cacheable | uncached |
| --- | --- | --- | --- | --- | --- | --- |
| 200,000 | `rlm` | 6 | 23,651 | 141,906 | 74.9% | 35,641 |
| 200,000 | `rlm-stable` | 6 | 24,052 | 144,312 | 75.0% | 36,043 |
| 8,192 | `rlm` | 6 | 23,651 | 141,906 | 74.9% | 35,641 |
| 8,192 | `rlm-stable` | 6 | 7,845 | 47,069 | 22.9% | 36,278 |

*cacheable* is the share of everything sent that a provider could serve from the
previous request's prefix: the longest common run of messages, and only when the
system prompt is byte-identical. *uncached* is the remainder — what is billed at
full rate.

*Could* serve, and that is load-bearing: this measures pH's own prompt assembly,
not any provider's cache. An OpenAI-compatible route caches implicitly and gets
this for free; Anthropic's does not, which is why P6-13 sends `cache_control`
markers and asserts the read as a number rather than trusting this table to
imply one.

## The workload, and what it is not

**There is no recorded RLM session to replay, and that is worth stating before
the findings rather than after.** P3-23 established that the prime-agent
fixtures are "the *coding* agent, not the RLM" — neither contains a single
`ipython` call. dsh's own `python-sdk-single-exe` snapshots *are* RLM-shaped, and
they call `cordis_define`, `cordis_run`, `workflow` and `cordis_undefine`:
dsh-only tools with no pH counterpart, so replaying them yields six resolution
failures rather than a trajectory. Recording a fresh one needs a provider key.

So the model's side is authored: three turns, each running one Code Mode cell
that reads a file. **Only the choices are authored.** The cell really executes,
`await tools.read(...)` really runs, and what lands in the log is the file's own
bytes — fabricating tool *results* would have made every token count a statement
about the fabrication. What this licenses is a claim about pH's own prompt
assembly under a representative shape, not a claim about how any model behaves.

Two context windows, because the answer differs between them. `ReplayAdapter`
defaults to 8 192; a first draft measured only that, where every request is over
budget and compaction fires every turn, and reported a permanently-compacting
session as though it were the ordinary case.

## Finding 1 — without stabilization, A12 holds exactly

Every request's cacheable prefix is its predecessor *in full*. Nothing is
rewritten, so a provider re-reads only what the turn added, and the hit rate is
74.9 % across the run. This is the structural property, priced.

## Finding 2 — nothing in `rlm` consults the context window

The two `rlm` rows are identical measurements. An unstabilized profile has no
mechanism that reacts to the budget, so a session that outgrows it simply keeps
growing: the 8 192 row sends 141 906 tokens against an 8 192-token window, and
against a real provider the last four requests would be refused outright. That
is the gap `stabilize` exists to close, and it is worth seeing as a number
rather than as a design intention.

## Finding 3 — with a large window, stabilization is a small net cost

+1.7 % tokens, no change to the hit rate. `stabilize` adds prompt sections and
subtracts nothing, because neither offload nor compaction engages. Reporting
stabilization as an unconditional saving would have been the comfortable answer
and the wrong one.

## Finding 4 — offload cannot reach a Code Mode result at shipped defaults

This is the finding the benchmark was not looking for. Two shipped thresholds do
not meet:

* the `rlm` bundle pins `code-runtime-python` to `maxValueBytes: 65536`, so a
  cell's value is capped at 64 KiB — about 17 000 tokens;
* `tool-result-offload` triggers at `TOOL_TOKEN_LIMIT_BEFORE_EVICT = 20_000`.

A Code Mode result is therefore capped *below* the threshold that would offload
it. And in these profiles almost nothing is native — `ipython` is the only
callable tool, which is the point of Code Mode — so `tool-result-offload` has
essentially nothing left to act on. It is not misconfigured; it is shadowed.

Whether that is right is a design question this report does not settle. The two
readings are that the kernel cap already does offload's job more cheaply (no
blob, no reference, no second round trip), or that the cap silently truncates
where offload would have preserved the whole result behind a reference. The
second is the one worth checking, because truncation is lossy and offload is
not.

## Finding 5 — under pressure, the trade is real in both directions

With the window exceeded every turn, `compaction-summarize` replaces the history
with a summary. Tokens per turn fall by 67 % (23 651 → 7 845). The prefix hit
rate falls with them, 74.9 % → 22.9 %, because the summary lands at position 0
and every cached byte before it is invalidated.

That is I4's surface `replace` against A12's prefix stability, and the two are
structurally opposed: a projection that rewrites history cannot also be a
prefix that never moves. Both invariants are correct; the cost of holding them
together is this number.

## Finding 6 — the saving is almost entirely cached tokens

Uncached tokens across all four rows: **35 641, 36 043, 35 641, 36 278** — a
spread of 1.8 %. The 67 % reduction in tokens *sent* is a reduction in tokens the
cache would have served for a fraction of the price.

What follows for a deployment:

* If the session fits the window, `stabilize` costs ~2 % and buys the gates —
  the human-in-the-loop rules, the limits, the fs permissions. Those are its
  value there, not tokens.
* If the session does not fit, `stabilize` is what makes it run at all, and the
  right expectation is *headroom*, not a smaller bill.
* A provider that does **not** discount cached input inverts finding 6 entirely:
  there, the 67 % is a real 67 %. The benchmark measures tokens, not any
  provider's price list.

## What this did not measure

* Any real model's behaviour — the trajectory is authored (see above).
* `input-offload`, which needs a large pasted *input*; every prompt here is one
  short sentence.
* Sessions long enough for compaction to run repeatedly under a large window,
  which is the regime a real long-running agent reaches and the one where
  finding 5's trade compounds.
* Cache *TTL*. A provider's prefix cache expires; every number here assumes the
  previous request's prefix is still warm, which is the best case.

## Re-deriving this report

```
uv run python tests/prefix_bench.py          # print the table
uv run python tests/prefix_bench.py --json   # rewrite tests/prefix_bench.json
uv run pytest tests/test_prefix_bench.py -q  # hold the six findings
```

Needs no `sources/` — the workload is authored, so this runs on a clean clone.
The table above is rendered from `tests/prefix_bench.json`, the run it records.
The tests assert the findings as **relationships rather than digits**: absolute
counts move with the length of the temporary directory, because the workspace
path is rendered into the session context snapshot and is therefore part of what
the model reads. A guard on 141 906 would fail on a machine with a longer
`TMPDIR`, which is a property of the runner and not of pH.
