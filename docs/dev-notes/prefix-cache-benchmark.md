# Prefix-cache hit rate and tokens/turn, with and without stabilization (P6-03)

A cache miss looks exactly like a cache hit, only on the invoice. A12 asserts
prefix stability *structurally* — `test_prefix_stability.py` checks that the
system prompt is byte-identical and that each request extends the last — but it
puts no number on what that is worth, and no number on what `stabilize` costs
or saves. This is that measurement.

**Headline: under context pressure, stabilization cuts tokens sent by 88 % and
keeps the prefix cache, so what a provider bills falls by as much.** The pressure
in this workload is one cell result a turn larger than the window, and
`compaction-summarize` clips each one before the request that would carry it.
With a window the session fits, stabilization costs about 2 %.

Until 2026-10-12 the same run summarized the history every turn instead: 67 %
fewer tokens sent, the prefix lost, and under 2 % off the bill. Finding 5 says
what changed, and when that trade still applies.

## The numbers

| context window | profile | requests | tokens/turn | sent | cacheable | uncached |
| --- | --- | --- | --- | --- | --- | --- |
| 200,000 | `rlm` | 6 | 23,878 | 143,268 | 75.0% | 35,868 |
| 200,000 | `rlm-stable` | 6 | 24,279 | 145,674 | 75.1% | 36,270 |
| 8,192 | `rlm` | 6 | 23,878 | 143,268 | 75.0% | 35,868 |
| 8,192 | `rlm-stable` | 6 | 2,846 | 17,074 | 75.9% | 4,120 |

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
75.0 % across the run. This is the structural property, priced.

## Finding 2 — nothing in `rlm` consults the context window

The two `rlm` rows are identical measurements. An unstabilized profile has no
mechanism that reacts to the budget, so a session that outgrows it simply keeps
growing: the 8 192 row sends 143 268 tokens against an 8 192-token window, and
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

The overflow clip is the exception, since 2026-10-12. It spills a result through
offload's own operation when the request carrying it would not fit the window
(finding 5), so at the 8 192 window every cell result here is spilled after all,
and at 200 000 none is.

## Finding 5 — under pressure from one result, the clip keeps the prefix

Each turn's cell result, about 16 600 tokens, is larger than the 8 192 window by
itself. At the next step's start `compaction-summarize` sees that the request
carrying it would fill the whole window, and clips it first: the result is spilled
to a file and replaced by a pointer before it is ever sent. Tokens per turn fall by
88 % (23 878 → 2 846), and the hit rate does not fall at all (75.9 %), because
nothing that was already sent is rewritten: each request still extends the last.

Before that check (2026-10-12), the request went out over the window, and a summary
replaced the history every turn: tokens per turn fell by 67 %, and the hit rate
with them, 74.9 % → 22.9 %, because the summary lands at position 0 and every
cached byte before it is invalidated. That is I4's surface `replace` against A12's
prefix stability, and the two are structurally opposed: a projection that rewrites
history cannot also be a prefix that never moves. The trade still applies when the
history itself outgrows the window; this workload no longer reaches it, since after
the clip three turns fit.

## Finding 6 — the saving reaches the bill

Uncached tokens: **35 868, 36 270, 35 868** for the three rows the clip leaves
alone, a spread of 1.1 %, and **4 120** for `rlm-stable` at 8 192. Because the clip
keeps the prefix, the tokens it saves are ones a provider would have billed at full
rate, not ones its cache would have served. (The summarizing run before it saved
almost only cached tokens: its four rows were within 1.8 % of each other.)

What follows for a deployment:

* If the session fits the window, `stabilize` costs ~2 % and buys the gates —
  the human-in-the-loop rules, the limits, the fs permissions. Those are its
  value there, not tokens.
* If one result does not fit, `stabilize` clips it before it is sent. The model
  reads a pointer to the file in its place, and the bill falls with the tokens.
* If the history does not fit, `stabilize` summarizes it, and the right
  expectation there is *headroom*, not a smaller bill, on a provider that
  discounts cached input. A provider that does **not** discount it makes that
  saving a real one too. The benchmark measures tokens, not any provider's price
  list.

## What this did not measure

* Any real model's behavior — the trajectory is authored (see above).
* `input-offload`, which needs a large pasted *input*; every prompt here is one
  short sentence.
* A history that outgrows the window, where a summary is the remedy and finding
  5's trade applies; this workload's pressure is one result a turn.
* Sessions long enough for compaction to run repeatedly under a large window,
  which is the regime a real long-running agent reaches and the one where
  that trade compounds.
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
the model reads. A guard on 143 268 would fail on a machine with a longer
`TMPDIR`, which is a property of the runner and not of pH.
