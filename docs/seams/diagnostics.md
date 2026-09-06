# `ctx.diagnostics` — what a row wants `ph doctor` to say about it

**Module:** `ph/seams/diagnostics.py` · **Row:** `diagnostics` · **Consumer:**
`ph doctor`

## Why it exists

Four rows in three packages arrived at this one by one: the containment tier, the
workspace kind and `repo_writable` per agent, a permission row's honest reach, and
the worker model.

`ph-app` **cannot import** `ph-stabilize` or `ph-rlm`. So without a seam, each of
them lands as a bespoke `ctx.<name>` the consumer has to know by heart — and the
consumer is *the one command whose entire job is to be complete*.

## This is `ctx.tui_status` minus the `Session`

Same registration shape, same `scope=`, same drop-a-raising-contributor rule,
same order-then-id sort. The difference is what it is read **for**:

| | answers | so it must be |
|---|---|---|
| a footer field | "where am I now", every spinner frame | **cheap** |
| a diagnostic | "what is this deployment", once, on request | free to stat a tree or spawn a subprocess |

Two seams rather than one parameterised seam, because a field that quietly became
expensive would take the footer down with it.

## Rows, not a sentence

`read()` returns `(label, value)` pairs — the shape a `ph doctor` section takes.

An **empty list keeps the section off the page entirely**, which is what lets a
healthy store contribute nothing: a section that appears on every run saying
"fine" is a section nobody reads, and then the run where it says something else
is missed too.

## The surface

```text
ctx.diagnostics.register(diagnostic, *, scope=None)   -> Disposer
await ctx.diagnostics.report()                        # what doctor prints
```

A `Diagnostic` is `id`, `title`, `order`, `read`.

```python
contribute(ctx, Diagnostic(
    id="session-lineage",
    title="Session lineage",
    read=partial(lineage_faults_of, store),
    order=20,
))
```

## State what is *not* enforced

This is where §5 rule 6 lands most often: a section that describes a tier is the
one place a person looks to check exactly what it bounds, so a diagnostic that
overstates is the single failure E1 exists to prevent.

`ph doctor` prints the retained-tree count **even when it is none**, because the
assumption a reader makes in the absence of a row is that nothing is
accumulating — which is precisely the assumption worth checking.

## What it does not do

* It does not fix anything, and it does not exit non-zero for a finding —
  `ph doctor` reports; a profile that will not *mount* is the separate loud
  failure.
* It does not run on a timer.
* It does not know about a terminal. `(label, value)` pairs render wherever.

## See also

[`ctx.tui_status`](tui_status.md) · [`ctx.invariants`](invariants.md) ·
`test_diagnostics.py`, `test_cli.py`
