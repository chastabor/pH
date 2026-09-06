# `ctx.invariants` — which invariants this deployment enforces, and whether they hold

**Module:** `ph/seams/invariants.py` · **Row:** `invariants` · **Consumers:**
`ph doctor`, the invariant rows themselves

## The gap this closes

pH's invariants are enforced by **rows**, and a row is optional.

So *"I3 holds"* was never a property of pH — it was a property of a profile that
happened to mount `agent-loop-invariant`, and nothing said which profiles those
were. A person reading `DESIGN.md` learned what pH promises, ran a profile that
promised less, and **had no way to find out**.

An invariant that is enforced now says so out loud, and one that nobody mounted
is **absent from the report** rather than silently assumed.

## Two kinds, and the difference is load-bearing

| kind | example | why |
|---|---|---|
| **inline** | I3 checks every request as it is built, and refuses | there is no state *between* requests to poll — the answer to "does it hold" is "every request so far was checked" |
| **pollable** | a projection either equals its fold right now or it does not | carries a `check` |

Reporting the two identically would be the overstatement E1 forbids **in both
directions**: an inline invariant reported as "holds" claims a check that did not
run, and a pollable one reported as "enforced" claims a guarantee about a file
nobody read.

So an `Invariant` carries `id`, `statement`, `order`, and a `check` that is
**optional** — its absence is the declaration that this one is inline.

## The surface

```text
ctx.invariants.register(invariant, *, scope=None)   -> Disposer
ctx.invariants.enforced()        # what this profile actually mounts
ctx.invariants.describe()        # for ph doctor
await ctx.invariants.verify()    # run every pollable check
```

## Registering one

```python
ctx.invariants.register(
    Invariant(
        id="session-invariant",
        statement="the transcript rebuilds from session.events alone",
        check=_check,          # omit for an inline invariant
    )
)
```

The shipped set: `session-invariant`, `tools-invariant`, `skills-invariant`,
`scope-invariant`, and the agent loop's `messages == derive_messages()`.

A row that enforces something worth promising should register — the report is
only as complete as what rows declare, and an unregistered enforcement is a
guarantee nobody can discover.

## What it does not do

* It does not enforce anything itself. It is a **registry and a report**; the
  rows do the enforcing.
* It does not make an invariant true. Mounting the row is what does that, which
  is why the report distinguishes enforced from absent rather than listing
  everything pH could promise.
* It does not run inline checks on demand — there is nothing to run.

## See also

[`ctx.diagnostics`](diagnostics.md) · `test_invariants.py`
