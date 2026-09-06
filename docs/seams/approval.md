# `ctx.approval` — asking a human, and failing closed when you cannot

**Module:** `ph/seams/approval.py` · **Row:** `approval` · **Consumers:** the tools
pipeline, `permissions-fs`, the RLM harness service

An ask turns into exactly one of four outcomes, and **only `allowed-once`
proceeds** (B3).

## Fail closed is the whole design

No answerer, an unmounted seam, a cancelled prompt, an exception inside an
answerer — every one of them denies. A permission system whose failure mode is
"allow" is not a permission system.

The consequence for a consumer: you never need a `try` around `request()` to stay
safe. What you do need is to distinguish *why* it did not grant.

## The four outcomes, and why they are four

```text
ApprovalOutcome = "allowed-once" | "rejected" | "cancelled" | "unavailable"
```

Only the first proceeds; the other three are distinct **on purpose**. A model
told "the user rejected this" can re-plan. One told "there is no approval channel"
knows the deployment is misconfigured rather than that a human said no.
Collapsing them makes a missing UI look like a decision.

Do not write those sentences yourself:

```python
from ph.seams.approval import denial_reason

reason = denial_reason(outcome, subject=f"the {name} tool")
```

`DENIAL_REASONS` exists because there are three consumers, and the first copy to
disagree was the one that collapsed all four outcomes into one string. `{subject}`
is the caller's, since a tool call, a path and a refinement are named differently
and only the caller knows which it is holding. An unknown answer reads as
`unavailable` — the fail-closed direction, and the honest one.

## Asking

```text
await ctx.approval.request(request, cancel=None)   # -> ApprovalAnswer
```

An `ApprovalRequest` carries what the human needs to decide and what the log
needs to record. The answer is one of the four outcomes, or one of two richer
replies a front end may return instead:

| answer | meaning |
|---|---|
| `allowed-once` | proceed, this once |
| `rejected` / `cancelled` / `unavailable` | do not proceed; see `denial_reason` |
| `Edited(arguments)` | proceed, but with *these* arguments — the human changed the call |
| `Responded(text)` | do not proceed; the human answered in words instead |

`Edited` is why an approval is not a boolean. The pipeline applies the
substitution before the body runs, and `PreparedCall.substituted` records that it
happened — so the log shows what actually ran rather than what was asked for.

## Answering

A front end registers an answerer:

```python
ctx.approval.register_answerer(answerer)
```

There is one channel, so a deployment configures its answerer once — the TUI
modal, the daemon's RPC, or a test's stub. `ctx.user_questions` is the sibling
seam for asking something that is *not* an approval; do not overload this one for
questions, because everything here is gated on a decision that must fail closed.

`set_policy(session, policy)` switches a session between `"ask"` and `"never"`.
It is read from the log rather than held in memory, so a resume keeps the posture
the person chose.

## Re-asking on resume falls out of the log

There is no pending-approvals table. `approval/asked` without a matching
`approval/decided` **is** the pending state, so a crash between the two leaves a
question a resumed session can find and put back to the human —
`pending_approvals(session)` is that fold.

That is also why both events are appended: the ask is durable evidence that the
harness stopped and waited, and a consumer that recorded only decisions would
lose every interrupted one.

| event | mode | |
|---|---|---|
| `approval/request` | waterfall | routes one prompt to an answerer; fails closed |
| `approval/asked` | *session event* | a human was asked |
| `approval/decided` | *session event* | what they said |

## Where the decision is made, and where it is not

The seam **routes and records**; it does not decide *what* needs approval. That
is a policy row's job, on `tools/pre-execute` (return `ask`) or `fs/*-intent`.
`ph-stabilize`'s `hitl` row is the shipped example, and `destructive.py` is its
classifier.

Two rules from the pipeline that matter here:

* **Approval runs before guards, and guards are the final word.** A monotonic
  guard is deny-only and runs *last*, so it overrides even an explicit human
  approval. Policy that must not be reorderable belongs in a guard, not a
  listener.
* **A gate that fires on routine work is one a person learns to approve without
  reading**, which `hitl` calls worse than no gate. When adding a rule, prefer
  narrow: `rm` gates on `-r` because recursion is the multiplier, and a single
  `rm -f` passing ungated is a stated cost rather than an oversight.

## The row

```yaml
- id: approval
  name: approval
```

No config. The seam is mounted everywhere; what varies is whether an answerer is
registered, and a profile with no front end simply denies — which is the correct
posture for `ph -p`, a scheduled tick or a sandboxed run.

## What it does not do

* It does not remember decisions. There is no "always allow" — `allowed-once` is
  the only grant, and a deployment that wants standing permission expresses it as
  policy (a preset, a rule) rather than as a remembered answer.
* It does not decide *what* to ask about.
* It does not queue. One request, one routing; a front end that has gone away
  produces `unavailable`, not a wait.

## See also

[`ctx.tools`](tools.md) · [Adding a seam](../cookbook/adding-a-seam.md) ·
`test_seams.py`, `test_hitl.py` (ph-stabilize)
