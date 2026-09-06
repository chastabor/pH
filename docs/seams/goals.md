# `ctx.goals` — an objective, a budget, and the gates that decide it

**Module:** `ph/seams/goals.py` · **Row:** `goals` · **Consumers:**
`/autonomous` and its `agent/turn-stopping` listener

An autonomous run is a loop with three ways to stop: **the gates pass**, **a
budget runs out**, or **a person cancels it**. This seam holds all three as facts
in the log, so the loop reading them can be restarted, resumed, or driven by a
daemon that was not running when the goal was set.

## Budgets are counted, not trusted

```text
Budget = max_continuations | max_turns | max_tokens | timeout_ms
```

Four limits rather than one, because they fail differently:

| limit | the failure it catches |
|---|---|
| `max_continuations` | a model looping on the same edit |
| `max_turns` | a model that cannot stop talking |
| `max_tokens` | a long context |
| `timeout_ms` | a hung gate |

One number could only ever catch whichever happened to bind first.

Every one is **folded from the session's own events**, never from a counter the
loop keeps: a run that survives a daemon restart must not come back with a fresh
allowance.

## Three outcomes, and they are not interchangeable

```text
Outcome = "achieved" | "budget_limited" | "abandoned"
```

* `achieved` — every gate passed;
* `budget_limited` — a budget ran out **with gates still failing**;
* `abandoned` — a person stopped it.

A loop reporting the second as the first would be **claiming work it did not
do**, which is the failure this whole layer exists to make impossible. It is also
why exhaustion is a *named* outcome rather than silence: "it stopped" and "it
stopped because it ran out" are different things to the person reading the trace,
and only one of them suggests raising the budget.

## A gate is a shell command and a fingerprint

Quality gates are the reason an autonomous run can be trusted to stop on its own:
`pytest`, `mypy`, whatever the deployment says "done" means.

Each result is recorded against the **tree hash** of the agent's work at the
moment it ran. So a gate that failed against a tree the agent has not changed is
**not run again** — the answer cannot have changed, and re-running a slow suite to
learn nothing is how a budget gets spent on nothing. An edit anywhere in the
worktree changes the hash and the gate runs.

`unchanged_failure(...)` is that check, and it is the reason `write-tree` was
extracted from the checkpoint machinery.

## Tools and commands, never host handlers (C2)

Everything here is reached through `ctx.goals` by a **command a person types** or
a **tool the model calls**, so the governed pipeline sees it. There is no
host-side path that sets a goal without a record.

## The surface

```text
ctx.goals.set(session, goal)          # -> the goal, recorded
ctx.goals.open(session)               # -> the live goal, if any
ctx.goals.record_gate(session, ...)   # a result, against a tree hash
ctx.goals.continued(session, ...)     # one more pass, counted
ctx.goals.settle(session, outcome)
ctx.goals.states(session)             # the fold
ctx.goals.unchanged_failure(...)      # may this gate be skipped?
```

A `Goal` is `id`, `objective`, `gates`, `budget`.

## Events

| event | |
|---|---|
| `goal/set` | the objective, the gates, the budget |
| `goal/gate` | one gate result, with the tree hash it ran against |
| `goal/continued` | one more pass, and what it spent |
| `goal/settled` | the outcome |

The fold across those is the whole state. `open_goal(session)` is what a resumed
daemon reads to discover it has work in progress, and the reason a goal survives
a restart without a table.

## What it does not do

* **It does not drive the loop, and nothing else has to.** `/autonomous`
  registers a policy on `agent/turn-stopping`: a turn ending with a goal still
  open is steered into the next step rather than allowed to stop, and the loop
  only breaks when the inbox is empty. The daemon drives `agent.run()` and so
  inherits the continuation without a loop of its own.
* It does not run gates. It records their results; the shell does the running,
  through [`ctx.shell`](../seams/README.md) and whatever confinement is mounted.
* It does not survive a fork with its allowance intact — the module's promise
  that a run cannot come back with a fresh allowance holds across *resume* and
  not across *fork*.

## See also

[`ctx.schedule`](schedule.md) · [`ctx.token_meter`](token_meter.md) ·
[`ctx.subagents`](subagents.md) · `test_goals.py`
