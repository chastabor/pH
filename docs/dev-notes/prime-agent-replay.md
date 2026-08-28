# Prime Agent trajectory replay (P3-23)

**What this is.** Two of prime-agent's own recorded sessions, read and compared
against the surface pH's `rlm` profile presents. The row's exit criterion is
"report checked in; unexpected diffs triaged" — this is the report, and
`packages/ph-rlm/tests/test_fixture_replay.py` re-derives its claims wherever
the vendored checkout exists.

**What it is not.** A re-run. Replaying a trajectory would need the model, the
key and the network; none is available to a test, and a recorded trajectory is
not a deterministic program anyway — the same prompt against the same model does
not reproduce the same tool calls. So the comparison is *structural*: reduce
each recorded session to the facts the port could change, and account for every
difference.

The fixtures live under `sources/`, a vendored reference checkout that is
deliberately not part of this repo. The tests skip without it; this report is
the durable artifact.

## The trajectories

| | `before-compaction.jsonl` | `large-session.jsonl` |
|---|---|---|
| records | 1 003 | 1 019 |
| assistant turns | 484 | 453 |
| user messages | 55 | 88 |
| tool results | 448 | 373 |
| tool calls | 454 | 391 |
| `bash` | 206 | 192 |
| `edit` | 125 | 146 |
| `read` | 107 | 50 |
| `write` | 16 | 3 |
| compactions | 2 | 0 |
| model changes | 5 | 1 |
| thinking-level changes | 5 | 103 |
| `bashExecution` (user shell) | 3 | 0 |

## Finding 1 — these are the *coding* agent, not the RLM

Neither fixture contains a single `ipython` call or an `rlm(...)` spawn. Both are
prime-agent's coding agent driving four native tools. That reframes the whole
exercise: the replay is a statement about **surface translation**, not about RLM
behaviour, and it exercises precisely the surface C1–C3 replaced.

It also means the two fixtures cannot show the RLM-specific diffs. That is
recorded below rather than glossed, because "the fixture did not exercise it" and
"there is no difference" are different claims.

## Finding 2 — every tool they called exists here

`bash`, `edit`, `read`, `write` are all registered tools in the shipped `rlm`
profile, asserted against the *mounted* registry rather than a list. So every
tool call in both trajectories is expressible. What changes is the route, which
is Finding 3.

## The diffs, triaged

Four. What the tests hold is not a list of diff *names* — the first draft did
that, comparing a set of literals against a set of literals written in the same
function, which could not fail — but a **coverage table over the fixtures' own
vocabulary**: every record type and every role either maps to a place in pH's
log or is named as a gap. A fixture bringing something unmapped fails there,
which is the version of "triaged" that survives someone adding one later.

### `one-cell-many-dispatches` — expected, and the point of the port

Prime Agent's 454 tool calls are 454 top-level model tool calls. Under Code Mode
the same work is *one* `ipython` call per cell plus one governed dispatch per
capability use. The number of **governance decisions is unchanged or higher** —
that is C2, and the governance gate proves it end to end — but the shape of the
log differs: `tool/call` counts drop, `tool/code-dispatch-start`/`tool/code-dispatch`
pairs appear, and a card that reported "1 tool call" now reports "1 cell, 12
governed calls".

Anyone diffing raw record counts between the two harnesses will see this and
should not read it as lost fidelity. The per-capability record is *finer* here,
not coarser; it moved from the model's call list to the dispatch log.

### `sdk-block` — expected, named in the row

Every pH request carries a generated SDK block describing the reachable
namespaces. Prime Agent's prompt had no equivalent — its single `ipython` tool
was described in prose. So a replayed request is strictly larger by that section,
and its content changes when a row registers a namespace. This is the diff the
plan predicted, and P3-09's gate ("SDK block lists every contributed namespace")
is what makes it deterministic rather than incidental.

### `access-default` — expected, **not observable in these fixtures**

pH gives a child `access="read"` unless it asks otherwise (E4), and records the
downgrade when no workspace tier can honour a `write`. Prime Agent had no such
concept. Neither fixture delegates, so neither shows it. The test asserts this
explicitly — if a future fixture spawns a child, the assertion fails and this
section needs rewriting rather than quietly becoming stale.

### `bashExecution` — expected, and a genuine capability gap

This is the one entry in the role table with no pH counterpart, and the test
asserts it is the *only* one.

`before-compaction.jsonl` carries three `bashExecution` records: a **user**
running a shell command in prime-agent's interactive UI, logged into the session
alongside the model's turns. pH has no counterpart. `ctx.commands` dispatches
without a model turn — that is the right mechanism — but no shipped command
shells out on the user's behalf, so nothing of the kind reaches the log.

Triage: this is a missing *feature*, not a porting error. It belongs with the
Phase 4 TUI work if it is wanted at all; the honest note is that a person running
`ls` in pH today does it outside the session, and the session does not know.

## What this exercise did *not* find

No tool prime-agent called is missing. No record type in either fixture is
unrepresentable in pH's log. The compaction case is present in
`before-compaction.jsonl` and maps onto `surface_op: replace` (I4) — the
mechanism pH uses for compaction, offload and rollback alike — so the one
structural operation both harnesses perform is performed the same way.

## Re-deriving this report

```
uv run pytest packages/ph-rlm/tests/test_fixture_replay.py -q
```

Skips cleanly without `sources/`. The numbers in the table come from
`fixture_replay.read_shape`, which is the same reduction the tests assert on.
