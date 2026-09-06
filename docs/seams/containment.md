# `ctx.containment` — which rung of the ladder this deployment asked for

**Module:** `ph/seams/containment.py` · **Row:** `containment` · **Consumers:**
`ctx.workspace`, `ph doctor`

§4.8's ladder is `advisory` → `worktree` → `sandbox`. Every rung is built; **none
of it is in force until something chooses**, and this is the choosing.

| rung | what it moves | what it does *not* bound |
|---|---|---|
| `advisory` | nothing — the agent works in the person's own checkout | anything |
| `worktree` | the cwd, to a checkout of its own | an absolute path |
| `sandbox` | the process's view of the filesystem | — (see [`ctx.sandbox`](sandbox.md)) |

## Two tiers, because the interesting default is not uniform

```yaml
- id: containment
  name: containment
  config:
    tier: advisory        # the *root* agent — the person's own session
    child_tier: worktree  # spawned children
    strict: false
```

A person running `rlm` is working in their own checkout and would be surprised to
find their agent editing a copy somewhere else — so the root stays `advisory`.
Its **children** are the fan-out hazard the tier exists for: eight of them
writing one tree concurrently is the case §4.8 opens with, so `child_tier`
defaults to `worktree`. One knob would have forced the wrong answer on one of
them.

`None` is **no opinion, not `advisory`**. Mounting this row must not opt a
profile out of a provider it deliberately layered; a profile says `advisory` when
it means "this agent stays in the person's checkout", which is a different
statement from never having chosen.

## Selection is per acquire, not per mounted row

The provider is registered whenever its row is layered. What `tier` decides is
whether a given caller *asks* for it.

That keeps one provider slot — two answers to "what runs this" is a contradiction
— while letting a parent and its children sit on different rungs, and it is why
`WorkspaceSeam.acquire` takes a tier rather than this module deciding which rows
exist.

## `strict` refuses to start, and refuses on `partial`

dsh's fail-closed `SANDBOX_UNAVAILABLE` posture, lifted from per-call to profile
start (Q10). An operator setting it is saying *"I do not want to run at all
unless confinement is real"* — so a `partial` backend is a **refusal**, not a
downgrade accepted quietly. A downgrade nobody notices is indistinguishable from
the thing they were trying to prevent.

The check runs **after** the profile is mounted rather than inside this row's
`apply`, because a backend may be layered after this row and a verdict computed
too early would refuse a deployment that is in fact confined.

## `TIERS` describes the bargain, and a provider may override its row

`TIERS` is keyed by rung, and the columns **belong to whoever occupies the rung,
not to its name**. Every provider at `worktree` inherited *"buys: collision
isolation and revertibility (fan-out safety, per-run checkpoints, /revert)"* —
true of a checkout and false of an overlay, which has no git tree to hash and
therefore never writes a restore point.

`ph doctor` was advertising a mechanism the mounted tier does not have, in the one
place a person looks to check exactly that — the single failure E1 exists to
prevent. So `DescribingProvider` is optional: a provider whose bargain *is* its
rung's says nothing and gets the stock row; one that differs describes itself,
and `tier` is on the Protocol so the override can be matched to the rung it
describes.

## The rule this seam exists to keep

**No tier is described as bounding writes it does not bound.** The tier table,
the permission row's validation, `ph doctor`, `/revert`'s output and the first-run
notice each carry their own caveat, because a caveat only in the docs is a defect
(§5 rule 6).

If you add a rung or a provider, the obligation is not to make it strong — it is
to make its description exactly as strong as it is.

## See also

[`ctx.sandbox`](sandbox.md) · [`ctx.workspace`](workspace.md) ·
`test_containment.py`, `test_containment_ladder.py` (which tests the *claims*,
not the mechanisms)
