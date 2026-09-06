# pH documentation

**Status:** P6-10 is complete — the cookbook is written and every seam has a
page. Both are pinned by `test_docs_seams.py`, so a new seam without an entry, or
a page naming a method that no longer exists, fails the suite.

## Where things are

| | |
|---|---|
| [Cookbook](cookbook/) | How to extend pH: a plugin, a tool, an adapter, a seam. Start here. |
| [Seams](seams/) | Reference for all 29 service keys — what each publishes, who may provide it, what it refuses. |
| [Skills](skills/) | Authoring skills and playbooks (P7-18). |
| [Dev notes](dev-notes/) | Per-phase records and design notes, including measurements and things that were tried and dropped. |

The *specification* is not here. [`DESIGN.md`](../DESIGN.md) says what pH is;
[`plans/`](../plans/) says why each decision fell where it did and what remains.
Where this documentation and those disagree, they win and this is a bug.

## Two things worth knowing before reading anything else

**Everything is a row.** There is no core with plugins bolted on: the agent loop,
the tool registry, the session store and the filesystem are all rows in a profile,
mounted by the same loader in file order. A capability you add is not a lesser
citizen than one that ships.

**A seam is three parts, and they are separable** (invariant I5) — the
*definition* (a service key and its Protocol), a *provider* (whatever implements
it), and the *consumers*. `ph-base` mounts several seams with no provider at all,
because "run a child agent" or "run this program" have genuinely different answers
in different deployments, and a harness that shipped one answer would have made
the choice for you.

Both facts have the same consequence for extending pH: the question is almost
never "where do I patch this" but "which of the three am I writing".
