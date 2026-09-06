# `ctx.skills` — the capability layer, and the boundary it must not cross

**Module:** `ph/seams/skills.py` · **Rows:** `skills`, `skills-progressive` ·
**Consumers:** the system prompt, `subagent-presets`, `tool-todo` (`ph-stabilize`)

A skill is a package a **distribution or a user** installed.

## The boundary (I7)

**The model cannot install one, and `/refine` cannot mint one.**

That is invariant I7, and it is the reason skills and the Continual Harness share
a word but not a mechanism: the knowledge layer writes **procedure**, never
**capability** (Q13). A harness where a model could add to its own capability
layer has no capability layer.

**Nothing is scanned by default.** A deployment names its directories. A
well-known path scanned at every start would make "install a skill" mean "drop a
file in a directory", which is precisely the boundary I7 draws.

## Progressive disclosure (G9)

The **catalog** goes in the prompt; the **body** stays on disk until the model
asks for it by name. `skills-progressive` is that row, and it lives in the same
module because "what a skill is" — the format, the limits, the registry, and what
the model is told about it — is one question.

**The catalog is a provider, not a fixed string.** It reads the registry at
assembly time, so a skill registered by *another* row appears in this catalog
rather than in a second one of its own, and no row has to be mounted before
another.

The cost is that registering a skill mid-session moves the cached prefix (A12) —
which is honest: the model was genuinely told something new.

## What a skill carries

| field | |
|---|---|
| `name`, `description` | the catalog entry |
| `path`, `source`, `version` | where it came from |
| `argument_hint` | the author's one line for the common case |
| `parameters` | declared inputs, validated before the body is rendered |
| `steps` | the playbook, seeded into the todo list |
| `allowed_tools` | what this skill may reach |
| `hint` | |

`parameters` and `steps` are what make a skill a *playbook* rather than a
document: a body may interpolate `{{parameters.x}}`, and `rendered_skill` returns
both the instructions and the steps with placeholders **already substituted** —
because an author writing `Run {{parameters.gate}}` in a step otherwise got the
literal placeholder in the todo list while the instructions got the value, which
is one procedure spelled two ways in the one place the model cannot go back and
check.

## The surface

```text
ctx.skills.register(skill, *, scope=None)    -> Disposer
ctx.skills.restrict(SkillRestriction(allow=..., deny=...), *, scope=...)
ctx.skills.list(scope=...)      ctx.skills.get(name, scope=...)
ctx.skills.reach(scope=...)     ctx.skills.admits(name, scope=...)
await ctx.skills.body(name)     # read on demand — the G9 half
```

`reach` is memoized, and `stale_reach` is what a runtime invariant checks: a
memoized reach must equal a freshly built one (P6-01, I6), because a cache that
drifts here silently widens or narrows what an agent may use.

Restrictions intersect and never widen — the same `NameFilter` rule
[`ctx.tools`](tools.md) uses, and for the same reason: a child's grant is
computed against its parent's, so an allow-list can only ever remove.

## Providing skills

Register into the seam; there is no provider Protocol to satisfy. Two shipped
sources:

* the built-in loader, reading `<dir>/<name>/SKILL.md` with validated
  frontmatter, from directories a deployment **names**;
* `rlm-skills-python`, which contributes skills from a Python package.

A row that discovers skills should register them at `apply` time and let the
catalog provider pick them up — do not build a second catalog.

## What it does not do

* It does not let anything install a skill at runtime (I7).
* It does not execute. A skill is instructions plus a declared reach; the tools
  it names are the ones that act.
* It does not enforce `allowed_tools` by itself — that is a restriction the
  mounting row applies, which is why `reach` is the value invariants check.

## See also

[`ctx.tools`](tools.md) · [`ctx.subagents`](subagents.md) (presets bind a name to
skills) · `docs/skills/README.md` for authoring · `test_skills_progressive.py`,
`test_skill_steps.py` (ph-stabilize)
