# Adding a plugin

A plugin is the unit everything else ships in. This page is the whole of it; the
other three recipes are about *what* you register once you have one.

## The smallest thing that works

```python
from ph.cordis import Context, plugin
from ph.wire import WireModel


class Config(WireModel):
    greeting: str = "hello"


@plugin("greeter", affects="environment", inject=["tools"], config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Whatever this row contributes, it contributes by registering."""
    ...
```

Four parts, and each has a consequence:

* **`"greeter"`** is the plugin's name — what a profile row's `name:` refers to.
* **`affects`** says which configuration owns the row's settings: `environment`
  (a session profile), `presentation` (the TUI's) or `deployment` (the daemon's).
  It has no default, so a row that does not say is caught where it is written.
* **`inject`** lists service keys this row needs. It gates activation: with
  `ctx.tools` absent from the profile, `apply` never runs. Ask for what you
  actually use, and nothing else — an over-broad `inject` makes your row silently
  absent in a deployment that would have been fine.
* **`config`** is a pydantic model. The row's `config:` block is validated into
  it before `apply` sees it, so the body never hand-checks its input. A row with
  no `config:` block gets the model's defaults, never `None`.

## Five ways to be a plugin

`normalize_plugin` is duck-typed on purpose: a third party should be able to
contribute a row without importing pH's decorator. All five are read by
`ph.cordis.plugin`, and `test_cordis_plugin.py` pins each:

| shape | how it is read |
|---|---|
| a decorated function | its stamped `PluginSpec` |
| a module with a decorated `apply` | the spec on `apply` — the entry-point case |
| an object with `name`/`apply`/`inject`/`Config` | those attributes; no pH import needed |
| a bare callable | named from `__name__` |
| a `PluginSpec` | itself, unchanged |

A `Config` attribute that is not a pydantic model is refused *at normalization*,
so the mistake is named as the plugin's rather than surfacing later as a
validation error against somebody's row.

## Getting it into a profile

Declare the entry point, in your package's `pyproject.toml`:

```toml
[project.entry-points."ph.plugins"]
greeter = "my_package.greeter:apply"
```

Then add a row. A row is `id` (unique in the profile), `name` (the plugin), and
optional `config`:

```yaml
- id: greeter
  name: greeter
  config:
    greeting: good morning
```

`id` and `name` are separate because one plugin may be mounted twice under two
ids with two configs — which is exactly how `llm-openai-compatible` serves
several routes.

Layer it onto a shipped profile rather than editing one:

```bash
phern --profile headless --patch '{insert: [{id: greeter, name: greeter}]}'
phern --dump-config --profile headless      # what actually composed, in order
phern config                                # every row's config model, with prose
```

`--patch` is the CLI layer, applied last. `{id: x, disabled: true}` removes a row,
`{id: x, config: {...}}` replaces its config.

## Disposal: you almost never write any

Every registration returns a disposer already owned by the calling scope. What
needs care is *artifacts* — anything outside the process:

```python
async def apply(ctx: Context, config: Config) -> None:
    path = await ctx.effect(_make_scratch, label="scratch")
```

`ctx.effect(enter)` calls `enter()`, which returns the release, and registers it
in one step — so a failure between acquiring and registering cannot leak. Scopes
dispose LIFO, children first. A disposer that raises is logged and the unwind
continues: one bad teardown must not strand every artifact registered before it.

## Who is running matters

Registrations record *who* made them (P6-12, P6-29). A body the seam invokes
later — a provider, a listener, a tool's `execute` — runs re-bound to the row that
contributed it, not to whatever happened to be running when it was called. You
get this for free by registering through the seam's own method; you lose it by
stashing a callable somewhere and invoking it yourself.

The practical consequence: a scoped registration (`scope=agent.ctx`) reaches only
that agent, and unwinds with it. Passing no scope means the mount's own, which is
the whole process. Those are different policies, and choosing by omission is how
a per-agent rule becomes a global one.

## Refusing to mount

A row that cannot honor its configuration refuses at `apply`, and it refuses with
`MountRefusal` (`ph.cordis`) rather than a bare exception. Every command that
mounts a profile turns that one type into a sentence and an exit code, and leaves
anything else as the traceback a bug deserves. `containment.strict` on a host with
no sandbox backend and the OTel exporter without its extra are the two shipped
examples (E8). Refuse at mount, not at first use: by then the agent is running and
"refuse to start" has already been disobeyed.

## When a child holds less of your row

A child runs on its parent's mount, so a profile its parent assigns it
(`profile="reviewer"`) is read against the parent's rows rather than mounted. A
row it drops takes the tools and skills that row registered. What a row it keeps
says is the parent's, unless the plugin declares how a child holds less of it:

```python
from ph.cordis import ChildLimit, ChildReach, NarrowingRefused


def narrows(mounted: Config, asked: Config, reach: ChildReach) -> ChildLimit:
    """The parent's row as it runs, the child's as its profile says it."""
    if asked.greeting != mounted.greeting:
        raise NarrowingRefused("its greeter greets differently, which a child cannot")
    return ChildLimit()


@plugin("greeter", affects="environment", config=Config, narrows=narrows)
async def apply(ctx: Context, config: Config) -> None: ...
```

`ChildLimit` is what the child is held to: a model key its parent lists, read-only,
skills withheld, or fewer writable directories. A limit of another kind is a field
there and on the child's grant, not something a narrower can say alone. `NarrowingRefused` refuses the
spawn, and names the row; a narrower refuses anything wider than the parent. The
narrower is typed against the same `config` model, so mypy holds the two together.
`models`, `sandbox-policy`, `sandbox-allow` and `skills-progressive` are the shipped
examples.

## Checklist

- [ ] `inject` lists what you use, and nothing more
- [ ] a deliberate refusal raises `MountRefusal`, so `phern -p` prints a sentence
- [ ] config is a `WireModel` with per-field docstrings (`phern config` prints them)
- [ ] artifacts acquired through `ctx.effect`
- [ ] the row is in a profile, and `--dump-config` shows it where you expect
- [ ] a test that mounts a profile, not one that calls `apply` directly
