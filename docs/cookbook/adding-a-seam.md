# Adding a seam

A seam is a new *kind* of capability, with the implementation left open. Add one
when the honest answer to "how should this work" is **different in different
deployments** — not merely configurable, but genuinely a different mechanism.

If there is one right answer, write a plugin instead. A seam with exactly one
possible provider is indirection that has to be read through for ever.

## Definition, provider, consumer (invariant I5)

Three parts, and they are separable on purpose:

* the **definition** — a service key, the Protocol a provider satisfies, and the
  events it declares. This is what `ph-base` mounts.
* a **provider** — whatever implements it. May live in another package, may be
  absent.
* the **consumers** — everything that calls `ctx.<key>`, none of which learns
  which provider answered.

`ph-base` mounts several seams with no provider at all — `ctx.subagents`,
`ctx.code_runtime`, `ctx.workspace` — and that is the design working. A deployment
that layers `rlm` gets a provider; one that does not has the seam refuse cleanly
rather than pretend.

## The definition

```python
@runtime_checkable
class Uploader(Protocol):
    async def upload(self, ref: AttachmentRef, content: bytes) -> FileHandle: ...


@dataclass(slots=True)
class UploadRegistry:
    """The service published as `ctx.uploads`."""
    ctx: Context
    ...


@plugin("uploads-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    ctx.provide("uploads", UploadRegistry(ctx=ctx, root=root))
```

`ctx.provide(key, service)` claims `ctx.<key>` in the provisioning realm and
returns a disposer the calling scope owns. A second claim of the same key in the
same realm raises `ServiceConflictError` naming who holds it — two answers to one
question is a contradiction, not a merge.

**Type the provider Protocol.** A backend whose method drifted would otherwise
fail at runtime inside somebody's `except` and be reported as "no provider".

## Taking registrations

Providers and contributions arrive through the seam, never by assignment. Three
claim helpers in `ph.seams._registry`, and the choice is a design statement:

| helper | shape | use when |
|---|---|---|
| `claim_slot(by, holder, attr, value)` | one | two answers is a contradiction — the runtime, the rebase resolver |
| `claim_key(owner, table, key, value)` | one per key | per-provider — an uploader per provider name |
| `claim_entry(owner, entries, value)` | a list | many contributions — walk screens, diagnostics |

`claim_entry` removes **by identity**, because frozen dataclasses compare by value
and two equal registrations would otherwise unregister the wrong one.

`claim_slot` takes the `Running` pair rather than a context, and holds it in
`<attr>_by`: a provider is a body the seam invokes *later*, so to enter the right
binding then it must have kept who registered it (P6-29). Invoke it with
`with running(entry.by):`.

## Declaring events

```python
events.declare(
    "uploads/stored", "emit",
    owner="ph.seams.uploads",
    doc="Bytes reached a provider's file API.",
)
```

The mode is fixed at declaration — `emit`, `waterfall`, `parallel`, `serial` —
and `ctx.<mode>` raises on a mismatch. `ph events --profile <name>` prints the
producer/consumer matrix, which is why `owner` and `doc` are worth writing.

Name events for the *fact*, not for who consumes them. `ctx.subagents` emits
`subagent/admitted` and `ph-app` reads it; had it been named for the reader, the
seam would depend on its consumer.

## Failing closed

The rule (§5.5) is that a seam refuses rather than degrades silently:

* no provider → say so, in the caller's own vocabulary, and let the caller
  decide. `handle_for` answers `None` when a provider has no uploader, because
  the caller's alternative — sending bytes inline — is what every route did
  before that seam existed.
* an unavailable *policy* answer denies. `containment.strict` on a host with no
  backend refuses to start rather than running unconfined.

Where you cannot fail closed, state it next to where a reader would assume
otherwise. `ctx.uploads` says in its own docstring that nothing prunes its cache
automatically and names the command that does — a caveat that lived only in the
docs would be a defect.

## Scope

Every registration method should take `scope=` and mean it. A registration scoped
to an agent reaches that agent and unwinds with it; one with no scope reaches the
whole process. `FsService.screen` makes `scope` **required** for exactly this
reason — omitting it there would not mean "clean up with the mount", it would mean
"screen every agent": the widest policy in the seam, chosen by forgetting.

## What to write down

A seam's module docstring is its specification, and the tree's are long on
purpose. Say what the seam is *not*, and why: `ctx.attachments` opens by
explaining why it is not `ctx.spill_store` despite identical mechanics, because
the lifecycles differ in the one way that matters. That paragraph is what stops
the next person merging them.

## Checklist

- [ ] there is genuinely more than one possible provider
- [ ] the Protocol is typed and `runtime_checkable` where it is checked
- [ ] registrations go through a claim helper, with a `label`
- [ ] provider bodies are invoked under `with running(entry.by)`
- [ ] events declared with `owner` and `doc`
- [ ] the no-provider path is defined, and refuses rather than pretends
- [ ] `scope=` on every registration, required where the wide default would be wrong
- [ ] a page in [`docs/seams/`](../seams/)
