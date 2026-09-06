# `ctx.workspace` — where an agent's writes land, and how honestly that is stated

**Module:** `ph/seams/workspace.py` · **Row:** `workspace-shared` (+ a tier row) ·
**Consumers:** the agent lifecycle, `ctx.fs`, `ctx.subprocess`, `/revert`,
`permissions-fs`

The seam the containment ladder hangs off (D21, §4.8). Its consumer is the
**agent lifecycle**, not a tool: an agent acquires a workspace, and `ctx.fs`'s
root and `ctx.subprocess`'s cwd resolve to `workspace.root` — which is what makes
a tier bound *authored* code rather than merely observe it.

## The one distinction everything else rests on

**`repo_writable` records which guarantee was obtained, never which was
requested.**

A caller asks for `access="write"` or `"read"`. That is a *request*. What comes
back is a `kind` and a `repo_writable`, and `repo_writable is False` **only when
a tier is enforcing it** — never as a statement of intent, never inferred from
the request. Wording anywhere — here, in `ph doctor`, in a config comment — that
blurs request and guarantee is a defect (§12 Q10).

The practical form: asking for `read` on a deployment with no confining tier gets
you `shared` and `repo_writable=True`, plus a logged notice. You asked; the
harness could not promise; it says so rather than pretending.

## Four invariants a consumer may rely on

* **There is always a workspace.** `acquire` never fails and never returns
  `None`. A provider that cannot serve a request *declines*, and the seam falls
  back to `shared` with a notice.
* **A workspace is an effect of the scope that took it** (I2). `acquire`
  registers teardown through `ctx.effect`, so a disposed agent scope unwinds it.
  With no `scope=`, a live agent named by `agent_id` owns it anyway; only an id
  the registry does not know falls back to the seam's default owner.
  That is the in-process half; the `workspace/acquired` + `workspace/disposed`
  pair is the crash half, reconciled at session open.
* **`scratch` is always present and always writable**, on every kind and every
  tier — and the *seam* creates it, one implementation rather than one per
  provider. It lives in pH's own state directory rather than inside the
  workspace, so it survives disposal as a session artifact.
* **Kind predicates are exhaustive `match`es, never membership tests**, so a
  seventh `WorkspaceKind` fails to type-check rather than silently classifying.

## The vocabulary

```text
WorkspaceKind = "shared" | "worktree" | "worktree-ephemeral"
              | "readonly-scratch" | "overlay" | "overlay-ephemeral"
WorkspaceAccess = "write" | "read"        # a request, not a guarantee
```

A `Workspace` is a value — what the provider decided, and nothing about who
holds it:

| field | |
|---|---|
| `root` | the agent's cwd; `ctx.fs`'s root and `ctx.subprocess`'s default cwd |
| `scratch` | always writable, outside `root` on purpose |
| `kind` | what the agent actually got |
| `repo_writable` | whether `root` can actually be written |
| `ref` | the git branch, when the kind has one |
| `env` | environment a runner should apply |

`env` is best-effort by construction: the read-only kinds point `TMPDIR`,
`PYTEST_ADDOPTS` and friends inside `scratch`, because build tools write into the
tree they are run against. A toolchain that insists on writing beside its sources
will still fail, and the answer is `access="write"` for that agent — **not a
weaker tier**.

## Using it

```text
await ctx.workspace.acquire(agent_id, ..., access="write")   # -> Workspace
ctx.workspace.of(agent_id)                                   # -> Workspace | None
await ctx.workspace.dispose(agent_id)
ctx.workspace.live()                                         # -> list[Workspace]
ctx.workspace.retain(agent_id, reason)                       # keep it as evidence
```

Most code does not call `acquire` — `workspace-lifecycle` does it on the agent's
behalf and wires the result into `ctx.fs.rebase`. What ordinary code wants is
`workspace_of(...)` to ask where an agent's writes are going.

## Providing a tier

```python
@runtime_checkable
class WorkspaceProvider(Protocol):
    tier: ContainmentTier            # which rung this provider occupies

    async def acquire(
        self,
        *,
        session_id: str,
        agent_id: str,
        base: Path,
        scratch: Path,
        access: WorkspaceAccess = "write",
    ) -> Workspace | None: ...       # None = decline
```

**One required method.** Register with
`ctx.workspace.register_provider(provider)`, and **returning `None` is how you
decline** — a host with no `git`, a request for a kind you do not serve — after
which the seam falls back rather than failing the agent.

Everything else is an **optional capability, declared as its own Protocol rather
than probed with `getattr`**:

| Protocol | method | why optional |
|---|---|---|
| `ReclaimingProvider` | `reclaim(record) -> bool` | an in-memory tier has nothing to release (F6) |
| `ExportingProvider` | `export(record) -> str` | `shared` has nothing to export; a discarding tier has nothing to offer |
| `CheckpointingProvider` | `capture(workspace) -> str \| None`, `restore(workspace, token)` | `shared` is the person's own checkout; an overlay's delta is a restore point nothing reads yet (P4-09) |
| `ArtifactProvider` | `refs(base)`, `strays(base)`, `discard(path)`, `delete_ref(base, ref, force=)`, `merge(base, ref)` | `shared` leaves no artifact to manage; five verbs because they are `/workspaces`' whole surface (E15) |
| `DescribingProvider` | (see [containment](containment.md)) | only when the bargain differs from its rung's |

The reason they are Protocols and not probes is worth copying into any seam you
write: **a `getattr` probe reports a provider whose method is *misnamed* as one
that cannot do the thing** — which hides a leak instead of closing it, and in
this package that failure mode already cost a day.

`reclaim` returns whether anything was **kept**, matching `Workspace.release`, so
the `workspace/disposed` a reconciliation writes says what an orderly one would.
`export` returns the git ref, so one verb serves both isolating tiers — a
worktree answers with the branch it has been committing to, an overlay builds one
out of its delta first, and `/workspaces` asks the seam rather than asking which
tier it is talking to.

`capture` returns **one opaque token serving two readers**: P4-09 stores it as a
restore point and P5-07 uses it as a *fingerprint*, so a quality gate that failed
against this exact state need not run again. One derivation, so a gate memo and a
`/revert` can never disagree about whether the work changed. What a token must
promise is narrow and exact — **equal tokens mean equal content** — and the
converse is not required: a tier whose token moves while the content stands still
only makes a gate re-run. The git tier's token is a tree hash, pinned under
`refs/<branch>/pre-run/<tree>` so `gc` cannot take it; the jj tier's is the
working-copy commit, kept reachable by the operation log with nothing to pin.

`ArtifactProvider` is `/workspaces`' whole surface, which is why it is five verbs
in one Protocol rather than several: a tier serves that command or it does not, and
a half-implemented one hands a person rows whose verbs then fail. The command used
to do all of it itself, in git — `git branch --list ph/*`, `git worktree list
--porcelain`, `git merge`, `git branch -d` — and that was wrong twice over.

**A jj workspace is not a git worktree**, so a stray one had no path, no `dirty`
and no verb that could remove it, while a live one read as unheld and lost the
refusal that protects it. And **jj embeds its own git**: the whole acquire,
bookmark and export flow runs against a `git` on `PATH` that exits 127, so a jj
deployment need not have the binary — and there the branch listing returned nothing
while `/workspaces` reported that nothing had been left behind. Silent, and the
worst kind of wrong.

Two guards are the tier's and are the safety of the verbs. `strays` filters by the
provider's own **root**: `jj workspace list` reports the person's own `default`
workspace at the repository root, and a `remove` that trusted it would delete their
repository. `refs` filters **nothing** — `BRANCH_PREFIX` is applied by the command,
because that prefix is what keeps it from offering to delete somebody's
`feature/x`, and a guard belongs with the verb it guards.

Every verb that acts returns `""` for "it happened" and otherwise the sentence a
person reads, rather than a bool. A conflicted merge is what proves that shape:
git exits non-zero and refuses, jj exits **zero**, records the conflict inside the
commit and writes markers into the file — so a caller reading an exit code reports
a clean merge the person discovers by opening the file.

Whether a workspace can have a restore point is `WorkspaceSeam.can_checkpoint`,
and it is two halves: the **kind** (never `shared` — that root is the person's own
checkout) and the **tier** (does the mounted provider implement the capability).
It replaced a kind-keyed predicate, which was a table of provider facts kept
somewhere no provider could see.

Shipped providers: `workspace-shared` (the floor, always available),
`workspace-git-worktree` (the `worktree` tier), `workspace-jj` (the same tier over
Jujutsu), `workspace-agentfs` (a copy-on-write overlay),
`workspace-readonly-scratch` (the `sandbox` rung's kind).

`workspace-jj` and `workspace-git-worktree` occupy one rung and differ in what a
child inherits, and both offer restore points. A git worktree branches from the parent's last **commit**, and a
live parent commits only at disposal — so for the whole of a session a child
starts from context the parent has moved past. jj's working copy *is* a commit, so
the same spawn starts from the parent's work in progress. Colocation keeps the
result a real git branch, which is what makes this a provider swap rather than a
migration. It never converts a repository: a base jj does not already manage is
declined, and the binary being absent declines the row at mount.

The mirror image is worth stating, because it is the same property with the sign
flipped: **a file sitting untracked in the base reaches a child's branch** under
jj and does not under git. jj put it in the base's working-copy commit before pH
looked, and children fork from that commit. What pH does keep out is the material
it copied in itself — a provisioned `.env` is not the agent's work, and jj is told
so through `snapshot.auto-track`.

The seam keeps the bookkeeping — which agent, which session, how to end it — so a
provider cannot half-implement the lifecycle.

## Retention, reconciliation and collection

A settled child's tree is what a parent needs to diagnose a failed run, so
`retain(agent_id, reason)` keeps one past disposal and records
`workspace/retained`. What that buys is evidence; what it sells is an unbounded
pile of checkouts. `ph workspaces gc` closes that trade — and **never
automatically**, on `ph attachments gc`'s precedent: sweeping retained trees at
startup would delete last night's failure exactly as somebody sat down to read it.

`reconcile(session)` runs at session open and is the crash half of cleanup: a
tree whose `workspace/acquired` has no `workspace/disposed` belonged to a process
that is gone.

## Events

| event | |
|---|---|
| `workspace/acquired` | an agent took one — with the kind it actually got |
| `workspace/disposed` | it ended; `kept` and `retained` say what survived |
| `workspace/retained` | kept deliberately, with a reason |
| `workspace/provisioned` | materials were placed in a fresh tree (E14) |

## What it does not enforce

* **It is not confinement.** A `worktree` bounds a *relative* write by moving the
  cwd; it does nothing about an absolute path, and nothing about a process that
  chooses to `chdir`. `ctx.sandbox` is the seam that makes an enforcement claim,
  and [`ctx.containment`](containment.md) is where the ladder's honesty lives.
* **It does not undo.** `/revert` restores a checkpointed tree; a tool that
  published a package or sent mail is listed as *not* covered, read from each
  tool's own `effects_confined_to_workspace` declaration.

## See also

[`ctx.containment`](containment.md) · [`ctx.sandbox`](sandbox.md) ·
[`ctx.fs`](fs.md) · `test_workspace.py`, `test_workspace_lifecycle.py`,
`test_containment_ladder.py`
