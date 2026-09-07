# A TypeScript code runtime — what exists, and what it would take (D16, P6-08)

**Status:** not built, and half-prepared for. `render_typescript_sdk` ships and is
registered on every mount; no runtime can select it. This note says why, what the
codebase has already paid toward it, and what is actually left — written because
"the TypeScript renderer" reads like a live feature in the source and is not one.

## What is true today

`ph.tools.sdk` has two renderers and `ph.tools.code_mode`'s `apply` registers
both:

```python
ctx.code_runtime.register_sdk_renderer("python", render_python_sdk)
ctx.code_runtime.register_sdk_renderer("typescript", render_typescript_sdk)
```

The one a model sees is chosen by the **runtime's** language, not by anything
about the schema:

```python
language = getattr(runtime, "language", "python")
renderer = ctx.code_runtime.sdk_renderer(language)
```

Every runtime pH ships is Python — `ph_rlm.kernel.manager` declares
`language: ClassVar[str] = "python"`, and `ph.testing.stub_runtime` the same — so
`render_typescript_sdk` is registered under a key nothing ever asks for. It is
reachable from tests and from nowhere else.

**It is not about the schema's language.** A binding's `parameters` is plain JSON
Schema, and each renderer translates the same declaration into its own syntax:
`{"type": "integer"}` is `limit: int = ...` in one and `limit?: number` in the
other. A Python program is never shown TypeScript.

## What is already paid for

Four things, which is why this is a smaller job than it looks:

* **Binding names are already portable.** `RESERVED_BINDING_NAMES` is
  `keyword.kwlist | _TYPESCRIPT_RESERVED`, and `validate_binding_name` runs on
  every namespace and binding at construction. Its docstring gives the reason —
  "one binding list must be valid against every backend, so a name reserved
  anywhere is reserved everywhere" — so a tool named `class` or `function` was
  refused years before anything could run it. A runtime arriving later does not
  discover that the existing tool surface is a syntax error.
* **The seam already selects, and fails loud.** A runtime whose language has no
  renderer raises at prompt assembly rather than emitting a listing in the wrong
  syntax, because a model handed the wrong one writes code that cannot run and
  the failure looks like a model problem.
* **The renderer itself is written and tested** (`test_sdk.py`), including the
  shapes that differ from Python: `declare const`, `?` for optional, the doc
  comment above the member, arguments as one object, and `unknown[]` rather than
  `any[]`.
* **`Isolation` already admits the tiers a JS runtime would want** —
  `in-process | thread | process | sandbox | remote`.

## What is actually needed

### 1. A provider with `language = "typescript"`

The `CodeRuntime` Protocol is four members:

```python
language: str
isolation: Isolation
persistence: Persistence


async def run(self, request: CodeRunRequest) -> CodeRunResult: ...
```

Registered with `ctx.code_runtime.register(provider)`. `CodeRunRequest` carries
`program`, `bindings`, `namespace` and `cancel_scope`; `CodeRunResult` carries
`logs`, `value`, `error`, `truncated`, `reset` and `displays`.

### 2. A host↔guest call bridge — the real work

This is the part the renderer does not imply. `CodeBinding.dispatch` is a
**Python callable** that re-enters the tool pipeline as a governed sub-call (C1).
An in-process Python runtime can call it directly; the shipped process runtime
carries `call`/`reply` frames over fd 3. A QuickJS or Node runtime can do
neither, so it needs its own transport, and that transport has to preserve what
`DispatchBridge` enforces:

* every `await` is individually governed, recorded, and refusable;
* a refusal fails the **whole program** (C3), rather than being catchable by the
  model's own code;
* `max_dispatches` and `max_spawns` bound one run (C4);
* arguments cross as lossless JSON — the guest is hostile and every inbound
  value is rebuilt, never trusted.

Deciding whether to reuse the fd-3 protocol or define a second one is the first
design question, and it is the one this note cannot answer from the Python side.

### 3. A persistence decision, which the seam enforces

`persistence: "namespace"` requires `declares_kernel_snapshots = True`, and
`register()` raises `PersistenceObligationError` otherwise — cross-call state
invisible to the log is the reason a persistent runtime was withheld in the first
place (D17). Two honest options:

* `persistence = "none"` — fresh context per run, no snapshot obligation, and the
  simplest thing that can ship;
* `persistence = "namespace"` — then a JS heap has to be serialised into
  `kernel/snapshot` events the way `ph_rlm.snapshot` does for `dill`, which is a
  second substantial piece of work and should not be smuggled in with the first.

### 4. An acceptance gate that does not exist yet

P6-08's gate reads "conformance suite passes against it." **That suite is not
language-neutral.** `packages/ph-rlm/tests/test_conformance.py` enumerates
`ph_runtime.protocol.FRAME_FIELDS` and `ph_rlm.snapshot.RESERVED_KINDS` — it is a
conformance suite for the *fd-3 protocol*, not for `CodeRuntime`. So a
TypeScript runtime needs either:

* to speak that same protocol, in which case the suite applies as written; or
* a new suite over the seam's own contract — `run()` against the `CodeRunResult`
  fields, the binding bridge's governance properties, and the SDK block the
  renderer produces for the mounted namespaces.

Whichever is chosen, saying so is part of the row: the gate as written is
currently a pointer to a test that would not run.

## What must not regress

* **Prefix stability (A12).** The SDK block is prompt text. A renderer whose
  output changes between requests moves the cached prefix for every turn after
  it, which is why the seam holds exactly one renderer per language and an absent
  one is a loud failure rather than a silent default.
* **C6.** The transport stays the only callable name; the listing is not a set of
  native tools. `code_only_rule(transport)` is rendered from the *profile's* name
  for it, so a runtime must not hard-code `run_code`.
* **One runtime at a time.** `ctx.code_runtime` holds a single provider, by
  design: "two answers to what runs this program is a contradiction, and a
  profile picks its tier (D16)." A TypeScript runtime is a profile choice, not an
  addition alongside Python.

## Why the renderer stays in the meantime

It is registered on every mount, so it is shipped code rather than something
behind a flag, and the seam's fail-loud lookup only pays off if the renderer it
finds is correct. Deleting it would make P6-08 re-derive it; keeping it costs one
module and one test file. What it must not do is read as evidence that TypeScript
Code Mode works — hence this note.
