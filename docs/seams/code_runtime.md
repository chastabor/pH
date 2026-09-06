# `ctx.code_runtime` — the seam definition only

**Module:** `ph/seams/code_runtime.py` · **Row:** `code-runtime` (definition only)
· **Provider:** `code-runtime-python` (`ph-rlm`) · **Consumers:** Code Mode's
`run_code` transport

What ships is the **contract**, and one assertion inside it that is easy to miss
and expensive to omit.

## The persistence obligation, checked at registration (D17)

> A provider declaring `persistence: "namespace"` **must** emit `kernel/snapshot`
> events — and that is checked when it registers, not the first time somebody
> forks a session and discovers the state was never durable.

dsh withheld a persistent Python REPL precisely because *"cross-call state would
be invisible to the log"*. pH admits one only from a provider that has promised,
at registration, to keep it visible:

```python
ctx.code_runtime.register(provider)     # raises PersistenceObligationError
```

A promise checked at runtime would be discovered by the person who lost work.

## The contract

```python
@runtime_checkable
class CodeRuntime(Protocol):
    language: str
    isolation: Isolation          # in-process | thread | process | sandbox | remote
    persistence: Persistence      # none | namespace

    async def run(self, request: CodeRunRequest) -> CodeRunResult: ...
```

`CodeRunRequest` carries `program`, `bindings`, `namespace` (`None` keeps the
fresh-per-run contract; a key selects a persistent one) and `cancel_scope`.

`CodeRunResult` carries `logs`, `value`, `error`, `truncated`, `reset` and
`displays`. Two of those are worth knowing:

* **`reset`** — the runtime died since the last run, so this one got a fresh,
  empty namespace. The notice in `logs` tells the *model*; this flag tells the
  card, because a fact recovered from prose would be forged by any program whose
  first output is the marker text.
* **`displays`** — rich payloads for a front end, carried separately because
  `logs` is for the model and a base64 PNG in the model's text costs a fortune
  and says nothing.

## Binding names are validated here

One `bindings` list has to be valid against **every** backend regardless of
`language`, so `RESERVED_BINDING_NAMES` is `keyword.kwlist | _TYPESCRIPT_RESERVED`
and every namespace and binding is checked at construction.

A name legal in Python and reserved in TypeScript would make a binding set
silently backend-specific — so a tool named `class` or `function` is refused now,
long before anything could run it.

## One provider, and one renderer per language

`ctx.code_runtime` holds **at most one provider**: two answers to "what runs this
program" is a contradiction, and a profile picks its tier (D16).

`register_sdk_renderer(language, renderer)` claims the `tools:sdk` prompt
renderer for one language. A runtime whose language has no renderer **fails
prompt assembly** rather than shipping a listing in the wrong syntax — a model
given the wrong one writes code that cannot run, and the failure looks like a
model problem.

`render_typescript_sdk` ships and is currently unselectable, since every runtime
pH ships is Python. See
[`dev-notes/typescript-code-runtime.md`](../dev-notes/typescript-code-runtime.md)
for what would make it live.

## What it does not do

* It does not govern the bindings — `DispatchBridge` does, re-entering
  [`ctx.tools`](tools.md)'s pipeline per `await` (C1).
* It does not confine. `isolation` is a *descriptor* a consumer branches on, not
  a claim this seam enforces; [`ctx.sandbox`](sandbox.md) makes enforcement
  claims.
* It ships no provider in `ph-base`.

## See also

[`ctx.tools`](tools.md) · [`ctx.sandbox`](sandbox.md) · `test_code_mode.py`,
`test_conformance.py` (ph-rlm)
