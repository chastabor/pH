# `ctx.credentials` — references travel, values do not

**Module:** `ph/seams/credentials.py` · **Row:** `credentials-env` ·
**Consumers:** every LLM adapter, at the edge and nowhere else

## The rule

> **Nothing above the adapter edge ever holds a secret value.**

A consumer asks for a `CredentialRef`, which is a **name**. Only the adapter
about to build an HTTP request resolves it, and only into a local variable that
goes out of scope with the request.

```python
# in a row's config: the *name* of a variable, never an interpolation
api_key_env: str = "ANTHROPIC_API_KEY"

# at the edge, and nowhere above it
secret = resolve_secret(ctx, config.api_key_env, config.provider)
```

That is what makes the guarantee **checkable rather than aspirational**: a
planted `FOO_API_KEY` must not appear in any event, any fd-3 frame, or any
child's environment — and a test asserts exactly that over a whole run.

A design that passed values around would need every future plugin author to be
careful. This one needs the adapter edge to be.

## `__repr__` is overridden

A secret that reaches a log via an **exception traceback** has still leaked. The
resolved value hides itself in `repr`, so a `raise` that formats its locals does
not undo the rest.

## The surface

```text
ctx.credentials.reference(name)      # -> CredentialRef; a name, not a value
ctx.credentials.resolve(ref)         # -> the secret, at the edge only
ctx.credentials.has(name)            # is it available, without reading it
ctx.credentials.require(name)        # resolve or refuse
ctx.credentials.provide_value(...)   # for a test, or a non-env source
```

A `CredentialRef` carries `name`, `source` and `description` — enough for
`ph doctor` to say *which variable* is missing without ever reading one.

## Failing

A missing credential is refused with the variable **named**:

```text
ANTHROPIC_API_KEY is not set, so provider "anthropic" cannot be called
```

The operator's next action is to set that variable, so the message is the
variable. `MISSING_CREDENTIAL` and `NO_CREDENTIALS` are distinct codes because
"the seam is not mounted" and "the value is absent" have different fixes.

## Providing another source

`credentials-env` reads the environment. A deployment with a secret manager
replaces the row: what the rest of pH depends on is the *reference* shape, not
where a value comes from. Nothing above the edge changes.

## What it does not do

* It does not cache. A resolve reads at the moment it is needed, so a rotated
  secret takes effect on the next request rather than the next restart.
* It does not pass values to children. [`ctx.subprocess`](subprocess.md) scrubs
  `*KEY*`, `*SECRET*`, `*TOKEN*`, `*PASSWORD*` from every child (I-4).
* It does not know what a credential is *for*. A row names the variable it needs.

## See also

[`ctx.subprocess`](subprocess.md) · [Adding an
adapter](../cookbook/adding-an-adapter.md) · `test_seams.py`,
`test_daemon_shell.py`
