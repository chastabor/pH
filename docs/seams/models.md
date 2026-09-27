# `ctx.models` — the models a profile lists, and which one a root runs on

**Module:** `ph/seams/models.py` · **Row:** `models` (environment) ·
**Consumers:** the daemon's supervisor, `phern -p` and `--mode rpc`, the TUI's `/model`

A route is a provider and a model. It used to be two free strings — the daemon's
`--provider`/`--model` for every root — so which models a session could reach was
nowhere a profile could say. The row lists them by key:

```yaml
- id: models
  config:
    default: main
    models:
      main: {provider: anthropic, model: claude-sonnet-5, reasoningEffort: medium}
      fast: {provider: anthropic, model: claude-haiku-4-5-20251001}
```

Each route may carry `reasoningEffort`, `temperature` and `maxTokens`, which a
root started on it gets as its agent settings. `default` must be a listed key; a
list may be empty, which is `ph-base`'s: it registers no adapter, so there is
nothing to list.

## One rule for a choice

`ModelList.resolve` is what the command line, `/model` and rpc all ask:

| asked for | runs on |
|---|---|
| nothing | the default |
| a key (`--model fast`, `/model fast`) | that entry — a key the profile does not list is refused, naming the ones it does |
| a whole route (`--provider p --model m`, `/model p/m`) | the entry it matches, else the route as given, with no key |

A person's own choice is not bounded by the list. What the list bounds is what an
*agent* may pick when it starts one (session profiles, S7b). `choose(ctx, choice)`
adds the fact only a mount knows: a provider no adapter serves is refused where it
is chosen, rather than at the first request as a turn that fails.

## The surface

```text
ctx.models.resolve(choice)   # -> ModelEntry(key, route); raises ModelChoiceError
ctx.models.entries()         # -> the listed models as a front end offers them, default first
ctx.models.default           # -> the default key, "" when nothing is listed
ctx.models.routes            # -> key -> ModelRoute
```

`ModelChoice` is a key or a whole route, never both; `ModelChoice.from_flags` reads
`--provider`/`--model` (`--model` alone is a key) and `ModelChoice.parse` reads
`/model`'s argument (a `/` means a route, split at the first one).

## Changing the route of a running root

`AgentDriver.reroute(options)` runs an agent on a new route from its next
request. Setting `options` alone would change nothing: past its first request an
agent seeds each one from the conversation's own `request/header`. The header the
next request logs, with reason `change`, is the log's record of the new route.

`ModelEntry` is the one shape for a route under its key — what `/model` offers, what
a choice resolves to, and what `daemon/status` says a new root starts on — and
`entry.options()` carries the key into `AgentOptions.model_key`, so the agent owns
both halves and nothing beside it has to be kept in step.

Over the daemon this is `session/model`, a mutation answered with the root's
description: the choice is resolved before the idempotence key is claimed, so a
refusal does not burn the retry. `session/new` takes a choice too, which is how the
TUI's `--provider`/`--model` reach its own session — a fresh root mounts on it, and
the daemon it spawns is started with no route.

**Not yet an override that survives a restart.** A resumed root starts on its
profile's default again. Logging the choice as an override, and rebuilding it on
restart, is S4 of `plans/Session_Profiles_Plan.md`.
