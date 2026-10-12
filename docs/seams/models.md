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
      main: {provider: anthropic, model: claude-sonnet-5}
      fast: {provider: anthropic, model: claude-haiku-4-5-20251001}
```

Each route may carry `reasoningEffort`, its `efforts`, `maxTokens` and the sampling settings model
makers recommend (`temperature`, `topP`, `topK`, `minP`, `presencePenalty`,
`frequencyPenalty`), which a root started on it gets as its agent settings, and so does a child an agent starts
on it, by key or as the whole route (`child_options`), rather than its parent's. A
child whose skill names a key this mount cannot run falls back to its parent's route
and says so (`ModelFallback`, in [subagents](subagents.md)), and a readmitted child
resolves its route again against the list as it is then, falling back the same way
whatever named it. `default` must be a listed key; a list may be empty, which is
`ph-base`'s: it registers no adapter, so there is nothing to list.

**Kinds are keys named for the job** (`MODEL_KINDS`), so a skill works on any profile:
`classify` for short replies at low effort with steady settings (sorting, tagging, yes
or no), and `light` for cheap general work. A profile serves a kind by listing it, on
whatever route does that job well there; a kind it does not list falls back to the
parent's model with a warning, the cue to set it up, and `phern doctor` lists under
"Model kinds" which it serves. Shipped profiles do not alias a kind to their default
model, which would hide that warning. Media are not kinds: a skill states what it hands
its child with `accepts`, whatever kind it asks for ([subagents](subagents.md)). ph-core
ships one skill that asks for a kind, `categorize` (`model: classify`, a checked
`{category, reason}` value), offered wherever a provider can start a child.

**A sampling setting a route's API cannot carry is dropped, not refused.** Each
adapter says which it takes (`ResolvedModel.sampling`): OpenAI-compatible servers take
`temperature`, `topP` and the penalties, and `topK` and `minP` where a profile says
its server does (`samplingExtras`, as llama.cpp and vLLM do); Anthropic takes
`temperature`, `topP` and `topK`; Google all but `minP`. The `llm` seam drops any other
before the request and warns once per route and setting, and `phern doctor` names it
under "Model settings". An unset setting is never sent, so the server's own flags
decide it.

**A route's `efforts` say what each reasoning effort sends on its API.** No adapter
sends an effort of its own, because each API spells it differently and the spelling
belongs to the model as much as to the wire: Qwen on llama.cpp turns its thinking off
with `chat_template_kwargs`, an OpenAI reasoning model takes `reasoning_effort`,
Anthropic a `thinking` budget, Gemini 2.5 `generationConfig.thinkingConfig`. So a
profile says it per model, as request body fields by effort name, and the adapter
merges the chosen effort's fields into its body key by key, after its own
(`wire_effort`):

```yaml
      local:
        provider: llama
        model: qwen3
        reasoningEffort: none
        efforts:
          none: {chat_template_kwargs: {enable_thinking: false}}
          high: {chat_template_kwargs: {enable_thinking: true}}
```

**Unset is the provider's default, and so is `default`.** Leaving the field out of the
request is how each of these APIs falls back to its own default, so an effort nobody
asked for sends nothing and warns of nothing. `default` (`DEFAULT_EFFORT`) asks for
that by name, over an effort the route sets: a route's `reasoningEffort`, a spawn's
effort or a skill's `effort` may name it, it sends nothing and warns of nothing, and
since it is the provider's and not pH's, `efforts` cannot define it.

**An effort asked for that the route has no mapping for is warned about.** One its
`efforts` do not define, its own or one a spawn or a skill asks of a child on it, is
not sent, so the provider's default applies, with a warning once per route and effort
and a line under "Model settings". An effort defined as `{}` sends nothing on purpose
and is not warned about. The fields are the route's the request is sent on, looked up
in the session's own list after any plugin moved it (`ModelList.efforts_for`, the
agent's key first where two keys list one route), so a route the list does not hold has
none and its effort is withheld.

**The request header records what was sent.** The agent loop withholds what a route
cannot carry, a sampling setting or an effort, before it logs `request/header`, so
the header's `config` is what went to the provider and its `withheld` holds each
setting asked for and not sent, by value. The next request asks for them again and
checks them against the route as it is then, so a route that comes to carry one sends
it without anyone setting it twice. The footer's effort is the header's, so it shows
none when the effort was withheld.

## One rule for a choice

`ModelList.resolve` is what the command line, `/model` and rpc all ask:

| asked for | runs on |
|---|---|
| nothing | the default |
| a key (`--model fast`, `/model fast`) | that entry — a key the profile does not list is refused, naming the ones it does |
| a whole route (`--provider p --model m`, `/model p/m`) | the entry it matches, else the route as given — which joins the session's list under its own name when chosen |

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

A `ModelChoiceError` is a sentence for a person. The refusal a caller may degrade
around is its subclass `UnrunnableChoice`, a well-formed choice this mount cannot run,
which also carries `reason`: `unlisted` (the list does not hold the key) or `unserved`
(no adapter here serves the route's provider). A child whose skill names such a key,
and any readmitted child, catches it to fall back to its parent's route
([subagents](subagents.md)).

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

**A choice is an override of the session's `models` row** (S4, `move_to`):
`/model`, `session/model` and each start's `--provider`/`--model` that differs make
the chosen entry the row's `default` through `ph.session_profile.override`, recorded
before it is made. A route the list did not hold joins it under a key made from its
own name (`route_key`, `fake/fake-9` → `fake-fake-9`), so the footer can name it and
the session's next start runs on it again.
