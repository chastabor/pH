"""`ctx.models` — the models a profile lists, and which one a root runs on.

A route is a provider and a model. It used to be two free strings — the daemon's
`--provider`/`--model` for every root, a child's own request else its parent's —
so which models a session could reach was nowhere a profile could say. A profile
now lists them by name (`plans/Session_Profiles_Plan.md`, item 8):

```yaml
- id: models
  config:
    default: main
    models:
      main: {provider: anthropic, model: claude-sonnet-5, reasoningEffort: medium}
      fast: {provider: anthropic, model: claude-haiku-4-5-20251001}
```

A root runs on `default` unless a person chose otherwise, and `ModelList.resolve`
is the one rule for what a choice means — the command line, `/model` and rpc all
ask it, so the three cannot come to disagree:

* nothing: the default;
* a key: that entry — and a key the profile does not list is refused, naming
  the ones it does;
* a whole route (`provider` and `model`): the entry it matches, else the route as
  given. A person's own choice is not bounded by the list; what the list bounds is
  what an *agent* may pick when it spawns one (S7b).

**Environment**, because the model is the first thing an agent's work depends
on, and a session's audit that could not say which it ran would be no audit.

The list is the person's, not a catalog pH keeps: `model_choices`' old refusal to
invent one that goes stale still holds, because nothing here is shipped as the
truth about a provider — a shipped profile's list is a default a person replaces.

@module ph.seams.models
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pydantic import Field, ValidationError, model_validator

from ..agent.types import AgentOptions
from ..cordis import Context, Profile, interpolate, plugin
from ..keys import LLM, MODELS
from ..wire import WireModel
from ._names import require_slug

__all__ = [
    "KEY_MAX",
    "Config",
    "ModelChoice",
    "ModelChoiceError",
    "ModelEntry",
    "ModelList",
    "ModelRoute",
    "apply",
    "choose",
]

KEY_MAX = 32
"""The longest key a profile may name a model by — a word a person types."""


class ModelChoiceError(ValueError):
    """A choice that names nothing this profile can run, in a sentence that says why."""


class ModelRoute(WireModel):
    """One route: the provider that serves it, the model, and its call settings."""

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: str | None = None
    temperature: float | None = None
    max_tokens: int | None = None

    @property
    def label(self) -> str:
        """`provider/model`, the spelling the picker and the footer use."""
        return f"{self.provider}/{self.model}"


class Config(WireModel):
    """Row config for `models`: the listed routes, by key, and which one is the default."""

    default: str = ""
    """The key a root runs on when nobody chose. Must be listed; empty only when
    nothing is."""

    models: dict[str, ModelRoute] = Field(default_factory=dict)
    """Every route this profile means, under the key a person, a skill or a spawn
    names it by. Replaced whole by a patch, like any row's config."""

    @model_validator(mode="after")
    def _default_is_listed(self) -> Config:
        for key in self.models:
            require_slug(key, maximum=KEY_MAX, kind="model key")
        if self.models and self.default not in self.models:
            raise ValueError(
                f'default "{self.default}" is not a listed model; the list is '
                f"{', '.join(sorted(self.models))}"
            )
        if self.default and not self.models:
            raise ValueError(f'default "{self.default}" names a model, and none is listed')
        return self


class ModelChoice(WireModel):
    """What somebody asked to run on: a listed key, a whole route, or neither.

    Neither is the default. The two are exclusive because they are two different
    requests — "the one this profile calls `fast`" and "exactly this route" — and
    a value that carried both would make the resolver pick which was meant.
    """

    key: str = ""
    provider: str = ""
    model: str = ""

    @model_validator(mode="after")
    def _one_request(self) -> ModelChoice:
        if self.key and (self.provider or self.model):
            raise ValueError("a model choice is a key or a route, not both")
        if bool(self.provider) != bool(self.model):
            raise ValueError("a route names both a provider and a model")
        return self

    @property
    def is_default(self) -> bool:
        """Whether this asks for nothing, so whoever holds a default decides."""
        return not (self.key or self.provider)

    @classmethod
    def from_flags(cls, provider: str | None, model: str | None) -> ModelChoice:
        """`--provider`/`--model` as a person types them.

        `--model` alone names a listed key, which is the short thing to type; the
        two together are a whole route. `--provider` alone would be a provider
        with no model, which nothing can run.
        """
        if provider is None:
            return cls(key=model or "")
        if not model:
            raise ModelChoiceError("--provider names a route, and needs --model beside it")
        return cls(provider=provider, model=model)

    @classmethod
    def parse(cls, text: str) -> ModelChoice:
        """`/model`'s argument and the picker's value: `key`, or `provider/model`.

        A key is a slug and so has no `/`, which is what tells the two apart. The
        split is at the *first* `/`, because a model name may carry more of them
        (`org/model` behind an OpenAI-compatible server).
        """
        stripped = text.strip()
        provider, slash, model = stripped.partition("/")
        if not slash:
            return cls(key=stripped)
        try:
            return cls(provider=provider, model=model)
        except ValidationError as error:
            raise ModelChoiceError(f'"{stripped}" is not a key or a provider/model') from error


class ModelEntry(WireModel):
    """A route under the key the profile lists it by — `""` for one it does not.

    What `/model` offers, what a choice resolves to, and what a daemon starts a root
    on: one shape for the one fact, so nothing converts between copies of it.
    """

    key: str
    route: ModelRoute
    default: bool = False

    def options(self) -> AgentOptions:
        """The agent settings this entry starts an agent with, its key included."""
        route = self.route
        return AgentOptions(
            provider=route.provider,
            model=route.model,
            max_tokens=route.max_tokens,
            temperature=route.temperature,
            reasoning_effort=route.reasoning_effort,
            model_key=self.key,
        )


@dataclass(frozen=True, slots=True)
class ModelList:
    """The listed models and the rule that turns a choice into a route."""

    config: Config = field(default_factory=Config)

    @classmethod
    def of(cls, profile: Profile) -> ModelList:
        """The list a composed profile names, read without mounting it.

        For a check made before anything mounts — `phern daemon` refusing a
        `--model` its profile does not list, rather than every root failing on
        it later. An absent or disabled row is the empty list, which is also what
        a mount without the row provides. Interpolated as a mount interpolates,
        so `${env:LLAMA_MODEL:-default}` is the model it names here too.
        """
        row = next((row for row in profile.enabled_rows() if row.name == "models"), None)
        if row is None:
            return cls()
        return cls(Config.model_validate(interpolate(row.config) or {}))

    @property
    def default(self) -> str:
        return self.config.default

    @property
    def routes(self) -> Mapping[str, ModelRoute]:
        return self.config.models

    def entry(self, key: str, route: ModelRoute) -> ModelEntry:
        return ModelEntry(key=key, route=route, default=bool(key) and key == self.default)

    def entries(self) -> list[ModelEntry]:
        """Every listed model, default first and the rest by key."""
        return sorted(
            (self.entry(key, route) for key, route in self.routes.items()),
            key=lambda entry: (not entry.default, entry.key),
        )

    def resolve(self, choice: ModelChoice) -> ModelEntry:
        """What `choice` runs on — the module docstring's three cases, and nothing else."""
        if choice.key:
            route = self.routes.get(choice.key)
            if route is None:
                listed = ", ".join(sorted(self.routes)) or "nothing"
                raise ModelChoiceError(
                    f'"{choice.key}" is not a model this profile lists (it lists {listed}); '
                    "name a whole route as provider/model to run one it does not"
                )
            return self.entry(choice.key, route)
        if choice.provider:
            matched = next(
                (
                    key
                    for key, route in self.routes.items()
                    if (route.provider, route.model) == (choice.provider, choice.model)
                ),
                None,
            )
            if matched is not None:
                return self.entry(matched, self.routes[matched])
            return self.entry("", ModelRoute(provider=choice.provider, model=choice.model))
        if not self.default:
            raise ModelChoiceError(
                "this profile lists no models: give it a `models` row with a default, "
                "or name a route with --provider and --model"
            )
        return self.entry(self.default, self.routes[self.default])


def choose(ctx: Context, choice: ModelChoice) -> ModelEntry:
    """`choice` resolved by this mount's list, on a provider an adapter here serves.

    The list's rule is `ModelList.resolve`'s. What is added is the one fact only a
    mount knows — which providers are registered — so a route nothing can run is
    refused where it is chosen rather than at its first request. A mount without
    the row resolves against the empty list: only a whole route runs there.
    """
    chosen = (ctx.get(MODELS) or ModelList()).resolve(choice)
    llm = ctx.get(LLM)
    served = llm.list_providers() if llm is not None else []
    if chosen.route.provider not in served:
        raise ModelChoiceError(
            f'no adapter here serves "{chosen.route.provider}" '
            f"(this profile serves {', '.join(served) or 'nothing'})"
        )
    return chosen


@plugin("models", affects="environment", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Provide the profile's model list."""
    ctx.provide(MODELS, ModelList(config))
