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
      main: {provider: anthropic, model: claude-sonnet-5}
      fast: {provider: anthropic, model: claude-haiku-4-5-20251001}
      local:
        provider: llama
        model: qwen3
        reasoningEffort: none
        efforts:
          none: {chat_template_kwargs: {enable_thinking: false}}
          high: {chat_template_kwargs: {enable_thinking: true}}
```

A route's `efforts` say what each reasoning effort sends on its API (`ModelRoute.efforts`);
an effort they do not define is not sent, with a warning.

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
from typing import Any, Literal, TypeAlias

from pydantic import Field, ValidationError, model_validator

from ..agent.types import AgentOptions
from ..cordis import (
    ChildLimit,
    ChildReach,
    Context,
    NarrowingRefused,
    Profile,
    Row,
    interpolate,
    plugin,
)
from ..keys import LLM, MODELS, MOUNT
from ..llm.types import DEFAULT_EFFORT, EffortFields, Sampling, unmapped_effort
from ..session import Session
from ..session_profile import OverrideSource, override
from ..wire import WireModel
from ._names import require_slug, slugify
from .diagnostics import Diagnostic, contribute

__all__ = [
    "KEY_MAX",
    "MODEL_KINDS",
    "Config",
    "ModelChoice",
    "ModelChoiceError",
    "ModelEntry",
    "ModelKind",
    "ModelList",
    "ModelRoute",
    "UnrunnableChoice",
    "UnrunnableReason",
    "apply",
    "choose",
    "models_row",
    "move_to",
    "route_key",
    "start_on",
]

KEY_MAX = 32
"""The longest key a profile may name a model by — a word a person types."""

ModelKind: TypeAlias = Literal["classify", "light"]
"""A kind of model a shipped skill may ask for by key (A2): named for the job, so a
skill works on any profile, rather than for a model, which only one profile has."""

MODEL_KINDS: Mapping[ModelKind, str] = {
    "classify": "short replies at low effort with steady settings: sorting, tagging, yes or no",
    "light": "cheap general work: a summary, a lookup, a first pass",
}
"""Each shared kind, and the job it names (A2).

A profile serves a kind by listing it as a key, on whatever route does that job well
there. One it does not list falls back to the parent's model with a warning
(`ModelFallback`), which is the cue to set it up; `phern doctor` lists which kinds the
profile serves. Shipped profiles do not alias a kind to their default model, since that
would hide the warning. The media a skill hands its child are not kinds: a skill states
them with `accepts`, whatever kind it asks for."""


UnrunnableReason: TypeAlias = Literal["unlisted", "unserved"]
"""Why a well-formed choice cannot run here: the list does not hold its key, or no
adapter here serves its route's provider."""


class ModelChoiceError(ValueError):
    """A choice that names nothing this profile can run, in a sentence that says why."""


class UnrunnableChoice(ModelChoiceError):
    """A well-formed choice this mount cannot run, with `reason` as a code: the one
    refusal a caller may degrade around rather than surface. A child whose skill's model
    cannot run, or any readmitted child whose model cannot, falls back to its parent's
    (`ph.seams.subagents.ModelFallback`)."""

    def __init__(self, message: str, *, reason: UnrunnableReason) -> None:
        super().__init__(message)
        self.reason = reason


class ModelRoute(WireModel):
    """One route: the provider that serves it, the model, and its call settings.

    `sampling` is what a model maker recommends, kept beside the model it was tuned for:
    `sampling: {temperature: 0.7, topP: 0.8}`. Each setting is sent where the route's
    API takes it and dropped with a warning where it does not (`ResolvedModel.sampling`);
    unset is unsent, so the server's own flags decide.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: str | None = None
    """The effort this route runs at, by a name its `efforts` defines, or `default`
    (`DEFAULT_EFFORT`); a spawn or a skill may ask a child for another. Unset sends
    none."""
    efforts: dict[str, dict[str, Any]] | None = None
    """What each reasoning effort sends on this model's API, by the effort's name: request
    body fields the adapter merges into its own (`wire_effort`), an object key by key.

    **Per model, and stated by the profile**, because the spelling is the model's as much
    as the wire's: Qwen on llama.cpp turns its thinking off with `chat_template_kwargs`,
    an OpenAI reasoning model takes `reasoning_effort`, Anthropic a `thinking` budget,
    Gemini 2.5 `generationConfig.thinkingConfig.thinkingBudget`. No adapter guesses one.
    An effort asked for that this does not name (the route's own, or one a spawn or a
    skill asks for) is not sent, with a warning, the request header's record
    (`EpochHeader.withheld`) and a line in `phern doctor`. `DEFAULT_EFFORT` is not pH's
    to define, so it is refused here. Absent rather than empty when unset, so a route's
    wire form is as it was.

    `dict[str, Any]` rather than `JsonObject`, which pydantic cannot build (`ph.json`)."""
    sampling: Sampling = Field(default_factory=Sampling, exclude_if=Sampling.is_empty)
    max_tokens: int | None = None

    @property
    def label(self) -> str:
        """`provider/model`, the spelling the picker and the footer use."""
        return f"{self.provider}/{self.model}"

    @model_validator(mode="after")
    def _efforts_are_named_and_not_the_provider_s(self) -> ModelRoute:
        for name in self.efforts or {}:
            require_slug(name, maximum=KEY_MAX, kind="effort")
        if self.efforts and DEFAULT_EFFORT in self.efforts:
            raise ValueError(
                f'efforts cannot define "{DEFAULT_EFFORT}": it is the provider\'s own '
                "default, which sends nothing"
            )
        return self


class Config(WireModel):
    """Row config for `models`: the listed routes, by key, and which one is the default."""

    default: str = ""
    """The key a root runs on when nobody chose. Must be listed; empty only when
    nothing is."""

    models: dict[str, ModelRoute] = Field(default_factory=dict)
    """Every route this profile means, under the key a person, a skill or a spawn
    names it by. Replaced whole by a patch, like any row's config."""

    def choosing(self, chosen: ModelEntry) -> Config:
        """This list with `chosen` as its default — what `/model` makes an override of.

        A route the list does not hold joins it under a key made from its own name
        (`route_key`), so the session's list says what it runs on, and choosing the
        same route again finds the same entry.
        """
        key = chosen.key or route_key(chosen.route)
        if not chosen.key:
            # A made-up key that a listed model already holds for another route gets a
            # number, rather than quietly naming that other route.
            count = 2
            while key in self.models and self.models[key] != chosen.route:
                key = f"{route_key(chosen.route)[: KEY_MAX - 3]}-{count}"
                count += 1
        models = dict(self.models)
        models.setdefault(key, chosen.route)
        return Config(default=key, models=models)

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

    @property
    def spelled(self) -> str:
        """As `/model` takes it: the key, or `provider/model`."""
        return self.key or f"{self.provider}/{self.model}"

    @property
    def flags(self) -> str:
        """As the command line takes it: `--model KEY`, or both flags."""
        return (
            f"--model {self.key}"
            if self.key
            else f"--provider {self.provider} --model {self.model}"
        )

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
    """A route under the key the profile lists it by — `""` for one it does not, until a
    person chooses it and it joins the session's list under its own name (`route_key`).

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
            sampling=route.sampling,
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
        row = models_row(profile)
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
                raise UnrunnableChoice(
                    f'"{choice.key}" is not a model this profile lists (it lists {listed}); '
                    "name a whole route as provider/model to run one it does not",
                    reason="unlisted",
                )
            return self.entry(choice.key, route)
        if choice.provider:
            matched = self.listed_as(choice.provider, choice.model)
            if matched is not None:
                return self.entry(matched, self.routes[matched])
            return self.entry("", ModelRoute(provider=choice.provider, model=choice.model))
        if not self.default:
            raise ModelChoiceError(
                "this profile lists no models: give it a `models` row with a default, "
                "or name a route with --provider and --model"
            )
        return self.entry(self.default, self.routes[self.default])

    def listed_as(self, provider: str, model: str, *, key: str = "") -> str | None:
        """The key this list holds the route `provider/model` under: `key` where that is
        the route it names, else the first that is, else `None` for a route it does not
        list."""
        return next(
            (
                name
                for name in (key, *self.routes)
                if (route := self.routes.get(name)) is not None
                and (route.provider, route.model) == (provider, model)
            ),
            None,
        )

    def efforts_for(
        self, provider: str, model: str, *, key: str = ""
    ) -> Mapping[str, EffortFields]:
        """What each reasoning effort sends on the route `provider/model`, by the entry
        this list holds it under (`listed_as`, `key` first, since two keys may list one
        route with different efforts). Nothing for a route it does not list."""
        listed = self.listed_as(provider, model, key=key)
        efforts = self.routes[listed].efforts if listed is not None else None
        return efforts or {}


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
        raise UnrunnableChoice(
            f'no adapter here serves "{chosen.route.provider}" '
            f"(this profile serves {', '.join(served) or 'nothing'})",
            reason="unserved",
        )
    return chosen


def route_key(route: ModelRoute) -> str:
    """A key for a route a person chose that no list held: its own name, as a slug."""
    return slugify(f"{route.provider}-{route.model}", maximum=KEY_MAX) or "chosen"


def models_row(profile: Profile) -> Row | None:
    """The row that lists `profile`'s models — found by its plugin's name, the way
    `/sandbox` finds its row, since the id is the profile's to choose."""
    return next((row for row in profile.enabled_rows() if row.name == "models"), None)


async def move_to(
    ctx: Context, session: Session, chosen: ModelEntry, *, source: OverrideSource, command: str
) -> ModelEntry:
    """Make `chosen` this session's model — an override of the `models` row (S4).

    Through the one door, so the log holds the choice before it is made and a restart
    runs on it again. Answers the entry under the key the session's list now gives
    it, which is how an unlisted route comes back with a name. A mount with no
    `models` row has no list to change: the route is the agent's alone, and is
    answered as chosen.
    """
    mount = ctx.get(MOUNT)
    row = models_row(mount.profile) if mount is not None else None
    listed = ctx.get(MODELS)
    if row is None or listed is None:
        return chosen
    config = listed.config.choosing(chosen)
    wire = config.model_dump(mode="json", by_alias=True)
    await override(ctx, session, row.id, wire, source=source, command=command)
    return ModelList(config).resolve(ModelChoice(key=config.default))


async def start_on(
    ctx: Context, session: Session, choice: ModelChoice, *, source: OverrideSource, command: str
) -> ModelEntry:
    """The entry a starting agent runs on: this start's `choice`, made the session's
    where it differs (`move_to`), else the session's own default.

    The one spelling for every host that starts an agent — the daemon's roots,
    `phern -p` and rpc — so what a start option means cannot differ between them.

    :raises ModelChoiceError: for a choice nothing here can run.
    """
    if choice.is_default:
        return choose(ctx, choice)
    return await move_to(ctx, session, choose(ctx, choice), source=source, command=command)


def narrows(mounted: Config, asked: Config, _reach: ChildReach) -> ChildLimit:
    """A child's `models` row names the model it runs on, which must be a key its
    parent's list holds: the child runs on its parent's adapters (S7b)."""
    key = asked.default
    if not key:
        return ChildLimit()
    try:
        ModelList(mounted).resolve(ModelChoice(key=key))
    except ModelChoiceError as error:
        raise NarrowingRefused(
            f"its models row runs on {key}, which its parent does not list: {error}"
        ) from error
    return ChildLimit(model_key=key)


def dropped_settings(ctx: Context, models: ModelList) -> list[tuple[str, str]]:
    """`phern doctor`'s rows for each listed route that sets what it cannot send, which
    every request on it goes without, with a warning: a sampling setting its API does
    not take (A1, `Sampling.uncarried`), where its adapter has said what it takes, and a
    reasoning effort its `efforts` do not define (`unmapped_effort`)."""
    llm = ctx.get(LLM)
    rows: list[tuple[str, str]] = []
    for key, route in models.routes.items():
        carried = llm.resolve_model(route.provider, route.model).sampling if llm else None
        if carried is not None and (dropped := route.sampling.uncarried(carried)):
            takes = ", ".join(sorted(carried)) or "none"
            rows.append((key, f"{', '.join(dropped)} dropped: {route.label} takes {takes}"))
        effort, efforts = route.reasoning_effort, route.efforts or {}
        if unmapped_effort(effort, efforts):
            defined = ", ".join(efforts) or "none"
            rows.append(
                (
                    key,
                    f"reasoningEffort {effort} not sent, so the provider's default applies: "
                    f"its efforts define {defined}",
                )
            )
    return rows


def served_kinds(models: ModelList) -> list[tuple[str, str]]:
    """`phern doctor`'s rows for each shared kind (`MODEL_KINDS`): the route this
    profile serves it on, or that a skill asking for it runs on its parent's model."""
    routes = models.routes
    return [
        (
            kind,
            routes[kind].label if kind in routes else f"not listed; {job}, on the parent's model",
        )
        for kind, job in MODEL_KINDS.items()
    ]


@plugin("models", affects="environment", config=Config, narrows=narrows)
async def apply(ctx: Context, config: Config) -> None:
    """Provide the profile's model list, and say in `phern doctor` which settings it
    sets that its routes cannot carry."""
    models = ModelList(config)
    ctx.provide(MODELS, models)
    contribute(
        ctx,
        Diagnostic(
            id="model-settings",
            title="Model settings",
            read=lambda: dropped_settings(ctx, models),
            order=7,
        ),
    )
    contribute(
        ctx,
        Diagnostic(
            id="model-kinds", title="Model kinds", read=lambda: served_kinds(models), order=8
        ),
    )
