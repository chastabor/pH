"""`ctx.models`: the models that a profile lists, and which one a root runs on.

A route is a provider and a model. It used to be two free strings: the
`--provider` and `--model` of the daemon for every root, and for a child its
own request or else the route of its parent. So no profile could say which
models a session could reach. A profile now lists them by name
(`plans/Session_Profiles_Plan.md`, item 8):

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

The `efforts` of a route say what each reasoning effort sends on the API of the
route (`ModelRoute.efforts`). An effort that they do not define is not sent,
and a warning says so.

A root runs on `default` unless a user chose otherwise. `ModelList.resolve` is
the one rule for what a choice means. The command line, `/model` and rpc all
ask it, so the three cannot come to disagree:

* Nothing: the default.
* A key: that entry. A key that the profile does not list is refused, and the
  refusal names the keys that it does list.
* A whole route (`provider` and `model`): the entry that it matches, or else
  the route as given. The list does not bound the choice of a user. It bounds
  what an *agent* may pick when it spawns a child (S7b).

The row affects the **environment**, because the model is the first thing that
the work of an agent depends on. An audit of a session that could not say which
model the session ran on would be no audit.

The list belongs to the user. It is not a catalog that pH keeps. The old
refusal of `model_choices` to invent a catalog that goes stale still holds,
because nothing here is shipped as the truth about a provider. The list of a
shipped profile is a default that a user replaces.

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
"""The longest key that a profile may name a model by. A key is a word that a
user types."""

ModelKind: TypeAlias = Literal["classify", "light"]
"""A kind of model that a shipped skill may ask for by key (A2). A kind is named
for the job, so the skill works on any profile. The name of a model would work
only on the one profile that has that model."""

MODEL_KINDS: Mapping[ModelKind, str] = {
    "classify": "short replies at low effort with steady settings: sorting, tagging, yes or no",
    "light": "cheap general work: a summary, a lookup, a first pass",
}
"""Each shared kind, and the job that it names (A2).

A profile serves a kind by listing it as a key, on whatever route does that job
well there. A kind that the profile does not list falls back to the model of
the parent with a warning (`ModelFallback`). The warning is the cue to set the
kind up, and `phern doctor` lists which kinds the profile serves. Shipped
profiles do not alias a kind to their default model, because that would hide
the warning. The media that a skill hands its child are not kinds: a skill
states them with `accepts`, whatever kind it asks for."""


UnrunnableReason: TypeAlias = Literal["unlisted", "unserved"]
"""Why a well-formed choice cannot run here: the list does not hold its key, or
no adapter here serves the provider of its route."""


class ModelChoiceError(ValueError):
    """A choice that names nothing this profile can run, in a sentence that says why."""


class UnrunnableChoice(ModelChoiceError):
    """A well-formed choice that this mount cannot run, with `reason` as a code.
    It is the one refusal that a caller may degrade around, where it would
    otherwise pass the refusal on. A child falls back to the model of its
    parent (`ph.seams.subagents.ModelFallback`) in two cases: the model that
    its skill names cannot run, or the child is readmitted and its model cannot
    run."""

    def __init__(self, message: str, *, reason: UnrunnableReason) -> None:
        super().__init__(message)
        self.reason = reason


class ModelRoute(WireModel):
    """One route: the provider that serves it, the model, and its call settings.

    `sampling` holds what the maker of a model recommends, kept beside the model
    that it was tuned for: `sampling: {temperature: 0.7, topP: 0.8}`. Each
    setting is sent where the API of the route takes it, and is dropped with a
    warning where the API does not (`ResolvedModel.sampling`). A setting that
    is unset is not sent, so the flags of the server itself decide.
    """

    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    reasoning_effort: str | None = None
    """The effort that this route runs at. It is a name that `efforts` defines,
    or `default` (`DEFAULT_EFFORT`). A spawn or a skill may ask a child for
    another effort. When this is unset, no effort is sent."""
    efforts: dict[str, dict[str, Any]] | None = None
    """What each reasoning effort sends on the API of this model, by the name of
    the effort. Each value holds request body fields that the adapter merges
    into its own body (`wire_effort`), and an object merges key by key.

    **The mapping is per model, and the profile states it**, because the
    spelling belongs to the model as much as to the wire. Qwen on llama.cpp
    turns its thinking off with `chat_template_kwargs`. An OpenAI reasoning
    model takes `reasoning_effort`, Anthropic takes a `thinking` budget, and
    Gemini 2.5 takes `generationConfig.thinkingConfig.thinkingBudget`. No
    adapter guesses one. An effort that is asked for and that this mapping does
    not name is not sent. That effort may be the effort of the route itself, or
    one that a spawn or a skill asks for. Four things report it: a warning, the
    record in the request header (`EpochHeader.withheld`), a line in
    `phern doctor` and, for a child, a note to its parent (`EffortUnsent`).
    `DEFAULT_EFFORT` is the default of the provider and pH does not define it,
    so it is refused here. The field is absent and not empty when it is unset,
    so the wire form of a route is as it was.

    The type is `dict[str, Any]` and not `JsonObject`, which pydantic cannot
    build (`ph.json`)."""
    sampling: Sampling = Field(default_factory=Sampling, exclude_if=Sampling.is_empty)
    max_tokens: int | None = None

    @property
    def label(self) -> str:
        """`provider/model`, the spelling that the picker and the footer use."""
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
    """The key that a root runs on when nobody chose one. It must be a listed
    key. It is empty only when no model is listed."""

    models: dict[str, ModelRoute] = Field(default_factory=dict)
    """Every route that this profile lists, under the key that a user, a skill
    or a spawn names it by. A patch replaces the whole mapping, as it does for
    the config of any row."""

    def choosing(self, chosen: ModelEntry) -> Config:
        """This list with `chosen` as its default. `/model` makes an override of
        the result.

        A route that the list does not hold joins it under a key that is made
        from the name of the route (`route_key`). So the list of the session
        says what the session runs on, and choosing the same route again finds
        the same entry.
        """
        key = chosen.key or route_key(chosen.route)
        if not chosen.key:
            # A listed model may already hold the made-up key for another route.
            # The key then gets a number, so that it does not name that other
            # route without notice.
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

    Neither means the default. A key and a route are exclusive because they are
    two different requests: "the one that this profile calls `fast`" and
    "exactly this route". A value that carried both would make the resolver
    pick which one was meant.
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
        """`--provider` and `--model` as a user types them.

        `--model` alone names a listed key, which is the short thing to type.
        The two together are a whole route. `--provider` alone would be a
        provider with no model, which nothing can run.
        """
        if provider is None:
            return cls(key=model or "")
        if not model:
            raise ModelChoiceError("--provider names a route, and needs --model beside it")
        return cls(provider=provider, model=model)

    @classmethod
    def parse(cls, text: str) -> ModelChoice:
        """The argument of `/model` and the value of the picker: `key`, or
        `provider/model`.

        A key is a slug and so has no `/`, which is how the two are told apart.
        The split is at the *first* `/`, because a model name may hold more of
        them (`org/model` behind an OpenAI-compatible server).
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
    """A route under the key that the profile lists it by. The key is `""` for a
    route that the profile does not list, until a user chooses the route and it
    joins the list of the session under its own name (`route_key`).

    It is what `/model` offers, what a choice resolves to, and what a daemon
    starts a root on. One shape holds the one fact, so nothing converts between
    copies of it.
    """

    key: str
    route: ModelRoute
    default: bool = False

    def options(self) -> AgentOptions:
        """The agent settings that this entry starts an agent with, its key
        included."""
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
        """The list that a composed profile names, read without mounting it.

        It is for a check that is made before anything mounts. `phern daemon`
        refuses a `--model` that its profile does not list, where every root
        would otherwise fail on it later. An absent or disabled row is the
        empty list, which is also what a mount without the row resolves against
        (`choose`). The config is interpolated as a mount interpolates it, so
        `${env:LLAMA_MODEL:-default}` names the same model here.
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
        """What `choice` runs on: the three cases of the module docstring, and
        nothing else."""
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
        """The key that this list holds the route `provider/model` under. It is
        `key` where `key` names that route, or else the first key that does. It
        is `None` for a route that the list does not hold."""
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
        """What each reasoning effort sends on the route `provider/model`, by the
        entry that this list holds the route under (`listed_as`). `key` is tried
        first, because two keys may list one route with different efforts. The
        result is empty for a route that the list does not hold."""
        listed = self.listed_as(provider, model, key=key)
        efforts = self.routes[listed].efforts if listed is not None else None
        return efforts or {}


def choose(ctx: Context, choice: ModelChoice) -> ModelEntry:
    """`choice` resolved by the list of this mount, on a provider that an adapter
    here serves.

    The rule of the list is `ModelList.resolve`. This function adds the one
    fact that only a mount knows, which is the set of registered providers. So
    a route that nothing can run is refused where it is chosen, and not at its
    first request. A mount without the row resolves against the empty list,
    and only a whole route runs there.
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
    """A key for a route that a user chose and that no list held: the name of the
    route, as a slug."""
    return slugify(f"{route.provider}-{route.model}", maximum=KEY_MAX) or "chosen"


def models_row(profile: Profile) -> Row | None:
    """The row that lists the models of `profile`. It is found by the name of its
    plugin, as `/sandbox` finds its row, because the profile chooses the id."""
    return next((row for row in profile.enabled_rows() if row.name == "models"), None)


async def move_to(
    ctx: Context, session: Session, chosen: ModelEntry, *, source: OverrideSource, command: str
) -> ModelEntry:
    """Make `chosen` the model of this session, as an override of the `models`
    row (S4).

    The change goes through `override` (`ph.session_profile`), the one function
    that changes the config of a row on a mount. So the log holds the choice
    before it is made, and a restart runs on it again. The function answers the
    entry under the key that the list of the session now gives it, which is how
    an unlisted route comes back with a name. A mount with no `models` row has
    no list to change. The route then belongs to the agent alone, and the
    function answers it as chosen.
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
    """The entry that a starting agent runs on. A start that names a `choice`
    makes it the model of the session (`move_to`). A start that names none runs
    on the default of the session itself.

    Every host that starts an agent calls this one function: the roots of the
    daemon, `phern -p` and rpc. So the meaning of a start option cannot differ
    between them.

    :raises ModelChoiceError: for a choice that nothing here can run.
    """
    if choice.is_default:
        return choose(ctx, choice)
    return await move_to(ctx, session, choose(ctx, choice), source=source, command=command)


def narrows(mounted: Config, asked: Config, _reach: ChildReach) -> ChildLimit:
    """The `models` row of a child names the model that the child runs on. That
    model must be a key that the list of its parent holds, because the child
    runs on the adapters of its parent (S7b)."""
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
    """The rows of `phern doctor` for each listed route that sets what it cannot
    send. Every request on the route goes without that setting, with a warning.
    There are two cases. The first is a sampling setting that the API of the
    route does not take (A1, `Sampling.uncarried`), where its adapter has said
    what it takes. The second is a reasoning effort that the `efforts` of the
    route do not define (`unmapped_effort`)."""
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
    """The rows of `phern doctor` for each shared kind (`MODEL_KINDS`): the route
    that this profile serves the kind on, or a note that a skill that asks for
    the kind runs on the model of its parent."""
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
    """Provide the model list of the profile. In `phern doctor`, report which
    settings the list sets that its routes cannot carry, and which shared kinds
    the profile serves."""
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
