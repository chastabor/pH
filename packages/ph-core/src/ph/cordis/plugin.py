"""Plugin identity: what a row mounts, and how its config is validated.

A plugin is a name, a list of injected service keys, an optional pydantic
config model, and an `apply(ctx, config)` body. It may be written as a
decorated function or as any object carrying those four attributes — the
shape is duck-typed by `normalize_plugin`, not enforced by a base class.

@module ph.cordis.plugin
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, overload

from pydantic import BaseModel

from .errors import LoaderError
from .key import ServiceKey, service_names

if TYPE_CHECKING:
    from .context import Context

__all__ = ["PluginSpec", "normalize_plugin", "plugin"]


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """One plugin's normalized identity."""

    name: str
    apply: Callable[..., Any]
    """Erased on purpose, and the erasure is the same one `ToolDefinition.output`
    carries: a spec lives in a table beside every other row, so the config type
    `plugin()` checked is not recoverable from here. The decorator's overload is
    what holds a body to the model it declares — this field is what the loader
    calls, and it calls every row through one signature."""
    inject: tuple[str, ...] = ()
    config_model: type[BaseModel] | None = None

    def resolve_config(self, raw: object) -> BaseModel | None:
        """Validate a row's raw config against the plugin's model.

        **A plugin without a model takes no config, and is handed `None`.** It
        used to be handed the row's config verbatim, which made a `config:` block
        under a row that never reads one a silent no-op: the profile said
        something, the mount agreed, and nothing happened. Fifty-two rows in this
        tree ignore their config; the four that read it raw now declare a model.
        So a non-empty config on a model-less row is refused, naming the row and
        the keys — the same sentence `extra="forbid"` gives a mistyped key under a
        row that *does* have a model. An empty mapping is treated as absent, since
        `config: {}` says nothing either way.
        """
        model = self.config_model
        if model is None:
            if raw is None or raw == {}:
                return None
            given = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise LoaderError(
                f'row "{self.name}" takes no config, but the profile gave it {given}; '
                "remove the config block, or address a row that has options"
            )
        if isinstance(raw, model):
            return raw
        if raw is None:
            return model()
        if isinstance(raw, BaseModel):
            return model.model_validate(raw.model_dump(by_alias=True))
        return model.model_validate(raw)


@overload
def plugin(
    name: str, *, inject: Sequence[str | ServiceKey[Any]] = ()
) -> Callable[
    [Callable[[Context, None], Awaitable[None]]], Callable[[Context, None], Awaitable[None]]
]: ...
@overload
def plugin[C: BaseModel](
    name: str, *, inject: Sequence[str | ServiceKey[Any]] = (), config: type[C]
) -> Callable[
    [Callable[[Context, C], Awaitable[None]]], Callable[[Context, C], Awaitable[None]]
]: ...
def plugin(
    name: str,
    *,
    inject: Sequence[str | ServiceKey[Any]] = (),
    config: type[BaseModel] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a function as a plugin body.

    ```python
    @plugin("session", inject=[LLM], config=SessionConfig)
    async def apply(ctx: Context, config: SessionConfig) -> None: ...
    ```

    **Overloaded on `config=`, so the body's second parameter is checked against
    the model the decorator names.** Forty-five rows declare a model and read
    `config.field` in the body; before this, a row that declared `Config` and
    annotated `config: OtherConfig` type-checked, and the first field read at
    mount was where the difference surfaced. A row with no model takes `None`,
    and `resolve_config` above is what makes that annotation true.
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__ph_plugin__ = PluginSpec(  # type: ignore[attr-defined]
            name=name,
            apply=fn,
            inject=service_names(inject),
            config_model=config,
        )
        return fn

    return decorate


def normalize_plugin(source: object) -> PluginSpec:
    """Coerce a decorated function, a module, or a plugin object into a spec."""
    if isinstance(source, PluginSpec):
        return source
    marked = getattr(source, "__ph_plugin__", None)
    if isinstance(marked, PluginSpec):
        return marked
    apply = getattr(source, "apply", None)
    if apply is None and callable(source):
        apply = source
    if apply is None:
        raise LoaderError(f"{source!r} is not a plugin: no apply() and not callable")
    marked = getattr(apply, "__ph_plugin__", None)
    if isinstance(marked, PluginSpec):
        return marked
    name = getattr(source, "name", None) or getattr(source, "__name__", None) or repr(source)
    config_model = getattr(source, "Config", None)
    if config_model is not None and not (
        isinstance(config_model, type) and issubclass(config_model, BaseModel)
    ):
        raise LoaderError(f'plugin "{name}" has a Config that is not a pydantic model')
    return PluginSpec(
        name=str(name),
        apply=apply,
        inject=service_names(getattr(source, "inject", ()) or ()),
        config_model=config_model,
    )
