"""Plugin identity: what a row mounts, and how its config is validated.

A plugin is a name, a list of injected service keys, an optional pydantic
config model, and an `apply(ctx, config)` body. It may be written as a
decorated function or as any object carrying those four attributes — the
shape is duck-typed by `normalize_plugin`, not enforced by a base class.

@module ph.cordis.plugin
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from .errors import LoaderError

__all__ = ["PluginSpec", "normalize_plugin", "plugin"]


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """One plugin's normalized identity."""

    name: str
    apply: Callable[..., Any]
    inject: tuple[str, ...] = ()
    config_model: type[BaseModel] | None = None

    def resolve_config(self, raw: Any) -> Any:
        """Validate a row's raw config against the plugin's model.

        A plugin without a model receives the row's config verbatim, which is
        how a plugin whose config is a plain mapping (or absent) works.
        """
        model = self.config_model
        if model is None or isinstance(raw, model):
            return raw
        if raw is None:
            return model()
        if isinstance(raw, BaseModel):
            return model.model_validate(raw.model_dump(by_alias=True))
        return model.model_validate(raw)


def plugin(
    name: str, *, inject: Sequence[str] = (), config: type[BaseModel] | None = None
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a function as a plugin body.

    ```python
    @plugin("session", inject=["llm"], config=SessionConfig)
    async def apply(ctx: Context, config: SessionConfig) -> None: ...
    ```
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__ph_plugin__ = PluginSpec(  # type: ignore[attr-defined]
            name=name, apply=fn, inject=tuple(inject), config_model=config
        )
        return fn

    return decorate


def normalize_plugin(source: Any) -> PluginSpec:
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
        inject=tuple(getattr(source, "inject", ()) or ()),
        config_model=config_model,
    )
