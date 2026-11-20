"""Plugin identity: what a row mounts, and how its config is validated.

A plugin has a name, a statement of what its settings shape (`Affects`), a list
of injected service keys, an optional pydantic config model, and an
`apply(ctx, config)` body. It may be written as a decorated function or as any
object that carries those attributes. `normalize_plugin` duck-types the shape,
and no base class enforces it.

@module ph.cordis.plugin
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias, get_args, overload

from pydantic import BaseModel

from ..wire import literal_lookup
from .child_limit import ChildLimit, Narrows
from .errors import LoaderError
from .key import ServiceKey, service_names

if TYPE_CHECKING:
    from .context import Context

__all__ = [
    "AFFECTS",
    "CONFIGURED_IN",
    "LOG_LAYER",
    "Affects",
    "LogLayer",
    "PluginSpec",
    "normalize_plugin",
    "plugin",
]

Affects: TypeAlias = Literal["environment", "presentation", "deployment"]
"""What the settings of a row shape, and therefore which configuration owns them.

* `environment`: what the agent sees or can do. This is its model, tools,
  skills, sandbox, workspace and posture. The profile of a session covers these
  rows: what the session starts with, what is overridden during it, and what a
  restart compares.
* `presentation`: only how a front end draws. This is screens and footer
  readings. The configuration of the TUI owns these rows.
* `deployment`: the machinery of the host itself. This is persistence,
  telemetry, invariants, doctor sections, the local stores, and the credential
  source. The configuration of the daemon owns these rows.

Every plugin declares its kind, and there is no default. A row that does not say
is therefore caught where it is written (the keyword is required by the
decorator), and is not guessed when a profile is split by owner.
`plans/Session_Profiles_Plan.md` is the design.
"""

AFFECTS: Mapping[str, Affects] = literal_lookup(Affects)
"""Every `Affects` value, keyed by its spelling. A lookup checks the word that a
plugin gave and also gives it its type."""

CONFIGURED_IN: Mapping[Affects, str] = {
    "environment": "a session profile, `$PH_HOME/profiles/<name>.yaml`",
    "presentation": "the TUI's settings, `$PH_HOME/tui.json`, where a screen is hidden or "
    "its key remapped",
    "deployment": "the daemon's configuration, `$PH_HOME/daemon.yaml`",
}
"""Where a user sets each kind, in the words that the refusal of a misplaced row
uses.

The values are words and not paths, because two of the three are files of a front
end and of a host, and `ph.cordis` never reads them. They are written one time,
so that the refusal of the loader and the documentation cannot name different
files for one kind.
"""

LogLayer: TypeAlias = Literal["storage", "telemetry", "export", "checkpoint"]
"""The place of a row in the log layer. The log layer is the rows that keep the log,
export it, and say when it reaches disk. It is beneath every row that writes to
the log.

Rows unwind in reverse, so the layer unwinds last, and in the reverse of this
order. The last barrier of the checkpoint policy flushes what the agents
appended as they let go, while every other row of the layer is still mounted. An
`export` row (a sink, or a redactor of what sinks receive) is there to handle
that tail. The telemetry seam outlives the `export` rows, because they register
with it. Storage unwinds last, so no flush announces a write after storage has
gone.

**The place is declared, so nobody has to remember where to write the row.**
The composition gathers the layer at the position of its first row, in this
order (`ph.cordis.loader`). A deployment therefore adds a sink by naming it. A
row appended at the end, which is all that a profile can do, is still placed
beneath the agents. A `storage` row with no `checkpoint` row gets a warning at
mount, because nothing then says when the log reaches disk.
"""

LOG_LAYER: tuple[LogLayer, ...] = get_args(LogLayer)

_LOG_LAYERS: Mapping[str, LogLayer] = literal_lookup(LogLayer)


@dataclass(frozen=True, slots=True)
class PluginSpec:
    """The normalized identity of one plugin."""

    name: str
    apply: Callable[..., Any]
    """The type is erased on purpose, in the same way as `ToolDefinition.output`. A
    spec is in a table beside every other row, so the config type that `plugin()`
    checked cannot be recovered from here. The overload of the decorator holds a
    body to the model that it declares. This field is what the loader calls, and
    the loader calls every row through one signature."""
    affects: Affects
    """The configuration that owns the settings of this row (`Affects`)."""
    inject: tuple[str, ...] = ()
    config_model: type[BaseModel] | None = None
    narrows: Callable[..., ChildLimit] | None = None
    """How a child holds less of this row (`ph.cordis.child_limit`). `None` means
    that a child holds this row as its parent runs it, or not at all. The type is
    erased as the type of `apply` is, and the same overload holds it to the
    config model."""
    log_layer: LogLayer | None = None
    """The place of this row in the log layer (`LogLayer`), or `None` for a row
    outside it."""

    def resolve_config(self, raw: object) -> BaseModel | None:
        """Validate the raw config of a row against the model of the plugin.

        **A plugin without a model takes no config, and receives `None`.** It
        used to receive the config of the row verbatim. A `config:` block under
        a row that never reads one was then a silent no-op: the profile said
        something, the mount agreed, and nothing happened. When this rule was
        written, fifty-two rows in this tree ignored their config, and the four
        rows that read it raw were given a model. So a config that is not empty,
        on a row with no model, is refused. The refusal names the row and the
        keys. It is the same sentence that `extra="forbid"` gives for a mistyped
        key under a row that *does* have a model. An empty mapping is treated as
        absent, because `config: {}` says nothing either way.
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
    name: str,
    *,
    affects: Affects,
    inject: Sequence[str | ServiceKey[Any]] = (),
    log_layer: LogLayer | None = None,
) -> Callable[
    [Callable[[Context, None], Awaitable[None]]], Callable[[Context, None], Awaitable[None]]
]: ...
@overload
def plugin[C: BaseModel](
    name: str,
    *,
    affects: Affects,
    inject: Sequence[str | ServiceKey[Any]] = (),
    config: type[C],
    narrows: Narrows[C] | None = None,
    log_layer: LogLayer | None = None,
) -> Callable[
    [Callable[[Context, C], Awaitable[None]]], Callable[[Context, C], Awaitable[None]]
]: ...
def plugin(
    name: str,
    *,
    affects: Affects,
    inject: Sequence[str | ServiceKey[Any]] = (),
    config: type[BaseModel] | None = None,
    narrows: Callable[..., ChildLimit] | None = None,
    log_layer: LogLayer | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Mark a function as a plugin body.

    ```python
    @plugin("session", affects="deployment", inject=[LLM], config=SessionConfig)
    async def apply(ctx: Context, config: SessionConfig) -> None: ...
    ```

    `affects` has no default (`Affects` gives the reason).

    **The decorator is overloaded on `config=`, so the second parameter of the
    body is checked against the model that the decorator names.** When the
    overload was written, forty-five rows declared a model and read
    `config.field` in the body. Before the overload, a row that declared `Config`
    and annotated `config: OtherConfig` passed the type check, and the difference
    first appeared at the first field read at mount. A row with no model takes
    `None`, and `resolve_config` above makes that annotation true.

    `narrows` says how the copy of this row in a child holds less than the copy
    in its parent (`ph.cordis.child_limit`). It takes the config model too, so it
    is accepted only with one: a row with no config has nothing to narrow except
    whether it runs.

    `log_layer` places a row that keeps, exports or flushes the log (`LogLayer`).
    """

    def decorate(fn: Callable[..., Any]) -> Callable[..., Any]:
        fn.__ph_plugin__ = PluginSpec(  # type: ignore[attr-defined]
            name=name,
            apply=fn,
            affects=affects,
            inject=service_names(inject),
            config_model=config,
            narrows=narrows,
            log_layer=log_layer,
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
    affects = AFFECTS.get(getattr(source, "affects", ""))
    if affects is None:
        raise LoaderError(
            f'plugin "{name}" does not say what its settings shape: give it `affects`, one '
            f"of {', '.join(sorted(AFFECTS))}"
        )
    declared = getattr(source, "log_layer", None)
    log_layer = None if declared is None else _LOG_LAYERS.get(declared)
    if declared is not None and log_layer is None:
        raise LoaderError(
            f'plugin "{name}" has log_layer {declared!r}; give it one of {", ".join(LOG_LAYER)}'
        )
    return PluginSpec(
        name=str(name),
        apply=apply,
        affects=affects,
        inject=service_names(getattr(source, "inject", ()) or ()),
        config_model=config_model,
        log_layer=log_layer,
    )
