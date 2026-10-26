"""Plugin identity: what a row mounts, and how its config is validated.

A plugin is a name, what its settings shape (`Affects`), a list of injected
service keys, an optional pydantic config model, and an `apply(ctx, config)`
body. It may be written as a decorated function or as any object carrying those
attributes — the shape is duck-typed by `normalize_plugin`, not enforced by a
base class.

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
"""What a row's settings shape, and so which configuration owns them.

* `environment` — what the agent sees or can do: its model, tools, skills, sandbox,
  workspace and posture. A session's profile covers these rows: what it starts
  with, what is overridden during it, and what a restart compares.
* `presentation` — only how a front end draws: screens and footer readings. The
  TUI's configuration owns these.
* `deployment` — the host's own machinery: persistence, telemetry, invariants,
  doctor sections, the local stores, the credential source. The daemon's own
  configuration owns these.

Declared by every plugin, with no default, so a row that does not say is caught
where it is written (the decorator's required keyword) rather than guessed at
when a profile is split by owner. `plans/Session_Profiles_Plan.md` is the design.
"""

AFFECTS: Mapping[str, Affects] = literal_lookup(Affects)
"""Every `Affects` by its spelling, so checking a plugin's word also types it."""

CONFIGURED_IN: Mapping[Affects, str] = {
    "environment": "a session profile, `$PH_HOME/profiles/<name>.yaml`",
    "presentation": "the TUI's settings, `$PH_HOME/tui.json`, where a screen is hidden or "
    "its key remapped",
    "deployment": "the daemon's configuration, `$PH_HOME/daemon.yaml`",
}
"""Where a person sets each kind, as the refusal of a misplaced row names it.

In words rather than paths, because two of the three are a front end's and a
host's files that `ph.cordis` never reads. Written once, so the loader's refusal
and the documentation cannot name different files for one kind.
"""

LogLayer: TypeAlias = Literal["storage", "telemetry", "export", "checkpoint"]
"""A row's place in the log layer: the rows that keep the log, export it and say when
it reaches disk, beneath every row that writes to it.

Rows unwind in reverse, so the layer goes last, and in reverse of this order. The
checkpoint policy's last barrier flushes what the agents appended as they let go,
while every other row here is still mounted. An `export` row, a sink or a redactor
of what sinks receive, is there to handle that tail. The telemetry seam outlives
them, since they register with it. Storage goes last, so no flush announces a write
after it went.

**Declared, so nobody has to remember where to write the row.** The composition
gathers the layer where its first row is written, in this order
(`ph.cordis.loader`), so a deployment adds a sink by naming it, and a row appended
at the end, which is all a profile can do, still lands beneath the agents. A
`storage` row with no `checkpoint` row is warned about at mount, since nothing then
says when the log reaches disk.
"""

LOG_LAYER: tuple[LogLayer, ...] = get_args(LogLayer)

_LOG_LAYERS: Mapping[str, LogLayer] = literal_lookup(LogLayer)


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
    affects: Affects
    """Which configuration owns this row's settings (`Affects`)."""
    inject: tuple[str, ...] = ()
    config_model: type[BaseModel] | None = None
    narrows: Callable[..., ChildLimit] | None = None
    """How a child holds less of this row (`ph.cordis.child_limit`), or `None` when
    it holds this row as its parent runs it or not at all. Erased as `apply` is, and
    held to the config model by the same overload."""
    log_layer: LogLayer | None = None
    """This row's place in the log layer (`LogLayer`), or `None` for a row outside it."""

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

    `affects` has no default (`Affects` says why).

    **Overloaded on `config=`, so the body's second parameter is checked against
    the model the decorator names.** Forty-five rows declare a model and read
    `config.field` in the body; before this, a row that declared `Config` and
    annotated `config: OtherConfig` type-checked, and the first field read at
    mount was where the difference surfaced. A row with no model takes `None`,
    and `resolve_config` above is what makes that annotation true.

    `narrows` is how a child's copy of this row holds less than its parent's
    (`ph.cordis.child_limit`), and takes the config model too, so it comes only with
    one: a row with no config has nothing to narrow but whether it runs.

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
