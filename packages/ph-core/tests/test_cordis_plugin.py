"""P6-02 — plugin identity: the five shapes a row may be written in.

`normalize_plugin` is the door every plugin comes through, and the *reason* it is
duck-typed rather than a base class is that a third party should be able to
contribute one without importing anything from pH: a module with an `apply`, an
object with three attributes, or a bare function all work. That generosity is
only safe if each shape is actually read the way its author expects, and until
this file only the decorated one was.

The other half is `resolve_config`, which decides what a plugin's `apply` is
handed. Four inputs reach it — nothing, a mapping, a model of the right type, a
model of a *different* type — and the last two are the ones worth pinning: one
must pass through untouched and one must be re-validated, and confusing them
either copies a config needlessly or hands a plugin somebody else's model.
"""

from __future__ import annotations

from typing import Any, ClassVar

import pytest
from pydantic import BaseModel

from ph.cordis import LoaderError
from ph.cordis.plugin import PluginSpec, normalize_plugin, plugin


class _Config(BaseModel):
    depth: int = 1


class _Other(BaseModel):
    depth: int = 9


async def _apply(ctx: Any, config: Any) -> None: ...


# ------------------------------------------------------------- the shapes --


def test_a_decorated_function_carries_its_own_spec() -> None:
    """The shipped form: the decorator stamps the spec onto the function, so the
    row's name lives beside the body rather than in a registry somewhere else."""

    @plugin("tool-fs", inject=["tools", "fs"], config=_Config)
    async def apply(ctx: Any, config: _Config) -> None: ...

    spec = normalize_plugin(apply)

    assert spec.name == "tool-fs"
    assert spec.inject == ("tools", "fs")
    assert spec.config_model is _Config
    assert spec.apply is apply


def test_a_spec_passes_through_itself() -> None:
    """Normalizing twice is not a copy. The loader normalizes what it is given
    and a caller may normalize before handing it over, so idempotence is what
    keeps the two from disagreeing about identity."""
    spec = PluginSpec(name="row", apply=_apply)

    assert normalize_plugin(spec) is spec


def test_a_module_with_a_decorated_apply_is_read_through_it() -> None:
    """The entry-point shape: `ph.plugins` names a *module*, and the decorator is
    on the function inside it. Reading the module's own attributes instead would
    name the row after the module and lose the `inject` list — which is what
    decides whether the row activates at all."""

    class _Module:
        @staticmethod
        @plugin("session", inject=["llm"], config=_Config)
        async def apply(ctx: Any, config: _Config) -> None: ...

    spec = normalize_plugin(_Module)

    assert spec.name == "session" and spec.inject == ("llm",)
    assert spec.config_model is _Config


def test_an_object_with_the_four_attributes_needs_no_import_from_ph() -> None:
    """The reason this is duck-typed at all.

    A third-party plugin can be an ordinary object — `name`, `apply`, `inject`,
    and a nested `Config` — with no dependency on pH's decorator. `Config` is the
    conventional attribute name because a class body is where a plugin author
    already puts its model.
    """

    class _Plugin:
        name = "third-party"
        inject: ClassVar[list[str]] = ["fs"]
        Config = _Config

        @staticmethod
        async def apply(ctx: Any, config: _Config) -> None: ...

    spec = normalize_plugin(_Plugin)

    assert spec.name == "third-party"
    assert spec.inject == ("fs",)
    assert spec.config_model is _Config


def test_a_bare_callable_is_a_plugin_named_after_itself() -> None:
    """The smallest thing that can be a row. No decorator, no attributes: a
    function that takes `(ctx, config)` is enough, and its `__name__` is the
    identity it gets."""

    async def scratch(ctx: Any, config: Any) -> None: ...

    spec = normalize_plugin(scratch)

    assert spec.name == "scratch" and spec.apply is scratch
    assert spec.inject == () and spec.config_model is None


def test_an_object_that_names_itself_beats_its_type_name() -> None:
    """`name` wins over `__name__`, so an instance contributed twice under two
    row ids is two rows rather than one — which is how a profile mounts the same
    plugin class against two different configs."""

    class _Plugin:
        name = "chosen"

        @staticmethod
        async def apply(ctx: Any, config: Any) -> None: ...

    assert normalize_plugin(_Plugin).name == "chosen"


# ------------------------------------------------------------ the refusals --


def test_something_that_is_not_a_plugin_says_so() -> None:
    """The message names the two things it looked for, because the author of a
    row that will not mount is reading this sentence to find out which one they
    forgot."""
    with pytest.raises(LoaderError, match="no apply\\(\\) and not callable"):
        normalize_plugin(object())


def test_a_config_that_is_not_a_model_is_refused_at_the_door() -> None:
    """A `Config` that is a dict or a dataclass would otherwise reach
    `resolve_config` and fail there — inside a mount, named as a validation
    error against a row rather than as the plugin's own mistake.
    """

    class _Plugin:
        name = "wrong-config"
        Config: ClassVar[dict[str, int]] = {"depth": 1}

        @staticmethod
        async def apply(ctx: Any, config: Any) -> None: ...

    with pytest.raises(LoaderError, match="Config that is not a pydantic model"):
        normalize_plugin(_Plugin)


# -------------------------------------------------------------- the config --


def test_a_plugin_without_a_model_receives_the_row_verbatim() -> None:
    """Which is how a row whose config is a plain mapping works at all: there is
    no model to validate against, so the plugin is handed exactly what the
    profile wrote."""
    spec = PluginSpec(name="row", apply=_apply)
    raw = {"anything": [1, 2]}

    assert spec.resolve_config(raw) is raw


def test_an_absent_config_becomes_the_model_s_defaults() -> None:
    """A row with no `config:` block is the common case, and it must reach
    `apply` as a *model* rather than as `None` — every plugin body reads
    `config.field`, and a row that omitted the block would otherwise be the one
    input that crashes it."""
    spec = PluginSpec(name="row", apply=_apply, config_model=_Config)

    resolved = spec.resolve_config(None)

    assert isinstance(resolved, _Config) and resolved.depth == 1


def test_a_mapping_is_validated_into_the_model() -> None:
    spec = PluginSpec(name="row", apply=_apply, config_model=_Config)

    assert spec.resolve_config({"depth": 4}) == _Config(depth=4)


def test_a_model_of_the_right_type_is_not_rebuilt() -> None:
    """Identity, not equality: re-validating would copy a model the caller may
    still hold a reference to, and a plugin that read its config back through
    that reference would be reading a different object."""
    spec = PluginSpec(name="row", apply=_apply, config_model=_Config)
    given = _Config(depth=7)

    assert spec.resolve_config(given) is given


def test_a_model_of_another_type_is_re_validated_through_its_aliases() -> None:
    """The daemon and the loader both pass models around, and the one that
    arrives is not always the one this plugin declared.

    Dumped `by_alias`, because that is the form the field names are declared in —
    a camelCase wire model re-validated through its Python names would silently
    lose every renamed field to its default.
    """
    spec = PluginSpec(name="row", apply=_apply, config_model=_Config)

    resolved = spec.resolve_config(_Other(depth=9))

    assert isinstance(resolved, _Config) and resolved.depth == 9
