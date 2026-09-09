"""P8-05 — `ph.keys` and the services the tree actually provides, held against each other.

A key module is a hand-written list, and the failure of a hand-written list is
quiet: a row starts providing `ctx.widgets`, nobody adds `WIDGETS`, and every
reader of it is back to `Any` without a word said. `ph.seams._registry` says why
such a list is worth a test rather than a habit; this is that test for the one
list every other typed read hangs off.

Both directions, for `test_docs_seams`' reason: a key with no provider is a
reader following a name to nothing, and a provider with no key is the hole this
phase exists to close.
"""

from __future__ import annotations

import ast
import re
from functools import cache
from pathlib import Path

import pytest

import ph.keys as core_keys
from ph.cordis import Context, ServiceKey, ServiceNotFoundError, service_name

REPO = Path(__file__).resolve().parents[3]
SOURCES = sorted(REPO.glob("packages/*/src/**/*.py"))

pytestmark = pytest.mark.anyio


@cache
def _provided_names() -> dict[str, set[str]]:
    """Every service name the tree provides, and the modules that provide it.

    Cached: both tests below ask, and the walk parses all 259 files under
    `packages/*/src` — 337 ms, which is half this module's runtime paid twice.
    """
    found: dict[str, set[str]] = {}
    for path in SOURCES:
        if path.name == "keys.py":
            continue
        module = str(path).split("src/")[1].removesuffix(".py").replace("/", ".")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "provide"
                and len(node.args) >= 2
            ):
                continue
            key = node.args[0]
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                found.setdefault(key.value, set()).add(module)
            elif isinstance(key, ast.Name):
                found.setdefault(key.id.lower(), set()).add(module)
    return found


@cache
def _declared_keys() -> dict[str, str]:
    """Every `NAME: ServiceKey[...] = ServiceKey("name")` in the tree, name -> module.

    Cached for `_provided_names`' reason: two callers, one answer, 259 files."""
    declared: dict[str, str] = {}
    pattern = re.compile(r'^([A-Z_]+): ServiceKey\[[^\]]+\] = ServiceKey\("([a-z_]+)"\)', re.M)
    for path in SOURCES:
        module = str(path).split("src/")[1].removesuffix(".py").replace("/", ".")
        for constant, name in pattern.findall(path.read_text(encoding="utf-8")):
            assert constant == name.upper(), f"{module}: {constant} spells {name!r}"
            declared[name] = module
    return declared


def test_every_provided_service_has_a_key() -> None:
    """The direction the phase is for: a provider with no key is an `Any` read."""
    provided = _provided_names()
    declared = _declared_keys()
    # `agent` is provided into each agent's own scope by the registry rather than
    # by a row; it has a key (`AGENT`) and is listed here for the reader.
    missing = sorted(name for name in provided if name not in declared)
    assert missing == [], f"provided but no ServiceKey declares it: {missing}"


def test_every_key_names_a_service_something_provides() -> None:
    """The other direction: a key nothing provides is a name that resolves to nothing."""
    provided = _provided_names()
    declared = _declared_keys()
    orphaned = sorted(name for name in declared if name not in provided)
    assert orphaned == [], f"declared but nothing provides it: {orphaned}"


def test_core_keys_are_all_exported_and_spelled_by_their_name() -> None:
    """`__all__` and the declarations, held together.

    Not a re-export rule — `no_implicit_reexport` governs *imported* names, and
    39 of the 41 keys are defined here, so it reaches only the two that are not:
    `MOUNT` and `PROJECT_ROOT`, both declared beside the `Profile.mount` that
    provides them. This is the
    repo's `__all__`-on-every-module convention made checkable for the one module
    whose contents are a list somebody extends by hand. The spelling rule is
    `_declared_keys`' job, over every source file rather than this one.
    """
    exported = set(core_keys.__all__)
    for attribute in dir(core_keys):
        if isinstance(getattr(core_keys, attribute), ServiceKey):
            assert attribute in exported, f"ph.keys.{attribute} is not in __all__"


async def test_a_key_and_its_string_are_one_registry_entry() -> None:
    """Interop is the migration's whole premise: a row still providing by string
    is read by a consumer asking with the key, and the other way round."""
    ctx = Context()
    ctx.provide("tools", "by-string")
    assert ctx.require(core_keys.TOOLS) == "by-string"  # type: ignore[comparison-overlap]
    assert ctx.get(core_keys.TOOLS) == "by-string"  # type: ignore[comparison-overlap]
    assert ctx.has(core_keys.TOOLS) and ctx.has("tools")
    assert service_name(core_keys.TOOLS) == "tools" == service_name("tools")
    await ctx.dispose()


async def test_require_refuses_by_name_and_get_answers_none() -> None:
    ctx = Context()
    assert ctx.get(core_keys.LLM) is None
    stand_in = object()
    assert ctx.get("llm", stand_in) is stand_in
    with pytest.raises(ServiceNotFoundError, match='no service "llm"'):
        ctx.require(core_keys.LLM)
    await ctx.dispose()


async def test_inject_accepts_keys_and_strings_alike() -> None:
    """`inject=[LLM]` and `inject=["llm"]` wait on the same name."""
    ctx = Context()
    seen: list[str] = []
    ctx.inject([core_keys.LLM, "sessions"], lambda scope: seen.append(scope.path))
    await ctx.reconcile()
    assert seen == []
    ctx.provide(core_keys.LLM, object())
    ctx.provide("sessions", object())
    await ctx.reconcile()
    assert len(seen) == 1
    await ctx.dispose()
