"""`ServiceKey[T]` — the name a service is provided under, carrying its type.

`Context.provide("llm", runtime)` and `ctx.llm` are a registry keyed by strings,
and a string carries no type: every seam call from every row in this tree was
`Any`, because `Context.__getattr__` is. A key is the string plus the type the
service is provided *as*, so `ctx.provide(LLM, runtime)` is checked against
`LlmRuntime` and `ctx.require(LLM)` hands back one — the last hop of a chain
that `ph.seams._registry` had already made "checked end to end" for tables and
slots, applied to the registry those tables hang off.

**A leaf module, and that is load-bearing.** `plugin.py` normalizes an `inject`
list of keys and `context.py` imports `plugin.py`, so the type both need cannot
live in either. The same reasoning scales up: `ph.keys` declares every core
key and imports nothing at runtime but this, which is what lets a consumer name
a service without importing the module that provides it — 145 such imports
would otherwise have been new, several of them cycles (plan P8-05).

@module ph.cordis.key
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

__all__ = ["ServiceKey", "service_name", "service_names"]


@dataclass(frozen=True, slots=True)
class ServiceKey[T]:
    """The name under which a `T` is provided.

    `T` is phantom — nothing here holds one — which is what makes a key free to
    declare in a module that never imports the type: under
    `from __future__ import annotations` the annotation `ServiceKey[FsService]`
    is a string mypy reads, and the value is `ServiceKey("fs")`.
    """

    name: str


def service_name(key: str | ServiceKey[Any]) -> str:
    """The registry's string for a key, whichever spelling arrived."""
    return key if isinstance(key, str) else key.name


def service_names(keys: Sequence[str | ServiceKey[Any]]) -> tuple[str, ...]:
    """`service_name` over a sequence — what an `inject` list becomes.

    Here rather than at each of the three entry points that take a key list
    (`plugin()`, `normalize_plugin`, `Context._register_dependent`), for the
    reason `ph.seams._names` gives about one regex written three times: the
    fourth caller copies whichever it reads first, and `normalize_plugin`'s
    `getattr(source, "inject", ()) or ()` was already a divergent variant.
    """
    return tuple([service_name(key) for key in keys])
