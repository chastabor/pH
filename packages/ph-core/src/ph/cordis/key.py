"""`ServiceKey[T]`: the name that a service is provided under, with its type.

`Context.provide("llm", runtime)` and `ctx.llm` are a registry keyed by strings,
and a string carries no type. Every seam call from every row in this tree was
`Any`, because `Context.__getattr__` is. A key is the string plus the type that
the service is provided *as*. So `ctx.provide(LLM, runtime)` is checked against
`LlmRuntime`, and `ctx.require(LLM)` returns one. `ph.seams._registry` had
already made tables and slots "checked end to end". The key is the last hop of
that chain, applied to the registry that those tables depend on.

**This is a leaf module, and other modules depend on that.** `plugin.py`
normalizes an `inject` list of keys and `context.py` imports `plugin.py`, so the
type that both need cannot be in either. The same reasoning applies one level
up. `ph.keys` lists every core key. At run time it imports only this module and
`ph.cordis.loader`, which declares `MOUNT` and `PROJECT_ROOT`. A consumer can
therefore name a service without importing the module that provides it.
Otherwise 145 such imports would have been new, and several of them would have
been cycles (plan P8-05).

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

    `T` is phantom: nothing here holds a `T`. A key is therefore free to declare
    in a module that never imports the type. Under
    `from __future__ import annotations`, the annotation `ServiceKey[FsService]`
    is a string that mypy reads, and the value is `ServiceKey("fs")`.
    """

    name: str


def service_name(key: str | ServiceKey[Any]) -> str:
    """The string that the registry uses for a key, for either spelling of the key."""
    return key if isinstance(key, str) else key.name


def service_names(keys: Sequence[str | ServiceKey[Any]]) -> tuple[str, ...]:
    """`service_name` over a sequence: what an `inject` list becomes.

    The function is here and not at each of the three entry points that take a
    key list (`plugin()`, `normalize_plugin`, `Context._register_dependent`).
    The reason is the one that `ph.seams._names` gives about hand-written
    copies of one regex: the fourth caller copies whichever version it reads
    first. The `getattr(source, "inject", ()) or ()` of `normalize_plugin` was
    already a divergent variant.
    """
    if isinstance(keys, str):
        # `str` satisfies `Sequence[str]`, so `inject="llm"` passes the type
        # check and iterates into `l`, `l`, `m` (A12). The row then waits for a
        # service named `l` that nothing provides. There is no error at compose
        # and none at mount, only a row that never applies. A string is the one
        # spelling that cannot be meant.
        raise TypeError(f'inject must be a list of service names, not the string "{keys}"')
    return tuple([service_name(key) for key in keys])
