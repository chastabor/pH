"""What the model sees in `globals()`: namespaces whose every call is governed.

The programming model is prime-agent's, deliberately and exactly — `await
rlm(...)`, `await agent_message.send(msg, receiver_role="parent")`, `await
tools.read(path=...)`, skills as pre-imported callables. That is prime-agent's
genuine contribution and it is what the RLM doctrine teaches, so it is preserved
verbatim.

What changed is everything underneath. Each of these calls marshals **one `call`
frame** to the host, where it re-enters the full tool pipeline as a sub-call and
settles as a durable `tool/code-dispatch` record. Prime-agent reached the host
over an `ipykernel.Comm`, which no `tools/pre-execute` listener, no approval and
no call limit ever observed; there is no such channel here, so the governed path
is not a convention but the only path that exists.

Unknown attributes and unknown keyword arguments fail loudly, with the available
names in the message. The reader is a model, and a silent `None` is how a cell
spends a turn discovering that a capability it invented does not exist.

@module ph_runtime.proxies
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from typing import Any

__all__ = ["Namespace", "RlmNamespace", "build_namespaces"]

Dispatch = Callable[[str, str, dict[str, Any]], Awaitable[Any]]

RLM_KWARGS = ("name", "model", "thinking", "access")
"""`rlm()`'s keyword arguments. Validated here as well as on the host so a typo
costs a `TypeError` in the cell rather than a round trip and a tool failure.

`access` is pH's one addition to prime-agent's contract, and its default is
`"read"` rather than prime-agent's implicit write (E4, Q11): a research child
that turns out to need writes costs one turn to re-spawn, where a writing child
that should not have written costs a review of every diff it produced."""


class Namespace:
    """A group of governed callables — `tools`, `agent_message`, `agent_observe`."""

    def __init__(self, name: str, bindings: Sequence[str], dispatch: Dispatch) -> None:
        self._name = name
        self._bindings = tuple(bindings)
        self._dispatch = dispatch

    def __repr__(self) -> str:
        return f"<{self._name}: {', '.join(self._bindings)}>"

    def __dir__(self) -> list[str]:
        return sorted(self._bindings)

    def __getattr__(self, name: str) -> Callable[..., Awaitable[Any]]:
        if name.startswith("_") or name not in self._bindings:
            raise AttributeError(
                f"{self._name}.{name} does not exist. Available: "
                f"{', '.join(sorted(self._bindings)) or '(none)'}"
            )

        async def call(**arguments: Any) -> Any:
            return await self._dispatch(self._name, name, arguments)

        call.__name__ = name
        call.__qualname__ = f"{self._name}.{name}"
        return call


class RlmNamespace(Namespace):
    """`rlm` is callable *and* a namespace: `await rlm(prompt)`, `rlm.find_models()`."""

    RUN = "run"

    async def __call__(self, prompt: str, **kwargs: Any) -> Any:
        unknown = sorted(set(kwargs) - set(RLM_KWARGS))
        if unknown:
            raise TypeError(
                f"rlm() got unexpected keyword argument(s) {', '.join(unknown)}; "
                f"accepted: {', '.join(RLM_KWARGS)}"
            )
        return await self._dispatch(self._name, self.RUN, {"prompt": prompt, **kwargs})


def build_namespaces(
    declared: Sequence[dict[str, Any]], dispatch: Dispatch
) -> dict[str, Namespace]:
    """Build one proxy per namespace the `boot` frame declared.

    Driven by the frame rather than by a list here, so a namespace a plugin adds
    reaches the cell without this module changing (I1, I7).
    """
    built: dict[str, Namespace] = {}
    for namespace in declared:
        name = namespace.get("name")
        if not isinstance(name, str) or not name:
            continue
        bindings = [
            binding["name"]
            for binding in namespace.get("bindings") or ()
            if isinstance(binding, dict) and isinstance(binding.get("name"), str)
        ]
        factory = RlmNamespace if name == "rlm" else Namespace
        built[name] = factory(name, bindings, dispatch)
    return built
