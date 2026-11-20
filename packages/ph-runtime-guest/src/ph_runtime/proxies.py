"""What the model sees in `globals()`: namespaces in which every call is governed.

The programming model is the model of prime-agent, deliberately and exactly:
`await rlm(...)`, `await agent_message.send(msg, receiver_role="parent")`, `await
tools.read(path=...)`, and skills as pre-imported callables. That model is the
real contribution of prime-agent, and it is what the RLM doctrine teaches, so it
is kept unchanged.

Everything below it changed. Each of these calls sends **one `call` frame** to
the host. There the call re-enters the full tool pipeline as a sub-call and
settles as a durable `tool/code-dispatch` record. Prime-agent reached the host
over an `ipykernel.Comm`, which no `tools/pre-execute` listener, no approval, and
no call limit ever observed. No such channel exists here. The governed path is
therefore not a convention. It is the only path.

Unknown attributes and unknown keyword arguments fail with an error that lists
the available names. The reader is a model. With a silent `None`, a cell would
use a turn to discover that a capability that the model invented does not exist.

**A binding is called in the way that the SDK listing shows it.** The listing
renders `async def tools.context_tombstone(sections: list, reason: str)`.
Positional arguments therefore bind to the parameters in the order that the
listing declares, and keyword arguments bind by name. The host sends that order
(`ph.tools.sdk.parameter_names`), so both sides read one source. A live model
wrote `tools.context_tombstone(['S12'], reason=...)` from the listing. At that
time this module accepted keywords only, so the call was refused, and the model
lost the step that it needed for the edit. A call with more positional arguments
than parameters, or with one parameter given both ways, is refused with a message
that shows the full call.

@module ph_runtime.proxies
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .channel import Model, is_model

__all__ = ["Namespace", "RlmNamespace", "build_namespaces"]

Dispatch = Callable[[str, str, dict[str, Any]], Awaitable[Any]]

RLM_KWARGS = ("name", "model", "output", "thinking", "access")
"""The keyword arguments of `rlm()`. They are validated here as well as on the
host, so that a typo costs a `TypeError` in the cell and not a round trip and a
tool failure.

`access` is the one addition of pH to the contract of prime-agent. Its default is
`"read"`, where prime-agent has an implicit write (E4, Q11). A research child
that needs writes costs one turn to spawn again. A writing child that should not
have written costs a review of every diff that it produced."""


class Namespace:
    """A group of governed callables, such as `tools`, `agent_message`, and
    `agent_observe`."""

    def __init__(
        self, name: str, bindings: Mapping[str, Sequence[str]], dispatch: Dispatch
    ) -> None:
        self._name = name
        self._parameters = {binding: tuple(names) for binding, names in bindings.items()}
        """Each binding, with its parameter names in the order that the host
        declared them."""
        self._bindings = tuple(self._parameters)
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

        parameters = self._parameters[name]

        async def call(*positional: object, **arguments: object) -> Any:  # noqa: ANN401
            if positional:
                arguments = _bound(f"{self._name}.{name}", parameters, positional, arguments)
            return await self._call(name, arguments)

        call.__name__ = name
        call.__qualname__ = f"{self._name}.{name}"
        return call

    async def _call(self, name: str, arguments: dict[str, object]) -> Any:  # noqa: ANN401
        """Send one call frame to the host. The wire decides how to spell a value
        that has no JSON form (`jsonable`)."""
        return await self._dispatch(self._name, name, arguments)


class RlmNamespace(Namespace):
    """`rlm` is callable *and* a namespace: `await rlm(prompt)`, `rlm.find_models()`.

    **A model class given as `output` comes back as an instance.**
    `rlm.run(..., output=Sort)` sends the JSON Schema of `Sort` to the host
    (`jsonable`). The child is held to that schema. The namespace remembers the
    class for that child. `await rlm.wait(handle)` then returns
    `Sort.model_validate(value)`, so pydantic checks the value one more time, in
    the terms of the cell.
    """

    RUN = "run"
    WAIT = "wait"

    def __init__(
        self, name: str, bindings: Mapping[str, Sequence[str]], dispatch: Dispatch
    ) -> None:
        super().__init__(name, bindings, dispatch)
        self._shapes: dict[str, type[Model]] = {}
        """The model class that each child must return its value as, keyed by child id."""

    async def __call__(self, prompt: str, **kwargs: object) -> Any:  # noqa: ANN401
        unknown = sorted(set(kwargs) - set(RLM_KWARGS))
        if unknown:
            raise TypeError(
                f"rlm() got unexpected keyword argument(s) {', '.join(unknown)}; "
                f"accepted: {', '.join(RLM_KWARGS)}"
            )
        return await self._call(self.RUN, {"prompt": prompt, **kwargs})

    async def _call(self, name: str, arguments: dict[str, object]) -> Any:  # noqa: ANN401
        if name == self.RUN:
            return await self._run(arguments)
        if name == self.WAIT:
            return await self._wait(arguments)
        return await super()._call(name, arguments)

    async def _run(self, arguments: dict[str, object]) -> Any:  # noqa: ANN401
        """Spawn a child. If `output` is a model class, remember it for the child
        that was admitted."""
        handle = await super()._call(self.RUN, arguments)
        shape = arguments.get("output")
        child = handle.get("childId") if isinstance(handle, Mapping) else None
        if is_model(shape) and isinstance(child, str):
            self._shapes[child] = shape
        return handle

    async def _wait(self, arguments: dict[str, object]) -> Any:  # noqa: ANN401
        """Return the value of one child as an instance of the class that was
        requested for it. With no class, return the value. For a child that was
        asked for no shape, return its answer."""
        child = arguments.get("child_id")
        if isinstance(child, Mapping):
            # The argument is a handle as `run` returned it, or the id of the child.
            child = child.get("childId")
        settled = await super()._call(self.WAIT, {**arguments, "child_id": child})
        if not isinstance(settled, Mapping):
            return settled
        value = settled.get("value")
        if value is None:
            return settled.get("answer")
        shape = self._shapes.get(child) if isinstance(child, str) else None
        return shape.model_validate(value) if shape is not None else value


def _bound(
    call: str,
    parameters: Sequence[str],
    positional: Sequence[object],
    arguments: dict[str, object],
) -> dict[str, object]:
    """Return `arguments` with the positional arguments bound to the parameters
    in declared order, or raise a refusal that shows the full call."""
    spelled = f"{call}({', '.join(parameters)})"
    if len(positional) > len(parameters):
        raise TypeError(
            f"{call}() takes {len(parameters)} argument(s), {len(positional)} given "
            f"positionally: {spelled}"
        )
    bound = dict(zip(parameters, positional, strict=False))
    twice = sorted(set(bound) & set(arguments))
    if twice:
        raise TypeError(f"{call}() got {', '.join(twice)} both positionally and by name: {spelled}")
    return {**bound, **arguments}


def build_namespaces(
    declared: Sequence[dict[str, Any]], dispatch: Dispatch
) -> dict[str, Namespace]:
    """Build one proxy for each namespace that the `boot` frame declared.

    The frame drives this, not a list in this module. A namespace that a plugin
    adds therefore reaches the cell without a change to this module (I1, I7).
    """
    built: dict[str, Namespace] = {}
    for namespace in declared:
        name = namespace.get("name")
        if not isinstance(name, str) or not name:
            continue
        bindings = {
            binding["name"]: [
                one for one in binding.get("parameters") or () if isinstance(one, str)
            ]
            for binding in namespace.get("bindings") or ()
            if isinstance(binding, dict) and isinstance(binding.get("name"), str)
        }
        factory = RlmNamespace if name == "rlm" else Namespace
        built[name] = factory(name, bindings, dispatch)
    return built
