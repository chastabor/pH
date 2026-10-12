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

**A binding is called the way the SDK listing shows it.** The listing renders
`async def tools.context_tombstone(sections: list, reason: str)`, so positional
arguments bind to the parameters in the order it declares them, and keywords by
name — the host sends that order (`ph.tools.sdk.parameter_names`), so the two read
one source. A live model wrote `tools.context_tombstone(['S12'], reason=...)` from
the listing and was refused when this took keywords only, losing the step it had to
edit in. More positionals than parameters, or one parameter given both ways, is
refused with the call spelled out.

@module ph_runtime.proxies
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any

from .channel import Model, is_model

__all__ = ["Namespace", "RlmNamespace", "build_namespaces"]

Dispatch = Callable[[str, str, dict[str, Any]], Awaitable[Any]]

RLM_KWARGS = ("name", "model", "output", "thinking", "access")
"""`rlm()`'s keyword arguments. Validated here as well as on the host so a typo
costs a `TypeError` in the cell rather than a round trip and a tool failure.

`access` is pH's one addition to prime-agent's contract, and its default is
`"read"` rather than prime-agent's implicit write (E4, Q11): a research child
that turns out to need writes costs one turn to re-spawn, where a writing child
that should not have written costs a review of every diff it produced."""


class Namespace:
    """A group of governed callables — `tools`, `agent_message`, `agent_observe`."""

    def __init__(
        self, name: str, bindings: Mapping[str, Sequence[str]], dispatch: Dispatch
    ) -> None:
        self._name = name
        self._parameters = {binding: tuple(names) for binding, names in bindings.items()}
        """Each binding, with its parameter names in the order the host declared them."""
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
        """One call frame to the host; a value with no JSON form is the wire's to spell
        (`jsonable`)."""
        return await self._dispatch(self._name, name, arguments)


class RlmNamespace(Namespace):
    """`rlm` is callable *and* a namespace: `await rlm(prompt)`, `rlm.find_models()`.

    **A model class as `output` comes back as an instance.** `rlm.run(..., output=Sort)`
    sends the host `Sort`'s JSON Schema (`jsonable`), which is what the child is held
    to, and remembers the class for that child; `await rlm.wait(handle)` then hands back
    `Sort.model_validate(value)`, so pydantic checks the value once more in the cell's
    own terms.
    """

    RUN = "run"
    WAIT = "wait"

    def __init__(
        self, name: str, bindings: Mapping[str, Sequence[str]], dispatch: Dispatch
    ) -> None:
        super().__init__(name, bindings, dispatch)
        self._shapes: dict[str, type[Model]] = {}
        """The model class each child was asked to return its value as, by child id."""

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
        """A spawn, with a model class `output` remembered for the child it admitted."""
        handle = await super()._call(self.RUN, arguments)
        shape = arguments.get("output")
        child = handle.get("childId") if isinstance(handle, Mapping) else None
        if is_model(shape) and isinstance(child, str):
            self._shapes[child] = shape
        return handle

    async def _wait(self, arguments: dict[str, object]) -> Any:  # noqa: ANN401
        """One child's value, as an instance of the class it was asked for; else the
        value, or the answer of a child asked for no shape."""
        child = arguments.get("child_id")
        if isinstance(child, Mapping):
            # A handle as `run` gave it, or the child's id.
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
    """`arguments`, with the positional ones bound to the parameters in declared order —
    or a refusal that spells the call out."""
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
    """Build one proxy per namespace the `boot` frame declared.

    Driven by the frame rather than by a list here, so a namespace a plugin adds
    reaches the cell without this module changing (I1, I7).
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
