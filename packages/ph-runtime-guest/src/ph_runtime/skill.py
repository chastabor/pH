"""Python skills as pre-imported callables.

This is a ported convention, not ported code (§6.8). The `rlm/skill.py` of
prime-agent makes a module with a `run()` function callable. A cell then writes
`await websearch(query=...)` and not `await websearch.run(query=...)`. The
wrapper keeps that convention and keeps attribute access working, so both
spellings mean the same thing.

An import that fails binds a **stub that explains itself**. It does not leave the
name undefined. A model that finds a name missing reads that as its own mistake,
and uses a turn to work around it. A stub that says "this skill failed to import,
and here is why" costs one line of context and no turn.

@module ph_runtime.skill
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Any, NoReturn

__all__ = ["UnavailableSkill", "wrap_skill_module"]


class _CallableModule:
    """A module whose `run()` can be reached by calling the module itself."""

    def __init__(self, module: ModuleType, entry: Callable[..., Any]) -> None:
        self._module = module
        self._entry = entry

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        return self._entry(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:  # noqa: ANN401
        return getattr(self._module, name)

    def __dir__(self) -> list[str]:
        return dir(self._module)

    def __repr__(self) -> str:
        return f"<skill {self._module.__name__}>"


class UnavailableSkill:
    """A skill that did not import. It says so when it is used and when it is printed."""

    def __init__(self, name: str, reason: str) -> None:
        self._name = name
        self._reason = reason

    def _fail(self) -> NoReturn:
        raise RuntimeError(
            f'the skill "{self._name}" is unavailable: {self._reason}. '
            "It is installed in the runtime venv but did not import; ask for it to be "
            "fixed rather than working around it."
        )

    def __call__(self, *_args: object, **_kwargs: object) -> NoReturn:
        self._fail()

    def __getattr__(self, name: str) -> NoReturn:
        if name.startswith("_"):
            raise AttributeError(name)
        self._fail()

    def __repr__(self) -> str:
        return f"<unavailable skill {self._name}: {self._reason}>"


def wrap_skill_module(module: ModuleType) -> Any:  # noqa: ANN401
    """Return the module, made callable when it has a `run()`. Otherwise return it
    unchanged."""
    entry = getattr(module, "run", None)
    return _CallableModule(module, entry) if callable(entry) else module
