"""Python skills as pre-imported callables.

Ported convention, not ported code (§6.8): prime-agent's `rlm/skill.py` makes a
module with a `run()` callable, so a cell writes `await websearch(query=...)`
rather than `await websearch.run(query=...)`. The wrapper keeps that, and keeps
attribute access working, so both spellings mean the same thing.

An import that fails binds a **stub that explains itself** instead of leaving the
name undefined. A model that finds a name missing reads it as its own mistake and
spends a turn working around it; a stub that says "this skill failed to import,
here is why" is one line of context and no wasted turn.

@module ph_runtime.skill
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

__all__ = ["UnavailableSkill", "wrap_skill_module"]


class _CallableModule:
    """A module whose `run()` is reachable by calling the module itself."""

    def __init__(self, module: ModuleType, entry: Any) -> None:
        self._module = module
        self._entry = entry

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._entry(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._module, name)

    def __dir__(self) -> list[str]:
        return dir(self._module)

    def __repr__(self) -> str:
        return f"<skill {self._module.__name__}>"


class UnavailableSkill:
    """A skill that did not import. Says so when used, and when printed."""

    def __init__(self, name: str, reason: str) -> None:
        self._name = name
        self._reason = reason

    def _fail(self) -> None:
        raise RuntimeError(
            f'the skill "{self._name}" is unavailable: {self._reason}. '
            "It is installed in the runtime venv but did not import; ask for it to be "
            "fixed rather than working around it."
        )

    def __call__(self, *_args: Any, **_kwargs: Any) -> Any:
        self._fail()

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._fail()

    def __repr__(self) -> str:
        return f"<unavailable skill {self._name}: {self._reason}>"


def wrap_skill_module(module: ModuleType) -> Any:
    """The module, callable when it offers a `run()`; otherwise unchanged."""
    entry = getattr(module, "run", None)
    return _CallableModule(module, entry) if callable(entry) else module
