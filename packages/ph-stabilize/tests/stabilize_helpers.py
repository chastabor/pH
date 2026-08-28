"""Shared scaffolding for the `stabilize` bundle's tests.

A module of its own rather than `conftest.py`, mirroring `ph-app`'s
`tui_helpers`. This package has **no** conftest, and deliberately: `sys.modules`
holds one slot named `conftest`, so adding a second one that pytest loads after
ph-rlm's took the slot from it and every `from conftest import ...` in that
suite failed to collect. Its one piece of shared setup is a plain function rather than a
fixture, so a test calls it by name instead of importing a decorated symbol it
then shadows with a parameter of the same name.
"""

from __future__ import annotations

from typing import Any

import pytest

from ph.bundles import BASE, HEADLESS
from ph.seams.spill import SpillStore
from ph_stabilize import BUNDLE

__all__ = ["PROFILE", "blob", "break_spill"]

PROFILE = [BASE, HEADLESS, BUNDLE]
"""Base, the fake adapter, and this bundle — what a profile layering it gets."""


def blob(size: int, *, lines: int = 40) -> str:
    """Exactly `size` characters over about `lines` lines.

    Exact, because the boundary tests turn on one character either side of a
    threshold, and a generator that overshot would be testing a number nobody
    chose. The line breaks are what give `content_preview` a middle to omit.
    """
    chunk = max(1, size // lines)
    return "\n".join("x" * chunk for _ in range(lines)).ljust(size, "y")[:size]


def break_spill(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every spill write fail, for the fail-open gates.

    Patched on the class: `SpillStore` is a slots dataclass, so the instance has
    no room for an override.
    """

    async def refuse(_self: Any, **_kwargs: Any) -> Any:
        raise OSError("no space left on device")

    monkeypatch.setattr(SpillStore, "save_text", refuse)
