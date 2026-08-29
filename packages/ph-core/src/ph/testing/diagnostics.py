"""Reading one section out of `ph doctor`'s report.

`report()` returns `list[(title, rows)]` — a shape three test modules in three
packages were unwrapping by hand, two of them raising a bare `KeyError` when a
title changed rather than saying which sections did exist. One helper, so a
renamed section fails with the list in the message.

@module ph.testing.diagnostics
"""

from __future__ import annotations

from typing import Any

__all__ = ["report_section"]


def report_section(ctx: Any, title: str) -> dict[str, str]:
    """One section of `ctx.diagnostics.report()`, as label → value."""
    sections = dict(ctx.diagnostics.report())
    assert title in sections, f"no {title!r} section in {list(sections)}"
    return dict(sections[title])
