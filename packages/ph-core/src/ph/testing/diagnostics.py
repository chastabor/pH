"""Reads one section from the report of `phern doctor`.

`report()` returns `list[(title, rows)]`. Three test modules in three packages
unwrapped that shape by hand. When a title changed, two of them raised a bare
`KeyError` and did not say which sections existed. With one helper, a renamed
section fails with the list of sections in the message.

@module ph.testing.diagnostics
"""

from __future__ import annotations

from ..cordis import Context
from ..keys import DIAGNOSTICS

__all__ = ["report_section"]


def report_section(ctx: Context, title: str) -> dict[str, str]:
    """One section of `ctx.diagnostics.report()`, as label → value."""
    sections = dict(ctx.require(DIAGNOSTICS).report())
    assert title in sections, f"no {title!r} section in {list(sections)}"
    return dict(sections[title])
