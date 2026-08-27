"""The one way these tests run a cell.

`run_tool` in `ph.testing` is generic over tools; the `{"program": ...}` shape is
`run_code`'s, so it lives here rather than being spelled out at every call site.
"""

from __future__ import annotations

from typing import Any

from ph.testing import run_tool
from ph.tools.registry import RUN_CODE


async def run_cell(
    ctx: Any,
    program: str,
    *,
    agent: Any,
    session: Any = None,
    call_id: str = "call-1",
    name: str = RUN_CODE,
) -> Any:
    """Execute one cell through the real transport and pipeline.

    `name` is for the profile that presents the transport as `ipython` —
    the reserved name stops resolving once the rename is mounted.
    """
    return await run_tool(
        ctx, name, {"program": program}, agent=agent, session=session, call_id=call_id
    )
