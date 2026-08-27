"""The one way these tests run a cell.

`run_tool` in `ph.testing` is generic over tools; the `{"program": ...}` shape is
`run_code`'s, so it lives here rather than being spelled out at every call site.
"""

from __future__ import annotations

from typing import Any

from ph.testing import run_tool
from ph.tools.registry import RUN_CODE

DISPATCH_START = "tool/code-dispatch-start"
DISPATCH_SETTLED = "tool/code-dispatch"
"""The two durable records C2 is about, named once rather than spelled as bare
strings in five test modules across two packages."""


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


async def run_ipython_cell(
    ctx: Any, program: str, *, agent: Any, session: Any = None, call_id: str = "c1"
) -> Any:
    """`run_cell` under the name the RLM profile presents (`ipython`).

    Three modules wanted this two-liner; the reserved `run_code` stops resolving
    once `rlm-presentation` is mounted, so a test on that profile has to name the
    transport the model sees.
    """
    from ph_rlm.presentation import IPYTHON

    return await run_cell(ctx, program, agent=agent, session=session, call_id=call_id, name=IPYTHON)


def dispatch_names(session: Any) -> list[str]:
    """The governed tools a session's cells dispatched, in submission order."""
    return [event.data["name"] for event in session.events if event.type == DISPATCH_START]


def settled_dispatches(session: Any) -> list[Any]:
    """The settled halves of those dispatches, in the order they settled."""
    return [event for event in session.events if event.type == DISPATCH_SETTLED]
