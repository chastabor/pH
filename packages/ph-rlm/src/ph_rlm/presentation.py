"""`rlm-presentation` — the one callable the RLM model sees (P3-09, C1/C6).

Prime Agent's model-facing registry has exactly one entry, `ipython`, and that
single fact is what C1-C3 exist to undo: one tool meant one permission
evaluation, one approval prompt and one call-limit tick per cell, no matter how
many files the cell wrote. Code Mode restores the per-capability boundary while
keeping the surface — so this row does not add a tool. It *renames the transport*
and states how a settled cell reads.

Three things, and the reason each is here rather than in `ph-core`:

* **The name and description.** `run_code` is the reserved transport name and
  stays reserved (nothing may occupy it, C6); `ipython` is how this profile
  presents it, with prime-agent's wording ported verbatim so a model that knows
  that surface finds the surface it knows. The mechanism is
  `ctx.tools.present_transport`, which is scoped like every other registration.

* **The result text.** Prime Agent concatenates
  `stdout + "\\n" + stderr + "\\n" + result + "\\n" + traceback`, and a model
  trained against that layout reads ours the same way. pH's runtime interleaves
  the two streams in arrival order rather than concatenating them separately —
  a cell that prints, warns, then prints again is *misread* when the warning is
  moved to the end — so what survives is the ordering of the four sections, not
  the stream split. That deviation is deliberate and stated here because it is
  the kind of thing a later reader would file as a porting bug.

* **The details payload.** `IpythonToolDetails` is what drives the code-cell
  widget (P3-19). It is computed from the durable result alone, like every other
  presentation projection, so a replayed cell draws the same card as a live one
  (A11).

@module ph_rlm.presentation
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ph.cordis import Context, plugin
from ph.text import count_of
from ph.tools.code_mode import CodeCellValue
from ph.tools.definition import ToolOutput, ToolResult, TransportPresentation, text_content
from ph.tools.presentation import ToolCallView, ToolResultView
from ph.wire import WireModel

__all__ = ["IPYTHON", "IPYTHON_DESCRIPTION", "IpythonToolDetails", "apply", "render_cell"]

IPYTHON = "ipython"
"""What the RLM profile calls the transport. `run_code` is still the reserved
name, and still what `register_transport` claims."""

IPYTHON_DESCRIPTION = (
    "Python scratchpad code or `%%bash` shell cells to execute in the agent "
    "kernel. Use the target project's own environment for project imports, "
    "tests, scripts, CLIs, and dependency checks instead of direct kernel "
    "imports."
)
"""Ported verbatim from prime-agent's `src/core/tools/ipython.ts`.

Verbatim on purpose: the description is the contract the model was trained
against, and paraphrasing it is a silent behaviour change with no test that
would catch it.

**It says `%%bash` and pH has no magics.** That is a known, deliberate
inconsistency, not an oversight — D19 removed the magic *because* it was the one
shell nothing could see, and this sentence is the trained contract rather than a
statement of fact. The correction is the doctrine's job (`ph_rlm.prompt`), which
states plainly that there are no magics and names `await tools.bash(...)`; the
conformance suite asserts the doctrine and the runtime agree. Do not "fix" this
string to match — changing it changes what the model was trained against, and
`test_presentation.py` pins it verbatim for that reason."""


class IpythonToolDetails(WireModel):
    """The durable card payload for one cell.

    Every field is derived from the settled result, never from live execution
    state — the widget renders this during streaming *and* during a replay, and
    the two must agree.
    """

    status: str
    """`ok` | `error` — what the model was told the cell did."""
    dispatches: int = 0
    """Binding calls this cell made. The number C2 exists to make non-zero: under
    prime-agent's single tool these were invisible."""
    truncated: bool = False
    attachments: int = 0
    """`display` payloads the program emitted. A count, not the payloads: the
    card shows how many there are and fetches one when asked."""
    reset: bool = False
    """The kernel had died and this cell got a fresh, empty namespace."""


def render_cell(_args: Any, value: Any) -> list[Any]:
    """Prime Agent's four sections, in its order, minus its stream split.

    Absent sections are dropped rather than left as blank lines: a model reading
    three empty lines above a traceback has to work out that they mean nothing.
    """
    parts: list[str] = []
    logs = str(value.get("logs") or "").rstrip()
    if logs:
        parts.append(logs)
    result = value.get("value")
    if result is not None:
        parts.append(f"[result] {result!r}")
    error = value.get("error")
    if error:
        parts.append(str(error))
    return text_content("\n".join(parts) if parts else "(no output)")


def cell_details(_args: Any, value: Any) -> Any:
    return IpythonToolDetails(
        status="error" if value.get("error") else "ok",
        dispatches=int(value.get("dispatches") or 0),
        truncated=bool(value.get("truncated")),
        attachments=len(value.get("displays") or ()),
        reset=bool(value.get("reset")),
    ).to_wire()


def _program(args: Any) -> str:
    """The cell text, from arguments that may not be a mapping.

    The guard is load-bearing: the TUI adapter feeds these `parse_arguments`
    output, which is the raw *string* when the model's JSON was malformed.
    """
    return str(args.get("program", "")) if hasattr(args, "get") else ""


def _present_call(args: Any) -> ToolCallView:
    # A bounded prefix: the card wants the program, not every line of a
    # generated file these views re-materialize per replayed cell. `input` is the
    # header line, `body` the program the code cell renders under it (P3-19).
    head = _program(args)[:2048]
    first = next((line for line in head.splitlines() if line.strip()), "")
    return ToolCallView(card="terminal", title=IPYTHON, input=first, body=head)


def _present_result(args: Any, result: ToolResult) -> ToolResultView:
    program = _program(args)
    count = program.count("\n") + (0 if program.endswith("\n") or not program else 1)
    subtitle = count_of(count, "line")
    return ToolResultView(
        card="terminal",
        title=IPYTHON,
        subtitle=subtitle,
        is_error=result.is_error,
        # `Mapping`, not `dict`: a live result's meta arrives frozen as a
        # `MappingProxyType`, which is a Mapping and is *not* a dict instance —
        # so a `dict` test silently dropped the payload the card is drawn from.
        meta=dict(result.meta) if isinstance(result.meta, Mapping) else None,
    )


@plugin("rlm-presentation", inject=["tools"])
async def apply(ctx: Context, _config: Any) -> None:
    """Present the transport as `ipython`, with the RLM cell projections.

    `present_as("code")` is *not* called here. The mode is the profile's, set by
    the bundle's `tools` row, and claiming it again would put two rows on one
    cell — where the first disposal clears what the second still wants.
    """
    ctx.tools.present_transport(
        TransportPresentation(
            name=IPYTHON,
            description=IPYTHON_DESCRIPTION,
            output=ToolOutput(
                schema=CodeCellValue,
                render=render_cell,
                presentation_meta=cell_details,
            ),
            present_call=_present_call,
            present_result=_present_result,
        )
    )
