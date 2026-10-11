"""`rlm-presentation` — the one callable the RLM model sees (P3-09, C1/C6).

Prime Agent's model-facing registry has exactly one entry, its `ipython` tool, and
that single fact is what C1-C3 exist to undo: one tool meant one permission
evaluation, one approval prompt and one call-limit tick per cell, no matter how
many files the cell wrote. Code Mode restores the per-capability boundary while
keeping the one-callable surface — so this row does not add a tool. It *renames
the transport* and states how a settled cell reads.

Three things, and the reason each is here rather than in `ph-core`:

* **The name and description.** `run_code` is the reserved transport name and
  stays reserved (nothing may occupy it, C6); `python` is how this profile
  presents it. **In pH's own words, describing pH's own runtime** — not
  prime-agent's: pH owns its interpreter rather than running IPython (D19), so
  prime-agent's name and its "`%%bash` shell cells" described an environment the
  model is not in, and the doctrine had to contradict the tool it was introducing.
  The model gets one account of where it is. The mechanism is
  `ctx.tools.present_transport`, which is scoped like every other registration.

* **The result text.** Prime Agent concatenates
  `stdout + "\\n" + stderr + "\\n" + result + "\\n" + traceback`, and a model
  trained against that layout reads ours the same way. pH's runtime interleaves
  the two streams in arrival order rather than concatenating them separately —
  a cell that prints, warns, then prints again is *misread* when the warning is
  moved to the end — so what survives is the ordering of the four sections, not
  the stream split. That deviation is deliberate and stated here because it is
  the kind of thing a later reader would file as a porting bug.

* **The details payload.** `CellDetails` is what drives the code-cell
  widget (P3-19). It is computed from the durable result alone, like every other
  presentation projection, so a replayed cell draws the same card as a live one
  (A11).

@module ph_rlm.presentation
"""

from __future__ import annotations

from typing import Any

from ph.cordis import Context, plugin
from ph.json import JsonObject, as_bool, as_int, as_str
from ph.keys import TOOLS
from ph.llm.types import ContentBlock
from ph.text import count_of
from ph.tools.code_mode import CodeCellValue
from ph.tools.definition import ToolOutput, ToolResult, TransportPresentation, text_content
from ph.tools.presentation import ToolCallView, ToolResultView
from ph.wire import WireModel
from ph_runtime.cell import MAGIC_PREFIXES

__all__ = ["MAGICS", "PYTHON", "PYTHON_DESCRIPTION", "CellDetails", "apply", "render_cell"]

PYTHON = "python"
"""What the RLM profile calls the transport. `run_code` is still the reserved
name, and still what `register_transport` claims.

Not prime-agent's `ipython`: the cell runs in pH's own interpreter, which has no
magics, no `!`, no Jupyter display hooks, so a name promising IPython promised an
environment the model is not in."""

MAGICS = ", ".join(f"`{prefix}`" for prefix in MAGIC_PREFIXES)
"""The prefixes the guest actually refuses, rendered for the model.

Derived rather than restated: a prefix added to the runtime would otherwise leave
the tool and the doctrine describing a different runtime from the one the model is
talking to — which is exactly the drift that had the doctrine promising `%%bash`
shell cells for a phase (P3-22)."""

PYTHON_DESCRIPTION = (
    "Run a Python program in your own interpreter and get back what it printed and "
    "the value of its last expression. Variables, imports and functions persist from "
    "one call to the next. This is plain Python run by pH, not IPython or Jupyter: "
    f"there are no magics ({MAGICS}). Everything else is an `await` on a binding the "
    "SDK lists, each one governed and recorded; a shell is "
    "`await tools.bash(command=...)`. The interpreter is pH's, not the project's "
    "environment: run the project's own tests, scripts and CLIs through `tools.bash`."
)
"""The transport as the model reads it: the one account of the runtime it has.

What persists, what is not there (the magics the guest refuses,
`ph_runtime.cell.MAGIC_PREFIXES`), how everything else is reached, and where the
project's own environment is. The doctrine (`ph_rlm.prompt`) points here rather than
saying it again, and states only the rules; the conformance suite holds this to the
runtime."""


class CellDetails(WireModel):
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


def render_cell(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    """Prime Agent's four sections, in its order, minus its stream split.

    Absent sections are dropped rather than left as blank lines: a model reading
    three empty lines above a traceback has to work out that they mean nothing.
    """
    parts: list[str] = []
    logs = as_str(value.get("logs")).rstrip()
    if logs:
        parts.append(logs)
    result = value.get("value")
    if result is not None:
        parts.append(f"[result] {result!r}")
    error = value.get("error")
    if error:
        parts.append(str(error))
    return text_content("\n".join(parts) if parts else "(no output)")


def cell_details(_args: JsonObject, value: Any) -> JsonObject:  # noqa: ANN401
    return CellDetails(
        status="error" if value.get("error") else "ok",
        dispatches=as_int(value.get("dispatches")),
        truncated=as_bool(value.get("truncated")),
        attachments=len(value.get("displays") or ()),
        reset=as_bool(value.get("reset")),
    ).to_wire()


def _program(args: JsonObject) -> str:
    """The cell text. Both entry points have narrowed the arguments already.

    This guarded `hasattr(args, "get")` for as long as `render_call_view` and
    `render_result_view` handed a presenter whatever `parse_arguments` returned
    — the raw *string* when the model's JSON was malformed. Both narrow with
    `as_obj` now, so a string arrives as no arguments and the guard was checking
    something that can no longer reach it.
    """
    return str(args.get("program", ""))


def _present_call(args: JsonObject) -> ToolCallView:
    # A bounded prefix: the card wants the program, not every line of a
    # generated file these views re-materialize per replayed cell. `input` is the
    # header line, `body` the program the code cell renders under it (P3-19).
    head = _program(args)[:2048]
    first = next((line for line in head.splitlines() if line.strip()), "")
    return ToolCallView(card="terminal", title=PYTHON, input=first, body=head)


def _present_result(args: JsonObject, result: ToolResult) -> ToolResultView:
    program = _program(args)
    count = program.count("\n") + (0 if program.endswith("\n") or not program else 1)
    subtitle = count_of(count, "line")
    return ToolResultView(
        card="terminal",
        title=PYTHON,
        subtitle=subtitle,
        is_error=result.is_error,
        # `dict(...)` because a live result's meta arrives frozen as a
        # `MappingProxyType` and `ToolResultView.meta` is a real dict; the
        # `isinstance` this used to guard with is now the declaration's job.
        meta=None if result.meta is None else dict(result.meta),
    )


@plugin("rlm-presentation", affects="environment", inject=[TOOLS])
async def apply(ctx: Context, config: None) -> None:
    """Present the transport as `python`, with the RLM cell projections.

    `present_as("code")` is *not* called here. The mode is the profile's, set by
    the bundle's `tools` row, and claiming it again would put two rows on one
    cell — where the first disposal clears what the second still wants.
    """
    ctx.require(TOOLS).present_transport(
        TransportPresentation(
            name=PYTHON,
            description=PYTHON_DESCRIPTION,
            output=ToolOutput(
                schema=CodeCellValue,
                render=render_cell,
                presentation_meta=cell_details,
            ),
            present_call=_present_call,
            present_result=_present_result,
        )
    )
