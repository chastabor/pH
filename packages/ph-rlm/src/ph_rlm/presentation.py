"""`rlm-presentation`: the one callable that the RLM model sees (P3-09, C1/C6).

The model-facing registry of prime-agent has exactly one entry, its `ipython`
tool. C1-C3 exist to undo the effect of that one fact. One tool meant one
permission evaluation, one approval prompt and one call-limit tick for each
cell, however many files the cell wrote. Code Mode restores the boundary for
each capability and keeps the surface of one callable. This row therefore does
not add a tool. It *renames the transport* and states how a settled cell reads.

The row does three things. Each has a reason to be here and not in `ph-core`:

* **The name and description.** `run_code` is the reserved transport name and
  stays reserved (nothing may occupy it, C6). `python` is the name under which
  this profile presents it. The description is **in the words of pH and
  describes the runtime of pH**, not the runtime of prime-agent. pH owns its
  interpreter and does not run IPython (D19). The name of prime-agent and its
  "`%%bash` shell cells" described an environment that the model is not in, and
  the doctrine had to contradict the tool that it introduced. The model now gets
  one account of where it is. The mechanism is `ctx.tools.present_transport`,
  which is scoped like every other registration.

* **The result text.** Prime-agent concatenates
  `stdout + "\\n" + stderr + "\\n" + result + "\\n" + traceback`, and a model
  that was trained against that layout reads the layout of pH in the same way.
  The runtime of pH interleaves the two streams in arrival order and does not
  concatenate them separately. A cell that prints, warns, then prints again is
  *misread* when the warning is moved to the end. What is kept is therefore the
  order of the four sections, not the stream split. That difference is
  deliberate. It is stated here because a later reader would otherwise report it
  as a porting bug.

* **The details payload.** `CellDetails` drives the code-cell widget (P3-19). It
  is computed from the durable result alone, like every other presentation
  projection. A replayed cell therefore draws the same card as a live cell
  (A11).

@module ph_rlm.presentation
"""

from __future__ import annotations

from typing import Any

from ph.characters import head
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
"""The name that the RLM profile gives the transport. `run_code` is still the
reserved name, and it is still the name that `register_transport` claims.

The name is not the `ipython` of prime-agent. The cell runs in the interpreter
of pH itself, which has no magics, no `!` and no Jupyter display hooks. A name
that promised IPython promised an environment that the model is not in."""

MAGICS = ", ".join(f"`{prefix}`" for prefix in MAGIC_PREFIXES)
"""The prefixes that the guest refuses, rendered for the model.

The text is derived and not typed again. Otherwise a prefix that was added to
the runtime would leave the tool and the doctrine describing a different runtime
from the one that the model talks to. That is the drift that made the doctrine
promise `%%bash` shell cells for a phase (P3-22)."""

PYTHON_DESCRIPTION = (
    "Run a Python program in your own interpreter and get back what it printed and "
    "the value of its last expression. Variables, imports and functions persist from "
    "one call to the next. This is plain Python run by pH, not IPython or Jupyter: "
    f"there are no magics ({MAGICS}). Everything else is an `await` on a binding the "
    "SDK lists, each one governed and recorded; a shell is "
    "`await tools.bash(command=...)`. The interpreter is pH's, not the project's "
    "environment: run the project's own tests, scripts and CLIs through `tools.bash`."
)
"""The transport as the model reads it: the one account that the model has of
the runtime.

It says what persists, what is absent (the magics that the guest refuses,
`ph_runtime.cell.MAGIC_PREFIXES`), how everything else is reached, and where the
environment of the project is. The doctrine (`ph_rlm.prompt`) points here and
does not repeat it. The doctrine states only the rules. The conformance suite
checks this text against the runtime."""


class CellDetails(WireModel):
    """The durable card payload for one cell.

    Every field is derived from the settled result, never from live execution
    state. The widget renders this payload during streaming *and* during a
    replay, and the two must agree.
    """

    status: str
    """`ok` | `error`: what the model was told that the cell did."""
    dispatches: int = 0
    """The binding calls that this cell made. C2 exists to make this number
    non-zero: under the single tool of prime-agent, these calls were invisible."""
    truncated: bool = False
    attachments: int = 0
    """The number of `display` payloads that the program emitted. It is a count
    and not the payloads: the card shows how many there are and fetches one on
    request."""
    reset: bool = False
    """The kernel had died and this cell got a fresh, empty namespace."""


def render_cell(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    """The four sections of prime-agent, in its order, without its stream split.

    Absent sections are dropped and not left as blank lines. A model that reads
    three empty lines above a traceback must work out that they mean nothing.
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
    """The cell text. Both entry points have already narrowed the arguments.

    This function used to guard with `hasattr(args, "get")`. At that time,
    `render_call_view` and `render_result_view` gave a presenter whatever
    `parse_arguments` returned, which was the raw *string* when the JSON of the
    model was malformed. Both now narrow with `as_obj`, so a string arrives as no
    arguments. The guard checked something that can no longer reach this
    function.
    """
    return str(args.get("program", ""))


def _present_call(args: JsonObject) -> ToolCallView:
    # A bounded prefix. The card needs the program, not every line of a generated
    # file that these views materialize again for each replayed cell. `input` is
    # the header line. `body` is the program that the code cell renders under it
    # (P3-19).
    shown = head(_program(args), 2048)
    first = next((line for line in shown.splitlines() if line.strip()), "")
    return ToolCallView(card="terminal", title=PYTHON, input=first, body=shown)


def _present_result(args: JsonObject, result: ToolResult) -> ToolResultView:
    program = _program(args)
    count = program.count("\n") + (0 if program.endswith("\n") or not program else 1)
    subtitle = count_of(count, "line")
    return ToolResultView(
        card="terminal",
        title=PYTHON,
        subtitle=subtitle,
        is_error=result.is_error,
        # `dict(...)` because the meta of a live result arrives frozen as a
        # `MappingProxyType`, and `ToolResultView.meta` is a real dict. An
        # `isinstance` check used to guard this. The declaration now does that
        # job.
        meta=None if result.meta is None else dict(result.meta),
    )


@plugin("rlm-presentation", affects="environment", inject=[TOOLS])
async def apply(ctx: Context, config: None) -> None:
    """Present the transport as `python`, with the RLM cell projections.

    `present_as("code")` is *not* called here. The mode belongs to the profile,
    and the `tools` row of the bundle sets it. A second claim would put two rows
    on one cell, and the first disposal would clear what the second row still
    needs.
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
