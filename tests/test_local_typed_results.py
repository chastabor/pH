"""Typed child results, live: a sorting child on the local model returns a checked value.

**Opt-in by `LLAMA_BASE_URL`**, and run by `./test.sh smoke`, as the rest of the live
gate is. What the scripted tests cannot show is whether a real model, told it owes a
result in a set shape, hands one back: through `tools.submit_result` from a cell, or as a
reply that is the JSON object. Either counts, and which one it took is in the child's own
log (`subagent/returned`'s `via`), so a model that stops using the tool shows up here as
a change in that, not as a failure.

Two sortings, each by a skill that declares its `output:`, the way a person would set one
up: a report sorted as a bug or a feature, and a picture sorted by its color, which the
child has to attach to see. Model-following, like the rest of the gate: each input has
one obvious answer, and what is asserted is the harness's part, a value that checks
against the skill's shape, plus that obvious answer.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import anyio
import pytest
from local_server import LIVE, live_agent

from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.keys import SESSIONS, SKILLS, SUBAGENTS
from ph.seams.skills import read_skill
from ph.seams.subagents import RETURNED, SubagentRequest, SubagentResult
from ph.testing import KIND_OUTPUT, MountProfile, not_none, write_skill
from ph_rlm.subagents import PROVIDER_NAME

pytestmark = [pytest.mark.anyio, LIVE]

CHILD_SECONDS = 420.0
"""How long one sorting child may take: a few steps on a 27B model on a laptop, where a
step that thinks to its token cap alone takes minutes. Measured at about a hundred
seconds for the report."""

KIND_SKILL = """\
Sort the report you are given. It is a `bug` when something that should work is
broken, and a `feature` when it asks for something new.
"""

COLOR_SKILL = """\
Sort the picture you are given by its color. Attach it to see it, with
`await tools.attach(path=...)` in a python cell, then return the color you see.
"""

COLOR_OUTPUT: dict[str, Any] = {
    "type": "object",
    "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}},
    "required": ["color"],
    "additionalProperties": False,
}
"""What a color-sorting child returns."""


def _swatch(path: Path, rgb: tuple[int, int, int], side: int = 64) -> None:
    """A solid-color PNG, written without an imaging library: one obvious answer."""
    row = b"\x00" + bytes(rgb) * side
    pixels = zlib.compress(row * side)

    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    header = struct.pack(">IIBBBBB", side, side, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", pixels) + chunk(b"IEND", b"")
    )


def _installed(ctx: Context, root: Path, name: str, body: str, output: dict[str, Any]) -> None:
    """A sorting skill as a person installs one: a `SKILL.md` with a body and its
    `output:` (JSON being YAML). The body matters: a child given a skill with nothing to
    read goes looking for its instructions on disk, which is most of what a run spends."""
    extra = f"output: {json.dumps(output)}"
    written = write_skill(root, name, description=name.replace("-", " "), body=body, extra=extra)
    found = not_none(read_skill(written / "SKILL.md"))
    assert found.output == output, f"{name}'s output was not read"
    ctx.require(SKILLS).register(found)


async def _sorted(
    ctx: Context, parent: ReactLoopAgent, prompt: str, sorting: str
) -> tuple[SubagentResult, str]:
    """One sorting child, run to its end: its result, and how it handed its value back."""
    run = await ctx.require(SUBAGENTS).start(
        PROVIDER_NAME, SubagentRequest(prompt=prompt, parent=parent, skills=(sorting,))
    )
    with anyio.fail_after(CHILD_SECONDS):
        result = await not_none(run.result)()
    child = not_none(ctx.require(SESSIONS).get(run.session_id))
    via = [str(event.data.get("via")) for event in child.events if event.type == RETURNED]
    return result, ",".join(via) or "nothing returned"


async def test_a_sorting_child_returns_its_report_s_kind_in_the_skill_s_shape(
    mount: MountProfile, route: str, tmp_path: Path
) -> None:
    """The text half: the skill's `output:` reaches the child, which returns a value the
    shape checks. Sabotage: drop the `output` from the skill, and the child owes nothing
    and returns no value."""
    ctx, parent = await live_agent(mount, route, "rlm", "rlm-live-sort-text")
    _installed(ctx, tmp_path / "skills", "sort-report", KIND_SKILL, KIND_OUTPUT)

    result, via = await _sorted(
        ctx,
        parent,
        "Sort this report as a bug or a feature: 'The app crashes with a stack trace "
        "every time I press Save.'",
        "sort-report",
    )

    assert result.status == "done", f"{result.error} (returned: {via}; said: {result.answer})"
    assert result.value is not None and result.value["kind"] == "bug", (result.value, via)


async def test_a_sorting_child_sees_a_picture_and_returns_its_color(
    mount: MountProfile, route: str, tmp_path: Path
) -> None:
    """The image half: the child attaches the picture to see it, on a route that takes
    images, and returns its color in the skill's shape."""
    _swatch(tmp_path / "swatch.png", (220, 20, 20))
    ctx, parent = await live_agent(mount, route, "rlm", "rlm-live-sort-image")
    _installed(ctx, tmp_path / "skills", "sort-color", COLOR_SKILL, COLOR_OUTPUT)

    result, via = await _sorted(
        ctx,
        parent,
        "Sort the picture swatch.png by its color.",
        "sort-color",
    )

    assert result.status == "done", f"{result.error} (returned: {via}; said: {result.answer})"
    assert result.value is not None and result.value["color"] == "red", (result.value, via)
