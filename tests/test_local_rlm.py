"""ph-rlm, live: Code Mode on a local server, end to end.

**Opt-in by `LLAMA_BASE_URL`**, like `test_local_server.py`, and run with it by
`./test.sh smoke`. What the fake adapter cannot show:

* **that every request offers one tool, `python`, and a real model calls it.** Until a
  live run found it, Code Mode sent no tool at all: the fake adapter asks for no
  schema, so nothing noticed that a real model had nothing to call.
* **that the interpreter keeps what a cell defined** for the next turn's cell.
* **that a tool called from inside a cell is its own governed call** — a
  `tool/code-dispatch` in the log — and its value reaches the cell.
* **that a shell command from a cell is a governed `bash` call** — pH runs plain
  Python, so there is no `%%bash` magic to go around the pipeline.
* **that `rlm.run` delegates to a child on the same model that can itself run code**,
  in its own log, and that its answer reaches the parent's next turn through the
  inbox. A child once inherited a grant that filtered its transport out, and was
  offered nothing to call.

**Model-following, as the rest of the live gate is**: each prompt names the code to
run, and a model that will not take that instruction is not one to run an agent on.
What the harness owes once the model does — the dispatch record, the interpreter's
state, the child's log, the inbox — is asserted, not the model's wording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from clm_helpers import model_text
from local_server import LIVE, cells_after, live_agent, output_of
from runtime_helpers import dispatch_names

from ph.agent_loop.driver import ReactLoopAgent
from ph.keys import SESSIONS, SUBAGENTS
from ph.llm.types import GenerateOptions
from ph.testing import MountProfile
from ph_rlm.presentation import PYTHON

pytestmark = [pytest.mark.anyio, LIVE]

CHILD_SECONDS = 300.0
"""How long a delegated child may take to settle: one agent loop on a 27B model."""


def _cells(agent: ReactLoopAgent, after: int) -> tuple[str, str]:
    """Every cell the model ran after seq `after`: their programs and their outputs,
    each joined — a model may look around in a cell of its own before the one it was
    told to run."""
    calls = cells_after(agent.session, after)
    assert all(call.data.get("name") == PYTHON for call in calls), [
        call.data.get("name") for call in calls
    ]
    programs = "\n".join(str(call.data.get("arguments")) for call in calls)
    return programs, "\n".join(output_of(agent.session, call) for call in calls)


async def test_the_interpreter_keeps_a_variable_for_the_next_turns_cell(
    mount: MountProfile, route: str
) -> None:
    """And every request on the way offers exactly one tool, `python`."""
    ctx, agent = await live_agent(mount, route, "rlm", "rlm-live-kernel")
    offered: list[list[str]] = []

    async def spy(options: GenerateOptions, next_: Any) -> Any:  # noqa: ANN401
        if options.is_loop_request:
            offered.append([tool.name for tool in options.tools])
        return await next_(options)

    ctx.on("llm/stream", spy)
    await agent.prompt(
        "Run one python cell: secret = 'quo' + 'kka'. Print nothing. "
        "Then reply with the single word: set."
    )
    asked = len(agent.session.events)

    await agent.prompt("Run one python cell: print(secret). Then reply with what it printed.")

    assert offered and all(tools == [PYTHON] for tools in offered), offered
    program, output = _cells(agent, asked)
    assert "quokka" not in program, "the cell defined the value again rather than reading it"
    assert "quokka" in output, output


async def test_a_tool_called_from_a_cell_is_its_own_governed_call(
    mount: MountProfile, route: str, tmp_path: Path
) -> None:
    (tmp_path / "note.txt").write_text("the secret word is marmalade\n", encoding="utf-8")
    _ctx, agent = await live_agent(mount, route, "rlm", "rlm-live-binding")

    await agent.prompt(
        "Run one python cell: print(await tools.read(path='note.txt')). "
        "Then reply with the secret word."
    )

    assert "read" in dispatch_names(agent.session), "the cell's read was not a tool call"
    _program, output = _cells(agent, 0)
    assert "marmalade" in output, output


async def test_a_shell_command_from_a_cell_is_a_governed_bash_call(
    mount: MountProfile, route: str
) -> None:
    """pH runs plain Python: a shell is `tools.bash`, governed like any other tool, and
    there is no `%%bash` magic to go around the pipeline — a cell that tries one is
    told so and pointed here."""
    _ctx, agent = await live_agent(mount, route, "rlm", "rlm-live-bash")

    await agent.prompt(
        "Run one python cell: print(await tools.bash(command='echo hello-from-bash')). "
        "Then reply with the single word: done."
    )

    assert "bash" in dispatch_names(agent.session), "the cell's shell command was not bash"
    _program, output = _cells(agent, 0)
    assert "hello-from-bash" in output, output


async def test_a_child_delegated_from_a_cell_runs_code_and_answers_its_parent(
    mount: MountProfile, route: str
) -> None:
    """The second agent loop: `rlm.run` admits a child and returns at once; the child
    runs on the same model, in its own log, runs a cell of its own and settles; its
    answer — a message it sent, or the notice of its last words — reaches the parent's
    next turn."""
    ctx, agent = await live_agent(mount, route, "rlm", "rlm-live-child")

    await agent.prompt(
        "Run one python cell: h = await rlm.run(prompt='Run one python cell: "
        "print(\\'marma\\' + \\'lade\\'). Then reply with what it printed.', "
        "name='scout'); print(h). Then reply with the single word: started."
    )

    assert "rlm_run" in dispatch_names(agent.session), "the cell's rlm.run was not dispatched"
    (run,) = ctx.require(SUBAGENTS).list(parent_id=agent.id)
    assert run.result is not None
    with anyio.fail_after(CHILD_SECONDS):
        outcome = await run.result()
    assert outcome.status == "done", f"the child ended {outcome.status}: {outcome.error}"
    child_log = ctx.require(SESSIONS).get(run.session_id)
    assert child_log is not None, "the child's own log is not in the store"
    child_cells = [one for one in child_log.select("tool/call") if one.data.get("name") == PYTHON]
    assert child_cells, "the child ran no cell: was it offered its transport?"
    asked = len(agent.session.events)

    await agent.prompt("What did the scout say? Reply with its word only.")

    assert any(
        event.type == "agent/inbox/spliced" and event.seq > asked for event in agent.session.events
    ), "the child's answer was never spliced into the parent's turn"
    assert "marmalade" in model_text(agent.session).lower()
