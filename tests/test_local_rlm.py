"""ph-rlm, live: Code Mode on a local server, end to end.

**Opt-in by `LLAMA_BASE_URL`**, like `test_local_server.py`, and run with it by
`./test.sh smoke`. What the fake adapter cannot show:

* **that the request offers one tool, `ipython`, and a real model calls it.** Until a
  live run found it, Code Mode sent no tool at all: the fake adapter asks for no
  schema, so nothing noticed that a real model had nothing to call.
* **that the kernel keeps what a cell defined** for the next turn's cell.
* **that a tool called from inside a cell is its own governed call** — a
  `tool/code-dispatch` in the log — and its value reaches the cell.
* **that a shell command from a cell is a governed `bash` call** — pH runs plain
  Python, so there is no `%%bash` magic to go around the pipeline.
* **that `rlm.run` delegates to a child on the same model**, the child answers in its
  own log, and the answer reaches the parent's next turn through its inbox.

**Model-following, as the rest of the live gate is**: each prompt names the code to
run, and a model that will not take that instruction is not one to run an agent on.
What the harness owes once the model does — the dispatch record, the kernel's state,
the child's log, the inbox — is asserted, not the model's wording.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
import pytest
from local_server import LIVE, calls_after, local_model, output_of

from ph.agent.types import AgentOptions
from ph.agent_loop.driver import ReactLoopAgent
from ph.cordis import Context
from ph.keys import AGENTS, SESSIONS, SUBAGENTS
from ph.llm.types import GenerateOptions, text_of
from ph.seams.subagents import ChildState
from ph.testing import MountProfile
from ph_app.profiles import PROFILE_DIR, resolve_profile
from ph_rlm.presentation import IPYTHON

pytestmark = [pytest.mark.anyio, LIVE]

CHILD_SECONDS = 300.0
"""How long a delegated child may take to settle: one agent loop on a 27B model."""

SETTLED = ("done", "error", "canceled")


@pytest.fixture
def route(monkeypatch: pytest.MonkeyPatch) -> str:
    """The model to ask for, with the route's credential defaulted (`local_model`)."""
    return local_model(monkeypatch)


async def _agent(mount: MountProfile, route: str, name: str) -> tuple[Context, ReactLoopAgent]:
    """The shipped `rlm` over the shipped llama route, resolved inside the test (the
    suite pins its own home per test; at import this would read the person's)."""
    ctx = await mount(profile=[*resolve_profile("rlm"), PROFILE_DIR / "llama.yaml"])
    session = ctx.require(SESSIONS).create(name)
    agent = ctx.require(AGENTS).create(
        session, AgentOptions(provider="llama", model=route, max_tokens=2048)
    )
    assert isinstance(agent, ReactLoopAgent)
    return ctx, agent


def _dispatched(agent: ReactLoopAgent, after: int) -> list[str]:
    """The governed tools cells dispatched after seq `after`, in submission order."""
    return [
        str(event.data.get("name"))
        for event in agent.session.select("tool/code-dispatch-start")
        if event.seq > after
    ]


def _cells(agent: ReactLoopAgent, after: int) -> tuple[str, str]:
    """Every cell the model ran after seq `after`: their programs and their outputs,
    each joined — a model may look around in a cell of its own before the one it was
    told to run."""
    calls = calls_after(agent.session, after)
    assert calls, "the model was told to run a cell by name and ran none"
    assert all(call.data.get("name") == IPYTHON for call in calls), [
        call.data.get("name") for call in calls
    ]
    programs = "\n".join(str(call.data.get("arguments")) for call in calls)
    return programs, "\n".join(output_of(agent.session, call) for call in calls)


async def test_the_model_is_offered_ipython_and_runs_a_cell(
    mount: MountProfile, route: str
) -> None:
    ctx, agent = await _agent(mount, route, "rlm-live-cell")
    offered: list[list[str]] = []

    async def spy(options: GenerateOptions, next_: Any) -> Any:  # noqa: ANN401
        if options.is_loop_request:
            offered.append([tool.name for tool in options.tools])
        return await next_(options)

    ctx.on("llm/stream", spy)

    await agent.prompt("Run one ipython cell: print(6 * 7). Then reply with the number.")

    assert offered and all(tools == [IPYTHON] for tools in offered), offered
    _program, output = _cells(agent, 0)
    assert "42" in output, output


async def test_the_kernel_keeps_a_variable_for_the_next_turns_cell(
    mount: MountProfile, route: str
) -> None:
    _ctx, agent = await _agent(mount, route, "rlm-live-kernel")
    await agent.prompt(
        "Run one ipython cell: secret = 'quo' + 'kka'. Print nothing. "
        "Then reply with the single word: set."
    )
    asked = len(agent.session.events)

    await agent.prompt("Run one ipython cell: print(secret). Then reply with what it printed.")

    program, output = _cells(agent, asked)
    assert "quokka" not in program, "the cell defined the value again rather than reading it"
    assert "quokka" in output, output


async def test_a_tool_called_from_a_cell_is_its_own_governed_call(
    mount: MountProfile, route: str, tmp_path: Path
) -> None:
    (tmp_path / "note.txt").write_text("the secret word is marmalade\n", encoding="utf-8")
    _ctx, agent = await _agent(mount, route, "rlm-live-binding")

    await agent.prompt(
        "Run one ipython cell: print(await tools.read(path='note.txt')). "
        "Then reply with the secret word."
    )

    assert "read" in _dispatched(agent, 0), "the cell's read was not dispatched as a tool call"
    _program, output = _cells(agent, 0)
    assert "marmalade" in output, output


async def test_a_shell_command_from_a_cell_is_a_governed_bash_call(
    mount: MountProfile, route: str
) -> None:
    """pH runs plain Python: a shell is `tools.bash`, governed like any other tool, and
    there is no `%%bash` magic to go around the pipeline — a cell that tries one is
    told so and pointed here."""
    _ctx, agent = await _agent(mount, route, "rlm-live-bash")

    await agent.prompt(
        "Run one ipython cell: print(await tools.bash(command='echo hello-from-bash')). "
        "Then reply with the single word: done."
    )

    assert "bash" in _dispatched(agent, 0), "the cell's shell command was not a bash call"
    _program, output = _cells(agent, 0)
    assert "hello-from-bash" in output, output


async def test_a_child_delegated_from_a_cell_answers_its_parent(
    mount: MountProfile, route: str
) -> None:
    """The second agent loop: `rlm.run` admits a child and returns at once; the child
    runs on the same model, in its own log, and settles; its answer — a message it
    sent, or the notice of its last words — reaches the parent's next turn."""
    ctx, agent = await _agent(mount, route, "rlm-live-child")

    await agent.prompt(
        "Run one ipython cell: h = await rlm.run(prompt='Reply with the single word: "
        "marmalade. Run no code.', name='scout'); print(h). Then reply with the single "
        "word: started."
    )

    assert "rlm_run" in _dispatched(agent, 0), "the cell's rlm.run was not dispatched"
    child = await _settled(ctx, agent)
    assert child.status == "done", f"the child ended {child.status}: {child.detail}"
    child_log = ctx.require(SESSIONS).get(child.session_id)
    assert child_log is not None, "the child's own log is not in the store"
    assert any(event.type == "turn/end" for event in child_log.events), "the child never ran"
    asked = len(agent.session.events)

    await agent.prompt("What did the scout say? Reply with its word only.")

    delivered = "\n".join(
        text_of(message.content) for message in agent.session.derive_messages()[-6:]
    )
    assert any(
        event.type == "agent/inbox/spliced" and event.seq > asked for event in agent.session.events
    ), "the child's answer was never spliced into the parent's turn"
    assert "marmalade" in delivered.lower(), delivered[-600:]


async def _settled(ctx: Context, agent: ReactLoopAgent) -> ChildState:
    """The parent's one child, once it has settled."""
    subagents = ctx.require(SUBAGENTS)
    with anyio.fail_after(CHILD_SECONDS):
        while True:
            children = list(subagents.children(agent.session.id).values())
            if children and children[0].status in SETTLED:
                return children[0]
            await anyio.sleep(1)
