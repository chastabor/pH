"""The first half of `test_mass_restart.py`: a deployment caught mid-work, then killed.

Run as its own process — `python tests/mass_restart_host.py <project> [keyed]` — so the
test can `SIGKILL` it: the case where no teardown of ours runs at all, which an
in-process "restart" cannot reach. `keyed` puts the two sub-agents on a route of their
own, `keyed`, which the restarting process gives a credential this one never needed
(T5). It puts one of each kind of in-flight work on disk, prints
the ids the test needs as one JSON line, and waits to be killed.

In flight when it prints:

* the root's turn, with a `write` recorded as started and its file on disk, parked
  in `tools/post-execute` so no result is ever appended;
* a `!!` command, opened through `ctx.intents` and on disk before it "runs";
* an approval put to an answerer that never answers;
* a keyed tool call, its effect opened and its body holding;
* two sub-agents under a concurrency of one: the first at the model, the second
  queued behind it.

Nothing here asserts anything; the test does, against what a fresh process makes of
the logs this one leaves behind.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import Any

import anyio

from ph.agent.types import AgentOptions
from ph.bundles import BASE, HEADLESS
from ph.cordis import Context, Profile, load_profile_documents
from ph.json import as_str
from ph.keys import AGENTS, APPROVAL, INTENTS, LLM, SESSIONS, SUBAGENTS, TOOLS
from ph.llm.adapter import ResolvedModel
from ph.llm.fake import FakeAdapter
from ph.llm.types import (
    BlockEnd,
    BlockStart,
    Finish,
    FinishReason,
    GenerateOptions,
    StreamChunk,
    ToolCallBlock,
)
from ph.seams.subagents import SubagentRequest, subagent_roster
from ph.session.kinds import SHELL_COMMAND
from ph.testing import simple_tool
from ph.tools import ToolExecution, ToolExecutionInput
from ph_rlm.subagents import PROVIDER_NAME

ROOT = "root"
PLAN = "the plan, written before the crash"
SCRIPTED = AgentOptions(provider="scripted", model="s1")
PROVIDER_ROW = {"id": "rlm-subagent-provider", "name": "rlm-subagent-provider"}


class Scripted:
    """The root asks for one `write`; every other request waits to be killed."""

    def __init__(self) -> None:
        self.asked = False

    async def stream(self, options: GenerateOptions) -> AsyncIterator[StreamChunk]:
        if options.session_id == ROOT and not self.asked:
            self.asked = True
            arguments = json.dumps({"path": "notes.md", "content": PLAN})
            yield BlockStart(index=0, block_type="tool-call")
            yield BlockEnd(index=0, block=ToolCallBlock(id="w1", name="write", arguments=arguments))
            yield Finish(reason=FinishReason(kind="tool-calls"))
            return
        await anyio.sleep_forever()
        yield Finish(reason=FinishReason(kind="stop"))  # pragma: no cover

    def resolve_model(self, provider: str, model: str) -> ResolvedModel:
        return ResolvedModel(context_window=8192)


async def _held(*_args: object) -> Any:  # noqa: ANN401
    await anyio.sleep_forever()


async def _park_the_write(
    execution: ToolExecution, result: object, next_: Callable[[], Awaitable[object]]
) -> object:
    """The `write` has landed; its result never reaches the log."""
    if execution.name == "write":
        await anyio.sleep_forever()
    return await next_()


async def main(project: Path, children_on: str) -> None:
    documents = load_profile_documents([BASE, HEADLESS])
    documents.append(("host-root", [{"id": "sandbox-local", "disabled": True}]))
    documents.append(("host-overlay", [dict(PROVIDER_ROW, config={"maxConcurrent": 1})]))
    ctx = Context()
    await Profile.from_documents(documents).mount(ctx, project=project)

    ctx.require(LLM).register_adapter(("scripted", "keyed"), Scripted())
    FakeAdapter.stream = _held  # type: ignore[assignment,method-assign]
    ctx.on("tools/post-execute", _park_the_write)
    ctx.require(TOOLS).register(
        simple_tool("send", _held, idempotency_key=lambda args: as_str(args.get("id")))
    )
    ctx.require(APPROVAL).register_answerer(_held)

    sessions = ctx.require(SESSIONS)
    root = sessions.create(ROOT)
    agent = ctx.require(AGENTS).create(root, SCRIPTED)

    async with anyio.create_task_group() as tasks:
        tasks.start_soon(agent.prompt, "write the plan down")
        await ctx.require(INTENTS).open(
            root, SHELL_COMMAND, {"command": "make release", "surface": False}
        )
        tasks.start_soon(
            lambda: ctx.require(APPROVAL).request(agent=agent, tool_name="publish", call_id="ask-1")
        )
        tasks.start_soon(
            lambda: ctx.require(TOOLS).execute(
                ToolExecutionInput(
                    call_id="fx-1",
                    name="send",
                    arguments={"id": "m1", "message": "the numbers"},
                    scope=agent.ctx,
                    session=root,
                    agent=agent,
                )
            )
        )
        provider, model = ("keyed", "k1") if children_on == "keyed" else (None, None)
        first, second = [
            await ctx.require(SUBAGENTS).start(
                PROVIDER_NAME,
                SubagentRequest(prompt=prompt, parent=agent, provider=provider, model=model),
            )
            for prompt in ("research the first thing", "research the second thing")
        ]

        def settled_into_place() -> bool:
            types = {event.type for event in root.events}
            roster = subagent_roster(root)
            return (
                {"tool/call", "approval/asked", "tool/effect", "shell/command"} <= types
                and (project / "notes.md").exists()
                and roster.get(first.id, {}).get("status") == "running"
                and roster.get(second.id, {}).get("status") == "queued"
            )

        with anyio.fail_after(30):
            while not settled_into_place():
                await anyio.sleep(0.02)
        for session in sessions.list():
            await sessions.flush(session)
        print(
            json.dumps(
                {
                    "root": ROOT,
                    "running": {"run": first.id, "session": first.session_id},
                    "queued": {"run": second.id, "session": second.session_id},
                }
            ),
            flush=True,
        )
        await anyio.sleep_forever()


if __name__ == "__main__":
    anyio.run(main, Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "scripted")
