"""`tool-bash`: run a command, with a description that says what the tool is.

The description matters as much as the code. A model that is told "run a shell
command" uses `cat` and `sed` in place of `read` and `edit`. Each of those calls
escapes the filesystem gates that make `edit` reviewable. So the description
points back at the specific tools. That is the only lever that a registry-side
design has over the behavior of the model (C1: "make the registered path the
convenient path").

@module ph.tools.builtin.bash_tool
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...cancel import Canceled
from ...characters import head
from ...cordis import Context, plugin
from ...json import JsonObject, as_str
from ...keys import SHELL, TOOLS
from ...llm.types import ContentBlock
from ...text import NO_OUTPUT, truncation_marker
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..presentation import ToolCallView, ToolResultView

__all__ = ["apply"]

TIMED_OUT = "[ph: timed out; the command was killed and this is what it had printed]"
"""How the `timeout_ms` of a caller appears to the reader of the result.

`!!` shares this text through `ph_app.shell`, for the same reason that it shares
the truncation sentence. The docstring of `shell_body` already promises that one
event does not get two renderings."""

DESCRIPTION = """Run a shell command and return its output.

Prefer the dedicated tools where one fits: `read` over `cat`, `edit` over `sed`,
`glob` over `find`, `grep` over `grep`. They are reviewable, they respect the
workspace root, and their results render as cards rather than raw text."""


class BashArgs(ToolModel):
    command: str = Field(description="The command line to run.")
    timeout_ms: int | None = Field(
        None, ge=100, le=600_000, description="Kill the command after this long."
    )
    description: str | None = Field(
        None, description="One short line about what this command is for."
    )


class BashValue(ToolModel):
    command: str
    exit_code: int
    stdout: str
    stderr: str
    confined_by: str | None = None
    dropped: int = 0
    """The bytes past the cap of the seam that nobody kept (P7-13)."""
    cap: int = 0
    timed_out: bool = False


def _render(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    parts: list[str] = []
    if value["stdout"]:
        parts.append(value["stdout"].rstrip())
    if value["stderr"]:
        parts.append(f"[stderr]\n{value['stderr'].rstrip()}")
    if value["timed_out"]:
        parts.append(TIMED_OUT)
    if value["dropped"]:
        # `truncation_marker` is used, and not a wording of this tool. The RLM
        # kernel and the guest runner already say this sentence for the same
        # event, and a reader who compares a transcript with a log must not find
        # two sentences (D4).
        parts.append(truncation_marker(value["dropped"], value["cap"]).strip())
    if value["exit_code"] != 0:
        parts.append(f"[exit {value['exit_code']}]")
    return text_content("\n".join(parts) if parts else NO_OUTPUT)


@plugin("tool-bash", affects="environment", inject=[TOOLS, SHELL])
async def apply(ctx: Context, config: None) -> None:
    """Register the bash tool."""

    async def run_command(args: BashArgs, run: ToolRunContext) -> dict[str, Any]:
        # The tool passes the agent and not a directory. `ctx.shell` resolves
        # the cwd and the workspace environment from the agent, so this tool
        # states who is running and does not derive again where (D21, E2).
        result = await ctx.require(SHELL).run(
            args.command,
            agent=run.agent,
            timeout_ms=args.timeout_ms,
            # **The cancellation of the call itself, passed down** (C7). Nothing
            # between the pipeline and the child watched it before. A command
            # with no `timeout_ms` therefore ran to completion, whatever
            # interrupted it. The child token of `tools-timeout` exists to
            # narrow exactly this, and it had nowhere to arrive.
            signal=run.signal,
        )
        if result.canceled:
            # **The pipeline has its own class for this** (N3). The child was
            # killed because somebody pressed stop. A returned value would
            # report the exit code of the kill as the exit code of the command:
            # `[exit -15]`, rendered to the model as a command that failed.
            # `registry._failure` maps `Canceled` to `aborted_result`, which is
            # the vocabulary that the rest of the loop already reads.
            raise Canceled(f"{args.command!r} was canceled")
        return {
            "command": args.command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "confined_by": result.confined_by,
            "dropped": result.dropped,
            "cap": result.cap,
            "timed_out": result.timed_out,
        }

    ctx.require(TOOLS).register(
        define_tool(
            "bash",
            DESCRIPTION,
            parameters=BashArgs,
            output=ToolOutput(schema=BashValue, render=_render),
            execute=run_command,
            # A shell command reaches past the tree by construction. `/revert`
            # restores the worktree, not the world.
            is_irreversible=True,
            present_call=lambda args: ToolCallView(
                card="terminal",
                title=as_str(args.get("description"), "Run command"),
                input=str(args.get("command", "")),
            ),
            present_result=lambda args, result: ToolResultView(
                card="terminal",
                title=as_str(args.get("description"), "Run command"),
                subtitle=head(str(args.get("command", "")), 120),
                is_error=result.is_error,
            ),
        )
    )
