"""`tool-bash` — run a command, honestly labelled.

The description matters as much as the code. A model told "run a shell command"
will reach for `cat` and `sed` over `read` and `edit`, and every one of those
calls escapes the filesystem gates that make `edit` reviewable. So the
description points back at the specific tools, which is the only lever a
registry-side design has over model behaviour (C1's "make the registered path
the convenient path").

@module ph.tools.builtin.bash_tool
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...cordis import Context, plugin
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..presentation import ToolCallView, ToolResultView

__all__ = ["apply"]

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


def _render(args: Any, value: Any) -> Any:
    parts: list[str] = []
    if value["stdout"]:
        parts.append(value["stdout"].rstrip())
    if value["stderr"]:
        parts.append(f"[stderr]\n{value['stderr'].rstrip()}")
    if value["exit_code"] != 0:
        parts.append(f"[exit {value['exit_code']}]")
    return text_content("\n".join(parts) if parts else "(no output)")


@plugin("tool-bash", inject=["tools", "shell"])
async def apply(ctx: Context, config: Any) -> None:
    """Register the bash tool."""

    async def run_command(args: BashArgs, run: ToolRunContext) -> Any:
        # The agent, not a directory: `ctx.shell` resolves the cwd and the
        # workspace environment from it, so this tool states who is running
        # rather than re-deriving where (D21, E2).
        result = await ctx.shell.run(args.command, agent=run.agent, timeout_ms=args.timeout_ms)
        return {
            "command": args.command,
            "exit_code": result.exit_code,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "confined_by": result.confined_by,
        }

    ctx.tools.register(
        define_tool(
            "bash",
            DESCRIPTION,
            parameters=BashArgs,
            output=ToolOutput(schema=BashValue, render=_render),
            execute=run_command,
            present_call=lambda args: ToolCallView(
                card="terminal",
                title=str(args.get("description") or "Run command"),
                input=str(args.get("command", "")),
            ),
            present_result=lambda args, result: ToolResultView(
                card="terminal",
                title=str(args.get("description") or "Run command"),
                subtitle=str(args.get("command", ""))[:120],
                is_error=result.is_error,
            ),
        )
    )
