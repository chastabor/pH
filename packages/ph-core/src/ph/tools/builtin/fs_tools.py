"""`tool-fs`: read, write, edit, glob, grep.

Each of these tools is a thin wrapper over `ctx.fs`, and that is the purpose.
The policy gates (`fs/write-intent`, `fs/edit-intent`), the read-before-edit
rule and the workspace root all live in the seam. A second editing tool, or a
Code Mode binding, therefore inherits them and does not implement them again.

The `output` declarations carry real schemas and not free text, so the durable
record is structured. A card renders from `content` plus `meta` with no access
to the live call. That is why a replayed session looks identical to the live
session.

Reads are classified as concurrency-safe. Writes and edits are not. A read
cannot disturb a sibling. Two edits to one file in the same batch are a race
whose outcome depends on scheduling, so they are serialized (B6).

@module ph.tools.builtin.fs_tools
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import anyio
from pydantic import Field

from ...agent.types import AgentHandle
from ...cordis import Context, plugin
from ...json import JsonObject, JsonValue, as_obj
from ...keys import FS, TOOLS
from ...llm.types import ContentBlock
from ...paths import canonical, is_under
from ...seams.workspace import workspace_leaks, workspace_of
from ...session import Session, SessionEvent
from ...text import count_of
from ..definition import (
    Done,
    NotDone,
    Reconciled,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    Unknown,
    define_tool,
    text_content,
)
from ..presentation import simple_views

__all__ = ["apply"]


class ReadArgs(ToolModel):
    path: str = Field(description="Path to read. Relative paths resolve against the workspace.")
    offset: int = Field(0, ge=0, description="First line to return (0-based).")
    limit: int = Field(2_000, ge=1, le=20_000, description="Maximum lines to return.")


class ReadValue(ToolModel):
    path: str
    text: str
    offset: int
    lines: int
    total_lines: int
    truncated: bool


class WriteArgs(ToolModel):
    path: str = Field(description="Path to write. Parent directories are created.")
    content: str = Field(description="Complete new file contents.")


class WriteValue(ToolModel):
    path: str
    bytes: int
    created: bool


class EditArgs(ToolModel):
    path: str = Field(description="Path to edit.")
    old_text: str = Field(description="Exact text to replace. Must be unique unless replace_all.")
    new_text: str = Field(description="Replacement text.")
    replace_all: bool = Field(False, description="Replace every occurrence instead of one.")


class EditValue(ToolModel):
    path: str
    replacements: int


class GlobArgs(ToolModel):
    pattern: str = Field(description="Glob pattern, e.g. '**/*.py'.")
    path: str | None = Field(
        None, description="Directory to search from. Defaults to the workspace root."
    )


class GlobValue(ToolModel):
    paths: list[str]
    truncated: bool


class GrepArgs(ToolModel):
    pattern: str = Field(description="Regular expression to search for.")
    path: str | None = Field(None, description="Directory to search from.")
    glob: str = Field("**/*", description="Which files to search.")


class GrepMatchValue(ToolModel):
    path: str
    line: int
    text: str


class GrepValue(ToolModel):
    matches: list[GrepMatchValue]
    truncated: bool


def _render_read(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    first = value["offset"] + 1
    last = value["offset"] + value["lines"]
    # The marker goes in the header, which is clearly the text of the harness.
    # When the marker was appended to the body, it touched the last byte of the
    # file (the terminator included, since J7), so this function had to know
    # where the window ended.
    more = f"; re-read with offset={last}" if value["truncated"] else ""
    header = f"{value['path']} (lines {first}-{last} of {value['total_lines']}{more})"
    return text_content(f"{header}\n{value['text']}")


def _holds_exactly(target: Path, expected: bytes) -> bool:
    """Whether `target` is a file that holds `expected`. The size is compared
    first, so a file of a different length is answered without a read."""
    if not target.is_file() or target.stat().st_size != len(expected):
        return False
    return target.read_bytes() == expected


def _render_write(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    verb = "Created" if value["created"] else "Wrote"
    return text_content(f"{verb} {value['path']} ({value['bytes']} bytes)")


def _render_edit(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    return text_content(
        f"Edited {value['path']} ({count_of(value['replacements'], 'replacement')})"
    )


def _render_glob(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    paths = value["paths"]
    if not paths:
        return text_content("No files matched.")
    suffix = "\n[truncated]" if value["truncated"] else ""
    return text_content(f"{len(paths)} match(es):\n" + "\n".join(paths) + suffix)


def _render_grep(_args: JsonObject, value: Any) -> list[ContentBlock]:  # noqa: ANN401
    matches = value["matches"]
    if not matches:
        return text_content("No matches.")
    listing = "\n".join(f"{m['path']}:{m['line']}: {m['text']}" for m in matches)
    suffix = "\n[truncated]" if value["truncated"] else ""
    return text_content(f"{len(matches)} match(es):\n{listing}{suffix}")


GLOB_LIMIT = 1_000
GREP_LIMIT = 200


@plugin("tool-fs", affects="environment", inject=[TOOLS, FS])
async def apply(ctx: Context, config: None) -> None:
    """Register the filesystem tools."""
    fs = ctx.require(FS)

    async def read(args: ReadArgs, run: ToolRunContext) -> dict[str, Any]:
        window = await fs.read(
            args.path,
            offset=args.offset,
            limit=args.limit,
            agent=run.agent,
            scope=run.scope,
            session=run.session,
        )
        return window.model_dump()

    async def write(args: WriteArgs, run: ToolRunContext) -> dict[str, Any]:
        written = await fs.write(
            args.path,
            args.content,
            agent=run.agent,
            scope=run.scope,
            session=run.session,
        )
        return {
            # The name that the workspace knows the file by, never the name on
            # the machine. See `FsService.named`.
            "path": fs.named(written.path, agent=run.agent),
            "bytes": written.bytes,
            "created": written.created,
        }

    async def reconciled_write(args: Any, _call: SessionEvent, session: Session) -> Reconciled:  # noqa: ANN401
        """Did a write that a crash interrupted land? The file says (P10-13).

        **Exact or `Unknown`.** The answer is done when the file holds the bytes
        of the call exactly: the effect is there, whoever finished it. The answer
        is not done when the file is missing or holds something else. Either the
        call never reached the file or somebody changed it since, and in both
        cases writing it again is what the call asked for. The path is resolved
        in the way that the call resolved it, against the root of the agent as
        its log records it: a fresh-root workspace that is still open, or else
        the root of the deployment. No agent exists to ask at resume. Anything
        that cannot be read is `Unknown`, which keeps the `TOOL_OUTCOME_UNKNOWN`
        text of repair.
        """
        body = as_obj(args)
        path, content = body.get("path"), body.get("content")
        if not isinstance(path, str) or not path or not isinstance(content, str):
            return Unknown()
        root = next(
            (one.root for one in workspace_leaks(session) if one.agent_id == session.id),
            fs.root,
        )
        target = fs.resolve(path, root=root)
        expected = content.encode("utf-8")
        try:
            landed = await anyio.to_thread.run_sync(_holds_exactly, target, expected)
        except OSError:
            return Unknown()
        if not landed:
            return NotDone()
        return Done({"path": fs.named(target, root=root), "bytes": len(expected), "created": False})

    def inside_workspace(args: JsonValue, agent: AgentHandle | None) -> bool:
        """Whether a `write` or an `edit` lands in the workspace of the agent (S4).

        The test uses where the path really resolves, with links followed as the
        write follows them. It tests against the tree that a restore resets. That
        tree is the workspace of the agent and not only its cwd. The two are the
        same only where the lifecycle rebases `ctx.fs`. An absolute path, an
        allowance or a link out of that tree is a write that a restore does not
        take back, so it is not covered. The checkpoint policy makes its record
        durable first, and `/revert` lists it as not undone. With no workspace
        there is no tree, and the answer is false.
        """
        path = as_obj(args).get("path")
        workspace = workspace_of(ctx, agent)
        if not isinstance(path, str) or not path or workspace is None:
            return False
        return is_under(canonical(fs.resolve(path, agent=agent)), canonical(workspace.root))

    async def edit(args: EditArgs, run: ToolRunContext) -> dict[str, Any]:
        count = await fs.edit(
            args.path,
            args.old_text,
            args.new_text,
            replace_all=args.replace_all,
            agent=run.agent,
            scope=run.scope,
            session=run.session,
        )
        return {"path": fs.named(args.path, agent=run.agent), "replacements": count}

    async def glob_tool(args: GlobArgs, run: ToolRunContext) -> dict[str, Any]:
        paths = await fs.glob(
            args.pattern, root=args.path, limit=GLOB_LIMIT, agent=run.agent, scope=run.scope
        )
        return {"paths": paths, "truncated": len(paths) >= GLOB_LIMIT}

    async def grep_tool(args: GrepArgs, run: ToolRunContext) -> dict[str, Any]:
        matches = await fs.grep(
            args.pattern,
            root=args.path,
            glob=args.glob,
            limit=GREP_LIMIT,
            agent=run.agent,
            scope=run.scope,
        )
        return {
            "matches": [match.model_dump() for match in matches],
            "truncated": len(matches) >= GREP_LIMIT,
        }

    # Each of these tools bounds its own output and offers the model a way to
    # ask for more. `read` takes an offset and a limit, `glob` and `grep` cap
    # their match lists, and `write` and `edit` return a confirmation. The
    # declaration lets the offload row (G2) leave their results inline, without
    # another package that keeps a list of the tool names of this package.
    for definition in (
        # The three tools that only look are `effect_free`. The two that write
        # answer `effects_confined_to_workspace` for each call (S4), because
        # their path can leave the tree. `/revert` therefore lists none of them
        # as a call that it failed to undo, unless one did reach past the tree
        # (N3).
        define_tool(
            "read",
            "Read a file, or a window of one. Prefer this over shelling out to cat.",
            parameters=ReadArgs,
            output=ToolOutput(schema=ReadValue, render=_render_read),
            execute=read,
            effect_free=True,
            self_limits=True,
            reads_paths=True,
            is_concurrency_safe=True,
            **simple_views("read", "Read", "path"),
        ),
        define_tool(
            "write",
            "Write a complete file. Creates parent directories.",
            parameters=WriteArgs,
            output=ToolOutput(schema=WriteValue, render=_render_write),
            execute=write,
            reconcile=reconciled_write,
            effects_confined_to_workspace=inside_workspace,
            self_limits=True,
            # The file body is a payload that this call delivered, and not an
            # instruction that the model refers back to. The file is also on
            # disk, where `read` can fetch it. Retained history may elide the
            # body under pressure.
            arguments_disposable=True,
            **simple_views("diff", "Write", "path"),
        ),
        define_tool(
            "edit",
            "Replace exact text in a file. Read the file first.",
            parameters=EditArgs,
            output=ToolOutput(schema=EditValue, render=_render_edit),
            execute=edit,
            effects_confined_to_workspace=inside_workspace,
            self_limits=True,
            arguments_disposable=True,
            **simple_views("diff", "Edit", "path"),
        ),
        define_tool(
            "glob",
            "Find files by glob pattern.",
            parameters=GlobArgs,
            output=ToolOutput(schema=GlobValue, render=_render_glob),
            execute=glob_tool,
            effect_free=True,
            self_limits=True,
            searches_paths=True,
            is_concurrency_safe=True,
            **simple_views("search", "Glob", "pattern"),
        ),
        define_tool(
            "grep",
            "Search file contents with a regular expression.",
            parameters=GrepArgs,
            output=ToolOutput(schema=GrepValue, render=_render_grep),
            execute=grep_tool,
            effect_free=True,
            self_limits=True,
            searches_paths=True,
            is_concurrency_safe=True,
            **simple_views("search", "Grep", "pattern"),
        ),
    ):
        ctx.require(TOOLS).register(definition)
