"""`tool-fs` — read, write, edit, glob, grep.

Every one of these is a thin shell over `ctx.fs`, and that is the point: the
policy gates (`fs/write-intent`, `fs/edit-intent`), the read-before-edit rule
and the workspace root all live in the seam, so a second editing tool — or a
Code Mode binding — inherits them instead of re-implementing them.

The `output` declarations carry real schemas rather than free text, so the
durable record is structured: a card renders from `content` plus `meta` with no
access to the live call, which is what lets a replayed session look identical to
the live one.

Reads classify as concurrency-safe; writes and edits do not. A read cannot
disturb a sibling, while two edits to one file in the same batch are a race
whose outcome depends on scheduling — so they serialize (B6).

@module ph.tools.builtin.fs_tools
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...cordis import Context, plugin
from ...text import count_of
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
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


def _render_read(args: Any, value: Any) -> Any:
    first = value["offset"] + 1
    last = value["offset"] + value["lines"]
    body = value["text"]
    if value["truncated"]:
        body += f"\n\n[truncated; re-read with offset={last}]"
    return text_content(f"{value['path']} (lines {first}-{last} of {value['total_lines']})\n{body}")


def _render_write(args: Any, value: Any) -> Any:
    verb = "Created" if value["created"] else "Wrote"
    return text_content(f"{verb} {value['path']} ({value['bytes']} bytes)")


def _render_edit(args: Any, value: Any) -> Any:
    return text_content(
        f"Edited {value['path']} ({count_of(value['replacements'], 'replacement')})"
    )


def _render_glob(args: Any, value: Any) -> Any:
    paths = value["paths"]
    if not paths:
        return text_content("No files matched.")
    suffix = "\n[truncated]" if value["truncated"] else ""
    return text_content(f"{len(paths)} match(es):\n" + "\n".join(paths) + suffix)


def _render_grep(args: Any, value: Any) -> Any:
    matches = value["matches"]
    if not matches:
        return text_content("No matches.")
    listing = "\n".join(f"{m['path']}:{m['line']}: {m['text']}" for m in matches)
    suffix = "\n[truncated]" if value["truncated"] else ""
    return text_content(f"{len(matches)} match(es):\n{listing}{suffix}")


GLOB_LIMIT = 1_000
GREP_LIMIT = 200


@plugin("tool-fs", inject=["tools", "fs"])
async def apply(ctx: Context, config: Any) -> None:
    """Register the filesystem tools."""
    fs = ctx.fs

    async def read(args: ReadArgs, run: ToolRunContext) -> Any:
        window = await fs.read(
            args.path,
            offset=args.offset,
            limit=args.limit,
            agent=run.agent,
            scope=run.scope,
            session=run.session,
        )
        return window.model_dump()

    async def write(args: WriteArgs, run: ToolRunContext) -> Any:
        target = fs.resolve(args.path, agent=run.agent)
        existed = target.exists()
        await fs.write(
            args.path, args.content, agent=run.agent, scope=run.scope, session=run.session
        )
        return {
            "path": str(target),
            "bytes": len(args.content.encode("utf-8")),
            "created": not existed,
        }

    async def edit(args: EditArgs, run: ToolRunContext) -> Any:
        count = await fs.edit(
            args.path,
            args.old_text,
            args.new_text,
            replace_all=args.replace_all,
            agent=run.agent,
            scope=run.scope,
            session=run.session,
        )
        return {"path": str(fs.resolve(args.path, agent=run.agent)), "replacements": count}

    async def glob_tool(args: GlobArgs, run: ToolRunContext) -> Any:
        paths = await fs.glob(
            args.pattern, root=args.path, limit=GLOB_LIMIT, agent=run.agent, scope=run.scope
        )
        return {"paths": paths, "truncated": len(paths) >= GLOB_LIMIT}

    async def grep_tool(args: GrepArgs, run: ToolRunContext) -> Any:
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

    # Every one of these bounds its own output and offers the model a way to
    # ask for more — `read` takes an offset and a limit, `glob`/`grep` cap
    # their match lists, `write`/`edit` return a confirmation. Declared so
    # the offload row (G2) can leave their results inline without another
    # package keeping a list of this package's tool names.
    for definition in (
        # `effects_confined_to_workspace` on all five: every effect these have is a
        # file inside the agent's workspace, or — for the three that only look —
        # no effect at all. `/revert` therefore does not list them as things it
        # failed to undo, which is what keeps the listing about the calls that
        # actually reached past the tree (N3).
        define_tool(
            "read",
            "Read a file, or a window of one. Prefer this over shelling out to cat.",
            parameters=ReadArgs,
            output=ToolOutput(schema=ReadValue, render=_render_read),
            execute=read,
            effects_confined_to_workspace=True,
            self_limits=True,
            is_concurrency_safe=True,
            **simple_views("read", "Read", "path"),
        ),
        define_tool(
            "write",
            "Write a complete file. Creates parent directories.",
            parameters=WriteArgs,
            output=ToolOutput(schema=WriteValue, render=_render_write),
            execute=write,
            effects_confined_to_workspace=True,
            self_limits=True,
            # The file body is a payload this call delivered, not an instruction
            # the model refers back to — and the file is on disk, where `read`
            # can fetch it. Retained history may elide it under pressure.
            arguments_disposable=True,
            **simple_views("diff", "Write", "path"),
        ),
        define_tool(
            "edit",
            "Replace exact text in a file. Read the file first.",
            parameters=EditArgs,
            output=ToolOutput(schema=EditValue, render=_render_edit),
            execute=edit,
            effects_confined_to_workspace=True,
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
            effects_confined_to_workspace=True,
            self_limits=True,
            is_concurrency_safe=True,
            **simple_views("search", "Glob", "pattern"),
        ),
        define_tool(
            "grep",
            "Search file contents with a regular expression.",
            parameters=GrepArgs,
            output=ToolOutput(schema=GrepValue, render=_render_grep),
            execute=grep_tool,
            effects_confined_to_workspace=True,
            self_limits=True,
            is_concurrency_safe=True,
            **simple_views("search", "Grep", "pattern"),
        ),
    ):
        ctx.tools.register(definition)
