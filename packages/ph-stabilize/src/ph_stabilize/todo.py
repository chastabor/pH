"""`tool-todo` — planning as a cognitive anchor (P4-01, G1).

Deep Agents' `TodoListMiddleware`, ported onto pH's seams rather than onto a
middleware stack: a tool on `ctx.tools`, a tool-owned `todo/write` event, a
prompt section, and a `tools/pre-execute` listener for the one enforcement rule.
That decomposition is D12 — a stabilization feature attaches to waterfalls,
never to the loop — and it is why this is a row a profile opts into rather than
a parameter on the driver.

**The list lives in the log and nowhere else.** `write_todos` replaces the whole
list and appends `todo/write {todos}`; the TUI sidebar and the prompt both fold
that event. There is no side table, which is what makes the list survive a
resume and a fork for free, and what makes the sidebar and the model's view one
projection rather than two that can disagree (A11).

**One call per model turn, enforced before any of them runs.** Upstream's rule
is that if a single assistant message contains more than one `write_todos` call
then *every* one of them fails — the list is whole-list-replacement, so two
calls in a turn make precedence ambiguous. pH's `tools/pre-execute` waterfall is
per call, so the batch question is answered from the log: the assistant message
that requested the calls is committed before any of them executes, and counting
`write_todos` in it gives every sibling the same answer before the first body
runs. That last part is the point — a rule that let the first call through and
failed the second would leave a list written by a call the model was told had
failed. The rule is the native transport's: a Code Mode cell orders its own
calls, so the ambiguity it guards against cannot arise there (see
`_parallel_write_todos`).

@module ph_stabilize.todo
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from ph.cordis import Context, plugin
from ph.llm.types import ToolCallBlock
from ph.session import Session, derive_event_message, thaw_json
from ph.system_prompt.assembly import (
    ORDER_TOOL_GUIDANCE,
    AssembleContext,
    PromptContext,
    PromptSection,
)
from ph.tools import ToolCallView, ToolResultView
from ph.tools.definition import (
    Deny,
    ToolExecution,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    define_tool,
    text_content,
)

__all__ = [
    "PARALLEL_CALL_ERROR",
    "TOOL_NAME",
    "WRITE_TODOS_SYSTEM_PROMPT",
    "apply",
    "render_todo_list",
    "todos_of",
]

TOOL_NAME = "write_todos"

GLYPHS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
"""How the list renders back to the model. Deliberately the same three-state
vocabulary the sidebar draws, so a person and the model are reading one list."""

# --------------------------------------------------------------------------
# Verbatim from `langchain.agents.middleware.todo` (checked against 1.3.18,
# which satisfies the plan's pinned `langchain>=1.3.17`). Copied rather than
# imported: ph does not depend on langchain, and these are prompt *text* — the
# thing most likely to be silently reworded by an upgrade, and the thing whose
# exact words the row's gate is about.
# --------------------------------------------------------------------------

WRITE_TODOS_SYSTEM_PROMPT = """## `write_todos`

You have access to the `write_todos` tool to help you manage and plan complex objectives.
Use this tool for complex objectives to ensure that you are tracking each necessary step.
This tool is very helpful for planning complex objectives, and for breaking down these larger complex objectives into smaller steps.

It is critical that you mark todos as completed as soon as you are done with a step. Do not batch up multiple steps before marking them as completed.
For simple objectives that only require a few steps, it is better to just complete the objective directly and NOT use this tool.
Writing todos takes time and tokens, use it when it is helpful for managing complex many-step problems! But not for simple few-step requests.

## Important To-Do List Usage Notes to Remember

- The `write_todos` tool should never be called multiple times in parallel.
- Don't be afraid to revise the To-Do list as you go. New information may reveal new tasks that need to be done, or old tasks that are irrelevant.

## Finishing a task

When you finish all work, write your final answer in the message AFTER your last `write_todos` call — not in the same turn as that call. Start the final message with the substantive content the user asked for — the data, computation, summary, or analysis. The user wants the result, not confirmation that the work is done."""  # noqa: E501

WRITE_TODOS_TOOL_DESCRIPTION = """Use this tool to create and manage a structured task list for your current work session. This helps you track progress and organize complex tasks.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool

Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool

1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool

It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
    - pending: Task not yet started
    - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
    - completed: Task finished successfully

2. **Task Management**:
    - Update task status in real-time as you work
    - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
    - Complete current tasks before starting new ones
    - Remove tasks that are no longer relevant from the list entirely
    - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
    - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress.

3. **Task Completion Requirements**:
    - ONLY mark a task as completed when you have FULLY accomplished it
    - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
    - When blocked, create a new task describing what needs to be resolved
    - Never mark a task as completed if:
        - There are unresolved issues or errors
        - Work is partial or incomplete
        - You encountered blockers that prevent completion
        - You couldn't find necessary resources or dependencies
        - Quality standards haven't been met

4. **Task Breakdown**:
    - Create specific, actionable items
    - Break complex tasks into smaller, manageable steps
    - Use clear, descriptive task names

Being proactive with task management ensures you complete all requirements successfully
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.

## When You Finish

`write_todos` tracks your work; it does not deliver the answer. Whatever the user asked for — computations, summaries, comparisons, data — must appear as text content in a message after your final `write_todos` call. Marking the last todo complete is not itself an answer to the user."""  # noqa: E501

PARALLEL_CALL_ERROR = (
    "Error: The `write_todos` tool should never be called multiple times "
    "in parallel. Please call it only once per model invocation to update "
    "the todo list."
)
"""Verbatim from upstream, and the row's gate. Assembled from the same three
fragments so a diff against `todo.py` stays readable."""


class TodoItem(ToolModel):
    """One entry. Upstream's `Todo` TypedDict, which is `{content, status}`.

    Deliberately no `activeForm`: deepagents' own tests carry that field but
    the pinned `langchain` middleware's schema does not, and pH renders a glyph
    beside the content in both the sidebar and the prompt — a second phrasing of
    the same task would be a field with no reader.
    """

    content: str = Field(description="What the step is.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        "pending", description="Where the step has got to."
    )


class WriteTodosArgs(ToolModel):
    todos: list[TodoItem] = Field(
        description="The ENTIRE list. This call replaces the previous one; there are no "
        "partial updates and no per-item edits."
    )


# ------------------------------------------------------------------ reading --


def todos_of(session: Session | None) -> list[dict[str, Any]]:
    """The current list: the last `todo/write` in the log, or nothing.

    Last-write-wins over an append-only log, which is the whole storage design —
    a fold rather than a table, so a fork inherits the list its prefix ended
    with and a resume shows what the person left. `Session.latest` is that
    question's own helper, incremental so a prompt assembled every step does not
    rescan a log that is mostly `assistant/chunk`s.
    """
    event = session.latest("todo/write") if session is not None else None
    if event is None:
        return []
    # `thaw_json` because the payload is frozen — a `MappingProxyType` is not a
    # `dict`, the bug this project has now been bitten by three times.
    thawed = thaw_json(event.data.get("todos"))
    return [item for item in thawed or () if isinstance(item, dict)]


def render_todo_list(todos: list[dict[str, Any]]) -> str:
    """The list as the model sees it between turns."""
    lines = [
        f"{GLYPHS.get(str(todo.get('status')), '[ ]')} {todo.get('content', '')}" for todo in todos
    ]
    return "## Todo list\n\n" + "\n".join(lines)


def _parallel_write_todos(session: Session | None) -> bool:
    """Whether the assistant message now being executed asked for two writes.

    Read from the log rather than from the batch, because pH's
    `tools/pre-execute` is per call while upstream's rule is about the *message*.
    The assistant message is committed before any of its tool calls runs, so
    every sibling gets this same answer before the first body executes — which
    is what makes "every one of them fails" true rather than "all but the first".
    `Session.latest` and `derive_event_message` own the scan and the payload
    shape; a third reader spelling either by hand is how they drift.

    Native transport only, and deliberately. The rule exists because two
    whole-list replacements in one *message* have no defined precedence. A Code
    Mode cell that calls `tools.write_todos` twice has one — the program's own
    execution order (the tool declares no concurrency safety, so even gathered
    dispatches serialize) — so there the last write wins by the program's own
    statement, and refusing would punish a cell for being unambiguous.
    """
    event = session.latest("assistant/message") if session is not None else None
    message = derive_event_message(event) if event is not None else None
    if message is None:
        return False
    calls = sum(
        1
        for block in message.content
        if isinstance(block, ToolCallBlock) and block.name == TOOL_NAME
    )
    return calls > 1


# ------------------------------------------------------------------- the row --


@plugin("tool-todo", inject=["tools", "system_prompt"])
async def apply(ctx: Context, config: Any) -> None:
    """Register the tool, its prompt section, its context and its one rule."""

    async def write_todos(args: WriteTodosArgs, run: ToolRunContext) -> Any:
        todos = [item.model_dump(mode="json") for item in args.todos]
        session = run.session
        if session is not None:
            # Tool-owned event: the tool that changed the list is the one that
            # records it, so "model-visible means logged" holds without the
            # pipeline learning what a todo is (I3).
            session.append("todo/write", {"todos": todos})
        return {"todos": todos}

    ctx.tools.register(
        define_tool(
            TOOL_NAME,
            WRITE_TODOS_TOOL_DESCRIPTION,
            parameters=WriteTodosArgs,
            output=ToolOutput(
                # The output is the input, accepted — same shape, so the same
                # model; a separate `Value` class would differ only by lacking
                # the field description.
                schema=WriteTodosArgs,
                # Upstream's own wording. The evals note a proposed neutral
                # "Todo list updated."; the pinned release still echoes the
                # list, and echoing it is what lets a model that batched several
                # changes see what actually landed.
                render=lambda args, value: text_content(f"Updated todo list to {value['todos']}"),
            ),
            execute=write_todos,
            present_call=lambda args: ToolCallView(
                title="Plan", input=render_todo_list(list(args.get("todos") or ()))
            ),
            present_result=lambda args, result: ToolResultView(
                title="Plan",
                subtitle=_counts(list(args.get("todos") or ())),
                is_error=result.is_error,
            ),
        )
    )

    async def refuse_parallel_calls(execution: ToolExecution, next_: Any) -> Any:
        if execution.name == TOOL_NAME and _parallel_write_todos(execution.session):
            return Deny(reason=PARALLEL_CALL_ERROR)
        return await next_(execution)

    ctx.on("tools/pre-execute", refuse_parallel_calls)

    ctx.system_prompt.section(
        PromptSection(
            name="write-todos",
            # After the tool guidance the catalog itself contributes: this is
            # advice about one tool, and it reads as a footnote to that list
            # rather than as a competing instruction.
            order=ORDER_TOOL_GUIDANCE + 50,
            text=WRITE_TODOS_SYSTEM_PROMPT,
        )
    )

    def current_list(assemble: AssembleContext) -> str:
        todos = todos_of(getattr(assemble.agent, "session", None))
        return render_todo_list(todos) if todos else ""

    # A `context`, not a `section`: it changes every time the model writes, and
    # a changing string in the cached prefix would invalidate the prefix on
    # every plan update (A12). Materialized after retained history, which is
    # also where it belongs to be read — the state as of now, after the
    # conversation that produced it.
    ctx.system_prompt.context(PromptContext(name="todos", text=current_list))


def _counts(todos: list[Any]) -> str:
    """`2 done · 1 doing · 3 to do`, in the order work leaves the list."""
    tally = Counter(str(todo.get("status")) for todo in todos if isinstance(todo, Mapping))
    labelled = (("completed", "done"), ("in_progress", "doing"), ("pending", "to do"))
    return " · ".join(f"{tally[status]} {label}" for status, label in labelled)
