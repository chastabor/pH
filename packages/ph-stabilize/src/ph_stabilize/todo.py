"""`tool-todo`: planning as a cognitive anchor (P4-01, G1).

This row ports `TodoListMiddleware` from Deep Agents. The port uses the seams of
pH and not a middleware stack. It consists of:

* a tool on `ctx.tools`
* a `todo/write` event that the tool owns
* a prompt section
* a `tools/pre-execute` listener for the one enforcement rule

That decomposition is D12: a stabilization feature attaches to waterfalls and
never to the loop. For that reason this is a row that a profile opts into, and
not a parameter on the driver.

**The port forked from upstream at P7-16, on purpose and in two places.**

First, `requires` declares what a step waits on. The upstream schema has no
field for that, and the upstream *prompt* works around the gap in prose ("When
blocked, create a new task describing what needs to be resolved"). That is a
dependency graph written as a naming convention, and nothing can read it, order
it or check it.

Second, a completed entry carries `worked`: the number of tools that the harness
saw run in the window in which the entry was finished. It is the one field in
the list that the model does not write, because a receipt that the claimant
issues is not a receipt.

The upstream prompt text and the parallel-call rule are still tracked verbatim
and still gated. The schema is no longer the upstream schema. The comparison
that produced both changes is recorded against `sources/OpenMonoAgent.ai`.

**The list is stored in the log and nowhere else.** `write_todos` replaces the
whole list and appends `todo/write {todos}`. The TUI sidebar and the prompt both
fold that event. There is no side table. For that reason the list survives a
resume and a fork at no extra cost, and the sidebar and the view of the model
are one projection and not two that can disagree (A11).

**One call in each model turn, enforced before any of the calls runs.** The
upstream rule is that if a single assistant message contains more than one
`write_todos` call, then *every* one of them fails. The tool replaces the whole
list, so two calls in one turn make precedence ambiguous. The
`tools/pre-execute` waterfall of pH runs for each call, so the log answers the
question about the batch. The assistant message that requested the calls is
committed before any of them executes. A count of the `write_todos` calls in
that message gives every sibling the same answer before the first body runs.
That timing is the purpose. A rule that let the first call through and failed
the second would leave a list that was written by a call that the model was told
had failed.

The rule applies to the native transport only. A Code Mode cell orders its own
calls, so the ambiguity that the rule guards against cannot occur there (see
`_parallel_write_todos`).

@module ph_stabilize.todo
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any, Literal

from pydantic import Field

from ph.cordis import Context, Next, plugin
from ph.json import as_int, as_seq, as_str, thaw_json
from ph.keys import SYSTEM_PROMPT, TOOLS
from ph.llm.types import Message, ToolCallBlock
from ph.session import Session, SessionEvent, derive_event_message, is_in_place_rewrite
from ph.session.writers import log_writer
from ph.system_prompt.assembly import (
    ORDER_TOOL_GUIDANCE,
    AssembleContext,
    PromptContext,
    PromptSection,
)
from ph.text import count_of
from ph.tools import TOOL_DISPATCH_EVENT_TYPES, ToolCallView, ToolResultView
from ph.tools.definition import (
    Deny,
    PreToolDecision,
    ToolExecution,
    ToolModel,
    ToolOutput,
    ToolRunContext,
    define_tool,
    text_content,
)

_LOG = log_writer(__name__)

__all__ = [
    "MAX_REQUIRES",
    "MAX_TODOS",
    "MAX_TODO_CONTENT",
    "PARALLEL_CALL_ERROR",
    "SKILL",
    "TOOL_NAME",
    "WRITE_TODOS_SYSTEM_PROMPT",
    "PlanError",
    "apply",
    "blocked_by",
    "outstanding_steps",
    "render_todo_list",
    "todos_of",
    "unevidenced",
]

TOOL_NAME = "write_todos"

MAX_TODO_CONTENT = 500
"""The maximum length of the text of one entry.

**The list is the one string written by the model that is included in every
later prompt.** It is a `context`, materialized again on each turn. After
compaction has shadowed the turns that wrote it, it is the only remaining
statement of the plan. An entry is therefore not a place to paste a diff. Text
that is unbounded here is unbounded in every request for the rest of the
session. That is the lesson of P7-13 one layer up: a bound belongs where the
unbounded thing is written, and not at each reader.

The bound is generous for the purpose of the tool. The upstream examples are
phrases like "Run the pre-flight check". Text that is too long is refused and
not clipped. A model that was told that its text was accepted, when the text was
truncated, would continue to believe that the list says something that it does
not say. The bound reaches the model as `maxLength` in the schema of the tool,
so it is a stated rule and not a surprise."""

SKILL = "skill"
"""The value of `source` on an entry that a skill added, and not the model.

**The harness issues it, like `worked`, and for the same reason.** If the model
could *label* an entry as the entry of a skill, it could also label that entry
as its own and then delete it. The arguments of the model therefore carry no
`source` at all. Each write attaches `source` again from the list that the log
already holds, and a write that does not carry a skill step forward is refused.

This enables one rule, and that rule is all of P7-18. The model may add entries
beside a seeded step and may mark a seeded step done. It may not drop, reorder
or reword one. That is the difference between a procedure and a suggestion.

There is one value and not a vocabulary. An entry that the model wrote carries
no `source` at all. There is no `"model"` constant, because the question that
every reader asks *is* whether `source` is present. A second value would be a
field that the model could set."""

MAX_REQUIRES = 20
"""The number of entries on which one entry may declare a dependency.

It is bounded for the same reason as `MAX_TODO_CONTENT`: the list is included in
every later prompt. It is low because a step that waits on twenty other steps
has not been broken down. The product is the thing that needs a bound: 100
entries with 20 references each is already more plan than this tool is for."""

MAX_TODOS = 100
"""The number of entries that one list may hold. `MAX_TODO_CONTENT` explains why
there is a bound.

The value is well past any real plan. The upstream guidance is to skip the tool
entirely for work that is "a few tool calls". This count is what bounds the
product. Without it, entries inside the length limit still grow the prompt
without end."""

GLYPHS = {"pending": "[ ]", "in_progress": "[~]", "completed": "[x]"}
"""How the list is rendered back to the model. The three states are on purpose
the same three that the sidebar draws, so a person and the model read one list."""

# --------------------------------------------------------------------------
# Verbatim from `langchain.agents.middleware.todo`, with one addition: the
# `## Dependencies` section at the end of the tool description is pH's own and
# describes `requires`. The text was checked against 1.3.18, which satisfies the
# pinned `langchain>=1.3.17` of the plan. It is copied and not imported, for two
# reasons. ph does not depend on langchain. These constants are also prompt
# *text*: an upgrade is most likely to reword that text silently, and the gate
# of the row is about its exact words.
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

`write_todos` tracks your work; it does not deliver the answer. Whatever the user asked for — computations, summaries, comparisons, data — must appear as text content in a message after your final `write_todos` call. Marking the last todo complete is not itself an answer to the user.

## Dependencies

If a step cannot start until another one is done, say so with `requires` rather than in prose: list the exact `content` of the entries it waits on. The list is checked when you write it — a name that is not in the list, a step that waits on itself, a cycle, or a step marked `in_progress` while something it waits on is unfinished are all refused, and nothing is written. `requires` is optional; leave it out when the order does not matter."""  # noqa: E501

PARALLEL_CALL_ERROR = (
    "Error: The `write_todos` tool should never be called multiple times "
    "in parallel. Please call it only once per model invocation to update "
    "the todo list."
)
"""Verbatim from upstream, and the gate of the row. It is assembled from the same
three fragments as upstream, so that a diff against the upstream `todo.py` stays
readable."""


class PlanError(ValueError):
    """The plan contradicts itself, so nothing is written.

    The *model* reads this refusal and can fix the plan on its next turn. The
    refusal has the shape that B5 normalizes: a raised error becomes an
    `is_error` result and not a denial. The model did not do anything forbidden.
    It wrote a plan that does not hold together, and the fix is a corrected plan.

    The plan is refused and not repaired, and the log is left unchanged. The
    reason is the same as for the parallel-call rule: if the written list
    differs from the list that the model asked for, the model reasons about it
    wrongly for the rest of the session."""


# **`TodoItem` is where the port forks from langchain** (P7-16). The upstream
# `Todo` is `{content, status}` and has no `requires`. The upstream prompt works
# around that absence in prose ("When blocked, create a new task describing what
# needs to be resolved"). That is a dependency graph written as a naming
# convention, and nothing can read it, order it or check it. A declared
# dependency can be checked: a reference to nothing, a self-reference and a cycle
# are all refused, and `blocked_by` answers what can really be started.
#
# There is still no `activeForm`, on purpose. The reason was never that upstream
# lacks it. pH renders a glyph beside the content in both the sidebar and the
# prompt, so a second phrasing of one task would be a field with no reader.
# `requires` has three readers.
#
# This history is a comment and not part of the class docstring. Pydantic copies
# the class docstring into the JSON schema of `write_todos`, so the model reads
# it on every request. The docstring therefore says only what an entry is.
class TodoItem(ToolModel):
    """One entry: what the step is, where it has got to, and what it waits on."""

    content: str = Field(
        description="What the step is.",
        max_length=MAX_TODO_CONTENT,
    )
    status: Literal["pending", "in_progress", "completed"] = Field(
        "pending", description="Where the step has got to."
    )
    requires: list[str] = Field(
        default_factory=list,
        description=(
            "The `content` of entries in THIS list that must be completed first. "
            "Use it instead of describing a blocker in prose. A name that is not "
            "in the list, a self-reference, or a cycle is refused."
        ),
        max_length=MAX_REQUIRES,
    )


class WriteTodosArgs(ToolModel):
    todos: list[TodoItem] = Field(
        description="The ENTIRE list. This call replaces the previous one; there are no "
        "partial updates and no per-item edits.",
        max_length=MAX_TODOS,
    )


# ------------------------------------------------------------------ reading --


def todos_of(session: Session | None) -> list[dict[str, Any]]:
    """The current list: the last `todo/write` in the log, or an empty list.

    The last write wins over an append-only log. That is the complete storage
    design: a fold and not a table. A fork therefore inherits the list that its
    prefix ended with, and a resume shows what the user left. `Session.latest`
    is the helper for this question. It is incremental, so a prompt that is
    assembled on every step does not rescan a log that is mostly
    `assistant/chunk` events.
    """
    event = session.latest("todo/write") if session is not None else None
    if event is None:
        return []
    # `thaw_json` because the payload is frozen. A `MappingProxyType` is not a
    # `dict`, and that bug has now occurred three times in this project.
    return _entries(thaw_json(event.data.get("todos")))


def steps_of(todos: list[dict[str, Any]]) -> list[str]:
    """The contents of the entries that a skill seeded, in order.

    Order matters as much as membership. A seeded procedure in a different order
    is a different procedure, and `requires` alone would not detect a swap of two
    steps that do not depend on each other.
    """
    return [_content(one) for one in todos if one.get("source") == SKILL]


def startable(todos: list[dict[str, Any]]) -> list[str]:
    """The entries that are not finished and wait on nothing unfinished.

    A `turn-stopping` listener asks this: does the plan say that some work can
    begin now? An empty answer while entries are still pending means that all
    remaining work is blocked. That is different from done. The listener then
    does nothing, and does not name work that the plan itself says cannot start.
    """
    waiting = blocked_by(todos)
    return [
        _content(one)
        for one in todos
        if one.get("status") != "completed" and _content(one) not in waiting
    ]


def outstanding_steps(session: Session | None) -> set[str]:
    """The contents of the seeded entries that are not finished, read **frozen**.

    A `turn-stopping` listener asks this one time in each turn for *every*
    session in the deployment, including the sessions that never read a skill.
    The answer therefore comes from the frozen payload, and the function does
    not thaw a list that it would then discard. `Session.latest` is an
    incremental fold (measured at 0.08 µs for any log length), so a session that
    never used the tool costs one dict lookup and nothing else.

    The function is here and not in `skill-steps`, because the meaning of "an
    entry a skill put there" belongs to this module. `steps_of`, `_carried` and
    this function are the three readers of `source`. A fourth version across a
    package boundary would let the sidebar and the steer disagree about what is
    outstanding.
    """
    previous = session.latest("todo/write") if session is not None else None
    return {
        _content(one)
        for one in _recorded(previous)
        if one.get("source") == SKILL and one.get("status") != "completed"
    }


def blocked_by(todos: list[dict[str, Any]]) -> dict[str, list[str]]:
    """For each entry, the declared dependencies that are not yet completed.

    `requires` exists for this fold. "What can really be started now" is a
    question about the whole list, and before this fold only prose could ask it.
    An entry that is missing from the result can be started.
    """
    done = {_content(one) for one in todos if one.get("status") == "completed"}
    waiting: dict[str, list[str]] = {}
    for todo in todos:
        pending = [name for name in _requires(todo) if name not in done]
        if pending:
            waiting[_content(todo)] = pending
    return waiting


def _entries(value: object) -> list[dict[str, Any]]:
    """The `todos` of a tool argument, as the list of entries that it claims to be.

    The reader here was `list(value or ())`. It had the failure that the
    docstring of `as_seq` describes: a `str` is a `Sequence`, so a `todos` field
    written as a string became a list of its *characters*, and every later rule
    read one-letter entries. `as_seq` refuses the string. The shape test then
    drops anything in the array that is not an entry. That is the same policy
    one level down: a row with the wrong shape costs only that row.
    """
    return [one for one in as_seq(value) if isinstance(one, dict)]


def unevidenced(todos: list[dict[str, Any]]) -> list[str]:
    """The completed entries for which the harness saw no work.

    The tool counts `worked`, and the model never supplies it. This is therefore
    the one thing in the list that the model cannot assert. An entry marked done
    in a window where the agent called no other tool is a claim with no
    evidence. That does not make the entry wrong, because a step can be "decide
    the approach". But a checklist that the model marks itself otherwise hides
    exactly this difference: a tick with work behind it, or a bare tick (P5-16).
    """
    return [
        _content(one) for one in todos if one.get("status") == "completed" and not one.get("worked")
    ]


def render_todo_list(todos: list[dict[str, Any]]) -> str:
    """The list as the model sees it between turns.

    A blocked entry says what it waits on. The list is the statement of the
    model itself, and the purpose of a declared dependency is a reminder at the
    moment when the model chooses what to do next. Nothing else is rendered. The
    receipt in `worked` is for a person who reads the card. To repeat it here
    would spend tokens on every turn to tell the model something that it cannot
    act on.
    """
    waiting = blocked_by(todos)
    lines = []
    for todo in todos:
        content = _content(todo)
        glyph = GLYPHS.get(as_str(todo.get("status")), "[ ]")
        blockers = waiting.get(content) if todo.get("status") != "completed" else None
        suffix = f" (waiting on: {', '.join(blockers)})" if blockers else ""
        lines.append(f"{glyph} {content}{suffix}")
    return "## Todo list\n\n" + "\n".join(lines)


def _checked(todos: list[dict[str, Any]]) -> None:
    """Refuse a plan that contradicts itself, before anything is written.

    A plan can contradict itself in three ways, and the model can only have
    meant each one by mistake:

    * A dependency on an entry that is not in the list.
    * A cycle. An entry that waits on itself is the one-node case.
    * An entry that claims to be in progress or finished while something that it
      *said* it waits on is not finished.

    The last check compares the model against its own statement. It is not a
    policy imposed on the model, and that keeps it on the right side of the line
    in P5-16. `requires` is optional for each entry. A model that finds the rule
    inconvenient does not declare the dependency, and then nothing here has an
    opinion.

    The function also refuses two entries with the same content, because
    `requires` names entries by their content.
    """
    edges: dict[str, list[str]] = {}
    for todo in todos:
        content = _content(todo)
        if content in edges:
            raise PlanError(
                f"two entries share the content {content!r}; `requires` names entries by "
                "their content, so each must be distinct"
            )
        edges[content] = _requires(todo)
    for content, wants in edges.items():
        for name in wants:
            if name not in edges:
                raise PlanError(f"{content!r} requires {name!r}, which is not in the list")

    # One path, appended and popped, and not a `walking` set beside a `trail`
    # list. Membership and the text of the cycle are the same question, and two
    # structures would ask it twice.
    path: list[str] = []
    settled: set[str] = set()

    def visit(node: str) -> None:
        if node in settled:
            return
        if node in path:
            raise PlanError(
                f"the plan has a cycle: {' -> '.join([*path[path.index(node) :], node])}"
            )
        path.append(node)
        for nxt in edges[node]:
            visit(nxt)
        path.pop()
        settled.add(node)

    for name in edges:
        visit(name)

    waiting = blocked_by(todos)
    for todo in todos:
        content = _content(todo)
        if todo.get("status") in ("in_progress", "completed") and content in waiting:
            raise PlanError(
                f"{content!r} is {todo.get('status')} but waits on "
                f"{', '.join(repr(name) for name in waiting[content])}, which is not completed"
            )


def _work_since(session: Session, since: int) -> int:
    """How many tools the agent called between `since` and now, ignoring this one.

    The count comes from `tool/call`. Since P7-15 that record exists only for a
    call that the pipeline let through and was about to run. The count is
    therefore work *attempted*, which is the correct thing to count for "did
    anything happen, or was a box ticked". `write_todos` is excluded, because
    otherwise every window would count itself.

    **Both records are counted, one for each transport.**
    `TOOL_DISPATCH_EVENT_TYPES` explains why they are the same fact. A count of
    `tool/call` alone scored a Code Mode cell that read nine files and edited
    three as *one* piece of work. The nudge for a window that did too little
    then fired at an agent that did plenty (D12).

    The count reads the tail and not the whole log. The previous `todo/write` is
    at most one turn back, and `events_from` copies only what follows it.
    """
    return sum(
        1
        for event in session.events_from(since + 1)
        if event.type in TOOL_DISPATCH_EVENT_TYPES and as_str(event.data.get("name")) != TOOL_NAME
    )


def _content(entry: Mapping[str, Any]) -> str:
    """The `content` of one entry, which is the join key of this module.

    Every rule here keys on it: the dict of `blocked_by`, the `seeded` map of
    `_carried`, the `was` map of `_witnessed`, and the edge table of `_checked`.
    It was written out fifteen times in two different forms. There is now one
    reader, for the reason that `_recorded` states: a key that is extracted in
    fifteen places can drift in fifteen places on the day that it is normalized
    (`.strip()`) or renamed.
    """
    return as_str(entry.get("content"))


def _requires(entry: Mapping[str, Any]) -> list[str]:
    """What this entry says it waits on, narrowed in the same way as `content`.

    The names were read with a bare `str()`. A `requires: [3]` then compared an
    invented `"3"` against a `done` set that was built with `as_str`: two
    narrowings of one field, five lines apart.
    """
    return [as_str(name) for name in as_seq(entry.get("requires"))]


def _recorded(previous: SessionEvent | None) -> list[Mapping[str, Any]]:
    """The entries that the last `todo/write` holds, read **frozen**.

    Both rules use this one read. `todos_of` would deep-copy the whole previous
    list. That was measured as 12x more expensive at the caps, and about 95% of
    the cost was the copy of `requires`. `_carried` and `_witnessed` together
    need four scalars from each entry. A thaw would therefore be paid twice for
    each write, to build something that neither of them mutates. A
    `MappingProxyType` answers `.get` as well as a `dict` does.
    """
    entries = as_seq(previous.data.get("todos")) if previous else ()
    return [one for one in entries if isinstance(one, Mapping)]


def _carried(
    recorded: list[Mapping[str, Any]], todos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Attach again what a skill seeded, and refuse a write that lost it (P7-18).

    **The arguments of the model never carry `source`.** It is read from the
    list that the log already holds and is put back here. A procedure therefore
    cannot be forged into existence or removed by a change of label. `worked`
    has the same shape, for the same reason: a claim that the claimant issues
    about itself is not evidence.

    Three things are refused: a dropped seeded step, reordered seeded steps, and
    a reworded seeded step. A reworded step reads as a drop plus an add, and
    that is what it is. Status is the one thing that the model *may* change,
    because the purpose is to mark progress. The `requires` of a seeded step is
    also put back, so the order that a skill declared cannot be edited out.

    A write to a list that no skill has seeded is not changed. That is every
    write in a deployment that installs no procedural skills.
    """
    seeded = {_content(one): one for one in recorded if one.get("source") == SKILL}
    if not seeded:
        return todos
    kept = [name for name in (_content(one) for one in todos) if name in seeded]
    if kept != list(seeded):
        # Two refusals, because these are two mistakes and the fix differs. A
        # reorder that reported every step as "dropped" told the model that all
        # of the steps were wrong, when only their order was wrong.
        lost = [name for name in seeded if name not in kept]
        if lost:
            raise PlanError(
                f"this plan drops {count_of(len(lost), 'step')} a skill set: "
                f"{', '.join(repr(name) for name in lost)}. Keep them, in order — you "
                "may add your own entries around them and mark these done."
            )
        raise PlanError(
            f"this plan reorders the {count_of(len(seeded), 'step')} a skill set. Keep "
            f"them in the order it declared: {', '.join(repr(name) for name in seeded)}. "
            "You may add your own entries around them and mark these done."
        )
    for todo in todos:
        before = seeded.get(_content(todo))
        if before is not None:
            todo["source"] = SKILL
            todo["requires"] = _requires(before)
    return todos


def _witnessed(
    session: Session,
    previous: SessionEvent | None,
    recorded: list[Mapping[str, Any]],
    todos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the receipt to the entries that have just been completed.

    **The one field in the list that the model does not write.** `worked` is
    what the harness saw happen in the window before this write. A completion
    with a zero in it is a claim with no evidence, and `unevidenced` reads it.
    The purpose is to attach it here and not accept it as an argument: a receipt
    that the claimant issues is not a receipt.

    The value is carried forward for entries that were already complete. A write
    replaces the whole list, and the evidence of an entry is about the window in
    which it was finished, not about this window. An entry that moves *back* out
    of `completed` loses the value. It is no longer a claim, so there is nothing
    to witness.
    """
    was = {_content(one): (one.get("status"), one.get("worked")) for one in recorded}
    worked = _work_since(session, previous.seq if previous else -1)
    for todo in todos:
        if todo.get("status") != "completed":
            continue
        before = was.get(_content(todo))
        todo["worked"] = as_int(before[1]) if before and before[0] == "completed" else worked
    return todos


def _executing_message(event: SessionEvent) -> Message | None:
    """One `assistant/message` event as the message that a batch would run from (L12).

    **A rewrite is not new work.** `SurfaceReplace` appends. An in-place rewrite
    of an *older* message therefore becomes the newest `assistant/message` in
    the log, while the message that is really executing is further back.
    Argument elision is the producer that does this. A reader that asks only for
    the latest event gets a near-copy of something that was answered several
    steps ago.

    The function uses `is_in_place_rewrite` and not
    `is_replacement_surface_event`, for the same reason as `persistence.repair`
    one package over: the narrow predicate is the one that means "not new work".
    The commit refuses any other assistant replacement
    (`surface._assert_assistant_rewrite`), so today the two predicates agree.

    This is a module-level function because `Session.projection` keys its fold
    on `(type, parse)`. A lambda here would build a fold for each call.
    """
    if is_in_place_rewrite(event):
        return None
    return derive_event_message(event)


def _parallel_write_todos(session: Session | None) -> bool:
    """Whether the assistant message that is now executing asked for more than
    one write.

    The answer is read from the log and not from the batch, because the
    `tools/pre-execute` waterfall of pH runs for each call, while the upstream
    rule is about the *message*. The assistant message is committed before any
    of its tool calls runs, so every sibling gets this same answer before the
    first body executes. That makes "every one of them fails" true, and not "all
    but the first". `Session.projection` and `derive_event_message` own the scan
    and the payload shape. A third reader that wrote either one by hand would
    let them drift apart.

    The rule applies to the native transport only, on purpose. It exists because
    two whole-list replacements in one *message* have no defined precedence. A
    Code Mode cell that calls `tools.write_todos` twice has a defined
    precedence: the execution order of the program. The tool declares no
    concurrency safety, so even gathered dispatches run one after another. There
    the last write wins by the statement of the program itself, and a refusal
    would punish a cell for being unambiguous.
    """
    message = session.projection("assistant/message", _executing_message) if session else None
    if message is None:
        return False
    calls = sum(
        1
        for block in message.content
        if isinstance(block, ToolCallBlock) and block.name == TOOL_NAME
    )
    return calls > 1


# ------------------------------------------------------------------- the row --


@plugin("tool-todo", affects="environment", inject=[TOOLS, SYSTEM_PROMPT])
async def apply(ctx: Context, config: None) -> None:
    """Register the tool, its prompt section, its context and its one rule."""

    async def write_todos(args: WriteTodosArgs, run: ToolRunContext) -> dict[str, Any]:
        todos = [item.model_dump(mode="json") for item in args.todos]
        session = run.session
        # One `latest` and one frozen read. Both rules below need a few scalars
        # from the previous list. Two folds for each write were the deep copy
        # that `_witnessed` had already been written to avoid.
        previous = session.latest("todo/write") if session is not None else None
        recorded = _recorded(previous)
        # Provenance before coherence. A write that lost a skill step is refused
        # for *that* reason, which the model can act on. Otherwise it would be
        # refused for whatever the dependency check finds in what is left.
        todos = _carried(recorded, todos)
        _checked(todos)
        if session is not None:
            todos = _witnessed(session, previous, recorded, todos)
            # A tool-owned event: the tool that changed the list is the one that
            # records it. "Model-visible means logged" then holds without the
            # pipeline learning what a todo is (I3).
            _LOG.append(session, "todo/write", {"todos": todos})
        return {"todos": todos}

    ctx.require(TOOLS).register(
        define_tool(
            TOOL_NAME,
            WRITE_TODOS_TOOL_DESCRIPTION,
            parameters=WriteTodosArgs,
            output=ToolOutput(
                # The output is the accepted input. It has the same shape, so it
                # uses the same model. A separate `Value` class would differ
                # only because it lacks the field description.
                schema=WriteTodosArgs,
                # The receipt reaches the card through the channel that was
                # built for it. `presentation_meta` is the durable payload of
                # the tool itself, so the rule "a completion with no work behind
                # it" is stated one time, in the package that counts it.
                # `present_result` receives the arguments and a `ToolResult`,
                # and never the value. Without this, the card would have to
                # derive the rule again from the log, in another package, from a
                # field whose meaning it would then hold a second copy of.
                presentation_meta=lambda args, value: {
                    "unevidenced": unevidenced(_entries(value.get("todos")))
                },
                # The wording of upstream. The evals note a proposed neutral
                # "Todo list updated.". The pinned release still echoes the
                # list, and the echo lets a model that batched several changes
                # see what was really written.
                render=lambda args, value: text_content(f"Updated todo list to {value['todos']}"),
            ),
            execute=write_todos,
            present_call=lambda args: ToolCallView(
                title="Plan", input=render_todo_list(_entries(args.get("todos")))
            ),
            present_result=lambda args, result: ToolResultView(
                title="Plan",
                subtitle=_counts(
                    _entries(args.get("todos")),
                    bare=len(as_seq((result.meta or {}).get("unevidenced"))),
                ),
                is_error=result.is_error,
            ),
        )
    )

    async def refuse_parallel_calls(
        execution: ToolExecution, next_: Next[PreToolDecision]
    ) -> PreToolDecision:
        if execution.name == TOOL_NAME and _parallel_write_todos(execution.session):
            return Deny(reason=PARALLEL_CALL_ERROR)
        return await next_(execution)

    ctx.on("tools/pre-execute", refuse_parallel_calls)

    ctx.require(SYSTEM_PROMPT).section(
        PromptSection(
            name="write-todos",
            # After the tool guidance that the catalog itself contributes. This
            # is advice about one tool, so it reads as a footnote to that list
            # and not as a competing instruction.
            order=ORDER_TOOL_GUIDANCE + 50,
            text=WRITE_TODOS_SYSTEM_PROMPT,
        )
    )

    def current_list(assemble: AssembleContext) -> str:
        todos = todos_of(assemble.session)
        return render_todo_list(todos) if todos else ""

    # A `context`, not a `section`. It changes every time the model writes, and
    # a changing string in the cached prefix would invalidate the prefix on
    # every plan update (A12). It is materialized after the retained history.
    # That is also the correct place to read it: the state as of now, after the
    # conversation that produced it.
    ctx.require(SYSTEM_PROMPT).context(PromptContext(name="todos", text=current_list))


def _counts(todos: list[dict[str, Any]], *, bare: int = 0) -> str:
    """`2 done · 1 doing · 3 to do`, in the order in which work leaves the list.

    When there is one, the line also gives what a person who watches a plan most
    wants to know: an entry that was ticked in a window where the agent ran
    nothing. This is the reader of `unevidenced`. It is the card and not the
    prompt, because this is a fact about the model that the model has no use for
    and cannot act on.
    """
    # No shape test. The caller narrows with `_entries`, which is the purpose of
    # that helper. The caller used `list(args.get("todos") or ())` before, and a
    # shape test here was necessary then.
    tally = Counter(as_str(todo.get("status")) for todo in todos)
    labeled = (("completed", "done"), ("in_progress", "doing"), ("pending", "to do"))
    line = " · ".join(f"{tally[status]} {label}" for status, label in labeled)
    return f"{line} · {bare} unevidenced" if bare else line
