"""`tool-todo` — planning as a cognitive anchor (P4-01, G1).

Deep Agents' `TodoListMiddleware`, ported onto pH's seams rather than onto a
middleware stack: a tool on `ctx.tools`, a tool-owned `todo/write` event, a
prompt section, and a `tools/pre-execute` listener for the one enforcement rule.
That decomposition is D12 — a stabilization feature attaches to waterfalls,
never to the loop — and it is why this is a row a profile opts into rather than
a parameter on the driver.

**Forked from upstream at P7-16, deliberately and in two places.** `requires`
declares what a step waits on, which upstream's schema has no room for and its
*prompt* works around in prose ("When blocked, create a new task describing what
needs to be resolved") — a dependency graph written as a naming convention that
nothing can read, order or check. And a completed entry carries `worked`, the
number of tools the harness saw run in the window it was finished in: the one
field in the list the model does not write, because a receipt the claimant issues
is not a receipt. Upstream's prompt text and the parallel-call rule are still
tracked verbatim and still gated; the schema is no longer theirs, and the
comparison that produced both is recorded against `sources/OpenMonoAgent.ai`.

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
from ph.text import count_of
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
"""How long one entry's text may be.

**The list is the one model-written string that rides every later prompt.** It is
a `context`, materialized fresh each turn and — after compaction has shadowed the
turns that wrote it — the only surviving statement of the plan. So an entry is
not a place to paste a diff into: unbounded here is unbounded in every request
for the rest of the session, which is P7-13's lesson one layer up (a bound
belongs where the unbounded thing is written, not at each reader).

Generous against what the tool is for — upstream's own examples are phrases like
"Run the pre-flight check" — and a refusal rather than a clip, because a model
told its text was accepted when it was truncated would go on believing the list
says something it does not. The bound reaches the model as `maxLength` in the
tool's own schema, so it is a stated rule rather than a surprise."""

SKILL = "skill"
"""What `source` says on an entry a skill put there, rather than the model.

**Harness-issued, like `worked`, and for the same reason**: an entry the model
could *label* as a skill's is one it could also label as its own and then delete.
So the model's arguments carry no `source` at all — this is re-attached on every
write from the list the log already holds, and a write that fails to carry a
skill's step forward is refused.

The rule it enables is the whole of P7-18: the model may add entries beside a
seeded step and may mark one done, and may not drop, reorder or reword one. That
is the difference between a procedure and a suggestion.

One value, not a vocabulary: an entry the model wrote carries no `source` at
all. There is no `"model"` constant because presence *is* the question every
reader asks, and a second value would be a field the model could set."""

MAX_REQUIRES = 20
"""How many entries one entry may declare a dependency on.

Bounded for `MAX_TODO_CONTENT`'s reason — the list rides every later prompt — and
low because a step waiting on twenty others is a step that has not been broken
down. The product is what needs bounding: 100 entries with 20 references each is already
more plan than any of this is for."""

MAX_TODOS = 100
"""How many entries one list may hold. See `MAX_TODO_CONTENT` for why bounded.

Well past any real plan — upstream's guidance is to skip the tool entirely for
work that is "a few tool calls" — and it is the count that bounds the product:
without it, entries inside the length limit still grow the prompt without end."""

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

`write_todos` tracks your work; it does not deliver the answer. Whatever the user asked for — computations, summaries, comparisons, data — must appear as text content in a message after your final `write_todos` call. Marking the last todo complete is not itself an answer to the user.

## Dependencies

If a step cannot start until another one is done, say so with `requires` rather than in prose: list the exact `content` of the entries it waits on. The list is checked when you write it — a name that is not in the list, a step that waits on itself, a cycle, or a step marked `in_progress` while something it waits on is unfinished are all refused, and nothing is written. `requires` is optional; leave it out when the order does not matter."""  # noqa: E501

PARALLEL_CALL_ERROR = (
    "Error: The `write_todos` tool should never be called multiple times "
    "in parallel. Please call it only once per model invocation to update "
    "the todo list."
)
"""Verbatim from upstream, and the row's gate. Assembled from the same three
fragments so a diff against `todo.py` stays readable."""


class PlanError(ValueError):
    """The plan contradicts itself, so nothing is written.

    A refusal the *model* reads and can fix on its next turn, in the shape B5
    normalizes — a raised error becomes an `is_error` result rather than a
    denial, because the model did not do anything forbidden: it wrote a plan
    that does not hold together, and the fix is a corrected plan.

    Refused rather than repaired, and the log left alone, for the reason the
    parallel-call rule refuses: a list written differently from the one the
    model asked for is a list it will reason about wrongly for the rest of the
    session."""


class TodoItem(ToolModel):
    """One entry: what the step is, where it has got to, and what it waits on.

    **This is where the port forks from langchain** (P7-16). Upstream's `Todo` is
    `{content, status}`, and `requires` is not in it — but the absence is what
    upstream's own prompt works around in prose ("When blocked, create a new task
    describing what needs to be resolved"), which is a dependency graph written
    as a naming convention that nothing can read, order or check. Declared, it is
    checkable: a reference to nothing, a self-reference and a cycle are all
    refused, and `blocked_by` answers what is actually startable.

    Still deliberately no `activeForm` — the reason there was never that upstream
    lacks it but that pH renders a glyph beside the content in both the sidebar
    and the prompt, so a second phrasing of one task would be a field with no
    reader. `requires` has three.
    """

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


def steps_of(todos: list[dict[str, Any]]) -> list[str]:
    """The contents of entries a skill seeded, in order.

    Order matters as much as membership: a seeded procedure reordered is a
    different procedure, and `requires` alone would not notice a swap between two
    steps that happen not to depend on each other.
    """
    return [str(one.get("content")) for one in todos if one.get("source") == SKILL]


def startable(todos: list[dict[str, Any]]) -> list[str]:
    """Entries that are not finished and are waiting on nothing unfinished.

    What a `turn-stopping` listener asks: is there work this plan says can begin
    right now? An empty answer with entries still pending means everything left
    is blocked — a different thing from being done, and the listener stands down
    rather than naming work the plan itself says cannot start.
    """
    waiting = blocked_by(todos)
    return [
        str(one.get("content"))
        for one in todos
        if one.get("status") != "completed" and str(one.get("content")) not in waiting
    ]


def outstanding_steps(session: Session | None) -> set[str]:
    """The contents of seeded entries that are not finished, read **frozen**.

    The question a `turn-stopping` listener asks once per turn for *every*
    session in the deployment, including the ones that never read a skill — so it
    is answered off the frozen payload rather than by thawing a list it is about
    to throw away. `Session.latest` is an incremental fold (measured 0.08 µs
    whatever the log length), so a session that never used the tool costs a dict
    lookup and nothing else.

    Here rather than in `skill-steps`, because "an entry a skill put there" is
    this module's sentence: `steps_of`, `_carried` and this are the three readers
    of `source`, and a fourth spelling across a package boundary is how the
    sidebar and the steer come to disagree about what is outstanding.
    """
    previous = session.latest("todo/write") if session is not None else None
    return {
        str(one.get("content"))
        for one in _recorded(previous)
        if one.get("source") == SKILL and one.get("status") != "completed"
    }


def blocked_by(todos: list[dict[str, Any]]) -> dict[str, list[str]]:
    """For each entry, the dependencies it declared that are not yet completed.

    The fold `requires` exists for: "what can actually be started now" is a
    question about the whole list, and before this it was a question only prose
    could ask. An entry missing from the result is startable.
    """
    done = {str(one.get("content")) for one in todos if one.get("status") == "completed"}
    waiting: dict[str, list[str]] = {}
    for todo in todos:
        pending = [str(name) for name in todo.get("requires") or () if str(name) not in done]
        if pending:
            waiting[str(todo.get("content"))] = pending
    return waiting


def unevidenced(todos: list[dict[str, Any]]) -> list[str]:
    """Completed entries the harness saw no work behind.

    `worked` is counted by the tool, never supplied by the model, so this is the
    one thing in the list the model cannot assert: an entry marked done in a
    window where the agent called no other tool is a claim with nothing behind
    it. It does not make the entry wrong — a step can be "decide the approach" —
    but the difference between a tick with work behind it and a bare tick is
    exactly what a checklist the model marks itself otherwise hides (P5-16).
    """
    return [
        str(one.get("content"))
        for one in todos
        if one.get("status") == "completed" and not one.get("worked")
    ]


def render_todo_list(todos: list[dict[str, Any]]) -> str:
    """The list as the model sees it between turns.

    Blocked entries say what they are waiting on, because the list is the model's
    own statement and the point of declaring a dependency is to be reminded of it
    at the moment of choosing what to do next. Nothing else is rendered — the
    receipt `worked` carries is for a person reading the card, and repeating it
    here would spend tokens every turn to tell the model something it cannot act
    on.
    """
    waiting = blocked_by(todos)
    lines = []
    for todo in todos:
        content = str(todo.get("content", ""))
        glyph = GLYPHS.get(str(todo.get("status")), "[ ]")
        blockers = waiting.get(content) if todo.get("status") != "completed" else None
        suffix = f" (waiting on: {', '.join(blockers)})" if blockers else ""
        lines.append(f"{glyph} {content}{suffix}")
    return "## Todo list\n\n" + "\n".join(lines)


def _checked(todos: list[dict[str, Any]]) -> None:
    """Refuse a plan that contradicts itself, before anything is written.

    Three ways it can, and each is a thing the model can only have meant by
    mistake: a dependency on an entry that is not in the list, a cycle — of which
    an entry waiting on itself is the one-node case — and an entry claiming to be
    under way or finished while something it *said* it waits on is not.

    That last one is a check of the model against its own statement rather than
    a policy imposed on it, which is what keeps this on the right side of
    P5-16's line. `requires` is optional per entry; a model that finds the rule
    inconvenient simply does not declare the dependency, and then nothing here
    has an opinion.
    """
    edges: dict[str, list[str]] = {}
    for todo in todos:
        content = str(todo.get("content", ""))
        if content in edges:
            raise PlanError(
                f"two entries share the content {content!r}; `requires` names entries by "
                "their content, so each must be distinct"
            )
        edges[content] = [str(name) for name in todo.get("requires") or ()]
    for content, wants in edges.items():
        for name in wants:
            if name not in edges:
                raise PlanError(f"{content!r} requires {name!r}, which is not in the list")

    # One path, appended and popped, rather than a `walking` set beside a `trail`
    # list: membership and the cycle's own text are the same question asked twice.
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
        content = str(todo.get("content", ""))
        if todo.get("status") in ("in_progress", "completed") and content in waiting:
            raise PlanError(
                f"{content!r} is {todo.get('status')} but waits on "
                f"{', '.join(repr(name) for name in waiting[content])}, which is not completed"
            )


def _work_since(session: Session, since: int) -> int:
    """How many tools the agent called between `since` and now, ignoring this one.

    Counted off `tool/call`, which since P7-15 exists only for a call the
    pipeline let through and was about to run — so this is work *attempted*,
    which is the honest thing to count for "did anything happen, or was a box
    ticked". `write_todos` is excluded or every window would score itself.

    From the tail rather than the whole log: the previous `todo/write` is at most
    a turn ago, and `events_from` copies what follows it.
    """
    return sum(
        1
        for event in session.events_from(since + 1)
        if event.type == "tool/call" and str(event.data.get("name")) != TOOL_NAME
    )


def _recorded(previous: Any) -> list[Mapping[str, Any]]:
    """The entries the last `todo/write` holds, read **frozen**.

    The one read both rules take. `todos_of` would deep-copy the whole previous
    list — measured 12x dearer at the caps, ~95% of it copying `requires` — and
    between them `_carried` and `_witnessed` want four scalars per entry, so a
    thaw would be paid twice per write to build something neither of them
    mutates. A `MappingProxyType` answers `.get` perfectly well.
    """
    entries = seq_of(previous.data.get("todos")) if previous else ()
    return [one for one in entries if isinstance(one, Mapping)]


def _carried(
    recorded: list[Mapping[str, Any]], todos: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Re-attach what a skill seeded, and refuse a write that lost it (P7-18).

    **The model's arguments never carry `source`.** It is read from the list the
    log already holds and put back here, so a procedure cannot be forged into
    existence or labelled away — the same shape `worked` uses, and for the same
    reason: a claim the claimant issues about itself is not evidence.

    Three things are refused: dropping a seeded step, reordering them, and
    rewording one — the last reads as a drop plus an add, which is what it is.
    Status is the one thing the model *may* change, because marking progress is
    the whole point. `requires` on a seeded step is put back too, so the order a
    skill declared cannot be edited out from underneath it.

    A write that has never seen a skill is untouched, which is every write in a
    deployment that installs no procedural skills.
    """
    seeded = {str(one.get("content")): one for one in recorded if one.get("source") == SKILL}
    if not seeded:
        return todos
    kept = [name for name in (str(one.get("content")) for one in todos) if name in seeded]
    if kept != list(seeded):
        # Two refusals, because they are two mistakes and the fix differs. A
        # reorder that named every step as "dropped" told the model all of them
        # were wrong when only their order was.
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
        before = seeded.get(str(todo.get("content")))
        if before is not None:
            todo["source"] = SKILL
            todo["requires"] = list(before.get("requires") or ())
    return todos


def _witnessed(
    session: Session,
    previous: Any,
    recorded: list[Mapping[str, Any]],
    todos: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach the receipt to entries that have just been completed.

    **The one field in the list the model does not write.** `worked` is what the
    harness saw happen in the window before this write; a completion with a zero
    in it is a claim with nothing behind it, and `unevidenced` is what reads it.
    Attaching it here rather than accepting it as an argument is the whole point:
    a receipt the claimant issues is not a receipt.

    Carried forward for entries that were already complete, because a write
    replaces the whole list and an entry's evidence is about the window it was
    finished in, not this one. An entry moved *back* out of `completed` loses it
    — it is no longer a claim, so there is nothing to witness.
    """
    was = {str(one.get("content")): (one.get("status"), one.get("worked")) for one in recorded}
    worked = _work_since(session, previous.seq if previous else -1)
    for todo in todos:
        if todo.get("status") != "completed":
            continue
        before = was.get(str(todo.get("content")))
        todo["worked"] = int(before[1] or 0) if before and before[0] == "completed" else worked
    return todos


def seq_of(value: Any) -> tuple[Any, ...]:
    """A frozen JSON array as a tuple, without thawing what is inside it."""
    return tuple(value) if isinstance(value, (list, tuple)) else ()


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
        # One `latest`, one frozen read: both rules below want a few scalars off
        # the previous list, and folding it twice per write was the deep copy
        # `_witnessed` had already been written to avoid.
        previous = session.latest("todo/write") if session is not None else None
        recorded = _recorded(previous)
        # Provenance before coherence: a write that lost a skill's step is
        # refused for *that*, which is actionable, rather than for whatever the
        # dependency check notices about the wreckage left behind.
        todos = _carried(recorded, todos)
        _checked(todos)
        if session is not None:
            todos = _witnessed(session, previous, recorded, todos)
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
                # The receipt reaches the card through the channel built for it:
                # `presentation_meta` is the tool's own durable payload, so the
                # rule "a completion with no work behind it" is stated once, in
                # the package that counts it. `present_result` is handed the
                # arguments and a `ToolResult` — never the value — so without
                # this the card would have to re-derive it from the log, in
                # another package, from a field whose meaning it would then own
                # a second copy of.
                presentation_meta=lambda args, value: {
                    "unevidenced": unevidenced(list(value.get("todos") or ()))
                },
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
                subtitle=_counts(
                    list(args.get("todos") or ()),
                    bare=len((result.meta or {}).get("unevidenced") or ()),
                ),
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


def _counts(todos: list[Any], *, bare: int = 0) -> str:
    """`2 done · 1 doing · 3 to do`, in the order work leaves the list.

    Plus, when there is one, the thing a person watching a plan most wants to
    know: an entry ticked in a window where the agent ran nothing. That is
    `unevidenced`'s reader — the card, not the prompt, because it is a fact about
    the model that the model has no use for and cannot act on.
    """
    tally = Counter(str(todo.get("status")) for todo in todos if isinstance(todo, Mapping))
    labelled = (("completed", "done"), ("in_progress", "doing"), ("pending", "to do"))
    line = " · ".join(f"{tally[status]} {label}" for status, label in labelled)
    return f"{line} · {bare} unevidenced" if bare else line
