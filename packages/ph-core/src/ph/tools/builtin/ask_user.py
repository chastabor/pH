"""`tool-ask-user` — the model asks the person a question (P7-09).

`ctx.user_questions` shipped with a definition, a front end and a modal, and no
caller: `grep '\\.ask('` across `packages/*/src` found nothing. This is the
producer, so the seam has all three parts and the whole path is exercised rather
than intended.

**Shipped disabled, and armed by the profile that has a screen.** The row is
`disabled: true` in `ph-base` and enabled by `tui.yaml` — the `tool-todo` idiom.
An unattended posture therefore never sees the tool at all: no schema in the
prompt, no turn spent calling it, nothing in the log. Paying nothing for a
capability the deployment cannot perform is the same rule `subagent-task`
follows when it registers nothing without a provider.

**Not gated by approval.** Asking permission to ask is one more interruption for
the same person, and the ask has to happen anyway for the model to be directed.

**Nobody attending is a result, not an error.** The tool answers with a sentence
the model can act on, the way an unreadable attachment degrades to a pointer, so
an interactive profile run with every front end detached — `/autonomous`, a
sandboxed run, a daemon whose UIs have closed — degrades instead of hanging.

@module ph.tools.builtin.ask_user
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...cordis import Context, plugin
from ...seams.user_questions import UserQuestion
from ..definition import ToolModel, ToolOutput, ToolRunContext, define_tool, text_content
from ..presentation import ToolCallView, ToolResultView

__all__ = ["apply"]

DESCRIPTION = """Ask the person running this session a question, and wait for their answer.

Use it when the work genuinely forks on something only they can settle — which
option they want, which of two files they meant, whether an assumption holds.
Do not use it for anything you can determine yourself from the code, the
repository or the conversation: every call stops the work and waits for a human.

Give `options` when the answer is a choice; they pick one, or several if
`multiSelect` is set, and the answer comes back as their choices joined by
commas. Leave `options` out for free text."""

UNATTENDED = (
    "Nobody is attending this session, so the question was not put to anyone. "
    "Continue without an answer: choose the most reasonable option yourself and "
    "state the assumption you made, or say what you would need in order to proceed."
)
"""What the model reads when there is no one to ask.

Phrased as an instruction rather than as an error because it arrives as a
successful result: "no answer" is this seam's defined failure mode, and a model
told only that something failed retries it."""


class AskUserArgs(ToolModel):
    question: str = Field(description="The question, written for a person to read.")
    options: list[str] | None = Field(
        None, description="The choices to offer. Omit for a free-text answer."
    )
    header: str | None = Field(
        None, description="A few words naming what is being decided, shown as a label."
    )
    multi_select: bool = Field(
        False, description="Allow more than one option to be chosen. Ignored without options."
    )


class AskUserValue(ToolModel):
    answer: str | None = None
    answered: bool


@plugin("tool-ask-user", inject=["tools", "user_questions"])
async def apply(ctx: Context, config: Any) -> None:
    """Register the question tool."""

    async def ask_user(args: AskUserArgs, run: ToolRunContext) -> Any:
        answer = await ctx.user_questions.ask(
            UserQuestion(
                question=args.question,
                options=args.options,
                header=args.header,
                multi_select=args.multi_select,
                # The call id, so the log's `askId` is the one string that
                # already joins `tool/call`, `tool/result` and the two
                # `question/*` records of this one exchange.
                ask_id=run.call_id,
            ),
            session=run.session,
        )
        return {"answer": answer, "answered": answer is not None}

    ctx.tools.register(
        define_tool(
            "ask_user",
            DESCRIPTION,
            parameters=AskUserArgs,
            output=ToolOutput(
                schema=AskUserValue,
                render=lambda args, value: text_content(
                    str(value["answer"]) if value["answered"] else UNATTENDED
                ),
            ),
            execute=ask_user,
            # It changes nothing and asks nobody's permission to run; what it
            # costs is a person's attention, which no gate can give back.
            is_irreversible=False,
            present_call=lambda args: ToolCallView(
                title=str(args.get("header") or "Question"),
                input=str(args.get("question", "")),
            ),
            present_result=lambda args, result: ToolResultView(
                title=str(args.get("header") or "Question"),
                subtitle=str(args.get("question", ""))[:120],
                is_error=result.is_error,
            ),
        )
    )
