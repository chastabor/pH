"""`tool-ask-user`: the model asks the user a question (P7-09).

`ctx.user_questions` shipped with a definition, a front end and a modal, but
with no caller: `grep '\\.ask('` across `packages/*/src` found nothing. This tool
is the producer. The seam therefore has all three parts, and the whole path is
exercised and not only intended.

**The row ships disabled, and the profile that has a screen enables it.** The
row is `disabled: true` in `ph-base`, and `tui.yaml` enables it. This is the
`tool-todo` idiom. An unattended posture therefore never sees the tool: there is
no schema in the prompt, no turn is spent on a call, and nothing is in the log.
A deployment pays nothing for a capability that it cannot perform.
`subagent-task` follows the same rule when it registers nothing without a
provider.

**Approval does not gate the tool.** A request for permission to ask is one more
interruption for the same user. The question must also be asked anyway, so that
the model gets its direction.

**An absent user is a result, not an error.** The tool answers with a sentence
that the model can act on, in the way that an unreadable attachment degrades to
a pointer. An interactive profile that runs with every front end detached
therefore degrades and does not hang. Examples are `/autonomous`, a sandboxed
run, and a daemon whose UIs have closed.

@module ph.tools.builtin.ask_user
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from ...characters import head
from ...cordis import Context, plugin
from ...json import as_str
from ...keys import TOOLS, USER_QUESTIONS
from ...seams.user_questions import AskResolution, UserQuestion
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

The text is an instruction and not an error, because it arrives as a successful
result. "No answer" is the defined failure mode of this seam, and a model that
is told only that something failed tries it again."""


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
    outcome: AskResolution = "answered"
    """How the ask ended, as reported by the seam that ran it. See `AskResolution`,
    which explains the closed set and the guess that it replaced (K7)."""


DECLINED = (
    "The question reached somebody and they did not answer it. Do not ask again: "
    "choose the most reasonable option yourself and state the assumption you made, "
    "or stop and say what you would need in order to proceed."
)
"""What the model reads when a person was asked and declined (K7).

This text is distinct from `UNATTENDED` because the two cases call for different
behavior, and only one of them had a sentence before. "Nobody is attending"
invites the model to continue alone. That is right when no one is there. It is
wrong when somebody has just dismissed the prompt. That person declined to
direct the model, and the one thing that the model must not do is ask again.
"""

CANCELED = (
    "The question was not put to anybody, because this turn was canceled while it "
    "was being asked. Stop rather than continuing on an assumption."
)
"""What the model reads when the ask was abandoned before delivery.

This is the one ending that is not about the attention of the user. To tell the
model to continue would tell it to work through a cancellation. To tell it that
somebody declined would invent a person who was never asked. The old attendance
guess said exactly that here.
"""

FAILED = (
    "The question could not be delivered: the harness failed while asking it. "
    "Treat it as unanswered — choose the most reasonable option yourself and state "
    "the assumption you made, or say what you would need in order to proceed."
)
"""What the model reads when an answerer raised.

This case has its own name and is not folded into `UNATTENDED`. "There is nobody
there" and "something here is broken" are different facts, and only the second
is worth showing to a person in the transcript."""

_SENTENCES: dict[AskResolution, str] = {
    "unattended": UNATTENDED,
    "declined": DECLINED,
    "canceled": CANCELED,
    "failed": FAILED,
}
"""One sentence for each way of not getting an answer.

This is a dict and not a chain of conditionals, so that a new resolution is a
new row. Nothing here makes the dict *exhaustive*. A dict literal is not checked
against the `Literal`, and the `.get` below falls back and does not raise. So
`test_every_way_a_question_can_end_reads_as_itself` walks `get_args`, and that
test is what catches a resolution that arrives without a sentence."""


def _rendered(value: Any) -> str:  # noqa: ANN401
    """The text of the result. `ask` used to fold four endings into `None`, and
    this function tells them apart again."""
    resolution: AskResolution = value["outcome"]
    if resolution == "answered":
        return as_str(value["answer"])
    # `.get` is used for the reason that `ph.wire.literal_lookup` exists. A
    # `Literal` constrains writers and is lost at the payload boundary. A reader
    # therefore holds what is really a `str`, and must have an answer for a value
    # that it does not know. `denial_reason` makes the same fallback on the
    # approval side.
    return _SENTENCES.get(resolution, UNATTENDED)


@plugin("tool-ask-user", affects="environment", inject=[TOOLS, USER_QUESTIONS])
async def apply(ctx: Context, config: None) -> None:
    """Register the question tool."""

    async def ask_user(args: AskUserArgs, run: ToolRunContext) -> dict[str, Any]:
        questions = ctx.require(USER_QUESTIONS)
        outcome = await questions.ask(
            UserQuestion(
                question=args.question,
                options=args.options,
                header=args.header,
                multi_select=args.multi_select,
                # The call id is used so that `askId` in the log is the one
                # string that already joins `tool/call`, `tool/result` and the
                # two `question/*` records of this one exchange.
                ask_id=run.call_id,
            ),
            session=run.session,
            cancel=run.signal,
        )
        return {"answer": outcome.answer, "outcome": outcome.resolution}

    ctx.require(TOOLS).register(
        define_tool(
            "ask_user",
            DESCRIPTION,
            parameters=AskUserArgs,
            output=ToolOutput(
                schema=AskUserValue,
                render=lambda args, value: text_content(_rendered(value)),
            ),
            execute=ask_user,
            # The tool changes nothing and needs no permission to run. Its cost
            # is the attention of a person, which no gate can give back.
            is_irreversible=False,
            present_call=lambda args: ToolCallView(
                title=as_str(args.get("header"), "Question"),
                input=str(args.get("question", "")),
            ),
            present_result=lambda args, result: ToolResultView(
                title=as_str(args.get("header"), "Question"),
                subtitle=head(str(args.get("question", "")), 120),
                is_error=result.is_error,
            ),
        )
    )
