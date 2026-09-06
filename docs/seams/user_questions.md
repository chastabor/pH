# `ctx.user_questions` — asking the human something that is not an approval

**Module:** `ph/seams/user_questions.py` · **Row:** `user-questions` ·
**Consumers:** `tool-ask-user`, the TUI modal, the daemon's ask desk

## Why this is not `ctx.approval`

The shapes differ, and sharing one seam would force one behaviour onto the other:

| | approval | question |
|---|---|---|
| about | a **specific pending call** | anything |
| shape | one-shot yes/no | free-form, or a choice |
| nobody there | **denies** — fails closed | answers `None` — the caller decides |

An approval that answered "no opinion" would be a permission system that fails
open. A question that *denied* would be a harness refusing to continue because
nobody was watching.

## A question is logged only when it is actually put to a person

This is the one rule here that is not obvious, and it follows from the failure
mode rather than from tidiness.

"Nobody could answer" resolves **instantly** to `None`. Appending around that
would write a question-and-refusal pair into the log of every unattended run — an
`/autonomous` turn inside an interactive profile, a `ph -p` against a profile that
armed the row — for an exchange that never happened. The log would then say a
person was asked and declined, which is a different and **false** claim.

So attendance is decided **first**:

* unattended → append nothing, return at once;
* deliverable → append `question/asked` *before* the waterfall, `question/answered`
  after (§5 rule 2, the same order `ApprovalService` uses).

A crash between them leaves the question in the log with no answer, which is
exactly the pending state `pending_questions` folds — the same log-as-state
design [`ctx.approval`](approval.md) uses.

## The surface

```text
await ctx.user_questions.ask(question, session=...)   # -> str | None
ctx.user_questions.register_answerer(answerer, reachable=...)
ctx.user_questions.attended()                          # is anyone there?
pending_questions(session)                             # the fold
```

A `UserQuestion` carries `question`, optional `options`, a `header`,
`multi_select`, and an `ask_id`.

`ask_id` is worth passing: `tool-ask-user` passes the tool's `call_id`, which
makes one string join `tool/call`, `tool/result` and both `question/*` records of
the same exchange.

## Nobody attending is a *result*, not an error

`ask` answers `None`, and the calling tool turns that into a sentence the model
can act on:

> Nobody is attending this session, so the question was not put to anyone.
> Continue without an answer: choose the most reasonable option yourself and
> state the assumption you made, or say what you would need in order to proceed.

Phrased as an instruction rather than an error **because it arrives as a
successful result**: "no answer" is this seam's defined outcome, and a model told
only that something failed will retry it.

## Answering

`register_answerer(answerer, reachable=...)` — `reachable` is what makes
`attended()` answerable *before* a question is asked, which is what the
append-nothing rule above depends on.

`tool-ask-user` ships **disabled** in `ph-base` and is armed by `tui.yaml`. An
unattended posture therefore never sees the tool at all: no schema in the prompt,
no turn spent calling it, nothing in the log. Paying nothing for a capability the
deployment cannot perform is the same rule `subagent-task` follows.

## What it does not do

* It does not gate anything. Nothing waits on a question except the tool that
  asked it — use [`ctx.approval`](approval.md) when the answer decides whether an
  action proceeds.
* It does not queue or retry. One ask, one routing.
* It is not asked permission to ask: gating a question behind an approval is one
  more interruption for the same person, and the ask has to happen anyway for the
  model to be directed.

## See also

[`ctx.approval`](approval.md) · [`ctx.commands`](commands.md) ·
`test_ask_user.py`, `test_daemon_asks.py`
