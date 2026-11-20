"""The intent kinds of ph-core: every pair that ph-core writes through `ctx.intents` (T4).

This leaf declares every kind. It also holds the pure pieces that each kind
needs: its key functions, its closer, and the payload builders. The closer and
the live settle of the seam both call those builders.

* `SHELL_COMMAND`: a shell command outside a tool call. It is the user's `!` or
  `!!`, or the gate of an autonomous goal. The pair is the command, then what
  it did.
* `COMMAND_RUN`: a slash command. The pair is its line, then how it ended.
* `UPLOAD`: a file that is about to leave this machine for a provider, then
  the fact that it did.
* `APPROVAL_ASK`: an approval, asked and then decided.
* `QUESTION_ASK`: a question put to a person, and its answer.
* `TOOL_DISPATCH`: a Code Mode sub-dispatch, started and then settled.
* `TOOL_EFFECT`: a call whose tool names its effect, opened and then settled.
* `CREDENTIAL_WAIT`: a session or a child held for a credential, then released.
* `WORKSPACE_RESTORE`: a workspace put back to a restore point, then whether
  it was.

**Why the kinds are in a leaf, and not in the seam that fills each pair.**
Repair settles only the kinds that are declared in the process that does the
resume. A kind is declared when its module is imported. When each kind lived
in its seam, repair had to import four seams itself, and only inside a
function. At module top that import was a cycle (`ph.orphans` →
`ph.persistence` → `repair` → `ph.seams.shell` → `ph.orphans`). Only a fresh
interpreter that imported in another order would hit the cycle. So whether a
resume settled a kind depended on what that process happened to import first.
This module imports nothing that a cycle could run through, and `ph.session`
imports it. So every process that holds a session log has the kinds of ph-core
declared. The import is static and at module top, so mypy sees it.

**What this module may import:** the standard library, `ph.session.intents`,
`ph.session.events`, `ph.session.writers` and `ph.json`. `test_intent_kinds.py`
tests that limit. It also tests that every `IntentKind(...)` in shipped code is
in a `kinds` leaf. A package brings its own leaf the same way: `ph_app.kinds`
holds `CLIENT_COMMAND`, and `ph_app` imports it.

**The seams write. This module only declares.** Each seam opens and settles its
pair through `ctx.intents`, with the payload built here. So the settle that
repair writes on resume and the settle that the live seam writes cannot drift
apart. The constants that the builders need are here too (`INTERRUPTED`,
`AskResolution`, `DISPATCH_INTERRUPTED`). The seams re-export `INTERRUPTED`
and `AskResolution`.

@module ph.session.kinds
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from ..json import JsonObject, JsonValue, as_bool, as_str
from .events import SessionEvent
from .intents import IntentKind, Unsettled, declare_intent, opened_seq, seq_field
from .writers import log_writer

__all__ = [
    "APPROVAL_ASK",
    "COMMAND_RUN",
    "CREDENTIAL_WAIT",
    "DISPATCH_INTERRUPTED",
    "DISPATCH_NOT_DONE",
    "DISPATCH_REF_KEYS",
    "INTERRUPTED",
    "KINDS",
    "QUESTION_ASK",
    "SESSION_HOLDER",
    "SHELL_COMMAND",
    "TOOL_DISPATCH",
    "TOOL_EFFECT",
    "UPLOAD",
    "WORKSPACE_RESTORE",
    "AskResolution",
    "approval_decided",
    "command_done",
    "credential_hold",
    "effect_settle",
    "hold_of",
    "question_answered",
    "restore_settled",
    "upload_settled",
]

_LOG = log_writer(__name__)
"""The writer of this leaf. Every kind below carries it, and the journal writes
the pairs of those kinds through it (T6)."""


def _ask_seq(event: SessionEvent) -> str | None:
    """The ask that a decision or an answer settles: its `askSeq`, as a key (T3)."""
    return seq_field(event, "askSeq")


def _key_field(event: SessionEvent) -> str:
    """The `key` in the payload of an intent. An effect and a hold are keyed by it."""
    return as_str(event.data.get("key"))


# -------------------------------------------------------------------- shell --


def _command_seq(event: SessionEvent) -> str | None:
    """The command that a `shell/result` settles: its `commandSeq`, as a key."""
    return seq_field(event, "commandSeq")


def _shell_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle for a command whose own result was never written.

    Two cases exist. It is `outcome-unknown` when the record was on disk and the
    harness then stopped. It is `not-started` when the record could not be
    written, so the child never spawned. This payload does not say which case
    it is. The journal or repair says that in the `unsettled` marker, and
    whichever of them writes this settle merges the marker in. `ok` is false in
    both cases, because nothing here saw the command succeed.
    """
    return {"commandSeq": opened.seq, "ok": False}


SHELL_COMMAND = declare_intent(
    IntentKind(
        opened="shell/command",
        settled="shell/result",
        # A command is keyed by its own seq: nothing else about it is unique.
        opened_key=opened_seq,
        settled_key=_command_seq,
        # The child may have run. This pair exists for a command that takes the
        # daemon down with it, and repair cannot know how far the command got.
        orphan="outcome-unknown",
        # On disk before the child starts (F9). A command whose record cannot be
        # written does not run.
        barrier="durable",
        closer=_shell_interrupted,
        writer=_LOG,
    )
)
"""A shell command run outside a tool call: the command, then what it did (P10-08).
It is the user's `!` or `!!`, or the gate of an autonomous goal (S16, `gate` on
the command record). Both go through `ShellService.run_recorded`. The kind is
declared here because repair must settle the pair on any resume that has ph-core."""


# ------------------------------------------------------------------ command --


def _run_seq(event: SessionEvent) -> str | None:
    """The command that a `command/done` settles: its `runSeq`, as a key."""
    return seq_field(event, "runSeq")


def command_done(
    *, name: str, run_seq: int, outcome: str, detail: str | None = None
) -> dict[str, Any]:
    """A `command/done` payload. The live ending and repair both build it here.

    `outcome` is `ok`, `error` or `canceled`. `runSeq` names the `command/run`
    that this ends (S16). It is the one key that two runs of one command never
    share.
    """
    data: dict[str, Any] = {"name": name, "outcome": outcome, "runSeq": run_seq}
    if detail is not None:
        data["detail"] = detail
    return data


def _command_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The ending of a command whose own ending was never written. It is an `error`,
    and its detail says which half is known. The `unsettled` marker says the same
    again, for a reader that folds."""
    detail = (
        "the command's record could not be written, so it did not run"
        if why == "not-started"
        else "the harness stopped while this command ran; whether it finished is unknown"
    )
    return command_done(
        name=as_str(opened.data.get("name")), run_seq=opened.seq, outcome="error", detail=detail
    )


COMMAND_RUN = declare_intent(
    IntentKind(
        opened="command/run",
        settled="command/done",
        opened_key=opened_seq,
        settled_key=_run_seq,
        # The body may have done anything that a command does, and `/revert`
        # rewrites the tree. Repair cannot know how far the body got.
        orphan="outcome-unknown",
        # On disk before the body runs (S16). Earlier, the effects of a command
        # came before any durable record of it. A command whose record cannot be
        # written does not run.
        barrier="durable",
        closer=_command_interrupted,
        writer=_LOG,
    )
)
"""A slash command: its line, then how it ended. The command or repair writes the ending."""


# ----------------------------------------------------------------- approval --

INTERRUPTED: Literal["interrupted"] = "interrupted"
"""The process died while a person was being asked (P5-13).

**Only repair writes this value. An answerer never does.** That separates it
from the four answers beside it in `ph.seams.approval.ApprovalOutcome`. Those
four say what happened when the question was *put*: somebody allowed it,
somebody refused, the work was canceled, or nobody could be asked. This one
says that the question was never resolved at all, because the process that
held it stopped existing. Repair records it on resume, so that the log holds no
open ask that no one can answer.

It is not `canceled`, which claims that somebody stopped the work. It is not
`unavailable`, which is the live answer when no front end takes the prompt, and
which is a *denial* that a turn continues from. The separate name has a purpose:
a person who reads a transcript can tell "I was asked and the daemon died" from
"I was asked and said no".
"""


def approval_decided(
    *,
    tool_name: str,
    call_id: str | None,
    outcome: str,
    ask_seq: int,
    automatic: bool,
    answer: JsonObject | None = None,
) -> dict[str, Any]:
    """An `approval/decided` payload. The live decision and repair both build it here.

    `outcome` is the one word of the answer (`answer_kind`). `answer` is what an
    answer that carries data adds: the substituted arguments of an edit, the
    message of a response. `askSeq` names the ask that this decides (T3). It is
    the one key that two asks of one tool never share.
    """
    data: dict[str, Any] = {"toolName": tool_name, "outcome": outcome, "askSeq": ask_seq}
    if answer:
        data.update(answer)
    if call_id is not None:
        data["callId"] = call_id
    if automatic:
        data["automatic"] = True
    return data


def _approval_closed(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The decision that nobody made, as it has always been written.

    `not-started` means that the ask never reached disk, so nobody was asked. It
    reads as the live `unavailable`, since no one could be asked.
    `outcome-unknown` is `INTERRUPTED`. It is recorded as automatic, because no
    person made it.
    """
    call_id = opened.data.get("callId")
    return approval_decided(
        tool_name=as_str(opened.data.get("toolName")),
        call_id=call_id if isinstance(call_id, str) else None,
        outcome="unavailable" if why == "not-started" else INTERRUPTED,
        ask_seq=opened.seq,
        automatic=why != "not-started",
    )


APPROVAL_ASK = declare_intent(
    IntentKind(
        opened="approval/asked",
        settled="approval/decided",
        # The ask is keyed by its own seq, and its decision by `askSeq` (T3).
        # Earlier the key was the call id, or the tool name when there was no
        # call id. Two asks of one tool then shared a key, and the first was
        # never settled. The Continual Harness asks `tool_name="refine"` with no
        # call id. `callId` and `toolName` stay in the payloads.
        opened_key=opened_seq,
        settled_key=_ask_seq,
        # A person may have decided on a screen whose answer never reached the
        # log. The outcome of the question is the part that is unknown.
        orphan="outcome-unknown",
        # On disk before anybody is asked (F8).
        barrier="durable",
        closer=_approval_closed,
        writer=_LOG,
    )
)
"""An approval: asked, then decided by a person, a policy, or repair (P10-09)."""


# ----------------------------------------------------------------- question --

AskResolution: TypeAlias = Literal["answered", "unattended", "declined", "canceled", "failed"]
"""Every way that one question can end. The set is closed, because a caller
renders each one.

`ask` used to fold all four failures into `None`. The caller then guessed which
failure it had been by sampling `attended`. That is a *live* probe, read at a
different moment from the moment at which the seam (`ph.seams.user_questions`)
checked. A canceled ask still reads as attended. So the guess said "somebody
was asked and declined" about a question that was put to nobody (K7, and the
same class of false story that K7 set out to remove)."""


def question_answered(
    *, ask_id: str | None, ask_seq: int, resolution: AskResolution, answer: str | None
) -> dict[str, Any]:
    """A `question/answered` payload that says *how* the question closed.

    `declined` is written beside `resolution` for every non-answer. Logs that
    were written before `resolution` existed carry only `declined`, so a reader
    still folds on it. Writing both keeps one reader and not two. `askSeq` names
    the ask that this answers (T3), so a question that is posed again under its
    old id is an intent of its own.
    """
    data: dict[str, Any] = {"askId": ask_id, "askSeq": ask_seq, "resolution": resolution}
    if resolution == "answered":
        data["answer"] = answer
    else:
        # Asked and *not* answered. This differs from never being asked, which
        # appends nothing at all. It is recorded so that the fold stops calling
        # the question pending.
        data["declined"] = True
    return data


def _question_closed(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The answer that nobody gave, as it has always been written.

    `not-started` is the barrier failing: the question could not be written, so
    it was not delivered. It closes `failed`, as a live ask does when the
    machinery does not reach a person. `outcome-unknown` is the case of repair:
    the process died while somebody may have been answering. So it deliberately
    writes **no** `declined`. Declined means "somebody was there and declined".
    To claim that of a person who was never reached is the false statement that
    the module docstring of `ph.seams.user_questions` refuses to make. The
    `unsettled` marker says that the ask was interrupted, as it does for every
    kind (T2).
    """
    ask_id = opened.data.get("askId")
    if why == "not-started":
        return question_answered(
            ask_id=ask_id if isinstance(ask_id, str) else None,
            ask_seq=opened.seq,
            resolution="failed",
            answer=None,
        )
    return {"askId": as_str(ask_id), "askSeq": opened.seq}


QUESTION_ASK = declare_intent(
    IntentKind(
        opened="question/asked",
        settled="question/answered",
        opened_key=opened_seq,
        settled_key=_ask_seq,
        orphan="outcome-unknown",
        # On disk before it is delivered (F8).
        barrier="durable",
        closer=_question_closed,
        writer=_LOG,
    )
)
"""A question put to a person, then answered, or settled by repair (P10-09)."""


# ----------------------------------------------------------------- dispatch --

DISPATCH_INTERRUPTED = (
    "The harness stopped while this call was running, so its result was never "
    "recorded. Its outcome is unknown."
)
"""The body of a dispatch that repair settled. Its card shows this text (P10-11)."""

DISPATCH_NOT_DONE = (
    "The harness stopped while this call was running, and the tool has since "
    "checked: it did not happen."
)
"""The body of a dispatch settled `not-started`. Only the check of its tool
writes one. The barrier of a dispatch belongs to the checkpoint policy, so the
journal never settles a dispatch as not-started (L6b)."""


DISPATCH_REF_KEYS: tuple[str, ...] = ("rootCallId", "parentCallId", "subCallId", "name")
"""The wire keys of `ph.tools.code_mode.CodeDispatchRef`. They are the identity
that the two records of a dispatch share, and the only fields that its readers
pair the records by.

The keys are spelled here because this leaf cannot import the model.
`test_code_mode.py` holds the two equal. So a field that is renamed on the model
fails a test, and does not leave the settle that repair writes unpaired."""


def _dispatch_identity(data: JsonObject) -> dict[str, JsonValue]:
    """The identity of a dispatch, read from one of its records."""
    return {key: data.get(key) for key in DISPATCH_REF_KEYS}


def _sub_call_id(event: SessionEvent) -> str:
    return as_str(event.data.get("subCallId"))


def _dispatch_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle for a dispatch whose own settle was never written.

    It has three parts: the identity of the start record, so that every reader
    pairs it by the fields that `CodeDispatchRef` names, an error, and a text
    body that says why. A dispatch card draws that body, so a crash no longer
    leaves a card running forever. The `unsettled` marker of repair says which
    half is known. The `tool/result` of the parent call belongs to the turn
    repair, as `TOOL_OUTCOME_UNKNOWN`.
    """
    text = DISPATCH_NOT_DONE if why == "not-started" else DISPATCH_INTERRUPTED
    return {
        **_dispatch_identity(opened.data),
        "isError": True,
        "content": [{"type": "text", "text": text}],
    }


def _dispatch_done(opened: SessionEvent, content: tuple[JsonValue, ...]) -> JsonObject:
    """The settle for a dispatch that its tool said, on resume, did happen (L6b). It
    carries the content that the tool rendered, as the own settle of the dispatch
    would have carried it."""
    return {**_dispatch_identity(opened.data), "isError": False, "content": list(content)}


def _dispatch_within(opened: SessionEvent) -> tuple[str, str]:
    """The top-level call that a dispatch ran inside, and the name of the dispatched tool."""
    return as_str(opened.data.get("rootCallId")), as_str(opened.data.get("name"))


TOOL_DISPATCH = declare_intent(
    IntentKind(
        opened="tool/code-dispatch-start",
        settled="tool/code-dispatch",
        opened_key=_sub_call_id,
        settled_key=_sub_call_id,
        # Recorded after the pipeline decided and before the binding ran. The
        # call may have happened.
        orphan="outcome-unknown",
        # The barrier of the checkpoint policy, before the body of the tool. The
        # policy skips the barrier where a restore covers the tool.
        barrier="tools-body",
        closer=_dispatch_interrupted,
        # A dispatched tool that can check its own effect is asked on resume, as
        # the tool of a top-level call is. The cell that it ran in says what it
        # found (L6b).
        reconciled=_dispatch_done,
        within=_dispatch_within,
        writer=_LOG,
    )
)
"""A Code Mode sub-dispatch: started, then settled by the pipeline or by repair."""


# ------------------------------------------------------------------- effect --


def effect_settle(
    key: str, call_id: str, *, is_error: bool, content: Sequence[JsonValue]
) -> JsonObject:
    """A `tool/effect-settled` payload, in wire form: whether the call failed, and the
    result that a repeat is handed. `outcome_of` says what the payload means."""
    return {"key": key, "callId": call_id, "isError": is_error, "content": list(content)}


def _effect_unknown(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle of a keyed call that nobody saw finish. Its effect may have
    happened. The `unsettled` marker says so, and whoever writes this settle
    merges the marker in."""
    return effect_settle(
        _key_field(opened), as_str(opened.data.get("callId")), is_error=True, content=()
    )


def _effect_failed(settled: SessionEvent) -> bool:
    """The call ran and reported an error. Only its tool knows the state of its far side."""
    return as_bool(settled.data.get("isError"))


TOOL_EFFECT = declare_intent(
    IntentKind(
        opened="tool/effect",
        settled="tool/effect-settled",
        opened_key=_key_field,
        settled_key=_key_field,
        orphan="outcome-unknown",
        # Opened ahead of the `tools/execute` waterfall. So the barrier of the
        # checkpoint policy on `tools/body` carries it to disk before the body
        # runs.
        barrier="tools-body",
        closer=_effect_unknown,
        writer=_LOG,
        failed=_effect_failed,
        # A failure, or an attempt that never started, is worth another attempt.
        # A done one is answered from the log. `reconcile` decides an unknown one.
        reopen=frozenset({"failed", "not-started"}),
    )
)
"""A call whose tool names its effect (`ToolDefinition.idempotency_key`, P10-12).
It is keyed by tool name and effect key, for the life of the log of the session."""


# ------------------------------------------------------------------- upload --

_UPLOAD_FIELDS: tuple[str, ...] = ("provider", "attachmentId", "mime", "name", "bytes")
"""What the two records of an upload share: which file, and to whom. The opening
record carries these fields so that its settle can say the same. That settle is
the live one, or the one that repair writes."""


def _upload_seq(event: SessionEvent) -> str | None:
    """The upload that an `attachment/uploaded` settles: its `uploadSeq`, as a key."""
    return seq_field(event, "uploadSeq")


def upload_settled(opened: SessionEvent, *, expires_at: int | None = None) -> dict[str, Any]:
    """An `attachment/uploaded` payload. The live settle and repair both build it here.

    It holds the file and the provider, copied from the opening record. It also
    holds the time at which the provider said that its copy stops working, if
    the provider said one.
    """
    data: dict[str, Any] = {key: opened.data.get(key) for key in _UPLOAD_FIELDS}
    data["uploadSeq"] = opened.seq
    if expires_at is not None:
        data["expiresAt"] = expires_at
    return data


def _upload_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle of an upload that nobody saw finish: the file and the provider.
    The `unsettled` marker says that the file *may* have left, and whoever writes
    this settle merges the marker in."""
    return upload_settled(opened)


UPLOAD = declare_intent(
    IntentKind(
        opened="attachment/uploading",
        settled="attachment/uploaded",
        opened_key=opened_seq,
        settled_key=_upload_seq,
        # The bytes may have reached the provider. A person who audits where
        # their data went needs a crash in the middle of an upload reported as
        # "maybe".
        orphan="outcome-unknown",
        # On disk before the bytes leave (S9). Earlier the record of an upload
        # came after the upload. A crash between the two left the provider with
        # a file that the log never said it was given. An upload that the log
        # cannot record does not happen.
        barrier="durable",
        closer=_upload_interrupted,
        writer=_LOG,
    )
)
"""A file that is about to leave this machine for the file API of a named
provider, then the fact that it did, or the "it may have" of repair (P7-03, S9)."""


# --------------------------------------------------------------- credential --

SESSION_HOLDER = "session"
"""The holder that a session waits as when its own route lacks the credential,
and not the route of one of its children."""


def credential_hold(holder: str, name: str) -> JsonObject:
    """A `credential/needed` payload, and its `credential/supplied`. Both have the
    same fields.

    `holder` is who waits: the run id of a child, or `SESSION_HOLDER` for the
    session that owns this log. `name` is the credential, **by name only** (I-3).
    The value is nowhere in the log, and that makes the record safe to keep.
    """
    return {"key": f"{holder}:{name}", "holder": holder, "name": name}


def hold_of(event: SessionEvent) -> tuple[str, str]:
    """The holder and the name of a hold record: `credential_hold`, read back."""
    return as_str(event.data.get("holder")), as_str(event.data.get("name"))


CREDENTIAL_WAIT = declare_intent(
    IntentKind(
        opened="credential/needed",
        settled="credential/supplied",
        opened_key=_key_field,
        settled_key=_key_field,
        # **The owner looks** (T5). Repair cannot know whether the name is here
        # now. The resume check can, and it asks again on every open. It settles
        # a hold whose name has arrived and starts the work of that hold. It
        # leaves a hold that still waits as it is. So a second restart before
        # the key arrives appends nothing.
        orphan="owner-settles",
        # Buffered with the records around it. Nothing acts on a hold, so nothing
        # needs it on disk first.
        barrier="buffered",
        writer=_LOG,
        # A hold that was released and is needed again (the key taken away, the
        # daemon restarted) is a new wait, and not the old wait answered.
        reopen=frozenset({"done"}),
    )
)
"""A session, or one of its children, held because its route names a credential
that this deployment cannot supply (T5). It is released when the name arrives."""


# ------------------------------------------------------------------ restore --


def _restoring_seq(event: SessionEvent) -> str | None:
    """The restore that a `workspace/restored` settles: its `restoringSeq`, as a key."""
    return seq_field(event, "restoringSeq")


def restore_settled(opened: SessionEvent, *, ok: bool, detail: str | None = None) -> dict[str, Any]:
    """A `workspace/restored` payload. The live settle and repair both build it here.
    It says whose workspace, which restore point, and whether the tree is now at
    that point."""
    data: dict[str, Any] = {
        "restoringSeq": opened.seq,
        "agentId": opened.data.get("agentId"),
        "tree": opened.data.get("tree"),
        "ok": ok,
    }
    if detail is not None:
        data["detail"] = detail
    return data


def _restore_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle of a restore that nobody saw finish. It is not `ok`. The
    `unsettled` marker says that the tree may be part-way between the two states,
    and whoever writes this settle merges the marker in."""
    return restore_settled(opened, ok=False)


WORKSPACE_RESTORE = declare_intent(
    IntentKind(
        opened="workspace/restoring",
        settled="workspace/restored",
        opened_key=opened_seq,
        settled_key=_restoring_seq,
        # A restore rewrites and deletes files. A restore that is cut short
        # leaves a tree that is neither the restore point nor what was there. A
        # reader must be told "maybe", and not the rollback that the caller set
        # out to make.
        orphan="outcome-unknown",
        # On disk before the tree is touched (S12). A restore that the log cannot
        # record does not run.
        barrier="durable",
        closer=_restore_interrupted,
        writer=_LOG,
    )
)
"""A workspace put back to a restore point, then whether it was (S12). The pair
is opened inside `WorkspaceSeam.restore`, so every caller records it one way:
`/revert`, or a retry of the supervisor. It is keyed by the seq of the opening
record."""


KINDS: tuple[IntentKind, ...] = (
    SHELL_COMMAND,
    COMMAND_RUN,
    APPROVAL_ASK,
    QUESTION_ASK,
    TOOL_DISPATCH,
    TOOL_EFFECT,
    UPLOAD,
    CREDENTIAL_WAIT,
    WORKSPACE_RESTORE,
)
"""The kinds of ph-core, as this leaf declares them. `isolated_intent_kinds` keeps these."""
