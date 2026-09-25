"""ph-core's intent kinds: every pair ph-core writes through `ctx.intents`, in one leaf (T4).

Five kinds, and the pure pieces each one needs — its key functions, its closer, and the
payload builders the closer and the seam's live settle both call:

* `SHELL_COMMAND` — a person's `!` or `!!`: the command, then what it did;
* `APPROVAL_ASK` — an approval, asked and then decided;
* `QUESTION_ASK` — a question put to a person, and its answer;
* `TOOL_DISPATCH` — a Code Mode sub-dispatch, started and then settled;
* `TOOL_EFFECT` — a call whose tool names its effect, opened and then settled;
* `CREDENTIAL_WAIT` — a session or a child held for a credential, then released.

**Why a leaf, and not the seam that fills each pair.** Repair settles only the kinds
that are declared in the process doing the resume, and a kind is declared when its
module is imported. When each kind lived in its seam, repair had to import four seams
itself, and only inside a function: at module top that import was a cycle
(`ph.orphans` → `ph.persistence` → `repair` → `ph.seams.shell` → `ph.orphans`), which
only a fresh interpreter importing in another order would hit. So whether a resume
settled a kind depended on what that process happened to import first. This module
imports nothing a cycle could run through, and `ph.session` imports it, so every
process that holds a session log has ph-core's kinds declared — by a static,
module-top import that mypy sees.

**What it may import:** the standard library, `ph.session.intents`, `ph.session.events`,
`ph.session.writers` and `ph.json`. `test_intent_kinds.py` holds that line, and holds
every `IntentKind(...)` in shipped code to a `kinds` leaf. A package brings its own
leaf the same way: `ph_app.kinds` holds `CLIENT_COMMAND`, and `ph_app` imports it.

**The seams write; this module only declares.** Each seam opens and settles its pair
through `ctx.intents`, with the payload built here, so the settle repair writes on
resume and the one the live seam writes cannot drift apart. Constants the builders
need live here too (`INTERRUPTED`, `AskResolution`, `DISPATCH_INTERRUPTED`), and the
seams re-export them.

@module ph.session.kinds
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, TypeAlias

from ..json import JsonObject, JsonValue, as_bool, as_str
from .events import SessionEvent
from .intents import IntentKind, Unsettled, declare_intent
from .writers import log_writer

__all__ = [
    "APPROVAL_ASK",
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
    "AskResolution",
    "approval_decided",
    "credential_hold",
    "effect_settle",
    "hold_of",
    "question_answered",
]

_LOG = log_writer(__name__)
"""This leaf's writer: every kind below carries it, and the journal writes their pairs
through it (T6)."""


def _seq_key(event: SessionEvent) -> str:
    """An intent keyed by the seq of the record that opened it: unique by construction."""
    return str(event.seq)


def _ask_seq(event: SessionEvent) -> str | None:
    """The ask a decision or an answer settles — its `askSeq`, as a key (T3)."""
    return _seq_field(event, "askSeq")


def _key_field(event: SessionEvent) -> str:
    """An intent keyed by the `key` its payload carries: an effect's, a hold's."""
    return as_str(event.data.get("key"))


def _seq_field(event: SessionEvent, name: str) -> str | None:
    seq = event.data.get(name)
    return str(seq) if isinstance(seq, int) and not isinstance(seq, bool) else None


# -------------------------------------------------------------------- shell --


def _command_seq(event: SessionEvent) -> str | None:
    """The command a `shell/result` settles — its `commandSeq`, as a key."""
    return _seq_field(event, "commandSeq")


def _shell_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle for a command whose own result was never written.

    Which half is known — `outcome-unknown` when the record was on disk and the
    harness then stopped, `not-started` when it could not be written so the child
    never spawned — is the journal's and repair's `unsettled` marker, merged in by
    whoever writes this. `ok` is false either way: nothing here saw it succeed.
    """
    return {"commandSeq": opened.seq, "ok": False}


SHELL_COMMAND = declare_intent(
    IntentKind(
        opened="shell/command",
        settled="shell/result",
        # A command is keyed by its own seq: nothing else about it is unique.
        opened_key=_seq_key,
        settled_key=_command_seq,
        # The child may have run: a command that takes the daemon down with it is
        # the case this pair exists for, and repair cannot know how far it got.
        orphan="outcome-unknown",
        # On disk before the child starts (F9): a command whose record cannot be
        # written does not run.
        barrier="durable",
        closer=_shell_interrupted,
        writer=_LOG,
    )
)
"""A person's `!` or `!!`: the command, then what it did (P10-08). Filled by
`ph_app.shell`; declared here because repair must settle the pair on any resume
that has ph-core, and ph-core cannot import the app."""


# ----------------------------------------------------------------- approval --

INTERRUPTED: Literal["interrupted"] = "interrupted"
"""The process died while a person was being asked (P5-13).

**Written only by repair, never by an answerer**, which is what separates it from
the four answers beside it in `ph.seams.approval.ApprovalOutcome`. Those four say
what happened when the question was *put*: somebody allowed it, somebody refused,
the work was canceled, or nobody could be asked. This one says the question was
never resolved at all, because the process holding it stopped existing — and it is
recorded on resume so that the log holds no open ask no one can answer.

Not `canceled`, which claims somebody stopped the work; not `unavailable`, which is
the live answer when no front end takes the prompt and is a *denial* a turn
continues from. Naming it apart is the point: a person reading a transcript can
tell "I was asked and the daemon died" from "I was asked and said no".
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
    """An `approval/decided` payload — the live decision's and repair's alike.

    `outcome` is the answer's one word (`answer_kind`); `answer` is what an answer
    that carries data adds — an edit's substituted arguments, a response's message.
    `askSeq` names the ask this decides (T3): the one key two asks of one tool never
    share.
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
    """The decision nobody made, as it has always been written.

    `not-started` — the ask never reached disk, so nobody was asked — reads as the
    live `unavailable`, since no one could be; `outcome-unknown` is `INTERRUPTED`,
    recorded automatically because no person made it.
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
        # Keyed by the ask's own seq, and its decision by `askSeq` (T3). By call id,
        # or tool name when there was none, two asks of one tool — the Continual
        # Harness asks `tool_name="refine"` with no call id — shared a key, and the
        # first was never settled. `callId` and `toolName` stay in the payloads.
        opened_key=_seq_key,
        settled_key=_ask_seq,
        # A person may have decided on a screen whose answer never reached the
        # log: the question's outcome is what is unknown.
        orphan="outcome-unknown",
        # On disk before anybody is asked (F8).
        barrier="durable",
        closer=_approval_closed,
        writer=_LOG,
    )
)
"""An approval: asked, then decided — by a person, a policy, or repair (P10-09)."""


# ----------------------------------------------------------------- question --

AskResolution: TypeAlias = Literal["answered", "unattended", "declined", "canceled", "failed"]
"""Every way one question can end. Closed, because a caller renders each.

`ask` used to fold all four failures into `None`, and the caller then guessed
which it had been by sampling `attended` — a *live* probe, read at a different
moment from the one this seam checked. A canceled ask still reads as attended,
so the guess said "somebody was asked and declined" about a question nobody was
put (K7, and the same false-story class K7 set out to remove)."""


def question_answered(
    *, ask_id: str | None, ask_seq: int, resolution: AskResolution, answer: str | None
) -> dict[str, Any]:
    """A `question/answered` payload that says *how* the question closed.

    `declined` is written beside `resolution` for every non-answer, because logs
    written before `resolution` existed carry only that — so a reader still folds on
    it, and writing both keeps one reader rather than two. `askSeq` names the ask
    this answers (T3), so a question re-posed under its old id is its own intent.
    """
    data: dict[str, Any] = {"askId": ask_id, "askSeq": ask_seq, "resolution": resolution}
    if resolution == "answered":
        data["answer"] = answer
    else:
        # Asked and *not* answered. Distinct from never being asked, which appends
        # nothing at all, and recorded so the fold stops calling it pending.
        data["declined"] = True
    return data


def _question_closed(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The answer nobody gave, as it has always been written.

    `not-started` is the barrier failing — the question could not be written, so it
    was not delivered, and closes `failed` as a live ask does when the machinery does
    not reach a person. `outcome-unknown` is repair's: the process died while
    somebody may have been answering, so it pointedly writes **no** `declined` —
    declined means "somebody was there and declined", and claiming that of a person
    who was never reached is the false statement `ph.seams.user_questions` opens by
    refusing to make. That it was interrupted is the `unsettled` marker's to say, as
    for every kind (T2).
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
        opened_key=_seq_key,
        settled_key=_ask_seq,
        orphan="outcome-unknown",
        # On disk before it is delivered (F8).
        barrier="durable",
        closer=_question_closed,
        writer=_LOG,
    )
)
"""A question put to a person, then answered — or settled by repair (P10-09)."""


# ----------------------------------------------------------------- dispatch --

DISPATCH_INTERRUPTED = (
    "The harness stopped while this call was running, so its result was never "
    "recorded. Its outcome is unknown."
)
"""The body of a dispatch settled by repair — what its card shows (P10-11)."""

DISPATCH_NOT_DONE = (
    "The harness stopped while this call was running, and the tool has since "
    "checked: it did not happen."
)
"""The body of a dispatch settled `not-started`. Only its tool's check writes one: a
dispatch's barrier is the checkpoint policy's, so the journal never settles one
not-started (L6b)."""


DISPATCH_REF_KEYS: tuple[str, ...] = ("rootCallId", "parentCallId", "subCallId", "name")
"""The wire keys of `ph.tools.code_mode.CodeDispatchRef` — the identity a dispatch's
two records share, and the only fields its readers pair them by.

Spelled here because this leaf cannot import the model; `test_code_mode.py` holds
the two equal, so a field renamed on the model fails a test rather than leaving the
settle repair writes unpaired."""


def _dispatch_identity(data: JsonObject) -> dict[str, JsonValue]:
    """A dispatch's identity, read off one of its records."""
    return {key: data.get(key) for key in DISPATCH_REF_KEYS}


def _sub_call_id(event: SessionEvent) -> str:
    return as_str(event.data.get("subCallId"))


def _dispatch_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The settle for a dispatch whose own was never written.

    The start record's identity, so every reader pairs it by the fields
    `CodeDispatchRef` names, an error, and a text body saying why — which is what a
    dispatch card draws, so a crash no longer leaves one running forever. Which half
    is known rides on repair's `unsettled` marker. The parent call's own
    `tool/result` is the turn repair's, `TOOL_OUTCOME_UNKNOWN`.
    """
    text = DISPATCH_NOT_DONE if why == "not-started" else DISPATCH_INTERRUPTED
    return {
        **_dispatch_identity(opened.data),
        "isError": True,
        "content": [{"type": "text", "text": text}],
    }


def _dispatch_done(opened: SessionEvent, content: tuple[JsonValue, ...]) -> JsonObject:
    """The settle for a dispatch its tool said, on resume, happened (L6b): the content
    the tool rendered, as the dispatch's own settle would have carried."""
    return {**_dispatch_identity(opened.data), "isError": False, "content": list(content)}


def _dispatch_within(opened: SessionEvent) -> tuple[str, str]:
    """The top-level call a dispatch ran inside, and its tool's name."""
    return as_str(opened.data.get("rootCallId")), as_str(opened.data.get("name"))


TOOL_DISPATCH = declare_intent(
    IntentKind(
        opened="tool/code-dispatch-start",
        settled="tool/code-dispatch",
        opened_key=_sub_call_id,
        settled_key=_sub_call_id,
        # Recorded once the pipeline decided and before the binding ran: the call
        # may have happened.
        orphan="outcome-unknown",
        # The checkpoint policy's barrier before tools execute, which it places
        # after every pre-execute gate and skips where a restore covers the tool.
        barrier="tools-execute",
        closer=_dispatch_interrupted,
        # A dispatched tool that can check its own effect is asked on resume, as a
        # top-level call's is, and the cell it ran in says what it found (L6b).
        reconciled=_dispatch_done,
        within=_dispatch_within,
        writer=_LOG,
    )
)
"""A Code Mode sub-dispatch: started, then settled — by the pipeline, or repair."""


# ------------------------------------------------------------------- effect --


def effect_settle(
    key: str, call_id: str, *, is_error: bool, content: Sequence[JsonValue]
) -> JsonObject:
    """A `tool/effect-settled` payload: whether it failed, and the result a repeat is
    handed, in wire form. What it means is `outcome_of`'s to say."""
    return {"key": key, "callId": call_id, "isError": is_error, "content": list(content)}


def _effect_unknown(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """A keyed call nobody saw finish: its effect may have happened — which the
    `unsettled` marker whoever writes this merges in says."""
    return effect_settle(
        _key_field(opened), as_str(opened.data.get("callId")), is_error=True, content=()
    )


def _effect_failed(settled: SessionEvent) -> bool:
    """The call ran and reported an error: its tool is the one that knows its far side."""
    return as_bool(settled.data.get("isError"))


TOOL_EFFECT = declare_intent(
    IntentKind(
        opened="tool/effect",
        settled="tool/effect-settled",
        opened_key=_key_field,
        settled_key=_key_field,
        orphan="outcome-unknown",
        # Opened ahead of the `tools/execute` waterfall, so the checkpoint
        # policy's barrier — after every pre-execute gate — carries it to disk
        # before the body runs.
        barrier="tools-execute",
        closer=_effect_unknown,
        writer=_LOG,
        failed=_effect_failed,
        # A failure, or an attempt that never started, is worth another; a done
        # one is answered from the log; an unknown one is `reconcile`'s to decide.
        reopen=frozenset({"failed", "not-started"}),
    )
)
"""A call whose tool names its effect (`ToolDefinition.idempotency_key`, P10-12).
Keyed by tool name and effect key, for the life of the session's log."""


# --------------------------------------------------------------- credential --

SESSION_HOLDER = "session"
"""The holder a session waits as when it is its own route that lacks the credential,
rather than one of its children's."""


def credential_hold(holder: str, name: str) -> JsonObject:
    """A `credential/needed` payload, and its `credential/supplied` — the same fields.

    `holder` is who waits: a child's run id, or `SESSION_HOLDER` for the session whose
    log this is. `name` is the credential, **by name only** (I-3): the value is
    nowhere in the log, which is what makes the record safe to keep.
    """
    return {"key": f"{holder}:{name}", "holder": holder, "name": name}


def hold_of(event: SessionEvent) -> tuple[str, str]:
    """A hold record's holder and name — `credential_hold`, read back."""
    return as_str(event.data.get("holder")), as_str(event.data.get("name"))


CREDENTIAL_WAIT = declare_intent(
    IntentKind(
        opened="credential/needed",
        settled="credential/supplied",
        opened_key=_key_field,
        settled_key=_key_field,
        # **The owner looks** (T5). Repair cannot know whether the name is here now;
        # the resume check can, and asks again on every open: a hold whose name has
        # arrived is settled and its work started, and one still waiting is left as
        # it is — so a second restart before the key arrives appends nothing.
        orphan="owner-settles",
        # With the records around it: nothing acts on a hold, so nothing needs it on
        # disk first.
        barrier="buffered",
        writer=_LOG,
        # A hold released and needed again — the key taken away, the daemon
        # restarted — is a new wait, not the old one answered.
        reopen=frozenset({"done"}),
    )
)
"""A session, or one of its children, held because its route names a credential this
deployment cannot supply (T5) — and released when the name arrives."""


KINDS: tuple[IntentKind, ...] = (
    SHELL_COMMAND,
    APPROVAL_ASK,
    QUESTION_ASK,
    TOOL_DISPATCH,
    TOOL_EFFECT,
    CREDENTIAL_WAIT,
)
"""ph-core's kinds, as this leaf declares them — what `isolated_intent_kinds` keeps."""
