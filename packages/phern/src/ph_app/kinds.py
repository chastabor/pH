"""`ph_app`'s intent kinds: every pair the app writes through `ctx.intents`, in one leaf (T4).

Two kinds, each with its keys and its closer:

* `CLIENT_COMMAND` — the record that makes a daemon verb idempotent (P5-02), and its
  outcome (P10-10);
* `SUPERVISOR_RESTORE` — a retry putting a root's tree back to its last restore
  point, and whether it did (S12).

**Why a leaf.** Repair settles the kinds declared in the process doing the resume, and a
kind is declared when its module is imported. ph-core cannot import the app, so the
app brings its own: `ph_app/__init__.py` imports this module, so any process that has
loaded any part of the app — the daemon, `phern -p`, the trajectory viewer — has the
app's kinds declared, by a static import mypy sees. A process that never loaded the
app has none of them, and repair refuses a log holding an open one by name
(`ph.session.known_event_types.INTENT_PAIRS`), rather than leaving it open.

It is also where a log type the app alone writes would be declared with
`declare_log_type` (P10-03), so one module says everything the app brings to the log.
None ships: decision 7 keeps the app's types in ph-core's vocabulary, which must read
them where the app is not installed. The first would add `ph.session.known_event_types`,
itself pure, to what a leaf may import.

**What it may import:** the standard library, `ph.session.intents`, `ph.session.events`,
`ph.session.writers` and `ph.json`, so importing it can never cycle back through the daemon.
`test_intent_kinds.py` holds that line, and ph-core's `INTENT_PAIRS` to what this leaf
declares.

@module ph_app.kinds
"""

from __future__ import annotations

from typing import Any

from ph.json import JsonObject, as_str
from ph.session.events import SessionEvent
from ph.session.intents import IntentKind, Unsettled, declare_intent, opened_seq, seq_field
from ph.session.writers import log_writer

__all__ = ["CLIENT_COMMAND", "SUPERVISOR_RESTORE", "command_settled", "restore_settled"]


_LOG = log_writer(__name__)
"""This leaf's writer, which `CLIENT_COMMAND` carries (T6)."""


def _command(event: SessionEvent) -> str:
    return as_str(event.data.get("command"))


def command_settled(command: str) -> JsonObject:
    """A `client/command-settled` payload: the verb's key, and nothing else — what it
    means is `outcome_of`'s to say, and a repeat is told only that."""
    return {"command": command}


def _unknown(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """A verb whose settle nobody wrote: a crash, or an `act` that raised.

    `unknown` either way, and that is the point (P10-10): the key was claimed, so the
    act may have begun, and a retry told `repeated` with no more than that would
    assume it finished. The `unsettled` marker whoever writes this merges in is what
    says so; `not-started` cannot reach here — the kind is `buffered` — and would
    tell a client the same thing: ask.
    """
    return command_settled(_command(opened))


CLIENT_COMMAND = declare_intent(
    IntentKind(
        opened="client/command",
        settled="client/command-settled",
        opened_key=_command,
        settled_key=_command,
        orphan="outcome-unknown",
        # **Buffered, not durable**, and that is P5-02's rule kept: the key goes
        # to disk with the act's own records — the flush `_mutate` makes before
        # it replies (F7), or an act's own barrier (`!!`) — never alone before
        # the act. A key made durable first would turn a crash *before* the act
        # into a refused retry for work that never began; a crash before any
        # flush re-runs the verb, which a transcript shows, where a dropped one
        # is invisible.
        barrier="buffered",
        closer=_unknown,
        writer=_LOG,
    )
)
"""The record that makes a mutating command idempotent (P5-02), and its outcome
(P10-10). Keyed by `clientId:commandId`; filled by `ph_app.daemon.server`."""


# ------------------------------------------------------------------ restore --


def _restoring_seq(event: SessionEvent) -> str | None:
    """The restore a `supervisor/restored` settles — its `restoringSeq`, as a key."""
    return seq_field(event, "restoringSeq")


def restore_settled(opened: SessionEvent, *, ok: bool, detail: str | None = None) -> dict[str, Any]:
    """A `supervisor/restored` payload — the ladder's and repair's alike: which
    attempt, which tree, and whether the tree is now that tree."""
    data: dict[str, Any] = {
        "restoringSeq": opened.seq,
        "attempt": opened.data.get("attempt"),
        "tree": opened.data.get("tree"),
        "ok": ok,
    }
    if detail is not None:
        data["detail"] = detail
    return data


def _restore_interrupted(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """A restore nobody saw finish: not `ok`, and — in the `unsettled` marker whoever
    writes this merges in — that the tree may be part-way between the two states."""
    return restore_settled(opened, ok=False)


SUPERVISOR_RESTORE = declare_intent(
    IntentKind(
        opened="supervisor/restoring",
        settled="supervisor/restored",
        opened_key=opened_seq,
        settled_key=_restoring_seq,
        # A restore rewrites and deletes files, so one cut short leaves a tree that
        # is neither the checkpoint nor what the crashed attempt left: a reader is
        # owed "maybe", not the rollback the retry set out to make.
        orphan="outcome-unknown",
        # On disk before the tree is touched (S12). The retry used to restore first
        # and record after, so a daemon that died restoring had rewritten the tree
        # with nothing in the log saying so. One the log cannot record is not run.
        barrier="durable",
        closer=_restore_interrupted,
        writer=_LOG,
    )
)
"""A retry putting a root's tree back to its last restore point, then whether it did
(S12). Keyed by the opening record's seq; filled by `ph_app.daemon.supervisor`."""
