"""The intent kinds of `ph_app`: every pair that the app writes through
`ctx.intents`, in one leaf (T4).

There is one kind today, `CLIENT_COMMAND`, with its key and its closer. It is
the record that makes a daemon verb idempotent (P5-02), and the outcome of that
verb (P10-10). The restore of a supervisor is `WORKSPACE_RESTORE` in core, and
the seam that restores opens it.

**Why a leaf.** Repair settles the kinds that are declared in the process that
does the resume, and a kind is declared when its module is imported. ph-core
cannot import the app, so the app brings its own kinds. `ph_app/__init__.py`
imports this module. Any process that has loaded any part of the app (the
daemon, `phern -p`, the trajectory viewer) therefore has the kinds of the app
declared, by a static import that mypy sees. A process that never loaded the app
has none of them. Repair then refuses a log that holds an open one, by name
(`ph.session.known_event_types.INTENT_PAIRS`), and does not leave it open.

This leaf is also where a log type that only the app writes would be declared
with `declare_log_type` (P10-03), so that one module says everything that the
app brings to the log. No such type ships. Decision 7 keeps the types of the app
in the vocabulary of ph-core, which must read them where the app is not
installed. The first such type would make this leaf import
`ph.session.known_event_types`. That module is pure, and the rule for leaves
already allows it.

**What it may import:** the standard library, `ph.session.intents`,
`ph.session.events`, `ph.session.writers` and `ph.json`, so importing it can
never cycle back through the daemon. `test_intent_kinds.py` checks that rule for
every leaf, and its list also allows `ph.session.known_event_types`. The same
test holds the `INTENT_PAIRS` of ph-core equal to what the leaves declare.

@module ph_app.kinds
"""

from __future__ import annotations

from ph.json import JsonObject, as_str
from ph.session.events import SessionEvent
from ph.session.intents import IntentKind, Unsettled, declare_intent
from ph.session.writers import log_writer

__all__ = ["CLIENT_COMMAND", "command_settled"]


_LOG = log_writer(__name__)
"""This leaf's writer, which `CLIENT_COMMAND` carries (T6)."""


def _command(event: SessionEvent) -> str:
    return as_str(event.data.get("command"))


def command_settled(command: str) -> JsonObject:
    """A `client/command-settled` payload: the key of the verb, and nothing else.
    `outcome_of` says what the settle means, and a repeat is told only that."""
    return {"command": command}


def _unknown(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """A verb whose settle nobody wrote: a crash, or an `act` that raised.

    The outcome is `unknown` in both cases, on purpose (P10-10). The key was
    claimed, so the act may have begun. A retry that was told only `repeated`
    would assume that the act finished. Whoever writes this settle merges in the
    `unsettled` marker, and that marker says so. `not-started` cannot reach
    here, because the kind is `buffered`. It would tell a client the same thing:
    ask.
    """
    return command_settled(_command(opened))


CLIENT_COMMAND = declare_intent(
    IntentKind(
        opened="client/command",
        settled="client/command-settled",
        opened_key=_command,
        settled_key=_command,
        orphan="outcome-unknown",
        # **Buffered, not durable.** This keeps the rule of P5-02: the key goes
        # to disk with the records of the act itself, and never alone before the
        # act. `_mutate` puts them on disk before it replies (F7, in `_acted`),
        # or the act makes its own barrier (`!!`). A key that was made durable
        # first would turn a crash *before* the act into a refused retry for
        # work that never began. A crash before any flush runs the verb again,
        # and a transcript shows that. A dropped verb is invisible.
        barrier="buffered",
        closer=_unknown,
        writer=_LOG,
    )
)
"""The record that makes a mutating command idempotent (P5-02), and its outcome
(P10-10). It is keyed by `clientId:commandId`, and `ph_app.daemon.server` fills
it."""
