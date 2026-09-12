"""The daemon's vocabulary as values: one `Verb` per method (P8-11, issue 74).

**What this closes.** A method's name, its params and its reply were three
values that had to agree with nothing checking that they did. The name was a
string literal at each call site, the params model was named again in the
server's dispatch table, and the reply was whatever the handler returned — so
`client.call("session/status", SessionParams(session_id=s))` type-checked and so
did `client.call("sessions/list", SessionParams(...))`. P8-07 typed what the
daemon receives and P8-09 typed the fields a client sends; neither could check
that a params model belonged to the method it was sent under, because nothing
held a name and a model together. A `Verb` is that holding, and this module is
where the holdings live.

**Read by both ends, which is why it is a leaf.** It imports `protocol`,
`params` and `payloads` and nothing else — no handlers, no client, no daemon —
so the server's tables are keyed off `verb.name` and a front end reads the same
constants without importing `ph_app.daemon.server` (`test_app_layering`).

**`Verb` is not for notifications.** A notice binds its name to its payload on
the payload itself (`SessionScoped.METHOD`), which is what let `Root.publish`
take one argument rather than two that had to agree. Owning those names here
would put that back. Verbs cover the request/reply half only; `NOTICES` and
`notice_of` keep the binding they have.

**Every method is here or it is unreachable**, and `VOCABULARY` is derived from
this list rather than written beside it — the three spellings the old
arrangement had (the table keys, the test's hand-written set, and the literals
at 26 call sites) become one.

@module ph_app.verbs
"""

from __future__ import annotations

from typing import Any

from ph.seams.schedule import Schedule

from .params import (
    CancelScheduleParams,
    CommandParams,
    CreateScheduleParams,
    HeldCredentialsParams,
    InitializeParams,
    NewSessionParams,
    PresetParams,
    PromptParams,
    PutAttachmentParams,
    ShellParams,
    SnapshotParams,
    StageParams,
    StoreCredentialParams,
)
from .payloads import (
    AttachmentStored,
    AttachReply,
    CommandShown,
    CredentialsHeldReply,
    CredentialStored,
    DaemonConfigReply,
    DaemonStatusReply,
    PresetApplied,
    RootDescription,
    RootListing,
    RootStatusReply,
    ScheduleCancelled,
    SessionBrowse,
    SessionCommandsNotice,
    SessionDetached,
    SessionReadingsReply,
    SessionSchedulesReply,
    SessionScreensNotice,
    SessionSkillsReply,
    SessionStagedNotice,
    SessionToolsReply,
    ShellReply,
    SnapshotPage,
)
from .protocol import CapabilityBlock, NoParams, Notify, SessionParams, Verb

__all__ = [
    "ATTACHMENT_PUT",
    "COMMANDS_LIST",
    "CREDENTIALS_HELD",
    "CREDENTIALS_STORE",
    "DAEMON_CONFIG",
    "DAEMON_HELLO",
    "DAEMON_STATUS",
    "INITIALIZE",
    "MUTATING",
    "SCHEDULE_CANCEL",
    "SCHEDULE_CREATE",
    "SCHEDULE_LIST",
    "SCREENS_LIST",
    "SESSIONS_BROWSE",
    "SESSIONS_LIST",
    "SESSION_ATTACH",
    "SESSION_CANCEL",
    "SESSION_COMMAND",
    "SESSION_DETACH",
    "SESSION_NEW",
    "SESSION_PRESET",
    "SESSION_PROMPT",
    "SESSION_READINGS",
    "SESSION_SHELL",
    "SESSION_SNAPSHOT",
    "SESSION_STAGE",
    "SESSION_STATUS",
    "SHUTDOWN",
    "SKILLS_LIST",
    "TOOLS_LIST",
    "UNKEYED",
    "VOCABULARY",
    "VOCABULARY_VERBS",
]

# --- the daemon itself ---------------------------------------------------------

INITIALIZE = Verb("initialize", InitializeParams, CapabilityBlock)
"""Trade capability blocks. The dsh SDK's name for it."""

DAEMON_HELLO = Verb("daemon/hello", InitializeParams, CapabilityBlock)
"""P5-01's name for `initialize`, and the same handler answers both.

Two verbs rather than one with two names: a verb *is* a name, and a client that
says either gets the same reply model — which is the fact worth declaring."""

DAEMON_CONFIG = Verb("daemon/config", NoParams, DaemonConfigReply)
DAEMON_STATUS = Verb("daemon/status", NoParams, DaemonStatusReply)
SHUTDOWN = Notify("shutdown", NoParams)
"""The one method with no reply, and a `Notify` so that stays true.

A request awaiting a reply would wait on a frame the daemon is concurrently
losing the ability to write, so "stop" is not a question and does not get an
id. As a `Verb` that was a convention rather than a rule:
`client.call(SHUTDOWN, NoParams())` type-checked, and every call site using
`notify` was luck. `Notify` says why the two doors are different types."""

# --- roots ---------------------------------------------------------------------

SESSIONS_LIST = Verb("sessions/list", NoParams, RootListing)
SESSIONS_BROWSE = Verb("sessions/browse", NoParams, SessionBrowse)
SESSION_NEW = Verb("session/new", NewSessionParams, RootDescription)
SESSION_ATTACH = Verb("session/attach", SessionParams, AttachReply)
SESSION_DETACH = Verb("session/detach", SessionParams, SessionDetached)
SESSION_STATUS = Verb("session/status", SessionParams, RootStatusReply)
SESSION_CANCEL = Verb("session/cancel", SessionParams, RootDescription)
SESSION_SNAPSHOT = Verb("session/snapshot", SnapshotParams, SnapshotPage)

# --- the read-only projections (P5-14) -----------------------------------------
# Two of the four answer with the *notice* type, because the reply and the
# notification are one shape and the server was the only party not saying so.

SESSION_READINGS = Verb("session/readings", SessionParams, SessionReadingsReply)
COMMANDS_LIST = Verb("commands/list", SessionParams, SessionCommandsNotice)
SCREENS_LIST = Verb("screens/list", SessionParams, SessionScreensNotice)
TOOLS_LIST = Verb("tools/list", SessionParams, SessionToolsReply)
SKILLS_LIST = Verb("skills/list", SessionParams, SessionSkillsReply)

# --- attachments and credentials -----------------------------------------------

ATTACHMENT_PUT = Verb("attachment/put", PutAttachmentParams, AttachmentStored)
CREDENTIALS_HELD = Verb("credentials/held", HeldCredentialsParams, CredentialsHeldReply)

# --- the schedule seam over the wire (P5-06, P5-10) ----------------------------

SCHEDULE_CREATE = Verb("schedule/create", CreateScheduleParams, Schedule)
SCHEDULE_CANCEL = Verb("schedule/cancel", CancelScheduleParams, ScheduleCancelled)
SCHEDULE_LIST = Verb("schedule/list", SessionParams, SessionSchedulesReply)

# --- the mutations -------------------------------------------------------------
# Every one of these goes through the daemon's idempotence guard, so a client
# calls them through `DaemonClient.mutate` and their replies are
# `R | MutationRepeated` — see `MUTATIONS` for why the key is claimed there and
# nowhere else.

SESSION_PROMPT = Verb("session/prompt", PromptParams, RootDescription)
SESSION_COMMAND = Verb("session/command", CommandParams, CommandShown)
SESSION_STAGE = Verb("session/stage", StageParams, SessionStagedNotice)
SESSION_SHELL = Verb("session/shell", ShellParams, ShellReply)
SESSION_PRESET = Verb("session/preset", PresetParams, PresetApplied)
CREDENTIALS_STORE = Verb("credentials/store", StoreCredentialParams, CredentialStored)

MUTATING = (
    SESSION_PROMPT,
    SESSION_COMMAND,
    SESSION_STAGE,
    SESSION_SHELL,
    SESSION_PRESET,
    CREDENTIALS_STORE,
)
"""The verbs that change a root under an idempotence key.

Named here rather than only in the server's table so a *client* can tell which
door a verb goes through — `mutate` stamps the key, `call` does not, and a
mutation sent through `call` silently loses the write-ahead guard. Deliberately
absent, and `MUTATIONS` says why: `attachment/put` (content-addressed, so a
retry is already a no-op) and `session/new` (`start` is idempotent by id).
"""

VOCABULARY_VERBS: tuple[Verb[Any, Any] | Notify[Any], ...] = tuple(
    one for one in tuple(globals().values()) if isinstance(one, Verb | Notify)
)
"""Every verb this module declares, read off the module itself.

Off the module rather than hand-listed, which is what the first draft did: a
22-name `READING` tuple that was precisely the declarations minus `MUTATING`,
so adding a verb meant three edits in this one file and the module docstring's
claim that "the three spellings become one" was optimistic by one. Missing the
tuple edit failed `test_daemon_methods` with "declared but not routed", which
was the wrong half of the wrong question.

`globals()` rather than `SessionScoped.__subclasses__`-style introspection for
the reason `test_payloads._notice_classes` gives about the same choice: a
module's own namespace is what this file wrote, where a process-global registry
is whatever the run happened to import.
"""

UNKEYED: tuple[Verb[Any, Any] | Notify[Any], ...] = tuple(
    one for one in VOCABULARY_VERBS if one not in MUTATING
)
"""Everything that is not a mutation. Together with `MUTATING`, the whole of it.

**Not `READING`**, which is what this was called: six of its members write —
`shutdown`, `session/new`, `session/cancel`, `attachment/put`,
`schedule/create` and `schedule/cancel` — so the name described a property most
of the set lacked. What they have in common is the one `MUTATIONS` names: no
idempotence key, so no write-ahead guard. `METHODS`' own docstring already had
the honest phrasing, "every method that is not a mutation".
"""

VOCABULARY = frozenset(verb.name for verb in VOCABULARY_VERBS)
"""Every method the daemon answers, **derived**.

It was a hand-written set in `test_daemon_methods` beside the server's two
tables — "two lists that must agree is one list checked", which was true as far
as it went and left the set itself as a third place to edit. Derived from the
verbs, the check that remains is the one worth making: a verb declared here and
never routed to a handler is unreachable, and `test_daemon_methods` fails
naming it.
"""

# Checked at import because the failure is otherwise silent: two verbs with one
# name would give the server's table one row, and whichever was declared second
# would answer for both — with the first verb's reply model still type-checking
# at every call site that used it. A comment rather than the docstring this
# once was: a string after an `assert` documents nothing, it is a no-op
# expression no doc tool renders.
assert len(VOCABULARY) == len(VOCABULARY_VERBS), (
    f"a verb name is declared twice: {sorted(verb.name for verb in VOCABULARY_VERBS)}"
)
