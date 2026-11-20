"""The vocabulary of the daemon as values: one `Verb` for each method (P8-11, issue 74).

**What this closes.** The name of a method, its params and its reply were three
values that had to agree, and nothing checked that they did. The name was a
string literal at each call site. The params model was named again in the
dispatch table of the server. The reply was whatever the handler returned. So
`client.call("session/status", SessionParams(session_id=s))` type-checked, and
so did `client.call("sessions/list", SessionParams(...))`. P8-07 typed what the
daemon receives, and P8-09 typed the fields that a client sends. Neither could
check that a params model belonged to the method that it was sent under, because
nothing held a name and a model together. A `Verb` holds them together, and this
module is where the verbs are declared.

**Both ends read this module, so it is a leaf.** From `ph_app` it imports
`protocol`, `params` and `payloads` and nothing else: no handlers, no client, no
daemon. The tables of the server are keyed by `verb.name`, and a front end reads
the same constants without importing `ph_app.daemon.server`
(`test_app_layering`).

**`Verb` is not for notifications.** A notice binds its name to its payload on
the payload itself (`SessionScoped.METHOD`). That let `Root.publish` take one
argument, and not two that had to agree. If this module owned those names, the
two arguments would come back. Verbs cover the request/reply half only.
`NOTICES` and `notice_of` keep the binding that they have.

**Every method is here, or it is unreachable.** `VOCABULARY` is derived from
this list and is not written beside it. The old arrangement had three spellings:
the table keys, the hand-written set of the test, and the literals at 26 call
sites. They are now one.

@module ph_app.verbs
"""

from __future__ import annotations

from typing import Any

from ph.seams.schedule import Schedule

from .params import (
    AdoptParams,
    BrowseParams,
    CancelScheduleParams,
    CommandParams,
    CreateScheduleParams,
    InitializeParams,
    ModelParams,
    NewSessionParams,
    OwedParams,
    PresetParams,
    ProbeParams,
    PromptParams,
    PutAttachmentParams,
    ResumeParams,
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
    DaemonLifetime,
    DaemonStatusReply,
    NewSessionReply,
    OwedListing,
    PresetApplied,
    ProbeReply,
    ResumeReport,
    RootDescription,
    RootListing,
    RootStatusReply,
    ScheduleCanceled,
    SessionBrowse,
    SessionChildrenNotice,
    SessionCommandsNotice,
    SessionDetached,
    SessionModelsReply,
    SessionPresetsReply,
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
    "DAEMON_HELLO",
    "DAEMON_LIFETIME",
    "DAEMON_PROBE",
    "DAEMON_STATUS",
    "INITIALIZE",
    "MODELS_LIST",
    "MUTATING",
    "PRESETS_LIST",
    "SCHEDULE_CANCEL",
    "SCHEDULE_CREATE",
    "SCHEDULE_LIST",
    "SCREENS_LIST",
    "SESSIONS_BROWSE",
    "SESSIONS_LIST",
    "SESSIONS_OWED",
    "SESSIONS_RESUME",
    "SESSION_ADOPT",
    "SESSION_ATTACH",
    "SESSION_CANCEL",
    "SESSION_CHILDREN",
    "SESSION_COMMAND",
    "SESSION_DETACH",
    "SESSION_MODEL",
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
"""The name that P5-01 gave `initialize`. The same handler answers both.

They are two verbs, and not one verb with two names, because a verb *is* a name.
A client that says either name gets the same reply model, and that is the fact
worth declaring."""

DAEMON_STATUS = Verb("daemon/status", NoParams, DaemonStatusReply)
DAEMON_LIFETIME = Verb("daemon/lifetime", NoParams, DaemonLifetime)
"""Why this daemon is still here, read one time at attach.

It is a verb as well as a notification. A client that connected between two
changes would otherwise draw nothing until something moved. `session/attach`
replies with a footer, and does not wait for the first `session.status`, for the
same reason."""
DAEMON_PROBE = Verb("daemon/probe", ProbeParams, ProbeReply)
"""One of the three orchestrator questions, over the socket, for
`phern agents probe` and an `exec` probe (P13-12). The same answers are served
over TCP when `--probe-bind` is given. This route is for a host with no port to
spare."""
SHUTDOWN = Notify("shutdown", NoParams)
"""The one method with no reply. It is a `Notify`, so that this stays true.

A request that awaits a reply would wait for a frame that the daemon is losing
the ability to write at the same moment. "Stop" is therefore not a question and
gets no id. When this was a `Verb`, that was a convention and not a rule:
`client.call(SHUTDOWN, NoParams())` type-checked, and every call site that used
`notify` was correct by luck. `Notify` says why the two routes are different
types."""

# --- roots ---------------------------------------------------------------------

SESSIONS_LIST = Verb("sessions/list", NoParams, RootListing)
SESSIONS_BROWSE = Verb("sessions/browse", BrowseParams, SessionBrowse)
SESSIONS_OWED = Verb("sessions/owed", OwedParams, OwedListing)
"""Every stored root that still owes work, and why (P13-02). A resumer asks this
before it asks for anything to be mounted."""
SESSIONS_RESUME = Verb("sessions/resume", ResumeParams, ResumeReport)
"""Mount the roots named, or every owed root, and say what became of each (P13-03)."""
SESSION_NEW = Verb("session/new", NewSessionParams, NewSessionReply)
SESSION_ATTACH = Verb("session/attach", SessionParams, AttachReply)
SESSION_DETACH = Verb("session/detach", SessionParams, SessionDetached)
SESSION_STATUS = Verb("session/status", SessionParams, RootStatusReply)
SESSION_CANCEL = Verb("session/cancel", SessionParams, RootDescription)
SESSION_SNAPSHOT = Verb("session/snapshot", SnapshotParams, SnapshotPage)

# --- the read-only projections (P5-14) -----------------------------------------
# Three of them answer with the *notice* type. The reply and the notification
# have one shape, and the server was the only party that did not say so.

SESSION_READINGS = Verb("session/readings", SessionParams, SessionReadingsReply)
COMMANDS_LIST = Verb("commands/list", SessionParams, SessionCommandsNotice)
SCREENS_LIST = Verb("screens/list", SessionParams, SessionScreensNotice)
SESSION_CHILDREN = Verb("session/children", SessionParams, SessionChildrenNotice)
"""Every sub-agent beneath a root, from their own logs (P11-07), for a client
that has just attached. `session.children` pushes the same list whenever it
moves."""
TOOLS_LIST = Verb("tools/list", SessionParams, SessionToolsReply)
SKILLS_LIST = Verb("skills/list", SessionParams, SessionSkillsReply)
PRESETS_LIST = Verb("presets/list", SessionParams, SessionPresetsReply)
MODELS_LIST = Verb("models/list", SessionParams, SessionModelsReply)

# --- attachments and credentials -----------------------------------------------

ATTACHMENT_PUT = Verb("attachment/put", PutAttachmentParams, AttachmentStored)
CREDENTIALS_HELD = Verb("credentials/held", SessionParams, CredentialsHeldReply)

# --- the schedule seam over the wire (P5-06, P5-10) ----------------------------

SCHEDULE_CREATE = Verb("schedule/create", CreateScheduleParams, Schedule)
SCHEDULE_CANCEL = Verb("schedule/cancel", CancelScheduleParams, ScheduleCanceled)
SCHEDULE_LIST = Verb("schedule/list", SessionParams, SessionSchedulesReply)

# --- the mutations -------------------------------------------------------------
# Every verb in `MUTATING` goes through the idempotence guard of the daemon. A
# client therefore calls them through `DaemonClient.mutate`, and their replies
# are `R | MutationRepeated`. See `MUTATIONS` for the reason why the key is
# claimed there and nowhere else. `credentials/store` and `session/adopt` are
# declared in this section and are not in `MUTATING`, which says why.

SESSION_PROMPT = Verb("session/prompt", PromptParams, RootDescription)
SESSION_COMMAND = Verb("session/command", CommandParams, CommandShown)
SESSION_STAGE = Verb("session/stage", StageParams, SessionStagedNotice)
SESSION_SHELL = Verb("session/shell", ShellParams, ShellReply)
SESSION_PRESET = Verb("session/preset", PresetParams, PresetApplied)
SESSION_MODEL = Verb("session/model", ModelParams, RootDescription)
CREDENTIALS_STORE = Verb("credentials/store", StoreCredentialParams, CredentialStored)
SESSION_ADOPT = Verb("session/adopt", AdoptParams, RootDescription)

MUTATING = (
    SESSION_PROMPT,
    SESSION_COMMAND,
    SESSION_STAGE,
    SESSION_SHELL,
    SESSION_PRESET,
    SESSION_MODEL,
)
"""The verbs that change a root under an idempotence key.

They are named here, and not only in the table of the server, so that a *client*
can tell which route a verb goes through. `mutate` stamps the key and `call`
does not, and a mutation that is sent through `call` silently loses the
write-ahead guard. Four verbs are absent on purpose:

- `attachment/put` is content-addressed, so a retry is already a no-op.
- `session/new` goes through `start`, which is idempotent by id.
- `credentials/store` stores one value one time when it is sent twice, and a key
  would outlive the value that it named. The value is in the memory of the
  daemon process, and after a restart the only way to bring it back is to send
  it again (T5).
- `session/adopt` makes one adoption when one version is accepted twice. The
  latest adoption is what the next start reads.

`MUTATIONS` gives the reasons for the first two, and
`_Connection._credentials_store` gives the reason for the third.
"""

VOCABULARY_VERBS: tuple[Verb[Any, Any] | Notify[Any], ...] = tuple(
    one for one in tuple(globals().values()) if isinstance(one, Verb | Notify)
)
"""Every verb that this module declares, read from the module itself.

The tuple is read from the module and is not listed by hand. The first draft
listed by hand: a 22-name `READING` tuple that was exactly the declarations
minus `MUTATING`. Adding a verb then meant three edits in this one file, and the
claim of the module docstring that the three spellings are now one was wrong by
one. A missed tuple edit failed `test_daemon_methods` with "declared but not
routed", which was the wrong half of the wrong question.

It uses `globals()` and not introspection in the style of
`SessionScoped.__subclasses__`. `test_payloads._notice_classes` gives the reason
for the same choice: the namespace of a module is what this file wrote, and a
process-global registry is whatever the run happened to import.
"""

UNKEYED: tuple[Verb[Any, Any] | Notify[Any], ...] = tuple(
    one for one in VOCABULARY_VERBS if one not in MUTATING
)
"""Every verb that is not a mutation. With `MUTATING`, this is the whole
vocabulary.

**It is not named `READING`**, which was its earlier name. Members of it write:
`shutdown`, `session/new`, `session/cancel`, `attachment/put`,
`schedule/create`, `schedule/cancel`, `credentials/store` and `session/adopt`.
So the name described a property most of the set lacked. What the members have
in common is the property that `MUTATIONS` names: no idempotence key, and so no
write-ahead guard. The docstring of `METHODS` already described the set
correctly, as "Every method that is not a mutation".
"""

VOCABULARY = frozenset(verb.name for verb in VOCABULARY_VERBS)
"""Every method that the daemon answers, **derived**.

It used to be a hand-written set in `test_daemon_methods`, beside the two tables
of the server. The reasoning was "two lists that must agree is one list
checked". That was true as far as it went, and it left the set itself as a third
place to edit. Now that the set is derived from the verbs, the check that
remains is the useful one: a verb that is declared here and never routed to a
handler is unreachable, and `test_daemon_methods` fails and names it.
"""

# This is checked at import, because the failure is otherwise silent. Two verbs
# with one name would give the table of the server one row, and the verb that
# was declared second would answer for both. The reply model of the first verb
# would still type-check at every call site that used it. This is a comment and
# not the docstring that it used to be: a string after an `assert` documents
# nothing, because it is a no-op expression that no doc tool renders.
assert len(VOCABULARY) == len(VOCABULARY_VERBS), (
    f"a verb name is declared twice: {sorted(verb.name for verb in VOCABULARY_VERBS)}"
)
