"""The event vocabulary that this build understands.

The persistence read path refuses to interpret a log that contains a type
outside this set, **unless** the event carries `ignorable`. A newer harness
probably wrote such a log. An unrecognized *required* event may change how the
rest of the log is read. So to skip it silently would reconstruct a wrong
session, and not only an incomplete one.

`test_log_writers.py` checks the set. It walks every write in the shipped code
of **every** package: ph-core and the bundles alike. A type that ships without
an entry here would be a log that this build could write and then refuse to
read. Whether a type is *ignorable* is a separate decision. The comment beside
the type argues it, and the tests of its producer check it.

**The write path refuses what the read path refuses** (F11). The append behind
every writer (`Session._append`, T6) raises `UnknownEventTypeError` for a type
outside the vocabulary. Earlier it wrote the type and left the log impossible
to open at the next resume. A package outside ph-core adds its type with
`declare_log_type`, at import, the way a bus event is declared with
`EventRegistry.declare`.

**Whether only an auditor reads a type is declared here too**
(`AUDIT_ONLY_SESSION_EVENT_TYPES`, `declare_log_type(audit_only=)`). A front
end draws no conversation row for such a type, and the auditor's view records
it. So a front end derives both of its tables from the vocabulary, and does not
list each type again in each table.

**Which fields of a type telemetry ships is declared here too**
(`EXPORTED_SESSION_EVENT_FIELDS`, `declare_log_type(exported=)`). A ledger
record names its event, and carries only the fields that its type declares. So
the telemetry seam never names a type or a payload key.

@module ph.session.known_event_types
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

__all__ = [
    "AUDIT_ONLY_SESSION_EVENT_TYPES",
    "EXPORTED_SESSION_EVENT_FIELDS",
    "IGNORABLE_SESSION_EVENT_TYPES",
    "INTENT_PAIRS",
    "KNOWN_SESSION_EVENT_TYPES",
    "WRITERS",
    "IntentPair",
    "LogTypeDeclaration",
    "LogTypeError",
    "UnknownEventTypeError",
    "declare_log_type",
    "declared_owner",
    "exported_fields",
    "is_audit_only",
    "is_ignorable",
    "is_known",
    "known_log_types",
    "written_by",
]

KNOWN_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # agent lifecycle
        "agent/inbox/spliced",
        "assistant/chunk",
        "assistant/message",
        "request/context",
        "request/header",
        "session/end-seed",
        "session/resumed",
        # A model call of a step, made again by the row named in `by` (P1). There
        # is one for each retry that the loop accepted, so `step/start` plus
        # these is every model call. Ignorable: a reader that skips it
        # undercounts the calls that a ceiling has seen. That is accounting. The
        # conversation is unchanged, because a failed attempt appends no message.
        "step/retry",
        # A session continued in a fresh file (§7 step 6). It is the terminal
        # record of the parent, and it names the log that carries on. It is the
        # forward half of a link. The backward half is the `parent_session`
        # header of the child. It is needed because a segment and a branch have
        # the same structure: both are a fork, and only this record says which
        # one a reader is looking at.
        "session/segmented",
        # A mutating command that a protocol client asked for. It is recorded so
        # that asking twice is safe across a restart (P5-02). It is not a slash
        # `command/*`. Its settle says that the verb finished. When repair
        # writes the settle, it says that a crash left the outcome unknown
        # (P10-10), and a retry is told that.
        "client/command",
        "client/command-settled",
        # The retry ladder of the supervisor (P5-04): a crashed root task that
        # is run again, the ladder giving up, and a retry that worked. These are
        # in the log and not in the memory of the supervisor, because "this root
        # tried three times and stopped" must survive the process that decided
        # it. A root that was resumed in the middle of the ladder and forgot
        # would start the count over and retry forever.
        #
        # They are `supervisor/*` and not `agent/*`. `ph.agent` owns that
        # namespace: its registry declares `agent/status`, `agent/error` and
        # `agent/inbox/*` with `owner="ph.agent"`. These are the records of the
        # supervisor *about* an agent, and not the records of the agent itself.
        # The earlier name `agent/failed` also read too much like
        # `agent/error`, which means something else.
        "supervisor/retry",
        "supervisor/failed",
        "supervisor/recovered",
        # A root released because it was idle (P5-05). It has its own record
        # because the alternative is an unexplained gap. A transcript that stops
        # for three days and then resumes reads as a crash. The
        # `session/resumed` on the way back says only that something was
        # resumed. It does not say that nothing had gone wrong.
        "supervisor/passivated",
        # The daemon can no longer be reached. This is written into every root
        # that it is running (P5-11). No connected client reads it: by the time
        # it is appended the socket is gone, so a client can neither be notified
        # nor connect to ask. The transcript is the surface that survives. A run
        # that went quiet at 18:04 says here that its supervisor lost its socket
        # at 18:04, and names the fix.
        "supervisor/unreachable",
        # A pollable invariant did not hold when the supervisor last asked (I6).
        # The finding is in the log and not in the stderr of the daemon, because
        # it is about bookkeeping that the same process owns. When a projection
        # no longer equals its fold, the account that the process gives of
        # itself is exactly what is in doubt.
        "supervisor/violated",
        # Future work that a root will do, and the claim that it is doing it
        # (P5-06). `schedule/tick` is appended *before* delivery. A tick that is
        # recorded and then lost to a crash costs one skipped run. A tick that
        # is delivered and then lost costs a repeated run, and a repeated
        # scheduled prompt bills twice and confuses the transcript.
        "schedule/created",
        "schedule/canceled",
        "schedule/tick",
        # The objective of an autonomous run, its spend, its quality gates and
        # how it ended (P5-07). `goal/settled` names *which* budget stopped the
        # run. "It stopped" and "it ran out of turns" are different things to
        # the reader of the trace, and only one of them suggests a larger
        # budget.
        "goal/set",
        "goal/continued",
        "goal/gate",
        "goal/settled",
        "step/end",
        "step/start",
        # Appended after the gate of the pipeline has decided and before the
        # body runs (B4, P7-15): after the *wait*, before the *act*. So a call
        # that is still waiting for a human when the process dies has no record,
        # because nothing happened. Repair reads that as `TOOL_NOT_STARTED`. The
        # arguments of this record are what ran. The arguments that the model
        # wrote are in the assistant message.
        "tool/call",
        "tool/result",
        "turn/end",
        "turn/start",
        "user/message",
        # policy and human-in-the-loop
        "approval/asked",
        "approval/decided",
        "approval/policy",
        # The model asks the user something that is not an approval (P7-09,
        # emitted by `tool-ask-user`). There are two events for the same reason
        # that approvals have two. The ask is appended before the waterfall
        # runs. So a crash while somebody was deciding leaves the question in
        # the log and does not lose it: `QUESTION_ASK` is open, and repair
        # settles it on resume.
        #
        # Only a question that was actually *delivered* appears at all. An
        # unattended ask resolves to "no answer" and appends nothing. So a log
        # never claims that a person was asked and declined when no person was
        # there.
        "question/asked",
        "question/answered",
        # A shell command run outside a tool call, and what it printed (P7-10).
        # Both records go through `ShellService.run_recorded`. The
        # `session/shell` verb of `phern` calls it for a command that the user
        # ran from the composer, and an autonomous goal calls it for a gate
        # (S16). There are two events for the same reason that `tool/call` and
        # `tool/result` are two. The command is appended before it runs. So a
        # command that hangs, or that takes the daemon down with it, still shows
        # in the log what was started. That is exactly the command worth knowing
        # about.
        #
        # These are invisible to the model because they are not among the three
        # surface-eligible types. See `SURFACE_EVENT_TYPES`, which defines and
        # explains that mechanism.
        "shell/command",
        "shell/result",
        # The gating *posture* (P4-05; emitted by `hitl` in ph-stabilize). It is
        # its own type and not a field on `approval/policy`, which
        # `permission-presets` also writes. One last-write-wins fold with two
        # writers, where one writer does not know that the field exists, is a
        # posture that silently reverts when someone switches preset. The type
        # is listed here because the reader that refuses an unknown type belongs
        # to ph-core.
        "approval/mode",
        # A slash command, on disk before its body runs and settled after (S16).
        # This is `COMMAND_RUN`. Repair closes its orphan, because a body may do
        # anything that a command does, and `/revert` rewrites the tree.
        "command/done",
        "command/run",
        # The full environment that a session started in (session profiles, S3):
        # every environment row through its model, and the layers of the user
        # that it came from. Required: a reader that skipped it would rebuild a
        # session in an environment that it never ran in.
        "profile/base",
        # One deviation from that base, as a profile entry (S4): a start option,
        # a slash command or a verb. Required, and written before the change is
        # made. A reader that skipped one would rebuild a session without an
        # allowance that it ran with.
        "profile/override",
        # A named profile that moved since the session started (S6): a version
        # accepted for the next start, a version that a person said no to, and
        # an override that stops applying. The first two are ignorable: a reader
        # that skipped them would start on the base that it already has, or ask
        # one more time. The clear is required: a reader that skipped it would
        # rebuild an override that is no longer in force.
        "profile/adopted",
        "profile/declined",
        "profile/override-cleared",
        # An adopted version that a start could not mount, taken back with the
        # reason. Required: a reader that skipped it would apply the adoption
        # again.
        "profile/withdrawn",
        # Changes that the log could not hold when they were made. They were
        # refused and so were never in force. Required: a reader that skipped it
        # would start the session on a change that the user was told was
        # refused.
        "profile/refused",
        # The environment of the session, saved as a named profile (S7).
        # Ignorable, because it changes nothing that the session runs with.
        "profile/saved",
        # The body of a skill, read at runtime and hashed for the audit (S8,
        # decision 12). Ignorable: the version of a skill never stops a task or
        # makes a restart ask.
        "skill/read",
        "permission/preset",
        "sandbox/mode",
        # A confined command was refused something (P6-38): a host that the
        # egress proxy would not carry, or a write that the kernel turned back.
        # The proxy knows the host. The kernel refuses in silence, so the
        # filesystem half is read from what the command printed, and the record
        # says so (`via`). Ignorable: the result of the command already shows
        # that it failed. This record is the account of *why*, and it carries
        # the `/sandbox` line that lifts the refusal.
        "sandbox/denied",
        # capability observations
        "fs/observed",
        # Where the writes of an agent land (D21, P4-07). This is a *pair*. An
        # `acquired` with no `disposed` is how a leaked workspace is detected at
        # session open, and that only works if both halves are in the
        # vocabulary.
        "workspace/acquired",
        "workspace/disposed",
        # A workspace put back to a restore point, and whether it was (S12).
        # This is `WORKSPACE_RESTORE`. It is a pair because the restore is an
        # act of its own: a restore that is cut short leaves a tree that is
        # neither the restore point nor what was there. The seam opens it,
        # whoever asked: `/revert` or a retry.
        "workspace/restoring",
        "workspace/restored",
        # Where a tier is *about to* put a fresh tree, written before the tier
        # makes one (S12). An `acquiring` with nothing after it is a tree that a
        # crash may have left half-made: a checkout, or a mounted overlay.
        # Reconcile needs the root to find it.
        "workspace/acquiring",
        # A tree marked as evidence. It is written when the mark is made, and
        # not only on the closing half (P6-28). A retention is decided because a
        # run went wrong, and the most complete way for a run to go wrong writes
        # no `disposed` at all. So the mark has to be durable before that.
        "workspace/retained",
        # Materials that did not reach a fresh workspace (E14). It is appended
        # only when something failed. A silent success is the ordinary case. One
        # event for each agent that says "nothing went wrong" is noise in a log
        # whose value is that everything in it happened.
        "workspace/provisioned",
        # The restore point of each run (E7). **Not** ignorable: a reader that
        # skipped it would offer `/revert` a session with no restore points
        # while the refs are in the repository. That build would misread
        # recoverable state as unrecoverable.
        "workspace/checkpoint",
        # retry
        "llm/retry",
        # Questions that a decision model answered on one of the routes of the
        # session, for the agent of this session (`LlmRuntime.decide`): which
        # route, how many questions, and what the provider said it cost, or why
        # the route would not decide. This is not the request of a turn, so no
        # `assistant/message` carries the spend. A budget and the account of a
        # child find the spend here (`SPENDING_TYPES`). Ignorable: nothing in
        # the conversation moves, and a reader that skips it loses the metering.
        "llm/decided",
        "llm/decision-failed",
        # Code Mode dispatch records (log-only; see ph.tools.code_mode)
        "tool/code-dispatch",
        "tool/code-dispatch-start",
        # A call whose tool names its effect (P10-12). It is opened before the
        # body and settled with what the body returned. So the same effect,
        # asked for again under a new call id, is answered from the log. **Not**
        # ignorable: a reader that skipped them would run an effect that the log
        # says happened.
        "tool/effect",
        "tool/effect-settled",
        # A session, or one of its children, that waits for a credential that
        # its route names and this deployment cannot supply (T5). It is opened
        # when the resume check holds it, and settled when the name arrives.
        # Names only, never values (I-3). Ignorable: a reader that skips them
        # starts the work and meets the missing key at its first request, which
        # is what every build did before.
        "credential/needed",
        "credential/supplied",
        # Persistent-kernel state (D17; emitted by the snapshot policy of
        # ph-rlm). These make `persistence: "namespace"` admissible at all: the
        # seam takes the promise of the provider at registration, and these
        # events are the promise being kept. They are listed here and not in
        # ph-rlm because the *reader* is ph-core. The seed path refuses a log
        # that carries a type this build does not know, so the vocabulary has
        # one home.
        "kernel/snapshot",
        "kernel/restored",
        # Delegation (P3-11; emitted by a `ctx.subagents` provider). The types
        # are named for the *seam*, and not for the bundle that ships the first
        # provider. The readers are ph-core and `ph_app`, and neither imports
        # ph-rlm. A second provider that emitted `rlm/…` would state a false
        # identity in an append-only log. The provider names itself in the
        # payload instead. Each record is in the **own log of the child**
        # (Phase 11). The state of a child *is* the fold of these records, with
        # no copy in the parent beside it. So each one is required reading: the
        # ladder counts starts from the statuses, and the value of a child is
        # its `returned`.
        "subagent/admitted",
        "subagent/deleted",
        "subagent/returned",
        "subagent/status",
        # The Continual Harness (P3-16). Its state *is* this fold, and there is
        # no side file to fall back on. So a reader that skipped one would show
        # a harness that the session does not have. A rollback is a refinement
        # with `rollbackOf` set. So there is one type, and not two fold cases
        # that have to agree about one operation.
        "harness/refined",
        # Considering and declining. Unlike the refinement itself, this changes
        # no state. It advances the auto-refine cooldown (H7). So it is
        # ignorable: a reader that skips it gets one extra review pass, and not
        # a harness that the session does not have.
        "harness/refine-considered",
        # The probe by H1 of the reference of a skill entry, and what it found
        # (S21). This is `HARNESS_PROBE`, declared in `ph_rlm.kinds`. Ignorable:
        # the refinement records what it refused. A reader that skips these
        # loses the account of the check, and not the harness.
        "harness/probe",
        "harness/probed",
        # Offloading (P4-02, G2; emitted by `tool-result-offload` in
        # `ph-stabilize`). The forwarding address for a result of which the
        # model was handed a preview. Ignorable: `tool/result` already carries
        # what the model saw. A reader that skips this loses the way back to the
        # original, and not the conversation. The `context/loaded` recipe passes
        # the same test.
        "offload/spilled",
        # The input side (G3): where a pasted message went. Ignorable for the
        # same reason. The surface `replace` beside it is what the model reads,
        # and that is a required `user/message`. So a reader that skips this
        # loses the forwarding address and nothing else.
        "offload/input-spilled",
        # Planning (P4-01, G1). `tool-todo` in `ph-stabilize` emits it, and so
        # does `skill-steps` when a skill that it read declares a procedure.
        # There are two writers of one event on purpose, because both mean "the
        # list is now this". The list *is* this fold, and there is no side
        # table. The list reaches the model through a prompt context, so a
        # reader that skipped it would assemble a different prompt from the one
        # that the session had. Required, not ignorable: the difference is
        # model-visible.
        "todo/write",
        # The nudge budget of the procedure that `skill-steps` seeded last
        # (D16): the `max-nudges` of the skill, or `null` for the default of the
        # row. Ignorable: a reader that skips it steers against the budget of
        # the profile instead. That changes how many nudges come *next*. Every
        # nudge that was already sent is a required `user/message` of its own.
        "skill-steps/budget",
        # What a decision model said about a turn that stopped with a procedure
        # unfinished, and the threshold at which `skill-steps` held its nudge.
        # Ignorable: a held nudge is a message that was never sent. So a reader
        # that skips this reads the same conversation, and loses why the turn
        # was allowed to end.
        "skill-steps/decided",
        # Compaction (P4-03, G4; emitted by `compaction-summarize` in
        # `ph-stabilize`). The accounting for one landed summary: what it
        # shadowed, what that cost, and which model wrote it. Ignorable: the
        # summary that the model actually reads is the `user/message` appended
        # immediately after this, and that is a required surface event. So a
        # reader that skips this loses the metering and the provenance, and not
        # the conversation. It is kept adjacent to its replacement on purpose.
        # The adjacency lets a consumer price a shadowed range without keeping a
        # price for each node.
        "compaction/summarized",
        # An attempt that changed nothing, and why. Also ignorable: nothing in
        # the derivation moved, so a reader that skips it reads the same
        # conversation. It only cannot say why the session was never compacted.
        "compaction/declined",
        # The model-free half of the same row: long tool-call arguments elided
        # from retained history. The elision itself is an `assistant/message`
        # surface `replace`, so a reader that skips this still derives exactly
        # what the model saw. It loses the statement of the harness about what
        # it took out and why, and not the message. Ignorable for that reason.
        # It is adjacent to nothing: it names the seqs that it rewrote and does
        # not rely on position, because a truncation pass rewrites several at
        # the same time.
        "compaction/args-truncated",
        # Hard boundaries (P4-04, G5; emitted by `limits` in `ph-stabilize`).
        # Why a turn stopped, or why a tool stopped being called. Both are
        # ignorable. The model read the denial as a `tool/result`, and a
        # rejected step logs its own `turn/end{blocked}`. A reader that skips
        # these loses the *reason*, which is accounting and not the
        # conversation.
        "limits/exceeded",
        "limits/breaker-tripped",
        # Media that could not be sent (P7-01). It is appended when the route
        # that a request went to would not take an attachment: a MIME type that
        # is not accepted, a file over the limit, bytes that are no longer on
        # disk. So "the model was not shown this" is a fact in the log, and not
        # a line in a process log that nobody reads. Ignorable: the model saw
        # the pointer that the adapter put in its place. That pointer is in the
        # `user/message` that already carried the attachment. So a reader that
        # skips this loses the account of *why*, and not the conversation.
        "attachment/degraded",
        # Media that was sent and is larger than the route can use (P7-03). It
        # is not a degradation: the model saw the picture and the turn is
        # correct. The bill is what is wrong. The surplus pixels are uploaded
        # again on every request of the session and discarded at the far end.
        # Only a person can act on that, which is why it is a record here and
        # not a line in a process log. Ignorable for the same reason as its
        # sibling, and more so: nothing about the conversation changed.
        "attachment/oversized",
        # Bytes left this machine for the file API of a named provider (P7-03).
        # The *handle* is a cache under `$PH_CACHE` and is deliberately not
        # here. It is a prediction with an expiry, and an append-only log cannot
        # take one back. But that a file was uploaded is a fact, it is
        # privacy-relevant, and it is what a person who audits where their data
        # went looks for. Ignorable: the model reads the same attachment in
        # both cases. `attachment/uploading` opens it (`UPLOAD`, S9), on disk
        # before the bytes leave.
        "attachment/uploaded",
        "attachment/uploading",
        # The recipe of the context loader (P3-17): which corpus a session was
        # told about, and the digest of the sources that it was built from.
        # Ignorable: a reader that skips it loses the note, and not the
        # conversation.
        "context/loaded",
    }
)

IGNORABLE_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "credential/needed",
        "credential/supplied",
        "kernel/snapshot",
        "kernel/restored",
        "harness/refine-considered",
        "harness/probe",
        "harness/probed",
        "context/loaded",
        "offload/spilled",
        "offload/input-spilled",
        "skill-steps/budget",
        "skill-steps/decided",
        "step/retry",
        "llm/decided",
        "llm/decision-failed",
        "compaction/summarized",
        "compaction/declined",
        "compaction/args-truncated",
        "attachment/degraded",
        "attachment/oversized",
        "attachment/uploaded",
        "attachment/uploading",
        # Protocol bookkeeping that a reader can skip without misreading
        # anything else. The turn that it deduplicated is in the log either way.
        "client/command",
        "client/command-settled",
        # Supervisor bookkeeping (P5-04), ignorable on the same terms as
        # `limits/*`. A reader that skips these loses the *account* of why a
        # root paused, gave up or came back. That is accounting and not the
        # conversation, and it derives no message. They are listed so that a
        # build without the ladder can still seed a log that a P5-04 daemon
        # touched. Otherwise it would refuse a whole session over records that
        # carry no conversational content.
        "supervisor/retry",
        "supervisor/failed",
        "supervisor/recovered",
        "supervisor/passivated",
        "supervisor/unreachable",
        "supervisor/violated",
        # Scheduler bookkeeping. A reader that skips these loses *why* a turn
        # started at 3am. That is accounting and not the conversation: the
        # prompt that the tick delivered is a `user/message` either way.
        "schedule/created",
        "schedule/canceled",
        "schedule/tick",
        # What ran outside a tool call: a command of the user from the composer,
        # or the gate of a goal. Ignorable: a reader that skips these loses the
        # account of what ran beside the conversation, and never the
        # conversation. The model is not shown these records either way, which
        # is the purpose of the type.
        "shell/command",
        "shell/result",
        # What the model asked a person and what they said. Ignorable, unlike
        # `approval/*`. The decision of an approval can carry substituted
        # arguments and so changes what ran. The answer to a question reaches
        # the model only as the `ask_user` tool result, which is a `tool/result`
        # either way. A reader that skips these loses the account of the
        # exchange, and not the conversation. And a build that does not know
        # these types has no `ask_user` to pose them again with.
        "question/asked",
        "question/answered",
        # Goal bookkeeping. A reader that skips these loses why the loop went
        # round again. That is accounting: the turns themselves are all there.
        "goal/continued",
        "goal/gate",
        "limits/exceeded",
        "limits/breaker-tripped",
        "sandbox/denied",
        "workspace/restoring",
        "workspace/restored",
        "workspace/acquiring",
        "workspace/acquired",
        "workspace/disposed",
        "workspace/retained",
        "workspace/provisioned",
        # A version accepted for the next start, or declined when a start asked
        # (S6). When a reader skips these, the session starts on the base that
        # it has, or is asked one more time.
        "profile/adopted",
        "profile/declined",
        # A save changes nothing the session runs with (S7).
        "profile/saved",
        # The audit of what a skill said when it was read (S8).
        "skill/read",
    }
)
"""Types that a *different* build may skip without misreading the rest of the log.

Ignorability is a property of the type, and not of the call. "A reader that
does not recognize this `type` may skip it" is true of every record of the type
or of none. So it is declared here, beside the type, and `Session._append`
stamps it. A flag on each call would let two call sites disagree about one
type. A forgotten flag would make an older build refuse a log that it could
have read.

The set is deliberately limited. An unrecognized *required* event must still
refuse the seed, because skipping one can change how everything after it reads.
Only purely informational records belong here, such as kernel state, metering
and supervisor bookkeeping.
"""

AUDIT_ONLY_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        # The prompt snapshot that each request was built from. The conversation
        # is the messages, and this is how they were assembled.
        "request/header",
        # A model call made again (P1). The transcript shows the answer, and
        # `llm/retry` already tells a reader that a transient failure was
        # retried.
        "step/retry",
        # Postures. Each is a reading that the owning row contributes to the
        # footer, and not a moment in the conversation. The auditor wants each
        # move, and who made it.
        "approval/mode",
        "approval/policy",
        "permission/preset",
        "sandbox/mode",
        # The environment (S3 to S7): the base, each deviation, and what a
        # person decided about a named profile that moved. The command or the
        # modal that asked is already the row in the conversation.
        "profile/base",
        "profile/override",
        "profile/adopted",
        "profile/declined",
        "profile/override-cleared",
        "profile/withdrawn",
        "profile/refused",
        "profile/saved",
        # Which text of a skill was read, by hash (S8): the `skill` call is the row.
        "skill/read",
        "fs/observed",
        "session/end-seed",
        # Where the writes of an agent land and what happened to the tree: why
        # it was kept (P6-28), what did not reach it, its restore points, and a
        # restore that is about to run (S12). The settle of that restore,
        # `workspace/restored`, is the row.
        "workspace/acquiring",
        "workspace/acquired",
        "workspace/disposed",
        "workspace/retained",
        "workspace/provisioned",
        "workspace/checkpoint",
        "workspace/restoring",
        # A schedule created or canceled (P5-06): the tick is the row.
        "schedule/created",
        "schedule/canceled",
        # The accounting of the goal loop. `goal/set` and `goal/settled` bracket
        # the run as rows. A continuation and a gate result show as turns and
        # tool output.
        "goal/continued",
        "goal/gate",
        # Which client asked for a turn and how it ended (P5-02, P10-10). It is
        # provenance to an auditor, and deduplication that a reader never sees.
        "client/command",
        "client/command-settled",
        # The effect record of a keyed call (P10-12). The call renders as its
        # tool card, and the result carries the fact that it was answered from
        # the log.
        "tool/effect",
        "tool/effect-settled",
        # Bytes about to leave for a provider (S9): `attachment/uploaded` is the row.
        "attachment/uploading",
        # The probe by H1 and what it found (S21). The row of the refinement
        # says what it refused.
        "harness/probe",
        "harness/probed",
        # The nudge budget of a skill (D16). Each nudge is a row. The number
        # behind the nudges is for the auditor.
        "skill-steps/budget",
        # Why a turn was allowed to end with a procedure unfinished, or was not.
        # The nudge that it decided for is the row.
        "skill-steps/decided",
        # What a decision cost, and a decision that failed. Whoever asked
        # records what the decision was for.
        "llm/decided",
        "llm/decision-failed",
        # The accounting for a compaction. Its summary already draws a row, as
        # the replacement `user/message` that carries it. `compaction/declined`
        # is not here: a compaction that did not happen leaves no row of its
        # own.
        "compaction/summarized",
        # The own records of a sub-agent (Phase 11). They are in its own log and
        # never in the log of a root. Its spawn and its revocation are tool
        # cards of the root, and its log is read in the auditor's view.
        "subagent/admitted",
        "subagent/status",
        "subagent/returned",
        "subagent/deleted",
    }
)
"""Types that only an auditor reads. A front end draws no row in the
conversation for one. The auditor's view records it from its payload, unless
that view has a handler of its own for the type.

**This property is beside ignorability, and not inside it.**
`attachment/uploaded` is ignorable and draws a row. `request/header` is required
and draws none. Ignorability is a promise to other builds about skipping a
record. This property says who reads a record.

Not every type without a row is here. `step/start` and `kernel/snapshot` draw
no row and no record either, because another record carries them or they are
too fine-grained to read. Each front end lists those for its own reasons.
"""

EXPORTED_SESSION_EVENT_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        # How a turn ended: who canceled it, what a stop said, or a failure.
        "turn/end": ("reason",),
        # A model call failed in a way worth another try, and which failure it was.
        "llm/retry": ("code",),
        # A route would not decide: which failure, and its sentence.
        "llm/decision-failed": ("code", "reason"),
        "compaction/declined": ("code", "reason"),
        "harness/refine-considered": ("reason",),
        "goal/settled": ("outcome", "detail"),
        "subagent/deleted": ("reason",),
        "sandbox/denied": ("kind", "message"),
        "profile/refused": ("reason",),
        "profile/withdrawn": ("reason",),
        # A root's recovery ladder, and what a supervisor found broken around it.
        "supervisor/retry": ("reason",),
        "supervisor/failed": ("reason",),
        "supervisor/unreachable": ("reason",),
        "supervisor/violated": ("violations",),
    }
)
"""The fields of the payload of a type that telemetry ships, each as `event.<field>`.

**The fields say why the work failed, was refused or stopped**, and nothing
else. An operator filters on that without the log at hand. Everything else
stays in the log. That is why a record carried no data at all before. It is
also why a type is listed here, and telemetry does not ship every field named
`reason`: the `reason` of `request/header` says which prompt snapshot it is,
and much of a payload is what the model or a person wrote.

Each name is one that the writers of its type write. `test_log_writers` reads
that from the source.
"""


# ------------------------------------------------------------- intent pairs --


@dataclass(frozen=True, slots=True)
class IntentPair:
    """What settles the intent that one type opens, and the leaf that declares its kind."""

    settled: str
    leaf: str
    """The module whose `IntentKind(...)` declares the pair: the `kinds` of a package."""


INTENT_PAIRS: Mapping[str, IntentPair] = MappingProxyType(
    {
        "approval/asked": IntentPair("approval/decided", "ph.session.kinds"),
        "credential/needed": IntentPair("credential/supplied", "ph.session.kinds"),
        "client/command": IntentPair("client/command-settled", "ph_app.kinds"),
        "command/run": IntentPair("command/done", "ph.session.kinds"),
        "harness/probe": IntentPair("harness/probed", "ph_rlm.kinds"),
        "attachment/uploading": IntentPair("attachment/uploaded", "ph.session.kinds"),
        "question/asked": IntentPair("question/answered", "ph.session.kinds"),
        "shell/command": IntentPair("shell/result", "ph.session.kinds"),
        "tool/code-dispatch-start": IntentPair("tool/code-dispatch", "ph.session.kinds"),
        "tool/effect": IntentPair("tool/effect-settled", "ph.session.kinds"),
        "workspace/restoring": IntentPair("workspace/restored", "ph.session.kinds"),
    }
)
"""Every type that opens an intent, by the type that opens it (T4).

**This table lets repair notice a kind that it was never given.** A kind is
declared by importing its leaf, and a process that never imported a package has
none of the kinds of that package. Before this table, repair settled what was
declared and silently left the rest open. An open `client/command`, resumed by
a process without `ph_app`, read as a command that was still running forever.
With the table, repair refuses such a resume and names the leaf
(`UndeclaredIntentError`).

The table is here, beside the vocabulary, for the reason that decision 7 gives.
The reader of ph-core already knows every type that a package writes. So it can
know which of them open an intent without importing the package.
`test_intent_kinds.py` holds this table equal to the kinds that the leaves
declare, in both directions: every `IntentKind(...)` in shipped code, its
settling type and its module.

Not enforced: an intent pair on a *declared* log type (`declare_log_type`). No
such pair ships. One would belong to a package, and a log that carries a
required type that this reader does not know is already refused at seed.
"""


# ---------------------------------------------------------- writers of record --


def _with_pairs(appended: Mapping[str, frozenset[str]]) -> Mapping[str, frozenset[str]]:
    """`appended`, plus each intent pair credited to the leaf that declares its kind.
    So `INTENT_PAIRS` is the one statement of which leaf writes which pair."""
    table = dict(appended)
    for opened, pair in INTENT_PAIRS.items():
        table[pair.leaf] = table.get(pair.leaf, frozenset()) | {opened, pair.settled}
    return table


_WRITTEN_BY: Mapping[str, frozenset[str]] = _with_pairs(
    {
        "ph.agent.inbox": frozenset({"agent/inbox/spliced"}),
        "ph.agent_loop.driver": frozenset(
            {
                "assistant/chunk",
                "assistant/message",
                "request/context",
                "request/header",
                "step/end",
                "step/retry",
                "step/start",
                "turn/end",
                "turn/start",
                "user/message",
            }
        ),
        "ph.llm.adapter": frozenset({"llm/decided", "llm/decision-failed"}),
        "ph.llm.media": frozenset({"attachment/degraded", "attachment/oversized"}),
        "ph.llm.retry": frozenset({"llm/retry"}),
        "ph.persistence.resume": frozenset({"session/resumed"}),
        "ph.seams.approval": frozenset({"approval/policy"}),
        "ph.seams.fs": frozenset({"fs/observed"}),
        "ph.seams.goals": frozenset({"goal/continued", "goal/gate", "goal/set", "goal/settled"}),
        "ph.seams.permission_presets": frozenset({"permission/preset"}),
        "ph.seams.sandbox": frozenset({"sandbox/denied", "sandbox/mode"}),
        "ph.seams.skills": frozenset({"skill/read"}),
        "ph.seams.schedule": frozenset({"schedule/canceled", "schedule/created", "schedule/tick"}),
        # Every record of a child, in the own log of the child, and from nowhere
        # else. The seam writes the admission. A provider reports its child
        # through the functions of that module (`record_waiting`,
        # `record_started`, `record_ended`, `record_deleted`, `record_returned`),
        # and each function keeps the rules of its record.
        "ph.seams.subagents": frozenset(
            {"subagent/admitted", "subagent/deleted", "subagent/returned", "subagent/status"}
        ),
        "ph.seams.workspace": frozenset(
            {
                "workspace/acquired",
                "workspace/acquiring",
                "workspace/checkpoint",
                "workspace/disposed",
                "workspace/provisioned",
                "workspace/retained",
            }
        ),
        "ph.session_profile": frozenset(
            {
                "profile/adopted",
                "profile/base",
                "profile/declined",
                "profile/override",
                "profile/override-cleared",
                "profile/refused",
                "profile/saved",
                "profile/withdrawn",
            }
        ),
        # Every surface replacement, from the one module that keeps their rules.
        # A substitution is a `user/message`. A rewrite in place is the
        # `assistant/message` or the `tool/result` that it rewrites. Compaction,
        # input-offload and ph-clm write a replacement through this module.
        # Each of them writes only its own records itself.
        "ph.session.revise": frozenset({"assistant/message", "tool/result", "user/message"}),
        "ph.session.store": frozenset({"session/end-seed", "session/segmented"}),
        "ph.tools.batch": frozenset({"tool/call", "tool/result"}),
        "ph_app.daemon.supervisor": frozenset(
            {
                "supervisor/failed",
                "supervisor/passivated",
                "supervisor/recovered",
                "supervisor/retry",
                "supervisor/unreachable",
                "supervisor/violated",
            }
        ),
        "ph_rlm.context_loader": frozenset({"context/loaded"}),
        "ph_rlm.harness": frozenset({"harness/refine-considered"}),
        "ph_rlm.harness.service": frozenset({"harness/refined"}),
        "ph_rlm.snapshot": frozenset({"kernel/restored", "kernel/snapshot"}),
        # Compaction rewrites the history of the model (a summary, an elided
        # call, a clipped result) through `ph.session.revise`, and records the
        # reason here.
        "ph_stabilize.compaction": frozenset(
            {"compaction/args-truncated", "compaction/declined", "compaction/summarized"}
        ),
        "ph_stabilize.hitl": frozenset({"approval/mode"}),
        # The preview of an offloaded paste goes through `ph.session.revise` too.
        "ph_stabilize.input_offload": frozenset({"offload/input-spilled"}),
        "ph_stabilize.limits": frozenset({"limits/breaker-tripped", "limits/exceeded"}),
        "ph_stabilize.offload": frozenset({"offload/spilled"}),
        # Two writers of one list on purpose: a procedure that a skill declares
        # is a todo list, and both rows mean "the list is now this".
        "ph_stabilize.skill_steps": frozenset(
            {"skill-steps/budget", "skill-steps/decided", "todo/write"}
        ),
        "ph_stabilize.todo": frozenset({"todo/write"}),
    }
)
"""Which shipped module appends which of the types of ph-core (P10-01). The rows
of the leaves come from `INTENT_PAIRS`: the pair of each kind, written through
`ctx.intents` (P10-08, T4).

**The table says who may write each type.** The refusal of an unknown type
(F11) cannot say that. It is the difference between "deliberately shared" and
"nobody noticed". The table is grouped by module, so that each shared type
carries its reason where it is shared.

**The writer of a module is minted from this table** (T6).
`log_writer(__name__)` hands a module a writer for exactly its row, and the
writer refuses any other type at the write (F12). So this table is enforced at
runtime, and not only read by a test. `test_log_writers.py` holds the table to
the code in both directions. A module that writes a type that its row does not
grant fails. A grant that nothing writes any more fails too.

**The table is lexical, not dynamic.** It names the module whose writer writes,
and not the row that happened to be running. `permission-presets` changes the
sandbox posture by calling `SandboxSeam.set_mode`, and the write belongs to the
seam.

**A pair written through `ctx.intents` belongs to the leaf that declares its
kind** (`IntentKind(opened=…, settled=…)` in the `kinds` of a package, T4). It
does not belong to the journal, which writes it through the writer of the leaf
(`IntentKind.writer`). It does not belong to the seam that holds the kind
either. The declaration is the one place where both types are named, and a kind
whose writer does not own both is refused. Not enforced: which modules may
*use* a declared kind. The walk sees the declaration, and not who passes it to
the journal.

Repair is not a writer here. Its closers are synthesized onto a seed
(`interrupted_turn_closers`) and never appended.
"""

WRITERS: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        kind: frozenset(module for module, kinds in _WRITTEN_BY.items() if kind in kinds)
        for kind in sorted(KNOWN_SESSION_EVENT_TYPES)
    }
)
"""`_WRITTEN_BY` inverted: each ph-core type, and the modules that append it."""


# ------------------------------------------------------------- declarations --


class UnknownEventTypeError(ValueError):
    """An append named a type this build would refuse to read back (F11)."""


class LogTypeError(ValueError):
    """A declaration that would make two statements of one type disagree."""


@dataclass(frozen=True, slots=True)
class LogTypeDeclaration:
    """A session event type that belongs to a package outside ph-core."""

    name: str
    owner: str
    """The module that appends the type. It is the producer of record of the type,
    as for a bus event. It is usually the declaring module. The leaf of a package
    may declare for the modules that write."""
    ignorable: bool
    audit_only: bool
    """Only an auditor reads it, as for `AUDIT_ONLY_SESSION_EVENT_TYPES`."""
    exported: tuple[str, ...] = ()
    """The fields telemetry ships, as for `EXPORTED_SESSION_EVENT_FIELDS`."""


_NAME = re.compile(r"[a-z][a-z0-9-]*(?:/[a-z0-9-]+)+")
"""`namespace/type`: the shape by which `ph.selectors` addresses the log vocabulary."""

_DECLARED: dict[str, LogTypeDeclaration] = {}
"""What packages outside ph-core declared. This is the one mutable table. A
question about known or ignorable goes to the frozen sets first, and then to
this table."""


def declare_log_type(
    name: str, *, owner: str, ignorable: bool, audit_only: bool, exported: tuple[str, ...] = ()
) -> LogTypeDeclaration:
    """Add a type to the vocabulary that this build writes and reads. Call at import.

    This is for a package outside ph-core. The frozen set cannot list the types
    of such a package without ph-core learning about the package. The own types
    of ph-core stay in `KNOWN_SESSION_EVENT_TYPES`, beside the reasoning that
    each one carries.

    **Ignorable is a promise to other builds**, as it is for the core set. A
    reader without the declaring package skips an ignorable record and refuses
    a required one, because skipping a required record can change how the rest
    of the log reads. So `ignorable=False` means that a log that carries the
    type opens only where the package is installed. That is the purpose of
    required.

    **Audit-only says who reads the type**, as `AUDIT_ONLY_SESSION_EVENT_TYPES`
    does for the types of ph-core: no conversation row, and a record in the
    auditor's view, derived from the payload. A front end can have no handler
    of its own for a type that it cannot import. So no front end shows a
    declared type that is not audit-only.

    **Exported names the fields that telemetry ships**, as
    `EXPORTED_SESSION_EVENT_FIELDS` does for the types of ph-core: the fields
    that say why the work failed, was refused or stopped.

    A declaration is refused, the way `EventRegistry.declare` refuses a bus
    event, for:

    * a name that ph-core already owns. One type has one home.
    * a second owner, or the same owner changing the ignorability of the type,
      who reads it, or what ships. Two statements of one type that disagree are
      a log that two builds read in two ways.
    * a name that is not `namespace/type`.

    Declaring the same thing twice returns the first declaration, so a module
    that is imported twice does not refuse itself.
    """
    if not _NAME.fullmatch(name):
        raise LogTypeError(f'"{name}" is not a log type name; it must be namespace/type')
    if not owner:
        raise LogTypeError(f'"{name}" needs an owner: the module that appends it')
    if name in KNOWN_SESSION_EVENT_TYPES:
        raise LogTypeError(f'"{name}" is a ph-core type; it cannot be declared again')
    existing = _DECLARED.get(name)
    if existing is not None:
        if existing.owner != owner:
            raise LogTypeError(
                f'"{name}" is already declared by "{existing.owner}"; '
                f'"{owner}" cannot declare it as well'
            )
        if existing.ignorable != ignorable:
            raise LogTypeError(
                f'"{name}" is already declared ignorable={existing.ignorable}; '
                "a type's ignorability cannot change"
            )
        if existing.audit_only != audit_only:
            raise LogTypeError(
                f'"{name}" is already declared audit_only={existing.audit_only}; '
                "who reads a type cannot change"
            )
        if existing.exported != exported:
            raise LogTypeError(
                f'"{name}" is already declared exported={existing.exported}; '
                "what a type ships cannot change"
            )
        return existing
    declaration = LogTypeDeclaration(
        name=name, owner=owner, ignorable=ignorable, audit_only=audit_only, exported=exported
    )
    _DECLARED[name] = declaration
    return declaration


def written_by(module: str) -> frozenset[str]:
    """The ph-core types that `module` is a writer of record for. Its writer may write these."""
    return _WRITTEN_BY.get(module, frozenset())


def declared_owner(event_type: str) -> str | None:
    """The module that declared `event_type` with `declare_log_type`, if one did."""
    declared = _DECLARED.get(event_type)
    return None if declared is None else declared.owner


def is_known(event_type: str) -> bool:
    """Whether this build writes and reads `event_type`: ph-core's, or declared."""
    return event_type in KNOWN_SESSION_EVENT_TYPES or event_type in _DECLARED


def known_log_types() -> frozenset[str]:
    """The whole vocabulary that this process writes and reads: the types of ph-core,
    and every type that a package has declared so far. The completeness check of
    a front end compares its tables with this."""
    return KNOWN_SESSION_EVENT_TYPES | frozenset(_DECLARED)


def is_audit_only(event_type: str) -> bool:
    """Whether only an auditor reads `event_type`: no conversation row, and a record
    in the auditor's view."""
    if event_type in AUDIT_ONLY_SESSION_EVENT_TYPES:
        return True
    declared = _DECLARED.get(event_type)
    return declared is not None and declared.audit_only


def exported_fields(event_type: str) -> tuple[str, ...]:
    """The fields of `event_type`'s payload that telemetry ships: ph-core's, or declared."""
    core = EXPORTED_SESSION_EVENT_FIELDS.get(event_type)
    if core is not None:
        return core
    declared = _DECLARED.get(event_type)
    return () if declared is None else declared.exported


def is_ignorable(event_type: str) -> bool:
    """Whether a build without this type may skip it. `Session._append` stamps it."""
    if event_type in IGNORABLE_SESSION_EVENT_TYPES:
        return True
    declared = _DECLARED.get(event_type)
    return declared is not None and declared.ignorable
