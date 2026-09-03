"""The event vocabulary this build understands.

The persistence read path refuses to interpret a log containing a type outside
this set **unless** the event carries `ignorable`. Such a log was likely written
by a newer harness, and an unrecognized *required* event may change how the rest
of the log is read — so silently skipping it would reconstruct a wrong session
rather than an incomplete one.

Kept beside the code that appends these types, and checked by a test that walks
every `append(` call site in `ph-core`: a type that ships without an entry here
would be a log this build could write and then refuse to read. That walker sees
only ph-core — a producer in another package (the subagent providers,
ph-stabilize's `tool-todo`) owes the same proof through its own bundle's tests,
which is the deal the per-type comments below record.

@module ph.session.known_event_types
"""

from __future__ import annotations

__all__ = ["IGNORABLE_SESSION_EVENT_TYPES", "KNOWN_SESSION_EVENT_TYPES"]

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
        # A session continued in a fresh file (§7 step 6). The parent's own
        # terminal record, naming the log that carries on — the forward half of
        # a link whose backward half is the child's `parent_session` header.
        # Needed because a segment and a branch are structurally identical: both
        # are a fork, and only this says which one a person is looking at.
        "session/segmented",
        # A mutating command a protocol client asked for, recorded so asking
        # twice is safe across a restart (P5-02). Not a slash `command/*`.
        "client/command",
        # The supervisor's retry ladder (P5-04): a crashed root task being run
        # again, the ladder giving up, and a retry that worked. In the log
        # rather than in supervisor memory, because "this root tried three times
        # and stopped" must survive the process that decided it — and a root
        # resumed mid-ladder that forgot would start the count over and retry
        # forever.
        #
        # `supervisor/*` and not `agent/*`: `ph.agent` owns that namespace (its
        # registry declares `agent/status`, `agent/error` and `agent/inbox/*`
        # with `owner="ph.agent"`), and these are the supervisor's records
        # *about* an agent rather than the agent's own. `agent/failed` also sat
        # one letter from `agent/error`, which means something else entirely.
        "supervisor/retry",
        "supervisor/failed",
        "supervisor/recovered",
        # A root released for being idle (P5-05). Its own record because the
        # alternative is an unexplained gap: a transcript that stops for three
        # days and resumes reads as a crash, and the `session/resumed` on the
        # way back says only that something was resumed, not that nothing had
        # gone wrong.
        "supervisor/passivated",
        # The daemon can no longer be reached, written into every root it is
        # running (P5-11). Its reader is nobody who is connected: by the time
        # this is appended the socket is gone, so a client can neither be
        # notified nor connect to ask. The transcript is the surface that
        # survives — a run that went quiet at 18:04 says here that its
        # supervisor lost its socket at 18:04, and names the fix.
        "supervisor/unreachable",
        # Future work a root will do, and the claim that it is doing it (P5-06).
        # `schedule/tick` is appended *before* delivery: a tick recorded and
        # lost to a crash costs one skipped run, while a tick delivered and lost
        # costs a repeated one — and repeating a scheduled prompt bills twice
        # and confuses the transcript.
        "schedule/created",
        "schedule/cancelled",
        "schedule/tick",
        "schedule/heartbeat",
        # An autonomous run's objective, its spend, its quality gates and how it
        # ended (P5-07). `goal/settled` names *which* budget stopped it, because
        # "it stopped" and "it ran out of turns" are different things to whoever
        # reads the trace, and only one of them suggests raising the budget.
        "goal/set",
        "goal/continued",
        "goal/gate",
        "goal/settled",
        "step/end",
        "step/start",
        "tool/call",
        "tool/result",
        "turn/end",
        "turn/start",
        "user/message",
        # policy and human-in-the-loop
        "approval/asked",
        "approval/decided",
        "approval/policy",
        # The model asking the person something that is not an approval (P7-09,
        # emitted by `tool-ask-user`). Two events for the reason approvals have
        # two: the ask is appended before the waterfall runs, so a crash while
        # somebody was deciding leaves the question in the log rather than losing
        # it — which is what `pending_questions` folds.
        #
        # Only a question that was actually *delivered* appears at all. An
        # unattended ask resolves to "no answer" without appending, so a log
        # never claims a person was asked and declined when no person was there.
        "question/asked",
        "question/answered",
        # A shell command a *person* ran from the composer, and what it printed
        # (P7-10, emitted by `ph-app`'s `session/shell`). Two events for the
        # reason `tool/call`/`tool/result` are two: the command is appended
        # before it runs, so one that hangs — or that takes the daemon down with
        # it — still shows in the log what was started, which is exactly the
        # command worth knowing about.
        #
        # Invisible to the model by not being one of the three surface-eligible
        # types — see `SURFACE_EVENT_TYPES`, which is where that mechanism is
        # defined and explained.
        "shell/command",
        "shell/result",
        # The gating *posture* (P4-05; emitted by ph-stabilize's `hitl`). Its own
        # type rather than a field on `approval/policy`, which `permission-presets`
        # also writes: one last-write-wins fold with two writers, one of which
        # does not know the field exists, is a posture that silently reverts when
        # someone switches preset. Listed here because the reader that refuses an
        # unknown type is ph-core's.
        "approval/mode",
        "command/done",
        "command/run",
        "permission/preset",
        "sandbox/mode",
        # capability observations
        "fs/observed",
        # Where an agent's writes land (D21, P4-07). A *pair*: an `acquired` with
        # no `disposed` is how a leaked workspace is detected at session open,
        # which only works if both halves are in the vocabulary.
        "workspace/acquired",
        "workspace/disposed",
        # A tree marked as evidence, written when the mark is made rather than
        # only on the closing half (P6-28). A retention is decided because a run
        # went wrong, and the most complete way for one to go wrong writes no
        # `disposed` at all — so the mark has to be durable before that.
        "workspace/retained",
        # Materials that did not reach a fresh workspace (E14). Appended only
        # when something failed — a silent success is the ordinary case, and one
        # event per agent saying "nothing went wrong" is noise in a log whose
        # whole value is that everything in it happened.
        "workspace/provisioned",
        # The per-run restore point (E7). **Not** ignorable: a reader that
        # skipped it would offer `/revert` a session with no restore points
        # while the refs are sitting in the repository, which is a build
        # misreading recoverable state as unrecoverable.
        "workspace/checkpoint",
        # retry
        "llm/retry",
        # Code Mode dispatch records (log-only; see ph.tools.code_mode)
        "tool/code-dispatch",
        "tool/code-dispatch-start",
        # Persistent-kernel state (D17; emitted by ph-rlm's snapshot policy).
        # These are what make `persistence: "namespace"` admissible at all: the
        # seam takes the provider's promise at registration, and these events are
        # the promise being kept. Listed here rather than in ph-rlm because the
        # *reader* is ph-core — a log carrying a type this build does not know is
        # refused on the seed path, so the vocabulary has one home.
        "kernel/snapshot",
        "kernel/restored",
        # Delegation (P3-11; emitted by a `ctx.subagents` provider). Named for
        # the *seam*, not for the bundle that ships the first provider: the
        # reader is ph-core and ph-app, neither of which depends on ph-rlm, and a
        # second provider emitting `rlm/…` would be lying about its identity in
        # an append-only log. The provider names itself in the payload instead.
        # The roster a parent reads back *is* this fold — there is no side
        # table — so the admission and the tombstone are required reading, while
        # status and usage attribution are informational.
        "subagent/admitted",
        "subagent/deleted",
        "subagent/status",
        "subagent/usage-attributed",
        # The Continual Harness (P3-16). Its state *is* this fold — there is no
        # side file to fall back on — so a reader that skipped one would show a
        # harness the session does not have. A rollback is a refinement with
        # `rollbackOf` set, so there is one type rather than two fold cases that
        # have to agree about one operation.
        "harness/refined",
        # Considering and declining. Unlike the refinement itself this changes no
        # state — it is what advances the auto-refine cooldown (H7) — so it is
        # ignorable: a reader that skips it gets one extra review pass, not a
        # harness the session does not have.
        "harness/refine-considered",
        # Offloading (P4-02, G2; emitted by `ph-stabilize`'s `tool-result-offload`).
        # The forwarding address for a result the model was handed a preview of.
        # Ignorable: `tool/result` already carries what the model saw, so a
        # reader that skips this loses the way back to the original, not the
        # conversation — the same test the `context/loaded` recipe passes.
        "offload/spilled",
        # And the input side (G3): where a pasted message went. Ignorable for
        # the same reason — the surface `replace` beside it is what the model
        # reads, and that is a required `user/message`, so a reader skipping
        # this loses the forwarding address and nothing else.
        "offload/input-spilled",
        # Planning (P4-01, G1; emitted by `ph-stabilize`'s `tool-todo`). The
        # list *is* this fold — there is no side table — and it reaches the
        # model through a prompt context, so a reader that skipped it would
        # assemble a different prompt than the session had. Required, not
        # ignorable: the difference is model-visible.
        "todo/write",
        # Compaction (P4-03, G4; emitted by `ph-stabilize`'s `compaction-summarize`).
        # The accounting for one landed summary: what it shadowed, what that
        # cost, and which model wrote it. Ignorable — the summary the model
        # actually reads is the `user/message` appended immediately after this,
        # a required surface event, so a reader that skips this loses the
        # metering and the provenance rather than the conversation. Kept
        # adjacent to its replacement on purpose: that adjacency is what lets a
        # consumer price a shadowed range without retaining per-node prices.
        "compaction/summarized",
        # An attempt that changed nothing, and why. Also ignorable: nothing in
        # the derivation moved, so a reader that skips it reads the same
        # conversation — it just cannot say why the session was never compacted.
        "compaction/declined",
        # The model-free half of the same row: long tool-call arguments elided
        # from retained history. The elision itself is an `assistant/message`
        # surface `replace`, so a reader that skips this still derives exactly
        # what the model saw — it loses the harness's statement of what it took
        # out and why, not the message. Ignorable for that reason, and adjacent
        # to nothing: it names the seqs it rewrote rather than relying on
        # position, because a truncation pass rewrites several at once.
        "compaction/args-truncated",
        # Hard boundaries (P4-04, G5; emitted by `ph-stabilize`'s `limits`).
        # Why a turn stopped, or why a tool stopped being called. Both ignorable:
        # the model read the denial as a `tool/result`, and a rejected step logs
        # its own `turn/end{blocked}` — a reader that skips these loses the
        # *reason*, which is accounting, not the conversation.
        "limits/exceeded",
        "limits/breaker-tripped",
        # Media that could not be sent (P7-01). Appended when the route a
        # request went to would not take an attachment — an unaccepted MIME, an
        # over-limit file, bytes no longer on disk — so that "the model was not
        # shown this" is a fact in the log rather than a line in a process log
        # nobody reads. Ignorable: the model saw the pointer the adapter put in
        # its place, and that rides the `user/message` it was already carried
        # by, so a reader skipping this loses the account of *why*, not the
        # conversation.
        "attachment/degraded",
        # Media that was sent and is larger than the route can use (P7-03).
        # Not a degradation: the model saw the picture and the turn is correct.
        # What is wrong is the bill — the surplus pixels are re-uploaded on every
        # request of the session and discarded at the far end — and a person is
        # the only one who can act on it, which is why it is a record here rather
        # than a line in a process log. Ignorable for the same reason as its
        # sibling, and more so: nothing about the conversation changed.
        "attachment/oversized",
        # Bytes left this machine for a named provider's file API (P7-03). The
        # *handle* is a cache under `$PH_CACHE` and deliberately not here — it is
        # a prediction with an expiry, and an append-only log cannot take one
        # back — but that a file was uploaded is a fact, it is privacy-relevant,
        # and it is what a person auditing where their data went comes for.
        # Ignorable: the model reads the same attachment either way.
        "attachment/uploaded",
        # The context loader's recipe (P3-17): which corpus a session was told
        # about, and the digest of the sources it was built from. Ignorable — a
        # reader that skips it loses the note, not the conversation.
        "context/loaded",
    }
)

IGNORABLE_SESSION_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "kernel/snapshot",
        "kernel/restored",
        # Status mirroring and usage bookkeeping. `subagent/admitted` and
        # `subagent/deleted` are deliberately NOT here: they are the roster's
        # only record of a child existing and of it being revoked, and a reader
        # that skipped either would show a parent the wrong family.
        "subagent/status",
        "subagent/usage-attributed",
        "harness/refine-considered",
        "context/loaded",
        "offload/spilled",
        "offload/input-spilled",
        "compaction/summarized",
        "compaction/declined",
        "compaction/args-truncated",
        "attachment/degraded",
        "attachment/oversized",
        "attachment/uploaded",
        # Protocol bookkeeping a reader can skip without misreading anything
        # else: the turn it deduplicated is in the log either way.
        "client/command",
        # Supervisor bookkeeping (P5-04), ignorable on the same terms as
        # `limits/*`: a reader skipping these loses the *account* of why a root
        # paused, gave up or came back — which is accounting, not the
        # conversation, and derives no message. Listed so a build without the
        # ladder can still seed a log a P5-04 daemon touched, rather than
        # hard-refusing a whole session over two records carrying no
        # conversational content.
        "supervisor/retry",
        "supervisor/failed",
        "supervisor/recovered",
        "supervisor/passivated",
        "supervisor/unreachable",
        # Scheduler bookkeeping: a reader skipping these loses *why* a turn
        # started at 3am, which is accounting rather than the conversation —
        # the prompt the tick delivered is a `user/message` either way.
        "schedule/created",
        "schedule/cancelled",
        "schedule/tick",
        "schedule/heartbeat",
        # What a person poked at from the composer. Ignorable: a reader skipping
        # these loses the account of what someone ran beside the conversation,
        # and never the conversation — the model was not shown it either way,
        # which is the whole point of the type.
        "shell/command",
        "shell/result",
        # What the model asked a person and what they said. Ignorable, unlike
        # `approval/*`: an approval's decision can carry substituted arguments
        # and so changes what ran, while a question's answer reaches the model
        # only as the `ask_user` tool result — which is a `tool/result` either
        # way. A reader that skips these loses the account of the exchange, not
        # the conversation. And a build that does not know these types has no
        # `ask_user` to re-pose them with.
        "question/asked",
        "question/answered",
        # Goal bookkeeping: a reader skipping these loses why the loop went
        # round again, which is accounting — the turns themselves are all there.
        "goal/continued",
        "goal/gate",
        "limits/exceeded",
        "limits/breaker-tripped",
        "workspace/acquired",
        "workspace/disposed",
        "workspace/retained",
        "workspace/provisioned",
    }
)
"""Types a *different* build may skip without misreading the rest of the log.

Ignorability is a property of the type, not of the call — "a reader that does
not recognize this `type` may skip it" is true of every record of the type or
none — so it is declared here, beside the type, and `Session.append` stamps it.
A per-call flag would let two call sites disagree about one type, and a
forgotten flag is an older build refusing a log it could have read.

The set is deliberately small: an unrecognized *required* event must still
refuse the seed, because skipping one can change how everything after it reads.
Only purely informational records — kernel state, subagent status, usage
attribution — belong here.
"""
