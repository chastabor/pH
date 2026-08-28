"""`session/event` in, transcript out.

The load-bearing decision (P2-01's gate): **the transcript is rebuilt from
`session.events`, never from `derive_messages()`.** The derivation is the *model's*
view, and compaction deliberately shadows what it replaced — so rebuilding a
resumed session from it would erase conversation the person sitting there
already read. The adapter therefore reads the log, and marks a compacted range
rather than dropping it.

Two modes, and the difference matters:

* **replay** (a resumed seed) uses `assistant/message`, the authoritative
  assembled text, and ignores `assistant/chunk` entirely;
* **live** streams the chunks and lets the message finalize them.

Feeding chunks on replay would rebuild a message the log already has, one delta
at a time, and any chunk lost to a crash would leave the transcript disagreeing
with the log.

`HANDLERS` is the closed list of what renders. Together with `RECORDLESS` — the
known types that are an auditor's records rather than a reader's (P3-24) — it
covers the log's whole vocabulary, and a test holds that equality so a new event
type cannot go silently unrendered.

@module ph_app.tui.adapter
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ph.seams.subagents import downgrade_text, fold_subagent_event
from ph.session import (
    Session,
    SessionEvent,
    is_in_place_rewrite,
    is_replacement_surface_event,
    thaw_json,
)
from ph.session.request_header import parse_request_context
from ph.text import count_of
from ph.tools import ToolResult, ToolResultView, parse_arguments

from .state import ChatItem, ItemRole, ToolCard, TuiState
from .wire import first, media_labels, obj, one_line, seq, text_of_wire

__all__ = ["HANDLERS", "RECORDLESS", "TuiEventAdapter"]


@dataclass(slots=True)
class TuiEventAdapter:
    """Folds session events into a `TuiState`."""

    state: TuiState = field(default_factory=TuiState)
    tools: Any = None
    """`ctx.tools`, when available, so a card can use the tool's own
    `present_call`/`present_result`. Absent is fine: the generic card renders
    from the log alone."""
    _fragment: int = 0

    # --------------------------------------------------------------- entry --

    def replay(self, session: Session) -> TuiState:
        """Rebuild the whole transcript from a stored log, in place."""
        self.state.reset()
        self._fragment = 0
        for event in session.events:
            self.apply(event, live=False)
        return self.state

    def apply(self, event: SessionEvent, *, live: bool = True) -> None:
        handler = HANDLERS.get(event.type)
        if handler is not None:
            handler(self, event, live)

    # ---------------------------------------------------------------- rows --

    def _row(
        self,
        prefix: str,
        role: ItemRole,
        text: str,
        event: SessionEvent,
        *,
        turn: int | None = None,
    ) -> ChatItem:
        """One settled row keyed by the event that produced it."""
        return self.state.add(
            ChatItem(
                key=f"{prefix}-{event.seq}",
                role=role,
                text=text,
                turn=self.state.turn if turn is None else turn,
                seq=event.seq,
            )
        )

    # ------------------------------------------------------------ messages --

    def _on_user_message(self, event: SessionEvent, live: bool) -> None:
        source = obj(event.data.get("source"))
        kind = source.get("kind")
        content = event.data.get("content")
        text = text_of_wire(content)
        # Attached files are part of what the person said. Left out, the row
        # would show a bare "what is this?" beside a model answering about an
        # image the transcript never mentions.
        attached = media_labels(content)
        if attached:
            text = "\n".join([text, *(f"[{label}]" for label in attached)]).strip()
        if is_replacement_surface_event(event):
            # Whatever its cause, a replacement shadows the rows it cites: they
            # stay above it, dimmed.
            self._mark_shadowed(event.source_event_seqs or ())
            # But only compaction is compaction. `input-offload` (P4-02) also
            # substitutes on the surface, and calling its preview "history
            # compacted" would tell the reader their conversation was summarized
            # when a paste was relocated. The discriminator is the log's own
            # attribution: a compaction summary declares `form: compaction`
            # (P4-03), which is a claim about the surface and not a colour.
            if source.get("form") == "compaction":
                self._row("compaction", "compaction", text or "(history compacted)", event)
                return
        self._row("msg", "context" if kind == "plugin" else "user", text, event)

    def _mark_shadowed(self, shadowed: tuple[int, ...]) -> None:
        """Mark the rows a replacement stands in for.

        They stay; they are just no longer what the model sees. Named for the
        mechanism rather than for compaction, because compaction is one cause
        of it and no longer the only one.
        """
        targets = set(shadowed)
        for item in self.state.items:
            if item.seq in targets:
                item.shadowed = True

    def _on_assistant_chunk(self, event: SessionEvent, live: bool) -> None:
        if not live:
            # The assembled `assistant/message` is authoritative on replay.
            return
        chunk = obj(event.data.get("chunk"))
        turn, step = int(event.data.get("turn", 0)), int(event.data.get("step", 0))
        kind = chunk.get("type")
        if kind == "text-delta":
            self._append_stream(turn, step, "assistant", str(chunk.get("text", "")), event.seq)
        elif kind == "reasoning-delta":
            self._append_stream(turn, step, "thinking", str(chunk.get("text", "")), event.seq)

    def _append_stream(self, turn: int, step: int, role: ItemRole, text: str, seq: int) -> None:
        if not text:
            return
        item = self.state.streaming_item(turn, step)
        if item is None or item.role != role:
            self._fragment += 1
            item = ChatItem(
                key=f"stream-{turn}-{step}-{self._fragment}",
                role=role,
                streaming=True,
                turn=turn,
                seq=seq,
            )
            self.state.begin_streaming(turn, step, item)
        item.text += text

    def _on_assistant_message(self, event: SessionEvent, live: bool) -> None:
        if is_in_place_rewrite(event):
            # A node replacing *itself* — argument truncation (P4-03) eliding a
            # long tool-call argument. The text is the model's own and is already
            # on screen, so rendering the replacement would show the assistant
            # speaking twice; the row it stands for is deliberately not dimmed,
            # because the message is still what the model sees. Falling through
            # would also let `_count_usage` reset the footer to an old turn's.
            #
            # Keyed on the shape rather than on "any replacement": a
            # *substitution* of an assistant message is a different event that
            # does remove conversation, and blanket-returning would drop it
            # silently — the mechanism-not-cause mistake this file already fixed
            # once, fifty lines up.
            return
        turn, step = int(event.data.get("turn", 0)), int(event.data.get("step", 0))
        blocks = obj(event.data.get("message")).get("content")
        streamed = self.state.end_streaming(turn, step)
        text = text_of_wire(blocks)
        thinking = text_of_wire(blocks, kind="reasoning")
        self._count_usage(obj(event.data.get("usage")))
        if live and streamed is not None:
            # Finalize what streamed rather than adding a duplicate row.
            streamed.text = text if streamed.role == "assistant" else thinking
            streamed.seq = event.seq
            if streamed.role == "thinking" and text:
                self._row("msg", "assistant", text, event, turn=turn)
            return
        if thinking:
            self._row("think", "thinking", thinking, event, turn=turn)
        if text:
            self._row("msg", "assistant", text, event, turn=turn)

    def _count_usage(self, usage: Mapping[str, Any]) -> None:
        """The provider's own count of the last request — the meter's `usage` baseline.

        Same four terms as `TokenMeter.baseline`; the meter's *estimate* branch
        is the meter's job, and a projection does not guess.
        """
        if not usage:
            return
        self.state.tokens = sum(
            int(usage.get(key) or 0)
            for key in ("inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens")
        )

    # --------------------------------------------------------------- tools --

    def _on_tool_call(self, event: SessionEvent, live: bool) -> None:
        call_id = str(event.data.get("callId"))
        name = str(event.data.get("name"))
        card = self.state.register_card(
            ToolCard(
                call_id=call_id,
                name=name,
                arguments=str(event.data.get("arguments", "")),
                title=name,
            )
        )
        self._present_call(card)
        self.state.add(
            ChatItem(
                key=f"tool-{call_id}", role="tool", tool=card, turn=self.state.turn, seq=event.seq
            )
        )

    def _present_call(self, card: ToolCard) -> None:
        """Ask the tool how its pending state looks, if it is registered here."""
        definition = self._definition(card.name)
        view = None
        if definition is not None and definition.present_call is not None:
            try:
                view = definition.present_call(parse_arguments(card.arguments))
            except Exception:
                view = None
        if view is None:
            card.subtitle = one_line(card.arguments)
            return
        card.title = view.title
        card.subtitle = view.input or view.subtitle or ""
        card.card = view.card
        card.input_text = view.body or ""

    def _on_tool_result(self, event: SessionEvent, live: bool) -> None:
        message = obj(event.data.get("message"))
        call_id = str(obj(message.get("source")).get("callId"))
        # One `tool_result` block carries both the text and the error flag; read
        # it once rather than indexing the content twice.
        result = first(message.get("content"))
        body = text_of_wire(result.get("content"))
        card = self.state.card(call_id)
        if card is None:
            self._row("tool", "tool", body, event)
            return
        card.settled = True
        card.is_error = bool(result.get("isError"))
        card.failure_kind = str(event.data.get("failureKind", ""))
        card.body = body
        self._present_result(card, event.data.get("meta"))

    def _present_result(self, card: ToolCard, meta: Any) -> None:
        definition = self._definition(card.name)
        if definition is None or definition.present_result is None:
            return
        try:
            view: ToolResultView | None = definition.present_result(
                parse_arguments(card.arguments),
                ToolResult(content=(), is_error=card.is_error, meta=meta),
            )
        except Exception:
            view = None
        if view is None:
            return
        card.title = view.title
        card.subtitle = view.subtitle or card.subtitle
        card.card = view.card
        card.details = dict(view.meta or {})

    def _definition(self, name: str) -> Any:
        if self.tools is None:
            return None
        try:
            return self.tools.get(name)
        except Exception:
            return None

    def _on_tool_code_dispatch_start(self, event: SessionEvent, live: bool) -> None:
        parent = self.state.card(str(event.data.get("parentCallId")))
        if parent is None:
            return
        name = str(event.data.get("name"))
        parent.dispatches.append(
            self.state.register_card(
                ToolCard(
                    call_id=str(event.data.get("subCallId")),
                    name=name,
                    arguments=one_line(json.dumps(thaw_json(event.data.get("arguments")))),
                    title=name,
                )
            )
        )

    def _on_tool_code_dispatch(self, event: SessionEvent, live: bool) -> None:
        dispatch = self.state.card(str(event.data.get("subCallId")))
        if dispatch is None:
            return
        dispatch.settled = True
        dispatch.is_error = bool(event.data.get("isError"))
        dispatch.body = text_of_wire(event.data.get("content"))

    # ------------------------------------------------------------ lifecycle --

    def _on_turn_start(self, event: SessionEvent, live: bool) -> None:
        self.state.turn = int(event.data.get("turn", 0))
        # Whatever was queued has been claimed into this turn.
        self.state.queued = 0

    def _on_turn_end(self, event: SessionEvent, live: bool) -> None:
        reason = obj(event.data.get("reason"))
        kind = reason.get("kind")
        if kind in ("completed", None):
            return
        if kind == "error":
            detail = obj(reason.get("error")).get("message", "the turn failed")
            self._row("err", "error", str(detail), event)
            return
        labels = {
            "aborted": "Interrupted.",
            "blocked": "Blocked by policy before the model was called.",
            "max-tokens": "Response hit the output-token ceiling.",
            "interrupted": "Recovered from an interrupted turn.",
        }
        self._row("notice", "notice", labels.get(str(kind), str(kind)), event)

    def _on_request_context(self, event: SessionEvent, live: bool) -> None:
        try:
            context = parse_request_context(event)
        except ValidationError:
            return
        self.state.provider = context.provider
        self.state.model = context.model
        self.state.context_window = context.context_window

    def _on_approval_asked(self, event: SessionEvent, live: bool) -> None:
        self._row("ask", "notice", f"Approval requested for {event.data.get('toolName')}.", event)

    def _on_approval_decided(self, event: SessionEvent, live: bool) -> None:
        outcome = str(event.data.get("outcome"))
        role: ItemRole = "notice" if outcome == "allowed-once" else "error"
        self._row("decided", role, f"{event.data.get('toolName')}: {outcome}", event)

    def _on_permission_preset(self, event: SessionEvent, live: bool) -> None:
        self.state.preset = str(event.data.get("preset", self.state.preset))

    def _on_sandbox_mode(self, event: SessionEvent, live: bool) -> None:
        self.state.sandbox_mode = str(event.data.get("mode", self.state.sandbox_mode))

    def _on_command_run(self, event: SessionEvent, live: bool) -> None:
        argument = str(event.data.get("argument", "")).strip()
        label = f"/{event.data.get('name')}" + (f" {argument}" if argument else "")
        self._row("cmd", "notice", label, event)

    def _on_command_done(self, event: SessionEvent, live: bool) -> None:
        if event.data.get("outcome") == "error":
            detail = str(event.data.get("detail", "the command failed"))
            self._row("cmderr", "error", detail, event)

    def _on_llm_retry(self, event: SessionEvent, live: bool) -> None:
        text = f"Retrying after {event.data.get('code')} (attempt {event.data.get('attempt')})."
        self._row("retry", "notice", text, event)

    def _on_kernel_restored(self, event: SessionEvent, live: bool) -> None:
        """A resumed namespace, but only when something did not come back.

        A clean restore is not news — the cell that follows simply works. A
        variable that is *gone* is news, because the model is about to reference
        a name that no longer exists and will read the failure as its own bug.
        """
        failed = [str(name) for name in seq(event.data.get("failed"))]
        if not failed:
            return
        restored = len(seq(event.data.get("restored")))
        self._row(
            "kernel",
            "notice",
            f"Restored {restored} kernel variable(s); {', '.join(failed)} could not be restored.",
            event,
        )

    def _on_harness_refined(self, event: SessionEvent, live: bool) -> None:
        """A refinement, rendered because it changes the model's own prompt.

        Not left to the auditor's view: `/refine` is only one way here — the
        planner refines at turn end with no command to show for it — and a user
        who cannot see the harness change cannot know why the next turn behaves
        differently. Rejections are on the same row: an edit the harness refused
        is the interesting half.
        """
        summary = str(event.data.get("summary") or "the harness")
        edits = len(seq(event.data.get("appliedEdits")))
        rejected = len(seq(event.data.get("rejected")))
        rolled = event.data.get("rollbackOf")
        text = (
            f"Rolled back {rolled}: {edits} edit(s) undone."
            if rolled
            else f"Refined the harness: {summary} ({edits} edit(s))."
        )
        if rejected:
            text = f"{text} {rejected} edit(s) refused."
        self._row("harness", "notice", text, event)

    def _on_harness_refine_considered(self, event: SessionEvent, live: bool) -> None:
        """A pass that decided not to refine — shown only when a human asked.

        The automatic passes (H7) are the auditor's business: one line every
        twenty-five turns saying nothing changed is noise. A `/refine` the user
        typed is the opposite — without this row the command would answer
        "refining in the background" and then never come back.
        """
        if event.data.get("trigger") != "user":
            return
        reason = str(event.data.get("reason") or "nothing to record")
        self._row("harness", "notice", f"No refinement: {reason}", event)

    def _on_context_loaded(self, event: SessionEvent, live: bool) -> None:
        """A loaded corpus — rendered only when there is something to say.

        The ordinary load is described in the system prompt already; what is news
        is the corpus having *changed* under a conversation that was told about
        it, or a source that could not be read.
        """
        note = str(event.data.get("note") or "")
        if not note:
            return
        self._row("context", "notice", note, event)

    def _on_compaction_args_truncated(self, event: SessionEvent, live: bool) -> None:
        """Long tool-call arguments elided from retained history (P4-03).

        A notice, not silence: the tool cards above still show the arguments as
        sent, and this is the only place the transcript can say the model is no
        longer being shown all of them.
        """
        seqs = seq(event.data.get("seqs"))
        saved = int(event.data.get("savedChars") or 0)
        self._row(
            "compaction",
            "notice",
            f"Shortened tool arguments in {count_of(len(seqs), 'message')} "
            f"(~{saved} characters) to make room.",
            event,
        )

    def _on_attachment_degraded(self, event: SessionEvent, live: bool) -> None:
        """Media a route would not take (P7-01).

        The one place a person finds out their diagram never reached the model.
        The adapter also logs it, but a process log is not where anyone looks to
        understand a conversation.
        """
        items = [obj(one) for one in seq(event.data.get("attachments"))]
        if not items:
            return
        names = ", ".join(str(one.get("name") or one.get("mime") or "?") for one in items)
        reason = str(items[0].get("reason") or "this model cannot read it")
        self._row("attachment", "notice", f"Not sent to the model: {names} — {reason}.", event)

    def _on_compaction_declined(self, event: SessionEvent, live: bool) -> None:
        """An automatic compaction that changed nothing, and why (P4-03).

        News, and the one place it can be news: the session is at its limit and
        stayed there, so the next thing the reader sees may be a turn that ends
        in a provider refusal. Its sibling `compaction/summarized` is
        record-less by contrast — the summary row the replacement produces is
        already the visible half of a compaction that worked.
        """
        reason = str(event.data.get("reason") or event.data.get("code") or "no reason given")
        self._row("compaction", "notice", f"Compaction declined: {reason}", event)

    def _on_subagent_admitted(self, event: SessionEvent, live: bool) -> None:
        """A delegation the human should see starting.

        Rendered rather than left to the panel because a spawn is a *decision*:
        it is the point at which work left this conversation, and reading the
        transcript later without it makes the child's eventual reply arrive from
        nowhere.
        """
        name = str(event.data.get("name") or event.data.get("runId") or "child")
        model = str(event.data.get("model") or "?")
        access = str(event.data.get("grantedAccess") or "read")
        self._fold_roster(event)
        text = f"Delegated to {name} on {model} ({access} workspace)."
        reason = event.data.get("downgradeReason")
        if reason:
            # The sentence is generated from the code, here and in the model's
            # own result, so the two never disagree and neither goes stale.
            text = f"{text} {downgrade_text(str(reason))}"
        self._row("subagent", "notice", text, event)

    def _fold_roster(self, event: SessionEvent) -> None:
        """Fold one delegation record through the *seam's* rule (A11).

        `fold_subagent_event` is `subagent_roster`'s own per-event step, so the
        panel and the roster the model reads cannot disagree about seeding,
        tombstones or `cause` — there is one rule, not two kept in step by a
        test. `SubagentRow` is the drawn projection of those rows, plus `tokens`,
        which is the panel's own addition because usage is not a roster fact.
        """
        fold_subagent_event(self.state.roster, event)
        self.state.sync_subagents()

    def _on_subagent_status(self, event: SessionEvent, live: bool) -> None:
        """A child moved. Panel only — see `RECORDLESS` for why not a row."""
        self._fold_roster(event)

    def _on_subagent_usage(self, event: SessionEvent, live: bool) -> None:
        """Attributed tokens, summed per child (P3-11).

        Not a roster fact: `subagent/usage-attributed` is deliberately outside
        the seam's `_ROSTER_TYPES`, and the sum is what the panel adds to it.
        """
        row = self.state.subagents.get(str(event.data.get("runId")))
        if row is None:
            return
        usage = obj(event.data.get("childUsage"))
        row.tokens += int(usage.get("inputTokens") or 0) + int(usage.get("outputTokens") or 0)

    def _on_subagent_deleted(self, event: SessionEvent, live: bool) -> None:
        """A revoked child. The transcript stays on disk; the row says it went."""
        run_id = str(event.data.get("runId") or "child")
        reason = str(event.data.get("reason") or "user")
        # A tombstone, not a removal — the seam's rule, applied by the seam.
        self._fold_roster(event)
        self._row("subagent", "notice", f"Revoked child {run_id} ({reason}).", event)

    def _on_todo_write(self, event: SessionEvent, live: bool) -> None:
        # Emitted by ph-stabilize's `tool-todo` (P4-01); folded here so the
        # sidebar and the model's prompt context read one list.
        self.state.todos = [thaw_json(todo) for todo in seq(event.data.get("todos"))]

    def _on_offload_spilled(self, event: SessionEvent, live: bool) -> None:
        """An oversized result or pasted message was relocated (P4-02).

        A notice rather than silence: the reader is looking at a preview where
        the tool produced far more, and the path is how they get the rest —
        the same thing the model was told.
        """
        locator = str(event.data.get("locator") or "")
        size = int(event.data.get("bytes") or 0)
        what = "Message" if event.type == "offload/input-spilled" else "Result"
        self._row(
            "offload",
            "notice",
            f"{what} too large ({count_of(size, 'byte')}); full text at {locator}",
            event,
        )

    def _on_agent_inbox_spliced(self, event: SessionEvent, live: bool) -> None:
        inserted = len(seq(event.data.get("inserted")))
        removed = int(event.data.get("removedCount", 0))
        self.state.queued = max(0, self.state.queued + inserted - removed)


Handler = Callable[[TuiEventAdapter, SessionEvent, bool], None]

HANDLERS: Mapping[str, Handler] = {
    "user/message": TuiEventAdapter._on_user_message,
    "assistant/chunk": TuiEventAdapter._on_assistant_chunk,
    "assistant/message": TuiEventAdapter._on_assistant_message,
    "tool/call": TuiEventAdapter._on_tool_call,
    "tool/result": TuiEventAdapter._on_tool_result,
    "tool/code-dispatch-start": TuiEventAdapter._on_tool_code_dispatch_start,
    "tool/code-dispatch": TuiEventAdapter._on_tool_code_dispatch,
    "turn/start": TuiEventAdapter._on_turn_start,
    "turn/end": TuiEventAdapter._on_turn_end,
    "request/context": TuiEventAdapter._on_request_context,
    "approval/asked": TuiEventAdapter._on_approval_asked,
    "approval/decided": TuiEventAdapter._on_approval_decided,
    "permission/preset": TuiEventAdapter._on_permission_preset,
    "sandbox/mode": TuiEventAdapter._on_sandbox_mode,
    "command/run": TuiEventAdapter._on_command_run,
    "command/done": TuiEventAdapter._on_command_done,
    "llm/retry": TuiEventAdapter._on_llm_retry,
    "agent/inbox/spliced": TuiEventAdapter._on_agent_inbox_spliced,
    "todo/write": TuiEventAdapter._on_todo_write,
    "offload/spilled": TuiEventAdapter._on_offload_spilled,
    "offload/input-spilled": TuiEventAdapter._on_offload_spilled,
    "compaction/declined": TuiEventAdapter._on_compaction_declined,
    "attachment/degraded": TuiEventAdapter._on_attachment_degraded,
    "compaction/args-truncated": TuiEventAdapter._on_compaction_args_truncated,
    "kernel/restored": TuiEventAdapter._on_kernel_restored,
    "harness/refined": TuiEventAdapter._on_harness_refined,
    "harness/refine-considered": TuiEventAdapter._on_harness_refine_considered,
    "context/loaded": TuiEventAdapter._on_context_loaded,
    "subagent/admitted": TuiEventAdapter._on_subagent_admitted,
    "subagent/deleted": TuiEventAdapter._on_subagent_deleted,
    "subagent/status": TuiEventAdapter._on_subagent_status,
    "subagent/usage-attributed": TuiEventAdapter._on_subagent_usage,
}
"""Event type → handler. Explicit, so the set of what renders is a value a test
can hold against the log's vocabulary rather than a naming convention."""

RECORDLESS: frozenset[str] = frozenset(
    {
        "request/header",
        "step/start",
        "step/end",
        "approval/policy",
        "fs/observed",
        "session/end-seed",
        "kernel/snapshot",
        "compaction/summarized",
    }
)
"""Known types that produce no transcript row on purpose. They are the auditor's
records — the prompt snapshot, step timings, policy changes — and belong to the
trajectory view (P3-24), not the conversation.

`kernel/snapshot` is here for a second reason as well as that one: there is one
per changed variable per cell, so rendering them would bury the conversation in
its own bookkeeping. Its companion `kernel/restored` *is* rendered, but only when
a variable failed to come back — see `_on_kernel_restored`.

`compaction/summarized` is here for a third reason again: it is the *accounting*
for a compaction, and the compaction itself already produced a row — the
replacement `user/message` the summary rides on. Rendering both would show one
event twice, once as the summary a person can read and once as a token count.
Its sibling `compaction/declined` is not record-less: a compaction that did not
happen leaves no row of its own, and the reader is about to hit the limit it
would have relieved.

`subagent/status` and `subagent/usage-attributed` were here until P3-19 gave them
their consumer. They still produce **no transcript row** — eight children ticking
through `queued → running → done` would push the conversation off screen — but
they are no longer record-less: they fold into `TuiState.subagents`, which the
sidebar's panel draws. `subagent/admitted` and `subagent/deleted` do both, and
the same split decides ignorability in `ph.session.known_event_types`."""
