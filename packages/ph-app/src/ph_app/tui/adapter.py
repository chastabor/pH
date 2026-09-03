"""`session/event` in, transcript out.

The load-bearing decision (P2-01's gate): **the transcript is rebuilt from
`session.events`, never from `derive_messages()`.** The derivation is the *model's*
view, and compaction deliberately shadows what it replaced — so rebuilding a
resumed session from it would erase conversation the person sitting there already
read. The adapter reads the log, and marks a compacted range rather than dropping
it.

Two modes, and the difference matters:

* **replay** (a resumed seed) uses `assistant/message`, the authoritative
  assembled text, and ignores `assistant/chunk` entirely;
* **live** streams the chunks and lets the message finalize them.

Feeding chunks on replay would rebuild a message the log already has one delta at
a time, and any chunk lost to a crash would leave the transcript disagreeing with
the log.

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
    SurfaceReplace,
    is_in_place_rewrite,
    is_replacement_surface_event,
    thaw_json,
)
from ph.session.request_header import parse_request_context
from ph.text import count_of
from ph.tools import ToolCallView, ToolResult, ToolResultView
from ph.tools.presentation import render_call_view, render_result_view

from ..wire import media_labels, obj, one_line, result_block, seq, text_of_wire
from .state import ChatItem, ItemRole, ToolCard, TuiState

__all__ = ["HANDLERS", "RECORDLESS", "TuiEventAdapter"]


def _arrived[View: (ToolCallView, ToolResultView)](model: type[View], sidecar: Any) -> View | None:
    """The daemon's rendered view, validated rather than trusted.

    The sidecar is JSON off a socket, and the two view models are what say which
    fields a card may set. One that does not parse renders as the generic card,
    which is what a front end with no daemon already does — a plain card is not a
    regression, a wrong one would be.
    """
    if not isinstance(sidecar, Mapping):
        return None
    try:
        return model.model_validate(dict(sidecar))
    except ValidationError:
        return None


@dataclass(slots=True)
class TuiEventAdapter:
    """Folds session events into a `TuiState`."""

    state: TuiState = field(default_factory=TuiState)
    tools: Any = None
    """`ctx.tools`, when the harness is in this process, so a card can use the
    tool's own `present_call`/`present_result`. `None` over a socket, where the
    daemon renders the same views and sends them instead — see `_sidecar`."""
    _sidecar: Any = None
    """The view the daemon rendered for the event being applied, if any.

    **The sidecar wins when there is one**, and there is one exactly when there
    is no registry to ask, so the two sources never compete. Held on the adapter
    for the duration of one `apply` rather than threaded through: `HANDLERS` is a
    table of `(self, event, live)` bodies, and widening every one of forty-eight
    of them to carry a value two of them read would put the cost of this feature
    everywhere it is not used."""
    _fragment: int = 0

    # --------------------------------------------------------------- entry --

    def replay(self, session: Session) -> TuiState:
        """Rebuild the whole transcript from a stored log, in place."""
        self.state.reset()
        self._fragment = 0
        for event in session.events:
            self.apply(event, live=False)
        return self.state

    def apply(self, event: SessionEvent, *, live: bool = True, presentation: Any = None) -> None:
        """Fold one event in, with whatever the daemon rendered for it.

        `presentation` is `None` for every in-process caller and for every event
        that is not a tool card, which is nearly all of them.
        """
        handler = HANDLERS.get(event.type)
        if handler is None:
            return
        self._sidecar = presentation
        try:
            handler(self, event, live)
        finally:
            self._sidecar = None

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
            # The **op's set**, not the citation. `source_event_seqs` may name
            # more than the replacement shadowed — the chunks a message was
            # built from — so dimming from it greys rows that are still live.
            operation = event.surface_op
            self._mark_shadowed(
                operation.replaces
                if isinstance(operation, SurfaceReplace)
                else tuple(event.source_event_seqs or ())
            )
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
        """How this pending call looks — as rendered, or as the tool would."""
        view = _arrived(ToolCallView, self._sidecar) or render_call_view(
            self.tools, card.name, card.arguments
        )
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
        result = result_block(message)
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
        view = _arrived(ToolResultView, self._sidecar) or render_result_view(
            self.tools,
            card.name,
            card.arguments,
            ToolResult(content=(), is_error=card.is_error, meta=meta),
        )
        if view is None:
            return
        card.title = view.title
        card.subtitle = view.subtitle or card.subtitle
        card.card = view.card
        card.details = dict(view.meta or {})

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

    def _on_question_asked(self, event: SessionEvent, live: bool) -> None:
        header = str(event.data.get("header") or "").strip()
        label = f"{header}: " if header else ""
        self._row("asked", "notice", f"{label}{event.data.get('question')}", event)

    def _on_question_answered(self, event: SessionEvent, live: bool) -> None:
        if event.data.get("declined"):
            # Asked and not answered. A row rather than silence: the transcript
            # otherwise shows a question and then the model carrying on, which
            # reads as the person having answered something invisible.
            self._row("answered", "notice", "No answer given.", event)
            return
        self._row("answered", "user", str(event.data.get("answer", "")), event)

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

    def _on_session_resumed(self, event: SessionEvent, live: bool) -> None:
        """Say that this transcript is a continuation, and whether it crashed.

        The person who opened this may not know a previous run existed, let
        alone that it stopped mid-turn — and the events above this row are
        somebody else's work, which changes how the ones below should be read.
        Worth a line for the same reason `llm/retry` is: quiet recovery that
        nobody is told about is indistinguishable from nothing having happened.
        """
        events = event.data.get("events", 0)
        if event.data.get("interrupted"):
            text = (
                f"Resumed after an interrupted run — {events} earlier events, "
                "and the turn that was open when it stopped has been closed."
            )
        else:
            text = f"Resumed an existing session — {events} earlier events."
        self._row("resumed", "notice", text, event)

    def _on_session_segmented(self, event: SessionEvent, live: bool) -> None:
        """Say that this log stops here, and where the work carries on.

        `_on_session_resumed` read the other way round, and needed for the same
        reason: a seam at the *end* of a transcript is as invisible as one at the
        start. Somebody scrolling to the last row has reached the end of a file,
        not the end of the run, and without this the two are identical on screen
        — which makes a routine segment look like a session that stopped.
        """
        continues = str(event.data.get("continues", ""))
        text = (
            f"Continued in session {continues} — this log ends here."
            if continues
            else "Continued in another session — this log ends here."
        )
        self._row("segmented", "notice", text, event)

    def _on_supervisor_retry(self, event: SessionEvent, live: bool) -> None:
        """This session's task crashed and is being run again (P5-04).

        Worth a row for the reason `session/resumed` and `llm/retry` are: a
        quiet recovery nobody is told about is indistinguishable from nothing
        having happened — and the transcript is about to resume mid-thought,
        which without this line reads as the agent losing its place.
        """
        # From the record, with no ladder length assumed: this may be
        # rendering a log a different build wrote, and a resumed transcript
        # should not be re-narrated with today's constants.
        attempt, of = event.data.get("attempt", "?"), event.data.get("of", "?")
        seconds = int(event.data.get("delayMs", 0)) / 1000
        restored = " after restoring the tree" if event.data.get("restored") else ""
        reason = str(event.data.get("reason", "")).strip()
        detail = f": {reason}" if reason else ""
        self._row(
            "retry",
            "notice",
            f"This session hit a problem — retrying in {seconds:g}s{restored} "
            f"(attempt {attempt} of {of}){detail}",
            event,
        )

    def _on_supervisor_failed(self, event: SessionEvent, live: bool) -> None:
        """The ladder is spent, and this session has stopped (P5-04).

        The loudest row this adapter draws, because it is the one that means no
        further work is coming. A root that stopped silently is one somebody
        waits on indefinitely.
        """
        attempts = event.data.get("attempts", "several")
        reason = str(event.data.get("reason", "")).strip() or "no reason recorded"
        self._row(
            "failed",
            "error",
            f"This session stopped after {attempts} failed attempts: {reason}",
            event,
        )

    def _on_supervisor_recovered(self, event: SessionEvent, live: bool) -> None:
        """A retry worked (P5-04).

        The close of a story the retry row opened. Without it a reader is left
        with "retrying…" as the last thing said about a session that has been
        working fine ever since.
        """
        after = event.data.get("afterAttempts", "")
        self._row("recovered", "notice", f"Recovered after {after} attempts", event)

    def _on_supervisor_passivated(self, event: SessionEvent, live: bool) -> None:
        """This session was released for being idle (P5-05).

        The row that stops a gap from reading as a crash. A transcript that
        halts for three days and picks up again looks like a failure unless
        something says the pause was deliberate — and the `session/resumed` on
        the way back says only that something resumed, not that nothing was
        wrong.
        """
        minutes = int(event.data.get("idleMs", 0)) // 60_000
        self._row(
            "passivated",
            "notice",
            f"Released after {minutes} minutes idle — it resumes on the next message",
            event,
        )

    def _on_supervisor_unreachable(self, event: SessionEvent, live: bool) -> None:
        """The daemon lost its socket while this session was running (P5-11).

        A row, and a row in the *conversation* rather than only in the auditor's
        view, because of who reads it and when: this record is written when no
        client can connect, so its reader is whoever opens the transcript
        afterwards asking why nothing answered. The advice rides along — it is
        one command, and a notice that named a problem without its fix would
        send that reader to a search engine.
        """
        advice = str(event.data.get("advice") or "")
        reason = str(event.data.get("reason") or "removed")
        was = "was removed" if reason == "removed" else "was replaced by another daemon's"
        self._row(
            "unreachable",
            "notice",
            f"The daemon's socket {was} — this session kept running, but clients "
            f"could not reach it{f'. Fix: {advice}' if advice else ''}",
            event,
        )

    def _on_schedule_tick(self, event: SessionEvent, live: bool) -> None:
        """A scheduled run started (P5-06).

        The row that answers "why did this session wake at 3am". Without it a
        transcript shows a turn nobody typed, which reads as the agent acting on
        its own — and `late` is said out loud because a tick that coalesced
        several missed fire times is a gap the reader can otherwise only infer
        from the clock.
        """
        due, fired = int(event.data.get("dueAt", 0)), int(event.data.get("firedAt", 0))
        late = (fired - due) // 1000
        delay = f", {late}s late" if late >= 1 else ""
        self._row("schedule", "notice", f"Scheduled run — {event.data.get('id', '')}{delay}", event)

    def _on_goal_set(self, event: SessionEvent, live: bool) -> None:
        """An autonomous run started, and what will decide it."""
        gates = event.data.get("gates") or ()
        decides = f" — gates: {', '.join(str(gate) for gate in gates)}" if gates else ""
        self._row(
            "goal", "notice", f"Working toward: {event.data.get('objective', '')}{decides}", event
        )

    def _on_goal_settled(self, event: SessionEvent, live: bool) -> None:
        """How the run ended, in the words that distinguish the three endings.

        `budget_limited` is not `achieved`, and a row that blurred them would
        report work the run did not do — which is the failure this whole layer
        exists to make impossible.
        """
        outcome = str(event.data.get("outcome", ""))
        detail = str(event.data.get("detail", "")).strip()
        said = {
            "achieved": "Goal achieved — every gate passed",
            "budget_limited": "Stopped: out of budget, with gates still failing",
            "abandoned": "Goal abandoned",
        }.get(outcome, f"Goal settled: {outcome}")
        text = f"{said}{f' ({detail})' if detail else ''}"
        self._row("goal", "notice" if outcome == "achieved" else "error", text, event)

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

    def _on_limits_exceeded(self, event: SessionEvent, live: bool) -> None:
        """Why the turn stopped (P4-04). The person's only account of it —
        `turn/end{blocked}` says that it stopped, not what stopped it."""
        self._row("limits", "notice", str(event.data.get("message") or "Limit reached."), event)

    def _on_breaker_tripped(self, event: SessionEvent, live: bool) -> None:
        """A tool taken out of service after repeated failure."""
        tool, failures = event.data.get("tool"), event.data.get("failures")
        self._row(
            "limits",
            "notice",
            f"Stopped calling {tool} after {count_of(int(failures or 0), 'failure')} in a row.",
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

    def _on_attachment_oversized(self, event: SessionEvent, live: bool) -> None:
        """Media that *was* sent, and is bigger than the route can use (P7-03).

        Its sibling above says a file never reached the model; this says one did
        and is costing more than it buys. Both are notices and neither is an
        error, but a reader must not confuse them — so the wording leads with
        what happened rather than with the file.
        """
        items = [obj(one) for one in seq(event.data.get("attachments"))]
        if not items:
            return
        first = items[0]
        names = ", ".join(str(one.get("name") or one.get("mime") or "?") for one in items)
        self._row(
            "attachment",
            "notice",
            f"Sent at more detail than this model uses: {names} — "
            f"{first.get('width')}x{first.get('height')}, "
            f"scaled to {first.get('usableEdge')} px on the long edge.",
            event,
        )

    def _on_attachment_uploaded(self, event: SessionEvent, live: bool) -> None:
        """A file this provider now holds a copy of (P7-03).

        Worth a row for one reason and it is not performance: a person should be
        able to see, in the conversation, that their video was handed to a named
        third party. The handle is deliberately absent — it is cache state, it
        expires, and it would read as something to keep.
        """
        name = str(event.data.get("name") or event.data.get("attachmentId") or "?")
        self._row(
            "attachment",
            "notice",
            f"Uploaded to {event.data.get('provider')}: {name}.",
            event,
        )

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
    "question/asked": TuiEventAdapter._on_question_asked,
    "question/answered": TuiEventAdapter._on_question_answered,
    "permission/preset": TuiEventAdapter._on_permission_preset,
    "sandbox/mode": TuiEventAdapter._on_sandbox_mode,
    "command/run": TuiEventAdapter._on_command_run,
    "command/done": TuiEventAdapter._on_command_done,
    "llm/retry": TuiEventAdapter._on_llm_retry,
    "session/resumed": TuiEventAdapter._on_session_resumed,
    "session/segmented": TuiEventAdapter._on_session_segmented,
    "supervisor/retry": TuiEventAdapter._on_supervisor_retry,
    "supervisor/failed": TuiEventAdapter._on_supervisor_failed,
    "supervisor/recovered": TuiEventAdapter._on_supervisor_recovered,
    "supervisor/passivated": TuiEventAdapter._on_supervisor_passivated,
    "supervisor/unreachable": TuiEventAdapter._on_supervisor_unreachable,
    "schedule/tick": TuiEventAdapter._on_schedule_tick,
    "goal/set": TuiEventAdapter._on_goal_set,
    "goal/settled": TuiEventAdapter._on_goal_settled,
    "agent/inbox/spliced": TuiEventAdapter._on_agent_inbox_spliced,
    "todo/write": TuiEventAdapter._on_todo_write,
    "offload/spilled": TuiEventAdapter._on_offload_spilled,
    "offload/input-spilled": TuiEventAdapter._on_offload_spilled,
    "compaction/declined": TuiEventAdapter._on_compaction_declined,
    "attachment/degraded": TuiEventAdapter._on_attachment_degraded,
    "attachment/oversized": TuiEventAdapter._on_attachment_oversized,
    "attachment/uploaded": TuiEventAdapter._on_attachment_uploaded,
    "limits/exceeded": TuiEventAdapter._on_limits_exceeded,
    "limits/breaker-tripped": TuiEventAdapter._on_breaker_tripped,
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
        # Creating, cancelling and heartbeating a schedule are not events in the
        # conversation — the *tick* is what a reader needs, and it has a row.
        "schedule/created",
        "schedule/cancelled",
        "schedule/heartbeat",
        # The loop's own bookkeeping. `goal/set` and `goal/settled` are rows —
        # they bracket the run — while a continuation and a gate result are
        # accounting the transcript already shows as turns and tool output.
        "goal/continued",
        "goal/gate",
        "request/header",
        "step/start",
        "step/end",
        "approval/mode",
        "approval/policy",
        "fs/observed",
        "workspace/acquired",
        "workspace/disposed",
        "workspace/retained",
        "workspace/provisioned",
        "workspace/checkpoint",
        "session/end-seed",
        # Protocol bookkeeping: which client asked for which turn. The turn
        # itself renders; who deduplicated it is not conversation.
        "client/command",
        "kernel/snapshot",
        "compaction/summarized",
    }
)
"""Known types that produce no transcript row on purpose — the auditor's records
(the prompt snapshot, step timings, policy changes), which belong to the
trajectory view (P3-24) rather than the conversation.

Three entries are here for reasons of their own:

* `kernel/snapshot` — one per changed variable per cell, so rendering them would
  bury the conversation in its own bookkeeping. Its companion `kernel/restored`
  *is* rendered, but only when a variable failed to come back.
* `compaction/summarized` — the *accounting* for a compaction, where the
  compaction itself already produced a row: the replacement `user/message` the
  summary rides on. Its sibling `compaction/declined` is **not** record-less,
  because a compaction that did not happen leaves no row of its own and the
  reader is about to hit the limit it would have relieved.
* `subagent/status` and `subagent/usage-attributed` produce no transcript row —
  eight children ticking through `queued → running → done` would push the
  conversation off screen — but they are not record-less: they fold into
  `TuiState.subagents`, which the sidebar's panel draws. `subagent/admitted` and
  `subagent/deleted` do both, and the same split decides ignorability in
  `ph.session.known_event_types`.
"""
