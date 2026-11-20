"""Fold `session/event` frames into the transcript.

The key decision is the gate of P2-01: **the transcript is rebuilt from
`session.events`, never from `derive_messages()`.** The derivation is the view
of the *model*, and compaction deliberately shadows what it replaced. A resumed
session that was rebuilt from the derivation would lose conversation that the
user had already read. The adapter reads the log, and it marks a compacted
range instead of dropping it.

There are two modes, and the difference matters:

* **replay** (a resumed seed) uses `assistant/message`, which is the
  authoritative assembled text, and ignores `assistant/chunk` entirely.
* **live** streams the chunks, and the message then finalizes them.

Chunks fed on replay would rebuild, one delta at a time, a message that the log
already has. Any chunk lost to a crash would then leave the transcript in
disagreement with the log.

**This is a log reader with a state and a step. It is the only reader in pH
that has no canonical replay to be checked against.** Every other reader has a
pure fold of the prefix, and its incremental steps must agree with that fold:
the `extend` of a `SessionFoldCache` against its `compute`, and `SurfaceManager`
against `fold_surface`. This reader cannot have one, because two of the inputs
that its steps consume are not in the log. `Frame.view` is a card that the
*daemon* rendered beside the event. `Frame.live` selects between two
deliberately different folds: a replay reads `assistant/message` and ignores the
chunks that a live fold streams, as the modes above say. So `replay` is a second
fold and not a check on `apply`. For the same reason this reader does not use
`SessionFoldCache`, whose one requirement is a pure fold of the prefix.

This is stated explicitly, because "state + step" is the shape that the rest of
the codebase checks against a replay. A reader who assumed that the pair agreed
here would be wrong, and nothing would catch the error. The position itself is
kept in `Followed.seen` and in the mirror `Session` that a remote front end
admits events into.

`RULES` is the closed list of what this front end reacts to. It has one row for
each event type, with the handler that folds the event and the surfaces that
the fold moves. `HANDLERS` is derived from it and names the types that render.
Three sets together cover the whole vocabulary of the log:

* The types in `RULES`.
* The types that the vocabulary declares as records for an auditor and not for
  a reader (`is_audit_only`, P3-24).
* `RECORDLESS`, the few types that are in neither set.

A test holds that equality, so a new event type cannot go unrendered without a
failure.

@module ph_app.tui.adapter
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from pydantic import ValidationError

from ph.json import JsonObject, as_bool, as_int, as_obj, as_seq, as_str, thaw_json
from ph.seams.approval import INTERRUPTED
from ph.seams.token_meter import reported_usage
from ph.session import (
    Session,
    SessionEvent,
    is_in_place_rewrite,
    is_replacement_surface_event,
    is_stand_in,
    settle_of,
    shadowed_by,
    unsettled_why,
)
from ph.session.kinds import SESSION_HOLDER, hold_of
from ph.session.request_header import parse_request_context
from ph.text import count_of, one_line
from ph.tools import ToolCallView, ToolResult, ToolResultView
from ph.tools.presentation import render_call_view, render_result_view
from ph.tools.registry import ToolRuntime

from ..shell import shell_body
from ..wire import media_labels, result_block, text_of_wire
from .state import ChatItem, ItemRole, Surface, ToolCard, TuiState

__all__ = [
    "HANDLERS",
    "RECORDLESS",
    "REPLAY",
    "RULES",
    "EventRule",
    "Frame",
    "TuiEventAdapter",
]


@dataclass(frozen=True, slots=True)
class Frame:
    """What travels *beside* one event through `apply` (P7-12).

    `HANDLERS` is a table of bodies, and each body takes the event and this
    frame. The frame carries two facts today. The first is whether the event
    arrives live or is replayed. The second is the card view that a daemon
    rendered for the event, for a front end that has no registry of its own to
    ask.

    The table already carried the first fact as a bare `bool`, which every
    handler declared and two handlers read. The second fact arrived as mutable
    state on the adapter, set at the top of `apply` and cleared in a `finally`.
    That was a second ad-hoc channel for the same kind of thing, and nothing
    type-checked it. A datum that belongs to one `apply` call is a parameter, so
    both facts are parameters now. The next such fact becomes a field here and
    not a third mechanism.

    `view` is **already typed**. It is validated at the wire edge
    (`ph_app.wire.view_of`) and never inside a fold, which is two layers below
    the wire.
    """

    live: bool = True
    view: ToolCallView | ToolResultView | None = None


REPLAY = Frame(live=False)
"""An event being folded back from a stored log."""


@dataclass(slots=True)
class TuiEventAdapter:
    """Folds session events into a `TuiState`."""

    state: TuiState = field(default_factory=TuiState)
    tools: ToolRuntime | None = None
    """`ctx.tools`, when the harness is in this process. A card can then use
    the `present_call`/`present_result` of the tool itself. It is `None` over a
    socket, where the daemon renders the same views and sends them beside the
    event. See `Frame.view`."""
    _fragment: int = 0
    touched: Surface = Surface.NOTHING
    """What has changed since the last draw asked. See `take_touched`.

    It starts at nothing. This field answers "what did the events that I just
    folded move", and before any event is folded the correct answer is none.
    The *first* draw belongs to the app: `PHTuiApp._dirty` starts at `ALL`. If
    the adapter also claimed the first draw, the first batch would report
    everything, whatever it contained."""

    # --------------------------------------------------------------- entry --

    def replay(self, session: Session) -> TuiState:
        """Rebuild the whole transcript from a stored log, in place."""
        self.state.reset()
        self._fragment = 0
        for event in session.events:
            self.apply(event, REPLAY)
        return self.state

    def apply(self, event: SessionEvent, frame: Frame = Frame()) -> None:
        """Fold one event in, with what traveled beside it."""
        rule = RULES.get(event.type)
        if rule is None:
            # Not ours: an auditor's record, a `RECORDLESS` type, or a type that
            # this build does not know. In each case nothing moved, so nothing
            # is redrawn for it.
            return
        self.touched |= rule.surfaces
        rule.handler(self, event, frame)

    def take_touched(self) -> Surface:
        """Return what has moved since the last call, and reset it.

        The caller reads this one time for each delivered batch and not for
        each event, because the draw is also coalesced for each batch. The
        caller asks "what do I have to redraw", and the answer is the union
        over everything that it just folded.
        """
        touched, self.touched = self.touched, Surface.NOTHING
        return touched

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

    def _on_user_message(self, event: SessionEvent, frame: Frame) -> None:
        source = as_obj(event.data.get("source"))
        kind = source.get("kind")
        content = event.data.get("content")
        text = text_of_wire(content)
        # Attached files are part of what the user said. Without them, the row
        # would show a bare "what is this?" beside a model that answers about an
        # image that the transcript never mentions.
        attached = media_labels(content)
        if attached:
            text = "\n".join([text, *(f"[{label}]" for label in attached)]).strip()
        if is_replacement_surface_event(event):
            # Whatever its cause, a replacement shadows the rows that it cites.
            # They stay above it, dimmed.
            # The shadowed rows come from the **set of the op**, not from the
            # citation. `source_event_seqs` may name more than the replacement
            # shadowed, such as the chunks that a message was built from.
            # Dimming from the citation therefore grays rows that are still
            # live.
            self._mark_shadowed(shadowed_by(event))
            # But only a stand-in is a compaction row. `input-offload` (P4-02)
            # also substitutes on the surface. If its preview said "history
            # compacted", the reader would think that the conversation was
            # summarized when a paste was relocated. The discriminator is the
            # attribution in the log itself. Text that stands in for
            # conversation (a compaction summary, or a revision by the model
            # itself) declares `form: compaction` (`is_stand_in`). That is a
            # claim about the surface and not a color.
            if is_stand_in(event):
                self._row("compaction", "compaction", text or "(history compacted)", event)
                return
        self._row("msg", "context" if kind == "plugin" else "user", text, event)

    def _mark_shadowed(self, shadowed: tuple[int, ...]) -> None:
        """Mark the rows that a replacement stands in for.

        The rows stay. They are only no longer what the model sees. The name
        describes the mechanism and not compaction, because compaction is one
        cause of shadowing and is no longer the only cause.
        """
        targets = set(shadowed)
        for item in self.state.items:
            if item.seq in targets:
                item.shadowed = True

    def _on_assistant_chunk(self, event: SessionEvent, frame: Frame) -> None:
        if not frame.live:
            # The assembled `assistant/message` is authoritative on replay.
            return
        chunk = as_obj(event.data.get("chunk"))
        turn, step = as_int(event.data.get("turn")), as_int(event.data.get("step"))
        kind = chunk.get("type")
        if kind == "text-delta":
            self._append_stream(turn, step, "assistant", as_str(chunk.get("text")), event.seq)
        elif kind == "reasoning-delta":
            self._append_stream(turn, step, "thinking", as_str(chunk.get("text")), event.seq)

    def _append_stream(self, turn: int, step: int, role: ItemRole, text: str, seq: int) -> None:
        if not text:
            return
        item = self.state.streaming_item(turn, step, role)
        if item is None:
            # **A second row, not a replacement** (H5). A step streams reasoning
            # and then text. Both rows stay open until the message of the step
            # settles them together. When the key was the step alone, the second
            # role evicted the first, so nothing ever settled the first row. The
            # thinking row kept `streaming=True`, and with it a live
            # `MarkdownStream` task, for the rest of the session.
            #
            # It is tempting to settle the first row *here*, on the switch, but
            # that is wrong. `_on_assistant_message` would then see only the
            # text row and find no row for the reasoning of the message. It
            # would add a *second* thinking row beside the one already on
            # screen.
            self._fragment += 1
            item = ChatItem(
                key=f"stream-{turn}-{step}-{self._fragment}",
                role=role,
                streaming=True,
                turn=turn,
                seq=seq,
            )
            self.state.begin_streaming(turn, step, role, item)
        item.text += text

    def _on_assistant_message(self, event: SessionEvent, frame: Frame) -> None:
        if is_in_place_rewrite(event):
            # A node that replaces *itself*: argument truncation (P4-03) elides
            # a long tool-call argument. The text is the model's own and is
            # already on screen, so a rendered replacement would show the
            # assistant speaking twice. The row that it stands for is
            # deliberately not dimmed, because the message is still what the
            # model sees. Falling through would also let the count in the footer
            # reset to the count of an old turn.
            #
            # The test is on the shape and not on "any replacement", although
            # the commit now refuses the other shape for an assistant message: a
            # substitution is a `user/message`
            # (`surface._assert_assistant_rewrite`). A blanket return here could
            # therefore only drop what that rule lets through. That is the
            # mechanism-not-cause mistake that this file already fixed one time,
            # about fifty lines above in `_mark_shadowed`.
            return
        turn, step = as_int(event.data.get("turn")), as_int(event.data.get("step"))
        blocks = as_obj(event.data.get("message")).get("content")
        streamed = self.state.end_streaming(turn, step)
        text = text_of_wire(blocks)
        thinking = text_of_wire(blocks, kind="reasoning")
        usage = reported_usage(event)
        if usage is not None:
            # This is the count that the provider itself gave for the last
            # request: the usage baseline of the meter at the moment of the
            # reply. The adjustment for what the surface did since then is the
            # job of the meter, and a projection does not guess
            # (`docs/seams/token_meter.md`, "Reading pressure").
            self.state.tokens = usage.total
        # Finalize what streamed, and do not add a duplicate row. The message is
        # authoritative, so each open row takes its own half of the message.
        shown: set[ItemRole] = set()
        if frame.live:
            for item in streamed:
                item.text = thinking if item.role == "thinking" else text
                item.seq = event.seq
            shown = {item.role for item in streamed}
        # A half that did *not* stream is still in the message. This is one
        # pair of lines and not two. When the pair was written twice, for the
        # streamed branch and for the replayed branch, the two copies came to
        # disagree. As a result, a step whose text streamed and whose reasoning
        # did not stream dropped the reasoning entirely. An empty `shown` is
        # the replay case.
        if thinking and "thinking" not in shown:
            self._row("think", "thinking", thinking, event, turn=turn)
        if text and "assistant" not in shown:
            self._row("msg", "assistant", text, event, turn=turn)

    # --------------------------------------------------------------- tools --

    def _on_tool_call(self, event: SessionEvent, frame: Frame) -> None:
        name = as_str(event.data.get("name"))
        card = self._card_row(
            ToolCard(
                call_id=as_str(event.data.get("callId")),
                name=name,
                arguments=as_str(event.data.get("arguments")),
                title=name,
            ),
            event,
        )
        self._present_call(card, frame.view)

    def _present_call(self, card: ToolCard, arrived: ToolCallView | ToolResultView | None) -> None:
        """Set how this pending call looks: as the daemon rendered it, or as the tool would.

        The arrived view wins when there is one. There is one exactly when
        there is no registry to ask, so the two sources never compete. The view
        is narrowed by type and not trusted. A result view that is sent beside a
        call has the wrong shape, so this falls back and does not draw a wrong
        card.
        """
        view = (
            arrived
            if isinstance(arrived, ToolCallView)
            else render_call_view(self.tools, card.name, card.arguments)
        )
        if view is None:
            card.subtitle = one_line(card.arguments)
            return
        card.title = view.title
        card.subtitle = view.input or view.subtitle or ""
        card.card = view.card
        card.input_text = view.body or ""

    def _on_tool_result(self, event: SessionEvent, frame: Frame) -> None:
        # `settle_of` is the one reader of a settle. It says which call this
        # is and whether the call failed. The text is the business of the card.
        call_id, is_error = settle_of(event) or ("", False)
        body = text_of_wire(result_block(as_obj(event.data.get("message"))).get("content"))
        card = self.state.card(call_id)
        if card is None:
            self._row("tool", "tool", body, event)
            return
        card.settled = True
        card.is_error = is_error
        card.failure_kind = as_str(event.data.get("failureKind"))
        card.body = body
        self._present_result(card, as_obj(event.data.get("meta")), frame.view)

    def _present_result(
        self,
        card: ToolCard,
        meta: JsonObject,
        arrived: ToolCallView | ToolResultView | None,
    ) -> None:
        view = (
            arrived
            if isinstance(arrived, ToolResultView)
            else render_result_view(
                self.tools,
                card.name,
                card.arguments,
                ToolResult(content=(), is_error=card.is_error, meta=meta),
            )
        )
        if view is None:
            return
        card.title = view.title
        card.subtitle = view.subtitle or card.subtitle
        card.card = view.card
        card.details = dict(view.meta or {})

    def _on_tool_code_dispatch_start(self, event: SessionEvent, frame: Frame) -> None:
        parent = self.state.card(as_str(event.data.get("parentCallId")))
        if parent is None:
            return
        name = as_str(event.data.get("name"))
        parent.dispatches.append(
            self.state.register_card(
                ToolCard(
                    call_id=as_str(event.data.get("subCallId")),
                    name=name,
                    arguments=one_line(json.dumps(thaw_json(event.data.get("arguments")))),
                    title=name,
                )
            )
        )

    def _on_tool_code_dispatch(self, event: SessionEvent, frame: Frame) -> None:
        dispatch = self.state.card(as_str(event.data.get("subCallId")))
        if dispatch is None:
            return
        dispatch.settled = True
        dispatch.is_error = as_bool(event.data.get("isError"))
        dispatch.body = text_of_wire(event.data.get("content"))

    # ------------------------------------------------------------ lifecycle --

    def _on_turn_start(self, event: SessionEvent, frame: Frame) -> None:
        self.state.turn = as_int(event.data.get("turn"))
        # Whatever was queued has been claimed into this turn.
        self.state.queued = 0

    def _on_turn_end(self, event: SessionEvent, frame: Frame) -> None:
        # Settle whatever this turn left open. A step that raised never appends
        # an `assistant/message`, so this is the only thing that stops the
        # animation of its row. See `TuiState.end_streaming`.
        self.state.end_streaming(as_int(event.data.get("turn")))
        reason = as_obj(event.data.get("reason"))
        kind = reason.get("kind")
        if kind in ("completed", None):
            return
        if kind == "error":
            detail = as_obj(reason.get("error")).get("message", "the turn failed")
            self._row("err", "error", str(detail), event)
            return
        labels = {
            "aborted": "Interrupted.",
            "blocked": "Blocked by policy before the model was called.",
            "max-tokens": "Response hit the output-token ceiling.",
            "interrupted": "Recovered from an interrupted turn.",
        }
        self._row("notice", "notice", labels.get(str(kind), str(kind)), event)

    def _on_request_context(self, event: SessionEvent, frame: Frame) -> None:
        try:
            context = parse_request_context(event)
        except ValidationError:
            return
        self.state.provider = context.provider
        self.state.model = context.model
        self.state.context_window = context.context_window

    def _on_approval_asked(self, event: SessionEvent, frame: Frame) -> None:
        self._row("ask", "notice", f"Approval requested for {event.data.get('toolName')}.", event)

    def _on_approval_decided(self, event: SessionEvent, frame: Frame) -> None:
        """Report how a request was answered, or for one outcome, that it was not.

        `INTERRUPTED` gets a sentence of its own because it is the only outcome
        that is not a decision. The other four say what happened when the
        question was *put*. This one says that repair settled the request on
        resume, because the process that held it stopped existing (P5-13). A
        user who read "edit: interrupted" would reasonably think that the edit
        was interrupted. That reading is wrong and is also the more alarming
        one, because nothing ran at all.

        The row is a notice and not an error, for the same reason: nothing
        failed here. The turn that the request belonged to is recorded
        separately as interrupted, and that record is where the news belongs.
        """
        outcome = as_str(event.data.get("outcome"))
        tool = as_str(event.data.get("toolName"))
        if outcome == INTERRUPTED:
            self._row(
                "decided",
                "notice",
                f"{tool}: you were still being asked when the harness stopped — "
                "nothing ran, and the question is closed",
                event,
            )
            return
        role: ItemRole = "notice" if outcome == "allowed-once" else "error"
        self._row("decided", role, f"{tool}: {outcome}", event)

    def _card_row(self, card: ToolCard, event: SessionEvent) -> ToolCard:
        """Register a card and place its row. This is the one place that does both.

        `state.card()` finds a card by its `call_id`, and the key of the row is
        `tool-` plus the same `call_id`. Two authors of that pair are two
        chances for a lookup to miss."""
        registered = self.state.register_card(card)
        self.state.add(
            ChatItem(
                key=f"tool-{card.call_id}",
                role="tool",
                tool=registered,
                turn=self.state.turn,
                seq=event.seq,
            )
        )
        return registered

    def _on_shell_command(self, event: SessionEvent, frame: Frame) -> None:
        """Draw a shell command that the user ran, as a card that has not settled.

        The card is a `ToolCard` with the `terminal` kind, which Code Mode
        already draws. So `!!ls` looks like what it is in both front ends. The
        card is keyed by the seq of the event, because a user can run the same
        command twice, and two cards are the true account of that.

        **The title names which of the two ran.** A `!` puts its output in front
        of the model and a `!!` never does. That difference is invisible in the
        command, the output and the exit code, which are the only three things
        that the card otherwise shows. A user who cannot tell the two apart on
        screen cannot tell whether the agent is about to read what they just
        ran.
        """
        command = as_str(event.data.get("command"))
        # The gate of an autonomous goal is recorded as the same pair (S16). The
        # harness ran it and the user did not, and the title says so.
        if event.data.get("gate"):
            title = "Gate"
        elif as_bool(event.data.get("surface")):
            title = "Shell → agent"
        else:
            title = "Shell"
        self._card_row(
            ToolCard(
                call_id=f"shell-{event.seq}",
                name="shell",
                arguments="",
                title=title,
                subtitle=command,
                card="terminal",
                input_text=command,
            ),
            event,
        )

    def _on_shell_result(self, event: SessionEvent, frame: Frame) -> None:
        """Settle the command that this event cites, as every other pair here is settled.

        The command is found through `commandSeq` in the event data. That is
        the same shape as the `callId` of `tool/result` and the `askId` of
        `question/answered`. An adapter field that held "the one in flight"
        would be wrong on a multiplexed session, where two attached UIs can each
        be running a command.
        """
        card = self.state.card(f"shell-{event.data.get('commandSeq')}")
        if card is None:
            return
        card.settled = True
        card.is_error = not event.data.get("ok", False)
        card.body = shell_body(event.data)

    def _on_question_asked(self, event: SessionEvent, frame: Frame) -> None:
        header = as_str(event.data.get("header")).strip()
        label = f"{header}: " if header else ""
        self._row("asked", "notice", f"{label}{event.data.get('question')}", event)

    def _on_question_answered(self, event: SessionEvent, frame: Frame) -> None:
        """Report what became of a question. An unanswered question never draws an answer.

        **A question that was open when the harness stopped is a notice, not a
        reply.** Repair settles a question that it finds open on resume, and
        that settle carries neither `declined` nor an `answer`. When the settle
        was read as the answer of the user, it drew an empty row in their name,
        which was a reply that nobody gave. The marker says so today
        (`unsettled`, T2). A 0.3.x repair wrote `interrupted` instead, and logs
        on disk still carry it, so both are read. A question that could not be
        written at all (`not-started`) is the `declined` case below: nobody was
        asked.
        """
        stopped = unsettled_why(event.data) == "outcome-unknown"
        if stopped or event.data.get("interrupted"):
            self._row(
                "answered",
                "notice",
                "You were still being asked this when the harness stopped — it was never "
                "answered, and the question is closed.",
                event,
            )
            return
        if event.data.get("declined"):
            # The question was asked and not answered. This draws a row, because
            # the transcript would otherwise show a question and then the model
            # continuing. That reads as if the user had answered something
            # invisible.
            #
            # **The kind of not-answered matters** (K7). "No answer given" about
            # an ask that the harness failed to deliver tells the user that they
            # declined something that they were never shown. That is the same
            # false story that the sentence for the model was fixed for.
            # `resolution` is absent from logs that were written before the seam
            # supplied it, and the older reading is the right default for those.
            failed = as_str(event.data.get("resolution")) == "failed"
            said = "pH could not deliver the question." if failed else "No answer given."
            self._row("answered", "notice", said, event)
            return
        self._row("answered", "user", as_str(event.data.get("answer")), event)

    def _on_command_run(self, event: SessionEvent, frame: Frame) -> None:
        argument = as_str(event.data.get("argument")).strip()
        label = f"/{event.data.get('name')}" + (f" {argument}" if argument else "")
        self._row("cmd", "notice", label, event)

    def _on_command_done(self, event: SessionEvent, frame: Frame) -> None:
        if event.data.get("outcome") == "error":
            detail = as_str(event.data.get("detail"), "the command failed")
            self._row("cmderr", "error", detail, event)

    def _on_llm_retry(self, event: SessionEvent, frame: Frame) -> None:
        text = f"Retrying after {event.data.get('code')} (attempt {event.data.get('attempt')})."
        self._row("retry", "notice", text, event)

    def _on_session_resumed(self, event: SessionEvent, frame: Frame) -> None:
        """Say that this transcript is a continuation, and whether the earlier run crashed.

        The user who opened this session may not know that a previous run
        existed, or that it stopped in the middle of a turn. The events above
        this row are the work of another run, and that changes how the events
        below must be read. The row is drawn for the same reason as the row for
        `llm/retry`: a quiet recovery that nobody is told about cannot be told
        apart from nothing having happened.
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

    def _on_credential_needed(self, event: SessionEvent, frame: Frame) -> None:
        """Say what is waiting, and for which name (T5).

        After a restart, a held child looks from here exactly like a slow child,
        and a held session looks like one that nobody has prompted. The name is
        what the user needs in order to act. The row never shows a value,
        because the record carries none (I-3).
        """
        holder, name = hold_of(event)
        who = _holder_label(holder)
        self._row(
            "credential",
            "notice",
            f"{who[:1].upper()}{who[1:]} is waiting for {name}, which is not set here — "
            "supply it to continue.",
            event,
        )

    def _on_credential_supplied(self, event: SessionEvent, frame: Frame) -> None:
        holder, name = hold_of(event)
        self._row(
            "credential",
            "notice",
            f"{name} is here now; {_holder_label(holder)} carries on.",
            event,
        )

    def _on_session_segmented(self, event: SessionEvent, frame: Frame) -> None:
        """Say that this log stops here, and where the work continues.

        This is `_on_session_resumed` read in the other direction, and it is
        needed for the same reason: a seam at the *end* of a transcript is as
        invisible as a seam at the start. A reader who scrolls to the last row
        has reached the end of a file, not the end of the run. Without this row
        the two look identical on screen, and a routine segment looks like a
        session that stopped.
        """
        continues = as_str(event.data.get("continues"))
        text = (
            f"Continued in session {continues} — this log ends here."
            if continues
            else "Continued in another session — this log ends here."
        )
        self._row("segmented", "notice", text, event)

    def _on_supervisor_retry(self, event: SessionEvent, frame: Frame) -> None:
        """The task of this session crashed and is being run again (P5-04).

        The row is drawn for the same reason as the rows for `session/resumed`
        and `llm/retry`: a quiet recovery that nobody is told about cannot be
        told apart from nothing having happened. The transcript is also about
        to resume in the middle of a thought, and without this line that reads
        as the agent losing its place.
        """
        # The numbers come from the record, and no ladder length is assumed.
        # This may render a log that a different build wrote, and a resumed
        # transcript must not be narrated again with the constants of today.
        attempt, of = event.data.get("attempt", "?"), event.data.get("of", "?")
        seconds = as_int(event.data.get("delayMs")) / 1000
        reason = as_str(event.data.get("reason")).strip()
        detail = f": {reason}" if reason else ""
        self._row(
            "retry",
            "notice",
            f"This session hit a problem — retrying in {seconds:g}s "
            f"(attempt {attempt} of {of}){detail}",
            event,
        )

    def _on_workspace_restored(self, event: SessionEvent, frame: Frame) -> None:
        """What the tree is after a restore: put back, left as it was, or unknown (S12).

        All three cases draw a row, whoever asked. The asker is `/revert`, or a
        retry, whose row just above says which attempt this is. Without the
        row, the work that follows reads as if it started from the restore
        point. The opening record draws nothing, because this settle says
        everything that the opening record would. The seam writes the settle,
        or repair does.
        """
        if unsettled_why(event.data) is not None:
            said = "The harness stopped while restoring the tree; it may be partly restored"
        elif event.data.get("ok"):
            said = "Restored the tree to its restore point"
        else:
            detail = as_str(event.data.get("detail")).strip()
            said = "The tree could not be restored; it stays as it was" + (
                f": {detail}" if detail else ""
            )
        self._row("restored", "notice", said, event)

    def _on_supervisor_failed(self, event: SessionEvent, frame: Frame) -> None:
        """The ladder is spent, and this session has stopped (P5-04).

        This is the loudest row that this adapter draws, because it is the one
        that means no further work is coming. When a root stops silently,
        somebody waits on it indefinitely.
        """
        attempts = event.data.get("attempts", "several")
        reason = as_str(event.data.get("reason")).strip() or "no reason recorded"
        self._row(
            "failed",
            "error",
            f"This session stopped after {attempts} failed attempts: {reason}",
            event,
        )

    def _on_supervisor_recovered(self, event: SessionEvent, frame: Frame) -> None:
        """A retry worked (P5-04).

        This row closes the story that the retry row opened. Without it, the
        last thing said about a session that has worked correctly ever since
        is "retrying…".
        """
        after = event.data.get("afterAttempts", "")
        self._row("recovered", "notice", f"Recovered after {after} attempts", event)

    def _on_supervisor_passivated(self, event: SessionEvent, frame: Frame) -> None:
        """This session was released because it was idle (P5-05).

        This row stops a gap from reading as a crash. A transcript that halts
        for three days and then continues looks like a failure unless something
        says that the pause was deliberate. The `session/resumed` on the way
        back says only that something resumed, and not that nothing was wrong.
        """
        minutes = as_int(event.data.get("idleMs")) // 60_000
        self._row(
            "passivated",
            "notice",
            f"Released after {minutes} minutes idle — it resumes on the next message",
            event,
        )

    def _on_supervisor_unreachable(self, event: SessionEvent, frame: Frame) -> None:
        """The daemon lost its socket while this session was running (P5-11).

        This is a row in the *conversation* and not only in the view of the
        auditor, because of who reads it and when. This record is written when
        no client can connect. Its reader is therefore whoever opens the
        transcript afterwards and asks why nothing answered. The advice is
        included in the row. It is one command, and a notice that named a
        problem without its fix would send that reader to a search engine.
        """
        advice = as_str(event.data.get("advice"))
        reason = as_str(event.data.get("reason"), "removed")
        was = "was removed" if reason == "removed" else "was replaced by another daemon's"
        self._row(
            "unreachable",
            "notice",
            f"The daemon's socket {was} — this session kept running, but clients "
            f"could not reach it{f'. Fix: {advice}' if advice else ''}",
            event,
        )

    def _on_supervisor_violated(self, event: SessionEvent, frame: Frame) -> None:
        """A pollable invariant did not hold when the supervisor last asked (I6).

        This is a conversation row and not an auditor-only row. The reason is
        the reverse of the reason in `_on_supervisor_unreachable`: the reader
        of that record arrives afterwards, and the reader of this one is *here
        now*. What has drifted is a projection of the log: the surface, the
        derivation, or a cached fold. So the correct thing to tell a user who
        reads a transcript is that the transcript may no longer be what the
        model saw.

        **The event carries both transitions**, and a first draft missed that
        half. `verify_root` writes the same type with an empty list when a root
        starts to hold again. An empty `violations` is therefore the *good*
        news. When it was rendered through the same sentence, the row said
        "stopped holding 1 invariant (an unnamed invariant)". That was
        backwards, and it was the more alarming reading of the two.

        The row shows the invariant ids and not the details. An id names which
        promise broke and fits in a row. The detail is a sentence about node
        counts, and it belongs in the log that the row points at.
        """
        broken = [
            as_str(as_obj(one).get("invariant")) for one in as_seq(event.data.get("violations"))
        ]
        if not broken:
            self._row("violated", "notice", "This deployment holds its invariants again", event)
            return
        self._row(
            "violated",
            "notice",
            f"This deployment stopped holding {count_of(len(broken), 'invariant')} "
            f"({', '.join(broken)}) — a projection of the log no longer equals the log; "
            "what is shown here may not be what the model sees",
            event,
        )

    def _on_schedule_tick(self, event: SessionEvent, frame: Frame) -> None:
        """A scheduled run started (P5-06).

        This row answers "why did this session wake at 3am". Without it, a
        transcript shows a turn that nobody typed, which reads as the agent
        acting on its own. The row states `late` explicitly, because a tick
        that coalesced several missed fire times is a gap that the reader can
        otherwise only infer from the clock.
        """
        due, fired = as_int(event.data.get("dueAt")), as_int(event.data.get("firedAt"))
        late = (fired - due) // 1000
        delay = f", {late}s late" if late >= 1 else ""
        self._row("schedule", "notice", f"Scheduled run — {event.data.get('id', '')}{delay}", event)

    def _on_goal_set(self, event: SessionEvent, frame: Frame) -> None:
        """An autonomous run started, and what will decide it."""
        gates = as_seq(event.data.get("gates"))
        decides = f" — gates: {', '.join(str(gate) for gate in gates)}" if gates else ""
        self._row(
            "goal", "notice", f"Working toward: {event.data.get('objective', '')}{decides}", event
        )

    def _on_goal_settled(self, event: SessionEvent, frame: Frame) -> None:
        """How the run ended, in words that distinguish the three endings.

        `budget_limited` is not `achieved`. A row that blurred them would
        report work that the run did not do. This whole layer exists to make
        that failure impossible.
        """
        outcome = as_str(event.data.get("outcome"))
        detail = as_str(event.data.get("detail")).strip()
        said = {
            "achieved": "Goal achieved — every gate passed",
            "budget_limited": "Stopped: out of budget, with gates still failing",
            "abandoned": "Goal abandoned",
        }.get(outcome, f"Goal settled: {outcome}")
        text = f"{said}{f' ({detail})' if detail else ''}"
        self._row("goal", "notice" if outcome == "achieved" else "error", text, event)

    def _on_kernel_restored(self, event: SessionEvent, frame: Frame) -> None:
        """A resumed namespace, drawn only when something did not come back.

        A clean restore is not news, because the cell that follows simply
        works. A variable that is *gone* is news. The model is about to
        reference a name that no longer exists, and it will read the failure as
        its own bug.
        """
        failed = [str(name) for name in as_seq(event.data.get("failed"))]
        if not failed:
            return
        restored = len(as_seq(event.data.get("restored")))
        self._row(
            "kernel",
            "notice",
            f"Restored {restored} kernel variable(s); {', '.join(failed)} could not be restored.",
            event,
        )

    def _on_harness_refined(self, event: SessionEvent, frame: Frame) -> None:
        """A refinement, rendered because it changes the prompt of the model itself.

        It is not left to the view of the auditor. `/refine` is only one way to
        reach this handler: the planner also refines at turn end, with no
        command to show for it. A user who cannot see the harness change cannot
        know why the next turn behaves differently. Rejections are on the same
        row, because an edit that the harness refused is the interesting half.
        """
        summary = as_str(event.data.get("summary"), "the harness")
        edits = len(as_seq(event.data.get("appliedEdits")))
        rejected = len(as_seq(event.data.get("rejected")))
        rolled = event.data.get("rollbackOf")
        text = (
            f"Rolled back {rolled}: {edits} edit(s) undone."
            if rolled
            else f"Refined the harness: {summary} ({edits} edit(s))."
        )
        if rejected:
            text = f"{text} {rejected} edit(s) refused."
        self._row("harness", "notice", text, event)

    def _on_harness_refine_considered(self, event: SessionEvent, frame: Frame) -> None:
        """A pass that decided not to refine, shown only when a human asked for it.

        The automatic passes (H7) are the business of the auditor. One line
        every twenty-five turns that says nothing changed is noise. A `/refine`
        that the user typed is the opposite case. Without this row, the command
        would answer "refining in the background" and then never report back.
        """
        if event.data.get("trigger") != "user":
            return
        reason = as_str(event.data.get("reason"), "nothing to record")
        self._row("harness", "notice", f"No refinement: {reason}", event)

    def _on_context_loaded(self, event: SessionEvent, frame: Frame) -> None:
        """A loaded corpus, rendered only when there is something to say.

        The system prompt already describes the ordinary load. The news is a
        corpus that *changed* under a conversation that was told about it, or a
        source that could not be read.
        """
        note = as_str(event.data.get("note"))
        if not note:
            return
        self._row("context", "notice", note, event)

    def _on_compaction_args_truncated(self, event: SessionEvent, frame: Frame) -> None:
        """Long tool-call arguments were elided from retained history (P4-03).

        This is a notice and not silence. The tool cards above still show the
        arguments as sent, and this row is the only place where the transcript
        can say that the model is no longer shown all of them.
        """
        seqs = as_seq(event.data.get("seqs"))
        saved = as_int(event.data.get("savedChars"))
        self._row(
            "compaction",
            "notice",
            f"Shortened tool arguments in {count_of(len(seqs), 'message')} "
            f"(~{saved} characters) to make room.",
            event,
        )

    def _on_limits_exceeded(self, event: SessionEvent, frame: Frame) -> None:
        """Why the turn stopped (P4-04). This row is the only account that the
        user gets: `turn/end{blocked}` says that the turn stopped, and not what
        stopped it."""
        self._row("limits", "notice", as_str(event.data.get("message"), "Limit reached."), event)

    def _on_sandbox_denied(self, event: SessionEvent, frame: Frame) -> None:
        """A boundary that the sandbox held (P6-38), and the `/sandbox` line that lifts it.

        The sentence comes from the seam (`Denial.message`). The record carries
        it, so every reader prints one account. The row is a notice and not an
        error: the result of the command already said that it failed, and this
        row is the account of why.
        """
        self._row(
            "sandbox",
            "notice",
            as_str(event.data.get("message"), "The sandbox refused something."),
            event,
        )

    def _on_breaker_tripped(self, event: SessionEvent, frame: Frame) -> None:
        """A tool taken out of service after repeated failure."""
        tool, failures = event.data.get("tool"), event.data.get("failures")
        self._row(
            "limits",
            "notice",
            f"Stopped calling {tool} after {count_of(as_int(failures), 'failure')} in a row.",
            event,
        )

    def _on_attachment_degraded(self, event: SessionEvent, frame: Frame) -> None:
        """Media that a route would not take (P7-01).

        This row is the one place where a user finds out that their diagram
        never reached the model. The `media-degrade` row in `ph.llm.media` also
        logs it, but nobody looks in a process log to understand a
        conversation.
        """
        items = [as_obj(one) for one in as_seq(event.data.get("attachments"))]
        if not items:
            return
        names = ", ".join(as_str(one.get("name") or one.get("mime"), "?") for one in items)
        reason = as_str(items[0].get("reason"), "this model cannot read it")
        self._row("attachment", "notice", f"Not sent to the model: {names} — {reason}.", event)

    def _on_attachment_oversized(self, event: SessionEvent, frame: Frame) -> None:
        """Media that *was* sent, and is bigger than the route can use (P7-03).

        Its sibling above says that a file never reached the model. This one
        says that a file did reach the model and costs more than it is worth.
        Both are notices and neither is an error, but a reader must not confuse
        them. So the wording starts with what happened and not with the file.
        """
        items = [as_obj(one) for one in as_seq(event.data.get("attachments"))]
        if not items:
            return
        first = items[0]
        names = ", ".join(as_str(one.get("name") or one.get("mime"), "?") for one in items)
        self._row(
            "attachment",
            "notice",
            f"Sent at more detail than this model uses: {names} — "
            f"{first.get('width')}x{first.get('height')}, "
            f"scaled to {first.get('usableEdge')} px on the long edge.",
            event,
        )

    def _on_attachment_uploaded(self, event: SessionEvent, frame: Frame) -> None:
        """A file that this provider now holds a copy of (P7-03).

        The row has one reason, and the reason is not performance. A user must
        be able to see, in the conversation, that their video was handed to a
        named third party. The handle is deliberately absent. It is cache state,
        it expires, and it would read as something to keep.
        """
        name = as_str(event.data.get("name") or event.data.get("attachmentId"), "?")
        # Repair settled this record (S9). The harness stopped in the middle of
        # the upload, so the provider may or may not hold the file. A user who
        # audits the upload must be told "maybe".
        said = (
            f"An upload to {event.data.get('provider')} was interrupted: {name} may have been sent."
            if unsettled_why(event.data) is not None
            else f"Uploaded to {event.data.get('provider')}: {name}."
        )
        self._row("attachment", "notice", said, event)

    def _on_compaction_declined(self, event: SessionEvent, frame: Frame) -> None:
        """An automatic compaction that changed nothing, and why (P4-03).

        This is news, and this row is the one place where it can be news. The
        session is at its limit and stayed there, so the next thing that the
        reader sees may be a turn that ends in a provider refusal. Its sibling
        `compaction/summarized` draws no row, by contrast. The vocabulary
        declares it an auditor's record, and the summary row that the
        replacement produces is already the visible half of a compaction that
        worked.
        """
        reason = as_str(event.data.get("reason") or event.data.get("code"), "no reason given")
        self._row("compaction", "notice", f"Compaction declined: {reason}", event)

    def _on_todo_write(self, event: SessionEvent, frame: Frame) -> None:
        # The `tool-todo` row of ph-stabilize emits this event (P4-01). It is
        # folded here so that the sidebar and the prompt context of the model
        # read one list.
        # Malformed entries are *dropped*, which matches
        # `ph_stabilize.todo.todos_of` over the same payload. When they were
        # narrowed to `{}` instead, the sidebar rendered a blank row for data
        # that the other reader discards.
        self.state.todos = [
            thaw_json(todo) for todo in as_seq(event.data.get("todos")) if isinstance(todo, Mapping)
        ]

    def _on_offload_spilled(self, event: SessionEvent, frame: Frame) -> None:
        """An oversized result or pasted message was relocated (P4-02).

        This is a notice and not silence. The reader sees a preview where the
        tool produced far more, and the path is how they get the rest. The
        model was told the same thing.
        """
        locator = as_str(event.data.get("locator"))
        size = as_int(event.data.get("bytes"))
        what = "Message" if event.type == "offload/input-spilled" else "Result"
        self._row(
            "offload",
            "notice",
            f"{what} too large ({count_of(size, 'byte')}); full text at {locator}",
            event,
        )

    def _on_agent_inbox_spliced(self, event: SessionEvent, frame: Frame) -> None:
        """Track the queue depth for the footer, and draw a row when pending input is dropped.

        **`outcome` is read here because nothing else reads it.** `InboxSplice`
        writes `"canceled"` precisely to distinguish input that was *dropped*
        from input that was *claimed*. Both remove messages, and on replay the
        key is the only thing that still knows which one happened. Without it,
        the count in the footer fell silently from 1 to 0. A prompt that the
        user had queued, or the output of a `!` that they ran, was gone, and
        nothing on screen had said so.

        Esc is the common way to reach this. `AgentDriver.cancel` clears the
        inbox unless the caller asks it not to, so interrupting a turn discards
        everything that waits behind it. That is the intended behavior. The
        notice does not veto it. The notice is the account of it.

        The count and not the content, because the splice event carries neither:
        it records coordinates, and the messages it removed are recoverable only
        by an observer synchronous with the mutation (`Inbox.splice` says so).
        """
        inserted = len(as_seq(event.data.get("inserted")))
        removed = as_int(event.data.get("removedCount"))
        self.state.queued = max(0, self.state.queued + inserted - removed)
        if removed and as_str(event.data.get("outcome")) == "canceled":
            self._row("inbox", "notice", f"{count_of(removed, 'pending message')} canceled", event)


Handler = Callable[[TuiEventAdapter, SessionEvent, Frame], None]


@dataclass(frozen=True, slots=True)
class EventRule:
    """One event type: what folds it, and what that fold moves on screen.

    **One row, because two tables drifted.** What an event *draws* and what it
    *changes* were declared separately for a day. In that day two entries went
    wrong in the same way. `subagent/admitted` drew "Delegated to X" and was
    declared as a panel change only, so the row would have waited in the fold
    until something unrelated marked the transcript. A handler and its surfaces
    cannot be edited separately when they are one value.

    `surfaces` defaults to `ALL`. That is the safe answer, and it is the reason
    that most rows below say nothing. A pane that is left stale shows an old
    answer, and a redraw that nobody needed costs one frame. An entry earns a
    narrower surface set. It does not lose a surface by default.

    Every rule has a handler. For a short time the two postures had rows here
    with no handler. They render nothing, and the footer still had to redraw
    when they moved. But a reading that changes is for the *daemon* to
    announce. The daemon pushes a fresh `session.status`, and the client marks
    both surfaces when one arrives. With a rule that had no handler, the client
    inferred from an event what the readings frame states outright, and it
    inferred the redraw without the new values.
    """

    handler: Handler
    surfaces: Surface = Surface.ALL


def _holder_label(holder: str) -> str:
    """The holder of a credential hold, as a sentence names it."""
    return "this session" if holder == SESSION_HOLDER else f"sub-agent {holder}"


RULES: Mapping[str, EventRule] = {
    "user/message": EventRule(TuiEventAdapter._on_user_message, Surface.TRANSCRIPT),
    "assistant/chunk": EventRule(TuiEventAdapter._on_assistant_chunk, Surface.TRANSCRIPT),
    "assistant/message": EventRule(
        TuiEventAdapter._on_assistant_message, Surface.TRANSCRIPT | Surface.FOOTER
    ),
    "tool/call": EventRule(TuiEventAdapter._on_tool_call),
    "tool/result": EventRule(TuiEventAdapter._on_tool_result),
    "tool/code-dispatch-start": EventRule(TuiEventAdapter._on_tool_code_dispatch_start),
    "tool/code-dispatch": EventRule(TuiEventAdapter._on_tool_code_dispatch),
    "turn/start": EventRule(TuiEventAdapter._on_turn_start, Surface.TRANSCRIPT | Surface.FOOTER),
    "turn/end": EventRule(TuiEventAdapter._on_turn_end, Surface.TRANSCRIPT | Surface.FOOTER),
    "request/context": EventRule(TuiEventAdapter._on_request_context, Surface.FOOTER),
    "approval/asked": EventRule(TuiEventAdapter._on_approval_asked),
    "approval/decided": EventRule(TuiEventAdapter._on_approval_decided),
    "shell/command": EventRule(TuiEventAdapter._on_shell_command),
    "shell/result": EventRule(TuiEventAdapter._on_shell_result),
    "question/asked": EventRule(TuiEventAdapter._on_question_asked),
    "question/answered": EventRule(TuiEventAdapter._on_question_answered),
    "sandbox/denied": EventRule(TuiEventAdapter._on_sandbox_denied),
    "command/run": EventRule(TuiEventAdapter._on_command_run),
    "command/done": EventRule(TuiEventAdapter._on_command_done),
    "llm/retry": EventRule(TuiEventAdapter._on_llm_retry),
    "session/resumed": EventRule(TuiEventAdapter._on_session_resumed),
    "session/segmented": EventRule(TuiEventAdapter._on_session_segmented),
    "credential/needed": EventRule(TuiEventAdapter._on_credential_needed),
    "credential/supplied": EventRule(TuiEventAdapter._on_credential_supplied),
    "supervisor/retry": EventRule(TuiEventAdapter._on_supervisor_retry),
    "workspace/restored": EventRule(TuiEventAdapter._on_workspace_restored),
    "supervisor/failed": EventRule(TuiEventAdapter._on_supervisor_failed),
    "supervisor/recovered": EventRule(TuiEventAdapter._on_supervisor_recovered),
    "supervisor/passivated": EventRule(TuiEventAdapter._on_supervisor_passivated),
    "supervisor/unreachable": EventRule(TuiEventAdapter._on_supervisor_unreachable),
    "supervisor/violated": EventRule(TuiEventAdapter._on_supervisor_violated),
    "schedule/tick": EventRule(TuiEventAdapter._on_schedule_tick),
    "goal/set": EventRule(TuiEventAdapter._on_goal_set),
    "goal/settled": EventRule(TuiEventAdapter._on_goal_settled),
    "agent/inbox/spliced": EventRule(TuiEventAdapter._on_agent_inbox_spliced),
    "todo/write": EventRule(TuiEventAdapter._on_todo_write, Surface.SIDEBAR),
    "offload/spilled": EventRule(TuiEventAdapter._on_offload_spilled),
    "offload/input-spilled": EventRule(TuiEventAdapter._on_offload_spilled),
    "compaction/declined": EventRule(TuiEventAdapter._on_compaction_declined),
    "attachment/degraded": EventRule(TuiEventAdapter._on_attachment_degraded),
    "attachment/oversized": EventRule(TuiEventAdapter._on_attachment_oversized),
    "attachment/uploaded": EventRule(TuiEventAdapter._on_attachment_uploaded),
    "limits/exceeded": EventRule(TuiEventAdapter._on_limits_exceeded),
    "limits/breaker-tripped": EventRule(TuiEventAdapter._on_breaker_tripped),
    "compaction/args-truncated": EventRule(TuiEventAdapter._on_compaction_args_truncated),
    "kernel/restored": EventRule(TuiEventAdapter._on_kernel_restored),
    "harness/refined": EventRule(TuiEventAdapter._on_harness_refined),
    "harness/refine-considered": EventRule(TuiEventAdapter._on_harness_refine_considered),
    "context/loaded": EventRule(TuiEventAdapter._on_context_loaded),
}
"""Every event type that this front end reacts to.

`HANDLERS` below is derived from it, so "what renders" stays a value that a
test can compare with the vocabulary of the log."""

HANDLERS: Mapping[str, Handler] = {name: rule.handler for name, rule in RULES.items()}
"""Event type → handler, derived from `RULES`.

It keeps its own name because the vocabulary gate compares against it. Also,
"does this render" is a question that a caller can then ask without unpacking a
rule."""

RECORDLESS: frozenset[str] = frozenset(
    {
        # Folded into what they time: the rows of the step itself, and the
        # `Timing` of the trajectory.
        "step/start",
        "step/end",
        # There is one for each changed variable in each cell, so rendering
        # them would bury the conversation in its own bookkeeping. Its
        # companion `kernel/restored` is rendered, but only when a variable
        # failed to come back.
        "kernel/snapshot",
    }
)
"""Known types that draw no transcript row and are also not records that only an
auditor reads. Another record carries them, or they are too fine-grained to read.

The records of the auditor (the prompt snapshot, policy changes, the accounting
of a compaction) draw no row either, but they are not listed here. The
vocabulary declares them (`is_audit_only`), and the types of a package with
them, so this front end needs no entry for a type that it cannot import.
"""
