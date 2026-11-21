"""The agent inbox: durable, replayable pending input.

The inbox has three delivery semantics, ported exactly. The difference between
them is what makes steering feel instant and injection feel invisible:

| Call | Lands at | Wakes an idle agent |
|---|---|---|
| `followup(msg)` | next **turn** | yes |
| `steer(msg)` | next **step** | yes |
| `inject(msg)` | next **step** | **no**: it waits for another message |

Every mutation is logged as `agent/inbox/spliced` *before* the projection
changes. A resumed agent therefore rebuilds its queue from the log, and does
not lose what the user typed before the crash.

**A claim is proposed, then taken** (S3). The driver has to decide about a
batch before the batch reaches the conversation. That decision,
`agent/pre-step`, is an await, and a listener in it may make a model call whose
barrier flushes the log. When the batch was taken first, a crash inside that
await left the prompt out of the inbox on disk and nowhere in the conversation.
So `propose` says what a step would take and takes nothing. `claim` takes it in
one `Session.batch()` together with the records that the caller appends. The
splice that removes a prompt from the inbox and the `user/message` that puts it
in the conversation are written together, or neither is written.

@module ph.agent.inbox
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import ConfigDict

from ..json import JsonObject
from ..llm.types import Message
from ..session import Session, SessionBatch, SessionEvent
from ..session.writers import log_writer
from ..wire import WireModel

_LOG = log_writer(__name__)

__all__ = ["Inbox", "InboxBatch", "InboxNotifications", "InboxTarget", "replay_pending"]

InboxTarget: TypeAlias = Literal["next-turn", "next-step"]


@dataclass(frozen=True, slots=True)
class InboxNotifications:
    """Live mirrors of durable inbox mutations."""

    inserted: Callable[[Message], None]
    discarded: Callable[[Message], None]
    claimed: Callable[[Message, int], None]


@dataclass(frozen=True, slots=True)
class InboxBatch:
    """What one step would take from the inbox: proposed, and not yet taken.

    `Inbox.propose` builds one and `Inbox.claim` takes it. Holding one takes
    nothing. The messages stay pending, on disk and in the projection, until
    the batch is claimed.
    """

    steps: tuple[Message, ...]
    """Every `next-step` message that was pending when the batch was proposed."""
    turn: tuple[Message, ...]
    """At a turn boundary, the one queued turn that the batch takes. Otherwise empty."""

    @property
    def messages(self) -> tuple[Message, ...]:
        """The batch, in the order in which the step sees it."""
        return (*self.steps, *self.turn)


class InboxSplice(WireModel):
    """One inbox mutation, as `agent/inbox/spliced` carries it.

    The payload is declared, as every other durable payload is (`UserQuestion`,
    `RequestContext`, `Message`). It is not built by hand and parsed by hand.
    `WireModel` owns the camelCase aliases and the frozen mapping of the log.
    The key that `splice` writes and the key that `_apply_splice` reads
    therefore cannot drift apart. `ph_app.tui.adapter` could validate against it
    too. It still reads `inserted` and `removedCount` by hand.

    `removed_count` is `None` and not `0` when it is absent, and `outcome`
    likewise, because `to_wire()` omits `None`. An insert-only splice must not
    start to emit `"removedCount": 0` into logs that never carried it.

    `extra="ignore"` overrides the `forbid` of `WireModel`. A log is read by
    builds older than the build that wrote it. A key that was added later must
    be skipped on replay, and must not condemn the whole session.
    """

    model_config = ConfigDict(extra="ignore")

    target: InboxTarget
    start: int
    inserted: list[Message]
    removed_count: int | None = None
    outcome: Literal["canceled"] | None = None
    """Whether the removed messages were canceled or consumed. Live, the
    difference is `_notify.discarded` against `_notify.claimed`. On replay this
    key is the only record of which one happened, and `ph_app.tui.adapter`
    reads it to report canceled messages."""


class Inbox:
    """A projection that replays the log one time and then applies each later splice."""

    __slots__ = ("_notify", "_session", "_state")

    def __init__(self, session: Session, notifications: InboxNotifications) -> None:
        self._session = session
        self._notify = notifications
        # Replay only the splices of this lifecycle. A fork inherits the
        # transcript of its parent, and not the unanswered queue of its parent.
        self._state: dict[str, list[Message]] = replay_pending(session.own_events())

    @property
    def next_turn(self) -> tuple[Message, ...]:
        return tuple(self._state["next-turn"])

    @property
    def next_step(self) -> tuple[Message, ...]:
        return tuple(self._state["next-step"])

    @property
    def has_pending(self) -> bool:
        return bool(self._state["next-turn"] or self._state["next-step"])

    def clear(self) -> None:
        """Durably cancel all pending input, next-step before next-turn."""
        self.splice("next-step", 0, len(self._state["next-step"]), [])
        self.splice("next-turn", 0, len(self._state["next-turn"]), [])

    def propose(self, target: InboxTarget) -> InboxBatch:
        """The batch that one step would take, without taking it.

        The batch always holds every pending `next-step` message. At a turn
        boundary it also holds exactly one queued turn. Claiming more than one
        turn would merge two user prompts into one model call.
        """
        turn = tuple(self._state["next-turn"][:1]) if target == "next-turn" else ()
        return InboxBatch(steps=tuple(self._state["next-step"]), turn=turn)

    @contextmanager
    def claim(self, proposed: InboxBatch, turn: int) -> Iterator[SessionBatch]:
        """Take `proposed` out of the inbox, in one batch with what the block appends.

        ```python
        with inbox.claim(proposed, turn) as batch:
            _LOG.append(batch, "user/message", message.to_wire(), SurfaceIntent("append"))
        ```

        The splices are written into the batch that the block receives, so they
        commit with the records of the block or not at all. A block that raises,
        a batch that the surface refuses, or a process that dies before the
        flush leaves every message pending. The projection changes and `claimed`
        fires only after the batch has committed. That is after the durable
        event, as for every splice here, so a `session/event` observer still
        sees the lists as they were.

        The block must be synchronous, as `Session.batch` requires.

        :raises RuntimeError: when the front of a queue is no longer what was
            proposed. Something took it in between, which a single driver
            cannot do to itself.
        """
        queues: tuple[tuple[InboxTarget, tuple[Message, ...]], ...] = (
            ("next-step", proposed.steps),
            ("next-turn", proposed.turn),
        )
        for target, messages in queues:
            front = self._state[target][: len(messages)]
            if [m.id for m in front] != [m.id for m in messages]:
                raise RuntimeError(
                    f"the {target} inbox changed after its batch was proposed, so it "
                    "cannot be claimed as proposed"
                )
        taken = [(target, len(messages)) for target, messages in queues if messages]
        with self._session.batch() as batch:
            for target, count in taken:
                splice = InboxSplice(target=target, start=0, inserted=[], removed_count=count)
                _LOG.append(batch, "agent/inbox/spliced", splice.to_wire())
            yield batch
        for target, count in taken:
            del self._state[target][:count]
        for message in proposed.messages:
            self._notify.claimed(message, turn)

    def take(self, proposed: InboxBatch, turn: int) -> None:
        """`claim` with nothing appended beside it. The batch is consumed and reaches
        no step: a step that a pre-step rejected, or a step that a pre-step emptied."""
        with self.claim(proposed, turn):
            pass

    def append(self, target: InboxTarget, message: Message) -> None:
        self.splice(target, len(self._state[target]), 0, [message])

    def splice(
        self, target: InboxTarget, start: int, delete_count: int, inserted: Sequence[Message]
    ) -> list[Message]:
        """Standard splice semantics, durably recorded. Removed messages are canceled."""
        pending = self._state[target]
        start = min(max(start, 0), len(pending))
        delete_count = min(max(delete_count, 0), len(pending) - start)
        if delete_count == 0 and not inserted:
            return []
        splice = InboxSplice(
            target=target,
            start=start,
            inserted=list(inserted),
            removed_count=delete_count or None,
            outcome="canceled" if delete_count else None,
        )
        # This is the one rule that a write can still break. The coordinates
        # are clamped above and the values are already typed, so there is
        # nothing here to parse.
        _unique_in(self._state, target, start, delete_count, inserted)
        # The durable event commits BEFORE the live projection changes. A
        # synchronous `session/event` observer therefore sees the lists as they
        # were before the splice, and can reconstruct exactly what was removed
        # from the normalized coordinates.
        _LOG.append(self._session, "agent/inbox/spliced", splice.to_wire())
        removed = pending[start : start + delete_count]
        pending[start : start + delete_count] = inserted
        for message in removed:
            self._notify.discarded(message)
        for message in inserted:
            self._notify.inserted(message)
        return list(removed)

    # ------------------------------------------------------------- internals --


def replay_pending(events: Iterable[SessionEvent]) -> dict[str, list[Message]]:
    """What is pending after replaying every `agent/inbox/spliced` among `events`.

    **This is a pure fold, and `Inbox` is built from it** (P13-01). A reader may
    hold a stored log and no `Session`: the live-roots index does, when it
    decides whether a root died with a prompt that nobody claimed. That reader
    asks the same question that the constructor asks. A second replay written
    there would be a second account of splice semantics, and it could drift
    from this one. The caller chooses the slice. `Inbox` replays only the
    splices of its own lifecycle (`own_events`), because a fork inherits the
    transcript of its parent and not the unanswered queue of its parent.

    :raises ValueError: for a splice that the queues cannot take, naming its seq.
    """
    state: dict[str, list[Message]] = {"next-turn": [], "next-step": []}
    for event in events:
        if event.type != "agent/inbox/spliced":
            continue
        try:
            _apply_splice(state, event.data)
        except ValueError as error:
            raise ValueError(
                f"invalid persisted inbox splice at session seq {event.seq}"
            ) from error
    return state


def _apply_splice(state: dict[str, list[Message]], splice: JsonObject) -> None:
    """Replay one logged splice into `state`, or refuse it."""
    parsed = InboxSplice.model_validate(splice)
    pending = state[parsed.target]
    removed_count = parsed.removed_count or 0
    # Check that the coordinates fall inside the queue that they name. Only
    # replay needs this. `splice` clamps both into range before it writes, so on
    # the write path these branches cannot fail, and a check there would read
    # as though they could.
    if (
        not 0 <= parsed.start <= len(pending)
        or not 0 <= removed_count <= len(pending) - parsed.start
    ):
        raise ValueError("invalid inbox splice")
    _unique_in(state, parsed.target, parsed.start, removed_count, parsed.inserted)
    pending[parsed.start : parsed.start + removed_count] = list(parsed.inserted)


def _unique_in(
    state: Mapping[str, Sequence[Message]],
    target: InboxTarget,
    start: int,
    removed_count: int,
    inserted: Sequence[Message],
) -> None:
    """Check that no message id would be pending twice, across both queues.

    This is the one rule that the write path and replay share. It takes messages
    and not a payload: `splice` holds the `Message` list that it just built, and
    has no reason to serialize it and read the ids back out.
    """
    pending = state[target]
    other: InboxTarget = "next-step" if target == "next-turn" else "next-turn"
    candidate = [
        *(m.id for m in pending[:start]),
        *(m.id for m in inserted),
        *(m.id for m in pending[start + removed_count :]),
        *(m.id for m in state[other]),
    ]
    if len(set(candidate)) != len(candidate):
        duplicate = next(i for i in candidate if candidate.count(i) > 1)
        raise ValueError(f'message "{duplicate}" is already pending')
