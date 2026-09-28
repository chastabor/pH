"""The agent inbox: durable, replayable pending input.

Three delivery semantics, ported exactly, because the difference is what makes
steering feel instant and injection feel invisible:

| call | lands at | wakes an idle agent |
|---|---|---|
| `followup(msg)` | next **turn** | yes |
| `steer(msg)` | next **step** | yes |
| `inject(msg)` | next **step** | **no** — it waits for another message |

Every mutation is logged as `agent/inbox/spliced` *before* the projection
changes, so a resumed agent reconstructs its queue from the log rather than
losing whatever the user typed before the crash.

**A claim is proposed, then taken** (S3). The driver has to decide about a batch —
`agent/pre-step` is an await, and a listener in it may make a model call whose
barrier flushes the log — before the batch reaches the conversation. Taken first,
a crash inside that await left the prompt out of the inbox on disk and nowhere in
the conversation. So `propose` says what a step would take without taking it, and
`claim` takes it in one `Session.batch()` with the records the caller appends beside
it: the splice that removes a prompt from the inbox and the `user/message` that puts
it in the conversation land together, or neither does.

@module ph.agent.inbox
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import ConfigDict

from ..json import JsonObject
from ..llm.types import Message
from ..session import Session, SessionBatch
from ..session.writers import log_writer
from ..wire import WireModel

_LOG = log_writer(__name__)

__all__ = ["Inbox", "InboxBatch", "InboxNotifications", "InboxTarget"]

InboxTarget: TypeAlias = Literal["next-turn", "next-step"]


@dataclass(frozen=True, slots=True)
class InboxNotifications:
    """Live mirrors of durable inbox mutations."""

    inserted: Callable[[Message], None]
    discarded: Callable[[Message], None]
    claimed: Callable[[Message, int], None]


@dataclass(frozen=True, slots=True)
class InboxBatch:
    """What one step would take from the inbox — proposed, and not yet taken.

    Built by `Inbox.propose` and taken by `Inbox.claim`. Holding one takes nothing:
    the messages stay pending, on disk and in the projection, until it is claimed.
    """

    steps: tuple[Message, ...]
    """Every `next-step` message pending when it was proposed."""
    turn: tuple[Message, ...]
    """At a turn boundary, the one queued turn it takes; otherwise empty."""

    @property
    def messages(self) -> tuple[Message, ...]:
        """The batch in the order the step is shown it."""
        return (*self.steps, *self.turn)


class InboxSplice(WireModel):
    """One inbox mutation, as `agent/inbox/spliced` carries it.

    Declared rather than hand-built and hand-parsed, the way every other durable
    payload is (`UserQuestion`, `RequestContext`, `Message`): `WireModel` owns the
    camelCase aliases and the log's frozen mapping, so `splice` writing a key and
    `_apply` reading one cannot drift, and `ph_app.tui.adapter` has something to
    validate against instead of re-spelling `inserted` and `removedCount`.

    `removed_count` is `None` rather than `0` when absent, and `outcome` likewise,
    because `to_wire()` omits `None` — an insert-only splice must not start
    emitting `"removedCount": 0` into logs that never carried it.

    `extra="ignore"` overrides `WireModel`'s `forbid`: a log is read by builds
    older than the one that wrote it, and a key added later must be skipped on
    replay rather than condemn the whole session.
    """

    model_config = ConfigDict(extra="ignore")

    target: InboxTarget
    start: int
    inserted: list[Message]
    removed_count: int | None = None
    outcome: Literal["canceled"] | None = None
    """Whether the removed messages were canceled or consumed. Written and not
    yet read back: live, the difference is `_notify.discarded` against
    `_notify.claimed`, and on replay this key is the only thing that still knows
    which happened."""


class Inbox:
    """A replay-once projection that incrementally consumes later splices."""

    __slots__ = ("_notify", "_session", "_state")

    def __init__(self, session: Session, notifications: InboxNotifications) -> None:
        self._session = session
        self._notify = notifications
        self._state: dict[str, list[Message]] = {"next-turn": [], "next-step": []}
        # Replay only this lifecycle's splices: a fork inherits its parent's
        # transcript, not its parent's unanswered queue.
        for event in session.events[session.header.seed_length or 0 :]:
            if event.type != "agent/inbox/spliced":
                continue
            try:
                self._apply(event.data)
            except ValueError as error:
                raise ValueError(
                    f"invalid persisted inbox splice at session seq {event.seq}"
                ) from error

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
        """The batch one step would take, without taking it.

        Always every pending `next-step` message, plus — at a turn boundary —
        exactly one queued turn. Claiming more than one turn would merge two
        user prompts into one model call.
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

        The splices are written into the batch the block receives, so they commit
        with the block's records or not at all: a block that raises, a batch the
        surface refuses, or a process that dies before the flush leaves every
        message pending. Only once the batch has committed does the projection
        change and `claimed` fire — after the durable event, as every splice here
        is, so a `session/event` observer still sees the lists as they were.

        Synchronous inside, as `Session.batch` requires.

        :raises RuntimeError: when the front of a queue is no longer what was
            proposed — something took it in between, which a single driver
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
        """`claim` with nothing beside it: the batch is consumed and reaches no step —
        a step a pre-step rejected, or one it emptied."""
        with self.claim(proposed, turn):
            pass

    def append(self, target: InboxTarget, message: Message) -> None:
        self.splice(target, len(self._state[target]), 0, [message])

    def splice(
        self, target: InboxTarget, start: int, delete_count: int, inserted: Sequence[Message]
    ) -> list[Message]:
        """Standard splice semantics, durably recorded; removed messages are canceled."""
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
        # The one rule a write can still break. The coordinates are clamped above
        # and the values are already typed, so there is nothing here to parse.
        self._unique(target, start, delete_count, inserted)
        # The durable event commits BEFORE the live projection mutates, so a
        # synchronous `session/event` observer sees the pre-splice lists and can
        # reconstruct exactly what was removed from the normalized coordinates.
        _LOG.append(self._session, "agent/inbox/spliced", splice.to_wire())
        removed = pending[start : start + delete_count]
        pending[start : start + delete_count] = inserted
        for message in removed:
            self._notify.discarded(message)
        for message in inserted:
            self._notify.inserted(message)
        return list(removed)

    # ------------------------------------------------------------- internals --

    def _apply(self, splice: JsonObject) -> None:
        """Replay one logged splice, or refuse the log."""
        parsed = InboxSplice.model_validate(splice)
        pending = self._state[parsed.target]
        removed_count = parsed.removed_count or 0
        self._placed(parsed.target, parsed.start, removed_count)
        self._unique(parsed.target, parsed.start, removed_count, parsed.inserted)
        pending[parsed.start : parsed.start + removed_count] = list(parsed.inserted)

    def _placed(self, target: InboxTarget, start: int, removed_count: int) -> None:
        """That the coordinates fall inside the queue they name.

        Replay only: `splice` clamps both into range before it writes, so on the
        write path these branches cannot fail and checking them there would read
        as though they could.
        """
        pending = self._state[target]
        if not 0 <= start <= len(pending) or not 0 <= removed_count <= len(pending) - start:
            raise ValueError("invalid inbox splice")

    def _unique(
        self, target: InboxTarget, start: int, removed_count: int, inserted: Sequence[Message]
    ) -> None:
        """That no message id would end up pending twice, across both queues.

        The one rule both paths share, so it takes messages rather than a payload:
        `splice` is holding the `Message` list it just built and has no reason to
        serialize it and read the ids back out.
        """
        pending = self._state[target]
        other: InboxTarget = "next-step" if target == "next-turn" else "next-turn"
        candidate = [
            *(m.id for m in pending[:start]),
            *(m.id for m in inserted),
            *(m.id for m in pending[start + removed_count :]),
            *(m.id for m in self._state[other]),
        ]
        if len(set(candidate)) != len(candidate):
            duplicate = next(i for i in candidate if candidate.count(i) > 1)
            raise ValueError(f'message "{duplicate}" is already pending')
