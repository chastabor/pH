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

@module ph.agent.inbox
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

from ..llm.types import Message
from ..session import Session

__all__ = ["Inbox", "InboxNotifications", "InboxTarget"]

InboxTarget: TypeAlias = Literal["next-turn", "next-step"]


@dataclass(frozen=True, slots=True)
class InboxNotifications:
    """Live mirrors of durable inbox mutations."""

    inserted: Callable[[Message], None]
    discarded: Callable[[Message], None]
    claimed: Callable[[Message, int], None]


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

    def claim(self, target: InboxTarget, turn: int) -> list[Message]:
        """Take the batch proposed for one step.

        Always every pending `next-step` message, plus — at a turn boundary —
        exactly one queued turn. Claiming more than one turn would merge two
        user prompts into one model call.
        """
        claimed = self._mutate("next-step", 0, len(self._state["next-step"]), [], False)
        if target == "next-turn":
            claimed.extend(self._mutate("next-turn", 0, 1, [], False))
        for message in claimed:
            self._notify.claimed(message, turn)
        return claimed

    def append(self, target: InboxTarget, message: Message) -> None:
        self.splice(target, len(self._state[target]), 0, [message])

    def splice(
        self, target: InboxTarget, start: int, delete_count: int, inserted: Sequence[Message]
    ) -> list[Message]:
        """Standard splice semantics, durably recorded; removed messages are cancelled."""
        return self._mutate(target, start, delete_count, list(inserted), True)

    # ------------------------------------------------------------- internals --

    def _mutate(
        self,
        target: InboxTarget,
        start: int,
        delete_count: int,
        inserted: list[Message],
        discard_removed: bool,
    ) -> list[Message]:
        pending = self._state[target]
        start = min(max(start, 0), len(pending))
        delete_count = min(max(delete_count, 0), len(pending) - start)
        if delete_count == 0 and not inserted:
            return []
        splice: dict[str, Any] = {
            "target": target,
            "start": start,
            "inserted": [message.to_wire() for message in inserted],
        }
        if delete_count:
            splice["removedCount"] = delete_count
            if discard_removed:
                splice["outcome"] = "canceled"
        self._validate(splice)
        # The durable event commits BEFORE the live projection mutates, so a
        # synchronous `session/event` observer sees the pre-splice lists and can
        # reconstruct exactly what was removed from the normalized coordinates.
        self._session.append("agent/inbox/spliced", splice)
        removed = pending[start : start + delete_count]
        pending[start : start + delete_count] = inserted
        if discard_removed:
            for message in removed:
                self._notify.discarded(message)
        for message in inserted:
            self._notify.inserted(message)
        return list(removed)

    def _apply(self, splice: Any) -> None:
        self._validate(splice)
        pending = self._state[splice["target"]]
        start = splice["start"]
        removed_count = splice.get("removedCount", 0)
        pending[start : start + removed_count] = [
            Message.model_validate(item) for item in splice["inserted"]
        ]

    def _validate(self, splice: Any) -> None:
        target = splice.get("target")
        if target not in ("next-turn", "next-step"):
            raise ValueError("invalid inbox splice target")
        pending = self._state[target]
        start = splice.get("start")
        removed_count = splice.get("removedCount", 0)
        if (
            not isinstance(start, int)
            or not 0 <= start <= len(pending)
            or not isinstance(removed_count, int)
            or not 0 <= removed_count <= len(pending) - start
        ):
            raise ValueError("invalid inbox splice")
        other = "next-step" if target == "next-turn" else "next-turn"
        candidate = [
            *(m.id for m in pending[:start]),
            *(item["id"] for item in splice["inserted"]),
            *(m.id for m in pending[start + removed_count :]),
            *(m.id for m in self._state[other]),
        ]
        if len(set(candidate)) != len(candidate):
            duplicate = next(i for i in candidate if candidate.count(i) > 1)
            raise ValueError(f'message "{duplicate}" is already pending')
