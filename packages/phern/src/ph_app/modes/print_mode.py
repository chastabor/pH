"""`phern -p "…"`: a one-shot question, a printed answer, and a JSONL log to inspect.

This is the smallest complete pH run: compose a profile, create a session, drive
one turn, print the assistant text. It is deliberately built on the same seams
that the TUI uses, so "does the harness work" and "does the front end work" stay
separate questions.

@module ph_app.modes.print_mode
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ph.cordis import Context, Profile
from ph.json import as_obj, as_str
from ph.keys import SESSION_PERSISTENCE
from ph.llm.types import text_of
from ph.persistence import unwritten_at_unwind
from ph.seams.models import ModelChoice
from ph.session import Session, derive_transcript

from ..runtime import prompted

__all__ = ["PrintResult", "run_print"]


@dataclass(slots=True)
class PrintResult:
    session_id: str
    text: str
    log_path: Path | None
    events: int
    ended: str
    """How the turn that this run drove finished. It gives the caller its exit
    code (C5).

    Before this field, a `-p` run whose turn ended in `error` printed whatever
    text had arrived and exited 0. A script that drove pH could not tell an
    answer from a provider outage: the failure was in the log, where a caller
    could not reach it. The field holds the kind and not a boolean, because
    `blocked` and `max-tokens` are not failures and a caller may still want to
    know which one it got.

    The type is a plain `str` and not the `Literal` of `TurnEndReason.kind`, for
    the same reason that the reader of the TUI and `Root.last_turn` both use
    plain strings. The value comes from a log, so it is whatever a past build
    wrote there, and a type that claimed otherwise would be a claim about every
    file on disk. The value is empty when the log has no `turn/end` at all.
    """

    failure: str
    """The message of the provider when `ended` is `error`. Empty otherwise."""

    unwritten: Mapping[str, str]
    """What the last write of the mount could not write, by session id, with the
    reason (`unwritten_at_unwind`). It is read after the mount is down, after
    everything above."""


async def run_print(
    profile: Profile,
    prompt: str,
    *,
    choice: ModelChoice = ModelChoice(),
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
) -> PrintResult:
    """Run one prompt to completion and return what the model said."""
    # `opened` is where the events of this run start. `prompted` resumes a
    # session that is already on disk, and the transcript is the *whole*
    # conversation. `phern -p --session x` therefore printed every answer that
    # the session had ever given, with the new one last, and grew by one
    # paragraph for each run. A one-shot run prints only what it produced.
    opened = 0

    def mark(_ctx: Context, session: Session) -> None:
        nonlocal opened
        opened = session.seq

    async with prompted(
        profile,
        prompt,
        choice=choice,
        session_id=session_id,
        attachments=attachments,
        before=mark,
    ) as (ctx, session):
        # This is the human transcript, not the model surface: what the user
        # was shown, with or without compaction.
        text = "\n".join(
            text_of(message.content)
            for message in derive_transcript(session.events_from(opened))
            if message.role == "assistant" and text_of(message.content)
        )
        persistence = ctx.get(SESSION_PERSISTENCE)
        # `locate` is used so that a backend with no per-session file reports
        # none, and not a path that nobody could open.
        log_path = None if persistence is None else persistence.locate(session.id)
        events = len(session.events)
        # The reason is read from the log and not from a listener. The log is
        # where the driver writes it, and a `finally` guarantees that it is
        # written: a turn that a cancellation killed still records how it
        # ended. A listener would have to survive the same cancellation to hear
        # it, so the listener would be the more fragile of the two.
        end = session.latest("turn/end")
        reason = as_obj(end.data.get("reason")) if end is not None else {}
    return PrintResult(
        session_id=session.id,
        text=text,
        log_path=log_path,
        events=events,
        ended=as_str(reason.get("kind")),
        failure=as_str(as_obj(reason.get("error")).get("message")),
        unwritten=unwritten_at_unwind(ctx),
    )
