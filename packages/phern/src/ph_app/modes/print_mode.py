"""`phern -p "…"` — one-shot question, printed answer, inspectable JSONL.

The smallest complete pH run: compose a profile, create a session, drive one
turn, print the assistant text. It is deliberately built on exactly the same
seams the TUI will use, so "does the harness work" and "does the front-end work"
stay separate questions.

@module ph_app.modes.print_mode
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from ph.cordis import Context, Profile
from ph.json import as_obj, as_str
from ph.keys import SESSION_PERSISTENCE
from ph.llm.types import text_of
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
    """How the turn this run drove finished — the caller's exit code (C5).

    A `-p` run whose turn ended in `error` printed whatever text had arrived and
    exited 0, so a script driving pH could not tell an answer from a provider
    outage: the failure was in the log and nowhere a caller could reach it. The
    kind rather than a boolean, because `blocked` and `max-tokens` are not
    failures and a caller may still want to know which one it got.

    A plain `str` and not the `TurnEndReason` Literal, for the reason the TUI's
    reader and `Supervisor.last_turn` are both plain strings: this comes off
    a log, so it is whatever a past build wrote there, and a type claiming
    otherwise would be a claim about every file on disk. Empty when the log has
    no `turn/end` at all.
    """

    failure: str
    """The provider's own words, when `ended` is `error` — and empty otherwise."""


async def run_print(
    profile: Profile,
    prompt: str,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
) -> PrintResult:
    """Run one prompt to completion and return what the model said."""
    # Where this run's own events start. `prompted` resumes a session that is
    # already on disk, and the transcript is the *whole* conversation — so
    # `phern -p --session x` printed every answer the session had ever given,
    # the new one last, growing by one paragraph per run. What a one-shot run
    # prints is what it produced.
    opened = 0

    def mark(_ctx: Context, session: Session) -> None:
        nonlocal opened
        opened = session.seq

    async with prompted(
        profile,
        prompt,
        provider=provider,
        model=model,
        session_id=session_id,
        attachments=attachments,
        before=mark,
    ) as (ctx, session):
        # The human transcript, not the model surface: what the user was shown,
        # compaction or not.
        text = "\n".join(
            text_of(message.content)
            for message in derive_transcript(session.events_from(opened))
            if message.role == "assistant" and text_of(message.content)
        )
        persistence = ctx.get(SESSION_PERSISTENCE)
        # Off the log rather than out of a listener, because that is where the
        # driver writes it and a `finally` is what guarantees it is written: a
        # turn killed by a cancellation still records how it ended, and a
        # listener that had to survive the same cancellation to hear it would be
        # the more fragile of the two.
        end = session.latest("turn/end")
        reason = as_obj(end.data.get("reason")) if end is not None else {}
        return PrintResult(
            session_id=session.id,
            text=text,
            # `locate`, so a backend with no per-session file reports none
            # rather than a path nobody could open.
            log_path=None if persistence is None else persistence.locate(session.id),
            events=len(session.events),
            ended=as_str(reason.get("kind")),
            failure=as_str(as_obj(reason.get("error")).get("message")),
        )
