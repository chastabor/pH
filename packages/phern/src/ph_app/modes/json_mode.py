"""`--mode json`: the session log, streamed, in camelCase (I-7).

This mode is not a rendering. It emits the *same* envelopes that the JSONL
holds, as they commit. That makes the mode useful for a wrapper: a caller that
consumes this stream and a caller that reads the stored log parse one format,
and dsh tooling reads both (Q2).

@module ph_app.modes.json_mode
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ph.cordis import Context, Profile
from ph.json import dumps
from ph.persistence import unwritten_at_unwind
from ph.seams.models import ModelChoice
from ph.session import Session, SessionEvent

from ..runtime import prompted

__all__ = ["JsonResult", "run_json"]


@dataclass(slots=True)
class JsonResult:
    session_id: str
    events: int
    unwritten: Mapping[str, str]
    """As `PrintResult.unwritten`."""


async def run_json(
    profile: Profile,
    prompt: str,
    *,
    choice: ModelChoice = ModelChoice(),
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
    out: TextIO | None = None,
) -> JsonResult:
    """Drive one prompt, emitting each committed event as one JSON line."""
    stream = out if out is not None else sys.stdout

    def attach(ctx: Context, session: Session) -> None:
        def emit(source: Session, event: SessionEvent) -> None:
            stream.write(f"{dumps(event.to_wire())}\n")
            stream.flush()

        stream.write(f"{dumps({'type': 'session/header', 'header': session.header.to_wire()})}\n")
        # Emit what this run committed before there was a stream to write to.
        # Opening a session records its base first (S3), so the log of a new
        # session begins before this listener does. Everything after the last
        # `session/resumed` belongs to this run, and that is all of a new log.
        # The stream is the log.
        for event in _opened_here(session):
            emit(session, event)
        ctx.on("session/event", emit)

    async with prompted(
        profile,
        prompt,
        choice=choice,
        session_id=session_id,
        attachments=attachments,
        before=attach,
    ) as (ctx, session):
        pass
    # The events are counted after teardown, not inside it. Scopes unwind on the
    # way out, and what they append (a released workspace, a reaped child) is
    # streamed like any other event. A count taken one line earlier would report
    # fewer events than the reader just saw.
    return JsonResult(
        session_id=session.id, events=len(session.events), unwritten=unwritten_at_unwind(ctx)
    )


def _opened_here(session: Session) -> tuple[SessionEvent, ...]:
    """Return the events that the open of this run committed: those after the
    history that it resumed, if any."""
    resumed = session.last_event_of("session/resumed")
    return session.events_from(resumed.seq + 1 if resumed is not None else 0)
