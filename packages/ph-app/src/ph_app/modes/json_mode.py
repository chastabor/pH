"""`--mode json` — the session log, streamed, camelCase (I-7).

Not a rendering: the *same* envelopes the JSONL holds, emitted as they commit.
That is what makes the mode useful for a wrapper — a caller consuming this and a
caller reading the stored log parse one format, and dsh tooling reads both (Q2).

@module ph_app.modes.json_mode
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

from ph.cordis import Context
from ph.session import Session, SessionEvent, dumps

from ..runtime import prompted

__all__ = ["JsonResult", "run_json"]


@dataclass(slots=True)
class JsonResult:
    session_id: str
    events: int


async def run_json(
    documents: list[Path],
    prompt: str,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    out: TextIO | None = None,
) -> JsonResult:
    """Drive one prompt, emitting each committed event as one JSON line."""
    stream = out if out is not None else sys.stdout

    def attach(ctx: Context, session: Session) -> None:
        def emit(source: Session, event: SessionEvent) -> None:
            stream.write(f"{dumps(event.to_wire())}\n")
            stream.flush()

        ctx.on("session/event", emit)
        stream.write(f"{dumps({'type': 'session/header', 'header': session.header.to_wire()})}\n")

    async with prompted(
        documents, prompt, provider=provider, model=model, session_id=session_id, before=attach
    ) as (_ctx, session):
        return JsonResult(session_id=session.id, events=len(session.events))
