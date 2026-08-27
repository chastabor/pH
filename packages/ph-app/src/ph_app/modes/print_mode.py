"""`ph -p "…"` — one-shot question, printed answer, inspectable JSONL.

The smallest complete pH run: compose a profile, create a session, drive one
turn, print the assistant text. It is deliberately built on exactly the same
seams the TUI will use, so "does the harness work" and "does the front-end work"
stay separate questions.

@module ph_app.modes.print_mode
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ph.llm.types import text_of
from ph.persistence import session_path

from ..runtime import prompted

__all__ = ["PrintResult", "run_print"]


@dataclass(slots=True)
class PrintResult:
    session_id: str
    text: str
    log_path: Path | None
    events: int


async def run_print(
    documents: list[Path],
    prompt: str,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
) -> PrintResult:
    """Run one prompt to completion and return what the model said."""
    async with prompted(
        documents, prompt, provider=provider, model=model, session_id=session_id
    ) as (ctx, session):
        # The human transcript, not the model surface: what the user was shown,
        # compaction or not.
        text = "\n".join(
            text_of(message.content)
            for message in session.transcript()
            if message.role == "assistant" and text_of(message.content)
        )
        persistence = ctx.get("session_persistence")
        return PrintResult(
            session_id=session.id,
            text=text,
            log_path=None if persistence is None else session_path(persistence.root, session.id),
            events=len(session.events),
        )
