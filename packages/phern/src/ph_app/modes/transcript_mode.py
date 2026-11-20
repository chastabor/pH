"""`--mode transcript`: what the user saw, not what the model sees.

The mode reads `session.transcript()`, so a compacted conversation still shows
the turns that the human actually had. `derive_messages()` here would erase
those turns as soon as a compaction replaces them (Phase 4). The model surface
deliberately shadows replaced ranges, and that is the wrong projection for a
reader.

@module ph_app.modes.transcript_mode
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, assert_never

from ph.cordis import Profile
from ph.llm.types import (
    MediaBlock,
    Message,
    ReasoningBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultBlock,
    text_of,
)
from ph.persistence import unwritten_at_unwind
from ph.seams.models import ModelChoice
from ph.text import block_marker, redacted_marker

from ..runtime import prompted

__all__ = ["TranscriptResult", "render_transcript", "run_transcript"]


@dataclass(slots=True)
class TranscriptResult:
    session_id: str
    text: str
    unwritten: Mapping[str, str]
    """As `PrintResult.unwritten`."""


_SPEAKER: dict[Literal["system", "user", "assistant"], str] = {
    "user": "you",
    "assistant": "pH",
    "system": "system",
}
"""Keyed on the union of `Message.role` itself, so the type checker enforces that
the table covers it. A `.get(role, role)` default here could never fire. It is
the same dead defense that this round removed one line below."""


def render_transcript(messages: tuple[Message, ...]) -> str:
    """Render messages as a readable transcript."""
    lines: list[str] = []
    for message in messages:
        for block in message.content:
            match block:
                case TextBlock():
                    speaker = (
                        "context" if message.source.kind == "plugin" else _SPEAKER[message.role]
                    )
                    lines.append(f"{speaker}: {block.text}")
                case ReasoningBlock():
                    # Ciphertext is not something the assistant said (G8).
                    said = redacted_marker() if block.redacted else block.text
                    lines.append(f"pH (thinking): {said}")
                case ToolCallBlock():
                    lines.append(f"pH → {block.name}({block.arguments})")
                case ToolResultBlock():
                    body = text_of(block.content, placeholder=block_marker)
                    marker = "!" if block.is_error else "←"
                    lines.append(f"{marker} {body}")
                case MediaBlock():
                    lines.append(f"{_SPEAKER[message.role]}: {block_marker(block.type)}")
                case _ as unhandled:
                    assert_never(unhandled)
    return "\n".join(lines)


async def run_transcript(
    profile: Profile,
    prompt: str,
    *,
    choice: ModelChoice = ModelChoice(),
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
) -> TranscriptResult:
    async with prompted(
        profile,
        prompt,
        choice=choice,
        session_id=session_id,
        attachments=attachments,
    ) as (ctx, session):
        text = render_transcript(session.transcript())
    return TranscriptResult(session_id=session.id, text=text, unwritten=unwritten_at_unwind(ctx))
