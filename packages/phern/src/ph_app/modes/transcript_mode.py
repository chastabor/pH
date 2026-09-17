"""`--mode transcript` — what a person saw, not what the model sees.

Reads `session.transcript()`, so a compacted conversation still shows the turns
the human actually had. Using `derive_messages()` here would erase them the
moment compaction lands (Phase 4) — the model surface deliberately shadows
replaced ranges, and that is the wrong projection for a reader.

@module ph_app.modes.transcript_mode
"""

from __future__ import annotations

from collections.abc import Sequence
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
from ph.text import block_marker

from ..runtime import prompted

__all__ = ["TranscriptResult", "render_transcript", "run_transcript"]


@dataclass(slots=True)
class TranscriptResult:
    session_id: str
    text: str


_SPEAKER: dict[Literal["system", "user", "assistant"], str] = {
    "user": "you",
    "assistant": "pH",
    "system": "system",
}
"""Keyed on `Message.role`'s own union, so the checker enforces that the table
covers it. A `.get(role, role)` default here could never fire — the same dead
defense this round removed one line below."""


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
                    lines.append(f"pH (thinking): {block.text}")
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
    provider: str,
    model: str,
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
) -> TranscriptResult:
    async with prompted(
        profile,
        prompt,
        provider=provider,
        model=model,
        session_id=session_id,
        attachments=attachments,
    ) as (_ctx, session):
        return TranscriptResult(session_id=session.id, text=render_transcript(session.transcript()))
