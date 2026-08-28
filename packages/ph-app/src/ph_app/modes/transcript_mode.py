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

from ph.llm.types import Message, ReasoningBlock, TextBlock, ToolCallBlock, ToolResultBlock, text_of

from ..runtime import prompted

__all__ = ["TranscriptResult", "render_transcript", "run_transcript"]


@dataclass(slots=True)
class TranscriptResult:
    session_id: str
    text: str


_SPEAKER = {"user": "you", "assistant": "pH", "system": "system"}


def render_transcript(messages: tuple[Message, ...]) -> str:
    """Render messages as a readable transcript."""
    lines: list[str] = []
    for message in messages:
        for block in message.content:
            if isinstance(block, TextBlock):
                source = getattr(message.source, "kind", "user")
                speaker = (
                    "context" if source == "plugin" else _SPEAKER.get(message.role, message.role)
                )
                lines.append(f"{speaker}: {block.text}")
            elif isinstance(block, ReasoningBlock):
                lines.append(f"pH (thinking): {block.text}")
            elif isinstance(block, ToolCallBlock):
                lines.append(f"pH → {block.name}({block.arguments})")
            elif isinstance(block, ToolResultBlock):
                body = text_of(block.content)
                marker = "!" if block.is_error else "←"
                lines.append(f"{marker} {body}")
    return "\n".join(lines)


async def run_transcript(
    documents: list[Path],
    prompt: str,
    *,
    provider: str,
    model: str,
    session_id: str | None = None,
    attachments: Sequence[Path] = (),
) -> TranscriptResult:
    async with prompted(
        documents,
        prompt,
        provider=provider,
        model=model,
        session_id=session_id,
        attachments=attachments,
    ) as (_ctx, session):
        return TranscriptResult(session_id=session.id, text=render_transcript(session.transcript()))
