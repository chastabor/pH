"""`ph.system_prompt` — prompt assembly (sections, contexts, tools, variables)."""

from __future__ import annotations

from .assembly import (
    AssembleContext,
    PromptAssembly,
    PromptContext,
    PromptSection,
    SystemPromptService,
    join_context_sections,
    render_context_sections,
    render_prompt,
)

__all__ = [
    "AssembleContext",
    "PromptAssembly",
    "PromptContext",
    "PromptSection",
    "SystemPromptService",
    "join_context_sections",
    "render_context_sections",
    "render_prompt",
]
