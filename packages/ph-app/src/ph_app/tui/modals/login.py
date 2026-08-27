"""Interactive login: a credential named, typed, and never written down.

`CredentialService.provide_value` exists for exactly this — "a test or an
interactive login" — so the modal's whole job is to get a value into the
process. Three things it deliberately does not do: echo the value, put it in the
session log, or write it to disk. A secret typed here lives as long as the
process and no longer, which is the honest contract for a value pH did not
persist and cannot rotate.

The candidate names come from the composed configuration rather than a list kept
here: an adapter row declares its own `apiKeyEnv`, so a provider added by a
plugin shows up without this module changing (I7).

@module ph_app.tui.modals.login
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Input, Static

from .base import Choice, PhModal

__all__ = ["LoginModal", "credential_choices", "credential_names"]

CREDENTIAL_CONFIG_KEY = "apiKeyEnv"
"""What an adapter row calls the environment variable holding its key."""


class LoginModal(PhModal[str | None]):
    """Take one secret. Masked, and gone when the process ends."""

    DEFAULT_CSS = """
    LoginModal { align: center middle; background: $background 60%; }
    LoginModal > #login {
        width: 68; max-width: 90%; height: auto;
        background: $ph-panel; border: round $ph-accent; padding: 1 2;
    }
    LoginModal #login-title { color: $ph-accent; }
    LoginModal #login-note { color: $ph-muted; margin: 0 0 1 0; }
    LoginModal #login-value { border: none; background: $ph-surface; }
    """

    def __init__(self, credential: str) -> None:
        super().__init__()
        # Not `self.name`: `DOMNode.name` is a read-only property.
        self.credential = credential

    def compose(self) -> ComposeResult:
        with Vertical(id="login"):
            yield Static(
                Content.from_markup("[b]$name[/b]", name=self.credential), id="login-title"
            )
            yield Static(
                Content("Held in this process only — not logged, not written to disk."),
                id="login-note",
            )
            yield Input(password=True, placeholder="paste the key", id="login-value")

    def on_mount(self) -> None:
        self.query_one("#login-value", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value.strip() or None)


def credential_names(rows: Iterable[Any]) -> list[str]:
    """Every credential the composed configuration names, in row order.

    Walks a row's config rather than matching on plugin names: the key is
    declared, so an adapter pH has never heard of is still covered.
    """
    found: list[str] = []
    for row in rows:
        for name in _walk(row.get("config") if isinstance(row, Mapping) else None):
            if name not in found:
                found.append(name)
    return found


def _walk(value: Any) -> Iterable[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == CREDENTIAL_CONFIG_KEY and isinstance(item, str) and item:
                yield item
            else:
                yield from _walk(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk(item)


def credential_choices(rows: Iterable[Any], credentials: Any) -> list[Choice]:
    """The credentials pH could use, marked with the ones it already has."""
    choices: list[Choice] = []
    for name in credential_names(rows):
        held = bool(credentials is not None and credentials.has(credentials.reference(name)))
        choices.append(
            Choice(value=name, label=name, detail="set" if held else "not set", marked=held)
        )
    return choices
