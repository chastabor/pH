"""Interactive login: a credential that is named, typed, and never written down.

`CredentialService.provide_value` exists for exactly this: "a test or an
interactive login". The only task of the modal is therefore to take a value for
the process that will hold it, which is the daemon. The modal deliberately does
not do three things: echo the value, put it in the session log, or write it to
disk. A secret that is typed here lives as long as that process and no longer.
That is the correct contract for a value that pH did not persist and cannot
rotate.

The candidate names come from the profile that the root is mounted on, and not
from a list that is kept here. An adapter row declares its own `apiKeyEnv`, so a
provider that a plugin adds appears without a change to this module (I7).

@module ph_app.tui.modals.login
"""

from __future__ import annotations

from collections.abc import Mapping

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.content import Content
from textual.widgets import Input, Static

from .base import Choice, PhModal

__all__ = ["LoginModal", "credential_choices"]


class LoginModal(PhModal[str | None]):
    """Take one secret. The input is masked, and the secret is gone when the
    process that holds it ends."""

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
                Content("Held in memory by the daemon only — not logged, not written to disk."),
                id="login-note",
            )
            yield Input(password=True, placeholder="paste the key", id="login-value")

    def on_mount(self) -> None:
        self.query_one("#login-value", Input).focus()

    async def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value.strip() or None)


def credential_choices(held: Mapping[str, bool]) -> list[Choice]:
    """The credentials that pH could use, with a mark on those that it already holds.

    The whole input is a `{name: held}` map, in the row order of the daemon.
    `credential_names` used to be here, and it walked the composed profile for
    `apiKeyEnv` keys. Every attached front end was therefore sent the entire
    configuration so that a modal could search it. `credentials_of` now answers
    both halves. See `ph_app.daemon.projections`.
    """
    return [
        Choice(value=name, label=name, detail="set" if is_held else "not set", marked=is_held)
        for name, is_held in held.items()
    ]
