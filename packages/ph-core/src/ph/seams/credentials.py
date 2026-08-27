"""`ctx.credentials` — references travel, values do not (I-3).

The rule: **nothing above the adapter edge ever holds a secret value.** A
consumer asks for a `CredentialRef`, which is a name; only the adapter about to
build an HTTP request resolves it, and only into a local variable.

That is what makes the guarantee checkable rather than aspirational — a planted
`FOO_API_KEY` must not appear in any event, any fd-3 frame, or any child's
environment, and the test asserts exactly that over a whole run. A design that
passed values around would need every future plugin author to be careful; this
one needs the adapter edge to be.

`__repr__` is overridden on the resolved value for the same reason: a secret
that reaches a log via an exception traceback has still leaked.

@module ph.seams.credentials
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from ..cordis import Context, plugin
from ..wire import WireModel

__all__ = ["CredentialRef", "CredentialService", "SecretValue", "apply"]

log = logging.getLogger("ph.seams.credentials")


class CredentialRef(WireModel):
    """A name for a secret. Safe to log, store, and send to a child."""

    name: str
    source: str = "env"
    description: str | None = None


@dataclass(frozen=True, slots=True)
class SecretValue:
    """A resolved secret, wrapped so it cannot be printed by accident."""

    ref: CredentialRef
    _value: str = field(repr=False)

    def reveal(self) -> str:
        """The value. The only call site should be an outgoing request."""
        return self._value

    def __repr__(self) -> str:
        return f"SecretValue({self.ref.name}, <redacted>)"

    def __str__(self) -> str:
        return self.__repr__()


@dataclass(slots=True)
class CredentialService:
    """The service published as `ctx.credentials`."""

    ctx: Context
    _overrides: dict[str, str] = field(default_factory=dict)

    def reference(
        self, name: str, *, source: str = "env", description: str | None = None
    ) -> CredentialRef:
        """Mint a reference. Cheap, and safe to hand anywhere."""
        return CredentialRef(name=name, source=source, description=description)

    def provide_value(self, name: str, value: str) -> None:
        """Register a value in-process, for a test or an interactive login."""
        self._overrides[name] = value

    def has(self, ref: CredentialRef) -> bool:
        return ref.name in self._overrides or ref.name in os.environ

    def resolve(self, ref: CredentialRef) -> SecretValue | None:
        """Resolve a reference. Called at the adapter edge and nowhere else."""
        value = self._overrides.get(ref.name) or os.environ.get(ref.name)
        return None if value is None else SecretValue(ref=ref, _value=value)

    def require(self, ref: CredentialRef) -> SecretValue:
        resolved = self.resolve(ref)
        if resolved is None:
            raise KeyError(
                f'credential "{ref.name}" is not available; set the environment '
                "variable or provide it through ctx.credentials"
            )
        return resolved


@plugin("credentials-env")
async def apply(ctx: Context, config: Any) -> None:
    """Mount the environment-backed credential resolver."""
    ctx.provide("credentials", CredentialService(ctx=ctx))
