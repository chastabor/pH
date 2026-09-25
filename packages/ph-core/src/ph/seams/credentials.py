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
from collections.abc import Mapping
from dataclasses import dataclass, field

from ..cordis import Context, plugin
from ..keys import CREDENTIALS, LLM
from ..session import Session, intents_of
from ..session.kinds import CREDENTIAL_WAIT, credential_hold, hold_of
from ..wire import WireModel

__all__ = [
    "CredentialRef",
    "CredentialService",
    "SecretValue",
    "apply",
    "hold_for_credential",
    "missing_credential",
    "waiting_for",
]

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

    def provided(self) -> tuple[str, ...]:
        """The **names** this process was handed, in the order they arrived.

        Names and never values — this class has one accessor that reveals a
        secret and it is `resolve`, at the adapter edge.

        For the picker, which lists what a deployment *names* in its profile: a
        credential typed into the login screen's free-text entry is named
        nowhere, so without this it vanished from the list the moment it was
        stored. The environment is deliberately not enumerated with it — every
        variable a shell exports is not a list of pH's credentials.
        """
        return tuple(self._overrides)

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


def missing_credential(ctx: Context, provider: str, model: str) -> str | None:
    """The credential the route `provider`/`model` names and `ctx` cannot supply.

    `None` when the route needs none, when it can be supplied, or when there is no
    adapter for it to ask — `resolve_model` then names no credential, and a route with
    no adapter fails for that reason, loudly; calling it a missing key would send a
    person looking for the wrong thing.

    **Names only** (I-3): the adapter says which name it resolves at its edge
    (`ResolvedModel.credential`), and `has` answers whether a value exists without
    handing one over. With no credential seam mounted, nothing can be supplied, so
    the name is missing. The question behind the resume check (T5): can this
    session run here, now?
    """
    llm = ctx.get(LLM)
    if llm is None:
        return None
    name = llm.resolve_model(provider, model).credential
    if not name:
        return None
    credentials = ctx.get(CREDENTIALS)
    if credentials is not None and credentials.has(credentials.reference(name)):
        return None
    return name


async def hold_for_credential(
    ctx: Context, session: Session, holder: str, provider: str, model: str
) -> str | None:
    """Hold `holder` on the credential its route names if `ctx` cannot supply it, or
    release it if it can — the log made to say which (T5). Returns the name it waits
    for, or `None`.

    The one check behind every resume: the daemon's for a root's own route, the
    subagent seam's for each child it would readmit. Logs nothing: the daemon says
    what waits, once per root, when the root comes back.
    """
    name = missing_credential(ctx, provider, model)
    await _record_wait(ctx, session, holder, name)
    return name


async def _record_wait(ctx: Context, session: Session, holder: str, name: str | None) -> None:
    """Make `session`'s log say what `holder` waits for now: `name`, or nothing (T5).

    A hold for any other name is settled — its name arrived, or the route stopped
    naming it — and a hold for `name` is opened unless one is open already, so
    asking again with nothing changed appends nothing. That is what lets every open
    of a session ask, a second restart before the key arrives included, without the
    log growing a record per restart.

    Through `ctx.intents`, as `CREDENTIAL_WAIT`'s records, never a direct append;
    names only (I-3).
    """
    journal = intents_of(ctx)
    for claim in journal.held(session, CREDENTIAL_WAIT):
        waited_by, waited = hold_of(claim.opened)
        if waited_by == holder and waited != name:
            journal.settle(session, claim, credential_hold(holder, waited))
    if name is not None:
        await journal.open_once(session, CREDENTIAL_WAIT, credential_hold(holder, name))


def waiting_for(ctx: Context, session: Session) -> Mapping[str, str]:
    """Who in `session`'s log waits for which credential: holder → name.

    One name per holder, since `_record_wait` settles a holder's other holds before
    it opens one. Read off the journal's cached index rather than refolded, so the
    doctor, a root's status and its startup summary cost what the log grew since.
    """
    return dict(
        hold_of(intent.opened) for intent in intents_of(ctx).pending(session, CREDENTIAL_WAIT)
    )


@plugin("credentials-env")
async def apply(ctx: Context, config: None) -> None:
    """Mount the environment-backed credential resolver."""
    ctx.provide(CREDENTIALS, CredentialService(ctx=ctx))
