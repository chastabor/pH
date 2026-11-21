"""`ctx.credentials`: references travel, values do not (I-3).

The rule: **nothing above the adapter edge ever holds a secret value.** A
consumer asks for a `CredentialRef`, which is a name. Only the adapter that is
about to build an HTTP request resolves it, and only into a local variable.

That makes the guarantee one that can be checked, and not only intended. A
planted `FOO_API_KEY` must not appear in any event, any fd-3 frame, or the
environment of any child, and the test asserts exactly that over a whole run.
A design that passed values around would need every future plugin author to be
careful. This design needs only the adapter edge to be careful.

`__repr__` is overridden on the resolved value for the same reason: a secret
that reaches a log through an exception traceback has still leaked.

@module ph.seams.credentials
"""

from __future__ import annotations

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
        """The **names** that this process was handed, in the order in which
        they arrived.

        It returns names and never values. This class has one accessor that
        reveals a secret, and that accessor is `resolve`, at the adapter edge.

        It is for the picker, which lists what a deployment *names* in its
        profile. A credential that is typed into the free-text entry of the
        login screen is named nowhere. Without this method it vanished from the
        list at the moment that somebody set it. The environment is
        deliberately not enumerated with it, because the variables that a shell
        exports are not a list of the credentials of pH.
        """
        return tuple(self._overrides)

    def has(self, ref: CredentialRef) -> bool:
        return ref.name in self._overrides or ref.name in os.environ

    def resolve(self, ref: CredentialRef) -> SecretValue | None:
        """Resolve a reference. Called at the adapter edge and nowhere else."""
        value = self._overrides.get(ref.name) or os.environ.get(ref.name)
        return None if value is None else SecretValue(ref=ref, _value=value)


def missing_credential(ctx: Context, provider: str, model: str) -> str | None:
    """The credential that the route `provider`/`model` names and that `ctx`
    cannot supply.

    The result is `None` when the route needs no credential, when `ctx` can
    supply it, or when there is no adapter to ask. In the last case
    `resolve_model` names no credential. A route with no adapter fails for that
    reason, and loudly. To call it a missing key would send a user to look for
    the wrong thing.

    **Names only** (I-3). The adapter says which name it resolves at its edge
    (`ResolvedModel.credential`), and `has` answers whether a value exists
    without handing one over. With no credential seam mounted, nothing can be
    supplied, so the name is missing. This is the question behind the resume
    check (T5): can this session run here, now?
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
    """Hold `holder` for the credential that its route names when `ctx` cannot
    supply it, or release `holder` when `ctx` can. The log is made to say which
    (T5). Returns the name that `holder` waits for, or `None`.

    This is the one check behind every resume. The daemon makes it for the
    route of a root itself, and the subagent seam makes it for each child that
    it would readmit. The function emits no log message: the daemon says what
    waits, one time for each root, when the root comes back.
    """
    name = missing_credential(ctx, provider, model)
    await _record_wait(ctx, session, holder, name)
    return name


async def _record_wait(ctx: Context, session: Session, holder: str, name: str | None) -> None:
    """Make the log of `session` say what `holder` waits for now: `name`, or
    nothing (T5).

    A hold for any other name is settled, because its name arrived or the route
    stopped naming it. A hold for `name` is opened unless one is open already,
    so asking again with nothing changed appends nothing. Every open of a
    session can therefore ask, including a second restart before the key
    arrives, and the log does not grow by one record for each restart.

    The records go through `ctx.intents`, as records of `CREDENTIAL_WAIT`, and
    never through a direct append. They hold names only (I-3).
    """
    journal = intents_of(ctx)
    for claim in journal.held(session, CREDENTIAL_WAIT):
        waited_by, waited = hold_of(claim.opened)
        if waited_by == holder and waited != name:
            journal.settle(session, claim, credential_hold(holder, waited))
    if name is not None:
        await journal.open_once(session, CREDENTIAL_WAIT, credential_hold(holder, name))


def waiting_for(ctx: Context, session: Session) -> Mapping[str, str]:
    """Who in the log of `session` waits for which credential, as holder → name.

    There is one name for each holder, because `_record_wait` settles the other
    holds of a holder before it opens one. The answer is read from the cached
    index of the journal and is not folded again. So the doctor, the status of
    a root and its startup summary cost only what the log grew since.
    """
    return dict(
        hold_of(intent.opened) for intent in intents_of(ctx).pending(session, CREDENTIAL_WAIT)
    )


@plugin("credentials-env", affects="deployment")
async def apply(ctx: Context, config: None) -> None:
    """Mount the environment-backed credential resolver."""
    ctx.provide(CREDENTIALS, CredentialService(ctx=ctx))
