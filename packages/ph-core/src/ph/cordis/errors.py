"""Error vocabulary for `ph.cordis`. @module ph.cordis.errors"""

from __future__ import annotations

__all__ = [
    "CordisError",
    "EventModeError",
    "InactiveScopeError",
    "LoaderError",
    "MountRefusal",
    "ServiceConflictError",
    "ServiceNotFoundError",
    "UndeclaredEventError",
]


class CordisError(Exception):
    """Base class for every failure raised by the plugin meta-framework."""


class ServiceNotFoundError(CordisError, AttributeError):
    """A `ctx.<key>` read found no provider at or above the reading scope.

    Subclasses :class:`AttributeError` so ``getattr(ctx, key, default)`` and
    ``hasattr`` behave the way callers expect for an absent service.
    """


class ServiceConflictError(CordisError):
    """A second provider claimed a service key already held in the same realm."""


class InactiveScopeError(CordisError):
    """A registration was attempted on a scope that has already been disposed."""


class UndeclaredEventError(CordisError):
    """An event was dispatched or listened to without a matching declaration."""


class EventModeError(CordisError):
    """An event was dispatched through a method other than its declared mode."""


class LoaderError(CordisError):
    """A profile could not be composed into rows."""


class MountRefusal(CordisError, RuntimeError):
    """A row declined to apply on purpose, with a sentence for the user (E8).

    A refusal is distinct from a *bug* in an `apply`, which stays a traceback. A
    refusal is a row that says the deployment cannot honor what the profile
    asked of it: `containment.strict` on a host with no sandbox backend, or a
    telemetry exporter without its extra. Every host that mounts owes the user
    that sentence and an exit code, and not the stack. One type lets each host
    catch the refusal without knowing which row refused. Refuse at mount and not
    at first use. At first use the agent is already running, and "refuse to
    start" has already been disobeyed.

    `RuntimeError` stays a base, so that callers that caught the old spelling
    still catch this one.

    The class has a `code` because the CLI is not the only host that mounts. The
    daemon mounts a profile per root, and `respond` reads `code` from any raised
    instance and puts it in `data.reason`. Without a `code`, a profile that
    refuses would reach a TUI or a `phern agents` client as a generic failure
    that the client cannot tell from a mistyped method.
    `ph.persistence.lease.SessionBusy` names itself for the same reason.
    """

    code = "profile_refused"
