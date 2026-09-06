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
    """A row declined to apply — on purpose, with a sentence for a person (E8).

    Distinct from a *bug* in an `apply`, which stays a traceback. A refusal is a
    row saying the deployment cannot honour what the profile asked of it:
    `containment.strict` on a host with no sandbox backend, a telemetry exporter
    without its extra. Every host that mounts owes the person that sentence and an
    exit code rather than the stack, and one type is what lets each of them catch
    it without knowing which row refused. Refuse at mount, not at first use — by
    then the agent is running and "refuse to start" has already been disobeyed.

    `RuntimeError` stays a base so callers that caught the old spelling still do.

    `code` because the CLI is not the only host that mounts: the daemon mounts a
    profile per root, and `respond` reads `code` off any raised instance to put
    it in `data.reason` — so without one a profile that refuses reaches a TUI or
    an `ph agents` client as a generic failure it cannot tell from a mistyped
    method. `ph.persistence.lease.SessionBusy` names itself for the same reason.
    """

    code = "profile_refused"
