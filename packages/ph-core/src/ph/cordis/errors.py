"""Error vocabulary for `ph.cordis`. @module ph.cordis.errors"""

from __future__ import annotations

__all__ = [
    "CordisError",
    "EventModeError",
    "InactiveScopeError",
    "LoaderError",
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
