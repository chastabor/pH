"""`ph.cordis` — the plugin meta-framework subset pH is built on (D1).

Everything in pH is a plugin; there is no privileged core to patch (invariant
I1). This package supplies the three things that makes possible: a context that
holds services, an event bus with four dispatch modes fixed by declaration, and
scopes whose disposal unwinds every registration and every acquired artifact
(invariant I2).
"""

from __future__ import annotations

from .context import (
    DEPLOYMENT,
    Boundary,
    Context,
    Deployment,
    Disposer,
    ForkScope,
    Hook,
    Listener,
    Running,
    boundary_of,
    chain_label,
    drop_dead_chains,
    is_bailed,
    maybe_await,
    running,
)
from .errors import (
    CordisError,
    EventModeError,
    InactiveScopeError,
    LoaderError,
    ServiceConflictError,
    ServiceNotFoundError,
    UndeclaredEventError,
)
from .events import DispatchMode, EventDeclaration, EventRegistry, events
from .loader import (
    ENTRY_POINT_GROUP,
    Loader,
    ProfileLayer,
    Row,
    compose_rows,
    import_plugin_modules,
    interpolate,
    resolve_plugin,
    safe_yaml_load,
)
from .plugin import PluginSpec, normalize_plugin, plugin

__all__ = [
    "DEPLOYMENT",
    "ENTRY_POINT_GROUP",
    "Boundary",
    "Context",
    "CordisError",
    "Deployment",
    "DispatchMode",
    "Disposer",
    "EventDeclaration",
    "EventModeError",
    "EventRegistry",
    "ForkScope",
    "Hook",
    "InactiveScopeError",
    "Listener",
    "Loader",
    "LoaderError",
    "PluginSpec",
    "ProfileLayer",
    "Row",
    "Running",
    "ServiceConflictError",
    "ServiceNotFoundError",
    "UndeclaredEventError",
    "boundary_of",
    "chain_label",
    "compose_rows",
    "drop_dead_chains",
    "events",
    "import_plugin_modules",
    "interpolate",
    "is_bailed",
    "maybe_await",
    "normalize_plugin",
    "plugin",
    "resolve_plugin",
    "running",
    "safe_yaml_load",
]
