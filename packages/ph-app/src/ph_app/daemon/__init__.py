"""The daemon: roots that outlive the clients watching them (P5-01).

@module ph_app.daemon
"""

from __future__ import annotations

from .client import DaemonClient
from .server import DaemonServer, serve
from .supervisor import Root, Supervisor

__all__ = ["DaemonClient", "DaemonServer", "Root", "Supervisor", "serve"]
