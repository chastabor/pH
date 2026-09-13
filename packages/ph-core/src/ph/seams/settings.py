"""`ctx.settings` — durable user preferences, read as data.

Deliberately separate from profile rows. A row is what a *deployment* composed
and is versioned with the code; a setting is what a *user* chose and survives
across profiles. Conflating them means either a user edit gets clobbered by an
upgrade, or a deployment cannot change a default.

@module ph.seams.settings
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import anyio

from ..cordis import Context, plugin
from ..json import PlainJsonValue, as_obj, loads, thaw_json
from ..keys import SETTINGS
from ..paths import default_home_path, write_text_under
from ..wire import WireModel

__all__ = ["SettingsService", "apply"]

log = logging.getLogger("ph.seams.settings")


@dataclass(slots=True)
class SettingsService:
    """The service published as `ctx.settings`."""

    ctx: Context
    path: Path
    _values: dict[str, PlainJsonValue] = field(default_factory=dict)
    _loaded: bool = False

    def load(self) -> dict[str, PlainJsonValue]:
        """The whole settings tree, read once.

        `as_obj` is not decoration: a settings file whose top level is an array
        parses fine, and the object this returns is the one `set` walks with
        `.get`. Typed as the mapping it has to be, a non-object file is the empty
        tree — the same answer the unreadable case already gives — rather than a
        list that reads as defaults and raises on the first write.
        """
        if not self._loaded:
            try:
                self._values = thaw_json(as_obj(loads(self.path.read_text(encoding="utf-8"))))
            except FileNotFoundError:
                self._values = {}
            except (json.JSONDecodeError, OSError):
                # A corrupt settings file must not stop the harness starting;
                # defaults are always a valid answer for a preference.
                log.warning("ph.seams.settings: %s is unreadable; using defaults", self.path)
                self._values = {}
            self._loaded = True
        return self._values

    def get(self, key: str, default: PlainJsonValue = None) -> PlainJsonValue:
        """Read a dotted key."""
        node: PlainJsonValue = self.load()
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    async def set(self, key: str, value: PlainJsonValue) -> None:
        """Write a dotted key and persist. Not `object`: the tree is dumped below."""
        values = self.load()
        node = values
        parts = key.split(".")
        for part in parts[:-1]:
            existing = node.get(part)
            if not isinstance(existing, dict):
                existing = {}
                node[part] = existing
            node = existing
        node[parts[-1]] = value
        text = json.dumps(values, indent=2, ensure_ascii=False)
        await anyio.to_thread.run_sync(write_text_under, self.path, text)


class Config(WireModel):
    path: str | None = None


@plugin("settings-local", config=Config)
async def apply(ctx: Context, config: Config) -> None:
    """Mount the local settings store."""
    path = default_home_path(config.path, "settings.json")
    ctx.provide(SETTINGS, SettingsService(ctx=ctx, path=path))
