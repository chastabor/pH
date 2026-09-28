"""`$PH_HOME/daemon.yaml` — the host's own configuration (decision 23).

Configuration has three owners, one per kind of row (`ph.cordis.plugin.Affects`):
a session's profile owns what the agent works in, the TUI's `tui.json` owns how a
front end draws, and this file owns the host's machinery — persistence,
telemetry, the job bound, the invariants — and where the other two are kept:

```yaml
paths:
  sessions: ~/.ph/sessions
  profiles: ~/.ph/profiles
rows:
  - id: jobs
    config:
      concurrency: {subagent: 8}
```

`rows` is a profile document in the profile grammar, composed after the shipped
layers and refused any row that is not `deployment`. It is not parsed here:
`compose_rows` owns that grammar, and a second reader of it is how the two come
to accept different things.

**Read once per process.** A daemon's roots all mount against the directories it
started with, so an edit takes effect at the next start — the rule a changed
named profile follows too (decision 17), and for the same reason: a daemon that
re-read its configuration while running would put two roots' logs in two places
without anybody having restarted anything.

**Nothing in it is a secret.** Credentials have their own seam; this file is
plain configuration a person can show someone else.

@module ph.host
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from pydantic import ValidationError

from .cordis import LoaderError
from .documents import decode_document
from .json import JsonValue
from .wire import WireModel, validation_summary

__all__ = ["HostConfig", "HostPaths", "host_config_path", "load_host_config"]


class HostPaths(WireModel):
    """Where the host keeps what the other two configurations write.

    Each is `~`-expanded, and a relative one is under `$PH_HOME`. Unset is the
    directory of that name under `$PH_HOME`, which is where both have always been.
    """

    sessions: str | None = None
    """Every session log. The persistence rows default to it, and so does the
    daemon's browse before its first root has mounted a store to ask."""

    profiles: str | None = None
    """Named profiles a person wrote, and the drop-ins pH writes beside them."""


@dataclass(frozen=True, slots=True)
class HostConfig:
    """`daemon.yaml`, read: the paths block, and the rows it sets."""

    paths: HostPaths = field(default_factory=HostPaths)
    rows: JsonValue = None
    """A profile document, still raw — see the module docstring."""


def host_config_path(home: Path) -> Path:
    return home / "daemon.yaml"


@cache
def load_host_config(home: Path) -> HostConfig:
    """`$PH_HOME/daemon.yaml`, or the empty configuration when there is none.

    Refused whole when any part is wrong, rather than applied in part: a paths
    block that did not parse would otherwise move the sessions directory back to
    its default and a daemon would start writing logs where nothing looks.
    """
    path = host_config_path(home)
    if not path.is_file():
        return HostConfig()
    # Strict, not `read_document`'s preference-file leniency: a host file that did
    # not read is a daemon that must not start, and this is the sentence for why.
    raw = decode_document(path)
    if raw is None:
        return HostConfig()
    if not isinstance(raw, Mapping):
        raise LoaderError(f"{path}: expected a mapping with `paths:` and `rows:`")
    unknown = sorted(set(raw) - {"paths", "rows"})
    if unknown:
        raise LoaderError(f"{path}: unknown keys {unknown}; the file holds `paths:` and `rows:`")
    try:
        paths = HostPaths.model_validate(raw.get("paths") or {})
    except ValidationError as error:
        said = validation_summary(error, root="the block")
        raise LoaderError(f"{path}: paths: {said}") from error
    return HostConfig(paths=paths, rows=raw.get("rows"))
