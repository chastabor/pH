"""`$PH_HOME/daemon.yaml`: the configuration of the host itself (decision 23).

Configuration has three owners, one for each kind of row
(`ph.cordis.plugin.Affects`). The profile of a session owns what the agent works
in. The `tui.json` of the TUI owns how a front end draws. This file owns the
machinery of the host: persistence, telemetry, the job bound, and the
invariants. It also owns where the other two are kept:

```yaml
paths:
  sessions: ~/.ph/sessions
  profiles: ~/.ph/profiles
rows:
  - id: jobs
    config:
      concurrency: {subagent: 8}
```

`rows` is a profile document in the profile grammar. It is composed after the
shipped layers, and any row in it that is not `deployment` is refused. This
module does not parse it. `compose_rows` owns that grammar, and a second reader
of the grammar would in time accept different things.

**The file is read one time in each process.** All roots of a daemon mount
against the directories that the daemon started with, so an edit takes effect at
the next start. A changed named profile follows the same rule (decision 17), for
the same reason. A daemon that read its configuration again while it ran would
put the logs of two roots in two places, although nobody restarted anything.

**Nothing in the file is a secret.** Credentials have their own seam. This file
is plain configuration that a user can show to another person.

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

    Each path is `~`-expanded, and a relative path is under `$PH_HOME`. An unset
    path means the directory of that name under `$PH_HOME`, which is where both
    directories have always been.
    """

    sessions: str | None = None
    """Every session log. The persistence rows default to this directory. So
    does the browse of the daemon, before its first root has mounted a store
    that it can ask."""

    profiles: str | None = None
    """The named profiles that a user wrote, and the drop-ins that pH wrote
    beside them before S4. pH reads those drop-ins until `phern profiles fold`
    folds them, and writes none now."""


@dataclass(frozen=True, slots=True)
class HostConfig:
    """`daemon.yaml` after it is read: the paths block, and the rows that it sets."""

    paths: HostPaths = field(default_factory=HostPaths)
    rows: JsonValue = None
    """A profile document, still raw. See the module docstring."""


def host_config_path(home: Path) -> Path:
    return home / "daemon.yaml"


@cache
def load_host_config(home: Path) -> HostConfig:
    """Read `$PH_HOME/daemon.yaml`, or return the empty configuration when there
    is no file.

    The file is refused whole when any part is wrong, and is not applied in
    part. Otherwise a paths block that did not parse would move the sessions
    directory back to its default, and a daemon would start to write logs where
    nothing looks.
    """
    path = host_config_path(home)
    if not path.is_file():
        return HostConfig()
    # Strict, and not the leniency that `read_document` has for preference files.
    # A host file that did not read is a daemon that must not start, and this is
    # the sentence for why.
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
