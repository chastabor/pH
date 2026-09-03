"""Which project roots a person has trusted (P5-14).

**Out of the TUI because the daemon is what mounts.** The question is the front
end's to *ask* — it has the modal, and only it has a person — but the answer
gates reading a project's `AGENTS.md`, its hooks and its configured plugins, and
all three are read where the profile is mounted. A gate enforced only in the
client is one that any other client walks past: `ph agents send` naming a new
session in an untrusted checkout would load it with nobody asked.

So the file is shared, `session/new` consults it, and the TUI still shows the
modal. Two readers, one of which refuses.

@module ph_app.trust
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from ph.paths import resolve_roots, write_text_under

__all__ = ["TRUST_FILE", "TrustAnswer", "TrustStore", "trust_path"]

TrustAnswer: TypeAlias = Literal["", "once", "always"]
"""What a front end says the person answered.

A closed set, declared once and imported by both ends: the wire carries it, the
daemon branches on it, and the modal's actions are named for it — so a typo is a
type error rather than a mount that silently refuses. `""` is "nobody was asked",
which is every non-interactive client.
"""

TRUST_FILE = "trust.json"


def trust_path(home: Path | None = None) -> Path:
    """Where the trusted roots are recorded.

    One spelling, because two readers disagree otherwise: the TUI resolves `home`
    from its own injectable option and the daemon from `$PH_HOME`, so a client
    given a different home wrote a file the daemon never read — and the gate
    passed for a directory nobody had vouched for. The `tui_settings_path(home)`
    idiom, applied to the one file two processes share.
    """
    return (home if home is not None else resolve_roots().home) / TRUST_FILE


@dataclass(slots=True)
class TrustStore:
    """Which project roots the user has trusted, kept outside the project.

    Deliberately not stored in the project: a file inside the repository could
    declare the repository trustworthy, which is the one thing the prompt is
    supposed to prevent.
    """

    path: Path

    def trusted(self, root: Path) -> bool:
        return str(root.resolve()) in self._load()

    def trust(self, root: Path) -> None:
        roots = self._load()
        roots.add(str(root.resolve()))
        write_text_under(self.path, json.dumps({"trusted": sorted(roots)}, indent=2) + "\n")

    def _load(self) -> set[str]:
        try:
            record = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return set()
        trusted = record.get("trusted")
        return (
            {item for item in trusted if isinstance(item, str)}
            if isinstance(trusted, list)
            else set()
        )
