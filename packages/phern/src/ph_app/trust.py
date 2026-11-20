"""Which project roots the user has trusted (P5-14).

**This module is outside the TUI because the daemon is what mounts.** The front
end *asks* the question: it has the modal, and only it has a user. But the
answer gates the reading of the `AGENTS.md` of a project, its hooks and its
configured plugins, and all three are read where the profile is mounted. A gate
that only the client enforces is a gate that any other client walks past.
`phern agents send` with a new session in an untrusted checkout would load the
project with nobody asked.

So the file is shared, `session/new` consults it, and the TUI still shows the
modal. There are two readers, and one of them refuses.

@module ph_app.trust
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypeAlias

from ph.paths import resolve_roots, write_atomic

__all__ = ["TRUST_FILE", "TrustAnswer", "TrustStore", "trust_path"]

TrustAnswer: TypeAlias = Literal["once", "always"]
"""What a front end reports as the answer of the user.

This is a closed set, declared one time and imported by both ends. The wire
carries it, the daemon branches on it, and the actions of the modal are named
for it. A typo is therefore a type error and not a mount that silently refuses.

"Nobody was asked" is not an answer, so it is not in the set. A client that has
not shown the modal sends no answer (`NewSessionParams.trust` is `None`), and
the daemon then checks the store. That is the case for every non-interactive
client.
"""

TRUST_FILE = "trust.json"


def trust_path(home: Path | None = None) -> Path:
    """Where the trusted roots are recorded.

    There is one spelling, because two readers disagree otherwise. The TUI
    resolves `home` from its own injectable option, and the daemon resolves it
    from `$PH_HOME`. A client that was given a different home therefore wrote a
    file that the daemon never read, and the gate passed for a directory that
    nobody had vouched for. This is the `tui_settings_path(home)` idiom, applied
    to the one file that two processes share.
    """
    return (home if home is not None else resolve_roots().home) / TRUST_FILE


@dataclass(slots=True)
class TrustStore:
    """Which project roots the user has trusted, kept outside the project.

    The record is deliberately not stored in the project. A file inside the
    repository could declare the repository trustworthy, and that is the one
    thing that the prompt must prevent.
    """

    path: Path

    def trusted(self, root: Path) -> bool:
        return str(root.resolve()) in self._load()

    def trust(self, root: Path) -> None:
        roots = self._load()
        roots.add(str(root.resolve()))
        # The write is atomic (O2). A second TUI and the daemon read this file
        # without coordination. A truncating write leaves a window in which the
        # file is neither the old set nor the new one. An unparseable trust file
        # that is read in that window is an untrusted root.
        write_atomic(self.path, json.dumps({"trusted": sorted(roots)}, indent=2) + "\n")

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
