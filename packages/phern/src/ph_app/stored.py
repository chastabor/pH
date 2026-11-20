"""A stored log, read with nothing mounted: what the views and exports over a file open.

The logs that are worth reading afterwards are often the ones that nobody can
reopen: a crashed run, the transcript of a child, a copy that someone sent. So
the trajectory viewer (`ph_app.tui.trajectory_app`) and the revision export
(`ph_app.revisions`) read the file itself, and its lineage, and do not ask a
daemon. This module does not import Textual, so a command that only reads a log
does not pay for a terminal UI.

@module ph_app.stored
"""

from __future__ import annotations

from pathlib import Path

from ph.paths import resolve_roots
from ph.persistence import materialize, read_session, repaired
from ph.persistence.jsonl import family_log, locate_session
from ph.session import Session, SessionEvent, SessionHeader

__all__ = ["load_session"]


def load_session(target: str, *, root: Path | None = None, home: Path | None = None) -> Session:
    """The log for a session id, or at a path to a stored log, with its lineage.

    A path is accepted so that a log outside the sessions directory (a fixture,
    a copy that someone sent) is readable without an install step first. `root`
    is the directory of the persistence row itself, when a caller knows it. The
    default is `sessions_dir()`, which is where that row puts logs unless a
    profile said otherwise.
    """
    candidate = Path(target)
    # A target that *looks* like a path is a path. When a missing file was
    # reinterpreted as a session id, the report said that `/nope.jsonl.jsonl`
    # was missing, which names a file that nobody typed.
    looks_like_path = candidate.suffix != "" or len(candidate.parts) > 1
    if looks_like_path:
        path = candidate
    else:
        roots = resolve_roots() if home is None else None
        base = root or (
            roots.sessions_dir() if roots is not None else Path(home or ".") / "sessions"
        )
        path = locate_session(base, target) or family_log(base / target, target)
    if not path.is_file():
        raise FileNotFoundError(f"no session log at {path}")
    header, events = _lineage_of(path)
    # `repaired`, as on a resume. A log whose tail was cut mid-turn is the main
    # case for the auditor, and a raw seed would refuse the very session that
    # someone opened this view to read.
    return Session(header.id, seed=repaired(events), header=header)


def _lineage_of(path: Path) -> tuple[SessionHeader, list[SessionEvent]]:
    """The log at `path`, plus whatever prefix it inherits (feasibility §5.3).

    **The auditor wants the lineage, not the segment.** A reference-forked child
    stores only its own events. The file alone would therefore show a history
    that begins in the middle, and `Session(seed=…)` would refuse it outright,
    because the seq of the first event would not be 0. The lineage is also what
    `f` forks from: the `source_seq` of a row is a position in the whole
    conversation, and not in whatever file happened to hold it.

    **The read goes through `materialize`, not through `JsonlSessionStore`**,
    even though the store is where `read` now does this. The store addresses a
    log as `<root>/<family>/<id>.jsonl`, and this view deliberately opens *any*
    path: a fixture or a copy that someone sent, as the docstring of
    `load_session` says. A read through the store would therefore quietly
    require a naming convention that the entry point does not have. The real
    risk of divergence was the lineage walk, and that is shared. `read_session`
    is the leaf read on both paths.

    Ancestors resolve by id **beside the target**. That is the only guess
    available with nothing mounted, and it is the right one for a lineage that
    was copied somewhere as a set. A chain whose ancestors are absent raises
    `LineageError`, which names the missing one. It does not render a partial
    history as if it were whole.
    """
    header, events = read_session(path)

    def read_one(
        session_id: str, upto: int | None = None, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        # The target is already read, and it is the only log whose *filename* is
        # known. Every log above it is addressed in the way that the store
        # addresses it.
        if session_id == header.id:
            return header, events
        return read_session(family_log(path.parent, session_id), upto=upto)

    return materialize(read_one, header.id)
