"""A stored log, read with nothing mounted: what the views and exports over a file open.

The logs worth reading after the fact are often the ones nobody can reopen — a crashed
run, a child's transcript, a copy someone sent — so the trajectory viewer
(`ph_app.tui.trajectory_app`) and the revision export (`ph_app.revisions`) read the file
itself, and its lineage, rather than asking a daemon. Textual-free, so a command that
only reads a log does not pay for a terminal UI.

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

    A path so a log outside the sessions directory — a fixture, a copy someone
    sent — is readable without being installed anywhere first. `root` is the
    persistence row's own directory when a caller knows it; the default is
    `sessions_dir()`, which is where that row puts logs unless a profile said
    otherwise.
    """
    candidate = Path(target)
    # A target that *looks* like a path is one: reinterpreting a missing file as
    # a session id reported `/nope.jsonl.jsonl` missing, naming a file nobody
    # typed.
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
    # `repaired`, like a resume: a log whose tail was cut mid-turn is the
    # auditor's headline case, and seeding it raw would refuse the very session
    # someone opened this view to read.
    return Session(header.id, seed=repaired(events), header=header)


def _lineage_of(path: Path) -> tuple[SessionHeader, list[SessionEvent]]:
    """The log at `path`, plus whatever prefix it inherits (feasibility §5.3).

    **The auditor wants the lineage, not the segment.** A reference-forked child
    stores only its own events, so reading the file alone would show a history
    beginning in the middle — and `Session(seed=…)` would refuse it outright,
    since the first event's seq would not be 0. It is also what `f` forks from:
    a row's `source_seq` is a position in the whole conversation, not in whatever
    file happened to hold it.

    **Through `materialize`, not through `JsonlSessionStore`**, even though the
    store is where `read` now does this. The store addresses a log as
    `<root>/<id>.jsonl`, and this view deliberately opens *any* path — "a
    fixture, a copy someone sent", per this module's own docstring — so routing
    through it would quietly require a naming convention the entry point does not
    have. What actually risked divergence was the lineage walk, and that is
    shared; `read_session` is the leaf read on both paths.

    Ancestors resolve by id **beside the target**, which is the only guess
    available with nothing mounted and the right one for a lineage that was
    copied somewhere as a set. A chain whose ancestors are absent raises
    `LineageError` naming the missing one, rather than rendering a partial
    history as if it were whole.
    """
    header, events = read_session(path)

    def read_one(
        session_id: str, upto: int | None = None, family: str | None = None
    ) -> tuple[SessionHeader, list[SessionEvent]]:
        # The target is already read, and it is the only one whose *filename* is
        # known; everything above it is addressed the way the store addresses it.
        if session_id == header.id:
            return header, events
        return read_session(family_log(path.parent, session_id), upto=upto)

    return materialize(read_one, header.id)
