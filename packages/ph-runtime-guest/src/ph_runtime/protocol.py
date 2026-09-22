"""The fd-3 frame vocabulary, guest side.

**This module is written twice on purpose — but not all of it, and the
asymmetry is the point.** Its twin is `ph_rlm.kernel.protocol`.

The four constants below are declared *here* and **imported** by the host. That
direction is the only one available: the guest runs in `$PH_CACHE/runtime-venv`
with almost nothing installed, and making it import the host package would put
the harness inside the process boundary that exists to keep the harness out. The
host is under no such restriction — it already depends on this package — so for
anything that is plain data with no typing to lose, one declaration beats two
held equal by a test.

Everything else is genuinely twinned, and `test_protocol_mirror.py` is what keeps
it honest (D7, D4). Two things, for two different reasons: every frame's required
and optional field set, because the host *derives* its own from `TypedDict`s for
mypy and a `TypedDict` cannot be derived from a runtime table; and the truncation
marker byte for byte, because the host re-exports ph-core's copy and this side may
not import ph-core at all — two implementations, which is the one thing a shared
declaration could never have reconciled.

**`FRAME_FIELDS` is the *only* declaration of the vocabulary on this side** — a
second one that no test compares is a copy that drifts, and one already had.

Because there are two definitions across the two *sides*, there is exactly one
owner of every default: the **host**. `boot` carries every limit as a required
field, so a guest has nothing to guess and a changed default cannot mean two
things at once.

Frame names and field names are camelCase on the wire, matching dsh's
`code-runtime-python` protocol so its mirror test remains a usable reference (Q2).
`call` carries the namespace under the key `global`, which is what dsh calls it —
a binding namespace *is* a global in the program — so the Python-side name is
`namespace` and the wire name is not.

@module ph_runtime.protocol
"""

from __future__ import annotations

from typing import Final

from ._json import as_str

__all__ = [
    "FD_ENV",
    "FRAME_BYTES_ENV",
    "FRAME_FIELDS",
    "GUEST_FRAMES",
    "HOST_FRAMES",
    "NAMESPACE_ENV",
    "PROTOCOL_FD",
    "PROTOCOL_VERSION",
    "UNPRODUCED_FRAMES",
    "as_str",
    "truncation_marker",
]

PROTOCOL_VERSION: Final = 4
"""Four, since `restore` was split into one frame per variable (O3).

`more` is required — `encode` drops only `None`, so a `bool` default is always
sent — and a guest that does not read it answers `done` on the first batch,
leaving the host settled with the namespace half restored and nothing said.
A required field added, which is the rule the number exists for.

Three, since `boot` gained a required `idleCpuSeconds` (M1).

The rule that moved it to two, applied again: a guest that cannot read the field
has no budget between runs, and a guest that reads a frame without it raises
before `boot-ack`. Neither side can ignore the difference, so the pairing is
refused at boot — and `venv._marker`, which keys on this number and carries no
content digest, rebuilds a warm `$PH_CACHE/runtime-venv` instead of booting last
week's guest against today's host. `ping`/`pong` needed no bump on their own:
an older pairing simply never sends or answers one.

Two, since `call` gained a required `run` (F4).

Moved by exactly the rule below: a **required field added**, which a mismatched
pairing would misread rather than ignore. A guest that does not send `run` has
its every binding call refused by a host that requires it — silently, as "not
available" — so the pairing has to be refused at `boot-ack` instead, and a warm
venv from before the change has to rebuild. That is what the number is for.

The original note, still the rule:

A version gate exists for two builds that have to understand each other, and
before a first release there are not two: the host and the guest ship together
out of this repo, and a frame set that changed changed on both sides in the same
commit. `boot` gaining a required `skills` field (P3-18) and `reply` gaining an
optional `name` were both recorded here as bumps, and neither ever gated
anything.

What it is for when that day comes: a `boot` the guest cannot serve is refused
at `boot-ack` rather than misread one frame at a time. Move it for a change that
would make a mismatched pairing *misread* — a required field added or removed, a
frame's meaning changed — not for one it can simply ignore, since an unknown key
is dropped and a missing one takes its default.

**Not a cache key**, though `venv._marker` digests it and a bump does force a
warm guest venv to rebuild. Reaching for it to ship an edit is how a number that
means "compatibility" ends up meaning "some byte changed".

**Sharing this declaration with the host does not make the two agree at
runtime**, and nothing about the import should be read as if it did: host and
guest import from *different* venvs, so they are two installations that can hold
different versions. What covers that is unchanged and lives elsewhere —
`kernel/venv.py`'s staleness marker rebuilds a venv whose protocol or guest
version moved, and `boot-ack` refuses a pairing that still disagrees."""

PROTOCOL_FD: Final = 3
"""Where the channel is by default. fd 0/1/2 stay the program's own, so a cell's
`print` and a grandchild's output do not have to be untangled from frames."""

FD_ENV: Final = "PH_RUNTIME_FD"
"""Overrides `PROTOCOL_FD`. `subprocess.pass_fds` keeps a descriptor at the
number it has in the *parent*, and re-numbering it to 3 in the child would need
a `preexec_fn` — which is unsafe in a threaded parent. So the host passes the
number instead of moving the descriptor."""

FRAME_BYTES_ENV: Final = "PH_RUNTIME_FRAME_BYTES"
"""The largest frame the guest's reader will assemble, sized by the host (F8).

The host derives it from `maxSnapshotBytes` — its own `frame_cap` — so both
readers are sized by one expression. In the environment rather than in `boot`
because the reader's limit is fixed when the connection opens, before any frame
has been read; `FD_ENV` crosses the same way for the same reason. Absent, the
guest keeps `channel.MAX_FRAME_BYTES`, which is what an older host expects."""

NAMESPACE_ENV: Final = "PH_NAMESPACE_ID"

HOST_FRAMES: Final = frozenset({"boot", "run", "reply", "restore", "cancel", "shutdown", "ping"})
GUEST_FRAMES: Final = frozenset(
    {"boot-ack", "call", "log", "display", "snapshot", "done", "fault", "pong"}
)

UNPRODUCED_FRAMES: Final = frozenset({"display"})
"""Frames defined here that nothing emits yet.

`display` is decoded by the host, collected, carried out through `CodeRunResult`
and counted on the code cell's card — but the guest has no `display()` in the
cell namespace, so nothing can send one. Declared beside the vocabulary rather
than in a test, because the person who adds `display()` reads this file and the
constraint has to be where they are looking. The conformance suite reads this
set; emptying it without a producer fails there."""


FRAME_FIELDS: Final[dict[str, tuple[frozenset[str], frozenset[str]]]] = {
    # `maxValueBytes` caps the cell's own value, which goes into the model's
    # context; `maxSnapshotBytes` caps one snapshotted variable, which goes into
    # the log. Different magnitudes, so different numbers: one figure would
    # either truncate a legitimate DataFrame or put megabytes in a prompt.
    "boot": (
        frozenset(
            {
                "type",
                "protocol",
                "cpuSeconds",
                # See `Runner._end_runaway` (M1): what a namespace may burn while
                # no cell is running, which is the window nothing else bounds.
                "idleCpuSeconds",
                "addressSpaceBytes",
                "maxLogBytes",
                "maxValueBytes",
                "maxSnapshotBytes",
                "namespaces",
                # Always sent, like `namespaces`: an empty list is a deployment
                # with no Python skills, which is a fact rather than an absence.
                "skills",
            }
        ),
        frozenset({"namespaceId"}),
    ),
    "run": (frozenset({"type", "id", "program"}), frozenset()),
    # `fatal` is C3 on the wire: the dispatch settled the whole run (a denial or
    # a budget), so the proxy raises what the program is not offered a chance to
    # catch — and the host aborts the run regardless of whether it tries.
    "reply": (frozenset({"type", "id", "ok"}), frozenset({"value", "message", "name", "fatal"})),
    # **One frame per variable, like `snapshot` coming back** (O3).
    # `maxSnapshotBytes` bounds each *value*; a frame carrying the whole
    # namespace was bounded by nothing, and the guest's reader is sized before
    # any frame arrives (`FRAME_BYTES_ENV`, F8) — so one value per frame is what
    # that size can promise to hold. `more` is the batch marker: the guest
    # accumulates and answers on the frame without it.
    "restore": (frozenset({"type", "id", "variables", "more"}), frozenset()),
    "cancel": (frozenset({"type"}), frozenset({"id"})),
    "shutdown": (frozenset({"type"}), frozenset()),
    "boot-ack": (frozenset({"type", "protocol", "python", "limits"}), frozenset()),
    "call": (frozenset({"type", "id", "run", "global", "name", "args"}), frozenset()),
    "log": (frozenset({"type", "stream", "text"}), frozenset({"truncated"})),
    "display": (frozenset({"type", "mime", "data"}), frozenset({"meta"})),
    "snapshot": (frozenset({"type", "id", "variables"}), frozenset()),
    "done": (frozenset({"type", "id"}), frozenset({"value", "error", "truncated"})),
    "fault": (frozenset({"type", "message"}), frozenset()),
    # **The loop's own clock** (M2). `ping` is answered by the guest's reader
    # task, so the round trip measures how far behind the guest's event loop
    # is — a cell burning CPU starves that task and the answer is late or
    # absent, which is a *load* reading and not a death certificate.
    #
    # `run` is the host's open run, and it rides on the *question* rather
    # than the answer. The host waits for `done` with no wall clock of its
    # own, so it asks the one party that knows: a guest neither running that
    # run nor owing it a `done` settles it on the spot, through the same
    # `_send_done` every other terminal path takes. The repair travels as an
    # ordinary `done`, which leaves the pong carrying nothing but the clock.
    "ping": (frozenset({"type", "id", "run"}), frozenset()),
    "pong": (frozenset({"type", "id"}), frozenset()),
}
"""frame type → (required fields, optional fields).

Data rather than docstrings, because this is what the mirror test compares. A
field added on one side and not the other is a failing assertion instead of a
frame the other half silently ignores."""


def truncation_marker(dropped: int, cap: int) -> str:
    """The text that stands in for output a cap discarded.

    Byte-identical on both sides by construction and by test (D4): the host
    counts what it received and the guest counts what it dropped, and a reader
    comparing a transcript to a log must not find two different sentences for
    the same event.
    """
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
