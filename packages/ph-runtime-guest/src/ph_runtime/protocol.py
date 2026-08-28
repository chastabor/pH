"""The fd-3 frame vocabulary, guest side.

**This module is written twice on purpose.** Its twin is
`ph_rlm.kernel.protocol`, and neither imports the other: the guest runs in
`$PH_CACHE/runtime-venv` with almost nothing installed, and making it import the
host package would put the harness inside the process boundary that exists to
keep the harness out. What keeps the two in step is `test_protocol_mirror.py`,
which compares `PROTOCOL_VERSION`, every frame's required and optional field
set, and the truncation marker byte for byte (D7, D4).

`FRAME_FIELDS` is the *only* declaration of the vocabulary on this side. There
were also `TypedDict`s for each frame; nothing consumed them at runtime, and
under `from __future__ import annotations` their `__required_keys__` cannot see
`NotRequired` — so they reported `namespaceId` as required while `FRAME_FIELDS`
had it optional. A third copy that no test compared had already drifted, which
is the argument against keeping it as documentation.

Because there are two definitions across the two *sides*, there is exactly one
owner of every default: the **host**. `boot` carries every limit as a required field, so a guest has
nothing to guess and a changed default cannot mean two things at once.

Frame names and field names are camelCase on the wire, matching dsh's
`code-runtime-python` protocol so its mirror test remains a usable reference
(Q2). `call` carries the namespace under the key `global`, which is what dsh
calls it — a binding namespace *is* a global in the program — so the Python-side
name is `namespace` and the wire name is not.

@module ph_runtime.protocol
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "FD_ENV",
    "FRAME_FIELDS",
    "GUEST_FRAMES",
    "HOST_FRAMES",
    "NAMESPACE_ENV",
    "PROTOCOL_FD",
    "PROTOCOL_VERSION",
    "truncation_marker",
]

PROTOCOL_VERSION: Final = 2
# 2: `boot` gained the required `skills` field (P3-18).
"""Bumped when a frame's field set changes. A `boot` the guest cannot serve is
refused at `boot-ack` rather than misread one frame at a time."""

PROTOCOL_FD: Final = 3
"""Where the channel is by default. fd 0/1/2 stay the program's own, so a cell's
`print` and a grandchild's output do not have to be untangled from frames."""

FD_ENV: Final = "PH_RUNTIME_FD"
"""Overrides `PROTOCOL_FD`. `subprocess.pass_fds` keeps a descriptor at the
number it has in the *parent*, and re-numbering it to 3 in the child would need
a `preexec_fn` — which is unsafe in a threaded parent. So the host passes the
number instead of moving the descriptor."""

NAMESPACE_ENV: Final = "PH_NAMESPACE_ID"

HOST_FRAMES: Final = frozenset({"boot", "run", "reply", "restore", "cancel", "shutdown"})
GUEST_FRAMES: Final = frozenset({"boot-ack", "call", "log", "display", "snapshot", "done", "fault"})


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
    "reply": (frozenset({"type", "id", "ok"}), frozenset({"value", "message", "fatal"})),
    "restore": (frozenset({"type", "id", "variables"}), frozenset()),
    "cancel": (frozenset({"type"}), frozenset({"id"})),
    "shutdown": (frozenset({"type"}), frozenset()),
    "boot-ack": (frozenset({"type", "protocol", "python", "limits"}), frozenset()),
    "call": (frozenset({"type", "id", "global", "name", "args"}), frozenset()),
    "log": (frozenset({"type", "stream", "text"}), frozenset({"truncated"})),
    "display": (frozenset({"type", "mime", "data"}), frozenset({"meta"})),
    "snapshot": (frozenset({"type", "id", "variables"}), frozenset()),
    "done": (frozenset({"type", "id"}), frozenset({"value", "error", "truncated"})),
    "fault": (frozenset({"type", "message"}), frozenset()),
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
