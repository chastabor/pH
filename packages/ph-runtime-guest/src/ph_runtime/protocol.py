"""The fd-3 frame vocabulary, guest side.

**This module is written twice on purpose, but only in part, and the difference
matters.** Its twin is `ph_rlm.kernel.protocol`.

The four constants below are declared *here*, and the host **imports** them. This
is the only direction that is possible. The guest runs in
`$PH_CACHE/runtime-venv` with almost nothing installed. If the guest imported the
host package, the harness would be inside the process boundary that exists to
keep it out. The host has no such restriction, because it already depends on this
package. For plain data with no typing to lose, one declaration is therefore
better than two declarations that a test holds equal.

Everything else is a real twin, and `test_protocol_mirror.py` keeps the two sides
equal (D7, D4). It compares two things, for two different reasons:

* The required and optional field set of every frame. The host *derives* its own
  set from `TypedDict`s for mypy, and a `TypedDict` cannot be derived from a
  runtime table.
* The truncation marker, byte for byte. The host re-exports the copy of ph-core,
  and this side may not import ph-core at all. There are therefore two
  implementations, which a shared declaration could never reconcile.

**`FRAME_FIELDS` is the *only* declaration of the vocabulary on this side.** A
second declaration that no test compares is a copy that drifts, and one such copy
already drifted.

Because there are two definitions across the two *sides*, each default has
exactly one owner: the **host**. `boot` carries every limit as a required field.
A guest therefore has nothing to guess, and a changed default cannot have two
meanings at the same time.

Frame names and field names are camelCase on the wire. They match the
`code-runtime-python` protocol of dsh, so that its mirror test stays usable as a
reference (Q2). `call` carries the namespace under the key `global`, which is the
dsh name: a binding namespace *is* a global in the program. The Python name is
therefore `namespace`, and the wire name is different.

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
"""The version is four since `restore` was split into one frame for each variable (O3).

`more` is a required field. `encode` drops only `None`, so a `bool` default is
always sent. A guest that does not read `more` answers `done` on the first batch.
The host is then settled with the namespace half restored and with no report of
it. A required field was added, and that is the rule that this number exists for.

The version became three when `boot` got a required `idleCpuSeconds` (M1).

This is the same rule that moved the version to two. A guest that cannot read the
field has no budget between runs. A guest that reads a frame without the field
raises before `boot-ack`. Neither side can ignore the difference, so the pairing
is refused at boot. `venv._marker` keys on this number and carries no content
digest. It therefore rebuilds a warm `$PH_CACHE/runtime-venv`, so that an old
guest is not booted against a new host. `ping` and `pong` needed no version
change on their own, because an older pairing never sends or answers one.

The version became two when `call` got a required `run` (F4).

The rule below moved it: a **required field was added**, which a mismatched
pairing would misread and not ignore. A host that requires `run` refuses every
binding call of a guest that does not send it. The refusal is silent and reads
as "not available". The pairing must therefore be refused at `boot-ack`, and a
warm venv from before the change must rebuild. This is the purpose of the number.

The original note is still the rule:

A version gate exists for two builds that must understand each other. Before a
first release there are not two builds. The host and the guest ship together from
this repo, and a change to the frame set changes both sides in the same commit.
Two changes were recorded here as version bumps and never gated anything: `boot`
got a required `skills` field (P3-18), and `reply` got an optional `name`.

When there are two builds, the gate has this purpose: a `boot` that the guest
cannot serve is refused at `boot-ack`, and is not misread one frame at a time.
Change the number for a change that would make a mismatched pairing *misread*: a
required field that is added or removed, or a frame whose meaning changes. Do not
change it for a change that a pairing can ignore, because an unknown key is
dropped and a missing key takes its default.

**The number is not a cache key.** `venv._marker` does digest it, and a bump does
force a warm guest venv to rebuild. But if the number is bumped to ship an edit,
a number that means "compatibility" starts to mean "some byte changed".

**Sharing this declaration with the host does not make the two sides agree at
runtime.** The import must not be read that way. The host and the guest import
from *different* venvs, so they are two installations that can hold different
versions. That case is handled elsewhere, as before. The staleness marker in
`kernel/venv.py` rebuilds a venv whose protocol version or guest version changed,
and `boot-ack` refuses a pairing that still disagrees."""

PROTOCOL_FD: Final = 3
"""The default descriptor of the channel. fd 0, 1, and 2 stay with the program, so
the `print` output of a cell and the output of a grandchild process do not need
to be separated from frames."""

FD_ENV: Final = "PH_RUNTIME_FD"
"""Overrides `PROTOCOL_FD`. `subprocess.pass_fds` keeps a descriptor at the number
that it has in the *parent*. To move it to 3 in the child would need a
`preexec_fn`, which is unsafe in a threaded parent. The host therefore passes
the number and does not move the descriptor."""

FRAME_BYTES_ENV: Final = "PH_RUNTIME_FRAME_BYTES"
"""The largest frame that the reader of the guest assembles. The host sets the
size (F8).

The host derives the size from `maxSnapshotBytes`, with its own `frame_cap`, so
one expression sizes both readers. The value is in the environment and not in
`boot`, because the limit of the reader is fixed when the connection opens,
before any frame is read. `FD_ENV` is passed the same way for the same reason.
When the variable is absent, the guest keeps `channel.MAX_FRAME_BYTES`, which is
what an older host expects."""

NAMESPACE_ENV: Final = "PH_NAMESPACE_ID"

HOST_FRAMES: Final = frozenset({"boot", "run", "reply", "restore", "cancel", "shutdown", "ping"})
GUEST_FRAMES: Final = frozenset(
    {"boot-ack", "call", "log", "display", "snapshot", "done", "fault", "pong"}
)

UNPRODUCED_FRAMES: Final = frozenset({"display"})
"""The frames that are defined here and that nothing emits yet.

The host decodes `display`, collects it, carries it out through `CodeRunResult`,
and counts it on the card of the code cell. But the guest has no `display()` in
the cell namespace, so nothing can send the frame. The set is declared beside
the vocabulary and not in a test, because the developer who adds `display()`
reads this file, and the constraint must be where that developer looks. The
conformance suite reads this set. If the set is emptied without a producer, the
suite fails."""


FRAME_FIELDS: Final[dict[str, tuple[frozenset[str], frozenset[str]]]] = {
    # `maxValueBytes` caps the value of the cell, which goes into the context of
    # the model. `maxSnapshotBytes` caps one snapshotted variable, which goes
    # into the log. The sizes differ greatly, so the numbers differ. One number
    # would either truncate a legitimate DataFrame or put megabytes in a prompt.
    "boot": (
        frozenset(
            {
                "type",
                "protocol",
                "cpuSeconds",
                # See `Runner._end_runaway` (M1). This is the CPU that a namespace
                # may use while no cell is running. Nothing else limits that time.
                "idleCpuSeconds",
                "addressSpaceBytes",
                "maxLogBytes",
                "maxValueBytes",
                "maxSnapshotBytes",
                "namespaces",
                # Always sent, like `namespaces`. An empty list means a deployment
                # with no Python skills, which is a fact and not a missing value.
                "skills",
            }
        ),
        frozenset({"namespaceId"}),
    ),
    "run": (frozenset({"type", "id", "program"}), frozenset()),
    # `fatal` is C3 on the wire. The dispatch settled the whole run (a denial or
    # a budget). The proxy therefore raises an exception that the program is not
    # meant to catch, and the host aborts the run even if the program catches it.
    "reply": (frozenset({"type", "id", "ok"}), frozenset({"value", "message", "name", "fatal"})),
    # **One frame for each variable, like `snapshot` in the other direction**
    # (O3). `maxSnapshotBytes` limits each *value*. A frame that carried the whole
    # namespace had no limit, and the reader of the guest is sized before any
    # frame arrives (`FRAME_BYTES_ENV`, F8). One value per frame is therefore
    # what that size can hold. `more` marks a batch: the guest accumulates, and
    # answers on the frame where `more` is false.
    "restore": (frozenset({"type", "id", "variables", "more"}), frozenset()),
    "cancel": (frozenset({"type"}), frozenset({"id"})),
    "shutdown": (frozenset({"type"}), frozenset()),
    "boot-ack": (frozenset({"type", "protocol", "python", "limits"}), frozenset()),
    "call": (frozenset({"type", "id", "run", "global", "name", "args"}), frozenset()),
    "log": (frozenset({"type", "stream", "text"}), frozenset({"truncated"})),
    "display": (frozenset({"type", "mime", "data"}), frozenset({"meta"})),
    "snapshot": (frozenset({"type", "id", "variables"}), frozenset()),
    "done": (frozenset({"type", "id"}), frozenset({"value", "error", "truncated", "dropped"})),
    "fault": (frozenset({"type", "message"}), frozenset()),
    # **The clock of the event loop** (M2). The reader task of the guest answers
    # `ping`, so the round trip measures how far behind the event loop of the
    # guest is. A cell that burns CPU starves that task, and the answer is late
    # or absent. That is a *load* reading and not proof of death.
    #
    # `run` is the open run of the host. It is carried on the *question* and not
    # on the answer. The host waits for `done` with no wall clock of its own, so
    # it asks the one party that knows. A guest that is not running that run and
    # does not owe it a `done` settles the run immediately, through the same
    # `_send_done` that every other terminal path uses. The repair travels as an
    # ordinary `done`, so the pong carries only the clock.
    "ping": (frozenset({"type", "id", "run"}), frozenset()),
    "pong": (frozenset({"type", "id"}), frozenset()),
}
"""Frame type → (required fields, optional fields).

This is data and not docstrings, because the mirror test compares it. A field
that is added on one side and not on the other is then a failing assertion, and
not a frame that the other half silently ignores."""


def truncation_marker(dropped: int | None, cap: int) -> str:
    """Return the text that stands in for output that a cap discarded.

    The text is byte-identical on both sides, by construction and by test (D4).
    The host counts what it received and the guest counts what it dropped. A
    reader who compares a transcript with a log must not find two different
    sentences for the same event. `dropped` is `None` while the stream is still
    running, because the total of the guest is sent on its `done` frame.
    """
    if dropped is None:
        return f"\n[ph: output truncated — the rest is not kept, cap {cap} bytes]\n"
    return f"\n[ph: output truncated — {dropped} bytes dropped, cap {cap} bytes]\n"
