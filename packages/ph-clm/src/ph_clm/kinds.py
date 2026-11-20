"""The kinds leaf of `ph_clm`: everything the bundle adds to the log, in one module (T4).

The bundle has no intent kinds. It has two log types, declared with
`declare_log_type`. Both are records that only an auditor reads, and both are
ignorable. The edit itself is a surface `replace`, a core event that any build
folds. A build without this package therefore derives the same context and skips
these records. For that reason the types are declared here and are not listed in
the vocabulary of ph-core, which keeps only the types that its own reader needs
(decision 7).

**Why this is a leaf module.** `ph_app.kinds` gives the reason, and it applies
here. A type is declared when its module is imported. `ph_clm/__init__.py`
therefore imports this module at module top, so any process that loaded any part
of the bundle has both types. A process that never loaded the bundle sees them as
ignorable types that it does not know. An example is the trajectory viewer of
phern, when it reads a stored log with nothing mounted. It still shows the
records from their payload.

**What this module may import:** the standard library and
`ph.session.known_event_types`. That module is pure, so importing it can never
cycle back through a seam. `test_intent_kinds.py` enforces this limit.

@module ph_clm.kinds
"""

from __future__ import annotations

from ph.session.known_event_types import declare_log_type

__all__ = ["DECLINED", "REVISED"]

REVISED = "clm/revised"
"""An edit that the model made to its own context: the sections that it took off
the surface, the node that stands for them, and the cost. The record lands in the
same batch as the revision that it describes."""

DECLINED = "clm/declined"
"""A context-file edit that was refused, and the reason. The receipt is in the
result of the call. This record is the copy for the auditor, and the reason is
the field that telemetry ships."""

declare_log_type(REVISED, owner="ph_clm.edits", ignorable=True, audit_only=True)
declare_log_type(
    DECLINED, owner="ph_clm.mirror", ignorable=True, audit_only=True, exported=("reason",)
)
