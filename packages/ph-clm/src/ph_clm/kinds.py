"""`ph_clm`'s kinds leaf: everything the bundle brings to the log, in one module (T4).

No intent kinds; two log types, declared with `declare_log_type`. Both are records only
an auditor reads, and both are ignorable: an edit itself is a surface `replace`, a core
event any build folds, so a build without this package derives the same context and
skips the account. So they are declared here rather than listed in ph-core's
vocabulary, which keeps only the types its own reader needs (decision 7).

**Why a leaf.** `ph_app.kinds` gives the reason, and it holds here: a type is declared
when its module is imported, so `ph_clm/__init__.py` imports this module at module top,
and any process that loaded any part of the bundle has both. A process that never did —
phern's trajectory viewer reading a stored log with nothing mounted — meets them as
ignorable types it does not know, and shows them from their payload all the same.

**What it may import:** the standard library and `ph.session.known_event_types`, which
is pure, so importing it can never cycle back through a seam. `test_intent_kinds.py`
holds that line.

@module ph_clm.kinds
"""

from __future__ import annotations

from ph.session.known_event_types import declare_log_type

__all__ = ["DECLINED", "REVISED"]

REVISED = "clm/revised"
"""A model's own context edit: which sections it took off the surface, the node standing
for them, and what that cost. Lands in the batch of the revision it describes."""

DECLINED = "clm/declined"
"""A context-file edit refused, and why. Its receipt is in the call's result; this is
the auditor's copy."""

declare_log_type(REVISED, owner="ph_clm.edits", ignorable=True, audit_only=True)
declare_log_type(DECLINED, owner="ph_clm.mirror", ignorable=True, audit_only=True)
