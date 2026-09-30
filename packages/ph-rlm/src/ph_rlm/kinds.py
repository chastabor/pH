"""`ph_rlm`'s intent kinds: every pair the bundle writes through `ctx.intents`, in one leaf (T4).

One kind today, `HARNESS_PROBE`: H1's check that a skill entry's reference resolves
(S21).

**Why a leaf.** `ph_app.kinds` gives the reason for the app, and it holds here:
repair settles the kinds declared in the process doing the resume, ph-core cannot
import the bundle, so the bundle brings its own, and `ph_rlm/__init__.py` imports
this module at module top. The pair's types are in ph-core's vocabulary
(`INTENT_PAIRS`), so a process without the bundle still reads the log, and refuses
by name to resume one holding an open probe.

**What it may import:** the standard library, `ph.session.intents`, `ph.session.events`,
`ph.session.writers` and `ph.json`, so importing it can never cycle back through a
seam. `test_intent_kinds.py` holds that line.

@module ph_rlm.kinds
"""

from __future__ import annotations

from ph.json import JsonObject
from ph.session.events import SessionEvent
from ph.session.intents import IntentKind, Unsettled, declare_intent, opened_seq, seq_field
from ph.session.writers import log_writer

__all__ = ["HARNESS_PROBE", "probe_settled"]

_LOG = log_writer(__name__)
"""This leaf's writer, which `HARNESS_PROBE` carries (T6)."""


def _probe_seq(event: SessionEvent) -> str | None:
    """The probe a `harness/probed` settles — its `probeSeq`, as a key."""
    return seq_field(event, "probeSeq")


def probe_settled(probe_seq: int, unresolved: str | None) -> JsonObject:
    """A `harness/probed` payload: which probe, and why its reference did not
    resolve — `None` when it did."""
    return {"probeSeq": probe_seq, "unresolved": unresolved}


def _unfinished(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """A probe whose own settle was never written: the harness stopped, or the run
    raised. Nothing saw the reference resolve, and the refinement it checked is
    written only after every probe has settled, so no entry rests on it."""
    return probe_settled(opened.seq, "the probe did not finish")


HARNESS_PROBE = declare_intent(
    IntentKind(
        opened="harness/probe",
        settled="harness/probed",
        opened_key=opened_seq,
        settled_key=_probe_seq,
        # The module may have been imported: its top level runs on import, and a
        # probe cut short cannot say how far it got.
        orphan="outcome-unknown",
        # On disk before the kernel imports anything: a probe whose record cannot be
        # written does not run.
        barrier="durable",
        closer=_unfinished,
        writer=_LOG,
    )
)
"""H1's probe: the reference it resolves, then what it found (S21). Opened and settled
by `HarnessService._probe`."""
