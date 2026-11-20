"""The intent kinds of `ph_rlm`: every pair that the bundle writes through
`ctx.intents`, in one leaf (T4).

There is one kind today, `HARNESS_PROBE`: the check of H1 that the reference of
a skill entry resolves (S21).

**Why a leaf.** `ph_app.kinds` gives the reason for the app, and the same reason
holds here. Repair settles the kinds that are declared in the process that does
the resume. ph-core cannot import the bundle, so the bundle brings its own
kinds, and `ph_rlm/__init__.py` imports this module at module top. The types of
the pair are in the vocabulary of ph-core (`INTENT_PAIRS`). A process without
the bundle therefore still reads the log, and it refuses by name to resume a log
that holds an open probe.

**What it may import:** the standard library, `ph.session.intents`,
`ph.session.events`, `ph.session.writers` and `ph.json`. An import of this
module can therefore never cycle back through a seam. `test_intent_kinds.py`
checks that rule.

@module ph_rlm.kinds
"""

from __future__ import annotations

from ph.json import JsonObject
from ph.session.events import SessionEvent
from ph.session.intents import IntentKind, Unsettled, declare_intent, opened_seq, seq_field
from ph.session.writers import log_writer

__all__ = ["HARNESS_PROBE", "probe_settled"]

_LOG = log_writer(__name__)
"""The writer of this leaf, which `HARNESS_PROBE` carries (T6)."""


def _probe_seq(event: SessionEvent) -> str | None:
    """The probe that a `harness/probed` settles: its `probeSeq`, as a key."""
    return seq_field(event, "probeSeq")


def probe_settled(probe_seq: int, unresolved: str | None) -> JsonObject:
    """A `harness/probed` payload: which probe, and why its reference did not
    resolve. The reason is `None` when the reference resolved."""
    return {"probeSeq": probe_seq, "unresolved": unresolved}


def _unfinished(opened: SessionEvent, why: Unsettled) -> JsonObject:
    """The payload for a probe whose own settle was never written: the harness
    stopped, or the run raised. Nothing saw the reference resolve. The refinement
    that the probe checked is written only after every probe has settled, so no
    entry depends on this probe."""
    return probe_settled(opened.seq, "the probe did not finish")


HARNESS_PROBE = declare_intent(
    IntentKind(
        opened="harness/probe",
        settled="harness/probed",
        opened_key=opened_seq,
        settled_key=_probe_seq,
        # The module may have been imported. Its top level runs on import, and a
        # probe that was cut short cannot say how far it got.
        orphan="outcome-unknown",
        # The record is on disk before the kernel imports anything. A probe whose
        # record cannot be written does not run.
        barrier="durable",
        closer=_unfinished,
        writer=_LOG,
    )
)
"""The probe of H1: the reference that it resolves, then what it found (S21).
`HarnessService._probe` opens and settles it."""
