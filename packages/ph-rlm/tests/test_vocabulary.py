"""Every type `ph-rlm` appends is one this build can read back.

`ph-core` has this test over its own tree, and the reason it exists applies
equally here: `KNOWN_SESSION_EVENT_TYPES` gates the *seed* path, so a type this
package can write and ph-core cannot read is a log pH produces and then refuses
to resume. That failure would be found by whoever resumed the session, not by
whoever added the type.

The set lives in ph-core because the reader does. This test is the other half:
the writer, checked against it.
"""

from __future__ import annotations

import re
from pathlib import Path

import ph_rlm
from ph.session import KNOWN_SESSION_EVENT_TYPES

APPEND = re.compile(r"""\.append\(\s*["']([a-z0-9/\-]+)["']""")


def _appended() -> set[str]:
    root = Path(ph_rlm.__path__[0])
    return {
        match
        for path in root.rglob("*.py")
        for match in APPEND.findall(path.read_text(encoding="utf-8"))
    }


def test_every_appended_type_is_known_to_the_reader() -> None:
    appended = _appended()
    assert appended, "the scan found no append call sites — the regex is stale"
    assert appended <= KNOWN_SESSION_EVENT_TYPES, appended - KNOWN_SESSION_EVENT_TYPES


def test_the_kernel_state_pair_is_in_the_vocabulary() -> None:
    """Named explicitly, because these two are what make D6's promise keepable."""
    assert {"kernel/snapshot", "kernel/restored"} <= KNOWN_SESSION_EVENT_TYPES
