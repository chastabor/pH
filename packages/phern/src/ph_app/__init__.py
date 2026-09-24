"""`ph_app` — the pH command line, and from Phase 2 the Textual TUI.

**Importing any of the app declares its intent kinds** (T4): `kinds` is imported here,
statically, so a process that loaded any part of the app — and so may resume a log the
app wrote — has `CLIENT_COMMAND` for repair to settle. See `ph_app.kinds`.
"""

from __future__ import annotations

from . import kinds as kinds

__all__: list[str] = ["kinds"]
