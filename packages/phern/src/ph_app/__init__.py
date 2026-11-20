"""`ph_app`: the pH command line and the Textual TUI.

**Importing any part of the app declares its intent kinds** (T4). `kinds` is
imported here, statically. A process that loaded any part of the app may resume a
log that the app wrote, and it therefore has `CLIENT_COMMAND` declared for repair
to settle. See `ph_app.kinds`.
"""

from __future__ import annotations

from . import kinds as kinds

__all__: list[str] = ["kinds"]
