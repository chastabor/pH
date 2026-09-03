"""The browser front end (P7-05). See `ph_app.web.serve`.

Nothing is imported here: `textual-serve` is an extra (`ph-app[web]`), so
importing this package must not require it. `ph_app.cli` imports `serve` inside
the `--mode web` branch, which is where a missing extra becomes an install line
rather than an `ImportError` a person has to read.

@module ph_app.web
"""

from __future__ import annotations

__all__: list[str] = []
