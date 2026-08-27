"""`python -m ph_runtime` — how the host spawns the guest."""

from __future__ import annotations

import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())
