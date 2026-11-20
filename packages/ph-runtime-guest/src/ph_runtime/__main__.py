"""`python -m ph_runtime`: the command that the host uses to start the guest."""

from __future__ import annotations

import sys

from .runner import main

if __name__ == "__main__":
    sys.exit(main())
