"""Shipped profile bundles. YAML rows, addressed by id."""

from __future__ import annotations

from pathlib import Path

BUNDLE_DIR = Path(__file__).parent

BASE = BUNDLE_DIR / "base.yaml"
HEADLESS = BUNDLE_DIR / "headless.yaml"

__all__ = ["BASE", "BUNDLE_DIR", "HEADLESS"]
