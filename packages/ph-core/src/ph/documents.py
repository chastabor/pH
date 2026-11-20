"""Reading a document that a person wrote: JSON or YAML, strictly or leniently.

This is the read side of `ph.paths.write_text_under`. It is also the one place
for a pattern that had been written out several times. `$PH_HOME/tui.json` and
`$PH_HOME/themes/theme-profile.yaml`, among others, each carried their own
chain: `read_text` → decode → `except FileNotFoundError` →
`except (decode, OSError)`. The log line and the set of caught exceptions
drifted between them.

**Two policies, one dispatch, and the difference is deliberate.**

`decode_document` raises. It is for a caller that has something to say about
*which* file failed and why: a theme directory where one bad file is skipped and
named, or a shipped document whose failure is a bug and not a preference.

`read_document` answers `None`. It is for a preference file, where the rule is
that *defaults are always a valid answer for a preference*. A document that
somebody hand-edited into invalid YAML costs them their customization and not
their session. An **absent** file is not logged, because that is the ordinary
first run. An unreadable file is logged, because a person who edited a file and
saw no change deserves the reason.

**The suffix decides the notation, and JSON keeps its own parser.** YAML would
accept a JSON document, but YAML 1.1 is not a strict superset. It refuses tab
indentation and is laxer about duplicate keys. A `.json` file that `json`
accepts and YAML does not would fail for a reason that its author could not see
in it.

@module ph.documents
"""

from __future__ import annotations

import logging
from json import JSONDecodeError
from pathlib import Path

from .cordis import LoaderError, safe_yaml_load
from .json import JsonValue, loads

__all__ = ["DOCUMENT_FAULTS", "decode_document", "read_document"]

log = logging.getLogger("ph.documents")

DOCUMENT_FAULTS: tuple[type[Exception], ...] = (
    LoaderError,
    JSONDecodeError,
    OSError,
    UnicodeDecodeError,
)
"""Everything that reading a document can fail with.

It is one tuple and not a hand-written `except` for each caller, because the set
cannot be guessed from the call site. `safe_yaml_load` raises `LoaderError` and
not `yaml.YAMLError`. A file with a bad byte raises `UnicodeDecodeError`, and
not the `OSError` that a reader expects. Both were missing from one of the three
chains that this module replaced.
"""


def decode_document(path: Path) -> JsonValue:
    """One document as data. It raises one of `DOCUMENT_FAULTS` when the
    document cannot be read.

    `safe_yaml_load` is the single YAML entry point of ph-core: no custom tags
    and no implicit date coercion. What comes back is therefore the same
    JSON-shaped tree that `loads` gives, and a caller does not need to know
    which notation it was given.
    """
    text = path.read_text(encoding="utf-8")
    return loads(text) if path.suffix == ".json" else safe_yaml_load(text, origin=str(path))


def read_document(path: Path) -> JsonValue | None:
    """One document as data, or `None` when there is not one to read.

    This is for a preference file. The module docstring says why the two
    policies differ and when to use `decode_document`.
    """
    try:
        return decode_document(path)
    except FileNotFoundError:
        # The ordinary first run. Nothing has gone wrong, and nothing is logged.
        return None
    except DOCUMENT_FAULTS as error:
        log.warning("ph.documents: %s is unreadable (%s); using defaults", path, error)
        return None
