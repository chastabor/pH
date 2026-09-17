"""Reading a document a person wrote: JSON or YAML, strictly or leniently.

The read side of `ph.paths.write_text_under`, and the third home for a shape that
had been written out three times — `$PH_HOME/tui.json`, `$PH_HOME/settings.json`
and `$PH_HOME/themes/theme-profile.yaml` each carried their own
`read_text` → decode → `except FileNotFoundError` → `except (decode, OSError)`
ladder, with the log line and the caught set drifting between them.

**Two policies, one dispatch, and the difference is deliberate.**

`decode_document` raises. It is for a caller that has something to say about
*which* file failed and why — a theme directory where one bad file is skipped and
named, or a shipped document whose failure is a bug rather than a preference.

`read_document` answers `None`. It is for a preference file, where the rule is the
one `ph.seams.settings` states: *defaults are always a valid answer for a
preference*, so a document somebody hand-edited into invalid YAML costs them
their customization and not their session. An **absent** file is not logged —
that is the ordinary first run — while an unreadable one is, because a person who
edited a file and saw no change deserves the reason.

**Notation is decided by the suffix, and JSON keeps its own parser.** YAML would
accept a JSON document, but YAML 1.1 is not a strict superset: it refuses tab
indentation and is laxer about duplicate keys, so a `.json` file that `json`
accepts and YAML does not would fail for a reason its author could not see in it.

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
"""Everything reading a document can fail with.

One tuple rather than a hand-written `except` per caller, because the set is not
guessable from the call site: `safe_yaml_load` raises `LoaderError` and not
`yaml.YAMLError`, and a file with a bad byte raises `UnicodeDecodeError` rather
than the `OSError` a reader expects. Both were missing from one of the three
ladders this module replaced.
"""


def decode_document(path: Path) -> JsonValue:
    """One document as data, raising `DOCUMENT_FAULTS` when it will not read.

    `safe_yaml_load` is ph-core's single YAML door — no custom tags, no implicit
    date coercion — so what comes back is the same JSON-shaped tree `loads` gives
    and a caller needs no idea which notation it was handed.
    """
    text = path.read_text(encoding="utf-8")
    return loads(text) if path.suffix == ".json" else safe_yaml_load(text, origin=str(path))


def read_document(path: Path) -> JsonValue | None:
    """One document as data, or `None` when there is not one to read.

    For a preference file: see this module's own docstring for why the two
    policies differ and when to reach for `decode_document` instead.
    """
    try:
        return decode_document(path)
    except FileNotFoundError:
        # The ordinary first run. Nothing has gone wrong and nothing is said.
        return None
    except DOCUMENT_FAULTS as error:
        log.warning("ph.documents: %s is unreadable (%s); using defaults", path, error)
        return None
