"""`ph.documents` — reading a document a person wrote, strictly or leniently.

The module exists because three readers had each written the same
`read_text` → decode → `except FileNotFoundError` → `except (decode, OSError)`
ladder, and they had drifted: the caught sets differed, and two of the three
would have crashed on a file with a bad byte rather than falling back. So the
tests worth having here are the ones about the *edges* the copies disagreed on —
which faults are caught, and what is said about each — rather than the happy path.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from ph.documents import DOCUMENT_FAULTS, decode_document, read_document


def test_both_notations_read_the_same_tree(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('{"x": [1, "two"]}', encoding="utf-8")
    (tmp_path / "a.yaml").write_text("x:\n  - 1\n  - two\n", encoding="utf-8")
    assert read_document(tmp_path / "a.json") == read_document(tmp_path / "a.yaml")


def test_a_yml_file_is_yaml_too(tmp_path: Path) -> None:
    (tmp_path / "a.yml").write_text("x: 1\n", encoding="utf-8")
    assert read_document(tmp_path / "a.yml") == {"x": 1}


def test_json_keeps_its_own_parser(tmp_path: Path) -> None:
    """The reason the `.json` branch exists rather than riding the YAML door.

    YAML 1.1 refuses tab indentation, so a perfectly good JSON document that
    happens to be tab-formatted — which is what most formatters emit — would fail
    for a reason its author cannot see in it.

    Sabotage: route `.json` through `safe_yaml_load` and this raises.
    """
    path = tmp_path / "tabbed.json"
    path.write_text('{\n\t"x": 1\n}', encoding="utf-8")
    assert decode_document(path) == {"x": 1}


def test_a_missing_document_is_none_and_says_nothing(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The ordinary first run. Nothing has gone wrong, so nothing is said.

    Sabotage: fold `FileNotFoundError` into the logged branch, and every first
    run warns about a file it was never going to find.
    """
    with caplog.at_level(logging.WARNING, logger="ph.documents"):
        assert read_document(tmp_path / "absent.yaml") is None
    assert not caplog.records


@pytest.mark.parametrize(
    ("name", "body"),
    [
        ("bad.json", '{"x": '),
        ("bad.yaml", "x: [unclosed\n"),
        ("bad.yml", "a:\n\tb: 1\n"),
    ],
)
def test_an_unreadable_document_is_none_and_names_the_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, name: str, body: str
) -> None:
    """A person who edited a file and saw no change deserves the reason."""
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="ph.documents"):
        assert read_document(path) is None
    assert str(path) in caplog.text


def test_a_document_that_is_not_utf8_is_caught(tmp_path: Path) -> None:
    """`UnicodeDecodeError` is not an `OSError`, which is what two of the three
    ladders this module replaced were missing — a stray byte crashed them.

    Sabotage: drop `UnicodeDecodeError` from `DOCUMENT_FAULTS`.
    """
    path = tmp_path / "latin.yaml"
    path.write_bytes(b"name: caf\xe9\n")
    assert read_document(path) is None
    assert UnicodeDecodeError in DOCUMENT_FAULTS


def test_a_directory_where_a_document_was_expected_is_caught(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").mkdir()
    assert read_document(tmp_path / "a.yaml") is None


def test_decode_raises_where_read_answers_none(tmp_path: Path) -> None:
    """The two policies, side by side — the whole reason both are exported.

    A caller with something to say about *which* file failed catches
    `DOCUMENT_FAULTS` itself; a preference file takes the `None`.
    """
    path = tmp_path / "bad.yaml"
    path.write_text("x: [unclosed\n", encoding="utf-8")
    with pytest.raises(DOCUMENT_FAULTS):
        decode_document(path)
    assert read_document(path) is None

    missing = tmp_path / "absent.json"
    with pytest.raises(FileNotFoundError):
        decode_document(missing)
    assert read_document(missing) is None


def test_a_yaml_fault_is_not_a_yaml_error(tmp_path: Path) -> None:
    """`safe_yaml_load` raises `LoaderError`, not `yaml.YAMLError`.

    Stated as a test because it is the fault a caller writing its own `except`
    gets wrong: the obvious `except yaml.YAMLError` never fires.
    """
    import yaml

    path = tmp_path / "bad.yaml"
    path.write_text("x: [unclosed\n", encoding="utf-8")
    with pytest.raises(Exception) as raised:
        decode_document(path)
    assert not isinstance(raised.value, yaml.YAMLError)
    assert isinstance(raised.value, DOCUMENT_FAULTS)


def test_a_document_is_data_not_code(tmp_path: Path) -> None:
    """The one policy that must survive going through this module."""
    path = tmp_path / "evil.yaml"
    path.write_text("x: !!python/object/apply:os.system ['echo hi']\n", encoding="utf-8")
    assert read_document(path) is None
